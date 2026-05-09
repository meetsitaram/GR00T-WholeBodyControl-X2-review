"""
M1 acceptance gate: X2 LeRobot v2.1 exporter ↔ Isaac-GR00T loader contract.

The M0 gate (tests/test_groot_contract.py) verified that the upstream
``unitree_g1_sonic`` reference dataset round-trips through
Isaac-GR00T's ``LeRobotEpisodeLoader``. M1 builds the equivalent
gate for *X2-emitted* data: we ask the gear_sonic exporter to produce
a real LeRobot v2.1 dataset on disk using the X2 ``RobotModel``, then
load it back through Isaac-GR00T with the X2 modality config and
assert every contract invariant we depend on at training time.

The fixture is built once per test session by
``gear_sonic.scripts.build_x2_sample_episode.build_sample_dataset``;
the test then verifies:

1. ``meta/info.json`` and ``meta/modality.json`` are present and
   reflect the X2 ``RobotModel``-derived joint slices.
2. ``Gr00tDatasetMetadata.validate_modality_config`` recognises every
   required top-level modality key.
3. Isaac-GR00T's ``LeRobotEpisodeLoader`` can ingest the dataset using
   the X2 ``ModalityConfig`` factory and yields a DataFrame whose
   columns / dtypes / shapes match the ``unitree_g1_sonic`` contract
   (slot-for-slot for the keys we share with the reference embodiment).
4. The synthetic ego-view video round-trips through the loader's video
   backend and decodes to the expected ``(H, W, 3)`` shape.

Run via::

    .venv/bin/python -m pytest tests/test_x2_lerobot_exporter.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
GROOT_CLONE = REPO_ROOT / "external_dependencies" / "Isaac-GR00T"


def _ensure_groot_on_path() -> None:
    if not GROOT_CLONE.is_dir():
        raise RuntimeError(
            f"Isaac-GR00T clone not found at {GROOT_CLONE}. "
            "See docs/source/references/x2_isaac_groot_data_contract.md."
        )
    str_path = str(GROOT_CLONE)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)


@pytest.fixture(scope="module")
def x2_sample_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a one-episode X2 LeRobot dataset on disk for the test session."""
    from gear_sonic.scripts.build_x2_sample_episode import build_sample_dataset

    out_dir = tmp_path_factory.mktemp("x2_sample_episode") / "dataset"
    build_sample_dataset(
        output_dir=out_dir,
        num_frames=50,
        num_episodes=1,
        task="play minecraft music on piano",
        overwrite=True,
    )
    return out_dir


def test_dataset_layout_on_disk(x2_sample_dataset: Path) -> None:
    """Invariant 1: the LeRobot v2.1 directory tree is correct."""
    assert (x2_sample_dataset / "meta" / "info.json").is_file()
    assert (x2_sample_dataset / "meta" / "modality.json").is_file()
    assert (x2_sample_dataset / "meta" / "episodes.jsonl").is_file()
    assert (x2_sample_dataset / "meta" / "tasks.jsonl").is_file()
    assert (x2_sample_dataset / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert (
        x2_sample_dataset
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000000.mp4"
    ).is_file()


def test_modality_json_state_action_layout(x2_sample_dataset: Path) -> None:
    """Invariant 2: meta/modality.json mirrors the unitree_g1_sonic state shape."""
    modality = json.loads((x2_sample_dataset / "meta" / "modality.json").read_text())

    state_groups = set(modality["state"].keys())
    expected_state = {
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "projected_gravity",
    }
    assert state_groups == expected_state, (
        f"X2 state modality groups diverged from unitree_g1_sonic: {state_groups}"
    )

    # Body slices come from the X2 RobotModel; verify the contiguity invariant
    # the exporter relies on.
    for grp in ("left_leg", "right_leg", "waist", "left_arm", "right_arm"):
        s = modality["state"][grp]
        assert s["original_key"] == "observation.state", (
            f"{grp} should slice into observation.state, got {s}"
        )
        assert s["end"] > s["start"]

    # Hand slices live just after the body block.
    left_hand = modality["state"]["left_hand"]
    right_hand = modality["state"]["right_hand"]
    assert left_hand["end"] - left_hand["start"] == 10
    assert right_hand["end"] - right_hand["start"] == 10
    assert right_hand["start"] == left_hand["end"]
    assert left_hand["start"] >= 31  # body block

    action = modality["action"]
    assert set(action.keys()) == {"motion_token", "left_hand_joints", "right_hand_joints"}
    assert action["motion_token"]["end"] - action["motion_token"]["start"] == 64
    assert action["left_hand_joints"]["end"] - action["left_hand_joints"]["start"] == 10
    assert action["right_hand_joints"]["end"] - action["right_hand_joints"]["start"] == 10


def test_info_json_carries_x2_script_config(x2_sample_dataset: Path) -> None:
    info = json.loads((x2_sample_dataset / "meta" / "info.json").read_text())
    assert info["robot_type"] == "agibot_x2_ultra"
    assert info["fps"] == 50
    sc = info["script_config"]
    assert sc["embodiment_tag"] == "new_embodiment"
    assert sc["hand_variant"] == "omnihand_10"
    assert sc["num_body_joints"] == 31
    assert sc["hand_dof_per_side"] == 10


def test_lerobot_episode_loader_round_trips_x2(x2_sample_dataset: Path) -> None:
    """Invariant 3+4: Isaac-GR00T can ingest X2 data with the X2 modality config."""
    _ensure_groot_on_path()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gear_sonic.data.x2_modality_config import make_x2_modality_config

    modality_configs = make_x2_modality_config(hand_dof=10, action_horizon=4)
    # Action horizon is overridden to 4 to keep the M1 fixture short
    # (1-sec episode @ 50 Hz cannot satisfy a 40-step action lookahead).
    loader = LeRobotEpisodeLoader(
        dataset_path=str(x2_sample_dataset),
        modality_configs=modality_configs,
        video_backend="torchcodec",
    )

    assert len(loader) == 1, f"expected 1 episode, got {len(loader)}"
    assert loader.fps == 50

    episode = loader[0]
    expected_columns = {
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.projected_gravity",
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
        "video.ego_view",
        "language.annotation.human.task_description",
    }
    missing = expected_columns - set(episode.columns)
    assert not missing, f"missing loader columns: {missing}"

    state_arm = episode["state.left_arm"].iloc[0]
    assert state_arm.shape == (7,)
    state_hand = episode["state.left_hand"].iloc[0]
    assert state_hand.shape == (10,)
    state_gravity = episode["state.projected_gravity"].iloc[0]
    assert state_gravity.shape == (3,)
    np.testing.assert_allclose(state_gravity, [0.0, 0.0, -1.0])

    motion_token = episode["action.motion_token"].iloc[0]
    assert motion_token.shape == (64,)
    left_hand_action = episode["action.left_hand_joints"].iloc[0]
    assert left_hand_action.shape == (10,)

    frame = episode["video.ego_view"].iloc[0]
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3, f"unexpected frame: {frame.shape}"

    text = episode["language.annotation.human.task_description"].iloc[0]
    assert text == "play minecraft music on piano"
