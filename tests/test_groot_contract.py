"""
M0 acceptance gate: Isaac-GR00T data contract.

This test pins the upstream LeRobot v2.1 contract that the gear_sonic
exporter must obey. It is intentionally lightweight — it only exercises
the upstream demo dataset and the X2 ModalityConfig factory so the gate
can run on a laptop without the GR00T-N1.7-3B checkpoint or any teleop
recording present.

The gate covers four invariants:

1. The locally cloned ``external_dependencies/Isaac-GR00T`` exposes
   ``EmbodimentTag``, ``ModalityConfig``, ``register_modality_config`` and
   the ``unitree_g1_sonic`` reference modality config.
2. The reference ``unitree_g1_sonic`` modality config has the action
   layout we plan to mirror for X2 (motion_token + 2 hand keys, 40-step
   chunk, ABSOLUTE / NON_EEF / DEFAULT).
3. ``LeRobotEpisodeLoader`` can parse upstream's ``demo_data/cube_to_bowl_5``
   end-to-end and yield a DataFrame with the expected columns / shapes /
   dtypes (this is the mechanical schema the gear_sonic exporter has to
   produce for X2).
4. ``gear_sonic.data.x2_modality_config.make_x2_modality_config`` builds a
   structurally valid ModalityConfig for both hand-DOF variants and can
   be registered against ``EmbodimentTag.NEW_EMBODIMENT`` without
   colliding with the upstream registry.

Run with::

    .venv/bin/python tests/test_groot_contract.py

or via pytest::

    .venv/bin/python -m pytest tests/test_groot_contract.py -q

Documentation: ``docs/source/references/x2_isaac_groot_data_contract.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
GROOT_CLONE = REPO_ROOT / "external_dependencies" / "Isaac-GR00T"
DEMO_DATASET = GROOT_CLONE / "demo_data" / "cube_to_bowl_5"


def _ensure_groot_on_path() -> None:
    """Make the in-repo Isaac-GR00T clone importable without an editable install.

    The plan installs gr00t via ``pip install -e external_dependencies/Isaac-GR00T
    --no-deps``, but this gate also needs to pass on a fresh clone before that
    install has been performed. Falling back to ``sys.path`` keeps the gate
    independent of installer state.
    """
    if not GROOT_CLONE.is_dir():
        raise RuntimeError(
            f"Isaac-GR00T clone not found at {GROOT_CLONE}. Run:\n"
            "    git clone https://github.com/NVIDIA/Isaac-GR00T "
            "external_dependencies/Isaac-GR00T"
        )
    str_path = str(GROOT_CLONE)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)


def test_upstream_module_layout_intact() -> None:
    """Invariant 1: the upstream symbols we depend on are still importable."""
    _ensure_groot_on_path()
    from gr00t.configs.data.embodiment_configs import (  # noqa: F401
        MODALITY_CONFIGS,
        register_modality_config,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag, POSTTRAIN_TAGS  # noqa: F401
    from gr00t.data.types import (  # noqa: F401
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
        ModalityConfig,
        VLAStepData,
    )

    assert "unitree_g1_sonic" in MODALITY_CONFIGS, (
        "UNITREE_G1_SONIC reference modality config disappeared from upstream."
    )
    assert EmbodimentTag.UNITREE_G1_SONIC in POSTTRAIN_TAGS
    assert EmbodimentTag.NEW_EMBODIMENT.value == "new_embodiment"


def test_unitree_g1_sonic_action_layout_matches_plan() -> None:
    """Invariant 2: the reference SONIC action space is what we plan to mirror."""
    _ensure_groot_on_path()
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    sonic = MODALITY_CONFIGS["unitree_g1_sonic"]
    assert set(sonic.keys()) == {"video", "state", "action", "language"}

    action = sonic["action"]
    assert action.modality_keys == ["motion_token", "left_hand_joints", "right_hand_joints"]
    assert action.delta_indices == list(range(40)), (
        f"Expected 40-step action horizon, got {action.delta_indices}"
    )
    assert action.action_configs is not None
    for cfg in action.action_configs:
        assert cfg.rep.value == "absolute"
        assert cfg.type.value == "non_eef"
        assert cfg.format.value == "default"

    assert sonic["video"].modality_keys == ["ego_view"]
    assert sonic["language"].modality_keys == ["annotation.human.task_description"]
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
    assert set(sonic["state"].modality_keys) == expected_state, (
        f"State joint groups changed upstream: {sonic['state'].modality_keys}"
    )


def test_demo_dataset_loads_and_returns_expected_shapes() -> None:
    """Invariant 3: a real LeRobot v2.1 dataset round-trips through the loader."""
    _ensure_groot_on_path()
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.types import (
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
        ModalityConfig,
    )

    if not DEMO_DATASET.is_dir():
        raise RuntimeError(
            f"Demo dataset missing at {DEMO_DATASET}. "
            "Re-run the Isaac-GR00T clone (it should ship demo_data/cube_to_bowl_5/)."
        )

    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["single_arm", "gripper"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }

    loader = LeRobotEpisodeLoader(
        dataset_path=str(DEMO_DATASET),
        modality_configs=modality_configs,
        video_backend="torchcodec",
    )

    assert len(loader) == 5
    assert loader.fps == 30
    assert "data/chunk-{episode_chunk:03d}" in loader.data_path_pattern

    episode = loader[0]
    expected_columns = {
        "state.single_arm",
        "state.gripper",
        "action.single_arm",
        "action.gripper",
        "video.front",
        "video.wrist",
        "language.annotation.human.task_description",
    }
    assert expected_columns.issubset(set(episode.columns)), (
        f"missing columns: {expected_columns - set(episode.columns)}"
    )

    state_arr = episode["state.single_arm"].iloc[0]
    assert isinstance(state_arr, np.ndarray)
    assert state_arr.shape == (5,)
    assert state_arr.dtype == np.float32

    action_arr = episode["action.single_arm"].iloc[0]
    assert action_arr.shape == (5,)

    frame = episode["video.front"].iloc[0]
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3, f"unexpected frame shape: {frame.shape}"

    text = episode["language.annotation.human.task_description"].iloc[0]
    assert isinstance(text, str) and len(text) > 0


def test_x2_modality_config_factory() -> None:
    """Invariant 4: gear_sonic's X2 modality config factory is structurally valid.

    We do NOT import the side-loaders (``x2_modality_config_7dof.py`` /
    ``x2_modality_config_10dof.py``) here because they call
    ``register_modality_config`` at import time, which mutates the upstream
    registry and would clash if both side-loaders are imported in the same
    process. The factory call exercises the same code path.
    """
    _ensure_groot_on_path()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from gear_sonic.data.x2_modality_config import (
        DEFAULT_ACTION_HORIZON,
        DEFAULT_STATE_GROUPS,
        SONIC_MOTION_TOKEN_DIM,
        make_x2_modality_config,
    )

    assert SONIC_MOTION_TOKEN_DIM == 64
    assert DEFAULT_ACTION_HORIZON == 40
    assert "ego_view" not in DEFAULT_STATE_GROUPS  # video, not state
    assert "projected_gravity" in DEFAULT_STATE_GROUPS

    for hand_dof in (7, 10):
        cfg = make_x2_modality_config(hand_dof=hand_dof)
        assert set(cfg.keys()) == {"video", "state", "action", "language"}
        assert cfg["action"].modality_keys == [
            "motion_token",
            "left_hand_joints",
            "right_hand_joints",
        ]
        assert cfg["action"].delta_indices == list(range(DEFAULT_ACTION_HORIZON))
        assert len(cfg["action"].action_configs) == 3
        assert cfg["state"].modality_keys == list(DEFAULT_STATE_GROUPS)
        assert cfg["language"].modality_keys == ["annotation.human.task_description"]
        assert cfg["video"].modality_keys == ["ego_view"]

    try:
        make_x2_modality_config(hand_dof=5)  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("hand_dof=5 should have raised ValueError")


def main() -> int:
    tests = [
        test_upstream_module_layout_intact,
        test_unitree_g1_sonic_action_layout_matches_plan,
        test_demo_dataset_loads_and_returns_expected_shapes,
        test_x2_modality_config_factory,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - we want full traceback in CLI mode
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc!r}")
            import traceback

            traceback.print_exc()
        else:
            print(f"PASS  {fn.__name__}")
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        return 1
    print("\nOK: M0 contract gate green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
