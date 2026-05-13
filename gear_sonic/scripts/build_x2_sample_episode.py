"""
Synthesise a sample LeRobot v2.1 episode for the AgiBot X2 Ultra SONIC VLA pipeline.

This is the M1 acceptance-gate driver. It produces a tiny end-to-end
dataset on disk that:

1. Uses the X2 ``RobotModel`` for body-DOF structure and joint naming
   (so dataset feature names trace back to the URDF).
2. Has the ``unitree_g1_sonic``-compatible action layout
   (``motion_token`` + ``left_hand_joints`` + ``right_hand_joints``).
3. Carries a synthetic ego-view video (a per-frame numbered gradient)
   so the LeRobot writer's video pipeline gets exercised end-to-end.
4. Writes a single language prompt (``"play minecraft music on piano"``)
   matching the smoke-test convention in
   ``docs/source/tutorials/vla_training.md``.

The resulting dataset is the fixture used by
``tests/test_x2_lerobot_exporter.py`` to verify Isaac-GR00T's loader
can ingest X2 data. It is intentionally small (1 episode, ~1 s @ 50 Hz)
so the gate runs in a few seconds on a laptop.

Usage::

    timeout 120 .venv/bin/python gear_sonic/scripts/build_x2_sample_episode.py \\
        --output-dir /tmp/x2_sample_episode

By default the script writes to ``outputs/x2_sample_episode_<timestamp>``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_x2_vla import (
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    FPS,
    HAND_DOF_OMNI,
    SONIC_MOTION_TOKEN_DIM,
    assemble_observation_state,
    get_features_x2_vla,
    get_modality_config_x2_vla,
    get_x2_robot_model,
)


DEFAULT_TASK = "play minecraft music on piano"
DEFAULT_NUM_FRAMES = 50  # 1 second at 50 Hz
DEFAULT_NUM_EPISODES = 1


def _make_synthetic_ego_view(frame_idx: int, num_frames: int) -> np.ndarray:
    """Generate a deterministic ego-view RGB frame for the given index.

    The frame encodes the frame index in its red channel (so a video
    decode round-trip can verify ordering) plus a slow blue ramp tied
    to the frame index so the video isn't all-zeros (LeRobot's stats
    computation rejects degenerate per-frame distributions).
    """
    frame = np.zeros((EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3), dtype=np.uint8)
    red_value = int(255 * frame_idx / max(num_frames - 1, 1))
    blue_value = int(255 * (1 - frame_idx / max(num_frames - 1, 1)))
    frame[..., 0] = red_value
    frame[..., 2] = blue_value

    yy, xx = np.mgrid[: EGO_VIEW_HEIGHT, : EGO_VIEW_WIDTH]
    frame[..., 1] = ((xx + yy + frame_idx * 7) % 256).astype(np.uint8)
    return frame


def _stand_pose(num_body: int) -> np.ndarray:
    """Return the trained-default stand pose used as the per-frame body anchor."""
    return np.zeros(num_body, dtype=np.float64)


def _hand_wiggle(frame_idx: int, num_frames: int, dof: int) -> np.ndarray:
    """Synthetic per-frame finger trajectory for the M1 sample episode.

    Sinusoidal sweep of small amplitude (~0.2 rad) so the hand-joint
    columns aren't constant (constant columns trip LeRobot's per-feature
    stats validators).
    """
    phase = 2 * np.pi * frame_idx / max(num_frames, 1)
    base = 0.2 * np.sin(phase)
    return base * np.linspace(0.5, 1.5, dof, dtype=np.float64)


def _build_one_episode(
    exporter: Gr00tDataExporter,
    robot_model,
    num_frames: int,
    task: str,
) -> None:
    num_body = robot_model.num_joints
    hand_dof = HAND_DOF_OMNI

    body_q = _stand_pose(num_body)

    for frame_idx in range(num_frames):
        left_hand_q = _hand_wiggle(frame_idx, num_frames, hand_dof)
        right_hand_q = _hand_wiggle(frame_idx + num_frames // 2, num_frames, hand_dof)

        observation_state = assemble_observation_state(
            robot_model, body_q, left_hand_q, right_hand_q
        )
        projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Action targets: the SONIC VLA pipeline emits a 64-D motion token
        # plus the next-step hand joint targets. For the smoke episode we
        # use a constant standing token (zeros) so downstream training
        # gets a stable target distribution but the pipeline still
        # exercises every action key.
        motion_token = np.zeros(SONIC_MOTION_TOKEN_DIM, dtype=np.float64)
        # v1 schema: action.body_q_mj is the canonical post-SONIC body
        # command surface. The smoke episode is synthetic (no real SONIC
        # rollout), so executed == pre_sonic == default stand pose
        # zero-fill, and sonic_correction_max_rad is exactly zero.
        commanded_body_q_mj = np.zeros(num_body, dtype=np.float64)
        action_left_hand = left_hand_q.copy()
        action_right_hand = right_hand_q.copy()

        ego_view = _make_synthetic_ego_view(frame_idx, num_frames)

        frame_data = {
            "observation.state": observation_state,
            "observation.projected_gravity": projected_gravity,
            "action.motion_token": motion_token,
            "action.body_q_mj": commanded_body_q_mj,
            "action.left_hand_joints": action_left_hand,
            "action.right_hand_joints": action_right_hand,
            "action.body_q_mj_pre_sonic": commanded_body_q_mj.copy(),
            "action.left_hand_joints_pre_sonic": action_left_hand.copy(),
            "action.right_hand_joints_pre_sonic": action_right_hand.copy(),
            "action.sonic_correction_max_rad": np.zeros(1, dtype=np.float32),
            "observation.images.ego_view": ego_view,
            "task": task,
        }
        exporter.add_frame(frame_data)

    exporter.save_episode()


def build_sample_dataset(
    output_dir: Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    task: str = DEFAULT_TASK,
    overwrite: bool = True,
) -> Path:
    """Build the X2 sample LeRobot v2.1 dataset.

    Args:
        output_dir: Directory where the LeRobot dataset will be written.
        num_frames: Frames per episode. Default 50 frames (1 s @ 50 Hz).
        num_episodes: Episodes to write. Default 1 (sufficient for the
            M1 acceptance gate).
        task: Language prompt stored in ``meta/tasks.jsonl``.
        overwrite: If ``True``, deletes ``output_dir`` before writing.

    Returns:
        The ``output_dir`` path (resolved, absolute).
    """
    output_dir = Path(output_dir).resolve()
    # Gr00tDataExporter.create owns the output directory lifecycle; if we
    # pre-mkdir it, the exporter falls into its "resume" branch and tries
    # to fetch metadata from HF Hub. Let the exporter's overwrite_existing
    # flag handle pre-existing directories instead.
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    robot_model = get_x2_robot_model("omnihand_10")
    features = get_features_x2_vla(robot_model)
    modality_config = get_modality_config_x2_vla(robot_model)

    exporter = Gr00tDataExporter.create(
        save_root=output_dir,
        fps=FPS,
        features=features,
        modality_config=modality_config,
        task=task,
        script_config={
            "robot_type": "agibot_x2_ultra",
            "embodiment_tag": "new_embodiment",
            "hand_variant": "omnihand_10",
            "num_body_joints": robot_model.num_joints,
            "hand_dof_per_side": HAND_DOF_OMNI,
            "fps": FPS,
            "synthetic": True,
        },
        robot_type="agibot_x2_ultra",
    )

    for _ in range(num_episodes):
        _build_one_episode(exporter, robot_model, num_frames, task)

    # Isaac-GR00T's LeRobotEpisodeLoader hard-requires meta/stats.json --
    # it computes per-feature mean/std/min/max/q01/q99 used by the trainer
    # for normalisation. Generate it here so the dataset is loader-ready
    # the moment build_sample_dataset returns. Lazy import keeps the
    # build script usable even when Isaac-GR00T isn't on the path.
    try:
        from gr00t.data.stats import generate_stats
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "Isaac-GR00T (gr00t.data.stats) is required to finalise the "
            "X2 sample dataset. Install via "
            "`uv pip install -e external_dependencies/Isaac-GR00T --no-deps "
            "--python .venv/bin/python` (see "
            "docs/source/references/x2_isaac_groot_data_contract.md)."
        ) from exc
    generate_stats(output_dir)

    print(f"[build_x2_sample_episode] wrote {num_episodes} episode(s) "
          f"of {num_frames} frames to {output_dir}")
    return output_dir


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write the LeRobot v2.1 dataset. Defaults to "
            "outputs/x2_sample_episode_<timestamp>."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Frames per episode (default: 50, = 1 s at 50 Hz).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Episodes to write (default: 1).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Language prompt (default matches smoke-test convention).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Disable the default behaviour of wiping output-dir before writing.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"x2_sample_episode_{timestamp}"

    build_sample_dataset(
        output_dir=output_dir,
        num_frames=args.num_frames,
        num_episodes=args.num_episodes,
        task=args.task,
        overwrite=not args.no_overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
