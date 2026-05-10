"""Kinematic-only MuJoCo replay of a recorded LeRobot v2.1 episode.

Reads ``action.body_q_mj`` plus ``action.left_hand_joints`` /
``action.right_hand_joints`` straight out of the recorded parquet and
faithfully replays them in a passive MuJoCo viewer with the floating
base pinned to the on-feet stand pose. Nothing is re-derived: no
SONIC, no policy, no Quest 3 reader, no IK. This is the offline
counterpart to :file:`teleop_x2_kinematic.py` -- pure ``mj_forward``
kinematics, the same way the smoketest renderer works.

Schema dispatch (v0 vs v1)
    * **v1 datasets** (``meta/dataset_format_version.json`` with
      ``version=1``): the canonical body action lives in
      ``action.body_q_mj``. SONIC-recorded datasets additionally have
      ``action.body_q_mj_pre_sonic`` for retargeter analysis; this
      replay reads only the canonical column.
    * **v0 datasets** (no version file, legacy schema): the canonical
      body action lives in ``action.commanded_body_q_mj``. The replay
      auto-falls-back to that column when ``action.body_q_mj`` is
      absent.

What it shows
    The exact joint trajectory the recorder wrote to disk. If the
    recording captured weird arm or hand commands, this viewer will
    show those weird commands. There is no smoothing, retargeting, or
    physics correction in the loop.

What it does NOT show
    SONIC tracking-policy output. The commanded ``body_q_mj`` is what
    the operator told the lower body to do, not what a stabilised
    deploy would have actually executed. For SONIC-loop replay you
    need the (deferred) ``replay_x2_sonic.py`` or the recipe-6.3
    ``parquet -> ZMQ -> deploy`` flow.

Embodiment dispatch
    Selected via ``--robot`` (default ``x2``). The flag resolves to
    an :class:`gear_sonic.utils.embodiment.EmbodimentConfig` via the
    registry. Adding a new robot is a config-only change in
    :file:`gear_sonic/utils/embodiment/<name>.py`. Today only ``x2``
    has a real config; ``g1`` registers a stub that fails fast at
    model-build time.

Example::

    python -m gear_sonic.scripts.replay_x2_kinematic \\
        --dataset x2_quest3_kinematic_v4 --episode 0

    python -m gear_sonic.scripts.replay_x2_kinematic \\
        --dataset /abs/path/to/dataset --episode 3 --rate 25.0 --loop
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Default chunk size for LeRobot v2.1 datasets when meta/info.json
# is missing or doesn't declare it. Matches the upstream default and
# the recorder's behaviour (see Gr00tDataExporter).
_DEFAULT_CHUNK_SIZE: int = 1000


# Required columns that this replay CLI consumes. Kept here (not in
# the embodiment config) because the schema is part of the LeRobot
# contract, not the robot. The body action column is one of two
# alternatives (v1 = ``action.body_q_mj``, v0 = ``action.commanded_body_q_mj``);
# we resolve to whichever is present at load time.
_BODY_ACTION_CANDIDATES: tuple[str, ...] = (
    "action.body_q_mj",            # v1 schema (post-SONIC canonical or kinematic)
    "action.commanded_body_q_mj",  # v0 legacy
)
_REQUIRED_HAND_COLUMNS = (
    "action.left_hand_joints",
    "action.right_hand_joints",
)


def _resolve_dataset_path(name_or_path: str) -> Path:
    """Resolve ``--dataset`` to a concrete dataset root directory.

    The argument may be either:

    * a path to an existing directory (absolute or relative to CWD), or
    * a short name resolved against ``<repo>/data/lerobot/<name>``.

    Raises:
        FileNotFoundError: if neither resolution finds a directory.
    """
    direct = Path(name_or_path).expanduser()
    if direct.is_dir():
        return direct.resolve()

    candidate = (REPO_ROOT / "data" / "lerobot" / name_or_path).resolve()
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"Dataset {name_or_path!r} not found. Tried:\n"
        f"  - {direct}\n"
        f"  - {candidate}\n"
        "Pass either an absolute path or a short name under "
        "data/lerobot/."
    )


def _read_chunk_size(dataset_root: Path) -> int:
    """Pull ``chunks_size`` out of ``meta/info.json``; fall back to 1000.

    LeRobot v2.1 declares the chunk size per-dataset so re-runs can
    change it without breaking older datasets. Reading it here keeps
    :func:`_episode_parquet_path` independent of any constant guess.
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return _DEFAULT_CHUNK_SIZE
    try:
        info = json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return _DEFAULT_CHUNK_SIZE
    chunk = info.get("chunks_size", _DEFAULT_CHUNK_SIZE)
    try:
        return int(chunk)
    except (TypeError, ValueError):
        return _DEFAULT_CHUNK_SIZE


def _episode_parquet_path(
    dataset_root: Path,
    episode: int,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Path:
    """Return the LeRobot v2.1 parquet path for ``episode``.

    Layout: ``<dataset_root>/data/chunk-{ep // chunk_size:03d}/episode_{ep:06d}.parquet``.
    """
    if episode < 0:
        raise ValueError(f"--episode must be >= 0; got {episode}")
    chunk_idx = episode // chunk_size
    return (
        dataset_root
        / "data"
        / f"chunk-{chunk_idx:03d}"
        / f"episode_{episode:06d}.parquet"
    )


def _load_and_validate_parquet(
    parquet_path: Path,
    *,
    num_body_dofs: int,
    num_hand_dof_per_side: int,
    require_omnihand: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Read the recorded parquet and validate its column shapes.

    Returns ``(body_q, left_hand_q, right_hand_q)``. The hand arrays
    are ``None`` when ``require_omnihand`` is False (the viewer just
    won't write hand DOFs).

    Raises:
        FileNotFoundError: if the parquet file is missing.
        ValueError: if any required column is missing or has wrong width.
    """
    import pyarrow.parquet as pq

    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Episode parquet not found at {parquet_path}. "
            "Was the dataset recorded with the LeRobot v2.1 layout?"
        )

    table = pq.read_table(parquet_path)

    body_col = next(
        (c for c in _BODY_ACTION_CANDIDATES if c in table.column_names),
        None,
    )
    if body_col is None:
        raise ValueError(
            f"Parquet {parquet_path} is missing the body action column. "
            f"Tried (in order): {_BODY_ACTION_CANDIDATES}. "
            f"Available: {table.column_names}"
        )
    missing = [c for c in _REQUIRED_HAND_COLUMNS if c not in table.column_names]
    if require_omnihand and missing:
        raise ValueError(
            f"Parquet {parquet_path} is missing required hand columns: {missing}. "
            f"Available: {table.column_names}"
        )

    def _stack(col: str) -> np.ndarray:
        return np.stack(table[col].to_numpy()).astype(np.float64)

    body_q = _stack(body_col)
    if body_q.ndim != 2 or body_q.shape[1] != num_body_dofs:
        raise ValueError(
            f"{body_col} has shape {body_q.shape}; "
            f"expected (num_frames, {num_body_dofs}) for the configured robot."
        )

    if not require_omnihand:
        return body_q, None, None

    left_q = _stack("action.left_hand_joints")
    right_q = _stack("action.right_hand_joints")
    if left_q.shape[1] != num_hand_dof_per_side:
        raise ValueError(
            f"action.left_hand_joints has width {left_q.shape[1]}; "
            f"expected {num_hand_dof_per_side} (cfg.num_hand_dof_per_side). "
            "Pass --no-omnihand if the dataset has no hand columns."
        )
    if right_q.shape[1] != num_hand_dof_per_side:
        raise ValueError(
            f"action.right_hand_joints has width {right_q.shape[1]}; "
            f"expected {num_hand_dof_per_side}."
        )
    # Per-frame row counts are guaranteed equal by parquet's table
    # invariant; we don't re-check here.
    return body_q, left_q, right_q


def _resolve_frame_window(
    num_frames: int,
    *,
    start_frame: int,
    end_frame: int,
) -> tuple[int, int]:
    """Resolve negative / out-of-range frame indices to a clamped ``[start, end)``.

    Raises:
        ValueError: if the resolved window is empty.
    """
    if num_frames <= 0:
        raise ValueError("Recorded episode has zero frames.")

    start = start_frame
    end = end_frame
    if end < 0:
        end = num_frames
    start = max(0, min(start, num_frames))
    end = max(0, min(end, num_frames))
    if start >= end:
        raise ValueError(
            f"Empty frame window: start={start_frame} end={end_frame} "
            f"resolves to [{start}, {end}) over {num_frames} frames."
        )
    return start, end


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name (resolved under data/lerobot/<name>) or an "
             "absolute / relative path to a LeRobot v2.1 dataset root. "
             "Required unless --parquet is set.",
    )
    p.add_argument(
        "--episode", type=int, default=None,
        help="Zero-indexed episode number. The CLI reads "
             "<dataset>/data/chunk-{ep//chunks_size:03d}/episode_{ep:06d}.parquet. "
             "Required unless --parquet is set.",
    )
    p.add_argument(
        "--parquet", type=Path, default=None,
        help="Direct path to a parquet file. Bypasses --dataset / --episode "
             "resolution; useful for replaying re-derived variants (e.g. "
             "episode_000000_hand_range_calibrated.parquet) without renaming "
             "the canonical file.",
    )
    p.add_argument(
        "--robot", type=str, default="x2",
        help="Embodiment key dispatched through "
             "gear_sonic.utils.embodiment.get_embodiment(). "
             "Default 'x2'. 'g1' is currently a stub that raises "
             "NotImplementedError at model-build time.",
    )
    p.add_argument(
        "--rate", type=float, default=50.0,
        help="Playback rate in Hz. Default matches the recorder's 50 Hz "
             "control rate.",
    )
    p.add_argument(
        "--start-frame", type=int, default=0,
        help="First frame to play (inclusive, zero-indexed).",
    )
    p.add_argument(
        "--end-frame", type=int, default=-1,
        help="Stop frame (exclusive). Use -1 for end-of-episode (default).",
    )
    p.add_argument(
        "--loop", action="store_true",
        help="Restart from --start-frame after reaching --end-frame.",
    )
    p.add_argument(
        "--with-omnihand", dest="with_omnihand", action="store_true",
        help="Load the X2 + OmniHand augmented MJCF and apply recorded "
             "hand joint commands. Default ON.",
    )
    p.add_argument(
        "--no-omnihand", dest="with_omnihand", action="store_false",
        help="Skip OmniHand augmentation. Hand columns in the parquet "
             "are ignored.",
    )
    p.set_defaults(with_omnihand=True)
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-second progress logs.",
    )
    return p.parse_args(argv)


def _run_viewer(
    *,
    cfg,
    body_q: np.ndarray,
    left_q: Optional[np.ndarray],
    right_q: Optional[np.ndarray],
    start: int,
    end: int,
    rate: float,
    loop: bool,
    quiet: bool,
) -> None:
    """Open a passive MuJoCo viewer and replay the recorded trajectory.

    Encapsulated as its own function so the test surface
    (:func:`_parse_args`, :func:`_load_and_validate_parquet`,
    :func:`_resolve_dataset_path`, etc.) stays viewer-free.
    """
    import mujoco
    import mujoco.viewer

    model, layout, body_qposadr = cfg.build_kinematic_model(
        with_omnihand=(left_q is not None and right_q is not None),
    )
    data = mujoco.MjData(model)

    apply_hand_fn = (
        cfg.apply_omnihand_fn
        if (layout is not None and left_q is not None and right_q is not None)
        else None
    )

    from gear_sonic.utils.teleop.x2_kinematic_view import set_kinematic_pose

    init_left = (
        left_q[start]
        if left_q is not None
        else np.zeros(cfg.num_hand_dof_per_side)
    )
    init_right = (
        right_q[start]
        if right_q is not None
        else np.zeros(cfg.num_hand_dof_per_side)
    )
    set_kinematic_pose(
        mujoco_mod=mujoco,
        model=model,
        data=data,
        body_q_mj=body_q[start],
        body_qposadr=body_qposadr,
        layout=layout,
        apply_hand_fn=apply_hand_fn,
        left_hand_q=init_left,
        right_hand_q=init_right,
        pelvis_pos_xyz=cfg.pelvis_pos_xyz,
        pelvis_quat_wxyz=cfg.pelvis_quat_wxyz,
    )

    period = 1.0 / max(rate, 1e-3)
    stop = {"flag": False}

    def _on_sigint(_sig, _frm):
        stop["flag"] = True

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            frame_idx = start
            wall_t0 = time.perf_counter()
            last_log_wall = wall_t0
            frames_since_log = 0
            while viewer.is_running() and not stop["flag"]:
                set_kinematic_pose(
                    mujoco_mod=mujoco,
                    model=model,
                    data=data,
                    body_q_mj=body_q[frame_idx],
                    body_qposadr=body_qposadr,
                    layout=layout,
                    apply_hand_fn=apply_hand_fn,
                    left_hand_q=(
                        left_q[frame_idx]
                        if left_q is not None
                        else np.zeros(cfg.num_hand_dof_per_side)
                    ),
                    right_hand_q=(
                        right_q[frame_idx]
                        if right_q is not None
                        else np.zeros(cfg.num_hand_dof_per_side)
                    ),
                    pelvis_pos_xyz=cfg.pelvis_pos_xyz,
                    pelvis_quat_wxyz=cfg.pelvis_quat_wxyz,
                )
                viewer.sync()

                frame_idx += 1
                frames_since_log += 1
                if frame_idx >= end:
                    if not loop:
                        break
                    frame_idx = start

                target_wall = wall_t0 + frames_since_log * period
                slack = target_wall - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)

                if not quiet:
                    now = time.perf_counter()
                    if now - last_log_wall >= 1.0:
                        elapsed_frames = frames_since_log
                        actual_hz = elapsed_frames / (now - last_log_wall)
                        print(
                            f"[replay-x2-kinematic] frame {frame_idx}/{end} "
                            f"({actual_hz:.1f} Hz)",
                            flush=True,
                        )
                        last_log_wall = now
                        frames_since_log = 0
                        wall_t0 = now
    finally:
        signal.signal(signal.SIGINT, prev_handler)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from gear_sonic.utils.embodiment import get_embodiment

    cfg = get_embodiment(args.robot)

    if args.parquet is not None:
        parquet_path = args.parquet.expanduser().resolve()
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"--parquet {parquet_path} does not exist."
            )
        if not args.quiet:
            print(
                f"[replay-x2-kinematic] parquet override={parquet_path} "
                f"robot={cfg.name} with_omnihand={args.with_omnihand} "
                f"rate={args.rate:.1f} Hz",
                flush=True,
            )
    else:
        if args.dataset is None or args.episode is None:
            raise SystemExit(
                "Error: --parquet, OR --dataset together with --episode, "
                "is required."
            )
        dataset_root = _resolve_dataset_path(args.dataset)
        chunk_size = _read_chunk_size(dataset_root)
        parquet_path = _episode_parquet_path(
            dataset_root, args.episode, chunk_size=chunk_size
        )
        if not args.quiet:
            print(
                f"[replay-x2-kinematic] dataset={dataset_root} "
                f"episode={args.episode} robot={cfg.name} "
                f"with_omnihand={args.with_omnihand} rate={args.rate:.1f} Hz",
                flush=True,
            )
            print(f"[replay-x2-kinematic] parquet={parquet_path}", flush=True)

    body_q, left_q, right_q = _load_and_validate_parquet(
        parquet_path,
        num_body_dofs=cfg.num_body_dofs,
        num_hand_dof_per_side=cfg.num_hand_dof_per_side,
        require_omnihand=args.with_omnihand,
    )

    start, end = _resolve_frame_window(
        body_q.shape[0],
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    if not args.quiet:
        print(
            f"[replay-x2-kinematic] frames=[{start}, {end}) "
            f"of {body_q.shape[0]} | loop={args.loop}",
            flush=True,
        )

    _run_viewer(
        cfg=cfg,
        body_q=body_q,
        left_q=left_q,
        right_q=right_q,
        start=start,
        end=end,
        rate=args.rate,
        loop=args.loop,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
