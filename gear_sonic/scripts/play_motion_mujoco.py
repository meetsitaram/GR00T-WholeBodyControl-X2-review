#!/usr/bin/env python3
"""Kinematic playback of a motion-lib PKL in the MuJoCo viewer.

No physics, no policy: at each frame we write the recorded ``root_trans_offset``,
``root_rot``, and per-joint ``dof`` straight into ``mj_data.qpos``, call
``mj_kinematics`` to refresh body transforms, and sync the viewer. This is
the cleanest way to *see* what a reference motion actually contains
(useful for sanity-checking that a "standing" clip really stands still and
that joint ordering / quaternion convention are right).

Usage:
    python gear_sonic/scripts/play_motion_mujoco.py \
        --motion gear_sonic/data/motions/x2_ultra_idle_stand.pkl

Optional:
    --mjcf PATH        Override MJCF (defaults to gear_sonic x2_ultra.xml)
    --speed 1.0        Playback speed multiplier (e.g. 0.25 for slow-mo)
    --loop / --no-loop Loop at end (default: loop)
    --start-frame N    Start from frame N (default 0)

Controls:
    SPACE - viewer pause/resume (the script keeps stepping; you just freeze
            the camera ergonomics provided by mujoco.viewer).
    Esc   - close the viewer / exit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MJCF = str(
    GEAR_SONIC_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
)

NUM_DOFS = 31


def _take_first_motion(pkl_data: dict):
    if not isinstance(pkl_data, dict) or not pkl_data:
        raise ValueError("PKL is not a non-empty dict-of-motions")
    name = next(iter(pkl_data))
    return name, pkl_data[name]


def _validate(motion: dict) -> None:
    for key in ("dof", "root_rot", "root_trans_offset", "fps"):
        if key not in motion:
            raise KeyError(
                f"motion missing required key {key!r}; "
                f"present: {sorted(motion.keys())}"
            )
    dof = np.asarray(motion["dof"])
    if dof.ndim != 2 or dof.shape[1] != NUM_DOFS:
        raise ValueError(f"dof must be (T, {NUM_DOFS}); got {dof.shape}")


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
    )


def play(
    motion_path: Path,
    mjcf_path: Path,
    speed: float = 1.0,
    loop: bool = True,
    start_frame: int = 0,
) -> int:
    print(f"[play_motion] loading {motion_path} ...", flush=True)
    data = joblib.load(motion_path)
    name, motion = _take_first_motion(data)
    _validate(motion)

    dof = np.asarray(motion["dof"], dtype=np.float64)
    root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    fps = float(motion["fps"])
    n_frames = int(dof.shape[0])

    print(
        f"[play_motion] motion '{name}': {n_frames} frames @ {fps:.2f} fps "
        f"({n_frames / fps:.2f} s)",
        flush=True,
    )
    print(f"[play_motion] loading MJCF {mjcf_path}", flush=True)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if model.nq < 7 + NUM_DOFS:
        print(
            f"[play_motion] ERROR: MJCF nq={model.nq} but expected at least "
            f"{7 + NUM_DOFS} (free root + {NUM_DOFS} DOF)",
            file=sys.stderr,
        )
        return 1

    data_mj = mujoco.MjData(model)
    frame_idx = max(0, min(start_frame, n_frames - 1))
    print(
        "[play_motion] opening viewer; press Esc in the window to exit.",
        flush=True,
    )

    target_dt = 1.0 / max(1e-6, fps * speed)
    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        while viewer.is_running():
            t_start = time.perf_counter()

            data_mj.qpos[:3] = root_pos[frame_idx]
            data_mj.qpos[3:7] = _xyzw_to_wxyz(root_quat_xyzw[frame_idx])
            data_mj.qpos[7 : 7 + NUM_DOFS] = dof[frame_idx]
            data_mj.qvel[:] = 0.0
            mujoco.mj_kinematics(model, data_mj)
            mujoco.mj_comPos(model, data_mj)
            viewer.sync()

            frame_idx += 1
            if frame_idx >= n_frames:
                if loop:
                    frame_idx = 0
                else:
                    print("[play_motion] reached end (no-loop); exiting.")
                    break

            elapsed = time.perf_counter() - t_start
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    print("[play_motion] viewer closed; exiting.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True, type=Path,
                   help="Motion-lib .pkl path.")
    p.add_argument("--mjcf", type=Path, default=Path(DEFAULT_MJCF),
                   help=f"MJCF to load (default: {DEFAULT_MJCF})")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0).")
    p.add_argument("--no-loop", dest="loop", action="store_false",
                   help="Stop at the end instead of looping.")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Start from this frame (default 0).")
    args = p.parse_args(argv)
    return play(
        motion_path=args.motion,
        mjcf_path=args.mjcf,
        speed=args.speed,
        loop=args.loop,
        start_frame=args.start_frame,
    )


if __name__ == "__main__":
    raise SystemExit(main())
