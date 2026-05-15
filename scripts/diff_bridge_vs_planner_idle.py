"""Byte-diff bridge _IdleStandLoop output vs the raw idle_stand PKL frames.

After the bridge fix on 2026-05-14 the bridge was supposed to be
byte-equivalent to the planner on the wire. The fact that
--vla-no-policy still leans ~28 deg while --planner-only stands at
grav_z=-1.00 means we're still emitting something different.

Step 1 of the diff: confirm bridge.current(t) matches the raw idle_stand
clip frames at the SAME index. If yes, the bridge's content is correct
and the divergence is downstream (recorder forward, deploy decode, or
proprioception). If no, the bridge's yaw alignment / resampling drifted
the frame content from the planner's reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PRIMITIVES_PKL = (
    REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_primitives.pkl"
)


def main() -> int:
    from gear_sonic.scripts.live_vla_publish_motion_token import (
        _load_idle_stand_loop,
        _NUM_FUTURE_SLOTS,
        _FUTURE_STEP_TICKS,
    )
    from gear_sonic.utils.planner.blending import (
        resample_motion_30_to_50hz,
        yaw_align_segment,
    )

    print(f"loading idle_stand from {PRIMITIVES_PKL}")
    raw = joblib.load(PRIMITIVES_PKL)
    idle = raw.get("idle_stand")
    if idle is None:
        print("  ERROR: no 'idle_stand' bin")
        return 1
    src_dof = np.asarray(idle["dof"], dtype=np.float32)
    src_quat = np.asarray(idle["root_rot_xyzw"], dtype=np.float32)
    src_trans = np.asarray(idle["root_trans"], dtype=np.float64)
    src_fps = float(idle["fps"])
    print(
        f"  PKL idle_stand: dof shape={src_dof.shape} fps={src_fps} "
        f"trans shape={src_trans.shape}"
    )

    if abs(src_fps - 50.0) < 0.5:
        rs_dof, rs_quat, rs_trans = src_dof, src_quat, src_trans
        print("  no resampling needed (PKL already @ 50 Hz)")
    else:
        rs_dof, rs_quat, rs_trans = resample_motion_30_to_50hz(
            src_dof, src_quat, src_trans, src_fps, 50.0
        )
        print(
            f"  resampled to 50 Hz: dof shape={rs_dof.shape}"
        )

    aligned_dof, aligned_quat, _ = yaw_align_segment(
        rs_dof, rs_quat, rs_trans, np.array([0.0, 0.0]), 0.0
    )
    print(
        f"  yaw-aligned to robot_yaw=0: dof shape={aligned_dof.shape}"
    )
    print(f"  aligned[0] dof[:5]={aligned_dof[0, :5]}")
    print(f"  aligned[0] quat   ={aligned_quat[0]}")

    bridge_loop = _load_idle_stand_loop(PRIMITIVES_PKL)
    print(
        f"  bridge loop: {bridge_loop.n_frames} frames, "
        f"future_step_ticks={_FUTURE_STEP_TICKS}, "
        f"future_slots={_NUM_FUTURE_SLOTS}"
    )

    bj0, bq0 = bridge_loop.current(0)
    print(f"  bridge[0]   dof[:5]={bj0[:5]}")
    print(f"  bridge[0]   quat   ={bq0}")

    if bridge_loop.n_frames != aligned_dof.shape[0]:
        print(
            f"\n  FRAME COUNT MISMATCH: bridge={bridge_loop.n_frames} "
            f"aligned={aligned_dof.shape[0]}"
        )
        return 2

    worst_dof = 0.0
    worst_quat = 0.0
    worst_dof_idx = -1
    worst_quat_idx = -1
    for t in range(bridge_loop.n_frames):
        bj, bq = bridge_loop.current(t)
        dof_d = float(np.abs(aligned_dof[t].astype(np.float32) - bj).max())
        quat_d = float(np.abs(aligned_quat[t].astype(np.float32) - bq).max())
        if dof_d > worst_dof:
            worst_dof = dof_d
            worst_dof_idx = t
        if quat_d > worst_quat:
            worst_quat = quat_d
            worst_quat_idx = t

    print(
        f"\n  bridge vs aligned over all {bridge_loop.n_frames} ticks:"
    )
    print(
        f"    worst dof delta:  {worst_dof:.6e} at tick={worst_dof_idx}"
    )
    print(
        f"    worst quat delta: {worst_quat:.6e} at tick={worst_quat_idx}"
    )
    if worst_dof < 1e-5 and worst_quat < 1e-5:
        print("    -> bridge IS byte-equivalent to the yaw-aligned clip")
    else:
        print("    -> bridge content DIFFERS from the planner's reference")

    return 0


if __name__ == "__main__":
    sys.exit(main())
