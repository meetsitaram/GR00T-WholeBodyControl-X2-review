"""Synthetic v1 calibration check — confirm the wrist-orientation path is wired.

We don't have a recaptured calibration with wrist quats yet, so this
script:
  1. Loads the legacy v0 YAML for position-fit data.
  2. Pulls a window of "arms-down" wrist quats from the recorded NPZ
     (frames where engaged=True and the operator's head yaw is near 0
     and the wrist is near the calibrated arms-down position).
  3. Builds a v1 calibration in-memory with non-identity wrist alignment
     quats derived from those samples.
  4. Replays the NPZ through the new solver with this v1 calibration to
     confirm:
       - the auto-disable does NOT fire (alignment is not identity)
       - the IK still converges (no exploding errors)
       - wrist joints actually move
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.operator_calibration import (
    OperatorCalibration,
    head_yaw_from_quat,
    robot_reference_wrist_quats,
    wrist_quat_to_head_yaw_frame,
    _quat_inverse_wxyz,
    _quat_multiply_wxyz,
)
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import VRArmTeleopCalibrated


def _build_vr_pose(d, i: int) -> np.ndarray:
    out = np.zeros((3, 7), dtype=np.float64)
    out[0, :3] = d["vr_left_wrist_pos"][i]
    out[0, 3:] = d["vr_left_wrist_quat"][i]
    out[1, :3] = d["vr_right_wrist_pos"][i]
    out[1, 3:] = d["vr_right_wrist_quat"][i]
    out[2, :3] = d["vr_head_pos"][i]
    out[2, 3:] = d["vr_head_quat"][i]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz", type=Path,
        default=Path("data/lerobot/x2_quest3_kinematic_v2/debug/teleop_episode_000000.npz"),
    )
    p.add_argument(
        "--calibration", type=Path,
        default=Path("data/operator_calibrations/default.yaml"),
    )
    args = p.parse_args()

    print(f"[v1-check] loading {args.npz}")
    d = np.load(args.npz, allow_pickle=True)
    n = int(d["num_frames"])

    print(f"[v1-check] loading {args.calibration}")
    cal = OperatorCalibration.load_yaml(args.calibration)

    # Synthesise non-identity wrist alignment from a window of "calm"
    # frames. We pick frames where the operator's wrist is close to the
    # calibrated arms-down position (in head-yaw frame) and where they
    # are not in a dropout. Then take the median wrist quaternion in
    # the head-yaw frame and pretend that's the operator's "arms-down
    # neutral wrist orientation". The alignment is then
    #     q_align = q_robot_arms_down * inverse(q_op_arms_down_in_head_yaw).
    arms_down_l = cal.measurements["arms_down"].left_wrist_mean
    arms_down_r = cal.measurements["arms_down"].right_wrist_mean

    op_l_hy = d["op_left_wrist_head_yaw"].astype(np.float64)
    op_r_hy = d["op_right_wrist_head_yaw"].astype(np.float64)
    eng = d["engaged"].astype(bool)

    # Filter: engaged + close to calibrated arms-down + neither side a
    # dropout (raw pos norm > 0.1 m, raw quat not identity).
    raw_l_pos_norm = np.linalg.norm(d["vr_left_wrist_pos"], axis=1)
    raw_r_pos_norm = np.linalg.norm(d["vr_right_wrist_pos"], axis=1)
    raw_l_q_diff = np.linalg.norm(
        d["vr_left_wrist_quat"].astype(np.float64) - np.array([1, 0, 0, 0]),
        axis=1,
    )
    raw_r_q_diff = np.linalg.norm(
        d["vr_right_wrist_quat"].astype(np.float64) - np.array([1, 0, 0, 0]),
        axis=1,
    )
    not_drop = (
        (raw_l_pos_norm > 0.1) & (raw_r_pos_norm > 0.1)
        & (raw_l_q_diff > 0.05) & (raw_r_q_diff > 0.05)
    )

    near_left = np.linalg.norm(op_l_hy - arms_down_l, axis=1) < 0.10
    near_right = np.linalg.norm(op_r_hy - arms_down_r, axis=1) < 0.10
    pick = eng & not_drop & near_left & near_right
    print(f"[v1-check] arms-down window samples: {pick.sum()} / {n}")
    if pick.sum() < 50:
        print("[v1-check] not enough arms-down samples; widening to engaged+not-drop only")
        pick = eng & not_drop

    # Compute median operator wrist quat in head-yaw frame.
    op_l_quats = []
    op_r_quats = []
    for i in np.where(pick)[0]:
        op_l_quats.append(
            wrist_quat_to_head_yaw_frame(
                d["vr_left_wrist_quat"][i].astype(np.float64),
                d["vr_head_quat"][i].astype(np.float64),
            )
        )
        op_r_quats.append(
            wrist_quat_to_head_yaw_frame(
                d["vr_right_wrist_quat"][i].astype(np.float64),
                d["vr_head_quat"][i].astype(np.float64),
            )
        )
    op_l_arr = np.asarray(op_l_quats)
    op_r_arr = np.asarray(op_r_quats)
    # Antipodal align then component-median.
    for arr in (op_l_arr, op_r_arr):
        ref = arr[0]
        for i in range(1, arr.shape[0]):
            if np.dot(arr[i], ref) < 0:
                arr[i] = -arr[i]
    op_l_med = np.median(op_l_arr, axis=0)
    op_r_med = np.median(op_r_arr, axis=0)
    op_l_med /= np.linalg.norm(op_l_med)
    op_r_med /= np.linalg.norm(op_r_med)
    print(f"[v1-check] op_arms_down_quat L: {op_l_med}")
    print(f"[v1-check] op_arms_down_quat R: {op_r_med}")

    # Build alignment.
    qref = robot_reference_wrist_quats()
    align_l = _quat_multiply_wxyz(qref["arms_down"]["left"], _quat_inverse_wxyz(op_l_med))
    align_r = _quat_multiply_wxyz(qref["arms_down"]["right"], _quat_inverse_wxyz(op_r_med))
    print(f"[v1-check] derived align L: {align_l}")
    print(f"[v1-check] derived align R: {align_r}")

    # Patch the calibration in-memory with the derived alignments.
    cal.fit["left"].wrist_alignment_quat = align_l
    cal.fit["right"].wrist_alignment_quat = align_r
    cal.measurements["arms_down"].left_wrist_quat_head_yaw = op_l_med
    cal.measurements["arms_down"].right_wrist_quat_head_yaw = op_r_med

    print("\n[v1-check] === replaying with v1+ calibration (rotation_weight=0.3) ===")
    teleop = VRArmTeleopCalibrated(
        calibration=cal,
        rotation_weight=0.3,
        null_space_gain=0.10,
    )
    print(f"[v1-check] teleop._rotation_weight = {teleop._rotation_weight} "
          f"(expected 0.3 if alignment is non-identity)")

    new_left_q = np.zeros((n, 7), dtype=np.float64)
    new_right_q = np.zeros((n, 7), dtype=np.float64)
    new_left_pos_err = np.zeros(n, dtype=np.float64)
    new_right_pos_err = np.zeros(n, dtype=np.float64)
    new_left_rot_err = np.zeros(n, dtype=np.float64)
    new_right_rot_err = np.zeros(n, dtype=np.float64)

    engaged_state = False
    for i in range(n):
        if eng[i] != engaged_state:
            teleop.set_engaged(bool(eng[i]))
            engaged_state = bool(eng[i])
        vr = _build_vr_pose(d, i)
        res = teleop.step(vr)
        new_left_q[i] = res.left_q
        new_right_q[i] = res.right_q
        new_left_pos_err[i] = res.left_ik.pos_err_m
        new_right_pos_err[i] = res.right_ik.pos_err_m
        new_left_rot_err[i] = res.left_ik.rot_err_rad
        new_right_rot_err[i] = res.right_ik.rot_err_rad

    print("\n[v1-check] IK position errors (engaged):")
    print(f"  L: median={np.median(new_left_pos_err[eng])*100:.2f} cm  "
          f"p95={np.percentile(new_left_pos_err[eng], 95)*100:.2f} cm  "
          f"max={new_left_pos_err[eng].max()*100:.2f} cm")
    print(f"  R: median={np.median(new_right_pos_err[eng])*100:.2f} cm  "
          f"p95={np.percentile(new_right_pos_err[eng], 95)*100:.2f} cm  "
          f"max={new_right_pos_err[eng].max()*100:.2f} cm")

    print("\n[v1-check] IK rotation errors (engaged):")
    print(f"  L: median={np.median(new_left_rot_err[eng]):.3f} rad  "
          f"p95={np.percentile(new_left_rot_err[eng], 95):.3f} rad")
    print(f"  R: median={np.median(new_right_rot_err[eng]):.3f} rad  "
          f"p95={np.percentile(new_right_rot_err[eng], 95):.3f} rad")

    print("\n[v1-check] Wrist joint motion (std, rad) on engaged:")
    for joint, name in ((4, "wrist_yaw"), (5, "wrist_pitch"), (6, "wrist_roll")):
        l_std = new_left_q[eng, joint].std()
        r_std = new_right_q[eng, joint].std()
        print(f"  {name:<12s} L_std={l_std:.3f}  R_std={r_std:.3f}")

    print("\n[v1-check] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
