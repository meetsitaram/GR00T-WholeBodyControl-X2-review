#!/usr/bin/env python3
"""Quantify how far an EXECUTED motion deviates from its REFERENCE (input).

This is the success / tracking-quality metric (Metric 2) for the G1 feasibility
experiment: given the reference motion the policy was asked to track and the
executed motion the robot actually produced (dumped by the ONNX im_eval sweep,
see g1_onnx_policy_shim.py), it forward-kinematics both through the G1 MJCF and
reports standard motion-tracking deviation metrics.

Metrics (per clip + aggregate):
  MPJPE_g   mean per-body position error, GLOBAL (mm) -- includes root drift.
  MPJPE_l   mean per-body position error, ROOT-RELATIVE (mm) -- pure pose fidelity
            (each frame's pelvis translation removed before comparing).
  root_xy   mean horizontal root-path deviation (m) + final-frame value.
  root_z    mean |height| deviation (m).
  joint_mae mean absolute per-joint angle deviation (deg).
  stride    executed / reference horizontal foot travel (understep < 1.0).

Usage:
  python gear_sonic/scripts/motion_deviation.py \
      --ref  gear_sonic/data/motions/g1_parity_walk.pkl \
      --exe  gear_sonic/data/motions/g1_onnx_executed/g1_executed_walk.pkl \
      --mjcf gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np

# 14 tracked bodies (== im_eval_callback body_names); MPJPE is computed over the
# subset that actually exists in the MJCF (names fall back gracefully).
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _resample_motion(m, n_target):
    """Resample a motion (dof/root) to n_target frames over normalized time.

    Needed because the executed dump is at the 50 Hz control rate while a
    bones-seed reference may be 120 fps -- they cover the same clip duration but
    have different frame counts, so we align by normalized time before comparing.
    """
    dof = np.asarray(m["dof"], np.float64)
    rr = np.asarray(m["root_rot"], np.float64)
    rp = np.asarray(m["root_trans_offset"], np.float64)
    n = len(dof)
    if n == n_target:
        return m
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, n_target)
    dof2 = np.stack([np.interp(dst, src, dof[:, i]) for i in range(dof.shape[1])], 1)
    rp2 = np.stack([np.interp(dst, src, rp[:, i]) for i in range(3)], 1)
    rr2 = np.stack([np.interp(dst, src, rr[:, i]) for i in range(4)], 1)
    rr2 /= np.linalg.norm(rr2, axis=1, keepdims=True) + 1e-9
    return {"dof": dof2, "root_rot": rr2, "root_trans_offset": rp2, "fps": m.get("fps")}


def fk_body_positions(model, data, motion, body_ids):
    """FK a motion -> (T, n_bodies, 3) world body positions."""
    dof = np.asarray(motion["dof"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)  # xyzw
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    n = dof.shape[0]
    nd = model.nq - 7
    out = np.empty((n, len(body_ids), 3), dtype=np.float64)
    for t in range(n):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = _wxyz(root_rot[t])
        data.qpos[7 : 7 + nd] = dof[t, :nd]
        data.qvel[:] = 0.0
        mujoco.mj_kinematics(model, data)
        for j, bid in enumerate(body_ids):
            out[t, j] = data.xpos[bid]
    return out, dof


def clip_metrics(model, ref, exe):
    data = mujoco.MjData(model)
    body_ids, body_names = [], []
    for nm in TRACKED_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            body_ids.append(bid)
            body_names.append(nm)
    pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
    foot_idx = [body_names.index(f) for f in FOOT_BODIES if f in body_names]

    # time-align: resample the reference to the executed frame count (control rate)
    ref = _resample_motion(ref, len(np.asarray(exe["dof"])))

    p_ref, dof_ref = fk_body_positions(model, data, ref, body_ids)
    p_exe, dof_exe = fk_body_positions(model, data, exe, body_ids)
    T = min(len(p_ref), len(p_exe))
    p_ref, p_exe, dof_ref, dof_exe = p_ref[:T], p_exe[:T], dof_ref[:T], dof_exe[:T]

    # global MPJPE (mm)
    d_g = np.linalg.norm(p_exe - p_ref, axis=-1)  # (T, B)
    mpjpe_g = d_g.mean() * 1000.0
    # root-relative MPJPE (remove pelvis translation each frame)
    pr_ref = p_ref - p_ref[:, pelvis_idx : pelvis_idx + 1, :]
    pr_exe = p_exe - p_exe[:, pelvis_idx : pelvis_idx + 1, :]
    mpjpe_l = np.linalg.norm(pr_exe - pr_ref, axis=-1).mean() * 1000.0
    # per-body worst
    per_body = d_g.mean(axis=0) * 1000.0

    # root path deviation
    root_ref = p_ref[:, pelvis_idx, :]
    root_exe = p_exe[:, pelvis_idx, :]
    root_xy = np.linalg.norm(root_exe[:, :2] - root_ref[:, :2], axis=-1)
    root_z = np.abs(root_exe[:, 2] - root_ref[:, 2])

    # joint MAE (deg), wrapped
    jerr = np.abs(((np.rad2deg(dof_exe - dof_ref) + 180) % 360) - 180)
    joint_mae = jerr.mean()
    per_joint = jerr.mean(axis=0)

    # stride ratio (horizontal foot travel exe/ref)
    def foot_travel(p):
        if not foot_idx:
            return np.nan
        tot = 0.0
        for fi in foot_idx:
            step = np.linalg.norm(np.diff(p[:, fi, :2], axis=0), axis=-1)
            tot += step.sum()
        return tot
    tr_ref, tr_exe = foot_travel(p_ref), foot_travel(p_exe)
    stride = tr_exe / tr_ref if tr_ref and tr_ref > 1e-6 else np.nan

    return {
        "frames": T,
        "mpjpe_g_mm": mpjpe_g,
        "mpjpe_l_mm": mpjpe_l,
        "root_xy_mean_m": float(root_xy.mean()),
        "root_xy_final_m": float(root_xy[-1]),
        "root_z_mean_m": float(root_z.mean()),
        "joint_mae_deg": float(joint_mae),
        "stride_ratio": float(stride),
        "per_body_mm": dict(zip(body_names, np.round(per_body, 1))),
        "worst_joint_deg": float(per_joint.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="reference (input) motion PKL")
    ap.add_argument("--exe", required=True, help="executed motion PKL")
    ap.add_argument("--mjcf", required=True)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    ref_data = joblib.load(args.ref)
    exe_data = joblib.load(args.exe)
    keys = [k for k in ref_data if k in exe_data]
    if not keys:
        raise SystemExit("no shared motion keys between --ref and --exe")

    print(f"{'clip':<26}{'frames':>7}{'MPJPE_g':>9}{'MPJPE_l':>9}"
          f"{'root_xy':>9}{'root_z':>8}{'jMAE':>7}{'stride':>8}")
    print(f"{'':<26}{'':>7}{'(mm)':>9}{'(mm)':>9}{'(m)':>9}{'(m)':>8}{'(deg)':>7}{'':>8}")
    agg = []
    for k in keys:
        m = clip_metrics(model, ref_data[k], exe_data[k])
        agg.append(m)
        print(f"{k[:25]:<26}{m['frames']:>7}{m['mpjpe_g_mm']:>9.1f}{m['mpjpe_l_mm']:>9.1f}"
              f"{m['root_xy_final_m']:>9.2f}{m['root_z_mean_m']:>8.3f}"
              f"{m['joint_mae_deg']:>7.2f}{m['stride_ratio']:>8.2f}")
    def mean(key):
        return float(np.nanmean([a[key] for a in agg]))
    print("-" * 82)
    print(f"{'MEAN':<26}{'':>7}{mean('mpjpe_g_mm'):>9.1f}{mean('mpjpe_l_mm'):>9.1f}"
          f"{mean('root_xy_final_m'):>9.2f}{mean('root_z_mean_m'):>8.3f}"
          f"{mean('joint_mae_deg'):>7.2f}{mean('stride_ratio'):>8.2f}")
    # worst bodies aggregated
    allbodies = {}
    for a in agg:
        for b, v in a["per_body_mm"].items():
            allbodies.setdefault(b, []).append(v)
    worst = sorted(((b, float(np.mean(v))) for b, v in allbodies.items()), key=lambda x: -x[1])
    print("\nworst tracked bodies (mean global err, mm):")
    for b, v in worst[:6]:
        print(f"  {b:<26}{v:>7.1f}")


if __name__ == "__main__":
    main()
