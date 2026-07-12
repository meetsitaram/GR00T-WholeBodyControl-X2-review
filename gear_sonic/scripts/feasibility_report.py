#!/usr/bin/env python3
"""Category-aware feasibility + tracking-quality report for the G1 ONNX sweep.

Combines Metric 1 (feasibility, from im_eval's metrics_eval.json) with Metric 2
(tracking deviation, from FK of the executed vs reference motions) into a single
per-clip report with a 4-way label, plus an aggregate distribution.

Labels (per feasible clip):
  CLEAN        pose tracked (upper & lower local MPJPE < --pose-mm) AND base stayed
               put (root drift < max(--drift-m, 2x reference root travel)). Prime data.
  BASE-MOBILE  pose tracked but the base drifted (e.g. stepped to balance a heavy
               manipulation). Keep the executed motion; flag as base-mobile.
  POOR-POSE    upright but couldn't reach the pose (high local MPJPE, e.g. a deep
               crouch). Keep as the achievable version; flag low-fidelity.
  INFEASIBLE   terminated (fell / diverged). Drop.

The gate is content-adaptive, not label-dependent: locomotion clips (reference
root travel large) are effectively judged on drift+stride; in-place clips
(manipulation / posture) tolerate drift up to the 0.5 m floor and are judged on
pose. The bones-seed category (via --cat-map) is carried through for grouping only.

Usage:
  python gear_sonic/scripts/feasibility_report.py \
      --ref-pkl  bones_sample.pkl \
      --exe-pkl  bones_executed.pkl \
      --metrics  bones_eval_out/metrics_eval.json \
      --mjcf     gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml \
      --cat-map  bones_sample_map.json \
      --out      bones_feasibility_report.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np

UPPER = ["torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
         "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]
LOWER = ["left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
         "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link"]
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
KEEP = {"CLEAN": "prime", "BASE-MOBILE": "keep+flag",
        "POOR-POSE": "keep(low-fi)", "INFEASIBLE": "drop"}


def _wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _resample(m, n):
    dof = np.asarray(m["dof"], np.float64)
    rr = np.asarray(m["root_rot"], np.float64)
    rp = np.asarray(m["root_trans_offset"], np.float64)
    k = len(dof)
    if k == n:
        return dof, rr, rp
    s, d = np.linspace(0, 1, k), np.linspace(0, 1, n)
    dof2 = np.stack([np.interp(d, s, dof[:, i]) for i in range(dof.shape[1])], 1)
    rp2 = np.stack([np.interp(d, s, rp[:, i]) for i in range(3)], 1)
    rr2 = np.stack([np.interp(d, s, rr[:, i]) for i in range(4)], 1)
    rr2 /= np.linalg.norm(rr2, axis=1, keepdims=True) + 1e-9
    return dof2, rr2, rp2


def fk(model, data, dof, rr, rp, body_ids):
    nd = model.nq - 7
    out = np.empty((len(dof), len(body_ids), 3))
    for t in range(len(dof)):
        data.qpos[:3] = rp[t]
        data.qpos[3:7] = _wxyz(rr[t])
        data.qpos[7 : 7 + nd] = dof[t, :nd]
        mujoco.mj_kinematics(model, data)
        for j, b in enumerate(body_ids):
            out[t, j] = data.xpos[b]
    return out


def local_mpjpe(P_e, P_r):  # P_*: (T, B+1, 3), last col = pelvis
    pe = P_e[:, :-1] - P_e[:, -1:]
    pr = P_r[:, :-1] - P_r[:, -1:]
    return np.linalg.norm(pe - pr, axis=-1).mean() * 1000.0


def horiz_travel(P):  # (T, B, 2/3) -> summed horizontal path over bodies
    return sum(np.linalg.norm(np.diff(P[:, i, :2], axis=0), axis=-1).sum() for i in range(P.shape[1]))


def label_clip(feasible, up, lo, drift, ref_travel, pose_mm, drift_m):
    if not feasible:
        return "INFEASIBLE"
    if up >= pose_mm or lo >= pose_mm:
        return "POOR-POSE"
    if drift >= max(drift_m, 2.0 * ref_travel):
        return "BASE-MOBILE"
    return "CLEAN"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-pkl", required=True)
    ap.add_argument("--exe-pkl", required=True)
    ap.add_argument("--metrics", required=True, help="im_eval metrics_eval.json")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--cat-map", default=None, help="json {clip: {category, bucket}}")
    ap.add_argument("--out", required=True, help="output per-clip CSV")
    ap.add_argument("--pose-mm", type=float, default=50.0)
    ap.add_argument("--drift-m", type=float, default=0.5)
    ap.add_argument("--target-fps", type=float, default=50.0, help="control/eval fps of executed dump")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    bid = lambda ns: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ns]
    up_ids, lo_ids, ft_ids = bid(UPPER), bid(LOWER), bid(FEET)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    ref = joblib.load(args.ref_pkl)
    exe = joblib.load(args.exe_pkl)
    md = json.load(open(args.metrics))["eval/all_metrics_dict"]
    term = {k: bool(t) for k, t in zip(md["motion_keys"], md["terminated"])}
    prog = {k: float(p) for k, p in zip(md["motion_keys"], md["progress"])}
    catmap = json.load(open(args.cat_map)) if args.cat_map else {}

    rows = []
    for k in ref:
        if k not in exe:
            continue
        ref_fps = float(ref[k].get("fps", 120.0))
        n_ref = len(np.asarray(ref[k]["dof"]))
        true_len = max(2, round(n_ref * args.target_fps / ref_fps))
        ed = np.asarray(exe[k]["dof"])
        tl = min(true_len, len(ed))
        e_dof, e_rr, e_rp = ed[:tl], np.asarray(exe[k]["root_rot"])[:tl], np.asarray(exe[k]["root_trans_offset"])[:tl]
        r_dof, r_rr, r_rp = _resample(ref[k], tl)

        Pe_up = fk(model, data, e_dof, e_rr, e_rp, up_ids + [pid])
        Pr_up = fk(model, data, r_dof, r_rr, r_rp, up_ids + [pid])
        Pe_lo = fk(model, data, e_dof, e_rr, e_rp, lo_ids + [pid])
        Pr_lo = fk(model, data, r_dof, r_rr, r_rp, lo_ids + [pid])
        up = local_mpjpe(Pe_up, Pr_up)
        lo = local_mpjpe(Pe_lo, Pr_lo)
        drift = float(np.linalg.norm(Pe_up[-1, -1, :2] - Pr_up[-1, -1, :2]))
        ref_travel = float(np.linalg.norm(r_rp[-1, :2] - r_rp[0, :2]))
        # stride only where the reference actually locomotes
        Pe_ft = fk(model, data, e_dof, e_rr, e_rp, ft_ids)
        Pr_ft = fk(model, data, r_dof, r_rr, r_rp, ft_ids)
        tr_ref, tr_exe = horiz_travel(Pr_ft), horiz_travel(Pe_ft)
        stride = tr_exe / tr_ref if (ref_travel > 0.5 and tr_ref > 1e-6) else float("nan")

        feasible = not term.get(k, True)
        lab = label_clip(feasible, up, lo, drift, ref_travel, args.pose_mm, args.drift_m)
        cat = catmap.get(k, {}).get("category", "")
        bucket = catmap.get(k, {}).get("bucket", "")
        rows.append({
            "clip": k, "category": cat, "bucket": bucket,
            "feasible": int(feasible), "progress": round(prog.get(k, 0.0), 3),
            "upper_local_mm": round(up, 1), "lower_local_mm": round(lo, 1),
            "root_drift_m": round(drift, 2), "ref_travel_m": round(ref_travel, 2),
            "stride_ratio": round(stride, 2) if stride == stride else "",
            "label": lab, "keep": KEEP[lab],
        })

    fields = ["clip", "category", "bucket", "feasible", "progress", "upper_local_mm",
              "lower_local_mm", "root_drift_m", "ref_travel_m", "stride_ratio", "label", "keep"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    n = len(rows)
    labels = collections.Counter(r["label"] for r in rows)
    print(f"\n{n} clips scored -> {args.out}")
    print("label distribution:")
    for lab in ("CLEAN", "BASE-MOBILE", "POOR-POSE", "INFEASIBLE"):
        c = labels.get(lab, 0)
        print(f"  {lab:<12} {c:>4}  ({100*c/n:>4.0f}%)   [{KEEP[lab]}]")
    feas = sum(r["feasible"] for r in rows)
    print(f"feasible (not terminated): {feas}/{n} ({100*feas/n:.0f}%)")
    if any(r["bucket"] for r in rows):
        print("\nby sample bucket:")
        for b in sorted({r["bucket"] for r in rows if r["bucket"]}):
            br = [r for r in rows if r["bucket"] == b]
            bl = collections.Counter(r["label"] for r in br)
            print(f"  {b:<6} n={len(br):<3} " + "  ".join(f"{k}={bl.get(k,0)}" for k in ("CLEAN","BASE-MOBILE","POOR-POSE","INFEASIBLE")))


if __name__ == "__main__":
    main()
