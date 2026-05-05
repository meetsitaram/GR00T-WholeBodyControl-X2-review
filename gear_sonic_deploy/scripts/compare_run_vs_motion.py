#!/usr/bin/env python3
"""Per-frame deviation: recorded run state vs motion-reference trajectory.

Loads a deploy ``run.npz`` and the kinematic motion PKL the policy was
tracking, aligns them on a common 30 Hz grid, and reports where (and on
which joints) the robot drifted away from the reference motion.

Outputs:
- ``summary.txt`` / ``summary.json`` -- per-DoF L2 deviation, worst frames
- ``deviation_heatmap.png`` -- frame x DoF abs-deviation, deg
- ``deviation_per_dof_bar.png`` -- per-DoF L2 deviation, deg
- ``deviation_top_frames.png`` -- whole-body deviation over time, with
  worst frames highlighted (with seconds + frame index annotations)
- ``worst_dof_traces.png`` -- ref vs measured for the 6 worst DoFs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# MJCF actuator order is the canonical 31-DoF order used by motion PKLs.
MJCF_ACTUATOR_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
]


def _detect_control_window(raw: dict) -> tuple[float, float]:
    """Find longest contiguous block where leg knee_kp ~ policy median."""
    t_cmd = raw["t_cmd_leg"].astype(np.float64)
    kp = raw["cmd_kp_leg"].astype(np.float64)
    n = t_cmd.size
    knee_kp = kp[:, 3]
    mid_lo, mid_hi = n // 4, max(n // 4 + 1, 3 * n // 4)
    pol_kp = float(np.median(knee_kp[mid_lo:mid_hi]))
    mask = np.abs(knee_kp - pol_kp) < 5.0
    if not mask.any():
        return float(t_cmd[0]), float(t_cmd[-1])
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]: starts = np.concatenate(([0], starts))
    if mask[-1]: ends = np.concatenate((ends, [n]))
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return float(t_cmd[starts[best]]), float(t_cmd[ends[best] - 1])


def _gather_state_in_mjcf_order(raw: dict, t_grid: np.ndarray) -> np.ndarray:
    """Resample state_pos for all four limb groups onto t_grid, reorder to MJCF order. Returns (T, 31) in radians."""
    out = np.zeros((t_grid.size, 31), dtype=np.float64)
    name_to_col = {n: i for i, n in enumerate(MJCF_ACTUATOR_ORDER)}
    for grp in ("leg", "waist", "arm", "head"):
        names = [str(x) for x in raw[f"joint_names_{grp}"]]
        t = raw[f"t_state_{grp}"].astype(np.float64)
        v = raw[f"state_pos_{grp}"].astype(np.float64)
        for j, jname in enumerate(names):
            out[:, name_to_col[jname]] = np.interp(t_grid, t, v[:, j])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Path to run.npz")
    ap.add_argument("--motion", required=True, help="Path to motion PKL")
    ap.add_argument("--motion-key", default=None,
                    help="Motion key (default: first entry)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--label", default="real",
                    help="Label for the run (used in plot titles)")
    ap.add_argument("--top-frames", type=int, default=12,
                    help="How many worst frames to report")
    ap.add_argument("--worst-dofs-plot", type=int, default=6,
                    help="How many worst DoFs to plot ref vs measured")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = dict(np.load(args.run, allow_pickle=True))
    motions = joblib.load(args.motion)
    keys = list(motions.keys())
    motion_key = args.motion_key or keys[0]
    if motion_key not in motions:
        cand = [k for k in keys if args.motion_key and args.motion_key in k]
        if len(cand) == 1:
            motion_key = cand[0]
        else:
            raise SystemExit(f"motion-key '{args.motion_key}' not in {keys[:6]}...")
    M = motions[motion_key]
    fps = float(M["fps"])
    ref_dof = M["dof"].astype(np.float64)  # (n_frames, 31)
    n_motion = ref_dof.shape[0]
    motion_dur = n_motion / fps

    cs, ce = _detect_control_window(raw)
    ctrl_dur = ce - cs
    use_dur = min(ctrl_dur, motion_dur)
    t_grid = cs + np.arange(int(np.floor(use_dur * fps))) / fps

    state = _gather_state_in_mjcf_order(raw, t_grid)
    ref = ref_dof[: t_grid.size]

    dev_rad = state - ref
    dev_deg = np.degrees(dev_rad)
    abs_dev_deg = np.abs(dev_deg)

    per_dof_l2 = np.sqrt(np.mean(dev_deg ** 2, axis=0))
    whole_body_deg_per_frame = np.linalg.norm(dev_deg, axis=1) / np.sqrt(31)

    order = np.argsort(-per_dof_l2)
    top_frames_idx = np.argsort(-whole_body_deg_per_frame)[: args.top_frames]
    top_frames_idx = np.sort(top_frames_idx)
    rel_t = (t_grid - cs)

    print(f"motion : {motion_key} ({motion_dur:.2f}s, {n_motion} frames @ {fps} Hz)")
    print(f"run    : {args.run}")
    print(f"CONTROL window: {cs:.2f}..{ce:.2f}s ({ctrl_dur:.2f}s)")
    print(f"compared frames: {t_grid.size} ({use_dur:.2f}s)")
    print()
    print("Per-DoF L2 deviation (state - reference), deg, sorted:")
    for j in order:
        print(f"  {MJCF_ACTUATOR_ORDER[j]:32s} {per_dof_l2[j]:7.2f}")
    print()
    total_l2 = float(np.sqrt(np.mean(dev_deg ** 2)))
    print(f"Total deviation L2 across all DoFs and frames: {total_l2:.2f} deg")
    print()
    print(f"Worst {args.top_frames} frames by whole-body deviation:")
    print(f"  {'idx':>5}  {'rel_t (s)':>10}  {'whole_body (deg)':>17}  worst-DoF (deg)")
    for fi in top_frames_idx:
        wj = int(np.argmax(abs_dev_deg[fi]))
        print(f"  {fi:5d}  {rel_t[fi]:10.2f}  {whole_body_deg_per_frame[fi]:17.2f}  {MJCF_ACTUATOR_ORDER[wj]} ({dev_deg[fi, wj]:+.1f})")

    summary = {
        "motion_key": motion_key,
        "motion_fps": fps,
        "motion_duration_s": motion_dur,
        "run_path": args.run,
        "control_window": [cs, ce],
        "compared_frames": int(t_grid.size),
        "compared_seconds": float(use_dur),
        "joint_names": MJCF_ACTUATOR_ORDER,
        "per_dof_l2_deg": per_dof_l2.tolist(),
        "total_rms_deg": total_l2,
        "worst_frames": [
            {
                "index": int(fi),
                "rel_t_s": float(rel_t[fi]),
                "whole_body_dev_deg": float(whole_body_deg_per_frame[fi]),
                "worst_dof": MJCF_ACTUATOR_ORDER[int(np.argmax(abs_dev_deg[fi]))],
                "worst_dof_value_deg": float(dev_deg[fi, int(np.argmax(abs_dev_deg[fi]))]),
            }
            for fi in top_frames_idx
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    txt = out_dir / "summary.txt"
    with txt.open("w") as f:
        f.write(f"motion: {motion_key}  ({motion_dur:.2f}s, {n_motion} frames @ {fps} Hz)\n")
        f.write(f"run   : {args.run}\n")
        f.write(f"label : {args.label}\n")
        f.write(f"CONTROL window: {cs:.2f}..{ce:.2f}s ({ctrl_dur:.2f}s)\n")
        f.write(f"compared {t_grid.size} frames ({use_dur:.2f}s)\n\n")
        f.write("per-DoF L2 deviation (deg), sorted by largest:\n")
        for j in order:
            f.write(f"  {MJCF_ACTUATOR_ORDER[j]:32s} {per_dof_l2[j]:7.2f}\n")
        f.write(f"\ntotal RMS across all 31 DoFs and {t_grid.size} frames: {total_l2:.2f} deg\n\n")
        f.write(f"worst {args.top_frames} frames (whole-body L2 deg):\n")
        for fi in top_frames_idx:
            wj = int(np.argmax(abs_dev_deg[fi]))
            f.write(f"  frame {int(fi):5d}  t={rel_t[fi]:6.2f}s  body_dev={whole_body_deg_per_frame[fi]:6.2f} deg  worst={MJCF_ACTUATOR_ORDER[wj]} ({dev_deg[fi,wj]:+.1f})\n")

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(abs_dev_deg.T, aspect="auto", origin="lower",
                   extent=[rel_t[0], rel_t[-1], -0.5, 30.5],
                   cmap="hot", vmin=0, vmax=min(60.0, float(np.percentile(abs_dev_deg, 99))))
    ax.set_yticks(np.arange(31))
    ax.set_yticklabels([n.replace("_joint", "") for n in MJCF_ACTUATOR_ORDER], fontsize=7)
    ax.set_xlabel("time since CONTROL start (s)")
    ax.set_title(f"Deviation |state - reference| (deg) -- {args.label} vs {motion_key}")
    fig.colorbar(im, ax=ax, label="abs deviation (deg)")
    for fi in top_frames_idx:
        ax.axvline(rel_t[fi], color="cyan", lw=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "deviation_heatmap.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(np.arange(31), per_dof_l2[order], color="tab:red", alpha=0.8)
    ax.set_yticks(np.arange(31))
    ax.set_yticklabels([MJCF_ACTUATOR_ORDER[j].replace("_joint", "") for j in order], fontsize=8)
    ax.set_xlabel("L2 deviation (deg)")
    ax.set_title(f"Per-DoF deviation -- {args.label} vs {motion_key}")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "deviation_per_dof_bar.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rel_t, whole_body_deg_per_frame, color="tab:blue", lw=0.8)
    ax.scatter(rel_t[top_frames_idx], whole_body_deg_per_frame[top_frames_idx],
               color="red", zorder=5, s=40, label=f"top-{args.top_frames} worst frames")
    for fi in top_frames_idx:
        ax.annotate(f"f{int(fi)}\n{rel_t[fi]:.1f}s",
                    (rel_t[fi], whole_body_deg_per_frame[fi]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")
    ax.set_xlabel("time since CONTROL start (s)")
    ax.set_ylabel("whole-body deviation (deg, RMS over 31 DoF)")
    ax.set_title(f"Whole-body deviation over time -- {args.label} vs {motion_key}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "deviation_top_frames.png", dpi=130)
    plt.close(fig)

    n_show = min(args.worst_dofs_plot, 31)
    worst = order[:n_show]
    rows = (n_show + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 2.5 * rows), sharex=True)
    axes = np.atleast_2d(axes).reshape(-1)
    for k, j in enumerate(worst):
        a = axes[k]
        a.plot(rel_t, np.degrees(ref[:, j]), color="black", lw=1.0, label="reference")
        a.plot(rel_t, np.degrees(state[:, j]), color="tab:red", lw=0.9, label="measured")
        a.set_title(f"{MJCF_ACTUATOR_ORDER[j]}  L2={per_dof_l2[j]:.1f} deg")
        a.grid(alpha=0.3)
        if k == 0:
            a.legend(loc="upper right", fontsize=8)
    for k in range(len(worst), len(axes)):
        axes[k].axis("off")
    axes[-2].set_xlabel("time since CONTROL start (s)")
    axes[-1].set_xlabel("time since CONTROL start (s)")
    fig.suptitle(f"{n_show} worst DoFs: reference vs measured -- {args.label}")
    fig.tight_layout()
    fig.savefig(out_dir / "worst_dof_traces.png", dpi=130)
    plt.close(fig)

    print(f"\n[done] wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
