#!/usr/bin/env python3
"""Compare a real-robot recorder npz against a sim-mode recorder npz.

Both are produced by ``deploy_x2.sh --record`` (the real one with mode ``local``
or ``onbot``, the sim one with mode ``sim``). The script:

  1. Detects the CONTROL window in each run (first published joint command
     to last, optionally trimming a few seconds of RAMP_OUT / HOLD_FOR_MC
     at the tail).
  2. Resamples both runs onto a common time grid (50 Hz default).
  3. Computes per-DoF tracking error (state - cmd), per-DoF L2 over the
     run, and the sim-vs-real cmd / state diff.
  4. Writes a summary JSON / TXT plus PNG figures highlighting the
     sim-to-real gap.

Usage:

    python gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py \\
        --real scratch/runs/x2_run_20260503_222045/run.npz \\
        --sim  /tmp/anchor_b_sim_<...>/run.npz \\
        --out  /tmp/sim_vs_real_anchor_b/

It also accepts ``--real real_a.npz --sim real_b.npz`` for real-vs-real
sanity checks (identical labels just say "A" and "B").

Outputs (under --out):

  summary.json            numeric stats (durations, per-DoF L2, L2 deltas)
  summary.txt             human-readable summary
  dof_pos_grid.png        31-DoF: cmd / state for both runs, time on x-axis
  dof_tracking_error.png  31-DoF: |state - cmd| traces
  dof_l2_bar.png          per-DoF L2 of tracking error, real vs sim
  imu_overlay.png         IMU roll/pitch/yaw + angular velocity overlay
  cmd_diff_heatmap.png    time x DoF heatmap of cmd_real - cmd_sim

The script does not require the two runs to have identical durations -- it
crops both to their detected CONTROL window and overlays in run-relative
time. If you want strict alignment (same playlist phase), pass
``--align-from-zero`` to anchor both runs at t=0 of CONTROL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib import is deferred so --help works without the GUI backend.

# X2 Ultra DoF order: per-group concat in (arm, head, leg, waist) order.
GROUPS = ("arm", "head", "leg", "waist")
GROUP_DOFS = {"arm": 14, "head": 2, "leg": 12, "waist": 3}
TOTAL_DOFS = sum(GROUP_DOFS.values())  # = 31


# ---------------------------------------------------------------------------
# Loading + control-window detection
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """Resampled, control-window-cropped view of a recorder npz."""

    path: Path
    label: str
    meta: Dict
    joint_names: List[str]            # length 31, in (arm, head, leg, waist) order
    t: np.ndarray                     # (T,) seconds, run-relative starting at 0
    cmd_pos: np.ndarray               # (T, 31)
    state_pos: np.ndarray             # (T, 31)
    cmd_kp: np.ndarray                # (T, 31)
    cmd_kd: np.ndarray                # (T, 31)
    imu_quat_wxyz: np.ndarray         # (T, 4)
    imu_angvel: np.ndarray            # (T, 3)
    control_start_wall: float         # original wall-clock t of CONTROL start
    control_end_wall: float           # original wall-clock t of CONTROL end (post-trim)
    detected_handoff_at: Optional[float]  # wall-clock t of mc_mode handoff if known
    duration_s: float                 # control window duration


def _load_raw(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    return out


def _concat_group_array(raw: dict, prefix: str, t_per_group: Dict[str, np.ndarray]
                        ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Stack four limb-group arrays (arm/head/leg/waist) onto a common time grid.

    Each group has its own t_<prefix>_<group> stamps and possibly different
    sample counts, so we resample each group to the union grid before stacking.
    Returns (t_union, stacked, joint_names).
    """
    cols: List[Tuple[np.ndarray, np.ndarray, List[str]]] = []
    for grp in GROUPS:
        t = t_per_group[grp].astype(np.float64)
        a = raw[f"{prefix}_{grp}"].astype(np.float64)
        names = list(raw[f"joint_names_{grp}"])
        cols.append((t, a, names))
    # Build a union time grid as the densest of the four.
    t_union = max((t for t, _, _ in cols), key=lambda x: x.size)
    out = np.empty((t_union.size, TOTAL_DOFS), dtype=np.float64)
    names_out: List[str] = []
    j = 0
    for t, a, names in cols:
        # Per-DoF interpolation onto t_union.
        for col in range(a.shape[1]):
            out[:, j] = np.interp(t_union, t, a[:, col])
            j += 1
        names_out.extend(names)
    return t_union, out, names_out


def _detect_control_window(raw: dict,
                           start_skip: float,
                           end_trim: float
                           ) -> Tuple[float, float, Optional[float]]:
    """Return (control_start_wall, control_end_wall, detected_handoff_at).

    Newer recordings (post smooth-handoff fix) capture WAIT_FOR_CONTROL,
    CONTROL, RAMP_OUT, and HOLD_FOR_MC inside one npz, with t_cmd_leg
    spanning the whole thing. Older recordings (pre-fix) only have CONTROL
    and RAMP_OUT.

    We separate the regions using the leg-knee kp value:
      - During WAIT_FOR_CONTROL and HOLD_FOR_MC the deploy node mirrors
        MC's STAND_DEFAULT kp/kd schedule -- typically knee kp ~ 150,
        with varying values for hip / ankle (std-of-row ~ 40).
      - During CONTROL the policy publishes its own (often uniform-ish)
        kp -- typically knee kp ~ 99 for sphere-feet, values inherited
        from the tuning yaml.

    Strategy: take the median kp_knee across the middle 60% of the run as
    the "policy kp" estimate (the run is dominated by CONTROL by length).
    Mark every sample within +/- 5 of that median as in-control. The
    longest contiguous in-control block is CONTROL.

    For sim recordings (which never enter MC modes) this collapses to
    "everything is in-control" and the function returns
    [t_cmd_leg[0], t_cmd_leg[-1]].
    """
    t_cmd_leg = raw["t_cmd_leg"].astype(np.float64)
    kp_leg = raw["cmd_kp_leg"].astype(np.float64)
    if t_cmd_leg.size == 0:
        raise SystemExit("No commands recorded -- did the deploy node ever enter CONTROL?")

    n = t_cmd_leg.size
    # knee column is index 3 of the 12-DoF leg block
    knee_kp = kp_leg[:, 3]
    mid_lo = n // 4
    mid_hi = max(mid_lo + 1, 3 * n // 4)
    policy_knee_kp = float(np.median(knee_kp[mid_lo:mid_hi]))
    in_control_mask = np.abs(knee_kp - policy_knee_kp) < 5.0

    if not in_control_mask.any():
        # Should never happen if the run had any CONTROL ticks; fall back
        # to the entire cmd window.
        cs_idx, ce_idx = 0, n - 1
    else:
        # Run-length encode in_control_mask, take the longest True run.
        diff = np.diff(in_control_mask.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1  # exclusive index
        if in_control_mask[0]:
            starts = np.concatenate(([0], starts))
        if in_control_mask[-1]:
            ends = np.concatenate((ends, [n]))
        if starts.size == 0:
            cs_idx, ce_idx = 0, n - 1
        else:
            lengths = ends - starts
            best = int(np.argmax(lengths))
            cs_idx = int(starts[best])
            ce_idx = int(ends[best] - 1)

    control_start = float(t_cmd_leg[cs_idx]) + float(start_skip)
    control_end   = float(t_cmd_leg[ce_idx]) - float(end_trim)

    # Optional cross-check: mc_mode trace (if present, post-handoff runs).
    handoff_at: Optional[float] = None
    if "mc_mode_str" in raw and "t_mc_mode" in raw:
        modes = list(raw["mc_mode_str"])
        t_modes = raw["t_mc_mode"].astype(np.float64)
        in_control_mc = False
        for m, tt in zip(modes, t_modes):
            if "JOINT" in str(m) or "PASSIVE" in str(m):
                in_control_mc = True
            elif in_control_mc and "STAND_DEFAULT" in str(m):
                handoff_at = float(tt)
                break

    if control_end <= control_start:
        raise SystemExit(
            f"Control window collapsed (start={control_start:.3f}s, "
            f"end={control_end:.3f}s) -- try smaller --start-skip / --end-trim."
        )

    return control_start, control_end, handoff_at


def load_run(path: Path,
             label: str,
             *,
             start_skip: float,
             end_trim: float,
             resample_hz: float
             ) -> Run:
    raw = _load_raw(path)
    meta = json.loads(str(raw["meta_json"])) if "meta_json" in raw else {}

    # Per-group time arrays.
    t_cmd = {grp: raw[f"t_cmd_{grp}"].astype(np.float64) for grp in GROUPS}
    t_state = {grp: raw[f"t_state_{grp}"].astype(np.float64) for grp in GROUPS}

    # Concat 31-DoF arrays.
    t_cmd_union, cmd_pos_full, joint_names = _concat_group_array(raw, "cmd_pos", t_cmd)
    _,           cmd_kp_full,  _           = _concat_group_array(raw, "cmd_kp",  t_cmd)
    _,           cmd_kd_full,  _           = _concat_group_array(raw, "cmd_kd",  t_cmd)
    t_state_union, state_pos_full, _       = _concat_group_array(raw, "state_pos", t_state)

    # IMU.
    t_imu = raw["t_imu"].astype(np.float64)
    imu_quat = raw["imu_quat_wxyz"].astype(np.float64)
    imu_angvel = raw["imu_angvel"].astype(np.float64)

    # Detect control window.
    cs_wall, ce_wall, handoff_at = _detect_control_window(
        raw, start_skip=start_skip, end_trim=end_trim
    )

    # Build common resampled grid in run-relative time.
    duration = ce_wall - cs_wall
    n_samples = max(2, int(np.ceil(duration * resample_hz)))
    t_rel = np.linspace(0.0, duration, n_samples, dtype=np.float64)
    t_abs = cs_wall + t_rel

    def resample(t_src: np.ndarray, a: np.ndarray) -> np.ndarray:
        out = np.empty((n_samples, a.shape[1]), dtype=np.float64)
        for c in range(a.shape[1]):
            out[:, c] = np.interp(t_abs, t_src, a[:, c])
        return out

    cmd_pos   = resample(t_cmd_union,   cmd_pos_full)
    cmd_kp    = resample(t_cmd_union,   cmd_kp_full)
    cmd_kd    = resample(t_cmd_union,   cmd_kd_full)
    state_pos = resample(t_state_union, state_pos_full)
    imu_quat_r = resample(t_imu, imu_quat)
    imu_angvel_r = resample(t_imu, imu_angvel)

    # Re-normalise quaternions after interpolation (linear interp breaks unit norm).
    norms = np.linalg.norm(imu_quat_r, axis=1, keepdims=True)
    norms = np.where(norms > 1e-9, norms, 1.0)
    imu_quat_r = imu_quat_r / norms

    return Run(
        path=path,
        label=label,
        meta=meta,
        joint_names=joint_names,
        t=t_rel,
        cmd_pos=cmd_pos,
        state_pos=state_pos,
        cmd_kp=cmd_kp,
        cmd_kd=cmd_kd,
        imu_quat_wxyz=imu_quat_r,
        imu_angvel=imu_angvel_r,
        control_start_wall=cs_wall,
        control_end_wall=ce_wall,
        detected_handoff_at=handoff_at,
        duration_s=duration,
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler angles (roll, pitch, yaw) from wxyz quaternion."""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1)


# ---------------------------------------------------------------------------
# Stats + plotting
# ---------------------------------------------------------------------------


def compute_stats(real: Run, sim: Run) -> dict:
    """Per-DoF L2 tracking error for each run, plus sim-vs-real diffs."""
    # Resample sim onto real's time grid (or vice versa) so diffs make sense.
    if real.t.size != sim.t.size or not np.allclose(real.t, sim.t):
        # Reinterpolate sim onto real's grid.
        t_target = real.t
        sim_cmd_pos   = np.empty((t_target.size, TOTAL_DOFS))
        sim_state_pos = np.empty((t_target.size, TOTAL_DOFS))
        for c in range(TOTAL_DOFS):
            sim_cmd_pos[:, c]   = np.interp(t_target, sim.t, sim.cmd_pos[:, c])
            sim_state_pos[:, c] = np.interp(t_target, sim.t, sim.state_pos[:, c])
        sim_imu_angvel = np.empty((t_target.size, 3))
        for c in range(3):
            sim_imu_angvel[:, c] = np.interp(t_target, sim.t, sim.imu_angvel[:, c])
        sim_quat = np.empty((t_target.size, 4))
        for c in range(4):
            sim_quat[:, c] = np.interp(t_target, sim.t, sim.imu_quat_wxyz[:, c])
        norms = np.linalg.norm(sim_quat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        sim_quat = sim_quat / norms
    else:
        sim_cmd_pos = sim.cmd_pos
        sim_state_pos = sim.state_pos
        sim_imu_angvel = sim.imu_angvel
        sim_quat = sim.imu_quat_wxyz

    real_track_err = real.state_pos - real.cmd_pos
    sim_track_err  = sim_state_pos - sim_cmd_pos

    real_l2 = np.sqrt(np.mean(real_track_err ** 2, axis=0))   # (31,)
    sim_l2  = np.sqrt(np.mean(sim_track_err ** 2, axis=0))    # (31,)

    cmd_diff   = real.cmd_pos   - sim_cmd_pos                  # (T, 31)
    state_diff = real.state_pos - sim_state_pos                # (T, 31)

    real_rpy = quat_to_rpy(real.imu_quat_wxyz)
    sim_rpy  = quat_to_rpy(sim_quat)

    return {
        "duration_s_real": float(real.duration_s),
        "duration_s_sim":  float(sim.duration_s),
        "duration_s_compare": float(real.t[-1] - real.t[0]),
        "n_samples": int(real.t.size),
        "joint_names": real.joint_names,
        "real_track_err_l2_per_dof": real_l2.tolist(),
        "sim_track_err_l2_per_dof":  sim_l2.tolist(),
        "real_track_err_l2_total":   float(np.sqrt(np.mean(real_track_err ** 2))),
        "sim_track_err_l2_total":    float(np.sqrt(np.mean(sim_track_err ** 2))),
        "cmd_diff_l2_per_dof":   np.sqrt(np.mean(cmd_diff   ** 2, axis=0)).tolist(),
        "state_diff_l2_per_dof": np.sqrt(np.mean(state_diff ** 2, axis=0)).tolist(),
        "cmd_diff_l2_total":   float(np.sqrt(np.mean(cmd_diff   ** 2))),
        "state_diff_l2_total": float(np.sqrt(np.mean(state_diff ** 2))),
        "real_rpy_max_abs_deg":  np.max(np.abs(real_rpy), axis=0).tolist(),
        "sim_rpy_max_abs_deg":   np.max(np.abs(sim_rpy),  axis=0).tolist(),
        "real_imu_angvel_l2_total":  float(np.sqrt(np.mean(real.imu_angvel ** 2))),
        "sim_imu_angvel_l2_total":   float(np.sqrt(np.mean(sim_imu_angvel ** 2))),
        # Stash the resampled sim arrays so the plotter can use them.
        "_sim_cmd_pos_aligned": sim_cmd_pos,
        "_sim_state_pos_aligned": sim_state_pos,
        "_sim_imu_angvel_aligned": sim_imu_angvel,
        "_sim_rpy_aligned": sim_rpy,
        "_real_rpy": real_rpy,
        "_cmd_diff": cmd_diff,
    }


def _grid_layout(n: int) -> Tuple[int, int]:
    """Return (rows, cols) for a balanced grid that fits n panels."""
    if n <= 6:
        return 1, n
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_dof_pos_grid(real: Run, sim: Run, stats: dict, out: Path,
                      label_real: str, label_sim: str) -> None:
    import matplotlib.pyplot as plt

    sim_cmd_pos = stats["_sim_cmd_pos_aligned"]
    sim_state_pos = stats["_sim_state_pos_aligned"]
    rows, cols = 5, 7  # 31 panels comfortably
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 1.8),
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()
    for j in range(TOTAL_DOFS):
        ax = axes[j]
        ax.plot(real.t, np.degrees(real.cmd_pos[:, j]),  color="#1f77b4",
                lw=0.8, ls="--", label=f"cmd ({label_real})")
        ax.plot(real.t, np.degrees(real.state_pos[:, j]), color="#1f77b4",
                lw=1.2, label=f"state ({label_real})")
        ax.plot(real.t, np.degrees(sim_cmd_pos[:, j]),    color="#ff7f0e",
                lw=0.8, ls="--", label=f"cmd ({label_sim})")
        ax.plot(real.t, np.degrees(sim_state_pos[:, j]),  color="#ff7f0e",
                lw=1.2, label=f"state ({label_sim})")
        ax.set_title(real.joint_names[j].replace("_joint", ""), fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.grid(alpha=0.3, lw=0.4)
    for j in range(TOTAL_DOFS, len(axes)):
        axes[j].axis("off")
    axes[0].legend(loc="upper left", fontsize=6, framealpha=0.6)
    fig.suptitle(f"DoF position (deg)  --  {label_real} vs {label_sim}",
                 fontsize=11)
    fig.supxlabel("time in CONTROL window (s)", fontsize=9)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_dof_tracking_error(real: Run, sim: Run, stats: dict, out: Path,
                            label_real: str, label_sim: str) -> None:
    import matplotlib.pyplot as plt

    sim_cmd_pos = stats["_sim_cmd_pos_aligned"]
    sim_state_pos = stats["_sim_state_pos_aligned"]
    real_err = np.degrees(real.state_pos - real.cmd_pos)
    sim_err  = np.degrees(sim_state_pos - sim_cmd_pos)

    rows, cols = 5, 7
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 1.8),
                             sharex=True, sharey=False, constrained_layout=True)
    axes = axes.flatten()
    for j in range(TOTAL_DOFS):
        ax = axes[j]
        ax.plot(real.t, real_err[:, j], color="#1f77b4", lw=0.9, label=label_real)
        ax.plot(real.t, sim_err[:, j],  color="#ff7f0e", lw=0.9, label=label_sim)
        ax.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax.set_title(real.joint_names[j].replace("_joint", ""), fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.grid(alpha=0.3, lw=0.4)
    for j in range(TOTAL_DOFS, len(axes)):
        axes[j].axis("off")
    axes[0].legend(loc="upper left", fontsize=6, framealpha=0.6)
    fig.suptitle("Tracking error (state - cmd, deg)", fontsize=11)
    fig.supxlabel("time in CONTROL window (s)", fontsize=9)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_dof_l2_bar(real: Run, stats: dict, out: Path,
                    label_real: str, label_sim: str) -> None:
    import matplotlib.pyplot as plt

    real_l2 = np.degrees(np.array(stats["real_track_err_l2_per_dof"]))
    sim_l2  = np.degrees(np.array(stats["sim_track_err_l2_per_dof"]))
    n = TOTAL_DOFS
    x = np.arange(n)
    w = 0.42
    fig, ax = plt.subplots(figsize=(max(11, n * 0.36), 4.5),
                           constrained_layout=True)
    ax.bar(x - w / 2, real_l2, width=w, color="#1f77b4", label=label_real)
    ax.bar(x + w / 2, sim_l2,  width=w, color="#ff7f0e", label=label_sim)
    short = [n.replace("_joint", "") for n in real.joint_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("RMS tracking error (deg)")
    ax.set_title("Per-DoF tracking-error RMS (state - cmd)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_imu_overlay(real: Run, sim: Run, stats: dict, out: Path,
                     label_real: str, label_sim: str) -> None:
    import matplotlib.pyplot as plt

    real_rpy = stats["_real_rpy"]
    sim_rpy = stats["_sim_rpy_aligned"]
    sim_angvel = stats["_sim_imu_angvel_aligned"]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6),
                             sharex=True, constrained_layout=True)
    rpy_titles = ["roll (deg)", "pitch (deg)", "yaw (deg)"]
    av_titles  = ["wx (rad/s)", "wy (rad/s)", "wz (rad/s)"]

    for k in range(3):
        ax = axes[0, k]
        ax.plot(real.t, np.degrees(real_rpy[:, k]),
                color="#1f77b4", lw=1.2, label=label_real)
        ax.plot(real.t, np.degrees(sim_rpy[:, k]),
                color="#ff7f0e", lw=1.2, label=label_sim)
        ax.set_title(f"IMU {rpy_titles[k]}")
        ax.grid(alpha=0.3)
    for k in range(3):
        ax = axes[1, k]
        ax.plot(real.t, real.imu_angvel[:, k],
                color="#1f77b4", lw=1.0, label=label_real)
        ax.plot(real.t, sim_angvel[:, k],
                color="#ff7f0e", lw=1.0, label=label_sim)
        ax.set_title(f"IMU angvel {av_titles[k]}")
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("time in CONTROL window (s)")
    fig.suptitle("IMU overlay (base orientation + angular velocity)")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_cmd_diff_heatmap(real: Run, stats: dict, out: Path,
                          label_real: str, label_sim: str) -> None:
    import matplotlib.pyplot as plt

    cmd_diff_deg = np.degrees(stats["_cmd_diff"])  # (T, 31)
    short = [n.replace("_joint", "") for n in real.joint_names]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    vmax = np.percentile(np.abs(cmd_diff_deg), 99)
    if vmax < 1e-9:
        vmax = 1.0
    im = ax.imshow(
        cmd_diff_deg.T, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(real.t[0], real.t[-1], -0.5, TOTAL_DOFS - 0.5),
    )
    ax.set_yticks(np.arange(TOTAL_DOFS))
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("time in CONTROL window (s)")
    ax.set_title(f"cmd diff (deg):  {label_real} - {label_sim}")
    cbar = fig.colorbar(im, ax=ax, label="cmd diff (deg)")
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------


def write_summary_files(real: Run, sim: Run, stats: dict, out_dir: Path,
                        label_real: str, label_sim: str) -> None:
    # Strip private numpy fields out of the JSON.
    public_stats = {k: v for k, v in stats.items() if not k.startswith("_")}
    summary = {
        "real": {
            "path": str(real.path),
            "label": label_real,
            "duration_s": real.duration_s,
            "control_start_wall": real.control_start_wall,
            "control_end_wall": real.control_end_wall,
            "detected_handoff_at": real.detected_handoff_at,
            "meta_json": real.meta,
        },
        "sim": {
            "path": str(sim.path),
            "label": label_sim,
            "duration_s": sim.duration_s,
            "control_start_wall": sim.control_start_wall,
            "control_end_wall": sim.control_end_wall,
            "detected_handoff_at": sim.detected_handoff_at,
            "meta_json": sim.meta,
        },
        "stats": public_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Human-readable summary.
    lines = []
    lines.append(f"Sim-to-Real Comparison")
    lines.append("=" * 60)
    lines.append(f"  REAL: {label_real}")
    lines.append(f"        {real.path}")
    lines.append(f"        CONTROL window: {real.duration_s:.2f}s "
                 f"({real.t.size} samples)")
    if real.detected_handoff_at is not None:
        lines.append(f"        handoff back to MC at wall t = "
                     f"{real.detected_handoff_at:.2f}s")
    lines.append(f"  SIM:  {label_sim}")
    lines.append(f"        {sim.path}")
    lines.append(f"        CONTROL window: {sim.duration_s:.2f}s "
                 f"({sim.t.size} samples)")
    lines.append("")
    lines.append("Tracking error RMS (state - cmd), deg:")
    lines.append(f"   {label_real:>30s}: {np.degrees(stats['real_track_err_l2_total']):.3f}")
    lines.append(f"   {label_sim:>30s}: {np.degrees(stats['sim_track_err_l2_total']):.3f}")
    lines.append("")
    lines.append("Sim-to-real diff (over aligned CONTROL window):")
    lines.append(f"   cmd_pos   real - sim  RMS = {np.degrees(stats['cmd_diff_l2_total']):.3f} deg")
    lines.append(f"   state_pos real - sim  RMS = {np.degrees(stats['state_diff_l2_total']):.3f} deg")
    lines.append("")
    lines.append("IMU angvel RMS:")
    lines.append(f"   {label_real:>30s}: {stats['real_imu_angvel_l2_total']:.4f} rad/s")
    lines.append(f"   {label_sim:>30s}: {stats['sim_imu_angvel_l2_total']:.4f} rad/s")
    lines.append("")
    lines.append("Top 8 DoFs by tracking-error gap (real RMS - sim RMS), deg:")
    real_l2 = np.degrees(np.array(stats["real_track_err_l2_per_dof"]))
    sim_l2  = np.degrees(np.array(stats["sim_track_err_l2_per_dof"]))
    diff = real_l2 - sim_l2
    order = np.argsort(-np.abs(diff))[:8]
    lines.append(f"   {'DoF':<32s} {'real':>8s} {'sim':>8s} {'real-sim':>10s}")
    for j in order:
        nm = real.joint_names[j].replace("_joint", "")
        lines.append(f"   {nm:<32s} {real_l2[j]:>8.3f} {sim_l2[j]:>8.3f} "
                     f"{diff[j]:>10.3f}")
    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(txt)
    print(txt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--real", required=True, type=Path,
                   help="Real-robot recorder npz")
    p.add_argument("--sim",  required=True, type=Path,
                   help="Sim-mode recorder npz (deploy_x2.sh sim --record)")
    p.add_argument("--out",  required=True, type=Path,
                   help="Output directory (created if missing)")
    p.add_argument("--label-real", default="real",
                   help="Legend label for the real run")
    p.add_argument("--label-sim",  default="sim",
                   help="Legend label for the sim run")
    p.add_argument("--start-skip", type=float, default=0.0,
                   help="Seconds to skip after the first published command "
                        "(useful to drop soft-start ramp). Default 0.")
    p.add_argument("--end-trim", type=float, default=2.0,
                   help="Seconds to trim from the end (drops RAMP_OUT and "
                        "HOLD_FOR_MC). Default 2.0. Set 0 to keep the tail.")
    p.add_argument("--resample-hz", type=float, default=50.0,
                   help="Common time-grid rate for the comparison (Hz). "
                        "Default 50 (matches policy tick).")
    p.add_argument("--skip-plots", action="store_true",
                   help="Only write summary.json/summary.txt; skip PNGs")
    args = p.parse_args()

    if not args.real.exists():
        sys.exit(f"--real not found: {args.real}")
    if not args.sim.exists():
        sys.exit(f"--sim not found: {args.sim}")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[load] real: {args.real}")
    real = load_run(args.real, args.label_real,
                    start_skip=args.start_skip,
                    end_trim=args.end_trim,
                    resample_hz=args.resample_hz)
    print(f"       CONTROL window {real.duration_s:.2f}s, "
          f"{real.t.size} samples @ {args.resample_hz:.0f} Hz")

    print(f"[load] sim:  {args.sim}")
    sim = load_run(args.sim, args.label_sim,
                   start_skip=args.start_skip,
                   end_trim=args.end_trim,
                   resample_hz=args.resample_hz)
    print(f"       CONTROL window {sim.duration_s:.2f}s, "
          f"{sim.t.size} samples @ {args.resample_hz:.0f} Hz")

    if real.joint_names != sim.joint_names:
        warnings.warn(
            "joint name lists differ between runs; comparison will assume "
            "positional alignment which is probably wrong.",
            stacklevel=2,
        )

    stats = compute_stats(real, sim)

    write_summary_files(real, sim, stats, args.out,
                        args.label_real, args.label_sim)

    if not args.skip_plots:
        print(f"[plot] writing PNGs to {args.out}")
        plot_dof_pos_grid(real, sim, stats,
                          args.out / "dof_pos_grid.png",
                          args.label_real, args.label_sim)
        plot_dof_tracking_error(real, sim, stats,
                                args.out / "dof_tracking_error.png",
                                args.label_real, args.label_sim)
        plot_dof_l2_bar(real, stats,
                        args.out / "dof_l2_bar.png",
                        args.label_real, args.label_sim)
        plot_imu_overlay(real, sim, stats,
                         args.out / "imu_overlay.png",
                         args.label_real, args.label_sim)
        plot_cmd_diff_heatmap(real, stats,
                              args.out / "cmd_diff_heatmap.png",
                              args.label_real, args.label_sim)
    print(f"[done] wrote outputs to {args.out}")


if __name__ == "__main__":
    main()
