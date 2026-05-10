"""
Offline analyser for the side-channel debug NPZ written by
``teleop_x2_kinematic.py``.

Goal: mechanically test whether the **play-area-frame** hypothesis
explains a "hands behind body" episode. The hypothesis is:

  WebXR ``local-floor`` reference space publishes wrist/head pose in
  Guardian-frame coordinates, NOT in the operator's body frame.
  ``VRArmTeleop.engage()`` snapshots wrist xyz in that frame and
  ``_compose_target`` applies subsequent deltas as if they were in
  the robot torso frame. As soon as the operator yaws their body
  inside the play area, the wrist-xyz delta no longer agrees with
  the operator's intended body-relative motion -- the robot reaches
  the wrong direction (often "behind").

What this script reports
------------------------

For each debug NPZ passed on the command line:

* engage yaw (head yaw at the first engaged frame, in degrees).
* yaw-vs-time histogram in 30-degree bins, so you can see how often
  the operator was rotated away from the engage yaw.
* peak |yaw - engage_yaw|. >= 30 degrees is enough to spoil
  body-relative reaching; >= 90 degrees pretty much guarantees
  hands-behind-body.
* Pearson correlation between (wrist_x rotated into engage frame)
  and (engage-frame intended forward axis). When the operator
  rotates and reaches "forward in body frame", the play-area-frame
  wrist x flips sign relative to the engage-frame wrist x. A
  correlation that drops well below 1.0 (or goes negative on body
  rotations) is direct evidence of the bug.
* IK residuals (mean, max). Spikes at the same time-stamps as the
  yaw rotations argue the IK is being asked for impossible targets.

Usage::

    python -m gear_sonic.scripts.analyze_teleop_debug \\
        data/lerobot/x2_quest3_kinematic_v0/debug/teleop_episode_000000.npz

    # multiple episodes (last argument can be a glob)
    python -m gear_sonic.scripts.analyze_teleop_debug \\
        data/lerobot/x2_quest3_kinematic_v0/debug/teleop_episode_000000.npz \\
        data/lerobot/x2_quest3_kinematic_v0/debug/teleop_episode_000001.npz

The script prints a verdict line at the bottom of each report:

    VERDICT: play-area-frame bug LIKELY (max body yaw drift ... deg,
    forward-axis correlation ...).

It is *intentionally* read-only and never writes to the NPZ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Extract yaw (rotation about robot-frame +Z) from a [w,x,y,z] quat array.

    Args:
        quat_wxyz: ``(T, 4)`` array of [w,x,y,z] quaternions in the robot
            (Z-up) frame, as written by ``compute_3pt_pose_from_quest3``.

    Returns:
        ``(T,)`` array of yaw angles in radians, wrapped to [-pi, pi].
    """
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(f"expected (T, 4) quat array; got {quat_wxyz.shape}")
    # scipy expects xyzw
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rot = sRot.from_quat(quat_xyzw)
    # ZYX intrinsic Euler: first angle is yaw about world-Z.
    yaw = rot.as_euler("ZYX", degrees=False)[:, 0]
    return yaw


def _rotate_z(vec_xyz: np.ndarray, yaw_rad: np.ndarray | float) -> np.ndarray:
    """Rotate ``(T, 3)`` xyz vectors by per-row yaw_rad about world-Z.

    A scalar ``yaw_rad`` rotates every row by the same angle.
    """
    if np.isscalar(yaw_rad):
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        rotmat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return vec_xyz @ rotmat.T
    yaw = np.asarray(yaw_rad, dtype=np.float64)
    if yaw.shape[0] != vec_xyz.shape[0]:
        raise ValueError(
            f"per-row yaw must have len T={vec_xyz.shape[0]}; got {yaw.shape}"
        )
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty_like(vec_xyz)
    out[:, 0] = c * vec_xyz[:, 0] - s * vec_xyz[:, 1]
    out[:, 1] = s * vec_xyz[:, 0] + c * vec_xyz[:, 1]
    out[:, 2] = vec_xyz[:, 2]
    return out


def _wrap_to_pi(rad: np.ndarray) -> np.ndarray:
    return (rad + np.pi) % (2 * np.pi) - np.pi


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Per-episode report
# ---------------------------------------------------------------------------


def _report_one(path: Path) -> dict:
    """Print a diagnostic block for one debug NPZ, return a verdict dict."""
    print("=" * 78)
    print(f"file: {path}")

    if not path.exists():
        print(f"  ERROR: file does not exist; skipping")
        return {"path": str(path), "ok": False, "reason": "missing"}

    d = np.load(path, allow_pickle=True)
    keys = list(d.files)
    needed = (
        "vr_head_quat",
        "vr_left_wrist_pos",
        "vr_right_wrist_pos",
        "engaged",
        "ik_left_pos_err_m",
        "ik_right_pos_err_m",
        "ik_left_target_pos",
        "ik_right_target_pos",
        "t_episode_s",
    )
    missing = [k for k in needed if k not in keys]
    if missing:
        print(f"  ERROR: missing required keys: {missing}")
        return {"path": str(path), "ok": False, "reason": f"missing={missing}"}

    t = d["t_episode_s"]
    engaged = d["engaged"].astype(bool)
    head_quat = d["vr_head_quat"]
    L_wrist = d["vr_left_wrist_pos"]
    R_wrist = d["vr_right_wrist_pos"]
    L_target = d["ik_left_target_pos"]
    R_target = d["ik_right_target_pos"]
    L_err = d["ik_left_pos_err_m"]
    R_err = d["ik_right_pos_err_m"]

    T = t.shape[0]
    fps = float(d["fps"]) if "fps" in keys else float("nan")
    task = str(d["task"]) if "task" in keys else "?"
    print(f"  task    : {task!r}")
    print(f"  frames  : {T}  (~{(T / fps) if fps == fps else float('nan'):.1f} s @ {fps:.1f} Hz)")
    print(f"  engaged : {engaged.sum()} / {T} frames ({engaged.mean()*100:.1f}%)")

    if engaged.sum() == 0:
        print("  WARNING: no engaged frames -- press A on the controller before reaching")
        return {"path": str(path), "ok": False, "reason": "no_engaged_frames"}

    # ── Yaw analysis ──────────────────────────────────────────────────────
    yaw = _yaw_from_quat_wxyz(head_quat)
    engage_idx = int(np.argmax(engaged))  # first True
    engage_yaw = float(yaw[engage_idx])
    yaw_drift = _wrap_to_pi(yaw - engage_yaw)
    yaw_drift_deg = np.degrees(yaw_drift)

    yaw_drift_engaged = yaw_drift_deg[engaged]
    peak_drift = float(np.max(np.abs(yaw_drift_engaged))) if yaw_drift_engaged.size else 0.0

    print(f"  engage  : t={t[engage_idx]:.2f}s  head_yaw={np.degrees(engage_yaw):+7.1f} deg")
    print(f"  yaw-drift (engaged frames):")
    print(f"    median |drift| = {np.median(np.abs(yaw_drift_engaged)):.1f} deg")
    print(f"    peak   |drift| = {peak_drift:.1f} deg")

    # Histogram in 30-deg bins
    bins = np.arange(-180, 181, 30)
    hist, _ = np.histogram(yaw_drift_engaged, bins=bins)
    if engaged.sum() > 0:
        pct = 100.0 * hist / engaged.sum()
        line = "    yaw bins   "
        for lo, p in zip(bins[:-1], pct):
            if p >= 1.0:
                line += f"[{lo:+4d}:{p:4.1f}%] "
        print(line.rstrip())

    # ── Forward-axis correlation ──────────────────────────────────────────
    # Project wrist xy into the engage-frame (rotate by -engage_yaw). If the
    # operator only translates wrists in body-frame (e.g. "reach forward"),
    # the engage-frame wrist xy should track those reaches faithfully. If
    # the play-area-frame bug is active, raw L_wrist[:, 0] and the
    # engage-frame x_body diverge whenever yaw drifts.
    L_eng = _rotate_z(L_wrist - L_wrist[engage_idx], -engage_yaw)
    R_eng = _rotate_z(R_wrist - R_wrist[engage_idx], -engage_yaw)
    L_raw = L_wrist - L_wrist[engage_idx]
    R_raw = R_wrist - R_wrist[engage_idx]

    # Mask to engaged frames for the correlation.
    mask = engaged.copy()
    if mask.sum() < 5:
        print("  too few engaged frames for correlation analysis")
    else:
        # The KEY diagnostic: when the body-frame x reach matches what the
        # robot was *commanded* (which uses raw L_wrist deltas without the
        # engage-yaw correction), correlation is ~1.0. Below 0.7 means the
        # play-area frame is rotating the operator's reaches away from the
        # robot's interpretation.
        cx_L = _safe_corr(L_eng[mask, 0], L_raw[mask, 0])
        cy_L = _safe_corr(L_eng[mask, 1], L_raw[mask, 1])
        cx_R = _safe_corr(R_eng[mask, 0], R_raw[mask, 0])
        cy_R = _safe_corr(R_eng[mask, 1], R_raw[mask, 1])
        print(
            f"  body-vs-play-area corr (1.0 = no bug, ~0 or negative = bug):\n"
            f"    L wrist  x={cx_L:+.3f}  y={cy_L:+.3f}\n"
            f"    R wrist  x={cx_R:+.3f}  y={cy_R:+.3f}"
        )

    # ── IK residual summary ───────────────────────────────────────────────
    print(
        f"  IK pos residual (engaged frames):\n"
        f"    L  mean={float(L_err[engaged].mean())*100:.2f} cm  "
        f"max={float(L_err[engaged].max())*100:.2f} cm\n"
        f"    R  mean={float(R_err[engaged].mean())*100:.2f} cm  "
        f"max={float(R_err[engaged].max())*100:.2f} cm"
    )

    # ── Verdict ───────────────────────────────────────────────────────────
    verdict = "PASS"
    reasons: list[str] = []
    if peak_drift > 30.0:
        verdict = "play-area-frame bug LIKELY"
        reasons.append(f"peak yaw drift {peak_drift:.0f} deg (>30 deg)")
    if peak_drift > 90.0:
        verdict = "play-area-frame bug VERY LIKELY"
        reasons.append("peak yaw drift exceeds 90 deg -- expect hands-behind-body")
    print(f"\n  VERDICT: {verdict}" + (" (" + "; ".join(reasons) + ")" if reasons else ""))

    return {
        "path": str(path),
        "ok": True,
        "frames": int(T),
        "engaged_frames": int(engaged.sum()),
        "engage_yaw_deg": float(np.degrees(engage_yaw)),
        "peak_yaw_drift_deg": peak_drift,
        "verdict": verdict,
        "reasons": reasons,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Analyse a teleop_x2_kinematic.py debug NPZ for the "
        "play-area-frame retargeting bug.",
    )
    p.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more debug NPZ files written under "
        "<lerobot-output-dir>/debug/teleop_episode_NNNNNN.npz",
    )
    args = p.parse_args()

    results = []
    for path in args.paths:
        try:
            results.append(_report_one(path))
        except Exception as exc:  # noqa: BLE001 -- defensive analysis tool
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results.append({"path": str(path), "ok": False, "reason": str(exc)})

    print("=" * 78)
    print(f"analysed {len(results)} file(s)")
    bad = [r for r in results if r.get("ok") and "play-area" in r.get("verdict", "")]
    if bad:
        print(f"  {len(bad)} file(s) flagged the play-area-frame bug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
