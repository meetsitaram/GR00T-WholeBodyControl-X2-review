"""Offline replay: feed a recorded debug NPZ through the new teleop solver.

Goal
----

Validate that the P0 + P0 + P1 fixes (controller-dropout rejection,
elbow-down null-space bias, head-yaw-corrected wrist quaternion) actually
reduce the failure modes the user observed in
``data/lerobot/x2_quest3_kinematic_v2/debug/teleop_episode_000000.npz``:

* "Hands behind the body" / unreachable IK targets — caused by raw
  ``(0, 0, 0)`` controller-dropout positions reaching the IK target.
* "Elbows facing reverse" — caused by the DLS picking arbitrary
  redundant-DOF branches when the target is unreachable / near-singular.
* "Wrists do not rotate" — caused by ``rotation_weight=0`` because the
  wrist quaternion was not aligned to the robot frame.

We replay the recorded raw VR samples through ``VRArmTeleopCalibrated``
(post-fix) and compare against the OLD ``ik_left_q_rad / ik_right_q_rad``
written by the original solver.

Usage::

    python scripts/replay_teleop_debug_with_fixes.py \\
        --npz data/lerobot/x2_quest3_kinematic_v2/debug/teleop_episode_000000.npz \\
        --calibration data/operator_calibrations/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gear_sonic.utils.teleop.operator_calibration import OperatorCalibration
from gear_sonic.utils.teleop.solver.arm.x2_arm_fk import arm_fk
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import (
    VRArmTeleopCalibrated,
    _is_controller_dropout,
    _is_twin_dropout,
)


def _build_vr_pose(d, i: int) -> np.ndarray:
    out = np.zeros((3, 7), dtype=np.float64)
    out[0, :3] = d["vr_left_wrist_pos"][i]
    out[0, 3:] = d["vr_left_wrist_quat"][i]
    out[1, :3] = d["vr_right_wrist_pos"][i]
    out[1, 3:] = d["vr_right_wrist_quat"][i]
    out[2, :3] = d["vr_head_pos"][i]
    out[2, 3:] = d["vr_head_quat"][i]
    return out


def _stats(name: str, x: np.ndarray) -> str:
    return (
        f"  {name:<26s} median={np.median(x)*100:7.2f} cm  "
        f"p95={np.percentile(x, 95)*100:7.2f} cm  "
        f"max={x.max()*100:7.2f} cm  "
        f"mean={x.mean()*100:7.2f} cm"
    )


def _branch_flip_count(q_series: np.ndarray, joint: int) -> int:
    """Count zero-crossings in a single joint over an engaged window.

    A "flip" is a sign change with magnitude ≥ 0.5 rad on either side
    within 5 ticks (i.e., a real 180-degree swing rather than a tiny
    crossing of zero on a slow drift). For 7-DOF arms with neutral q
    near zero, ``shoulder_yaw`` (joint 2) and ``elbow`` (joint 3) are
    the channels where elbow-flipping shows up.
    """
    if q_series.shape[0] < 5:
        return 0
    s = q_series[:, joint]
    crossings = 0
    last_extreme = s[0]
    for i in range(1, len(s)):
        if np.sign(s[i]) != np.sign(last_extreme) and abs(s[i] - last_extreme) > 0.8:
            crossings += 1
            last_extreme = s[i]
        elif abs(s[i]) > abs(last_extreme):
            last_extreme = s[i]
    return crossings


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument(
        "--calibration", type=Path,
        default=Path("data/operator_calibrations/default.yaml"),
    )
    p.add_argument(
        "--rotation-weight", type=float, default=0.3,
        help="Forwarded to VRArmTeleopCalibrated. Auto-disabled by the "
             "teleop class if the loaded calibration has no wrist "
             "alignment quat (legacy v0 YAML).",
    )
    p.add_argument(
        "--null-space-gain", type=float, default=0.10,
        help="Null-space bias gain (default 0.10). 0 disables.",
    )
    args = p.parse_args()

    print(f"\n[replay] loading {args.npz}", flush=True)
    d = np.load(args.npz, allow_pickle=True)
    n = int(d["num_frames"]) if "num_frames" in d.files else d["t_episode_s"].shape[0]
    engaged_old = d["engaged"].astype(bool)
    print(f"[replay] frames: {n}, engaged: {engaged_old.sum()} ({100*engaged_old.mean():.1f}%)")

    print(f"[replay] loading {args.calibration}")
    cal = OperatorCalibration.load_yaml(args.calibration)
    print(f"[replay]   schema_version={cal.schema_version}, operator_id={cal.operator_id}")
    has_alignment = False
    for side in ("left", "right"):
        q = cal.fit[side].wrist_alignment_quat
        is_ident = float(np.linalg.norm(q - np.array([1.0, 0, 0, 0]))) < 1e-3
        print(f"[replay]   {side} alignment_quat = {q}  (identity={is_ident})")
        has_alignment = has_alignment or not is_ident
    if not has_alignment:
        print(
            "[replay] NOTE: legacy v0 calibration -- wrist orientation IK "
            "will be auto-disabled."
        )

    # ── 1) Dropout statistics on raw VR data ─────────────────────────
    print("\n[replay] === raw VR dropout statistics ===")
    drops_l = np.zeros(n, dtype=bool)
    drops_r = np.zeros(n, dtype=bool)
    twin = np.zeros(n, dtype=bool)
    for i in range(n):
        drops_l[i] = _is_controller_dropout(
            d["vr_left_wrist_pos"][i].astype(np.float64),
            d["vr_left_wrist_quat"][i].astype(np.float64),
        )
        drops_r[i] = _is_controller_dropout(
            d["vr_right_wrist_pos"][i].astype(np.float64),
            d["vr_right_wrist_quat"][i].astype(np.float64),
        )
        twin[i] = _is_twin_dropout(
            d["vr_left_wrist_pos"][i].astype(np.float64),
            d["vr_right_wrist_pos"][i].astype(np.float64),
        )
    eng = engaged_old
    print(f"  L dropouts (engaged frames):        {drops_l[eng].sum():5d} / {eng.sum():5d} "
          f"({100*drops_l[eng].mean():.2f}%)")
    print(f"  R dropouts (engaged frames):        {drops_r[eng].sum():5d} / {eng.sum():5d} "
          f"({100*drops_r[eng].mean():.2f}%)")
    print(f"  twin dropouts (engaged frames):     {twin[eng].sum():5d} / {eng.sum():5d} "
          f"({100*twin[eng].mean():.2f}%)")
    any_drop = (drops_l | drops_r | twin) & eng
    print(f"  ANY dropout (engaged frames):       {any_drop.sum():5d} / {eng.sum():5d} "
          f"({100*any_drop.mean():.2f}%)")

    # ── 2) Replay through the new solver ─────────────────────────────
    print("\n[replay] === replaying through new VRArmTeleopCalibrated ===")
    teleop = VRArmTeleopCalibrated(
        calibration=cal,
        rotation_weight=args.rotation_weight,
        null_space_gain=args.null_space_gain,
    )

    new_left_q = np.zeros((n, 7), dtype=np.float64)
    new_right_q = np.zeros((n, 7), dtype=np.float64)
    new_left_target = np.zeros((n, 3), dtype=np.float64)
    new_right_target = np.zeros((n, 3), dtype=np.float64)
    new_left_pos_err = np.zeros(n, dtype=np.float64)
    new_right_pos_err = np.zeros(n, dtype=np.float64)
    new_left_held = np.zeros(n, dtype=bool)
    new_right_held = np.zeros(n, dtype=bool)
    new_left_drop = np.zeros(n, dtype=bool)
    new_right_drop = np.zeros(n, dtype=bool)

    # Match the original engagement state. The fix should improve quality
    # within engaged windows, not change when we engage.
    engaged_state = False
    for i in range(n):
        if engaged_old[i] != engaged_state:
            teleop.set_engaged(bool(engaged_old[i]))
            engaged_state = bool(engaged_old[i])
        vr = _build_vr_pose(d, i)
        res = teleop.step(vr)
        new_left_q[i] = res.left_q
        new_right_q[i] = res.right_q
        new_left_target[i] = res.left_target_pos
        new_right_target[i] = res.right_target_pos
        new_left_pos_err[i] = res.left_ik.pos_err_m
        new_right_pos_err[i] = res.right_ik.pos_err_m
        new_left_held[i] = bool(res.left_target_held)
        new_right_held[i] = bool(res.right_target_held)
        new_left_drop[i] = bool(res.left_dropout)
        new_right_drop[i] = bool(res.right_dropout)

    # ── 3) Per-tick IK position error (engaged only) ─────────────────
    print("\n[replay] === IK position error on engaged frames ===")
    print("  OLD solver (recorded):")
    print(_stats("L_pos_err",
                 d["ik_left_pos_err_m"][eng].astype(np.float64)))
    print(_stats("R_pos_err",
                 d["ik_right_pos_err_m"][eng].astype(np.float64)))
    print("  NEW solver (replay):")
    print(_stats("L_pos_err", new_left_pos_err[eng]))
    print(_stats("R_pos_err", new_right_pos_err[eng]))

    # ── 4) Held-target frames (proves dropout rejection is firing) ──
    print("\n[replay] === dropout rejection ===")
    held_l = new_left_held & eng
    held_r = new_right_held & eng
    print(f"  L target held due to dropout: {held_l.sum():5d} / {eng.sum():5d} "
          f"({100*held_l.mean():.2f}%)")
    print(f"  R target held due to dropout: {held_r.sum():5d} / {eng.sum():5d} "
          f"({100*held_r.mean():.2f}%)")

    # The interesting metric: on dropout frames, did the OLD solver
    # produce worse IK errors than the NEW solver?
    drop_mask = (new_left_drop | new_right_drop) & eng
    if drop_mask.sum() > 50:
        print(f"\n  Dropout-frame IK error comparison ({drop_mask.sum()} frames):")
        print("    OLD (raw target):")
        print(_stats("    L_pos_err",
                     d["ik_left_pos_err_m"][drop_mask].astype(np.float64)))
        print(_stats("    R_pos_err",
                     d["ik_right_pos_err_m"][drop_mask].astype(np.float64)))
        print("    NEW (held target):")
        print(_stats("    L_pos_err", new_left_pos_err[drop_mask]))
        print(_stats("    R_pos_err", new_right_pos_err[drop_mask]))

    # ── 5) Null-space / branch-flip metric ───────────────────────────
    print("\n[replay] === elbow / shoulder-yaw branch flips ===")
    old_left_q = d["ik_left_q_rad"].astype(np.float64)
    old_right_q = d["ik_right_q_rad"].astype(np.float64)
    # Restrict to engaged windows; concatenate them.
    def _engaged_segments(arr, mask):
        return arr[mask]
    old_l_eng = _engaged_segments(old_left_q, eng)
    old_r_eng = _engaged_segments(old_right_q, eng)
    new_l_eng = _engaged_segments(new_left_q, eng)
    new_r_eng = _engaged_segments(new_right_q, eng)

    for joint, name in ((2, "shoulder_yaw"), (3, "elbow")):
        old_l = _branch_flip_count(old_l_eng, joint)
        old_r = _branch_flip_count(old_r_eng, joint)
        new_l = _branch_flip_count(new_l_eng, joint)
        new_r = _branch_flip_count(new_r_eng, joint)
        print(f"  {name:<14s} OLD: L={old_l:3d}  R={old_r:3d}    "
              f"NEW: L={new_l:3d}  R={new_r:3d}")

    # ── 6) Distance from preferred (arms-down) posture ───────────────
    print("\n[replay] === q distance from arms-down preferred posture (rad) ===")
    q_pref_l = teleop._left_solver.q_preferred
    q_pref_r = teleop._right_solver.q_preferred
    old_dist_l = np.linalg.norm(old_l_eng - q_pref_l, axis=1)
    old_dist_r = np.linalg.norm(old_r_eng - q_pref_r, axis=1)
    new_dist_l = np.linalg.norm(new_l_eng - q_pref_l, axis=1)
    new_dist_r = np.linalg.norm(new_r_eng - q_pref_r, axis=1)
    print(f"  OLD solver:  L mean={old_dist_l.mean():.3f}  median={np.median(old_dist_l):.3f}  "
          f"R mean={old_dist_r.mean():.3f}  median={np.median(old_dist_r):.3f}")
    print(f"  NEW solver:  L mean={new_dist_l.mean():.3f}  median={np.median(new_dist_l):.3f}  "
          f"R mean={new_dist_r.mean():.3f}  median={np.median(new_dist_r):.3f}")

    # ── 7) Wrist motion (joints 4/5/6) ───────────────────────────────
    print("\n[replay] === wrist motion (joints 4-6) std rad ===")
    for joint, name in ((4, "wrist_yaw"), (5, "wrist_pitch"), (6, "wrist_roll")):
        old_l_std = old_l_eng[:, joint].std()
        old_r_std = old_r_eng[:, joint].std()
        new_l_std = new_l_eng[:, joint].std()
        new_r_std = new_r_eng[:, joint].std()
        print(f"  {name:<12s} OLD: L_std={old_l_std:.3f}  R_std={old_r_std:.3f}    "
              f"NEW: L_std={new_l_std:.3f}  R_std={new_r_std:.3f}")
    if not has_alignment:
        print("  (NEW wrist motion will look identical to OLD since")
        print("   orientation IK was auto-disabled for legacy v0 YAML.)")

    print("\n[replay] DONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
