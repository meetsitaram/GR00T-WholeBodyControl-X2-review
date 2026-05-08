#!/usr/bin/env python3
"""Headless survival benchmark for a SONIC checkpoint over many motion clips.

For every motion in a PKL (or every motion matching ``--filter``), RSI-init
the X2 Ultra into MuJoCo, run the policy in-process, and measure how long the
robot stays upright before triggering an IsaacLab-style fall reset
(``pelvis_z < --fall-height`` or ``gravity_body[z] > --fall-tilt-cos``).

Reuses the actor / observation / PD-control plumbing from
``eval_x2_mujoco.py`` — no viewer, no rendering, so we can sweep hundreds of
clips in a few minutes on CPU.

Outputs a leaderboard sorted by survival time + writes a CSV report.

Example:
    conda run -n env_isaaclab --no-capture-output python \\
        gear_sonic/scripts/benchmark_motions_mujoco.py \\
        --checkpoint $HOME/x2_cloud_checkpoints/run-20260420_083925/last.pt \\
        --motion gear_sonic/data/motions/x2_ultra_bones_seed.pkl \\
        --filter "standing__" \\
        --max-seconds 6.0 \\
        --report /tmp/x2_survival_standing.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.scripts.eval_x2_mujoco import (  # noqa: E402
    ACTION_SCALE,
    CONTROL_DT,
    DECIMATION,
    DEFAULT_DOF,
    JOINT_TO_ACTUATOR,
    KD,
    KP,
    IL_TO_MJ_DOF,
    MJ_TO_IL_DOF,
    MJCF_PATH,
    MUJOCO_JOINT_NAMES,
    NUM_DOFS,
    SIM_DT,
    ProprioceptionBuffer,
    build_tokenizer_obs,
    compute_motion_state,
    load_actor_from_checkpoint,
    quat_rotate_inverse,
)


# ---- Per-joint-group PD scaling.
# The X2 deployment uses KP = armature × ω² and KD = 2·ζ·armature·ω, i.e. the
# *training-equivalent* PD on every joint. The working G1 sim2sim pipeline
# (`gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`) ships
# *two* PD sets — `JOINT_KP/KD` (training-style) and `MOTOR_KP/KD` (deployment,
# hand-tuned higher on legs/ankles, lower on waist) — and only the MOTOR_*
# values are applied to the MuJoCo `ctrl`. The asymmetry is the standard fix
# for the implicit-vs-explicit actuator gap (G5 in `sim2sim_mujoco.md`):
# IsaacLab integrates PD against the joint-space inertia + armature implicitly,
# so the same numerical KP behaves stiffer than the explicit `ctrl`-driven
# torque MuJoCo applies. Bumping the deployed KP/KD recovers that authority
# without retraining.
#
# Match patterns are evaluated against the MuJoCo joint name (with
# `left_/right_` and `_joint` stripped). First match wins, so patterns are
# ordered most-specific-first ("ankle" before generic catch-alls).
_PD_GROUPS = [
    ("hip", "leg"),
    ("knee", "knee"),
    ("ankle", "ankle"),
    ("waist", "waist"),
    ("shoulder", "arm"),
    ("elbow", "arm"),
    ("wrist", "wrist"),
    ("head", "head"),
]


def _build_pd_scale_arrays(args) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build per-DOF KP/KD scale arrays from the CLI group-level scales.

    Returns (kp_scale, kd_scale, summary_lines) where the arrays have shape
    (NUM_DOFS,) in MuJoCo joint order. summary_lines is a human-readable
    description of which non-1.0 scales were applied (for the run header).
    """
    kp_scale = np.full(NUM_DOFS, float(args.kp_scale), dtype=np.float64)
    kd_scale = np.full(NUM_DOFS, float(args.kd_scale), dtype=np.float64)

    group_kp = {
        "leg": float(args.kp_scale_leg),
        "knee": float(args.kp_scale_knee),
        "ankle": float(args.kp_scale_ankle),
        "waist": float(args.kp_scale_waist),
        "arm": float(args.kp_scale_arm),
        "wrist": float(args.kp_scale_wrist),
        "head": float(args.kp_scale_head),
    }
    group_kd = {
        "leg": float(args.kd_scale_leg),
        "knee": float(args.kd_scale_knee),
        "ankle": float(args.kd_scale_ankle),
        "waist": float(args.kd_scale_waist),
        "arm": float(args.kd_scale_arm),
        "wrist": float(args.kd_scale_wrist),
        "head": float(args.kd_scale_head),
    }

    for i, jname in enumerate(MUJOCO_JOINT_NAMES):
        short = jname.replace("left_", "").replace("right_", "").replace("_joint", "")
        for token, group in _PD_GROUPS:
            if token in short:
                kp_scale[i] *= group_kp[group]
                kd_scale[i] *= group_kd[group]
                break

    summary: list[str] = []
    if abs(args.kp_scale - 1.0) > 1e-9:
        summary.append(f"kp×{args.kp_scale:g}")
    if abs(args.kd_scale - 1.0) > 1e-9:
        summary.append(f"kd×{args.kd_scale:g}")
    for group, kp_v in group_kp.items():
        kd_v = group_kd[group]
        if abs(kp_v - 1.0) > 1e-9 or abs(kd_v - 1.0) > 1e-9:
            summary.append(f"{group}(kp×{kp_v:g},kd×{kd_v:g})")
    return kp_scale, kd_scale, summary


def _apply_init_state(mj_model, mj_data, motion_state):
    s = motion_state
    mj_data.qpos[0] = 0.0
    mj_data.qpos[1] = 0.0
    mj_data.qpos[2] = float(s["root_pos_w"][2])
    mj_data.qpos[3:7] = s["root_quat_w_wxyz"]
    mj_data.qpos[7:7 + NUM_DOFS] = s["joint_pos_mj"]
    mj_data.qvel[0:3] = s["root_lin_vel_w"]
    mj_data.qvel[3:6] = quat_rotate_inverse(
        s["root_quat_w_wxyz"], s["root_ang_vel_w"]
    )
    mj_data.qvel[6:6 + NUM_DOFS] = s["joint_vel_mj"]
    mj_data.xfrc_applied[:] = 0
    mujoco.mj_forward(mj_model, mj_data)


def run_one(actor, mj_model, mj_data, motion_data, args, device, kp_used, kd_used):
    """Roll out the policy on a single-clip motion_data dict.

    Returns (survival_seconds, fall_reason_or_None).
    """
    motion_entry = motion_data[next(iter(motion_data))]
    total_frames = motion_entry["dof"].shape[0]
    motion_fps = float(motion_entry["fps"])

    init_frame = int(args.init_frame)
    init_state = compute_motion_state(motion_data, init_frame, motion_fps)
    # Wipe any leftover MuJoCo state from the previous clip in this sweep —
    # actuator state, mj_data.time, contact buffers etc. otherwise persist
    # and skew the early-step dynamics of the next motion's rollout.
    mujoco.mj_resetData(mj_model, mj_data)
    _apply_init_state(mj_model, mj_data, init_state)

    prop_buf = ProprioceptionBuffer()
    last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
    # First-order EMA on the joint-target action emitted by the policy.
    # Output of policy on tick t:  raw_t        (NUM_DOFS,)
    # PD target on tick t:         filt_t = α·raw_t + (1-α)·filt_{t-1}
    # α = 1.0 → no filter (legacy behaviour, this branch returns raw_t).
    # The filter is applied AFTER the policy forward pass, so the
    # next-step proprioception sees the *filtered* action as `last_action`,
    # mirroring what the policy would receive in IsaacLab if a filter were
    # applied there.
    filt_action_mj = None
    sim_time = float(init_frame) / motion_fps
    episode_start = sim_time
    step_count = 0

    while True:
        # Match eval_x2_mujoco.py exactly: snap motion_time to the nearest
        # frame so tokenizer_obs is sampled on discrete frame boundaries.
        motion_time = sim_time
        motion_frame = int(motion_time * motion_fps) % total_frames
        motion_time = motion_frame / motion_fps

        qpos_j = mj_data.qpos[7:7 + NUM_DOFS].copy()
        qvel_j = mj_data.qvel[6:6 + NUM_DOFS].copy()
        base_quat = mj_data.qpos[3:7].copy()
        base_angvel = mj_data.qvel[3:6].copy()

        dof_pos_il = qpos_j[IL_TO_MJ_DOF]
        dof_vel_il = qvel_j[IL_TO_MJ_DOF]
        action_il_prev = last_action_mj[IL_TO_MJ_DOF]

        gravity = quat_rotate_inverse(base_quat, np.array([0., 0., -1.]))
        dof_pos_rel_il = dof_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]

        prop_buf.append(gravity, base_angvel, dof_pos_rel_il, dof_vel_il, action_il_prev)
        proprioception = prop_buf.get_flat()
        tokenizer_obs = build_tokenizer_obs(
            motion_data, motion_time, base_quat, motion_fps)

        with torch.no_grad():
            prop_t = torch.from_numpy(proprioception).unsqueeze(0).to(device)
            tok_t = torch.from_numpy(tokenizer_obs).unsqueeze(0).to(device)
            action_il_t = actor(prop_t, tok_t).squeeze(0).cpu().numpy()

        action_mj = action_il_t[MJ_TO_IL_DOF]
        alpha = float(args.action_lpf_alpha)
        if alpha < 1.0:
            if filt_action_mj is None:
                filt_action_mj = action_mj.copy()
            else:
                filt_action_mj = alpha * action_mj + (1.0 - alpha) * filt_action_mj
            action_mj_used = filt_action_mj
        else:
            action_mj_used = action_mj
        last_action_mj = action_mj_used.copy()
        target_pos = DEFAULT_DOF + action_mj_used * ACTION_SCALE

        for _ in range(DECIMATION):
            torque = kp_used * (target_pos - mj_data.qpos[7:7 + NUM_DOFS]) \
                   - kd_used * mj_data.qvel[6:6 + NUM_DOFS]
            for j in range(NUM_DOFS):
                mj_data.ctrl[JOINT_TO_ACTUATOR[j]] = torque[j]
            mujoco.mj_step(mj_model, mj_data)

        sim_time += CONTROL_DT
        step_count += 1

        pelvis_z = float(mj_data.qpos[2])
        grav_z = float(gravity[2])
        elapsed = sim_time - episode_start

        if pelvis_z < args.fall_height:
            return elapsed, f"pelvis_z={pelvis_z:.2f}"
        if grav_z > args.fall_tilt_cos:
            return elapsed, f"tilt grav_z={grav_z:+.2f}"
        if elapsed >= args.max_seconds:
            return elapsed, None  # survived to time-limit


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion", required=True,
                        help="PKL with one or many motions")
    parser.add_argument("--filter", default="",
                        help="Only test motion keys containing this substring "
                        "(e.g. 'standing__' or 'idle')")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N motions (0 = all)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="If >=0, deterministically shuffle the matched "
                        "keys with this RNG seed BEFORE applying --limit. "
                        "Use this to draw a reproducible random subsample "
                        "(e.g. --seed 0 --limit 50 picks the same 50 random "
                        "motions every run). Default -1 = no shuffle, "
                        "preserve PKL key order (legacy behavior).")
    parser.add_argument("--max-seconds", type=float, default=6.0,
                        help="Cap per-clip rollout (default 6.0s).")
    parser.add_argument("--init-frame", type=int, default=0)
    parser.add_argument("--fall-height", type=float, default=0.4)
    parser.add_argument("--fall-tilt-cos", type=float, default=-0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default="",
                        help="Optional CSV path to write per-motion results.")
    parser.add_argument("--top", type=int, default=30,
                        help="Print top-N most-stable in summary (default 30)")

    # ---- Post-training (deployment-only) physics knobs.
    # All four below override mj_model.opt fields after MJCF parse, so they
    # change deployment behaviour without touching the MJCF or the policy.
    parser.add_argument("--impratio", type=float, default=-1.0,
                        help="Override mj_model.opt.impratio (MuJoCo default 1). "
                        "Higher values increase friction-cone resolution at "
                        "stiff contacts (commonly tested values: 10, 100). "
                        "<0 = leave at MJCF/MuJoCo default.")
    parser.add_argument("--cone", default="",
                        choices=["", "pyramidal", "elliptic"],
                        help="Override mj_model.opt.cone. MuJoCo default "
                        "is pyramidal; elliptic matches PhysX's cone "
                        "shape more closely. '' = leave at MJCF default.")
    parser.add_argument("--solver-iters", type=int, default=-1,
                        help="Override mj_model.opt.iterations (Newton inner "
                        "iterations). MuJoCo default 100. <0 = leave alone.")
    parser.add_argument("--action-lpf-alpha", type=float, default=1.0,
                        help="EMA alpha applied to the policy's joint target "
                        "before PD: filt = α·raw + (1-α)·prev_filt. "
                        "1.0 = no filter (default), smaller = more smoothing. "
                        "Use this to attenuate policy chatter at deployment "
                        "without retraining.")

    # ---- Per-joint-group PD scaling. Mirrors the JOINT_KP→MOTOR_KP gap
    # baked into the working G1 sim2sim YAML. All multiplicative; final per-DOF
    # scale = global × group. See _build_pd_scale_arrays for joint→group
    # mapping.
    parser.add_argument("--kp-scale", type=float, default=1.0,
                        help="Global multiplier applied to KP on every DOF "
                        "(stacked with per-group --kp-scale-*).")
    parser.add_argument("--kd-scale", type=float, default=1.0,
                        help="Global multiplier applied to KD on every DOF.")
    for group, default_help in [
        ("leg",   "hip_yaw/roll/pitch"),
        ("knee",  "knee"),
        ("ankle", "ankle_pitch/roll"),
        ("waist", "waist_yaw/pitch/roll"),
        ("arm",   "shoulder + elbow"),
        ("wrist", "wrist_yaw/pitch/roll"),
        ("head",  "head_yaw/pitch"),
    ]:
        parser.add_argument(f"--kp-scale-{group}", type=float, default=1.0,
                            help=f"KP multiplier on {group} group "
                                 f"({default_help}). Default 1.0.")
        parser.add_argument(f"--kd-scale-{group}", type=float, default=1.0,
                            help=f"KD multiplier on {group} group "
                                 f"({default_help}). Default 1.0.")
    args = parser.parse_args()

    print(f"Loading actor from {args.checkpoint} ...", flush=True)
    actor = load_actor_from_checkpoint(args.checkpoint, args.device)
    print("  Actor loaded.", flush=True)

    print(f"Loading motions from {args.motion} ...", flush=True)
    all_motions = joblib.load(args.motion)
    keys = [k for k in all_motions.keys() if args.filter in k]
    n_match = len(keys)
    if args.seed >= 0:
        rng = random.Random(args.seed)
        rng.shuffle(keys)
    if args.limit > 0:
        keys = keys[:args.limit]
    sub_msg = f" (seed={args.seed} subsample)" if args.seed >= 0 else ""
    print(f"  {len(all_motions)} total → {n_match} match filter "
          f"'{args.filter}' → {len(keys)} selected{sub_msg}",
          flush=True)
    if not keys:
        print("Nothing to benchmark. Exiting.")
        return

    print("Loading MuJoCo model ...", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_model.opt.timestep = SIM_DT
    overrides = []
    if args.impratio >= 0:
        mj_model.opt.impratio = float(args.impratio)
        overrides.append(f"impratio={args.impratio}")
    if args.cone:
        mj_model.opt.cone = (mujoco.mjtCone.mjCONE_PYRAMIDAL
                             if args.cone == "pyramidal"
                             else mujoco.mjtCone.mjCONE_ELLIPTIC)
        overrides.append(f"cone={args.cone}")
    if args.solver_iters > 0:
        mj_model.opt.iterations = int(args.solver_iters)
        overrides.append(f"iter={args.solver_iters}")
    if args.action_lpf_alpha < 1.0:
        overrides.append(f"lpf_alpha={args.action_lpf_alpha}")
    if overrides:
        print(f"  Post-training overrides: {', '.join(overrides)}", flush=True)

    kp_scale, kd_scale, pd_summary = _build_pd_scale_arrays(args)
    kp_used = (KP * kp_scale).astype(np.float64)
    kd_used = (KD * kd_scale).astype(np.float64)
    if pd_summary:
        print(f"  PD scaling: {', '.join(pd_summary)}", flush=True)
        print("  Per-DOF effective KP/KD (scale ≠ 1.0 only):", flush=True)
        for i, jname in enumerate(MUJOCO_JOINT_NAMES):
            if abs(kp_scale[i] - 1.0) > 1e-9 or abs(kd_scale[i] - 1.0) > 1e-9:
                print(f"    {jname:<28s}  KP {KP[i]:7.2f} → {kp_used[i]:7.2f}"
                      f"  (×{kp_scale[i]:.3g})   "
                      f"KD {KD[i]:6.3f} → {kd_used[i]:6.3f}"
                      f"  (×{kd_scale[i]:.3g})",
                      flush=True)
    mj_data = mujoco.MjData(mj_model)

    results = []
    t0 = time.time()
    for idx, k in enumerate(keys, start=1):
        single = {k: all_motions[k]}
        motion_dur = single[k]["dof"].shape[0] / float(single[k]["fps"])
        try:
            survived, fall = run_one(
                actor, mj_model, mj_data, single, args, args.device,
                kp_used, kd_used,
            )
        except Exception as e:  # noqa: BLE001
            survived, fall = 0.0, f"ERROR {e}"
        status = "SURVIVED" if fall is None else f"FELL ({fall})"
        elapsed = time.time() - t0
        rate = idx / elapsed if elapsed > 0 else 0.0
        eta = (len(keys) - idx) / rate if rate > 0 else 0.0
        print(f"[{idx:>4d}/{len(keys)}] {k:<70s}  "
              f"clip={motion_dur:5.1f}s  surv={survived:5.2f}s  {status}   "
              f"({rate:.1f} clip/s, eta {eta:.0f}s)", flush=True)
        results.append({
            "motion": k,
            "clip_seconds": round(motion_dur, 3),
            "survival_seconds": round(survived, 3),
            "fall_reason": fall or "",
        })

    total = time.time() - t0
    print("\n" + "=" * 80)
    print(f"Benchmarked {len(results)} motions in {total/60:.1f} min")
    survived_n = sum(1 for r in results if r["fall_reason"] == "")
    print(f"  survived to --max-seconds={args.max_seconds:.1f}s: "
          f"{survived_n}/{len(results)} ({100*survived_n/len(results):.1f}%)")

    sorted_res = sorted(results, key=lambda r: -r["survival_seconds"])
    print(f"\n--- Top {min(args.top, len(sorted_res))} most stable ---")
    for r in sorted_res[:args.top]:
        tag = "OK " if r["fall_reason"] == "" else "FELL"
        print(f"  {tag}  {r['survival_seconds']:5.2f}s  {r['motion']}")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["motion", "clip_seconds",
                                              "survival_seconds", "fall_reason"])
            w.writeheader()
            for r in sorted_res:
                w.writerow(r)
        print(f"\nWrote per-motion CSV report → {args.report}")


if __name__ == "__main__":
    main()
