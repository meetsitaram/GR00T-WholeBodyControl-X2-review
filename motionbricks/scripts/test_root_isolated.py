"""Per-model isolation test for the root model.

Loads the X2 (or G1) MotionBricks stack via the production loader, seeds
``NeuralPlannerCore`` with a real motion clip, then issues velocity
intents and reads back BOTH:

  * The **root model's own prediction** of the future global root trajectory
    (captured by monkey-patching ``motion_inference._predict_root_trajectories``
    so we see exactly what the root backbone produced before pose+VQVAE
    consume it).
  * The **end-to-end integrated MuJoCo qpos** the full planner publishes
    (i.e. what ``x2_kplanner.py`` would push on the wire).

That gives us both layers from the plan:

  * **Diagnostic (physical units)** — body-frame forward / lateral / dyaw /
    hip-Z drift on this clip. Answers "does the robot walk?" on its own.
  * **Comparison (dimensionless)** — slope = forward_m / (vx * horizon_s),
    tracking_ratio = achieved / commanded. Should approach 1.0 on a
    properly-trained stack regardless of skeleton. This is the axis we
    compare X2 vs G1 on.

Modes:

  * ``single`` — one velocity intent, one report.
  * ``cold_sweep`` — vx ∈ {0.0, 0.2, 0.4, 0.6, 0.8} m/s from a stationary
    context. Tabulates forward_m vs commanded. **A flat curve = root
    model can't initiate motion**, the exact failure from prior
    sessions.
  * ``warm_sweep`` — same intent grid but seeded from a mid-walk context
    (frame 50 of a walking clip). Tells us whether the model can extend
    an existing trajectory even if it can't cold-start.

The script is canonicalized to **body-frame** displacement: regardless of
the seed clip's world orientation, forward = +X in the robot's start
frame. So a forward-walking clip facing world -Y produces
``pred_forward_m`` ≈ +2.6 m, not "+dy or -dy" depending on how the clip
was recorded.

Usage::

    PYTHONPATH="${PWD}/motionbricks:${PWD}" python motionbricks/scripts/test_root_isolated.py \\
        --ckpt-set x2 \\
        --fixture walking \\
        --mode cold_sweep \\
        --save-npz out/per_model_report/root_x2_walking_cold.npz \\
        --report-json out/per_model_report/root_x2_walking_cold.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "motionbricks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "motionbricks"))


# ---------------------------------------------------------------------------
# Fixture registry. Plan-fixed canonical X2 clips; G1 fixtures map to clip
# indices inside G1-clip.ckpt (resolved at load time, no PKL required).
# ---------------------------------------------------------------------------


FIXTURES = {
    "x2": {
        "walking": {
            "kind": "x2_pkl",
            "pkl": "gear_sonic/data/motions/x2_ultra_locowalk.pkl",
            "clip_key": "Loop_Forward_Walk_001__A018",
        },
        "stationary": {
            "kind": "x2_pkl",
            "pkl": "gear_sonic/data/motions/x2_ultra_locowalk.pkl",
            "clip_key": "Idle_Right_001__A019",
        },
    },
    "g1": {
        "walking": {
            "kind": "g1_clip",
            "g1_clip_path": "motionbricks/out/G1-clip.ckpt",
            "clip_idx": 11,
        },
        "stationary": {
            "kind": "g1_clip",
            "g1_clip_path": "motionbricks/out/G1-clip.ckpt",
            "clip_idx": 0,
        },
    },
}


# Velocity-sweep grid (m/s for vel_x; fixed at yaw_rate=0, vel_z=0).
SWEEP_VX = (0.0, 0.2, 0.4, 0.6, 0.8)


# ---------------------------------------------------------------------------
# Quaternion + yaw helpers (copied from replay_pkl_through_kplanner.py so this
# script has no extra import surface). MuJoCo qpos is wxyz; PKL root_rot is
# xyzw -- match the convention each consumer expects.
# ---------------------------------------------------------------------------


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        axis=-1,
    )


def _wxyz_to_yaw_rad(quat_wxyz: np.ndarray) -> np.ndarray:
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _project_world_to_body(dx_w: float, dy_w: float, yaw0_rad: float) -> tuple[float, float]:
    """Rotate world XY displacement back into the body frame at yaw0."""
    c, s = np.cos(-yaw0_rad), np.sin(-yaw0_rad)
    return c * dx_w - s * dy_w, s * dx_w + c * dy_w


# ---------------------------------------------------------------------------
# Fixture loaders -> seed qpos
# ---------------------------------------------------------------------------


def _load_x2_fixture_qpos(pkl_path: Path, clip_key: str) -> tuple[np.ndarray, float]:
    """Build [T, 38] MuJoCo qpos from an X2 PKL clip."""
    payload = joblib.load(pkl_path)[clip_key]
    trans = np.asarray(payload["root_trans_offset"], dtype=np.float32)
    rot_xyzw = np.asarray(payload["root_rot"], dtype=np.float32)
    dof = np.asarray(payload["dof"], dtype=np.float32)
    rot_wxyz = _quat_xyzw_to_wxyz(rot_xyzw)
    qpos = np.concatenate([trans, rot_wxyz, dof], axis=-1).astype(np.float32)
    fps = float(payload.get("fps", 30))
    return qpos, fps


def _load_g1_fixture_qpos(g1_clip_path: Path, clip_idx: int) -> tuple[np.ndarray, float]:
    """Read [T, 36] G1 MuJoCo qpos from G1-clip.ckpt.

    G1-clip.ckpt is the `clip_holder_G1` state_dict that
    `interactive_demo_g1.py` consumes; it stores qpos as wxyz quat
    already, so no permutation is needed. ``num_frames_per_clip[i]``
    tells us how many of the 150 slots are real.
    """
    sd = torch.load(g1_clip_path, map_location="cpu", weights_only=False)
    qpos_all = sd["mujoco_qpos"].numpy()  # [N, 150, 36]
    nfpc = sd["num_frames_per_clip"].numpy()  # [N]
    if clip_idx < 0 or clip_idx >= qpos_all.shape[0]:
        raise IndexError(f"G1 clip_idx={clip_idx} out of range [0, {qpos_all.shape[0]})")
    n = int(nfpc[clip_idx])
    return qpos_all[clip_idx, :n].astype(np.float32), 30.0


# ---------------------------------------------------------------------------
# Planner load (X2 today; G1 is a follow-up phase)
# ---------------------------------------------------------------------------


def _load_planner(ckpt_set: str, device: str):
    if ckpt_set == "x2":
        from motionbricks.motion_backbone.inference.load_x2_planner import (
            X2PlannerPaths,
            load_x2_planner,
        )
        paths = X2PlannerPaths.default()
        paths.validate()
        return load_x2_planner(paths, device=device)
    if ckpt_set == "g1":
        from motionbricks.motion_backbone.inference.load_g1_planner import (
            G1PlannerPaths,
            load_g1_planner,
        )
        paths = G1PlannerPaths.default()
        paths.validate()
        return load_g1_planner(paths, device=device)
    raise ValueError(f"Unknown ckpt-set: {ckpt_set!r}")


# ---------------------------------------------------------------------------
# Root-output capture
# ---------------------------------------------------------------------------


def _install_root_capture(inferencer):
    """Wrap inferencer._predict_root_trajectories so we can read out the
    normalized root predictions and their context inputs from the outside.

    Returns the dict that gets populated on every predict() call. Caller
    should ``del captured[k]`` between sweep steps if memory matters.
    """
    captured: dict = {}
    orig = inferencer._predict_root_trajectories

    def wrapped(batch, config):
        out = orig(batch, config)
        captured["root_input_global_normalized"] = batch["global_root_values"].detach().cpu()
        captured["root_input_local_normalized"] = batch["local_root_values"].detach().cpu()
        captured["pred_num_tokens"] = out[0].detach().cpu()
        captured["pred_global_root_values_normalized"] = out[1].detach().cpu()
        captured["pred_local_root_values_normalized"] = out[2].detach().cpu()
        return out

    inferencer._predict_root_trajectories = wrapped
    return captured


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _unnormalize_global_root(captured: dict, inferencer) -> np.ndarray:
    """Return predicted global root values in physical units (m, cos/sin heading).

    Output shape: [T_predicted, 5] = [x, y_up, z, cos_heading, sin_heading]
    where T_predicted = pred_num_tokens * 4.
    """
    pred_norm = captured["pred_global_root_values_normalized"]  # [1, max_T, 5]
    num_tok = int(captured["pred_num_tokens"][0])
    horizon_frames = num_tok * 4
    pred_unnorm = inferencer.global_motion_rep.unnormalize(pred_norm).numpy()
    return pred_unnorm[0, :horizon_frames]  # [T, 5]


def _root_metrics_from_canonical_pred(
    pred_global: np.ndarray, fps: float, intent: dict
) -> dict:
    """Compute physical-unit and dimensionless metrics from the root output.

    The root model's input is recentered (xz starts at 0,0) and
    canonicalized (initial heading = 0), so the output ``pred_global``
    is already in body-frame coordinates: +X = forward, +Z = lateral
    left, +Y = vertical.

    Args:
        pred_global: [T, 5] = [x, y_up, z, cos_heading, sin_heading]
        fps: planner output FPS (typically 30)
        intent: dict with vel_x, vel_z, yaw_rate, hip_h (commanded)

    Returns:
        Dict of metrics keyed by name.
    """
    horizon_s = pred_global.shape[0] / float(fps)
    # Physical (body frame -- input was canonicalized to face +X).
    pred_forward_m = float(pred_global[-1, 0] - pred_global[0, 0])
    pred_lateral_m = float(pred_global[-1, 2] - pred_global[0, 2])
    pred_height_m = float(pred_global[-1, 1] - pred_global[0, 1])
    final_yaw = float(np.arctan2(pred_global[-1, 4], pred_global[-1, 3]))
    initial_yaw = float(np.arctan2(pred_global[0, 4], pred_global[0, 3]))
    # Unwrap delta to (-pi, pi).
    dyaw = (final_yaw - initial_yaw + np.pi) % (2.0 * np.pi) - np.pi
    pred_dyaw_deg = float(np.degrees(dyaw))

    # Dimensionless: slope = forward / (commanded vx * horizon).
    # For a perfectly tracking root model, slope -> 1.0.
    commanded_dy = float(intent["vel_x"]) * horizon_s
    slope_forward = (
        pred_forward_m / commanded_dy
        if abs(commanded_dy) > 1e-6
        else None
    )

    return {
        "horizon_s": horizon_s,
        "horizon_frames": int(pred_global.shape[0]),
        "pred_forward_m": pred_forward_m,
        "pred_lateral_m": pred_lateral_m,
        "pred_height_m": pred_height_m,
        "pred_dyaw_deg": pred_dyaw_deg,
        "slope_forward": slope_forward,
    }


def _e2e_metrics_from_qpos(
    integrated_qpos: np.ndarray, intent: dict, fps: float
) -> dict:
    """End-to-end metrics from the planner's published MuJoCo qpos.

    Unlike the root-only metric, this comes AFTER pose+VQVAE decode and
    AFTER uncanonicalization. To make body-frame metrics meaningful we
    project the world-frame XY delta back to the body frame using the
    initial yaw of the integrated trajectory.
    """
    if integrated_qpos.shape[0] < 2:
        return {}
    horizon_s = (integrated_qpos.shape[0] - 1) / float(fps)
    trans = integrated_qpos[:, :3]
    quat_wxyz = integrated_qpos[:, 3:7]
    yaw = _wxyz_to_yaw_rad(quat_wxyz)
    yaw_unwrap = np.unwrap(yaw)
    yaw0 = float(yaw_unwrap[0])
    dx_w = float(trans[-1, 0] - trans[0, 0])
    dy_w = float(trans[-1, 1] - trans[0, 1])
    forward_body, lateral_body = _project_world_to_body(dx_w, dy_w, yaw0)
    hip_z_mean = float(trans[:, 2].mean())
    hip_z_std = float(trans[:, 2].std())
    achieved_dyaw_deg = float(np.degrees(yaw_unwrap[-1] - yaw_unwrap[0]))

    commanded_dy = float(intent["vel_x"]) * horizon_s
    tracking_ratio = (
        float(forward_body / commanded_dy)
        if abs(commanded_dy) > 1e-6
        else None
    )

    return {
        "horizon_s": horizon_s,
        "horizon_frames": int(integrated_qpos.shape[0]),
        "achieved_forward_m": float(forward_body),
        "achieved_lateral_m": float(lateral_body),
        "achieved_dyaw_deg": achieved_dyaw_deg,
        "hip_z_mean_m": hip_z_mean,
        "hip_z_std_m": hip_z_std,
        "tracking_ratio_forward": tracking_ratio,
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def _run_one_intent(
    planner,
    captured: dict,
    seed_qpos_np: np.ndarray,
    seed_frame_offset: int,
    intent: dict,
    fps: float,
    device: str,
    inferencer,
) -> dict:
    """Reset the planner from a seed window, run one replan, collect metrics."""
    # Seed buffer with a contiguous window from the clip starting at seed_frame_offset.
    # NUM_MIN_FRAMES_IN_BUFFER = 64 in NeuralPlannerCore; reset() will tile if shorter.
    n_avail = seed_qpos_np.shape[0] - seed_frame_offset
    seed_window = seed_qpos_np[seed_frame_offset : seed_frame_offset + min(n_avail, 64)]
    seed_t = torch.from_numpy(seed_window).to(device)
    planner.reset(seed_t)

    intent_t = torch.tensor(
        [intent["yaw_rate"], intent["vel_x"], intent["vel_z"], intent["hip_h"]],
        device=device,
        dtype=torch.float32,
    )
    planner.replan_with_velocity(intent_t)

    # Root-only metrics (from captured root output).
    pred_global = _unnormalize_global_root(captured, inferencer)
    root_only = _root_metrics_from_canonical_pred(pred_global, fps, intent)

    # End-to-end metrics from the integrated qpos buffer.
    integrated = planner.frames["mujoco_qpos"][0].detach().cpu().numpy()
    e2e = _e2e_metrics_from_qpos(integrated, intent, fps)

    return {
        "intent": intent,
        "root_only": root_only,
        "e2e": e2e,
        "_pred_global_root_unnorm": pred_global,
        "_integrated_qpos": integrated,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-model isolation test for the root model.",
    )
    p.add_argument(
        "--ckpt-set",
        choices=("x2", "g1"),
        default="x2",
        help="Which trained stack to load. G1 is a follow-up step; only X2 is wired today.",
    )
    p.add_argument(
        "--fixture",
        choices=("walking", "stationary"),
        default="walking",
        help="Canonical fixture. Walking: forward-locomotion clip. Stationary: idle clip.",
    )
    p.add_argument(
        "--mode",
        choices=("single", "cold_sweep", "warm_sweep"),
        default="single",
        help=(
            "single: one intent (use --vx / --vz / --yaw-rate / --hip-h). "
            "cold_sweep: vx in (0.0, 0.2, 0.4, 0.6, 0.8) m/s from frame 0 of fixture. "
            "warm_sweep: same intent grid but seeded from frame 50 of fixture (mid-walk)."
        ),
    )
    p.add_argument("--vx", type=float, default=0.4, help="Forward velocity intent (m/s).")
    p.add_argument("--vz", type=float, default=0.0, help="Lateral velocity intent (m/s).")
    p.add_argument(
        "--yaw-rate", type=float, default=0.0, help="Yaw rate intent (rad/s)."
    )
    p.add_argument(
        "--hip-h", type=float, default=None,
        help="Hip-height intent (m). Default: clip's mean hip Z over the seed window.",
    )
    p.add_argument(
        "--seed-frame", type=int, default=0,
        help="Frame index of fixture clip to use as seed start (overrides mode defaults).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--save-npz", type=Path, default=None,
        help="Optional output NPZ with the full predicted root + integrated qpos.",
    )
    p.add_argument(
        "--report-json", type=Path, default=None,
        help="Optional output JSON with the metrics tables for the runner to consume.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    fixture_spec = FIXTURES[args.ckpt_set][args.fixture]
    if fixture_spec["kind"] == "x2_pkl":
        pkl = REPO_ROOT / fixture_spec["pkl"]
        clip_key = fixture_spec["clip_key"]
        print(f"[fixture] {args.ckpt_set} {args.fixture}: {pkl.name}::{clip_key}")
        seed_qpos_np, fps = _load_x2_fixture_qpos(pkl, clip_key)
    else:
        g1_path = REPO_ROOT / fixture_spec["g1_clip_path"]
        idx = fixture_spec["clip_idx"]
        print(f"[fixture] {args.ckpt_set} {args.fixture}: {g1_path.name}[{idx}]")
        seed_qpos_np, fps = _load_g1_fixture_qpos(g1_path, idx)
    print(f"[fixture] T={seed_qpos_np.shape[0]} frames @ {fps} fps "
          f"(dur={seed_qpos_np.shape[0] / fps:.2f} s, qpos dim={seed_qpos_np.shape[-1]})")

    print(f"[load] ckpt-set={args.ckpt_set} on device={args.device}")
    planner = _load_planner(args.ckpt_set, args.device)
    inferencer = planner._inferencer
    captured = _install_root_capture(inferencer)

    # Determine sweep + seed offsets per mode.
    if args.mode == "single":
        intents = [
            {
                "yaw_rate": args.yaw_rate,
                "vel_x": args.vx,
                "vel_z": args.vz,
                "hip_h": args.hip_h
                if args.hip_h is not None
                else float(seed_qpos_np[: max(1, args.seed_frame + 4), 2].mean()),
            }
        ]
        seed_offset = args.seed_frame
    elif args.mode == "cold_sweep":
        # Cold start: always seed from frame 0 (or wherever user pointed).
        seed_offset = args.seed_frame
        hip_h = (
            args.hip_h if args.hip_h is not None
            else float(seed_qpos_np[: 4, 2].mean())
        )
        intents = [
            {"yaw_rate": 0.0, "vel_x": vx, "vel_z": 0.0, "hip_h": hip_h}
            for vx in SWEEP_VX
        ]
    elif args.mode == "warm_sweep":
        # Mid-clip: prefer frame 50 if available, else fall back to clip midpoint.
        T = seed_qpos_np.shape[0]
        seed_offset = args.seed_frame if args.seed_frame > 0 else min(50, max(0, T // 2))
        if seed_offset + 4 > T:
            seed_offset = max(0, T - 4)
        hip_h = (
            args.hip_h if args.hip_h is not None
            else float(seed_qpos_np[seed_offset : seed_offset + 4, 2].mean())
        )
        intents = [
            {"yaw_rate": 0.0, "vel_x": vx, "vel_z": 0.0, "hip_h": hip_h}
            for vx in SWEEP_VX
        ]
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    print(f"[mode] {args.mode} seed_offset={seed_offset} n_intents={len(intents)}")

    rows = []
    raw_per_intent: list[dict] = []
    for i, intent in enumerate(intents):
        # Clear capture between intents so a stale read can't fool us.
        captured.clear()
        row = _run_one_intent(
            planner, captured, seed_qpos_np, seed_offset, intent, fps, args.device, inferencer
        )
        raw_per_intent.append(row)
        rows.append({"intent": row["intent"], "root_only": row["root_only"], "e2e": row["e2e"]})

    # Pretty stdout: a focused two-column table.
    print()
    print("=" * 92)
    print(f"  Root-model isolation report  (ckpt={args.ckpt_set}, "
          f"fixture={args.fixture}, mode={args.mode})")
    print("=" * 92)
    print(f"  {'vx_cmd':>7} | {'fwd_m':>7} {'lat_m':>7} {'dyaw_deg':>9} "
          f"{'slope':>7} | {'e2e_fwd':>8} {'tracking':>9} {'hipZ_std_cm':>11}")
    print(f"  {'-'*7:>7} | {'-'*7:>7} {'-'*7:>7} {'-'*9:>9} {'-'*7:>7} | "
          f"{'-'*8:>8} {'-'*9:>9} {'-'*11:>11}")
    for r in rows:
        vx = r["intent"]["vel_x"]
        ro = r["root_only"]
        e2e = r["e2e"]
        slope_str = f"{ro['slope_forward']:.2f}" if ro.get("slope_forward") is not None else "  --"
        tr_str = f"{e2e['tracking_ratio_forward']:.2f}" if e2e.get("tracking_ratio_forward") is not None else "  --"
        print(
            f"  {vx:>7.2f} | {ro['pred_forward_m']:>7.3f} {ro['pred_lateral_m']:>7.3f} "
            f"{ro['pred_dyaw_deg']:>9.2f} {slope_str:>7} | "
            f"{e2e.get('achieved_forward_m', float('nan')):>8.3f} {tr_str:>9} "
            f"{e2e.get('hip_z_std_m', float('nan')) * 100:>11.2f}"
        )
    print("=" * 92)
    print(
        "  fwd_m / lat_m / dyaw_deg : root-model output (body frame, canonicalized)\n"
        "  slope          : pred_forward_m / (vx_cmd * horizon_s); ideal ~1.0\n"
        "  e2e_fwd        : full-stack integrated forward distance (body frame)\n"
        "  tracking       : achieved_forward_m / commanded_dy; ideal ~1.0\n"
        "  hipZ_std_cm    : hip-Z drift, falling-over alarm"
    )

    # Save report + NPZ.
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "ckpt_set": args.ckpt_set,
            "fixture": args.fixture,
            "mode": args.mode,
            "fps": fps,
            "seed_offset": seed_offset,
            "fixture_spec": fixture_spec,
            "sweep": rows,
        }
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {args.report_json}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        # Pad ragged predictions to a uniform [N_intents, T_max, ...] array.
        T_max_root = max(r["_pred_global_root_unnorm"].shape[0] for r in raw_per_intent)
        T_max_e2e = max(r["_integrated_qpos"].shape[0] for r in raw_per_intent)
        pred_root = np.full(
            (len(raw_per_intent), T_max_root, 5), np.nan, dtype=np.float32
        )
        pred_e2e = np.full(
            (len(raw_per_intent), T_max_e2e, raw_per_intent[0]["_integrated_qpos"].shape[1]),
            np.nan,
            dtype=np.float32,
        )
        for i, r in enumerate(raw_per_intent):
            t_r = r["_pred_global_root_unnorm"].shape[0]
            t_e = r["_integrated_qpos"].shape[0]
            pred_root[i, :t_r] = r["_pred_global_root_unnorm"]
            pred_e2e[i, :t_e] = r["_integrated_qpos"]
        np.savez(
            args.save_npz,
            ckpt_set=args.ckpt_set,
            fixture=args.fixture,
            mode=args.mode,
            fps=fps,
            intents=np.array([
                [r["intent"]["yaw_rate"], r["intent"]["vel_x"],
                 r["intent"]["vel_z"], r["intent"]["hip_h"]]
                for r in raw_per_intent
            ], dtype=np.float32),
            pred_global_root_body_frame=pred_root,
            integrated_qpos_world_frame=pred_e2e,
            seed_qpos_world_frame=seed_qpos_np[seed_offset : seed_offset + 8],
        )
        print(f"[npz]    wrote {args.save_npz}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
