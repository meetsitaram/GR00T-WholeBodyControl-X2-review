"""Rule out training bugs without retraining the kplanner.

The trained kplanner produces gait-amplitude joint motion but does not
follow velocity intent: forward intents drift sideways with strong yaw,
left/right turns are asymmetric, etc (see
``compare_kplanner_vs_heuristic.py`` for the side-by-side). Before
spending 2 days retraining, this script answers two diagnostic
questions in a few minutes:

1. **Behavior grid.** For each of 5 canonical velocity intents
   (idle, forward, backward, turn-left, turn-right), how does the
   model's INTEGRATED output compare to the commanded intent? A
   well-trained model produces output velocity that matches commanded
   velocity within a small tolerance. Wrong-sign / zero / drift-
   dominated output means the intent-to-motion mapping is broken
   end-to-end (either training data didn't cover that intent or
   supervision was buggy).

2. **Training data distribution.** Iterate over every clip in the
   source motion library (``x2_ultra_locowalk.pkl`` by default) and
   compute its mean yaw_rate, vel_x, vel_z, hip_h. Plot a histogram-
   like summary so you can see whether the training set actually
   contains forward walking with zero yaw / pure left turns / pure
   right turns / etc. A model can only learn intents that appear in
   the data; if 90% of clips are left-biased, the model will have a
   left-yaw drift floor like the one we see at inference.

Pass criteria for both:

  * Behavior grid: each row's ``|output - commanded|`` should be small.
    A pass row means the model honors that direction; a fail row
    pinpoints which intent direction the model can't track.
  * Data distribution: each commanded direction should have at least
    a few hundred clips with the matching sign on the relevant axis.
    A "zero coverage" warning on (say) forward-with-zero-yaw means
    the training data didn't teach the model what that looks like.

Output is a single table + plain-English verdict. No fancy plotting
so this runs unattended on any host with the kplanner installed.

Run::

  source .venv/bin/activate && \\
    PYTHONPATH="${PWD}/motionbricks:${PWD}" \\
    python motionbricks/scripts/check_kplanner_training.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "motionbricks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "motionbricks"))


# ---------------------------------------------------------------------------
# Behavior grid: ask the model to honor each canonical intent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentSpec:
    label: str
    intent: tuple[float, float, float, float]  # yaw_rate, vel_x, vel_z, hip_h
    expect_axis: str  # 'yaw' | 'vx' | 'vz' | 'none'
    expect_sign: int  # +1, -1, 0 (0 = expect ~zero motion)


_CANONICAL_INTENTS: list[IntentSpec] = [
    IntentSpec("idle      ", (0.0, 0.0, 0.0, 0.95), "none", 0),
    IntentSpec("fwd_walk  ", (0.0, 0.5, 0.0, 0.95), "vx", +1),
    IntentSpec("back_walk ", (0.0, -0.35, 0.0, 0.95), "vx", -1),
    IntentSpec("turn_left ", (1.5, 0.0, 0.0, 0.95), "yaw", +1),
    IntentSpec("turn_right", (-1.5, 0.0, 0.0, 0.95), "yaw", -1),
    IntentSpec("side_left ", (0.0, 0.0, 0.4, 0.95), "vz", +1),
    IntentSpec("side_right", (0.0, 0.0, -0.4, 0.95), "vz", -1),
]


def _wxyz_to_yaw_unwrapped(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return yaw (rad) per frame, unwrapped so net rotation is recoverable."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.unwrap(yaw)


def _run_intent(
    planner,
    intent: tuple[float, float, float, float],
    n_frames: int,
    device: str,
    warmup_qpos: np.ndarray,
) -> dict[str, float]:
    """Run one velocity intent through the kplanner and measure output.

    Returns net (dx, dy, dyaw_deg) over ``n_frames`` plus avg rates.
    """
    planner.reset(torch.from_numpy(warmup_qpos))
    velocity = torch.tensor(list(intent), dtype=torch.float32, device=device)
    planner.replan_with_velocity(velocity)  # force first replan to use intent

    out = np.zeros((n_frames, 38), dtype=np.float32)
    for i in range(n_frames):
        if planner.should_replan():
            planner.replan_with_velocity(velocity)
        out[i] = planner.get_next_frame().detach().cpu().numpy()

    trans = out[:, :3]
    quat_wxyz = out[:, 3:7]
    yaw_unwrap = _wxyz_to_yaw_unwrapped(quat_wxyz)
    duration_s = (n_frames - 1) / 50.0  # OUTPUT_FPS
    dx = float(trans[-1, 0] - trans[0, 0])
    dy = float(trans[-1, 1] - trans[0, 1])
    dyaw_rad = float(yaw_unwrap[-1] - yaw_unwrap[0])
    return {
        "dx_m": dx,
        "dy_m": dy,
        "dyaw_rad": dyaw_rad,
        "avg_vx": dx / duration_s if duration_s > 0 else 0.0,
        "avg_vy": dy / duration_s if duration_s > 0 else 0.0,
        "avg_yaw_rate": dyaw_rad / duration_s if duration_s > 0 else 0.0,
        # The model's "in body frame" velocity is hard to recover when
        # yaw is also drifting; we report avg world (vx,vy) and let the
        # caller interpret.
    }


def _verdict_for_row(
    spec: IntentSpec,
    out: dict[str, float],
    yaw_tol_rad_s: float,
    trans_tol_m_s: float,
) -> tuple[str, str]:
    """Pass / fail + short reason for a single intent.

    A row PASSES if either:
      * spec.expect_sign is 0 AND all output rates are below tolerance
        (idle stays still).
      * spec.expect_sign matches the SIGN of the relevant output axis
        AND the magnitude is within 50% of commanded (sane direction
        and approximate magnitude).
    """
    yaw = out["avg_yaw_rate"]
    vx = out["avg_vx"]
    vy = out["avg_vy"]
    cmd_yaw, cmd_vx, cmd_vz, _ = spec.intent

    if spec.expect_axis == "none":
        if (
            abs(yaw) < yaw_tol_rad_s
            and abs(vx) < trans_tol_m_s
            and abs(vy) < trans_tol_m_s
        ):
            return "PASS", "idle output stays under tolerance"
        return "FAIL", (
            f"idle commanded but yaw={yaw:+.2f} rad/s, "
            f"(vx,vy)=({vx:+.2f},{vy:+.2f}) m/s — drift floor"
        )

    if spec.expect_axis == "yaw":
        ok_sign = np.sign(yaw) == spec.expect_sign
        ok_mag = abs(yaw) >= 0.5 * abs(cmd_yaw)
        if ok_sign and ok_mag:
            return "PASS", (
                f"yaw_rate={yaw:+.2f} rad/s matches commanded "
                f"{cmd_yaw:+.2f} rad/s"
            )
        return "FAIL", (
            f"yaw_rate={yaw:+.2f} rad/s vs commanded {cmd_yaw:+.2f} "
            f"rad/s (sign_ok={ok_sign}, mag_ok={ok_mag})"
        )

    if spec.expect_axis == "vx":
        ok_sign = np.sign(vx) == spec.expect_sign
        ok_mag = abs(vx) >= 0.3 * abs(cmd_vx)
        if ok_sign and ok_mag:
            return "PASS", (
                f"world vx={vx:+.2f} m/s matches commanded "
                f"vel_x={cmd_vx:+.2f} m/s"
            )
        return "FAIL", (
            f"world vx={vx:+.2f} m/s vs commanded vel_x={cmd_vx:+.2f} "
            f"m/s (sign_ok={ok_sign}, mag_ok={ok_mag})"
        )

    if spec.expect_axis == "vz":
        # Lateral commanded via vel_z (motion-rep Y-up); deploy publishes
        # MuJoCo (Z-up), so the lateral world axis is +Y. Compare
        # ``avg_vy`` to commanded ``vel_z`` for sign.
        ok_sign = np.sign(vy) == spec.expect_sign
        ok_mag = abs(vy) >= 0.3 * abs(cmd_vz)
        if ok_sign and ok_mag:
            return "PASS", (
                f"world vy={vy:+.2f} m/s matches commanded "
                f"vel_z={cmd_vz:+.2f} m/s"
            )
        return "FAIL", (
            f"world vy={vy:+.2f} m/s vs commanded vel_z={cmd_vz:+.2f} "
            f"m/s (sign_ok={ok_sign}, mag_ok={ok_mag})"
        )

    return "FAIL", f"unknown expect_axis={spec.expect_axis!r}"


def _run_behavior_grid(args) -> tuple[list[tuple[IntentSpec, dict, str, str]], str]:
    """Run all canonical intents through the model and collect results."""
    from gear_sonic.scripts.x2_kplanner import _build_default_warmup_qpos
    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths,
        load_x2_planner,
    )

    # Probe device.
    device = args.device
    if device != "cpu":
        if not torch.cuda.is_available():
            print(f"[device] {device!r} requested but CUDA unavailable; using cpu")
            device = "cpu"
        else:
            try:
                _ = (torch.zeros(1, device=device) + 1).cpu()
            except RuntimeError as exc:
                print(f"[device] CUDA probe failed: {exc}; falling back to cpu")
                device = "cpu"

    print(f"[device] using {device}")
    defaults = X2PlannerPaths.default()
    paths = X2PlannerPaths(
        vqvae_ckpt=args.vqvae_ckpt or defaults.vqvae_ckpt,
        pose_ckpt=args.pose_ckpt or defaults.pose_ckpt,
        root_ckpt=args.root_ckpt or defaults.root_ckpt,
        vqvae_version_dir=defaults.vqvae_version_dir,
        pose_version_dir=defaults.pose_version_dir,
        root_version_dir=defaults.root_version_dir,
    )
    print(f"[loader] checkpoints in use:")
    print(f"           vqvae = {paths.vqvae_ckpt}")
    print(f"           pose  = {paths.pose_ckpt}")
    print(f"           root  = {paths.root_ckpt}")
    planner = load_x2_planner(
        paths,
        device=device,
        replan_threshold_frames=16,
    )
    warmup_qpos = _build_default_warmup_qpos()

    results: list[tuple[IntentSpec, dict, str, str]] = []
    n_frames = int(round(args.duration_s * 50.0))
    for spec in _CANONICAL_INTENTS:
        out = _run_intent(
            planner=planner,
            intent=spec.intent,
            n_frames=n_frames,
            device=device,
            warmup_qpos=warmup_qpos,
        )
        verdict, reason = _verdict_for_row(
            spec, out,
            yaw_tol_rad_s=args.idle_yaw_tol,
            trans_tol_m_s=args.idle_trans_tol,
        )
        results.append((spec, out, verdict, reason))
    return results, device


def _print_behavior_grid(results: list[tuple[IntentSpec, dict, str, str]]) -> int:
    """Pretty-print the grid and return the count of failing rows."""
    print("\n" + "=" * 88)
    print("BEHAVIOR GRID — does the model honor each canonical intent?")
    print("=" * 88)
    print(
        f"{'intent':12s} {'cmd (yaw,vx,vz)':>22s} "
        f"{'out (yaw,vx,vy)':>22s}  verdict  reason"
    )
    print("-" * 88)
    n_fail = 0
    for spec, out, verdict, reason in results:
        cmd_str = (
            f"({spec.intent[0]:+.2f},{spec.intent[1]:+.2f},"
            f"{spec.intent[2]:+.2f})"
        )
        out_str = (
            f"({out['avg_yaw_rate']:+.2f},{out['avg_vx']:+.2f},"
            f"{out['avg_vy']:+.2f})"
        )
        if verdict != "PASS":
            n_fail += 1
        print(f"{spec.label} {cmd_str:>22s} {out_str:>22s}  {verdict:7s}  {reason}")
    print("=" * 88)
    return n_fail


# ---------------------------------------------------------------------------
# Training data distribution
# ---------------------------------------------------------------------------


def _clip_velocity_stats(payload: dict, target_fps: int = 30) -> dict[str, float]:
    """Compute per-clip mean yaw_rate (rad/s), vx (m/s), vy (m/s), hip_z (m).

    The training PKL stores ``root_trans_offset`` in world frame (m) and
    ``root_rot`` as a quaternion (x, y, z, w convention from the curator).
    The yaw is the rotation around world +Z in MuJoCo Z-up; differenced
    across frames and scaled by fps gives angular velocity.
    """
    trans = payload["root_trans_offset"]
    rot = payload["root_rot"]
    fps = float(payload.get("fps", target_fps))
    if trans.shape[0] < 2:
        return {
            "mean_yaw_rate_rad_s": 0.0,
            "mean_vx_m_s": 0.0,
            "mean_vy_m_s": 0.0,
            "mean_hip_z_m": float(trans[0, 2]) if trans.shape[0] else 0.0,
            "n_frames": int(trans.shape[0]),
        }
    # Quat xyzw -> yaw around +Z.
    x = rot[:, 0]
    y = rot[:, 1]
    z = rot[:, 2]
    w = rot[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw = np.unwrap(yaw)
    dyaw_total = float(yaw[-1] - yaw[0])
    dx = float(trans[-1, 0] - trans[0, 0])
    dy = float(trans[-1, 1] - trans[0, 1])
    duration_s = (trans.shape[0] - 1) / fps if fps > 0 else 0.0
    if duration_s <= 0:
        return {
            "mean_yaw_rate_rad_s": 0.0,
            "mean_vx_m_s": 0.0,
            "mean_vy_m_s": 0.0,
            "mean_hip_z_m": float(trans[:, 2].mean()),
            "n_frames": int(trans.shape[0]),
        }
    return {
        "mean_yaw_rate_rad_s": dyaw_total / duration_s,
        "mean_vx_m_s": dx / duration_s,
        "mean_vy_m_s": dy / duration_s,
        "mean_hip_z_m": float(trans[:, 2].mean()),
        "n_frames": int(trans.shape[0]),
    }


def _bucketize(values: np.ndarray, edges: list[float]) -> list[int]:
    """Return per-bucket counts for ``values`` partitioned by ``edges``."""
    edges_full = [-np.inf, *edges, np.inf]
    counts = []
    for lo, hi in zip(edges_full[:-1], edges_full[1:]):
        counts.append(int(np.logical_and(values >= lo, values < hi).sum()))
    return counts


def _print_distribution(stats: list[dict[str, float]]) -> int:
    """Pretty-print distribution histograms and return # of coverage gaps."""
    yaw_rates = np.array([s["mean_yaw_rate_rad_s"] for s in stats])
    vxs = np.array([s["mean_vx_m_s"] for s in stats])
    vys = np.array([s["mean_vy_m_s"] for s in stats])
    print("\n" + "=" * 88)
    print(f"TRAINING DATA DISTRIBUTION — {len(stats)} clips")
    print("=" * 88)

    def _row(name: str, vals: np.ndarray, edges: list[float], unit: str) -> int:
        counts = _bucketize(vals, edges)
        labels = (
            [f"<{edges[0]:+.2f}"]
            + [f"[{lo:+.2f},{hi:+.2f})"
               for lo, hi in zip(edges[:-1], edges[1:])]
            + [f">={edges[-1]:+.2f}"]
        )
        print(f"\n  {name} ({unit}):")
        print(f"    mean={vals.mean():+.3f}  median={np.median(vals):+.3f}  "
              f"stdev={vals.std():.3f}  "
              f"min={vals.min():+.3f}  max={vals.max():+.3f}")
        gaps = 0
        for label, c in zip(labels, counts):
            bar = "#" * min(50, c // max(1, len(stats) // 100))
            pct = 100.0 * c / max(1, len(stats))
            warn = ""
            if c < max(20, len(stats) // 500):
                warn = "  <-- UNDERREPRESENTED"
                gaps += 1
            print(f"      {label:>18s} : {c:6d} ({pct:5.2f}%) {bar}{warn}")
        return gaps

    n_gaps = 0
    n_gaps += _row(
        "mean yaw_rate", yaw_rates,
        [-0.5, -0.1, 0.1, 0.5], "rad/s",
    )
    n_gaps += _row(
        "mean vel_x (world)", vxs,
        [-0.2, -0.05, 0.05, 0.2], "m/s",
    )
    n_gaps += _row(
        "mean vel_y (world, lateral)", vys,
        [-0.2, -0.05, 0.05, 0.2], "m/s",
    )
    print("=" * 88)
    return n_gaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--motion-lib-pkl", type=Path,
        default=REPO_ROOT / "gear_sonic" / "data" / "motions"
                / "x2_ultra_locowalk.pkl",
        help="source motion library the kplanner was trained on",
    )
    p.add_argument(
        "--duration-s", type=float, default=2.0,
        help="how long to integrate each behavior-grid intent",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--vqvae-ckpt", type=Path, default=None,
        help=(
            "override VQVAE checkpoint. Default = X2PlannerPaths.default() "
            "(current pinned step). Useful for testing intermediate "
            "checkpoints during a 200k retrain: pass step=125k, 150k, "
            "175k to watch the behavior grid converge before committing "
            "to the full 200k."
        ),
    )
    p.add_argument(
        "--pose-ckpt", type=Path, default=None,
        help="override pose checkpoint; see --vqvae-ckpt.",
    )
    p.add_argument(
        "--root-ckpt", type=Path, default=None,
        help="override root checkpoint; see --vqvae-ckpt.",
    )
    p.add_argument(
        "--idle-yaw-tol", type=float, default=0.1,
        help="rad/s; idle row passes only if |yaw_rate| stays below this",
    )
    p.add_argument(
        "--idle-trans-tol", type=float, default=0.1,
        help="m/s; idle row passes only if |vx|,|vy| stay below this",
    )
    p.add_argument(
        "--skip-behavior", action="store_true",
        help="skip the kplanner inference grid (data distribution only)",
    )
    p.add_argument(
        "--skip-distribution", action="store_true",
        help="skip the training data distribution (behavior grid only)",
    )
    p.add_argument(
        "--json-out", type=Path, default=None,
        help="optional path to dump the full results (for CI / regression)",
    )
    args = p.parse_args()

    full_results: dict = {"args": {k: str(v) for k, v in vars(args).items()}}

    n_fail_behavior = 0
    behavior_summary: list[dict] = []
    if not args.skip_behavior:
        results, device_used = _run_behavior_grid(args)
        n_fail_behavior = _print_behavior_grid(results)
        for spec, out, verdict, reason in results:
            behavior_summary.append({
                "label": spec.label.strip(),
                "intent": list(spec.intent),
                "out": out,
                "verdict": verdict,
                "reason": reason,
            })
        full_results["behavior"] = {
            "device_used": device_used,
            "rows": behavior_summary,
            "n_fail": n_fail_behavior,
        }

    n_gaps_distribution = 0
    if not args.skip_distribution:
        print(f"\n[distribution] loading {args.motion_lib_pkl} ...")
        raw = joblib.load(args.motion_lib_pkl)
        print(f"[distribution] {len(raw)} clips loaded")
        stats = [_clip_velocity_stats(payload) for payload in raw.values()]
        n_gaps_distribution = _print_distribution(stats)
        full_results["distribution"] = {
            "n_clips": len(stats),
            "n_underrepresented_buckets": n_gaps_distribution,
        }

    print("\n" + "=" * 88)
    print("OVERALL VERDICT")
    print("=" * 88)
    if not args.skip_behavior:
        if n_fail_behavior == 0:
            print(
                "  [behavior]    PASS  the model honors every canonical "
                "intent. If teleop still misbehaves, the bug is in the "
                "inference plumbing (manager / canonicalize / publish), "
                "not in the trained weights."
            )
        else:
            print(
                f"  [behavior]    FAIL  {n_fail_behavior} of "
                f"{len(_CANONICAL_INTENTS)} intents are not honored. "
                "The trained model has not learned the intent surface."
            )
    if not args.skip_distribution:
        if n_gaps_distribution == 0:
            print(
                "  [data]        PASS  every velocity bucket has decent "
                "training-set coverage. The data balance is not the "
                "limiting factor; suspect training hyperparameters / "
                "loss formulation / undertrained checkpoints instead."
            )
        else:
            print(
                f"  [data]        FAIL  {n_gaps_distribution} velocity "
                "buckets are underrepresented in the training set. The "
                "model literally cannot learn intents it never saw. Add "
                "more clips covering those buckets and retrain."
            )
    print("=" * 88)
    print("Next steps:")
    if not args.skip_behavior and n_fail_behavior > 0:
        print(
            "  * Re-examine the failing intents above. If forward+backward\n"
            "    BOTH fail with same magnitude offset, suspect a sign bug.\n"
            "    If only ONE direction works, suspect data imbalance.\n"
            "  * Check the [data] section: does the training set actually\n"
            "    contain clips matching the failing intent direction?"
        )
    if not args.skip_distribution and n_gaps_distribution > 0:
        print(
            "  * Curate or generate additional clips covering the\n"
            "    underrepresented buckets. Re-train (the 2-day run) only\n"
            "    AFTER the data spans the full intent surface."
        )
    if (
        not args.skip_behavior
        and n_fail_behavior == 0
        and not args.skip_distribution
        and n_gaps_distribution == 0
    ):
        print(
            "  * Both gates pass. The training is healthy. If teleop\n"
            "    still misbehaves, instrument the inference path:\n"
            "    canonicalize / uncanonicalize axis conventions, the\n"
            "    velocity intent vector sent to the model, and the\n"
            "    quat permutation at publish (wxyz vs xyzw)."
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(full_results, indent=2, default=str))
        print(f"\n[json] full results written to {args.json_out}")

    return 0 if (n_fail_behavior == 0 and n_gaps_distribution == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
