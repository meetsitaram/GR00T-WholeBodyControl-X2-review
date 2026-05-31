"""Side-by-side diagnostic: kplanner forward-walk output vs heuristic clip.

Goal: when the heuristic planner moves the robot correctly but the
kplanner spins / freezes / bends the torso for the same intent, narrow
the root cause to one of three regions:

  1. Joint reference quality. The neural model's predicted joint
     angles for ``fwd_step`` may be out-of-distribution for the SONIC
     policy (timing, amplitude, phase). Symptom: kplanner per-joint
     range / mean differs sharply from the curated ``fwd_walk_standard``
     clip the policy is known to track.
  2. Root-yaw drift. The model's predicted root_quat may drift
     cumulatively across replans, making the published reference yaw
     rotate even when commanded yaw_rate = 0. Symptom: kplanner net
     dyaw over 1 s is large (>5 deg) while the heuristic clip has
     dyaw ~0.
  3. Root-XY consistency. The kplanner's published root_trans may
     deviate from the integrated commanded velocity, which the deploy
     uses as the world-frame target. Symptom: kplanner dx over 1 s is
     near zero (model produced stepping joints but no integrated
     translation) while heuristic has dx ~0.5 m.

The script does NOT run the deploy or the SONIC policy. It only checks
what the kplanner *would publish* against what the heuristic publishes.

Run::

  python -m motionbricks.scripts.compare_kplanner_vs_heuristic \
      --intent-yaw-rate 0.0 \
      --intent-vel-x 0.5 \
      --duration-s 1.0

or for a turn-left::

  python -m motionbricks.scripts.compare_kplanner_vs_heuristic \
      --intent-yaw-rate 1.5 \
      --intent-vel-x 0.0 \
      --heuristic-bin turn_left_deg_45

The script prints per-joint min / max / mean for both sources plus the
yaw drift and XY trajectory of each, so a one-line diff makes the
failure region obvious.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "motionbricks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "motionbricks"))


def _quat_wxyz_to_yaw_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Extract yaw (rotation around +Z in MuJoCo) from a wxyz quat batch.

    quat_wxyz: (T, 4) array. Returns (T,) in degrees, wrapped to [-180, 180).
    """
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    # yaw = atan2(2 (w z + x y), 1 - 2 (y^2 + z^2)).
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad = np.arctan2(sin_yaw, cos_yaw)
    return np.degrees(yaw_rad)


def _quat_xyzw_to_yaw_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    """Same as ``_quat_wxyz_to_yaw_deg`` but for xyzw-ordered quats."""
    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    z = quat_xyzw[:, 2]
    w = quat_xyzw[:, 3]
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad = np.arctan2(sin_yaw, cos_yaw)
    return np.degrees(yaw_rad)


def _resolve_device(requested: str) -> str:
    """Validate ``requested`` torch device, falling back to cpu when needed.

    The kplanner is happiest on a CUDA GPU with kernels matching the
    installed driver's compute capability (sm_120 for RTX 5090), but
    the diagnostic should also be usable under interpreters that
    weren't built with the right wheel. Detect the mismatch up front
    and fall back to ``cpu`` with a loud notice so the user doesn't
    chase a ``no kernel image is available`` traceback halfway into a
    50-frame loop.
    """
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print(f"[device] {requested!r} requested but torch.cuda.is_available() "
              "is False; falling back to CPU.")
        return "cpu"
    try:
        # Allocating a tiny tensor on the device and running a kernel
        # is the cheapest way to surface sm_120-vs-PyTorch wheel
        # mismatches before the real prediction starts.
        probe = torch.zeros(1, device=requested)
        _ = (probe + 1).cpu()
    except RuntimeError as exc:
        print(f"[device] CUDA probe failed on {requested!r}: {exc}")
        print("[device] Likely a PyTorch wheel without sm_120 kernels. "
              "Falling back to CPU. To use the GPU, activate the "
              "kplanner's python (e.g. `source .venv/bin/activate` or "
              "`~/miniconda3/envs/env_isaaclab/bin/python ...`).")
        return "cpu"
    return requested


def _run_kplanner_forward(
    warmup_qpos: np.ndarray,
    intent: tuple[float, float, float, float],
    n_frames: int,
    device: str,
) -> np.ndarray:
    """Predict ``n_frames`` of forward-walk qpos via the trained kplanner.

    Returns an ``(n_frames, 38)`` array: ``[root_xyz(3), root_wxyz(4),
    joints(31)]`` per frame. This is exactly what the daemon publishes
    out of the ring buffer at 50 Hz, so the joint slice [7:] is
    directly comparable to ``Primitive.dof`` from the curator PKL.
    """
    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths,
        load_x2_planner,
    )

    paths = X2PlannerPaths.default()
    planner = load_x2_planner(paths, device=device, replan_threshold_frames=16)
    planner.reset(torch.from_numpy(warmup_qpos))

    out = np.zeros((n_frames, 38), dtype=np.float32)
    velocity = torch.tensor(list(intent), dtype=torch.float32, device=device)

    # ``reset`` tiles ``warmup_qpos`` to ``NUM_MIN_FRAMES_IN_BUFFER`` (64)
    # copies, so ``frames_remaining`` starts HIGH (64) right out of
    # reset. The daemon's worker thread only fires ``replan_with_velocity``
    # when ``frames_remaining <= REPLAN_THRESHOLD_FRAMES`` (16). If we
    # just pop in a loop without forcing a replan, the first ~48 pops
    # return the tiled stand pose, not the model's prediction. Force
    # one replan at the start so the buffer's content reflects the
    # ``velocity`` intent we're trying to diagnose; subsequent
    # threshold-based replans keep it fresh past frame 52.
    planner.replan_with_velocity(velocity)
    written = 0
    while written < n_frames:
        if planner.should_replan():
            planner.replan_with_velocity(velocity)
        qpos = planner.get_next_frame().detach().cpu().numpy()
        out[written] = qpos.astype(np.float32)
        written += 1
    return out


def _load_heuristic_clip(
    primitives_pkl: Path, bin_name: str, n_frames: int
) -> dict[str, np.ndarray]:
    """Load the first ``n_frames`` of a curator PKL bin at native fps.

    Returns ``{"dof": (n, 31), "root_rot_xyzw": (n, 4), "root_trans": (n, 3)}``.
    Resampling to OUTPUT_FPS=50 (what the heuristic actually publishes)
    is left to the caller's downstream comparison; we keep the raw
    clip here to make the diff between native motion and 50 Hz publish
    detectable.
    """
    raw = joblib.load(primitives_pkl)
    if bin_name not in raw:
        raise KeyError(
            f"bin {bin_name!r} not in {primitives_pkl}; available keys "
            f"(first 20): {list(raw.keys())[:20]}"
        )
    payload = raw[bin_name]
    dof = np.asarray(payload["dof"], dtype=np.float32)
    rot = np.asarray(payload["root_rot_xyzw"], dtype=np.float32)
    trans = np.asarray(payload["root_trans"], dtype=np.float64)
    n = min(n_frames, dof.shape[0])
    return {
        "dof": dof[:n],
        "root_rot_xyzw": rot[:n],
        "root_trans": trans[:n],
        "fps": float(payload["fps"]),
    }


def _summarize_joints(
    dof: np.ndarray, label: str
) -> None:
    """Print per-joint min / max / mean + global ranges."""
    print(f"\n[{label}] joint statistics over {dof.shape[0]} frames:")
    print(f"  global min  = {dof.min():+.4f}")
    print(f"  global max  = {dof.max():+.4f}")
    print(f"  global mean = {dof.mean():+.4f}")
    print(f"  per-joint stdev (sum) = {dof.std(axis=0).sum():.4f}  "
          "(higher = more motion)")
    # Top 5 most-active joints
    stds = dof.std(axis=0)
    top = np.argsort(stds)[::-1][:5]
    print("  top 5 most-active joints (idx: stdev rad):")
    for idx in top:
        print(f"    j{idx:02d}: {stds[idx]:.4f}")


def _summarize_root(
    quat_wxyz_or_xyzw: np.ndarray, trans: np.ndarray, *,
    wxyz: bool, label: str, fps: float,
) -> None:
    """Print yaw drift and XY trajectory deltas."""
    if wxyz:
        yaw_deg = _quat_wxyz_to_yaw_deg(quat_wxyz_or_xyzw)
    else:
        yaw_deg = _quat_xyzw_to_yaw_deg(quat_wxyz_or_xyzw)
    yaw_unwrapped = np.unwrap(np.radians(yaw_deg))
    dyaw_deg = np.degrees(yaw_unwrapped[-1] - yaw_unwrapped[0])
    yaw_osc_deg = float(np.degrees(yaw_unwrapped.max() - yaw_unwrapped.min()))
    dx_m = float(trans[-1, 0] - trans[0, 0])
    dy_m = float(trans[-1, 1] - trans[0, 1])
    duration_s = (len(yaw_deg) - 1) / fps if fps > 0 else 0.0
    print(f"\n[{label}] root trajectory over {len(yaw_deg)} frames "
          f"({duration_s:.2f} s at {fps:.1f} fps):")
    print(f"  net dyaw       = {dyaw_deg:+.2f} deg")
    print(f"  yaw osc range  = {yaw_osc_deg:.2f} deg "
          f"(peak-to-peak across the window)")
    print(f"  net dx (world) = {dx_m:+.3f} m")
    print(f"  net dy (world) = {dy_m:+.3f} m")
    if duration_s > 0:
        print(f"  avg vx (world) = {dx_m / duration_s:+.3f} m/s")
        print(f"  avg vy (world) = {dy_m / duration_s:+.3f} m/s")
        print(f"  avg yaw rate   = "
              f"{(dyaw_deg / duration_s):+.2f} deg/s "
              f"({np.radians(dyaw_deg / duration_s):+.3f} rad/s)")


def _build_default_warmup_qpos() -> np.ndarray:
    """Same 38-D vector the kplanner uses for IDLE_LOOP / warmup.

    Imports from ``x2_kplanner`` so this script stays in sync with the
    daemon's notion of "default stand pose".
    """
    from gear_sonic.scripts.x2_kplanner import _build_default_warmup_qpos
    return _build_default_warmup_qpos()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intent-yaw-rate", type=float, default=0.0,
                   help="commanded yaw_rate channel (rad/s)")
    p.add_argument("--intent-vel-x", type=float, default=0.5,
                   help="commanded vel_x channel (m/s)")
    p.add_argument("--intent-vel-z", type=float, default=0.0,
                   help="commanded vel_z channel (m/s, lateral)")
    p.add_argument("--intent-hip-h", type=float, default=0.95,
                   help="commanded hip-height channel (m)")
    p.add_argument("--duration-s", type=float, default=1.0,
                   help="how many seconds of motion to predict / compare")
    p.add_argument("--planner-fps", type=float, default=50.0,
                   help="kplanner publish fps (matches OUTPUT_FPS)")
    p.add_argument("--heuristic-bin", default="fwd_walk_standard",
                   help="bin name in x2_planner_primitives.pkl to compare to")
    p.add_argument(
        "--primitives-pkl",
        type=Path,
        default=REPO_ROOT
        / "gear_sonic"
        / "data"
        / "motions"
        / "x2_planner_primitives.pkl",
    )
    p.add_argument(
        "--device", default="cuda",
        help="torch device for the planner forward pass",
    )
    args = p.parse_args()

    n_frames = int(round(args.duration_s * args.planner_fps))
    intent = (
        float(args.intent_yaw_rate),
        float(args.intent_vel_x),
        float(args.intent_vel_z),
        float(args.intent_hip_h),
    )
    device = _resolve_device(args.device)
    print("=" * 72)
    print("Forward-walk diagnostic: kplanner vs heuristic")
    print(f"  intent vector  = (yaw_rate={intent[0]:+.2f} rad/s, "
          f"vel_x={intent[1]:+.2f} m/s, vel_z={intent[2]:+.2f} m/s, "
          f"hip_h={intent[3]:.2f} m)")
    print(f"  duration       = {args.duration_s:.2f} s "
          f"({n_frames} frames at {args.planner_fps:.1f} fps)")
    print(f"  heuristic bin  = {args.heuristic_bin!r}")
    print(f"  device         = {device} (requested {args.device!r}, "
          f"python={sys.executable})")
    print("=" * 72)

    print("\n[kplanner] loading model + running prediction...")
    warmup_qpos = _build_default_warmup_qpos()
    kpl = _run_kplanner_forward(
        warmup_qpos=warmup_qpos,
        intent=intent,
        n_frames=n_frames,
        device=device,
    )
    print(f"[kplanner] qpos[0] dof slice [7:14] = "
          f"{kpl[0, 7:14].round(3).tolist()}")
    print(f"[kplanner] qpos[{n_frames-1}] dof slice [7:14] = "
          f"{kpl[-1, 7:14].round(3).tolist()}")

    print("\n[heuristic] loading curator PKL...")
    heur = _load_heuristic_clip(
        primitives_pkl=args.primitives_pkl,
        bin_name=args.heuristic_bin,
        n_frames=n_frames,
    )
    print(f"[heuristic] clip fps = {heur['fps']:.1f}, "
          f"frames used = {heur['dof'].shape[0]}")

    _summarize_joints(kpl[:, 7:], label="kplanner")
    _summarize_joints(heur["dof"], label=f"heuristic/{args.heuristic_bin}")

    _summarize_root(
        kpl[:, 3:7], kpl[:, :3],
        wxyz=True, label="kplanner", fps=args.planner_fps,
    )
    _summarize_root(
        heur["root_rot_xyzw"], heur["root_trans"],
        wxyz=False,
        label=f"heuristic/{args.heuristic_bin}",
        fps=heur["fps"],
    )

    print("\n" + "=" * 72)
    print("Diagnostic guidance:")
    print(
        "  * If kplanner per-joint stdev is FAR below heuristic "
        "  -> model isn't producing visible gait amplitude\n"
        "    (regardless of yaw); SONIC has nothing to track.\n"
        "  * If kplanner net dx is near zero but joint stdev matches\n"
        "    -> model produces stepping joints but no integrated XY\n"
        "    translation; the policy can't initiate world motion.\n"
        "  * If kplanner net dyaw is large (>5 deg) with vy=0 intent\n"
        "    -> root_quat drift dominates; that's the spinning bug.\n"
        "  * If all three match the heuristic but the deploy still\n"
        "    fails -> issue is downstream (deploy, manager, SONIC obs).\n"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
