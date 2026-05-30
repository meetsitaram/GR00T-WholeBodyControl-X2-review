"""Replay a known PKL motion clip through the kplanner.

This is the "is the model memorizing training data?" test. The previous
smoke test (``smoke_test_x2_inference.py``) only validated TENSOR
SHAPES, not motion content. This script does a real content check:

1. Pick a clip from the training set (default: a forward-walk clip
   from ``x2_ultra_locowalk.pkl``).
2. Build MuJoCo qpos = ``[trans(3), wxyz_quat(4), dof(31)]`` for the
   clip's first 4 frames. (The PKL stores ``root_rot`` as xyzw; we
   permute to wxyz here before seeding the planner.)
3. Compute the clip's actual velocity profile over frames [0, 4] and
   use that as the ``velocity_intent`` for the kplanner. So we ask
   the model the EXACT velocity that was in the source training
   motion.
4. Seed ``NeuralPlannerCore`` with the 4 real-clip frames as context.
5. Predict the next ~52 frames.
6. Compare predicted frames to the clip's actual next 50 frames:
   - per-joint RMS error
   - root XY trajectory deviation
   - root yaw drift

What the result tells you:

* **RMS error < ~0.15 rad on joints AND trajectory matches.** The
  model has memorized / generalized the training data correctly. If
  the daemon still misbehaves at runtime, the bug is in the
  inference plumbing (seeding from default_angles instead of real
  context, wrong velocity_intent convention, etc).

* **Joint motion plausible but trajectory diverges.** The model
  honors local pose but not root velocity. Indicates weak supervision
  on ``target_local_root_values`` -- the loss formulation needs more
  weight on root motion.

* **Joint motion looks wrong (RMS > 0.5 rad) and trajectory random.**
  The model has not learned the data at all. Training is broken
  (loss not converging, wrong target, wrong architecture).

Run::

  source .venv/bin/activate && \\
    PYTHONPATH="${PWD}/motionbricks:${PWD}" \\
    python motionbricks/scripts/replay_pkl_through_kplanner.py
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


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        axis=-1,
    )


def _quat_to_yaw_rad(quat_xyzw: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> yaw (rad) around world +Z."""
    x = quat_xyzw[..., 0]
    y = quat_xyzw[..., 1]
    z = quat_xyzw[..., 2]
    w = quat_xyzw[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wxyz_to_yaw_rad(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> yaw (rad) around world +Z."""
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _build_clip_qpos(payload: dict) -> tuple[np.ndarray, float]:
    """Build (T, 38) MuJoCo qpos array for a PKL clip.

    PKL layout (from ``x2_ultra_locowalk.pkl`` curator):
        root_trans_offset : (T, 3) world m
        root_rot          : (T, 4) xyzw
        dof               : (T, 31) rad
        fps               : int
    """
    trans = np.asarray(payload["root_trans_offset"], dtype=np.float32)
    rot_xyzw = np.asarray(payload["root_rot"], dtype=np.float32)
    dof = np.asarray(payload["dof"], dtype=np.float32)
    if rot_xyzw.shape[-1] != 4:
        raise ValueError(f"root_rot has wrong last dim: {rot_xyzw.shape}")
    rot_wxyz = _quat_xyzw_to_wxyz(rot_xyzw)
    qpos = np.concatenate([trans, rot_wxyz, dof], axis=-1)
    return qpos.astype(np.float32), float(payload.get("fps", 30))


def _intent_from_clip(qpos: np.ndarray, fps: float, n_context: int) -> tuple[float, float, float, float]:
    """Compute (yaw_rate, vel_x_body, vel_z_body, hip_h) from a clip's first frames.

    Matches the convention the model was trained on (motion-rep Y-up;
    we project MuJoCo Z-up world translations into MuJoCo X / Y, and
    the model maps these to motion-rep vel_x / vel_z via the standard
    converter). We compute velocities in the BODY FRAME of the first
    frame so the intent is rotation-invariant.
    """
    if qpos.shape[0] <= n_context:
        raise ValueError("clip too short to compute intent")
    quat_wxyz = qpos[:n_context, 3:7]
    trans = qpos[:n_context, :3]
    yaw_rad = _wxyz_to_yaw_rad(quat_wxyz)
    yaw_unwrap = np.unwrap(yaw_rad)
    duration_s = (n_context - 1) / fps if fps > 0 else 1.0 / 30.0
    dyaw = float(yaw_unwrap[-1] - yaw_unwrap[0])
    dx_world = float(trans[-1, 0] - trans[0, 0])
    dy_world = float(trans[-1, 1] - trans[0, 1])
    # Project into BODY FRAME at the starting yaw.
    yaw0 = float(yaw_unwrap[0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    dx_body = c * dx_world - s * dy_world
    dy_body = s * dx_world + c * dy_world
    hip_h = float(trans[:, 2].mean())
    yaw_rate = dyaw / duration_s
    # Channel convention (matches NeuralPlannerCore._predict_with_velocity):
    #   vel_x = motion-rep X = MuJoCo body-Y = LATERAL  -> dy_body
    #   vel_z = motion-rep Z = MuJoCo body-X = FORWARD  -> dx_body
    vel_x_lateral = dy_body / duration_s
    vel_z_forward = dx_body / duration_s
    return yaw_rate, vel_x_lateral, vel_z_forward, hip_h


def _per_joint_rms(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return per-joint RMS error over the time axis."""
    n = min(pred.shape[0], actual.shape[0])
    diff = pred[:n] - actual[:n]
    return np.sqrt((diff ** 2).mean(axis=0))


def _trajectory_summary(qpos: np.ndarray) -> dict:
    yaw = np.unwrap(_wxyz_to_yaw_rad(qpos[:, 3:7]))
    return {
        "dyaw_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "dx_m": float(qpos[-1, 0] - qpos[0, 0]),
        "dy_m": float(qpos[-1, 1] - qpos[0, 1]),
        "hip_z_mean_m": float(qpos[:, 2].mean()),
    }


def _instant_intent_from_clip(
    qpos: np.ndarray,
    fps: float,
    frame_idx: int,
    window: int = 8,
) -> tuple[float, float, float, float]:
    """Compute a rolling-window intent at ``frame_idx``.

    Uses frames ``[frame_idx - window/2, frame_idx + window/2]`` so the
    intent reflects the velocity AROUND the current playback position,
    in body frame.
    """
    half = window // 2
    lo = max(0, frame_idx - half)
    hi = min(qpos.shape[0] - 1, frame_idx + half)
    if hi <= lo:
        return 0.0, 0.0, 0.0, float(qpos[frame_idx, 2])
    quat_wxyz = qpos[lo:hi + 1, 3:7]
    trans = qpos[lo:hi + 1, :3]
    yaw = np.unwrap(_wxyz_to_yaw_rad(quat_wxyz))
    duration_s = (hi - lo) / fps if fps > 0 else 1.0 / 30.0
    dyaw = float(yaw[-1] - yaw[0])
    dx_world = float(trans[-1, 0] - trans[0, 0])
    dy_world = float(trans[-1, 1] - trans[0, 1])
    yaw0 = float(yaw[0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    dx_body = c * dx_world - s * dy_world
    dy_body = s * dx_world + c * dy_world
    hip_h = float(trans[:, 2].mean())
    yaw_rate = dyaw / duration_s
    # Channel convention (matches NeuralPlannerCore._predict_with_velocity):
    #   vel_x = motion-rep X = MuJoCo body-Y = LATERAL  -> dy_body
    #   vel_z = motion-rep Z = MuJoCo body-X = FORWARD  -> dx_body
    vel_x_lateral = dy_body / duration_s
    vel_z_forward = dx_body / duration_s
    return yaw_rate, vel_x_lateral, vel_z_forward, hip_h


def _run_full_clip_replay(
    planner,
    qpos_clip: np.ndarray,
    fps: float,
    n_context: int,
    device: str,
    override: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive the planner for the full duration of the clip.

    Seed with the clip's first ``n_context`` frames, then ask for
    one frame at a time. Whenever the planner asks to replan, feed
    it the clip's instantaneous rolling velocity at the current
    playback position.
    """
    total = qpos_clip.shape[0]
    n_predict = total - n_context
    print(f"  full-clip       = seed {n_context}, predict {n_predict} frames "
          f"(~{n_predict / fps:.2f}s @ {fps:.0f}fps)")

    seed_qpos = torch.from_numpy(qpos_clip[:n_context])
    planner.reset(seed_qpos)
    if override is not None:
        intent0 = override
    else:
        intent0 = _instant_intent_from_clip(qpos_clip, fps, n_context)
    print(f"  intent[t=0]     = (yaw_rate={intent0[0]:+.3f}, "
          f"vel_x={intent0[1]:+.3f}, vel_z={intent0[2]:+.3f}, "
          f"hip_h={intent0[3]:.3f})")
    planner.replan_with_velocity(
        torch.tensor(list(intent0), dtype=torch.float32, device=device),
    )

    pred = np.zeros((n_predict, 38), dtype=np.float32)
    intent_log = []
    for i in range(n_predict):
        playback_frame = n_context + i
        if planner.should_replan():
            if override is not None:
                intent = override
            else:
                intent = _instant_intent_from_clip(
                    qpos_clip, fps, playback_frame,
                )
            intent_log.append((playback_frame, intent))
            planner.replan_with_velocity(
                torch.tensor(list(intent), dtype=torch.float32, device=device),
            )
        pred[i] = planner.get_next_frame().detach().cpu().numpy()

    print(f"  replans         = {len(intent_log)} over {n_predict} frames")
    if intent_log:
        log_steps = max(1, len(intent_log) // 5)
        for j, (pf, it) in enumerate(intent_log):
            if j % log_steps == 0 or j == len(intent_log) - 1:
                print(f"    @frame={pf:4d}  yaw_rate={it[0]:+.3f}  "
                      f"vel_x={it[1]:+.3f}  vel_z={it[2]:+.3f}  "
                      f"hip_h={it[3]:.3f}")

    actual = qpos_clip[n_context : n_context + n_predict]
    return pred, actual


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--motion-lib-pkl", type=Path,
        default=REPO_ROOT / "gear_sonic" / "data" / "motions"
                / "x2_ultra_locowalk.pkl",
    )
    p.add_argument(
        "--clip-key", default=None,
        help="bin name to replay. Defaults to the first forward-walk clip.",
    )
    p.add_argument(
        "--n-context", type=int, default=4,
        help="number of clip frames to seed the planner with (matches "
             "NUM_FRAMES_PER_TOKEN=4)",
    )
    p.add_argument(
        "--mode",
        choices=("full-clip", "window"),
        default="full-clip",
        help="full-clip: drive planner for the whole clip duration using "
             "the clip's rolling velocity as intent (full memorization "
             "check). window: only replay a single ``--n-predict`` window "
             "starting at ``--window-start``.",
    )
    p.add_argument(
        "--n-predict", type=int, default=50,
        help="(window mode) frames of prediction to compare against clip",
    )
    p.add_argument(
        "--window-start", type=int, default=0,
        help="(window mode) frame index to start the context window from",
    )
    p.add_argument(
        "--override-intent", type=str, default=None,
        help="Comma-separated (yaw_rate, vel_x, vel_z, hip_h) intent that "
             "overrides the intent derived from the clip. Forces the same "
             "intent for every replan (full-clip mode) or for the single "
             "window (window mode).",
    )
    p.add_argument(
        "--save-npz", type=Path, default=None,
        help="Optional path to save predicted vs actual qpos arrays.",
    )
    p.add_argument(
        "--mask-start-keyframes", type=int, default=None,
        help="DIAGNOSTIC: downgrade the start-window keyframe density to N "
             "(N=1 or 2) to match the keyframe_num_warmup_steps schedule at "
             "training step <= 100k. Default behavior (None) feeds 4-of-4 "
             "keyframes which is what production inference does and what "
             "the 100k checkpoint has never been trained on.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--vqvae-ckpt", type=Path, default=None,
        help="override VQVAE checkpoint (default = X2PlannerPaths.default())",
    )
    p.add_argument(
        "--pose-ckpt", type=Path, default=None,
    )
    p.add_argument(
        "--root-ckpt", type=Path, default=None,
    )
    args = p.parse_args()

    print("=" * 88)
    print("PKL replay through kplanner")
    print("=" * 88)

    raw = joblib.load(args.motion_lib_pkl)
    if args.clip_key is None:
        for k in raw.keys():
            if "forward" in k.lower() and "_M" not in k:
                args.clip_key = k
                break
        if args.clip_key is None:
            args.clip_key = next(iter(raw.keys()))
    if args.clip_key not in raw:
        raise KeyError(f"Clip {args.clip_key!r} not in {args.motion_lib_pkl}")
    payload = raw[args.clip_key]
    qpos_clip, fps = _build_clip_qpos(payload)
    print(f"  clip            = {args.clip_key}")
    print(f"  fps             = {fps:.1f}, total frames = {qpos_clip.shape[0]}")
    print(f"  trans frame[0]  = {qpos_clip[0, :3].round(3).tolist()}")
    print(f"  trans frame[N]  = {qpos_clip[-1, :3].round(3).tolist()}")
    print(f"  mode            = {args.mode}")

    if args.override_intent is not None:
        override = tuple(float(x) for x in args.override_intent.split(","))
        if len(override) != 4:
            raise ValueError(
                "--override-intent must be 4 comma-separated values: "
                "yaw_rate,vel_x,vel_z,hip_h"
            )
    else:
        override = None

    # Device probe.
    device = args.device
    if device != "cpu":
        if not torch.cuda.is_available():
            print(f"[device] {device!r} unavailable, falling back to cpu")
            device = "cpu"
        else:
            try:
                _ = (torch.zeros(1, device=device) + 1).cpu()
            except RuntimeError as exc:
                print(f"[device] CUDA probe failed ({exc}); cpu")
                device = "cpu"
    print(f"  device          = {device}")

    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths,
        load_x2_planner,
    )

    defaults = X2PlannerPaths.default()
    paths = X2PlannerPaths(
        vqvae_ckpt=args.vqvae_ckpt or defaults.vqvae_ckpt,
        pose_ckpt=args.pose_ckpt or defaults.pose_ckpt,
        root_ckpt=args.root_ckpt or defaults.root_ckpt,
        vqvae_version_dir=defaults.vqvae_version_dir,
        pose_version_dir=defaults.pose_version_dir,
        root_version_dir=defaults.root_version_dir,
    )
    print(f"  vqvae_ckpt      = {paths.vqvae_ckpt}")
    print(f"  pose_ckpt       = {paths.pose_ckpt}")
    print(f"  root_ckpt       = {paths.root_ckpt}")

    planner = load_x2_planner(paths, device=device, replan_threshold_frames=16)

    if args.mask_start_keyframes is not None:
        n_keep = int(args.mask_start_keyframes)
        print(f"  [diag] downgrading start-window to {n_keep}/4 keyframes "
              "(simulating training-time keyframe density)")

        def _hook(has_g, has_l, has_p, NUM_FT):
            # Keep the FIRST n_keep positions in the start window (deterministic
            # for reproducibility), mask the remaining start-window positions.
            # The end window is left at its default (no global / no pose;
            # full has_local_root_values from velocity_intent).
            new_g = has_g.clone()
            new_l = has_l.clone()
            new_p = has_p.clone()
            for i in range(n_keep, NUM_FT):
                new_g[:, i] = False
                new_p[:, i] = False
                if i < NUM_FT - 1:
                    new_l[:, i] = False
            return new_g, new_l, new_p

        planner.diagnostic_mask_hook = _hook

    if args.mode == "full-clip":
        pred, actual = _run_full_clip_replay(
            planner, qpos_clip, fps, args.n_context, device, override,
        )
    else:
        if args.window_start + args.n_context >= qpos_clip.shape[0]:
            raise ValueError(
                f"window_start={args.window_start} + n_context="
                f"{args.n_context} exceeds clip length {qpos_clip.shape[0]}"
            )
        seed_slice = qpos_clip[
            args.window_start : args.window_start + args.n_context
        ]
        if override is not None:
            intent = override
            print(f"  intent (override) = "
                  f"(yaw_rate={intent[0]:+.3f}, vel_x={intent[1]:+.3f}, "
                  f"vel_z={intent[2]:+.3f}, hip_h={intent[3]:.3f})")
        else:
            intent = _intent_from_clip(
                qpos_clip[args.window_start:], fps, args.n_context,
            )
            print(f"  intent (derived)  = "
                  f"(yaw_rate={intent[0]:+.3f}, vel_x={intent[1]:+.3f}, "
                  f"vel_z={intent[2]:+.3f}, hip_h={intent[3]:.3f})")
        seed_qpos = torch.from_numpy(seed_slice)
        planner.reset(seed_qpos)
        velocity = torch.tensor(list(intent), dtype=torch.float32, device=device)
        planner.replan_with_velocity(velocity)
        pred = np.zeros((args.n_predict, 38), dtype=np.float32)
        for i in range(args.n_predict):
            if planner.should_replan():
                planner.replan_with_velocity(velocity)
            pred[i] = planner.get_next_frame().detach().cpu().numpy()
        actual_start = args.window_start + args.n_context
        actual = qpos_clip[actual_start : actual_start + args.n_predict]

    if args.mode == "window" and actual.shape[0] < args.n_predict:
        print(f"  (clip only has {actual.shape[0]} frames after context; "
              "comparing what's available)")
    n = min(pred.shape[0], actual.shape[0])
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            pred=pred[:n],
            actual=actual[:n],
            clip_key=args.clip_key,
            fps=fps,
        )
        print(f"  saved npz       = {args.save_npz}")

    # Joint RMS error.
    joint_rms = _per_joint_rms(pred[:, 7:], actual[:, 7:])
    rms_total = float(np.sqrt((joint_rms ** 2).mean()))
    rms_max = float(joint_rms.max())
    rms_top5 = np.argsort(joint_rms)[::-1][:5]

    print("\n" + "=" * 88)
    print(f"PREDICTION vs CLIP — {n} frames of comparison")
    print("=" * 88)
    print(f"\n  joint RMS error (rad):")
    print(f"    overall RMS     = {rms_total:.4f}")
    print(f"    worst joint RMS = {rms_max:.4f}")
    print(f"    top 5 joints    = "
          + ", ".join(f"j{idx:02d}({joint_rms[idx]:.3f})"
                       for idx in rms_top5))

    pred_traj = _trajectory_summary(pred[:n])
    actual_traj = _trajectory_summary(actual[:n])

    print("\n  trajectory (over the {0} comparison frames):".format(n))
    print(f"    actual dyaw  = {actual_traj['dyaw_deg']:+8.2f} deg | "
          f"pred dyaw  = {pred_traj['dyaw_deg']:+8.2f} deg | "
          f"err = {pred_traj['dyaw_deg'] - actual_traj['dyaw_deg']:+8.2f} deg")
    print(f"    actual dx_m  = {actual_traj['dx_m']:+8.3f}     | "
          f"pred dx_m  = {pred_traj['dx_m']:+8.3f}     | "
          f"err = {pred_traj['dx_m'] - actual_traj['dx_m']:+8.3f}")
    print(f"    actual dy_m  = {actual_traj['dy_m']:+8.3f}     | "
          f"pred dy_m  = {pred_traj['dy_m']:+8.3f}     | "
          f"err = {pred_traj['dy_m'] - actual_traj['dy_m']:+8.3f}")
    print(f"    actual hip_z = {actual_traj['hip_z_mean_m']:+8.3f}     | "
          f"pred hip_z = {pred_traj['hip_z_mean_m']:+8.3f}     | "
          f"err = {pred_traj['hip_z_mean_m'] - actual_traj['hip_z_mean_m']:+8.3f}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if rms_total < 0.15 and abs(
        pred_traj["dx_m"] - actual_traj["dx_m"]
    ) < 0.2 and abs(pred_traj["dyaw_deg"] - actual_traj["dyaw_deg"]) < 10.0:
        print(
            "  PASS  the model reproduces the training clip with low error.\n"
            "        If the daemon misbehaves at runtime, look at the\n"
            "        inference plumbing: warmup seed (default_angles vs\n"
            "        in-distribution context), velocity_intent convention,\n"
            "        quat layout at publish."
        )
        return 0
    if rms_total < 0.30:
        print(
            f"  PARTIAL  joint RMS={rms_total:.3f} (poses plausible) but\n"
            "         trajectory diverges. Local pose is learned but root\n"
            "         motion is not -- suspect weak supervision on\n"
            "         target_local_root_values in the root model's loss."
        )
        return 1
    print(
        f"  FAIL  joint RMS={rms_total:.3f} (~poses random) and trajectory\n"
        "       does not match the source clip. The model has not learned\n"
        "       to reproduce its own training data. Likely causes:\n"
        "       - undertrained (root/pose at 100k of 200k recommended)\n"
        "       - loss diverged at some point (check wandb)\n"
        "       - data prep / channel-order bug in supervision target"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
