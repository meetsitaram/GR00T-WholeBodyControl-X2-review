"""Probe: does the root model under-track because of constraint masking?

`NeuralPlannerCore` feeds only target VELOCITY as a constraint in its
8-frame predict window (target_pos and target_pose are masked off).
`full_navigation_agent` (the G1 reference demo) keeps ALL three target
constraints ON by default (``target_root_realignment=True``).

This probe tests each combination on the SAME root model and the SAME
seed, measuring forward displacement at a single fixed intent. If the
slope jumps from ~0.05 to ~1.0 when we enable the target_pos
constraint, we've found the bug.

Modes:

  * ``velocity_only`` — NeuralPlannerCore's current behavior:
        has_global_root_values[:, -4:] = False
        has_local_poses[:, -4:] = False
        has_local_root_values[:, -4:] = True
  * ``velocity_plus_target_pos`` — add the target_pos constraint:
        has_global_root_values[:, -4:] = True (target_pos from
            seed_pos + velocity * horizon)
        has_local_poses[:, -4:] = False
        has_local_root_values[:, -4:] = True
  * ``demo_full`` — match full_agent default:
        all three has_* = True for the target window
        target_pos from seed_pos + velocity * horizon
        target_pose from seed clip's actual frame 8 (the "what the
            real motion looks like 8 frames ahead")

A simple table of (mode, slope) per intent reveals the fix.

Usage::

    PYTHONPATH="${PWD}/motionbricks:${PWD}" python \\
        motionbricks/scripts/probe_root_constraint_modes.py \\
        --ckpt-set g1 --fixture walking --vx 0.4
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


# Reuse the fixture conventions from test_root_isolated.py to keep the
# probe and the per-model test in lockstep.
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


def _quat_xyzw_to_wxyz(q):
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def _load_seed_qpos(ckpt_set: str, fixture: str) -> tuple[np.ndarray, float]:
    spec = FIXTURES[ckpt_set][fixture]
    if spec["kind"] == "x2_pkl":
        payload = joblib.load(REPO_ROOT / spec["pkl"])[spec["clip_key"]]
        trans = np.asarray(payload["root_trans_offset"], dtype=np.float32)
        rot_xyzw = np.asarray(payload["root_rot"], dtype=np.float32)
        dof = np.asarray(payload["dof"], dtype=np.float32)
        rot_wxyz = _quat_xyzw_to_wxyz(rot_xyzw)
        qpos = np.concatenate([trans, rot_wxyz, dof], axis=-1).astype(np.float32)
        return qpos, float(payload.get("fps", 30))
    sd = torch.load(REPO_ROOT / spec["g1_clip_path"], map_location="cpu", weights_only=False)
    n = int(sd["num_frames_per_clip"].numpy()[spec["clip_idx"]])
    return sd["mujoco_qpos"][spec["clip_idx"], :n].numpy().astype(np.float32), 30.0


def _load_planner(ckpt_set: str, device: str):
    if ckpt_set == "x2":
        from motionbricks.motion_backbone.inference.load_x2_planner import (
            X2PlannerPaths, load_x2_planner,
        )
        return load_x2_planner(X2PlannerPaths.default(), device=device)
    if ckpt_set == "g1":
        from motionbricks.motion_backbone.inference.load_g1_planner import (
            G1PlannerPaths, load_g1_planner,
        )
        return load_g1_planner(G1PlannerPaths.default(), device=device)
    raise ValueError(ckpt_set)


# ---------------------------------------------------------------------------
# Core experiment: build the 8-frame constraint window for each mode and
# call _predict_root_trajectories directly.
# ---------------------------------------------------------------------------


def _build_context_8frame_window(
    planner,
    seed_qpos_np: np.ndarray,
    seed_offset: int,
    target_offset_frames: int,
    intent: tuple[float, float, float, float],
    device: str,
    target_horizon_s: float = 2.0,
):
    """Build the 8-frame unnormalized constraint window for each mode.

    Returns:
        global_root_values [1, 8, 5]   (unnormalized)
        local_root_values  [1, 8, 4]   (unnormalized)
        local_poses        [1, 8, 304] (unnormalized)
        target_pos_seed_from_clip   [1, 3]   actual future world-frame xyz from clip
                                              at seed_offset + target_offset_frames
    """
    inferencer = planner._inferencer
    converter = planner._converter
    NUM_FT = planner.NUM_FRAMES_PER_TOKEN
    fps = float(inferencer.local_motion_rep.fps)

    # Take 4 context frames + 4 target frames (mid-clip) so target has
    # real-data joint poses to anchor demo_full mode.
    ctx_qpos = seed_qpos_np[seed_offset : seed_offset + NUM_FT]
    if ctx_qpos.shape[0] < NUM_FT:
        raise ValueError(f"need {NUM_FT} context frames, got {ctx_qpos.shape[0]}")
    tgt_idx = seed_offset + target_offset_frames
    if tgt_idx + NUM_FT > seed_qpos_np.shape[0]:
        # Pad with last available frame.
        tgt_qpos = np.tile(seed_qpos_np[-1:], (NUM_FT, 1))
    else:
        tgt_qpos = seed_qpos_np[tgt_idx : tgt_idx + NUM_FT]

    ctx_t = torch.from_numpy(ctx_qpos).to(device).float().unsqueeze(0)  # [1, 4, qpos_dim]
    tgt_t = torch.from_numpy(tgt_qpos).to(device).float().unsqueeze(0)

    # ---- Canonicalize the context (mirror NeuralPlannerCore behavior) ----
    # We want the model input to be in canonicalized frame (initial pos =
    # (0, y_up, 0), initial heading = 0). We canonicalize ctx in-place
    # and apply the SAME canonicalization transform to tgt so the
    # relative pose is preserved.
    from motionbricks.geometry.quaternions import matrix_to_quaternion
    from motionbricks.motionlib.core.utils.rotations import quaternion_to_matrix
    from motionbricks.motion_backbone.inference.neural_planner import (
        angle_to_Z_rotation_matrix,
    )
    first_frame_position = (
        ctx_t[:, 0, :3].clone() * torch.tensor([[1.0, 1.0, 0.0]], device=device)
    )
    first_frame_rot = quaternion_to_matrix(ctx_t[:, 0, 3:7].clone())
    first_frame_heading = torch.atan2(first_frame_rot[:, 1, 0], first_frame_rot[:, 0, 0])
    first_frame_heading[first_frame_heading.isnan()] = 0.0
    R_h = angle_to_Z_rotation_matrix(first_frame_heading)
    R_h_inv = R_h.transpose(-2, -1)

    for buf in (ctx_t, tgt_t):
        new_pos = torch.matmul(
            R_h_inv[:, None, :, :],
            (buf[:, :, :3].clone() - first_frame_position)[..., None],
        )[..., 0]
        new_rot = torch.matmul(
            R_h_inv[:, None, :, :],
            quaternion_to_matrix(buf[:, :, 3:7]),
        )
        buf[:, :, 3:7] = matrix_to_quaternion(new_rot)
        buf[:, :, :3] = new_pos

    # ---- Convert canonicalized qpos to joint positions/rotations ----
    ctx_jp, ctx_jr = converter.convert_mujoco_qpos_to_motion_transforms(ctx_t)
    tgt_jp, tgt_jr = converter.convert_mujoco_qpos_to_motion_transforms(tgt_t)
    # ctx_jp: [B, 4, J, 3], ctx_jr: [B, 4, J, 3, 3]

    # ---- Build context constraint values (mirror NeuralPlannerCore lines 308-352) ----
    root_idx = 0
    ctx_root_pos = ctx_jp[:, :, root_idx, :]    # [B, 4, 3]
    ctx_rot_angle = torch.atan2(
        ctx_jr[:, :, root_idx, 0, 2], ctx_jr[:, :, root_idx, 2, 2]
    )  # [B, 4]
    ctx_global_root = torch.cat(
        [ctx_root_pos,
         torch.cos(ctx_rot_angle)[..., None],
         torch.sin(ctx_rot_angle)[..., None]],
        dim=-1,
    )  # [B, 4, 5]

    ctx_local_root = torch.zeros([1, NUM_FT, 4], device=device)
    ctx_local_root[:, : NUM_FT - 1, 0] = (
        ((ctx_rot_angle[:, 1:] - ctx_rot_angle[:, :-1] + torch.pi) % (2 * torch.pi))
        - torch.pi
    ) * fps
    ctx_local_root[:, : NUM_FT - 1, 1:3] = (
        ctx_root_pos[:, 1:, [0, 2]] - ctx_root_pos[:, :-1, [0, 2]]
    ) * fps
    ctx_local_root[:, : NUM_FT - 1, 3] = ctx_global_root[:, : NUM_FT - 1, 1]
    ctx_local_root[:, NUM_FT - 1, :] = ctx_local_root[:, NUM_FT - 2, :]

    from motionbricks.motionlib.core.utils.rotations import matrix_to_cont6d
    joint_pos_rel = ctx_jp[:, :, 1:, :].clone()
    joint_pos_rel[..., 0] = ctx_jp[:, :, 1:, 0] - ctx_jp[:, :, :1, 0]
    joint_pos_rel[..., 2] = ctx_jp[:, :, 1:, 2] - ctx_jp[:, :, :1, 2]
    joint_rot_6d = matrix_to_cont6d(ctx_jr)
    ctx_local_poses = torch.cat(
        [joint_pos_rel.view([1, NUM_FT, -1]),
         joint_rot_6d.view([1, NUM_FT, -1])],
        dim=-1,
    )

    # ---- Build target constraint values ----
    yaw_rate, vx, vz, hip_h = intent
    intent_t = torch.tensor([yaw_rate, vx, vz, hip_h], device=device, dtype=torch.float32)

    # Target velocity: broadcast intent (NeuralPlannerCore style).
    tgt_local_root = intent_t[None, None, :].expand([1, NUM_FT, 4]).contiguous()

    # Target global root values: position the target window at the END
    # of an EXPECTED full-horizon prediction (~2 s by default). The
    # last 4 frames of the 8-frame constraint window represent "where
    # the robot should END UP", not "the next 4 frames". So target_pos
    # = last_context_pos + velocity_intent * target_horizon_s.
    implied_target_x = ctx_root_pos[:, -1:, 0] + intent_t[1] * target_horizon_s
    implied_target_z = ctx_root_pos[:, -1:, 2] + intent_t[2] * target_horizon_s
    implied_target_y = torch.full_like(implied_target_x, intent_t[3])  # hip height
    implied_target_pos = torch.cat(
        [implied_target_x, implied_target_y, implied_target_z], dim=-1
    )  # [B, 1, 3]
    implied_target_pos = implied_target_pos.expand([1, NUM_FT, 3])
    implied_target_heading_angle = (
        ctx_rot_angle[:, -1:] + intent_t[0] * target_horizon_s
    )
    implied_target_heading_angle = implied_target_heading_angle.expand([1, NUM_FT])
    tgt_global_root = torch.cat(
        [implied_target_pos,
         torch.cos(implied_target_heading_angle)[..., None],
         torch.sin(implied_target_heading_angle)[..., None]],
        dim=-1,
    )

    # Target local poses from the actual clip future frame (for demo_full mode).
    tgt_root_pos = tgt_jp[:, :, root_idx, :]
    tgt_joint_pos_rel = tgt_jp[:, :, 1:, :].clone()
    tgt_joint_pos_rel[..., 0] = tgt_jp[:, :, 1:, 0] - tgt_root_pos[..., :1]
    tgt_joint_pos_rel[..., 2] = tgt_jp[:, :, 1:, 2] - tgt_root_pos[..., 2:3]
    tgt_joint_rot_6d = matrix_to_cont6d(tgt_jr)
    tgt_local_poses = torch.cat(
        [tgt_joint_pos_rel.view([1, NUM_FT, -1]),
         tgt_joint_rot_6d.view([1, NUM_FT, -1])],
        dim=-1,
    )

    global_root_values = torch.cat([ctx_global_root, tgt_global_root], dim=1)
    local_root_values = torch.cat([ctx_local_root, tgt_local_root], dim=1)
    local_poses = torch.cat([ctx_local_poses, tgt_local_poses], dim=1)

    return (global_root_values, local_root_values, local_poses,
            ctx_root_pos[:, -1, :])  # the canonicalized last-context root pos


def _run_one_mode(
    planner, inferencer, mode: str,
    global_root_values, local_root_values, local_poses,
    intent: tuple[float, float, float, float],
    fps: float,
):
    """Run _predict_root_trajectories with mode-specific masks."""
    NUM_FT = planner.NUM_FRAMES_PER_TOKEN
    device = global_root_values.device
    has_global = torch.ones_like(global_root_values[:, :, 0], dtype=torch.bool)
    has_local = torch.ones_like(local_root_values[:, :, 0], dtype=torch.bool)
    has_poses = torch.ones_like(local_poses[:, :, 0], dtype=torch.bool)
    # Always mask the last context-frame velocity (no t+1 to difference).
    has_local[:, NUM_FT - 1] = False

    if mode == "velocity_only":
        has_global[:, -NUM_FT:] = False
        has_poses[:, -NUM_FT:] = False
        # has_local target stays True
    elif mode == "velocity_plus_target_pos":
        # has_global target stays True (constraint added back)
        has_poses[:, -NUM_FT:] = False
        # has_local target stays True
    elif mode == "demo_full":
        pass  # all three constraints ON for the target window
    else:
        raise ValueError(mode)

    # Build the normalized batch the way motion_inference.predict step 1 does.
    EPS = 1e-5
    # Recenter global xz to start at 0,0 (predict() step 1).
    g2d = inferencer.global_motion_rep.indices["global_root_pos_2d"]
    g = global_root_values.clone()
    g[:, :, g2d] -= g[:, :1, g2d]

    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    local_pose_feat_idx = extract_feature_from_motion_rep(
        torch.zeros([1, 1, len(inferencer.local_motion_rep.indices["all"])]),
        inferencer.local_motion_rep,
        inferencer.INTERNAL_POSE_FEATURE_MODE,
        fetch_feat_idx=True,
    )
    global_height = g[:, :, inferencer.global_motion_rep.indices["global_root_pos"][[1]]]
    mean = inferencer.local_motion_rep.stats.mean[None, None, local_pose_feat_idx].to(device)
    std = inferencer.local_motion_rep.stats.std[None, None, local_pose_feat_idx].to(device)
    local_poses_normed = (
        torch.cat([global_height, local_poses], dim=-1) - mean
    ) / torch.sqrt(std ** 2 + EPS)

    batch = {
        "global_root_values": inferencer.global_motion_rep.normalize(g).float(),
        "local_root_values": inferencer.local_motion_rep.normalize(local_root_values).float(),
        "local_poses": local_poses_normed.float(),
        "has_global_root_values": has_global,
        "has_local_root_values": has_local,
        "has_local_poses": has_poses,
        "text_embeddings": None,
        "has_text_embeddings": None,
        "num_tokens": torch.full(
            [1, 1], inferencer._root_model.backbone_net.MASKED_NUM_TOKENS,
            dtype=torch.int, device=device,
        ),
        "allowed_pred_num_tokens": None,
    }
    pred_num_tokens, pred_global_root, _pred_local_root = (
        inferencer._predict_root_trajectories(batch, config={})
    )

    pred_global_root_unnorm = (
        inferencer.global_motion_rep.unnormalize(pred_global_root).detach().cpu().numpy()
    )
    horizon_frames = int(pred_num_tokens[0].cpu()) * NUM_FT
    pred = pred_global_root_unnorm[0, :horizon_frames]  # [T, 5]

    horizon_s = horizon_frames / fps
    pred_forward_m = float(pred[-1, 0] - pred[0, 0])
    pred_lateral_m = float(pred[-1, 2] - pred[0, 2])
    commanded_dy = intent[1] * horizon_s
    slope = pred_forward_m / commanded_dy if abs(commanded_dy) > 1e-6 else float("nan")

    return {
        "mode": mode,
        "horizon_frames": horizon_frames,
        "horizon_s": horizon_s,
        "pred_forward_m": pred_forward_m,
        "pred_lateral_m": pred_lateral_m,
        "slope": slope,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-set", choices=("x2", "g1"), default="g1")
    ap.add_argument("--fixture", choices=("walking", "stationary"), default="walking")
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--vz", type=float, default=0.0)
    ap.add_argument("--yaw-rate", type=float, default=0.0)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Start frame for the 4 context frames.")
    ap.add_argument("--target-offset-frames", type=int, default=20,
                    help="Frames after seed_offset to pull target_local_poses from. "
                         "Must be small enough that fixture has these frames. "
                         "Used only in demo_full mode.")
    ap.add_argument("--target-horizon-s", type=float, default=2.0,
                    help="Distance (in seconds) to place the implied target_pos ahead "
                         "of the last context frame. The model uses this to size its "
                         "predicted horizon.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    seed_qpos, fps = _load_seed_qpos(args.ckpt_set, args.fixture)
    print(f"[fixture] {args.ckpt_set}/{args.fixture}: T={seed_qpos.shape[0]} "
          f"frames @ {fps} fps (qpos dim={seed_qpos.shape[-1]})")
    planner = _load_planner(args.ckpt_set, args.device)
    inferencer = planner._inferencer
    hip_h_default = float(seed_qpos[args.seed_offset : args.seed_offset + 4, 2].mean())
    intent = (args.yaw_rate, args.vx, args.vz, hip_h_default)
    print(f"[intent] yaw_rate={intent[0]:.3f}, vx={intent[1]:.3f}, "
          f"vz={intent[2]:.3f}, hip_h={intent[3]:.3f}")

    g, l_root, l_poses, last_ctx_pos = _build_context_8frame_window(
        planner, seed_qpos, args.seed_offset, args.target_offset_frames,
        intent, args.device, target_horizon_s=args.target_horizon_s,
    )
    print(f"[ctx] last-context canonical root pos: {last_ctx_pos[0].cpu().numpy()}")

    rows = []
    for mode in ("velocity_only", "velocity_plus_target_pos", "demo_full"):
        try:
            r = _run_one_mode(planner, inferencer, mode, g, l_root, l_poses, intent, fps)
        except Exception as e:
            r = {"mode": mode, "error": str(e)}
        rows.append(r)

    print()
    print("=" * 78)
    print(f"  Constraint-mode probe  (ckpt={args.ckpt_set}, "
          f"fixture={args.fixture}, vx={args.vx})")
    print("=" * 78)
    print(f"  {'mode':30s} {'horizon_s':>10} {'fwd_m':>8} {'lat_m':>8} {'slope':>8}")
    print(f"  {'-'*30:30s} {'-'*10:>10} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
    for r in rows:
        if "error" in r:
            print(f"  {r['mode']:30s} ERROR: {r['error']}")
        else:
            print(f"  {r['mode']:30s} {r['horizon_s']:>10.2f} "
                  f"{r['pred_forward_m']:>8.3f} {r['pred_lateral_m']:>8.3f} "
                  f"{r['slope']:>8.2f}")
    print("=" * 78)
    print("  velocity_only            = NeuralPlannerCore default "
          "(masks target_pos + target_pose)")
    print("  velocity_plus_target_pos = adds the target_global_root_values constraint back")
    print("  demo_full                = matches full_agent target_root_realignment=True")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
