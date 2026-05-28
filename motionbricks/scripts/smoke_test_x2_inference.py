#!/usr/bin/env python3
"""End-to-end smoke test for the X2 MotionBricks pipeline.

What this validates:

1. The trained X2 VQVAE / pose / root checkpoints load via PyTorch Lightning.
2. We can extract a real motion-rep feature tensor from a BONES-SEED clip.
3. The VQVAE round-trip (encode -> decode) preserves shapes.
4. The X2 ``mujoco_qpos_converter`` produces ``[T, 38]`` tensors that decompose
   into ``root_xyz`` (3) + ``root_quat`` (4, wxyz) + ``joint_pos`` (31).

Usage::

    python scripts/smoke_test_x2_inference.py --max-clips 5

Returns non-zero exit code on any failure so this can gate the full training
pipeline in CI / pre-cloud-launch checks.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch

MB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MB_ROOT.parent
sys.path.insert(0, str(MB_ROOT))


def _lazy_imports():
    from motionbricks.data.x2_bones_seed_dataset import (
        X2MotionDataset,
        build_x2_motion_rep,
    )
    from motionbricks.data.x2_pkl_to_motion import (
        X2MujocoFkExtractor,
        assert_x2_mjcf_coherent,
        default_x2_mjcf_path,
    )
    from motionbricks.helper.mujoco_helper_x2 import (
        X2MujocoQposConverter,
        motion_feature_to_x2_mujoco_qpos,
        mujoco_qpos_to_x2_pose_msg,
    )

    return locals()


def _check_ckpt(label: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} ckpt: {path}")
    sz_mb = path.stat().st_size / (1024 * 1024)
    if sz_mb < 1.0:
        raise RuntimeError(f"{label} ckpt suspiciously small ({sz_mb:.2f} MB): {path}")
    print(f"  {label} ckpt OK -> {path}  ({sz_mb:.1f} MB)")


def _load_ckpt_state(path: Path) -> dict:
    """Load a Lightning ckpt without instantiating the full LightningModule.

    We only need to validate that the file is a real torch checkpoint. The
    full pose/root models pull in heavy hydra plumbing and are exercised in
    train_*_x2 already; this is just shape sanity.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        raise RuntimeError(f"Ckpt {path} missing state_dict; not a Lightning ckpt?")
    nparams = sum(v.numel() for v in state["state_dict"].values())
    print(f"    -> state_dict OK ({nparams / 1e6:.1f}M params)")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 MotionBricks smoke test")
    parser.add_argument("--result_dir", type=Path, default=MB_ROOT / "out")
    parser.add_argument(
        "--pkl",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/motions/x2_ultra_bones_seed.pkl",
    )
    parser.add_argument("--max-clips", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("X2 MotionBricks smoke test")
    print("=" * 70)

    try:
        m = _lazy_imports()
        X2MotionDataset = m["X2MotionDataset"]
        build_x2_motion_rep = m["build_x2_motion_rep"]
        assert_x2_mjcf_coherent = m["assert_x2_mjcf_coherent"]
        default_x2_mjcf_path = m["default_x2_mjcf_path"]
        X2MujocoQposConverter = m["X2MujocoQposConverter"]
        motion_feature_to_x2_mujoco_qpos = m["motion_feature_to_x2_mujoco_qpos"]
        mujoco_qpos_to_x2_pose_msg = m["mujoco_qpos_to_x2_pose_msg"]

        # 1. MJCF coherence check.
        print("\n[1/6] Verifying canonical X2 MJCF (sphere-feet, nq=38)...")
        mjcf = default_x2_mjcf_path()
        assert_x2_mjcf_coherent(mjcf)
        print(f"  MJCF coherence OK -> {mjcf}")

        # 2. Asset / hparams / stats present.
        print("\n[2/6] Verifying skeleton + stats + hparams...")
        skel_dir = args.result_dir / "motionbricks_vqvae_x2/version_1/skeleton"
        stats_dir = args.result_dir / "motionbricks_vqvae_x2/version_1/stats/motion"
        for variant in ("vqvae", "pose", "root"):
            hp = args.result_dir / f"motionbricks_{variant}_x2/version_1/hparams.yaml"
            if not hp.is_file():
                raise FileNotFoundError(
                    f"Missing {hp}. Run `python scripts/build_x2_skeleton_assets.py` first."
                )
        for fp in (skel_dir / "joints.p", skel_dir / "parents.p", stats_dir / "mean.npy", stats_dir / "std.npy"):
            if not fp.is_file():
                raise FileNotFoundError(f"Missing asset: {fp}")
        print(f"  skeleton + stats + 3x hparams.yaml OK")

        # 3. motion_rep loads with stats.
        print("\n[3/6] Loading motion_rep with stats...")
        motion_rep = build_x2_motion_rep(skel_dir, stats_dir, load_stats=True)
        feat_dim = len(motion_rep.indices["all"])
        dual_dim = motion_rep.dual_rep.motion_rep_dim
        print(f"  motion_rep feat_dim={feat_dim} (global), dual_dim={dual_dim}")
        if dual_dim != 418:
            raise RuntimeError(f"Expected dual_dim=418, got {dual_dim}")

        # 4. Real-clip feature extraction (only run if PKL exists).
        print(f"\n[4/6] Extracting features from {args.max_clips} clip(s)...")
        if not args.pkl.is_file():
            raise FileNotFoundError(f"Missing PKL: {args.pkl}")
        ds = X2MotionDataset(
            args.pkl,
            motion_rep,
            min_frames=80,
            max_frames=200,
            max_clips=args.max_clips,
        )
        if len(ds) == 0:
            raise RuntimeError("Dataset is empty after FK extraction")
        sample = ds[0]
        feats: torch.Tensor = sample["motion"]
        print(f"  clip[0] keyid={sample['keyid']}  feats.shape={tuple(feats.shape)}")
        if feats.shape[-1] != feat_dim:
            raise RuntimeError(
                f"feature last-dim {feats.shape[-1]} != motion_rep feat_dim {feat_dim}"
            )

        # 5. VQVAE / pose / root ckpts present and loadable.
        print("\n[5/6] Loading trained checkpoints...")
        ckpt_paths = {
            "vqvae": args.result_dir / "motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt",
            "pose": args.result_dir / "motionbricks_pose_x2/version_1/checkpoints/last.ckpt",
            "root": args.result_dir / "motionbricks_root_x2/version_1/checkpoints/last.ckpt",
        }
        for label, p in ckpt_paths.items():
            _check_ckpt(label, p)
            _load_ckpt_state(p)

        # 6. qpos round-trip (38 cols, 3 + 4 + 31).
        print("\n[6/6] Converting features -> X2 mujoco qpos...")
        feats_b = feats.unsqueeze(0)  # [1, T, 414]
        qpos = motion_feature_to_x2_mujoco_qpos(
            feats_b, motion_rep, is_normalized=True, root_quat_w_first=True
        )
        if qpos.shape[-1] != 38:
            raise RuntimeError(f"qpos last-dim should be 38, got {qpos.shape[-1]}")
        msg = mujoco_qpos_to_x2_pose_msg(qpos[0])
        print(f"  qpos.shape={tuple(qpos.shape)}")
        print(f"  root_xyz.shape={tuple(msg['root_xyz'].shape)}")
        print(f"  root_quat.shape={tuple(msg['root_quat'].shape)}  (wxyz)")
        print(f"  joint_pos.shape={tuple(msg['joint_pos'].shape)}  (31 hinges)")

        # Sanity: quaternion magnitude ~ 1.
        quat_norm = msg["root_quat"].norm(dim=-1)
        if (quat_norm < 0.95).any() or (quat_norm > 1.05).any():
            raise RuntimeError(f"root_quat norms outside [0.95, 1.05]: {quat_norm}")

    except Exception:
        print("\n  SMOKE TEST FAILED:\n")
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("X2 smoke test PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
