#!/usr/bin/env python3
"""Visual gate: trained X2 VQVAE encode-decode reconstruction.

Loads the trained X2 VQVAE checkpoint
(``out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt``), runs a real
BONES-SEED clip through the network (encoder -> codebook quantize -> decoder),
and plays both the **ground truth** and the **VQVAE reconstruction** in the
same MuJoCo viewer with the M key toggling between them.

This is the *first* model-in-the-loop visual:

  * If the network is well-trained, RT (recon) should track GT closely.
  * After only 200 smoke steps it WILL look terrible (limbs flying, drift,
    likely jittery feet) -- that is the expected, informative outcome that
    confirms (a) checkpoint loads correctly, (b) the inference path runs
    end-to-end, (c) the qpos converter accepts the network's output, (d)
    the only remaining work is more training compute.

Controls:
    M           toggle GT <-> RECON
    SPACE       pause
    R           restart
    LEFT/RIGHT  scrub +-10 frames

Usage::

    DISPLAY=:0 conda run -n motionbricks --no-capture-output python \\
        scripts/visualize_x2_vqvae_reconstruction.py \\
        --motion-key Relaxed_walk_forward
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

MB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MB_ROOT.parent
sys.path.insert(0, str(MB_ROOT))

from motionbricks.data.x2_bones_seed_dataset import build_x2_motion_rep  # noqa: E402
from motionbricks.data.x2_pkl_to_motion import (  # noqa: E402
    X2MujocoFkExtractor,
    assert_x2_mjcf_coherent,
    default_x2_mjcf_path,
)
from motionbricks.helper.data_training_util import (  # noqa: E402
    extract_feature_from_motion_rep,
)
from motionbricks.helper.mujoco_helper_x2 import X2MujocoQposConverter  # noqa: E402

NUM_DOFS = 31


def find_motion_key(lib: dict, query: str | None) -> str:
    keys = sorted(lib.keys())
    if query is None:
        return keys[0]
    if query in lib:
        return query
    matches = [k for k in keys if query.lower() in k.lower()]
    if not matches:
        raise KeyError(
            f"Motion key '{query}' not found. {len(keys)} entries; first few: "
            + ", ".join(keys[:6])
            + (" ..." if len(keys) > 6 else "")
        )
    return matches[0]


def load_vqvae(ckpt_path: Path, motion_rep, device: str = "cuda"):
    """Build a fresh VQVAE matching the X2 hparams and load smoke checkpoint."""
    hparams = OmegaConf.load(ckpt_path.parent.parent / "hparams.yaml")
    vqvae = instantiate(
        hparams.model.pose_vqvae_network,
        motion_rep=motion_rep.dual_rep.local_motion_rep,
        _recursive_=False,
    ).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    pose_net_sd = {
        k[len("pose_net."):]: v for k, v in sd.items() if k.startswith("pose_net.")
    }
    if not pose_net_sd:
        raise RuntimeError(f"No pose_net.* keys in {ckpt_path}")
    missing, unexpected = vqvae.load_state_dict(pose_net_sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return vqvae


def gt_local_features(
    clip: dict,
    motion_rep,
    extractor: X2MujocoFkExtractor,
) -> torch.Tensor:
    """Extract the normalized 413-D local rep features for one clip."""
    inp = extractor.clip_to_input_dict(clip, subsample=1)
    dual_feats = motion_rep.dual_rep(
        inp, to_normalize=True, return_numpy=False
    ).squeeze(0)  # [T, 418], normalized
    local_idx = motion_rep.dual_rep.indices["local_rep"]
    return dual_feats[:, local_idx]  # [T, 413], normalized


def features_to_qpos(
    feats_local_normalized: torch.Tensor,  # [B, T, 413] normalized local rep
    motion_rep,
    converter: X2MujocoQposConverter,
) -> np.ndarray:
    """Local rep -> global rep -> mujoco qpos (B=1)."""
    if feats_local_normalized.dim() == 2:
        feats_local_normalized = feats_local_normalized.unsqueeze(0)
    T = feats_local_normalized.shape[1]
    lengths = torch.tensor([T], device=feats_local_normalized.device)
    feats_global = motion_rep.dual_rep.local_to_global(
        feats_local_normalized,
        is_normalized=True,
        to_normalize=False,
        lengths=lengths,
    )  # [1, T, 414], unnormalized global rep
    qpos = converter.convert_motion_features_to_mujoco_qpos(
        feats_global.float(),
        motion_rep,
        is_normalized=False,
        root_quat_w_first=True,
    )
    return qpos[0].detach().cpu().numpy().astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/motions/x2_ultra_bones_seed.pkl",
    )
    parser.add_argument(
        "--motion-key", default=None, help="Motion key (substring match)"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=MB_ROOT / "out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--start-mode", choices=("gt", "recon"), default="gt"
    )
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Truncate clip to this many frames (VQVAE down_t=2 needs len%4==0).",
    )
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(
            f"VQVAE checkpoint not found: {args.ckpt}\n"
            "Run scripts/train_vqvae_x2.py first."
        )

    mjcf = default_x2_mjcf_path()
    print(f"X2 MJCF: {mjcf}")
    assert_x2_mjcf_coherent(mjcf)
    print("  -> sphere-feet coherence OK")

    skel_dir = MB_ROOT / "out/motionbricks_vqvae_x2/version_1/skeleton"
    stats_dir = MB_ROOT / "out/motionbricks_vqvae_x2/version_1/stats/motion"
    print("Loading motion_rep + qpos converter ...")
    motion_rep = build_x2_motion_rep(skel_dir, stats_dir, load_stats=True).to(
        args.device
    )
    extractor = X2MujocoFkExtractor(mjcf)
    converter = X2MujocoQposConverter(motion_rep, xml_path=str(mjcf)).to(args.device)

    print(f"Loading VQVAE checkpoint: {args.ckpt}")
    vqvae = load_vqvae(args.ckpt, motion_rep, device=args.device)

    print(f"Loading PKL: {args.pkl}")
    lib = joblib.load(args.pkl)
    key = find_motion_key(lib, args.motion_key)
    clip = lib[key]
    fps = float(clip["fps"])
    n_frames_full = int(clip["dof"].shape[0])
    n_frames = min(n_frames_full, args.max_frames)
    n_frames = (n_frames // 4) * 4  # VQVAE down_t=2 wants multiple of 4
    if n_frames < 32:
        raise RuntimeError(f"clip too short ({n_frames_full} frames)")
    clip_trunc = {
        "dof": clip["dof"][:n_frames],
        "root_trans_offset": clip["root_trans_offset"][:n_frames],
        "root_rot": clip["root_rot"][:n_frames],
        "fps": clip["fps"],
    }
    print(f"Clip: {key}  ({n_frames}/{n_frames_full} frames @ {fps:g} fps)")

    print("Computing GT and VQVAE-reconstructed qpos sequences ...")
    with torch.no_grad():
        gt_local = gt_local_features(clip_trunc, motion_rep, extractor).to(
            args.device
        )  # [T, 413] normalized

        # external_cond mode comes from the VQVAE hparams; for X2 / G1 it is
        # ``root_without_hip_height_without_heading`` -> a 2-D feature tied to
        # local_root_vel. We mirror what motion_inference does internally so
        # the reconstruction here matches what the full planner would feed at
        # decode time.
        local_motion_rep = motion_rep.dual_rep.local_motion_rep
        external_cond = extract_feature_from_motion_rep(
            gt_local.unsqueeze(0),
            local_motion_rep,
            "root_without_hip_height_without_heading",
        )

        has_target_cond = torch.ones(
            [1, gt_local.shape[0]], dtype=torch.bool, device=args.device
        )

        out = vqvae.forward(
            gt_local.unsqueeze(0),
            target_cond=gt_local.unsqueeze(0),
            has_target_cond=has_target_cond,
            external_cond=external_cond,
        )
        recon_local = out["recon_state"]  # [1, T, 413]
        if recon_local.shape[-1] != gt_local.shape[-1]:
            raise RuntimeError(
                f"VQVAE recon_state has {recon_local.shape[-1]} dims, "
                f"expected {gt_local.shape[-1]}. The 'pose' decoder mode is "
                "expected to return the full local rep."
            )

        qpos_gt = features_to_qpos(gt_local, motion_rep, converter)
        qpos_recon = features_to_qpos(recon_local, motion_rep, converter)

    perplex = float(out.get("perplexity", torch.tensor(float("nan"))).mean())
    err = float(np.abs(qpos_gt - qpos_recon).max())
    print(
        f"  qpos shape: {qpos_gt.shape}  "
        f"max |gt - recon| = {err:.3e} m/rad   "
        f"vqvae perplexity = {perplex:.2f}"
    )
    print("  (at 200 smoke steps perplexity is ~1 and error is huge - that's expected)")

    print(f"Loading MuJoCo model: {mjcf}")
    mj_model = mujoco.MjModel.from_xml_path(str(mjcf))
    mj_data = mujoco.MjData(mj_model)
    pelvis_id = mj_model.body("pelvis").id

    mode = [args.start_mode]  # "gt" or "recon"
    paused = [False]
    cur_frame = [0]

    def apply_frame(f: int) -> None:
        f = int(f) % n_frames
        q = qpos_gt if mode[0] == "gt" else qpos_recon
        mj_data.qpos[: 7 + NUM_DOFS] = q[f]
        mj_data.qvel[:] = 0.0
        mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(mj_model, mj_data)

    def key_callback(keycode):
        import glfw

        if keycode == glfw.KEY_M:
            mode[0] = "recon" if mode[0] == "gt" else "gt"
            print(
                f"[mode] {'GT (PKL)' if mode[0] == 'gt' else 'RECON (VQVAE)'}",
                flush=True,
            )
            apply_frame(cur_frame[0])
        elif keycode == glfw.KEY_SPACE:
            paused[0] = not paused[0]
            print("Paused" if paused[0] else "Resumed", flush=True)
        elif keycode == glfw.KEY_R:
            cur_frame[0] = 0
            apply_frame(0)
        elif keycode == glfw.KEY_LEFT:
            cur_frame[0] = max(0, cur_frame[0] - 10)
            apply_frame(cur_frame[0])
        elif keycode == glfw.KEY_RIGHT:
            cur_frame[0] = min(n_frames - 1, cur_frame[0] + 10)
            apply_frame(cur_frame[0])

    apply_frame(0)
    init_root_z = float(qpos_gt[0, 2])

    print(
        "\n=== X2 VQVAE reconstruction viewer ===\n"
        f"  starting in {'GT (PKL)' if mode[0] == 'gt' else 'RECON (VQVAE)'}\n"
        "  M      toggle GT <-> RECON\n"
        "  SPACE  pause\n"
        "  R      restart\n"
        "  LEFT/RIGHT  scrub +- 10 frames\n",
        flush=True,
    )

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0.0, 0.0, init_root_z]
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id

        frame_dt = 1.0 / (fps * max(args.speed, 1e-6))
        wall_start = time.time()
        wall_frame_origin = cur_frame[0]

        while viewer.is_running():
            if paused[0]:
                viewer.sync()
                time.sleep(0.02)
                continue

            elapsed = time.time() - wall_start
            target_frame = wall_frame_origin + int(elapsed / frame_dt)
            if not args.no_loop:
                target_frame = target_frame % n_frames
            elif target_frame >= n_frames:
                paused[0] = True
                print("End of clip.", flush=True)
                continue

            if target_frame != cur_frame[0]:
                cur_frame[0] = target_frame
                apply_frame(target_frame)

            viewer.sync()
            time.sleep(min(frame_dt, 0.02))

    return 0


if __name__ == "__main__":
    rc = main()
    # Skip Python teardown of the GLFW/EGL context: under Wayland + NVIDIA
    # the destructor races with the driver and segfaults on exit. The viewer
    # has already closed cleanly at this point so a hard exit is safe.
    os._exit(rc or 0)
