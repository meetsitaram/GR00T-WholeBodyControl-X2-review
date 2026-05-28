#!/usr/bin/env python3
"""Visual verification: replay an X2 BONES-SEED clip through the MotionBricks pipeline.

Plays a chosen motion-lib clip in MuJoCo two ways in a single viewer:

  * RAW    — qpos taken directly from the SONIC PKL (ground truth).
  * RT     — qpos reconstructed via the MotionBricks pipeline:
             FK -> dual_rep features (418-D) -> normalize -> unnormalize ->
             X2MujocoQposConverter -> qpos[T, 38].

Press ``M`` to toggle which version drives the on-screen X2 (avoid TAB —
MuJoCo's viewer reserves it for the side-panel UI).
Press ``SPACE`` to pause, ``R`` to restart, ``LEFT/RIGHT`` to scrub.

If the round-trip is faithful, you should not see a visible difference between
RAW and RT. Any drift = a bug in the FK / motion-rep / converter chain.

Usage::

    DISPLAY=:1 conda run -n motionbricks --no-capture-output python \\
        scripts/visualize_x2_round_trip.py --motion-key Loop_Walk_Forward

    # default (first clip):
    DISPLAY=:1 conda run -n motionbricks --no-capture-output python \\
        scripts/visualize_x2_round_trip.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import torch

MB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MB_ROOT.parent
sys.path.insert(0, str(MB_ROOT))

from motionbricks.data.x2_bones_seed_dataset import build_x2_motion_rep  # noqa: E402
from motionbricks.data.x2_pkl_to_motion import (  # noqa: E402
    X2MujocoFkExtractor,
    assert_x2_mjcf_coherent,
    default_x2_mjcf_path,
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


def raw_qpos_from_clip(clip: dict, *, canonicalize: bool = True) -> np.ndarray:
    """Build qpos[T, 38] directly from the PKL fields (no MotionBricks).

    If ``canonicalize`` is True (default), apply the same first-frame
    canonicalization that MotionBricks performs internally:
    place pelvis XY at origin and rotate every frame so frame 0 faces MuJoCo
    +X (the conventional forward axis). This makes the raw qpos directly
    comparable to the round-trip qpos.
    """
    from scipy.spatial.transform import Rotation as Rot

    dof = np.asarray(clip["dof"], dtype=np.float64)  # (T, 31), MuJoCo order
    root_pos = np.asarray(clip["root_trans_offset"], dtype=np.float64)  # (T, 3)
    root_xyzw = np.asarray(clip["root_rot"], dtype=np.float64)  # (T, 4) scipy xyzw
    T = dof.shape[0]
    qpos = np.zeros((T, 7 + NUM_DOFS), dtype=np.float64)
    qpos[:, :3] = root_pos
    # MuJoCo wants wxyz
    qpos[:, 3] = root_xyzw[:, 3]
    qpos[:, 4:7] = root_xyzw[:, :3]
    qpos[:, 7:] = dof

    if canonicalize:
        # Yaw of frame 0 in MuJoCo (rotation around Z).
        yaw0 = Rot.from_quat(root_xyzw[0]).as_euler("ZYX")[0]
        Rz_inv = Rot.from_euler("Z", -yaw0)
        # Derotate position about the initial XY (anchored at origin).
        xy = root_pos[:, :2] - root_pos[0, :2]
        xyz = np.column_stack([xy, root_pos[:, 2]])
        xyz = Rz_inv.apply(xyz)
        qpos[:, :3] = xyz
        # Derotate root rotation.
        rot_world = Rot.from_quat(root_xyzw)
        rot_canon = Rz_inv * rot_world
        canon_xyzw = rot_canon.as_quat()
        qpos[:, 3] = canon_xyzw[:, 3]
        qpos[:, 4:7] = canon_xyzw[:, :3]
    return qpos


def round_trip_qpos_from_clip(
    clip: dict,
    motion_rep,
    extractor: X2MujocoFkExtractor,
    converter: X2MujocoQposConverter,
    use_normalized_round_trip: bool = True,
) -> np.ndarray:
    """Run the full MotionBricks pipeline on a single clip and return qpos[T, 38]."""
    inp = extractor.clip_to_input_dict(clip, subsample=1)
    feats = motion_rep.dual_rep(inp, to_normalize=False, return_numpy=False)
    if feats.dim() == 3:
        feats = feats.squeeze(0)

    # Optional: round-trip through normalize / unnormalize to exercise stats.
    # Use the *global rep* path for the qpos converter (matches inference time).
    global_indices = motion_rep.dual_rep.indices["global_rep"]
    global_feats = feats[..., global_indices]  # [T, 414]
    if use_normalized_round_trip:
        normalized = motion_rep.normalize(global_feats)
        global_feats = motion_rep.unnormalize(normalized)

    qpos = converter.convert_motion_features_to_mujoco_qpos(
        global_feats.unsqueeze(0).float(),  # [1, T, 414]
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
        "--motion-key",
        default=None,
        help="Motion key (substring match allowed). Default: first clip.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--start-mode",
        choices=("raw", "rt"),
        default="raw",
        help="Which version to show first (TAB toggles).",
    )
    parser.add_argument(
        "--skip-stats-norm",
        action="store_true",
        help="Skip the normalize -> unnormalize step in the round-trip pipeline. "
        "Useful for isolating FK vs stats issues.",
    )
    parser.add_argument(
        "--no-loop", action="store_true", help="Stop after one playthrough"
    )
    parser.add_argument(
        "--debug-rt-offset",
        type=float,
        default=0.0,
        metavar="METERS",
        help="Add a constant XY offset (in meters) to the RT qpos so the toggle "
        "is visually obvious. Default 0 -> RAW and RT should look identical. "
        "Set e.g. 0.5 to confirm the M key actually swaps data paths.",
    )
    args = parser.parse_args()

    mjcf = default_x2_mjcf_path()
    print(f"X2 MJCF: {mjcf}")
    assert_x2_mjcf_coherent(mjcf)
    print("  -> sphere-feet coherence OK (nq=38, njnt=32, sphere geoms >= 24)")

    skel_dir = MB_ROOT / "out/motionbricks_vqvae_x2/version_1/skeleton"
    stats_dir = MB_ROOT / "out/motionbricks_vqvae_x2/version_1/stats/motion"
    if not skel_dir.is_dir() or not stats_dir.is_dir():
        raise FileNotFoundError(
            "X2 skeleton / stats not built. Run "
            "`python scripts/build_x2_skeleton_assets.py` first."
        )

    print("Loading X2 motion_rep + qpos converter ...")
    motion_rep = build_x2_motion_rep(skel_dir, stats_dir, load_stats=True)
    extractor = X2MujocoFkExtractor(mjcf)
    converter = X2MujocoQposConverter(motion_rep, xml_path=str(mjcf))

    print(f"Loading PKL: {args.pkl}")
    lib = joblib.load(args.pkl)
    key = find_motion_key(lib, args.motion_key)
    clip = lib[key]
    fps = float(clip["fps"])
    n_frames = int(clip["dof"].shape[0])
    print(f"Clip: {key}  ({n_frames} frames @ {fps:g} fps)")
    if clip["dof"].shape[1] != NUM_DOFS:
        raise RuntimeError(f"Expected 31 DOFs, got {clip['dof'].shape[1]}")

    print("Computing both qpos sequences ...")
    qpos_raw = raw_qpos_from_clip(clip)
    qpos_rt = round_trip_qpos_from_clip(
        clip,
        motion_rep,
        extractor,
        converter,
        use_normalized_round_trip=not args.skip_stats_norm,
    )
    if qpos_raw.shape != qpos_rt.shape:
        raise RuntimeError(
            f"qpos shape mismatch: raw={qpos_raw.shape}, rt={qpos_rt.shape}"
        )
    err = float(np.abs(qpos_raw - qpos_rt).max())
    print(
        f"  qpos shape: {qpos_raw.shape} (T x 38)  "
        f"max |raw - rt| across all dims = {err:.3e}"
    )
    print(
        "  Note: the root quaternion can flip sign without changing pose; "
        "ignore differences exclusively in qpos[3:7] sign."
    )
    if args.debug_rt_offset != 0.0:
        qpos_rt = qpos_rt.copy()
        qpos_rt[:, 0] += args.debug_rt_offset
        qpos_rt[:, 1] += args.debug_rt_offset
        print(
            f"  [debug] applied +{args.debug_rt_offset:+.2f} m XY offset to RT "
            "to make the M-toggle visually obvious."
        )

    print(f"Loading MuJoCo model: {mjcf}")
    mj_model = mujoco.MjModel.from_xml_path(str(mjcf))
    mj_data = mujoco.MjData(mj_model)
    pelvis_id = mj_model.body("pelvis").id

    mode = [args.start_mode]  # "raw" or "rt"
    paused = [False]
    cur_frame = [0]

    def apply_frame(f: int) -> None:
        f = int(f) % n_frames
        q = qpos_raw if mode[0] == "raw" else qpos_rt
        mj_data.qpos[: 7 + NUM_DOFS] = q[f]
        mj_data.qvel[:] = 0.0
        mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(mj_model, mj_data)

    def key_callback(keycode):
        import glfw

        if keycode == glfw.KEY_M:
            mode[0] = "rt" if mode[0] == "raw" else "raw"
            print(f"[mode] {'RAW (PKL)' if mode[0] == 'raw' else 'RT  (MotionBricks)'}",
                  flush=True)
            apply_frame(cur_frame[0])
        elif keycode == glfw.KEY_SPACE:
            paused[0] = not paused[0]
            print("Paused" if paused[0] else "Resumed", flush=True)
        elif keycode == glfw.KEY_R:
            cur_frame[0] = 0
            apply_frame(0)
            print("[reset] frame 0", flush=True)
        elif keycode == glfw.KEY_LEFT:
            cur_frame[0] = max(0, cur_frame[0] - 10)
            apply_frame(cur_frame[0])
        elif keycode == glfw.KEY_RIGHT:
            cur_frame[0] = min(n_frames - 1, cur_frame[0] + 10)
            apply_frame(cur_frame[0])

    apply_frame(0)
    init_root_z = float(qpos_raw[0, 2])

    print(
        "\n=== X2 round-trip viewer ===\n"
        f"  starting in {'RAW (PKL)' if mode[0] == 'raw' else 'RT  (MotionBricks)'}\n"
        "  M      toggle RAW <-> RT  (NOT tab — MuJoCo reserves TAB for UI)\n"
        "  SPACE  pause / resume\n"
        "  R      restart from frame 0\n"
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
                print("End of clip.", flush=True)
                paused[0] = True
                continue

            if target_frame != cur_frame[0]:
                cur_frame[0] = target_frame
                apply_frame(target_frame)

            viewer.sync()
            time.sleep(min(frame_dt, 0.02))

    return 0


if __name__ == "__main__":
    rc = main()
    # Hard exit to avoid a Wayland + NVIDIA + GLFW teardown segfault when
    # Python's GC unwinds the viewer's GL context. The viewer has already
    # closed by this point, so it's safe.
    os._exit(rc or 0)
