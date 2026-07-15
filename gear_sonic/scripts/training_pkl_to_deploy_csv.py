#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201
"""Convert a G1 training-format motion PKL into a deploy `reference/<clip>/` CSV dir.

The C++ G1 deploy (`g1_deploy_onnx_ref`) plays back motions from CSV directories
(`joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, `metadata.txt`),
NOT from PKLs. This script bridges the gap in one hop: it takes the motion_lib
training PKL produced by `convert_soma_csv_to_motion_lib.py`
(`{root_trans_offset, root_rot(xyzw), pose_aa, dof(29, MuJoCo order), fps}`),
runs MuJoCo forward-kinematics on the G1 29-DOF MJCF to get world body poses,
resamples to the 50 Hz deploy control rate, reorders joints MuJoCo->IsaacLab,
and writes the CSV layout the deploy's `MotionDataReader` expects.

MuJoCo FK is used (not Humanoid_Batch) because it needs no extra deps beyond the
sim venv and its body frames match the sim2sim deploy exactly (verified: the G1
29-DOF MJCF reproduces the reference lower-body `body_pos` to float precision).

Usage:
    .venv_sim/bin/python gear_sonic/scripts/training_pkl_to_deploy_csv.py \
        --pkl /tmp/g1_squat_trainpkl.pkl \
        --out-dir gear_sonic_deploy/reference
    # -> gear_sonic_deploy/reference/<pkl_name>/<motion_name>/{joint_pos,joint_vel,body_pos,body_quat}.csv + metadata.txt
"""

import argparse
import os

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# --- G1 29-DOF constants -----------------------------------------------------
# MJ_TO_IL[i] = IsaacLab index feeding MuJoCo slot i  (mj = il[MJ_TO_IL]).
# Inverse recovers IsaacLab order from MuJoCo order:  il = mj[IL_FROM_MJ].
G1_MJ_TO_IL = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19,
     21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int32,
)
IL_FROM_MJ = np.argsort(G1_MJ_TO_IL)

G1_NUM_DOF = 29
DEPLOY_FPS = 50  # deploy control rate (control_dt_ = 0.02 s)

# The 14 tracked bodies the deploy reference format stores, in CSV column order.
# `MJ_BODY_IDS` index into the g1_29dof MJCF body table (world=0, pelvis=1, ...);
# `META_INDICES` are the parallel IsaacLab body indices written to metadata.txt.
# Order: pelvis, L/R (hip_roll, knee, ankle_roll), torso, L/R (shoulder_roll, elbow, wrist_yaw).
G1_TRACKED_MJ_BODY_IDS = np.array(
    [1, 3, 5, 7, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30], dtype=np.int32
)
G1_TRACKED_META_INDICES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]

DEFAULT_MJCF = "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"


def resample(root_t, root_r_xyzw, dof, fps_src, fps_tgt):
    """Resample (root_trans, root_rot xyzw, dof) to fps_tgt. Linear for trans/dof, slerp for rot."""
    T = dof.shape[0]
    if T < 2 or fps_src == fps_tgt:
        return root_t, root_r_xyzw, dof
    dur = (T - 1) / fps_src
    n_tgt = int(round(dur * fps_tgt)) + 1
    t_src = np.arange(T) / fps_src
    t_tgt = np.linspace(0.0, dur, n_tgt)
    root_t_r = np.stack([np.interp(t_tgt, t_src, root_t[:, i]) for i in range(3)], axis=1)
    dof_r = np.stack([np.interp(t_tgt, t_src, dof[:, i]) for i in range(dof.shape[1])], axis=1)
    root_r_r = Slerp(t_src, Rotation.from_quat(root_r_xyzw))(t_tgt).as_quat()
    return (
        root_t_r.astype(np.float32),
        root_r_r.astype(np.float32),
        dof_r.astype(np.float32),
    )


def mujoco_fk(model, data, root_t, root_r_xyzw, dof):
    """Forward kinematics -> world (pos, quat wxyz) for the 14 tracked bodies. Returns (T,14,3),(T,14,4)."""
    T = dof.shape[0]
    root_r_wxyz = root_r_xyzw[:, [3, 0, 1, 2]]
    body_pos = np.zeros((T, len(G1_TRACKED_MJ_BODY_IDS), 3), dtype=np.float32)
    body_quat = np.zeros((T, len(G1_TRACKED_MJ_BODY_IDS), 4), dtype=np.float32)
    for t in range(T):
        data.qpos[0:3] = root_t[t]
        data.qpos[3:7] = root_r_wxyz[t]
        data.qpos[7 : 7 + G1_NUM_DOF] = dof[t]
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[G1_TRACKED_MJ_BODY_IDS]
        body_quat[t] = data.xquat[G1_TRACKED_MJ_BODY_IDS]
    return body_pos, body_quat


def write_csv(path, array2d, headers):
    """Write (T, C) array as a headered CSV, 6 decimals (matches convert_motions.py)."""
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in array2d:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def write_metadata(path, name, T):
    """Write metadata.txt. The reader parses ONLY the single line after 'Body part indexes:'."""
    idx_line = "[" + " ".join(str(i) for i in G1_TRACKED_META_INDICES) + "]"
    with open(path, "w") as f:
        f.write(f"Metadata for: {name}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n")
        f.write(idx_line + "\n\n")
        f.write(f"Total timesteps: {T}\n\n")
        f.write("Data arrays summary:\n")
        f.write(f"  joint_pos: ({T}, {G1_NUM_DOF}) (float32)\n")
        f.write(f"  joint_vel: ({T}, {G1_NUM_DOF}) (float32)\n")
        nb = len(G1_TRACKED_META_INDICES)
        f.write(f"  body_pos_w: ({T}, {nb}, 3) (float32)\n")
        f.write(f"  body_quat_w: ({T}, {nb}, 4) (float32)\n")
        f.write(f"  _body_indexes: ({nb},) (int64)\n")


def convert_motion(name, entry, model, data, out_dir, fps_tgt):
    """Convert one training-PKL entry to a deploy CSV directory."""
    root_t = np.asarray(entry["root_trans_offset"], dtype=np.float32)
    root_r = np.asarray(entry["root_rot"], dtype=np.float32)  # xyzw
    dof_mj = np.asarray(entry["dof"], dtype=np.float32)[:, :G1_NUM_DOF]  # MuJoCo order
    fps_src = int(entry.get("fps", fps_tgt))

    root_t, root_r, dof_mj = resample(root_t, root_r, dof_mj, fps_src, fps_tgt)
    T = dof_mj.shape[0]

    # joint_pos: MuJoCo -> IsaacLab order (deploy expects IsaacLab-order joint columns)
    joint_pos_il = dof_mj[:, IL_FROM_MJ]
    joint_vel_il = (np.gradient(joint_pos_il, axis=0) * fps_tgt).astype(np.float32)

    body_pos, body_quat = mujoco_fk(model, data, root_t, root_r, dof_mj)

    os.makedirs(out_dir, exist_ok=True)
    write_csv(
        os.path.join(out_dir, "joint_pos.csv"),
        joint_pos_il,
        [f"joint_{i}" for i in range(G1_NUM_DOF)],
    )
    write_csv(
        os.path.join(out_dir, "joint_vel.csv"),
        joint_vel_il,
        [f"joint_vel_{i}" for i in range(G1_NUM_DOF)],
    )
    bp = body_pos.reshape(T, -1)
    write_csv(
        os.path.join(out_dir, "body_pos.csv"),
        bp,
        [f"body_{i // 3}_{'xyz'[i % 3]}" for i in range(bp.shape[1])],
    )
    bq = body_quat.reshape(T, -1)
    write_csv(
        os.path.join(out_dir, "body_quat.csv"),
        bq,
        [f"body_{i // 4}_{'wxyz'[i % 4]}" for i in range(bq.shape[1])],
    )
    write_metadata(os.path.join(out_dir, "metadata.txt"), name, T)
    print(f"  {name}: {T} frames @ {fps_tgt} Hz (src {fps_src}) -> {out_dir}")
    return T


def main():
    ap = argparse.ArgumentParser(description="G1 training PKL -> deploy reference CSV dir")
    ap.add_argument("--pkl", required=True, help="training-format motion PKL (joblib dict)")
    ap.add_argument(
        "--out-dir",
        default="gear_sonic_deploy/reference",
        help="base reference dir; clip subdir is created under it",
    )
    ap.add_argument(
        "--clip-name",
        default=None,
        help="clip subdir name (default: PKL basename without extension)",
    )
    ap.add_argument("--mjcf", default=DEFAULT_MJCF, help="G1 29-DOF MJCF for FK")
    ap.add_argument("--fps", type=int, default=DEPLOY_FPS, help="target output fps (default 50)")
    args = ap.parse_args()

    clip_name = args.clip_name or os.path.splitext(os.path.basename(args.pkl))[0]
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    if model.nq != 7 + G1_NUM_DOF:
        raise SystemExit(f"MJCF nq={model.nq}, expected {7 + G1_NUM_DOF} (7 floating + {G1_NUM_DOF})")

    motions = joblib.load(args.pkl)
    clip_dir = os.path.join(args.out_dir, clip_name)
    print(f"Converting {len(motions)} motion(s) from {args.pkl}")
    print(f"Output: {clip_dir}/<motion>/")
    for name, entry in motions.items():
        convert_motion(name, entry, model, data, os.path.join(clip_dir, name), args.fps)
    print(f"Done. Play with:  deploy.sh sim --input-type keyboard --motion-data reference/{clip_name}/")


if __name__ == "__main__":
    main()
