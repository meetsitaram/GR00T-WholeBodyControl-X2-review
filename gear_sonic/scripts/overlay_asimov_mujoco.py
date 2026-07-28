#!/usr/bin/env python3
"""MuJoCo overlay viewer for Asimov eval results: solid = EXECUTED policy
motion, translucent blue ghost = reference motion. Mirrors the X2 tool
overlay_run_mujoco.py (dual-robot MjSpec attach scene).

Inputs:
  --dump   traj_rank0.pkl written by im_eval with IM_EVAL_DUMP_TRAJ=<dir>
           (needs the exec_qpos field: root pos+quat wxyz + joint_pos in
           IsaacLab order, added 2026-07-27)
  --ref    the motion pkl the eval ran against (reference dof/root, MuJoCo order)
  --clip   motion key to show (default: first in dump)
  --save-video <out.mp4>  offscreen render instead of live viewer

Live viewer (needs display):
    python gear_sonic/scripts/overlay_asimov_mujoco.py --dump <dir>/traj_rank0.pkl \
        --ref gear_sonic/data/motions/asimov_walk1.pkl [--clip <key>] [--speed 1.0]
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import time

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MJCF = os.path.join(_REPO, "gear_sonic/data/assets/robot_description/mjcf/asimov.xml")

# IsaacLab -> MuJoCo dof gather (dof_mj[i] = dof_il[src]); generated arrays in
# robots/asimov.py ("isaaclab_to_mujoco_dof" = IL source index for MJ slot i).
ASIMOV_IL_SRC_FOR_MJ = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2,
                        6, 10, 14, 18, 22, 5, 9, 13, 17, 21]


def build_scene():
    solid = mujoco.MjSpec.from_file(MJCF)
    ghost = mujoco.MjSpec.from_file(MJCF)
    for g in ghost.geoms:
        g.rgba = [0.35, 0.55, 1.0, 0.35]
        g.contype = 0
        g.conaffinity = 0
    frame = solid.worldbody.add_frame()
    frame.attach_body(ghost.worldbody.first_body(), "ref_", "")
    model = solid.compile()
    return model, mujoco.MjData(model)


def set_qpos(model, data, prefix, root7, dof_mj):
    j0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base")
    adr = model.jnt_qposadr[j0]
    data.qpos[adr:adr + 7] = root7                     # pos + quat wxyz
    for k in range(23):
        jn = prefix + MJ_JOINT_NAMES[k]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = dof_mj[k]


MJ_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    d = pickle.load(open(args.dump, "rb"))
    keys = d.get("motion_keys") or []
    exec_list = d.get("exec_qpos")
    assert exec_list, "dump has no exec_qpos — rerun eval with the extended callback"
    idx = keys.index(args.clip) if (args.clip and args.clip in keys) else 0
    key = keys[idx] if keys else args.clip
    ex = np.asarray(exec_list[idx])                    # (T, 7+23) IL dof order
    fps = float(d.get("fps", 50))

    ref = joblib.load(args.ref)
    rkey = key if key in ref else list(ref)[0]
    re_ = ref[rkey]
    rdof = re_["dof"]                                  # (Tr, 23) MJ order, rad
    rroot = np.concatenate(
        [re_["root_trans_offset"],
         np.asarray(re_["root_rot"])[:, [3, 0, 1, 2]]], axis=1)  # xyzw->wxyz
    rfps = float(re_["fps"])

    model, data = build_scene()
    T = len(ex)
    print(f"[overlay] clip {key}: {T} exec frames @{fps}fps | ref {len(rdof)} @{rfps}fps")
    print("[overlay] ghost blue = reference; solid = EXECUTED policy motion")

    def frame_at(t_sec):
        fe = min(int(t_sec * fps), T - 1)
        fr = min(int(t_sec * rfps), len(rdof) - 1)
        e = ex[fe]
        dof_mj = e[7:][ASIMOV_IL_SRC_FOR_MJ]
        set_qpos(model, data, "", e[:7], dof_mj)
        set_qpos(model, data, "ref_", rroot[fr], rdof[fr])
        mujoco.mj_forward(model, data)

    dur = (T - 1) / fps
    if args.save_video:
        step_t = 1.0 / 60.0
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{args.width}x{args.height}", "-r", "60", "-i", "-",
             "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
             args.save_video], stdin=subprocess.PIPE)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 3.0, -12.0, 135.0
        with mujoco.Renderer(model, height=args.height, width=args.width) as ren:
            t = 0.0
            while t <= dur:
                frame_at(t)
                cam.lookat = [data.qpos[0], data.qpos[1], 0.7]
                ren.update_scene(data, camera=cam)
                ff.stdin.write(ren.render().tobytes())
                t += step_t
        ff.stdin.close()
        ff.wait()
        print(f"[overlay] wrote {args.save_video}")
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = ((time.time() - t0) * args.speed) % dur
            frame_at(t)
            viewer.sync()
            time.sleep(0.005)


if __name__ == "__main__":
    main()
