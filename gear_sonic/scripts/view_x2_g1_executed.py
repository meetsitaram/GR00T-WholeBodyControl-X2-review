#!/usr/bin/env python3
"""X2 and G1 EXECUTED motions side by side in ONE MuJoCo viewer.

WHY
---
Reviewing "X2@ckpt vs stock-G1 on the same dance" previously needed two
processes (the C++ G1 deploy stack + the X2 python evaluator) in two windows
with hand-synced clip selection. This replays what each policy ACTUALLY DID —
captured executed trajectories — in a single scene on a shared clock:

  X2 side: NPZ from ``eval_x2_mujoco.py --record``  (qpos stream, 50 Hz)
  G1 side: CSV from the ONNX im_eval sweep with ``G1_SHIM_RECORD_DIR`` set
           (soma order: root cm + euler deg + 29 joint deg, 50 Hz)

KINEMATIC playback, like view_multi_qpos.py: no physics, no policy — the
captured qpos is written straight into the model. What you see is exactly the
behavior the eval metrics scored.

Usage:
    .venv/bin/python gear_sonic/scripts/view_x2_g1_executed.py \
        --x2-dir out/eval_v5/x2_executed_npz \
        --g1-dir out/eval_v5/g1_executed_csv \
        [--ratings out/eval_v5/ub_shortlist_ratings.json] [--clip <substr>]

Keys: SPACE pause | N/B next/prev clip | R restart | LEFT/RIGHT scrub 1 s.
Clips = sorted intersection of the two dirs; each side freezes on its last
frame if it ended (fell) before the other — a fall is visible as one robot
frozen mid-collapse while the other keeps dancing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
X2_MJCF = str(REPO / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml")
G1_MJCF = str(REPO / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml")
FPS = 50.0

# soma CSV joint column order written by g1_onnx_policy_shim._record_step
G1_SOMA_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def build_scene(spacing: float):
    """One world: X2 at y=0 (prefix x2_), G1 at y=spacing (prefix g1_)."""
    parent = mujoco.MjSpec()
    parent.compiler.degree = False
    parent.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[12, 12, 0.1],
        rgba=[0.32, 0.34, 0.36, 1.0])
    for mjcf, prefix, y in ((X2_MJCF, "x2_", 0.0), (G1_MJCF, "g1_", spacing)):
        child = mujoco.MjSpec.from_file(mjcf)
        frame = parent.worldbody.add_frame(pos=[0.0, y, 0.0])
        frame.attach_body(child.worldbody.first_body(), prefix, "")
    return parent.compile()


def robot_layout(model, prefix):
    """(qpos_root_adr, [joint qpos adrs by name]) for one attached robot."""
    root_adr = None
    joint_adr = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name or not name.startswith(prefix):
            continue
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            root_adr = model.jnt_qposadr[j]
        else:
            joint_adr[name[len(prefix):]] = model.jnt_qposadr[j]
    if root_adr is None:
        raise RuntimeError(f"no free joint with prefix {prefix}")
    return root_adr, joint_adr


def load_x2_npz(path: Path):
    """-> (T, 38) qpos [xyz, wxyz, 31 dof] at 50 Hz."""
    z = np.load(path, allow_pickle=True)
    joints = np.asarray(z["qpos"], dtype=np.float64)
    if "root" not in z or len(z["root"]) != len(joints):
        raise SystemExit(
            f"{path.name} lacks the root stream — re-record with the current "
            "eval_x2_mujoco.py (its --record now saves root qpos[0:7] too).")
    full = np.concatenate([np.asarray(z["root"], dtype=np.float64), joints], axis=1)
    return full, list(z["joint_names"])


def load_g1_csv(path: Path):
    """soma CSV -> root xyz (m), quat wxyz, dof (29, rad) at 50 Hz."""
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    root = d[:, 1:4] / 100.0                       # cm -> m
    quat_xyzw = R.from_euler("xyz", d[:, 4:7], degrees=True).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    dof = np.deg2rad(d[:, 7:])
    return root, quat_wxyz, dof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2-dir", type=Path, required=True)
    ap.add_argument("--g1-dir", type=Path, required=True)
    ap.add_argument("--ratings", type=Path, default=None,
                    help="optional ratings json; shown next to the clip name")
    ap.add_argument("--clip", default=None,
                    help="only clips whose name contains this substring")
    ap.add_argument("--spacing", type=float, default=1.25)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    x2_clips = {p.stem: p for p in args.x2_dir.glob("*.npz")}
    g1_clips = {p.stem: p for p in args.g1_dir.glob("*.csv")}
    names = sorted(set(x2_clips) & set(g1_clips))
    if args.clip:
        names = [n for n in names if args.clip.lower() in n.lower()]
    if not names:
        raise SystemExit(
            f"no common clips (x2: {len(x2_clips)}, g1: {len(g1_clips)}, "
            f"filter: {args.clip!r})")
    ratings = json.load(open(args.ratings)) if args.ratings else {}

    model = build_scene(args.spacing)
    data = mujoco.MjData(model)
    x2_root, x2_jadr = robot_layout(model, "x2_")
    g1_root, g1_jadr = robot_layout(model, "g1_")
    g1_soma_adr = [g1_jadr[j] for j in G1_SOMA_JOINTS if j in g1_jadr]
    if len(g1_soma_adr) != len(G1_SOMA_JOINTS):
        missing = [j for j in G1_SOMA_JOINTS if j not in g1_jadr]
        raise RuntimeError(f"G1 MJCF is missing soma joints: {missing}")

    state = {"idx": 0, "frame": 0, "paused": False, "jump": 0, "scrub": 0,
             "restart": False}

    def key_cb(keycode):
        import glfw
        if keycode == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif keycode == glfw.KEY_N:
            state["jump"] = 1
        elif keycode == glfw.KEY_B:
            state["jump"] = -1
        elif keycode == glfw.KEY_R:
            state["restart"] = True
        elif keycode == glfw.KEY_RIGHT:
            state["scrub"] = int(FPS)
        elif keycode == glfw.KEY_LEFT:
            state["scrub"] = -int(FPS)

    def load_clip(i):
        name = names[i]
        x2_qpos, x2_names = load_x2_npz(x2_clips[name])
        # X2 npz joint order == the attached x2_ model's joint order (same MJCF)
        g1_root_t, g1_quat_t, g1_dof_t = load_g1_csv(g1_clips[name])
        # The X2 eval set trims clips (12 s cap) while G1 CSVs come from the
        # raw sources (up to ~57 s). The X2 window IS the comparison window:
        # end the shared clock exactly there, cutting any longer G1 tail —
        # otherwise X2 reads as "cut off" while G1 dances on into material
        # the X2 eval never contained. A G1 that fell earlier stays frozen
        # mid-fall until the X2 side finishes.
        T = len(x2_qpos)
        tag = f" [{ratings[name]}]" if name in ratings else ""
        print(f"\n=== CLIP {i + 1}/{len(names)}: {name}{tag}  "
              f"x2 {len(x2_qpos) / FPS:.1f}s | g1 {len(g1_root_t) / FPS:.1f}s ===",
              flush=True)
        return dict(name=name, T=T, x2=x2_qpos,
                    g1=(g1_root_t, g1_quat_t, g1_dof_t))

    clip = load_clip(0)
    print("Keys: SPACE pause | N/B next/prev | R restart | LEFT/RIGHT scrub 1s",
          flush=True)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb,
                                      show_left_ui=False, show_right_ui=False) as v:
        v.cam.distance, v.cam.elevation = 4.5, -15
        v.cam.lookat[:] = [0.0, args.spacing / 2, 0.7]
        last = time.time()
        while v.is_running():
            now = time.time()
            if state["jump"]:
                state["idx"] = (state["idx"] + state["jump"]) % len(names)
                state["jump"] = 0
                clip = load_clip(state["idx"])
                state["frame"] = 0
            if state["restart"]:
                state["restart"] = False
                state["frame"] = 0
            if state["scrub"]:
                state["frame"] = int(np.clip(state["frame"] + state["scrub"],
                                             0, clip["T"] - 1))
                state["scrub"] = 0

            f = state["frame"]
            fx = min(f, len(clip["x2"]) - 1)          # freeze at last frame if ended
            data.qpos[x2_root:x2_root + len(clip["x2"][fx])] = clip["x2"][fx]
            g1_root_t, g1_quat_t, g1_dof_t = clip["g1"]
            fg = min(f, len(g1_root_t) - 1)
            # free-joint qpos is WORLD-anchored — the attach frame's Y offset
            # does not apply to it, so the separation is added to the data here.
            data.qpos[g1_root:g1_root + 3] = g1_root_t[fg]
            data.qpos[g1_root + 1] += args.spacing
            data.qpos[g1_root + 3:g1_root + 7] = g1_quat_t[fg]
            for a, val in zip(g1_soma_adr, g1_dof_t[fg]):
                data.qpos[a] = val
            mujoco.mj_kinematics(model, data)
            v.sync()

            if not state["paused"]:
                adv = (now - last) * FPS * args.speed
                if adv >= 1.0:
                    state["frame"] = int(state["frame"] + adv)
                    last = now
                    if state["frame"] >= clip["T"]:   # clip done -> next
                        state["idx"] = (state["idx"] + 1) % len(names)
                        clip = load_clip(state["idx"])
                        state["frame"] = 0
            else:
                last = now
            time.sleep(1.0 / (FPS * 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
