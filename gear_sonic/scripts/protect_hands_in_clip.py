#!/usr/bin/env python3
"""Hold the hands pointing FORWARD in world frame during a floor recovery.

Problem (measured on ``recover_side_R``): while getting up, the robot pushes off
its hands. FK puts the right wrist link at z = -0.094 m -- below the floor plane
-- so the palm/fingers carry body weight. The soft hands are not built for that.

Fix: for every frame where the robot is still down, re-solve the THREE wrist
joints so the hand's pointing axis is horizontal and aligned with the robot's
heading, instead of aimed at the ground. Contact then lands on the rigid wrist /
forearm rather than the palm. As the robot stands up the constraint is ramped
out and the clip returns to its captured wrist motion.

Geometry: the hand link's collision geom sits at local z = -0.075 (half-length
0.107), so the hand points along the link's LOCAL -Z. That is the axis we aim.

Only wrist_yaw / wrist_pitch / wrist_roll are touched -- shoulders, elbows,
legs and root are untouched, so the gross recovery mechanics are preserved.
Joint limits are respected (wrist_pitch is only +-0.56 rad, so the solve leans
on yaw and roll). The result is ramped with a half-cosine and the per-frame jump
is reported: a step in the reference is exactly what destabilises SONIC.

    python gear_sonic/scripts/protect_hands_in_clip.py \
        --in /tmp/kneel_and_recover.pkl --clip recover_side_R \
        --out /tmp/kneel_and_recover_safe.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_x2_mujoco import MJCF_PATH, NUM_DOFS  # noqa: E402

WRIST = {"L": (19, 20, 21), "R": (26, 27, 28)}          # yaw, pitch, roll
HAND_BODY = {"L": "left_wrist_roll_link", "R": "right_wrist_roll_link"}
HAND_AXIS_LOCAL = np.array([0.0, 0.0, -1.0])            # hand points along -Z


def hand_dir_and_z(model, data, dof, rot, tr, f, side):
    """World unit vector the hand points along, and the hand link's world z."""
    data.qpos[0:3] = tr[f]
    q = rot[f]
    data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
    data.qpos[7 : 7 + NUM_DOFS] = dof
    mujoco.mj_forward(model, data)
    b = data.body(HAND_BODY[side])
    R = np.array(b.xmat).reshape(3, 3)
    return R @ HAND_AXIS_LOCAL, float(b.xpos[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="/tmp/kneel_and_recover.pkl")
    ap.add_argument("--clip", default="recover_side_R")
    ap.add_argument("--out", default="/tmp/kneel_and_recover_safe.pkl")
    ap.add_argument("--up-z", type=float, default=0.55,
                    help="pelvis z above which the robot counts as 'up'; the "
                         "constraint ramps out here")
    ap.add_argument("--ramp-frames", type=int, default=25)
    args = ap.parse_args()

    lib = joblib.load(args.src)
    clip = lib[args.clip]
    dof0 = np.asarray(clip["dof"], dtype=np.float64)
    rot = np.asarray(clip["root_rot"], dtype=np.float64)
    tr = np.asarray(clip["root_trans_offset"], dtype=np.float64)
    n = dof0.shape[0]
    dof = dof0.copy()

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    lims = {}
    for s, idx in WRIST.items():
        lims[s] = np.array([model.jnt_range[model.joint(
            f"{'left' if s == 'L' else 'right'}_{j}_joint").id]
            for j in ("wrist_yaw", "wrist_pitch", "wrist_roll")])

    # "Forward" = robot heading projected to the ground plane, per frame.
    def forward_dir(f):
        q = rot[f]  # xyzw
        yaw = np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]),
                         1 - 2 * (q[1] ** 2 + q[2] ** 2))
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    before_z = {s: np.zeros(n) for s in WRIST}
    after_z = {s: np.zeros(n) for s in WRIST}
    before_dot = {s: np.zeros(n) for s in WRIST}
    after_dot = {s: np.zeros(n) for s in WRIST}

    # weight: 1 while down, ramped to 0 as the robot stands
    w = np.clip((args.up_z - tr[:, 2]) / max(1e-6, args.up_z * 0.35), 0.0, 1.0)
    if args.ramp_frames > 1:
        k = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 2 * args.ramp_frames + 1)))
        w = np.convolve(w, k / k.sum(), mode="same")
    w = np.clip(w, 0.0, 1.0)

    for f in range(n):
        tgt = forward_dir(f)
        for s, idx in WRIST.items():
            d0, z0 = hand_dir_and_z(model, data, dof0[f], rot, tr, f, s)
            before_z[s][f] = z0
            before_dot[s][f] = float(np.dot(d0, np.array([0, 0, -1.0])))
            if w[f] < 1e-3:
                after_z[s][f], after_dot[s][f] = z0, before_dot[s][f]
                continue

            def resid(x, _f=f, _s=s, _idx=idx, _tgt=tgt):
                trial = dof[_f].copy()
                trial[list(_idx)] = x
                d, _ = hand_dir_and_z(model, data, trial, rot, tr, _f, _s)
                return d - _tgt

            x0 = dof0[f, list(idx)]
            sol = least_squares(resid, x0, bounds=(lims[s][:, 0], lims[s][:, 1]),
                                max_nfev=40, xtol=1e-3, ftol=1e-3)
            blended = x0 * (1 - w[f]) + sol.x * w[f]
            dof[f, list(idx)] = np.clip(blended, lims[s][:, 0], lims[s][:, 1])
            d1, z1 = hand_dir_and_z(model, data, dof[f], rot, tr, f, s)
            after_z[s][f] = z1
            after_dot[s][f] = float(np.dot(d1, np.array([0, 0, -1.0])))

    print(f"  {args.clip}: {n} frames, constraint active on "
          f"{int((w > 0.5).sum())} frames ({(w > 0.5).sum()/50:.1f}s)\n")
    print(f"  {'':<6}{'min hand z':>22}{'max downward-ness':>26}")
    print(f"  {'':<6}{'before':>11}{'after':>11}{'before':>13}{'after':>13}")
    for s in ("L", "R"):
        print(f"  {s:<6}{before_z[s].min():>11.3f}{after_z[s].min():>11.3f}"
              f"{before_dot[s].max():>13.2f}{after_dot[s].max():>13.2f}")
    print("  (downward-ness = hand axis . (0,0,-1); 1.0 = straight down, "
          "0 = horizontal)")
    jump = float(np.abs(np.diff(dof, axis=0)).max())
    print(f"\n  max per-frame joint jump: {jump:.3f} rad "
          f"({'OK' if jump < 0.30 else '*** CHECK ***'})")

    lib[args.clip + "_handsafe"] = {
        "dof": dof.astype(np.float32), "root_rot": rot.astype(np.float32),
        "root_trans_offset": tr.astype(np.float32), "fps": clip.get("fps", 50.0)}
    joblib.dump(lib, args.out)
    print(f"  wrote {args.out} (added '{args.clip}_handsafe'; original kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
