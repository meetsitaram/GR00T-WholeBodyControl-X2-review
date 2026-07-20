#!/usr/bin/env python3
"""View N motion trajectories side by side in ONE MuJoCo viewer.

WHY
---
This is KINEMATIC playback: the qpos stream is written straight into the
model and rendered. No policy, no SONIC, no physics stepping. So what you see
is exactly what the planner emitted -- the reference, not a robot's attempt to
track it. That distinction is the whole point: when the live stack misbehaves
we can never tell by eye whether the reference was bad or the tracker failed.
Here there is no tracker.

Robots are laid out along Y with a fixed spacing so the same instant is
comparable across settings, and each keeps its own world XY translation so
divergence in travel and heading is visible as separation over time.

    # planner settings from the intent probe
    python gear_sonic/scripts/view_multi_qpos.py \
        --npz out/frame_eval/intent_probe/*.npz

    # include the real clip as a reference robot (leftmost)
    python gear_sonic/scripts/view_multi_qpos.py --npz out/.../*.npz \
        --ref gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gear_sonic.scripts.kplanner_frame_eval import (  # noqa: E402
    _mjcf_path, load_qpos)


def build_scene(mjcf: str, n: int, spacing: float):
    """One world containing n prefixed copies of the robot."""
    parent = mujoco.MjSpec()
    parent.compiler.degree = False
    # a visible ground so foot height is judgeable by eye
    parent.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[12, 12, 0.1],
        rgba=[0.32, 0.34, 0.36, 1.0])
    for i in range(n):
        child = mujoco.MjSpec.from_file(mjcf)
        frame = parent.worldbody.add_frame(pos=[0.0, i * spacing, 0.0])
        body = child.worldbody.first_body()
        frame.attach_body(body, f"r{i}_", "")
    return parent.compile()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="*", type=Path, default=[])
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--ref-key", default=None)
    ap.add_argument("--mjcf", default=None)
    ap.add_argument("--spacing", type=float, default=1.4)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true", default=True)
    args = ap.parse_args()

    srcs = []
    if args.ref:
        q, fps = load_qpos(args.ref, args.ref_key)
        srcs.append((f"REF {args.ref.stem[:28]}", q, fps))
    for p in sorted(args.npz):
        q, fps = load_qpos(p, None)
        srcs.append((p.stem.split("__")[-1][:28], q, fps))
    if not srcs:
        raise SystemExit("nothing to view: pass --npz and/or --ref")

    mjcf = _mjcf_path(args.mjcf)
    n = len(srcs)
    model = build_scene(mjcf, n, args.spacing)
    data = mujoco.MjData(model)

    # qpos slice per robot, located by its prefixed free joint
    starts = []
    for i in range(n):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r{i}_")
        if j < 0:                      # free joint may inherit the body name
            j = next(k for k in range(model.njnt)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, k)
                         or "").startswith(f"r{i}_"))
        starts.append(int(model.jnt_qposadr[j]))
    width = min(np.diff(starts).min() if n > 1 else model.nq, 38)

    fps = srcs[0][2]
    T = max(s[1].shape[0] for s in srcs)
    print(f"  {n} robots, {width} qpos each, spacing {args.spacing} m")
    for i, (lbl, q, f) in enumerate(srcs):
        print(f"    y={i*args.spacing:+.1f}  {lbl}  ({q.shape[0]} frames @ {f:g}fps)")
    print("  space = pause, right-arrow = step while paused")

    with mujoco.viewer.launch_passive(model, data) as v:
        t0, k = time.time(), 0
        while v.is_running():
            k = int((time.time() - t0) * fps * args.speed)
            if k >= T:
                if not args.loop:
                    break
                t0, k = time.time(), 0
            for i, (_lbl, q, _f) in enumerate(srcs):
                f = min(k, q.shape[0] - 1)
                a = starts[i]
                w = min(width, q.shape[1])
                data.qpos[a:a + w] = q[f, :w]
                data.qpos[a + 1] += i * args.spacing      # lay out along Y
            mujoco.mj_kinematics(model, data)
            v.sync()
            time.sleep(1.0 / (fps * args.speed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
