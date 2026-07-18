#!/usr/bin/env python3
"""Kinematic side-by-side of idle-anchor poses in MuJoCo (no physics, no policy).

Purpose: eyeball the standing pose an anchor commands BEFORE it reaches hardware.
The v2 anchor carries a ~0.39 rad hip_pitch stagger (one leg planted forward) plus
~0.37 rad of hip_yaw on that leg, which reads on the robot as "the front right leg
is way forward and it wants to stay in that odd position".

Purely kinematic: qpos is set directly and mj_forward runs, so what you see is the
commanded pose itself, with no tracking error or controller behaviour mixed in.

    python gear_sonic/scripts/view_idle_anchor.py                 # v2 vs v3
    python gear_sonic/scripts/view_idle_anchor.py --anchors a.pkl b.pkl

Keys: TAB / SPACE = cycle anchors, R = reset camera, ESC = quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_x2_mujoco import IL_TO_MJ_DOF, MJCF_PATH, NUM_DOFS  # noqa: E402

MOTIONS = REPO / "gear_sonic" / "data" / "motions"
DEFAULT_ANCHORS = [
    MOTIONS / "kplanner_idle_anchor_g1teleop_v2.pkl",
    MOTIONS / "kplanner_idle_anchor_g1teleop_v3.pkl",
]
LEG_NAMES = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
# roll/yaw/ankle_roll flip across the sagittal plane; pitch/knee stay.
MIRROR = np.array([1, -1, -1, 1, 1, -1])


def load_anchor(path: Path) -> tuple[str, np.ndarray]:
    lib = joblib.load(path)
    key = next(iter(lib))
    dof = np.asarray(lib[key]["dof"], dtype=np.float64).reshape(-1)
    if dof.shape[0] != NUM_DOFS:
        raise ValueError(f"{path.name}: expected {NUM_DOFS} dof, got {dof.shape[0]}")
    return key, dof


def report(name: str, dof: np.ndarray) -> None:
    left, right = dof[0:6], dof[6:12]
    asym = np.abs(left - MIRROR * right)
    verdict = "SYMMETRIC" if asym.max() < 0.05 else "*** ASYMMETRIC ***"
    print(f"\n  {name}  -> {verdict} (max {asym.max():.3f} rad)")
    print(f"    hip_pitch stagger (leg-forward): {abs(left[0] - right[0]):.3f} rad")
    for i, n in enumerate(LEG_NAMES):
        mark = "  <--" if asym[i] > 0.05 else ""
        print(f"      {n:<12} L={left[i]:+.3f}  R={right[i]:+.3f}  asym={asym[i]:.3f}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=Path, nargs="*", default=DEFAULT_ANCHORS)
    ap.add_argument("--height", type=float, default=0.72, help="pelvis z for viewing")
    args = ap.parse_args()

    anchors = []
    for p in args.anchors:
        if not p.is_file():
            print(f"  skip (missing): {p}")
            continue
        key, dof = load_anchor(p)
        anchors.append((p.name, key, dof))
        report(p.name, dof)
    if not anchors:
        print("no anchors loaded", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    idx = {"i": 0}

    def apply(i: int) -> None:
        name, key, dof = anchors[i]
        data.qpos[:] = 0.0
        data.qpos[2] = args.height          # lift the pelvis so the pose is visible
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # NO permutation. The runtime consumes this file verbatim --
        # pc2_kplanner_onnx._qpos_from_deploy_pkl_frame does `qpos[7:38] = f0_dof`
        # -- so the anchor's dof is ALREADY in MuJoCo joint order. Applying
        # IL_TO_MJ_DOF (in either direction) scrambles the joints and renders a
        # pose that exists in no motion. Match the runtime exactly, or this
        # viewer shows something the robot will never do.
        data.qpos[7 : 7 + NUM_DOFS] = dof
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)      # kinematics only -- never mj_step
        print(f"\n>>> showing [{i + 1}/{len(anchors)}] {name}  ({key})", flush=True)

    def key_cb(keycode: int) -> None:
        import glfw
        if keycode in (glfw.KEY_TAB, glfw.KEY_SPACE):
            idx["i"] = (idx["i"] + 1) % len(anchors)
            apply(idx["i"])

    apply(0)
    print("\n  TAB / SPACE = next anchor, ESC = quit")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -12
        viewer.cam.azimuth = 135          # 3/4 view: leg stagger is clearest here
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
