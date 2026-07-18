#!/usr/bin/env python3
"""Kinematically play the planner's direction clips, cycling with TAB.

Companion to ``view_idle_anchor.py``. Where that shows a single standing pose,
this plays the motions the pad actually commands -- forward / back / stride
left-right / turn left-right -- so a bad stance or an asymmetric gait is visible
before it reaches hardware.

Purely kinematic: qpos is set per frame and mj_forward runs. There is no physics
and no policy, so what you see is the REFERENCE the planner emits, not what sonic
manages to track. That separation is the point -- it tells you whether a problem
is in the reference or in the tracking.

Joint order note: clip ``dof`` is written straight into ``qpos[7:38]`` with NO
permutation, exactly as the runtime does in
``pc2_kplanner_onnx._qpos_from_deploy_pkl_frame``. Applying a joint remap here
renders a pose that exists in no motion -- verified the hard way.

    python gear_sonic/scripts/view_direction_clips.py                  # /tmp/dirclips
    python gear_sonic/scripts/view_direction_clips.py --clips a.pkl b.pkl

Keys: TAB / SPACE = next clip, P = pause, R = restart clip, ESC = quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_x2_mujoco import MJCF_PATH, NUM_DOFS  # noqa: E402

DEFAULT_DIR = Path("/tmp/dirclips")
ORDER = ["forward", "back", "stride_left", "stride_right", "turn_left", "turn_right"]


def load_clips(paths: list[Path]) -> list[tuple[str, dict]]:
    out = []
    for p in paths:
        if not p.is_file():
            print(f"  skip (missing): {p}")
            continue
        lib = joblib.load(p)
        for key, clip in lib.items():
            out.append((key, clip))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, nargs="*", default=None)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate multiplier")
    args = ap.parse_args()

    paths = args.clips or [args.dir / f"{n}.pkl" for n in ORDER]
    clips = load_clips(paths)
    if not clips:
        print("no clips loaded", file=sys.stderr)
        return 1

    print(f"loaded {len(clips)} clips:")
    for key, c in clips:
        dof = np.asarray(c["dof"])
        tr = np.asarray(c["root_trans_offset"])
        disp = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
        print(f"   {key:<16} {dof.shape[0]:>4}f  disp={disp:.2f}m  fps={c.get('fps')}")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    st = {"clip": 0, "frame": 0, "paused": False}

    def apply(ci: int, fi: int) -> None:
        _, c = clips[ci]
        dof = np.asarray(c["dof"], dtype=np.float64)
        rot = np.asarray(c["root_rot"], dtype=np.float64)     # xyzw
        tr = np.asarray(c["root_trans_offset"], dtype=np.float64)
        fi = fi % dof.shape[0]
        data.qpos[0:3] = tr[fi]
        q = rot[fi]
        data.qpos[3:7] = [q[3], q[0], q[1], q[2]]             # xyzw -> wxyz
        data.qpos[7 : 7 + NUM_DOFS] = dof[fi]                 # verbatim, MJ order
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def announce() -> None:
        key, c = clips[st["clip"]]
        n = np.asarray(c["dof"]).shape[0]
        print(f"\n>>> [{st['clip'] + 1}/{len(clips)}] {key}  ({n} frames)", flush=True)

    def key_cb(keycode: int) -> None:
        import glfw
        if keycode in (glfw.KEY_TAB, glfw.KEY_SPACE):
            st["clip"] = (st["clip"] + 1) % len(clips)
            st["frame"] = 0
            announce()
        elif keycode == glfw.KEY_P:
            st["paused"] = not st["paused"]
            print("  paused" if st["paused"] else "  playing", flush=True)
        elif keycode == glfw.KEY_R:
            st["frame"] = 0

    apply(0, 0)
    announce()
    print("\n  TAB / SPACE = next clip, P = pause, R = restart, ESC = quit")

    import time
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -12
        viewer.cam.azimuth = 135
        while viewer.is_running():
            _, c = clips[st["clip"]]
            fps = float(c.get("fps", 50.0)) * max(0.05, args.speed)
            apply(st["clip"], st["frame"])
            # track the pelvis so the robot does not walk out of frame
            viewer.cam.lookat[:] = data.qpos[0:3]
            viewer.sync()
            if not st["paused"]:
                st["frame"] += 1
            time.sleep(1.0 / fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
