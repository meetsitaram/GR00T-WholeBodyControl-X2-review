"""Cycle through ALL clips in a motion-lib PKL in ONE MuJoCo viewer.

Plays each clip in sequence and AUTO-ADVANCES to the next at clip end, so a
whole generated batch is reviewable in one shot -- no relaunch per clip. The
tracking camera follows the pelvis; the current key is printed on every switch.

Keys:
    SPACE   pause / resume
    N / P   next / previous clip
    R       restart current clip
    , / .   slower / faster
    1..9    jump to clip N

    python gear_sonic/scripts/cycle_x2_motions.py \
        --motion out/kplanner_gen_proof_g1/proof_batch_g1retarget.pkl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np

GEAR_SONIC_ROOT = Path(__file__).resolve().parents[2]
MJCF_PATH = str(
    GEAR_SONIC_ROOT / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
)
NUM_DOFS = 31


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--motion", required=True, help="motion-lib PKL (multi-clip)")
    ap.add_argument("--loops-per-clip", type=int, default=1,
                    help="loops before auto-advancing to the next clip (default 1)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1x)")
    ap.add_argument("--sort", action="store_true", help="sort clip keys alphabetically")
    ap.add_argument("--mjcf", default=MJCF_PATH, help="robot MJCF (default X2; pass g1_29dof.xml for G1)")
    ap.add_argument("--num-dofs", type=int, default=NUM_DOFS, help="joint DOFs (X2=31, G1=29)")
    args = ap.parse_args()
    mjcf_path = args.mjcf
    ndof = args.num_dofs

    data = joblib.load(args.motion)
    keys = sorted(data.keys()) if args.sort else list(data.keys())
    if not keys:
        raise SystemExit(f"no clips in {args.motion}")
    print(f"Loaded {len(keys)} clips from {args.motion}", flush=True)

    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    pelvis_id = mj_model.body("pelvis").id

    st = {"ci": 0, "frame": 0, "paused": False, "loops": 0, "speed": float(args.speed)}
    arr: dict = {}

    def load_clip(ci: int) -> None:
        st["ci"] = ci % len(keys)
        m = data[keys[st["ci"]]]
        if m["dof"].shape[1] != ndof:
            raise RuntimeError(f"{keys[st['ci']]}: expected {ndof} DOFs, got {m['dof'].shape[1]}")
        arr["dof"] = np.asarray(m["dof"], np.float64)
        arr["pos"] = np.asarray(m["root_trans_offset"], np.float64)
        arr["quat_xyzw"] = np.asarray(m["root_rot"], np.float64)
        arr["fps"] = float(m["fps"])
        arr["n"] = m["dof"].shape[0]
        st["frame"] = 0
        st["loops"] = 0
        print(f">>> [{st['ci']+1}/{len(keys)}] {keys[st['ci']]:34s} "
              f"{arr['n']}f @ {arr['fps']:g}fps = {arr['n']/arr['fps']:.1f}s", flush=True)

    def apply(f: int) -> None:
        f = int(f) % arr["n"]
        mj_data.qpos[0:3] = arr["pos"][f]
        mj_data.qpos[3] = arr["quat_xyzw"][f, 3]      # wxyz <- xyzw
        mj_data.qpos[4:7] = arr["quat_xyzw"][f, 0:3]
        mj_data.qpos[7:7 + ndof] = arr["dof"][f]
        mj_data.qvel[:] = 0.0
        mj_data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

    clock = {"start": time.time(), "origin": 0}

    def resync() -> None:
        clock["start"] = time.time()
        clock["origin"] = st["frame"]

    def key_cb(keycode: int) -> None:
        import glfw
        if keycode == glfw.KEY_SPACE:
            st["paused"] = not st["paused"]; resync()
            print("paused" if st["paused"] else "resumed", flush=True)
        elif keycode == glfw.KEY_N:
            load_clip(st["ci"] + 1); resync()
        elif keycode == glfw.KEY_P:
            load_clip(st["ci"] - 1); resync()
        elif keycode == glfw.KEY_R:
            st["frame"] = 0; st["loops"] = 0; resync()
        elif keycode == glfw.KEY_COMMA:
            st["speed"] = max(0.1, st["speed"] / 1.5); print(f"speed {st['speed']:.2f}x", flush=True)
        elif keycode == glfw.KEY_PERIOD:
            st["speed"] = min(4.0, st["speed"] * 1.5); print(f"speed {st['speed']:.2f}x", flush=True)
        elif glfw.KEY_1 <= keycode <= glfw.KEY_9:
            target = keycode - glfw.KEY_1
            if target < len(keys):
                load_clip(target); resync()

    load_clip(0)
    apply(0)
    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_cb, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        print("keys: SPACE pause | N/P next/prev clip | R restart | ,/. speed | 1-9 jump",
              flush=True)
        while viewer.is_running():
            if not st["paused"]:
                elapsed = (time.time() - clock["start"]) * arr["fps"] * st["speed"]
                f = clock["origin"] + elapsed
                if f >= arr["n"]:
                    st["loops"] += 1
                    if st["loops"] >= args.loops_per_clip:
                        load_clip(st["ci"] + 1)
                    else:
                        st["frame"] = 0
                    resync()
                    f = 0
                st["frame"] = int(f) % arr["n"]
            apply(st["frame"])
            viewer.sync()
            time.sleep(1.0 / max(arr["fps"] * st["speed"], 1.0))


if __name__ == "__main__":
    main()
