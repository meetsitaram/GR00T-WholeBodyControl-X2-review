"""View a saved kplanner replay alongside the source PKL clip.

This loads two X2 MJCF instances into one MuJoCo scene -- "actual" on
the left (the source motion clip) and "pred" on the right (what the
kplanner produced when given the clip's velocity intent) -- and plays
them synchronously frame-by-frame in a passive viewer.

Use this after running ``replay_pkl_through_kplanner.py --save-npz
/tmp/replay_fwd.npz`` to inspect the side-by-side trajectory and
confirm the diagnostic numbers visually.

Controls (focus the viewer window):

    SPACE   pause / resume playback
    R       reset playback to frame 0
    [ / ]   step one frame backward / forward when paused
    , / .   slow down / speed up the playback rate (0.25x .. 4x)

Usage::

  source .venv/bin/activate && \\
    PYTHONPATH="${PWD}/motionbricks:${PWD}" \\
    python motionbricks/scripts/view_kplanner_replay.py \\
      --npz /tmp/replay_fwd.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_mjcf() -> Path:
    return (
        REPO_ROOT
        / "gear_sonic"
        / "data"
        / "assets"
        / "robot_description"
        / "mjcf"
        / "x2_ultra.xml"
    )


def _build_scene(mjcf_path: Path, separation_m: float) -> mujoco.MjModel:
    """Compose a scene with two prefixed X2 instances, side-by-side."""
    parent = mujoco.MjSpec()
    parent.modelname = "x2_kplanner_replay"

    parent.option.timestep = 1.0 / 30.0
    parent.option.gravity = [0.0, 0.0, 0.0]

    parent.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], castshadow=False)
    parent.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.1],
        rgba=[0.85, 0.85, 0.85, 1.0],
    )

    actual_frame = parent.worldbody.add_frame(pos=[0.0, +separation_m / 2.0, 0.0])
    pred_frame = parent.worldbody.add_frame(pos=[0.0, -separation_m / 2.0, 0.0])

    child_a = mujoco.MjSpec.from_file(str(mjcf_path))
    child_b = mujoco.MjSpec.from_file(str(mjcf_path))

    parent.attach(child_a, prefix="actual_", frame=actual_frame)
    parent.attach(child_b, prefix="pred_", frame=pred_frame)

    return parent.compile()


def _root_qpos_addr(model: mujoco.MjModel, prefix: str) -> int:
    name = f"{prefix}floating_base_joint"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"Joint {name!r} not found in compiled model.")
    return int(model.jnt_qposadr[jid])


def _slice_qpos_into_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actual_qpos38: np.ndarray,
    pred_qpos38: np.ndarray,
    actual_addr: int,
    pred_addr: int,
    separation_m: float,
) -> None:
    """Write a single (38,) qpos for each robot into ``data.qpos``.

    The attach() frame offsets the world by +-separation/2 in Y. We
    keep the qpos translation in the clip's own world frame and let
    the attach-frame offset render them side-by-side.
    """
    data.qpos[actual_addr : actual_addr + 38] = actual_qpos38
    data.qpos[pred_addr : pred_addr + 38] = pred_qpos38


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True,
                   help="path to a .npz saved by replay_pkl_through_kplanner.py")
    p.add_argument("--mjcf", type=Path, default=None,
                   help="X2 MJCF path (defaults to canonical sphere-feet MJCF)")
    p.add_argument("--separation", type=float, default=2.0,
                   help="meters between the two robots in Y (default 2.0)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="playback fps; overrides the npz fps if provided")
    args = p.parse_args()

    if not args.npz.is_file():
        raise FileNotFoundError(f"NPZ not found: {args.npz}")
    blob = np.load(args.npz, allow_pickle=True)
    pred = np.asarray(blob["pred"], dtype=np.float32)
    actual = np.asarray(blob["actual"], dtype=np.float32)
    n = min(pred.shape[0], actual.shape[0])
    pred, actual = pred[:n], actual[:n]
    clip_key = str(blob["clip_key"]) if "clip_key" in blob.files else "?"
    saved_fps = float(blob["fps"]) if "fps" in blob.files else 30.0
    fps = args.fps if args.fps else saved_fps

    print(f"[viewer] clip = {clip_key}")
    print(f"[viewer] frames = {n}, fps = {fps:.1f} (saved={saved_fps:.1f})")
    print(f"[viewer] pred shape = {pred.shape}, actual shape = {actual.shape}")

    mjcf = args.mjcf or _default_mjcf()
    if not mjcf.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf}")
    print(f"[viewer] MJCF = {mjcf}")
    print(f"[viewer] separation = {args.separation:.2f} m  "
          "(actual = +Y / left, pred = -Y / right)")

    model = _build_scene(mjcf, args.separation)
    data = mujoco.MjData(model)
    actual_addr = _root_qpos_addr(model, "actual_")
    pred_addr = _root_qpos_addr(model, "pred_")
    print(f"[viewer] qpos addr: actual={actual_addr}, pred={pred_addr}, "
          f"model.nq={model.nq}")

    frame_idx = 0
    paused = False
    speed_mult = 1.0
    speeds = [0.25, 0.5, 1.0, 2.0, 4.0]

    def key_cb(keycode: int) -> None:
        nonlocal frame_idx, paused, speed_mult
        try:
            c = chr(keycode)
        except ValueError:
            return
        if c == " ":
            paused = not paused
            print(f"[viewer] paused={paused}")
        elif c == "R":
            frame_idx = 0
            print(f"[viewer] reset to frame 0")
        elif c == "[":
            frame_idx = max(0, frame_idx - 1)
            print(f"[viewer] frame {frame_idx}")
        elif c == "]":
            frame_idx = min(n - 1, frame_idx + 1)
            print(f"[viewer] frame {frame_idx}")
        elif c == ",":
            i = speeds.index(speed_mult) if speed_mult in speeds else 2
            speed_mult = speeds[max(0, i - 1)]
            print(f"[viewer] speed = {speed_mult}x")
        elif c == ".":
            i = speeds.index(speed_mult) if speed_mult in speeds else 2
            speed_mult = speeds[min(len(speeds) - 1, i + 1)]
            print(f"[viewer] speed = {speed_mult}x")

    print("[viewer] launching... SPACE=pause, R=reset, [/]=step, ,/.=speed")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]

        prev = time.time()
        while viewer.is_running():
            _slice_qpos_into_model(
                model, data,
                actual_qpos38=actual[frame_idx],
                pred_qpos38=pred[frame_idx],
                actual_addr=actual_addr,
                pred_addr=pred_addr,
                separation_m=args.separation,
            )
            mujoco.mj_forward(model, data)
            viewer.sync()

            now = time.time()
            dt_target = 1.0 / (fps * speed_mult)
            sleep_s = max(0.0, dt_target - (now - prev))
            if sleep_s > 0:
                time.sleep(sleep_s)
            prev = time.time()

            if not paused:
                frame_idx += 1
                if frame_idx >= n:
                    frame_idx = 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
