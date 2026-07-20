#!/usr/bin/env python3
"""Superimpose planner reference vs robot-executed motion from a captured run.

WHY
---
The capture pipeline records BOTH sides of the tracking contract:

  frame_tape.f32   what the kplanner daemon published to SONIC (the reference)
  joint_pos.csv    what the robot actually did (SONIC's output, measured)

Numbers (leg-vel spikes, swing events) say WHERE they diverge; this shows HOW.
Two copies of the X2 are rendered in one scene on a shared clock:

  ghost (translucent blue)  = planner reference wire
  solid                     = robot measured joints + IMU orientation

Root translation is shared (the planner's) because the telemetry has no
odometry; orientation of the solid robot comes from the IMU, yaw-aligned to
the reference at the window start. So: watch LIMBS and TILT, not XY drift.

    python gear_sonic/scripts/overlay_run_mujoco.py \
        docs/experiments/robot_runs/20260719_223057_walks_residual_stumbles \
        --window 452 460          # first back walk, tape-clock seconds

The robot->tape clock offset is auto-derived by cross-correlating hip pitch
(reference vs measured); pass --clock-offset to pin it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MJCF = str(
    GEAR_SONIC_ROOT / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
)
NUM_DOFS = 31
FRAME_REC = 40  # tm, branch, xy, z, quat_xyzw, jpos31


def _load_frame_tape(path: Path):
    raw = np.fromfile(path, dtype=np.float32)
    n = raw.size // FRAME_REC
    f = raw[: n * FRAME_REC].reshape(n, FRAME_REC).astype(np.float64)
    return {
        "tm": f[:, 0], "branch": f[:, 1], "xy": f[:, 2:4], "z": f[:, 4],
        "quat_xyzw": f[:, 5:9], "jpos": f[:, 9:40],
    }


def _load_csv(path: Path):
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


def _auto_offset(tape, jp_t, jp0) -> float:
    """Robot->tape clock offset by cross-correlating hip pitch."""
    wt = np.arange(tape["tm"][0], tape["tm"][-1], 0.02)
    ws = np.interp(wt, tape["tm"], tape["jpos"][:, 0])
    it = np.arange(jp_t[0], jp_t[-1], 0.02)
    isig = np.interp(it, jp_t, jp0)
    base = it[0] - wt[0]
    n = min(len(ws), len(isig))
    best, boff = -np.inf, 0.0
    for off in np.arange(-40, 40, 0.1):
        sh = int(off / 0.02)
        a = ws[max(0, -sh): n - max(0, sh)]
        b = isig[max(0, sh): n - max(0, -sh)]
        k = min(len(a), len(b))
        if k < 1000:
            continue
        c = np.corrcoef(a[:k] - a[:k].mean(), b[:k] - b[:k].mean())[0, 1]
        if c > best:
            best, boff = c, off
    return base + boff


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _quat_mul(a, b):  # wxyz
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _yaw_of(q_wxyz):
    w, x, y, z = q_wxyz
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_dual_model(mjcf_path: str):
    import mujoco

    ref = mujoco.MjSpec.from_file(mjcf_path)
    act = mujoco.MjSpec.from_file(mjcf_path)
    # ghost styling on the reference copy: translucent blue, no material
    for g in ref.geoms:
        g.material = ""
        g.rgba = [0.35, 0.55, 1.0, 0.35]
    for s in (ref, act):
        for g in s.geoms:
            g.contype = 0
            g.conaffinity = 0
    frame = act.worldbody.add_frame()
    act.attach(ref, prefix="ref_", frame=frame)
    act.visual.global_.offwidth = max(act.visual.global_.offwidth, 1280)
    act.visual.global_.offheight = max(act.visual.global_.offheight, 720)
    model = act.compile()

    def qpos_addr(prefix):
        import mujoco as mj
        free = None
        dof_adr = []
        for j in range(model.njnt):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or ""
            is_ref = name.startswith("ref_")
            if (prefix == "ref_") != is_ref:
                continue
            if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                free = model.jnt_qposadr[j]
            else:
                dof_adr.append(model.jnt_qposadr[j])
        return free, np.array(dof_adr, dtype=int)

    return model, qpos_addr("ref_"), qpos_addr("")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                   help="tape-clock window in seconds (default: whole tape)")
    p.add_argument("--mjcf", default=DEFAULT_MJCF)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--clock-offset", type=float, default=None,
                   help="robot_clock - tape_clock; auto if omitted")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--save-video", type=Path, default=None,
                   help="render offscreen to this mp4 instead of a viewer")
    args = p.parse_args(argv)

    tape = _load_frame_tape(args.run_dir / "frame_tape.f32")
    jp = _load_csv(args.run_dir / "joint_pos.csv")
    imu = _load_csv(args.run_dir / "imu.csv")
    jp_t = jp.iloc[:, 0].to_numpy()
    jp_q = jp.iloc[:, 1: 1 + NUM_DOFS].to_numpy()
    imu_t = imu.iloc[:, 0].to_numpy()
    imu_q = imu[["qw", "qx", "qy", "qz"]].to_numpy()

    off = args.clock_offset
    if off is None:
        print("[overlay] deriving clock offset (hip-pitch cross-correlation)...",
              flush=True)
        off = _auto_offset(tape, jp_t, jp_q[:, 0])
    print(f"[overlay] robot->tape clock offset: {off:.2f}s", flush=True)
    jp_t = jp_t - off
    imu_t = imu_t - off

    t0 = args.window[0] if args.window else tape["tm"][0]
    t1 = args.window[1] if args.window else tape["tm"][-1]
    sel = (tape["tm"] >= t0) & (tape["tm"] <= t1)
    if sel.sum() < 5:
        print(f"[overlay] ERROR: no tape frames in window {t0}..{t1}",
              file=sys.stderr)
        return 1
    tm = tape["tm"][sel]
    print(f"[overlay] window {tm[0]:.2f}..{tm[-1]:.2f}s "
          f"({sel.sum()} wire ticks)", flush=True)

    # robot joints/orientation interpolated onto wire ticks
    act_j = np.stack([np.interp(tm, jp_t, jp_q[:, k])
                      for k in range(NUM_DOFS)], axis=1)
    act_quat = np.stack([np.interp(tm, imu_t, imu_q[:, k])
                         for k in range(4)], axis=1)
    act_quat /= np.linalg.norm(act_quat, axis=1, keepdims=True)

    ref_quat = np.stack([_xyzw_to_wxyz(q) for q in tape["quat_xyzw"][sel]])
    # yaw-align IMU to reference at window start (IMU yaw datum is arbitrary)
    dyaw = _yaw_of(ref_quat[0]) - _yaw_of(act_quat[0])
    align = _yaw_quat(dyaw)
    act_quat = np.stack([_quat_mul(align, q) for q in act_quat])

    ref_pos = np.column_stack([tape["xy"][sel], tape["z"][sel]])
    ref_j = tape["jpos"][sel]
    branch = tape["branch"][sel]

    import mujoco
    import mujoco.viewer

    model, (ref_free, ref_dofs), (act_free, act_dofs) = \
        build_dual_model(args.mjcf)
    if len(ref_dofs) != NUM_DOFS or len(act_dofs) != NUM_DOFS:
        print(f"[overlay] ERROR: dof mismatch ref={len(ref_dofs)} "
              f"act={len(act_dofs)} vs {NUM_DOFS}", file=sys.stderr)
        return 1
    data = mujoco.MjData(model)

    n = len(tm)

    def set_frame(i):
        data.qpos[ref_free: ref_free + 3] = ref_pos[i]
        data.qpos[ref_free + 3: ref_free + 7] = ref_quat[i]
        data.qpos[ref_dofs] = ref_j[i]
        data.qpos[act_free: act_free + 3] = ref_pos[i]  # shared root
        data.qpos[act_free + 3: act_free + 7] = act_quat[i]
        data.qpos[act_dofs] = act_j[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    if args.save_video:
        import imageio
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body(
            [model.body(b).name for b in range(model.nbody)
             if "pelvis" in (model.body(b).name or "")
             and not model.body(b).name.startswith("ref_")][0]).id
        cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0
        renderer = mujoco.Renderer(model, height=480, width=768)
        writer = imageio.get_writer(str(args.save_video), fps=50,
                                    codec="libx264", quality=7)
        for i in range(n):
            set_frame(i)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            if i % 500 == 0:
                print(f"[overlay] rendered {i}/{n}", flush=True)
        writer.close()
        renderer.close()
        print(f"[overlay] wrote {args.save_video} ({n} frames @ 50fps)")
        return 0

    i = 0
    target_dt = 0.02 / max(1e-6, args.speed)
    print("[overlay] ghost blue = planner reference; solid = robot measured. "
          "Esc to exit.", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_start = time.perf_counter()
            set_frame(i)
            viewer.sync()
            i += 1
            if i >= n:
                if args.loop:
                    i = 0
                else:
                    print("[overlay] end of window; close viewer to exit.")
                    i = n - 1
                    time.sleep(0.25)
            rem = target_dt - (time.perf_counter() - t_start)
            if rem > 0:
                time.sleep(rem)
    _ = branch  # branch stream available for future HUD use
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
