#!/usr/bin/env python3
"""Offscreen-render a SONIC policy rollout on X2 Ultra to an MP4.

Same RSI + PD-control loop as ``eval_x2_mujoco.py`` but renders every control
step with ``mujoco.Renderer`` instead of the interactive viewer, then muxes
the frames into an MP4 with imageio/ffmpeg.

✅ Faithful PT eval: this script imports :class:`UniversalTokenActor` from
   :mod:`eval_x2_mujoco`, which has been verified to match the live
   ``UniversalTokenModule`` to ~3.6e-7 rad on the iter-2000 sphere-feet
   checkpoint (validated against a fresh
   ``gear_sonic/scripts/dump_isaaclab_step0.py`` dump). So the rollouts
   recorded here track the deployed policy faithfully. For ONNX-driven
   evaluation, use :mod:`eval_x2_mujoco_onnx` instead.

Example:
    conda run -n env_isaaclab --no-capture-output python \\
        gear_sonic/scripts/record_x2_eval_mujoco.py \\
        --checkpoint $HOME/x2_cloud_checkpoints/run-20260420_083925/last.pt \\
        --motion   gear_sonic/data/motions/x2_ultra_take_a_sip.pkl \\
        --out      /tmp/x2_take_a_sip.mp4 \\
        --duration 8.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import csv

import imageio
import mujoco
import mujoco.viewer  # noqa: F401  -- needed for --onscreen live viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[2]


def load_deploy_tuning(yaml_path: str):
    """Mirror the C++ deploy's target conditioning from a real_deploy_tuning
    YAML: per-group kp/kd scales on the TRAINING PD base, per-joint
    max-target-deviation clamps, and per-group first-order target LPF.

    Returns (kp, kd, dev, lpf_alpha, action_clip); dev/lpf_alpha are (31,)
    arrays (dev=inf and alpha=1.0 disable their stages).
    """
    import yaml as _yaml
    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)

    def scale_for(jname, prefix):
        # most-specific key wins: ankle_pitch > ankle > (hip for kd) ...
        pats = ["ankle_pitch", "ankle_roll", "waist_yaw", "knee", "hip",
                "shoulder", "elbow", "wrist", "head"]
        for p in pats:
            if p in jname and f"{prefix}_{p}" in cfg:
                return float(cfg[f"{prefix}_{p}"])
        if ("waist_pitch" in jname or "waist_roll" in jname) \
                and f"{prefix}_waist_pr" in cfg:
            return float(cfg[f"{prefix}_waist_pr"])
        return 1.0

    def group_of(jname):
        if any(p in jname for p in ("hip", "knee", "ankle")):
            return "leg"
        if "waist" in jname:
            return "waist"
        if "head" in jname:
            return "head"
        return "arm"

    kp = np.zeros(NUM_DOFS)
    kd = np.zeros(NUM_DOFS)
    dev = np.full(NUM_DOFS, np.inf)
    alpha = np.ones(NUM_DOFS)
    for i, jname in enumerate(MUJOCO_JOINT_NAMES):
        kp_train = kd_train = 0.0
        for key, arm in ARMATURES.items():
            if key in jname:
                kp_train = arm * NATURAL_FREQ ** 2
                kd_train = 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ
                break
        kp[i] = kp_train * scale_for(jname, "kp_scale")
        kd[i] = kd_train * scale_for(jname, "kd_scale")
        g = group_of(jname)
        d = cfg.get(f"max_target_dev_{g}", cfg.get("max_target_dev"))
        if d is not None:
            dev[i] = float(d)
        f_hz = cfg.get(f"target_lpf_hz_{g}", cfg.get("target_lpf_hz", 0.0))
        f_hz = float(f_hz or 0.0)
        if f_hz > 0.0:
            w = 2.0 * np.pi * f_hz * CONTROL_DT
            alpha[i] = w / (1.0 + w)
    return kp, kd, dev, alpha, float(cfg.get("action_clip", np.inf))
sys.path.insert(0, str(REPO))

from gear_sonic.scripts.eval_x2_mujoco import (  # noqa: E402
    ACTION_SCALE,
    ARMATURES,
    CONTROL_DT,
    DAMPING_RATIO,
    DECIMATION,
    DEFAULT_DOF,
    JOINT_TO_ACTUATOR,
    KD,
    KP,
    MUJOCO_JOINT_NAMES,
    NATURAL_FREQ,
    IL_TO_MJ_DOF,
    MJ_TO_IL_DOF,
    MJCF_PATH,
    NUM_DOFS,
    SIM_DT,
    ProprioceptionBuffer,
    build_tokenizer_obs,
    compute_motion_state,
    load_actor_from_checkpoint,
    load_playlist_motion_data,
    quat_rotate_inverse,
)


def apply_init_state(mj_model, mj_data, motion_state):
    s = motion_state
    mj_data.qpos[0] = 0.0
    mj_data.qpos[1] = 0.0
    mj_data.qpos[2] = float(s["root_pos_w"][2])
    mj_data.qpos[3:7] = s["root_quat_w_wxyz"]
    mj_data.qpos[7:7 + NUM_DOFS] = s["joint_pos_mj"]
    mj_data.qvel[0:3] = s["root_lin_vel_w"]
    mj_data.qvel[3:6] = quat_rotate_inverse(
        s["root_quat_w_wxyz"], s["root_ang_vel_w"]
    )
    mj_data.qvel[6:6 + NUM_DOFS] = s["joint_vel_mj"]
    mj_data.xfrc_applied[:] = 0
    mujoco.mj_forward(mj_model, mj_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    motion_grp = parser.add_mutually_exclusive_group(required=True)
    motion_grp.add_argument("--motion",
                             help="Single-clip motion-lib PKL (first key is used).")
    motion_grp.add_argument(
        "--playlist",
        help="Warehouse playlist YAML (resolved via _warehouse_playlist."
             "build_concat). Mutually exclusive with --motion.",
    )
    parser.add_argument("--out", required=False, default=None,
                        help="Output MP4 path (required unless --onscreen/--no-render).")
    parser.add_argument("--onscreen", action="store_true",
                        help="Live interactive MuJoCo viewer (launch_passive) at "
                             "real-time pacing, looping the clip. No MP4. Loads the "
                             ".pt directly -- no ONNX/docker. SPACE pause, R restart.")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Seconds of rollout to record (default 10).")
    parser.add_argument("--init-frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cam-azimuth", type=float, default=120.0)
    parser.add_argument("--cam-elevation", type=float, default=-20.0)
    parser.add_argument("--cam-distance", type=float, default=3.0)
    parser.add_argument("--render-fps", type=int, default=50,
                        help="Output video FPS (default 50 = 1 frame/control step).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mjcf", default=None,
                        help="Override MJCF path. Used by Phase 5 sim2sim "
                             "ablation audits to A/B variant MJCFs without "
                             "editing the canonical x2_ultra.xml. Defaults to "
                             "MJCF_PATH from eval_x2_mujoco.")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip MP4 writing; just run the sim and report "
                             "fall time. ~5-10x faster for headless A/B audits.")
    parser.add_argument("--tuning-yaml", default=None,
                        help="real_deploy_tuning YAML (e.g. walking_soft_kp"
                             ".yaml). Applies the DEPLOY's kp/kd scales, "
                             "max-target-dev clamps and target LPF so the "
                             "sim conditions the policy like the robot does.")
    parser.add_argument("--wholebody-rate-limit", type=float, default=0.0,
                        help="If >0, cap the L2 norm of the per-tick "
                             "conditioned-target CHANGE across all 31 joints "
                             "to this many rad/s (whole vector scaled down, "
                             "direction preserved). Whole-body complement to "
                             "the per-joint max_target_dev clamps; 0 = off. "
                             "PROTOTYPE for evaluation — calibrate against "
                             "known-good recoveries before any deploy port.")
    parser.add_argument("--auto-reset", action="store_true",
                        help="On fall (pelvis<0.4), RSI back onto the "
                             "reference at the current frame and continue "
                             "(long robot-parallel rollouts).")
    parser.add_argument("--traj-csv", default=None,
                        help="If set, dump per-control-step trajectory CSV: "
                             "t, robot_{x,y,z,yaw_deg}, ref_{x,y,z,yaw_deg}, "
                             "ref_motion_frame, pelvis_z. Used to debug heading/"
                             "direction drift between IsaacLab and MuJoCo.")
    args = parser.parse_args()

    print(f"Loading actor from {args.checkpoint} ...", flush=True)
    actor = load_actor_from_checkpoint(args.checkpoint, args.device)
    print("  Actor loaded.", flush=True)

    if args.playlist is not None:
        print(f"Loading playlist from {args.playlist} ...", flush=True)
        motion_data = load_playlist_motion_data(args.playlist)
    else:
        print(f"Loading motion from {args.motion} ...", flush=True)
        import joblib  # local to defer
        motion_data = joblib.load(args.motion)
    mk = next(iter(motion_data))
    motion_entry = motion_data[mk]
    total_frames = motion_entry["dof"].shape[0]
    motion_fps = float(motion_entry["fps"])
    print(f"  {mk}: {total_frames} frames @ {motion_fps:.0f} fps "
          f"= {total_frames/motion_fps:.1f}s", flush=True)

    mjcf_path = args.mjcf or MJCF_PATH
    print(f"Loading MuJoCo model from {mjcf_path} ...", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_model.opt.timestep = SIM_DT
    # Bump offscreen framebuffer to match requested render size. The X2 MJCF
    # ships with the MuJoCo default (640x480) which caps the Renderer.
    mj_model.vis.global_.offwidth = max(args.width, int(mj_model.vis.global_.offwidth))
    mj_model.vis.global_.offheight = max(args.height, int(mj_model.vis.global_.offheight))
    mj_data = mujoco.MjData(mj_model)
    pelvis_id = mj_model.body("pelvis").id

    init_state = compute_motion_state(motion_data, int(args.init_frame), motion_fps)
    init_root_z = float(init_state["root_pos_w"][2])
    apply_init_state(mj_model, mj_data, init_state)

    viewer = None
    if args.onscreen:
        renderer = None
        cam = None
        viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, show_left_ui=False, show_right_ui=False)
        viewer.cam.azimuth = args.cam_azimuth
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.distance = args.cam_distance
        viewer.cam.lookat[:] = [0.0, 0.0, init_root_z]
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
    elif args.no_render:
        renderer = None
        cam = None
    else:
        if args.out is None:
            raise SystemExit("--out is required unless --onscreen or --no-render")
        renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
        cam = mujoco.MjvCamera()
        cam.azimuth = args.cam_azimuth
        cam.elevation = args.cam_elevation
        cam.distance = args.cam_distance
        cam.lookat[:] = [0.0, 0.0, init_root_z]
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = pelvis_id

    prop_buf = ProprioceptionBuffer()
    last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
    n_resets = 0
    sim_time = float(args.init_frame) / motion_fps

    # Deploy-twin target conditioning (gains + clamps + LPF). Defaults are
    # the training-equivalent PD with no conditioning — the historical
    # behavior of this script.
    kp_arr, kd_arr = KP, KD
    dev_arr = np.full(NUM_DOFS, np.inf)
    lpf_alpha = np.ones(NUM_DOFS)
    action_clip = np.inf
    if args.tuning_yaml:
        kp_arr, kd_arr, dev_arr, lpf_alpha, action_clip = \
            load_deploy_tuning(args.tuning_yaml)
        print(f"Deploy tuning: {args.tuning_yaml} "
              f"(kp leg {kp_arr[0]:.0f}, kd hip {kd_arr[0]:.1f}, "
              f"dev leg {dev_arr[0]:.2f} rad, "
              f"lpf leg alpha {lpf_alpha[0]:.2f}, clip {action_clip})",
              flush=True)
    target_filt = None  # first tick (and every auto-reset) re-seeds

    # One video frame per control step (50 Hz). If --render-fps differs, we
    # sub-sample control steps to match (simple integer stride).
    control_hz = int(round(1.0 / CONTROL_DT))
    if args.render_fps > control_hz:
        print(f"WARNING: --render-fps {args.render_fps} > control rate "
              f"{control_hz}; clamping.", flush=True)
        args.render_fps = control_hz
    stride = max(1, control_hz // args.render_fps)
    effective_fps = control_hz // stride

    total_steps = int(args.duration * control_hz)
    # Pre-extract motion world-frame xy + yaw for reference logging. The motion
    # was recorded with a non-zero starting world XY (we discard it at RSI by
    # zeroing qpos[0:2]) so we shift by `ref_xy0` to put the ref trajectory in
    # the same robot-spawned frame (origin = motion frame init_frame world XY).
    ref_world_xy = np.asarray(motion_entry["root_trans_offset"], dtype=np.float64)[:, :2]
    ref_xy0 = ref_world_xy[int(args.init_frame)].copy()
    ref_world_quat_xyzw = np.asarray(motion_entry["root_rot"], dtype=np.float64)
    ref_world_yaw = Rot.from_quat(ref_world_quat_xyzw).as_euler("ZYX")[:, 0]
    csv_fp = None
    csv_writer = None
    if args.traj_csv is not None:
        csv_fp = open(args.traj_csv, "w", newline="")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow([
            "step", "t",
            "robot_x", "robot_y", "robot_z", "robot_yaw_deg",
            "ref_x", "ref_y", "ref_z", "ref_yaw_deg",
            "ref_motion_frame", "pelvis_z",
            "robot_lin_vel_x_w", "robot_lin_vel_y_w",
            "tilt_deg", "gyro_norm",
        ] + [f"a_il_{k}" for k in range(NUM_DOFS)]
          + [f"qvel_mj_{k}" for k in range(NUM_DOFS)])
        print(f"Trajectory CSV -> {args.traj_csv}", flush=True)

    if args.onscreen:
        print(f"Rolling out {args.duration:.1f}s ({total_steps} control steps), "
              f"LIVE viewer @ real-time (looping)", flush=True)
        writer = None
    elif args.no_render:
        print(f"Rolling out {args.duration:.1f}s ({total_steps} control steps), "
              f"NO render (fall-time only)", flush=True)
        writer = None
    else:
        print(f"Rolling out {args.duration:.1f}s ({total_steps} control steps), "
              f"writing {args.out} @ {effective_fps} fps "
              f"({total_steps // stride} frames, ~{args.width}x{args.height})",
              flush=True)
        writer = imageio.get_writer(
            args.out, fps=effective_fps, codec="libx264",
            macro_block_size=1, quality=8,
        )

    fall_frame = None
    import time as _wallclock
    _wall0 = _wallclock.time()
    try:
        for step in range(total_steps):
            motion_time = sim_time
            motion_frame = int(motion_time * motion_fps) % total_frames
            motion_time = motion_frame / motion_fps

            qpos_j = mj_data.qpos[7:7 + NUM_DOFS].copy()
            qvel_j = mj_data.qvel[6:6 + NUM_DOFS].copy()
            base_quat = mj_data.qpos[3:7].copy()
            base_angvel = mj_data.qvel[3:6].copy()

            dof_pos_il = qpos_j[IL_TO_MJ_DOF]
            dof_vel_il = qvel_j[IL_TO_MJ_DOF]
            action_il_prev = last_action_mj[IL_TO_MJ_DOF]

            gravity = quat_rotate_inverse(base_quat, np.array([0., 0., -1.]))
            dof_pos_rel_il = dof_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]

            prop_buf.append(gravity, base_angvel, dof_pos_rel_il, dof_vel_il, action_il_prev)
            proprioception = prop_buf.get_flat()
            tokenizer_obs = build_tokenizer_obs(
                motion_data, motion_time, base_quat, motion_fps)

            with torch.no_grad():
                prop_t = torch.from_numpy(proprioception).unsqueeze(0).to(args.device)
                tok_t = torch.from_numpy(tokenizer_obs).unsqueeze(0).to(args.device)
                action_il_t = actor(prop_t, tok_t).squeeze(0).cpu().numpy()

            if np.isfinite(action_clip):
                action_il_t = np.clip(action_il_t, -action_clip, action_clip)
            action_mj = action_il_t[MJ_TO_IL_DOF]
            last_action_mj = action_mj.copy()
            target_pos = DEFAULT_DOF + action_mj * ACTION_SCALE
            # Deploy-side conditioning: clamp the target's deviation from the
            # MEASURED joint position, then first-order low-pass — exactly
            # the two stages the C++ deploy runs before its PD loop.
            qpos_meas = mj_data.qpos[7:7 + NUM_DOFS]
            target_pos = qpos_meas + np.clip(
                target_pos - qpos_meas, -dev_arr, dev_arr)
            if target_filt is None:
                target_filt = target_pos.copy()
            if args.wholebody_rate_limit > 0.0:
                # Whole-body rate budget: scale the aggregate target step
                # (vs the last conditioned target) so its L2 norm never
                # exceeds the budget — direction preserved, coordination
                # intact, magnitude bounded.
                step_vec = target_pos - target_filt
                n = float(np.linalg.norm(step_vec))
                budget = args.wholebody_rate_limit * CONTROL_DT
                if n > budget:
                    target_pos = target_filt + step_vec * (budget / n)
            target_filt = target_filt + lpf_alpha * (target_pos - target_filt)
            target_pos = target_filt

            for _ in range(DECIMATION):
                torque = kp_arr * (target_pos - mj_data.qpos[7:7 + NUM_DOFS]) \
                       - kd_arr * mj_data.qvel[6:6 + NUM_DOFS]
                for j in range(NUM_DOFS):
                    mj_data.ctrl[JOINT_TO_ACTUATOR[j]] = torque[j]
                mujoco.mj_step(mj_model, mj_data)

            sim_time += CONTROL_DT

            pelvis_z = float(mj_data.qpos[2])
            if pelvis_z < 0.40:
                if fall_frame is None or args.auto_reset:
                    print(f"  [fall] at step {step}, t={step*CONTROL_DT:.2f}s "
                          f"(pelvis_z={pelvis_z:.2f})", flush=True)
                if fall_frame is None:
                    fall_frame = step
                if args.auto_reset:
                    # RSI back onto the reference at the CURRENT motion frame,
                    # like eval's auto-reset: rollout continues robot-parallel.
                    rs = compute_motion_state(motion_data, motion_frame,
                                              motion_fps)
                    apply_init_state(mj_model, mj_data, rs)
                    prop_buf = ProprioceptionBuffer()
                    last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
                    target_filt = None
                    n_resets += 1

            if csv_writer is not None:
                # Robot world-frame state (qpos[0:2] starts at 0 by RSI).
                rx, ry, rz = (float(mj_data.qpos[0]), float(mj_data.qpos[1]),
                              float(mj_data.qpos[2]))
                bq_wxyz = mj_data.qpos[3:7]
                bq_xyzw = [bq_wxyz[1], bq_wxyz[2], bq_wxyz[3], bq_wxyz[0]]
                robot_yaw = Rot.from_quat(bq_xyzw).as_euler("ZYX")[0]
                # Motion-side reference (in robot-spawn frame, i.e. shifted by
                # ref_xy0 so frame=init_frame lands at xy=(0,0) just like the
                # robot does). Note: we don't rotate the motion XY into the
                # robot's spawn yaw — we keep both in the original motion-world
                # yaw frame so drift in robot yaw is visible directly.
                ref_xy_now = ref_world_xy[motion_frame] - ref_xy0
                ref_z = float(motion_entry["root_trans_offset"][motion_frame, 2])
                ref_yaw = float(ref_world_yaw[motion_frame])
                csv_writer.writerow([
                    step, f"{step*CONTROL_DT:.4f}",
                    f"{rx:.5f}", f"{ry:.5f}", f"{rz:.5f}",
                    f"{np.degrees(robot_yaw):.3f}",
                    f"{ref_xy_now[0]:.5f}", f"{ref_xy_now[1]:.5f}",
                    f"{ref_z:.5f}", f"{np.degrees(ref_yaw):.3f}",
                    motion_frame, f"{rz:.5f}",
                    f"{float(mj_data.qvel[0]):.5f}",
                    f"{float(mj_data.qvel[1]):.5f}",
                    f"{np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))):.3f}",
                    f"{float(np.linalg.norm(base_angvel)):.4f}",
                ] + [f"{v:.5f}" for v in action_il_t]
                  + [f"{v:.5f}" for v in mj_data.qvel[6:6 + NUM_DOFS]])

            if writer is not None and step % stride == 0:
                renderer.update_scene(mj_data, camera=cam)
                frame = renderer.render()
                writer.append_data(frame)

            if args.onscreen:
                if not viewer.is_running():
                    break
                viewer.sync()
                _target = _wall0 + (step + 1) * CONTROL_DT
                _dt = _target - _wallclock.time()
                if _dt > 0:
                    _wallclock.sleep(_dt)

            if step % 50 == 0:
                print(f"  step {step}/{total_steps}  t={step*CONTROL_DT:.2f}s  "
                      f"pelvis_z={float(mj_data.qpos[2]):.3f}", flush=True)

        if args.onscreen and viewer is not None:
            print("clip done -- holding viewer open; close the window to exit",
                  flush=True)
            while viewer.is_running():
                viewer.sync()
                _wallclock.sleep(0.03)
    finally:
        if viewer is not None:
            viewer.close()
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if csv_fp is not None:
            csv_fp.close()

    tag = "survived" if fall_frame is None else f"fell @ {fall_frame*CONTROL_DT:.2f}s"
    out_msg = f". Wrote {args.out}" if writer is not None else ""
    print(f"\nDone. {tag}{out_msg}", flush=True)


if __name__ == "__main__":
    main()
