# Capturing planner/teleop motion → motion-lib PKL (with world coordinates)

`record_motion_to_pkl.py` captures a **live sim** motion into a training-format
motion PKL — including the **real world trajectory** (XY translation + heading),
which the LeRobot-episode route (`lerobot_episode_to_motion_pkl.py`) cannot
recover because episodes only store joint commands, not the pelvis world pose.

## Where world coordinates come from

The sim bridge publishes the ground-truth pelvis pose on a ZMQ PUB
(topic `robot_pose`, default port **5570**): `pelvis_qpos_wxyz = [x, y, z, qw, qx,
qy, qz]` (the MuJoCo free-joint qpos). The deploy's `<robot>_debug` stream
(port **5557**) carries IMU orientation + joints but **not** world XY — that is
why the separate `robot_pose` stream is required.

| Robot | `robot_pose` PUB source | Enabled by |
|---|---|---|
| **X2** | `gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py` | on by default (`--robot-pose-pub-port 5570`) |
| **G1** | `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` (added) | **opt-in**: `GEAR_SONIC_ROBOT_POSE_ZMQ_PORT=5570` (or config `ROBOT_POSE_ZMQ_PORT`) |

The output PKL: `root_trans_offset` ← world xyz, `root_rot` ← world quat
(wxyz→xyzw), `dof` ← `body_q_measured` (MuJoCo order), plus `pose_aa`,
`smpl_joints`, `fps` — identical schema to `convert_soma_csv_to_motion_lib.py`.

## G1 — capture a locomotion mode from the STOCK stack

The G1 planner is **stock Unitree code** (Phase-1 stack: `run_sim_loop.py` +
`deploy.sh sim` + `quest3_manager_thread_server.py`). It selects named
locomotion modes **at runtime on the controller** (gamepad / Quest3), not via a
launch flag — the flag form (`--planner-mode`) is X2-only. From the deploy's
`gamepad.hpp` (`planner_use_movement_mode` / `LocomotionMode`):

- **L1 / R1** cycle the mode: **0=idle, 1=slow walk, 2=walk, 3=run, 4=boxing**, …
- **L2 / R2** adjust speed / height.

So you capture a named style by selecting it live, then recording. No stack
changes needed beyond enabling the pose PUB.

1. **Enable the G1 pose PUB** (in the terminal that launches the G1 sim). This is
   the ONLY change vs. the stock Phase-1 stack:

   ```bash
   export GEAR_SONIC_ROBOT_POSE_ZMQ_PORT=5570
   ```

2. **Launch the stock G1 stack**, then on the controller select the mode you want
   (L1/R1 → e.g. slow walk / walk / run) and drive the robot.

3. **Run the sidecar** (separate terminal; start it while the robot is walking):

   ```bash
   .venv/bin/python gear_sonic/scripts/record_motion_to_pkl.py --robot g1 \
       --duration 20 \
       --out gear_sonic/data/motions/g1_recorded/slow_walk_001.pkl \
       --motion-key slow_walk_001
   ```

   Ctrl-C stops early and still writes what was captured. It prints the frame
   count and **meters traveled in world XY** — a nonzero value confirms world
   coordinates were captured.

## X2 — same, no env var needed

The X2 bridge already publishes `robot_pose`. Unlike G1, the X2 planner is
custom (`x2_kplanner.py`) and exposes a discrete style selector
`--planner-mode {idle | slow_walk | walk | run_proxy}`, so you can capture each
named style directly. Then:

```bash
.venv/bin/python gear_sonic/scripts/record_motion_to_pkl.py --robot x2 \
    --duration 20 \
    --out gear_sonic/data/motions/x2_recorded/walk_001.pkl --motion-key walk_001
```

## Play / verify a captured clip

```bash
python -m gear_sonic.scripts.play_gesture --pkl <out.pkl>     # in-stack playback
# or preview PKL→CSV via the Phase-2 shim + visualize_motion.py
```

## Notes

- Sampling: the sidecar subscribes to both PUBs, timestamps on receive, and
  resamples both to `--fps` (default 50) on a common grid (linear interp for
  translation/joints, slerp for rotation). Async publish rates are fine.
- `body_q_measured` (what the robot did) is preferred; falls back to `body_q`.
- Hands are not captured (the debug body stream is 29/31-DoF body only), matching
  the existing recorded-gesture PKLs.
- Verified offline end-to-end (synthetic `robot_pose` + `<robot>_debug` PUBs →
  correct world translation + schema). Live capture needs the sim stack running.
