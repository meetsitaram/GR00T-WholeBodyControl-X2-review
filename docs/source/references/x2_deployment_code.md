# X2 Ultra C++ Deploy Program Flow

This document describes the AgiBot X2 Ultra deployment binary, its arguments,
and how its components map to the training pipeline. It complements
[deployment_code.md](deployment_code.md) (which covers the legacy G1 deploy)
by focusing on what is *different* for the 31-DOF X2 Ultra over ROS 2 / AimDK.

The package lives at
`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/` and was added in Phase 3
of [`x2-ultra-onnx-deploy_9dde7da2.plan.md`](.cursor/plans/x2-ultra-onnx-deploy_9dde7da2.plan.md).

For an end-to-end "first time on the real robot" walkthrough, see
[`x2_first_real_robot.md`](../user_guide/x2_first_real_robot.md).

## Architecture overview

```
[AimDK joint state x4 + IMU] ─► RobotState (MJ-ordered, mutex-guarded)
                                     │
                          50 Hz control timer
                                     ▼
       IL-remap ► ProprioceptionBuffer.Append (990 D, oldest-first)
                                     │
                BuildTokenizerObs (680 D, ONNX grouped layout)
                                     │
                  OnnxActor.Infer ─► action_il[31]
                                     │
            scale & MJ-remap ► target_pos_mj[31]
                                     │
                       ApplySafetyStack
                       ├─ tilt watchdog
                       ├─ soft-start ramp
                       └─ dry-run gain mute
                                     ▼
                              SafeCommand (latched)
                                     │
                          500 Hz writer timer
                                     ▼
[JointCommandArray /aima/hal/joint/{leg,waist,arm,head}/command]
```

### Key differences from the G1 deploy

| Aspect | G1 deploy | X2 deploy |
| ------ | --------- | --------- |
| Robot SDK | Unitree DDS (raw `LowState` / `LowCmd`) | ROS 2 + `aimdk_msgs` (`JointStateArray` / `JointCommandArray`) |
| Inference engine | TensorRT (FP16 optional) | ONNX Runtime CPU (fused encoder+decoder) |
| Joint count | 29 | 31 |
| Topic fan-out | 2 channels (low_state, low_cmd) | 4 joint groups (leg, waist, arm, head) + 1 IMU |
| Reference motion | MotionDataReader (PKL stream) | StandStillReference / PklMotionReference (X2M2 file) |
| Threading | DDS recurrent threads (4) | ROS 2 MultiThreadedExecutor + 2 wall timers |
| Logger | G1 StateLogger (Dex3 hands, encoder mode, etc.) | X2 DeployLogger (5 CSVs, no hands) |
| Safety scaffolding | bolted on per-launcher | First-class (Phase 4): `--dry-run`, soft-start, tilt watchdog |

### ROS 2 topic contract

Cross-referenced against:

- `agibot-x2-references/lx2501_3-v0.9.0.4/topics_and_services` (the
  authoritative topic registry shipped with the SDK).
- `dev/Interface/control_mod/joint_control.html` (joint command/state
  schema + per-group ordering).
- `dev/Interface/hal/sensor.html` (IMU topic table).

| Direction | Topic | Type | QoS hint | Rate |
| --------- | ----- | ---- | -------- | ---- |
| sub | `/aima/hal/joint/leg/state` | `aimdk_msgs/JointStateArray` | TRANSIENT_LOCAL | ~200 Hz |
| sub | `/aima/hal/joint/waist/state` | `aimdk_msgs/JointStateArray` | TRANSIENT_LOCAL | ~200 Hz |
| sub | `/aima/hal/joint/arm/state` | `aimdk_msgs/JointStateArray` | TRANSIENT_LOCAL | ~200 Hz |
| sub | `/aima/hal/joint/head/state` | `aimdk_msgs/JointStateArray` | TRANSIENT_LOCAL | ~200 Hz |
| sub | `/aima/hal/imu/torso/state` | `sensor_msgs/Imu` | TRANSIENT_LOCAL | 500 Hz |
| pub | `/aima/hal/joint/leg/command` | `aimdk_msgs/JointCommandArray` | (matches sub) | 500 Hz |
| pub | `/aima/hal/joint/waist/command` | `aimdk_msgs/JointCommandArray` | (matches sub) | 500 Hz |
| pub | `/aima/hal/joint/arm/command` | `aimdk_msgs/JointCommandArray` | (matches sub) | 500 Hz |
| pub | `/aima/hal/joint/head/command` | `aimdk_msgs/JointCommandArray` | (matches sub) | 500 Hz |

`AimdkIo` declares all subs/pubs with `rclcpp::SensorDataQoS()`
(BEST_EFFORT + VOLATILE, depth=10). This matches the reference
`echo_imu_data.cpp` example in the SDK and is QoS-compatible with the
firmware's `TRANSIENT_LOCAL` publishers (subscriber gets live data only,
no late-join replay).

> **IMU spelling caveat.** The official `topics_and_services` registry,
> `sensor.html`, and the `agibot-x2-monitor` bridge config use
> `/aima/hal/imu/torso/state`. The SDK example sources
> (`echo_imu_data.cpp` / `.py`) carry the typo `torse` in their comments.
> The deploy defaults to `torso` and exposes `--imu-topic` so an operator
> can override at runtime if a particular firmware build actually
> publishes to `torse`.

> **Topics the deploy intentionally does *not* touch.**
> `/aima/mc/locomotion/velocity` is the high-level walking-velocity
> channel arbitrated by AimDK's MC layer. The whole-body policy *is* the
> locomotion controller, so we bypass MC arbitration entirely and stream
> joint commands directly to HAL. Before launching the deploy, the MC
> module must be either stopped (`aima em stop-app mc`, the FAQ-prescribed
> recipe) or switched to `JOINT_DEFAULT` mode — see
> [x2_first_real_robot.md](../user_guide/x2_first_real_robot.md) for the
> exact pre-flight steps.
>
> Hand joints live on the separate `/aima/hal/joint/hand/{state,command}`
> topics and are not part of the 31-DOF policy contract; this deploy
> ignores them.

### Where the deploy can run

Both deployment topologies in `dev/quick_start/prerequisites.html` are
supported with no code changes:

- **On-bot (PC2, `10.0.1.41`)**: zero ethernet jitter, recommended for
  production. Build with `colcon build --packages-select agi_x2_deploy_onnx_ref`
  in the on-bot workspace and run from there.
- **Laptop over wired ethernet (`10.0.1.2/24` ↔ rear SDK port)**: cleanest
  for first bring-up and debugging. Bandwidth budget for our 5 subs + 4
  pubs is ~1.9 MB/s, well inside gigabit. Adds ~0.1-0.5 ms typical jitter.

**Strict prohibitions** carried over from `dev/end_notes.html`:

- Never run on PC1 (`10.0.1.40`, the motion-control unit).
- Never use Wi-Fi for the control loop. Wi-Fi is "for SSH debugging only."
- Cameras shouldn't be subscribed cross-host (90 MB/s raw). N/A for this
  deploy, but worth knowing if you bolt on perception later.

### Joint contract

The deploy uses **two joint orderings** internally and remaps explicitly via
the codegen'd tables in `policy_parameters.hpp`:

- **MuJoCo (MJ) order** — the kinematic-tree BFS order used by the URDF /
  MJCF and by AimDK's per-group topic ordering. Lookup table:
  `mujoco_joint_names[31]`. The four AimDK joint groups slice this 31-D
  vector contiguously: `[0..12) = leg`, `[12..15) = waist`,
  `[15..29) = arm`, `[29..31) = head`. AimdkIo cross-checks this on the
  inaugural state callback and aborts hard on any name mismatch.

- **IsaacLab (IL) order** — the order the policy's encoder and decoder
  consume internally. Permutation tables: `isaaclab_to_mujoco[31]` and
  `mujoco_to_isaaclab[31]`.

Per-group ordering verified against `joint_control.html`:

| Group | Length | Order | Notes |
| ----- | ------ | ----- | ----- |
| leg | 12 | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll | left first, then right |
| waist | 3 | yaw, pitch, roll | doc table has `wrist_*` typos; actual joint names are `waist_*_joint` per MJCF |
| arm | 14 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_pitch, wrist_roll | left first, then right |
| head | 2 | yaw, pitch | **only `head_yaw_joint` is physically actuated** — current firmware silently drops `head_pitch_joint` commands |

The 31-D shape is preserved end-to-end (so the policy doesn't need to be
retrained when head pitch eventually ships); inert head_pitch values are
just absorbed by the firmware.

Observation construction (proprioception, tokenizer, action output) is
exclusively in IL order. Hardware IO (state subs, command pubs) is
exclusively in MJ order. Conversion happens in exactly two places:

1. **Obs side** (50 Hz control tick): `dof_pos_il[il] = qpos_mj[isaaclab_to_mujoco[il]]`
2. **Action side** (50 Hz control tick): `target_mj[mj] = default_mj[mj] + action_il[mujoco_to_isaaclab[mj]] * action_scale_mj[mj]`

### PD gains and the implicit-PD assumption

The trained policy is calibrated for IsaacLab's *implicit* PD model with
specific kp/kd values per joint. Those exact values are codegen'd into
`kps[]` and `kds[]` of `policy_parameters.hpp` and shipped in every
`JointCommand.stiffness` / `JointCommand.damping` field. The X2 firmware
then applies them on its own 1 kHz inner loop; the policy never directly
torques the joints.

> The `ankle KP × 1.5` MuJoCo deployment-side scaling, which helped during
> sim2sim experiments, is **not** baked into the deploy — that is a
> MuJoCo-only artifact. See the FAQ at the bottom for why this is correct.

## CLI

```bash
ros2 run agi_x2_deploy_onnx_ref x2_deploy_onnx_ref \
    --model PATH \
    [--motion PATH] \
    [--autostart-after SECONDS] \
    [--dry-run] \
    [--tilt-cos COS] \
    [--ramp-seconds SEC] \
    [--log-dir PATH] \
    [--intra-op-threads N]
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--model PATH` | (required) | Fused `g1+g1_dyn` ONNX exported from training (`model_step_NNNNNN_g1.onnx`). Must accept input `[1, 1670]` and emit `[1, 31]`. |
| `--motion PATH` | StandStill | Optional X2M2 reference motion. With `StandStill` the tokenizer always sees the trained default standing pose. |
| `--autostart-after SEC` | -1 (operator) | Auto-transition WAIT_FOR_CONTROL → CONTROL after N seconds. With `<0`, type `go` on stdin instead. |
| `--dry-run` | off | Publish stiffness=0/damping=0 — full pipeline runs, no torque. **Mandatory for first-power-on.** |
| `--tilt-cos COS` | -0.3 | Tilt-watchdog threshold on body-frame gravity[2]. Same convention as `eval_x2_mujoco --fall-tilt-cos`. |
| `--ramp-seconds SEC` | 2.0 | Blend duration from default-pose to policy-output on entry to CONTROL. |
| `--log-dir PATH` | (off) | Per-tick CSVs (one per channel) written here. |
| `--intra-op-threads N` | 1 | ONNX session threads. |
| `--imu-topic NAME` | `/aima/hal/imu/torso/state` | Override IMU topic (e.g. `/aima/hal/imu/torse/state` on firmware shipped with the SDK-example typo). |

## State machine

```
INIT ─(all 5 sources fresh)─► WAIT_FOR_CONTROL ─(go)─► CONTROL
                                                        │
                                              (tilt watchdog trips)
                                                        ▼
                                                    SAFE_HOLD
```

| State | Publishing? | Exit condition |
| ----- | ----------- | -------------- |
| INIT | no | `AimdkIo::AllStateFresh(0.5)` reports all 5 sources within 500 ms |
| WAIT_FOR_CONTROL | no | `--autostart-after` elapses, or operator `RequestGo()` (stdin `go`) |
| CONTROL | yes (50/500 Hz) | Tilt watchdog trip, ONNX failure, or process kill |
| SAFE_HOLD | yes (latched) | Process restart only |

## Observation construction

See [`obs_config_x2_ultra.yaml`](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/config/obs_config_x2_ultra.yaml)
for the spec. Total input width: `1670 = 680 (tokenizer) + 990 (proprioception)`.

### Proprioception (990 D)

`HISTORY_LEN = 10` frames, oldest-first per term. Term order **must** match
IsaacLab's `PolicyCfg` dataclass attribute order (not YAML order):

```
ang_vel(3 ) × 10 = 30
jpos_rel(31) × 10 = 310    (joint_pos_il - default_angles[il])
jvel(31) × 10 = 310
last_action(31) × 10 = 310
gravity(3 ) × 10 = 30
─────────────────────────
                    990
```

The C++ `ProprioceptionBuffer` mirrors `IsaacLab CircularBuffer`: on the
first `Append()` after `Reset()`, **all 10 slots are broadcast-filled with
the first sample**. Without this priming, frames 0..8 would be zero on the
first inference step, which is OOD for any policy trained with history.

### Tokenizer (680 D, ONNX grouped layout)

```
command_flat(620) || ori_flat(60)
```

where

- `command_flat = [ jpos_il_f1(31) | jvel_il_f1(31) | ... | jpos_il_f10(31) | jvel_il_f10(31) ]`
- `ori_flat     = [ rot6d_f1(6)  | ... | rot6d_f10(6) ]`

Each future frame is sampled from the active `ReferenceMotion` at
`t_future_k = current_time + (k+1) * 0.1 s`. The 6-D rotation per frame is
the first two rows of `(cur_root_quat^-1 * future_root_quat).as_matrix()`,
both quaternions in xyzw scipy convention.

> The Python eval (`eval_x2_mujoco_onnx.py`) also accepts the *interleaved*
> PT-style layout and converts it via `_interleaved_to_grouped`. The C++
> deploy skips that intermediate entirely and writes the grouped layout
> directly. Feeding the interleaved layout to the ONNX session diverges by
> O(1) immediately (verified empirically during Phase 0).

## Safety stack

Three layers, applied in order on every 50 Hz tick by `ApplySafetyStack()`:

1. **TiltWatchdog** — checks `gravity_body[2] > --tilt-cos`. On trip,
   latches "hold default angles, 4× damping" (no policy commands ever
   again until process restart).
2. **SoftStartRamp** — for the first `--ramp-seconds` of CONTROL state,
   blends `target = (1 - α) * default_angles + α * policy_target` with
   `α = clamp((now - control_entry) / ramp_seconds, 0, 1)`.
3. **Dry-run** — if `--dry-run`, all `stiffness` / `damping` fields in the
   published commands are zeroed at the very last step (after target
   computation), so the full pipeline is exercised but no torque is applied.

## Logging

When `--log-dir PATH` is set, the deploy writes 5 CSV files into PATH:

| File | Schema |
| ---- | ------ |
| `tick.csv` | `t,ramp_alpha,dry_run,tilt_trip,reason` |
| `target_pos.csv` | `t,target_<joint_name>...` (31 joints, MJ order) |
| `joint_pos.csv` | `t,q_<joint_name>...` |
| `joint_vel.csv` | `t,dq_<joint_name>...` |
| `action_il.csv` | `t,a_il_0..30` (IL order; remap via `mujoco_to_isaaclab` to align) |
| `imu.csv` | `t,qw,qx,qy,qz,wx,wy,wz` |

Logger is mutex-guarded, fixed-precision, ~6 MB per hour at 50 Hz.

## Build

```bash
# Inside a colcon workspace that already has aimdk_msgs available:
colcon build --packages-select agi_x2_deploy_onnx_ref \
    --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
```

For an offline syntax check (no ROS 2 / no ONNX Runtime install):

```bash
cd gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref
cmake -S . -B build -DAGI_X2_OFFLINE_SYNTAX_CHECK=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The offline build compiles only the obs builders + `test_obs_builder` (a
self-contained gtest-style runner that covers quat math, proprioception
priming/aging, and tokenizer sizing/layout).

## Re-exporting the deploy ONNX from a checkpoint

The deploy node consumes a single fused ONNX
(`gear_sonic_deploy/models/x2_sonic_16k.onnx`) that maps a flat 1670-D obs
vector to a 31-D joint-position target. The fused graph contains the G1
encoder, the FSQ quantizer, and the `g1_dyn` decoder all in one file; its
input layout is the same per-frame interleaved view that
[`dump_isaaclab_step0.py`](../../../gear_sonic/scripts/dump_isaaclab_step0.py)
captures from training.

When you train a new policy (or want to re-export an older one), use the
helper at [`gear_sonic_deploy/scripts/reexport_x2_onnx.sh`](../../../gear_sonic_deploy/scripts/reexport_x2_onnx.sh):

```bash
conda activate env_isaaclab   # the helper will activate it for you if missing
cd <repo root>

./gear_sonic_deploy/scripts/reexport_x2_onnx.sh \
    $HOME/x2_cloud_checkpoints/run-20260420_083925
```

That single command runs both phases and writes the validated ONNX into the
deploy slot:

| Phase | Inner script | What it does |
| ----- | ------------ | ------------ |
| 1 | [`gear_sonic.scripts.dump_isaaclab_step0`](../../../gear_sonic/scripts/dump_isaaclab_step0.py) | Spins IsaacLab once with the chosen `last.pt`, captures the encoder input + decoder action mean for step 0 to `${DUMP_PATH}` (default `/tmp/x2_step0_isaaclab_lastpt.pt`). |
| 2 | [`gear_sonic.scripts.reexport_x2_g1_onnx`](../../../gear_sonic/scripts/reexport_x2_g1_onnx.py) | Loads the same checkpoint, wraps `actor_module` in `FusedG1Wrapper` (mirrors training's forward exactly), exports with `do_constant_folding=False` so FSQ rounding survives, and validates the new ONNX against the dump. Refuses to promote unless `max\|onnx − pt\| < ${MAX_DIFF}` (default `1e-3` rad). |

The previous ONNX is preserved at
`gear_sonic_deploy/models/x2_sonic_16k.onnx.broken_export.<timestamp>` before
the new one replaces it, so we never silently break a working deploy.

Useful environment variables:

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `CONDA_ENV` | `env_isaaclab` | Conda env to activate (must have IsaacLab + onnxruntime). |
| `DUMP_PATH` | `/tmp/x2_step0_isaaclab_lastpt.pt` | Where to write phase-1 dump. |
| `MAX_DIFF` | `1e-3` | Refuse to promote if `max\|onnx − pt\|` exceeds this (radians). |
| `SKIP_DUMP=1` | unset | Reuse an existing dump (only safe if it was made from this exact checkpoint). |

To export a *specific* checkpoint instead of `last.pt`, append a Hydra
override after the output path:

```bash
./gear_sonic_deploy/scripts/reexport_x2_onnx.sh \
    $HOME/x2_cloud_checkpoints/run-20260420_083925 \
    /tmp/x2_sonic_16k_test.onnx \
    +checkpoint=$HOME/x2_cloud_checkpoints/run-20260420_083925/model_step_014000.pt
```

### Why this exists

The first deploy ONNX (shipped Apr 21) was produced by
`inference_helpers.export_universal_token_module_as_onnx`, whose wrapper
laid out the encoder input as `[cmd_block(620) | ori_block(60)]` while the
trained encoder consumes the per-frame interleaved view
`[cmd_f0(62) | ori_f0(6) | cmd_f1(62) | ori_f1(6) | ...]`. The resulting
ONNX diverged from the PyTorch policy by up to **9.77 rad per joint** in a
side-by-side IsaacSim comparison (see `logs/x2/onnx_vs_pt.csv`). The
re-export helper above reduces that divergence to `~1e-6 rad` at scale,
because `FusedG1Wrapper` shares its layout with the dump path that we
already trust as ground truth.

## FAQ

**Q: Should the deploy use the MuJoCo PD scaling that fixed our sim2sim
runs (e.g. ankle kp × 1.5)?**

No. Those scalings are MuJoCo-specific compensations for the explicit-PD
mismatch between MuJoCo and IsaacLab's implicit-PD model. The X2 firmware
runs an implicit-PD inner loop that is much closer to IsaacLab than to
MuJoCo, so the deploy ships the trained gains unmodified. If real-robot
behavior shows a similar mismatch, you would tune `kps[]` / `kds[]` at the
codegen step (and regenerate the header), not at runtime.

**Q: Why is the writer 500 Hz when the policy only runs at 50 Hz?**

The policy target only changes at 50 Hz, but the firmware's joint inner
loop wants commands at its own rate. Republishing the same SafeCommand at
500 Hz keeps the firmware's command-age watchdog satisfied and lets it
treat the target as a continuous setpoint instead of a step function.

**Q: Where is Ruckig?**

The X2 motocontrol example uses Ruckig to plan smooth waypoint-to-waypoint
trajectories. The deploy doesn't need it: the policy already produces a
new target every 20 ms, well within the joints' tracking bandwidth. We may
add an optional Ruckig smoothing layer later if the robot exhibits
high-frequency jitter on the trained targets.

**Q: How do I move from `--dry-run` to torque?**

Verify CSV logs from a dry-run session look reasonable
(`target_pos` close to `q`, `tilt_trip` never set, no warnings in the
node log), then drop `--dry-run` and run with the operator gate (no
`--autostart-after`). See [`x2_first_real_robot.md`](../user_guide/x2_first_real_robot.md)
for the full bring-up checklist.
