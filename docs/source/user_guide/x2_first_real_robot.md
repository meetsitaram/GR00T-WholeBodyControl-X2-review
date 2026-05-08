# First Real-Robot Bring-Up: AgiBot X2 Ultra

This is the operator-facing checklist for taking a freshly-trained X2 Ultra
policy from the simulator to the actual hardware for the first time. It
assumes you have already completed:

- A training run that converged in IsaacLab (e.g. the cloud
  `run-20260420_083925` 16k checkpoint).
- A successful Phase 0 ONNX export (`model_step_NNNNNN_g1.onnx`) with
  `eval_x2_mujoco_onnx.py` parity ≤ 1e-4.
- A regenerated `policy_parameters.hpp` (Phase 1) and a green build of the
  `agi_x2_deploy_onnx_ref` ROS 2 package (Phase 3).

For the architecture / CLI reference, see
[`x2_deployment_code.md`](../references/x2_deployment_code.md). For the
underlying deployment plan, see
[`x2-ultra-onnx-deploy_9dde7da2.plan.md`](../../../.cursor/plans/x2-ultra-onnx-deploy_9dde7da2.plan.md).

## Pre-flight checklist (before powering motors)

### Hardware

- [ ] Robot is on a gantry / hoisted by an overhead crane with the body
      vertical and feet ~1 cm above the floor (so a fall doesn't damage
      anything).
- [ ] All joint covers and end-effectors are installed.
- [ ] E-stop is in operator's hand and tested (press, release, confirm
      motors disengage).
- [ ] Robot is on its own dedicated network switch — no
      multi-master DDS surprises from the lab.

### Software prerequisites

- [ ] Latest X2 firmware is flashed and the AimDK middleware is running
      (`ros2 topic list | grep aima/hal` shows joint group + IMU topics).
- [ ] ONNX Runtime ≥ 1.16 installed at `/opt/onnxruntime`. See the
      [install snippet](x2_sonic_deploy_real.md#install-onnx-runtime-one-off-both-topologies)
      in the runbook if it isn't there yet.
- [ ] `aimdk_msgs` built into the colcon workspace and sourced.
- [ ] `agi_x2_deploy_onnx_ref` built clean against this firmware. The
      easiest way is `./gear_sonic_deploy/deploy_x2.sh local --model … --build-only`
      (or `onbot` to build on PC2).
- [ ] The exported ONNX (`model_step_NNNNNN_g1.onnx`) and (optional) X2M2
      reference motion are in known paths on the on-bot computer.

> **Host OS note.** All `ros2 run` / `colcon build` snippets below assume
> Ubuntu 22.04 + ROS 2 Humble. If your laptop is on Ubuntu 24.04 you have
> two choices: (a) **Topology B (on-bot)** — PC2 / Orin NX is already on
> 22.04 + Humble, do everything there over ssh; (b) the
> `gear_sonic_deploy/docker_x2/` container, which wraps a 22.04 + Humble
> + ONNX Runtime image. Use `./docker_x2/enter_sim.sh` for sim DDS or
> `./docker_x2/get_x2_sonic_ready.sh` for real-robot DDS (the latter layers
> `docker-compose.real.yml` to set `ROS_DOMAIN_ID=0`, drop loopback
> isolation, and pin DDS to `enp10s0`). The real-robot DDS path is modeled
> on the working setup in `agitbot-x2-record-and-replay` and is the
> intended laptop-side bring-up flow now. See the
> [Environment](x2_sonic_deploy_real.md#environment) note in the runbook.

### Sanity ros2 topic checks

With AimDK running but no policy yet, verify all five sources the deploy
needs are alive (cross-referenced to the SDK's `topics_and_services`
registry and `dev/Interface/control_mod/joint_control.html`):

```bash
ros2 topic list | grep -E '/aima/hal/(joint|imu)'
# Expect:
#   /aima/hal/joint/leg/state     /aima/hal/joint/leg/command
#   /aima/hal/joint/waist/state   /aima/hal/joint/waist/command
#   /aima/hal/joint/arm/state     /aima/hal/joint/arm/command
#   /aima/hal/joint/head/state    /aima/hal/joint/head/command
#   /aima/hal/imu/torso/state     /aima/hal/imu/chest/state

ros2 topic hz /aima/hal/joint/leg/state
# Expect: ~200 Hz (firmware default; see joint_control.html)

ros2 topic hz /aima/hal/imu/torso/state
# Expect: ~500 Hz (per sensor.html)

ros2 topic echo /aima/hal/joint/leg/state --once | head -40
# Expect: 12 joints, names = left_hip_pitch_joint .. right_ankle_roll_joint
```

If `/aima/hal/imu/torso/state` is silent but `/aima/hal/imu/torse/state`
(typo) is publishing, you're on a firmware that shipped with the
SDK-example name. Re-run the deploy with
`--imu-topic /aima/hal/imu/torse/state`.

If any joint topic is missing or the joint names don't match, **do not
proceed** — `AimdkIo::ValidateJointNames` will hard-abort on the first
state callback, which is exactly what we want, but it's nicer to catch it
here. Verified topic / joint name conventions:

| Group | Topic | Joint names (MJ order) |
| ----- | ----- | ---------------------- |
| leg | `/aima/hal/joint/leg/{state,command}` | 12: `left_hip_pitch..left_ankle_roll, right_hip_pitch..right_ankle_roll` |
| waist | `/aima/hal/joint/waist/{state,command}` | 3: `waist_yaw, waist_pitch, waist_roll` |
| arm | `/aima/hal/joint/arm/{state,command}` | 14: `left_shoulder_pitch..left_wrist_roll, right_shoulder_pitch..right_wrist_roll` |
| head | `/aima/hal/joint/head/{state,command}` | 2: `head_yaw, head_pitch` (only yaw is actuated on current firmware) |

### Where to run the deploy: on-bot vs laptop-over-ethernet

Both topologies are officially supported (`dev/quick_start/prerequisites.html`):

| Topology | Pros | Cons | When to use |
| -------- | ---- | ---- | ----------- |
| **On-bot** (PC2, Jetson Orin NX, `10.0.1.41`) | No network in the loop. Lowest latency. Survives cable yanks. | Slower iteration, heavier ssh-based debugging. | Production, extended runs. |
| **Laptop over wired ethernet** (host PC, `10.0.1.2/24`) | Fast iteration. Easy to attach a debugger / kill instantly. Don't have to deploy a binary into the robot. | Adds ~0.1-0.5 ms ethernet jitter. Loses control if cable yanks. | First bring-up, debugging. |

Network setup for the laptop topology (per
`dev/about_agibot_X2/SDK_interface.html` + `dev/quick_start/prerequisites.html`):

1. Direct cable from your laptop to one of the two **rear RJ45 SDK
   development ethernet ports** (gigabit, wired to PC2/PC3).
2. Configure the laptop NIC as static IP `10.0.1.2/24`.
3. From the laptop, `ros2 topic list | grep /aima/hal` should show every
   robot topic via DDS auto-discovery.

**Strict prohibitions** (from `dev/end_notes.html`):

- **Never run our deploy on PC1** (the motion-control unit, `10.0.1.40`).
  AgiBot considers this "strictly prohibited to avoid safety risks" because
  PC1 hosts the high-real-time inner motion loop. PC2 (`10.0.1.41`) is the
  designated dev unit.
- **Never use Wi-Fi for control.** Per `prerequisites.html`: "Wi-Fi should
  only be used for SSH debugging." The 50/500 Hz control loop will not
  tolerate wifi jitter.

Cross-host bandwidth budget (well inside gigabit headroom for our usage):

| Topic | Direction (laptop side) | Rate × payload | Bandwidth |
| ----- | ----------------------- | -------------- | --------- |
| `/aima/hal/joint/{leg,waist,arm,head}/state` | sub | 200 Hz × ~600 B | ~120 KB/s |
| `/aima/hal/imu/torso/state` | sub | 500 Hz × 64 B | ~32 KB/s |
| `/aima/hal/joint/{leg,waist,arm,head}/command` | pub | 500 Hz × 31 × ~50 B | ~1.7 MB/s |

Two cross-host caveats that *don't* affect this deploy but are good to
know for future work:

- Cameras don't work cross-host. `sensor.html` says: "Raw image bandwidth
  is about 90 MB/s — use only on the same compute unit, do not subscribe
  across units." If you ever add vision, run on PC2 or use the
  `/compressed` topics.
- ROS 2 services are flaky cross-host. Repeated warning in
  `joint_control.html` / `modeswitch.html` / `MC_control.html`: "standard
  ROS DO NOT handle cross-host service (request-response) well." Our hot
  path is all topics, so this only affects the operator's optional
  `SetMcAction` call (use the retry pattern from `set_mc_action.cpp`:
  250 ms timeout × 8 retries) and the `aima em` admin commands (which we
  do via ssh anyway).

### Disable MC so HAL joint commands take effect

Joint commands published to `/aima/hal/joint/*/command` are **only honoured
when nothing else is fighting the firmware's joint loop**. Two sources of
conflict to neutralize:

1. **MC's own controllers** (`STAND_DEFAULT`, `LOCOMOTION_DEFAULT`, …)
   actively drive joints and will overwrite our targets.
2. **MC arbitration** of high-level intents (RC stick, app, vr) into
   `/aima/mc/locomotion/velocity`, which then becomes joint targets.

Two valid recipes, both backed by the official docs. Pick one:

**Option A — Stop the MC module entirely (recommended for whole-body policy).**
This is the FAQ-prescribed approach (`dev/faq/index.html`):

> Q: There is no response when I directly control the motor
> A: If you are controlling the HAL layer directly, the MC module must be
> stopped. Use `aima em stop-app mc` to stop the MC module. Restart your
> program afterward.

```bash
# Run this over ssh on PC2:
ssh agi@10.0.1.41
aima em stop-app mc
# MC is now gone for this session. HAL accepts our commands directly.
# Restart whatever was running before; in our case, launch the deploy.
```

**Pre-condition:** the standing controller is part of MC. Stopping MC
means the robot will fall under its own weight unless it is supported.
Always either:
- have the robot on a gantry with feet ≤ 1 cm off the floor, OR
- have it sitting / lying / hand-supported with motors disengaged.

To restart MC later (e.g. to use the built-in standing or RC walking again):
```bash
aima em start-app mc
```

**Option B — Keep MC running, switch it to `JOINT_DEFAULT` (Position-Controlled Stand).**
Less invasive — MC stays around as a safety net but accepts HAL position
commands as the active source. Reference: `modeswitch.html`.

```bash
# From the X2 SDK workspace (not this deploy package):
ros2 run examples set_mc_action JD     # JD = JOINT_DEFAULT

# verify:
ros2 service call /aimdk_5Fmsgs/srv/GetMcAction \
    aimdk_msgs/srv/GetMcAction "{request: {}}"
# expect info.action_desc == "JOINT_DEFAULT"
```

For our whole-body policy that runs its own balance, Option A is cleaner
(no chance of competing controllers waking up, no need to register an MC
input source). Option B is closer to a research-friendly "joint-position
puppet" mode and is fine when you don't trust the policy yet.

The deploy intentionally does **not** call `SetMcAction` or `aima em`
itself — both have to be operator-driven (the `aima em` command is
shell-only, not exposed as a service), and an explicit pre-flight is also
a useful safety gate.

### Make sure no other input source is publishing

Whether you choose Option A or B, also check that no operator is
accidentally driving the robot from elsewhere:

```bash
# List currently registered MC input sources (RC, app_proxy, vr, ...):
ros2 service call /aimdk_5Fmsgs/srv/GetCurrentInputSource \
    aimdk_msgs/srv/GetCurrentInputSource "{request: {}}"

# If a stick / app is registered and active, either zero its commands or
# disable it via SetMcInputSource (DISABLE = 2002). See MC_control.html.
```

## Step 1 — Dry-run with motors OFF

This is the most important step. Run the entire policy pipeline with the
robot fully powered down (or motors disengaged) and check the published
commands look sensible.

Recommended (wrapper):

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/run-20260420_083925/exported/model_step_016000_g1.onnx \
    --motion gear_sonic_deploy/data/motions_x2m2/x2_ultra_standing_only.x2m2 \
    --dry-run --autostart-after 5 \
    --log-dir /tmp/x2_dryrun_$(date +%Y%m%d_%H%M%S)
```

(`onbot` instead of `local` to build + run on PC2.) Manual form, after
`source ~/agi_ws/install/setup.bash`:

```bash
ros2 run agi_x2_deploy_onnx_ref x2_deploy_onnx_ref \
    --model /opt/x2_models/model_step_016000_g1.onnx \
    --dry-run \
    --autostart-after 5 \
    --log-dir /tmp/x2_dryrun_$(date +%Y%m%d_%H%M%S)
```

Expected on-screen log timeline:

```
[INFO] x2_deploy_onnx_ref starting [DRY-RUN] autostart=5.000000s
[INFO] Loaded ONNX: /opt/x2_models/model_step_016000_g1.onnx (input='actor_obs' [1, 1670])
[INFO] AimdkIo: leg joint names validated against mujoco_joint_names [0..12).
[INFO] AimdkIo: waist joint names validated against mujoco_joint_names [12..15).
[INFO] AimdkIo: arm joint names validated against mujoco_joint_names [15..29).
[INFO] AimdkIo: head joint names validated against mujoco_joint_names [29..31).
[INFO] INIT -> WAIT_FOR_CONTROL (all state sources fresh)
[WARN] Autostart elapsed (5.00s) -> CONTROL [DRY-RUN]
[INFO] CONTROL tick=50 policy_t=0.98s alpha=0.49 grav_z=-0.99
[INFO] CONTROL tick=100 policy_t=1.98s alpha=1.00 grav_z=-0.99
...
```

### What to verify in `/tmp/x2_dryrun_*`

Pick the most recent dry-run log directory and inspect the CSVs:

```bash
LOGDIR=$(ls -td /tmp/x2_dryrun_* | head -n1)
echo "Inspecting: $LOGDIR"

python3 - <<PY
import csv, os
LOGDIR = os.environ.get("LOGDIR") or "$LOGDIR"

with open(f"{LOGDIR}/joint_pos.csv") as f:
    print("joint_pos rows:", sum(1 for _ in f) - 1)

with open(f"{LOGDIR}/target_pos.csv") as f:
    print("target_pos rows:", sum(1 for _ in f) - 1)

with open(f"{LOGDIR}/tick.csv") as f:
    header = next(f).strip().split(",")
    idx = header.index("tilt_trip")
    flags = [int(row.split(",")[idx]) for row in f]
print("tilt_trip ever set:", any(flags))
PY
```

| Check | Expected |
| ----- | -------- |
| `joint_pos.csv` row count | ≥ 50 × seconds_running |
| `target_pos.csv` row count | same as joint_pos |
| `target_pos` ≈ `joint_pos` | yes, when robot is held by gantry near default pose (delta < 5 deg per joint) |
| `tilt_trip` ever set | **no** (if yes, IMU sign convention is wrong — see Troubleshooting) |
| `imu.csv` `qw,qx,qy,qz` | unit-norm, slow-changing |
| `imu.csv` `wx,wy,wz` | small values when robot is hung still |

If any of these checks fail, **stop and debug** before powering motors.

## Step 2 — First powered run, gantry-held

Same command as Step 1 but **drop `--dry-run`** and use the operator gate
(no `--autostart-after`).

> **Foreground only.** The operator gate reads `go` from stdin. Don't run
> the deploy under `nohup`, `&`, or any tooling that detaches stdin — it'll
> hang in `WAIT_FOR_CONTROL` indefinitely.

Recommended (wrapper):

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/run-20260420_083925/exported/model_step_016000_g1.onnx \
    --motion gear_sonic_deploy/data/motions_x2m2/x2_ultra_standing_only.x2m2 \
    --tilt-cos -0.3 --ramp-seconds 2.0 \
    --log-dir /tmp/x2_powered_$(date +%Y%m%d_%H%M%S)
```

Manual form:

```bash
ros2 run agi_x2_deploy_onnx_ref x2_deploy_onnx_ref \
    --model /opt/x2_models/model_step_016000_g1.onnx \
    --tilt-cos -0.3 \
    --ramp-seconds 2.0 \
    --log-dir /tmp/x2_powered_$(date +%Y%m%d_%H%M%S)
```

Pre-flight one more time before typing `go`:

- [ ] Robot is on the gantry, feet still off the floor.
- [ ] Operator can reach E-stop within < 0.5 s.
- [ ] Power up motors via AimDK service (motors will hold their current pose).
- [ ] Confirm log shows `INIT -> WAIT_FOR_CONTROL` (all state sources fresh).
- [ ] Use AimDK to gently move the robot into the trained default standing
      pose (left/right hip pitch ≈ -18°, knees ≈ +38°, ankles ≈ -21°,
      shoulders ≈ ±11°/-11°, elbows ≈ -34°). The closer you start to the
      default pose, the smaller the soft-start ramp jolt.

When ready, in the deploy node's stdin:

```
go
```

Watch the operator console:

```
[operator] go received; transitioning on next tick.
[INFO] CONTROL tick=50 policy_t=0.99s alpha=0.50 grav_z=-0.99
```

For the first ~2 s the soft-start ramp blends the policy target with the
default pose. After that, the full policy is in command. The robot should
hold a stable standing pose, with small joint corrections at 50 Hz.

If the tilt watchdog fires, you'll see:

```
[FATAL] tilt watchdog tripped: gravity_body[z]=+0.10 > threshold -0.30 (~84 deg from upright) -> SAFE_HOLD
```

The deploy will then hold default angles with 4× damping. Slowly lower the
robot back to a vertical pose using the gantry, kill the deploy, and start
over with a new log directory.

## Step 3 — First steps off the gantry

Only attempt this once you've had at least 30 s of stable standing on the
gantry across multiple runs and the CSVs look clean.

- [ ] Lower the gantry slowly until the robot's feet take its weight.
- [ ] Operator's hand is on the E-stop.
- [ ] If you'd like to add explicit walking commands, swap in a recorded
      walking PKL via the X2M2 export and pass `--motion`. Otherwise the
      `StandStill` reference will keep telling the policy "stay put", which
      is a fine first test.

> **Sim2real readiness check.** Before attempting Step 3, the same
> checkpoint should hold the standing pose for ≥ 30 s in the closed-loop
> MuJoCo sim (`./deploy_x2.sh sim --sim-viewer …`). If the policy
> collapses in MuJoCo within seconds, it will collapse on the real robot
> too — fix sim first. As of this writing the 16k checkpoint of
> `run-20260420_083925` only achieves ~2 s of free-stand in MuJoCo with
> the stitched idle motion, so do not skip the sim verification just
> because the gantry-held Step 2 looks calm.

## Troubleshooting

### Watchdog trips immediately on `go`

- IMU sign convention is wrong. Verify `imu.csv` from a dry run: when the
  robot is upright, `qw` should be ≈ +1 and `qx,qy,qz` ≈ 0. If that's the
  case, then `body_frame_gravity_from_quat_wxyz` will produce `[0, 0, -1]`.
  If the IMU reports orientation in xyzw, the conversion in
  `aimdk_io.cpp::on_imu` will load it incorrectly — fix that one place.

### "Joint commands are being published but the robot doesn't move"

You're probably not in `JOINT_DEFAULT` mode. Check with
`GetMcAction` (see Step 0 above). The other modes either ignore HAL joint
commands (`PASSIVE_DEFAULT`, `DAMPING_DEFAULT`) or run a competing
controller (`STAND_DEFAULT`, `LOCOMOTION_DEFAULT`) that overrides them.

### "head_pitch target_pos is changing in CSV but the head doesn't tilt"

That's expected — `head_pitch_joint` is part of the 31-DOF training
contract but is not physically actuated on current X2 Ultra firmware
("only yaw now, and pitch is unavailable" per `joint_control.html`).
The deploy publishes the policy's pitch target into
`/aima/hal/joint/head/command`, but the firmware silently drops it. Don't
try to "fix" it by zeroing the slot — leaving it as-is keeps the obs/action
shape stable for when pitch firmware ships.

### `AimdkIo: <group> joint name mismatch`

The firmware's published joint order on a `/state` topic doesn't match
`mujoco_joint_names[]`. Either:
1. Update the firmware to publish in the documented MJ order, or
2. Regenerate `policy_parameters.hpp` with the firmware's actual order
   (the codegen script reads from `eval_x2_mujoco.py`'s `MJ_JOINTS` — adjust
   that one constant).

Do **not** "fix" this by reshuffling at runtime. The whole point of the
hard-abort is to prevent silent observation corruption.

### Robot tilts forward / backward immediately on `go`

This is the same sim2sim divergence we saw in MuJoCo. The trained policy
expects implicit-PD dynamics; if the firmware uses explicit PD with
different effective gains, the robot may overshoot. First mitigation:
boost `--ramp-seconds` to 5 s or 10 s so the robot has time to settle
before the policy is fully in command. If the issue persists, the proper
fix is to retune `kps[]` / `kds[]` at the codegen step to match the
firmware's effective gains, regenerate the header, and rebuild.

### Policy looks stuck / actions are tiny

- Verify the ONNX export is the *fused* `g1+g1_dyn` graph, not just the
  decoder. The deploy will refuse to load anything with > 1 input or
  width != 1670.
- Verify proprioception is being primed: if `prop_buf_` was somehow
  reset mid-run, the first tick after reset would broadcast-fill the
  buffer. The deploy never resets `prop_buf_` after entering CONTROL, so
  this should be impossible — file a bug if you see it.

### Policy outputs are wildly large / saturate `--max-target-dev`

- Almost always means the deployed ONNX has drifted from the trained
  PyTorch policy (usually because the encoder input layout differs
  between the exporter and the C++ deploy). Re-export with the validated
  helper:

  ```bash
  conda activate env_isaaclab
  ./gear_sonic_deploy/scripts/reexport_x2_onnx.sh \
      $HOME/x2_cloud_checkpoints/<your-run>
  ```

  It refuses to overwrite the deploy ONNX unless the new one matches the
  `.pt` to within `1e-3` rad on a captured IsaacLab step-0 ground truth.
  See [`x2_deployment_code.md → Re-exporting the deploy ONNX from a checkpoint`](../references/x2_deployment_code.md#re-exporting-the-deploy-onnx-from-a-checkpoint)
  for what the helper does and why it exists.

### How do I shut down safely?

Ctrl-C in the deploy terminal will cleanly exit. The motors will retain
their last command (the SafeCommand, which is bounded). For a fully clean
stop, kill the AimDK joint commander first, then Ctrl-C the deploy.

## What's deferred to the next phase

Phase 6 (post-bring-up) will add:

- A `freq_test` micro-binary mirroring the G1 deploy's existence, for
  measuring inference latency without hardware.
- Optional Ruckig smoothing on the 500 Hz writer if the trained targets
  exhibit high-frequency jitter at the joints.
- A pluggable input-interface layer (gamepad / VR / external command
  topic) so the operator can change the reference motion without
  restarting the deploy.

For now, intentionally minimal — the goal of Phase 3+4+5 is to get to
"first stable standing on a gantry" with the smallest possible code
surface area.
