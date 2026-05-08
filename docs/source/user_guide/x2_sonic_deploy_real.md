# Running a SONIC-Trained Model on the AgiBot X2 Ultra

This is the end-to-end runbook for taking a SONIC training checkpoint and
driving the real X2 Ultra with it. It covers both supported topologies:

- **Topology A — laptop over wired ethernet**: faster iteration, easiest
  to attach a debugger, recommended for first bring-up.
- **Topology B — on-bot Orin NX (PC2)**: zero ethernet jitter, recommended
  for production / extended runs.

Companion documents:
- [`x2_first_real_robot.md`](x2_first_real_robot.md) — operator safety
  checklist and troubleshooting (read this *first* if you've never moved
  this robot before).
- [`x2_deployment_code.md`](../references/x2_deployment_code.md) — reference
  for the deploy package's architecture, CLI, observation/action contract,
  and ROS 2 topics.

The deploy package itself is `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/`.

## TL;DR

> **The fast path is `gear_sonic_deploy/deploy_x2.sh`.** It bundles
> Steps 5–7 (build + ssh stop-mc + run) and is the most-tested entry point
> on this repo. The manual `ros2 run` form is documented below for when
> you need to bypass the wrapper. See the
> [`deploy_x2.sh` quick reference](#deploy_x2sh-the-fast-path) further
> down.

```bash
# (0) Pick a checkpoint dir.  Adjust to YOUR cloud run.
RUN=$HOME/x2_cloud_checkpoints/run-20260420_083925
ONNX=$RUN/exported/model_step_016000_g1.onnx
PT=$RUN/model_step_016000.pt

# (1) Export ONNX from the trained checkpoint  (dev box with CUDA + IsaacLab)
python gear_sonic/eval_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_x2_ultra \
    +checkpoint=$PT \
    +headless=True ++num_envs=1 \
    +export_onnx_only=true
# Produces:  $RUN/exported/model_step_016000_g1.onnx   (1670 -> 31)

# (2) Sim parity sanity check  (same dev box, MuJoCo, no robot needed)
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
    --onnx $ONNX \
    --motion gear_sonic/data/motions/x2_ultra_standing_only.pkl \
    --compare-pt $PT \
    --parity-threshold 1e-4 --no-viewer --max-episode 30.0
# Verdict line should say: PASS

# (3) Codegen the C++ joint maps + kp/kd from eval_x2_mujoco.py constants
#     -> writes gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp
python gear_sonic_deploy/scripts/codegen_x2_policy_parameters.py --check
# If --check exits non-zero, drop --check to regenerate, then rebuild the deploy.

# (4) Convert reference motion PKL -> X2M2 binary  (optional; default is StandStill)
python gear_sonic_deploy/scripts/export_motion_for_deploy.py \
    --in  gear_sonic/data/motions/x2_ultra_standing_only.pkl \
    --out gear_sonic_deploy/data/motions_x2m2/x2_ultra_standing_only.x2m2

# (5+6+7) One-shot via the wrapper (recommended). Hoist the robot on a gantry first.
./gear_sonic_deploy/deploy_x2.sh local \
    --model  $ONNX \
    --motion gear_sonic_deploy/data/motions_x2m2/x2_ultra_standing_only.x2m2 \
    --dry-run --autostart-after 5 \
    --log-dir /tmp/x2_dryrun_$(date +%Y%m%d_%H%M%S)
```

When the dry-run CSVs look clean (see
[bring-up checklist](x2_first_real_robot.md#step-1--dry-run-with-motors-off)),
drop `--dry-run` and use the operator gate (omit `--autostart-after`, type `go`
in the foreground terminal).

### `deploy_x2.sh` — the fast path

`gear_sonic_deploy/deploy_x2.sh` wraps the manual flow:

| Mode | What it does |
| ---- | ------------ |
| `local`  (default) | colcon build on this box; talks to the robot via DDS over the wired SDK ethernet |
| `onbot`  | rsync the package to PC2, build there over ssh, run there |
| `sim`    | build locally + launch the MuJoCo bridge in the background; isolated loopback DDS so it can't touch a real robot |

It does the colcon build, optionally `ssh`es in to `aima em stop-app mc`,
prints the resolved `ros2 run` command, and execs it. Operator pre-flight
(gantry, E-stop, motors-on) is still on you. Run `./deploy_x2.sh --help`
for the full flag list.

### Environment

This repo has been developed and tested on Ubuntu 24.04 hosts. The
documented `ros2 run` / `colcon build` commands assume **Ubuntu 22.04 +
ROS 2 Humble**. You have three options:

- **Topology B (on-bot, recommended)** — PC2 / Orin NX (`10.0.1.41`)
  ships with Ubuntu 22.04 + ROS Humble per `dev/quick_start/prerequisites.html`.
  Build and run on PC2.
- **Docker container on your laptop** — `gear_sonic_deploy/docker_x2/` is a
  Ubuntu 22.04 + Humble + ONNX Runtime + MuJoCo image with persistent
  colcon volumes. Two entry points:
    - `./docker_x2/enter_sim.sh` — sim DDS (loopback isolated, pair with
      `deploy_x2.sh sim`).
    - `./docker_x2/get_x2_sonic_ready.sh` — real-robot DDS (overlays
      `docker-compose.real.yml`, sets `ROS_DOMAIN_ID=0`,
      `ROS_LOCALHOST_ONLY=0`, pins CycloneDDS to `enp10s0`, and uses
      `network_mode: host` so it shares the host's SDK ethernet — modeled
      on the working setup in
      [`agitbot-x2-record-and-replay`](https://github.com/Bot-Land-Inc/agitbot-x2-record-and-replay)).
    Pre-flight on the host: `enp10s0` static `10.0.1.2/24`, SDK cable in,
    `ping 10.0.1.41` succeeds.
- **Native Ubuntu 22.04 laptop** — same commands work as written.

## Prerequisites

You need a converged SONIC training checkpoint that was trained with the
universal-token actor (encoder + decoder) on the X2 Ultra embodiment. If
your `eval_x2_mujoco_onnx.py` parity check passes, you're good.

| Need | Where it lives | How to get it |
| ---- | -------------- | ------------- |
| Trained `.pt` checkpoint | `~/x2_cloud_checkpoints/run-YYYYMMDD_HHMMSS/model_step_NNNNNN.pt` | Output of `eval_agent_trl.py` training (synced from cloud node) |
| Fused ONNX (`*_g1.onnx`) | `~/x2_cloud_checkpoints/run-.../exported/model_step_NNNNNN_g1.onnx` | `eval_agent_trl.py ... +export_onnx_only=true` (Step 1 below) |
| `policy_parameters.hpp` | `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/` | `gear_sonic_deploy/scripts/codegen_x2_policy_parameters.py` (Phase 1; Step 3 below) |
| `agi_x2_deploy_onnx_ref` package | `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/` | This repo |
| `aimdk_msgs` ROS 2 pkg | `agibot-x2-references/lx2501_3-v0.9.0.4/src/aimdk_msgs/` | `colcon build --packages-select aimdk_msgs` |
| ONNX Runtime ≥ 1.16 (C++) | `/opt/onnxruntime/` (default) | Install snippet in [Step 5](#step-5--build-the-deploy) |

**Hardware:**

- AgiBot X2 Ultra running firmware that publishes the documented
  `/aima/hal/joint/{leg,waist,arm,head}/{state,command}` and
  `/aima/hal/imu/torso/state` topics (verify with `ros2 topic list`).
- Gantry / overhead support sufficient to keep the robot vertical with feet
  ≤ 1 cm off the floor.
- Operator E-stop within reach, tested.
- Either:
  - **Topology A**: a laptop with a free RJ45 port and an ethernet cable; OR
  - **Topology B**: ssh access into the on-bot Orin NX (PC2, `10.0.1.41`).

## Step 1 — Export the fused ONNX

The C++ deploy expects a single fused encoder+decoder ONNX file with
input shape `[1, 1670]` and output shape `[1, 31]`. The training script
already knows how to make this.

From your dev box, with the IsaacLab + SONIC environment activated:

```bash
python gear_sonic/eval_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_x2_ultra \
    +checkpoint=$HOME/x2_cloud_checkpoints/run-YYYYMMDD_HHMMSS/model_step_016000.pt \
    +headless=True ++num_envs=1 \
    +export_onnx_only=true
```

Notes:
- The Hydra arg syntax is `+exp=...`, `+checkpoint=...`, `+export_onnx_only=true`
  (with the leading `+`), not `--config`. There is **no `x2_eval.yaml`**;
  the X2 experiment is `+exp=manager/universal_token/all_modes/sonic_x2_ultra`.
- The `.pt` files live directly in `<run-dir>/`, not in a `checkpoints/`
  subdirectory; the file is `model_step_NNNNNN.pt`, not
  `checkpoint_step_NNNNNN.pt`.

Output (look for these lines in the run log):

```
Exported encoders ONNX to <experiment_dir>/exported/model_step_016000_encoder.onnx
Exported decoder ONNX to  <experiment_dir>/exported/model_step_016000_decoder.onnx
Exported policy as onnx to: <experiment_dir>/exported
```

The deploy uses **`model_step_NNNNNN_g1.onnx`** (the `g1` encoder fused
with the `g1_dyn` decoder — the X2 Ultra is a "g1-class" embodiment in
SONIC's terminology). The `_smpl.onnx` and `_teleop.onnx` variants are for
other input modalities and must not be used here.

## Step 2 — Verify ONNX/PT parity in MuJoCo

Before touching the robot, prove that the exported ONNX produces actions
that match the original PyTorch checkpoint to within numerical noise. This
is the single best protection against silent observation-builder bugs.

```bash
RUN=$HOME/x2_cloud_checkpoints/run-20260420_083925     # <-- adjust to your run
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
    --onnx       $RUN/exported/model_step_016000_g1.onnx \
    --motion     gear_sonic/data/motions/x2_ultra_standing_only.pkl \
    --compare-pt $RUN/model_step_016000.pt \
    --parity-csv logs/x2/parity_pt_vs_onnx.csv \
    --parity-threshold 1e-4 \
    --no-viewer --max-episode 30.0
```

Expected verdict line at the end of the run:

```
Verdict:               PASS  (mean |a_pt - a_onnx|_inf below 1e-4)
```

If this fails, **stop here**. The ONNX is buggy; do not proceed to robot.
Common causes: wrong tokenizer layout (PT-interleaved vs ONNX-grouped),
proprioception-buffer aging mismatch, joint-remap typo. See
[`x2_deployment_code.md`](../references/x2_deployment_code.md#observation-construction)
for the contract.

## Step 3 — Regenerate `policy_parameters.hpp` (Phase 1)

The C++ deploy doesn't read training configs at runtime. Joint maps, kp/kd
values, action scales, and default angles are baked into a single header
file so they're impossible to silently mismatch with the ONNX.

The codegen script reads `gear_sonic/scripts/eval_x2_mujoco.py` (the
Python source of truth for X2 constants — joint names, IL/MJ remaps,
armature-based PD gains, action scales, default standing pose) and emits
the deterministic C++ header.

```bash
# Verify the on-disk header is in sync with eval_x2_mujoco.py:
python gear_sonic_deploy/scripts/codegen_x2_policy_parameters.py --check

# If --check exits non-zero, regenerate and rebuild:
python gear_sonic_deploy/scripts/codegen_x2_policy_parameters.py
```

The header is written to
`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp`
by default; pass `--output PATH` to override. After regeneration, rerun
the colcon build (Step 5).

## Step 4 — Convert reference motion (optional)

The deploy's tokenizer needs a future-frame source. Two options:

- **`StandStillReference`** (default if you don't pass `--motion`): always
  returns the trained default standing pose. Best starting point for first
  bring-up — the policy will just hold position.
- **`PklMotionReference`** loading a custom `X2M2` binary: actual reference
  trajectory (e.g. the `neutral_walk_test.csv` from the AgiBot bones-seed
  reference, or any trajectory you preprocessed during training).

To convert a SONIC training PKL into the C++ reader's binary format:

```bash
python gear_sonic_deploy/scripts/export_motion_for_deploy.py \
    --in  gear_sonic/data/motions/x2_ultra_standing_only.pkl \
    --out gear_sonic_deploy/data/motions_x2m2/x2_ultra_standing_only.x2m2
```

(For Topology B the output also has to be rsynced onto PC2; see Step 5.)
The X2M2 file format is documented in the
[deploy package README](https://github.com/...../agi_x2_deploy_onnx_ref/README.md#reference-motion-file-format-x2m2).

## Step 5 — Build the deploy

The build target depends on your topology. **Easiest path: skip this
section and let `deploy_x2.sh local` (or `onbot`) do the build for you.**
The notes below document what the wrapper does under the hood.

### Install ONNX Runtime (one-off, both topologies)

The C++ deploy links against ONNX Runtime ≥ 1.16. If `/opt/onnxruntime`
doesn't exist, install it once:

```bash
ORT_VERSION=1.16.3
ORT_TGZ=onnxruntime-linux-x64-${ORT_VERSION}.tgz
wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}
sudo tar -xzf ${ORT_TGZ} -C /opt
sudo ln -sfn /opt/onnxruntime-linux-x64-${ORT_VERSION} /opt/onnxruntime
rm ${ORT_TGZ}
# Verify:
ls /opt/onnxruntime/lib/libonnxruntime.so*
```

For aarch64 (PC2 / Orin NX), substitute `linux-aarch64` for `linux-x64`
in both URLs.

### Topology A — build on your laptop (cross-host control)

Prerequisites on the laptop:
- Ubuntu 22.04 LTS, ROS 2 Humble (matches what AgiBot ships). See the
  [Environment](#environment) note in the TL;DR if your laptop is on
  Ubuntu 24.04 — use the `docker_x2/` container or switch to Topology B.
- ONNX Runtime ≥ 1.16 at `/opt/onnxruntime` (snippet above).
- `aimdk_msgs` built into the laptop's colcon workspace (just clone
  `lx2501_3-v0.9.0.4/src/aimdk_msgs/` into `~/agi_ws/src/`).

```bash
cd ~/agi_ws
colcon build --packages-select aimdk_msgs
colcon build --packages-select agi_x2_deploy_onnx_ref \
    --base-paths /path/to/gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref \
    --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
source install/setup.bash
```

### Topology B — build on PC2 (Orin NX)

PC2 already has Ubuntu 22.04, ROS 2 Humble, and the SDK dependencies
preinstalled per `dev/quick_start/prerequisites.html` ("SDK-on-device
mode"). You only need to drop in the deploy package and build.

```bash
# (a) rsync source from your dev box.  Both packages are required:
ROBOT=agi@10.0.1.41
ssh $ROBOT 'mkdir -p ~/x2_deploy_ws/src /opt/x2_models /opt/x2_motions'
rsync -avz --delete \
    gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref \
    gear_sonic_deploy/src/common \
    $ROBOT:~/x2_deploy_ws/src/
# Also rsync aimdk_msgs if it isn't already in PC2's overlay:
rsync -avz --delete \
    agibot-x2-references/lx2501_3-v0.9.0.4/src/aimdk_msgs \
    $ROBOT:~/x2_deploy_ws/src/

# (b) rsync the runtime artefacts (ONNX + reference motion + ORT):
rsync -avz \
    $HOME/x2_cloud_checkpoints/run-20260420_083925/exported/model_step_016000_g1.onnx \
    $ROBOT:/opt/x2_models/
rsync -avz \
    gear_sonic_deploy/data/motions_x2m2/x2_ultra_standing_only.x2m2 \
    $ROBOT:/opt/x2_motions/
# If /opt/onnxruntime does not exist on PC2, run the install snippet above
# inside an ssh session (use the linux-aarch64 tarball).

# (c) Build on PC2:
ssh $ROBOT '
    cd ~/x2_deploy_ws &&
    source /opt/ros/humble/setup.bash &&
    colcon build --packages-select aimdk_msgs &&
    colcon build --packages-select agi_x2_deploy_onnx_ref \
        --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
'
```

`./deploy_x2.sh onbot --model … --motion …` automates (a)+(c) plus the
subsequent `ros2 run`. Use the manual flow if you need to debug a build
failure on PC2.

## Step 6 — Network setup

### Topology A — Laptop

Per `dev/quick_start/prerequisites.html` and `dev/about_agibot_X2/SDK_interface.html`:

1. Direct ethernet cable from your laptop to one of the **two rear RJ45
   SDK development ports**. Both are gigabit and wired to the development
   computing unit (PC2) and the interaction unit (PC3).
2. Configure your laptop NIC as a static IP **`10.0.1.2/24`**, gateway
   blank.
3. Verify reachability:
   ```bash
   ping -c 3 10.0.1.41           # PC2 (development)
   # Should NOT need to ping 10.0.1.40 (PC1, motion control) for any reason.
   ```
4. Verify DDS discovery:
   ```bash
   ros2 topic list | grep -E '/aima/hal/(joint|imu)'
   ```
   You should see all 8 joint topics plus `/aima/hal/imu/torso/state`. If
   not, the firmware is using the SDK-example typo and you need
   `--imu-topic /aima/hal/imu/torse/state` later.

**Wi-Fi is NOT supported for the control loop.** Per the docs:
"Wi-Fi should only be used for SSH debugging."

### Topology B — On-bot

Skip — everything is on-loopback inside PC2. Just `source` the workspace
and you're done.

## Step 7 — Pre-flight: stop the MC module

The whole-body policy *is* the locomotion controller, so we need to take
the AimDK MC layer (which normally arbitrates locomotion intents) out of
the loop. Per the official FAQ (`dev/faq/index.html`):

> Q: There is no response when I directly control the motor
> A: If you are controlling the HAL layer directly, the MC module must be
> stopped. Use `aima em stop-app mc` to stop the MC module.

```bash
ssh agi@10.0.1.41
aima em stop-app mc
```

**Critical safety pre-condition:** stopping MC also stops the standing
controller. The robot will drop under its own weight unless it is already
supported. Always either:
- have it on a gantry with feet ≤ 1 cm off the floor, OR
- have it sitting / lying down with motors disengaged.

To re-enable MC later (e.g. to use the built-in standing or RC walking):

```bash
aima em start-app mc
```

Alternative (less invasive): keep MC running, switch it to `JOINT_DEFAULT`
via `ros2 run examples set_mc_action JD`. See
[`x2_first_real_robot.md`](x2_first_real_robot.md#disable-mc-so-hal-joint-commands-take-effect)
for the full discussion of the two recipes.

## Step 8 — Run the deploy

Same command for both topologies. **Recommended path: use
`deploy_x2.sh`** (it sources the workspace, picks the right topology, and
forwards every flag below to the underlying `ros2 run`). The manual
`ros2 run` form is shown for reference.

> **Foreground only.** Step 2 below uses an interactive operator gate
> (typing `go` on stdin). Do not launch the deploy under `nohup`,
> backgrounding (`&`), or `tmux send-keys` from outside — it'll hang in
> `WAIT_FOR_CONTROL` forever. Run it in a normal foreground terminal you
> can type into.

### Dry-run (no torque, motors must be powered for state to publish)

Wrapper form:

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model  $HOME/x2_cloud_checkpoints/run-20260420_083925/exported/model_step_016000_g1.onnx \
    --motion gear_sonic_deploy/data/motions_x2m2/x2_ultra_standing_only.x2m2 \
    --dry-run --autostart-after 5 \
    --log-dir /tmp/x2_dryrun_$(date +%Y%m%d_%H%M%S)
```

Manual form (after `source ~/agi_ws/install/setup.bash`):

```bash
ros2 run agi_x2_deploy_onnx_ref x2_deploy_onnx_ref \
    --model  /opt/x2_models/model_step_016000_g1.onnx \
    --motion /opt/x2_motions/x2_ultra_standing_only.x2m2 \
    --dry-run \
    --autostart-after 5 \
    --log-dir /tmp/x2_dryrun_$(date +%Y%m%d_%H%M%S)
```

`--dry-run` zeros every `stiffness` / `damping` field in the published
joint commands, so the firmware will not generate any torque. Every other
piece of the pipeline runs end-to-end (state ingestion, observation
construction, ONNX inference, safety stack, command publishing) — this is
your no-risk wiring test.

Expected log timeline:

```
[INFO] x2_deploy_onnx_ref starting [DRY-RUN] autostart=5.000000s
[INFO] Loaded ONNX: /opt/x2_models/model_step_016000_g1.onnx (input='actor_obs' [1, 1670])
[INFO] Reference motion: PklMotionReference '/opt/x2_motions/x2_ultra_standing_only.x2m2'
[INFO] AimdkIo: leg joint names validated against mujoco_joint_names [0..12).
[INFO] AimdkIo: waist joint names validated against mujoco_joint_names [12..15).
[INFO] AimdkIo: arm joint names validated against mujoco_joint_names [15..29).
[INFO] AimdkIo: head joint names validated against mujoco_joint_names [29..31).
[INFO] INIT -> WAIT_FOR_CONTROL (all state sources fresh)
[WARN] Autostart elapsed (5.00s) -> CONTROL [DRY-RUN]
[INFO] CONTROL tick=50 policy_t=0.98s alpha=0.49 grav_z=-0.99
```

If you see `AimdkIo: <group> joint name mismatch`, **stop**. The firmware
is publishing joints in a different order than the codegen header
expects — see
[Troubleshooting](x2_first_real_robot.md#aimdkio-group-joint-name-mismatch)
in the bring-up doc.

Verify the dry-run CSVs as described in the
[bring-up checklist Step 1](x2_first_real_robot.md#step-1--dry-run-with-motors-off).

### Powered run (operator gate, recommended for first torque)

After at least one clean dry-run with sensible CSVs:

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model  $HOME/x2_cloud_checkpoints/run-20260420_083925/exported/model_step_016000_g1.onnx \
    --motion gear_sonic_deploy/data/motions_x2m2/x2_ultra_standing_only.x2m2 \
    --tilt-cos -0.3 --ramp-seconds 2.0 \
    --log-dir /tmp/x2_powered_$(date +%Y%m%d_%H%M%S)
```

(Equivalent manual form: same `ros2 run` invocation as the dry-run above,
minus `--dry-run` and `--autostart-after`, plus `--tilt-cos -0.3
--ramp-seconds 2.0`.)

No `--dry-run`, no `--autostart-after`. The deploy will get to
`WAIT_FOR_CONTROL`, log "type 'go' to enter CONTROL", and wait for stdin.

Operator's pre-flight before typing `go`:

- [ ] Robot is on the gantry, feet still off the floor.
- [ ] Operator can reach E-stop within < 0.5 s.
- [ ] Power up motors via AimDK service (motors will hold their current
      pose because we're in JOINT_DEFAULT or MC is stopped).
- [ ] Use AimDK to gently move the robot into the trained default standing
      pose (so the soft-start ramp doesn't have to make a big jump).

When ready, in the deploy node's terminal:

```
go
```

Watch for:

```
[operator] go received; transitioning on next tick.
[INFO] CONTROL tick=50 policy_t=0.99s alpha=0.50 grav_z=-0.99
```

For the first ~2 s the soft-start ramp (`--ramp-seconds`) blends the
policy's target with the default pose. After that, the full policy is in
command. The robot should hold a stable standing pose.

Tilt watchdog trips look like:

```
[FATAL] tilt watchdog tripped: gravity_body[z]=+0.10 > threshold -0.30
        (~84 deg from upright) -> SAFE_HOLD
```

`SAFE_HOLD` latches "hold default angles, 4× damping" — operator must
lower the robot to vertical with the gantry, then kill and restart the
deploy to clear it.

## Step 9 — Tear down

```bash
# In the deploy terminal:
Ctrl-C

# On PC2 via ssh:
aima em start-app mc           # restore MC if you stopped it
```

Motors retain their last command (the bounded SafeCommand) until MC takes
back over. For a fully torque-free state, kill the AimDK joint commander
*before* Ctrl-C'ing the deploy, or call
`SetMcAction PASSIVE_DEFAULT` after restarting MC.

## Topology comparison summary

| Concern | Topology A (laptop) | Topology B (on-bot PC2) |
| ------- | ------------------- | ----------------------- |
| Build location | Laptop | PC2 (`ssh agi@10.0.1.41`) |
| Run location | Laptop | PC2 |
| ONNX file location | Laptop disk | `/opt/x2_models/...` on PC2 (rsync over) |
| Latency | gigabit + ~0.1-0.5 ms jitter | loopback, no extra jitter |
| Bandwidth | ~1.9 MB/s well inside 1 Gbps | irrelevant |
| Cameras | **Cannot subscribe cross-host** (90 MB/s raw); use `/compressed` if needed | Native, no problem |
| Failure mode if cable yanks | Subs go silent → INIT → SAFE_HOLD via watchdog | N/A (no cable) |
| Iteration speed | Fast — recompile + rerun locally | Slower — rsync + rebuild on PC2 |
| Best for | First bring-up, debugging, demos | Production runs, leaving the robot operating overnight |

Both topologies use the **exact same binary, same CLI, same logs**. There
is no code branch for "laptop vs on-bot"; ROS 2 DDS handles the rest.

## What the deploy intentionally does *not* do

- **Walking velocity command in.** This deploy assumes the policy gets all
  guidance from the reference-motion stream. There's no `cmd_vel`-style
  input. To add operator velocity input later, plumb a `geometry_msgs/Twist`
  subscriber into the deploy and feed it into the obs builder; for now,
  use `--motion` with a precomputed reference trajectory.
- **Hand control.** The 31-DOF policy doesn't include omnihand fingers;
  those live on `/aima/hal/joint/hand/{state,command}` and are owned by
  the AimDK hand driver. Run `hand_control` separately if you need them.
- **Auto mode-switching.** The deploy never calls `aima em` or
  `SetMcAction` — those are operator pre-flight steps. Cross-host service
  calls are flaky per AgiBot's own docs, and an explicit pre-flight is
  also a useful safety gate.
- **Camera streams.** Vision integration is out of scope for this deploy.
  See the cross-host bandwidth caveat in the comparison table above.

## Quick troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `INIT` never advances to `WAIT_FOR_CONTROL` | Some `/aima/hal/joint/*/state` topic is silent, or IMU spelling is wrong | `ros2 topic hz` each one; if IMU is on `/aima/hal/imu/torse/state`, restart with `--imu-topic /aima/hal/imu/torse/state` |
| `AimdkIo: <group> joint name mismatch` | Firmware joint order ≠ `policy_parameters.hpp` | Either re-flash matching firmware or regenerate the header from the firmware's actual joint order (Step 3) |
| Joint commands are publishing but the robot doesn't move | MC is still controlling joints | `aima em stop-app mc` (Step 7) or switch to `JOINT_DEFAULT` |
| Tilt watchdog fires immediately on `go` | IMU sign / quaternion convention wrong | Check `imu.csv` from a dry run (qw≈+1 when upright); fix `aimdk_io.cpp::on_imu` if conversion is inverted |
| `head_pitch` target changes but head doesn't tilt | Expected — head pitch is not actuated on current firmware | Leave alone; documented in `joint_control.html` |
| Policy outputs are tiny / robot is sluggish | Wrong ONNX file (e.g. `_decoder.onnx` instead of `_g1.onnx`) | Re-export with the fused `g1+g1_dyn` graph (Step 1); the deploy refuses anything that isn't `[1, 1670] → [1, 31]` |
| Sim parity fails (`eval_x2_mujoco_onnx.py` verdict FAIL) | Tokenizer layout or proprioception buffer mismatch | Don't deploy. Fix the export pipeline first; check `x2_deployment_code.md` obs section for the exact contract |

For deeper investigation, see the
[Troubleshooting section in the bring-up doc](x2_first_real_robot.md#troubleshooting).

## What changes between training runs

If you train a new checkpoint, the only artefacts that need to update are:

1. **`model_step_NNNNNN_g1.onnx`** (re-export per Step 1).
2. **`policy_parameters.hpp`** *only if* the training config changed any of:
   joint names/order, kp/kd values, action scales, or default angles.
   Identical training config + new weights = same header, just rebuild
   isn't needed.
3. **Reference motion `.x2m2`** *only if* you switched to a different
   reference clip.

Re-run the parity check (Step 2) after every export. Don't skip it — it's
the cheapest insurance against silent regressions.
