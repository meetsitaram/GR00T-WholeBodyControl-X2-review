# Pick and Place Commands

=============== do not auto edit this section ===============
### start sonic on PC2 : Robogym Wifi
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

### stop sonic
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32

### run teleop stack with recording enabled
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --with-record \
    --head-cameras \
    --output-dir data/lerobot/x2_pick_and_place_soda_can \
    --task "pick up the mini soda can with your left hand and place it in the open black container on the right"

### camera access on PC2
gear_sonic_deploy/scripts/x2_pc2_cameras.sh status --host 192.168.86.32
gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal --host 192.168.86.32

### replay episode 
./gear_sonic/scripts/view_x2_recorded_dataset.sh --dataset x2_grab_a_drink --episode 6

### run vla on x2-real
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model /home/stickbot/Projects/GR00T-WholeBodyControl/data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"

### run vla on x2-sim
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"

### pure IK kinematic teleop - no planner, no sonic
.venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic     --output-dir /tmp/ik_debug_20260607     --task "ik debug"     --rate 50

### operator calibration
.venv/bin/python -m gear_sonic.scripts.vr_operator_calibrate --operator-id default



=============== end of pure manual notes section ===============

---

## Recording variants (drop-in replacements for the "start local stack" step)

Your usual `run_x2_quest3_planner_stack.sh --pc2-host …` is teleop-only
(no parquet writes). To record a LeRobot dataset for VLA training,
replace that one line with one of the variants below. The other two
lines in your runbook (the SONIC start on PC2 + the SONIC stop) stay
exactly the same.

### Record without head cameras (legacy single-camera schema)

Writes `observation.images.ego_view` (MuJoCo render) only.

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --with-record \
    --output-dir data/lerobot/x2_pick_place_v1 \
    --task "pick up the apple and place it in the bowl"
```

### Record WITH the three real PC2 head cameras (recommended)

Writes four video tracks per episode:
`observation.images.{ego_view,head_front,stereo_left,stereo_right}`.
`--head-cameras` auto-launches the PC2 ROS→ZMQ bridge over SSH
against `--pc2-host` before spawning the recorder; `--camera-host`
defaults to `--pc2-host` so no extra plumbing.

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --with-record \
    --head-cameras \
    --output-dir data/lerobot/x2_pick_place_cams_v1 \
    --task "pick up the apple and place it in the bowl"
```

### Robocasa scene mode (auto-fills the task from the scene)

Drops the `--task` flag — `RobocasaTaskMirror` pulls the canonical
instruction from the scene metadata. Add `--head-cameras` here too if
you want the four-track schema.

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --with-record \
    --head-cameras \
    --robocasa-env X2PickPlaceApple \
    --output-dir data/lerobot/x2_pick_place_apple_v1
```

---

## Pre-flight if the PC2 head cameras came up missing

The IMX900 stereo pair sometimes loses to `orbbec_camera` in the boot
Argus race. Check + restart-hal + verify, in that order:

```sh
gear_sonic_deploy/scripts/x2_pc2_cameras.sh status      --host 192.168.86.32
gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal --host 192.168.86.32   # only if stereo pubs=0
gear_sonic_deploy/scripts/x2_pc2_cameras.sh grab        --host 192.168.86.32   # one JPEG per cam back to laptop
```

You only need to do this on first boot of the day (or after a manual
`aima em` bounce). Once `status` shows `pubs=1` on all four head
topics, the bridge will Just Work from then on.

## Camera bridge cleanup

`--head-cameras` deliberately leaves the bridge running on PC2 after
the recorder exits so back-to-back record sessions don't pay the
cold-start. Tear it down at end of day with:

```sh
gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-stop --host 192.168.86.32
```

---

## Autonomous VLA replay on the real robot

Drop-in replacement for the "start local stack" line in the manual
section above: same SONIC daemon on PC2 (`x2_pc2_daemons.sh start`),
same camera bridge — only the laptop-side script switches between
**teleop** (`run_x2_quest3_planner_stack.sh`), **recording**
(`...planner_stack.sh --with-record --head-cameras`), and
**autonomous VLA** (the launcher below).

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"
```

That's it — one command. Preflight (PC2 ping + x2_debug + cameras +
model files + bridge python) runs **inside** the launcher before any
pose target leaves the laptop. If the camera bridge isn't already up
from your recording session, the preflight auto-SSHes to PC2 and
runs `x2_pc2_cameras.sh serve` for you. A 5-second red safety banner
counts down before the bridge goes live so you can Ctrl-C if anything
looks off.

> Legacy flag `--sonic-checkpoint` (env: `SONIC_CHECKPOINT`) still
> works — the launcher accepts it as a deprecated alias and prints a
> one-shot warning. Migrate at your convenience.

Full operator runbook (safety checklist, knobs, troubleshooting,
postmortem):
[`docs/source/tutorials/x2_vla_runtime.md`](docs/source/tutorials/x2_vla_runtime.md).
For the cross-mode architecture (why this script feeds the same SONIC
daemon as the teleop + recording launchers), see
[`docs/source/references/x2_sonic_runtime_architecture.md`](docs/source/references/x2_sonic_runtime_architecture.md).

Stop / cleanup (PC2 daemons + cameras keep running across runs;
intentional — that's what lets you flip between teleop / recording /
inference modes without restarting SONIC):

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh stop \
    --run-dir /tmp/x2_vla_runtime-YYYYMMDD_HHMMSS
```

---

## VLA body modes (VR locomotion / arm-manipulation analog)

Quest 3 teleop uses `LOCOMOTION` (planner drives the base) vs
`ARM_MANIPULATION` (arms from VR IK, base planted). The VLA bridge
exposes two modes via `--body-mode` (default: `manipulation`):

| VLA `--body-mode` | VR analog | Body on wire | Decoder |
|---|---|---|---|
| `manipulation` (default) | `ARM_MANIPULATION` | decode arms/head/hands; freeze `legs,waist` | required |
| `locomotion` | `LOCOMOTION` | full-body decode | required |

**Typical sim / real run** (manipulation is the default — omit `--body-mode`):

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"
```

**Full body** (when ready to test locomotion):

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --body-mode locomotion \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"
```

**Runtime mode switch** (no restart): pass
`--mode-control-file /tmp/vla_body_mode` and write one word per line:

```sh
echo manipulation > /tmp/vla_body_mode   # arms+hands, legs frozen (default)
echo locomotion > /tmp/vla_body_mode     # full-body decode
```

Optional: `--freeze-body-groups legs,waist,head` overrides which groups
stay pinned in manipulation mode (default freeze: `legs,waist`).

---

## Sim-first VLA debugging (run *this* before powering the robot)

When the powered run looks broken (vibration, divergent joint targets,
unexpected motion), do the exact same VLA inference loop in
simulation first — same checkpoint, same modality, same wire shaping —
and watch the MuJoCo viewer. No risk to hardware.

Same launcher as real robot — just **omit** `--pc2-host` (like quest3):

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"
```

What the sim path mirrors from the real-robot path:

| Surface | Same as real? | Notes |
|---|---|---|
| Wire (`pose` on `:5556`, `x2_debug` on `:5557`) | yes | byte-identical schema |
| Action surface (motion_token + L/R hand joints) | yes | same three heads |
| SONIC motion-token decoder | yes | same `.pt` via `--motion-token-decoder` |
| Ramp-in + LPF (`VLA_RAMP_IN_TICKS`, `VLA_TARGET_LPF_HZ`) | yes | same defaults |
| Proprio assembly (990-D) | yes | from sim's `x2_debug` |
| Cameras | **degenerate stereo** | sim renders `stereo_head_front` once and aliases as both `stereo_left` and `stereo_right` — enough to validate the loop, not enough to validate visual reasoning |
| Hand bridge | yes (Docker-side) | finger commands go through `--sim-with-omnihand` |
| Body control | yes (Docker-side) | C++ deploy runs the fused SONIC ONNX |

Stop (also fixes wedged :5556/:5557 from a prior sim run):

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh stop
```

After each sim run, stats land under `/tmp/x2_vla_runtime-<timestamp>/`:

| File | What it captures |
|---|---|
| `x2_debug_trace.csv` | Per-tick deploy telemetry (joints, grav, last_action) |
| `x2_debug_summary.json` | Run aggregates (max drift, safety events) |
| `vla_chunks/chunk_*.npz` | Per-inference VLA I/O (tokens, camera frames, proprio) |
| `bridge.log` / `deploy.log` | Full stdout from bridge + sim deploy |

The launcher prints chunk aggregates at the end. Re-inspect anytime:

```sh
.venv/bin/python scripts/inspect_vla_chunks.py /tmp/x2_vla_runtime-*/vla_chunks
```

Full architecture (incl. the ghost camera adapter table):
[`docs/source/references/x2_sonic_runtime_architecture.md`](docs/source/references/x2_sonic_runtime_architecture.md)
section 7.

---

## Why we add `--lock-head-straight` to the SONIC start

### The observation

With SONIC running, you cannot move the head by hand — it feels stiff.
Confirmed:

* Head yaw motor works (free movement with SONIC down, and via the AgiBot app)
* SONIC was actively holding head yaw at ~+20° (commanded +0.50 rad,
  measured +0.345 rad)
* Head pitch is not actuated in firmware regardless

### Root cause

The deploy publishes full PD commands (kp≈16.8, kd≈1.07) on
`/aima/hal/joint/head/command` at 50 Hz. The policy's head target was
drifting off-center, and the deploy held it there stiffly.

Sending head joints through the ZMQ pose stream (VR / VLA path) does
**not** override this — the policy decides head motor targets, and
there is no head bypass (unlike the wrist bypass).

### The fix

`--max-target-dev-head` already exists in the C++ safety stack: it
clamps the policy's head target to within ±N radians of the trained
default (yaw=0, pitch=0). Setting it to a small positive value
(`0.01` ≈ 0.6°) effectively locks the head straight ahead. The
trained default *is* straight-ahead, so clamping there just keeps the
policy from drifting off-center.

### What we shipped

A convenience flag on `x2_pc2_daemons.sh`:

```sh
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start ... --lock-head-straight
```

It expands to `--max-target-dev-head 0.01` on `deploy_x2.sh`, which
overrides the tuning YAML's default of `0.50` (~29° sweep). Override
the clamp value via env var `LOCK_HEAD_STRAIGHT_RAD` if needed (e.g.,
loosen to `0.05` ≈ 3°).

### What it does and doesn't do

| Effect | Result |
|---|---|
| Locks head yaw near 0 (straight) during SONIC | yes |
| Removes head stiffness | no — head is still PD-held, just at center |
| Enables operator head control during teleop | no — would need a head bypass like wrists |
| Fixes head pitch (up/down) | no — firmware limitation |

---

## Heading stability on VLA start (bridge-side fixes 2026-06-07)

If, on the **first** VLA start of a SONIC session, the robot turned
sharply toward "world yaw=0" (where the deploy was first started) —
e.g. -45° starting heading rotated right to neutral, or -180° heading
spun nearly all the way around — that was the **bootstrap
yaw-reference race**, now fixed in the bridge:

| Bridge knob (auto, no flag) | Why it matters |
|---|---|
| Withhold pose publish until first `x2_debug` arrives | Until the bridge knows the robot's measured heading, every `root_quat_xyzw` it could ship is identity (yaw=0) — and the deploy was *already in CONTROL*, so it would lock onto that phantom yaw=0 reference. The C++ deploy has a bootstrap-safe override that uses the measured quat as the reference when `LastReceivedMonotonicS() < 0`; the bridge now keeps that escape hatch active by not sending a phantom first frame. |
| Live yaw-rebase on `root_quat_xyzw` + future window | Once `x2_debug` is alive, every wire frame's `root_quat` = `R_z(measured_yaw)`. Mirrors the dataset recorder's `_compute_idle_root_quat_xyzw`. |
| Surgical `waist_yaw` (MJ slot 12) pinned to measured during freeze | `idle_stand[0]`'s waist_yaw is ~33° off the trained default — freezing the whole "legs+waist" group to the clip without this pin would drag the robot to that 33° offset every run. Pin only the dominant heading effector; keep clip jitter on the other frozen DOFs so the policy's balance signal is preserved. |

Bridge log markers to look for:

```text
[live-VLA] withholding pose publish until first x2_debug frame arrives ...
[live-VLA] first pose publish (tick=N); x2_debug seen, root_quat now tracks live heading.
[live-VLA] root_quat yaw-rebase ACTIVE: ... (yaw=+178.2deg)
```

If you ever see a sharp turn on VLA start again, grep `bridge.log` for
those three lines — their absence (or reversed ordering) is the smoking
gun. Validate from multiple starting headings (e.g. -135°, +90°, ±180°)
to confirm the bridge is heading-agnostic on first publish.