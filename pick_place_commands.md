# Pick and Place Commands

### Action
the pick and place action sequence is raise left hand sideways to shoulder height, then move forward and up to avoid hitting the table while raising the arm, all while closing the fingers. then open the fingers, and slowlly move the arm to wards the soda can to grab it and place it in the black container. 

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
./gear_sonic/scripts/view_x2_recorded_dataset.sh --dataset x2_pick_and_place_soda_can --episode 17

### run vla on x2-real
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model /home/stickbot/Projects/GR00T-WholeBodyControl/data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "pick up the mini soda can with your left hand and place it in the open black container on the right"

### run vla on x2-real with record option
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "pick up the mini soda can with your left hand and place it in the open black container on the right" \
    --vla-max-wire-dev-from-body 1.5 \
    --vla-target-lpf-hz 5.0 \
    --vla-future-lpf-hz 5.0 \
    --vla-hand-lpf-hz 10.0 \
    --vla-max-wire-step 0.07 \
    --with-record \
    --output-dir data/lerobot/x2_pick_and_place_soda_can_n17_50k_v1_rollouts \
    --task "pick up the mini soda can with your left hand and place it in the open black container on the right"

### run vla on x2-real RAW (no bridge wire filters; policy output published as-is)
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "pick up the mini soda can with your left hand and place it in the open black container on the right" \
    --vla-raw


### run vla on x2-sim (robocasa scene: X2PickPlaceApple|X2PickPlaceBowl|X2PickPlaceCube)
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --robocasa-env X2PickPlaceApple \
    --model /home/stickbot/Projects/GR00T-WholeBodyControl/data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "pick up the mini soda can with your left hand and place it in the open black container on the right"

### pure IK kinematic teleop - no planner, no sonic
.venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic     --output-dir /tmp/ik_debug_20260607     --task "ik debug"     --rate 50

### operator calibration
.venv/bin/python -m gear_sonic.scripts.vr_operator_calibrate --operator-id default

### groot training
conda activate env_isaaclab
PYTHONPATH=external_dependencies/Isaac-GR00T:. python \
    external_dependencies/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path data/lerobot/x2_pick_and_place_soda_can \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path gear_sonic/data/x2_modality_config_omnihand_stereo.py \
    --num-gpus 1 \
    --max-steps 50000 \
    --save-steps 10000 \
    --output-dir data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --use-wandb \
    --wandb-project x2-pick-and-place-soda-can \
    --experiment-name x2_pick_and_place_soda_can_n17_50k_v1

### diagnose: FK raw policy intent vs delivered wire from chunk dumps
.venv-viewer/bin/python -m gear_sonic.scripts.diagnose_vla_chunk_fk \
    --chunk-dir /tmp/x2_vla_runtime-LATEST/vla_chunks


### human-in-loop commands (sim)
# --enable-takeover auto-promotes both loopback ports
# (POSE_PROXY_OVERRIDE_PORT=5560, VLA_CONTROL_PORT=5559); no need to
# re-pass them. Pass --vla-control-port 0 only if you want to
# disable the bridge cold-restart on operator release (the wire
# will snap to the mid-decode chunk; only useful for smoke tests).
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --enable-takeover \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --robocasa-env X2PickPlaceCube \
    --prompt "pick up the red cube and drop it into the blue bowl"

./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --no-deploy \
    --takeover \
    --no-x2-debug-bridge --preserve-arms-on-engage


### human-in-loop commands (real)
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --enable-takeover \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "pick up the mini soda can with your left hand and place it in the open black container on the right" \
    --vla-max-wire-dev-from-body 1.5 \
    --vla-target-lpf-hz 10.0 \
    --vla-future-lpf-hz 10.0 \
    --vla-hand-lpf-hz 10.0 \
    --vla-max-wire-step 0.07 \
    --with-record \
    --output-dir data/lerobot/x2_pick_and_place_soda_can_n17_50k_v1_rollouts \
    --task "pick up the mini soda can with your left hand and place it in the open black container on the right"

./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --no-deploy \
    --takeover \
    --preserve-arms-on-engage

./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32


=============== end of pure manual notes section ===============

---

## Notes on the manual-section recipes (archived, not used at runtime)

### `run vla on x2-real with record option (v3 LPF balance)` — tuning history

Same checkpoint, three iterations to land on the v3 LPF/step values:

| Variant | `--vla-target-lpf-hz` / `--vla-future-lpf-hz` | `--vla-max-wire-step` | Outcome |
|---|---|---|---|
| v1 | 10 | 0.10 | arm rose nicely but a 2.5 Hz limit cycle grew exponentially after ~8 s (run 091517) |
| v2 | 3  | 0.04 | limit cycle gone (loop gain << 1) but legitimate rise also over-damped — arm never reached forward-up |
| v3 | 5  | 0.07 | balanced. 5 Hz LPF attenuates the 2.5 Hz limit cycle by -1 dB (factor 0.89); combined with 0.07 rad/tick = 3.5 rad/s velocity cap (vs the limit cycle's 5 rad/s peak) this gives ~2x margin. Legitimate ramps (~0.5 rad/s rise = 0.01 rad/tick) sail through unaffected |

Tuning ladder if v3 misbehaves:
- Oscillations return → tighten `--vla-max-wire-step` to 0.05 first (preserves phase response of legitimate motion); only drop LPF below 4 Hz if the step cap isn't enough.
- Rise is still too slow → try `--vla-target-lpf-hz 7` next (back toward 10 incrementally), keep step at 0.07.

#### v3 + closed-loop tracking feedback (2026-06-10 follow-up 11, Step 1)

The 2026-06-10 PM run reproduced oscillations on identical v3 defaults
that had been stable the day before — confirming static tuning can't
cover the open-loop bridge's sensitivity to battery sag / motor
temperature drift / SONIC PID variation. The structural fix is
closed-loop tracking feedback (see milestone
`2026-06-10_vla_closed_loop_wire.md` for the full design).

**Step 1 rollout** (current real-robot recipe above): keeps every
v3 default (`--vla-target-lpf-hz 5`, `--vla-future-lpf-hz 5`,
`--vla-max-wire-step 0.07`) AND adds `--vla-tracking-feedback`. This
is **additive** — feedback throttles per-arm-joint when the actuator
is lagging; otherwise it's a no-op and the wire matches the v3 path.
The point of running both in parallel is to isolate the feedback's
contribution: if Step 1 reproduces v3's oscillations, the feedback
law is at fault; if Step 1 is smoother, the closed loop is helping.

Operator-visible telemetry: pub-tick log gains `tf_throttle=N/14`
(N out of 14 arm joints actively protected this tick). N=0 most of
the time = feedback is a no-op. Sustained high N = actuator is
saturating; investigate.

**Step 2** (separate commit after 2+ successful Step 1 runs): default
flips ON, v3 statics relax (LPF 5→8, blend 40→10, step-cap 0.07→0.05).

### `run vla on x2-real RAW` — what `--vla-raw` actually disables

Disables every wire-shaping knob the bridge layers on top of the policy output: body / future / hand LPFs, per-tick step clamps, chunk-blend windows, and the dev-from-body cap.

Intentionally **kept on** under `--vla-raw`, do NOT remove:
- `--vla-max-action-il 8.0` (default) — clips the policy's `last_action` proprio echo, NOT the wire. Disabling it triggers a proprio-feedback runaway (`last_action` grows unboundedly → policy predicts ever-larger `action_il` → repeat) that blew the wire to `body_Δ=90+ rad` within 50 chunks in a real run.
- `--vla-ramp-in-ticks 75` / `--vla-decode-delay-ticks 150` — one-shot hand-off smoothing at the start of the run.
- `--vla-body-mode manipulation` — pins legs+waist columns of the wire to `idle_stand`. The physical torso can still tilt under inertial reaction from rapidly chasing arm targets, but the *command* for those joints is held.

PC2's `--max-target-dev` guard on the SONIC tracker remains active as the last line of defense. Expect jerky chunk-boundary jumps and keep the E-stop in hand. Add `--with-record / --output-dir / --task` to capture the rollout for offline analysis.

### `run vla on x2-sim (robocasa scene …)` — distribution caveat

This policy was trained on real PC2 head-cam stereo of an actual soda can in your workspace. A robocasa kitchen scene is out-of-distribution visually. Use this as a sanity check of arm motion only, not as a pick-success eval (the can is not in any of these scenes).

### `groot training` — wandb requirement

`--use-wandb` is required to get loss / lr / grad-norm curves on W&B. Without it, HF Trainer sets `report_to="none"` and only the local text log captures progress (painful to monitor for an 11-hour run). `--experiment-name` sets the W&B run name (defaults to `output-dir` basename).

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

## Manual takeover during VLA (operator nudges, no restart)

If VLA gets stuck (e.g. arm hovers forward-up and refuses to descend
on the soda can), the operator can grab the wire via VR teleop,
re-position the arm, then let go — the mux emits an edge event
that triggers a bridge cold-restart so the next decoded chunk
ramps in from the operator's hand-off pose. No process restarts.

> **2026-06-11 update.** The single-process `x2_pose_proxy.py` was
> split into a laptop-side `x2_pose_mux` (does the merge / engagement
> ramp / `vla_control` edges) plus a slim PC2-side `x2_pose_watchdog`
> (renamed from `x2_pose_proxy.py`; does only the fallback ladder).
> See [`2026-06-11_pose_mux_split.md`](docs/source/user_guide/milestones/2026-06-11_pose_mux_split.md).
> Operator-visible behaviour is unchanged. The big delta in the runbook:
> the `POSE_PROXY_OVERRIDE_*` / `POSE_PROXY_TELEOP_MODE_*` /
> `POSE_PROXY_CONTROL_*` env vars **no longer go to PC2**. Setting any
> of them in PC2's environment now fails the daemon launch (the
> 2026-06-11 migration error). They drive the laptop-side mux via
> `run_x2_vla_runtime.sh --enable-takeover` instead.

**Wire topology** (2026-06-11 split):

```text
LAPTOP                                            PC2
══════                                            ═══

bridge :5571 (internal) ─┐
                         ├─► mux  ═══wifi═══►  watchdog  ──► deploy
recorder :5560 (override)─┘   *:5556          *:5558      :5558
                              (canonical)
                              │
                              ▼
                         vla_control PUB :5559
                         (loopback; bridge SUBs)
```

* The bridge moves from `*:5556` to `*:5571` (internal loopback) so
  the mux can bind the canonical `*:5556` instead. PC2's watchdog
  (and any other external SUB) continues SUBing at `LAPTOP:5556` —
  transparent to them.
* `:5571` (not `:5570`) was picked deliberately. `:5570` is owned by
  the kplanner stack's `x2_debug_to_robot_pose_bridge` (publishes the
  `robot_pose` topic); putting the VLA bridge there would zmq-bind-
  collide whenever the operator runs both `run_x2_vla_runtime.sh
  --enable-takeover` and `run_x2_quest3_planner_stack.sh` together
  (the takeover flow).
* Operator pose stays on laptop loopback. Only the merged wire
  crosses wifi.

### One-time PC2 daemon setup

Bring up the PC2 watchdog + deploy + hand bridge + motor monitor as
usual. **Do not set any `POSE_PROXY_OVERRIDE_*` / `POSE_PROXY_TELEOP_*` /
`POSE_PROXY_CONTROL_*` env vars** — those moved to the laptop on
2026-06-11 and now hard-fail the daemon launch with a migration
pointer:

```sh
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight
```

Watchdog log should show on startup:

```text
[pose_watchdog] upstream:   tcp://192.168.86.22:5556 topic='pose'
[pose_watchdog] downstream: tcp://*:5558 topic='pose'
[pose_watchdog] x2_debug:   tcp://127.0.0.1:5557 (yaw rebase)
```

(No `override SUB` / `vla_control PUB` lines anymore — the watchdog
doesn't know about either.)

### Start the VLA bridge with manual takeover

The `--enable-takeover` master switch tells the laptop launcher to
spawn `x2_pose_mux` locally, flip the bridge's pose PUB to the
internal `:5571`, and **auto-promote** the two laptop-loopback ports
to their canonical defaults:

* `POSE_PROXY_OVERRIDE_PORT` → `5560` (mux SUBs the recorder's
  operator pose here)
* `VLA_CONTROL_PORT` → `5559` (mux PUBs override-engaged / released
  edges; bridge SUBs on loopback for cold-restart on release)

Both are now purely on the laptop after the 2026-06-11 split, so the
operator no longer has to re-pass them on every invocation. Pass
`--vla-control-port 0` if you want to explicitly disable cold-restart
(operator override still works, but on release the wire snaps to
whatever VLA chunk is mid-decode — only useful for arbitration smoke
tests, not production).

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --enable-takeover \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "pick up the mini soda can with your left hand and place it in the open black container on the right"
```

Launcher log should show:

```text
[vla-runtime] real-robot manual-takeover plumbing ON:
[vla-runtime]   bridge :5571 -> mux *:5556 -> PC2 watchdog 192.168.86.32:5558 -> PC2 deploy
[vla-runtime]   override SUB: 127.0.0.1:5560 (recorder operator pose)
[vla-runtime]   vla_control PUB: 127.0.0.1:5559 -> bridge SUB (cold-restart on release)
[vla-runtime] spawning pose mux -> /tmp/x2_vla_runtime-…/pose_mux.log
[vla-runtime] pose_mux.pid = 12345
```

(If you see `WARN: vla_control DISABLED ...` instead of the
`vla_control PUB:` line, you passed `--vla-control-port 0`. Operator
override will still work but release-snap is back on the table.)

Mux log should show:

```text
[pose_mux] primary SUB: tcp://127.0.0.1:5571 topic='pose'
[pose_mux] override SUB: tcp://127.0.0.1:5560 topic='pose' (stale_ms=200)
[pose_mux] out PUB:     tcp://*:5556 topic='pose'
[pose_mux] vla_control PUB: tcp://127.0.0.1:5559 topic='vla_control'
```

Bridge log should show:

```text
[live-VLA] vla_control SUB connected: tcp://127.0.0.1:5559 topic='vla_control'
[live-VLA] vla_control SUB enabled (host=127.0.0.1 port=5559 …)
```

(Note: the bridge's `vla_control` host defaults to `127.0.0.1` now —
the mux is local. Pre-2026-06-11 runbooks that passed
`--vla-control-host <PC2_IP>` still work but cross wifi for no reason.)

### Start the teleop stack alongside (publishes operator wire on :5560)

The Quest3 manager + planner + recorder run as a separate process
group. Use the `--takeover` shortcut (= `--pose-port 5560`) so the
recorder PUBs the operator's wire into the mux's override SUB
instead of fighting the bridge for `:5556`. Pass `--no-deploy` so
we don't spawn a competing C++ deploy.

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --takeover \
    --no-deploy
```

Look for this on the stack's stderr:

```text
[quest3_stack] manual-takeover wiring ON:
[quest3_stack]   recorder pose PUB -> tcp://*:5560 (the mux's --override-port)
[quest3_stack]   prerequisite: run_x2_vla_runtime.sh --enable-takeover MUST be running with the same --override-port
```

If you don't see that line, the recorder is PUBing to `:5556` (the
mux's input is starved) and engagement won't work.

### Operator workflow

1. VLA runs autonomously; arm gets stuck.
2. Operator engages teleop the **usual** way — press `A+B+X+Y` on the
   Quest 3 controllers to enter `LOCOMOTION`, or just hold trigger
   to enter `ARM_MANIPULATION`. The Quest3 manager is unchanged; it
   just publishes its arm/finger commands as always, which the
   recorder packs into pose frames on :5560.
3. The proxy detects fresh frames on the override port and **prefers
   them** over the bridge's frames. The deploy commands the operator's
   pose. The proxy emits `override_engaged` on the vla_control PUB;
   the bridge stops sending decoded chunks (it ships the measured
   pose instead so the wire stays alive).
4. Operator hand-positions the arm and **releases** by pressing
   `A+B+X+Y` again to drop the manager back to `OFF` (or kills the
   teleop stack as a backstop).
5. The proxy detects the disengage from the manager's **`stream_mode`
   broadcast** (the canonical, 2026-06-10-follow-up path):
   - **Mode-gated engagement** (recommended; default in sim, opt-in
     on real robot via `POSE_PROXY_TELEOP_MODE_PORT`). The proxy SUBs
     to the Quest3 manager's `stream_mode` topic (manager's
     `--recorder-pub-port`, default `5564`, msgpack payload with
     `mode: "OFF" | "LOCOMOTION" | "ARM_MANIPULATION"`). Engagement
     is **STRICT**: override frames are forwarded iff the mode signal
     is fresh AND `mode != "OFF"`. Motion-hysteresis / frozen
     detection are bypassed entirely, so **holding the controller
     still in ARM_MANIPULATION no longer flickers release** (the
     2026-06-10 user bug). On `mode == "OFF"`, the proxy fires
     `override_released` on the **next tick** — no debounce, the
     operator's button press IS the truth.
   - **Fail-closed semantics**: if the mode signal goes stale past
     `--teleop-mode-stale-ms` (default 1 s = ~50 manager ticks @
     50 Hz), engagement is **BLOCKED** — a dead manager hands the
     wire back to VLA within ~1 s instead of silently falling back
     to flicker-prone heuristics.
   - **Legacy motion-hysteresis path** (only when
     `--teleop-mode-port` is unset or msgpack is missing): see the
     pre-2026-06-10 `--override-frozen-ticks` /
     `--override-engage-motion-ticks` flags below. Kept for replay
     smoke tests and for older deployments that don't run the
     manager. **This path WILL flicker** when the operator holds
     the controller still — use mode-gating in production.
   - **Silence backstop**: if the entire teleop stack dies (Ctrl-C,
     kernel panic, network drop), the override SUB stops receiving
     frames and `--override-stale-ms` (default 200 ms) triggers
     release. Acts as the catastrophic-loss escape hatch, NOT the
     normal disengage path.

   Either release path emits the same `override_released` JSON event,
   which now carries an optional `release_pose` field with the
   operator's last commanded body + hand joints (`joint_pos_mj`,
   `left_hand_joints`, `right_hand_joints`). The bridge consumes
   the released edge, triggers a cold restart (clears ramp / LPF /
   chunk-blend state, pins the chunk-id baseline so any in-flight
   pre-override chunk is discarded), seeds the chunk blend's "from"
   anchors from the operator pose, and holds the wire **at the
   operator's last commanded pose** for `--vla-cold-restart-hold-ticks`
   ticks (default 25 = 500 ms). Falls back to x2_debug's measured
   pose only if the proxy didn't ship a payload (legacy proxy or
   smoke test with `--override-engage-motion-ticks 0`).
6. The next freshly decoded chunk ramps in from the operator's
   hand-off pose. VLA continues autonomously.

### What to look for in the logs

Proxy (with `--teleop-mode-port 5564` — the canonical 2026-06-10
follow-up path):

```text
[pose_proxy] teleop_mode SUB: tcp://127.0.0.1:5564 topic='stream_mode' (stale_ms=1000) -- STRICT mode-gated engage (motion-hysteresis bypassed)
[pose_proxy] tick=2401 state=LIVE mode=blend upstream_age=20ms override(active=False age=2ms fwd=0 eng=0 rel=0) gate(mode=OFF age=18ms msgs=1183 fail=0 OFF)
[pose_proxy] state: LIVE -> OVERRIDE (operator teleop override engaged; forwarding override port frames verbatim)
[pose_proxy] tick=2440 state=OVERRIDE mode=blend upstream_age=20ms override(active=True age=2ms fwd=39 eng=1 rel=0) gate(mode=ARM_MANIPULATION age=18ms msgs=1222 fail=0)
[pose_proxy] state: OVERRIDE -> LIVE (upstream pose frames flowing again after 0 ms gap)
[pose_proxy] tick=2490 state=LIVE mode=blend upstream_age=20ms override(active=False age=2ms fwd=89 eng=1 rel=1) gate(mode=OFF age=18ms msgs=1272 fail=0 OFF)
```

The `gate(mode=…)` field on the status line is the smoking gun for
mode-gated engagement: `mode=OFF` ⇒ proxy refuses to engage,
`mode=ARM_MANIPULATION`/`LOCOMOTION` ⇒ proxy engaged. If you see
`STALE` in the tag (e.g. `gate(mode=OFF age=2400ms msgs=42 fail=0 STALE)`)
the manager has gone silent for `> --teleop-mode-stale-ms` and the
proxy is refusing to engage — fail-closed.

Legacy path (no `--teleop-mode-port`) instead emits the frame-equality
detector lines:

```text
[pose_proxy] override frozen detected (streak=10 ticks >= threshold=10, L2=0.000000 <= tol=0.0001); forcing release without waiting for SUB silence
```

If you see `gate(...)` instead of `frozen(...)`/`moving(...)` the
strict mode path is active. Only fall back to the frozen detector
for replay smoke tests where no manager is running.

Bridge:

```text
[live-VLA] vla_control: override_engaged (proxy_ts=12345.678) -- pausing decoded chunks, holding measured pose on wire
[live-VLA] vla_control: override_released (proxy_ts=12347.890) -- cold restart armed; clearing ramp / LPF / chunk state on next tick; release_pose has joint_pos_mj+left_hand_joints+right_hand_joints; bridge will hold at operator pose
[live-VLA] cold-restart fired tick=11234 baseline_chunk=47 hold_ticks=25; will hold wire at operator's last commanded pose (body+left_hand+right_hand) before any decoded chunk re-engages
```

If the bridge log instead shows `release_pose: no release_pose; bridge
will hold at x2_debug measured pose (legacy)` followed by `will hold
wire at x2_debug measured pose (legacy fallback)`, the proxy didn't
ship a release_pose payload — either you're running an older proxy
binary or the override-port chain is mis-wired so the proxy never
saw the operator's frames. Either case will cause the visible
"pose reset" the engage hysteresis was added to address; check that
the recorder is publishing on the proxy's `--override-port` and that
the proxy reports `eng >= 1` in its status line.

### Disabling without uninstalling

Drop `--enable-takeover` from `run_x2_vla_runtime.sh` (or pass
`--no-takeover` to explicitly negate a sticky env var). With takeover
off, the bridge re-binds the canonical `:5556` directly, the mux is
not spawned, and the runtime degrades to the pre-2026-06-11
autonomous-only topology. PC2's watchdog SUBs the same
`LAPTOP:5556` URL either way and forwards bytes verbatim — no PC2
restart is needed when you flip takeover on or off.

### Sim-only variant (no PC2, no real robot)

In sim mode (`--pc2-host` omitted) the same `--enable-takeover`
flag spawns BOTH the laptop-side `x2_pose_mux` AND a loopback
`x2_pose_watchdog` so the wire topology mirrors the real-robot
diagram with `127.0.0.1` everywhere: `bridge :5571 → mux :5556 →
watchdog :5558 → sim deploy`. Legacy autonomous-only sim runs
(without `--enable-takeover`) are byte-for-byte unchanged from the
pre-takeover behaviour: the bridge binds `:5556` directly and the
sim deploy SUBs from there.

**Terminal A — VLA runtime + sim deploy + sim mux + sim watchdog (one command):**

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --enable-takeover \
    --vla-control-port 5559 \
    --pose-proxy-override-port 5560 \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --robocasa-env X2PickPlaceCube \
    --prompt "pick up the red cube and drop it into the blue bowl"
```

(The MuJoCo passive viewer + OmniHand compose are sim defaults. Add
`--no-sim-viewer` for headless runs or `--no-sim-with-omnihand` to fall
back to the bare X2 stub fingers. When `--sim-with-omnihand` is on
AND `--enable-takeover` is set, the launcher also redirects the
OmniHand ZMQ subscriber from its default `localhost:5556` to the
sim watchdog's downstream port — so operator finger commands ride
the same arbitrated wire as the body joints. Without this redirect
the OmniHand SUB would stay pinned to the bridge-then-mux output at
`:5556` and always show the merged wire fingers regardless of
override engagement.)

**Wrist bypass (OFF by default — default flipped to `ik` and reverted on 2026-06-10
evening).** The launcher defaults `--wrist-bypass off`. This means the
SONIC tracker drives the wrist actuators directly, which per
`wrist_bypass.hpp` pins `*_wrist_pitch` at a near-static comfort pose
and `*_wrist_roll` at the asymmetric joint-range tight side
**regardless** of what the IK reference (operator command, VLA chunk)
asks for. This is intentional after the 12:26 incident: setting
`--wrist-bypass ik` makes the deploy force-write
`target_pos_mj[{20,21,27,28}]` to the wire's `joint_pos_mj` BEFORE the
safety stack, but the launcher's default wire content (idle_stand clip
wrists at startup, VLA chunk wrists during inference) is ~1.8 rad
(~103°) away from SONIC's natural pinned pose. Without
`--max-target-dev` configured (launcher default: empty), the only
per-tick rate limit is the soft-start blend, and you'll see a wrist
swing on startup that the operator on 2026-06-10 12:26 correctly
reported as "the hand slammed into the table".

If you want the wrist to actually respond to the operator's VR wrist
gesture (or VLA wrist tokens), pass BOTH flags together:

```
--wrist-bypass ik \
  --deploy-extra-arg --max-target-dev --deploy-extra-arg 0.05
```

(0.05 rad/tick at 50 Hz = 2.5 rad/s ≈ 1.3 s for a full wrist swing —
aggressive enough to keep up with hand gestures, slow enough to not
slam.) The launcher banner shows DANGER text when `ik` is selected so
you can't miss it. See the 2026-06-10 milestone document
(`docs/source/user_guide/milestones/2026-06-10_vla_manual_takeover.md`,
follow-up 4) for the full postmortem and the path back to a safe `ik`
default.

**Smooth handoff guard (ON by default since 2026-06-10 evening).** When
override releases (mode → OFF), the bridge does NOT immediately ramp
back to the idle clip. Instead it holds the wire at the operator's
hand-off pose for a MINIMUM of `--vla-cold-restart-hold-ticks` (default
25 = 500 ms, bridges the proxy's HOLD ladder seamlessly) and a MAXIMUM
of `--vla-handoff-max-hold-ticks` (default 200 = 4 s, safety cap). In
between, the hold releases as soon as the first **eligible** chunk
arrives (`chunk_id > cold_restart_baseline`, i.e. a chunk the model
decoded AFTER the operator released the wire). When that first chunk
fires, the ramp + body LPF + both hand LPFs are seeded from the
operator's hand-off pose — so the wire interpolates `operator →
decoded` over `--vla-ramp-in-ticks` (default 75 = 1.5 s) without an
idle-clip detour. Watch the bridge log for:

```text
[live-VLA] cold-restart fired tick=… min_hold_ticks=25 max_hold_ticks=200; will hold wire at operator's last commanded pose (body+left_hand+right_hand) until first eligible chunk (chunk_id > K) decodes, capped by max-hold safety
[live-VLA] cold-restart handoff: first eligible chunk decoded (chunk_id=K+1 > baseline=K); releasing wire hold at tick=…, ramping into VLA from operator pose
[live-VLA] cold-restart handoff: ramp + LPF seeded from operator pose (ramp_ticks=75); VLA wire re-engaging from hand-off pose without idle-clip detour
```

If you ever see `WARNING: cold-restart handoff safety cap reached`, the
decoder is wedged (stuck inference, proprio starvation, or only
zero-token chunks). The wire releases to idle to avoid sitting at the
operator pose forever; investigate the inference thread.

Tuning: bump `VLA_HANDOFF_MAX_HOLD_TICKS=400` (or pass
`--vla-handoff-max-hold-ticks 400`) if you have a slower inference
setup. Setting it to `0` disables the guard entirely and reverts to
the legacy 2026-06-10 behaviour (snap to idle on minimum-hold expiry;
not recommended).

**Finger telemetry.** The recorder's subscribe-mode status line now
shows `hand|L|=N.NNN(manager|zero-fallback) hand|R|=…` so you can tell
"manager publishes zero hand_q because the operator hasn't pulled
triggers" apart from "the wire is broken". If you see
`hand|L|=0.000(manager)` while in ARM_MANIPULATION, **the operator
isn't pulling the triggers** — the manager's retargeter only emits
non-zero finger curls in response to the VR controller's analog
triggers. If you see `hand|L|=0.000(zero-fallback)`, the manager hasn't
yet published `hand_finger_cmd` (still in startup, or you haven't
entered ARM_MANIPULATION yet — `arm_targets` and `hand_finger_cmd` only
flow once you've engaged arm IK via the A button after B → ARM_MAN).

Engage / release tuning (all optional; defaults are tuned for the
ARM_MANIPULATION ↔ OFF gesture workflow).

**Strict mode-gated engagement (canonical 2026-06-10 follow-up;
ON by default in sim):**
- `--pose-proxy-teleop-mode-port PORT` — port of the Quest3 manager's
  `stream_mode` PUB (default `5564`, matches the manager's
  `--recorder-pub-port`). When > 0 the proxy uses the operator's
  button presses as the engagement source of truth and **bypasses**
  motion-hysteresis entirely. Set to `-1` to fall back to the legacy
  motion-hysteresis path (only useful for replay smoke tests where
  no manager is running — will flicker if operator holds controller
  still in ARM_MANIPULATION).
- `--pose-proxy-teleop-mode-host HOST` — manager host (default
  `127.0.0.1`). Set to the laptop's address when the manager and
  proxy are on different machines.
- `--pose-proxy-teleop-mode-topic TOPIC` — ZMQ topic prefix
  (default `stream_mode`; matches the manager's
  `--stream-mode-topic`).
- `--pose-proxy-teleop-mode-stale-ms MS` — fail-closed window
  (default 1000 = ~50 manager ticks @ 50 Hz). When the mode signal
  has been silent longer than this, **engagement is BLOCKED** — a
  dead manager hands the wire back to VLA within ~1 s.

**Legacy motion-hysteresis flags (ignored when
`--pose-proxy-teleop-mode-port > 0`):**
- `--pose-proxy-override-engage-motion-ticks N` — symmetric engage
  hysteresis. Require N consecutive moving frames before engage
  (default 10 = 200 ms @ 50 Hz, mirrors `--override-frozen-ticks`).
  Set to 0 for the legacy single-frame engage.
- `--pose-proxy-override-frozen-l2-tol R` — joint-space tolerance
  for "frozen" frames (default `5e-3` rad ≈ 0.3° total motion;
  bumped from `1e-4` on 2026-06-10).
- `--pose-proxy-override-frozen-ticks N` — number of consecutive
  frozen frames before fire `override_released` (default 10).

Available robocasa scenes (built into `gear_sonic/data/assets/robocasa_scenes/`):
`X2PickPlaceCube` (cube → bowl, canonical), `X2PickPlaceApple` (apple → bowl),
`X2PickPlaceBowl` (bowl → target). None match the soda-can training data, so
the VLA's task success will be poor — that's expected and **fine for testing
the takeover loop**: we only need graspable-object chunks flowing so the
operator can engage teleop on top. To add a new scene, see
`gear_sonic/scripts/build_x2_robocasa_scene_xml.py`'s `_KNOWN_ENVS` registry.

(Env-var form `ENABLE_TAKEOVER=1 VLA_CONTROL_PORT=5559 POSE_PROXY_OVERRIDE_PORT=5560 ./run_x2_vla_runtime.sh ...` still works as a fallback for tmux-env propagation; the CLI flags take precedence when both are set.)

(Omit `--pc2-host` → SIM mode. With `--enable-takeover`, the launcher
spawns `bridge :5571 → mux :5556 → loopback watchdog :5558 → sim
deploy`. The banner shows
`Sim pose pipeline: ON (loopback) bridge :5571 -> mux *:5556 -> watchdog *:5558 -> deploy`
when it took effect; check `${RUN_DIR}/pose_mux.log` and
`${RUN_DIR}/pose_watchdog.log` for the per-process confirmation
lines.)

**Terminal B — Quest 3 teleop without spawning a competing sim deploy:**

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --no-deploy \
    --takeover \
    --no-x2-debug-bridge
```

Why each flag is required:
- `--no-deploy` → terminal A already owns the sim deploy on `:5558` SUB; without this we'd spawn a competing one.
- `--takeover` (= `--pose-port 5560`) → recorder PUBs the operator
  wire to the mux's override SUB instead of fighting the merged
  wire for `:5556`. Use `--takeover-port PORT` to override the
  default `5560`.
- `--no-x2-debug-bridge` → the sim deploy's `x2_mujoco_ros_bridge.py`
  already binds `:5570` for `robot_pose` natively. Spawning the
  planner stack's `x2_debug → robot_pose` bridge would clash on
  `:5570` (`Address already in use`). The kplanner's pose-feedback
  SUB still connects to `:5570` (default) and gets `robot_pose`
  from the sim deploy directly.

(On the **real robot**, replace `--no-x2-debug-bridge` with
`--x2-debug-bridge-host <PC2_IP>` because the C++ deploy on PC2 does
NOT run the MuJoCo Python bridge; the planner stack's
`x2_debug → robot_pose` translator IS needed there.)

The recorder PUBs the operator's wire on `tcp://localhost:5560`, the
loopback proxy SUBs there, arbitrates against the bridge's :5556, and
emits the `vla_control` events the bridge cold-restarts on. Operator
workflow + log markers are identical to the real-robot section above
(just `127.0.0.1` instead of `192.168.86.32`).

Tear-down: Ctrl-C in terminal B (recorder saves the episode), then
Ctrl-C in terminal A — `stop_all` kills the sim proxy after the sim
deploy is down so the loopback ports come back cleanly for the next
run. `kill_stale_sim_processes` also `fuser -k`'s 5558/5559/5560 if
anything leaks.

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