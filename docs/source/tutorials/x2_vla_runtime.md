# X2 VLA runtime — autonomous Isaac-GR00T VLA (sim + real robot)

Autonomous closed-loop deployment of a fine-tuned Isaac-GR00T VLA.
One launcher — ``gear_sonic/scripts/run_x2_vla_runtime.sh`` — same
pattern as ``run_x2_quest3_planner_stack.sh``:

* **Sim (default):** omit ``--pc2-host``. Spawns the bridge + local
  ``deploy_x2.sh sim --vla`` with MuJoCo ghost cameras. Debug here
  before powering the real robot.
* **Real robot:** pass ``--pc2-host <PC2_IP>``. Spawns only the bridge;
  assumes ``x2_pc2_daemons.sh start`` is already running on PC2 with
  the camera ZMQ bridge live.

> ⚠️ **Real hardware.** This runbook controls a powered humanoid via a
> neural policy. Treat every line as load-bearing; do **not** skip the
> safety checklist below.

For the engineering reference (ZMQ port catalogue, end-to-end topology,
encoder/decoder contracts) see:

- [`X2 SONIC Runtime Architecture`](../references/x2_sonic_runtime_architecture.md)
  — the cross-mode overview that shows how this runbook's laptop process
  slots into the same PC2 daemons that drive teleop and recording.
- [`X2 Quest 3 Planner Stack — System Architecture`](../references/x2_quest3_planner_stack_architecture.md)
  — the parent system reference; the VLA topology is a strict subset
  (no VR, no recorder, no planner).
- [`X2 Split-topology PC2 daemons`](../references/x2_split_deploy_pc2.md)
  — what `x2_pc2_daemons.sh` brings up on the robot.
- [`X2 VLA motion_token Decoder`](../references/x2_vla_motion_token_decoder.md)
  — why the laptop bridge needs the SONIC `.pt` checkpoint (the
  `--motion-token-decoder` flag) or the body will stay at idle.
- [`VLA Training`](vla_training.md) — how the fine-tuned checkpoint
  this runbook deploys was trained.

---

## 60-second TL;DR

Same `x2_pc2_daemons.sh start` you already run for teleop / recording
keeps SONIC alive on PC2. Switch laptop-side scripts to flip modes:

| Mode             | Laptop-side command                                                       |
| ---------------- | -------------------------------------------------------------------------- |
| Teleop only      | `run_x2_quest3_planner_stack.sh --pc2-host …`                            |
| Recording        | `run_x2_quest3_planner_stack.sh --with-record --head-cameras --pc2-host …` |
| Autonomous VLA   | `run_x2_vla_runtime.sh` (this runbook)                        |

```bash
# Sim first (omit --pc2-host) — safe debugging in MuJoCo viewer
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"

# Real robot (after sim looks sane)
# 1. PC2 daemons (idempotent; usually already running from your last session)
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start

# 2. Single command on the laptop. Preflight + camera auto-start + 5 s
#    safety banner + bounded 30 s live run, all in one.
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"
```

> The legacy flag name `--sonic-checkpoint` (env: `SONIC_CHECKPOINT`)
> is still accepted as a deprecated alias and prints a one-shot
> warning. Migrate scripts to `--motion-token-decoder` (env:
> `MOTION_TOKEN_DECODER`) when convenient.

What the launcher does in that single call:

1. **Preflight** — ping PC2, probe `x2_debug` + cameras over ZMQ,
   verify model + modality + motion-token decoder + bridge python are
   all in place, confirm local pose-PUB port is free.
2. **Camera auto-start** — if the cameras ZMQ stream is silent,
   auto-SSH to PC2 and run `x2_pc2_cameras.sh serve`. Mirrors the
   recording flow's `--head-cameras` auto-start. Usually a no-op
   since recording left the bridge up.
3. **Red safety banner** — 5 second countdown showing the resolved
   model / prompt / ports / safety knobs. Ctrl-C here aborts before
   any pose target leaves the laptop.
4. **Live VLA bridge** — bounded by `MAX_DURATION` (default 30 s),
   logs streamed to stdout, chunks dumped to `${RUN_DIR}/vla_chunks/`
   for postmortem. Ctrl-C → SIGTERM → 10 s graceful → SIGKILL.

---

## Topology

```text
        +-----------------+         tcp://*:5556 (pose)
        |   LAPTOP        | ─────────────────────────────────────►
        |  this bridge    |          (PC2 pose proxy SUBs here)
        |  - live_vla_..  |                       │
        |  - SUB x2_debug |◄── tcp://PC2:5557 ◄───┤
        |  - SUB cameras  |◄── tcp://PC2:5555 ◄───┤
        +-----------------+                       │
                                                  ▼
                                      +-------------------+
                                      |   PC2 (Orin NX)   |
                                      | x2_pc2_daemons.sh |
                                      |  - pose proxy 5558|
                                      |  - deploy --vla   |
                                      |  - hand bridge    |
                                      |  - motor monitor  |
                                      | x2_pc2_cameras.sh |
                                      |  - ZMQ PUB 5555   |
                                      +-------------------+
```

Key ports (defaults; override via env or flags):

| Endpoint                          | Host    | Port | Bind/Connect | Notes                                                      |
| --------------------------------- | ------- | ---- | ------------ | ---------------------------------------------------------- |
| Bridge `pose` PUB                 | laptop  | 5556 | bind `*`     | PC2 pose proxy SUBs from `tcp://${LAPTOP_HOST}:5556`      |
| PC2 pose proxy → deploy           | PC2     | 5558 | localhost    | deploy SUBs `tcp://localhost:5558`                         |
| PC2 deploy `x2_debug` PUB         | PC2     | 5557 | `0.0.0.0`    | bridge SUBs `tcp://PC2:5557`                              |
| PC2 camera bridge ZMQ PUB         | PC2     | 5555 | `0.0.0.0`    | bridge SUBs `tcp://PC2:5555` (msgpack ImageMessageSchema) |
| Laptop manager `pose_resume` PUB  | laptop  | 5566 | bind `*`     | *unused in VLA mode* — kept reserved for SAFE_IDLE recovery |

---

## Prerequisites

### 1. Hardware

- Wired SDK LAN (laptop on `10.0.1.50/24`, PC2 on `10.0.1.41`). WiFi
  works but adds 80 ms+ jitter on top of the 50 Hz pose loop; the
  pose proxy on PC2 will mask short drops but you'll feel any sustained
  glitching as a sluggish policy response.
- Robot gantry strap engaged at ≥ 80 % body support. The first powered
  VLA run **will** spike target deltas during the cold-start (~3 s);
  the band catches the drift before the legs do. Once you have one
  bounded run end-to-end you can crank the band down for the next.
- E-stop on the SDK MC reachable. The launcher does NOT touch MC; it
  publishes pose targets only. The deploy's RAMP_OUT path is the only
  motion-side abort.

### 2. PC2 daemons up

```bash
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start
```

This spawns four tmux sessions on PC2:

- `x2_pose_proxy` — SUBs to your laptop's `tcp://${LAPTOP_HOST}:5556`,
  re-PUBs to `tcp://localhost:5558`, falls back to a baked idle-stand
  `.x2m2` clip if your laptop goes silent (so the deploy never starves).
- `x2_deploy` — `agi_x2_deploy_onnx_ref` in `--vla` mode. SUBs
  `tcp://localhost:5558`, PUBs `x2_debug` on `tcp://0.0.0.0:5557`,
  optionally SUBs `pose_resume` on `tcp://${LAPTOP_HOST}:5566` for
  SAFE_IDLE recovery.
- `x2_hand_bridge` — translates the laptop's `hand_finger_cmd` ZMQ
  stream into `/aima/hal/joint/{left_hand,right_hand}/command` ROS
  topics. The VLA bridge does not publish on this stream; the hand
  joints come over the same `pose` topic via `left_hand_joints` /
  `right_hand_joints` payload fields.
- `x2_motor_monitor` — continuous motor-state recorder (writes
  `motor_monitor_*.jsonl` on PC2 and PUBs a compact summary on
  `tcp://0.0.0.0:5567`).

Verify with:

```bash
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh status
```

All four entries should be `RUNNING`. Tail any one with
`./gear_sonic_deploy/scripts/x2_pc2_daemons.sh logs deploy` (`deploy`
is the noisiest and the one to watch when troubleshooting).

> **The deploy waits at a Y/n safety gate** inside its tmux pane.
> Attach with `./x2_pc2_daemons.sh attach deploy` to type `y`, OR pass
> `--no-confirm` when starting daemons (only when you're already
> comfortable with the bring-up). The deploy then sits in `STANDBY`
> until the start-trigger sentinel fires (handled automatically by the
> daemons wrapper).

### 3. PC2 cameras up — **auto-started by the launcher**

You do **not** need to manually call `x2_pc2_cameras.sh serve` for
this runbook. The launcher's preflight probes
`tcp://${PC2_HOST}:5555` and, if silent, auto-SSHes to PC2 and runs:

```bash
./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve --host $PC2_HOST
```

This mirrors what `run_x2_quest3_planner_stack.sh --head-cameras` does
during recording, and the camera bridge is deliberately left running
across recorder / VLA sessions (see the cleanup section at the bottom
of `pick_place_commands.md`) — so in practice this auto-start is a
no-op except on the very first run after a PC2 reboot.

The bridge subscribes to:

- `head_front` — Orbbec Gemini 335 RGB (USB 3)
- `stereo_left` / `stereo_right` — Sony IMX900 GMSL stereo head

resized to 640×480 at the source and republished as a merged
`ImageMessageSchema` on `tcp://*:5555` (msgpack-encoded). The
`omnihand_stereo` modality config in this repo expects exactly
`stereo_left` + `stereo_right`.

Manual ops (only if you want to inspect or manage the bridge
directly):

```bash
./gear_sonic_deploy/scripts/x2_pc2_cameras.sh status      # which topics have publishers
./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-log   # tail the bridge log on PC2
./gear_sonic_deploy/scripts/x2_pc2_cameras.sh grab        # download one JPEG per cam
./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-stop  # tear down (end-of-day)
```

If you re-flash PC2 or `status` shows one of the IMX900 stereo topics
missing, run `restart-hal` to bounce `hal_sensor_orin` and re-arm the
Argus subscribers, *then* re-run the launcher (which will re-trigger
the camera serve auto-start).

If you want to manage the bridge entirely by hand, disable the
auto-start with `--no-cameras-autostart` (or `CAMERAS_AUTOSTART=0`).

### 4. Model checkpoint + SONIC decoder on the laptop

The launcher needs **two** artifacts:

| Artifact                          | Default lookup                                          | What it does                          |
| --------------------------------- | ------------------------------------------------------- | ------------------------------------- |
| `MODEL_DIR`                       | *(required, no default)*                                | Fine-tuned GR00T-N1.7 checkpoint (`model.safetensors` + `experiment_cfg/` + `processor/`) |
| `MOTION_TOKEN_DECODER`            | Auto-resolved from siblings of `MODEL_DIR`              | Bridge-side SONIC `g1_dyn` decoder `.pt`; decodes the VLA's predicted motion_token chunk back into a body-pose trajectory so the body actually moves. Same `.pt` the recorder loads as an encoder during data capture — see [`X2 SONIC Runtime Architecture`](../references/x2_sonic_runtime_architecture.md) section 6. |

If `--motion-token-decoder` is empty, the launcher prints a loud
warning and runs anyway — but the deploy's C++ side **ignores** the
wire's `motion_token` field (see
[`X2 VLA motion_token Decoder`](../references/x2_vla_motion_token_decoder.md))
so only the OmniHand fingers will move under VLA authority and the body
will stay at idle stand. For the canonical grab-a-drink demo, point
`--motion-token-decoder` at the 25k pretrain that pairs with the
deploy ONNX currently running on PC2:

```bash
--motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

The legacy flag `--sonic-checkpoint` (env: `SONIC_CHECKPOINT`) is
still accepted; it prints a one-shot deprecation warning and forwards
its value as the decoder path.

### 5. Bridge python env

The launcher defaults to
`~/miniconda3/envs/env_isaaclab/bin/python` because the base `.venv`
torch (2.6+cu124) crashes on Blackwell GPUs (sm_120); `env_isaaclab`
ships torch 2.7+cu128 which works on the RTX 5090. Override with
`--bridge-py /path/to/python` when running on a different GPU stack.

---

## Bring-up sequence

Two steps. The launcher folds preflight, camera auto-start, safety
banner, and live run into a single call.

```bash
# Step 1 — PC2 daemons (idempotent; safe to re-run if already up).
# Skip this step if SONIC is already running from a prior teleop /
# recording session in this boot.
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh status   # all four RUNNING

# Step 2 — single command on the laptop.
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"

# Ctrl-C to abort cleanly OR wait for MAX_DURATION to elapse.
# The launcher SIGTERMs the bridge, gives it 10 s for RAMP_OUT, then
# SIGKILLs. The deploy on PC2 catches the silent wire and falls back
# to the baked idle-stand via the pose proxy.
```

Diagnostic-only sub-commands (you'll rarely need these):

```bash
# Dry-run the preflight without launching the bridge. Side-effect:
# may auto-start the PC2 camera bridge if it's silent.
./gear_sonic/scripts/run_x2_vla_runtime.sh preflight

# Silent-wire smoke test (no policy load; useful after a checkpoint
# swap to confirm the wire format hasn't drifted).
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --skip-preflight --max-duration 10 -- --no-policy

# Explicit teardown when the launcher's tail / Ctrl-C path got
# orphaned. Normally Ctrl-C handles this.
./gear_sonic/scripts/run_x2_vla_runtime.sh stop \
    --run-dir /tmp/x2_vla_runtime-YYYYMMDD_HHMMSS
```

The very first live run prints a **red safety banner** with the
resolved configuration and waits 5 s before going live (override with
`FAST_ABORT=1`). Take that window to glance at:

- The robot's posture (should be the trained idle stand).
- The gantry slack (≥ 80 % body support).
- PC2 deploy log (`x2_pc2_daemons.sh logs deploy`) — last line
  should be a steady `setpoint` print every ~100 ms.

If anything looks off, Ctrl-C aborts before any pose target leaves the
laptop.

---

## Safety checklist

Print this and tick before every powered run.

- [ ] Gantry strap engaged, ≥ 80 % body support, gantry rails clear.
- [ ] E-stop within arm's reach of the operator.
- [ ] PC2 daemons all four RUNNING (`x2_pc2_daemons.sh status`).
- [ ] PC2 cameras log shows all three keys received per second
      (`x2_pc2_cameras.sh serve-log` — look for `head_front=15.0Hz, stereo_left=15.0Hz, stereo_right=15.0Hz`).
- [ ] `run_x2_vla_runtime.sh preflight` returned 0.
- [ ] First bring-up of the day → `MAX_DURATION=30` (default).
- [ ] No one in the robot's reach radius (sweep + manipulation cone of
      ~1.2 m forward of the chest plate).
- [ ] Bench objects in their trained positions (the model was trained
      on a fixed table layout; expect OOD chaos at unfamiliar offsets).
- [ ] Laptop on AC power (the bridge holds the GPU at near 100 %
      utilisation during inference; battery throttling drops inference
      to 1-2 Hz and the deploy's pose-ref watchdog will SAFE_IDLE).

---

## Iteration knobs

All knobs are CLI flags. Matching env var names (e.g. `MODEL_DIR`,
`PROMPT`) are accepted as fallbacks if the flag is omitted — handy for
shell aliases or wrapper scripts — but the flag form is the canonical
operator interface.

| Flag                          | Env fallback             | Default                                                         | When to tune                                                          |
| ----------------------------- | ------------------------ | --------------------------------------------------------------- | --------------------------------------------------------------------- |
| `--model PATH`                | `MODEL_DIR`              | *(required)*                                                    | Switch checkpoints to A-B test fine-tunes.                            |
| `--prompt STR`                | `PROMPT`                 | `"grab a drink"`                                                | Exactly the training prompt is safest; novel prompts are OOD.         |
| `--pc2-host HOST`             | `PC2_HOST`               | `10.0.1.41`                                                     | WiFi address differs; check `x2_pc2_daemons.sh print-env`.            |
| `--motion-token-decoder PATH` | `MOTION_TOKEN_DECODER`   | Auto-resolved (siblings of `--model`, then 25k canonical)       | Body locomotion check — empty = body stays at idle. Legacy alias: `--sonic-checkpoint` / `SONIC_CHECKPOINT` (deprecated). |
| `--max-duration SEC`          | `MAX_DURATION`           | `30`                                                            | Increase once you trust the policy + you can reach the E-stop in 1 s. |
| `--inference-min-period-s S`  | `INFERENCE_MIN_PERIOD_S` | `0.8`                                                           | Match the action chunk horizon (40 steps × 0.02 s). Drop only after profiling the GPU. |
| `--rate HZ`                   | `RATE`                   | `50`                                                            | Deploy expects 50 Hz; rarely changed.                                 |
| `--cameras-staleness-s S`     | `CAMERAS_STALENESS_S`    | `2.0`                                                           | The publisher runs at 15 Hz; 2 s = 30 missed frames. Drop only if you've measured the actual cam jitter. |
| `--run-dir DIR`               | `RUN_DIR`                | `/tmp/x2_vla_runtime-<timestamp>`                             | Pin the log + chunk dump location for postmortem.                     |
| `--bridge-py PATH`            | `BRIDGE_PY`              | `~/miniconda3/envs/env_isaaclab/bin/python`                     | Override on hosts where `env_isaaclab` lives elsewhere or the GPU needs a different torch. |
| `--modality-config PATH`      | `MODALITY_CONFIG`        | `gear_sonic/data/x2_modality_config_omnihand_stereo.py`         | Switch to `..._10dof.py` only if you trained against the single `ego_view` MuJoCo render (sim path; cameras-source ghost). |
| `--pc2-user NAME`             | `PC2_USER`               | `run`                                                           | The SSH user on PC2; rarely changed.                                  |
| `--skip-preflight`            | `SKIP_PREFLIGHT=1`       | off                                                             | Bypass connectivity probes. **Do not use on a powered robot.**       |
| `--fast-abort`                | `FAST_ABORT=1`           | off                                                             | Skip the 5 s safety banner countdown.                                 |
| `--no-cameras-autostart`      | `CAMERAS_AUTOSTART=0`    | off (auto-start enabled)                                        | Manage `x2_pc2_cameras.sh serve` manually instead of letting the launcher auto-SSH. |
| `--vla-ramp-in-ticks N` *(bridge passthrough)* | `VLA_RAMP_IN_TICKS` | `25` (= 0.5 s @ 50 Hz)                                    | Ticks over which the wire linearly interpolates from idle stand to the first decoded VLA pose on the idle→VLA transition. Pass `0` to disable (NOT recommended on a powered robot — see [`x2_vla_motion_token_decoder.md`](../references/x2_vla_motion_token_decoder.md) §9.1). |
| `--vla-target-lpf-hz HZ` *(bridge passthrough)* | `VLA_TARGET_LPF_HZ` | `8.0`                                                       | One-pole low-pass cutoff on the wire's current `joint_pos_mj` slot. Transparent for natural reach trajectories (<3 Hz spectral content); attenuates inter-chunk discontinuities. Pass `0` to disable. |

Less-common knobs (use `--help` for the full list): `--cameras-warmup-s`,
`SONIC_DECODER_DEVICE`, `DUMP_CHUNKS_EVERY`.

---

## Troubleshooting

### Preflight: PC2 ping fails

You're on the wrong network. Check `ip a` for a `10.0.1.50/24`
interface. If you're on WiFi, pass `--pc2-host <wifi_ip>` (find it
with `x2_pc2_daemons.sh print-env`).

### Preflight: x2_debug silent on tcp://PC2:5557

The deploy is not running on PC2 (or it's stuck at the Y/n gate). Run
`./gear_sonic_deploy/scripts/x2_pc2_daemons.sh status` then
`./gear_sonic_deploy/scripts/x2_pc2_daemons.sh logs deploy` to see the
last 50 lines. If the deploy is sitting at `Confirm start (Y/n):`,
attach the tmux session with `./x2_pc2_daemons.sh attach deploy` and
press `y`.

### Preflight: cameras silent on tcp://PC2:5555

Two common cases:

1. The ROS->ZMQ bridge isn't running. Run
   `./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve` then
   `serve-log` to confirm.
2. One of the cameras failed to initialise. Look for missing topics
   with `./gear_sonic_deploy/scripts/x2_pc2_cameras.sh status`; if the
   IMX900 stereo topics are absent, run `restart-hal` to bounce the
   Argus subscribers.

### Bridge dies during camera warm-up with "missing required keys"

The modality config expects `stereo_left` + `stereo_right` but the
publisher is only emitting `head_front`. Either:

- Re-run `x2_pc2_cameras.sh serve` (it picks up missing IMX900 cameras
  after a `restart-hal`), OR
- Switch the modality config to `..._10dof.py` (single `ego_view`) —
  but note this is a sim-only path; the publisher does not emit
  `ego_view` so this will fail too. The right fix is always to
  recover the stereo stream.

### Robot stays at idle stand, only fingers move

You're missing the motion-token decoder. Pass
`--motion-token-decoder /path/to/model_step_NNNNN.pt` (or set
`MOTION_TOKEN_DECODER`). The deploy's C++ side ignores the wire's
`motion_token` field by design; the laptop bridge has to decode the
motion_token back to a body-pose trajectory and ship that as
`joint_pos_mj` for the body to move. See
[`X2 VLA motion_token Decoder`](../references/x2_vla_motion_token_decoder.md).

### Inference period balloons to 2+ s, body twitches

GPU throttling. Plug into AC. If you're already on AC, check
`nvidia-smi dmon -s ucvm` for thermal throttling and lower
`INFERENCE_MIN_PERIOD_S` only after the GPU is stable.

### Body shakes / vibrates immediately when VLA takes over

Two root causes are common — the bridge logs distinguish them:

1. **Decoder fed bad proprio.** The bridge writes a `raw_Δ` and
   `wire_Δ` field per print tick once VLA is driving. If `raw_Δ`
   is consistently ≥0.3 rad regardless of prompt, the SONIC
   pose decoder is OOD. Confirm the `x2_debug` stream is
   carrying `body_dq` and `base_ang_vel` (sniff with
   `gear_sonic/scripts/dump_x2_debug.py --port 5557 --topic x2_debug`).
   The bridge falls back to zero proprio with a `WARN: live
   proprio assembly failed` if assembly throws, which surfaces
   in `bridge.log`.
2. **Ramp / LPF disabled.** Verify the launcher banner prints
   `Wire safety : ramp-in 25 ticks (= 0.50s), target LPF 8.0 Hz`.
   If you've passed `--vla-ramp-in-ticks 0` or
   `--vla-target-lpf-hz 0`, the wire passes the full decoded step
   through and the deploy will chase a 0.4+ rad jump on every
   joint at once. The 2026-06-07 powered run that motivated this
   safety net is documented in
   [`x2_vla_motion_token_decoder.md`](../references/x2_vla_motion_token_decoder.md)
   §9.1.

### Robot turns sharply toward "world yaw=0" on first VLA start

If the **first** VLA start of a SONIC session yanks the robot toward a
specific world heading (the orientation it had when SONIC was first
booted) — e.g. a robot started at -45° rotates right to neutral, or a
robot started at -180° spins nearly all the way around to neutral —
that's the **bootstrap yaw-reference race**. Three bridge-side fixes
landed 2026-06-07 that close it:

1. **Withhold pose publish until first `x2_debug` arrives.** Until the
   bridge has at least one measured `base_quat` sample, every
   `root_quat_xyzw` it could ship is the dataclass default = identity
   = world yaw=0. Publishing that to a deploy that's *already in
   CONTROL* (the PC2 SONIC daemon persists across teleop/VLA sessions)
   would lock the policy onto a phantom yaw=0 reference. The C++
   deploy has a bootstrap-safe path: when
   `ZmqPoseInputSource::LastReceivedMonotonicS() < 0` it substitutes
   the measured quat as the orientation reference and the policy
   holds whatever heading the body is in. The bridge now keeps that
   escape hatch active by not sending a phantom first frame.

2. **Live yaw-rebase on `root_quat_xyzw` + future window.** Once
   `x2_debug` is alive, the publisher overrides both the current and
   the 9-slot future `root_quat_xyzw` with `R_z(measured_yaw)` every
   tick. Mirrors `X2DatasetRecorder._compute_idle_root_quat_xyzw`,
   which already proved correct on the recorder path.

3. **Surgical `waist_yaw` (MJ slot 12) pin to measured during freeze.**
   In `manipulation` mode the bridge freezes `legs,waist` to the
   `idle_stand` clip for balance. But `idle_stand[0]`'s waist_yaw is
   ~33° off the trained default (see `_IdleStandLoop` docstring), and
   `waist_yaw_joint` is the dominant heading-correction effector — so
   freezing the whole group to the clip would drive a steady-state
   ~33° heading drift every run. The fix pins only `waist_yaw` to the
   live measured value while every other frozen DOF (`waist_pitch`,
   `waist_roll`, hip/knee/ankle) keeps the clip jitter the policy was
   trained on.

Bridge log markers to look for in `${RUN_DIR}/bridge.log`:

```text
[live-VLA] withholding pose publish until first x2_debug frame arrives ...
[live-VLA] first pose publish (tick=N); x2_debug seen, root_quat now tracks live heading.
[live-VLA] root_quat yaw-rebase ACTIVE: ... (yaw=+178.2deg)
```

If you ever see a sharp first-start turn again, grep for those three
lines — their absence (or reversed ordering) is the smoking gun.
Validate from multiple starting headings (e.g. -135°, +90°, ±180°) to
confirm the bridge is heading-agnostic on first publish.

Reference (why `waist_yaw` is special among the frozen DOFs): the
recorder side has the same machinery in
[`gear_sonic/utils/teleop/x2_dataset_recorder.py`](https://github.com/agibot-tech/GR00T-WholeBodyControl/blob/main/gear_sonic/utils/teleop/x2_dataset_recorder.py),
`_compute_idle_root_quat_xyzw` — the bridge port adds the
`waist_yaw`-only joint pin because the bridge ships
`joint_pos_mj` (which the recorder didn't have a freeze-to-clip
pathology to compensate for).

### Bridge prints "cameras stale (>2.0s old)" repeatedly

The PC2 camera bridge is alive but the publisher's `head_front` /
`stereo_*` ROS topics aren't getting frames. SSH to PC2 and check
`x2_pc2_cameras.sh serve-log` for `dropped_zmq` counts or decode
errors. Usually `restart-hal` fixes a missing IMX900.

### Deploy SAFE_IDLEs mid-run

The pose proxy fell back to the baked idle clip because the bridge
went silent for > 100 ms. Common causes:

- Laptop GPU throttled; inference period blew past 2 s.
- WiFi packet drop. Move to wired, or accept the proxy fallback as
  intended.
- Bridge process crashed. Check `${RUN_DIR}/bridge.log` for the
  traceback.

The deploy will *automatically* recover once the wire flows again
(no `pose_resume` needed in this VLA path; SAFE_IDLE here means
"tracker uses the baked idle frame", not "MC takeover").

### Postmortem: I want to see what the policy saw

Every run writes `${RUN_DIR}/vla_chunks/chunk_*.npz` containing both
the **observation** the policy received (body_q_mj, base_quat_wxyz,
hand state, every camera frame keyed by modality) and the **action**
it produced (motion_token, left/right hand joints). Inspect with:

```bash
python -c "
import numpy as np
d = np.load('/tmp/x2_vla_runtime-.../vla_chunks/chunk_00000.npz')
print(list(d.keys()))
print('cameras:', [k for k in d.keys() if k.startswith('view_')])
print('token  :', d['token'].shape, 'norm', np.linalg.norm(d['token'][0]))
"
```

Or scrub a sequence visually:

```bash
.venv/bin/python -m gear_sonic.scripts.inspect_vla_chunks /tmp/x2_vla_runtime-.../vla_chunks
```

---

## Stop / cleanup

```bash
# Graceful: SIGTERM + 10 s wait + SIGKILL if needed
./gear_sonic/scripts/run_x2_vla_runtime.sh stop \
    --run-dir /tmp/x2_vla_runtime-YYYYMMDD_HHMMSS
```

The launcher's foreground process also handles Ctrl-C the same way.

The PC2 daemons keep running across bridge runs — that's intentional;
stopping the deploy means restarting the MC handshake. Stop them
explicitly only when you're done for the session:

```bash
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop
./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-stop
```

---

## Related runbooks

- [`X2 Dataset Record and Replay`](x2_dataset_record_and_replay.md) —
  capturing the demonstrations the policy was trained on.
- [`X2 Head Cameras Recording`](x2_head_cameras_recording.md) — the
  same PC2 camera publisher this runbook subscribes to.
- [`VLA Training`](vla_training.md) — fine-tuning a checkpoint for
  this runbook to deploy.
- [`X2 Quest 3 Planner Stack Cheat Sheet`](x2_quest3_planner_stack_cheatsheet.md)
  — the operator cheat sheet for the upstream teleop / recording stack
  this runbook is the autonomous counterpart of.
