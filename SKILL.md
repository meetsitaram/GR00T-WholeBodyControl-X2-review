---
name: x2-sonic-stack
description: Set up and operate the AgiBot X2 SONIC whole-body-control stacks — headless policy eval, gamepad-driven sim, VR teleop in sim, and real-robot deployment
---

# X2 SONIC stack — operator skill

Everything needed to go from a fresh clone to X2 walking in MuJoCo sim, and
from sim to the real robot. Written to be followed top-to-bottom by a human
or an agent; each stage is independently verifiable.

## 1. Install (once)

One command from a fresh clone (see `SETUP.md` for the validated details):

```bash
git clone <this-repo> && cd <repo> && bash install_scripts/setup_x2.sh
```

The script is idempotent (safe to re-run) and does, in order: `git lfs pull`,
`python3.10` venv at `.venv/`, the validated pip sequence (CPU torch wheel
first so pip never pulls the multi-GB CUDA wheel, then `gear_sonic[sim]`,
`onnxruntime`, `motionbricks`, `huggingface_hub[cli]`), the X2 model download
into the model cache (skipped when already present), and an import + model
verification block. Add `--with-docker` to also build the
`gear_sonic_deploy/docker_x2` sim deploy image (needed only for the full
deploy loop, 3c/3d — not for headless eval). The deploy build needs
AgiBot's `aimdk_msgs` interface package — `setup_x2.sh --with-docker`
fetches it automatically from AgiBot's official SDK artifact
(`AIMDK_SDK_URL` to override; manual fallback in
`gear_sonic_deploy/thirdparty/aimdk_msgs/README.md`).

GPU torch is only needed for training; every runtime below works on CPU.

After install, activate the venv once per shell — every `python`/`bash`
command below assumes it (or prefix commands with `.venv/bin/`):

```bash
source .venv/bin/activate
```

## 2. Models — fixed cache location, found automatically

Weights are NOT in this repo. They live in the **SONIC model cache**:

```
$SONIC_HOME/                    # default ~/.cache/sonic (env-overridable)
├── x2/                         # X2 artifacts (this repo's stacks)
│   ├── sonic_policy/x2_sonic_policy.onnx        # fused SONIC tracking policy (deploy)
│   ├── sonic_policy/x2_sonic_policy.pt          # source ckpt (parity checks / finetune)
│   ├── kplanner_onnx/x2_kplanner_template.onnx  # torch-free planner graph (template mode)
│   ├── kplanner_onnx/x2_kplanner_velocity.onnx  # legacy velocity-mode graph
│   └── kplanner_torch/{vqvae,pose,root}/...     # planner finetuning tier (+ x2_clip.ckpt)
└── g1/                         # G1 artifacts (nvidia/GEAR-SONIC), same downloader
```

`setup_x2.sh` populates `x2/` for you. One download entrypoint serves every
embodiment (future ones — e.g. asimov — slot in as `$SONIC_HOME/asimov`):

```bash
.venv/bin/python download_from_hf.py --robot x2    # tinkerbuggy/sonic-x2 -> $SONIC_HOME/x2
.venv/bin/python download_from_hf.py --robot g1    # nvidia/GEAR-SONIC   -> $SONIC_HOME/g1
```

The X2 HF repo (`tinkerbuggy/sonic-x2`) is private during review; public at
release — until then request access and `hf auth login`. Every deployment
artifact there is md5-verified byte-identical to the copies running on the
reference robot.

**The stacks find models in the cache automatically — model flags are only
needed to override.** Resolution order everywhere: explicit CLI arg / env >
`$SONIC_X2_MODELS` (redirects just the x2 subtree) > `$SONIC_HOME/x2` >
legacy machine-local paths. Example override:

```bash
# explicit model override (any stack; same for --onnx on the eval script):
bash gear_sonic/scripts/run_x2_pkl_direct_stack.sh --model /path/to/other_policy.onnx
```

With the cache populated the quest3 stack also defaults `KPLANNER_ONNX` to
the cached template graph (the recommended, robot-identical planner path);
set `KPLANNER_ONNX=""` explicitly to force the torch kplanner tier instead.

## 3. Sim — fastest first

### 3a. Watch the policy in MuJoCo (seconds to start, no docker)

The repo ships ready-to-play motion pkls under `gear_sonic/data/motions/`:
`x2_dances_easy.pkl` (short dance set) and a **self-generated motion library recorded from our real robots** (no
external mocap provenance):
- `x2_recorded_gestures/` + `x2_recorded_gestures/mc/` — 63 gestures from
  the live X2 (waves, hearts, peace, bows, claps, fist bumps, hugs, ...)
- `g1_recorded_walks/` — 34 teleop-driven walking sessions recorded on a
  real Unitree G1 (VR- and keyboard-piloted; slow/medium walks +
  walk-and-manipulate)
- `x2_g1_recorded_walks/` — the same 34 sessions retargeted to X2, so the
  identical motion plays on either embodiment

Play one in a live viewer:

```bash
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
  --motion gear_sonic/data/motions/x2_dances_easy.pkl
# or a robot-recorded gesture:
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
  --motion gear_sonic/data/motions/x2_recorded_gestures/left_wave_high_001.pkl
# walking: forward walk loop:
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
  --motion gear_sonic/data/motions/x2_walk_examples/relaxed_walk_forward.pkl
# or a real-robot teleop walking session:
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
  --motion gear_sonic/data/motions/x2_g1_recorded_walks/slow_walk_medium_vr_001.pkl
```

Headless CI-style run with a tracking summary instead:

```bash
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
  --motion gear_sonic/data/motions/x2_dances_easy.pkl \
  --no-viewer --total-sim-seconds 20
```
`--onnx` is auto-resolved from the model cache; pass it only to override.

The viewer runs with the REAL ROBOT's deployment settings by default —
tuning preset `walk_101.yaml` (per-group PD trim, target clamps, target
LPF, action clip), the same preset the robot deploys with, so what you
see is what the robot does. Pass `--tuning ''` for raw training-parity
gains (parity/eval baselines; note some walking clips fall without the
deploy trims), or `--tuning <path>` for another preset.
If the viewer feels slow/laggy: physics runs 50 Hz with the window
redrawn every 2nd step; set `VIEWER_RENDER_STRIDE=1` for full-rate
rendering on fast GPUs, and `ORT_NUM_THREADS` (default 2) for the policy
session.
Expected: episodes end `motion_end` (no falls), pelvis ~0.58 m. Drop
`--no-viewer` for a live window. Parity check: add
`--compare-pt ~/.cache/sonic/x2/sonic_policy/x2_sonic_policy.pt --max-episode 10`
(expect max action delta ≤1e-4).

### 3b. Preflight the full stack without launching anything
```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --validate-only --no-record --no-x2-debug-bridge
```
Checks imports, assets, planner graphs, calibration. Fix everything it
flags before a real launch.

### 3c. Gamepad-driven sim (full deploy loop, needs the docker image)

Plug in an Xbox-class pad (or DualSense) and:

```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pad-only \
    --no-record --no-x2-debug-bridge            # pad, no headset
# or with a Quest 3 alongside (pad drives until VR engages):
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pad-and-vr \
    --no-record --no-x2-debug-bridge
```

Full control list — deadman, sticks, speed nudges, dance/combat/gesture
clip banks, e-stop — in **section 3f below**. The clip banks default to
the real robot's ritual bindings.

Variant with robot-parity md5 checks (validates the planner + policy dirs
against what a reachable robot actually runs):

```bash
bash gear_sonic/scripts/sim_onnx_planner.sh <PC2_IP>
```
Models auto-resolve (md5 candidate pool includes the model cache; override
with `SONIC_MODEL=` / `PLANNER_MODEL=` envs). Spawns MuJoCo sim + the C++
SONIC deploy node (docker) + the ONNX kplanner + pad bridge. The script
identity-checks the planner dir by md5 against the robot manifest when
reachable — sim must validate the SAME planner the robot runs, not a stale
local copy.

### 3d. Direct PKL playback in sim (no VR, no pad)
```bash
# terminal 1 — bring up sim deploy + recorder (MuJoCo window opens):
bash gear_sonic/scripts/run_x2_pkl_direct_stack.sh

# terminal 2 — trigger clips:
python gear_sonic/scripts/play_gesture.py --list                 # catalog
python gear_sonic/scripts/play_gesture.py wave_high     # named gesture
python gear_sonic/scripts/play_gesture.py --pkl gear_sonic/data/motions/x2_dances_easy.pkl
python gear_sonic/scripts/play_locomotion.py --list              # shipped walk pkls
python gear_sonic/scripts/play_locomotion.py --pkl <walk_or_turn.pkl>  # walks keep authored heading
python gear_sonic/scripts/play_gesture.py --release              # abort / release held pose
```

### 3e. Quest 3 VR teleop in sim
```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh   # sim is the default
```
Quest 3 over WebXR; operator calibration file required
(`data/operator_calibrations/<id>.yaml`, create via
`vr_operator_calibrate.py`; `default.yaml` ships). Headset connection
steps, the A+B+X+Y engage chord, and every button/stick binding are in
**section 3f below** (authoritative state machine:
`gear_sonic/utils/teleop/vr/button_state_machine.py`). Add `--pad-and-vr`
to keep a gamepad active alongside the headset — the pad drives until the
VR operator engages.

### 3f. Controls reference (sim and real robot — same bindings)

**Gamepad** (Xbox-class; DualSense works in sim):
- **Hold L2+R2 (deadman) to drive** — release = robot idles immediately
- Left stick: walk forward/back + strafe · Right stick: turn
- **X / Y buttons: nudge the forward-speed setpoint** down/up
  (default 0.5 m/s, clamp 0.2–1.0)
- Clip banks (deadman RELEASED; identical to the robot's ritual bindings,
  defined in `x2_pc2/ritual_start_demo.sh`, sim defaults in the quest3
  stack): **L1+Y** easy dances · **L1+X** combat (shadow boxing) ·
  **L1+B** medium dances · **L1+A** gestures (wave/kiss/five/shake/
  turn-waves) — each press advances its bank; **L1+R1** stops the clip
- Right stick with deadman RELEASED: LEFT/RIGHT = instant 270-degree
  in-place turns, UP (hold 2 s) = relaxed walk forward
- Clips live in `gear_sonic/data/motions/x2_pad_banks.pkl`; override the
  banks with `PAD_CLIP_PKL` / `PAD_CLIP_KEYS[_X|_M|_G|_TURN]` envs, or
  `PAD_CLIP_PKL=""` to disarm the chords
- **L1+R1+L2+R2 all held = e-stop chord** — commands go dead instantly
- Stick-up = forward is the default (`--invert-ly` is on); the bridge
  refuses to start if multiple pads are connected or the trigger axes
  don't rest where the driver mapping expects (see startup log)

**Quest 3 VR**:
- Open the printed WebXR URL in the headset browser (accept the
  self-signed cert) — the stack prints it at launch
- **Engage with the controller chord (A+B+X+Y)** — until engaged, VR
  publishes nothing and the pad (if running) keeps control
- Sticks = locomotion intent through the planner; hand poses = arm
  targets (tethered mode)
- Headset stream silent >0.5 s → inputs forced neutral (safety);
  disengage chord hands control back cleanly

**Untethered robot ignition** (no laptop, pad only): hold
**L1+R1+L2+R2 for 3 s** → 3 rumble buzzes (ARMED) → press **Y within
8 s** → the onboard ritual starts SONIC. No rumble = the pad daemon
isn't running or the pad isn't bonded — check
`/home/run/gear-sonic/log/pad_autopair.log` on PC2.

## 4. Real robot

Network — two common setups; `<PC2_IP>` below is whichever applies:
- **Robot on your wifi** (the usual case): PC2 joins your router and gets a
  DHCP address (find it on your router's client list, e.g.
  `192.168.x.x`). Your workstation is on the same wifi; use that address
  as `<PC2_IP>` and your workstation's wifi address as `<LAPTOP_IP>`.
- **Tethered / robot's internal subnet**: the onboard PCs live on the
  vendor-defined subnet (PC2 = 10.0.1.41); your workstation joins the
  robot's network and uses `--pc2-host 10.0.1.41`.

Sequence:
1. **Preflight** on the workstation: step 3b, plus `x2_preflight.py` on the
   robot side (ROS 2 / rclpy — runs on PC2, not locally), and
   `gear_sonic_deploy/scripts/pc2_preflight.sh` against the robot.
2. **PC2 bringup** (one-time provisioning + per-session daemons):
   `gear_sonic_deploy/scripts/pc2_bringup.sh` provisions the onboard PC2
   staging tree (`/home/run/gear-sonic/` — models, planner graphs, runtimes);
   `x2_pc2_daemons.sh start` brings up the per-session daemons and
   `x2_pc2_cameras.sh` the head-camera bridge. Verify the pose proxy and
   planner runtime are up before proceeding.
3. **Launch**: same entrypoints as sim with `--pc2-host` — no local deploy
   is spawned; the recorder's PUB binds LAN-visible and PC2 connects out.
4. **Verify before moving**: robot holds idle stand; planner md5 identity
   check passes; telemetry posture sane.

### Real-robot command reference (`<PC2_IP>` / `<LAPTOP_IP>` per Network above)

```bash
# PC2 per-session daemons — run these FIRST (and 'stop' when done).
# Defaults: model + walk_101 tuning auto-resolve; everything else defaults.
bash gear_sonic_deploy/scripts/x2_pc2_daemons.sh start     --attach --pc2-host <PC2_IP> --laptop-host <LAPTOP_IP>
bash gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host <PC2_IP>
bash gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve --host <PC2_IP>      # head cams (optional)

# Direct PKL play on the REAL robot (same two-terminal flow as sim):
bash gear_sonic/scripts/run_x2_pkl_direct_stack.sh --pc2-host <PC2_IP>
python gear_sonic/scripts/play_gesture.py wave_high     # terminal 2

# Quest 3 VR teleop on the real robot:
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pc2-host <PC2_IP>

# Gamepad driving with the certifiable sim-parity check (md5-verifies the
# planner + policy against what PC2 actually runs before anything moves):
bash gear_sonic/scripts/sim_onnx_planner.sh <PC2_IP>
bash gear_sonic/scripts/sim_onnx_planner.sh --pc2-host <PC2_IP> --vr   # VR variant
```

Operator controls are IDENTICAL to sim — the gamepad and Quest 3
bindings in **section 3f** (deadman, speed nudges, clip banks, e-stop,
VR engage chord) apply unchanged on the real robot; the untethered
pad-only ignition ritual is also in 3f.

Same safety rituals apply regardless of entrypoint: preflight first, MC
handoff before killing anything, telemetry before re-engaging.

### Safety rituals (non-negotiable)
- **Never kill a live deploy without a motion-controller handoff.** SONIC
  expects a continuous reference stream; killing the upstream mid-run leaves
  the robot tracking stale references. Hand off to the vendor MC (or e-stop)
  FIRST, then stop processes.
- The deploy's pose-ref watchdog trips to SAFE_IDLE after 0.5 s of missing
  reference — that is a safety feature; only disable it
  (`--disable-pose-ref-watchdog`) in local sim.
- After any kill: verify from telemetry (posture, joint torques) — not from
  process state — before re-engaging.
- Watch lower-limb thermals on long sessions; duty-cycle demos.

## 5. External asset sources (when downloads fail or links move)

`setup_x2.sh` fails loudly when a download breaks. Direct URLs go stale —
each asset below lists the **vendor page to search for the current link**,
and the env var to point the installer at it:

| asset | fetched to | env override | if the link moved, look here |
|---|---|---|---|
| SONIC models (policy + planner) | `~/.cache/sonic/x2` | `SONIC_X2_MODELS` / HF repo | https://huggingface.co/tinkerbuggy/sonic-x2 |
| X2 robot meshes (45 STL) | `.../urdf/x2_ultra/meshes/` | `X2_URDF_URL` | https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html |
| AimDK SDK (`aimdk_msgs`) | `gear_sonic_deploy/thirdparty/aimdk_msgs/` | `AIMDK_SDK_URL` | https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html and https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/robot_specifications.html |
| OmniHand meshes | `.../omnihand/meshes/` (2 custom ones in-repo) | `OMNIHAND_URDF_URL` | https://www.agibot.com.cn/DOCS/OS/Omnihand-O10 |

## 5b. Troubleshooting

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: onnxruntime` | `pip install onnxruntime` (not in extras) |
| Motion pkls are 130-byte text files | LFS pointers — `git lfs pull` |
| Port 5556 in use at launch | another stack instance — stack's `--cleanup-only` kills stack ports (careful: `fuser -k`), or pass `--pose-port` |
| validate-only fails wanting a robot host | add `--no-x2-debug-bridge` |
| sim behaves unlike robot | planner identity mismatch — let the script auto-select the dir that md5-matches the robot, don't pin stale graphs |
| recorder import-fails on `lerobot` | only `--with-record` needs it; default teleop path runs without |

## 6. VLA workflow (record → train → run)

```bash
# RECORD demonstrations (VR teleop + LeRobot dataset writing):
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --with-record   # sim
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pc2-host <PC2_IP> --with-record

# REPLAY a recorded dataset (kinematic or through the policy):
bash gear_sonic/scripts/run_x2_replay_stack.sh --dataset <lerobot_dataset_dir>

# TRAIN the GR00T VLA on your recordings:
bash gear_sonic/scripts/vla/train_groot_vla.sh --dataset <lerobot_dataset_dir> \
    --modality-config gear_sonic/data/x2_modality_config_10dof.py

# RUN VLA inference driving the robot/sim:
bash gear_sonic/scripts/vla/run_x2_vla_runtime.sh --checkpoint <finetuned_ckpt_dir>          # sim
bash gear_sonic/scripts/vla/run_x2_vla_runtime.sh --checkpoint <ckpt> --pc2-host <PC2_IP>    # real
python gear_sonic/scripts/vla/mock_vla_publish_stand_token.py   # wire test without a model
```
