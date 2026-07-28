---
name: x2-sonic-stack
description: Set up and operate the AgiBot X2 SONIC whole-body-control stacks — headless policy eval, gamepad-driven sim, VR teleop in sim, and real-robot deployment
---

# X2 SONIC stack — operator skill

Everything needed to go from a fresh clone to X2 walking in MuJoCo sim, and
from sim to the real robot. Written to be followed top-to-bottom by a human
or an agent; each stage is independently verifiable.

## 1. Install (once)

See `SETUP.md` for the validated details. Short form:

```bash
git clone <this-repo> && cd <repo> && git lfs install && git lfs pull
python3.10 -m venv .venv && source .venv/bin/activate && pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # BEFORE gear_sonic, else CUDA wheel
pip install -e "./gear_sonic[sim]"
pip install onnxruntime                                             # not in any extra; required
pip install -e ./motionbricks
```

GPU torch is only needed for training; every runtime below works on CPU.

## 2. Models — download once, point the stacks at them

Weights are NOT in this repo. They are hosted on Hugging Face at
**`tinkerbuggy/sonic-x2`** (private during review; public at release — until
then request access or use `hf auth login` with a granted token). Every
deployment artifact there is md5-verified byte-identical to the copies
running on the reference robot.

```bash
pip install -U "huggingface_hub[cli]"
hf download tinkerbuggy/sonic-x2 --local-dir "$HOME/x2_models"
export CKPT="$HOME/x2_models"
```

Layout you get:

```
$CKPT/sonic_policy/x2_sonic_policy.onnx        # fused SONIC tracking policy (deploy)
$CKPT/sonic_policy/x2_sonic_policy.pt          # source ckpt (parity checks / finetune)
$CKPT/kplanner_onnx/x2_kplanner_template.onnx  # torch-free planner graph (template mode)
$CKPT/kplanner_onnx/x2_kplanner_velocity.onnx  # legacy velocity-mode graph
$CKPT/kplanner_torch/{vqvae,pose,root}/...     # planner finetuning tier (+ x2_clip.ckpt)
```

**Wiring the stacks to your download** (script defaults reference the
maintainers' machines — always pass these explicitly):

```bash
# quest3 VR stack (sim or --pc2-host):
KPLANNER_ONNX=$CKPT/kplanner_onnx/x2_kplanner_template.onnx \
  bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
       --model $CKPT/sonic_policy/x2_sonic_policy.onnx

# pkl-direct stack:
bash gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
     --model $CKPT/sonic_policy/x2_sonic_policy.onnx

# headless policy eval:
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
       --onnx $CKPT/sonic_policy/x2_sonic_policy.onnx --motion <pkl> --no-viewer
```

Without `KPLANNER_ONNX` the quest3 stack tries the torch kplanner, which
needs the `kplanner_torch/` tier on disk at the paths in
`motionbricks/out/` — the ONNX graph is the recommended (robot-identical)
path.

## 3. Sim — fastest first

### 3a. Headless policy eval (seconds to start, no docker)
```bash
python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
  --onnx $CKPT/sonic_policy/x2_sonic_policy.onnx \
  --motion gear_sonic/data/motions/x2_dances_easy.pkl \
  --no-viewer --total-sim-seconds 20
```
Expected: episodes end `motion_end` (no falls), pelvis ~0.58 m. Drop
`--no-viewer` for a live window. Parity check: add
`--compare-pt $CKPT/sonic_policy/x2_sonic_policy.pt --max-episode 10`
(expect max action delta ≤1e-4).

### 3b. Preflight the full stack without launching anything
```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --validate-only --no-x2-debug-bridge
```
Checks imports, assets, planner graphs, calibration. Fix everything it
flags before a real launch.

### 3c. Gamepad-driven sim (full deploy loop, needs the docker image)
```bash
KPLANNER_ONNX=$CKPT/kplanner_onnx bash gear_sonic/scripts/sim_onnx_planner.sh \
  MODEL=$CKPT/sonic_policy/x2_sonic_policy.onnx
```
Spawns MuJoCo sim + the C++ SONIC deploy node (docker) + the ONNX kplanner +
pad bridge. The script identity-checks the planner dir by md5 against the
robot manifest when reachable — sim must validate the SAME planner the robot
runs, not a stale local copy.

### 3d. Direct PKL playback in sim (no VR, no pad)
```bash
# terminal 1 — bring up sim deploy + recorder (MuJoCo window opens):
bash gear_sonic/scripts/run_x2_pkl_direct_stack.sh --model $CKPT/sonic_policy/x2_sonic_policy.onnx

# terminal 2 — trigger clips:
python gear_sonic/scripts/play_gesture.py --list                 # catalog
python gear_sonic/scripts/play_gesture.py sit_stand_sit_A538     # named gesture
python gear_sonic/scripts/play_gesture.py --pkl gear_sonic/data/motions/x2_dances_easy.pkl
python gear_sonic/scripts/play_locomotion.py --pkl <walk_or_turn.pkl>  # walks keep authored heading
python gear_sonic/scripts/play_gesture.py --release              # abort / release held pose
```

### 3e. Quest 3 VR teleop in sim
```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh   # sim is the default
```
Quest 3 over WebXR; operator calibration file required
(`data/operator_calibrations/<id>.yaml`, create via
`vr_operator_calibrate.py`; `default.yaml` ships). While driving: VR sticks
= locomotion intent through the kplanner; controller buttons per
`gear_sonic/utils/teleop/vr/button_state_machine.py`.

## 4. Real robot

Network: the robot's onboard PCs live on the vendor-defined subnet
(PC2 = 10.0.1.41). Your workstation joins the robot's network; nothing else
changes — the same stack scripts take `--pc2-host 10.0.1.41`.

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

### Real-robot command reference (`<PC2_IP>` = robot's LAN address)

```bash
# Direct PKL play on the REAL robot (same two-terminal flow as sim):
bash gear_sonic/scripts/run_x2_pkl_direct_stack.sh --pc2-host <PC2_IP>
python gear_sonic/scripts/play_gesture.py sit_stand_sit_A538     # terminal 2

# Quest 3 VR teleop on the real robot:
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pc2-host <PC2_IP>

# Gamepad driving with the certifiable sim-parity check (md5-verifies the
# planner + policy against what PC2 actually runs before anything moves):
bash gear_sonic/scripts/sim_onnx_planner.sh <PC2_IP>
bash gear_sonic/scripts/sim_onnx_planner.sh --pc2-host <PC2_IP> --vr   # VR variant
```

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

## 5. Troubleshooting

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
bash gear_sonic/scripts/train_groot_vla.sh --dataset <lerobot_dataset_dir> \
    --modality-config gear_sonic/data/x2_modality_config_10dof.py

# RUN VLA inference driving the robot/sim:
bash gear_sonic/scripts/run_x2_vla_runtime.sh --checkpoint <finetuned_ckpt_dir>          # sim
bash gear_sonic/scripts/run_x2_vla_runtime.sh --checkpoint <ckpt> --pc2-host <PC2_IP>    # real
python gear_sonic/scripts/mock_vla_publish_stand_token.py   # wire test without a model
```
