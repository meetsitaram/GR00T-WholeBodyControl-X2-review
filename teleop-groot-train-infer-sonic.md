# Plan: Integrating AgiBot X2 Ultra into the SONIC + GR00T N1.7 Pipeline

> End-to-end plan for getting AgiBot X2 Ultra (30 DOF) running through the full loop:
> **retarget motions → train SONIC controller → VR teleop → collect demos → fine-tune GR00T N1.7 VLA → deploy autonomous policy on the robot via SONIC.**

**Target embodiment:** AgiBot X2 Ultra
- 30 actuated DOF: 1 neck + 7×2 arms + 3 waist + 6×2 legs
- 1.31 m, ~39 kg, 558 mm arm reach
- Onboard compute: 2× RK3588 + NVIDIA Orin NX (157 TOPS)
- Sensors: 3D LiDAR, RGB-D, front stereo RGB, rear RGB, head touch
- SDK: AimDK_X2 (task-level)
- Foundation model on-device: GO-1 (we ignore this — we're using GR00T N1.7 instead)

**Repos involved:**
- [`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl) — SONIC training, teleop, data collection, C++ deployment
- [`NVIDIA/Isaac-GR00T`](https://github.com/NVIDIA/Isaac-GR00T) — VLA post-training (N1.7) and inference

---

## Architectural overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      OFFLINE: TRAINING                              │
│                                                                     │
│  Bones-SEED BVH ──▶ SOMA Retargeter ──▶ X2 Ultra PKL motions        │
│                                            │                        │
│                                            ▼                        │
│  PPO + Universal Token (Isaac Lab) ──▶ SONIC X2-Ultra checkpoint    │
│                                            │                        │
│                                            ▼                        │
│                                    ONNX export ──▶ C++ deploy       │
└─────────────────────────────────────────────────────────────────────┘
                                            │
┌─────────────────────────────────────────────────────────────────────┐
│                      ONLINE: DATA COLLECTION                        │
│                                            ▼                        │
│   PICO VR ──SMPL──▶ SONIC controller ──▶ X2 Ultra hardware          │
│                              │                  │                   │
│                              │                  ├── ego camera      │
│                              │                  └── wrist cameras   │
│                              ▼                                      │
│                  run_data_exporter.py                               │
│                              │                                      │
│                              ▼                                      │
│                  LeRobot v2.1 dataset (parquet + mp4)               │
└─────────────────────────────────────────────────────────────────────┘
                                            │
┌─────────────────────────────────────────────────────────────────────┐
│                      OFFLINE: VLA POST-TRAINING                     │
│                                            ▼                        │
│   Isaac-GR00T launch_finetune.py ──▶ N1.7 fine-tuned for X2 Ultra   │
└─────────────────────────────────────────────────────────────────────┘
                                            │
┌─────────────────────────────────────────────────────────────────────┐
│                      ONLINE: AUTONOMOUS DEPLOYMENT                  │
│                                            ▼                        │
│   Camera + state ──▶ Gr00tPolicy ──SMPL/joints──▶ SONIC ──▶ Robot   │
└─────────────────────────────────────────────────────────────────────┘
```

The single most important architectural insight: **the SONIC controller treats teleop and VLA the same.** Both produce SMPL pose targets / joint setpoints over the same ZMQ interface. So once SONIC is trained for X2 Ultra, swapping the human (PICO) for the VLA (N1.7) is a matter of changing the upstream publisher — not rewriting the controller.

---

## Phase 0 — Prerequisites

### Hardware
- AgiBot X2 Ultra robot
- Workstation: Linux, ≥1× CUDA GPU with ≥24 GB VRAM (RTX 4090, A6000, or H100). For from-scratch SONIC training, ≥8 GPUs is realistic; ≥64 is what NVIDIA used.
- PICO 4 / PICO 4 Pro / PICO Neo VR headset for teleop
- Network: workstation and robot on same LAN (the camera server runs on-robot, everything else off-robot)
- Luxonis OAK cameras (head ego-view + optional wrist cams) — the X2 Ultra's stock RGB-D + LiDAR is **not** the tested camera path. Plan to mount OAKs alongside, or write a driver shim.

### Software / accounts
- Hugging Face account + accepted EULA for `nvidia/GEAR-SONIC` and `nvidia/GR00T-N1.7`
- Bones-SEED dataset access ([huggingface.co/datasets/bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed))
- Isaac Lab 2.3 installed (separate install — not bundled in either repo)
- AgiBot AimDK_X2 SDK + access to robot's low-level joint API (this is the riskiest software prerequisite — see Phase 8)

### Critical unknowns to resolve before starting
- [ ] **URDF availability.** Does AgiBot publish an open URDF for X2 Ultra, or is it locked behind a partner agreement? If locked, this plan stalls at Phase 1. NVIDIA's H2 example assumes you have a clean URDF + meshes.
- [ ] **Low-level joint control access.** SONIC outputs joint position targets at ~50 Hz. AimDK_X2 is "task-level," which suggests it may *not* expose direct joint setpoint streaming. Confirm with AgiBot whether there's a low-level API (likely on the Xyber-Edge controller) and whether you can bypass GO-1.
- [ ] **Motor specs for KP/KD tuning.** Need armature (rotor inertia), peak torque per joint, and gear ratio per joint group. Datasheet says 120 Nm peak joint torque but that's a single number; you need the breakdown.
- [ ] **Foot/contact geometry.** SONIC's reward terms reference ankle roll links. X2's foot design must be checked against this assumption.

If any of those four are blocked, stop here and resolve before writing code.

---

## Phase 1 — Robot model files (Step 1 of the new-embodiment guide)

**Goal:** Get a clean URDF + MJCF for X2 Ultra into `gear_sonic/data/assets/robot_description/`.

### Tasks
1. Obtain X2 Ultra URDF + meshes from AgiBot.
2. If only ROS package format is provided, convert `package://` paths to relative `meshes/...` paths.
3. Convert URDF → MJCF (MuJoCo XML). Tools:
   - `mujoco.MjModel.from_xml_path` won't read URDF directly — use `urdf2mjcf` or write the MJCF by hand using a Unitree humanoid as template.
   - **Verify** that joint names and tree structure match exactly between URDF and MJCF. Mismatches here cause silent corruption later.
4. Place files:
   ```
   gear_sonic/data/assets/robot_description/
   ├── urdf/agibot_x2_ultra/
   │   ├── agibot_x2_ultra.urdf
   │   └── meshes/
   └── mjcf/
       └── agibot_x2_ultra.xml
   ```
5. **Smoke test:** load the URDF in Isaac Lab and the MJCF in MuJoCo, render a default standing pose, eyeball that they look the same.

### Acceptance
- Robot loads in both simulators without errors.
- Standing pose looks identical side-by-side.
- Joint count = 30, body count = 31 (30 + pelvis root). Confirm.

---

## Phase 2 — Robot configuration (Step 2)

**Goal:** Create `gear_sonic/envs/manager_env/robots/agibot_x2_ultra.py`.

This is the most failure-prone file in the entire pipeline. Budget real time for it.

### 2a. Joint and body order mappings
Isaac Lab and MuJoCo traverse the kinematic tree differently. Build four index arrays:
- `AGIBOT_ISAACLAB_TO_MUJOCO_DOF` (length 30)
- `AGIBOT_MUJOCO_TO_ISAACLAB_DOF` (length 30)
- `AGIBOT_ISAACLAB_TO_MUJOCO_BODY` (length 31)
- `AGIBOT_MUJOCO_TO_ISAACLAB_BODY` (length 31)

Procedure:
1. Load URDF in Isaac Lab; print `articulation.joint_names` and `articulation.body_names`.
2. Load MJCF in MuJoCo; print joint names from `model.joint(i).name` and body names from `model.body(i).name`.
3. Compute `to_mujoco[i] = mujoco_index_of(isaaclab_names[i])` and the inverse.
4. **Validate by round-tripping a known pose.** Set joint values in IsaacLab order, reorder to MuJoCo, check that `mj_forward` produces the expected end-effector positions.

> Skipping this validation = scrambled observations during training and a policy that "trains" but produces gibberish. This has bitten everyone who's done this.

### 2b. Actuator KP/KD (per motor group)
SONIC uses implicit PD actuators. Group joints by motor type — X2 Ultra likely has at least:
- **Legs** (hip pitch/roll/yaw, knee, ankle pitch/roll) — strongest, e.g. 120 Nm peak
- **Waist** (3 DOF) — strong
- **Arms** (shoulder, elbow, wrist) — medium, with weaker wrist
- **Neck** (1 DOF) — weakest

Starting formulas from the H2 example:
```python
NATURAL_FREQ = 10 * 2.0 * 3.1415926535   # 10 Hz
DAMPING_RATIO = 2.0                       # overdamped
KP = armature * NATURAL_FREQ**2
KD = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
```

Tuning notes:
- Get armature (rotor inertia) per motor from AgiBot. If unavailable, estimate from torque + speed curve.
- If training crashes immediately ("robot explodes"), reduce KP or raise KD.
- Effort limits per joint must match real motor specs — sim-to-real failure mode is otherwise hidden.

### 2c. `ArticulationCfg`
Mirror the H2 pattern. Key fields:
- `pos=(0.0, 0.0, 1.10)` — set so feet are slightly above ground (X2 is 1.31 m, hip ≈ 0.7-0.8 m, allow margin)
- `joint_pos`: stable standing pose. Get from AgiBot's calibration default or capture one in MuJoCo.
- `actuators`: one `ImplicitActuatorCfg` per motor group with effort/stiffness/damping/armature dicts.

### 2d. Action scale
```python
AGIBOT_ACTION_SCALE = {
    j: effort_limit[j] / stiffness[j] for j in joint_names
}
```

### 2e. Register
- Add import to `gear_sonic/envs/manager_env/robots/__init__.py`
- Add entry to `robot_mapping` dict in `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` around line 998:
  ```python
  "agibot_x2_ultra": {
      "robot_cfg": agibot_x2_ultra.AGIBOT_CFG,
      "action_scale": agibot_x2_ultra.AGIBOT_ACTION_SCALE,
      "isaaclab_to_mujoco_mapping": agibot_x2_ultra.AGIBOT_ISAACLAB_TO_MUJOCO_MAPPING,
  }
  ```

### Acceptance
- `num_envs=1 headless=False` launches without errors.
- Robot loads, stands stably for ≥5 seconds in zero-action mode (with gravity).
- No body-name errors in logs.

---

## Phase 3 — Order converter (Step 3)

Add `AgibotX2UltraConverter` to `gear_sonic/trl/utils/order_converter.py`. Mirror the H2 pattern. Important fields specific to X2:

- `VR_3POINTS_BODY_NAMES`: pick the 3 bodies the VR system tracks. Convention is `[torso/chest, left_wrist, right_wrist]`. Use AgiBot's body names exactly.
- `FOOT_BODY_NAMES`: the two ankle/foot links used for foot contact and termination logic.

Use lazy imports inside `__init__` to dodge circular-import issues.

---

## Phase 4 — Body name compatibility audit (Step 4)

This is the step that *seems* trivial and bites you anyway. The training configs reference Unitree-specific body names. You must either rename your URDF bodies to match (cleaner) or override every reference (more invasive).

### Files to audit
- `config/manager_env/commands/terms/motion.yaml` — `anchor_body`, `vr_3point_body`, `reward_point_body`, `body_names` (14 tracked bodies)
- `config/manager_env/terminations/terms/ee_body_pos_adaptive.yaml`
- `config/manager_env/terminations/terms/foot_pos_xyz.yaml`
- `config/manager_env/rewards/terms/undesired_contacts.yaml`
- `config/manager_env/rewards/terms/anti_shake_ang_vel.yaml`

### Strategy
1. Run training with `num_envs=1 headless=False`. Isaac Lab will throw clear errors on missing body names, one at a time.
2. For each error, override in the experiment config (Phase 6) rather than editing the shared YAMLs:
   ```yaml
   manager_env:
     rewards:
       anti_shake_ang_vel:
         params:
           body_names: ["agibot_left_wrist_link", "agibot_right_wrist_link", "agibot_neck_link"]
   ```
3. Document every override in the experiment config with a comment so the diff vs. G1/H2 is auditable.

### X2-specific gotchas
- X2 Ultra has a **1-DOF neck** (G1 has none). The `head_link` references in G1 configs need to map to the neck link or be removed.
- X2 has a **3-DOF waist** (G1 has 3 too — likely fine, but verify joint axis conventions).
- 7-DOF arm (G1 has 7 too — likely fine, but the wrist body naming is the most common divergence point).

---

## Phase 5 — Motion data (Step 5) — **the biggest time sink**

**Goal:** Produce a directory of PKL motion files in MuJoCo joint/body order, retargeted to X2 Ultra's 30-DOF skeleton.

### Tasks
1. **Download Bones-SEED BVH files** from `huggingface.co/datasets/bones-studio/seed`.
2. **Retarget BVH → X2 Ultra** using SOMA Retargeter:
   - Author a JSON config mapping SMPL joints to X2 Ultra joints (this is the manual part — you specify which SMPL joint drives which robot joint and any axis flips/offsets).
   - Run retargeting in batch. Expect: hours-to-days for the full ~142K motions on a single GPU.
   - Use SOMA's viewer to QA a sample of retargeted motions before committing to the full batch.
3. **Convert to PKL** via the repo's data processor:
   ```bash
   python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
       --input /path/to/retargeted_csvs/ \
       --output data/agibot_motions/robot \
       --fps 30 --fps_source 120 --individual --num_workers 16
   ```
4. **Filter physically infeasible motions:**
   ```bash
   python gear_sonic/data_process/filter_and_copy_bones_data.py \
       --source data/agibot_motions/robot \
       --dest data/agibot_motions/robot_filtered
   ```
5. **Generate mirrored variants** (`_M.pkl`) — doubles dataset size, improves symmetry.
6. **SMPL data**: download Bones-SEED SMPL data from `nvidia/GEAR-SONIC` HF repo. The motion *keys* must match across SMPL and robot files. If they don't, set `smpl_motion_file: dummy` in config (weaker but works).

### Output PKL schema
```python
{
    "motion_name": {
        "root_trans_offset": np.ndarray,  # (T, 3)
        "pose_aa":            np.ndarray, # (T, 31, 3) — 31 bodies for X2 Ultra
        "dof":                np.ndarray, # (T, 30) — MuJoCo order
        "root_rot":           np.ndarray, # (T, 4) wxyz
        "smpl_joints":        np.ndarray, # (T, 24, 3) or zeros
        "fps": 30,
    }
}
```

### Acceptance
- ≥10K filtered, retargeted motions for X2 Ultra
- Motions visually plausible when replayed in MuJoCo
- No NaNs, no joint-limit violations, no ground-clipping pelvis trajectories

---

## Phase 6 — Experiment config (Step 6)

Create `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_agibot_x2_ultra.yaml`.

Start by copying `sonic_release.yaml` (the G1 release config) and modify:

```yaml
# @package _global_
defaults:
  - /algo: ppo_im_phc
  - /manager_env: base_env
  # ... keep the rest from sonic_release.yaml

project_name: TRL_AgibotX2Ultra_Track

manager_env:
  config:
    robot:
      type: agibot_x2_ultra            # must match robot_mapping key

  commands:
    motion:
      motion_lib_cfg:
        motion_file: null              # provided on CLI
        asset:
          assetFileName: "agibot_x2_ultra.xml"

  # Body name overrides from Phase 4 audit go here:
  rewards:
    anti_shake_ang_vel:
      params:
        body_names: [...]              # X2 Ultra-specific
  # ... etc.
```

Fields to review:
- `reward_point_body` / `reward_point_body_offset` — bodies used in tracking reward
- `vr_3point_body` / `vr_3point_body_offset` — for VR teleop
- `upper_body_augment_prefixes` — drop or rewrite based on motion data naming
- All body names that errored in Phase 4

---

## Phase 7 — SONIC training (Step 7)

### 7a. Smoke test
```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_agibot_x2_ultra \
    num_envs=16 headless=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/agibot_motions/robot_filtered
```

Watch for: robot stands, plays motions, doesn't immediately fall through floor.

### 7b. Full training
```bash
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_agibot_x2_ultra \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/agibot_motions/robot_filtered
```

Or finetune from the released G1 checkpoint (faster, but quality depends on how similar X2 Ultra's kinematics are to G1's — the Universal Token is supposed to bridge this, but it's an empirical bet):
- Download checkpoint via `python download_from_hf.py --training`
- Add `++load_checkpoint=path/to/sonic_release.pt` (or whatever the flag is — check `train_agent_trl.py --help`)

### 7c. Compute budget reality check
- 8 GPUs: days to a usable checkpoint, weeks to a strong one
- 64+ GPUs: NVIDIA's setup; expect ~1-2 weeks to publication quality
- 1 GPU: feasible only with the finetune-from-G1 path, and quality will be visibly worse

### 7d. Export to ONNX
After training, export the policy + Universal Token + planner to ONNX. The export script is in `gear_sonic/scripts/` (look for `export_*.py`). Verify ONNX runtime can consume the file and produce identical actions to PyTorch on a held-out batch.

### Acceptance
- Tracking reward plateaus above some threshold (compare to released G1 checkpoint metrics in the paper)
- Sim2sim (Isaac Lab → MuJoCo) eval passes — robot can replay teleop poses in MuJoCo using the trained policy
- ONNX export reproduces PyTorch predictions to within float tolerance

---

## Phase 8 — C++ deployment to real hardware

**Goal:** Get the C++ runtime in `gear_sonic_deploy/` talking to the X2 Ultra robot.

### 8a. Robot interface layer (the hard part)
The C++ deploy stack assumes Unitree's SDK conventions: a low-level joint command interface that accepts position targets at ≥200 Hz and returns joint state. AimDK_X2 is task-level. You will need to:

1. **Reach out to AgiBot for the low-level API.** The Xyber-Edge controller almost certainly exposes one — they just don't advertise it because they want you using GO-1. Push.
2. Write an adapter in `gear_sonic_deploy/` translating SONIC's joint command struct to AimDK_X2 (or whatever low-level API you get).
3. Translate IMU + joint state from the robot's format back to SONIC's expected struct.

This phase is gated entirely on what AgiBot will give you. Allocate generous schedule slack.

### 8b. Camera server on-robot
The X2 Ultra has an Orin NX with 157 TOPS — perfect for the camera server. Either:
- Mount Luxonis OAK cameras alongside the stock cameras and run NVIDIA's tested camera server unmodified, **or**
- Write a camera driver that publishes the X2's stock RGB-D + RGB cameras over the same ZMQ schema (ego + optional wrist), then point the workstation at it.

The first option is lower-risk and faster. Do that first; circle back to native cameras only if there's a specific reason.

```bash
# On-robot (Orin NX, after cloning the repo)
bash install_scripts/install_camera_server.sh
```

### 8c. Configure deploy.sh
Update `gear_sonic_deploy/scripts/setup_env.sh` and any robot-specific configs to point at:
- The X2 Ultra ONNX checkpoint
- The X2 Ultra observation config (joint count, body names)
- The X2 Ultra IP address

### 8d. First teleop run
Follow `tutorials/vr_wholebody_teleop.html` end-to-end:
1. Start camera server on-robot (auto-starts if installed as systemd service)
2. Start C++ deploy on workstation
3. Start PICO teleop streamer
4. Put on the headset, calibrate, drive the robot

### Acceptance
- Robot tracks operator's torso + arms in real-time without falling
- Latency feels acceptable (<150 ms end-to-end)
- No motor faults during a 5-minute continuous teleop session

---

## Phase 9 — Demonstration data collection

**Goal:** Record LeRobot v2.1 datasets ready for GR00T N1.7 fine-tuning.

### 9a. Setup data collection venv
```bash
bash install_scripts/install_data_collection.sh
```

### 9b. Plan the task taxonomy
Pick 3-5 simple tasks for the first round. Good first targets:
- `"pick up the cup"`
- `"hand over the cup"`
- `"place the cup on the table"`

Avoid bimanual or whole-body locomotion tasks for v0 — diagnose any teleop quirks on simple ones first.

### 9c. Record per task
```bash
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host <ROBOT_IP> \
    --task-prompt "pick up the cup" \
    --record-wrist-cameras \
    --dataset-name agibot_x2_ultra_pick_cup_v0
```

Target: ~50-200 episodes per task for an initial fine-tune. More is always better.

Recording controls:
- **Left grip + A** to start/stop episode
- **Left grip + B** to discard
- Or `c` / `x` over keyboard ZMQ

### 9d. Post-process
```bash
source .venv_data_collection/bin/activate
# Per-task cleaning
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/agibot_x2_ultra_pick_cup_v0 \
    --output-path outputs/agibot_x2_ultra_pick_cup_v0_clean

# Multi-task merge (if cross-task fine-tuning)
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/task_a_clean outputs/task_b_clean outputs/task_c_clean \
    --output-path outputs/agibot_x2_ultra_merged
```

### Acceptance
- ≥50 clean episodes per task after stale-SMPL filtering
- All episodes share the same `script_config` (merge step verifies this)
- Spot-check 5 random episodes — no glitches, no zero-pose blocks, task completes successfully on the recording

---

## Phase 10 — Register X2 Ultra as a GR00T embodiment

**Goal:** Teach Isaac-GR00T's data layer about the X2 Ultra.

GR00T N1.7 ships with pre-registered embodiments: `UNITREE_G1`, `LIBERO_PANDA`, `OXE_WIDOWX`, etc. Non-pretrained embodiments are tagged `new_embodiment`. We register a custom one.

### 10a. Modality config
Author a JSON modality config describing X2 Ultra's:
- State channels: `joint_position` (30,), `joint_velocity` (30,), `body_rotation_6d` (6,), `projected_gravity` (3,)
- Action channels: `joint_position` (30,), `body_rotation_6d` (6,)
- Image channels: `ego_view` (480, 640, 3), `left_wrist` (optional), `right_wrist` (optional)

Reference: `getting_started/finetune_new_embodiment.md` in `Isaac-GR00T`. Mirror the structure of the `UNITREE_G1` modality config but adjust dimensions.

### 10b. Embodiment tag registration
Either pass `--embodiment-tag new_embodiment` on the CLI (simplest), or register a named tag (`AGIBOT_X2_ULTRA`) in `gear_sonic/data/embodiment_tags.py` (or wherever the registry lives — check `Isaac-GR00T` source).

### 10c. Validate dataset loadability
```bash
python scripts/load_dataset.py \
    --dataset-path /path/to/outputs/agibot_x2_ultra_merged \
    --plot-state-action --video-backend torchvision_av
```

Check: state and action plots look continuous, video plays, no shape mismatches.

---

## Phase 11 — Fine-tune GR00T N1.7

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7 \
    --dataset-path /path/to/outputs/agibot_x2_ultra_merged \
    --embodiment-tag new_embodiment \
    --modality-config-path configs/agibot_x2_ultra_modality.json \
    --num-gpus 1 \
    --output-dir checkpoints/n17_agibot_x2_ultra_v0 \
    --max-steps 10000 \
    --global-batch-size 32
```

Knobs:
- `--max-steps 2000` for a quick first pass; `10000`+ for serious runs
- `--lora-rank 64 --lora-alpha 128` if VRAM-constrained (full finetune is recommended)
- `--no-tune-diffusion-model` halves VRAM by freezing the action head — only the VLM trains
- `--action-horizon 8` is the default — keep unless you have a reason

### Validation
```bash
python scripts/deployment/standalone_inference_script.py \
    --model-path checkpoints/n17_agibot_x2_ultra_v0 \
    --dataset-path /path/to/outputs/agibot_x2_ultra_merged \
    --embodiment-tag new_embodiment \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

Compare predicted vs. ground-truth actions on held-out episodes. Look for: roughly correct trajectory shape; if you see ~constant outputs, the embodiment registration is wrong.

### Acceptance
- Open-loop eval action MSE in a sane range (compare to the demo dataset numbers in N1.7 docs)
- Action distributions look like the recorded ones, not zero-centered noise

---

## Phase 12 — Closed-loop autonomous deployment

**Goal:** Replace the PICO publisher with the GR00T policy publisher. Robot acts autonomously.

### 12a. Wire up Gr00tPolicy
The Isaac-GR00T repo's `Gr00tPolicy` class wraps inference. You need to write a small server that:
1. Subscribes to the camera ZMQ stream (port 5555) on the workstation
2. Subscribes to the robot state stream from C++ deploy (port 5557)
3. Runs Gr00tPolicy on each tick (or every N ticks for action chunks)
4. Publishes the predicted joint targets / SMPL poses to the same ZMQ topic the PICO teleop streamer normally uses (port 5556)

This server replaces `pico_manager_thread_server.py` in the data-collection setup. It's ~200 lines of Python.

### 12b. (Optional) TensorRT acceleration
N1.7 has a TensorRT path documented in `Isaac-GR00T/deployment_scripts/`. On the Orin NX it's worth doing — PyTorch latency may be too high for 50 Hz control loops; TRT typically gets you 2-3× speedup.

### 12c. First autonomous run
1. Robot powered on, camera server running
2. C++ deploy running on workstation
3. Run the Gr00tPolicy server with the fine-tuned checkpoint
4. Speak/type the task prompt: `"pick up the cup"`
5. Stand back

### Acceptance
- Robot performs the task autonomously, multiple times, with reasonable success rate (anything >30% is encouraging for a v0)
- No falls, no motor faults
- Latency is stable (no growing queues)

---

## Phase 13 — Iterate

Realistic improvement loop:
1. **More data** (almost always the answer for behavioral problems)
2. **Better prompts** — try per-frame staged prompts during recording
3. **Co-train with G1 data** — the universal action interface is supposed to allow this; you can mix the released G1 demonstration data with X2 Ultra data in `LeRobotMixtureDataset`
4. **Tune the SONIC controller** — if SONIC is the bottleneck (jittery, slow tracking), retrain or finetune with more motions
5. **Camera placement experiments** — ego-view height, FOV, exposure
6. **Action chunking horizon** — try 16 vs 8 vs 4

---

## Risk register (read this before starting)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| AgiBot won't release URDF / low-level API | Medium | **Blocking** | Confirm before starting Phase 1 |
| Joint/body order mappings wrong, found late | High | High | Validation in Phase 2a is non-negotiable |
| Motor specs unavailable, KP/KD tuning is guesswork | High | Medium | Conservative damping (ratio 2-3), low natural freq |
| Retargeting quality poor due to skeleton differences | Medium | High | QA sample before full batch, iterate JSON config |
| Compute budget insufficient for from-scratch SONIC | High | Medium | Finetune from released G1 checkpoint instead |
| C++ deploy adapter to AgiBot SDK is more work than expected | High | Medium | Allocate 2-4 weeks; this is the #1 schedule risk |
| Data collection produces low-quality demos due to teleop UX issues | Medium | Medium | Pilot 5 episodes, review, refine before recording 100s |
| GR00T N1.7 doesn't transfer well from G1 pretraining | Low-Medium | Medium | Try LoRA, more data, mixed-embodiment training |

---

## Suggested timeline (one engineer, full-time)

| Phase | Optimistic | Realistic | Pessimistic |
|---|---|---|---|
| 0. Prereqs (especially AgiBot conversations) | 1 wk | 2-3 wks | blocked |
| 1. Robot model files | 2 days | 1 wk | 2 wks |
| 2. Robot config | 3 days | 1-2 wks | 3 wks |
| 3. Order converter | 1 day | 2 days | 1 wk |
| 4. Body name audit | 1 day | 3 days | 1 wk |
| 5. Motion data | 1 wk | 2-3 wks | 1 mo+ |
| 6. Experiment config | 1 day | 3 days | 1 wk |
| 7. SONIC training | 1 wk | 2-3 wks | 1 mo+ |
| 8. C++ deploy | 1 wk | 3-4 wks | blocked on SDK |
| 9. Data collection | 1 wk | 2-3 wks | 1 mo+ |
| 10. GR00T embodiment registration | 2 days | 1 wk | 2 wks |
| 11. N1.7 fine-tune | 3 days | 1 wk | 2 wks |
| 12. Autonomous deployment | 1 wk | 2 wks | 1 mo+ |
| **Total** | **~2 mo** | **~4-5 mo** | **6+ mo or blocked** |

---

## Appendix A — Useful references

- SONIC paper: https://arxiv.org/abs/2511.07820
- New embodiments guide: https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/new_embodiments.html
- Data collection guide: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html
- GR00T N1.7 blog: https://huggingface.co/blog/nvidia/gr00t-n1-7
- SOMA Retargeter: https://github.com/NVIDIA/soma-retargeter
- Bones-SEED dataset: https://huggingface.co/datasets/bones-studio/seed
- H2 reference implementation: see `gear_sonic/envs/manager_env/robots/h2.py` and `sonic_h2.yaml`
- Decoupled WBC (alternative controller, fallback option): `decoupled_wbc/` in the WBC repo

## Appendix B — H2 vs X2 Ultra topology comparison

| | Unitree H2 | AgiBot X2 Ultra |
|---|---|---|
| Total DOF | 31 | 30 |
| Bodies | 32 | 31 |
| Arm DOF (each) | 7 | 7 |
| Leg DOF (each) | 6 | 6 |
| Waist DOF | 3 | 3 |
| Neck DOF | 1 (head_yaw_link) | 1 |
| Height | similar | 1.31 m |

X2 Ultra is structurally close to H2 — close enough that **starting from `sonic_h2.yaml` rather than `sonic_release.yaml`** may be the path of least resistance. Verify body naming overlap before committing.

---

*Last updated: 2026-05-07 — initial draft. Update as decisions land.*
