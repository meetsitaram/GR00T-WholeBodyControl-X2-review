

### launch in mujoco
cd ~/Projects/GR00T-WholeBodyControl
gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
  --model ~/x2_cloud_checkpoints/arm_dynamics_v3_1500/exported/model_step_001500_g1.onnx
  
cd ~/Projects/GR00T-WholeBodyControl
python -m gear_sonic.scripts.play_locomotion \
  --pkl gear_sonic/data/motions/g1_recorded_x2/slow_walk_slow_keyboard_001.pkl


cd ~/Projects/GR00T-WholeBodyControl
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
  --model ~/x2_cloud_checkpoints/arm_dynamics_v3_1500/exported/model_step_001500_g1.onnx

### ealier models
  ~/x2_eval_tracking/slow_motion_test/slow_manip_iter3000_g1.onnx

# X2
### Terminal 1 — start the X2 stack with the 2k model:
gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
  --model ~/x2_cloud_checkpoints/arm_dynamics_v3_1500/exported/model_step_001500_g1.onnx

### Terminal 2 — feed any X2 motion PKL:
python -m gear_sonic.scripts.play_locomotion --pkl <X2_clip.pkl>

# G1
cd ~/Projects/GR00T-WholeBodyControl
what are you doing? we already 
### One-time per clip — convert a G1 motion PKL -> deploy reference CSVs:
.venv_sim/bin/python gear_sonic/scripts/training_pkl_to_deploy_csv.py --pkl <G1_clip.pkl> --fps 50
#   (writes gear_sonic_deploy/reference/<clip>/)

### Terminal 1 — G1 sim:
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

### Terminal 2 — G1 deploy (stock SONIC) playing the motion:
cd gear_sonic_deploy && source scripts/setup_env.sh

bash deploy.sh sim --input-type keyboard
bash deploy.sh sim --motion-data reference/<clip>/

### T3  (capture REAL world root xyz + joints -> motion-lib pkl)
.venv/bin/python gear_sonic/scripts/record_motion_to_pkl.py \
  --robot g1 --out gear_sonic/data/motions/g1_recorded_x2/sample_kplanner_motions.pkl \
  --motion-key walk_001 --duration N

### slow walk tests
cd ~/Projects/GR00T-WholeBodyControl
gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
  --model ~/x2_cloud_checkpoints/arm_dynamics_v3_1500/exported/model_step_001500_g1.onnx

cd ~/Projects/GR00T-WholeBodyControl
python -m gear_sonic.scripts.play_locomotion \
  --pkl gear_sonic/data/motions/g1_recorded_x2/slow_walk_slow_keyboard_001.pkl

### --------------


Relaxed_walk_forward_001__A057
body_search_001__A054
big_heavy_one_hand_behind_high_to_behind_high_R_001__A524
change_idle_crouch_right_to_idle_crouch_101__A125
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight


kneeling_start_001__A021
kneeling_loop_001__A025_M
kneeling_stop_001__A025_M

cigarette_pick_up_R_002__A459_M

cd /home/stickbot/Projects/GR00T-WholeBodyControl
gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
  --model ~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/exported/model_step_001376_g1.onnx

-- model fine tuned with recorded motions via g1 kplanner teleop
cd /home/stickbot/Projects/GR00T-WholeBodyControl
gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
  --model ~/x2_cloud_checkpoints/teleop_finetune_v1/exported/model_step_002000_g1.onnx



python -m gear_sonic.scripts.play_locomotion \
  --pkl gear_sonic/data/motions/g1_recorded_x2/slow_walk_slow_keyboard_001.pkl 

-- compare g1 and x2 planner outputs
source .venv/bin/activate
PYTHONPATH="${PWD}/motionbricks:${PWD}" python motionbricks/scripts/view_e2e_x2_vs_g1.py \
  --x2-npz out/e2e_headtohead/e2e_x2_forward.npz \
  --g1-npz out/e2e_headtohead/e2e_g1_forward.npz

### launch in isacclab
cd ~/Projects/GR00T-WholeBodyControl
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y DISPLAY=:1
PY=~/miniconda3/envs/env_isaaclab/bin/python
CKPT=~/x2_cloud_checkpoints/executed_feasible_v1/sonic_x2_ultra_executed_feasible_executed_feasible_v1-20260712_190745/model_step_002000.pt
MOTION=~/x2_eval_tracking/slow_motion_test/slow_walk_slow_keyboard_001.pkl


CKPT=~/x2_cloud_checkpoints/slow_manip_focus_iter3000/model_step_003000.pt

$PY gear_sonic/eval_agent_trl.py \
  +checkpoint=$CKPT \
  ++headless=False ++num_envs=1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION



### launch kinematic viewer
cd ~/Projects/GR00T-WholeBodyControl
export MUJOCO_GL=glfw DISPLAY=:1
python gear_sonic/scripts/play_x2_motion_mujoco.py \
  --motion gear_sonic/data/motions/demo_v1_sources/combat_chain_matched/shadow_boxing_R_003__A359__x2_chain_matched.pkl



### slow motion training setup
Focused corpus: gear_sonic/data/motions/x2_slow_manip_focus.pkl — the 34 teleop clips (27 slow/manip + 7 walks), ~14 min of motion.
Config: sonic_x2_ultra_slow_manip_focus.yaml — inherits executed-feasible, changes only:
tracking_body_linvel std 1.0 → 0.25 (the dead-band fix)
tracking_anchor_pos weight 0.5 → 2.0, std 0.3 → 0.15 (catch-up signal)
small-subset tweaks: uniform_sampling_rate 0.1 → 0.25, save_interval 500 → 250, num_steps_per_env 24 → 32
Warm-start: the 2k checkpoint.

### commands - vr mode
cd /home/stickbot/Projects/GR00T-WholeBodyControl
source .venv_sim/bin/activate
export GEAR_SONIC_ROBOT_POSE_ZMQ_PORT=5570
python gear_sonic/scripts/run_sim_loop.py

cd /home/stickbot/Projects/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim --input-type zmq_manager


cd /home/stickbot/Projects/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/quest3_manager_thread_server.py


cd /home/stickbot/Projects/GR00T-WholeBodyControl
.venv/bin/python gear_sonic/scripts/record_motion_to_pkl.py \
  --robot g1 --out data_backups/stride_ab/vr_slowwalk_0p3.pkl \
  --motion-key vr_slowwalk_0p3 --duration 12


--record
cd /home/stickbot/Projects/GR00T-WholeBodyControl
.venv/bin/python gear_sonic/scripts/record_motion_to_pkl.py \
  --robot g1 --out data_backups/stride_ab/vr_slowwalk_0p3.pkl \
  --motion-key vr_slowwalk_0p3 --duration 25

REC() { .venv_teleop/bin/python gear_sonic/scripts/record_motion_to_pkl.py   --robot g1 --g1-csv /tmp/g1_ft/$1.csv --out /tmp/g1_ft/$1.pkl --duration 25 --motion-key $1; }

REC slow_walk_slow_vr_001
REC slow_walk_slow_vr_002
REC slow_walk_medium_vr_001
REC slow_walk_medium_vr_002
REC walk_vr_001
REC walk_vr_002
REC walk_vr_003


### TODO — validate later (x2_kplanner changes, not yet live-tested)
Launch the X2 VR stack (see "commands - vr mode" above) and check:
1. VR forward velocity is now DISCRETE: snaps to 0.3 or 0.5 m/s only (was
   continuous 0.3–0.6). Hysteresis deadband around stick-mag 0.6 (±0.04) ->
   confirm no flicker between 0.3/0.5 when the stick hovers near the edge.
   Backward / lateral / yaw stay analog.
2. (pending impl) kplanner warmup/idle pose -> neutral_idle_loop_001__A076.
   Expect a nicer idle standing pose (natural arms), BUT a one-time ~16°
   arm settle + 4.3 cm hip rise at planner engage (SAFE_IDLE=training_default
   -> warmup mismatch). Confirm that startup settle is acceptable.
   NOTE: to make it zero-settle later, also set the C++ deploy SAFE_IDLE PD
   target to neutral_idle_loop.

# also flagged to validate: ~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/exported/model_step_001376_g1.onnx


### Retraining run Spec
Config: sonic_x2_ultra_slow_manip_rebalance.yaml
Full corpus (35,974 clips), warm-start from 2k
Reward fix: linvel std 0.25, anchor_pos 2.0/0.15
Sampler guards (your call): uniform_sampling_rate 0.25 + use_failure_rate_decay true — auto-focus the slow clips without re-cratering general feasibility
8 procs × 12,288 envs, 30k iters, W&B on


### encoder logs

Setting encode_mode for 13 loaded reference motions...
  Motion 'tired_forward_lunge_R_001__A359_M' encode_mode set to: 0
  Motion 'dance_in_da_party_001__A464_M' encode_mode set to: 0
  Motion 'dance_in_da_party_001__A464' encode_mode set to: 0
  Motion 'neutral_kick_R_001__A543' encode_mode set to: 0
  Motion 'tired_one_leg_jumping_R_001__A359' encode_mode set to: 0
  Motion 'macarena_001__A545' encode_mode set to: 0
  Motion 'macarena_001__A545_M' encode_mode set to: 0
  Motion 'neutral_kick_R_001__A543_M' encode_mode set to: 0
  Motion 'forward_lunge_R_001__A359_M' encode_mode set to: 0
  Motion 'squat_001__A359' encode_mode set to: 0
  Motion 'walking_quip_360_R_002__A428' encode_mode set to: 0
  Motion 'tired_one_leg_jumping_R_001__A359_M' encode_mode set to: 0
  Motion 'walking_quip_360_R_002__A428_M' encode_mode set to: 0
Planner motion encode_mode set to: 0

