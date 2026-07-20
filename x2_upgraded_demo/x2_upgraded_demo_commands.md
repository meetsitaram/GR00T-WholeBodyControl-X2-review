================================================================

## Real Robot Mode
(for sim-only mode, remove ip address args)

### Start sonic on robot pc2
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery_loose.yaml \
    --lock-head-straight

./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walk_101.yaml \
    --lock-head-straight

### run the vr planner stack
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 


### play pkl motions
python -m gear_sonic.scripts.play_locomotion \
  --pkl gear_sonic/data/motions/x2_dances_easy.pkl \
  --motion-key dance_party_hips_003__A467

### To Stop sonic on robot (*** this will collapse the robot and needs to be held)
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32
          
===================================================================

### various dance moves in kinematic viewer
-- pure kinematics
python gear_sonic/scripts/play_x2_motion_mujoco.py   --motion gear_sonic/data/motions/x2_ultra_dances_showcase.pkl   --anchor-xy

===================================================================

### validation
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --cleanup-only 

###### validate the latesrt models are on the robot
cd /home/stickbot/Projects/GR00T-WholeBodyControl

ALLOW_MISMATCH=1 \
  SONIC_MODEL=~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/exported/softland_4800_g1.onnx \
  PLANNER_MODEL=~/x2_cloud_checkpoints/planner_onnx_fixedscratch_p500k \
  ./gear_sonic/scripts/sim_onnx_planner.sh --pc2-host 192.168.86.32


ALLOW_MISMATCH=1 \
  SONIC_MODEL=~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/exported/softland_4800_g1.onnx \
  PLANNER_MODEL=~/x2_cloud_checkpoints/planner_onnx_ft \
  ./gear_sonic/scripts/sim_onnx_planner.sh --pc2-host 192.168.86.32




MODEL=~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/exported/softland_4800_g1.onnx \
  ./gear_sonic/scripts/sim_onnx_planner.sh --pc2-host 192.168.86.32





### various dance moves with sonic in mujoco

cd /home/stickbot/Projects/GR00T-WholeBodyControl
.venv/bin/python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint  /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/model_step_004800.pt \
  --wrist-ref --motions gear_sonic/data/motions/x2_dances_easy.pkl \
  --clip dance_party_hips_003__A467

--checkpoint /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/model_step_002000.pt
--checkpoint  /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/model_step_004800.pt
--onnx -onnx /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/exported/softland_4800_g1.onnx


cd /home/stickbot/Projects/GR00T-WholeBodyControl
.venv/bin/python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/model_step_004800.pt \
  --wrist-ref --motions gear_sonic/data/motions/x2_dances_easy.pkl



-- medium dance_latino_chase_mambo_kicks_R_fast_001__A314 (needs space - feet hitting each other on the robot, but ok)
-- medium dance_retro_jazz_cross_step_180_R_001__A314 (needs space in front)
-- medium dance_western_country_lasso_R_fast_002__A306 (needs space to the right)

-- difficult dance_retro_twist_step_variation_R_fast_002__A314 (in place)
-- difficult dance_latino_chase_mambo_pivot_R_001__A313 (needs extra side space)
-- difficult dance_basic_turn_v1_270_R_002__A309 (needs extra side space)
-- most difficult dance_retro_twist_step_variation_R_fast_002__A314 (single leg stance) (needs space)

/home/stickbot/x2_cloud_checkpoints/dance_v1_3250_full/dance_warmstart_3250.pt
/home/stickbot/x2_cloud_checkpoints/dance_175030_step2000_full/model_step_002000.pt

cd /home/stickbot/Projects/GR00T-WholeBodyControl
.venv/bin/python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint /home/stickbot/x2_cloud_checkpoints/dance_175030_step2000_full/model_step_002000.pt \
  --wrist-ref --motions gear_sonic/data/motions/x2_dances_easy.pkl 


  /home/stickbot/x2_cloud_checkpoints/dance_v3_3k_full/dance_v3_3k.pt
  /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/model_step_002000.pt
  ~/x2_cloud_checkpoints/g1teleop_overnight/sonic/snapshots/exported/ft_2082_g1.onnx

gear_sonic/data/motions/x2_dances_easy.pkl 
gear_sonic/data/motions/x2_dances_medium.pkl 
gear_sonic/data/motions/x2_dances_difficult.pkl 

### lower volume
./gear_sonic_deploy/scripts/x2_pc3_audio.sh volume 70 --pc2-host 192.168.86.32


### Easy (14)
dance_hiphop_stick_n_roll_dancehall_R_loop_003__A324
dance_vouge_butterfly_step_180_R_fast_002__A319
dance_vouge_vouge_pose_iv_R_001__A316
egipt_dance_R_001__A438
victory_dance_asarahe_180_R_004__A324
dance_distraction_dance_001__A466
dance_distraction_dance_001__A466_M
dance_freedom_wheels_001__A464
dance_freedom_wheels_001__A464_M
dance_freedom_wheels_001__A465
dance_freedom_wheels_001__A465_M
dance_party_hips_003__A464
dance_party_hips_003__A465
dance_party_hips_003__A467

### Medium (17)
dance_disco_fever_001__A465
dance_hiphop_bart_simpson_R_fast_001__A319
dance_hiphop_funky_guitar_R_fast_001__A319
dance_hiphop_indiana_step_double_R_003__A314
dance_latino_chase_mambo_kicks_R_fast_001__A314
dance_latino_kick_kick_R_001__A313
dance_retro_jazz_cross_step_180_R_001__A314
dance_western_country_lasso_R_fast_002__A306
dance_dip_001__A465
dance_distraction_dance_001__A464
dance_distraction_dance_001__A464_M
dance_distraction_dance_001__A467
dance_im_diamond_002__A464
dance_im_diamond_002__A464_M
dance_im_diamond_002__A465
dance_im_diamond_002__A465_M
dance_im_diamond_002__A466

### Difficult (10)
dance_basic_turn_v1_270_R_002__A309
dance_basic_turn_v1_180_R_loop_fast_003__A324
dance_basic_turn_v1_360_R_loop_fast_004__A322
dance_jazz_hands_002__A467
dance_latino_chase_mambo_pivot_R_001__A313
dance_retro_twist_step_variation_R_fast_002__A314
dance_statuesque_opt_2_002__A467
krakowiak_R_002__A100
lasso_dance_R_002__A167
dance_distraction_dance_001__A465



### misc
KPLANNER_FIXED_FWD_MPS=0.5 ./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
  --duration 0 \
  --model /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/snapshots/exported/ft_2082_g1.onnx \
  --kplanner-vqvae-ckpt /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/kplanner/vqvae/checkpoints/model-step=0250000.ckpt \
  --kplanner-pose-ckpt  /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/kplanner/pose/checkpoints/model-step=0080000.ckpt \
  --kplanner-root-ckpt  /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/kplanner/root/checkpoints/model-step=0250000.ckpt \
  --kplanner-planner-mode walk \
  --kplanner-python /home/stickbot/miniconda3/envs/env_isaaclab/bin/python


# planner (ONNX) + your pad bridge + sim consumer, standard sim ports (sim on pc)
cd ~/Projects/GR00T-WholeBodyControl

# ONNX planner (the deployed artifact) + MuJoCo sim + local pad
./gear_sonic/scripts/sim_onnx_planner.sh



===================================================================
### BEST KNOWN WALK CONFIG (2026-07-16 evening, user-validated "most decent walk so far")
# slow_walk template (baked from our slow_walk_0.3_001 recording) + 0.3 setpoint.
# X/Y buttons in locomotion adjust speed -/+0.1 (range 0.2-1.0); changes logged
# as "speed setpoint -> X" + per-replan "Replanning ... target_vel(fwd)".
KPLANNER_FIXED_FWD_MPS=0.3 ./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
  --duration 0 \
  --model /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/snapshots/exported/walkft_3065_g1.onnx \
  --kplanner-vqvae-ckpt /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/kplanner/vqvae/checkpoints/model-step=0250000.ckpt \
  --kplanner-pose-ckpt  /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/kplanner/pose/checkpoints/model-step=0080000.ckpt \
  --kplanner-root-ckpt  /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/kplanner/root/checkpoints/model-step=0250000.ckpt \
  --kplanner-planner-mode slow_walk \
  --kplanner-python /home/stickbot/miniconda3/envs/env_isaaclab/bin/python
# NOTE: pose ckpt is the 80k intermediate -- swap to the 250k final tomorrow morning.
# Template pairing guide: slow_walk<->0.2-0.4, walk<->0.4-0.6, run_proxy<->0.7-1.0.

===================================================================
### PAD-ONLY DRIVING (2026-07-16, WORKING — "it moved. yehh")
# Terminal 1 — stack with gamepad instead of Quest (add --pc2-host 192.168.86.32 for real robot):
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
  --duration 0 \
  --pad-only \
  --model /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/sonic/snapshots/exported/walkft_3065_g1.onnx \
  --kplanner-vqvae-ckpt /home/stickbot/x2_cloud_checkpoints/kplanner_g1ret/vqvae/vqvae_g1ret_250k.ckpt \
  --kplanner-pose-ckpt  /home/stickbot/x2_cloud_checkpoints/kplanner_g1ret/pose/pose_g1ret_250k.ckpt \
  --kplanner-root-ckpt  /home/stickbot/x2_cloud_checkpoints/g1teleop_overnight/kplanner/root/checkpoints/model-step=0250000.ckpt \
  --kplanner-python /home/stickbot/miniconda3/envs/env_isaaclab/bin/python
# Terminal 2 — dance/gesture buttons + e-stop chord:
.venv/bin/python gear_sonic/scripts/play_xbox_controller.py
# CONTROLS: hold L2+R2 = deadman -> L-stick walk, R-stick turn; L1/R1 tap = speed -/+0.1
#           (triggers released) A/B/X/Y +modifiers = gestures/dances; L1+R1+L2+R2 = E-STOP
#           avoid D-pad while triggers held (fires canned walk clips)
# Robot AP staged: aima nm start-ap  -> SSID X2-ROBOT / x2demo2026 (stop: aima nm stop-ap)

# 2026-07-16 addendum: unified idle/walk arm pose (fixes idle<->walk arm snap):
#   add to any planner stack launch:
#   --kplanner-warmup-qpos /home/stickbot/Projects/GR00T-WholeBodyControl/gear_sonic/data/motions/kplanner_idle_anchor_g1teleop.pkl
#   (anchor = slow_walk_0.3_001 frame 61 = the slow_walk template seed frame;
#    NOTE env var KPLANNER_WARMUP_QPOS does NOT work -- the stack resets it; CLI flag only.
#    Polish TODO: update PC2 idle_stand.x2m2 to match, so feed-stall blends stay in-stance.)
