================================================================

## Real Robot Mode
(for sim-only mode, remove ip address args)

### Start sonic on robot pc2
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery_loose.yaml \
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

### various dance moves with sonic in mujoco

cd /home/stickbot/Projects/GR00T-WholeBodyControl
.venv/bin/python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint /home/stickbot/x2_cloud_checkpoints/dance_v1_3250_full/dance_warmstart_3250.pt \
  --wrist-ref --motions gear_sonic/data/motions/x2_dances_easy.pkl \
  --clip dance_party_hips_003__A467

/home/stickbot/x2_cloud_checkpoints/dance_175030_step2000_full/model_step_002000.pt

cd /home/stickbot/Projects/GR00T-WholeBodyControl
.venv/bin/python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint /home/stickbot/x2_cloud_checkpoints/dance_175030_step2000_full/model_step_002000.pt \
  --wrist-ref --motions gear_sonic/data/motions/x2_dances_easy.pkl 

gear_sonic/data/motions/x2_dances_easy.pkl 
gear_sonic/data/motions/x2_dances_medium.pkl 
gear_sonic/data/motions/x2_dances_difficult.pkl 
                                

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