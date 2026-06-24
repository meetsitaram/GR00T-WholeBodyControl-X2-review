### run in kinematic viewer
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl \
    --no-loop

### run with sonic in sim directly (without full stack piipeline)

##### v2 — walk forward -> turn 90 -> turn 90 -> walk back (~34.5 s; ends ~29 cm off origin)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_walk_demo_v2.pkl \
    --sim-viewer --no-confirm \
    --max-duration 40


### run with sonic in sim and full stack
gear_sonic/scripts/run_x2_pkl_direct_stack.sh

python -m gear_sonic.scripts.play_locomotion --pkl gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl
