
### teleop  or record episode
```sh
 .venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic     --output-dir data/lerobot/x2_quest3_kinematic_v4     --task "yaw-sweep diagnostic"     --rate 50     --hand-input max
```

### replay episode
```sh
.venv/bin/python -m gear_sonic.scripts.replay_x2_kinematic \
    --dataset x2_quest3_kinematic_v4 \
    --episode 0

# Body-only (skip OmniHand augmentation)
.venv/bin/python -m gear_sonic.scripts.replay_x2_kinematic \
    --dataset x2_quest3_kinematic_v4 --episode 0 --no-omnihand
```

### build bones seed motion lib files
```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/data_process/build_x2_bones_seed_motion_lib.py
```

###standing gestures showcase launch in sim
```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
conda run -n env_isaaclab --no-capture-output python gear_sonic/scripts/eval_x2_mujoco.py \
    --checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.pt \
    --playlist gear_sonic/data/motions/playlists/showcase_v1.yaml
```