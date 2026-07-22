source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd ~/Projects/GR00T-WholeBodyControl
KP_PAD=1 KP_HIDE_TERRAIN=1 DISPLAY=:1 ~/projects/g1-kitchen-sim/.venv/bin/python -u -m gear_sonic.scripts.run_x2_kplanner_env2 \
  --run-dir ~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528 \
  --onnx ~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/exported/softland_4800_g1.onnx \
  --vqvae-ckpt ~/x2_cloud_checkpoints/fixed_scratch/vqvae/model-step=0300000.ckpt \
  --pose-ckpt ~/x2_cloud_checkpoints/fixed_scratch/pose_500k/model-step=0500000.ckpt \
  --root-ckpt ~/x2_cloud_checkpoints/fixed_scratch/root/model-step=0300000.ckpt \
  --planner-mode slow_walk \
  +num_envs=1 +headless=False \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/motions/x2_sonic_feasible_stand_single.pkl \
  ++manager_env.config.world_usd=/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_splat.usdz \
  ++manager_env.config.world_collision_usd=/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_collision.usd \
  '++manager_env.config.world_pos=[-19.99,-75.96,0.0]' \
  ++manager_env.terminations.ee_body_pos=null \
  ++manager_env.terminations.foot_pos_xyz=null \
  ++manager_env.terminations.anchor_ori_full=null \
  2>&1 | tee /tmp/claude-1000/kpe2_kitchen.log


source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd ~/Projects/GR00T-WholeBodyControl
KP_PAD=1 KP_HIDE_TERRAIN=1 KP_HEAD_CAM=1 DISPLAY=:1 ~/projects/g1-kitchen-sim/.venv/bin/python -u -m gear_sonic.scripts.run_x2_kplanner_env2 \
  --run-dir ~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528 \
  --onnx ~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/exported/softland_4800_g1.onnx \
  --vqvae-ckpt ~/x2_cloud_checkpoints/fixed_scratch/vqvae/model-step=0300000.ckpt \
  --pose-ckpt ~/x2_cloud_checkpoints/fixed_scratch/pose_500k/model-step=0500000.ckpt \
  --root-ckpt ~/x2_cloud_checkpoints/fixed_scratch/root/model-step=0300000.ckpt \
  --planner-mode slow_walk \
  +num_envs=1 +headless=False \
  ++manager_env.config.enable_cameras=true \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/motions/x2_sonic_feasible_stand_single.pkl \
  ++manager_env.config.world_usd=/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_splat.usdz \
  ++manager_env.config.world_collision_usd=/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_collision.usd \
  '++manager_env.config.world_pos=[-19.99,-75.96,0.0]' \
  ++manager_env.terminations.ee_body_pos=null \
  ++manager_env.terminations.foot_pos_xyz=null \
  ++manager_env.terminations.anchor_ori_full=null \
  2>&1 | tee /tmp/claude-1000/kpe2_kitchen.log

KP_REPLAN_THRESH=32
