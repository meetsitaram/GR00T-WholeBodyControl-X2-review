#!/usr/bin/env bash
# Baseline sanity: native clip playback, NO planner/ring/idle-writer.
# If this walks and the planner rig doesn't, the fault is in the planner
# path — the split that root-caused the base-family bug (2026-07-21).
set -euo pipefail
cd "$(dirname "$0")/../../.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

SONIC_DIR=~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528
MOTION="${1:-gear_sonic/data/motions/x2_ultra_walk_forward.pkl}"

DISPLAY="${DISPLAY:-:1}" \
~/projects/g1-kitchen-sim/.venv/bin/python -u -m gear_sonic.scripts.eval_x2_isaacsim_onnx \
  --run-dir "$SONIC_DIR" \
  --onnx "$SONIC_DIR/exported/softland_4800_g1.onnx" \
  +num_envs=1 +headless=False \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION" \
  2>&1 | tee /tmp/claude-1000/eval_walk_user.log
