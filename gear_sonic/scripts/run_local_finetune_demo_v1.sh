#!/bin/bash
# Local single-GPU fine-tune of the X2 Ultra SONIC policy on the demo-v1
# motion corpus (101 entries: body_check + combat + dances + mc_gestures +
# retargeted walks + sitstand). Warm-starts from the H200 25k sphere-feet
# checkpoint and runs 4k more PPO iterations.
#
# Sized for a 32 GB RTX 5090:
#   - num_envs       = 3072
#   - num_iterations = 4000
#   - num_processes  = 1 (single-GPU)
#   - wandb          = True (logged to TRL_X2Ultra_DemoV1 project)
#
# Usage (detached, no tmux needed):
#
#   setsid nohup bash gear_sonic/scripts/run_local_finetune_demo_v1.sh \
#       </dev/null >/dev/null 2>&1 &
#   echo $! > ~/sonic_demo.pid
#   tail -f ~/sonic_demo.log
#
# The training script handles its own logging via tee to LOG_FILE inside
# run_smoke_8gpu.sh, so don't redirect again from outside (would double
# every line in the log file).
#
# Stop with: kill $(cat ~/sonic_demo.pid)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LAUNCH_TS=$(date +%Y%m%d_%H%M%S)

export MOTION_FILE="gear_sonic/data/motions/x2_ultra_demo_v1.pkl"
export EXP_NAME="sonic_x2_ultra_demo_v1"
export NUM_PROCESSES=1
export NUM_ENVS=3072
export NUM_ITERS=4000
export USE_WANDB=True
export EXTRA_FLAGS="+checkpoint=$HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt"
export LOG_FILE="$HOME/sonic_demo_${LAUNCH_TS}.log"

# Refresh the convenience symlink so `tail -f ~/sonic_demo.log` always tails
# the most recent run.
ln -sfn "$LOG_FILE" "$HOME/sonic_demo.log"

exec bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
