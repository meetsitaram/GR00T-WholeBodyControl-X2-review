#!/bin/bash
# Local single-GPU fine-tune of the X2 Ultra SONIC policy on the demo-v2
# motion corpus (95 entries: demo-v1 minus the 6 sit-on-chair variants).
# Continues from the demo-v1 final checkpoint (model_step_004000.pt) and
# adds another 4k PPO iterations under a wider domain-randomization
# envelope (2× observation noise + ±15% per-episode KP/KD scaling).
#
# Sized for a 32 GB RTX 5090:
#   - num_envs       = 3072
#   - num_iterations = 4000
#   - num_processes  = 1 (single-GPU)
#   - wandb          = True (logged to TRL_X2Ultra_DemoV2 project)
#
# Usage (detached, no tmux needed):
#
#   setsid nohup bash gear_sonic/scripts/run_local_finetune_demo_v2.sh \
#       </dev/null >/dev/null 2>&1 &
#   echo $! > ~/sonic_demo_v2.pid
#   tail -f ~/sonic_demo_v2.log
#
# The training script handles its own logging via tee to LOG_FILE inside
# run_smoke_8gpu.sh, so don't redirect again from outside (would double
# every line in the log file).
#
# Stop with: kill $(cat ~/sonic_demo_v2.pid)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LAUNCH_TS=$(date +%Y%m%d_%H%M%S)

# Continue from the demo-v1 final checkpoint. If you ever need to restart
# from the warm-start instead, point CHECKPOINT at the H200 25k sphere-feet
# checkpoint and rename the run directory in the project name.
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/logs_rl/TRL_X2Ultra_DemoV1/manager/universal_token/all_modes/sonic_x2_ultra_demo_v1_demo_v1-20260623_231221/model_step_004000.pt}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: warm-start checkpoint not found: $CHECKPOINT" >&2
    echo "       Set CHECKPOINT=<path> to override." >&2
    exit 1
fi

export MOTION_FILE="gear_sonic/data/motions/x2_ultra_demo_v2.pkl"
export EXP_NAME="sonic_x2_ultra_demo_v2"
export NUM_PROCESSES=1
export NUM_ENVS=3072
export NUM_ITERS=4000
export USE_WANDB=True
export EXTRA_FLAGS="+checkpoint=$CHECKPOINT"
export LOG_FILE="$HOME/sonic_demo_v2_${LAUNCH_TS}.log"

ln -sfn "$LOG_FILE" "$HOME/sonic_demo_v2.log"

exec bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
