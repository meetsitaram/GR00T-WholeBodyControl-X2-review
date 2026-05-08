#!/bin/bash
# 8-GPU smoke test for the X2 Ultra training stack.
#
# Verifies the full pipeline (Isaac Sim init -> env build -> Hydra config ->
# accelerate distributed launch -> PPO loop) on every GPU of a cloud node,
# without committing to a multi-hour real run.
#
# Defaults are tuned for an 8x H200 (or H100 80GB) SXM node:
#   - 200 PPO iterations  (~3 min wall-clock on H200, ~$2 of GPU time)
#   - 4096 environments per GPU (32K total)
#   - W&B disabled
#   - motion library = stand-still smoke PKL produced by
#     ``gear_sonic/scripts/cloud/build_stand_idle_smoke.py``
#
# Usage (run on the cloud node, from the repo root):
#
#   # one-time: build the smoke PKL from the unpacked BONES-SEED bundle
#   python gear_sonic/scripts/cloud/build_stand_idle_smoke.py
#
#   # launch the smoke (detached, in tmux):
#   tmux new -d -s smoke "bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh"
#   tmux a -t smoke         # attach to watch
#   tail -f ~/smoke.log     # ...or tail the log file
#
# Override knobs:
#   NUM_PROCESSES   number of GPUs                      (default: 8)
#   NUM_ENVS        envs per GPU                        (default: 4096)
#   NUM_ITERS       PPO iterations                      (default: 200)
#   MOTION_FILE     motion-lib PKL                      (default: stand-idle smoke)
#   USE_WANDB       True/False                          (default: False)
#   LOG_FILE        where to tee stdout                 (default: ~/smoke.log)
#   EXP_NAME        Hydra +exp= leaf                    (default: sonic_x2_ultra_bones_seed)
#                   Resolved as +exp=manager/universal_token/all_modes/$EXP_NAME
#   EXTRA_FLAGS     additional Hydra flags appended raw (default: empty)
#                   e.g. EXTRA_FLAGS="+checkpoint=/path/to/model_step_NNNNNN.pt"
#                        EXTRA_FLAGS="++use_wandb=True ++algo.config.lr=5e-5"

set -euo pipefail

NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_ENVS=${NUM_ENVS:-4096}
NUM_ITERS=${NUM_ITERS:-200}
MOTION_FILE=${MOTION_FILE:-gear_sonic/data/motions/x2_ultra_stand_idle_smoke.pkl}
USE_WANDB=${USE_WANDB:-False}
LOG_FILE=${LOG_FILE:-$HOME/smoke.log}
EXP_NAME=${EXP_NAME:-sonic_x2_ultra_bones_seed}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date) === SMOKE START"
echo "  num_processes : $NUM_PROCESSES"
echo "  num_envs/proc : $NUM_ENVS"
echo "  iterations    : $NUM_ITERS"
echo "  motion_file   : $MOTION_FILE"
echo "  use_wandb     : $USE_WANDB"
echo "  exp_name      : $EXP_NAME"
echo "  extra_flags   : ${EXTRA_FLAGS:-<none>}"

# Activate env_isaaclab (matches the install in train-on-cloud.md).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

# Repo root is two levels up from this script (gear_sonic/scripts/cloud/...).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
echo "  repo root     : $REPO_ROOT"

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "ERROR: motion file not found: $MOTION_FILE"
  echo "       run: python gear_sonic/scripts/cloud/build_stand_idle_smoke.py"
  exit 1
fi

# Mitigate fragmentation OOMs and pre-accept Omniverse EULAs (no-op if already
# accepted; safe to set every launch).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

# shellcheck disable=SC2086  # EXTRA_FLAGS is intentionally word-split.
accelerate launch --num_processes="$NUM_PROCESSES" \
  gear_sonic/train_agent_trl.py \
  --config-name=base \
  "+exp=manager/universal_token/all_modes/$EXP_NAME" \
  ++num_envs="$NUM_ENVS" \
  ++headless=True \
  ++use_wandb="$USE_WANDB" \
  ++algo.config.num_learning_iterations="$NUM_ITERS" \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION_FILE" \
  $EXTRA_FLAGS

echo "=== $(date) === SMOKE DONE"
