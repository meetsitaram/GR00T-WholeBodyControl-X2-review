#!/bin/bash
# Root fine-tune Round 3 — continuation of step-300K -> step-315K loco fine-tune.
#
# Why this exists: the previous fine-tune resumed from step-300K and ran the
# original cosine schedule's dead tail (LR ~ 2e-6) for 15K steps. Open-loop
# forward-walk metrics did improve (dy_err 0.39 -> 0.15 m, dyaw_err 27 deg -> 0
# deg) but closed-loop sim showed regressions (fwd throughput -18%, lateral
# drift +1.3 m). Continuing the same recipe further would do nothing because
# the cosine schedule is exhausted.
#
# This script does a weights-only init from the 315K checkpoint, then starts
# a fresh optimizer + LR schedule with a much smaller peak LR. The intent is
# a gentle continuation that keeps the yaw gains and tries to claw back the
# forward throughput / lateral stability we lost.
#
# Launch (on the cloud node):
#   tmux new -d -s root_ft2 "bash motionbricks/scripts/cloud/run_root_finetune_v2.sh"
#   tmux a -t root_ft2     # attach to watch
#
# Outputs:
#   - new ckpts: motionbricks/out/motionbricks_root_x2/version_1/checkpoints_ft2/
#   - log:       ~/root_ft2.log
#   - W&B run:   TRL_X2Ultra_Planner / root_x2_ft2_lr1e5_from315k

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate motionbricks

INIT_FROM=${INIT_FROM:-motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0315000.ckpt}
PKL=${PKL:-gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2.pkl}
FILTER=${FILTER:-loco}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_EVERY=${SAVE_EVERY:-1000}
BATCH_PER_GPU=${BATCH_PER_GPU:-32}
NUM_WORKERS=${NUM_WORKERS:-6}
MIN_FRAMES=${MIN_FRAMES:-80}
MAX_FRAMES=${MAX_FRAMES:-200}
NUM_GPUS=${NUM_GPUS:-8}
PEAK_LR=${PEAK_LR:-1.0e-5}        # 10% of the original 1e-4 — gentle continuation
WARMUP_STEPS=${WARMUP_STEPS:-1000} # short warmup, model is already trained
FINAL_LR=${FINAL_LR:-5.0e-7}      # cosine floor: ~5% of peak, still has gradient signal
CKPT_SUBDIR=${CKPT_SUBDIR:-checkpoints_ft2}
WANDB_NAME=${WANDB_NAME:-root_x2_ft2_lr1e5_from315k}
LOG_FILE=${LOG_FILE:-$HOME/root_ft2.log}

if [[ ! -f "$INIT_FROM" ]]; then
  echo "ERROR: --init-from ckpt not found: $INIT_FROM" >&2
  exit 1
fi
if [[ ! -f "$PKL" ]]; then
  echo "ERROR: PKL not found: $PKL" >&2
  exit 1
fi

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date) === ROOT FT2 START"
echo "  init_from    : $INIT_FROM"
echo "  pkl          : $PKL"
echo "  filter       : $FILTER"
echo "  max_steps    : $MAX_STEPS  (fresh counter, ckpts: step=0001000..step=00${MAX_STEPS}.ckpt)"
echo "  save_every   : $SAVE_EVERY"
echo "  ckpt_subdir  : $CKPT_SUBDIR"
echo "  peak_lr      : $PEAK_LR"
echo "  warmup_steps : $WARMUP_STEPS"
echo "  final_lr     : $FINAL_LR"
echo "  batch/gpu    : $BATCH_PER_GPU"
echo "  num_workers  : $NUM_WORKERS"
echo "  frames       : [$MIN_FRAMES, $MAX_FRAMES]"
echo "  num_gpus     : $NUM_GPUS"
echo "  wandb_name   : $WANDB_NAME"
echo "  log_file     : $LOG_FILE"

# Thread caps matching run_planner_train.sh (avoid 21*nproc thread storm).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29501}  # different from prior run, in case it's still bound

python motionbricks/scripts/train_root_x2.py \
  --pkl "$PKL" \
  --filter "$FILTER" \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_PER_GPU" \
  --num_workers "$NUM_WORKERS" \
  --min_frames "$MIN_FRAMES" \
  --max_frames "$MAX_FRAMES" \
  --save-every-n-steps "$SAVE_EVERY" \
  --ckpt-subdir "$CKPT_SUBDIR" \
  --devices "$NUM_GPUS" \
  --num-nodes 1 \
  --no-progress-bar \
  --init-from "$INIT_FROM" \
  --peak-lr "$PEAK_LR" \
  --warmup-steps "$WARMUP_STEPS" \
  --final-lr "$FINAL_LR" \
  --use-wandb \
  --wandb-project TRL_X2Ultra_Planner \
  --wandb-name "$WANDB_NAME"

echo "=== $(date) === ROOT FT2 DONE"
