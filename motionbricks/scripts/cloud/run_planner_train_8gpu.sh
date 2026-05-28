#!/bin/bash
# Multi-GPU training pipeline for the X2 MotionBricks planner.
#
# Sequentially trains the three model components on a single node:
#   1. VQVAE     (motion tokenizer, no other model dependency)
#   2. Pose      (transformer backbone, conditions on the trained VQVAE)
#   3. Root      (continuous root motion, independent of VQVAE)
#
# The pose stage waits for VQVAE to finish (it loads the VQVAE checkpoint).
# Root is independent and could run in parallel, but on an 8x GPU node
# nothing parallelizes for free — we keep it sequential to maximize per-stage
# DDP throughput. Set RUN_ROOT=0 / RUN_POSE=0 to skip stages.
#
# Defaults are tuned for an 8x H200 (or H100 80GB) SXM node, full Path A run
# on the BONES-SEED corpus. For a smoke test, prefer
# ``run_planner_smoke_8gpu.sh`` instead.
#
# Usage (run on the cloud node, from the repo root):
#
#   # one-time: build skeleton assets + stats (single-process, ~30s)
#   python motionbricks/scripts/build_x2_skeleton_assets.py
#
#   # launch the full pipeline (detached, in tmux):
#   tmux new -d -s plan_train "bash motionbricks/scripts/cloud/run_planner_train_8gpu.sh"
#   tmux a -t plan_train         # attach to watch
#   tail -f ~/plan_train.log     # ...or tail the log file
#
# Override knobs (env vars):
#   NUM_GPUS               GPUs to use                              (default: 8)
#   PKL                    motion-lib PKL                           (default: x2_ultra_bones_seed.pkl)
#   FILTER                 dataset filter ('none' or 'loco')        (default: none)
#   MAX_CLIPS              cap dataset size for debug                (default: <unset> = all)
#   MIN_FRAMES/MAX_FRAMES  per-clip frame range                      (default: 80 / 200)
#   BATCH_PER_GPU          batch size per GPU per stage              (default: 4)
#   NUM_WORKERS            DataLoader workers per rank               (default: 4)
#   VQVAE_STEPS            VQVAE training steps                      (default: 500000)
#   POSE_STEPS             pose backbone training steps              (default: 200000)
#   ROOT_STEPS             root backbone training steps              (default: 200000)
#   SAVE_EVERY             checkpoint every N steps (all stages)     (default: 5000)
#   USE_WANDB              1=enable W&B, 0=disable                   (default: 0)
#   WANDB_PROJECT          W&B project                               (default: TRL_X2Ultra_Planner)
#   RUN_VQVAE              1=run VQVAE, 0=skip                       (default: 1)
#   RUN_POSE               1=run pose, 0=skip                        (default: 1)
#   RUN_ROOT               1=run root, 0=skip                        (default: 1)
#   VQVAE_CKPT             path to existing VQVAE ckpt for pose      (default: out/.../last.ckpt)
#   RESUME_VQVAE           ckpt to resume VQVAE from                 (default: <unset>)
#   RESUME_POSE            ckpt to resume pose from                  (default: <unset>)
#   RESUME_ROOT            ckpt to resume root from                  (default: <unset>)
#   LOG_FILE               where to tee stdout                       (default: ~/plan_train.log)
#   STAGE_LOG_DIR          per-stage log dir                         (default: ~/plan_train_logs)

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
PKL=${PKL:-gear_sonic/data/motions/x2_ultra_bones_seed.pkl}
FILTER=${FILTER:-none}
MAX_CLIPS=${MAX_CLIPS:-}
MIN_FRAMES=${MIN_FRAMES:-80}
MAX_FRAMES=${MAX_FRAMES:-200}
BATCH_PER_GPU=${BATCH_PER_GPU:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
VQVAE_STEPS=${VQVAE_STEPS:-500000}
POSE_STEPS=${POSE_STEPS:-200000}
ROOT_STEPS=${ROOT_STEPS:-200000}
SAVE_EVERY=${SAVE_EVERY:-5000}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-TRL_X2Ultra_Planner}
RUN_VQVAE=${RUN_VQVAE:-1}
RUN_POSE=${RUN_POSE:-1}
RUN_ROOT=${RUN_ROOT:-1}
VQVAE_CKPT=${VQVAE_CKPT:-}
RESUME_VQVAE=${RESUME_VQVAE:-}
RESUME_POSE=${RESUME_POSE:-}
RESUME_ROOT=${RESUME_ROOT:-}
LOG_FILE=${LOG_FILE:-$HOME/plan_train.log}
STAGE_LOG_DIR=${STAGE_LOG_DIR:-$HOME/plan_train_logs}

mkdir -p "$STAGE_LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date) === PLANNER TRAIN START"
echo "  num_gpus       : $NUM_GPUS"
echo "  pkl            : $PKL"
echo "  filter         : $FILTER"
echo "  max_clips      : ${MAX_CLIPS:-<all>}"
echo "  frames         : [$MIN_FRAMES, $MAX_FRAMES]"
echo "  batch_per_gpu  : $BATCH_PER_GPU"
echo "  num_workers    : $NUM_WORKERS"
echo "  vqvae_steps    : $VQVAE_STEPS"
echo "  pose_steps     : $POSE_STEPS"
echo "  root_steps     : $ROOT_STEPS"
echo "  save_every     : $SAVE_EVERY"
echo "  use_wandb      : $USE_WANDB"
echo "  run vqvae/pose/root : $RUN_VQVAE/$RUN_POSE/$RUN_ROOT"
echo "  log file       : $LOG_FILE"
echo "  stage logs     : $STAGE_LOG_DIR"

# Activate the motionbricks conda env (matches bootstrap_planner_node.sh).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate motionbricks

# Repo root is two levels up from this script (motionbricks/scripts/cloud/...).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
echo "  repo_root      : $REPO_ROOT"

if [[ ! -f "$PKL" ]]; then
  echo "ERROR: motion-lib PKL not found: $PKL" >&2
  echo "       did you scp + tar -xzf the planner bundle?" >&2
  exit 1
fi

# Build skeleton + stats once if missing (single-process work, ~30 s).
ASSETS_VERSION_DIR="motionbricks/out/motionbricks_vqvae_x2/version_1"
if [[ ! -f "${ASSETS_VERSION_DIR}/hparams.yaml" ]]; then
  echo "INFO: skeleton assets missing — running build_x2_skeleton_assets.py first"
  python motionbricks/scripts/build_x2_skeleton_assets.py
fi

# Mitigate fragmentation OOMs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DDP rendezvous over loopback (single-node, multi-GPU).
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

WANDB_FLAGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_FLAGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
fi

MAX_CLIPS_FLAGS=()
if [[ -n "$MAX_CLIPS" ]]; then
  MAX_CLIPS_FLAGS+=(--max_clips "$MAX_CLIPS")
fi

#-------------------------------------------------------------------------------
# Stage 1 — VQVAE
#-------------------------------------------------------------------------------
if [[ "$RUN_VQVAE" == "1" ]]; then
  echo
  echo "=== $(date) === STAGE 1: VQVAE"
  STAGE_LOG="$STAGE_LOG_DIR/vqvae.log"
  RESUME_FLAGS=()
  [[ -n "$RESUME_VQVAE" ]] && RESUME_FLAGS+=(--resume "$RESUME_VQVAE")
  python motionbricks/scripts/train_vqvae_x2.py \
    --pkl "$PKL" \
    --filter "$FILTER" \
    --max_steps "$VQVAE_STEPS" \
    --batch_size "$BATCH_PER_GPU" \
    --num_workers "$NUM_WORKERS" \
    --min_frames "$MIN_FRAMES" \
    --max_frames "$MAX_FRAMES" \
    --save-every-n-steps "$SAVE_EVERY" \
    --no-progress-bar \
    --devices "$NUM_GPUS" \
    --num-nodes 1 \
    "${MAX_CLIPS_FLAGS[@]}" \
    "${WANDB_FLAGS[@]}" \
    "${RESUME_FLAGS[@]}" \
    2>&1 | tee "$STAGE_LOG"
  echo "=== $(date) === STAGE 1 DONE -> $STAGE_LOG"
else
  echo "SKIPPING stage 1 (RUN_VQVAE=$RUN_VQVAE)"
fi

#-------------------------------------------------------------------------------
# Stage 2 — pose backbone (depends on VQVAE)
#-------------------------------------------------------------------------------
if [[ "$RUN_POSE" == "1" ]]; then
  echo
  echo "=== $(date) === STAGE 2: pose"
  STAGE_LOG="$STAGE_LOG_DIR/pose.log"
  if [[ -z "$VQVAE_CKPT" ]]; then
    VQVAE_CKPT="motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt"
  fi
  if [[ ! -f "$VQVAE_CKPT" ]]; then
    echo "ERROR: pose stage needs a VQVAE checkpoint at $VQVAE_CKPT" >&2
    echo "       set VQVAE_CKPT or run with RUN_VQVAE=1 first" >&2
    exit 1
  fi
  RESUME_FLAGS=()
  [[ -n "$RESUME_POSE" ]] && RESUME_FLAGS+=(--resume "$RESUME_POSE")
  python motionbricks/scripts/train_pose_x2.py \
    --pkl "$PKL" \
    --filter "$FILTER" \
    --max_steps "$POSE_STEPS" \
    --batch_size "$BATCH_PER_GPU" \
    --num_workers "$NUM_WORKERS" \
    --min_frames "$MIN_FRAMES" \
    --max_frames "$MAX_FRAMES" \
    --save-every-n-steps "$SAVE_EVERY" \
    --vqvae-ckpt "$VQVAE_CKPT" \
    --no-progress-bar \
    --devices "$NUM_GPUS" \
    --num-nodes 1 \
    "${MAX_CLIPS_FLAGS[@]}" \
    "${WANDB_FLAGS[@]}" \
    "${RESUME_FLAGS[@]}" \
    2>&1 | tee "$STAGE_LOG"
  echo "=== $(date) === STAGE 2 DONE -> $STAGE_LOG"
else
  echo "SKIPPING stage 2 (RUN_POSE=$RUN_POSE)"
fi

#-------------------------------------------------------------------------------
# Stage 3 — root backbone (independent)
#-------------------------------------------------------------------------------
if [[ "$RUN_ROOT" == "1" ]]; then
  echo
  echo "=== $(date) === STAGE 3: root"
  STAGE_LOG="$STAGE_LOG_DIR/root.log"
  RESUME_FLAGS=()
  [[ -n "$RESUME_ROOT" ]] && RESUME_FLAGS+=(--resume "$RESUME_ROOT")
  python motionbricks/scripts/train_root_x2.py \
    --pkl "$PKL" \
    --filter "$FILTER" \
    --max_steps "$ROOT_STEPS" \
    --batch_size "$BATCH_PER_GPU" \
    --num_workers "$NUM_WORKERS" \
    --min_frames "$MIN_FRAMES" \
    --max_frames "$MAX_FRAMES" \
    --save-every-n-steps "$SAVE_EVERY" \
    --no-progress-bar \
    --devices "$NUM_GPUS" \
    --num-nodes 1 \
    "${MAX_CLIPS_FLAGS[@]}" \
    "${WANDB_FLAGS[@]}" \
    "${RESUME_FLAGS[@]}" \
    2>&1 | tee "$STAGE_LOG"
  echo "=== $(date) === STAGE 3 DONE -> $STAGE_LOG"
else
  echo "SKIPPING stage 3 (RUN_ROOT=$RUN_ROOT)"
fi

echo
echo "=== $(date) === PLANNER TRAIN COMPLETE"
echo "Checkpoints:"
ls -lh motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt 2>/dev/null || true
ls -lh motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/last.ckpt 2>/dev/null || true
ls -lh motionbricks/out/motionbricks_root_x2/version_1/checkpoints/last.ckpt 2>/dev/null || true
