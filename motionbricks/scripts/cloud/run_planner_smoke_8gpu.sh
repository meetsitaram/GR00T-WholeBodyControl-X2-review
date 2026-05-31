#!/bin/bash
# 8-GPU smoke test for the X2 MotionBricks planner training stack.
#
# Verifies the full pipeline (build_x2_skeleton_assets → DDP-aware
# train_vqvae_x2 → train_pose_x2 → train_root_x2) on every GPU of a cloud
# node, without committing to a multi-hour real run.
#
# Defaults are tuned for an 8x H200 (or H100 80GB) SXM node:
#   - 30-clip planner-smoke PKL (built via build_planner_smoke_pkl.py)
#   - 1000 VQVAE steps + 500 pose steps + 500 root steps  (~3 min total)
#   - batch_per_gpu=2 (1024 effective with 8 GPUs ✕ 64-frame chunks)
#   - W&B disabled
#
# Usage (run on the cloud node, from the repo root):
#
#   # one-time: build the planner smoke PKL from the unpacked BONES-SEED bundle
#   python motionbricks/scripts/cloud/build_planner_smoke_pkl.py
#
#   # one-time: build skeleton + stats
#   python motionbricks/scripts/build_x2_skeleton_assets.py
#
#   # launch the smoke (detached, in tmux):
#   tmux new -d -s smoke "bash motionbricks/scripts/cloud/run_planner_smoke_8gpu.sh"
#   tmux a -t smoke              # attach to watch, Ctrl-b d to detach
#   tail -f ~/plan_smoke.log     # ...or tail the log file
#
# Override knobs (env vars passed straight through to run_planner_train.sh):
#   NUM_GPUS, PKL, BATCH_PER_GPU, NUM_WORKERS, USE_WANDB, WANDB_PROJECT,
#   VQVAE_STEPS, POSE_STEPS, ROOT_STEPS, SAVE_EVERY, LOG_FILE, STAGE_LOG_DIR

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Smoke defaults (override the train-stage env vars).
export NUM_GPUS=${NUM_GPUS:-8}
export PKL=${PKL:-gear_sonic/data/motions/x2_ultra_planner_smoke.pkl}
export FILTER=${FILTER:-none}
export BATCH_PER_GPU=${BATCH_PER_GPU:-2}
export NUM_WORKERS=${NUM_WORKERS:-2}
export VQVAE_STEPS=${VQVAE_STEPS:-1000}
export POSE_STEPS=${POSE_STEPS:-500}
export ROOT_STEPS=${ROOT_STEPS:-500}
export SAVE_EVERY=${SAVE_EVERY:-500}
export USE_WANDB=${USE_WANDB:-0}
export LOG_FILE=${LOG_FILE:-$HOME/plan_smoke.log}
export STAGE_LOG_DIR=${STAGE_LOG_DIR:-$HOME/plan_smoke_logs}

# If the smoke PKL doesn't exist yet, build it on the fly. Cheap (<10 s).
if [[ ! -f "$REPO_ROOT/$PKL" ]]; then
  echo "INFO: $PKL not found; building it from the BONES-SEED PKL"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate motionbricks
  (cd "$REPO_ROOT" && python motionbricks/scripts/cloud/build_planner_smoke_pkl.py)
fi

# Delegate to the full training launcher with smoke-tier step counts.
bash "$(dirname "${BASH_SOURCE[0]}")/run_planner_train.sh"
