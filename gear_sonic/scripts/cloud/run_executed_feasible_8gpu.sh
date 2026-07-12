#!/bin/bash
# 8-GPU launcher for the X2 SONIC fine-tune on the G1-SONIC EXECUTED-FEASIBLE
# corpus (gear_sonic/data/motions/x2_sonic_executed_feasible.pkl, 35,974 clips).
#
# Warm-starts (weight-only) from the chain_matched_v3 iter-1376 checkpoint.
# See gear_sonic/config/exp/manager/universal_token/all_modes/
#     sonic_x2_ultra_executed_feasible.yaml
# for the corpus provenance + reward-weight caveat, and
#   docs/experiments/x2_sonic_executed_feasible_finetune.md
# for the full run plan.
#
# Pre-launch checklist (cloud node):
#   - bootstrap_fresh_node.sh has run (env_isaaclab + IsaacLab installed)
#   - gear_sonic/data/motions/x2_sonic_executed_feasible.pkl on disk (~4.1 GB)
#   - the config yaml above is on disk (from the bundle or git pull)
#   - the warm-start .pt is at CHECKPOINT below (~383 MB, scp'd separately)
#   - W&B login present in ~/.netrc (or set USE_WANDB=False)
#
# Usage on the cloud node, from the repo root:
#
#   # ~3-min smoke first (10 iters) — proves the pipeline before committing ~20h
#   NUM_ITERS=10 USE_WANDB=False LOG_FILE=$HOME/smoke.log \
#     bash gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh
#
#   # real run
#   tmux new -d -s ef "bash gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh"
#   tmux a -t ef
#
# Override knobs (env vars):
#   NUM_PROCESSES  number of GPUs                (default: 8)
#   NUM_ENVS       envs per GPU                  (default: 12288; H100 80GB safe)
#   NUM_ITERS      PPO iterations               (default: 8000)
#   MOTION_FILE    motion-lib PKL               (default: executed_feasible)
#   USE_WANDB      True/False                   (default: True)
#   LOG_FILE       where to tee stdout          (default: ~/executed_feasible.log)
#   CHECKPOINT     warm-start .pt path          (default: v3 iter-1376)
#   EXTRA_FLAGS    extra Hydra flags, appended raw

set -euo pipefail

export NUM_PROCESSES=${NUM_PROCESSES:-8}
export NUM_ENVS=${NUM_ENVS:-12288}
export NUM_ITERS=${NUM_ITERS:-30000}
export MOTION_FILE=${MOTION_FILE:-gear_sonic/data/motions/x2_sonic_executed_feasible.pkl}
export USE_WANDB=${USE_WANDB:-True}
export LOG_FILE=${LOG_FILE:-$HOME/executed_feasible.log}
export EXP_NAME=sonic_x2_ultra_executed_feasible
CHECKPOINT=${CHECKPOINT:-$HOME/x2_cloud_checkpoints/chain_matched_v3_iter_001376/model_step_001376.pt}

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "FATAL: warm-start checkpoint not found: ${CHECKPOINT}" >&2
  echo "       scp it from your workstation, e.g.:" >&2
  echo "         scp ~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/model_step_001376.pt \\" >&2
  echo "             ubuntu@<this node>:${CHECKPOINT}" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ ! -f "${REPO_ROOT}/${MOTION_FILE}" ]]; then
  echo "FATAL: motion PKL not found: ${REPO_ROOT}/${MOTION_FILE}" >&2
  echo "       scp it from your workstation (~4.1 GB), e.g.:" >&2
  echo "         scp ${MOTION_FILE} \\" >&2
  echo "             ubuntu@<this node>:${REPO_ROOT}/${MOTION_FILE}" >&2
  exit 1
fi

EXTRA_FLAGS_BASE="+checkpoint=${CHECKPOINT}"
export EXTRA_FLAGS="${EXTRA_FLAGS_BASE} ${EXTRA_FLAGS:-}"

echo "=== executed-feasible 8-GPU launcher ==="
echo "  exp           : ${EXP_NAME}"
echo "  num_processes : ${NUM_PROCESSES}"
echo "  num_envs/proc : ${NUM_ENVS}"
echo "  num_iters     : ${NUM_ITERS}"
echo "  motion_file   : ${MOTION_FILE}"
echo "  checkpoint    : ${CHECKPOINT}"
echo "  use_wandb     : ${USE_WANDB}"
echo "  log_file      : ${LOG_FILE}"
echo "  extra_flags   : ${EXTRA_FLAGS}"
echo ""

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_smoke_8gpu.sh"
