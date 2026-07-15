#!/bin/bash
# 8-GPU launcher for the chain_matched v3 RECOVERY continuation run.
#
# Resumes from the v2 run's iter-4000 checkpoint on a curriculum-filtered
# corpus with boosted orientation reward weights. See header of
#   gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_bones_seed_chain_matched_v3_recovery.yaml
# for the full rationale and what-to-watch metrics.
#
# Pre-launch checklist:
#   - bootstrap_fresh_node.sh has run on this node (or you're reusing a node)
#   - gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2_easy.pkl
#     is on disk (scp from workstation; ~2.9 GB)
#   - model_step_004000.pt from the v2 run is at the warm-start path below
#   - W&B login present in ~/.netrc
#
# Usage on the cloud node, from the repo root:
#
#   tmux new -d -s v3_recovery "bash gear_sonic/scripts/cloud/run_v3_recovery_8gpu.sh"
#   tmux a -t v3_recovery
#
# Override knobs:
#   NUM_PROCESSES   number of GPUs                       (default: 8)
#   NUM_ENVS        envs per GPU                         (default: 12288)
#   NUM_ITERS       PPO iterations                       (default: 8000)
#   MOTION_FILE     motion-lib PKL                       (default: easy filter)
#   USE_WANDB       True/False                           (default: True)
#   LOG_FILE        where to tee stdout                  (default: ~/v3_recovery.log)
#   CHECKPOINT      warm-start path                      (default: v2 iter-4000)
#   EXTRA_FLAGS     additional Hydra flags appended raw

set -euo pipefail

export NUM_PROCESSES=${NUM_PROCESSES:-8}
export NUM_ENVS=${NUM_ENVS:-12288}
export NUM_ITERS=${NUM_ITERS:-8000}
export MOTION_FILE=${MOTION_FILE:-gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2_easy.pkl}
export USE_WANDB=${USE_WANDB:-True}
export LOG_FILE=${LOG_FILE:-$HOME/v3_recovery.log}
export EXP_NAME=sonic_x2_ultra_bones_seed_chain_matched_v3_recovery
CHECKPOINT=${CHECKPOINT:-$HOME/x2_cloud_checkpoints/chain_matched_v2_iter_004000/model_step_004000.pt}

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "FATAL: warm-start checkpoint not found: ${CHECKPOINT}" >&2
  echo "       scp it from your workstation, e.g.:" >&2
  echo "         scp ~/x2_cloud_checkpoints/chain_matched_v2_iter_004000/model_step_004000.pt \\" >&2
  echo "             ubuntu@<this node>:${CHECKPOINT}" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ ! -f "${REPO_ROOT}/${MOTION_FILE}" ]]; then
  echo "FATAL: motion PKL not found: ${REPO_ROOT}/${MOTION_FILE}" >&2
  echo "       scp it from your workstation (~2.9 GB), e.g.:" >&2
  echo "         scp ${MOTION_FILE} \\" >&2
  echo "             ubuntu@<this node>:${REPO_ROOT}/${MOTION_FILE}" >&2
  exit 1
fi

EXTRA_FLAGS_BASE="+checkpoint=${CHECKPOINT}"
export EXTRA_FLAGS="${EXTRA_FLAGS_BASE} ${EXTRA_FLAGS:-}"

echo "=== v3 recovery 8-GPU launcher ==="
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
