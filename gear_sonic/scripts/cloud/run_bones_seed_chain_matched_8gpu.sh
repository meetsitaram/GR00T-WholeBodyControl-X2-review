#!/bin/bash
# 8-GPU launcher for the chain_matched + sphere-feet continuation run.
#
# Wraps ``run_smoke_8gpu.sh`` with defaults baked in for the
# ``sonic_x2_ultra_bones_seed_chain_matched_sphere_feet`` experiment:
#   - NUM_ITERS=15000  (~29 h on 8× H200, ~$1k of GPU time)
#   - NUM_ENVS=12288   (down from 16384 to leave VRAM headroom for the
#                       3.8 GB chain_matched_v2 PKL)
#   - MOTION_FILE      = x2_ultra_bones_seed_chain_matched_v2.pkl
#   - +checkpoint      = H200 25k sphere-feet warm-start
#   - W&B enabled
#
# Usage (on the cloud node, from the repo root, after bootstrap + bundle):
#
#   tmux new -d -s chain_matched "bash gear_sonic/scripts/cloud/run_bones_seed_chain_matched_8gpu.sh"
#   tmux a -t chain_matched
#
# Override knobs (passed through to run_smoke_8gpu.sh):
#   NUM_PROCESSES   number of GPUs                      (default: 8)
#   NUM_ENVS        envs per GPU                        (default: 12288)
#   NUM_ITERS       PPO iterations                      (default: 15000)
#   MOTION_FILE     motion-lib PKL                      (default: chain_matched_v2)
#   USE_WANDB       True/False                          (default: True)
#   LOG_FILE        where to tee stdout                 (default: ~/chain_matched.log)
#   CHECKPOINT      warm-start path                     (default: H200 25k sphere-feet)
#   EXTRA_FLAGS     additional Hydra flags appended raw (default: empty;
#                   appended AFTER the +checkpoint flag below)
#
# Pre-launch checklist (each item must be GREEN):
#   - bootstrap_fresh_node.sh has run to completion (pulls the sphere URDF
#     and all other tracked files via git clone)
#   - Side-channel bundle extracted: x2_ultra_bones_seed_chain_matched_v2.pkl
#     and x2_ultra_body_check.pkl are in gear_sonic/data/motions/
#     (both are gitignored, must scp from workstation)
#   - model_step_025000.pt is at $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/
#   - 200-iter smoke (bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh) finished cleanly

set -euo pipefail

export NUM_PROCESSES=${NUM_PROCESSES:-8}
export NUM_ENVS=${NUM_ENVS:-12288}
export NUM_ITERS=${NUM_ITERS:-15000}
export MOTION_FILE=${MOTION_FILE:-gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2.pkl}
export USE_WANDB=${USE_WANDB:-True}
export LOG_FILE=${LOG_FILE:-$HOME/chain_matched.log}
export EXP_NAME=sonic_x2_ultra_bones_seed_chain_matched_sphere_feet
CHECKPOINT=${CHECKPOINT:-$HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt}

# Pre-flight: refuse to launch if the warm-start checkpoint isn't on disk —
# losing 29 h of training to a typo'd path is painful.
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "FATAL: warm-start checkpoint not found: ${CHECKPOINT}" >&2
  echo "       scp it from your workstation, e.g.:" >&2
  echo "         scp -r ~/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501 \\" >&2
  echo "             ubuntu@<this node>:~/x2_cloud_checkpoints/" >&2
  exit 1
fi

# Pre-flight: confirm the chain_matched PKL is on disk too.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ ! -f "${REPO_ROOT}/${MOTION_FILE}" ]]; then
  echo "FATAL: motion PKL not found: ${REPO_ROOT}/${MOTION_FILE}" >&2
  echo "       Extract the side-channel bundle, e.g.:" >&2
  echo "         cd ${REPO_ROOT} && tar -xzf ~/x2_chainmatched_bundle.tar.gz" >&2
  exit 1
fi

# Stitch +checkpoint into EXTRA_FLAGS so run_smoke_8gpu.sh passes it through.
EXTRA_FLAGS_BASE="+checkpoint=${CHECKPOINT}"
export EXTRA_FLAGS="${EXTRA_FLAGS_BASE} ${EXTRA_FLAGS:-}"

echo "=== chain_matched 8-GPU launcher ==="
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
