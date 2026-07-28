#!/bin/bash
# Elastic multi-node SONIC training on a preemptible GPU cluster.
#
# Run the SAME script on EVERY node (reserved-main + preemptible workers):
#
#   tmux new -d -s elastic "bash gear_sonic/scripts/cloud/run_elastic_multinode.sh"
#
# Fault-tolerance model (torchrun elastic, c10d rendezvous):
#   - --nnodes=MIN:MAX — training runs while >=MIN nodes are joined; a
#     preempted node triggers a restart-from-last-checkpoint with the
#     surviving membership; a node coming back triggers the same to scale up.
#   - The rendezvous endpoint lives on the RESERVED node (never preempted).
#     Workers joining/leaving never kill the rendezvous itself.
#   - Every membership change = all workers restart and resume from
#     $EXPERIMENT_DIR/last.pt (saved every 50 iters by ModelSaveCallback)
#     on the shared filesystem. wandb re-attaches via meta.yaml in the same
#     dir, so the dashboard x-axis is continuous.
#   - The outer supervisor loop re-arms torchrun if it ever exits nonzero
#     (e.g. restarts exhausted while <MIN nodes were alive); once the node
#     count recovers, training resumes automatically.
#
# Knobs (env):
#   RDZV_ENDPOINT    rendezvous host:port (default 10.0.0.37:29400 = reserved-main)
#   NNODES_MIN/MAX   elastic node range                (default 3:4)
#   NPROC_PER_NODE   GPUs per node                     (default 8)
#   NUM_ENVS         envs per GPU                      (default 8192)
#   NUM_ITERS        PPO iterations                    (default 200000)
#   EXP_NAME         Hydra +exp= leaf                  (default sonic_x2_ultra)
#   MOTION_FILE      motion-lib PKL                    (default shared-FS executed corpus)
#   EXPERIMENT_DIR   pinned run dir on shared FS — SAME on all nodes; new run =
#                    new dir (stale meta.yaml would re-attach the old wandb run)
#   USE_WANDB        True/False                        (default True)
#   MAX_RESTARTS     torchrun restart budget per agent (default 100)
#   SONIC_PG_TIMEOUT_S  NCCL process-group timeout     (default 600 here; a dead
#                    node aborts survivors' collectives after this, vs 100 min stock)
#   EXTRA_FLAGS      extra Hydra flags, word-split

set -uo pipefail

RDZV_ENDPOINT=${RDZV_ENDPOINT:?set to <reserved-node-private-ip>:29400}
NNODES_MIN=${NNODES_MIN:-3}
NNODES_MAX=${NNODES_MAX:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NUM_ENVS=${NUM_ENVS:-8192}
NUM_ITERS=${NUM_ITERS:-200000}
EXP_NAME=${EXP_NAME:-sonic_x2_ultra}
MOTION_FILE=${MOTION_FILE:?path to motion-lib pkl on the shared filesystem}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-/mnt/shared/ckpts/elastic_run1}
USE_WANDB=${USE_WANDB:-True}
MAX_RESTARTS=${MAX_RESTARTS:-100}
# RDZV_IS_HOST=1 on the node whose IP is in RDZV_ENDPOINT. torchrun's
# auto-detection resolves the machine hostname (127.0.1.1 on cloud images), so
# it never matches the endpoint IP and no node binds the store without this.
RDZV_IS_HOST=${RDZV_IS_HOST:-0}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
LOG_FILE=${LOG_FILE:-$HOME/elastic.log}

export SONIC_PG_TIMEOUT_S=${SONIC_PG_TIMEOUT_S:-600}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# INFO prints the selected inter-node transport at init ("NCCL INFO Using
# network IB" vs Socket) — the only way to confirm the IB rails are in use.
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

exec > >(tee -a "$LOG_FILE") 2>&1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== $(date) === ELASTIC SUPERVISOR START on $(hostname)"
echo "  rdzv          : $RDZV_ENDPOINT  nnodes $NNODES_MIN:$NNODES_MAX x $NPROC_PER_NODE gpu"
echo "  experiment_dir: $EXPERIMENT_DIR"
echo "  motion_file   : $MOTION_FILE   envs/gpu: $NUM_ENVS"

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "ERROR: motion file not found: $MOTION_FILE"
  exit 1
fi
mkdir -p "$EXPERIMENT_DIR"

attempt=0
while true; do
  attempt=$((attempt + 1))
  RESUME_FLAGS=""
  if [[ -f "$EXPERIMENT_DIR/last.pt" ]]; then
    RESUME_FLAGS="+resume=True"
    echo "=== $(date) === attempt $attempt: resuming from $EXPERIMENT_DIR/last.pt"
  else
    echo "=== $(date) === attempt $attempt: fresh start (no last.pt yet)"
  fi

  # shellcheck disable=SC2086  # EXTRA_FLAGS/RESUME_FLAGS intentionally word-split.
  torchrun \
    --nnodes="$NNODES_MIN:$NNODES_MAX" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$RDZV_ENDPOINT" \
    --rdzv-id=sonic_elastic \
    --rdzv-conf="is_host=$RDZV_IS_HOST" \
    --max-restarts="$MAX_RESTARTS" \
    gear_sonic/train_agent_trl.py \
    --config-name=base \
    "+exp=manager/universal_token/all_modes/$EXP_NAME" \
    ++num_envs="$NUM_ENVS" \
    ++headless=True \
    ++use_wandb="$USE_WANDB" \
    ++experiment_dir="$EXPERIMENT_DIR" \
    ++algo.config.num_learning_iterations="$NUM_ITERS" \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION_FILE" \
    $RESUME_FLAGS $EXTRA_FLAGS
  rc=$?

  if [[ $rc -eq 0 ]]; then
    echo "=== $(date) === training finished cleanly, supervisor exiting"
    break
  fi
  echo "=== $(date) === torchrun exited rc=$rc; re-arming in 30s (attempt $attempt done)"
  sleep 30
done
