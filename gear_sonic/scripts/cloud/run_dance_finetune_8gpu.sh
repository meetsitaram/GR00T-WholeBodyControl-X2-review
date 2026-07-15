#!/bin/bash
# X2 Ultra dance fine-tune, 8-GPU. Warm-starts from the arm-dynamics v3 last.pt
# (iter ~1500) and fine-tunes on the 79-clip dance corpus with the wrist
# velocity-limit fix in x2_ultra.py already applied. See
# sonic_x2_ultra_dance_finetune.yaml + project_x2_wrist_investigation.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root

# Activate env_isaaclab (accelerate + isaaclab live here; matches run_smoke_8gpu.sh)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_ENVS=${NUM_ENVS:-12288}
NUM_ITERS=${NUM_ITERS:-30000}
BASE_CORPUS=${BASE_CORPUS:-gear_sonic/data/motions/x2_sonic_executed_feasible.pkl}
CHECKPOINT=${CHECKPOINT:-$HOME/GR00T-WholeBodyControl/logs_rl/TRL_X2Ultra_ArmDynamics/manager/universal_token/all_modes/sonic_x2_ultra_arm_dynamics_finetune_arm_dynamics_v3-20260713_221531/last.pt}
LOG=${LOG:-$HOME/dance_finetune.log}

for f in "$BASE_CORPUS" gear_sonic/data/motions/x2_all_dances_finetune.pkl "$CHECKPOINT"; do
  [[ -f "$f" ]] || { echo "FATAL: missing required file: $f" >&2; exit 1; }
done

# Sanity: confirm the wrist velocity-limit fix is present (this run depends on it)
if ! grep -q '"\.\*_wrist_pitch_joint": 20.944' gear_sonic/envs/manager_env/robots/x2_ultra.py; then
  echo "WARN: wrist velocity_limit_sim fix (20.944) NOT found in x2_ultra.py -- dance wrists will not track." >&2
fi

echo "=== X2 dance fine-tune (8-GPU) ==="
echo "  exp         : sonic_x2_ultra_dance_finetune  (project TRL_X2Ultra_Dance)"
echo "  warm-start  : $CHECKPOINT"
echo "  base corpus : $BASE_CORPUS"
echo "  dance set   : x2_all_dances_finetune.pkl (79 clips, finetune_sample_rate 0.7)"
echo "  num_envs    : $NUM_ENVS    iters: $NUM_ITERS"
echo "  log         : $LOG"

nohup accelerate launch --num_processes="$NUM_PROCESSES" \
  gear_sonic/train_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/all_modes/sonic_x2_ultra_dance_finetune \
  ++num_envs="$NUM_ENVS" \
  ++headless=True \
  ++use_wandb=True \
  ++algo.config.num_learning_iterations="$NUM_ITERS" \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$BASE_CORPUS" \
  +checkpoint="$CHECKPOINT" \
  > "$LOG" 2>&1 &
echo "launched accelerate PID $!  ->  tail -f $LOG"
