#!/bin/bash
# X2 Ultra multi-skill fine-tune, 8-GPU. Warm-starts from the 175030 4k checkpoint
# and fine-tunes on the 182-clip v3 corpus (dance + shadow-boxing + slow-walk) with
# the manufacturer-datasheet effort/velocity limits in x2_ultra.py (waist 24,
# wrist 4.8/4.188). See sonic_x2_ultra_dance_finetune.yaml +
# project_x2_motor_datasheet_effort_fix.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root

# Activate env_isaaclab (accelerate + isaaclab live here; matches run_smoke_8gpu.sh)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_ENVS=${NUM_ENVS:-12288}
NUM_ITERS=${NUM_ITERS:-30000}
BASE_CORPUS=${BASE_CORPUS:-gear_sonic/data/motions/x2_sonic_executed_feasible.pkl}
CHECKPOINT=${CHECKPOINT:-$HOME/GR00T-WholeBodyControl/logs_rl/TRL_X2Ultra_Dance/manager/universal_token/all_modes/sonic_x2_ultra_dance_finetune_dance_finetune_v1-20260714_175030/model_step_004000.pt}
LOG=${LOG:-$HOME/dance_finetune.log}

for f in "$BASE_CORPUS" gear_sonic/data/motions/x2_all_finetune_v3.pkl "$CHECKPOINT"; do
  [[ -f "$f" ]] || { echo "FATAL: missing required file: $f" >&2; exit 1; }
done

# Sanity: confirm the manufacturer effort/velocity limits are present (this run depends on them)
if ! grep -q '"\.\*_wrist_pitch_joint": 4.188' gear_sonic/envs/manager_env/robots/x2_ultra.py; then
  echo "WARN: wrist velocity_limit_sim (4.188, manufacturer sheet) NOT found in x2_ultra.py." >&2
fi
if ! grep -q 'effort_limit_sim=24.0' gear_sonic/envs/manager_env/robots/x2_ultra.py; then
  echo "WARN: waist effort_limit_sim (24.0, manufacturer sheet) NOT found in x2_ultra.py." >&2
fi

echo "=== X2 dance fine-tune (8-GPU) ==="
echo "  exp         : sonic_x2_ultra_dance_finetune  (project TRL_X2Ultra_Dance)"
echo "  warm-start  : $CHECKPOINT"
echo "  base corpus : $BASE_CORPUS"
echo "  finetune set: x2_all_finetune_v3.pkl (182 clips: dance+shadow+slow-walk, sample_rate 0.3)"
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
