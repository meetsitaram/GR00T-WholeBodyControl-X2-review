#!/bin/bash
# X2 Ultra dance fine-tune V5 (v3+UltimateBots+gestures, 275 clips), 8-GPU.
# Warm-starts from softland_4800_g1 (model_step_004800.pt). Override CHECKPOINT= if path differs.
# and fine-tunes on the 182-clip v3 corpus (dance + shadow-boxing + slow-walk) with
# the manufacturer-datasheet effort/velocity limits in x2_ultra.py (waist 24,
# wrist 4.8/4.188). See sonic_x2_ultra_dance_finetune.yaml +
# project_x2_motor_datasheet_effort_fix.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root

# Activate env_isaaclab (accelerate + isaaclab live here; matches run_smoke_8gpu.sh)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_ENVS=${NUM_ENVS:-12288}
NUM_ITERS=${NUM_ITERS:-3000}
BASE_CORPUS=${BASE_CORPUS:-/mnt/ckpt/corpus/x2_sonic_executed_feasible.pkl}
CHECKPOINT=${CHECKPOINT:-/mnt/ckpt/sonic/v5_resume/model_step_002000.pt}
LOG=${LOG:-$HOME/dance_finetune_resume.log}

for f in "$BASE_CORPUS" gear_sonic/data/motions/x2_all_finetune_v5.pkl "$CHECKPOINT"; do
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
echo "  exp         : sonic_x2_ultra_dance_finetune_v5  (project TRL_X2Ultra_Dance)"
echo "  warm-start  : $CHECKPOINT"
echo "  base corpus : $BASE_CORPUS"
echo "  finetune set: x2_all_finetune_v5.pkl (182 clips: dance+shadow+slow-walk, sample_rate 0.3)"
echo "  num_envs    : $NUM_ENVS    iters: $NUM_ITERS"
echo "  log         : $LOG"


# --- CORPUS CONSISTENCY GUARD -------------------------------------------------
# The bug this catches: +exp points at a config whose fine_tune_dataset.motion_file
# is NOT the corpus we intend, so the run silently trains on the wrong clips.
# Assert the +exp leaf config actually references the expected corpus; abort loud.
EXPECTED_FT="x2_all_finetune_v5.pkl"
EXP_LEAF=$(grep -oE "[+]exp=manager/[^ ]*" "$0" | sed "s#.*/##")
EXP_CFG="gear_sonic/config/exp/manager/universal_token/all_modes/${EXP_LEAF}.yaml"
if [ ! -f "$EXP_CFG" ]; then echo "ABORT: +exp config not found: $EXP_CFG" >&2; exit 9; fi
if ! grep -q "$EXPECTED_FT" "$EXP_CFG"; then
  echo "ABORT: +exp=$EXP_LEAF does NOT point fine_tune at $EXPECTED_FT." >&2
  echo "       Refusing to launch on the wrong corpus." >&2
  echo "       ($EXP_CFG fine_tune_dataset.motion_file must reference $EXPECTED_FT)" >&2
  exit 9
fi
echo "  corpus guard OK: +exp=$EXP_LEAF -> $EXPECTED_FT"
# -----------------------------------------------------------------------------
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup accelerate launch --num_processes="$NUM_PROCESSES" \
  gear_sonic/train_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/all_modes/sonic_x2_ultra_dance_finetune_v5 \
  ++num_envs="$NUM_ENVS" \
  ++headless=True \
  ++use_wandb=True \
  ++algo.config.num_learning_iterations="$NUM_ITERS" \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$BASE_CORPUS" \
  ++base_dir=/mnt/ckpt/logs_sonic \
  +checkpoint="$CHECKPOINT" \
  +resume=true \
  > "$LOG" 2>&1 &
echo "launched accelerate PID $!  ->  tail -f $LOG"
