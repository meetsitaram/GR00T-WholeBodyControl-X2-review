#!/usr/bin/env bash
# Six-robot showcase: policy-generated route clips played natively, one env
# per destination, identical spawn pose, presentation cameras/colors/orbs.
# RECORD=1 additionally captures the screen to an mp4 via ffmpeg x11grab.
set -euo pipefail
cd "$(dirname "$0")/../../.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

SONIC_DIR=~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528
KITCHEN=~/projects/x2-kitchen-sim/assets/kitchen

export KP_MOTION_FILE="${MOTION_FILE:-gear_sonic/data/motions/kp6_converge.pkl}"
export KP_ORIGINS_JSON="${MOTION_FILE:-gear_sonic/data/motions/kp6_converge.pkl}.origins.json"

# NO_WORLD=1: bare ground, no splat/collision USD — isolates multi-robot
# physics from world-mesh (splat fuzz) issues
# NO_FALL_RESET=1: disable the anchor_pos (height) fall-reset termination
WORLD_KEYS=()
[[ -n "${NO_FALL_RESET:-}" ]] && WORLD_KEYS+=("++manager_env.terminations.anchor_pos=null")
# EXTRA_KEYS: space-separated additional hydra overrides
[[ -n "${EXTRA_KEYS:-}" ]] && read -ra _EK <<< "$EXTRA_KEYS" && WORLD_KEYS+=("${_EK[@]}")
if [[ -z "${NO_WORLD:-}" ]]; then
  WORLD_KEYS+=(
    "++manager_env.config.world_usd=$KITCHEN/kitchen_splat.usdz"
    "++manager_env.config.world_collision_usd=$KITCHEN/kitchen_collision.usd"
  )
fi
DISPLAY="${DISPLAY:-:1}" \
~/projects/g1-kitchen-sim/.venv/bin/python -u gear_sonic/scripts/x2-navigation/video_showcase_rig.py \
  --run-dir "$SONIC_DIR" \
  --onnx "$SONIC_DIR/exported/softland_4800_g1.onnx" \
  --chase-env "${CHASE_ENV:-1}" \
  +num_envs="${NUM_ENVS:-6}" +headless=False \
  ++manager_env.config.replicate_physics=false \
  ++manager_env.config.env_spacing=0.0 \
  ++env_spacing=0.0 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="${MOTION_FILE:-gear_sonic/data/motions/kp6_converge.pkl}" \
  ++manager_env.commands.motion.motion_lib_cfg.fine_tune_dataset.enable=false \
  ++manager_env.commands.motion.debug_vis=false \
  "${WORLD_KEYS[@]}" \
  '++manager_env.config.world_pos=[-19.99,-75.96,0.0]' \
  ++manager_env.terminations.ee_body_pos=null \
  ++manager_env.terminations.foot_pos_xyz=null \
  ++manager_env.terminations.anchor_ori_full=null \
  2>&1 | tee /tmp/claude-1000/showcase.log
