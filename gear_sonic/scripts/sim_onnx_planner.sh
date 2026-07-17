#!/bin/bash
# Standard SIM launch: ONNX planner (the exact graph deployed on PC2) +
# MuJoCo sim + local gamepad. SIM-ONLY -- no --pc2-host, so the pose wire
# binds 127.0.0.1 and the robot cannot subscribe.
#
#   ./gear_sonic/scripts/sim_onnx_planner.sh              # template graph (slow_walk)
#   PLANNER=velocity ./gear_sonic/scripts/sim_onnx_planner.sh   # velocity graph
#
# Controls: L2 = deadman drive (locked 0.3), L1+Y/A = cycle 14 dances,
# L1+B = stop. Needs a pad visible to pygame BEFORE launch (--pad-only
# tears the stack down otherwise) and the env_isaaclab conda env.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT_ROOT="${CKPT_ROOT:-$HOME/x2_cloud_checkpoints}"
GRAPH="${PLANNER:-template}"   # template | velocity
DANCES="$CKPT_ROOT/dances_x2m2"

export KPLANNER_ONNX="$CKPT_ROOT/planner_onnx/x2_planner_${GRAPH}.onnx"
export KPLANNER_DANCES_DIR="$DANCES"
export KPLANNER_FIXED_FWD_MPS=0.3
export PAD_LOCK_SPEED=1
export PAD_DEADMAN=left
export PAD_CLIP_PKL=gear_sonic/data/motions/x2_dances_easy.pkl
export PAD_CLIP_KEYS="$(ls "$DANCES" | sed 's/\.x2m2$//' | paste -sd,)"

MODE_FLAG=()
[[ "$GRAPH" == "template" ]] && MODE_FLAG=(--kplanner-planner-mode "${MODE:-slow_walk}")

exec ./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
  --duration 0 --pad-only "${MODE_FLAG[@]}" \
  --kplanner-warmup-qpos gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v2.pkl \
  --model "$CKPT_ROOT/g1teleop_overnight/sonic/snapshots/exported/walkft_3065_g1.onnx" \
  --kplanner-python "$HOME/miniconda3/envs/env_isaaclab/bin/python"
