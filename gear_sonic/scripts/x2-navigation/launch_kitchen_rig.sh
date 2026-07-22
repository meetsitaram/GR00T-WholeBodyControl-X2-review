#!/usr/bin/env bash
# Kitchen driving rig — the session-validated launch (2026-07-22).
# Pad -> pad_locomotion_bridge(:5563) -> embedded sim-clocked kplanner ->
# ring -> SONIC softland_4800 -> PhysX, in the Scaniverse kitchen splat.
#
# Prereqs:
#   - pad_locomotion_bridge.py running (DISPLAY=:1, PUB :5563)
#   - Isaac launch protocol: kill prior instances BY PID, verify VRAM < 3GB
#
# Knobs (env vars):
#   KP_REPLAN_THRESH=32   half-chunk planner commitment (~1.07s; default 2)
#   KP_HEAD_CAM=1         robot-eye camera + snapshots (default on here)
#   KP_SNAP_DIR=...       snapshot dir (default /tmp/claude-1000/kp_head_cam)
#   NO_MARKERS=1          debug_vis off (clean camera views)
#   echo <name> > /tmp/claude-1000/kp_label   -> capture waypoint at robot pose
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

SONIC_DIR=~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528
FS=~/x2_cloud_checkpoints/fixed_scratch
KITCHEN=~/projects/x2-kitchen-sim/assets/kitchen

EXTRA=()
[[ -n "${NO_MARKERS:-}" ]] && EXTRA+=("++manager_env.commands.motion.debug_vis=false")

KP_PAD=1 KP_HIDE_TERRAIN=1 KP_HEAD_CAM="${KP_HEAD_CAM:-1}" DISPLAY="${DISPLAY:-:1}" \
~/projects/g1-kitchen-sim/.venv/bin/python -u gear_sonic/scripts/x2-navigation/run_x2_kplanner_env2.py \
  --run-dir "$SONIC_DIR" \
  --onnx "$SONIC_DIR/exported/softland_4800_g1.onnx" \
  --vqvae-ckpt "$FS/vqvae/model-step=0300000.ckpt" \
  --pose-ckpt "$FS/pose_500k/model-step=0500000.ckpt" \
  --root-ckpt "$FS/root/model-step=0300000.ckpt" \
  --planner-mode slow_walk \
  +num_envs=1 +headless=False \
  ++manager_env.config.enable_cameras=true \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/motions/x2_sonic_feasible_stand_single.pkl \
  ++manager_env.config.world_usd="$KITCHEN/kitchen_splat.usdz" \
  ++manager_env.config.world_collision_usd="$KITCHEN/kitchen_collision.usd" \
  '++manager_env.config.world_pos=[-19.99,-75.96,0.0]' \
  ++manager_env.terminations.ee_body_pos=null \
  ++manager_env.terminations.foot_pos_xyz=null \
  ++manager_env.terminations.anchor_ori_full=null \
  "${EXTRA[@]}" \
  2>&1 | tee /tmp/claude-1000/kpe2_kitchen.log
