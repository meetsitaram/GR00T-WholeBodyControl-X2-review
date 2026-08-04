#!/usr/bin/env bash
# run_x2_vr_robot_teleop.sh -- laptop-side VR teleop against the REAL robot.
#
# This is the SPLIT topology, completely different from the sim-only
# run_x2_quest3_planner_stack.sh (which spawns planner+deploy+recorder
# locally):
#
#   LAPTOP                                   PC2 (robot, via ritual)
#   quest3_manager_x2  --planner_cmd:5563--> pc2_kplanner_onnx
#     (this script)    --arm_targets:5572--> (planner + SONIC deploy +
#     Quest3 WebXR app <--pose:5556--------- pose watchdog + pad bridge
#      on the headset)                        all started by the PAD
#                                             IGNITION RITUAL on PC2)
#
# The robot side must already be up (pad ritual: hold L1+R1+L2+R2 3s ->
# ARMED rumbles -> Y). This script only runs the laptop VR manager and
# connects it to the robot's planner. Killing this script never kills
# the robot stack -- the planner's command watchdog idles the robot
# when the VR stream goes silent.
#
# Usage:
#   gear_sonic/scripts/run_x2_vr_robot_teleop.sh [ROBOT_IP]
#   ROBOT_IP defaults to $X2_ROBOT_IP, then ${X2_PC2_HOST} (agibot-pc2).
#
# Speed: the planner's walk speed is set robot-side by the ritual env
# (KPLANNER_FIXED_FWD_MPS, default 0.3 m/s since 2026-08-04 -- 0.4 was
# too fast indoors). VR speed nudges still apply on top.
#
# After every (re)start of this script: RE-ENTER the WebXR page in the
# headset -- the WS drops on manager restart and buttons go nowhere
# until the page reconnects (see manager.log 'Client connected').
set -euo pipefail

ROBOT_IP="${1:-${X2_ROBOT_IP:-${X2_PC2_HOST}}}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LOG="/tmp/vr_manager_$(date +%Y%m%d_%H%M%S).log"
echo "[vr-robot-teleop] robot=$ROBOT_IP  log=$LOG"
echo "[vr-robot-teleop] reminder: re-enter the WebXR page in the headset now."

exec python gear_sonic/scripts/quest3_manager_x2.py \
    --planner-cmd-connect --planner-cmd-host "$ROBOT_IP" \
    --arm-connect "$ROBOT_IP:5572" \
    --preserve-arms-on-engage \
    --engage-pose-sub-host "$ROBOT_IP" --engage-pose-sub-port 5556 \
    2>&1 | tee "$LOG"
