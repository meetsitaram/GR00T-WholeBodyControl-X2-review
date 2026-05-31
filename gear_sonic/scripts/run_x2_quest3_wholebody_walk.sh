#!/usr/bin/env bash
# X2 Quest 3 whole-body-walking stack runner.
#
# Thin exec-wrapper around ``run_x2_quest3_planner_stack.sh`` that pins
# the walking-friendly defaults validated on 2026-05-30:
#
#   LOCO_DECOUPLED_ARMS=0   -- recorder ignores the manager's frozen
#                              arm pose during LOCOMOTION and lets the
#                              kplanner's gait-coupled arm swing flow
#                              through to the deploy. The natural
#                              counter-rotation restores angular-
#                              momentum coupling the policy was
#                              trained on (``x2_ultra_locowalk``) and
#                              improves forward-walking quality.
#
# Use this script for sessions where you just want the robot to WALK
# (commute between waypoints, demos, gait-quality tuning) and you are
# NOT carrying a payload between manipulation spots. If you need the
# ARM_MANIPULATION -> LOCOMOTION arm-hold workflow (e.g. holding a
# tool while walking), use the main stack instead:
#
#   ./gear_sonic/scripts/run_x2_quest3_planner_stack.sh
#
# All wrapper flags accepted by ``run_x2_quest3_planner_stack.sh`` are
# forwarded transparently. The operator can still override the
# LOCO_DECOUPLED_ARMS default from the command line, e.g. for an A/B
# regression check:
#
#   ./gear_sonic/scripts/run_x2_quest3_wholebody_walk.sh \
#       --loco-decoupled-arms 1   # restore frozen-arms override
#
# Implementation note: this is a one-line ``exec`` so any future
# improvements to ``run_x2_quest3_planner_stack.sh`` (new flags,
# bug fixes, port changes, etc.) are inherited automatically without
# requiring this script to track them. Do NOT add stack-level logic
# here -- if you need to change planner / deploy behaviour, change it
# in the underlying script so the manipulation stack benefits too.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_STACK="${SCRIPT_DIR}/run_x2_quest3_planner_stack.sh"

if [[ ! -x "${MAIN_STACK}" ]]; then
    echo "[whole-body-walk] FATAL: cannot find or exec ${MAIN_STACK}" >&2
    exit 1
fi

# Pin walking-friendly defaults. ``${VAR:-default}`` form means an
# operator-supplied env var still wins, so an A/B check is still
# `LOCO_DECOUPLED_ARMS=1 ./run_x2_quest3_wholebody_walk.sh`.
export LOCO_DECOUPLED_ARMS="${LOCO_DECOUPLED_ARMS:-0}"

echo "[whole-body-walk] LOCO_DECOUPLED_ARMS=${LOCO_DECOUPLED_ARMS} (planner-coupled arms)"
echo "[whole-body-walk] forwarding $# arg(s) to ${MAIN_STACK}"

exec "${MAIN_STACK}" "$@"
