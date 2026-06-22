#!/usr/bin/env bash
# Live replay of a recorded LeRobot v2.1 episode through the real X2
# (or sim) deploy via the standard pose ZMQ topic on port 5556.
#
# This is the SONIC-loop counterpart to ``replay_x2_kinematic.py`` (MuJoCo
# viewer only). The deploy + SONIC tracking decoder consume the wire
# exactly as they would during a live VLA / teleop session, so the robot
# physically re-executes the trajectory.
#
# Usage
# -----
#
#     ./gear_sonic/scripts/replay_x2_dataset.sh \\
#         --pc2-host 192.168.86.32 \\
#         --dataset x2_reach_and_retract_v1 \\
#         --episode 0
#
# Prereq: SONIC daemons already running on PC2, configured to SUB at
# this laptop's IP:
#
#     ./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \\
#         --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \\
#         --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \\
#         --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \\
#         --lock-head-straight
#
# Safety
# ------
#
# * Banner + 3-second countdown before publishing starts. Ctrl-C in the
#   countdown window aborts cleanly.
# * Ctrl-C during playback sends ~0.5 s of last-frame "hold" then exits.
#   SONIC's safety stack decays PD gains in ~200 ms so this is a soft
#   stop.
# * Object position on the table MUST match the recording; otherwise
#   the hand swings into empty air or collides.
#
# All args are forwarded to the underlying Python entry-point. Run
#
#     ./gear_sonic/scripts/replay_x2_dataset.sh --help
#
# for the full flag list (--countdown, --rate-scale, --loop, --dry-run).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# .venv (the planner/teleop env) ships pyzmq + pyarrow + numpy and is
# the same env the live VLA bridge and Quest 3 manager run in, so the
# wire envelope is guaranteed to be byte-identical to what the deploy
# normally sees.
VENV_PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
    echo "Error: ${VENV_PY} not found." >&2
    echo "Activate / install the .venv first (uv sync or install_scripts/)." >&2
    exit 2
fi

exec "${VENV_PY}" -m gear_sonic.scripts.replay_x2_dataset "$@"
