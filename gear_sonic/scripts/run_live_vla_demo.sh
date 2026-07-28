#!/usr/bin/env bash
# Deprecated wrapper — sim VLA now lives in run_x2_vla_runtime.sh
# (omit --pc2-host for sim, same pattern as run_x2_quest3_planner_stack.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[DEPRECATED] run_live_vla_demo.sh forwards to run_x2_vla_runtime.sh" >&2
echo "             Omit --pc2-host for sim; pass --pc2-host for real robot." >&2
exec "${SCRIPT_DIR}/run_x2_vla_runtime.sh" "$@"
