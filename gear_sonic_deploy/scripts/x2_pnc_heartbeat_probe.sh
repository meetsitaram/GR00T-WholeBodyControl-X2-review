#!/usr/bin/env bash
# x2_pnc_heartbeat_probe.sh -- thin docker wrapper around
# x2_pnc_heartbeat_probe.py. Auto-launches inside docker_x2/x2sim with
# the same DDS env (ROS_LOCALHOST_ONLY=0, ROS_DOMAIN_ID=0) the
# input-source probe used.
#
# Usage:
#
#   ./gear_sonic_deploy/scripts/x2_pnc_heartbeat_probe.sh \
#       --gantry-confirmed \
#       --log-json scratch/probes/pnc_heartbeat_$(date +%Y%m%d_%H%M%S).json
#
# Pre-condition: MC must be in PASSIVE_DEFAULT or DAMPING_DEFAULT (so
# the publishd zero-kp commands don't conflict with active balancing).
# The wrapper does *not* enforce this -- the Python script does, and
# refuses to run otherwise (override with --allow-balancing-mode).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_CWD="$(pwd)"

NC='\033[0m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'

NO_DOCKER=0
PASS_ARGS=()
ORIG_ARGS=("$@")
LOG_JSON=""

while (( $# > 0 )); do
    case "$1" in
        --no-docker) NO_DOCKER=1; shift ;;
        --log-json)  LOG_JSON="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --log-json=*) LOG_JSON="${1#--log-json=}"; PASS_ARGS+=("$1"); shift ;;
        -h|--help)
            sed -n '/^# x2_pnc_heartbeat_probe.sh/,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
                | sed -e '$d' -e 's/^# \{0,1\}//'
            exec python3 "$SCRIPT_DIR/x2_pnc_heartbeat_probe.py" --help 2>/dev/null \
                || true
            exit 0
            ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

# ────────────────────────────────────────────────────────────────────────
# Persistent-path guard for --log-json (only checked inside docker).
# ────────────────────────────────────────────────────────────────────────
is_host_persistent_path() {
    local p="$1"
    [[ "${X2_PNC_HB_IN_DOCKER:-0}" != "1" ]] && return 0
    [[ -z "$p" ]] && return 0
    local host_home="${X2_HOST_HOME:-$HOME}"
    case "$p" in
        /workspace/sonic|/workspace/sonic/*)   return 0 ;;
        "$host_home"|"$host_home"/*)           return 0 ;;
        *)                                     return 1 ;;
    esac
}

# ────────────────────────────────────────────────────────────────────────
# docker auto-relaunch
# ────────────────────────────────────────────────────────────────────────
maybe_relaunch_in_docker() {
    if [[ -d /workspace/sonic ]] || [[ -n "${X2_PNC_HB_IN_DOCKER:-}" ]]; then
        return 0
    fi
    if (( NO_DOCKER == 1 )); then
        return 0
    fi
    if ! command -v docker &>/dev/null; then
        echo -e "${YELLOW}Note: 'docker' not in PATH; pnc-heartbeat probe needs ros2 + aimdk_msgs.${NC}" >&2
        return 0
    fi
    local compose_dir="$DEPLOY_DIR/docker_x2"
    if [[ ! -f "$compose_dir/docker-compose.yml" ]]; then
        echo -e "${YELLOW}Note: $compose_dir/docker-compose.yml not found.${NC}" >&2
        return 0
    fi

    local domain_id="${X2_REAL_DOMAIN_ID:-0}"
    local tty_args=("-T")
    if [[ -t 0 && -t 1 ]]; then tty_args=(); fi

    echo -e "${BLUE}[auto-docker]${NC} re-exec inside docker_x2/x2sim (pnc heartbeat probe)"
    cd "$compose_dir"
    export X2_PNC_HB_IN_DOCKER=1
    exec docker compose run --rm \
        "${tty_args[@]}" \
        -e "ROS_LOCALHOST_ONLY=0" \
        -e "ROS_DOMAIN_ID=$domain_id" \
        -e "X2_PNC_HB_IN_DOCKER=1" \
        -e "X2_HOST_HOME=$HOME" \
        -v "$HOME:$HOME:rw" \
        -w "$USER_CWD" \
        x2sim \
        bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec "$@"' \
        bash "${BASH_SOURCE[0]}" "$@"
}

maybe_relaunch_in_docker "${ORIG_ARGS[@]}"

# ────────────────────────────────────────────────────────────────────────
# Now inside docker (or --no-docker). Validate persistent log path.
# ────────────────────────────────────────────────────────────────────────
if [[ -n "$LOG_JSON" ]]; then
    # Resolve relative paths against operator's invocation cwd.
    if [[ "$LOG_JSON" != /* ]]; then
        LOG_JSON="$USER_CWD/$LOG_JSON"
        # Re-write the corresponding entry in PASS_ARGS.
        for ((i=0; i<${#PASS_ARGS[@]}; i++)); do
            if [[ "${PASS_ARGS[$i]}" == "--log-json" ]]; then
                PASS_ARGS[$((i+1))]="$LOG_JSON"
            elif [[ "${PASS_ARGS[$i]}" == --log-json=* ]]; then
                PASS_ARGS[$i]="--log-json=$LOG_JSON"
            fi
        done
    fi
    if ! is_host_persistent_path "$LOG_JSON"; then
        echo -e "${RED}ERROR: --log-json '$LOG_JSON' not on a host-persistent mount.${NC}" >&2
        echo -e "${YELLOW}       Use scratch/probes/... or \$HOME/...${NC}" >&2
        exit 1
    fi
fi

PYBIN="${PYTHON:-python3}"
exec "$PYBIN" "$SCRIPT_DIR/x2_pnc_heartbeat_probe.py" "${PASS_ARGS[@]}"
