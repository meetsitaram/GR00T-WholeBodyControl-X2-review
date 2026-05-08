#!/usr/bin/env bash
# x2_mc_mode_probe.sh -- empirically label MC modes by walking the robot
# through every known mode while x2_record_real_run.py captures the bus.
#
# Usage (from any host shell -- auto-relaunches inside docker_x2/x2sim):
#
#   ./gear_sonic_deploy/scripts/x2_mc_mode_probe.sh \
#       --out scratch/runs/mc_mode_probe_$(date +%Y%m%d_%H%M%S)/run.npz \
#       --gantry-confirmed
#
# What it does:
#   1. Captures the initial MC mode (so we can restore on exit).
#   2. Backgrounds x2_record.sh with --track-mc-mode (default ON).
#   3. Walks the modes:
#        STAND_DEFAULT  (baseline; operator-prompt mid-dwell to nudge)
#        DAMPING_DEFAULT
#        PASSIVE_DEFAULT
#        DAMPING_DEFAULT
#        STAND_DEFAULT  (recovery)
#        JOINT_DEFAULT  (observe only -- we don't publish anything)
#        STAND_DEFAULT  (final)
#   4. Restores the captured initial mode.
#   5. SIGINTs the recorder so it finalizes the .npz + prints the per-mode
#      label-inference table.
#
# This is read-only on the bus side: no HAL joint commands are published
# at any point. The only state changes are SetMcAction calls, all of
# which are reverted at exit.
#
# IMPORTANT SAFETY: PASSIVE_DEFAULT puts the robot in zero-torque mode
# (it WILL collapse if not supported). DAMPING_DEFAULT does not actively
# balance. The robot must be on the gantry with feet =< 1 cm off the
# floor, OR fully supported by an operator. Gate is enforced by
# --gantry-confirmed. There is no way to skip the gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_CWD="$(pwd)"

NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'

# ────────────────────────────────────────────────────────────────────────
# Defaults / CLI parsing
# ────────────────────────────────────────────────────────────────────────
OUT_PATH=""
GANTRY_CONFIRMED=0
DWELL_S="6"
PUSH_DWELL_S="8"
NOTE="mc_mode_probe"
PRE_HOLD_S="2"  # let recorder subscribers attach before first SetMcAction
SET_TIMEOUT_S="0.25"
SET_RETRIES="8"
NO_DOCKER=0

# Preserve original argv before the parser shifts through it, so the docker
# re-exec forwards the same args to the inner invocation.
ORIG_ARGS=("$@")

print_help() {
    sed -n '/^# x2_mc_mode_probe.sh/,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '$d' -e 's/^# \{0,1\}//'
    cat <<EOF

Flags:
  --out PATH               Output .npz path. REQUIRED. Must resolve to a
                           host-persistent location (rejected if the path
                           lives on the docker container's ephemeral
                           writable layer; use scratch/runs/... or \$HOME/...).
  --gantry-confirmed       Operator confirms the robot is gantry-supported.
                           Required.
  --dwell-s SECS           Per-mode dwell time. Default: ${DWELL_S}.
  --push-dwell-s SECS      Dwell during the STAND_DEFAULT push test.
                           Default: ${PUSH_DWELL_S}.
  --note STR               Note string forwarded to recorder meta_json.
                           Default: '${NOTE}'.
  --no-docker              Skip the docker auto-relaunch (assumes the
                           current shell already has ROS 2 + aimdk_msgs
                           sourced and the recorder Python deps importable).
  -h, --help               Show this help.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --out)               OUT_PATH="$2"; shift 2 ;;
        --out=*)             OUT_PATH="${1#--out=}"; shift ;;
        --gantry-confirmed)  GANTRY_CONFIRMED=1; shift ;;
        --dwell-s)           DWELL_S="$2"; shift 2 ;;
        --dwell-s=*)         DWELL_S="${1#--dwell-s=}"; shift ;;
        --push-dwell-s)      PUSH_DWELL_S="$2"; shift 2 ;;
        --push-dwell-s=*)    PUSH_DWELL_S="${1#--push-dwell-s=}"; shift ;;
        --note)              NOTE="$2"; shift 2 ;;
        --note=*)            NOTE="${1#--note=}"; shift ;;
        --no-docker)         NO_DOCKER=1; shift ;;
        -h|--help)           print_help; exit 0 ;;
        *)
            echo "ERROR: unknown flag '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

# ────────────────────────────────────────────────────────────────────────
# Persistent-path guard (mirrors x2_record.sh::is_host_persistent_path).
# Container's --rm reaps everything outside /workspace/sonic and $HOME.
# ────────────────────────────────────────────────────────────────────────
is_host_persistent_path() {
    local p="$1"
    [[ "${X2_MC_PROBE_IN_DOCKER:-0}" != "1" ]] && return 0
    [[ -z "$p" ]] && return 1
    local host_home="${X2_HOST_HOME:-$HOME}"
    case "$p" in
        /workspace/sonic|/workspace/sonic/*)   return 0 ;;
        "$host_home"|"$host_home"/*)           return 0 ;;
        *)                                     return 1 ;;
    esac
}

assert_host_persistent_path() {
    local label="$1"
    local p="$2"
    if is_host_persistent_path "$p"; then
        return 0
    fi
    echo -e "${RED}ERROR: $label resolves to '$p' inside the docker container.${NC}" >&2
    echo -e "${RED}       That path is on the container's ephemeral writable layer${NC}" >&2
    echo -e "${RED}       (--rm reaps it on exit), so the recording would be lost${NC}" >&2
    echo -e "${RED}       the moment the run finishes.${NC}" >&2
    echo "" >&2
    echo -e "${YELLOW}       Persistent locations:${NC}" >&2
    echo -e "${YELLOW}         /workspace/sonic/scratch/runs/...${NC}" >&2
    echo -e "${YELLOW}         \$HOME/...${NC}" >&2
    exit 1
}

# ────────────────────────────────────────────────────────────────────────
# docker auto-relaunch (mirrors x2_record.sh)
# ────────────────────────────────────────────────────────────────────────
maybe_relaunch_in_docker() {
    if [[ -d /workspace/sonic ]] || [[ -n "${X2_MC_PROBE_IN_DOCKER:-}" ]]; then
        return 0
    fi
    if (( NO_DOCKER == 1 )); then
        return 0
    fi
    if ! command -v docker &>/dev/null; then
        echo -e "${YELLOW}Note: 'docker' not in PATH and ROS doesn't look sourced;${NC}" >&2
        echo -e "${YELLOW}      x2_mc_mode_probe.sh needs rclpy + aimdk_msgs.${NC}" >&2
        echo -e "${YELLOW}      Install docker + run docker_x2, or source ROS 2${NC}" >&2
        echo -e "${YELLOW}      and re-run with --no-docker.${NC}" >&2
        return 0
    fi
    local compose_dir="$DEPLOY_DIR/docker_x2"
    if [[ ! -f "$compose_dir/docker-compose.yml" ]]; then
        echo -e "${YELLOW}Note: $compose_dir/docker-compose.yml not found; aborting auto-docker.${NC}" >&2
        return 0
    fi

    local domain_id="${X2_REAL_DOMAIN_ID:-0}"
    local tty_args=("-T")
    if [[ -t 0 && -t 1 ]]; then
        tty_args=()
    fi

    echo -e "${BLUE}[auto-docker]${NC} re-exec inside docker_x2/x2sim (mode probe)"
    echo -e "${BLUE}[auto-docker]${NC} mounting \$HOME ($HOME) at $HOME"
    echo -e "${BLUE}[auto-docker]${NC} real-DDS env: ROS_LOCALHOST_ONLY=0, ROS_DOMAIN_ID=$domain_id"
    echo ""

    local script_in_container="/workspace/sonic/gear_sonic_deploy/scripts/$(basename "${BASH_SOURCE[0]}")"
    cd "$compose_dir"
    export X2_MC_PROBE_IN_DOCKER=1
    exec docker compose run --rm \
        "${tty_args[@]}" \
        -e "ROS_LOCALHOST_ONLY=0" \
        -e "ROS_DOMAIN_ID=$domain_id" \
        -e "X2_MC_PROBE_IN_DOCKER=1" \
        -e "X2_HOST_HOME=$HOME" \
        -v "$HOME:$HOME:rw" \
        -w "$USER_CWD" \
        x2sim \
        bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec "$@"' \
        bash "$script_in_container" "$@"
}

maybe_relaunch_in_docker "${ORIG_ARGS[@]}"

# ────────────────────────────────────────────────────────────────────────
# Validations
# ────────────────────────────────────────────────────────────────────────
if [[ -z "$OUT_PATH" ]]; then
    echo -e "${RED}ERROR: --out PATH is required.${NC}" >&2
    print_help >&2
    exit 2
fi

# Resolve relative paths against the operator's invocation cwd (preserved
# across docker re-exec via -w).
if [[ "$OUT_PATH" != /* ]]; then
    OUT_PATH="$USER_CWD/$OUT_PATH"
fi
assert_host_persistent_path "--out" "$OUT_PATH"

if (( GANTRY_CONFIRMED == 0 )); then
    cat >&2 <<EOF
${RED}ERROR: --gantry-confirmed is required.${NC}

This probe walks the robot through PASSIVE_DEFAULT (zero torque -- robot
will collapse if not supported) and DAMPING_DEFAULT (no active balance).
The robot MUST be on the gantry with feet =< 1 cm off the floor, OR
fully supported by an operator.

Re-run with --gantry-confirmed once that's true.
EOF
    exit 2
fi

if ! command -v ros2 &>/dev/null; then
    echo -e "${RED}ERROR: 'ros2' not in PATH. Source ROS 2 Humble + aimdk_msgs first.${NC}" >&2
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────
# MC service helpers (cross-host pattern: 250 ms timeout x 8 retries,
# matches the X2 SDK example_pkg::set_mc_action.cpp recipe).
# ────────────────────────────────────────────────────────────────────────
SETMC="/aimdk_5Fmsgs/srv/SetMcAction"
GETMC="/aimdk_5Fmsgs/srv/GetMcAction"

mc_get_action() {
    # Prints the current action_desc string on stdout. Returns 0 on
    # success, 1 if every retry failed.
    local out=""
    for _ in $(seq 1 "$SET_RETRIES"); do
        if out=$(timeout "$SET_TIMEOUT_S" \
                ros2 service call "$GETMC" \
                aimdk_msgs/srv/GetMcAction \
                "{request: {}}" 2>/dev/null); then
            # The reply YAML carries either info.action_desc (newer) or
            # action_desc (older). awk picks whichever non-empty value
            # appears on the first matching line.
            local mode
            mode=$(echo "$out" | awk -F': ' '
                /action_desc:/ {
                    gsub(/[\r\n"\047]/, "", $2)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
                    if ($2 != "") { print $2; exit }
                }
            ')
            if [[ -n "$mode" ]]; then
                echo "$mode"
                return 0
            fi
        fi
        sleep 0.1
    done
    return 1
}

mc_set_action() {
    # $1 = mode string (e.g. STAND_DEFAULT). Returns 0 on success.
    local mode="$1"
    for _ in $(seq 1 "$SET_RETRIES"); do
        if timeout "$SET_TIMEOUT_S" \
                ros2 service call "$SETMC" \
                aimdk_msgs/srv/SetMcAction \
                "{request: {action_desc: '$mode'}}" \
                2>/dev/null \
                | grep -q -E '(success.*[Tt]rue|status.*0)'; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# ────────────────────────────────────────────────────────────────────────
# Recorder lifecycle
# ────────────────────────────────────────────────────────────────────────
RECORDER="$DEPLOY_DIR/scripts/x2_record_real_run.py"
RECORDER_PID=""

start_recorder() {
    mkdir -p "$(dirname "$OUT_PATH")"
    echo -e "${BLUE}[recorder]${NC} backgrounding -> $OUT_PATH"
    python3 "$RECORDER" \
        --out "$OUT_PATH" \
        --note "$NOTE" \
        --track-mc-mode \
        --mc-poll-hz 5.0 \
        --status-period 1.0 &
    RECORDER_PID=$!
    sleep 0.5
    if ! kill -0 "$RECORDER_PID" 2>/dev/null; then
        echo -e "${RED}[recorder] exited immediately. Aborting probe.${NC}" >&2
        RECORDER_PID=""
        exit 1
    fi
    echo -e "${BLUE}[recorder]${NC} pid=$RECORDER_PID; pre-attach hold ${PRE_HOLD_S}s"
    sleep "$PRE_HOLD_S"
}

stop_recorder() {
    [[ -z "$RECORDER_PID" ]] && return 0
    if kill -0 "$RECORDER_PID" 2>/dev/null; then
        echo -e "${BLUE}[recorder]${NC} sending SIGINT (pid $RECORDER_PID); will print summary"
        kill -INT "$RECORDER_PID" 2>/dev/null || true
        wait "$RECORDER_PID" 2>/dev/null || true
    fi
    RECORDER_PID=""
}

# ────────────────────────────────────────────────────────────────────────
# Restore-on-exit trap. Always lands the robot back in the mode it was
# in when the script started, even on Ctrl-C / failure. Important: never
# leave the robot in PASSIVE_DEFAULT or DAMPING_DEFAULT after our exit.
# ────────────────────────────────────────────────────────────────────────
INITIAL_MODE=""
RESTORE_DONE=0

restore_initial_mode() {
    local rc=$?
    if (( RESTORE_DONE == 1 )); then return $rc; fi
    RESTORE_DONE=1
    if [[ -n "$INITIAL_MODE" ]]; then
        echo ""
        echo -e "${YELLOW}[cleanup]${NC} restoring MC -> $INITIAL_MODE"
        if mc_set_action "$INITIAL_MODE"; then
            echo -e "${GREEN}[cleanup]${NC} restored OK."
        else
            echo -e "${RED}[cleanup]${NC} FAILED to restore $INITIAL_MODE."
            echo -e "${RED}           Manually: ros2 service call $SETMC \\${NC}"
            echo -e "${RED}             aimdk_msgs/srv/SetMcAction${NC}"
            echo -e "${RED}             \"{request: {action_desc: '$INITIAL_MODE'}}\"${NC}"
        fi
    fi
    stop_recorder
    return $rc
}
trap restore_initial_mode EXIT INT TERM

# ────────────────────────────────────────────────────────────────────────
# Mode walk
# ────────────────────────────────────────────────────────────────────────
echo -e "${BOLD}=== X2 MC mode probe ===${NC}"
echo -e "Output: $OUT_PATH"
echo -e "Dwell:  ${DWELL_S}s per mode  (push window: ${PUSH_DWELL_S}s)"
echo ""

echo -e "${BLUE}[step 0]${NC} capturing initial MC mode ..."
if INITIAL_MODE="$(mc_get_action)"; then
    echo -e "${GREEN}[step 0]${NC} initial mode: $INITIAL_MODE"
else
    echo -e "${RED}ERROR: GetMcAction failed -- cannot reach $GETMC.${NC}" >&2
    echo -e "${RED}       Check that MC is running and ROS_DOMAIN_ID matches.${NC}" >&2
    exit 1
fi
echo ""

start_recorder

walk_step() {
    # $1 = mode, $2 = dwell_s, [$3 = mid-dwell prompt]
    local mode="$1" dwell="$2" prompt="${3:-}"
    echo ""
    echo -e "${BOLD}[walk]${NC} -> ${mode} (dwell ${dwell}s)"
    if ! mc_set_action "$mode"; then
        echo -e "${RED}[walk] SetMcAction $mode failed; aborting walk early.${NC}" >&2
        return 1
    fi
    if [[ -n "$prompt" ]]; then
        local half=$(awk "BEGIN{printf \"%.2f\", ${dwell}/2}")
        sleep "$half"
        echo -e "${YELLOW}[OPERATOR] $prompt${NC}"
        sleep "$half"
    else
        sleep "$dwell"
    fi
    return 0
}

# Sequence. Order rationale:
#   - start in STAND_DEFAULT so we have a "baseline active stand" segment
#     that includes a operator-prompted nudge (for disturbance-rejection).
#   - descend through DAMPING_DEFAULT into PASSIVE_DEFAULT so the
#     transition zero-torque entry/exit is visible.
#   - climb back up to STAND_DEFAULT.
#   - then JOINT_DEFAULT for an "MC yields the bus alone" reading
#     (we don't publish; we just observe what JD does on its own).
#   - end in STAND_DEFAULT before the trap restores the captured initial
#     mode (often will be STAND_DEFAULT itself, in which case the trap is
#     a no-op).
walk_step STAND_DEFAULT     "$PUSH_DWELL_S" "Give the robot a gentle 1s sideways nudge NOW."
walk_step DAMPING_DEFAULT   "$DWELL_S"
walk_step PASSIVE_DEFAULT   "$DWELL_S"
walk_step DAMPING_DEFAULT   "$DWELL_S"
walk_step STAND_DEFAULT     "$DWELL_S"
walk_step JOINT_DEFAULT     "$DWELL_S"
walk_step STAND_DEFAULT     "$DWELL_S"

echo ""
echo -e "${GREEN}[walk]${NC} mode walk complete; restoring initial mode + finalizing recorder."
# Trap handles the rest.
exit 0
