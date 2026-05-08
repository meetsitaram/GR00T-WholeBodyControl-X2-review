#!/usr/bin/env bash
# x2_mc_input_source_probe.sh -- empirical confirmation of MC's
# InputManager arbitration API.
#
# Answers three open questions from
# scratch/probes/mc_introspect_*/FINDINGS.md without publishing on the
# joint-command bus (so MC's 200 ms expiration_time watchdog is the
# safety net at every step):
#
#   Q1  Does SetMcInputSource MODIFY+ENABLE on the existing 'pnc' slot
#       actually claim the bus, and does GetCurrentInputSource report it?
#       Does the 200 ms watchdog auto-reclaim if we don't heartbeat?
#
#   Q2  Is the source registry locked to the five names hard-coded in
#       mc.yaml (rc / vr / app_proxy / interaction / pnc), or does
#       SetMcInputSource ADD with a fresh name (e.g. 'sonic_policy')
#       work? If it works, what's the ENABLE / DISABLE / DELETE
#       round-trip look like?
#
#   Q3  If mc.yaml is fixed and we'd need to ship a modified copy,
#       what's the path on PC1 / PC2, who can write it, and how is
#       MC restarted (systemd unit? em start_app? ad-hoc launcher?)
#
# Usage (from any host shell -- auto-relaunches inside docker_x2/x2sim):
#
#   ./gear_sonic_deploy/scripts/x2_mc_input_source_probe.sh \
#       --out-dir scratch/probes/mc_input_source_$(date +%Y%m%d_%H%M%S) \
#       --gantry-confirmed
#
#   # Optional: also skim PC1/PC2 filesystem for mc.yaml + MC supervisor.
#   #   --ssh-pc HOST 'sshpass-style password' (or rely on key auth).
#   ./gear_sonic_deploy/scripts/x2_mc_input_source_probe.sh \
#       --out-dir scratch/probes/mc_input_source_$(date +%Y%m%d_%H%M%S) \
#       --gantry-confirmed \
#       --ssh-pc run@10.0.1.41 \
#       --ssh-password 1
#
# What gets written (under --out-dir):
#
#   00_meta.txt                  context (timestamps, ROS env, git sha)
#   01_baseline_get_mc_action.txt
#   01_baseline_get_input.txt
#   02_pnc_modify_resp.txt       Test B step 1
#   02_pnc_get_after_modify.txt
#   02_pnc_enable_resp.txt       Test B step 2
#   02_pnc_get_after_enable_T+050ms.txt
#   02_pnc_get_after_enable_T+500ms.txt
#   02_pnc_get_after_enable_T+1500ms.txt
#   02_pnc_disable_resp.txt
#   02_pnc_get_after_disable.txt
#   03_sonic_add_resp.txt        Test C
#   03_sonic_enable_resp.txt     (only if ADD succeeded)
#   03_sonic_get_after_enable.txt
#   03_sonic_disable_resp.txt
#   03_sonic_delete_resp.txt
#   04_pc_skim.txt               Test D (only with --ssh-pc)
#   99_FINDINGS.md               auto-generated answers to Q1/Q2/Q3
#
# Safety: NO joint-command publishes. SetMcInputSource and
# GetCurrentInputSource only. If anything goes sideways, MC's
# expiration_time (200 ms) reclaims the bus.

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
# Defaults / CLI
# ────────────────────────────────────────────────────────────────────────
OUT_DIR=""
GANTRY_CONFIRMED=0
NO_DOCKER=0
SSH_PC=""              # e.g. run@10.0.1.41
SSH_PASSWORD=""        # optional; if set, uses sshpass
PNC_PRIORITY=40        # mc.yaml default
PNC_TIMEOUT=200        # ms; matches expiration_time
SONIC_NAME="sonic_policy"
SONIC_PRIORITY=45
SONIC_TIMEOUT=200
RPC_TIMEOUT="3.0"      # ros2 service call wall timeout (discovery+call+exit)
RPC_RETRIES=3

# Preserve the operator's original argv before the parser shifts through
# it, so the docker re-exec can forward the same args.
ORIG_ARGS=("$@")

print_help() {
    sed -n '/^# x2_mc_input_source_probe.sh/,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '$d' -e 's/^# \{0,1\}//'
    cat <<EOF

Flags:
  --out-dir PATH           Output directory. REQUIRED. Must resolve to a
                           host-persistent location (rejected if the path
                           lives on the docker container's ephemeral
                           writable layer; use scratch/probes/... or
                           \$HOME/...).
  --gantry-confirmed       Operator confirms the robot is gantry-supported
                           or in a non-balancing MC mode (PASSIVE_DEFAULT
                           / DAMPING_DEFAULT). REQUIRED.
  --pnc-priority N         Priority used for the pnc MODIFY in Test B.
                           Default: ${PNC_PRIORITY} (mc.yaml canonical).
  --pnc-timeout MS         Timeout (ms) used for the pnc MODIFY. Default:
                           ${PNC_TIMEOUT} (matches expiration_time).
  --sonic-name STR         Name used in Test C (ADD with a fresh name).
                           Default: '${SONIC_NAME}'.
  --sonic-priority N       Priority used in Test C. Default: ${SONIC_PRIORITY}.
  --sonic-timeout MS       Timeout used in Test C. Default: ${SONIC_TIMEOUT}.
  --ssh-pc USER@HOST       Run Test D: ssh into this host and skim
                           filesystem for mc.yaml + MC supervisor.
                           Off by default. Common value: run@10.0.1.41.
  --ssh-password STR       Optional password for sshpass. If not set,
                           assumes key-based auth.
  --no-docker              Skip the docker auto-relaunch (assumes the
                           current shell already has ROS 2 + aimdk_msgs
                           sourced).
  -h, --help               Show this help.

What's safe / what isn't:

  Tests A, B, C are state-changing on MC's arbitration table but DO NOT
  publish on /aima/hal/joint/*/command. If the call accidentally hangs
  or this script segfaults, MC's expiration_time (200 ms) will reclaim
  the bus and the robot stays in whatever mode it was in. The robot
  itself is not commanded to move.

  Test D is read-only (ssh into a remote host and run ls / find / cat /
  ps / systemctl-status / journalctl). It writes nothing on the remote
  side.

  --gantry-confirmed is required regardless. The 200 ms gap during
  which pnc owns the bus but doesn't publish is short enough that
  motors hold their last command, but if MC is currently
  STAND_DEFAULT and weight-bearing, you still want the gantry catching
  the robot if anything misbehaves.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --out-dir)           OUT_DIR="$2"; shift 2 ;;
        --out-dir=*)         OUT_DIR="${1#--out-dir=}"; shift ;;
        --gantry-confirmed)  GANTRY_CONFIRMED=1; shift ;;
        --pnc-priority)      PNC_PRIORITY="$2"; shift 2 ;;
        --pnc-priority=*)    PNC_PRIORITY="${1#--pnc-priority=}"; shift ;;
        --pnc-timeout)       PNC_TIMEOUT="$2"; shift 2 ;;
        --pnc-timeout=*)     PNC_TIMEOUT="${1#--pnc-timeout=}"; shift ;;
        --sonic-name)        SONIC_NAME="$2"; shift 2 ;;
        --sonic-name=*)      SONIC_NAME="${1#--sonic-name=}"; shift ;;
        --sonic-priority)    SONIC_PRIORITY="$2"; shift 2 ;;
        --sonic-priority=*)  SONIC_PRIORITY="${1#--sonic-priority=}"; shift ;;
        --sonic-timeout)     SONIC_TIMEOUT="$2"; shift 2 ;;
        --sonic-timeout=*)   SONIC_TIMEOUT="${1#--sonic-timeout=}"; shift ;;
        --ssh-pc)            SSH_PC="$2"; shift 2 ;;
        --ssh-pc=*)          SSH_PC="${1#--ssh-pc=}"; shift ;;
        --ssh-password)      SSH_PASSWORD="$2"; shift 2 ;;
        --ssh-password=*)    SSH_PASSWORD="${1#--ssh-password=}"; shift ;;
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
# Persistent-path guard (mirrors x2_record.sh / x2_mc_introspect.sh).
# ────────────────────────────────────────────────────────────────────────
is_host_persistent_path() {
    local p="$1"
    [[ "${X2_INPUTSRC_IN_DOCKER:-0}" != "1" ]] && return 0
    [[ -z "$p" ]] && return 1
    local host_home="${X2_HOST_HOME:-$HOME}"
    case "$p" in
        /workspace/sonic|/workspace/sonic/*)   return 0 ;;
        "$host_home"|"$host_home"/*)           return 0 ;;
        *)                                     return 1 ;;
    esac
}

assert_host_persistent_path() {
    local p="$1"
    if is_host_persistent_path "$p"; then
        return 0
    fi
    echo -e "${RED}ERROR: --out-dir resolves to '$p' inside docker.${NC}" >&2
    echo -e "${RED}       Container ephemeral layer would lose the dump.${NC}" >&2
    echo -e "${YELLOW}       Use scratch/probes/... or \$HOME/...${NC}" >&2
    exit 1
}

# ────────────────────────────────────────────────────────────────────────
# docker auto-relaunch
# ────────────────────────────────────────────────────────────────────────
maybe_relaunch_in_docker() {
    if [[ -d /workspace/sonic ]] || [[ -n "${X2_INPUTSRC_IN_DOCKER:-}" ]]; then
        return 0
    fi
    if (( NO_DOCKER == 1 )); then
        return 0
    fi
    if ! command -v docker &>/dev/null; then
        echo -e "${YELLOW}Note: 'docker' not in PATH; probe needs ros2 + aimdk_msgs.${NC}" >&2
        echo -e "${YELLOW}      Source ROS 2 manually + use --no-docker, or install docker.${NC}" >&2
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

    echo -e "${BLUE}[auto-docker]${NC} re-exec inside docker_x2/x2sim (input-source probe)"
    local script_in_container="/workspace/sonic/gear_sonic_deploy/scripts/$(basename "${BASH_SOURCE[0]}")"
    cd "$compose_dir"
    export X2_INPUTSRC_IN_DOCKER=1
    exec docker compose run --rm \
        "${tty_args[@]}" \
        -e "ROS_LOCALHOST_ONLY=0" \
        -e "ROS_DOMAIN_ID=$domain_id" \
        -e "X2_INPUTSRC_IN_DOCKER=1" \
        -e "X2_INPUTSRC_TEST_D_DONE=${X2_INPUTSRC_TEST_D_DONE:-0}" \
        -e "X2_HOST_HOME=$HOME" \
        -v "$HOME:$HOME:rw" \
        -w "$USER_CWD" \
        x2sim \
        bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec "$@"' \
        bash "$script_in_container" "$@"
}

# ────────────────────────────────────────────────────────────────────────
# Test D (SSH skim) runs on the HOST -- the docker_x2/x2sim image does
# not ship `ssh` / `sshpass`. We drop the dump into --out-dir before
# re-exec so it's still captured in the same probe directory. The
# inner (in-docker) invocation skips this branch via X2_INPUTSRC_TEST_D_DONE.
# ────────────────────────────────────────────────────────────────────────
host_run_test_d() {
    [[ -z "$SSH_PC" ]] && return 0
    [[ "${X2_INPUTSRC_TEST_D_DONE:-0}" == "1" ]] && return 0
    [[ "${X2_INPUTSRC_IN_DOCKER:-0}" == "1" ]] && return 0  # never run inside docker

    if ! command -v ssh &>/dev/null; then
        echo -e "${YELLOW}[host] ssh not available; skipping Test D.${NC}" >&2
        return 0
    fi

    # Resolve the out-dir before the persistent-path guard runs (we
    # only need it to be a writable absolute path on the host).
    local host_out="$OUT_DIR"
    if [[ "$host_out" != /* ]]; then
        host_out="$USER_CWD/$host_out"
    fi
    mkdir -p "$host_out"

    local SSH_BIN=(ssh)
    local SSH_OPTS=(-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new
                    -o PreferredAuthentications=password,publickey)
    if [[ -n "$SSH_PASSWORD" ]]; then
        if command -v sshpass &>/dev/null; then
            SSH_BIN=(sshpass -p "$SSH_PASSWORD" ssh)
            SSH_OPTS+=(-o BatchMode=no)
        else
            echo -e "${YELLOW}[host] sshpass not in PATH; trying key auth.${NC}" >&2
            SSH_OPTS+=(-o BatchMode=yes)
        fi
    else
        SSH_OPTS+=(-o BatchMode=yes)
    fi
    local SSH_CMD=("${SSH_BIN[@]}" "${SSH_OPTS[@]}" "$SSH_PC")

    echo -e "${BLUE}[host]${NC} Test D: SSH skim of $SSH_PC -> $host_out/04_pc_skim.txt"
    {
        echo "# Test D: ssh ${SSH_PC}: mc.yaml + MC supervisor skim"
        echo "# t_wall: $(date -Iseconds)"
        echo "# (run on host -- docker_x2 has no ssh)"
        echo "# ----------------------------------------------------------"
        echo ""
        echo "── uname / version ──"
        "${SSH_CMD[@]}" 'uname -a; cat /etc/aima-version 2>/dev/null; head -5 /etc/os-release 2>/dev/null' 2>&1 \
            || echo "(ssh failed)"
        echo ""
        echo "── mc.yaml location and ownership ──"
        "${SSH_CMD[@]}" '
            for d in /agibot/software/mc_param /opt/aima/mc_param /etc/aima/mc_param; do
                if [ -d "$d" ]; then
                    echo "── $d ──"
                    find "$d" -maxdepth 4 -type f -name "mc.yaml" -printf "%p  uid=%u  mode=%m  mtime=%t\n" 2>/dev/null
                fi
            done
            echo ""
            echo "── full filesystem search ──"
            find / -maxdepth 6 -name mc.yaml 2>/dev/null | head -20
        ' 2>&1 || true
        echo ""
        echo "── can current ssh user write to mc.yaml? ──"
        "${SSH_CMD[@]}" '
            f=$(find / -maxdepth 6 -name mc.yaml 2>/dev/null | head -1)
            if [ -z "$f" ]; then
                echo "(mc.yaml not found)"
            else
                echo "candidate: $f"
                ls -la "$f"
                if [ -w "$f" ]; then
                    echo "  -> WRITABLE by current user ($(whoami))"
                else
                    echo "  -> NOT writable by current user ($(whoami))"
                fi
                echo "  parent dir:  $(stat -c "uid=%U gid=%G mode=%a" "$(dirname "$f")")"
            fi
        ' 2>&1 || true
        echo ""
        echo "── MC processes ──"
        "${SSH_CMD[@]}" "ps -eo pid,user,etime,cmd | grep -iE 'mc_(ros2|main)|mc_em|aimrt' | grep -v grep" 2>&1 || true
        echo ""
        echo "── MC systemd units (any unit named 'mc' or 'aima*') ──"
        "${SSH_CMD[@]}" '
            (systemctl list-units --type=service --no-pager 2>/dev/null \
                | grep -iE "(^|[[:space:]])(mc|aima|aimrt|em)[-_.]" \
                || echo "(no matching units)")
            echo ""
            for unit in mc aimrt-mc aima-mc em em-mc; do
                systemctl status "$unit" --no-pager 2>/dev/null | head -10 || true
            done
        ' 2>&1 || true
        echo ""
        echo "── EM start_app / stop_app launcher hint ──"
        "${SSH_CMD[@]}" '
            for d in /agibot/software/em /opt/aima/em /etc/aima/em; do
                if [ -d "$d" ]; then
                    echo "── $d ──"
                    ls -la "$d/bin/cfg" 2>/dev/null | head -20
                    ls -la "$d/bin"     2>/dev/null | head -20
                fi
            done
        ' 2>&1 || true
        echo ""
        echo "── how is mc launched? (find launcher script referencing mc.yaml) ──"
        "${SSH_CMD[@]}" '
            mc_yaml=$(find / -maxdepth 6 -name mc.yaml 2>/dev/null | head -1)
            [ -z "$mc_yaml" ] && exit 0
            echo "search target: $mc_yaml"
            for d in /agibot/software /etc /opt/aima; do
                [ -d "$d" ] || continue
                grep -rln --binary-files=without-match -- "$(basename "$mc_yaml")" "$d" 2>/dev/null | head -10
            done
        ' 2>&1 || true
        echo ""
        echo "── tail of MC log (best-effort) ──"
        "${SSH_CMD[@]}" '
            for f in ~/aima_logs/mc.log /var/log/aima/mc.log /tmp/mc.log /agibot/log/mc.log; do
                if [ -f "$f" ]; then
                    echo "── $f ──"
                    tail -n 60 "$f" 2>/dev/null
                    break
                fi
            done
            (journalctl -u mc -n 60 --no-pager 2>/dev/null || true)
        ' 2>&1 || true
    } >"$host_out/04_pc_skim.txt" 2>&1
    export X2_INPUTSRC_TEST_D_DONE=1
}

host_run_test_d
maybe_relaunch_in_docker "${ORIG_ARGS[@]}"

# ────────────────────────────────────────────────────────────────────────
# Validations
# ────────────────────────────────────────────────────────────────────────
if [[ -z "$OUT_DIR" ]]; then
    echo -e "${RED}ERROR: --out-dir PATH is required.${NC}" >&2
    print_help >&2
    exit 2
fi

if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$USER_CWD/$OUT_DIR"
fi
assert_host_persistent_path "$OUT_DIR"

if (( GANTRY_CONFIRMED == 0 )); then
    {
        echo -e "${RED}ERROR: --gantry-confirmed is required.${NC}"
        echo ""
        echo "This probe makes state-changing SetMcInputSource calls (no joint"
        echo "publishes, but MC's arbitration table is poked). The 200 ms watchdog"
        echo "is the safety net, but the robot must be on the gantry or in a"
        echo "non-balancing MC mode (PASSIVE_DEFAULT / DAMPING_DEFAULT)."
        echo ""
        echo "Re-run with --gantry-confirmed once that's true."
    } >&2
    exit 2
fi

if ! command -v ros2 &>/dev/null; then
    echo -e "${RED}ERROR: 'ros2' not in PATH. Source ROS 2 Humble + aimdk_msgs first.${NC}" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo -e "${BOLD}=== X2 MC input-source probe ===${NC}"
echo -e "out-dir:           $OUT_DIR"
echo -e "pnc params:        priority=$PNC_PRIORITY timeout=${PNC_TIMEOUT}ms"
echo -e "sonic_policy ADD:  name=$SONIC_NAME priority=$SONIC_PRIORITY timeout=${SONIC_TIMEOUT}ms"
echo -e "ssh skim:          ${SSH_PC:-<disabled>}"
echo ""

# ────────────────────────────────────────────────────────────────────────
# 00 -- meta
# ────────────────────────────────────────────────────────────────────────
{
    echo "# x2_mc_input_source_probe dump"
    echo "t_wall_iso:        $(date -Iseconds)"
    echo "t_wall_unix:       $(date +%s)"
    echo "host_uname:        $(uname -a)"
    echo "ros_distro:        ${ROS_DISTRO:-<unset>}"
    echo "ros_domain_id:     ${ROS_DOMAIN_ID:-<unset>}"
    echo "ros_localhost:     ${ROS_LOCALHOST_ONLY:-<unset>}"
    echo "rmw:               ${RMW_IMPLEMENTATION:-<default>}"
    echo "pnc_priority:      $PNC_PRIORITY"
    echo "pnc_timeout_ms:    $PNC_TIMEOUT"
    echo "sonic_name:        $SONIC_NAME"
    echo "sonic_priority:    $SONIC_PRIORITY"
    echo "sonic_timeout_ms:  $SONIC_TIMEOUT"
    echo "ssh_pc:            ${SSH_PC:-<disabled>}"
    echo "git_sha:           $(git -C "$DEPLOY_DIR/.." rev-parse --short HEAD 2>/dev/null || echo '<not-a-repo>')"
} >"$OUT_DIR/00_meta.txt"

# ────────────────────────────────────────────────────────────────────────
# Service helpers
# ────────────────────────────────────────────────────────────────────────
GET_MC="/aimdk_5Fmsgs/srv/GetMcAction"
GET_IS="/aimdk_5Fmsgs/srv/GetCurrentInputSource"
SET_IS="/aimdk_5Fmsgs/srv/SetMcInputSource"

# ros2 service call returns Python-repr-style output, e.g.
#   response:
#   aimdk_msgs.srv.GetMcAction_Response(header=...code=0...action_desc='PASSIVE_DEFAULT'...)
# We detect success by looking for the literal 'response:' line followed
# by the typed-name token. We disable pipefail inside the helper so
# `timeout` killing ros2 *after* a successful reply doesn't poison the
# return code.
rpc_call() {
    # rpc_call OUT_FILE SVC_PATH SVC_TYPE PAYLOAD
    local out_file="$1" svc_path="$2" svc_type="$3" payload="$4"
    : >"$out_file"
    local i
    for i in $(seq 1 "$RPC_RETRIES"); do
        set +o pipefail
        timeout "$RPC_TIMEOUT" \
            ros2 service call "$svc_path" "$svc_type" "$payload" 2>&1 \
            >>"$out_file"
        set -o pipefail
        # Reply is considered valid if the file contains a 'response:'
        # line + the typed-name token from the service type. Discovery
        # failures emit "rcl node's context is invalid" or "service
        # not available" -- don't match.
        local typename
        typename=$(echo "$svc_type" | sed 's|.*/||')
        if grep -qE "${typename}_Response\(" "$out_file"; then
            return 0
        fi
        echo "# (attempt $i: no valid reply yet; retrying)" >>"$out_file"
        sleep 0.2
    done
    return 1
}

rpc_get_mc_action() {
    rpc_call "$1" "$GET_MC" "aimdk_msgs/srv/GetMcAction" "{request: {}}"
}

rpc_get_input_source() {
    rpc_call "$1" "$GET_IS" "aimdk_msgs/srv/GetCurrentInputSource" "{request: {}}"
}

# rpc_set_input_source ACTION_VALUE NAME PRIORITY TIMEOUT_MS OUT_FILE
#   ACTION_VALUE: 1001 ADD | 1002 MODIFY | 1003 DELETE | 2001 ENABLE | 2002 DISABLE
rpc_set_input_source() {
    local action_val="$1" name="$2" priority="$3" timeout_ms="$4" out_file="$5"
    local payload
    payload=$(printf "{request: {header: {stamp: {sec: 0, nanosec: 0}}}, action: {value: %s}, input_source: {name: '%s', priority: %s, timeout: %s}}" \
        "$action_val" "$name" "$priority" "$timeout_ms")
    rpc_call "$out_file" "$SET_IS" "aimdk_msgs/srv/SetMcInputSource" "$payload"
}

# ros2 service call emits Python-repr replies. Examples we parse:
#   ...CommonTaskResponse(header=ResponseHeader(stamp=...,code=0)...
#       state=aimdk_msgs.msg.CommonState(value=0)),
#       input_source=aimdk_msgs.msg.McInputSource(name='pnc', priority=40, timeout=200))
#
# Notes from the first dry run:
#   * MC populates response.header.code reliably (0 = success, nonzero =
#     server-side error). state.value is *always* 0 on this firmware --
#     the SUCCESS=1 path of CommonState appears unimplemented. So we
#     promote `code` to the primary success signal.
#   * GetCurrentInputSource returns name='' until a source has actually
#     published a heartbeat; pure ENABLE without publish never appears
#     in the current_input_source.

# Pull the LAST occurrence of a regex capture from a file (last attempt
# wins on retries).
last_capture() {
    local pattern="$1" file="$2"
    grep -oE "$pattern" "$file" 2>/dev/null | tail -n 1
}

# response.header.code (integer, '0' on success).
parse_code() {
    last_capture 'code=-?[0-9]+' "$1" | sed 's/.*=//'
}

# state.value (integer; effectively always 0 on this firmware).
parse_state_value() {
    last_capture 'CommonState\(value=-?[0-9]+\)' "$1" | sed 's/.*=//; s/)//'
}

state_name() {
    case "$1" in
        0)   echo "UNKNOWN" ;;
        1)   echo "SUCCESS" ;;
        2)   echo "FAILURE" ;;
        3)   echo "ABORTED" ;;
        4)   echo "TIMEOUT" ;;
        5)   echo "INVALID" ;;
        6)   echo "IN_MANUAL" ;;
        100) echo "NOT_READY" ;;
        200) echo "PENDING" ;;
        300) echo "CREATED" ;;
        400) echo "RUNNING" ;;
        *)   echo "<unmapped:$1>" ;;
    esac
}

# Promote 'code=0' to a coarse OK/FAIL string; falls back to UNKNOWN.
code_verdict() {
    local code="$1"
    if [[ -z "$code" ]]; then
        echo "UNKNOWN"
    elif [[ "$code" == "0" ]]; then
        echo "OK"
    else
        echo "ERR(code=$code)"
    fi
}

# input_source.name from GetCurrentInputSource reply (Python repr).
parse_input_source_name() {
    last_capture "McInputSource\(name='[^']*'" "$1" | sed "s/.*name='//; s/'$//"
}

# action_desc from GetMcAction reply.
parse_action_desc() {
    last_capture "action_desc='[^']*'" "$1" | sed "s/.*action_desc='//; s/'$//"
}

# ────────────────────────────────────────────────────────────────────────
# Cleanup trap: best-effort DISABLE pnc + DELETE+DISABLE sonic_policy
# regardless of how we exit. Failures here are logged and ignored.
# ────────────────────────────────────────────────────────────────────────
CLEANUP_NEEDED_PNC=0
CLEANUP_NEEDED_SONIC=0

cleanup() {
    local rc=$?
    echo ""
    echo -e "${YELLOW}[cleanup]${NC} restoring arbitration state (rc=$rc)"
    if (( CLEANUP_NEEDED_PNC )); then
        rpc_set_input_source 2002 "pnc" "$PNC_PRIORITY" "$PNC_TIMEOUT" \
            "$OUT_DIR/_cleanup_pnc_disable.txt" || true
    fi
    if (( CLEANUP_NEEDED_SONIC )); then
        rpc_set_input_source 2002 "$SONIC_NAME" "$SONIC_PRIORITY" "$SONIC_TIMEOUT" \
            "$OUT_DIR/_cleanup_sonic_disable.txt" || true
        rpc_set_input_source 1003 "$SONIC_NAME" "$SONIC_PRIORITY" "$SONIC_TIMEOUT" \
            "$OUT_DIR/_cleanup_sonic_delete.txt" || true
    fi
}
trap cleanup EXIT INT TERM

# ════════════════════════════════════════════════════════════════════════
# Test A -- baseline (read-only)
# ════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}── Test A: baseline ──${NC}"

if rpc_get_mc_action "$OUT_DIR/01_baseline_get_mc_action.txt"; then
    BASELINE_MODE=$(parse_action_desc "$OUT_DIR/01_baseline_get_mc_action.txt")
else
    BASELINE_MODE="<rpc-failed>"
fi
echo -e "  current MC mode:   ${BASELINE_MODE:-<empty>}"

if rpc_get_input_source "$OUT_DIR/01_baseline_get_input.txt"; then
    BASELINE_OWNER=$(parse_input_source_name "$OUT_DIR/01_baseline_get_input.txt")
else
    BASELINE_OWNER="<rpc-failed>"
fi
echo -e "  current bus owner: ${BASELINE_OWNER:-<empty>}"
echo ""

# Extra warning if we're in a balancing mode and operator ticked the
# gantry box -- we trust them, but worth printing.
case "$BASELINE_MODE" in
    STAND_DEFAULT|LOCOMOTION_DEFAULT)
        echo -e "${YELLOW}  note:${NC} MC is currently '$BASELINE_MODE' (weight-bearing)."
        echo -e "${YELLOW}        The 200 ms gap during pnc-claim with no publish is${NC}"
        echo -e "${YELLOW}        the only risk; motor controllers should hold last command.${NC}"
        echo -e "${YELLOW}        Trusting --gantry-confirmed and proceeding.${NC}"
        ;;
esac

# ════════════════════════════════════════════════════════════════════════
# Test B -- claim/release the existing 'pnc' slot, never publish
# ════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}── Test B: pnc MODIFY -> ENABLE -> watchdog poll -> DISABLE ──${NC}"

# B.1 MODIFY pnc (param-only update; no claim yet)
echo -e "  B.1 MODIFY pnc priority=$PNC_PRIORITY timeout=${PNC_TIMEOUT}ms"
rpc_set_input_source 1002 "pnc" "$PNC_PRIORITY" "$PNC_TIMEOUT" \
    "$OUT_DIR/02_pnc_modify_resp.txt" || true
B1_CODE=$(parse_code "$OUT_DIR/02_pnc_modify_resp.txt")
B1_STATE=$(parse_state_value "$OUT_DIR/02_pnc_modify_resp.txt")
echo -e "      reply: code=${B1_CODE:-?} verdict=$(code_verdict "${B1_CODE:-}") state.value=${B1_STATE:-?}"

rpc_get_input_source "$OUT_DIR/02_pnc_get_after_modify.txt" || true
B1_OWNER=$(parse_input_source_name "$OUT_DIR/02_pnc_get_after_modify.txt")
echo -e "      bus owner after MODIFY: '${B1_OWNER:-}'"

# B.2 ENABLE pnc (claim the bus). We DO NOT publish anything on
# /aima/hal/joint/*/command -- so the 200 ms expiration_time should
# arbitrate us back out shortly (if registration *itself* is the
# heartbeat) or never appear in current_input_source at all (if
# publishing is the activator). We measure this.
echo -e "  B.2 ENABLE pnc -- claim bus (no joint publishes will occur)"
CLEANUP_NEEDED_PNC=1
rpc_set_input_source 2001 "pnc" "$PNC_PRIORITY" "$PNC_TIMEOUT" \
    "$OUT_DIR/02_pnc_enable_resp.txt" || true
B2_CODE=$(parse_code "$OUT_DIR/02_pnc_enable_resp.txt")
B2_STATE=$(parse_state_value "$OUT_DIR/02_pnc_enable_resp.txt")
echo -e "      reply: code=${B2_CODE:-?} verdict=$(code_verdict "${B2_CODE:-}") state.value=${B2_STATE:-?}"

# Poll GetCurrentInputSource at three offsets after ENABLE:
#   T+50 ms   -- before watchdog could possibly have expired
#   T+500 ms  -- well past expiration_time=200ms
#   T+1500 ms -- comfortably stale
sleep 0.05
rpc_get_input_source "$OUT_DIR/02_pnc_get_after_enable_T+050ms.txt" || true
B2_OWNER_50=$(parse_input_source_name "$OUT_DIR/02_pnc_get_after_enable_T+050ms.txt")
echo -e "      T+50ms   owner: '${B2_OWNER_50:-}'"

sleep 0.45
rpc_get_input_source "$OUT_DIR/02_pnc_get_after_enable_T+500ms.txt" || true
B2_OWNER_500=$(parse_input_source_name "$OUT_DIR/02_pnc_get_after_enable_T+500ms.txt")
echo -e "      T+500ms  owner: '${B2_OWNER_500:-}'"

sleep 1.0
rpc_get_input_source "$OUT_DIR/02_pnc_get_after_enable_T+1500ms.txt" || true
B2_OWNER_1500=$(parse_input_source_name "$OUT_DIR/02_pnc_get_after_enable_T+1500ms.txt")
echo -e "      T+1500ms owner: '${B2_OWNER_1500:-}'"

# B.3 DISABLE pnc -- explicit cleanup even if the watchdog already did.
echo -e "  B.3 DISABLE pnc -- explicit release"
rpc_set_input_source 2002 "pnc" "$PNC_PRIORITY" "$PNC_TIMEOUT" \
    "$OUT_DIR/02_pnc_disable_resp.txt" || true
B3_CODE=$(parse_code "$OUT_DIR/02_pnc_disable_resp.txt")
B3_STATE=$(parse_state_value "$OUT_DIR/02_pnc_disable_resp.txt")
echo -e "      reply: code=${B3_CODE:-?} verdict=$(code_verdict "${B3_CODE:-}") state.value=${B3_STATE:-?}"
CLEANUP_NEEDED_PNC=0  # explicitly released

rpc_get_input_source "$OUT_DIR/02_pnc_get_after_disable.txt" || true
B3_OWNER=$(parse_input_source_name "$OUT_DIR/02_pnc_get_after_disable.txt")
echo -e "      bus owner after DISABLE: '${B3_OWNER:-}'"
echo ""

# ════════════════════════════════════════════════════════════════════════
# Test C -- ADD a fresh source name 'sonic_policy'
# ════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}── Test C: ADD '$SONIC_NAME' (registry mutability) ──${NC}"

echo -e "  C.1 ADD $SONIC_NAME priority=$SONIC_PRIORITY timeout=${SONIC_TIMEOUT}ms"
rpc_set_input_source 1001 "$SONIC_NAME" "$SONIC_PRIORITY" "$SONIC_TIMEOUT" \
    "$OUT_DIR/03_sonic_add_resp.txt" || true
C1_CODE=$(parse_code "$OUT_DIR/03_sonic_add_resp.txt")
C1_STATE=$(parse_state_value "$OUT_DIR/03_sonic_add_resp.txt")
echo -e "      reply: code=${C1_CODE:-?} verdict=$(code_verdict "${C1_CODE:-}") state.value=${C1_STATE:-?}"

# Treat ADD as accepted if the response.header.code is 0. The
# state.value field appears unimplemented on this firmware (always 0).
SONIC_REGISTRY_MUTABLE="unknown"
if [[ "$C1_CODE" == "0" ]]; then
    SONIC_REGISTRY_MUTABLE="yes"
    CLEANUP_NEEDED_SONIC=1

    echo -e "  C.2 ADD accepted (code=0) -- exercising ENABLE / verify / DISABLE / DELETE"
    rpc_set_input_source 2001 "$SONIC_NAME" "$SONIC_PRIORITY" "$SONIC_TIMEOUT" \
        "$OUT_DIR/03_sonic_enable_resp.txt" || true
    C2_CODE=$(parse_code "$OUT_DIR/03_sonic_enable_resp.txt")
    echo -e "      ENABLE  code=${C2_CODE:-?} verdict=$(code_verdict "${C2_CODE:-}")"

    sleep 0.05
    rpc_get_input_source "$OUT_DIR/03_sonic_get_after_enable.txt" || true
    C2_OWNER=$(parse_input_source_name "$OUT_DIR/03_sonic_get_after_enable.txt")
    echo -e "      bus owner: '${C2_OWNER:-}'"

    rpc_set_input_source 2002 "$SONIC_NAME" "$SONIC_PRIORITY" "$SONIC_TIMEOUT" \
        "$OUT_DIR/03_sonic_disable_resp.txt" || true
    C3_CODE=$(parse_code "$OUT_DIR/03_sonic_disable_resp.txt")
    echo -e "      DISABLE code=${C3_CODE:-?} verdict=$(code_verdict "${C3_CODE:-}")"

    rpc_set_input_source 1003 "$SONIC_NAME" "$SONIC_PRIORITY" "$SONIC_TIMEOUT" \
        "$OUT_DIR/03_sonic_delete_resp.txt" || true
    C4_CODE=$(parse_code "$OUT_DIR/03_sonic_delete_resp.txt")
    echo -e "      DELETE  code=${C4_CODE:-?} verdict=$(code_verdict "${C4_CODE:-}")"
    CLEANUP_NEEDED_SONIC=0
else
    SONIC_REGISTRY_MUTABLE="no"
    echo -e "  C.x ADD rejected (code!=0) -- registry appears locked to mc.yaml entries."
    echo -e "      (Will fall back to claiming existing 'pnc' slot.)"
fi
echo ""

# Test D was run on the host (before docker re-exec); see host_run_test_d().
if [[ -n "$SSH_PC" ]]; then
    if [[ -s "$OUT_DIR/04_pc_skim.txt" ]]; then
        echo -e "${BOLD}── Test D: SSH skim of $SSH_PC (ran on host) ──${NC}"
        echo "  see: $OUT_DIR/04_pc_skim.txt"
        echo ""
    else
        echo -e "${YELLOW}── Test D: requested but no 04_pc_skim.txt was produced.${NC}"
        echo -e "${YELLOW}   ssh probably isn't available on the host. Re-run on a host${NC}"
        echo -e "${YELLOW}   with ssh + sshpass installed.${NC}"
        echo ""
    fi
fi

# ════════════════════════════════════════════════════════════════════════
# 99 -- auto-generated FINDINGS.md
# ════════════════════════════════════════════════════════════════════════
{
    echo "# x2_mc_input_source_probe -- auto-generated findings"
    echo ""
    echo "**Date**: $(date -Iseconds)"
    echo "**Robot baseline at probe time**: ${BASELINE_MODE:-<unknown>} (input_source.name='${BASELINE_OWNER:-}')"
    echo ""
    echo "**Note on the success signal**: this MC firmware leaves"
    echo "\`response.state.value\` at 0 (UNKNOWN) on every reply, but populates"
    echo "\`response.header.code\` (0 = OK). The probe therefore promotes \`code\`"
    echo "to the primary verdict; \`state.value\` is shown for completeness."
    echo ""
    echo "## Q1: Does SetMcInputSource on 'pnc' work?"
    echo ""
    echo "| Step | code | verdict | state.value | bus owner observed |"
    echo "|------|------|---------|-------------|--------------------|"
    echo "| MODIFY pnc           | ${B1_CODE:-?} | $(code_verdict "${B1_CODE:-}") | ${B1_STATE:-?} | '${B1_OWNER:-}' |"
    echo "| ENABLE pnc (T+0)     | ${B2_CODE:-?} | $(code_verdict "${B2_CODE:-}") | ${B2_STATE:-?} | T+50ms='${B2_OWNER_50:-}' |"
    echo "| ENABLE pnc           |     |       |     | T+500ms='${B2_OWNER_500:-}' |"
    echo "| ENABLE pnc           |     |       |     | T+1500ms='${B2_OWNER_1500:-}' |"
    echo "| DISABLE pnc          | ${B3_CODE:-?} | $(code_verdict "${B3_CODE:-}") | ${B3_STATE:-?} | '${B3_OWNER:-}' |"
    echo ""
    if [[ "$B1_CODE" == "0" && "$B2_CODE" == "0" && "$B3_CODE" == "0" ]]; then
        echo "**Verdict**: MC accepts MODIFY / ENABLE / DISABLE on the existing"
        echo "\`pnc\` slot (all replies code=0). The arbitration API is alive."
    else
        echo "**Verdict**: at least one MODIFY/ENABLE/DISABLE returned a non-zero code."
        echo "Inspect 02_pnc_*.txt; rerun with the robot in a non-balancing mode"
        echo "(PASSIVE_DEFAULT) before drawing conclusions."
    fi
    echo ""
    echo "**Bus-owner observation**:"
    if [[ -z "${B2_OWNER_50}" && -z "${B2_OWNER_500}" && -z "${B2_OWNER_1500}" ]]; then
        echo "  GetCurrentInputSource returned name='' at every poll, even right"
        echo "  after ENABLE. That means **publishing IS the activator**: a source"
        echo "  doesn't appear in current_input_source until at least one fresh"
        echo "  command has flowed on /aima/hal/joint/*/command within the last"
        echo "  expiration_time. That's the cleanest possible heartbeat model for"
        echo "  our smooth-handoff -- our 50 Hz publish rate (20 ms period) is"
        echo "  comfortably tighter than the 200 ms watchdog, so MC will own the"
        echo "  bus by default and let us win whenever we publish."
    elif [[ "${B2_OWNER_50}" == "pnc" && -z "${B2_OWNER_500}" ]]; then
        echo "  Owner was 'pnc' at T+50ms but empty by T+500ms -- the watchdog"
        echo "  reclaimed the bus when no command flowed. **Publishing IS the"
        echo "  heartbeat.** Smooth-handoff design unchanged."
    elif [[ "${B2_OWNER_1500}" == "pnc" ]]; then
        echo "  Owner stayed 'pnc' through T+1500ms despite no joint publishes."
        echo "  **Registration alone is the heartbeat** -- we'd need a separate"
        echo "  keep-alive RPC (or rely on \`timeout\` field per-call) to release"
        echo "  the bus on a crash. Worth a follow-up test."
    else
        echo "  Mixed/ambiguous owner sequence: T+50ms='${B2_OWNER_50}'"
        echo "  T+500ms='${B2_OWNER_500}' T+1500ms='${B2_OWNER_1500}'."
        echo "  Re-read 02_pnc_get_after_enable_*.txt for the raw replies."
    fi
    echo ""
    echo "## Q2: Can we ADD a fresh source name?"
    echo ""
    echo "Attempted: ADD name='$SONIC_NAME' priority=$SONIC_PRIORITY timeout=${SONIC_TIMEOUT}ms"
    echo "Reply: code=${C1_CODE:-?} verdict=$(code_verdict "${C1_CODE:-}") state.value=${C1_STATE:-?}"
    echo "Registry mutable: **$SONIC_REGISTRY_MUTABLE**"
    echo ""
    if [[ "$SONIC_REGISTRY_MUTABLE" == "yes" ]]; then
        echo "**Verdict**: registry accepts new entries. We can ship our own"
        echo "named source with custom priority. Recommended name: '$SONIC_NAME',"
        echo "priority 45 (just above pnc=40, below interaction=50)."
    else
        echo "**Verdict**: registry is locked to mc.yaml entries. We must claim"
        echo "the existing 'pnc' slot (priority 40) -- which is fine, since it's"
        echo "the canonical autonomy slot."
    fi
    echo ""
    echo "## Q3: How is mc.yaml managed?"
    echo ""
    if [[ -n "$SSH_PC" ]]; then
        echo "See \`04_pc_skim.txt\` for full skim of $SSH_PC."
        echo ""
        echo "Key things to look for in 04_pc_skim.txt:"
        echo "  - 'WRITABLE by current user' vs 'NOT writable' line"
        echo "  - The 'MC processes' block: who launches mc_ros2_node?"
        echo "  - The 'systemd units' block: is there a unit we can restart?"
        echo "  - The 'how is mc launched' block: which script references mc.yaml?"
    else
        echo "Test D was skipped (no --ssh-pc). To answer Q3, re-run with:"
        echo "  --ssh-pc run@10.0.1.41 --ssh-password 1"
        echo "(or use ssh keys + drop --ssh-password)."
    fi
    echo ""
    echo "## Files in this dump"
    (cd "$OUT_DIR" && find . -type f -printf "%P\n" | sort)
} >"$OUT_DIR/99_FINDINGS.md"

echo -e "${GREEN}done.${NC}"
echo "  ${BOLD}Dump:${NC}     $OUT_DIR"
echo "  ${BOLD}Findings:${NC} $OUT_DIR/99_FINDINGS.md"
