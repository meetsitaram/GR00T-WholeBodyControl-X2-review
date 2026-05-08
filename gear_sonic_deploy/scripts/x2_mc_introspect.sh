#!/usr/bin/env bash
# x2_mc_introspect.sh -- read-only enumeration of the MC surface on PC1.
#
# Goal: find out if there's a gentler MC handoff than stop_app/start_app
# without changing any state. Dumps everything we can observe about MC's
# ROS surface, HTTP admin surface, and (optionally, opt-in) PC2's
# filesystem view of the SDK overlay, into one timestamped directory.
#
# Usage (from any host shell -- auto-relaunches inside docker_x2/x2sim):
#
#   ./gear_sonic_deploy/scripts/x2_mc_introspect.sh \
#       --out-dir scratch/probes/mc_introspect_$(date +%Y%m%d_%H%M%S)
#
# Defaults:
#   - PC1 EM HTTP server: http://10.0.1.40:50080  (motion-control unit)
#   - PC2 SDK dev unit:   agi@10.0.1.41           (only used with --ssh-pc2)
#
# Read-only. NO start_app / stop_app / SetMcAction calls -- only
# ros2 service|topic|node *list/info, GetMcAction, and HTTP GETs.

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
OUT_DIR=""
PC1_HTTP_HOST="${X2_PC1_HTTP_HOST:-10.0.1.40}"
PC1_HTTP_PORT="${X2_PC1_HTTP_PORT:-50080}"
SSH_PC2=""
NO_DOCKER=0
PROBE_STATE_VERBS=0
ASSUME_YES=0

# Preserve the operator's original argv before we shift through it for parsing,
# so the docker re-exec can forward the same args to the inner invocation.
ORIG_ARGS=("$@")

print_help() {
    sed -n '/^# x2_mc_introspect.sh/,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '$d' -e 's/^# \{0,1\}//'
    cat <<EOF

Flags:
  --out-dir PATH     Output directory. REQUIRED. Must resolve to a
                     host-persistent location.
  --pc1-host HOST    PC1 (motion-control unit) IP for HTTP enumeration.
                     Default: ${PC1_HTTP_HOST} (env X2_PC1_HTTP_HOST).
  --pc1-port PORT    PC1 EM HTTP port. Default: ${PC1_HTTP_PORT}.
  --ssh-pc2 USER\@HOST
                     If set, also ssh into PC2 and skim the SDK overlay
                     and aimdk_msgs install for service-type definitions
                     and config files. Off by default. Common value:
                     agi@10.0.1.41.
  --no-docker        Skip docker auto-relaunch.
  --probe-state-verbs
                     Also send HEAD requests to speculative state-changing
                     HTTP verbs (start_app, stop_app, pause_app, etc.) to
                     learn whether they exist. HEAD is supposed to be
                     read-only but a buggy server could in principle treat
                     it as POST. Off by default. Will prompt y/N before
                     issuing the requests unless --yes is also passed.
  --yes              Skip the interactive y/N confirmation for
                     --probe-state-verbs. Only use this in non-interactive
                     contexts (CI, scripts).
  -h, --help         Show this help.

NOTE: this script does NOT ssh into PC1 (10.0.1.40). Per the X2 SDK
docs, "never run our deploy on PC1". Read-only HTTP introspection of
PC1's EM port is the sanctioned way to query MC. PC2 (10.0.1.41) is
the dev unit and is fair game with --ssh-pc2.

Default behaviour is read-only on every axis: only ros2 list/info /
GetMcAction (read), HTTP GETs on introspection-only verbs, and (with
--ssh-pc2) ls/find/head/ps/journalctl on the PC2 SDK overlay.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --out-dir)         OUT_DIR="$2"; shift 2 ;;
        --out-dir=*)       OUT_DIR="${1#--out-dir=}"; shift ;;
        --pc1-host)        PC1_HTTP_HOST="$2"; shift 2 ;;
        --pc1-host=*)      PC1_HTTP_HOST="${1#--pc1-host=}"; shift ;;
        --pc1-port)        PC1_HTTP_PORT="$2"; shift 2 ;;
        --pc1-port=*)      PC1_HTTP_PORT="${1#--pc1-port=}"; shift ;;
        --ssh-pc2)         SSH_PC2="$2"; shift 2 ;;
        --ssh-pc2=*)       SSH_PC2="${1#--ssh-pc2=}"; shift ;;
        --no-docker)       NO_DOCKER=1; shift ;;
        --probe-state-verbs) PROBE_STATE_VERBS=1; shift ;;
        --yes|-y)          ASSUME_YES=1; shift ;;
        -h|--help)         print_help; exit 0 ;;
        *)
            echo "ERROR: unknown flag '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

# ────────────────────────────────────────────────────────────────────────
# Persistent-path guard (mirrors x2_record.sh).
# ────────────────────────────────────────────────────────────────────────
is_host_persistent_path() {
    local p="$1"
    [[ "${X2_INTROSPECT_IN_DOCKER:-0}" != "1" ]] && return 0
    [[ -z "$p" ]] && return 1
    # Inside docker $HOME is /root by default. We mount the operator's host
    # $HOME at the same absolute path via the auto-docker code path; that
    # original path is forwarded as X2_HOST_HOME so this check can match
    # against the actual mount, not the container's user-home.
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
    if [[ -d /workspace/sonic ]] || [[ -n "${X2_INTROSPECT_IN_DOCKER:-}" ]]; then
        return 0
    fi
    if (( NO_DOCKER == 1 )); then
        return 0
    fi
    if ! command -v docker &>/dev/null; then
        echo -e "${YELLOW}Note: 'docker' not in PATH; introspection wants ros2 + curl.${NC}" >&2
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

    echo -e "${BLUE}[auto-docker]${NC} re-exec inside docker_x2/x2sim (introspect)"
    local script_in_container="/workspace/sonic/gear_sonic_deploy/scripts/$(basename "${BASH_SOURCE[0]}")"
    cd "$compose_dir"
    export X2_INTROSPECT_IN_DOCKER=1
    exec docker compose run --rm \
        "${tty_args[@]}" \
        -e "ROS_LOCALHOST_ONLY=0" \
        -e "ROS_DOMAIN_ID=$domain_id" \
        -e "X2_INTROSPECT_IN_DOCKER=1" \
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
if [[ -z "$OUT_DIR" ]]; then
    echo -e "${RED}ERROR: --out-dir PATH is required.${NC}" >&2
    print_help >&2
    exit 2
fi

if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$USER_CWD/$OUT_DIR"
fi
assert_host_persistent_path "$OUT_DIR"

mkdir -p "$OUT_DIR"
echo -e "${BOLD}=== X2 MC introspection ===${NC}"
echo -e "out_dir: $OUT_DIR"
echo -e "PC1 HTTP: http://${PC1_HTTP_HOST}:${PC1_HTTP_PORT}"
echo -e "PC2 ssh:  ${SSH_PC2:-<disabled>}"
echo ""

# Convenience: every command's stdout+stderr is teed into a per-step
# file under $OUT_DIR. Use `step LABEL FILE -- command...` to wrap.
step() {
    local label="$1" out="$2"
    shift 2
    if [[ "$1" != "--" ]]; then
        echo "step: missing -- after FILE" >&2
        return 1
    fi
    shift
    echo -e "${BLUE}[step]${NC} $label -> $(basename "$out")"
    {
        echo "# label:   $label"
        echo "# command: $*"
        echo "# t_wall:  $(date -Iseconds)"
        echo "# ----------------------------------------------------------"
        if "$@"; then
            local rc=0
        else
            local rc=$?
            echo "# (command exited rc=$rc)"
        fi
    } >"$out" 2>&1 || true
}

# ────────────────────────────────────────────────────────────────────────
# 00 -- meta
# ────────────────────────────────────────────────────────────────────────
{
    echo "# x2_mc_introspect dump"
    echo "t_wall_iso:        $(date -Iseconds)"
    echo "t_wall_unix:       $(date +%s)"
    echo "host_uname:        $(uname -a)"
    echo "ros_distro:        ${ROS_DISTRO:-<unset>}"
    echo "ros_domain_id:     ${ROS_DOMAIN_ID:-<unset>}"
    echo "ros_localhost:     ${ROS_LOCALHOST_ONLY:-<unset>}"
    echo "rmw:               ${RMW_IMPLEMENTATION:-<default>}"
    echo "pc1_http:          http://${PC1_HTTP_HOST}:${PC1_HTTP_PORT}"
    echo "ssh_pc2:           ${SSH_PC2:-<disabled>}"
    echo "git_sha:           $(git -C "$DEPLOY_DIR/.." rev-parse --short HEAD 2>/dev/null || echo '<not-a-repo>')"
} >"$OUT_DIR/00_meta.txt"

# ────────────────────────────────────────────────────────────────────────
# 01 -- ROS 2 node enumeration
# ────────────────────────────────────────────────────────────────────────
if command -v ros2 &>/dev/null; then
    step "ros2 node list" "$OUT_DIR/01_ros2_node_list.txt" -- ros2 node list
    # Capture node info for every MC-related node. We only run `ros2 node info`
    # against names matching the pattern; non-matching nodes are listed but not
    # introspected (keeps the dump small).
    {
        echo "# ros2 node info <name> for every node matching mc|aima|aimdk"
        echo "# ----------------------------------------------------------"
    } >"$OUT_DIR/01_ros2_node_info.txt"
    if [[ -s "$OUT_DIR/01_ros2_node_list.txt" ]]; then
        # Skip the metadata header lines (start with '#').
        while IFS= read -r node; do
            [[ -z "$node" ]] && continue
            [[ "$node" =~ ^# ]] && continue
            if echo "$node" | grep -qiE '(mc|aima|aimdk)'; then
                {
                    echo ""
                    echo "════════════════════════════════════════════════════════════"
                    echo "node: $node"
                    echo "════════════════════════════════════════════════════════════"
                    timeout 5 ros2 node info "$node" 2>&1 || echo "(node info failed for $node)"
                } >>"$OUT_DIR/01_ros2_node_info.txt"
            fi
        done < "$OUT_DIR/01_ros2_node_list.txt"
    fi

    step "ros2 service list -t" "$OUT_DIR/02_ros2_service_list.txt" \
        -- ros2 service list -t
    step "ros2 topic list -t" "$OUT_DIR/02_ros2_topic_list.txt" \
        -- ros2 topic list -t

    # Live publishers on the joint command bus, per group. Tells us who is
    # writing to /aima/hal/joint/<group>/command right now -- expected: at
    # least the HAL bridge, possibly MC if it's a co-publisher in the
    # current mode.
    {
        echo "# ros2 topic info --verbose for each /aima/hal/joint/<g>/command"
        echo "# ----------------------------------------------------------"
        for grp in leg waist arm head; do
            echo ""
            echo "── /aima/hal/joint/${grp}/command ──"
            timeout 5 ros2 topic info "/aima/hal/joint/${grp}/command" --verbose 2>&1 \
                || echo "(topic info failed for $grp)"
        done
    } >"$OUT_DIR/03_topic_info_command.txt"

    # Current MC mode (read-only Get). Documents what mode the dump was
    # captured in -- analytical context for everything else.
    {
        echo "# Current MC action_desc (read-only)"
        echo "# ----------------------------------------------------------"
        timeout 3 ros2 service call \
            /aimdk_5Fmsgs/srv/GetMcAction \
            aimdk_msgs/srv/GetMcAction \
            "{request: {}}" 2>&1 || echo "(GetMcAction failed)"
    } >"$OUT_DIR/03_get_mc_action.txt"
else
    echo "ros2 not in PATH" >"$OUT_DIR/01_ros2_skipped.txt"
fi

# ────────────────────────────────────────────────────────────────────────
# 04 -- HTTP admin enumeration on PC1's EM port
#
# Split into two lists:
#   * INTROSPECTION_VERBS -- read-only by definition; we GET them.
#   * STATE_VERBS         -- speculative app-control verbs that *might*
#                            exist. GET is *probably* safe (a sane HTTP
#                            server returns 405 Method Not Allowed for
#                            POST-only routes), but a buggy server could
#                            treat GET like POST and trigger the action.
#                            We HEAD them (not GET) only when the operator
#                            opts in via --probe-state-verbs and the y/N
#                            confirmation passes. HEAD returns headers
#                            without invoking handler logic on RFC-7231
#                            -compliant servers.
# ────────────────────────────────────────────────────────────────────────
HTTP_BASE="http://${PC1_HTTP_HOST}:${PC1_HTTP_PORT}"

INTROSPECTION_VERBS=(
    "/"
    "/json"
    "/json/"
    "/json/help"
    "/json/version"
    "/json/list_apps"
    "/json/status"
    "/json/get_app_status"
    "/json/list_endpoints"
    "/json/list_services"
    "/json/info"
    "/json/list"
    "/metrics"
    "/healthz"
    "/api"
    "/api/v1"
)

STATE_VERBS=(
    "/json/start_app"
    "/json/stop_app"
    "/json/pause_app"
    "/json/resume_app"
    "/json/suspend_app"
    "/json/idle_app"
    "/json/restart_app"
)

http_probe() {
    # $1 = method (GET|HEAD), $2 = url path, $3 = output file
    local method="$1" verb="$2" out="$3"
    {
        echo "# ${method} ${HTTP_BASE}${verb}"
        echo "# ----------------------------------------------------------"
        if command -v curl &>/dev/null; then
            local extra=()
            [[ "$method" == "HEAD" ]] && extra=(-I)
            curl -sS -i --connect-timeout 2 --max-time 5 -X "$method" \
                "${extra[@]}" "${HTTP_BASE}${verb}" 2>&1 || echo "(curl failed)"
        else
            python3 - "$method" "${HTTP_BASE}${verb}" <<'PY' 2>&1 || echo "(python http failed)"
import sys, urllib.request, urllib.error, socket
method, url = sys.argv[1], sys.argv[2]
try:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        print("HTTP", r.status, r.reason)
        for k, v in r.getheaders():
            print(f"{k}: {v}")
        if method != "HEAD":
            print()
            print(r.read().decode("utf-8", errors="replace")[:4000])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.reason)
    for k, v in e.headers.items():
        print(f"{k}: {v}")
    if method != "HEAD":
        print()
        try:
            print(e.read().decode("utf-8", errors="replace")[:4000])
        except Exception:
            pass
except (urllib.error.URLError, socket.timeout) as e:
    print(f"NETWORK ERROR: {e}")
PY
        fi
    } >"$out" 2>&1 || true
}

mkdir -p "$OUT_DIR/04_http"
for verb in "${INTROSPECTION_VERBS[@]}"; do
    fname=$(echo "$verb" | sed 's|^/||; s|/|__|g')
    [[ -z "$fname" ]] && fname="root"
    http_probe GET "$verb" "$OUT_DIR/04_http/get_${fname}.txt"
done

# State-changing verbs: opt-in + confirmation gate.
if (( PROBE_STATE_VERBS == 1 )); then
    {
        echo ""
        echo -e "${YELLOW}[probe-state-verbs]${NC} About to send HEAD requests to:"
        for verb in "${STATE_VERBS[@]}"; do
            echo -e "${YELLOW}    ${HTTP_BASE}${verb}${NC}"
        done
        echo -e "${YELLOW}HEAD is supposed to be read-only on RFC-7231 servers,${NC}"
        echo -e "${YELLOW}but a buggy server could in principle treat HEAD as POST${NC}"
        echo -e "${YELLOW}and trigger the action. If MC is currently running and the${NC}"
        echo -e "${YELLOW}server is well-behaved, the worst case is a 405 response.${NC}"
        echo -e "${YELLOW}If it's buggy, /json/stop_app HEAD might stop MC.${NC}"
    } >&2
    if (( ASSUME_YES == 0 )); then
        echo -en "${BOLD}Proceed with HEAD probe of state-changing verbs? [y/N] ${NC}" >&2
        read -r reply
        if [[ ! "$reply" =~ ^[Yy] ]]; then
            echo -e "${YELLOW}[probe-state-verbs] declined; skipping.${NC}" >&2
            PROBE_STATE_VERBS=0
        fi
    else
        echo -e "${YELLOW}[probe-state-verbs] --yes given; proceeding without prompt.${NC}" >&2
    fi
fi

if (( PROBE_STATE_VERBS == 1 )); then
    for verb in "${STATE_VERBS[@]}"; do
        fname=$(echo "$verb" | sed 's|^/||; s|/|__|g')
        http_probe HEAD "$verb" "$OUT_DIR/04_http/head_${fname}.txt"
    done
else
    {
        echo "# State-changing HTTP verbs were NOT probed (default safe mode)."
        echo "# Re-run with --probe-state-verbs to opt in. Verbs that would be"
        echo "# probed:"
        for verb in "${STATE_VERBS[@]}"; do
            echo "#   ${HTTP_BASE}${verb}"
        done
    } >"$OUT_DIR/04_http/_state_verbs_skipped.txt"
fi

# ────────────────────────────────────────────────────────────────────────
# 05 -- aimdk_msgs service-type definitions (in-container, free of ssh)
# ────────────────────────────────────────────────────────────────────────
{
    echo "# All service types under aimdk_msgs"
    echo "# ----------------------------------------------------------"
    if command -v ros2 &>/dev/null; then
        timeout 5 ros2 interface package aimdk_msgs 2>&1 \
            || echo "(ros2 interface package aimdk_msgs failed)"
        echo ""
        echo "# Definitions of *McAction* / *Source* / *Mode* / *Action*"
        echo "# ----------------------------------------------------------"
        ros2 interface package aimdk_msgs 2>/dev/null | \
            grep -iE '(mcaction|source|mode|action|input|priority|arbiter|register|pause|resume|suspend|idle|hold)' \
            | while IFS= read -r ifc; do
                echo ""
                echo "── $ifc ──"
                timeout 3 ros2 interface show "$ifc" 2>&1 | head -100 \
                    || echo "(ros2 interface show $ifc failed)"
            done
    else
        echo "(ros2 not in PATH)"
    fi
} >"$OUT_DIR/05_aimdk_interfaces.txt"

# ────────────────────────────────────────────────────────────────────────
# 06 -- Optional: ssh into PC2 for SDK overlay skim
# ────────────────────────────────────────────────────────────────────────
if [[ -n "$SSH_PC2" ]]; then
    SSH_OPTS=(-o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
    {
        echo "# ssh ${SSH_PC2}: PC2 SDK overlay skim"
        echo "# ----------------------------------------------------------"
        echo ""
        echo "── uname / version ──"
        ssh "${SSH_OPTS[@]}" "$SSH_PC2" 'uname -a; cat /etc/aima-version 2>/dev/null; cat /etc/os-release | head -5' 2>&1 \
            || echo "(ssh failed)"
        echo ""
        echo "── /opt/aima* and /etc/aima* listing ──"
        ssh "${SSH_OPTS[@]}" "$SSH_PC2" '
            for d in /opt/aima /etc/aima /home/agi/aima_config /home/agi/agi_ws /opt/ros/humble/share/aimdk_msgs /var/log/aima; do
                if [ -d "$d" ]; then
                    echo "── $d ──"
                    ls -la "$d" 2>/dev/null | head -50
                    echo ""
                else
                    echo "── $d  (does not exist)"
                fi
            done
        ' 2>&1 || echo "(ssh listing failed)"
        echo ""
        echo "── small config files (head -n 100, files <16 KiB) ──"
        ssh "${SSH_OPTS[@]}" "$SSH_PC2" '
            for d in /opt/aima /etc/aima /home/agi/aima_config; do
                [ -d "$d" ] || continue
                find "$d" -maxdepth 4 -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.conf" -o -name "*.toml" -o -name "*.ini" \) -size -16k 2>/dev/null \
                    | while read -r f; do
                        echo ""
                        echo "─── $f ───"
                        head -n 100 "$f" 2>/dev/null
                    done
            done
        ' 2>&1 || true
        echo ""
        echo "── MC processes (filtered) ──"
        ssh "${SSH_OPTS[@]}" "$SSH_PC2" "ps -ef | grep -iE '(mc|aimdk|aima)' | grep -v grep" 2>&1 || true
        echo ""
        echo "── MC log tail (best-effort) ──"
        ssh "${SSH_OPTS[@]}" "$SSH_PC2" '
            for f in ~/aima_logs/mc.log /var/log/aima/mc.log /tmp/mc.log; do
                if [ -f "$f" ]; then
                    echo "── $f ──"
                    tail -n 200 "$f" 2>/dev/null
                    break
                fi
            done
            journalctl --user -u mc -n 100 --no-pager 2>/dev/null \
                || journalctl -u mc -n 100 --no-pager 2>/dev/null \
                || echo "(no journalctl mc unit accessible)"
        ' 2>&1 || true
    } >"$OUT_DIR/06_pc2_skim.txt"
fi

# ────────────────────────────────────────────────────────────────────────
# 99 -- Auto-triage: grep the dump for arbitration / source / pause
#       keywords and produce a one-page findings summary at the top.
# ────────────────────────────────────────────────────────────────────────
KEYWORDS=(
    "InputSource" "Source" "Sources" "Priority" "Arbiter" "Arbitration"
    "Pause" "Resume" "Suspend" "Hold" "Yield" "Cede" "Idle" "Standby"
    "Register" "register" "Subscribe" "Attach"
    "pause_app" "suspend_app" "idle_app" "hold_app" "restart_app"
    "JOINT_DEFAULT" "STAND_DEFAULT" "PASSIVE_DEFAULT" "DAMPING_DEFAULT"
    "LOCOMOTION_DEFAULT"
    "SafetyController" "WatchdogController" "input_source" "input_id"
)

{
    echo "# x2_mc_introspect findings summary"
    echo "# Generated: $(date -Iseconds)"
    echo ""
    echo "## Verdict"
    echo "(fill in by hand after review)"
    echo ""
    echo "## Keyword scan across the dump"
    echo ""
    printf "%-22s  %6s  %s\n" "keyword" "hits" "first 3 (file:line)"
    printf "%-22s  %6s  %s\n" "----------------------" "------" "-------------------------------"
    for kw in "${KEYWORDS[@]}"; do
        # rg / grep: search every file in the dump except this findings
        # report itself.
        local_hits=$(grep -rni --exclude="99_findings.md" -- "$kw" "$OUT_DIR" 2>/dev/null || true)
        n=$(echo "$local_hits" | grep -c . || true)
        if (( n == 0 )); then
            continue
        fi
        first3=$(echo "$local_hits" | head -n 3 | sed "s|$OUT_DIR/||g" | tr '\n' ';' | sed 's/;$//')
        printf "%-22s  %6d  %s\n" "$kw" "$n" "$first3"
    done
    echo ""
    echo "## Files in this dump"
    (cd "$OUT_DIR" && find . -type f -printf "%P\n" | sort)
} >"$OUT_DIR/99_findings.md"

echo ""
echo -e "${GREEN}done.${NC}"
echo "  ${BOLD}Dump:${NC}     $OUT_DIR"
echo "  ${BOLD}Findings:${NC} $OUT_DIR/99_findings.md"
echo ""
echo "  Recommended next step: read 99_findings.md first; if any non-zero"
echo "  hit count appears under InputSource / Source / Pause / Suspend /"
echo "  Yield / Arbiter, follow the (file:line) hint to investigate. If"
echo "  every keyword is zero, MC has no documented gentler-handoff API"
echo "  and we proceed with the smooth-handoff plan via JOINT_DEFAULT +"
echo "  stop_app/start_app."
