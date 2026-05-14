#!/usr/bin/env bash
# X2 Quest 3 planner-driven teleop / record stack runner (Phase 0).
#
# Spawns the full 4-process Phase 0 stack as CHILD processes of this
# wrapper, in the only order that's safe (deploy first, then planner,
# then manager, then recorder), with readiness markers between each
# step and a trap-cleaned reverse-order shutdown.
#
# Stack composition (port + topic contract):
#
#                      ┌──────────────────────┐
#                      │     Quest 3 (WebXR)  │
#                      └──────────┬───────────┘
#                          WS:8765/HTTPS:8443
#                                 ▼
#   ┌──────────────────────────────────────────────────┐
#   │  quest3_manager_x2  (IK + finger filter + intent)│
#   └──────────────┬───────────────────┬───────────────┘
#       planner_cmd│5563         5564 │ arm_targets +
#                  ▼                  │ hand_finger_cmd +
#   ┌──────────────────────────┐      │ stream_mode +
#   │  x2_heuristic_planner    │      │ recorder_cmd
#   │  (state machine)         │      │
#   └──────────────┬───────────┘      ▼
#                  │5565       ┌──────────────────────┐
#                  └──body_pose▶  record_x2_dataset   │
#                              │  (subscribe-mode     │
#                              │   merger + writer)   │
#                              └─────┬────────────────┘
#                              5556 │  ▲ 5557
#                              pose │  │ x2_debug
#                                   ▼  │
#                          ┌──────────────────────┐
#                          │  deploy_x2.sh sim    │
#                          │  (SONIC + MuJoCo)    │
#                          └──────────────────────┘
#
# Why this exists: the 4-process stack has a strict spawn-order +
# settle requirement (see run_planner_smoke.sh header for the
# rationale), and an even stricter shutdown order (recorder must drain
# its episode buffer BEFORE the deploy goes silent, otherwise the
# operator loses the last clip). Without a wrapper the operator has to
# juggle 4 terminals + readiness markers + Ctrl-C order; this script
# encodes all of that as one command with one trap.
#
# Usage:
#   gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
#       [--duration N] [--with-record] [--output-dir PATH] [--task STR]
#       [--robocasa-env {none,X2PickPlaceCube,X2PickPlaceBowl,X2PickPlaceApple}]
#       [--scene-xml-path PATH] [--episode-seed N]
#       [--model PATH] [--calibration PATH] [--operator-id NAME]
#       [--no-deploy] [--no-sim-viewer] [--sim-profile {parity,manual}]
#       [--apply-curl-compensation] [--apply-oppose-compensation]
#       [--no-apply-curl-compensation] [--no-apply-oppose-compensation]
#       [--rate FLOAT] [--log-dir PATH] [--cleanup-only] [--validate-only]
#
# Defaults to --teleop-only (no dataset writes). Pass --with-record
# (along with --output-dir and --task) to capture a LeRobot v2.1
# episode through the subscribe-mode pipeline.
#
# Examples:
#   # Smoke test (no writes), 5 min teleop session:
#   ./run_x2_quest3_planner_stack.sh --duration 300
#
#   # Recorded session writing into data/lerobot/x2_phase0_smoke_v0:
#   ./run_x2_quest3_planner_stack.sh --duration 600 --with-record \
#       --output-dir data/lerobot/x2_phase0_smoke_v0 \
#       --task "phase 0 subscribe-mode smoke"
#
#   # Robocasa scene (table + cube), recording episodes. Task auto-fills
#   # from the env's canonical instruction so --task can be omitted:
#   ./run_x2_quest3_planner_stack.sh --duration 1200 --with-record \
#       --output-dir data/lerobot/x2_pick_place_cube_v0 \
#       --robocasa-env X2PickPlaceCube
#
#   # Robocasa scene with reproducible per-episode placement:
#   ./run_x2_quest3_planner_stack.sh --duration 600 --with-record \
#       --output-dir data/lerobot/x2_pick_place_cube_seed42 \
#       --robocasa-env X2PickPlaceCube \
#       --episode-seed 42
#
#   # Stack against a deploy that's already running externally
#   # (skips spawning deploy_x2.sh):
#   ./run_x2_quest3_planner_stack.sh --no-deploy --duration 120
#
#   # Headless / CI variant (no MuJoCo viewer, headset still required):
#   ./run_x2_quest3_planner_stack.sh --no-sim-viewer --duration 60
#
#   ./run_x2_quest3_planner_stack.sh --cleanup-only   # free ports + kill orphans
#
#   # Sanity-check args without launching anything (prints the resolved
#   # banner, runs port + scene-XML checks, exits 0 if clean):
#   ./run_x2_quest3_planner_stack.sh --validate-only --robocasa-env X2PickPlaceCube
#
# Pre-flight (verified before any spawn):
#   - data/operator_calibrations/<operator-id>.yaml exists
#   - All 5 ports (5556, 5557, 5563, 5564, 5565) are free
#   - +2 more ports (5559, 5560) are free in --robocasa-env mode
#   - The ONNX model file exists (when deploy is being spawned)
#   - The planner primitives PKL + bins YAML exist
#   - The scene MJCF exists (when --robocasa-env is set); build with:
#       python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env <ENV>
#   - --with-record requires --output-dir; also requires --task UNLESS
#     --robocasa-env is set (then it auto-fills from scene metadata)
#
# Finger compensation defaults:
#   - flat-floor mode  : both compensations OFF unless --apply-* set
#   - robocasa mode    : both compensations ON  unless --no-apply-* set
#     (robocasa episodes are power-grasp pick-and-place; compensations
#      help the OmniHand fully close on a small object)
#
# All children inherit a fresh process group so SIGTERM cascades
# without orphaning grandchildren.

set -u
set -o pipefail

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRIMITIVES_PKL="${REPO_ROOT}/gear_sonic/data/motions/x2_planner_primitives.pkl"
BINS_YAML="${REPO_ROOT}/gear_sonic/data/motions/x2_planner_bins.yaml"
DEPLOY_SH="${REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

# PID files we need to clean up (planner writes its own; we own the rest).
PLANNER_PID_FILE="/tmp/x2_heuristic_planner.pid"

# --------------------------------------------------------------------------
# Port + topic contract (mirrored in --help so operators don't have to
# grep). Hardcoded because they're the wire-level handshake between the
# 4 processes; bumping any of these means bumping every consumer too.
# --------------------------------------------------------------------------

POSE_PORT=5556          # recorder PUB -> deploy SUB
POSE_TOPIC="pose"
DEBUG_PORT=5557         # deploy PUB -> recorder SUB
DEBUG_TOPIC="x2_debug"
PLANNER_CMD_PORT=5563   # manager PUB -> planner SUB
PLANNER_CMD_TOPIC="planner_cmd"
ARM_HANDS_PORT=5564     # manager PUB -> recorder SUB (4 multiplexed topics)
BODY_POSE_PORT=5565     # planner PUB -> recorder SUB
BODY_POSE_TOPIC="body_pose"
# Robocasa scene mode (Phase 1): two extra ports the deploy bridge
# binds when --sim-mjcf points at a robocasa scene XML.
SCENE_STATE_PORT=5559   # deploy bridge PUB -> recorder SUB (per-tick obj qpos)
SCENE_RESET_PORT=5560   # recorder PUB -> deploy bridge SUB (re-randomise objs)

# --------------------------------------------------------------------------
# CLI defaults
# --------------------------------------------------------------------------

DURATION_S=600
WITH_RECORD=0
OUTPUT_DIR=""
TASK=""
# Robocasa scene mode (off by default; legacy flat-floor recordings).
# When ROBOCASA_ENV != "none" we resolve a scene XML, forward
# --sim-mjcf to deploy_x2.sh so the bridge loads the same MJCF the
# recorder will mirror via RobocasaTaskMirror, and forward
# --robocasa-env / --episode-seed / scene ports to record_x2_dataset.
ROBOCASA_ENV="none"
SCENE_XML_PATH=""       # explicit override; auto-resolved from env name when empty
EPISODE_SEED=""         # numeric; empty = numpy global RNG
WITH_DEPLOY=1
SIM_VIEWER=1
SIM_PROFILE="parity"
SIM_RSI_PKL="${REPO_ROOT}/data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_rsi_anchor.pkl"
SIM_MODEL="${X2_PLANNER_SMOKE_MODEL:-/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx}"
SIM_CAM_TRACK_BODY="pelvis"
SIM_CAM_DISTANCE="3.5"
SIM_CAM_ELEVATION="-12"
SIM_CAM_AZIMUTH="135"
WRIST_BYPASS="ik"
WARMUP_QUIET_STAND_S="2.0"
RATE="50"
# Finger compensations: -1 = unspecified by operator, 0 = explicit off,
# 1 = explicit on. We use a tristate so the robocasa-default-on
# behaviour kicks in only when the operator hasn't expressed an
# opinion, leaving --no-apply-* as a clean override path.
APPLY_CURL_COMP=-1
APPLY_OPPOSE_COMP=-1
OPERATOR_ID="default"
CALIBRATION_PATH=""     # resolved after CLI parse from OPERATOR_ID
QUEST3_WS_PORT=8765
QUEST3_HTTP_PORT=8443
LOG_DIR=""              # auto-resolved to /tmp/<script>-<timestamp>
CLEANUP_ONLY=0
VALIDATE_ONLY=0         # exit 0 right after pre-flight; spawns nothing
SIDECAR_LOG=""          # default: <log-dir>/manager_sidecar.jsonl

usage() {
    # Print every comment line from "# Usage:" until the first
    # non-comment line (i.e. the whole hand-written help block above).
    awk '/^# Usage:/,/^[^#]/{ if ($0 ~ /^[^#]/) exit; sub(/^# ?/, ""); print }' "$0" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION_S="$2"; shift 2 ;;
        --with-record) WITH_RECORD=1; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --model) SIM_MODEL="$2"; shift 2 ;;
        --calibration) CALIBRATION_PATH="$2"; shift 2 ;;
        --operator-id) OPERATOR_ID="$2"; shift 2 ;;
        --no-deploy) WITH_DEPLOY=0; shift ;;
        --no-sim-viewer) SIM_VIEWER=0; shift ;;
        --sim-profile) SIM_PROFILE="$2"; shift 2 ;;
        --sim-rsi-pkl) SIM_RSI_PKL="$2"; shift 2 ;;
        --sim-cam-track-body) SIM_CAM_TRACK_BODY="$2"; shift 2 ;;
        --sim-cam-distance) SIM_CAM_DISTANCE="$2"; shift 2 ;;
        --sim-cam-elevation) SIM_CAM_ELEVATION="$2"; shift 2 ;;
        --sim-cam-azimuth) SIM_CAM_AZIMUTH="$2"; shift 2 ;;
        --no-sim-cam-track) SIM_CAM_TRACK_BODY=""; shift ;;
        --wrist-bypass) WRIST_BYPASS="$2"; shift 2 ;;
        --warmup-quiet-stand-s) WARMUP_QUIET_STAND_S="$2"; shift 2 ;;
        --rate) RATE="$2"; shift 2 ;;
        --apply-curl-compensation) APPLY_CURL_COMP=1; shift ;;
        --apply-oppose-compensation) APPLY_OPPOSE_COMP=1; shift ;;
        --no-apply-curl-compensation) APPLY_CURL_COMP=0; shift ;;
        --no-apply-oppose-compensation) APPLY_OPPOSE_COMP=0; shift ;;
        --quest3-ws-port) QUEST3_WS_PORT="$2"; shift 2 ;;
        --quest3-http-port) QUEST3_HTTP_PORT="$2"; shift 2 ;;
        --robocasa-env) ROBOCASA_ENV="$2"; shift 2 ;;
        --scene-xml-path) SCENE_XML_PATH="$2"; shift 2 ;;
        --episode-seed) EPISODE_SEED="$2"; shift 2 ;;
        --scene-state-port) SCENE_STATE_PORT="$2"; shift 2 ;;
        --scene-reset-port) SCENE_RESET_PORT="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --sidecar-log) SIDECAR_LOG="$2"; shift 2 ;;
        --cleanup-only) CLEANUP_ONLY=1; shift ;;
        --validate-only) VALIDATE_ONLY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="/tmp/x2_quest3_planner_stack-$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"

if [[ -z "${CALIBRATION_PATH}" ]]; then
    CALIBRATION_PATH="${REPO_ROOT}/data/operator_calibrations/${OPERATOR_ID}.yaml"
fi

if [[ -z "${SIDECAR_LOG}" ]]; then
    SIDECAR_LOG="${LOG_DIR}/manager_sidecar.jsonl"
fi

# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_DIM=$'\e[2m'; C_RESET=$'\e[0m'
log()  { printf '%s[stack %s]%s %s\n' "${C_GREEN}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
warn() { printf '%s[stack %s WARN]%s %s\n' "${C_YELLOW}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
err()  { printf '%s[stack %s ERROR]%s %s\n' "${C_RED}" "$(date +%H:%M:%S)" "${C_RESET}" "$*" >&2; }
dim()  { printf '%s%s%s\n' "${C_DIM}" "$*" "${C_RESET}"; }

# --------------------------------------------------------------------------
# Process / port helpers (lifted from run_planner_smoke.sh idioms)
# --------------------------------------------------------------------------

port_in_use() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nPi ":${port}" >/dev/null 2>&1
        return $?
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -n tcp "${port}" >/dev/null 2>&1
        return $?
    fi
    "${PYTHON}" - <<EOF
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", ${port}))
    sys.exit(1)
except OSError:
    sys.exit(0)
finally:
    s.close()
EOF
}

kill_pid_quiet() {
    local pid="$1"
    [[ -z "$pid" ]] && return 0
    kill -0 "$pid" 2>/dev/null || return 0
    local label="${2:-pid $pid}"
    log "  killing ${label} (pid=${pid})"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    warn "  force-killing ${label} (pid=${pid})"
    kill -KILL "$pid" 2>/dev/null || true
}

# Send SIGINT first (lets the recorder drain its parquet buffer),
# escalate to SIGTERM after grace_s, SIGKILL after grace_s+5.
kill_pgid_graceful() {
    local pgid="$1"
    local label="${2:-pgid $pgid}"
    local grace_s="${3:-5}"
    [[ -z "$pgid" ]] && return 0
    if ! kill -0 "-${pgid}" 2>/dev/null; then
        return 0
    fi
    log "  SIGINT ${label} (pgid=${pgid}; grace ${grace_s}s)"
    kill -INT -- "-${pgid}" 2>/dev/null || true
    local i=0
    while (( i < grace_s * 2 )); do
        kill -0 "-${pgid}" 2>/dev/null || return 0
        sleep 0.5
        i=$((i + 1))
    done
    warn "  SIGTERM ${label} after ${grace_s}s grace"
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    sleep 2
    if kill -0 "-${pgid}" 2>/dev/null; then
        warn "  SIGKILL ${label} (last resort)"
        kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
}

free_port() {
    local port="$1"
    if ! port_in_use "${port}"; then
        return 0
    fi
    log "freeing port ${port}..."
    if command -v fuser >/dev/null 2>&1; then
        fuser -k -TERM -n tcp "${port}" 2>/dev/null || true
        sleep 0.5
        fuser -k -KILL -n tcp "${port}" 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
        local pids
        pids="$(lsof -nPiTCP:"${port}" -sTCP:LISTEN -t || true)"
        for p in ${pids}; do kill_pid_quiet "${p}" "stale on :${port}"; done
    fi
}

cleanup_stale() {
    log "cleanup_stale: PID file=${PLANNER_PID_FILE} ports=${POSE_PORT},${DEBUG_PORT},${PLANNER_CMD_PORT},${ARM_HANDS_PORT},${BODY_POSE_PORT}"
    if [[ -f "${PLANNER_PID_FILE}" ]]; then
        local stale_pid
        stale_pid="$(cat "${PLANNER_PID_FILE}" 2>/dev/null || true)"
        kill_pid_quiet "${stale_pid}" "stale planner"
        rm -f "${PLANNER_PID_FILE}"
    fi
    free_port "${POSE_PORT}"
    free_port "${PLANNER_CMD_PORT}"
    free_port "${ARM_HANDS_PORT}"
    free_port "${BODY_POSE_PORT}"
    # Sweep stale x2sim docker containers from previous deploy runs.
    # See preflight_docker_cleanup() for the failure mode this fixes.
    preflight_docker_cleanup
    # We don't free DEBUG_PORT (5557) here because the docker sweep
    # above already releases it via container shutdown; freeing it
    # blindly would yank a legitimate parallel deploy_x2.sh out from
    # under another shell.
}

# --------------------------------------------------------------------------
# Docker-aware deploy cleanup. The deploy_x2.sh sim path runs INSIDE a
# docker_x2-x2sim-run-<hex> container (via `docker compose run`).
# Signals from the host bash do NOT reliably propagate into the
# container -- the compose-run shell can race with our SIGTERM and exit
# before delivering the signal to PID 1 inside the container. The
# observable symptom is that the MuJoCo viewer + bridge keep running
# AFTER our wrapper has exited, holding ports 5556/5557 forever
# (and resisting host-side ``pkill`` because the container processes
# run as root).
#
# Lifted verbatim from record_x2_dataset.sh (which has solved this
# the hard way in production for months). Two phases:
#   1. Find the container by parsing the
#      ``docker_x2-x2sim-run-<hex>`` line that compose writes to the
#      deploy log on bring-up.
#   2. ``docker stop --timeout 5`` it explicitly. Succeeds even if
#      the host-side bash already exited or is wedged.
#
# Also called proactively at startup (preflight_docker_cleanup) and
# when the user passes --cleanup-only, so a previous SIGKILL'd run
# can't strand a container holding the deploy ports.
# --------------------------------------------------------------------------
stop_deploy_container() {
    local deploy_log="$1"
    [[ -z "${deploy_log}" || ! -f "${deploy_log}" ]] && return 0
    if ! command -v docker >/dev/null 2>&1; then
        return 0
    fi
    local deploy_container
    deploy_container="$(grep -oE 'docker_x2-x2sim-run-[a-f0-9]+' "${deploy_log}" 2>/dev/null | tail -1 || true)"
    if [[ -z "${deploy_container}" ]]; then
        return 0
    fi
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${deploy_container}"; then
        return 0
    fi
    log "  docker stop ${deploy_container} (--timeout 5)"
    docker stop --timeout 5 "${deploy_container}" >/dev/null 2>&1 || true
}

preflight_docker_cleanup() {
    if ! command -v docker >/dev/null 2>&1; then
        return 0
    fi
    # Earlier versions of this filter used ``--filter ancestor=x2sim``,
    # which matched zero containers in production: the actual image tag
    # is ``gr00t-x2sim:latest`` (see gear_sonic_deploy/docker_x2/
    # docker-compose.yml) and the running container is named
    # ``docker_x2-x2sim-run-<hex>`` by ``docker compose run``. We now
    # match on container *name* (``x2sim-run``), which is stable across
    # image-tag renames and matches both the upstream tag and any
    # branch-built variant. Keep the legacy ancestor filter as a
    # second pass so we never miss a container that was started with
    # an unusual name override.
    local stale
    stale="$(
        {
            docker ps --filter name=x2sim-run --format '{{.Names}}' 2>/dev/null
            docker ps --filter ancestor=gr00t-x2sim --format '{{.Names}}' 2>/dev/null
            docker ps --filter ancestor=x2sim --format '{{.Names}}' 2>/dev/null
        } | sort -u
    )"
    if [[ -z "${stale}" ]]; then
        return 0
    fi
    warn "stale x2sim container(s) detected from previous run; stopping:"
    while IFS= read -r c; do
        [[ -z "$c" ]] && continue
        warn "  - ${c}"
        docker stop --timeout 5 "${c}" >/dev/null 2>&1 || true
    done <<< "${stale}"
}

# --------------------------------------------------------------------------
# wait_for_log_marker: tails ${log_path} for ${marker} until ${pid} is
# alive AND the marker appears, or ${timeout_s} elapses. Returns 0/1.
# --------------------------------------------------------------------------
wait_for_log_marker() {
    local log_path="$1"
    local pid="$2"
    local marker="$3"
    local timeout_s="${4:-60}"
    local label="${5:-process}"
    local start_ts
    start_ts=$(date +%s)
    while :; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            err "${label} died during bring-up (pid=${pid}); tail of log:"
            tail -n 40 "${log_path}" >&2 || true
            return 1
        fi
        if [[ -f "${log_path}" ]] && grep -F -q "${marker}" "${log_path}" 2>/dev/null; then
            return 0
        fi
        local now elapsed
        now=$(date +%s)
        elapsed=$((now - start_ts))
        if (( elapsed > timeout_s )); then
            err "${label} did not log '${marker}' within ${timeout_s}s; tail:"
            tail -n 40 "${log_path}" >&2 || true
            return 1
        fi
        sleep 0.5
    done
}

# --------------------------------------------------------------------------
# Cleanup trap (set BEFORE any child is spawned). Reverse-order shutdown:
#   recorder -> manager -> planner -> deploy
# This ordering is critical:
#   - recorder must drain its episode buffer + close parquet writers
#     BEFORE upstream stops feeding it (otherwise auto-flush logic in
#     X2DatasetRecorder may have a partial last frame).
#   - manager goes next so the planner stops getting commands and
#     transitions to idle_stand cleanly.
#   - planner shuts down before the deploy so the deploy's pose SUB
#     sees a clean disconnect (vs. a stale wire).
#   - deploy last; its trap restarts MC + tears down docker.
# --------------------------------------------------------------------------

DEPLOY_PID=""
DEPLOY_LOG=""
PLANNER_PID=""
MANAGER_PID=""
RECORDER_PID=""
RECORDER_PGID=""

cleanup_children() {
    log "shutting down children (reverse spawn order)..."
    if [[ -n "${RECORDER_PGID}" ]]; then
        # Recorder wrapped in setsid -> pgid==pid. Use SIGINT+grace so
        # the LeRobot writer can flush a last episode.
        kill_pgid_graceful "${RECORDER_PGID}" "recorder" 8
    fi
    kill_pid_quiet "${MANAGER_PID}"  "manager"
    kill_pid_quiet "${PLANNER_PID}"  "planner"
    if [[ -n "${DEPLOY_PID}" ]] && kill -0 "${DEPLOY_PID}" 2>/dev/null; then
        # SIGINT (not SIGTERM) so deploy_x2.sh's RAMP_OUT + viewer
        # teardown hooks fire. Then wait briefly for it to exit on its
        # own; the docker container shutdown happens below regardless.
        log "  SIGINT deploy host-side bash (pid=${DEPLOY_PID})"
        kill -INT "${DEPLOY_PID}" 2>/dev/null || true
        local i=0
        while (( i < 10 )); do
            kill -0 "${DEPLOY_PID}" 2>/dev/null || break
            sleep 0.5
            i=$((i + 1))
        done
        kill_pid_quiet "${DEPLOY_PID}" "deploy host-bash"
    fi
    # Always sweep the docker container even if the host bash already
    # exited cleanly -- compose can race with our signal and leak the
    # container as a host-visible orphan that holds the deploy ports.
    stop_deploy_container "${DEPLOY_LOG}"
    rm -f "${PLANNER_PID_FILE}"
}
trap 'cleanup_children; exit 130' INT TERM
trap 'cleanup_children' EXIT

# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

if [[ "${CLEANUP_ONLY}" -eq 1 ]]; then
    cleanup_stale
    log "cleanup-only: done"
    trap - EXIT
    exit 0
fi

# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------

if [[ ! -f "${PRIMITIVES_PKL}" ]]; then
    err "primitives PKL not found: ${PRIMITIVES_PKL}"
    err "run: ${PYTHON} -m gear_sonic.scripts.curate_x2_primitives"
    exit 1
fi
if [[ ! -f "${BINS_YAML}" ]]; then
    err "bins YAML not found: ${BINS_YAML}"
    exit 1
fi
if [[ ! -f "${CALIBRATION_PATH}" ]]; then
    err "operator calibration not found: ${CALIBRATION_PATH}"
    err "run: ${PYTHON} -m gear_sonic.scripts.vr_operator_calibrate --operator-id ${OPERATOR_ID}"
    exit 1
fi
if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    if [[ ! -x "${DEPLOY_SH}" ]]; then
        err "deploy_x2.sh not found / not executable: ${DEPLOY_SH}"
        exit 1
    fi
    if [[ -z "${SIM_MODEL}" || ! -f "${SIM_MODEL}" ]]; then
        err "deploy needs --model PATH (or X2_PLANNER_SMOKE_MODEL env)."
        err "tried: ${SIM_MODEL:-<unset>}"
        exit 1
    fi
fi
if [[ "${WITH_RECORD}" -eq 1 ]]; then
    if [[ -z "${OUTPUT_DIR}" ]]; then
        err "--with-record requires --output-dir"
        exit 1
    fi
    # In robocasa mode the recorder auto-fills --task from the env's
    # canonical instruction (see record_x2_dataset.py main()), so the
    # operator can omit --task. Outside robocasa mode we still demand
    # an explicit task string -- the LeRobot v2.1 schema requires one
    # per episode and we don't want to silently default it.
    if [[ -z "${TASK}" && "${ROBOCASA_ENV}" == "none" ]]; then
        err "--with-record requires --task (or pass --robocasa-env to auto-fill from scene metadata)"
        exit 1
    fi
fi

# Robocasa scene resolution. Mirrors record_x2_dataset.sh so the two
# entrypoints stay in lock-step on which env names exist and where
# their MJCFs live.
ROBOCASA_SCENE_XML=""
case "${ROBOCASA_ENV}" in
    none) ;;
    X2PickPlaceCube|X2PickPlaceBowl|X2PickPlaceApple)
        if [[ -n "${SCENE_XML_PATH}" ]]; then
            ROBOCASA_SCENE_XML="${SCENE_XML_PATH}"
        else
            ROBOCASA_SCENE_XML="${REPO_ROOT}/gear_sonic/data/assets/robocasa_scenes/${ROBOCASA_ENV}.xml"
        fi
        if [[ ! -f "${ROBOCASA_SCENE_XML}" ]]; then
            err "scene MJCF for --robocasa-env ${ROBOCASA_ENV} not found at"
            err "  ${ROBOCASA_SCENE_XML}"
            err "Build it via:"
            err "  ${PYTHON} -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env ${ROBOCASA_ENV}"
            exit 1
        fi
        # Robocasa scenes are always built on top of the OmniHand-
        # augmented MJCF and the deploy bridge unconditionally needs the
        # OmniHand to be present, so --sim-with-omnihand is already on
        # in our DEPLOY_ARGS. (record_x2_dataset.sh enforces the same
        # invariant.)
        ;;
    *)
        err "--robocasa-env must be one of 'none', 'X2PickPlaceCube', 'X2PickPlaceBowl', 'X2PickPlaceApple' (got '${ROBOCASA_ENV}')"
        exit 1
        ;;
esac

# Finger compensations: in robocasa mode the operator is doing power
# grasps on a cube / bowl, which is exactly the case the recorder's
# own log hint flags as "turn ON if fingers don't fully close".
# Default both to ON when the operator hasn't expressed an opinion,
# and surface the decision in the banner so it's never a surprise.
# Outside robocasa mode the historical default of OFF is preserved.
if [[ -n "${ROBOCASA_SCENE_XML}" ]]; then
    if [[ "${APPLY_CURL_COMP}"   -eq -1 ]]; then APPLY_CURL_COMP=1;   fi
    if [[ "${APPLY_OPPOSE_COMP}" -eq -1 ]]; then APPLY_OPPOSE_COMP=1; fi
else
    if [[ "${APPLY_CURL_COMP}"   -eq -1 ]]; then APPLY_CURL_COMP=0;   fi
    if [[ "${APPLY_OPPOSE_COMP}" -eq -1 ]]; then APPLY_OPPOSE_COMP=0; fi
fi

# Port pre-check. Any of these in use means we'd silently re-bind the
# wrong process; refuse early instead. Scene-state / scene-reset ports
# are only checked when we're actually using a robocasa scene -- on a
# vanilla flat-floor recording the deploy bridge doesn't bind them.
declare -A PORT_LABELS=(
    ["${POSE_PORT}"]="recorder->deploy pose (5556)"
    ["${PLANNER_CMD_PORT}"]="manager->planner planner_cmd (5563)"
    ["${ARM_HANDS_PORT}"]="manager->recorder arm/hands (5564)"
    ["${BODY_POSE_PORT}"]="planner->recorder body_pose (5565)"
)
PORTS_TO_CHECK=("${POSE_PORT}" "${PLANNER_CMD_PORT}" "${ARM_HANDS_PORT}" "${BODY_POSE_PORT}")
if [[ -n "${ROBOCASA_SCENE_XML}" ]]; then
    PORT_LABELS["${SCENE_STATE_PORT}"]="bridge->recorder scene_state (5559)"
    PORT_LABELS["${SCENE_RESET_PORT}"]="recorder->bridge scene_reset (5560)"
    PORTS_TO_CHECK+=("${SCENE_STATE_PORT}" "${SCENE_RESET_PORT}")
fi
for port in "${PORTS_TO_CHECK[@]}"; do
    if port_in_use "${port}"; then
        err "port ${port} (${PORT_LABELS[$port]}) is in use."
        err "run: $0 --cleanup-only"
        exit 1
    fi
done
# Sweep stale x2sim docker containers from previous runs BEFORE we
# probe DEBUG_PORT. Without this a container leaked from a previous
# SIGKILL'd run holds :5557 and our pre-flight refuses to start.
if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    preflight_docker_cleanup
fi

# DEBUG_PORT (5557) belongs to the deploy; it'll bind it itself. We
# only assert it's free if we're about to spawn the deploy ourselves.
if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    if port_in_use "${DEBUG_PORT}"; then
        err "port ${DEBUG_PORT} (deploy x2_debug PUB) is in use."
        err "run: $0 --cleanup-only or shut down the existing deploy first."
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------

cat <<EOF
${C_GREEN}┌──────────────────────────────────────────────────────────────────────┐
│  X2 Quest 3 planner-driven stack runner (Phase 0)                    │
└──────────────────────────────────────────────────────────────────────┘${C_RESET}
  log dir          : ${LOG_DIR}
  duration         : ${DURATION_S}s
  mode             : $([[ "${WITH_RECORD}" -eq 1 ]] && echo "RECORD -> ${OUTPUT_DIR}" || echo "TELEOP-ONLY")
  task             : ${TASK:-(none -- robocasa auto-fills from scene metadata)}
  scene            : $([[ "${ROBOCASA_ENV}" == "none" ]] && echo "(flat floor, no robocasa scene)" || echo "${ROBOCASA_ENV} -> ${ROBOCASA_SCENE_XML}")
  finger comp      : curl=$([[ "${APPLY_CURL_COMP}" -eq 1 ]] && echo on || echo off)  oppose=$([[ "${APPLY_OPPOSE_COMP}" -eq 1 ]] && echo on || echo off)$([[ -n "${ROBOCASA_SCENE_XML}" ]] && echo "  (robocasa default; pass --no-apply-{curl,oppose}-compensation to override)" || echo "  (pass --apply-{curl,oppose}-compensation to enable)")
  deploy           : $([[ "${WITH_DEPLOY}" -eq 1 ]] && echo "ON  (sim --vla, profile=${SIM_PROFILE}, viewer=$([[ "${SIM_VIEWER}" -eq 1 ]] && echo on || echo off))" || echo "OFF (assume external)")
  ONNX model       : ${SIM_MODEL}
  operator         : ${OPERATOR_ID} (${CALIBRATION_PATH})
  WebXR endpoint   : wss://<host>:${QUEST3_WS_PORT}, https://<host>:${QUEST3_HTTP_PORT}
  ports            : pose=${POSE_PORT}  x2_debug=${DEBUG_PORT}  planner_cmd=${PLANNER_CMD_PORT}
                     arm/hands=${ARM_HANDS_PORT}  body_pose=${BODY_POSE_PORT}$([[ -n "${ROBOCASA_SCENE_XML}" ]] && echo "
                     scene_state=${SCENE_STATE_PORT}  scene_reset=${SCENE_RESET_PORT}" || true)
  manager sidecar  : ${SIDECAR_LOG}
EOF
echo

# --------------------------------------------------------------------------
# --validate-only short-circuit. Exits cleanly right after pre-flight
# (port checks, scene XML resolution, finger-comp default-on, banner
# print) without spawning any of the four child processes. Two
# legitimate use cases:
#
#  (1) Operators sanity-check a long argument list before committing
#      to a full run -- "did the wrapper actually pick up my
#      --robocasa-env, did it free the right ports, what does the
#      banner look like, did it auto-enable finger compensation".
#      Cheap to invoke, leaves zero processes behind.
#
#  (2) The CLI test suite (tests/test_run_x2_quest3_planner_stack_cli
#      .py) needs to assert that --with-record + --robocasa-env
#      passes the task-required gate WITHOUT actually launching the
#      planner / manager / recorder. A 30 s subprocess timeout in
#      pytest would otherwise SIGKILL the wrapper bash and leave the
#      three Python children orphaned under PID 1, holding the Phase
#      0 ports until the operator hunts them down with lsof. (We
#      learned this the hard way 2026-05-13 -- three back-to-back
#      "port 5556 in use" failures were caused by exactly this leak
#      from CI runs of the test.)
# --------------------------------------------------------------------------
if [[ "${VALIDATE_ONLY}" -eq 1 ]]; then
    log "validate-only: pre-flight passed; exiting before any spawn."
    trap - EXIT
    exit 0
fi

# --------------------------------------------------------------------------
# Step 1 — Spawn deploy (FIRST). Wait for "Launching ..." marker, then
# settle 2 s. Same boot pattern as record_x2_dataset.sh and
# run_planner_smoke.sh --with-deploy.
# --------------------------------------------------------------------------

if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_LOG="${LOG_DIR}/deploy.log"
    DEPLOY_DURATION_S=$(( DURATION_S + 30 ))

    # Parity profile needs the bridge RSI PKL. Auto-bake on first use
    # so a fresh checkout / primitive rebuild doesn't fail at boot.
    if [[ "${SIM_PROFILE}" == "parity" ]]; then
        if [[ ! -f "${SIM_RSI_PKL}" ]]; then
            log "RSI anchor PKL not found at ${SIM_RSI_PKL}; baking now ..."
            if ! "${PYTHON}" -m gear_sonic.scripts.bake_planner_rsi_anchor \
                    --primitives-pkl "${PRIMITIVES_PKL}" \
                    --bins-yaml "${BINS_YAML}" \
                    --out "${SIM_RSI_PKL}" \
                    >>"${LOG_DIR}/rsi_anchor_bake.log" 2>&1; then
                err "failed to bake RSI anchor; see ${LOG_DIR}/rsi_anchor_bake.log"
                exit 1
            fi
        fi
        log "parity RSI source: ${SIM_RSI_PKL}"
    fi

    DEPLOY_ARGS=(
        sim
        --no-confirm
        --vla
        --vla-zmq-host 127.0.0.1
        --vla-zmq-port "${POSE_PORT}"
        --vla-zmq-topic "${POSE_TOPIC}"
        --vla-debug-port "${DEBUG_PORT}"
        --vla-debug-topic "${DEBUG_TOPIC}"
        --sim-profile "${SIM_PROFILE}"
        --sim-with-omnihand
        --wrist-bypass "${WRIST_BYPASS}"
        --model "${SIM_MODEL}"
        --autostart-after 0
        --max-duration "${DEPLOY_DURATION_S}"
    )
    if [[ "${SIM_PROFILE}" == "parity" ]]; then
        DEPLOY_ARGS+=(--motion "${SIM_RSI_PKL}")
    fi
    # Robocasa: forward --sim-mjcf so the deploy bridge loads the same
    # scene XML the recorder will mirror via RobocasaTaskMirror. The
    # bridge auto-discovers the .json sidecar next to the .xml and uses
    # it to resolve scene-object freejoint qpos addresses for its
    # scene_state PUB and scene_reset SUB plumbing.
    if [[ -n "${ROBOCASA_SCENE_XML}" ]]; then
        DEPLOY_ARGS+=(--sim-mjcf "${ROBOCASA_SCENE_XML}")
    fi
    if [[ "${SIM_VIEWER}" -eq 1 ]]; then
        DEPLOY_ARGS+=(--sim-viewer)
        if [[ -n "${SIM_CAM_TRACK_BODY}" ]]; then
            DEPLOY_ARGS+=(
                --sim-cam-track-body "${SIM_CAM_TRACK_BODY}"
                --sim-cam-distance "${SIM_CAM_DISTANCE}"
                --sim-cam-elevation "${SIM_CAM_ELEVATION}"
                --sim-cam-azimuth "${SIM_CAM_AZIMUTH}"
            )
        fi
    fi

    log "Step 1/4 — spawning deploy_x2.sh sim --vla -> ${DEPLOY_LOG}"
    "${DEPLOY_SH}" "${DEPLOY_ARGS[@]}" >"${DEPLOY_LOG}" 2>&1 &
    DEPLOY_PID=$!

    log "  waiting for deploy 'Launching ...' marker (up to 180s)..."
    if ! wait_for_log_marker "${DEPLOY_LOG}" "${DEPLOY_PID}" "Launching ..." 180 "deploy"; then
        exit 1
    fi
    log "  deploy READY (pid=${DEPLOY_PID}); settle 2s before planner ..."
    sleep 2.0
else
    log "Step 1/4 — deploy spawn SKIPPED (--no-deploy). Recorder will publish 'pose' on :${POSE_PORT} regardless; the deploy you have running externally must be subscribed there."
fi

# --------------------------------------------------------------------------
# Step 2 — Spawn planner (SECOND). Configure for Phase 0 wire:
#   - PUB body_pose@5565 (instead of legacy direct-to-deploy pose@5556)
#   - SUB planner_cmd@5563 (manager drives state machine via Quest3 inputs)
# Wait for the "Phase 0 mode" log line so we know body_pose PUB is live.
# --------------------------------------------------------------------------

PLANNER_LOG="${LOG_DIR}/planner.log"

PLANNER_ARGS=(
    -m gear_sonic.scripts.x2_heuristic_planner
    --primitives "${PRIMITIVES_PKL}"
    --bins "${BINS_YAML}"
    --pub-host 127.0.0.1
    --body-pose-port "${BODY_POSE_PORT}"
    --zmq-cmd-host localhost
    --zmq-cmd-port "${PLANNER_CMD_PORT}"
    --zmq-cmd-topic "${PLANNER_CMD_TOPIC}"
    --warmup-quiet-stand-s "${WARMUP_QUIET_STAND_S}"
    --pid-file "${PLANNER_PID_FILE}"
    --duration-s "${DURATION_S}"
)

log "Step 2/4 — spawning x2_heuristic_planner -> ${PLANNER_LOG}"
"${PYTHON}" "${PLANNER_ARGS[@]}" >"${PLANNER_LOG}" 2>&1 &
PLANNER_PID=$!

log "  waiting for planner 'Phase 0 mode' marker (up to 30s)..."
if ! wait_for_log_marker "${PLANNER_LOG}" "${PLANNER_PID}" \
        "Phase 0 mode: publishing 'body_pose'" 30 "planner"; then
    exit 1
fi
log "  planner READY (pid=${PLANNER_PID}); settle 0.5s before manager ..."
sleep 0.5

# --------------------------------------------------------------------------
# Step 3 — Spawn manager (THIRD). Quest3Reader + IntentDecoder +
# Retargeter live here. Manager waits internally for the WebXR client
# to connect; we don't gate the recorder on that (operator can start
# the recorder while still strapping the headset on).
# --------------------------------------------------------------------------

MANAGER_LOG="${LOG_DIR}/manager.log"

MANAGER_ARGS=(
    -m gear_sonic.scripts.quest3_manager_x2
    --ws-port "${QUEST3_WS_PORT}"
    --http-port "${QUEST3_HTTP_PORT}"
    --calibration "${CALIBRATION_PATH}"
    --planner-cmd-host '*'
    --planner-cmd-port "${PLANNER_CMD_PORT}"
    --planner-cmd-topic "${PLANNER_CMD_TOPIC}"
    --recorder-pub-host '*'
    --recorder-pub-port "${ARM_HANDS_PORT}"
    --rate "${RATE}"
    --sidecar-log "${SIDECAR_LOG}"
)
# Finger-curl / thumb-oppose compensations live on the MANAGER in
# subscribe-mode (the manager owns the Retargeter; the recorder just
# forwards what arrives on hand_finger_cmd). Forwarding these flags
# only to the recorder -- as we did pre-2026-05-13 -- silently no-ops
# them, which the operator notices as "fingers don't fully close on
# a power-grasp pick-and-place" (exact symptom from the recorder log
# hint). Forward to BOTH ends: the manager so it actually applies,
# the recorder so it stamps the per-frame metadata correctly.
if [[ "${APPLY_CURL_COMP}" -eq 1 ]]; then
    MANAGER_ARGS+=(--apply-curl-compensation)
fi
if [[ "${APPLY_OPPOSE_COMP}" -eq 1 ]]; then
    MANAGER_ARGS+=(--apply-oppose-compensation)
fi

log "Step 3/4 — spawning quest3_manager_x2 -> ${MANAGER_LOG}"
"${PYTHON}" "${MANAGER_ARGS[@]}" >"${MANAGER_LOG}" 2>&1 &
MANAGER_PID=$!

# Manager prints its bind log very early; wait briefly for the recorder
# PUB bind so the recorder doesn't slow-joiner-drop the first batch.
log "  waiting for manager 'recorder PUB bound' marker (up to 30s)..."
if ! wait_for_log_marker "${MANAGER_LOG}" "${MANAGER_PID}" \
        "recorder PUB bound at" 30 "manager"; then
    # Older builds use slightly different wording; fall back to a
    # generic "Quest3ManagerX2 ready" check.
    if ! wait_for_log_marker "${MANAGER_LOG}" "${MANAGER_PID}" \
            "Quest3ManagerX2" 5 "manager"; then
        exit 1
    fi
fi
log "  manager READY (pid=${MANAGER_PID}); settle 0.5s before recorder ..."
sleep 0.5

cat <<EOF

${C_YELLOW}┌──────────────────────────────────────────────────────────────────────┐
│  ACTION REQUIRED: open WebXR client on the Quest 3                   │
│    Browser URL:  https://<workstation-IP>:${QUEST3_HTTP_PORT}                       │
│    Accept the self-signed certificate, tap "Enter VR".               │
│                                                                       │
│  Mode chord:                                                         │
│    A+B+X+Y           OFF <-> LOCOMOTION (back to OFF from any mode)  │
│    B (in LOCOMOTION) LOCOMOTION <-> ARM_MANIPULATION                 │
│  LOCOMOTION (push the stick the way you want the robot to move):    │
│    L stick fwd       1 stride forward in world (logged: back_step)   │
│    L stick back      1 stride backward in world (logged: fwd_step)   │
│    L stick fwd + A   continuous walk forward (logged: walk/backward) │
│    L stick back + A  continuous walk backward (logged: walk/forward) │
│    L stick L/R       side_left_step / side_right_step                │
│    R stick L/R hard  turn_left_45deg / turn_right_45deg              │
│    R stick L/R hard + X held   turn_left_90deg / turn_right_90deg    │
│  Continuous waist (v7; soft R stick, position-mapped, slewed 60deg/s):│
│    R stick fwd / back  forward / backward lean (clamp +/-20deg)      │
│    R stick L/R soft    torso twist L/R       (clamp +/-40deg)        │
│    R stick L/R + A     lateral lean L/R sway (clamp +/-10deg)        │
│  Note: the planner-log "command" appears flipped vs. the operator    │
│  intent because the curated bins were authored in a body frame       │
│  rotated 180 deg from the bridge's RSI init. End-to-end behaviour    │
│  is correct (push the stick the way you want the robot to move).    │
│  ARM_MANIPULATION (arms track VR + recorder buttons):                │
│    A press           engage / disengage arm IK                       │
│    X press           start episode (--with-record only)              │
│    Y press           stop & save episode (--with-record only)        │
│  (B-single still toggles LOCOMOTION <-> ARM_MANIPULATION; no chord.) │
│  Stick clicks (v7.1; LOCOMOTION + ARM_MAN; idle in OFF):             │
│    L thumbstick click   cycle deploy MuJoCo viewer fixed cameras     │
│                         (sends ']' via xdotool; needs xdotool)       │
│    R thumbstick click   FREEZE / RELEASE waist hold at current pose  │
│                         (planner stays in STATIC_HOLD while frozen)  │
│                                                                       │
│  Full reference: docs/source/tutorials/x2_quest3_planner_stack_     │
│                  cheatsheet.md                                       │
└──────────────────────────────────────────────────────────────────────┘${C_RESET}
EOF

# --------------------------------------------------------------------------
# Step 4 — Spawn recorder (LAST). Subscribe-only mode merges
# planner.body_pose + manager.arm_targets/hand_finger_cmd and
# republishes 'pose' on ${POSE_PORT} for the deploy. Episode lifecycle
# is driven by recorder_cmd from the manager.
# --------------------------------------------------------------------------

RECORDER_LOG="${LOG_DIR}/recorder.log"

RECORDER_ARGS=(
    -m gear_sonic.scripts.record_x2_dataset
    --body-pose-source zmq
    --arm-targets-source zmq
    --body-pose-sub-host localhost
    --body-pose-sub-port "${BODY_POSE_PORT}"
    --body-pose-sub-topic "${BODY_POSE_TOPIC}"
    --arm-and-hands-sub-host localhost
    --arm-and-hands-sub-port "${ARM_HANDS_PORT}"
    --pub-host '*'
    --pub-port "${POSE_PORT}"
    --pub-topic "${POSE_TOPIC}"
    --sub-host localhost
    --sub-port "${DEBUG_PORT}"
    --sub-topic "${DEBUG_TOPIC}"
    --rate "${RATE}"
)
if [[ "${WITH_RECORD}" -eq 1 ]]; then
    RECORDER_ARGS+=(--output-dir "${OUTPUT_DIR}")
    # Task may be empty in robocasa mode (recorder auto-fills from
    # the scene metadata). Forward only when set so the recorder's
    # auto-fill path runs.
    if [[ -n "${TASK}" ]]; then
        RECORDER_ARGS+=(--task "${TASK}")
    fi
else
    RECORDER_ARGS+=(--teleop-only)
fi
if [[ "${APPLY_CURL_COMP}" -eq 1 ]]; then
    RECORDER_ARGS+=(--apply-curl-compensation)
fi
if [[ "${APPLY_OPPOSE_COMP}" -eq 1 ]]; then
    RECORDER_ARGS+=(--apply-oppose-compensation)
fi
# Robocasa scene mode: forward the env name and explicit XML path so
# the recorder doesn't have to re-resolve from its own scenes dir,
# plus the per-side ZMQ port pair for scene_state / scene_reset.
# Only meaningful with --with-record (the recorder ignores robocasa
# fields in --teleop-only mode), but harmless to forward anyway --
# the scene-state SUB just sits idle when no episode is open.
if [[ -n "${ROBOCASA_SCENE_XML}" ]]; then
    RECORDER_ARGS+=(
        --robocasa-env "${ROBOCASA_ENV}"
        --scene-xml-path "${ROBOCASA_SCENE_XML}"
        --scene-state-sub-host localhost
        --scene-state-sub-port "${SCENE_STATE_PORT}"
        --scene-reset-pub-host '*'
        --scene-reset-pub-port "${SCENE_RESET_PORT}"
    )
    if [[ -n "${EPISODE_SEED}" ]]; then
        RECORDER_ARGS+=(--episode-seed "${EPISODE_SEED}")
    fi
fi

log "Step 4/4 — spawning record_x2_dataset (subscribe mode) -> ${RECORDER_LOG}"
# setsid so the recorder is its own session leader; we use SIGINT on
# the whole pgid in cleanup so the LeRobot writer can drain its open
# episode buffer cleanly.
setsid "${PYTHON}" "${RECORDER_ARGS[@]}" >"${RECORDER_LOG}" 2>&1 < /dev/null &
RECORDER_PID=$!
RECORDER_PGID="${RECORDER_PID}"

log "  waiting for recorder 'first body_pose received' marker (up to 60s)..."
if ! wait_for_log_marker "${RECORDER_LOG}" "${RECORDER_PID}" \
        "first body_pose received" 60 "recorder"; then
    err "Recorder never received body_pose. Likely causes:"
    err "  - planner died after step 2 (check ${PLANNER_LOG})"
    err "  - port :${BODY_POSE_PORT} bound by something else"
    err "  - planner --body-pose-port flag wrong (check ${PLANNER_LOG})"
    exit 1
fi

log "  recorder READY (pid=${RECORDER_PID}); merge+publish loop active"

# --------------------------------------------------------------------------
# Run loop — wait for the planner (which has --duration-s baked in) OR
# any child to die, whichever comes first. Prints periodic health.
# --------------------------------------------------------------------------

cat <<EOF

${C_GREEN}=== Stack is LIVE. Logs streaming under ${LOG_DIR}/ ===${C_RESET}
  Tail any of:
    tail -f ${LOG_DIR}/deploy.log
    tail -f ${LOG_DIR}/planner.log
    tail -f ${LOG_DIR}/manager.log
    tail -f ${LOG_DIR}/recorder.log
    tail -f ${SIDECAR_LOG}

Manager events streamed below as [mgr] (mode + button feedback).
Recorder events streamed below as [recorder] (episode lifecycle + on-disk paths).
Ctrl-C to shut down.
EOF

# Mirror manager log to foreground so the operator sees real-time
# button feedback (mode transitions, [A] arm tracking, OFF-mode
# hints, etc.) without needing a second terminal. We keep only
# manager-emitted lines (filtered on the 'quest3_manager_x2' marker
# the Python logger inserts) and rewrite the verbose timestamp
# prefix to a short '[mgr]' tag.
tail -F -n 0 "${MANAGER_LOG}" 2>/dev/null \
    | grep --line-buffered -F 'quest3_manager_x2' \
    | sed -u -E 's/^\[[^]]+ INFO quest3_manager_x2\][[:space:]]*/[mgr] /' &
MANAGER_TAIL_PID=$!

# Mirror a TIGHT subset of the recorder log too. We deliberately do
# NOT mirror everything from recorder.log (it carries periodic
# status, sonic-correction warnings, scene_state SUB chatter, etc
# that would drown out the [mgr] stream). We keep:
#   - episode-lifecycle button echoes:  [recorder] [X] ...  /  [Y] ...
#   - episode-saved on-disk paths:      [recorder]     parquet -> ...
#                                       [recorder]     mp4     -> ...
#   - auto-discard / drop diagnostics:  [recorder] [auto-discard] ...
#
# The leading "[recorder]" prefix is kept verbatim so it visually
# pairs with the existing "[mgr]" tag in the foreground stream.
# Coordinated with the print() call sites in
# gear_sonic/utils/teleop/x2_dataset_recorder.py:_stop_episode and
# _start_episode -- if you change the indent or the "->" arrow style
# in those prints, update the regex here in the same commit.
tail -F -n 0 "${RECORDER_LOG}" 2>/dev/null \
    | grep --line-buffered -E '^\[recorder\] (\[(X|Y|auto-discard)\]|    )' &
RECORDER_TAIL_PID=$!

trap 'cleanup_children; kill_pid_quiet "${MANAGER_TAIL_PID}" "manager-tail"; kill_pid_quiet "${RECORDER_TAIL_PID}" "recorder-tail"; exit 130' INT TERM
trap 'cleanup_children; kill_pid_quiet "${MANAGER_TAIL_PID}" "manager-tail"; kill_pid_quiet "${RECORDER_TAIL_PID}" "recorder-tail"' EXIT

START_TS=$(date +%s)
LAST_HEARTBEAT_TS=${START_TS}
while :; do
    # Any child died -> bail (cleanup trap fires).
    for pid_var in DEPLOY_PID PLANNER_PID MANAGER_PID RECORDER_PID; do
        pid="${!pid_var}"
        [[ -z "${pid}" ]] && continue
        if ! kill -0 "${pid}" 2>/dev/null; then
            err "${pid_var} (pid=${pid}) exited prematurely. Tail of its log:"
            case "${pid_var}" in
                DEPLOY_PID)   tail -n 30 "${DEPLOY_LOG:-/dev/null}" >&2 || true ;;
                PLANNER_PID)  tail -n 30 "${PLANNER_LOG}"  >&2 || true ;;
                MANAGER_PID)  tail -n 30 "${MANAGER_LOG}"  >&2 || true ;;
                RECORDER_PID) tail -n 30 "${RECORDER_LOG}" >&2 || true ;;
            esac
            exit 1
        fi
    done

    NOW_TS=$(date +%s)
    ELAPSED=$((NOW_TS - START_TS))
    if (( ELAPSED >= DURATION_S )); then
        log "duration ${DURATION_S}s reached; shutting down stack ..."
        exit 0
    fi
    if (( NOW_TS - LAST_HEARTBEAT_TS >= 30 )); then
        LAST_HEARTBEAT_TS=${NOW_TS}
        log "alive: t=${ELAPSED}s/${DURATION_S}s  pids[deploy=${DEPLOY_PID:-skipped} planner=${PLANNER_PID} manager=${MANAGER_PID} recorder=${RECORDER_PID}]"
    fi
    sleep 1
done
