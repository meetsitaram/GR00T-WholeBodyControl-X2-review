#!/usr/bin/env bash
# X2 PKL-driven planner-stack runner.
#
# Sister wrapper of ``run_x2_quest3_planner_stack.sh``: same kplanner +
# deploy stack, but with a recorded PKL motion clip in place of the
# Quest 3 + manager input source. Use this to isolate
# "kplanner -> deploy" bugs from "Quest 3 input dynamics" bugs --
# both wrappers go through the same ``planner_cmd:5563`` ZMQ wire, so
# anything that walks here but fails under Quest 3 teleop points at the
# input chain (intent decoder / debounce / hysteresis), not the model.
#
# Launcher family (single-shell deploy + foreground client + trap-cleaned
# reverse-order shutdown):
#
#     teleop          run_x2_quest3_planner_stack.sh
#     recording       run_x2_quest3_planner_stack.sh --with-record --head-cameras
#     autonomous VLA  run_x2_vla_runtime.sh
#     dataset replay  run_x2_replay_stack.sh
#     pkl-driven      run_x2_pkl_planner_stack.sh        (this file)
#
# Lightweight topology (no recorder, no manager):
#
#                       ┌────────────────────────┐
#                       │  x2_pkl_command_source │  (this script's job)
#                       │  reads PKL, computes   │
#                       │  per-frame velocity    │
#                       └────────────┬───────────┘
#                          planner_cmd│ 5563
#                                     ▼
#                       ┌────────────────────────┐
#                       │      x2_kplanner       │
#                       │  VQVAE + pose + root   │
#                       └────────────┬───────────┘
#                                    │ pose (5556)
#                                    ▼
#                       ┌────────────────────────┐
#                       │   deploy_x2.sh sim     │
#                       │   SONIC + MuJoCo       │
#                       └────────────────────────┘
#
# The kplanner publishes ``pose`` (NOT ``body_pose``) directly to the
# deploy on :5556 to skip the recorder dependency for the L2 isolation
# test. The deploy must therefore be spawned with the same
# ``--disable-pose-ref-watchdog`` flag the production wrapper uses for
# local-sim runs (see run_x2_quest3_planner_stack.sh's pose-ref watchdog
# rationale).
#
# Usage:
#   gear_sonic/scripts/run_x2_pkl_planner_stack.sh
#       --pkl PATH
#       [--clip-id ID] [--start-frame N] [--num-frames N] [--loop]
#       [--duration N] [--rate-hz HZ]
#       [--with-deploy | --no-deploy]
#       [--no-sim-viewer]
#       [--sim-profile {parity,manual}]
#       [--kplanner-vqvae-ckpt PATH] [--kplanner-pose-ckpt PATH]
#       [--kplanner-root-ckpt PATH] [--kplanner-warmup-qpos PATH]
#       [--kplanner-device DEV] [--kplanner-replan-threshold-frames N]
#       [--kplanner-python PATH]
#       [--model PATH]
#       [--velocity-window N]
#       [--with-capture] [--capture-out DIR]
#       [--no-pose-feedback] [--pose-feedback-host H] [--pose-feedback-port P]
#       [--pose-feedback-topic T] [--pose-feedback-max-age-s S]
#       [--cleanup-only]
#       [--log-dir PATH]
#
# Examples:
#   # L2 isolation: replay a forward-walk clip through kplanner -> deploy.
#   ./run_x2_pkl_planner_stack.sh \
#       --pkl gear_sonic/data/motions/x2_ultra_locowalk.pkl \
#       --clip-id Loop_Forward_Walk_001__A018 \
#       --duration 30 --loop
#
#   # L2 headless variant (no MuJoCo window):
#   ./run_x2_pkl_planner_stack.sh \
#       --pkl gear_sonic/data/motions/x2_ultra_locowalk.pkl \
#       --no-sim-viewer --duration 30
#
#   # L2 with motion capture + auto-comparison: subscribes to the
#   # bridge's robot_pose PUB (5570) + the deploy's x2_debug PUB (5557)
#   # for the duration of the run, then prints a sim-vs-pkl table:
#   ./run_x2_pkl_planner_stack.sh \
#       --pkl gear_sonic/data/motions/x2_ultra_locowalk.pkl \
#       --clip-id Loop_Forward_Walk_001__A018 \
#       --duration 30 --loop --with-capture
#
#   # Cleanup orphaned children from a previous run:
#   ./run_x2_pkl_planner_stack.sh --cleanup-only

set -u
set -o pipefail

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_SH="${REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi
# Same Blackwell-aware override pattern as run_x2_quest3_planner_stack.sh.
KPLANNER_PYTHON="${KPLANNER_PYTHON:-${PYTHON}}"

PLANNER_PID_FILE="/tmp/x2_kplanner.pid"

# --------------------------------------------------------------------------
# Port + topic contract. Mirrors run_x2_quest3_planner_stack.sh so the
# C++ deploy needs no special configuration to talk to this stack.
# --------------------------------------------------------------------------

POSE_PORT=5556          # kplanner PUB (pose topic) -> deploy SUB
POSE_TOPIC="pose"
DEBUG_PORT=5557         # deploy PUB -> (unused here; reserved)
DEBUG_TOPIC="x2_debug"
PLANNER_CMD_PORT=5563   # pkl_command_source PUB -> kplanner SUB
PLANNER_CMD_TOPIC="planner_cmd"

# --------------------------------------------------------------------------
# CLI defaults
# --------------------------------------------------------------------------

PKL=""
CLIP_ID=""
START_FRAME=0
NUM_FRAMES=""
LOOP=0
DURATION_S=0           # 0 = unlimited
RATE_HZ=50
WITH_DEPLOY=1
SIM_VIEWER=1
SIM_PROFILE="parity"
VELOCITY_WINDOW=8
LOG_DIR=""
CLEANUP_ONLY=0
WITH_CAPTURE=0
CAPTURE_OUT=""
CONSTANT_INTENT=""
USE_MEAN_INTENT=0

# Closed-loop pose feedback: ON by default. The kplanner subscribes to
# the sim bridge's robot_pose:5570 PUB and reseeds its 4-frame root
# context from the robot's actually-observed pelvis pose just before
# each replan. Pass --no-pose-feedback to fall back to the open-loop
# behaviour (regression-test the diagnostic baseline).
WITH_POSE_FEEDBACK=1
POSE_FEEDBACK_HOST="127.0.0.1"
POSE_FEEDBACK_PORT="5570"
POSE_FEEDBACK_TOPIC="robot_pose"
POSE_FEEDBACK_MAX_AGE_S="0.5"
# 'full_root' overwrites xyz + quat (default). 'quat_only' overwrites
# just the quat -- preserves the planner's xy overshoot which the
# diagnostic runs showed actually HELPS forward tracking.
POSE_RESEED_SCOPE="full_root"

KPLANNER_VQVAE_CKPT=""
KPLANNER_POSE_CKPT=""
KPLANNER_ROOT_CKPT=""
KPLANNER_WARMUP_QPOS=""
KPLANNER_DEVICE="cuda"
KPLANNER_REPLAN_THRESHOLD_FRAMES="16"
KPLANNER_YAW_LOCK_EPSILON="0.0"
# Cold-start velocity ramp time constant (s). PKL replay uses
# constant-intent / mean-intent which already produces a steady
# velocity, so the ramp matters less here than for Quest 3 -- but
# leaving it accessible lets capture sweeps quantify the effect
# without re-editing x2_kplanner.py. Default empty -> daemon default
# (0.20 s); set to 0 to bypass the ramp entirely.
KPLANNER_COLD_START_RAMP_TAU_S=""

WARMUP_QUIET_STAND_S="2.0"
SIM_RSI_PKL="${REPO_ROOT}/data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_kplanner_rsi_anchor.pkl"
SIM_MODEL="${X2_PLANNER_SMOKE_MODEL:-/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx}"
SIM_CAM_TRACK_BODY="pelvis"
SIM_CAM_DISTANCE="3.5"
SIM_CAM_ELEVATION="-12"
SIM_CAM_AZIMUTH="135"

usage() {
    awk '/^# Usage:/,/^[^#]/{ if ($0 ~ /^[^#]/) exit; sub(/^# ?/, ""); print }' "$0" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pkl) PKL="$2"; shift 2 ;;
        --clip-id) CLIP_ID="$2"; shift 2 ;;
        --start-frame) START_FRAME="$2"; shift 2 ;;
        --num-frames) NUM_FRAMES="$2"; shift 2 ;;
        --loop) LOOP=1; shift ;;
        --duration) DURATION_S="$2"; shift 2 ;;
        --rate-hz) RATE_HZ="$2"; shift 2 ;;
        --velocity-window) VELOCITY_WINDOW="$2"; shift 2 ;;
        --no-deploy) WITH_DEPLOY=0; shift ;;
        --with-deploy) WITH_DEPLOY=1; shift ;;
        --no-sim-viewer) SIM_VIEWER=0; shift ;;
        --sim-profile) SIM_PROFILE="$2"; shift 2 ;;
        --sim-rsi-pkl) SIM_RSI_PKL="$2"; shift 2 ;;
        --kplanner-vqvae-ckpt) KPLANNER_VQVAE_CKPT="$2"; shift 2 ;;
        --kplanner-pose-ckpt)  KPLANNER_POSE_CKPT="$2";  shift 2 ;;
        --kplanner-root-ckpt)  KPLANNER_ROOT_CKPT="$2";  shift 2 ;;
        --kplanner-warmup-qpos) KPLANNER_WARMUP_QPOS="$2"; shift 2 ;;
        --kplanner-device) KPLANNER_DEVICE="$2"; shift 2 ;;
        --kplanner-replan-threshold-frames) KPLANNER_REPLAN_THRESHOLD_FRAMES="$2"; shift 2 ;;
        --kplanner-yaw-lock-epsilon) KPLANNER_YAW_LOCK_EPSILON="$2"; shift 2 ;;
        --kplanner-cold-start-ramp-tau-s) KPLANNER_COLD_START_RAMP_TAU_S="$2"; shift 2 ;;
        --kplanner-python) KPLANNER_PYTHON="$2"; shift 2 ;;
        --model) SIM_MODEL="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --with-capture) WITH_CAPTURE=1; shift ;;
        --no-capture) WITH_CAPTURE=0; shift ;;
        --capture-out) CAPTURE_OUT="$2"; WITH_CAPTURE=1; shift 2 ;;
        --constant-intent) CONSTANT_INTENT="$2"; shift 2 ;;
        --use-mean-intent) USE_MEAN_INTENT=1; shift ;;
        --no-pose-feedback) WITH_POSE_FEEDBACK=0; shift ;;
        --with-pose-feedback) WITH_POSE_FEEDBACK=1; shift ;;
        --pose-feedback-host) POSE_FEEDBACK_HOST="$2"; shift 2 ;;
        --pose-feedback-port) POSE_FEEDBACK_PORT="$2"; shift 2 ;;
        --pose-feedback-topic) POSE_FEEDBACK_TOPIC="$2"; shift 2 ;;
        --pose-feedback-max-age-s) POSE_FEEDBACK_MAX_AGE_S="$2"; shift 2 ;;
        --pose-reseed-scope) POSE_RESEED_SCOPE="$2"; shift 2 ;;
        --cleanup-only) CLEANUP_ONLY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="/tmp/x2_pkl_planner_stack-$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"

if [[ "${WITH_CAPTURE}" -eq 1 && -z "${CAPTURE_OUT}" ]]; then
    CAPTURE_OUT="${LOG_DIR}/capture"
fi

# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_RESET=$'\e[0m'
log()  { printf '%s[pkl-stack %s]%s %s\n' "${C_GREEN}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
warn() { printf '%s[pkl-stack %s WARN]%s %s\n' "${C_YELLOW}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
err()  { printf '%s[pkl-stack %s ERROR]%s %s\n' "${C_RED}" "$(date +%H:%M:%S)" "${C_RESET}" "$*" >&2; }

# --------------------------------------------------------------------------
# Process / port helpers (forked from run_x2_quest3_planner_stack.sh)
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
    log "cleanup_stale: PID files=${PLANNER_PID_FILE} ports=${POSE_PORT},${PLANNER_CMD_PORT}"
    if [[ -f "${PLANNER_PID_FILE}" ]]; then
        local stale_pid
        stale_pid="$(cat "${PLANNER_PID_FILE}" 2>/dev/null || true)"
        kill_pid_quiet "${stale_pid}" "stale kplanner (${PLANNER_PID_FILE})"
        rm -f "${PLANNER_PID_FILE}"
    fi
    free_port "${POSE_PORT}"
    free_port "${PLANNER_CMD_PORT}"
    # Sweep stale x2sim docker containers (forked verbatim).
    preflight_docker_cleanup
}

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
# Cleanup trap (set BEFORE any child is spawned). Reverse-order shutdown
# (pkl-source -> kplanner -> deploy) mirrors the spawn order so the
# deploy never sees a half-baked planner_cmd stream during teardown.
# --------------------------------------------------------------------------

DEPLOY_PID=""
DEPLOY_LOG=""
PLANNER_PID=""
PKL_SOURCE_PID=""
CAPTURE_PID=""

cleanup_children() {
    log "shutting down children (reverse spawn order)..."
    kill_pid_quiet "${PKL_SOURCE_PID}" "pkl-command-source"
    kill_pid_quiet "${PLANNER_PID}"     "kplanner"
    # The capture side-car finishes on SIGTERM by closing its sockets
    # and emitting its comparison. SIGINT gives it a chance to write
    # the NPZ + JSON before exit.
    if [[ -n "${CAPTURE_PID}" ]] && kill -0 "${CAPTURE_PID}" 2>/dev/null; then
        log "  SIGINT capture side-car (pid=${CAPTURE_PID})"
        kill -INT "${CAPTURE_PID}" 2>/dev/null || true
        local i=0
        while (( i < 20 )); do
            kill -0 "${CAPTURE_PID}" 2>/dev/null || break
            sleep 0.5
            i=$((i + 1))
        done
        kill_pid_quiet "${CAPTURE_PID}" "capture"
    fi
    if [[ -n "${DEPLOY_PID}" ]] && kill -0 "${DEPLOY_PID}" 2>/dev/null; then
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

if [[ -z "${PKL}" ]]; then
    err "--pkl PATH is required."
    err "Examples (forward-walk clips):"
    err "  --pkl gear_sonic/data/motions/x2_ultra_locowalk.pkl --clip-id Loop_Forward_Walk_001__A018"
    err "  --pkl gear_sonic/data/motions/x2_ultra_casual_walk_v2.pkl --clip-id casual_walk__v2__tight_cycle"
    exit 1
fi
if [[ ! -f "${PKL}" ]]; then
    err "--pkl not found: ${PKL}"
    exit 1
fi

# kplanner checkpoint preflight (mirrors run_x2_quest3_planner_stack.sh).
KPL_VQVAE_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0500000.ckpt"
KPL_POSE_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/model-step=0500000.ckpt"
KPL_ROOT_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0300000.ckpt"
for ck in "${KPLANNER_VQVAE_CKPT:-${KPL_VQVAE_DEFAULT}}" \
          "${KPLANNER_POSE_CKPT:-${KPL_POSE_DEFAULT}}" \
          "${KPLANNER_ROOT_CKPT:-${KPL_ROOT_DEFAULT}}"; do
    if [[ ! -f "${ck}" ]]; then
        err "kplanner checkpoint not found: ${ck}"
        err "Override via --kplanner-{vqvae,pose,root}-ckpt PATH."
        exit 1
    fi
done

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

# Sweep stale containers BEFORE probing DEBUG_PORT (forked rationale
# from run_x2_quest3_planner_stack.sh).
if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    preflight_docker_cleanup
fi

# Port pre-check.
for port in "${POSE_PORT}" "${PLANNER_CMD_PORT}"; do
    if port_in_use "${port}"; then
        err "port ${port} is in use. Run: $0 --cleanup-only"
        exit 1
    fi
done
if [[ "${WITH_DEPLOY}" -eq 1 ]] && port_in_use "${DEBUG_PORT}"; then
    err "port ${DEBUG_PORT} (deploy x2_debug PUB) is in use."
    err "Run: $0 --cleanup-only or shut down the existing deploy first."
    exit 1
fi

# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------

cat <<EOF
${C_GREEN}┌──────────────────────────────────────────────────────────────────────┐
│  X2 PKL-driven planner stack runner                                  │
│  (kplanner -> deploy; no manager, no recorder, no Quest 3)           │
└──────────────────────────────────────────────────────────────────────┘${C_RESET}
  log dir          : ${LOG_DIR}
  duration         : $([[ "${DURATION_S}" -eq 0 ]] && echo "unlimited (run until Ctrl-C)" || echo "${DURATION_S}s")
  pkl              : ${PKL}
  clip-id          : ${CLIP_ID:-(auto-pick forward-walk clip)}
  start-frame      : ${START_FRAME}
  num-frames       : ${NUM_FRAMES:-(all remaining)}
  loop             : $([[ "${LOOP}" -eq 1 ]] && echo on || echo off)
  rate             : ${RATE_HZ} Hz
  velocity-window  : ${VELOCITY_WINDOW} frames
  deploy           : $([[ "${WITH_DEPLOY}" -eq 1 ]] && echo "ON  (sim --vla, profile=${SIM_PROFILE}, viewer=$([[ "${SIM_VIEWER}" -eq 1 ]] && echo on || echo off))" || echo "OFF (assume external deploy SUBs pose:${POSE_PORT})")
  kplanner device  : ${KPLANNER_DEVICE}  python: ${KPLANNER_PYTHON}
  ports            : pose=${POSE_PORT}  x2_debug=${DEBUG_PORT}  planner_cmd=${PLANNER_CMD_PORT}
  capture          : $([[ "${WITH_CAPTURE}" -eq 1 ]] && echo "ON  -> ${CAPTURE_OUT}" || echo "OFF (pass --with-capture to record sim motion + compare to pkl)")
  pose feedback    : $([[ "${WITH_POSE_FEEDBACK}" -eq 1 ]] && echo "ON  (closed-loop reseed scope=${POSE_RESEED_SCOPE} from ${POSE_FEEDBACK_HOST}:${POSE_FEEDBACK_PORT}/${POSE_FEEDBACK_TOPIC}, max_age=${POSE_FEEDBACK_MAX_AGE_S}s)" || echo "OFF (open-loop baseline; pass --no-pose-feedback to keep this)")
EOF
echo

# --------------------------------------------------------------------------
# Step 1/3 — Deploy (docker sim + ONNX). Mirrors the quest3 wrapper's
# Step 1, except we pass --disable-pose-ref-watchdog because the
# kplanner publishes directly to pose:5556 (no recorder merging idle
# frames during cold-start) and the watchdog otherwise trips during
# the planner's 10-15 s model load.
# --------------------------------------------------------------------------

if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_LOG="${LOG_DIR}/deploy.log"
    if (( DURATION_S > 0 )); then
        DEPLOY_DURATION_S=$(( DURATION_S + 30 ))
    else
        DEPLOY_DURATION_S=0
    fi

    # Bake kplanner RSI anchor PKL so the deploy spawns at the same pose
    # the kplanner's first publish tick will emit. Always re-bake (it's
    # <300 ms) so changes to --kplanner-warmup-qpos take effect.
    if [[ "${SIM_PROFILE}" == "parity" ]]; then
        log "baking kplanner RSI anchor PKL -> ${SIM_RSI_PKL}"
        KPL_BAKE_ARGS=(-m gear_sonic.scripts.bake_kplanner_rsi_anchor
                       --out "${SIM_RSI_PKL}")
        if [[ -n "${KPLANNER_WARMUP_QPOS}" ]]; then
            KPL_BAKE_ARGS+=(--warmup-qpos-path "${KPLANNER_WARMUP_QPOS}")
        fi
        if ! "${PYTHON}" "${KPL_BAKE_ARGS[@]}" \
                >>"${LOG_DIR}/rsi_anchor_bake.log" 2>&1; then
            err "failed to bake kplanner RSI anchor; see ${LOG_DIR}/rsi_anchor_bake.log"
            exit 1
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
        --wrist-bypass ik
        --model "${SIM_MODEL}"
        --autostart-after 0
        # Pose-ref watchdog OFF for local sim: the kplanner takes 10-15 s
        # to load and the watchdog's 0.5 s SAFE_IDLE trip would collapse
        # the robot during cold-start. Forked rationale from
        # run_x2_quest3_planner_stack.sh.
        --deploy-extra-arg --disable-pose-ref-watchdog
    )
    if (( DEPLOY_DURATION_S > 0 )); then
        DEPLOY_ARGS+=(--max-duration "${DEPLOY_DURATION_S}")
    fi
    if [[ "${SIM_PROFILE}" == "parity" ]]; then
        DEPLOY_ARGS+=(--motion "${SIM_RSI_PKL}")
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

    log "Step 1/3 — spawning deploy_x2.sh sim --vla -> ${DEPLOY_LOG}"
    "${DEPLOY_SH}" "${DEPLOY_ARGS[@]}" >"${DEPLOY_LOG}" 2>&1 &
    DEPLOY_PID=$!

    log "  waiting for deploy 'Launching ...' marker (up to 180s)..."
    if ! wait_for_log_marker "${DEPLOY_LOG}" "${DEPLOY_PID}" "Launching ..." 180 "deploy"; then
        exit 1
    fi
    log "  deploy READY (pid=${DEPLOY_PID}); settle 2s before kplanner ..."
    sleep 2.0
else
    log "Step 1/3 — deploy spawn SKIPPED (--no-deploy). External deploy must SUB pose on :${POSE_PORT}."
fi

# --------------------------------------------------------------------------
# Step 1b/3 — Optional motion capture side-car. Started AFTER deploy comes
# up (so the bridge's robot_pose PUB:5570 + x2_debug:5557 are bound) and
# BEFORE the kplanner so we don't miss any pose samples. Subscribes to
# both, dumps NPZ, then prints sim-vs-pkl comparison at the end.
# --------------------------------------------------------------------------

if [[ "${WITH_CAPTURE}" -eq 1 ]]; then
    mkdir -p "${CAPTURE_OUT}"
    CAPTURE_LOG="${LOG_DIR}/capture.log"
    CAPTURE_ARGS=(
        "${SCRIPT_DIR}/capture_pkl_replay_motion.py"
        --pkl "${PKL}"
        --start-frame "${START_FRAME}"
        --velocity-window "${VELOCITY_WINDOW}"
        --pose-host 127.0.0.1
        --pose-port 5570
        --debug-host 127.0.0.1
        --debug-port "${DEBUG_PORT}"
        --debug-topic "${DEBUG_TOPIC}"
        --output-dir "${CAPTURE_OUT}"
    )
    if [[ -n "${CLIP_ID}" ]]; then
        CAPTURE_ARGS+=(--clip-id "${CLIP_ID}")
    fi
    if [[ -n "${NUM_FRAMES}" ]]; then
        CAPTURE_ARGS+=(--num-frames "${NUM_FRAMES}")
    fi
    if [[ "${LOOP}" -eq 1 ]]; then
        CAPTURE_ARGS+=(--loop)
    fi
    # Give the capture a bit more wall-clock than the source so we
    # observe the final ~1s of robot settling after the pkl source
    # drains. With --duration 0 the capture runs until SIGINT from
    # cleanup_children.
    if (( DURATION_S > 0 )); then
        CAPTURE_ARGS+=(--duration "$(( DURATION_S + 5 ))")
    fi
    log "Step 1b/3 — spawning capture side-car -> ${CAPTURE_LOG}"
    "${PYTHON}" "${CAPTURE_ARGS[@]}" >"${CAPTURE_LOG}" 2>&1 &
    CAPTURE_PID=$!
    log "  capture pid=${CAPTURE_PID}; waiting 1s for SUB sockets to connect..."
    sleep 1.0
    if ! kill -0 "${CAPTURE_PID}" 2>/dev/null; then
        err "capture side-car died during startup; tail of ${CAPTURE_LOG}:"
        tail -n 40 "${CAPTURE_LOG}" >&2 || true
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# Step 2/3 — kplanner. Publishes ``pose`` topic (NOT ``body_pose``)
# directly to the deploy on :5556 -- there is no recorder to merge.
# SUB ``planner_cmd`` on :5563 from the pkl_command_source.
# --------------------------------------------------------------------------

PLANNER_LOG="${LOG_DIR}/planner.log"

PLANNER_ARGS=(
    -m gear_sonic.scripts.x2_kplanner
    --pub-host 127.0.0.1
    --pub-port "${POSE_PORT}"
    --zmq-cmd-host localhost
    --zmq-cmd-port "${PLANNER_CMD_PORT}"
    --zmq-cmd-topic "${PLANNER_CMD_TOPIC}"
    --warmup-quiet-stand-s "${WARMUP_QUIET_STAND_S}"
    --pid-file "${PLANNER_PID_FILE}"
    --duration-s "${DURATION_S}"
    --device "${KPLANNER_DEVICE}"
    --replan-threshold-frames "${KPLANNER_REPLAN_THRESHOLD_FRAMES}"
    --yaw-lock-epsilon "${KPLANNER_YAW_LOCK_EPSILON}"
)
if [[ -n "${KPLANNER_COLD_START_RAMP_TAU_S}" ]]; then
    PLANNER_ARGS+=(--cold-start-ramp-tau-s "${KPLANNER_COLD_START_RAMP_TAU_S}")
fi
if [[ "${WITH_POSE_FEEDBACK}" -eq 1 ]]; then
    PLANNER_ARGS+=(
        --pose-feedback-host "${POSE_FEEDBACK_HOST}"
        --pose-feedback-port "${POSE_FEEDBACK_PORT}"
        --pose-feedback-topic "${POSE_FEEDBACK_TOPIC}"
        --pose-feedback-max-age-s "${POSE_FEEDBACK_MAX_AGE_S}"
        --pose-reseed-scope "${POSE_RESEED_SCOPE}"
    )
fi
if [[ -n "${KPLANNER_VQVAE_CKPT}" ]]; then
    PLANNER_ARGS+=(--vqvae-ckpt "${KPLANNER_VQVAE_CKPT}")
fi
if [[ -n "${KPLANNER_POSE_CKPT}" ]]; then
    PLANNER_ARGS+=(--pose-ckpt "${KPLANNER_POSE_CKPT}")
fi
if [[ -n "${KPLANNER_ROOT_CKPT}" ]]; then
    PLANNER_ARGS+=(--root-ckpt "${KPLANNER_ROOT_CKPT}")
fi
if [[ -n "${KPLANNER_WARMUP_QPOS}" ]]; then
    PLANNER_ARGS+=(--warmup-qpos-path "${KPLANNER_WARMUP_QPOS}")
fi

log "Step 2/3 — spawning x2_kplanner -> ${PLANNER_LOG}"
log "  using python: ${KPLANNER_PYTHON}"
# Inject PYTHONPATH so motionbricks imports work regardless of which
# interpreter ${KPLANNER_PYTHON} resolves to. Same pattern as the
# quest3 wrapper.
KPLANNER_PYTHONPATH="${REPO_ROOT}/motionbricks:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
log "  PYTHONPATH=${KPLANNER_PYTHONPATH}"
PYTHONPATH="${KPLANNER_PYTHONPATH}" \
    "${KPLANNER_PYTHON}" "${PLANNER_ARGS[@]}" >"${PLANNER_LOG}" 2>&1 &
PLANNER_PID=$!

log "  waiting for kplanner ready marker (up to 90s; first replan is ~10s)..."
if ! wait_for_log_marker "${PLANNER_LOG}" "${PLANNER_PID}" \
        "first replan complete" 90 "kplanner"; then
    exit 1
fi
log "  kplanner READY (pid=${PLANNER_PID}); settle 0.5s before pkl source ..."
sleep 0.5

# --------------------------------------------------------------------------
# Step 3/3 — PKL command source. PUBs ``planner_cmd`` on :5563 carrying
# the PKL's recorded per-frame velocity in the ``target_velocity`` JSON
# field. The kplanner short-circuits its intent dispatcher on that
# field and feeds the raw velocity into the neural model.
# --------------------------------------------------------------------------

PKL_SOURCE_LOG="${LOG_DIR}/pkl_source.log"
PKL_SOURCE_ARGS=(
    "${SCRIPT_DIR}/x2_pkl_command_source.py"
    --pkl "${PKL}"
    --bind "tcp://*:${PLANNER_CMD_PORT}"
    --topic "${PLANNER_CMD_TOPIC}"
    --rate-hz "${RATE_HZ}"
    --start-frame "${START_FRAME}"
    --velocity-window "${VELOCITY_WINDOW}"
    --print-every 25
)
if [[ -n "${CLIP_ID}" ]]; then
    PKL_SOURCE_ARGS+=(--clip-id "${CLIP_ID}")
fi
if [[ -n "${NUM_FRAMES}" ]]; then
    PKL_SOURCE_ARGS+=(--num-frames "${NUM_FRAMES}")
fi
if [[ "${LOOP}" -eq 1 ]]; then
    PKL_SOURCE_ARGS+=(--loop)
fi
if [[ -n "${CONSTANT_INTENT}" ]]; then
    PKL_SOURCE_ARGS+=(--constant-intent "${CONSTANT_INTENT}")
fi
if [[ "${USE_MEAN_INTENT}" -eq 1 ]]; then
    PKL_SOURCE_ARGS+=(--use-mean-intent)
fi

log "Step 3/3 — spawning x2_pkl_command_source -> ${PKL_SOURCE_LOG}"
"${PYTHON}" "${PKL_SOURCE_ARGS[@]}" >"${PKL_SOURCE_LOG}" 2>&1 &
PKL_SOURCE_PID=$!

log "  waiting for pkl source 'ZMQ PUB bound' marker (up to 30s)..."
if ! wait_for_log_marker "${PKL_SOURCE_LOG}" "${PKL_SOURCE_PID}" \
        "ZMQ PUB bound" 30 "pkl-command-source"; then
    exit 1
fi
log "  pkl source READY (pid=${PKL_SOURCE_PID}); the stack is live."

# --------------------------------------------------------------------------
# Main wait: block until duration elapses OR a child dies OR Ctrl-C.
# Mirrors the quest3 wrapper's wait loop, just over the lighter child
# set (no manager / recorder).
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Helpers for distinguishing a planned end-of-duration exit from a real
# crash. The kplanner and deploy are both spawned with ``--duration-s
# ${DURATION_S}``, so when ${DURATION_S} > 0 the EXPECTED behaviour is
# that they exit at the ${DURATION_S} mark on their own clock --
# typically a few seconds BEFORE the wrapper's wall-clock deadline,
# because the wrapper's deadline only starts after the
# deploy/kplanner/pkl-source spawn sequence finishes (5-15s of startup
# overhead). Without this check the wrapper used to mislabel that
# normal early exit as ``ERROR kplanner died`` even though everything
# ran successfully and the capture script produced its full verdict.
#
# These helpers harvest the child's actual exit code via ``wait`` (only
# works for direct children of this shell, which all of our & spawns
# are) and report success vs crash distinctly.

# Returns 0 if pid is still alive, 1 if it exited cleanly, 2 if it died with
# nonzero rc. Sets ``__exit_rc`` to the captured exit code when applicable.
# Note: ``$?`` after ``if cmd; then ... fi`` is reset to 0 by bash, so we
# must read ``$?`` immediately after the bare ``wait`` call -- not after
# any ``if`` test on it.
__exit_rc=""
check_child_alive_or_finished() {
    local pid="$1"
    if kill -0 "${pid}" 2>/dev/null; then
        __exit_rc=""
        return 0
    fi
    # Already exited. Reap it to get the actual rc. ``wait`` on a PID
    # that's no longer a known child of this shell returns 127; treat
    # that the same as "exited cleanly, code unknown" rather than as
    # a crash so we don't false-alarm on already-reaped children.
    wait "${pid}" 2>/dev/null
    local rc=$?
    if (( rc == 127 )); then
        __exit_rc=0
        return 1
    fi
    __exit_rc="${rc}"
    if (( rc == 0 )); then
        return 1
    fi
    return 2
}

main_wait_loop() {
    local end_ts="$1"  # 0 = run forever; non-zero = unix timestamp deadline
    local check_rc
    while :; do
        if (( end_ts > 0 )) && (( $(date +%s) >= end_ts )); then
            log "duration ${DURATION_S}s elapsed; shutting down."
            return 0
        fi

        if [[ -n "${DEPLOY_PID}" ]]; then
            check_child_alive_or_finished "${DEPLOY_PID}"
            check_rc=$?
            if (( check_rc == 1 )) && (( DURATION_S > 0 )); then
                log "deploy finished normally (pid=${DEPLOY_PID}, exit=${__exit_rc:-0}, --duration-s ${DURATION_S} elapsed); ending session."
                return 0
            elif (( check_rc == 2 )); then
                err "deploy died (pid=${DEPLOY_PID}, exit=${__exit_rc}); see ${DEPLOY_LOG}"
                return 1
            fi
        fi

        check_child_alive_or_finished "${PLANNER_PID}"
        check_rc=$?
        if (( check_rc == 1 )) && (( DURATION_S > 0 )); then
            log "kplanner finished normally (pid=${PLANNER_PID}, exit=${__exit_rc:-0}, --duration-s ${DURATION_S} elapsed); ending session."
            return 0
        elif (( check_rc == 2 )); then
            err "kplanner died (pid=${PLANNER_PID}, exit=${__exit_rc}); see ${PLANNER_LOG}"
            return 1
        fi

        check_child_alive_or_finished "${PKL_SOURCE_PID}"
        check_rc=$?
        if (( check_rc == 1 )); then
            # Clean exit. Either --loop is off (one pass done) or the
            # downstream kplanner ended its session and the source
            # noticed -- either way, recording is complete.
            log "pkl source finished normally (pid=${PKL_SOURCE_PID}, exit=${__exit_rc:-0}); ending session."
            return 0
        elif (( check_rc == 2 )); then
            err "pkl source died unexpectedly (pid=${PKL_SOURCE_PID}, exit=${__exit_rc}); see ${PKL_SOURCE_LOG}"
            return 1
        fi

        # Capture side-car may exit before us if its --duration hit
        # (we asked it to capture DURATION_S+5). Treat that as a normal
        # end-of-recording, not a failure.
        if [[ -n "${CAPTURE_PID}" ]]; then
            check_child_alive_or_finished "${CAPTURE_PID}"
            check_rc=$?
            if (( check_rc != 0 )); then
                log "capture side-car exited (recording complete, exit=${__exit_rc:-0}); see ${CAPTURE_LOG}"
                CAPTURE_PID=""
            fi
        fi

        sleep 1
    done
}

if (( DURATION_S > 0 )); then
    log "running for ${DURATION_S}s (Ctrl-C to stop early) ..."
    end_ts=$(( $(date +%s) + DURATION_S ))
    main_wait_loop "${end_ts}"
else
    log "running forever (Ctrl-C to stop) ..."
    main_wait_loop 0
fi
WAIT_LOOP_RC=$?

# The cleanup_children trap will fire on exit; just propagate the wait
# loop's rc as the script's overall rc so callers can shell-test for
# success vs failure.
exit "${WAIT_LOOP_RC}"
