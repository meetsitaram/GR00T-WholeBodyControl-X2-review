#!/usr/bin/env bash
# X2 dataset-replay stack runner.
#
# Sister wrapper of ``run_x2_pkl_planner_stack.sh`` and
# ``run_x2_vla_runtime.sh``: same single-shell ergonomics, but the
# foreground client is :mod:`gear_sonic.scripts.replay_x2_dataset`
# (a recorded LeRobot v2.1 episode) instead of the kplanner or the
# VLA bridge. Use this to validate end-to-end through SONIC -> motors
# without a headset or a policy in the loop:
#
#     teleop          run_x2_quest3_planner_stack.sh
#     pkl-driven      run_x2_pkl_planner_stack.sh
#     autonomous VLA  run_x2_vla_runtime.sh
#     dataset replay  run_x2_replay_stack.sh        (this file)
#
# Lightweight topology (no manager, no recorder, no planner, no policy):
#
#                       +-----------------------+
#                       | replay_x2_dataset.py  |  reads parquet,
#                       | (this script's job)   |  PUBs joint_pos_mj
#                       |                       |  + v5 future window
#                       +-----------+-----------+
#                                   | pose (5556)
#                                   v
#                       +-----------------------+
#                       |    deploy_x2.sh sim   |  SONIC + MuJoCo,
#                       |        (optional)     |  or assume PC2 deploy
#                       +-----------------------+
#
# The replayer publishes the v5-promoted ``pose`` envelope (current
# joint_pos_mj + 9-slot future window) directly to ``:5556``, so the
# deploy must be spawned with ``--disable-pose-ref-watchdog`` for the
# local-sim case (same rationale as the pkl wrapper: there is no
# upstream merge process, so the cold-start gap before warm-up trips
# the 0.5 s SAFE_IDLE watchdog otherwise).
#
# Three modes:
#
#   * Sim (default)        - spawns ``deploy_x2.sh sim --vla --sim-with-omnihand``
#                            on localhost, then runs the replayer against it.
#                            Safe place to validate the parquet + v5 wiring
#                            before powering the real robot.
#
#   * Real robot           - pass ``--pc2-host <PC2_IP>``. Skips spawning a
#                            sim deploy; assumes ``x2_pc2_daemons.sh start``
#                            is already running on PC2. The replayer's PUB
#                            binds locally and PC2's pose proxy connects out
#                            to it. The host is informational on the wire
#                            (replay logs it in its banner).
#
#   * External deploy      - pass ``--no-deploy``. Like real-robot mode but
#                            without the PC2_HOST annotation; useful when
#                            you brought up your own deploy in another shell
#                            and just want this wrapper to manage the
#                            replayer's lifecycle.
#
# Usage:
#   gear_sonic/scripts/run_x2_replay_stack.sh
#       --dataset NAME_OR_PATH
#       [--episode N]
#       [--rate HZ] [--rate-scale FLOAT] [--loop]
#       [--countdown S] [--hold-on-exit S]
#       [--with-deploy | --no-deploy | --pc2-host HOST]
#       [--no-sim-viewer] [--sim-profile {handoff,parity,manual}]
#       [--model PATH]
#       [--with-rerun]
#       [--duration N]
#       [--cleanup-only]
#       [--log-dir PATH]
#
# Examples:
#   # Sim smoke: bring up sim deploy + replay episode 0 of a recording.
#   ./run_x2_replay_stack.sh --dataset x2_reach_and_retract_v1 --episode 0
#
#   # Sim smoke + recorded-camera viewer side-by-side. The rerun GUI
#   # shows the operator's original cameras + recorded body trajectory;
#   # the MuJoCo viewer shows the live deploy replaying the same wire.
#   ./run_x2_replay_stack.sh --dataset x2_reach_and_retract_v1 --episode 0 \
#       --with-rerun
#
#   # Loop a recording forever so you can debug the deploy / sim viewer.
#   ./run_x2_replay_stack.sh --dataset x2_reach_and_retract_v1 --episode 0 \
#       --loop
#
#   # Half-speed first pass on the real robot (operator has e-stop in
#   # reach; PC2 daemons started separately via x2_pc2_daemons.sh start).
#   # --with-rerun lets you eyeball the recorded cameras while the real
#   # robot physically replays the trajectory.
#   ./run_x2_replay_stack.sh --dataset x2_reach_and_retract_v1 --episode 0 \
#       --pc2-host 192.168.86.32 --rate-scale 0.5 --with-rerun
#
#   # Sanity-check args + load the parquet without binding ZMQ.
#   ./run_x2_replay_stack.sh --dataset x2_reach_and_retract_v1 --episode 0 \
#       --no-deploy --dry-run
#
#   # Free ports + kill orphan deploy/replay processes from a crashed run.
#   ./run_x2_replay_stack.sh --cleanup-only

set -u
set -o pipefail

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_SH="${REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh"
RERUN_SH="${SCRIPT_DIR}/view_x2_recorded_dataset.sh"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

# --------------------------------------------------------------------------
# Port + topic contract. Mirrors run_x2_pkl_planner_stack.sh so the C++
# deploy needs no special configuration to talk to this stack.
# --------------------------------------------------------------------------

POSE_PORT=5556          # replay PUB (pose topic) -> deploy SUB
POSE_TOPIC="pose"
DEBUG_PORT=5557         # deploy PUB -> (unused here; reserved)
DEBUG_TOPIC="x2_debug"

# --------------------------------------------------------------------------
# CLI defaults
# --------------------------------------------------------------------------

DATASET=""
EPISODE=0
RATE=""                 # "" = let replay use the dataset's native fps
RATE_SCALE=1.0
LOOP=0
COUNTDOWN=""            # "" = let replay use its default (3.0 s)
HOLD_ON_EXIT=""         # "" = let replay use its default (0.5 s)
DRY_RUN=0

WITH_DEPLOY=1
SIM_VIEWER=1
# ``handoff`` (not ``parity``): replay has no baked RSI motion PKL to
# pass via ``--motion``, and ``handoff`` is the documented "final sim
# gate before powered runs" profile -- bridge starts at DEFAULT_DOF
# (matches real-robot MC handoff), elastic band stays on through the
# soft-start ramp so the body cannot tip while the replay's
# ``--countdown`` warm-up transitions from default_angles to frame 0
# of the recording. ``parity`` would 502 here because it requires
# ``--motion`` for RSI; the pkl wrapper avoids that by baking a
# kplanner anchor PKL up-front, which is meaningless for dataset replay.
SIM_PROFILE="handoff"
SIM_MODEL="${X2_PLANNER_SMOKE_MODEL:-/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx}"
SIM_CAM_TRACK_BODY="pelvis"
SIM_CAM_DISTANCE="3.5"
SIM_CAM_ELEVATION="-12"
SIM_CAM_AZIMUTH="135"
PC2_HOST=""

# Rerun viewer for the recorded camera MP4s + scalar timeline (body_q,
# hand joints, wrist FK trace) of the SAME episode being replayed. Lets
# the operator eyeball-compare "what the operator originally saw + what
# the recorded body did" against the live sim/robot. The rerun GUI
# process is spawned by view_x2_recorded_dataset.py via rr.init(
# spawn=True) and OUTLIVES this wrapper -- intentional, so you can
# scrub the recording after the live run ends. Default off because
# the viewer needs the dedicated ``.venv-viewer/`` interpreter and
# adds ~5-30 s of cold-load time at startup.
WITH_RERUN=0

DURATION_S=0            # 0 = unlimited; replay has its own end-of-episode signal
LOG_DIR=""
CLEANUP_ONLY=0

usage() {
    awk '/^# Usage:/,/^[^#]/{ if ($0 ~ /^[^#]/) exit; sub(/^# ?/, ""); print }' "$0" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --episode) EPISODE="$2"; shift 2 ;;
        --rate) RATE="$2"; shift 2 ;;
        --rate-scale) RATE_SCALE="$2"; shift 2 ;;
        --loop) LOOP=1; shift ;;
        --countdown) COUNTDOWN="$2"; shift 2 ;;
        --hold-on-exit) HOLD_ON_EXIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --with-deploy) WITH_DEPLOY=1; PC2_HOST=""; shift ;;
        --no-deploy) WITH_DEPLOY=0; PC2_HOST=""; shift ;;
        --pc2-host) PC2_HOST="$2"; WITH_DEPLOY=0; shift 2 ;;
        --no-sim-viewer) SIM_VIEWER=0; shift ;;
        --sim-viewer) SIM_VIEWER=1; shift ;;
        --sim-profile) SIM_PROFILE="$2"; shift 2 ;;
        --model) SIM_MODEL="$2"; shift 2 ;;
        --with-rerun) WITH_RERUN=1; shift ;;
        --no-rerun) WITH_RERUN=0; shift ;;
        --duration) DURATION_S="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --cleanup-only) CLEANUP_ONLY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="/tmp/x2_replay_stack-$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# LAN isolation for the replay pose PUB.
#
# PC2's x2_pose_proxy is a long-lived daemon started once via
# ``x2_pc2_daemons.sh start --laptop-host <LAPTOP_IP> ...``. It SUBs
# ``<LAPTOP_IP>:${POSE_PORT}`` over wifi and stays connected across
# laptop sessions. ZMQ SUBs connect out, our PUB just binds and
# accepts whatever attaches.
#
# That means a sim replay that binds the pose PUB on ``*`` (all
# interfaces) silently delivers the wire to PC2 too. The real robot
# starts tracking the sim replay even though the operator never
# passed ``--pc2-host``. Mirrors the 2026-06-23 fix in
# ``run_x2_vla_runtime.sh`` (same root cause, same gating shape).
#
# Fix: gate the replay's pose PUB bind on PC2_HOST. Without
# ``--pc2-host``: bind loopback so the wire is unreachable from PC2.
# With ``--pc2-host``: bind '*' so the always-on PC2 pose proxy can
# attach. Override-able via ``PUB_BIND_HOST=*`` env for the rare
# cross-host sim-mode debug case.
if [[ -n "${PC2_HOST}" ]]; then
    : "${PUB_BIND_HOST:=*}"
else
    : "${PUB_BIND_HOST:=127.0.0.1}"
fi

# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_RESET=$'\e[0m'
log()  { printf '%s[replay-stack %s]%s %s\n' "${C_GREEN}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
warn() { printf '%s[replay-stack %s WARN]%s %s\n' "${C_YELLOW}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
err()  { printf '%s[replay-stack %s ERROR]%s %s\n' "${C_RED}" "$(date +%H:%M:%S)" "${C_RESET}" "$*" >&2; }

# --------------------------------------------------------------------------
# Process / port helpers (forked verbatim from run_x2_pkl_planner_stack.sh
# so behaviour stays in lockstep across the wrapper family).
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

cleanup_stale() {
    log "cleanup_stale: port=${POSE_PORT}"
    free_port "${POSE_PORT}"
    # No need to free DEBUG_PORT here: the replay client is pure PUB, only
    # the deploy binds DEBUG_PORT. Sweep stale x2sim docker containers so
    # the next run's deploy spawn isn't fighting an orphan.
    preflight_docker_cleanup
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
# Cleanup trap (set BEFORE any child is spawned). Reverse-order shutdown:
# replay first (so its hold_on_exit ramp-down completes against a still-
# alive deploy), THEN deploy. SIGINT (not TERM) to the replay so its
# Python SIGINT handler fires and the last frames hold the pose cleanly.
# --------------------------------------------------------------------------

DEPLOY_PID=""
DEPLOY_LOG=""
REPLAY_PID=""
REPLAY_LOG=""
RERUN_PID=""
RERUN_LOG=""

cleanup_children() {
    log "shutting down children (reverse spawn order)..."
    # Rerun viewer parent is short-lived (loads + flushes then exits);
    # usually already gone by teardown. Its spawned GUI is a separate
    # process by design and STAYS UP so the operator can scrub the
    # recording after the live run ends. We only reap our direct child.
    if [[ -n "${RERUN_PID}" ]] && kill -0 "${RERUN_PID}" 2>/dev/null; then
        log "  killing rerun loader parent (pid=${RERUN_PID}; GUI stays open)"
        kill_pid_quiet "${RERUN_PID}" "rerun loader"
    fi
    if [[ -n "${REPLAY_PID}" ]] && kill -0 "${REPLAY_PID}" 2>/dev/null; then
        log "  SIGINT replay (pid=${REPLAY_PID}) -- letting hold_on_exit ramp-down complete"
        kill -INT "${REPLAY_PID}" 2>/dev/null || true
        local i=0
        while (( i < 20 )); do
            kill -0 "${REPLAY_PID}" 2>/dev/null || break
            sleep 0.5
            i=$((i + 1))
        done
        kill_pid_quiet "${REPLAY_PID}" "replay"
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

if [[ -z "${DATASET}" ]]; then
    err "--dataset NAME_OR_PATH is required."
    err "Examples:"
    err "  --dataset x2_reach_and_retract_v1 --episode 0"
    err "  --dataset /abs/path/to/lerobot/dataset --episode 3"
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

if [[ "${WITH_RERUN}" -eq 1 ]]; then
    if [[ ! -x "${RERUN_SH}" ]]; then
        err "view_x2_recorded_dataset.sh not found / not executable: ${RERUN_SH}"
        exit 1
    fi
    # The wrapper checks ``.venv-viewer/bin/python`` itself and prints a
    # pointer to install_scripts/install_viewer.sh if missing; we let
    # it bail with that helpful message rather than re-implementing the
    # check here.
fi

# Sweep stale containers BEFORE probing DEBUG_PORT.
if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    preflight_docker_cleanup
fi

# Port pre-check.
if port_in_use "${POSE_PORT}"; then
    err "port ${POSE_PORT} (pose PUB) is in use. Run: $0 --cleanup-only"
    exit 1
fi
if [[ "${WITH_DEPLOY}" -eq 1 ]] && port_in_use "${DEBUG_PORT}"; then
    err "port ${DEBUG_PORT} (deploy x2_debug PUB) is in use."
    err "Run: $0 --cleanup-only or shut down the existing deploy first."
    exit 1
fi

# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------

if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_DESC="ON  (sim --vla, profile=${SIM_PROFILE}, viewer=$([[ "${SIM_VIEWER}" -eq 1 ]] && echo on || echo off))"
elif [[ -n "${PC2_HOST}" ]]; then
    DEPLOY_DESC="OFF -- real-robot mode (PC2 deploy at ${PC2_HOST} SUBs pose:${POSE_PORT})"
else
    DEPLOY_DESC="OFF -- assume external deploy SUBs pose:${POSE_PORT}"
fi

cat <<EOF
${C_GREEN}+----------------------------------------------------------------------+
|  X2 dataset-replay stack runner                                      |
|  (replay -> deploy; no planner, no manager, no recorder, no policy)  |
+----------------------------------------------------------------------+${C_RESET}
  log dir          : ${LOG_DIR}
  duration         : $([[ "${DURATION_S}" -eq 0 ]] && echo "unlimited (run until replay ends OR Ctrl-C)" || echo "${DURATION_S}s wall-clock cap")
  dataset          : ${DATASET}
  episode          : ${EPISODE}
  rate             : ${RATE:-(dataset native fps)} Hz  scale=${RATE_SCALE}
  loop             : $([[ "${LOOP}" -eq 1 ]] && echo on || echo off)
  countdown        : ${COUNTDOWN:-(replay default 3.0)} s
  hold-on-exit     : ${HOLD_ON_EXIT:-(replay default 0.5)} s
  dry-run          : $([[ "${DRY_RUN}" -eq 1 ]] && echo on || echo off)
  deploy           : ${DEPLOY_DESC}
  rerun viewer     : $([[ "${WITH_RERUN}" -eq 1 ]] && echo "ON  (recorded cameras + scalars; GUI outlives this wrapper)" || echo "OFF (pass --with-rerun to spawn alongside the live stack)")
  ports            : pose=${POSE_PORT}  x2_debug=${DEBUG_PORT}
  pose PUB bind    : ${PUB_BIND_HOST}:${POSE_PORT}$( [[ "${PUB_BIND_HOST}" == "127.0.0.1" ]] && echo "  (LAN-isolated: PC2 cannot SUB even if x2_pose_proxy is up)" || echo "  (LAN-visible: PC2 pose proxy SUBs this stream over wifi)" )
EOF
echo

# --------------------------------------------------------------------------
# Step 0/2 -- Rerun viewer for the recorded episode. Spawned BEFORE the
# deploy so its 5-30 s cold-load (decode video metadata + send columnar
# scalars) overlaps with the deploy bring-up instead of stacking on
# top. The loader python process is short-lived; the rerun GUI it
# spawns is a separate process by design and outlives this wrapper.
# --------------------------------------------------------------------------

if [[ "${WITH_RERUN}" -eq 1 ]]; then
    RERUN_LOG="${LOG_DIR}/rerun.log"
    RERUN_ARGS=(--dataset "${DATASET}" --episode "${EPISODE}")
    log "Step 0/2 -- spawning rerun viewer -> ${RERUN_LOG}"
    log "  (recorded camera MP4s + body_q + hand joints + wrist FK trace; GUI outlives this wrapper)"
    "${RERUN_SH}" "${RERUN_ARGS[@]}" >"${RERUN_LOG}" 2>&1 &
    RERUN_PID=$!
    log "  rerun loader pid=${RERUN_PID}; following with: tail -F ${RERUN_LOG}"
fi

# --------------------------------------------------------------------------
# Step 1/2 -- Deploy (docker sim + ONNX). Skipped in --no-deploy /
# --pc2-host modes. We pass ``--disable-pose-ref-watchdog`` because the
# replayer publishes directly to pose:5556 (no recorder merging idle
# frames during cold-start) and the watchdog would otherwise trip
# during the replay's --countdown phase. Same rationale the pkl wrapper
# documents.
# --------------------------------------------------------------------------

if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_LOG="${LOG_DIR}/deploy.log"
    if (( DURATION_S > 0 )); then
        DEPLOY_DURATION_S=$(( DURATION_S + 30 ))
    else
        DEPLOY_DURATION_S=0
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
        --deploy-extra-arg --disable-pose-ref-watchdog
    )
    if (( DEPLOY_DURATION_S > 0 )); then
        DEPLOY_ARGS+=(--max-duration "${DEPLOY_DURATION_S}")
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

    log "Step 1/2 -- spawning deploy_x2.sh sim --vla -> ${DEPLOY_LOG}"
    "${DEPLOY_SH}" "${DEPLOY_ARGS[@]}" >"${DEPLOY_LOG}" 2>&1 &
    DEPLOY_PID=$!

    log "  waiting for deploy 'Launching ...' marker (up to 180s)..."
    if ! wait_for_log_marker "${DEPLOY_LOG}" "${DEPLOY_PID}" "Launching ..." 180 "deploy"; then
        exit 1
    fi
    log "  deploy READY (pid=${DEPLOY_PID}); settle 2s before replay ..."
    sleep 2.0
else
    if [[ -n "${PC2_HOST}" ]]; then
        log "Step 1/2 -- deploy spawn SKIPPED (--pc2-host ${PC2_HOST}). PC2 daemons must already be running."
    else
        log "Step 1/2 -- deploy spawn SKIPPED (--no-deploy). External deploy must SUB pose on :${POSE_PORT}."
    fi
fi

# --------------------------------------------------------------------------
# Step 2/2 -- Replayer. PUBs ``pose`` on :5556 with the v5 future window
# (joint_pos_mj + joint_pos_mj_future + joint_vel_mj_future + sibling
# fields) so the deploy actually tracks the recorded trajectory instead
# of back-filling with the trained idle stand pose. See the docstring
# at the top of replay_x2_dataset.py for the v5-promotion rationale.
# --------------------------------------------------------------------------

REPLAY_LOG="${LOG_DIR}/replay.log"
REPLAY_ARGS=(
    -u
    -m gear_sonic.scripts.replay_x2_dataset
    --dataset "${DATASET}"
    --episode "${EPISODE}"
    --pub-host "${PUB_BIND_HOST}"
    --pub-port "${POSE_PORT}"
    --pub-topic "${POSE_TOPIC}"
)
if [[ -n "${RATE}" ]]; then
    REPLAY_ARGS+=(--rate "${RATE}")
fi
if [[ -n "${RATE_SCALE}" ]]; then
    REPLAY_ARGS+=(--rate-scale "${RATE_SCALE}")
fi
if [[ "${LOOP}" -eq 1 ]]; then
    REPLAY_ARGS+=(--loop)
fi
if [[ -n "${COUNTDOWN}" ]]; then
    REPLAY_ARGS+=(--countdown "${COUNTDOWN}")
fi
if [[ -n "${HOLD_ON_EXIT}" ]]; then
    REPLAY_ARGS+=(--hold-on-exit "${HOLD_ON_EXIT}")
fi
if [[ -n "${PC2_HOST}" ]]; then
    REPLAY_ARGS+=(--pc2-host "${PC2_HOST}")
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
    REPLAY_ARGS+=(--dry-run)
fi

log "Step 2/2 -- spawning replay_x2_dataset -> ${REPLAY_LOG}"
"${PYTHON}" "${REPLAY_ARGS[@]}" >"${REPLAY_LOG}" 2>&1 &
REPLAY_PID=$!

if [[ "${DRY_RUN}" -eq 1 ]]; then
    # Dry-run prints the banner and exits ~immediately; just wait for it
    # and pass through its exit code (we never bound the deploy in this
    # mode in practice -- the user typically pairs --dry-run with
    # --no-deploy -- but if they didn't, the trap will still tear down
    # the deploy on exit).
    log "  dry-run: waiting for replay to print stats + exit ..."
    wait "${REPLAY_PID}"
    REPLAY_RC=$?
    log "  dry-run replay exited rc=${REPLAY_RC}; tail of ${REPLAY_LOG}:"
    tail -n 30 "${REPLAY_LOG}" || true
    exit "${REPLAY_RC}"
fi

log "  waiting for replay 'PUB bound on' marker (up to 30s)..."
if ! wait_for_log_marker "${REPLAY_LOG}" "${REPLAY_PID}" \
        "PUB bound on" 30 "replay"; then
    exit 1
fi
log "  replay READY (pid=${REPLAY_PID}); the stack is live."
log "  follow replay output with: tail -F ${REPLAY_LOG}"

# --------------------------------------------------------------------------
# Main wait: block until the replayer finishes (the typical end-of-run
# signal -- one pass for non-looping, Ctrl-C for looping), the deploy
# dies, or a wall-clock --duration cap fires.
# --------------------------------------------------------------------------

__exit_rc=""
check_child_alive_or_finished() {
    local pid="$1"
    if kill -0 "${pid}" 2>/dev/null; then
        __exit_rc=""
        return 0
    fi
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
                log "deploy finished normally (pid=${DEPLOY_PID}, exit=${__exit_rc:-0}, --max-duration elapsed); ending session."
                return 0
            elif (( check_rc == 2 )); then
                err "deploy died (pid=${DEPLOY_PID}, exit=${__exit_rc}); see ${DEPLOY_LOG}"
                return 1
            fi
        fi

        check_child_alive_or_finished "${REPLAY_PID}"
        check_rc=$?
        if (( check_rc == 1 )); then
            log "replay finished normally (pid=${REPLAY_PID}, exit=${__exit_rc:-0}); ending session."
            return 0
        elif (( check_rc == 2 )); then
            err "replay died unexpectedly (pid=${REPLAY_PID}, exit=${__exit_rc}); see ${REPLAY_LOG}"
            return 1
        fi

        sleep 1
    done
}

if (( DURATION_S > 0 )); then
    log "running for up to ${DURATION_S}s wall-clock (Ctrl-C to stop early) ..."
    end_ts=$(( $(date +%s) + DURATION_S ))
    main_wait_loop "${end_ts}"
else
    log "running until replay finishes or Ctrl-C ..."
    main_wait_loop 0
fi
WAIT_LOOP_RC=$?

# The cleanup_children trap will fire on exit; just propagate the wait
# loop's rc as the script's overall rc so callers can shell-test for
# success vs failure.
exit "${WAIT_LOOP_RC}"
