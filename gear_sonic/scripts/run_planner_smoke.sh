#!/usr/bin/env bash
# X2 heuristic planner smoke runner.
#
# Spawns the planner, an optional debug capture, and (when --with-deploy is
# given) the X2 sim deploy as CHILD processes of this script. A trap-based
# cleanup ensures every child is killed on any exit path -- normal, SIGINT,
# SIGTERM, or shell error.
#
# Why this exists: iterating on the planner against the C++ deploy easily
# leaves zombie ros2/zmq/python processes if you Ctrl-C the wrong terminal.
# This wrapper is the canonical entry point for "policy in the loop"
# validation (planner plan section D.6) and the only thing the docs / tests
# tell operators to use.
#
# Usage:
#   gear_sonic/scripts/run_planner_smoke.sh [--demo PATH.yaml] [--duration N]
#                                           [--with-deploy] [--with-debug]
#                                           [--with-viewer] [--pub-port N]
#                                           [--cleanup-only] [--keyboard]
#                                           [--no-sim-viewer]
#                                           [--sim-profile {parity,manual,gantry,handoff}]
#                                           [--sim-rsi-pkl PATH]
#
# Default --sim-profile is `parity`: bridge RSIs from the canonical
# planner anchor PKL (auto-baked on first use), elastic band off,
# robot spawns on the floor. Pass `--sim-profile manual` if you need
# the legacy band-catch-from-air behaviour.
#
# Examples:
#   # Scripted demo end-to-end with SONIC physics + MuJoCo viewer:
#   ./run_planner_smoke.sh --demo gear_sonic/data/scripted_demos/forward_back_turn.yaml \
#                          --with-deploy --duration 30
#
#   # Keyboard teleop with SONIC physics + MuJoCo viewer (W/A/S/D/Q/E/...):
#   ./run_planner_smoke.sh --with-deploy --keyboard --duration 120
#
#   # Lightweight: planner + standalone kinematic viewer (no physics, no SONIC):
#   ./run_planner_smoke.sh --demo gear_sonic/data/scripted_demos/forward_back_turn.yaml \
#                          --with-viewer
#
#   # Headless CI / regression run (no windows):
#   ./run_planner_smoke.sh --duration 5 --keyboard --with-deploy --no-sim-viewer
#
#   ./run_planner_smoke.sh --cleanup-only        # kill stale planners + free ports
#
# --with-viewer opens a MuJoCo window subscribing to the planner's pose
# stream so you can see the planner's KINEMATIC output (no physics).
#
# --with-deploy spawns the SONIC sim deploy in a docker container; it
# subscribes to the planner's pose topic and runs the trained policy
# closed-loop in MuJoCo with full physics. By default we also pass
# --sim-viewer + --autostart-after 0 so a window pops with the robot
# tracked by SONIC; pass --no-sim-viewer to suppress for CI / headless.
#
# Pre-flight:
#   - Verifies the publish port is free (refuses to start otherwise; the
#     planner CLI also refuses but we surface a clearer error).
#   - Verifies the primitives PKL exists (run curate_x2_primitives first if not).
#
# All children inherit a fresh process group so SIGTERM cascades without
# orphaning grandchildren.

set -u
set -o pipefail

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRIMITIVES_PKL="${REPO_ROOT}/gear_sonic/data/motions/x2_planner_primitives.pkl"
BINS_YAML="${REPO_ROOT}/gear_sonic/data/motions/x2_planner_bins.yaml"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

PID_FILE="/tmp/x2_heuristic_planner.pid"
LOG_DIR="${REPO_ROOT}/data/sim_to_real_anchors/planner_smoke"

# --------------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------------

DEMO=""
DURATION_S=20
PUB_PORT=5556
WITH_DEPLOY=0
WITH_DEBUG=0
WITH_VIEWER=0
KEYBOARD=0
CLEANUP_ONLY=0
SIM_VIEWER=1            # opens MuJoCo window when --with-deploy is set
# Default to PARITY profile (RSI from a stand-pose PKL, elastic band
# OFF, ramp 0, autostart 0) so the planner-driven sim mirrors
# `deploy_x2.sh sim --motion <pkl>` exactly: robot spawns ON THE FLOOR
# at the planner's canonical anchor frame, no air-drop, no band drama.
# `--sim-profile manual` is still available for legacy band-catch behaviour.
SIM_PROFILE="parity"
# RSI source PKL for parity profile. Bridge spawns at this PKL's frame 0
# pose. The default PKL is auto-baked from
# `gear_sonic.scripts.bake_planner_rsi_anchor` so frame 0 is bit-identical
# to the planner's canonical first-frame emit (joints = idle_stand[0],
# quat = identity yaw, pelvis_z from clip = ~0.685m). This guarantees:
#   bridge spawn pose == warmup wire content == state-machine first frame
# -> no tracker-error fight on tick 0.
SIM_RSI_PKL="${REPO_ROOT}/data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_rsi_anchor.pkl"
# ONNX checkpoint to feed --with-deploy. Override with --model PATH or
# the X2_PLANNER_SMOKE_MODEL env var. The default points at the H200
# 25k-iter sphere-feet checkpoint we've been validating against (same
# one bake_primitive_for_deploy.py uses in its example output).
SIM_MODEL="${X2_PLANNER_SMOKE_MODEL:-/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx}"
# When the deploy is run with --sim-viewer, lock the MuJoCo camera onto
# the robot's pelvis so a multi-primitive sequence (which can drift
# meters across the floor) stays in frame. Override with the matching
# CLI flags below if you want a different framing or no tracking.
SIM_CAM_TRACK_BODY="pelvis"
SIM_CAM_DISTANCE="3.5"
SIM_CAM_ELEVATION="-12"
SIM_CAM_AZIMUTH="135"

usage() {
    grep -E '^# (Usage|Examples)' "$0" | sed 's/^# //' >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --demo) DEMO="$2"; shift 2 ;;
        --duration) DURATION_S="$2"; shift 2 ;;
        --pub-port) PUB_PORT="$2"; shift 2 ;;
        --with-deploy) WITH_DEPLOY=1; shift ;;
        --with-debug) WITH_DEBUG=1; shift ;;
        --with-viewer) WITH_VIEWER=1; shift ;;
        --keyboard) KEYBOARD=1; shift ;;
        --cleanup-only) CLEANUP_ONLY=1; shift ;;
        --no-sim-viewer) SIM_VIEWER=0; shift ;;
        --sim-profile) SIM_PROFILE="$2"; shift 2 ;;
        --sim-rsi-pkl) SIM_RSI_PKL="$2"; shift 2 ;;
        --model) SIM_MODEL="$2"; shift 2 ;;
        --sim-cam-track-body) SIM_CAM_TRACK_BODY="$2"; shift 2 ;;
        --sim-cam-distance) SIM_CAM_DISTANCE="$2"; shift 2 ;;
        --sim-cam-elevation) SIM_CAM_ELEVATION="$2"; shift 2 ;;
        --sim-cam-azimuth) SIM_CAM_AZIMUTH="$2"; shift 2 ;;
        --no-sim-cam-track) SIM_CAM_TRACK_BODY=""; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

log() { printf '[planner-smoke %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
err() { printf '[planner-smoke %s ERROR] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

port_in_use() {
    # Returns 0 if port is in use, 1 if free.
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nPi ":${port}" >/dev/null 2>&1
        return $?
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -n tcp "${port}" >/dev/null 2>&1
        return $?
    fi
    # Fallback: probe via Python.
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
    log "  killing pid $pid"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    log "  force-killing pid $pid"
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
        for p in ${pids}; do kill_pid_quiet "${p}"; done
    fi
}

cleanup_stale() {
    log "cleanup_stale: PID file=${PID_FILE} port=${PUB_PORT}"
    if [[ -f "${PID_FILE}" ]]; then
        local stale_pid
        stale_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        kill_pid_quiet "${stale_pid}"
        rm -f "${PID_FILE}"
    fi
    free_port "${PUB_PORT}"
}

# --------------------------------------------------------------------------
# Cleanup trap (set BEFORE any child is spawned)
# --------------------------------------------------------------------------

CHILD_PIDS=()
cleanup_children() {
    log "shutting down children..."
    # Reverse-order so the deploy goes down before the planner that feeds it.
    for ((i=${#CHILD_PIDS[@]}-1; i>=0; i--)); do
        kill_pid_quiet "${CHILD_PIDS[$i]}"
    done
    CHILD_PIDS=()
    rm -f "${PID_FILE}"
}
trap 'cleanup_children; exit 130' INT TERM
trap 'cleanup_children' EXIT

# --------------------------------------------------------------------------
# wait_for_deploy_ready: tail $1 (deploy log) for the bridge's
# "Launching ..." marker that indicates the C++ deploy + bridge are up
# and ready to consume pose frames. Returns 0 on ready, 1 on timeout
# or early death.
#
# This mirrors the wait loop in record_x2_dataset.sh and
# browse_x2_planner_primitives._wait_for_deploy_ready, which has been
# the proven boot pattern for "land the robot gently on the floor"
# for months -- because nothing is on the ZMQ pose wire while the
# C++ deploy is booting + the band is releasing, the policy uses its
# internal default_angles target and the band drop is the ONLY
# transient the policy has to absorb. (vs. the old smoke flow:
# planner spawns first and starts streaming idle_stand BEFORE the
# deploy is up -> policy tries to track a moving reference during
# the band drop -> robot launches into the air.)
# --------------------------------------------------------------------------
wait_for_deploy_ready() {
    local deploy_log="$1"
    local deploy_pid="$2"
    local timeout_s="${3:-180}"
    local marker="${4:-Launching ...}"
    local start_ts
    start_ts=$(date +%s)
    while :; do
        if ! kill -0 "${deploy_pid}" 2>/dev/null; then
            err "deploy died during bring-up; tail of log:"
            tail -n 40 "${deploy_log}" >&2 || true
            return 1
        fi
        if [[ -f "${deploy_log}" ]] && grep -F -q "${marker}" "${deploy_log}" 2>/dev/null; then
            return 0
        fi
        local now elapsed
        now=$(date +%s)
        elapsed=$((now - start_ts))
        if (( elapsed > timeout_s )); then
            err "deploy did not reach '${marker}' within ${timeout_s}s; tail of log:"
            tail -n 40 "${deploy_log}" >&2 || true
            return 1
        fi
        sleep 0.5
    done
}

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
if port_in_use "${PUB_PORT}"; then
    err "publish port ${PUB_PORT} is in use; run: $0 --cleanup-only"
    exit 1
fi
mkdir -p "${LOG_DIR}"

# --------------------------------------------------------------------------
# Spawn order (CRITICAL): when --with-deploy is set, the deploy MUST come
# up first, reach 'Launching ...' (bridge + C++ deploy alive, sim profile
# fully resolved, init pose applied, viewer window open), and then settle
# for ~2 s before the planner starts publishing on the ZMQ pose wire.
#
# What "good" looks like under --sim-profile parity (the default):
#   * Bridge RSIs from SIM_RSI_PKL frame 0 -> robot spawns ON THE FLOOR
#     in the planner's canonical anchor pose (joints from idle_stand[0],
#     identity-yaw quat, pelvis_z ~ 0.685m).
#   * Elastic band OFF, ramp_seconds 0, autostart 0 -- mirrors what
#     `deploy_x2.sh sim --motion <pkl>` does in legacy playback mode.
#   * Nothing on the ZMQ wire during boot.
#   * After "Launching ..." marker + 2s settle, planner starts publishing
#     the SAME anchor frame frozen for `--warmup-quiet-stand-s` seconds
#     (default 2.0 s), then transitions to live state-machine output.
#
# Bit-identical bridge spawn / warmup wire / state-machine first frame
# means there's no tracker-error fight on tick 0; the robot stays planted.
#
# Legacy behavior (--sim-profile manual) is preserved as an opt-in:
# spawns at sim_init_pose=default (pelvis 0.85m air-drop) with the
# elastic band catching the body, then auto-releasing 1s after the first
# deploy command. Tolerable for some debug flows but the robot visibly
# launches from the air.
#
# Standalone (no --with-deploy) keeps the legacy ordering: planner
# first, then optional viewer.
# --------------------------------------------------------------------------

DEPLOY_PID=""
if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_SH="${REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh"
    if [[ ! -x "${DEPLOY_SH}" ]]; then
        err "deploy_x2.sh not found / not executable: ${DEPLOY_SH}"
        exit 1
    fi
    DEPLOY_LOG="${LOG_DIR}/deploy_$(date +%Y%m%d_%H%M%S).log"
    if [[ -z "${SIM_MODEL}" || ! -f "${SIM_MODEL}" ]]; then
        err "deploy needs --model PATH (or X2_PLANNER_SMOKE_MODEL env)."
        err "tried: ${SIM_MODEL:-<unset>}"
        exit 1
    fi
    # Parity profile requires --motion <pkl> for bridge RSI even in --vla
    # mode (the C++ deploy ignores --motion under --vla but the bridge
    # uses MOTION_SOURCE for RSI). Auto-bake the canonical anchor PKL
    # if it's missing so a fresh checkout (or one where primitives were
    # rebuilt) just works.
    if [[ "${SIM_PROFILE}" == "parity" ]]; then
        if [[ ! -f "${SIM_RSI_PKL}" ]]; then
            log "RSI anchor PKL not found at ${SIM_RSI_PKL}; baking now..."
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
    # Deploy runs for the full duration plus a small headroom so it
    # doesn't terminate mid-clip if the planner takes a moment to come
    # up. The planner is the wall-clock keeper at the bottom of the
    # script.
    DEPLOY_DURATION_S=$(( DURATION_S + 5 ))
    DEPLOY_ARGS=(
        sim
        --no-confirm
        --vla
        --vla-zmq-host 127.0.0.1
        --vla-zmq-port "${PUB_PORT}"
        --sim-profile "${SIM_PROFILE}"
        --model "${SIM_MODEL}"
        --autostart-after 0
        --max-duration "${DEPLOY_DURATION_S}"
    )
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
            log "spawning deploy_x2.sh sim FIRST (viewer ON, cam->${SIM_CAM_TRACK_BODY}, profile=${SIM_PROFILE}) -> ${DEPLOY_LOG}"
        else
            log "spawning deploy_x2.sh sim FIRST (viewer ON, free cam, profile=${SIM_PROFILE}) -> ${DEPLOY_LOG}"
        fi
    else
        log "spawning deploy_x2.sh sim FIRST (viewer OFF, profile=${SIM_PROFILE}) -> ${DEPLOY_LOG}"
    fi
    "${DEPLOY_SH}" "${DEPLOY_ARGS[@]}" >"${DEPLOY_LOG}" 2>&1 &
    DEPLOY_PID=$!
    CHILD_PIDS+=("${DEPLOY_PID}")

    log "waiting for deploy to reach 'Launching ...' (up to 180s)..."
    if ! wait_for_deploy_ready "${DEPLOY_LOG}" "${DEPLOY_PID}" 180 "Launching ..."; then
        err "deploy did not become ready; aborting."
        exit 1
    fi
    log "deploy ready (pid=${DEPLOY_PID}); sleeping 2.0s settle before starting planner."
    sleep 2.0
fi

# --------------------------------------------------------------------------
# Spawn planner
# --------------------------------------------------------------------------

PLANNER_LOG="${LOG_DIR}/planner_$(date +%Y%m%d_%H%M%S).log"
log "spawning planner -> ${PLANNER_LOG}"
PLANNER_ARGS=(
    -m gear_sonic.scripts.x2_heuristic_planner
    --primitives "${PRIMITIVES_PKL}"
    --bins "${BINS_YAML}"
    --pub-host "127.0.0.1"
    --pub-port "${PUB_PORT}"
    --pid-file "${PID_FILE}"
    --duration-s "${DURATION_S}"
)
[[ -n "${DEMO}" ]] && PLANNER_ARGS+=(--demo "${DEMO}")
[[ "${KEYBOARD}" -eq 1 ]] && PLANNER_ARGS+=(--keyboard)

cd "${REPO_ROOT}"
if [[ "${KEYBOARD}" -eq 1 ]]; then
    "${PYTHON}" "${PLANNER_ARGS[@]}" 2>&1 | tee "${PLANNER_LOG}" &
else
    "${PYTHON}" "${PLANNER_ARGS[@]}" >"${PLANNER_LOG}" 2>&1 &
fi
PLANNER_PID=$!
CHILD_PIDS+=("${PLANNER_PID}")

# Wait for the planner to come up (PID file written).
log "waiting for planner pid=${PLANNER_PID} to write PID file..."
for _ in $(seq 1 100); do
    [[ -f "${PID_FILE}" ]] && break
    if ! kill -0 "${PLANNER_PID}" 2>/dev/null; then
        err "planner exited early; tail of log:"
        tail -n 40 "${PLANNER_LOG}" >&2 || true
        exit 1
    fi
    sleep 0.05
done
if [[ ! -f "${PID_FILE}" ]]; then
    err "planner did not write ${PID_FILE} within 5s"
    exit 1
fi
log "planner up (pid=${PLANNER_PID})."

# --------------------------------------------------------------------------
# Optional: dump_x2_debug
# --------------------------------------------------------------------------

if [[ "${WITH_DEBUG}" -eq 1 ]]; then
    DEBUG_LOG_DIR="${LOG_DIR}/debug_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${DEBUG_LOG_DIR}"
    log "spawning dump_x2_debug -> ${DEBUG_LOG_DIR}"
    "${PYTHON}" -m gear_sonic.scripts.dump_x2_debug \
        --output-dir "${DEBUG_LOG_DIR}" \
        --duration "${DURATION_S}" \
        >"${LOG_DIR}/debug.log" 2>&1 &
    DEBUG_PID=$!
    CHILD_PIDS+=("${DEBUG_PID}")
fi

# --------------------------------------------------------------------------
# Optional: live MuJoCo viewer (subscribes to the pose stream)
# --------------------------------------------------------------------------

if [[ "${WITH_VIEWER}" -eq 1 ]]; then
    VIEWER_LOG="${LOG_DIR}/viewer_$(date +%Y%m%d_%H%M%S).log"
    log "spawning live MuJoCo viewer -> ${VIEWER_LOG}"
    "${PYTHON}" -m gear_sonic.scripts.view_x2_planner_mujoco \
        --from-zmq "127.0.0.1:${PUB_PORT}" \
        --duration-s "${DURATION_S}" \
        >"${VIEWER_LOG}" 2>&1 &
    VIEWER_PID=$!
    CHILD_PIDS+=("${VIEWER_PID}")
fi

# --------------------------------------------------------------------------
# Wait for the planner to finish (it's the wall-clock keeper)
# --------------------------------------------------------------------------

log "waiting for planner to finish (duration=${DURATION_S}s)..."
wait "${PLANNER_PID}"
PLANNER_RC=$?
log "planner exited rc=${PLANNER_RC}"

# Trap-based cleanup will handle the rest.
exit "${PLANNER_RC}"
