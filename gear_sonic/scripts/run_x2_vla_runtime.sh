#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# X2 VLA runtime — operator-side launcher for autonomous Isaac-GR00T VLA.
#
# Sits alongside ``run_x2_quest3_planner_stack.sh`` but is NOT named
# "planner stack" because the VLA pipeline has no planner: the GR00T
# policy predicts SONIC ``motion_token`` chunks directly from cameras +
# language prompt, and this script decodes those tokens back into
# ``joint_pos_mj`` on the wire. Both launchers feed the same downstream
# SONIC daemon on PC2; the operator just switches which one they run:
#
#     teleop          run_x2_quest3_planner_stack.sh
#     recording       run_x2_quest3_planner_stack.sh --with-record --head-cameras
#     autonomous VLA  run_x2_vla_runtime.sh        (this file)
#
# The SONIC tracker (the C++ deploy on PC2 with the fused
# encoder+FSQ+decoder ONNX) stays alive across mode switches via
# ``gear_sonic_deploy/scripts/x2_pc2_daemons.sh start`` — it's the
# core robotic stack and you don't have to restart it when flipping
# modes.
#
# One launcher, two surfaces — same pattern as
# ``run_x2_quest3_planner_stack.sh``:
#
#   * **Sim (default)** — omit ``--pc2-host``. Spawns the VLA bridge +
#     a local ``deploy_x2.sh sim --vla`` docker stack on localhost.
#     Cameras come from the MuJoCo ghost renderer (stereo keys via
#     ``_GhostCameraProvider``). Safe place to debug policy output
#     before powering the real robot.
#
#   * **Real robot** — pass ``--pc2-host <PC2_IP>``. Spawns only the
#     bridge on the laptop; assumes ``x2_pc2_daemons.sh start`` is
#     already running on PC2. Cameras come from the PC2 ZMQ bridge.
#
# Topology (real robot, ``--pc2-host``):
#
#     +-----------------+           tcp://*:5556 (pose)
#     |   LAPTOP        |  ─────────────────────────────────────────►
#     |  this bridge    |             (PC2 pose proxy SUBs here)
#     |  - live_vla_..  |                          │
#     |  - SUB x2_debug |◄─── tcp://PC2:5557 ◄────┤
#     |  - SUB cameras  |◄─── tcp://PC2:5555 ◄────┤
#     +-----------------+                          │
#                                                  ▼
#                                       +-------------------+
#                                       |   PC2 (Orin NX)   |
#                                       |  x2_pc2_daemons:  |
#                                       |   - pose proxy    |
#                                       |   - deploy --vla  |
#                                       |   - hand bridge   |
#                                       |   - motor monitor |
#                                       |  x2_pc2_cameras:  |
#                                       |   - camera ZMQ    |
#                                       +-------------------+
#
# Prereqs (run these on PC2 BEFORE this launcher; see runbook
# ``docs/source/tutorials/x2_vla_runtime.md`` for the
# bring-up order):
#
#   1. ``./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start``
#        -- starts pose proxy + deploy --vla + hand bridge + motor monitor.
#        Must be run explicitly; this launcher does NOT auto-start the
#        deploy (it blocks on a Y/n safety gate + calls SDK MC stop_app,
#        both side effects the operator has to opt into).
#
#   2. PC2 camera bridge. Auto-started by THIS launcher (idempotent, via
#      SSH against PC2) whenever the preflight probe finds tcp://PC2:5555
#      silent. Mirrors how ``run_x2_quest3_planner_stack.sh --head-cameras``
#      auto-launches the same bridge during recording. Pass
#      ``--no-cameras-autostart`` if you want to manage it manually with
#      ``./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve`` --
#      typically the bridge is already running from a prior recording
#      session (the recorder deliberately leaves it up between runs).
#
#   3. Confirm the deploy passed its Y/n safety gate AND the operator hit
#      the start-trigger sentinel (the SDK MC has been stopped and the
#      deploy is actively tracking the trained idle stand).
#
# Usage:
#   # Sim (default — no --pc2-host)
#   ./gear_sonic/scripts/run_x2_vla_runtime.sh \
#       --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
#       --motion-token-decoder $HOME/x2_cloud_checkpoints/.../model_step_025000.pt \
#       --prompt "grab the can from the table"
#
#   # Real robot
#   ./gear_sonic/scripts/run_x2_vla_runtime.sh \
#       --pc2-host 192.168.86.32 \
#       --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
#       --motion-token-decoder $HOME/x2_cloud_checkpoints/.../model_step_025000.pt \
#       --prompt "grab the can from the table"
#
#   ./gear_sonic/scripts/run_x2_vla_runtime.sh stop --run-dir /tmp/x2_vla_runtime-...
#
# Knobs (override via env or flags):
#
#   MODEL_DIR              # required (or pass --model)
#   PROMPT                 # language instruction (default: "grab a drink")
#   PC2_HOST               # unset = sim; set via --pc2-host for real robot
#   PC2_DEBUG_PORT         # default 5557 (PC2 deploy x2_debug PUB)
#   PC2_CAMERAS_PORT       # default 5555 (PC2 cameras PUB)
#   LAPTOP_POSE_PORT       # default 5556 (this bridge's pose PUB)
#   MODALITY_CONFIG        # default gear_sonic/data/x2_modality_config_omnihand_stereo.py
#
#   MOTION_TOKEN_DECODER   # bridge-side SONIC decoder .pt that translates
#                          # the VLA's predicted motion_token chunks back
#                          # into joint_pos_mj on the wire. NOT the same
#                          # as the PC2 deploy ONNX! That one is fused
#                          # encoder+FSQ+decoder and runs inside
#                          # x2_pc2_daemons. This .pt is just the g1_dyn
#                          # DECODER weights. Required because the C++
#                          # deploy explicitly ignores the wire's
#                          # motion_token field (see
#                          # zmq_pose_input_source.hpp:22-25); without
#                          # this decoder, the body stays at idle and
#                          # only the OmniHand fingers move.
#                          # ``SONIC_CHECKPOINT`` is accepted as a
#                          # deprecated alias (one-shot warning).
#   SONIC_DECODER_DEVICE   # default cpu (sub-ms, no GPU needed)
#   MAX_DURATION           # default 300 seconds (5 min)
#   INFERENCE_MIN_PERIOD_S # default 0.8 s (= one VLA chunk @ 50 Hz)
#   RATE                   # default 50 Hz (deploy control loop)
#   CAMERAS_STALENESS_S    # default 2.0 s (publisher runs at 15 Hz)
#   CAMERAS_WARMUP_S       # default 15 s
#   DUMP_CHUNKS_EVERY      # default 1 (every chunk; diagnostic-heavy)
#   RUN_DIR                # default /tmp/x2_vla_runtime-<timestamp>
#   BRIDGE_PY              # default ~/miniconda3/envs/env_isaaclab/bin/python
#   SKIP_PREFLIGHT         # "" = run all preflight checks (default).
#                            Set to "1" to bypass connectivity probes
#                            (NOT recommended; defeats the dry-run safety net).
#   CAMERAS_AUTOSTART      # default 1: auto-SSH to PC2 and launch
#                            x2_pc2_cameras.sh serve if the camera ZMQ
#                            stream is silent during preflight. Set to 0
#                            (or pass --no-cameras-autostart) to skip.
#   PC2_USER               # default 'run' (the on-PC2 SSH user)
#
# Safety contract:
#
# * **Sim:** spawns ``deploy_x2.sh sim --vla`` only after the bridge logs
#   ``policy ready`` (GR00T loaded, idle_stand wire live). Forwards
#   ``--disable-pose-ref-watchdog`` and defaults ``parity`` profile +
#   ``--autostart-after 0`` — same sequencing as quest3 VLA (spawn deploy
#   late, not at the 2 s PUB-bind).
# * **Real robot:** does NOT start the deploy. Assumes PC2 daemons +
#   cameras are already up. Preflight checks fail-fast if either is
#   silent so we never wedge the deploy at a stale wire.
# * MAX_DURATION defaults to 300 s (5 min). Pass MAX_DURATION=0 only after you've watched a
#   bounded run end-to-end and you trust the policy + the operator can
#   reach the deploy's HOLD_FOR_MC handoff.
# * The launcher prints a SAFETY BANNER and waits 5 s before going live
#   so the operator can abort with Ctrl-C if anything looks off
#   (override via FAST_ABORT=1; not recommended).
# * On Ctrl-C: SIGTERM the bridge, wait up to 10 s for graceful exit
#   (RAMP_OUT -> stand handoff), then SIGKILL the residue.
#
# Exit status: 0 on clean shutdown, non-zero on preflight failure /
# bridge crash / kill timeout.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NC=$'\033[0m'
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'

# ----- knobs (override via env) -----------------------------------------
: "${MODEL_DIR:=}"
: "${PROMPT:=grab the can from the table}"
: "${PC2_HOST:=}"
: "${PC2_DEBUG_PORT:=5557}"
: "${PC2_CAMERAS_PORT:=5555}"
: "${LAPTOP_POSE_PORT:=5556}"
: "${MODALITY_CONFIG:=gear_sonic/data/x2_modality_config_omnihand_stereo.py}"
: "${MOTION_TOKEN_DECODER:=}"
: "${SONIC_CHECKPOINT:=}"
: "${SONIC_DECODER_DEVICE:=cpu}"
# Wire-shaping safety knobs that protect the body from step-input
# shoves on the idle->VLA transition and from inter-chunk seams.
# See gear_sonic/scripts/live_vla_publish_motion_token.py --help for
# the full rationale; the defaults match what we ship in the live
# bridge.
: "${VLA_RAMP_IN_TICKS:=75}"
: "${VLA_TARGET_LPF_HZ:=2.0}"
: "${VLA_FUTURE_LPF_HZ:=2.0}"
: "${VLA_HAND_LPF_HZ:=1.0}"
: "${VLA_HAND_CHUNK_BLEND_TICKS:=30}"
: "${VLA_MAX_HAND_STEP:=0.08}"
: "${VLA_MAX_WIRE_DEV_FROM_BODY:=0.18}"
: "${VLA_MAX_WIRE_STEP:=0.035}"
: "${VLA_CHUNK_BLEND_TICKS:=40}"
: "${VLA_MAX_ACTION_IL:=8.0}"
: "${VLA_DECODE_DELAY_TICKS:=150}"
: "${VLA_BODY_MODE:=manipulation}"
: "${VLA_MODE_CONTROL_FILE:=}"
: "${VLA_FREEZE_BODY_GROUPS:=}"
: "${MAX_DURATION:=300}"
: "${INFERENCE_MIN_PERIOD_S:=1.5}"
: "${RATE:=50}"
: "${CAMERAS_STALENESS_S:=2.0}"
: "${CAMERAS_WARMUP_S:=15}"
: "${DUMP_CHUNKS_EVERY:=1}"
: "${RUN_DIR:=}"
: "${BRIDGE_PY:=${HOME}/miniconda3/envs/env_isaaclab/bin/python}"
: "${SKIP_PREFLIGHT:=}"
: "${FAST_ABORT:=}"
: "${VLA_DEVICE:=cuda:0}"
: "${EMBODIMENT_TAG:=new_embodiment}"
: "${CAMERAS_AUTOSTART:=1}"
: "${PC2_USER:=run}"

# Sim-only knobs (ignored when --pc2-host is set)
: "${SIM_MODEL:=${HOME}/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx}"
: "${SIM_PROFILE:=parity}"
: "${SIM_VIEWER:=1}"
: "${SIM_AUTOSTART_AFTER:=0}"
: "${SIM_RSI_PKL:=${REPO_ROOT}/data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_rsi_anchor.pkl}"
: "${PRIMITIVES_PKL:=${REPO_ROOT}/gear_sonic/data/motions/x2_planner_primitives.pkl}"
: "${BINS_YAML:=${REPO_ROOT}/gear_sonic/data/motions/x2_planner_bins.yaml}"
: "${SIM_WITH_OMNIHAND:=1}"
: "${SIM_MAX_TARGET_DEV:=}"
: "${SIM_DEPLOY_TARGET_LPF_HZ:=}"
: "${RENDER_WIDTH:=640}"
: "${RENDER_HEIGHT:=480}"
: "${VIDEO_OUT:=}"
: "${VIDEO_FRONT_OUT:=}"

DEPLOY_SH="${REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh"

usage() {
    sed -n '2,90p' "$0"
    cat <<'EOF'

Flags (preferred over env vars):
  --model PATH                 HuggingFace fine-tune checkpoint dir (required)
  --motion-token-decoder PATH  SONIC .pt for token→pose decode
  --prompt TEXT                Language instruction
  --pc2-host HOST              Real robot: PC2 IP (omit for sim)
  --modality-config PATH       Modality side-load
  --max-duration SEC           Control window (sim deploy / real bridge)
  --run-dir PATH               Log output directory
  --onnx PATH                  SONIC ONNX for sim deploy (sim only)
  --sim-profile PROFILE        Sim deploy profile: handoff|parity|gantry (sim)
  --no-sim-viewer              Disable MuJoCo passive viewer (sim)
  --vla-ramp-in-ticks N        Bridge ramp-in ticks (default: 75)
  --vla-target-lpf-hz HZ       Body wire LPF (default: 2.0)
  --vla-future-lpf-hz HZ       Future-window LPF (default: 2.0)
  --vla-hand-lpf-hz HZ         Hand wire LPF (default: 1.0)
  --vla-hand-chunk-blend-ticks N Hand inter-chunk blend (default: 30)
  --vla-max-hand-step R          Max per-tick hand step (default: 0.08)
  --vla-max-wire-dev-from-body R Max wire pose deviation from body_q (default: 0.18)
  --vla-max-wire-step R          Max per-tick wire joint step (default: 0.035)
  --vla-decode-delay-ticks N     Idle ticks before VLA decode (default: 150)
  --body-mode MODE               manipulation | locomotion (VR analog)
  --mode-control-file PATH       Runtime mode switch file (one word per line)
  --freeze-body-groups LIST      Override manipulation freeze groups (default legs,waist)
  --render-width / --render-height   Ghost camera resolution (sim)
  --video-out PATH             Record ego_view MP4 (sim, ghost mode)
  --video-front-out PATH       Record third-person MP4 (sim)
  -h, --help                   Show this help

Commands:
  run (default)                Start VLA runtime (sim or real per --pc2-host)
  preflight                    Dry-run checks only
  stop                         Stop bridge (+ sim deploy if run-dir has deploy.pid)
EOF
    exit 0
}

# ----- arg parse --------------------------------------------------------
CMD="run"
ARGS=()
# Track whether the operator hit the deprecated flag / env so we warn once.
DEPRECATED_SONIC_CHECKPOINT_USED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        run|stop|preflight)
            CMD="$1"; shift ;;
        --model)            MODEL_DIR="$2"; shift 2 ;;
        --prompt)           PROMPT="$2"; shift 2 ;;
        --pc2-host)         PC2_HOST="$2"; shift 2 ;;
        --sim)              PC2_HOST=""; shift ;;
        --modality-config)  MODALITY_CONFIG="$2"; shift 2 ;;
        --onnx|--sim-model) SIM_MODEL="$2"; shift 2 ;;
        --sim-profile)      SIM_PROFILE="$2"; shift 2 ;;
        --sim-rsi-pkl)      SIM_RSI_PKL="$2"; shift 2 ;;
        --no-sim-viewer)    SIM_VIEWER=0; shift ;;
        --autostart-after)  SIM_AUTOSTART_AFTER="$2"; shift 2 ;;
        --max-target-dev)   SIM_MAX_TARGET_DEV="$2"; shift 2 ;;
        --deploy-target-lpf-hz) SIM_DEPLOY_TARGET_LPF_HZ="$2"; shift 2 ;;
        --vla-ramp-in-ticks) VLA_RAMP_IN_TICKS="$2"; shift 2 ;;
        --vla-target-lpf-hz) VLA_TARGET_LPF_HZ="$2"; shift 2 ;;
        --vla-future-lpf-hz) VLA_FUTURE_LPF_HZ="$2"; shift 2 ;;
        --vla-hand-lpf-hz) VLA_HAND_LPF_HZ="$2"; shift 2 ;;
        --vla-hand-chunk-blend-ticks) VLA_HAND_CHUNK_BLEND_TICKS="$2"; shift 2 ;;
        --vla-max-hand-step) VLA_MAX_HAND_STEP="$2"; shift 2 ;;
        --vla-max-wire-dev-from-body) VLA_MAX_WIRE_DEV_FROM_BODY="$2"; shift 2 ;;
        --vla-max-wire-step) VLA_MAX_WIRE_STEP="$2"; shift 2 ;;
        --vla-chunk-blend-ticks) VLA_CHUNK_BLEND_TICKS="$2"; shift 2 ;;
        --vla-max-action-il) VLA_MAX_ACTION_IL="$2"; shift 2 ;;
        --vla-decode-delay-ticks) VLA_DECODE_DELAY_TICKS="$2"; shift 2 ;;
        --body-mode)          VLA_BODY_MODE="$2"; shift 2 ;;
        --mode-control-file) VLA_MODE_CONTROL_FILE="$2"; shift 2 ;;
        --hands-only)
            warn "--hands-only is deprecated and ignored (fingers-only mode removed)."
            warn "  Use --body-mode manipulation (default): arms+hands, legs/waist frozen."
            shift ;;
        --freeze-body-groups) VLA_FREEZE_BODY_GROUPS="$2"; shift 2 ;;
        --sonic-decoder-device) SONIC_DECODER_DEVICE="$2"; shift 2 ;;
        --render-width)     RENDER_WIDTH="$2"; shift 2 ;;
        --render-height)    RENDER_HEIGHT="$2"; shift 2 ;;
        --video-out)        VIDEO_OUT="$2"; shift 2 ;;
        --video-front-out)  VIDEO_FRONT_OUT="$2"; shift 2 ;;
        --motion-token-decoder)
            MOTION_TOKEN_DECODER="$2"; shift 2 ;;
        --sonic-checkpoint)
            # Deprecated alias kept for backwards compat. Forwarded into
            # MOTION_TOKEN_DECODER iff the canonical flag/env wasn't
            # already set, so an operator passing both wins on the new
            # name.
            DEPRECATED_SONIC_CHECKPOINT_USED=1
            if [[ -z "$MOTION_TOKEN_DECODER" ]]; then
                MOTION_TOKEN_DECODER="$2"
            fi
            SONIC_CHECKPOINT="$2"
            shift 2 ;;
        --max-duration)     MAX_DURATION="$2"; shift 2 ;;
        --inference-min-period-s) INFERENCE_MIN_PERIOD_S="$2"; shift 2 ;;
        --rate)             RATE="$2"; shift 2 ;;
        --cameras-staleness-s) CAMERAS_STALENESS_S="$2"; shift 2 ;;
        --run-dir)          RUN_DIR="$2"; shift 2 ;;
        --bridge-py)        BRIDGE_PY="$2"; shift 2 ;;
        --skip-preflight)   SKIP_PREFLIGHT=1; shift ;;
        --fast-abort)       FAST_ABORT=1; shift ;;
        --no-cameras-autostart) CAMERAS_AUTOSTART=0; shift ;;
        --pc2-user)         PC2_USER="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  ARGS+=("$1"); shift ;;
    esac
done

# ----- defaults derived after parsing -----------------------------------
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="/tmp/x2_vla_runtime-$(date +%Y%m%d_%H%M%S)"
fi

# Backwards-compat env: if MOTION_TOKEN_DECODER wasn't given but
# SONIC_CHECKPOINT was, treat the latter as the decoder path and
# remember to print the deprecation warning below.
if [[ -z "$MOTION_TOKEN_DECODER" && -n "$SONIC_CHECKPOINT" ]]; then
    MOTION_TOKEN_DECODER="$SONIC_CHECKPOINT"
    DEPRECATED_SONIC_CHECKPOINT_USED=1
fi

# Auto-derive motion-token decoder sibling next to MODEL_DIR if not given:
#   ${MODEL_DIR}/sonic_checkpoint.pt   (legacy filename; kept for
#                                       cohabiting with older training runs)
#   ${MODEL_DIR}/motion_token_decoder.pt
#   or fall back to the canonical 25k pretrain
if [[ -z "$MOTION_TOKEN_DECODER" && -n "$MODEL_DIR" ]]; then
    for cand in \
        "${MODEL_DIR}/motion_token_decoder.pt" \
        "${MODEL_DIR}/sonic_checkpoint.pt" \
        "${MODEL_DIR}/model_step_025000.pt" \
        "${HOME}/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt"; do
        if [[ -f "$cand" ]]; then
            MOTION_TOKEN_DECODER="$cand"
            break
        fi
    done
fi

# Sim when PC2_HOST is empty; real robot when --pc2-host (or PC2_HOST env) is set.
if [[ -n "$PC2_HOST" ]]; then
    SIM_MODE=0
    DEBUG_SUB_HOST="$PC2_HOST"
else
    SIM_MODE=1
    DEBUG_SUB_HOST="localhost"
fi

PID_FILE_BRIDGE="${RUN_DIR}/bridge.pid"
LOG_FILE_BRIDGE="${RUN_DIR}/bridge.log"
PID_FILE_DEPLOY="${RUN_DIR}/deploy.pid"
LOG_FILE_DEPLOY="${RUN_DIR}/deploy.log"

# ----- helpers ----------------------------------------------------------

log()   { echo -e "${CYAN}[vla-runtime]${NC} $*"; }
ok()    { echo -e "${GREEN}[vla-runtime]${NC} $*"; }
warn()  { echo -e "${YELLOW}[vla-runtime]${NC} $*" >&2; }
err()   { echo -e "${RED}[vla-runtime]${NC} $*" >&2; }

# wait_for_log_marker: tails ${log_path} for ${marker} until ${pid} is
# alive AND the marker appears, or ${timeout_s} elapses. Returns 0/1.
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

ensure_parity_rsi_pkl() {
    if [[ "${SIM_PROFILE}" != "parity" ]]; then
        return 0
    fi
    if [[ -f "${SIM_RSI_PKL}" ]]; then
        log "parity RSI anchor: ${SIM_RSI_PKL}"
        return 0
    fi
    log "parity RSI anchor missing; baking -> ${SIM_RSI_PKL}"
    mkdir -p "$(dirname "${SIM_RSI_PKL}")"
    if ! "${BRIDGE_PY}" -m gear_sonic.scripts.bake_planner_rsi_anchor \
            --primitives-pkl "${PRIMITIVES_PKL}" \
            --bins-yaml "${BINS_YAML}" \
            --out "${SIM_RSI_PKL}" \
            >>"${RUN_DIR}/rsi_anchor_bake.log" 2>&1; then
        err "failed to bake parity RSI anchor; see ${RUN_DIR}/rsi_anchor_bake.log"
        return 1
    fi
    ok "parity RSI anchor baked: ${SIM_RSI_PKL}"
}

stop_bridge() {
    if [[ ! -f "$PID_FILE_BRIDGE" ]]; then
        log "no bridge.pid in $RUN_DIR; nothing to stop."
        return 0
    fi
    local pid
    pid="$(cat "$PID_FILE_BRIDGE")"
    if ! kill -0 "$pid" 2>/dev/null; then
        log "bridge pid $pid already dead; cleaning up pidfile."
        rm -f "$PID_FILE_BRIDGE"
        return 0
    fi
    log "SIGTERM bridge pid=$pid; waiting up to 10s for clean exit …"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            ok "bridge stopped cleanly."
            rm -f "$PID_FILE_BRIDGE"
            return 0
        fi
    done
    warn "bridge pid=$pid still alive after 10s; SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
    sleep 1
    rm -f "$PID_FILE_BRIDGE"
}

stop_deploy() {
    if [[ -f "$PID_FILE_DEPLOY" ]]; then
        local dpid
        dpid="$(cat "$PID_FILE_DEPLOY")"
        if kill -0 "$dpid" 2>/dev/null; then
            log "SIGTERM deploy wrapper pid=$dpid …"
            kill -TERM "$dpid" 2>/dev/null || true
            sleep 2
            kill -KILL "$dpid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE_DEPLOY"
    fi
}

stop_dump() {
    if [[ -f "${RUN_DIR}/dump.pid" ]]; then
        local dpid
        dpid="$(cat "${RUN_DIR}/dump.pid")"
        kill -TERM "$dpid" 2>/dev/null || true
        rm -f "${RUN_DIR}/dump.pid"
    fi
}

# Nuke any leftover sim/VLA processes regardless of which RUN_DIR they
# were spawned from. The dockerized deploy keeps :5557 bound even after
# the bridge exits; a second run's preflight must reach here or ports
# stay wedged forever.
kill_stale_sim_processes() {
    log "cleaning stale VLA / sim deploy processes …"
    pkill -TERM -f "live_vla_publish_motion_token" 2>/dev/null || true
    pkill -TERM -f "gear_sonic.scripts.dump_x2_debug" 2>/dev/null || true
    pkill -TERM -f "deploy_x2.sh.*sim" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "live_vla_publish_motion_token" 2>/dev/null || true
    pkill -KILL -f "gear_sonic.scripts.dump_x2_debug" 2>/dev/null || true
    pkill -KILL -f "deploy_x2.sh.*sim" 2>/dev/null || true
    # deploy_x2.sh uses gr00t-x2sim:latest (host networking). Older
    # docs referenced ancestor=x2sim; match both image + name patterns.
    local cid=""
    for filt in "name=x2-x2sim" "ancestor=gr00t-x2sim" "ancestor=x2sim"; do
        local found
        found="$(docker ps -q --filter "$filt" 2>/dev/null || true)"
        if [[ -n "$found" ]]; then
            cid="${cid} ${found}"
        fi
    done
    cid="$(echo "$cid" | tr ' ' '\n' | sort -u | tr '\n' ' ')"
    if [[ -n "${cid// }" ]]; then
        log "stopping sim docker container(s): ${cid}"
        docker kill $cid 2>/dev/null || true
        sleep 3
    fi
    if ss -tln 2>/dev/null | grep -qE ":${LAPTOP_POSE_PORT}\b|:${PC2_DEBUG_PORT}\b"; then
        if command -v fuser >/dev/null 2>&1; then
            warn "ZMQ ports still bound; fuser -k on :${LAPTOP_POSE_PORT} and :${PC2_DEBUG_PORT}"
            fuser -k "${LAPTOP_POSE_PORT}/tcp" 2>/dev/null || true
            fuser -k "${PC2_DEBUG_PORT}/tcp" 2>/dev/null || true
            sleep 2
        fi
    fi
}

stop_all() {
    stop_bridge
    stop_deploy
    stop_dump
    kill_stale_sim_processes
}

# ----- preflight --------------------------------------------------------

preflight_pc2_reachable() {
    log "[preflight] ping ${PC2_HOST} (single packet, 2s timeout) …"
    if ping -c 1 -W 2 "$PC2_HOST" >/dev/null 2>&1; then
        ok "  PC2 ${PC2_HOST} reachable."
    else
        err "  PC2 ${PC2_HOST} did NOT respond to ping."
        err "  - Are you on the right network? (wired SDK port = 10.0.1.50/24)"
        err "  - Is PC2 powered on? Try 'x2_pc2_daemons.sh print-env'."
        return 1
    fi
}

preflight_zmq_probe() {
    # Probe a ZMQ PUB endpoint by attempting a 5-second SUB. Returns 0 if
    # at least one message arrived. We do it via the existing .venv +
    # pyzmq rather than dragging in a separate tool.
    local host="$1" port="$2" topic_label="$3" timeout_s="${4:-5}"
    log "[preflight] subscribe to ${topic_label} on tcp://${host}:${port} (≤${timeout_s}s) …"
    if "${REPO_ROOT}/.venv/bin/python" - <<EOF
import sys, time
try:
    import zmq
except ImportError as exc:
    print(f"FAIL: pyzmq missing in .venv: {exc}", flush=True)
    sys.exit(2)
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.SUB)
sock.setsockopt(zmq.SUBSCRIBE, b"")
sock.setsockopt(zmq.LINGER, 0)
sock.setsockopt(zmq.RCVTIMEO, 250)
sock.connect("tcp://${host}:${port}")
deadline = time.monotonic() + ${timeout_s}.0
n_msgs = 0
while time.monotonic() < deadline:
    try:
        sock.recv(flags=0)
        n_msgs += 1
        if n_msgs >= 1:
            break
    except zmq.Again:
        pass
sock.close(linger=0)
if n_msgs >= 1:
    print(f"OK: received {n_msgs} message(s) from ${topic_label}", flush=True)
    sys.exit(0)
print(f"FAIL: no messages from ${topic_label} on tcp://${host}:${port} within ${timeout_s}s", flush=True)
sys.exit(1)
EOF
    then
        ok "  ${topic_label} stream live."
        return 0
    else
        err "  ${topic_label} stream silent on tcp://${host}:${port}."
        return 1
    fi
}

preflight_model_dir() {
    log "[preflight] check model checkpoint ${MODEL_DIR} …"
    if [[ -z "$MODEL_DIR" ]]; then
        err "  MODEL_DIR is empty. Pass --model PATH or set MODEL_DIR env."
        return 1
    fi
    if [[ ! -d "$MODEL_DIR" ]]; then
        err "  MODEL_DIR=${MODEL_DIR} is not a directory."
        return 1
    fi
    local missing=0
    # ``model.safetensors`` + ``experiment_cfg/`` are mandatory. The
    # processor surface is either a flat ``processor_config.json``
    # (the HF Trainer per-checkpoint layout, what
    # ``data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000``
    # looks like on disk) or a full ``processor/`` directory (what the
    # run root has) -- accept either.
    for f in model.safetensors experiment_cfg; do
        if [[ ! -e "${MODEL_DIR}/${f}" ]]; then
            err "  missing required entry: ${MODEL_DIR}/${f}"
            missing=1
        fi
    done
    if [[ ! -d "${MODEL_DIR}/processor" && ! -f "${MODEL_DIR}/processor_config.json" ]]; then
        err "  missing processor surface: neither ${MODEL_DIR}/processor/"
        err "  nor ${MODEL_DIR}/processor_config.json exists."
        err "  HF Trainer per-checkpoint dirs ship processor_config.json;"
        err "  the training-run root ships the full processor/ dir."
        missing=1
    fi
    if [[ "$missing" -ne 0 ]]; then
        err "  model checkpoint dir incomplete; aborting."
        return 1
    fi
    ok "  model checkpoint present."
}

preflight_modality_config() {
    log "[preflight] check modality config ${MODALITY_CONFIG} …"
    if [[ ! -f "${REPO_ROOT}/${MODALITY_CONFIG}" && ! -f "${MODALITY_CONFIG}" ]]; then
        err "  modality config not found: ${MODALITY_CONFIG}"
        return 1
    fi
    ok "  modality config present."
}

preflight_motion_token_decoder() {
    log "[preflight] check motion-token decoder checkpoint …"
    if [[ -z "$MOTION_TOKEN_DECODER" ]]; then
        warn "  no motion-token decoder found / specified."
        warn "  Without a decoder the C++ deploy IGNORES the wire's"
        warn "  motion_token field (zmq_pose_input_source.hpp:22-25), so the"
        warn "  body will stay at idle_stand even with VLA running -- only"
        warn "  the OmniHand fingers will move under VLA authority."
        warn "  Pass --motion-token-decoder /path/to/model_step_NNNNN.pt to"
        warn "  enable the body-pose decoder."
        return 0  # non-fatal, but loud
    fi
    if [[ ! -f "$MOTION_TOKEN_DECODER" ]]; then
        err "  motion-token decoder not found: $MOTION_TOKEN_DECODER"
        return 1
    fi
    ok "  motion-token decoder present: $MOTION_TOKEN_DECODER"
}

preflight_bridge_py() {
    log "[preflight] check bridge python interpreter ${BRIDGE_PY} …"
    if [[ ! -x "$BRIDGE_PY" ]]; then
        err "  BRIDGE_PY=${BRIDGE_PY} not found / not executable."
        err "  Default is ~/miniconda3/envs/env_isaaclab/bin/python -- override"
        err "  with --bridge-py /path/to/python if conda lives elsewhere."
        return 1
    fi
    local py_imports="import transformers, torch, zmq, msgpack, msgpack_numpy"
    if [[ "$SIM_MODE" -eq 1 ]]; then
        py_imports+=", mujoco"
    fi
    if ! "$BRIDGE_PY" -c "$py_imports" 2>/dev/null; then
        err "  BRIDGE_PY=${BRIDGE_PY} is missing one of:"
        if [[ "$SIM_MODE" -eq 1 ]]; then
            err "    transformers / torch / pyzmq / msgpack / msgpack-numpy / mujoco"
        else
            err "    transformers / torch / pyzmq / msgpack / msgpack-numpy"
        fi
        "$BRIDGE_PY" -c "$py_imports" 2>&1 | sed 's/^/    /' >&2 || true
        return 1
    fi
    if [[ "$SIM_MODE" -eq 0 ]]; then
        if ! PYTHONPATH="${REPO_ROOT}/external_dependencies/Isaac-GR00T:${REPO_ROOT}" \
                "$BRIDGE_PY" -c "from gear_sonic.camera.composed_camera import ComposedCameraClientSensor" 2>/dev/null; then
            err "  BRIDGE_PY=${BRIDGE_PY} cannot import gear_sonic.camera.composed_camera:"
            PYTHONPATH="${REPO_ROOT}/external_dependencies/Isaac-GR00T:${REPO_ROOT}" \
                "$BRIDGE_PY" -c "from gear_sonic.camera.composed_camera import ComposedCameraClientSensor" 2>&1 \
                | sed 's/^/    /' >&2 || true
            return 1
        fi
    fi
    ok "  bridge python OK."
}

preflight_sim_onnx() {
    log "[preflight] check sim SONIC ONNX ${SIM_MODEL} …"
    if [[ ! -f "$SIM_MODEL" ]]; then
        err "  --onnx ${SIM_MODEL} does not exist."
        return 1
    fi
    if [[ ! -x "$DEPLOY_SH" ]]; then
        err "  deploy script not executable: ${DEPLOY_SH}"
        return 1
    fi
    ok "  sim ONNX + deploy script present."
}

preflight_sim_ports_free() {
    log "[preflight] sim ZMQ ports :${LAPTOP_POSE_PORT} and :${PC2_DEBUG_PORT} must be free …"
    if ss -tln 2>/dev/null | grep -qE ":${LAPTOP_POSE_PORT}\b|:${PC2_DEBUG_PORT}\b"; then
        warn "  port(s) in use; stopping stale sim processes …"
        kill_stale_sim_processes
    fi
    if ss -tln 2>/dev/null | grep -qE ":${LAPTOP_POSE_PORT}\b|:${PC2_DEBUG_PORT}\b"; then
        err "  ports still bound after cleanup. Inspect with:"
        err "    ss -tlnp | grep -E ':${LAPTOP_POSE_PORT}|:${PC2_DEBUG_PORT}'"
        err "  Or run: ./gear_sonic/scripts/run_x2_vla_runtime.sh stop"
        return 1
    fi
    ok "  sim ZMQ ports free."
}

preflight_local_ports_free() {
    log "[preflight] local port :${LAPTOP_POSE_PORT} must be free …"
    if ss -tln 2>/dev/null | grep -qE ":${LAPTOP_POSE_PORT}\b"; then
        err "  port ${LAPTOP_POSE_PORT} is already bound on this laptop."
        err "  Check 'ss -tlnp | grep :${LAPTOP_POSE_PORT}' and stop the"
        err "  conflicting process (likely a stale bridge from a prior run)."
        return 1
    fi
    ok "  port :${LAPTOP_POSE_PORT} free."
}

autostart_pc2_cameras() {
    # Mirror the recording flow: ``--head-cameras`` in
    # run_x2_quest3_planner_stack.sh auto-runs x2_pc2_cameras.sh serve
    # over SSH and the bridge is deliberately left running after each
    # session. If we land here with cameras silent, it usually means
    # PC2 was rebooted or no one recorded today -- a single SSH'd
    # ``serve`` brings it back.
    local cam_sh="${REPO_ROOT}/gear_sonic_deploy/scripts/x2_pc2_cameras.sh"
    if [[ ! -x "$cam_sh" ]]; then
        err "  x2_pc2_cameras.sh not executable at $cam_sh"
        return 1
    fi
    log "  auto-starting PC2 camera bridge over SSH (idempotent; bounces" \
        "any prior instance) …"
    if X2_PC2_HOST="$PC2_HOST" X2_PC2_CAM_PORT="$PC2_CAMERAS_PORT" \
            "$cam_sh" serve --host "$PC2_HOST" --user "$PC2_USER" \
                            --port "$PC2_CAMERAS_PORT"; then
        ok "  PC2 camera bridge launched on tcp://${PC2_HOST}:${PC2_CAMERAS_PORT}"
        log "  re-probing camera stream (≤8s) …"
        if preflight_zmq_probe "$PC2_HOST" "$PC2_CAMERAS_PORT" "cameras (ZMQ)" 8; then
            return 0
        fi
        err "  PC2 camera bridge launched but stream still silent."
        err "  Check: '$cam_sh status --host $PC2_HOST' and"
        err "         '$cam_sh serve-log --host $PC2_HOST'"
        return 1
    fi
    err "  x2_pc2_cameras.sh serve failed (SSH or missing topics)."
    err "  Fix manually, or pass --no-cameras-autostart to skip the auto-launch."
    return 1
}

preflight_all() {
    local fails=0
    if [[ "$SIM_MODE" -eq 1 ]]; then
        log "preflight mode: SIM (local deploy_x2.sh + ghost cameras)"
        preflight_sim_ports_free     || fails=$((fails+1))
        preflight_sim_onnx           || fails=$((fails+1))
    else
        log "preflight mode: REAL ROBOT (PC2 daemons + ZMQ cameras)"
        preflight_pc2_reachable    || fails=$((fails+1))
        preflight_zmq_probe "$PC2_HOST" "$PC2_DEBUG_PORT"   "x2_debug"        6 || fails=$((fails+1))
        if ! preflight_zmq_probe "$PC2_HOST" "$PC2_CAMERAS_PORT" "cameras (ZMQ)" 8; then
            if [[ "$CAMERAS_AUTOSTART" -eq 1 ]]; then
                log "[preflight] cameras silent -- attempting auto-start …"
                if ! autostart_pc2_cameras; then
                    fails=$((fails+1))
                fi
            else
                err "  --no-cameras-autostart set; not auto-launching."
                err "  Run: gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve" \
                    "--host $PC2_HOST"
                fails=$((fails+1))
            fi
        fi
        preflight_local_ports_free     || fails=$((fails+1))
    fi
    preflight_model_dir            || fails=$((fails+1))
    preflight_modality_config      || fails=$((fails+1))
    if [[ -z "$MOTION_TOKEN_DECODER" ]]; then
        if [[ "$SIM_MODE" -eq 1 ]]; then
            err "  --motion-token-decoder is required in sim (body will not move without it)."
            fails=$((fails+1))
        else
            preflight_motion_token_decoder || true
        fi
    elif [[ ! -f "$MOTION_TOKEN_DECODER" ]]; then
        err "  motion-token decoder not found: $MOTION_TOKEN_DECODER"
        fails=$((fails+1))
    else
        ok "  motion-token decoder present: $MOTION_TOKEN_DECODER"
    fi
    preflight_bridge_py            || fails=$((fails+1))
    if [[ $fails -ne 0 ]]; then
        err "preflight FAILED with ${fails} issue(s). Fix the items above and rerun."
        err "(Bypass at your own risk with SKIP_PREFLIGHT=1; not recommended.)"
        return 1
    fi
    ok "preflight all checks passed."
}

spawn_sim_telemetry() {
    local dump_duration=$(( MAX_DURATION + 20 ))
    local dump_py="${REPO_ROOT}/.venv/bin/python"
    local default_pose_npy="${RUN_DIR}/default_pose_mj.npy"
    if [[ ! -x "$dump_py" ]]; then
        warn "telemetry dump skipped: ${dump_py} not found"
        return 0
    fi
    # Bake trained stand pose once per run so summary JSON can report drift.
    DEFAULT_POSE_NPY="$default_pose_npy" PYTHONPATH="${REPO_ROOT}" \
        "$dump_py" - <<'PY' 2>/dev/null || true
import numpy as np, os
from gear_sonic.utils.teleop.sonic_token_to_pose_decoder import X2_DEFAULT_ANGLES_MJ
np.save(os.environ["DEFAULT_POSE_NPY"], np.asarray(X2_DEFAULT_ANGLES_MJ, dtype=np.float32))
PY
    log "spawning x2_debug telemetry -> ${RUN_DIR}/x2_debug_trace.csv"
    nohup "$dump_py" -m gear_sonic.scripts.dump_x2_debug \
        --host localhost \
        --port "$PC2_DEBUG_PORT" \
        --topic x2_debug \
        --duration "$dump_duration" \
        --rate "$RATE" \
        --quiet \
        --default-pose "$default_pose_npy" \
        --json-out "${RUN_DIR}/x2_debug_summary.json" \
        --csv-out "${RUN_DIR}/x2_debug_trace.csv" \
        > "${RUN_DIR}/dump.log" 2>&1 &
    echo $! > "${RUN_DIR}/dump.pid"
    ok "  dump.pid = $(cat "${RUN_DIR}/dump.pid")"
}

print_sim_artifacts() {
    echo
    log "=== sim run artifacts (for iteration) ==="
    log "  bridge log       : ${LOG_FILE_BRIDGE}"
    log "  deploy log       : ${LOG_FILE_DEPLOY}"
    log "  x2_debug CSV     : ${RUN_DIR}/x2_debug_trace.csv"
    log "    (per-tick: body_q, body_dq, grav, last_action, control_tick)"
    log "  x2_debug summary : ${RUN_DIR}/x2_debug_summary.json"
    log "  VLA chunk dumps  : ${RUN_DIR}/vla_chunks/"
    log "  telemetry log    : ${RUN_DIR}/dump.log"
    if [[ -f "${RUN_DIR}/x2_debug_summary.json" ]]; then
        echo
        log "x2_debug run summary:"
        sed 's/^/    /' "${RUN_DIR}/x2_debug_summary.json" || true
    fi
    if compgen -G "${RUN_DIR}/vla_chunks/chunk_*.npz" >/dev/null 2>&1; then
        echo
        log "VLA chunk aggregates:"
        "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/inspect_vla_chunks.py" \
            "${RUN_DIR}/vla_chunks" --first 3 --last 3 2>/dev/null \
            | sed 's/^/    /' || warn "inspect_vla_chunks failed (see dump.log)"
    fi
    echo
    log "Quick plots: import ${RUN_DIR}/x2_debug_trace.csv in pandas/matplotlib"
    log "  or: python -c \"import pandas as pd; df=pd.read_csv('${RUN_DIR}/x2_debug_trace.csv'); print(df.describe())\""
}

spawn_sim_deploy() {
    local deploy_args=(
        sim --vla
        --model "$SIM_MODEL"
        --autostart-after "$SIM_AUTOSTART_AFTER"
        --max-duration "$MAX_DURATION"
        --no-confirm
        --vla-zmq-host 127.0.0.1
        --vla-zmq-port "$LAPTOP_POSE_PORT"
        --vla-zmq-topic pose
        --vla-debug-port "$PC2_DEBUG_PORT"
        --vla-debug-topic x2_debug
        --sim-profile "$SIM_PROFILE"
        # Pose-ref watchdog OFF for local sim: bridge cold-start + docker
        # SUB wiring can exceed the 0.5 s SAFE_IDLE trip (quest3/pkl paths
        # do the same). Without this the robot collapses before SONIC loads.
        --deploy-extra-arg --disable-pose-ref-watchdog
    )
    if [[ "$SIM_PROFILE" == "parity" ]]; then
        deploy_args+=(--motion "$SIM_RSI_PKL")
    fi
    if [[ "$SIM_VIEWER" -eq 1 ]]; then
        deploy_args+=(--sim-viewer)
    fi
    if [[ "$SIM_WITH_OMNIHAND" -eq 1 ]]; then
        deploy_args+=(--sim-with-omnihand)
    fi
    if [[ -n "${SIM_MAX_TARGET_DEV:-}" ]]; then
        deploy_args+=(--max-target-dev "$SIM_MAX_TARGET_DEV")
    fi
    if [[ -n "${SIM_DEPLOY_TARGET_LPF_HZ:-}" ]]; then
        deploy_args+=(--target-lpf-hz "$SIM_DEPLOY_TARGET_LPF_HZ")
    fi
    log "spawning sim deploy -> ${LOG_FILE_DEPLOY}"
    log "  CMD: bash ${DEPLOY_SH} ${deploy_args[*]}"
    nohup bash "$DEPLOY_SH" "${deploy_args[@]}" \
        > "$LOG_FILE_DEPLOY" 2>&1 &
    echo $! > "$PID_FILE_DEPLOY"
    ok "deploy.pid = $(cat "$PID_FILE_DEPLOY")"
}

# ----- subcommands ------------------------------------------------------

if [[ "$CMD" == "stop" ]]; then
    stop_all
    exit 0
fi

if [[ "$CMD" == "preflight" ]]; then
    log "preflight only -- no bridge will be started."
    preflight_all
    exit $?
fi

# ----- run --------------------------------------------------------------

mkdir -p "$RUN_DIR"

if [[ "$DEPRECATED_SONIC_CHECKPOINT_USED" -eq 1 ]]; then
    warn "--sonic-checkpoint / SONIC_CHECKPOINT is DEPRECATED on this launcher."
    warn "  The canonical name is --motion-token-decoder (env: MOTION_TOKEN_DECODER)."
    warn "  The old name is forwarded as the decoder path for now; update your"
    warn "  scripts to use --motion-token-decoder. See docs/source/references/"
    warn "  x2_sonic_runtime_architecture.md for the naming rationale."
fi

if [[ -z "$SKIP_PREFLIGHT" ]]; then
    if ! preflight_all; then
        exit 1
    fi
else
    warn "SKIP_PREFLIGHT=1 -- skipping connectivity / model / decoder probes."
    warn "  This bypasses the dry-run safety net. Do not use this on a powered robot"
    warn "  unless you have just run preflight successfully in this session."
fi

# Normalize deprecated body-mode aliases.
if [[ "$VLA_BODY_MODE" == "upper_body" ]]; then
    warn "body mode 'upper_body' is deprecated; use 'manipulation'."
    VLA_BODY_MODE="manipulation"
fi

# ----- safety banner ----------------------------------------------------

echo
if [[ "$SIM_MODE" -eq 1 ]]; then
    echo -e "${CYAN}${BOLD}===============================================================${NC}"
    echo -e "${CYAN}${BOLD}           VLA RUNTIME — SIMULATION (MuJoCo + ghost cams)      ${NC}"
    echo -e "${CYAN}${BOLD}===============================================================${NC}"
else
    echo -e "${RED}${BOLD}===============================================================${NC}"
    echo -e "${RED}${BOLD}                ABOUT TO GO LIVE ON THE REAL ROBOT             ${NC}"
    echo -e "${RED}${BOLD}===============================================================${NC}"
fi
cat <<EOF
  ${BOLD}Surface         ${NC}: $([[ "$SIM_MODE" -eq 1 ]] && echo "sim (local deploy + ghost cameras)" || echo "real robot (PC2 daemons)")
  ${BOLD}Run dir         ${NC}: ${RUN_DIR}
  ${BOLD}Model           ${NC}: ${MODEL_DIR}
  ${BOLD}Prompt          ${NC}: "${PROMPT}"
EOF
if [[ "$SIM_MODE" -eq 1 ]]; then
    cat <<EOF
  ${BOLD}Sim ONNX        ${NC}: ${SIM_MODEL}
  ${BOLD}Sim profile     ${NC}: ${SIM_PROFILE} (viewer=$([[ "$SIM_VIEWER" -eq 1 ]] && echo on || echo off))
  ${BOLD}x2_debug SUB    ${NC}: tcp://localhost:${PC2_DEBUG_PORT}
  ${BOLD}Cameras         ${NC}: ghost MuJoCo renderer (modality-driven stereo keys)
EOF
else
    cat <<EOF
  ${BOLD}PC2 host        ${NC}: ${PC2_HOST}
  ${BOLD}Pose PUB (LAN)  ${NC}: tcp://*:${LAPTOP_POSE_PORT}  (PC2 pose proxy SUBs here)
  ${BOLD}x2_debug SUB    ${NC}: tcp://${PC2_HOST}:${PC2_DEBUG_PORT}
  ${BOLD}Cameras SUB     ${NC}: tcp://${PC2_HOST}:${PC2_CAMERAS_PORT}
EOF
fi
cat <<EOF
  ${BOLD}Modality config ${NC}: ${MODALITY_CONFIG}
  ${BOLD}Body mode       ${NC}: ${VLA_BODY_MODE}$( [[ -n "$VLA_MODE_CONTROL_FILE" ]] && echo " (runtime switch: ${VLA_MODE_CONTROL_FILE})" )
  ${BOLD}Token decoder   ${NC}: ${MOTION_TOKEN_DECODER:-DISABLED (body will track idle_stand only)}
  ${BOLD}Wire safety     ${NC}: ramp ${VLA_RAMP_IN_TICKS}t bodyLPF ${VLA_TARGET_LPF_HZ}Hz futLPF ${VLA_FUTURE_LPF_HZ}Hz handLPF ${VLA_HAND_LPF_HZ}Hz handBlend ${VLA_HAND_CHUNK_BLEND_TICKS}t handStep≤${VLA_MAX_HAND_STEP} bodyStep≤${VLA_MAX_WIRE_STEP} body≤${VLA_MAX_WIRE_DEV_FROM_BODY} blend ${VLA_CHUNK_BLEND_TICKS}t
  ${BOLD}Deploy clamp    ${NC}: max-target-dev ${SIM_MAX_TARGET_DEV:-off} deploy-LPF ${SIM_DEPLOY_TARGET_LPF_HZ:-off} Hz (sim)
  ${BOLD}Rate            ${NC}: ${RATE} Hz publisher, ${INFERENCE_MIN_PERIOD_S}s min inference period
  ${BOLD}Max duration    ${NC}: ${MAX_DURATION} s
EOF
if [[ "$SIM_MODE" -eq 0 ]]; then
    echo "  ${BOLD}Camera staleness${NC}: ${CAMERAS_STALENESS_S} s"
fi
echo

if [[ -z "${FAST_ABORT:-}" ]]; then
    if [[ "$SIM_MODE" -eq 0 ]]; then
        echo -ne "  ${YELLOW}Ctrl-C now to abort; auto-continue in 5s${NC}"
        for _ in 1 2 3 4 5; do echo -n "."; sleep 1; done
        echo
    fi
fi

# ----- spawn the bridge -------------------------------------------------

BRIDGE_DURATION="$MAX_DURATION"
if [[ "$SIM_MODE" -eq 1 ]]; then
    BRIDGE_DURATION=0
fi

BRIDGE_ARGS=(
    -m gear_sonic.scripts.live_vla_publish_motion_token
    --model-path "$MODEL_DIR"
    --prompt "$PROMPT"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config "$MODALITY_CONFIG"
    --device "$VLA_DEVICE"
    --pub-host '*'
    --pub-port "$LAPTOP_POSE_PORT"
    --pub-topic pose
    --sub-host "$DEBUG_SUB_HOST"
    --sub-port "$PC2_DEBUG_PORT"
    --sub-topic x2_debug
    --rate "$RATE"
    --duration "$BRIDGE_DURATION"
    --inference-min-period-s "$INFERENCE_MIN_PERIOD_S"
    --print-every 50
    --render-width "$RENDER_WIDTH"
    --render-height "$RENDER_HEIGHT"
    --dump-chunks-dir "${RUN_DIR}/vla_chunks"
    --dump-chunks-every "$DUMP_CHUNKS_EVERY"
)
if [[ "$SIM_MODE" -eq 1 ]]; then
    BRIDGE_ARGS+=(--cameras-source ghost)
    if [[ -n "$VIDEO_OUT" ]]; then
        BRIDGE_ARGS+=(--video-out "$VIDEO_OUT")
    fi
    if [[ -n "$VIDEO_FRONT_OUT" ]]; then
        BRIDGE_ARGS+=(--video-front-out "$VIDEO_FRONT_OUT")
    fi
else
    BRIDGE_ARGS+=(
        --cameras-source zmq
        --cameras-zmq-host "$PC2_HOST"
        --cameras-zmq-port "$PC2_CAMERAS_PORT"
        --cameras-staleness-s "$CAMERAS_STALENESS_S"
        --cameras-warmup-s "$CAMERAS_WARMUP_S"
    )
fi
BRIDGE_ARGS+=(--vla-body-mode "$VLA_BODY_MODE")
if [[ -n "$VLA_FREEZE_BODY_GROUPS" ]]; then
    BRIDGE_ARGS+=(--vla-freeze-body-groups "$VLA_FREEZE_BODY_GROUPS")
fi
if [[ -n "$VLA_MODE_CONTROL_FILE" ]]; then
    BRIDGE_ARGS+=(--vla-mode-control-file "$VLA_MODE_CONTROL_FILE")
fi
BRIDGE_ARGS+=(
    --vla-hand-lpf-hz "$VLA_HAND_LPF_HZ"
    --vla-hand-chunk-blend-ticks "$VLA_HAND_CHUNK_BLEND_TICKS"
    --vla-max-hand-step "$VLA_MAX_HAND_STEP"
)
if [[ -n "$MOTION_TOKEN_DECODER" ]]; then
    BRIDGE_ARGS+=(
        --motion-token-decoder "$MOTION_TOKEN_DECODER"
        --sonic-decoder-device "$SONIC_DECODER_DEVICE"
        --vla-ramp-in-ticks "$VLA_RAMP_IN_TICKS"
        --vla-target-lpf-hz "$VLA_TARGET_LPF_HZ"
        --vla-future-lpf-hz "$VLA_FUTURE_LPF_HZ"
        --vla-max-wire-dev-from-body "$VLA_MAX_WIRE_DEV_FROM_BODY"
        --vla-max-wire-step "$VLA_MAX_WIRE_STEP"
        --vla-chunk-blend-ticks "$VLA_CHUNK_BLEND_TICKS"
        --vla-max-action-il "$VLA_MAX_ACTION_IL"
        --vla-decode-delay-ticks "$VLA_DECODE_DELAY_TICKS"
    )
elif [[ "$VLA_BODY_MODE" == "locomotion" && -z "$MOTION_TOKEN_DECODER" ]]; then
    : # no decoder: body tracks idle_stand; hands move only if VLA runs
fi
# Forward any unrecognised CLI tail as passthrough to the bridge.
if [[ ${#ARGS[@]} -gt 0 ]]; then
    BRIDGE_ARGS+=("${ARGS[@]}")
fi

mkdir -p "${RUN_DIR}/vla_chunks"

log "spawning bridge -> ${LOG_FILE_BRIDGE}"
log "  CMD: ${BRIDGE_PY} ${BRIDGE_ARGS[*]}"

# TQDM_DISABLE=1: stdout is redirected to bridge.log, so any tqdm
# bar (HF safetensors materialisation, accelerate load_checkpoint,
# anywhere downstream) writes carriage-returns into the file instead
# of overwriting the same terminal line. Result: ~thousand
# "Loading weights: 100% ..." snippets glued into one logical log
# line, drowning out actual signal. Disabling tqdm globally for the
# bridge subprocess keeps the log scannable; the operator-facing
# launcher messages still emit normally.
PYTHONPATH="${REPO_ROOT}/external_dependencies/Isaac-GR00T:${REPO_ROOT}" \
MUJOCO_GL=egl \
TQDM_DISABLE=1 \
nohup "$BRIDGE_PY" "${BRIDGE_ARGS[@]}" \
    > "$LOG_FILE_BRIDGE" 2>&1 &
BRIDGE_PID=$!
echo "$BRIDGE_PID" > "$PID_FILE_BRIDGE"

cleanup() {
    log "caught signal — tearing down …"
    stop_all
}
trap cleanup INT TERM

log "bridge.pid = ${BRIDGE_PID}"
log "  log     = ${LOG_FILE_BRIDGE}"
log "  chunks  = ${RUN_DIR}/vla_chunks/"
log "waiting for bridge pose PUB bind marker (≤180s) …"
if ! wait_for_log_marker "$LOG_FILE_BRIDGE" "$BRIDGE_PID" \
        "pose PUB bound on" 180 "bridge"; then
    err "bridge never bound pose PUB — check $LOG_FILE_BRIDGE"
    stop_all
    exit 1
fi
ok "bridge PUB live (bootstrap idle_stand wire active)"

if [[ "$SIM_MODE" -eq 1 ]]; then
    log "waiting for bridge model load (policy ready, ≤180s) …"
    if ! wait_for_log_marker "$LOG_FILE_BRIDGE" "$BRIDGE_PID" \
            "policy ready" 180 "bridge"; then
        err "bridge never finished GR00T load — check $LOG_FILE_BRIDGE"
        stop_all
        exit 1
    fi
    ok "bridge policy ready; spawning sim deploy"

    if ! ensure_parity_rsi_pkl; then
        stop_all
        exit 1
    fi

    spawn_sim_deploy

    DEPLOY_PID="$(cat "$PID_FILE_DEPLOY")"
    log "waiting for deploy Launching marker (≤180s) …"
    if ! wait_for_log_marker "$LOG_FILE_DEPLOY" "$DEPLOY_PID" \
            "Launching ..." 180 "deploy"; then
        stop_all
        exit 1
    fi
    ok "deploy READY (SONIC sim loaded); settle 2s before telemetry"
    sleep 2
    spawn_sim_telemetry
fi

ok "bridge live; following logs (Ctrl-C to stop)"
echo

if [[ "$SIM_MODE" -eq 1 && -f "$PID_FILE_DEPLOY" ]]; then
    DEPLOY_PID="$(cat "$PID_FILE_DEPLOY")"
    tail -n 0 -F --pid="$DEPLOY_PID" "$LOG_FILE_BRIDGE" 2>/dev/null \
        | stdbuf -oL grep -E 'pub tick|inference|ghost|raw_Δ|wire_Δ|deploy_alive|render error' \
        | sed -u 's/^/[bridge] /' &
    TAIL_BRIDGE_PID=$!
    tail -n 0 -F --pid="$DEPLOY_PID" "$LOG_FILE_DEPLOY" 2>/dev/null \
        | stdbuf -oL grep -E 'CONTROL|grav_z|HANDOFF|band release|tilt|max-duration' \
        | sed -u 's/^/[deploy] /' &
    TAIL_DEPLOY_PID=$!

    wait "$DEPLOY_PID" 2>/dev/null || true
    EXIT_CODE=$?

    log "deploy exited; tearing down …"
    kill "$TAIL_BRIDGE_PID" "$TAIL_DEPLOY_PID" 2>/dev/null || true
    wait "$TAIL_BRIDGE_PID" "$TAIL_DEPLOY_PID" 2>/dev/null || true
    # Let the telemetry dumper flush its CSV/JSON (runs MAX_DURATION+20s).
    if [[ -f "${RUN_DIR}/dump.pid" ]]; then
        dump_pid="$(cat "${RUN_DIR}/dump.pid")"
        dump_wait=$(( MAX_DURATION + 25 ))
        for _ in $(seq 1 "$dump_wait"); do
            kill -0 "$dump_pid" 2>/dev/null || break
            sleep 1
        done
    fi
    stop_all
    print_sim_artifacts
else
    tail -F --pid="$BRIDGE_PID" "$LOG_FILE_BRIDGE" &
    TAIL_PID=$!
    wait "$BRIDGE_PID" 2>/dev/null || true
    EXIT_CODE=$?
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
fi

if [[ $EXIT_CODE -eq 0 ]]; then
    ok "run finished cleanly."
else
    warn "run exited with code $EXIT_CODE -- check ${LOG_FILE_BRIDGE}"
fi
if [[ "$SIM_MODE" -eq 1 ]]; then
    log "run dir kept at ${RUN_DIR} for postmortem / iteration"
else
    log "run dir kept at ${RUN_DIR} for postmortem"
fi
exit "$EXIT_CODE"
