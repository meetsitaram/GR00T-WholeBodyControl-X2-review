#!/usr/bin/env bash
# X2 Quest 3 planner-driven teleop / record stack runner (Phase 0).
#
# Spawns child processes in a safe order with readiness markers between
# steps and a trap-cleaned reverse-order shutdown.
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
#       [--sonic-checkpoint PATH] [--no-sonic-checkpoint]
#       [--sonic-tokenizer-device DEV] [--encoder-config PATH]
#       [--planner-demo PATH.yaml]
#       [--rate FLOAT] [--log-dir PATH] [--cleanup-only] [--validate-only]
#       [--vla-bridge MODEL_DIR --vla-prompt STR
#        [--vla-device DEV] [--vla-rate FLOAT] [--vla-inference-period-s S]
#        [--vla-python PATH] [--vla-max-target-dev RAD] [--vla-target-lpf-hz HZ]]
#
# Defaults to --teleop-only (no dataset writes). Pass --with-record
# (along with --output-dir and --task) to capture a LeRobot v2.1
# episode through the subscribe-mode pipeline.
#
# Scripted-motion mode: pass --planner-demo PATH.yaml to pre-load a
# scripted-demo YAML (same schema as gear_sonic/data/scripted_demos/
# *.yaml; see x2_heuristic_planner.md "Scripted demo gallery") into
# the planner's command queue at startup. Once the deploy + planner +
# manager + recorder come up and the warmup completes, the planner
# plays through the YAML commands in order; when the queue empties it
# falls back to IDLE_LOOP (idle_stand) and stays there until the
# operator takes over via the Quest 3 headset (button chord A+B+X+Y
# enters LOCOMOTION). The first VR-driven planner_cmd that arrives
# during playback REPLACES the pending YAML queue (the operator's
# intent always wins), so a half-played demo is interruptable.
# Mutually exclusive with --vla-bridge (VLA mode replaces the
# planner with the VLA bridge; there's no command queue to seed).
#
# VLA closed-loop mode: pass --vla-bridge MODEL_DIR --vla-prompt STR
# (with --robocasa-env, which is REQUIRED here). The heuristic planner
# is omitted; the wrapper spawns deploy + quest3_manager_x2 +
# record_x2_dataset (subscribe, --teleop-only) + live_vla_publish_motion_token.
# The bridge PUBs ``body_pose`` on :5565 like the planner; the recorder
# merges Quest/manager streams and PUBs ``pose`` on :5556 to the deploy,
# including the same idle-stand pre-body_pose behaviour as Phase 0.
# The Quest 3 headset is optional (manager waits internally until WebXR
# connects; without a headset the recorder uses the VLA body's arm slice).
#
# Examples:
#   # Smoke test (no writes), 5 min teleop session:
#   ./run_x2_quest3_planner_stack.sh --duration 300
#
#   # Run forever (no auto-shutdown; Ctrl-C to stop):
#   ./run_x2_quest3_planner_stack.sh --duration 0
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
#   # Closed-loop VLA demo (deploy + manager + recorder merge pipe +
#   # VLA bridge on body_pose; no heuristic planner; headset optional):
#   ./run_x2_quest3_planner_stack.sh --duration 60 \
#       --robocasa-env X2PickPlaceApple \
#       --vla-bridge /tmp/x2_pick_place_apple_v1_run1 \
#       --vla-prompt "pick up the apple from the table"
#
#   # Play a scripted YAML demo at startup, then idle-stand and wait
#   # for VR takeover (operator straps on the headset and chord-presses
#   # A+B+X+Y to enter LOCOMOTION whenever they're ready):
#   ./run_x2_quest3_planner_stack.sh --duration 0 \
#       --planner-demo gear_sonic/data/scripted_demos/eleven_motion_sequence.yaml
#
# Pre-flight (verified before any spawn):
#   - data/operator_calibrations/<operator-id>.yaml exists
#   - Ports 5556 (recorder pose PUB), 5557 (deploy x2_debug), 5564
#     (manager arm/hands PUB), 5565 (VLA body_pose PUB) are free; in
#     VLA mode 5563 is unused (no planner). In --robocasa-env mode also
#     5559 + 5560.
#   - The ONNX model file exists (when deploy is being spawned)
#   - The planner primitives PKL + bins YAML exist
#   - The scene MJCF exists (when --robocasa-env is set); build with:
#       python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env <ENV>
#   - --with-record requires --output-dir; also requires --task UNLESS
#     --robocasa-env is set (then it auto-fills from scene metadata)
#   - SONIC tokenizer .pt is auto-resolved from --model (strip
#     /exported/ + _g1.onnx, append .pt). Required to be present unless
#     --no-sonic-checkpoint is passed (smoke-test escape hatch -- the
#     resulting dataset's action.motion_token will be all zeros and is
#     NOT VLA-trainable). Override the auto-resolved path explicitly
#     with --sonic-checkpoint PATH.
#   - Encoder-observation YAML
#     (gear_sonic/data/encoder/x2_observation_config.yaml) -- pinned
#     by default; the recorder's inline tokenizer uses it to build the
#     same 680-D 10-frame future window the deploy actor's internal
#     encoder consumes. Pass --encoder-config '' to deliberately fall
#     back to the deprecated freeze-pose path (kept for backward compat
#     with the v0 direct-mode loop; semantically incorrect for VLA).
#   - VLA mode: --vla-bridge MODEL_DIR must point at a HuggingFace-style
#     finetune checkpoint (model.safetensors + processor/ + experiment
#     _cfg/). --vla-prompt is required. --robocasa-env must be set so
#     the deploy spawns with the same scene the model was trained on.
#     The bridge is launched out of the env_isaaclab conda env (override
#     with --vla-python /path/to/python). Planner + primitives preflight
#     are skipped; manager + recorder still run (subscribe merge to :5556).
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

DURATION_S=0  # 0 = unlimited (run until Ctrl-C). Pass --duration N for a fixed N-sec cap.
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
SIM_PROFILE_EXPLICIT=0   # set to 1 when the operator passes --sim-profile
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
# SONIC inline tokenizer (.pt) settings. SONIC_CHECKPOINT_MODE:
#   "auto"     -- resolve sibling .pt from --model (default)
#   "explicit" -- operator passed --sonic-checkpoint PATH
#   "off"      -- operator passed --no-sonic-checkpoint (escape hatch
#                 for smoke tests; resulting dataset is not VLA-trainable)
SONIC_CHECKPOINT_MODE="auto"
SONIC_CHECKPOINT=""           # resolved during preflight (auto or explicit)
SONIC_TOKENIZER_DEVICE="cpu"
# Recorder's inline OnlineSonicTokenizer was previously cuda:0, but the .venv
# PyTorch (2.6.0+cu124) is built for sm_50..sm_90 and crashes on Blackwell
# (RTX 5090, sm_120) with 'no kernel image is available for execution on the
# device' the very first time the encoder runs (~1 s after first body_pose).
# CPU adds ~1 ms per tick which is well under the 20 ms 50 Hz budget per the
# OnlineSonicTokenizer module docstring; pass --sonic-tokenizer-device cuda:0
# explicitly if you've upgraded .venv to a Blackwell-capable PyTorch.
# Encoder-observation YAML: drives the inline tokenizer's gather (the
# 680-D 10-frame future window the SONIC encoder was trained on). The
# canonical config ships at gear_sonic/data/encoder/x2_observation_
# config.yaml. Pass --encoder-config '' to fall back to the deprecated
# freeze-pose path (current body_q tiled 11 times).
ENCODER_CONFIG="${REPO_ROOT}/gear_sonic/data/encoder/x2_observation_config.yaml"
OPERATOR_ID="default"
CALIBRATION_PATH=""     # resolved after CLI parse from OPERATOR_ID
QUEST3_WS_PORT=8765
QUEST3_HTTP_PORT=8443
LOG_DIR=""              # auto-resolved to /tmp/<script>-<timestamp>
CLEANUP_ONLY=0
VALIDATE_ONLY=0         # exit 0 right after pre-flight; spawns nothing
SIDECAR_LOG=""          # default: <log-dir>/manager_sidecar.jsonl
# Optional scripted-demo YAML pre-loaded into the planner's command
# queue at startup. When set the planner is spawned with --demo PATH,
# which appends the YAML's commands to the planner queue at boot. The
# state machine then plays them through (returning to idle_stand at
# the end of the queue) before sitting in IDLE_LOOP waiting for the
# manager's planner_cmd stream. The first interactive VR command
# (source="zmq" via replace_pending) drops any still-pending YAML
# entries, so a half-played demo is preemptible the moment the
# operator takes manual control. Mutually exclusive with VLA mode
# (no heuristic planner is spawned in that path).
PLANNER_DEMO=""

# --------------------------------------------------------------------------
# VLA closed-loop mode (NEW). When --vla-bridge MODEL_DIR is passed,
# the wrapper REPLACES the heuristic_planner + quest3_manager +
# recorder trio with a single live_vla_publish_motion_token bridge
# process, reusing all the deploy / robocasa / scene-XML / port-
# preflight infra. --vla-prompt is required; --robocasa-env is required
# (must match the scene the model was trained on); the Quest 3 headset
# is NOT used. Defaults match the proven values from
# run_live_vla_demo.sh (max_target_dev=0.10 -> 6 deg per-joint clamp,
# target_lpf_hz=4.0 -> rolls off the inference chunk-boundary
# saw-tooth, inference_period=0.8 s -> consumes the full 40-step
# horizon at 50 Hz).
# --------------------------------------------------------------------------
VLA_BRIDGE_MODEL=""             # presence enables VLA mode
# Bridge-side SONIC token decoder. Routes the VLA's predicted
# motion_token chunks through the same g1_dyn decoder the deploy ONNX
# uses internally and publishes the result as joint_pos_mj on the wire,
# so the body actually moves under VLA authority (the C++ deploy
# explicitly ignores the wire's motion_token field per
# zmq_pose_input_source.hpp:22-25). Empty + auto-resolve from SIM_MODEL
# below; pass --vla-bridge-sonic-checkpoint /path/to/.pt to override
# explicitly, or --no-vla-bridge-sonic-checkpoint to disable (body will
# stay at idle_stand even with VLA running).
VLA_BRIDGE_SONIC_CKPT=""
VLA_BRIDGE_SONIC_CKPT_MODE="auto"  # {auto, explicit, off}
VLA_BRIDGE_SONIC_DECODER_DEVICE="cpu"  # decoder is ~5 M params, sub-ms on CPU
VLA_PROMPT=""                   # required when VLA mode is active
VLA_DEVICE="cuda:0"
VLA_BRIDGE_RATE="50"
VLA_INFERENCE_PERIOD_S="0.8"
# Per-inference VLA I/O dump. When set, the bridge writes a .npz per
# Nth chunk containing both the *input* sent to GR00T (ego_view RGB,
# body_q_mj, base_quat_wxyz, hand state) and the *output* it produced
# (motion_token chunk, left/right hand chunks). Useful for offline
# inspection when "the robot stands but doesn't move" -- you can see
# whether the VLA is producing meaningful tokens or just rehearsing
# idle. Empty = no dump. Default below: auto-enables to LOG_DIR/vla_chunks
# whenever a log dir is provided.
VLA_DUMP_CHUNKS_DIR=""
VLA_DUMP_CHUNKS_EVERY="1"
VLA_BRIDGE_PYTHON=""            # auto-resolves to env_isaaclab/bin/python
VLA_MAX_TARGET_DEV="0.10"       # forwarded to deploy --max-target-dev
VLA_TARGET_LPF_HZ="4.0"         # forwarded to deploy --target-lpf-hz
# Deploy-side smoothing filters in VLA mode. Off by default since
# 2026-05-14 (PM): empirically the LPF + clamp prevent the policy from
# reacting fast enough to gravity drift during the ~22s VLA cold-start
# window, so the robot collapses at t=5..6s. Opt back in (e.g. for a
# well-trained checkpoint that produces visible chunk-boundary
# saw-tooth) via --vla-deploy-filters.
VLA_DEPLOY_FILTERS=0
VLA_NO_POLICY=0                 # 1 -> bridge runs publisher + x2_debug SUB only,
                                #      no Gr00tPolicy load / inference. Use to
                                #      validate the deploy / recorder sequence
                                #      without paying the model-load cost or
                                #      letting policy actions move the robot.

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
        --sim-profile) SIM_PROFILE="$2"; SIM_PROFILE_EXPLICIT=1; shift 2 ;;
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
        --sonic-checkpoint)
            SONIC_CHECKPOINT_MODE="explicit"
            SONIC_CHECKPOINT="$2"
            shift 2 ;;
        --no-sonic-checkpoint)
            SONIC_CHECKPOINT_MODE="off"
            SONIC_CHECKPOINT=""
            shift ;;
        --sonic-tokenizer-device) SONIC_TOKENIZER_DEVICE="$2"; shift 2 ;;
        --encoder-config) ENCODER_CONFIG="$2"; shift 2 ;;
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
        --planner-demo) PLANNER_DEMO="$2"; shift 2 ;;
        --vla-bridge) VLA_BRIDGE_MODEL="$2"; shift 2 ;;
        --vla-bridge-sonic-checkpoint)
            VLA_BRIDGE_SONIC_CKPT_MODE="explicit"
            VLA_BRIDGE_SONIC_CKPT="$2"
            shift 2
            ;;
        --no-vla-bridge-sonic-checkpoint)
            VLA_BRIDGE_SONIC_CKPT_MODE="off"
            VLA_BRIDGE_SONIC_CKPT=""
            shift
            ;;
        --vla-bridge-sonic-decoder-device)
            VLA_BRIDGE_SONIC_DECODER_DEVICE="$2"
            shift 2
            ;;
        --vla-prompt) VLA_PROMPT="$2"; shift 2 ;;
        --vla-device) VLA_DEVICE="$2"; shift 2 ;;
        --vla-rate) VLA_BRIDGE_RATE="$2"; shift 2 ;;
        --vla-inference-period-s) VLA_INFERENCE_PERIOD_S="$2"; shift 2 ;;
        --vla-dump-chunks-dir) VLA_DUMP_CHUNKS_DIR="$2"; shift 2 ;;
        --vla-dump-chunks-every) VLA_DUMP_CHUNKS_EVERY="$2"; shift 2 ;;
        --vla-python) VLA_BRIDGE_PYTHON="$2"; shift 2 ;;
        --vla-max-target-dev) VLA_MAX_TARGET_DEV="$2"; VLA_DEPLOY_FILTERS=1; shift 2 ;;
        --vla-target-lpf-hz) VLA_TARGET_LPF_HZ="$2"; VLA_DEPLOY_FILTERS=1; shift 2 ;;
        --vla-deploy-filters) VLA_DEPLOY_FILTERS=1; shift ;;
        --vla-no-policy) VLA_NO_POLICY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

# Convenience boolean: set iff --vla-bridge MODEL_DIR was passed OR
# --vla-no-policy was passed (latter is a deploy-sequence smoke test
# that uses the bridge as a planner-shaped idle source without loading
# the model). Used everywhere downstream to gate planner-trio code paths.
if [[ -n "${VLA_BRIDGE_MODEL}" || "${VLA_NO_POLICY}" -eq 1 ]]; then
    VLA_MODE=1
else
    VLA_MODE=0
fi

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
# Cleanup trap (set BEFORE any child is spawned). Reverse-order shutdown
# matches spawn order last-to-first. Phase 0: recorder -> manager ->
# planner -> deploy. VLA: recorder -> VLA bridge -> manager -> deploy.
# --------------------------------------------------------------------------

DEPLOY_PID=""
DEPLOY_LOG=""
PLANNER_PID=""
MANAGER_PID=""
RECORDER_PID=""
RECORDER_PGID=""
VLA_BRIDGE_PID=""
VLA_BRIDGE_PGID=""

cleanup_children() {
    log "shutting down children (reverse spawn order)..."
    if [[ -n "${RECORDER_PGID}" ]]; then
        # Recorder wrapped in setsid -> pgid==pid. Use SIGINT+grace so
        # the LeRobot writer can flush a last episode.
        kill_pgid_graceful "${RECORDER_PGID}" "recorder" 8
    fi
    kill_pid_quiet "${MANAGER_PID}"  "manager"
    kill_pid_quiet "${PLANNER_PID}"  "planner"
    # VLA bridge runs in its own session (setsid) so we can SIGINT the
    # whole process group -- the bridge spawns helper threads that
    # render videos / write MP4s / etc, and a clean SIGINT lets them
    # close encoders. Grace 5s.
    if [[ -n "${VLA_BRIDGE_PGID}" ]]; then
        kill_pgid_graceful "${VLA_BRIDGE_PGID}" "vla-bridge" 5
    fi
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

# Planner primitives + bins are only consumed by the heuristic_planner
# (Step 2/4) and the parity-mode RSI bake (Step 1/4 sub-step). In VLA
# mode the planner is replaced and the deploy is forced to non-parity
# (no RSI PKL needed) -- skip the existence checks so a fresh checkout
# missing curated primitives can still drive a closed-loop demo.
if [[ "${VLA_MODE}" -eq 0 ]]; then
    if [[ ! -f "${PRIMITIVES_PKL}" ]]; then
        err "primitives PKL not found: ${PRIMITIVES_PKL}"
        err "run: ${PYTHON} -m gear_sonic.scripts.curate_x2_primitives"
        exit 1
    fi
    if [[ ! -f "${BINS_YAML}" ]]; then
        err "bins YAML not found: ${BINS_YAML}"
        exit 1
    fi
fi
# Operator calibration is consumed by quest3_manager_x2 (Phase 0 step 3
# and VLA mode step 2). Required even when the Quest is not connected —
# the manager still loads the YAML for retarget defaults.
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
# --planner-demo: validate YAML existence + reject in VLA mode (no
# heuristic planner is spawned there; the bridge owns body_pose and
# has no command queue to seed). Keep this gate close to the rest of
# the input-file checks so a stale absolute path fails fast at boot
# rather than after deploy / planner spawn (which would burn the
# ~30 s sim docker bring-up before surfacing the typo).
if [[ -n "${PLANNER_DEMO}" ]]; then
    if [[ "${VLA_MODE}" -eq 1 ]]; then
        err "--planner-demo is incompatible with --vla-bridge / --vla-no-policy"
        err "(VLA mode replaces the heuristic planner with the live VLA bridge;"
        err "there is no planner command queue to seed with a YAML demo)."
        exit 1
    fi
    if [[ ! -f "${PLANNER_DEMO}" ]]; then
        err "--planner-demo YAML not found: ${PLANNER_DEMO}"
        err "Available curated demos:"
        for f in "${REPO_ROOT}"/gear_sonic/data/scripted_demos/*.yaml; do
            [[ -f "$f" ]] && err "  - ${f#${REPO_ROOT}/}"
        done
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# VLA-mode preflight. Runs AFTER the deploy / record gate above so the
# operator gets the standard "missing --output-dir" / "missing model"
# errors first if applicable.
# --------------------------------------------------------------------------
if [[ "${VLA_MODE}" -eq 1 ]]; then
    if [[ -z "${VLA_PROMPT}" && "${VLA_NO_POLICY}" -eq 0 ]]; then
        err "--vla-bridge requires --vla-prompt STR (the language instruction"
        err "passed to the VLA on every inference; e.g. \"pick up the apple from the table\")"
        err "Pass --vla-no-policy if you only want the bridge's idle wire (no inference)."
        exit 1
    fi
    if [[ "${VLA_NO_POLICY}" -eq 0 && ! -d "${VLA_BRIDGE_MODEL}" ]]; then
        err "--vla-bridge model directory not found: ${VLA_BRIDGE_MODEL}"
        err "Expected a HuggingFace-style fine-tune dir containing"
        err "  model.safetensors  processor/  experiment_cfg/"
        err "Pass --vla-no-policy if you only want the bridge's idle wire (no inference)."
        exit 1
    fi
    if [[ "${ROBOCASA_ENV}" == "none" && "${VLA_NO_POLICY}" -eq 0 ]]; then
        err "--vla-bridge requires --robocasa-env (must match the scene the model"
        err "was trained on, e.g. --robocasa-env X2PickPlaceApple)"
        err "Pass --vla-no-policy if you only want the bridge's idle wire (no inference);"
        err "the scene match is irrelevant when no model is loaded, and dropping the"
        err "robocasa env removes the table that was propping up a falling robot."
        exit 1
    fi
    if [[ "${WITH_RECORD}" -eq 1 ]]; then
        warn "--with-record is ignored in --vla-bridge mode (this is a closed-loop"
        warn "demo, not a data-capture session)."
        WITH_RECORD=0
    fi
    # Resolve the bridge python. Defaults to env_isaaclab/bin/python --
    # same convention as run_live_vla_demo.sh (see CONDA_ENV_BRIDGE
    # there). We use an absolute path so an active .venv in the parent
    # shell can't shadow the conda interpreter.
    if [[ -z "${VLA_BRIDGE_PYTHON}" ]]; then
        VLA_BRIDGE_PYTHON="${HOME}/miniconda3/envs/env_isaaclab/bin/python"
    fi
    if [[ ! -x "${VLA_BRIDGE_PYTHON}" ]]; then
        err "VLA bridge python not found / not executable: ${VLA_BRIDGE_PYTHON}"
        err "Pass --vla-python /path/to/env_isaaclab/bin/python (or activate"
        err "the env and re-run)."
        exit 1
    fi
    # Force SONIC tokenizer + encoder-config OFF in VLA mode -- the
    # recorder runs --teleop-only (no dataset); loading the tokenizer
    # would add startup cost for no parquet labels.
    SONIC_CHECKPOINT_MODE="off"
    SONIC_CHECKPOINT=""
    ENCODER_CONFIG=""

    # Bridge-side SONIC token decoder resolution. Independent of the
    # recorder's tokenizer (which is forced OFF above): we still need
    # the same .pt's *decoder* weights to translate the VLA's predicted
    # motion_token back into joint_pos_mj on the wire. Without this the
    # C++ deploy would re-tokenise idle_stand on every tick (the wire's
    # motion_token field is documented as "logged but otherwise unused"
    # in zmq_pose_input_source.hpp:22-25), and the body would never move.
    case "${VLA_BRIDGE_SONIC_CKPT_MODE}" in
        auto)
            # Mirror the recorder's auto-resolve: strip /exported/ from
            # the SIM_MODEL ONNX path and replace _g1.onnx with .pt.
            VLA_BRIDGE_SONIC_CKPT="${SIM_MODEL/\/exported\//\/}"
            VLA_BRIDGE_SONIC_CKPT="${VLA_BRIDGE_SONIC_CKPT%_g1.onnx}.pt"
            if [[ ! -f "${VLA_BRIDGE_SONIC_CKPT}" ]]; then
                err "Bridge-side SONIC .pt not found at auto-resolved path:"
                err "  ${VLA_BRIDGE_SONIC_CKPT}"
                err "(resolved from --model ${SIM_MODEL})"
                err ""
                err "Without it the bridge will publish idle_stand for"
                err "joint_pos_mj on every tick (motion_token is logged but"
                err "ignored by the C++ deploy), and the body will NOT move"
                err "under VLA control. Hand DOFs (AimDK passthrough) still"
                err "follow the VLA chunk."
                err ""
                err "Fix one of:"
                err "  - point the .pt at the right place via:"
                err "      --vla-bridge-sonic-checkpoint /path/to/model_step_NNNNN.pt"
                err "  - opt out (smoke tests only) via:"
                err "      --no-vla-bridge-sonic-checkpoint"
                exit 1
            fi
            ;;
        explicit)
            if [[ ! -f "${VLA_BRIDGE_SONIC_CKPT}" ]]; then
                err "Bridge-side SONIC .pt not found at explicit path:"
                err "  ${VLA_BRIDGE_SONIC_CKPT}"
                exit 1
            fi
            ;;
        off)
            VLA_BRIDGE_SONIC_CKPT=""
            ;;
        *)
            err "internal: unexpected VLA_BRIDGE_SONIC_CKPT_MODE=${VLA_BRIDGE_SONIC_CKPT_MODE}"
            exit 1
            ;;
    esac
    if [[ "${VLA_NO_POLICY}" -eq 1 ]]; then
        # No VLA inference -> nothing to decode -> bridge keeps the
        # idle_stand wire (which is what makes --vla-no-policy stable).
        VLA_BRIDGE_SONIC_CKPT=""
    fi
    # Sim profile remap for VLA mode.
    #
    # Historically VLA mode forced parity -> handoff because the bridge
    # only published a static DEFAULT_STAND_POSE wire reference and could
    # not byte-match the planner's idle_stand[0] anchor that parity spawns
    # at -- a parity spawn would have caused a same-tick fight on the legs
    # (~33 deg knee delta) and an ankle slap.
    #
    # Since 2026-05-14 the bridge replays the planner's idle_stand clip
    # via _IdleStandLoop (load_idle_stand_loop -> yaw_align_segment), so
    # the wire content is byte-equivalent to what --planner-only emits.
    # That makes parity spawn = wire content, exactly the same invariant
    # the heuristic planner relies on for upright stand.
    #
    # As of 2026-05-14 (afternoon): the auto-remap to handoff is REMOVED.
    # The trigger here is the same fall mode we hit in --vla-no-policy
    # before parity/no-LPF/no-clamp was made the default for that mode.
    # Under handoff the elastic band auto-releases ~4 s after the first
    # deploy command (deploy log: "auto-release 4s after first deploy
    # command"), and from that tick onward the deploy's onboard SONIC
    # tracking policy is the ONLY thing keeping the robot up. With
    # --max-target-dev 0.10 + --target-lpf-hz 4.0 active the SONIC
    # policy's per-tick action delta is clamped to ~6 deg and low-passed
    # at 4 Hz, which empirically is too sluggish for it to compensate
    # for the post-band gravity drift on top of the idle_stand reference
    # -- robot tipped at t=5..6 s and settled at grav_z=-0.90 propped on
    # the table (verified 2026-05-14 with /tmp/x2_vla_viewer). The VLA
    # is not implicated: the bridge cold-start (~22 s to inference
    # #0001) does overlap the band release window, but the same
    # configuration also fell in --vla-no-policy mode where the VLA is
    # never even loaded -- it's purely the SONIC policy being unable to
    # react fast enough through the LPF + clamp.
    #
    # Empirical fix: match --vla-no-policy's deploy config (parity, no
    # LPF, no clamp) so VLA mode has the same chance of standing through
    # the cold-start window as --vla-no-policy does. Pass --sim-profile
    # handoff explicitly if you specifically want the gantry+band
    # bring-up (e.g. real-robot prep).
    :  # parity is honoured as-is in VLA mode now
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
# In VLA mode the bridge PUBs body_pose on :5565 (planner role); the
# recorder PUBs pose on :5556; manager still PUBs arm/hands on :5564.
# Port 5563 (planner_cmd) is unused (no heuristic planner).
declare -A PORT_LABELS=(
    ["${POSE_PORT}"]="recorder->deploy pose (5556)"
)
PORTS_TO_CHECK=("${POSE_PORT}")
if [[ "${VLA_MODE}" -eq 0 ]]; then
    PORT_LABELS["${PLANNER_CMD_PORT}"]="manager->planner planner_cmd (5563)"
    PORT_LABELS["${ARM_HANDS_PORT}"]="manager->recorder arm/hands (5564)"
    PORT_LABELS["${BODY_POSE_PORT}"]="planner->recorder body_pose (5565)"
    PORTS_TO_CHECK+=("${PLANNER_CMD_PORT}" "${ARM_HANDS_PORT}" "${BODY_POSE_PORT}")
else
    PORT_LABELS["${ARM_HANDS_PORT}"]="manager->recorder arm/hands (5564)"
    PORT_LABELS["${BODY_POSE_PORT}"]="vla-bridge->recorder body_pose (5565)"
    PORTS_TO_CHECK+=("${ARM_HANDS_PORT}" "${BODY_POSE_PORT}")
fi
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
# SONIC tokenizer .pt resolution + preflight.
#
# The recorder's inline OnlineSonicTokenizer (encodes commanded body_q
# into action.motion_token for VLA training) needs the .pt sibling of
# the deploy ONNX. Convention:
#   ONNX:  /.../<run>/exported/model_step_NNNNN_g1.onnx
#   PT:    /.../<run>/model_step_NNNNN.pt
# i.e. strip /exported/ + the _g1.onnx suffix, append .pt.
#
# Modes:
#   auto     (default)            -- resolve from --model, must exist
#   explicit (--sonic-checkpoint) -- operator-supplied path, must exist
#   off      (--no-sonic-checkpoint) -- skip tokenizer; dataset zeros
#
# Auto-resolution failure (file missing) is fatal in 'auto' mode unless
# the operator opts out with --no-sonic-checkpoint -- this is the
# "never silently produce a zero-token dataset" guarantee.
#
# Skipped in VLA mode: SONIC_CHECKPOINT_MODE is forced off above, so the
# case below is a no-op even if reached.
# --------------------------------------------------------------------------
case "${SONIC_CHECKPOINT_MODE}" in
    auto)
        # Strip /exported/ and replace _g1.onnx with .pt.
        SONIC_CHECKPOINT="${SIM_MODEL/\/exported\//\/}"
        SONIC_CHECKPOINT="${SONIC_CHECKPOINT%_g1.onnx}.pt"
        if [[ ! -f "${SONIC_CHECKPOINT}" ]]; then
            err "SONIC tokenizer .pt not found at auto-resolved path:"
            err "  ${SONIC_CHECKPOINT}"
            err "(resolved from --model ${SIM_MODEL})"
            err ""
            err "Without it the recorder will write action.motion_token = zeros"
            err "and the dataset will NOT be VLA-trainable."
            err ""
            err "Fix one of:"
            err "  - point the .pt at the right place via:"
            err "      --sonic-checkpoint /path/to/model_step_NNNNN.pt"
            err "  - opt out (smoke tests only) via:"
            err "      --no-sonic-checkpoint"
            exit 1
        fi
        ;;
    explicit)
        if [[ ! -f "${SONIC_CHECKPOINT}" ]]; then
            err "SONIC tokenizer .pt not found at --sonic-checkpoint path:"
            err "  ${SONIC_CHECKPOINT}"
            exit 1
        fi
        ;;
    off)
        SONIC_CHECKPOINT=""
        ;;
    *)
        err "internal: unexpected SONIC_CHECKPOINT_MODE=${SONIC_CHECKPOINT_MODE}"
        exit 1
        ;;
esac

# Encoder-observation YAML preflight. Only needed when the tokenizer
# is going to run; an empty value (--encoder-config '') intentionally
# disables the multi-frame path and is forwarded as an omitted flag
# below.
if [[ -n "${SONIC_CHECKPOINT}" && -n "${ENCODER_CONFIG}" ]]; then
    if [[ ! -f "${ENCODER_CONFIG}" ]]; then
        err "Encoder-observation YAML not found at:"
        err "  ${ENCODER_CONFIG}"
        err ""
        err "The default ships at"
        err "  gear_sonic/data/encoder/x2_observation_config.yaml"
        err "Restore it from git or pass --encoder-config '' to fall"
        err "back to the deprecated freeze-pose path."
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------

if [[ "${VLA_MODE}" -eq 1 ]]; then
cat <<EOF
${C_GREEN}┌──────────────────────────────────────────────────────────────────────┐
│  X2 Quest 3 stack runner -- VLA closed-loop mode                    │
│  (heuristic_planner omitted; manager + recorder merge + VLA bridge)  │
└──────────────────────────────────────────────────────────────────────┘${C_RESET}
  log dir          : ${LOG_DIR}
  duration         : $([[ "${DURATION_S}" -eq 0 ]] && echo "unlimited (run until Ctrl-C)" || echo "${DURATION_S}s")
  scene            : ${ROBOCASA_ENV} -> ${ROBOCASA_SCENE_XML}
  deploy           : $([[ "${WITH_DEPLOY}" -eq 1 ]] && echo "ON  (sim --vla, profile=${SIM_PROFILE}, viewer=$([[ "${SIM_VIEWER}" -eq 1 ]] && echo on || echo off))" || echo "OFF (assume external)")
  ONNX model       : ${SIM_MODEL}
  --max-target-dev : ${VLA_MAX_TARGET_DEV:-(bypass)}
  --target-lpf-hz  : ${VLA_TARGET_LPF_HZ:-(bypass)}
  VLA mode         : $([[ "${VLA_NO_POLICY}" -eq 1 ]] && echo "NO-POLICY (idle wire smoke; bridge skips Gr00tPolicy load + inference)" || echo "closed-loop (Gr00tPolicy live)")
  VLA model dir    : $([[ "${VLA_NO_POLICY}" -eq 1 ]] && echo "(skipped: --vla-no-policy)" || echo "${VLA_BRIDGE_MODEL}")
  VLA prompt       : $([[ "${VLA_NO_POLICY}" -eq 1 ]] && echo "(skipped: --vla-no-policy)" || echo "\"${VLA_PROMPT}\"")
  VLA device       : ${VLA_DEVICE}
  VLA python       : ${VLA_BRIDGE_PYTHON}
  VLA pub rate     : ${VLA_BRIDGE_RATE} Hz
  VLA inf period   : ${VLA_INFERENCE_PERIOD_S} s  (50 Hz x 40-step horizon = 0.8 s)
  ports            : pose=${POSE_PORT}  body_pose=${BODY_POSE_PORT}  arm/hands=${ARM_HANDS_PORT}  x2_debug=${DEBUG_PORT}$([[ -n "${ROBOCASA_SCENE_XML}" ]] && echo "
                     scene_state=${SCENE_STATE_PORT}  scene_reset=${SCENE_RESET_PORT}" || true)
  WebXR (optional) : wss://<host>:${QUEST3_WS_PORT}, https://<host>:${QUEST3_HTTP_PORT}
EOF
else
cat <<EOF
${C_GREEN}┌──────────────────────────────────────────────────────────────────────┐
│  X2 Quest 3 planner-driven stack runner (Phase 0)                    │
└──────────────────────────────────────────────────────────────────────┘${C_RESET}
  log dir          : ${LOG_DIR}
  duration         : $([[ "${DURATION_S}" -eq 0 ]] && echo "unlimited (run until Ctrl-C)" || echo "${DURATION_S}s")
  mode             : $([[ "${WITH_RECORD}" -eq 1 ]] && echo "RECORD -> ${OUTPUT_DIR}" || echo "TELEOP-ONLY")
  task             : ${TASK:-(none -- robocasa auto-fills from scene metadata)}
  scene            : $([[ "${ROBOCASA_ENV}" == "none" ]] && echo "(flat floor, no robocasa scene)" || echo "${ROBOCASA_ENV} -> ${ROBOCASA_SCENE_XML}")
  planner demo     : ${PLANNER_DEMO:-(none -- planner sits in IDLE_LOOP at startup, awaits VR planner_cmd)}
  finger comp      : curl=$([[ "${APPLY_CURL_COMP}" -eq 1 ]] && echo on || echo off)  oppose=$([[ "${APPLY_OPPOSE_COMP}" -eq 1 ]] && echo on || echo off)$([[ -n "${ROBOCASA_SCENE_XML}" ]] && echo "  (robocasa default; pass --no-apply-{curl,oppose}-compensation to override)" || echo "  (pass --apply-{curl,oppose}-compensation to enable)")
  deploy           : $([[ "${WITH_DEPLOY}" -eq 1 ]] && echo "ON  (sim --vla, profile=${SIM_PROFILE}, viewer=$([[ "${SIM_VIEWER}" -eq 1 ]] && echo on || echo off))" || echo "OFF (assume external)")
  ONNX model       : ${SIM_MODEL}
  motion_token     : $([[ -n "${SONIC_CHECKPOINT}" ]] && echo "ON  (${SONIC_CHECKPOINT}, ${SONIC_TOKENIZER_DEVICE})" || echo "DISABLED (action.motion_token = zeros; dataset will NOT be VLA-trainable)")
  encoder_config   : $([[ -n "${SONIC_CHECKPOINT}" && -n "${ENCODER_CONFIG}" ]] && echo "${ENCODER_CONFIG}, modes=[retargeted_body_q], multi-frame 10x68 -> 680-D" || ([[ -n "${SONIC_CHECKPOINT}" ]] && echo "DEPRECATED freeze-pose (--encoder-config '' was passed)" || echo "(unused; tokenizer DISABLED above)"))
  operator         : ${OPERATOR_ID} (${CALIBRATION_PATH})
  WebXR endpoint   : wss://<host>:${QUEST3_WS_PORT}, https://<host>:${QUEST3_HTTP_PORT}
  ports            : pose=${POSE_PORT}  x2_debug=${DEBUG_PORT}  planner_cmd=${PLANNER_CMD_PORT}
                     arm/hands=${ARM_HANDS_PORT}  body_pose=${BODY_POSE_PORT}$([[ -n "${ROBOCASA_SCENE_XML}" ]] && echo "
                     scene_state=${SCENE_STATE_PORT}  scene_reset=${SCENE_RESET_PORT}" || true)
  manager sidecar  : ${SIDECAR_LOG}
EOF
fi
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
# Deploy (docker sim + ONNX).
#
# Phase 0: boot deploy FIRST (planner / recorder SUB x2_debug from it).
#
# VLA closed-loop: **defer** deploy until after manager → VLA → recorder
# so tcp://127.0.0.1:${POSE_PORT} already has a live ``pose`` PUB (idle
# stand from the recorder merge loop) before the C++ ``--vla`` ZMQ SUB
# connects — avoids sim boot with no reference feed.
# --------------------------------------------------------------------------

if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_LOG="${LOG_DIR}/deploy.log"
    # DURATION_S=0 means "run forever". Skip --max-duration entirely so
    # deploy_x2.sh inherits its own no-limit behaviour (matches the same
    # convention live_vla_publish_motion_token.py uses for --duration 0).
    if (( DURATION_S > 0 )); then
        DEPLOY_DURATION_S=$(( DURATION_S + 30 ))
    else
        DEPLOY_DURATION_S=0
    fi

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
    )
    if (( DEPLOY_DURATION_S > 0 )); then
        DEPLOY_ARGS+=(--max-duration "${DEPLOY_DURATION_S}")
    fi
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
    # VLA mode: --max-target-dev and --target-lpf-hz are now OPT-IN
    # (used to be auto-passed). These knobs apply to the deploy's
    # onboard SONIC tracking policy (NOT the VLA): the LPF smooths the
    # SONIC policy's joint-target stream and the clamp bounds its
    # per-tick delta. The SONIC policy runs every tick regardless of
    # whether the wire carries a non-zero motion token, and during the
    # ~22 s bridge cold-start the wire is just idle_stand + zero token
    # (byte-equivalent to --vla-no-policy mode). The fall we observed
    # at t=5..6 s came from those LPF + clamp settings making the SONIC
    # policy too sluggish to compensate for post-band gravity drift
    # after handoff auto-releases the elastic band at t=4 s. Same fall
    # reproduces in --vla-no-policy with handoff + LPF + clamp, where
    # the VLA is never loaded -- so the trigger is the SONIC policy
    # being throttled, not anything about the VLA cold-start. Verified
    # 2026-05-14 with /tmp/x2_vla_viewer: grav_z dropped -1.00 -> -0.93
    # between t=4 s and t=6 s, settled at -0.90 propped on the table.
    # Removing LPF/clamp (and remapping to parity above) keeps the
    # robot upright at grav_z=-1.00 through cold-start AND through live
    # VLA inference.
    #
    # If you have a well-trained checkpoint and see chunk-boundary
    # saw-tooth in the viewer (40-step horizon @ 50 Hz -> ~1.25 Hz
    # pulse), opt back in via:
    #   --vla-max-target-dev 0.10  --vla-target-lpf-hz 4.0
    # (these are the run_live_vla_demo.sh tested values). The wrapper
    # honours whatever you pass; we just stopped auto-passing them so
    # cold-start VLA matches --vla-no-policy stability by default.
    #
    # NOTE: VLA_MAX_TARGET_DEV / VLA_TARGET_LPF_HZ defaults are still
    # 0.10 / 4.0 in this file (line ~281); flip the gate condition to
    # be explicit-opt-in by checking VLA_MAX_TARGET_DEV_EXPLICIT etc.
    # if you want to make them only fire when CLI-provided.
    if [[ "${VLA_MODE}" -eq 1 && "${VLA_NO_POLICY}" -eq 0 \
          && "${VLA_DEPLOY_FILTERS}" -eq 1 ]]; then
        if [[ -n "${VLA_MAX_TARGET_DEV}" ]]; then
            DEPLOY_ARGS+=(--max-target-dev "${VLA_MAX_TARGET_DEV}")
        fi
        if [[ -n "${VLA_TARGET_LPF_HZ}" ]]; then
            DEPLOY_ARGS+=(--target-lpf-hz "${VLA_TARGET_LPF_HZ}")
        fi
    fi

    if [[ "${VLA_MODE}" -eq 0 ]]; then
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
        log "VLA mode: deploy deferred — recorder will publish idle pose on :${POSE_PORT} before sim starts."
    fi
else
    if [[ "${VLA_MODE}" -eq 1 ]]; then
        log "Step 1/4 — deploy spawn SKIPPED (--no-deploy). Recorder will PUB pose on :${POSE_PORT}; VLA must PUB body_pose on :${BODY_POSE_PORT}; external deploy must SUB pose on :${POSE_PORT}."
    else
        log "Step 1/4 — deploy spawn SKIPPED (--no-deploy). Recorder will publish 'pose' on :${POSE_PORT} regardless; the deploy you have running externally must be subscribed there."
    fi
fi

if [[ "${VLA_MODE}" -eq 0 ]]; then

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
# Optional --demo: pre-loads the YAML's command sequence into the
# planner's queue at boot. Commands play in order through the FSM
# (idle -> blend -> primitive -> blend -> idle ...) and the queue
# drains naturally back to IDLE_LOOP. The first manager-emitted
# planner_cmd that lands while the queue is non-empty calls
# replace_pending() and drops anything still pending so the operator
# always wins (see x2_heuristic_planner.py "Source semantics").
if [[ -n "${PLANNER_DEMO}" ]]; then
    PLANNER_ARGS+=(--demo "${PLANNER_DEMO}")
fi

log "Step 2/4 — spawning x2_heuristic_planner -> ${PLANNER_LOG}"
if [[ -n "${PLANNER_DEMO}" ]]; then
    log "  (--planner-demo ${PLANNER_DEMO}: queue will drain to idle_stand"
    log "   on its own; first VR planner_cmd preempts via replace_pending)"
fi
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
# Audio cues for X/Y in ARM_MAN ('Recording.' / 'Saved.') are gated on
# --recorder-enabled to stop the headset from announcing a save in
# --teleop-only sessions where no parquet is being written. The
# manager publishes recorder_cmd either way; this only mutes the
# audio path. Set iff --with-record was requested -- the recorder
# is the source of truth for whether a save actually landed and we
# want the operator's audio feedback to track that.
if [[ "${WITH_RECORD}" -eq 1 ]]; then
    MANAGER_ARGS+=(--recorder-enabled)
else
    MANAGER_ARGS+=(--no-recorder-enabled)
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
│    (v7.2: lateral lean / roll removed -- A+R-stick is unreachable    │
│     mid-lean with a single right thumb. Chain twist+lean instead.)   │
│  v7.2: R-stick lean+twist now ALSO active in ARM_MANIPULATION        │
│    (walk / turn still LOCO-only -- arm IK targets ride the torso so  │
│     leaning extends reach, but a base translation would break IK).   │
│  Note: the planner-log "command" appears flipped vs. the operator    │
│  intent because the curated bins were authored in a body frame       │
│  rotated 180 deg from the bridge's RSI init. End-to-end behaviour    │
│  is correct (push the stick the way you want the robot to move).    │
│  ARM_MANIPULATION (arms track VR + recorder buttons):                │
│    A press           engage / disengage arm IK                       │
│    X press           start episode (--with-record only)              │
│    Y press           stop & save episode (--with-record only)        │
│    R stick           SAME lean / twist as in LOCOMOTION (v7.2)       │
│    L stick           NO-OP (walk / step gated to LOCO mode)          │
│  (B-single still toggles LOCOMOTION <-> ARM_MANIPULATION; no chord.) │
│  Headset audio for X/Y ('Recording.' / 'Saved.') is gated on the     │
│  manager's --recorder-enabled flag (set iff --with-record). In       │
│  TELEOP-ONLY runs no audio fires on X/Y so the operator doesn't      │
│  get a false ACK; the [recorder] log line is the ground truth.       │
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
# Forward the resolved SONIC tokenizer .pt + device. When mode==off
# we deliberately omit --sonic-checkpoint so the recorder's None
# default kicks in and the one-shot warning fires (matched to the
# DISABLED banner line above). When ON, both flags ride together so
# the device choice is respected.
if [[ -n "${SONIC_CHECKPOINT}" ]]; then
    RECORDER_ARGS+=(
        --sonic-checkpoint "${SONIC_CHECKPOINT}"
        --sonic-tokenizer-device "${SONIC_TOKENIZER_DEVICE}"
    )
fi
# Forward the encoder-observation YAML so the inline tokenizer drives
# the multi-frame (real planner future) gather instead of falling back
# to the deprecated freeze-pose path. An empty value (operator passed
# --encoder-config '') deliberately omits the flag so the recorder's
# subscribe-mode loop emits the freeze-pose deprecation warning.
if [[ -n "${ENCODER_CONFIG}" ]]; then
    RECORDER_ARGS+=(--encoder-config "${ENCODER_CONFIG}")
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

fi  # end of: if [[ "${VLA_MODE}" -eq 0 ]] ; (planner + manager + recorder trio)

# --------------------------------------------------------------------------
# VLA mode (no heuristic planner): quest3_manager + live_vla bridge
# (body_pose @ :5565, same packed wire as Phase 0) + record_x2_dataset
# subscribe merge to ``pose`` @ :5556. Matches default idle-stand +
# 50 Hz publish path before the first body_pose arrives.
# --------------------------------------------------------------------------
if [[ "${VLA_MODE}" -eq 1 ]]; then
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
    if [[ "${APPLY_CURL_COMP}" -eq 1 ]]; then
        MANAGER_ARGS+=(--apply-curl-compensation)
    fi
    if [[ "${APPLY_OPPOSE_COMP}" -eq 1 ]]; then
        MANAGER_ARGS+=(--apply-oppose-compensation)
    fi
    if [[ "${WITH_RECORD}" -eq 1 ]]; then
        MANAGER_ARGS+=(--recorder-enabled)
    else
        MANAGER_ARGS+=(--no-recorder-enabled)
    fi

    log "Step 1/4 — spawning quest3_manager_x2 -> ${MANAGER_LOG}"
    "${PYTHON}" "${MANAGER_ARGS[@]}" >"${MANAGER_LOG}" 2>&1 &
    MANAGER_PID=$!

    log "  waiting for manager 'recorder PUB bound' marker (up to 30s)..."
    if ! wait_for_log_marker "${MANAGER_LOG}" "${MANAGER_PID}" \
            "recorder PUB bound at" 30 "manager"; then
        if ! wait_for_log_marker "${MANAGER_LOG}" "${MANAGER_PID}" \
                "Quest3ManagerX2" 5 "manager"; then
            exit 1
        fi
    fi
    log "  manager READY (pid=${MANAGER_PID}); settle 0.5s before VLA bridge ..."
    sleep 0.5

    VLA_BRIDGE_LOG="${LOG_DIR}/vla_bridge.log"

    VLA_BRIDGE_ARGS=(
        -m gear_sonic.scripts.live_vla_publish_motion_token
        --device "${VLA_DEVICE}"
        --pub-host '*'
        --pub-port "${BODY_POSE_PORT}"
        --pub-topic "${BODY_POSE_TOPIC}"
        --sub-host localhost
        --sub-port "${DEBUG_PORT}"
        --sub-topic "${DEBUG_TOPIC}"
        --rate "${VLA_BRIDGE_RATE}"
        --duration 0
        --inference-min-period-s "${VLA_INFERENCE_PERIOD_S}"
        --print-every 50
    )
    if [[ "${VLA_NO_POLICY}" -eq 1 ]]; then
        # --vla-no-policy keeps the bridge's idle wire LIVE: deploy is a
        # tracker, not a self-stabiliser, so the policy needs a stable
        # reference on the wire every tick or the robot falls in ~1 s
        # (verified empirically 2026-05-14 with --silent-wire +
        # --no-idle-publish: robot tilted to grav_z=-0.55 at policy_t=1.00s
        # and tripped the tilt watchdog at 75 deg). The bridge's
        # _IdleStandLoop replays the planner's idle_stand primitive in
        # the same shape build_pose_payload emits, so the wire content is
        # supposed to be byte-equivalent to --planner-only. Pass
        # --silent-wire explicitly only when you want a 'no upstream'
        # falsifier run (the deploy's prefill is NOT enough on its own).
        VLA_BRIDGE_ARGS+=(--no-policy)
    else
        VLA_BRIDGE_ARGS+=(--model-path "${VLA_BRIDGE_MODEL}" --prompt "${VLA_PROMPT}")
        if [[ -n "${VLA_BRIDGE_SONIC_CKPT}" ]]; then
            VLA_BRIDGE_ARGS+=(
                --sonic-checkpoint "${VLA_BRIDGE_SONIC_CKPT}"
                --sonic-decoder-device "${VLA_BRIDGE_SONIC_DECODER_DEVICE}"
            )
            log "  VLA bridge-side SONIC pose decoder: ${VLA_BRIDGE_SONIC_CKPT} (device=${VLA_BRIDGE_SONIC_DECODER_DEVICE})"
        else
            log "  VLA bridge-side SONIC pose decoder: DISABLED (body will track idle_stand only)"
        fi
        # Default chunk dump alongside the rest of the per-run logs so
        # the operator can sanity-check VLA I/O after the fact (see
        # scripts/inspect_vla_chunks.py for a quick summary). Skipped in
        # --vla-no-policy (no inference happens, would just write empty
        # safe-idle chunks) and overridable via --vla-dump-chunks-dir.
        if [[ -z "${VLA_DUMP_CHUNKS_DIR}" && -n "${LOG_DIR}" ]]; then
            VLA_DUMP_CHUNKS_DIR="${LOG_DIR}/vla_chunks"
        fi
        if [[ -n "${VLA_DUMP_CHUNKS_DIR}" ]]; then
            mkdir -p "${VLA_DUMP_CHUNKS_DIR}"
            VLA_BRIDGE_ARGS+=(
                --dump-chunks-dir "${VLA_DUMP_CHUNKS_DIR}"
                --dump-chunks-every "${VLA_DUMP_CHUNKS_EVERY}"
            )
            log "  VLA chunk I/O dump -> ${VLA_DUMP_CHUNKS_DIR} (every ${VLA_DUMP_CHUNKS_EVERY} chunk)"
        fi
    fi

    log "Step 2/4 — spawning live_vla_publish_motion_token -> ${VLA_BRIDGE_LOG}"
    log "  (body_pose PUB on :${BODY_POSE_PORT} binds immediately; 50 Hz bootstrap until policy loads)"
    setsid env \
        PYTHONPATH="${REPO_ROOT}/external_dependencies/Isaac-GR00T:${REPO_ROOT}" \
        MUJOCO_GL=egl \
        "${VLA_BRIDGE_PYTHON}" "${VLA_BRIDGE_ARGS[@]}" \
        >"${VLA_BRIDGE_LOG}" 2>&1 < /dev/null &
    VLA_BRIDGE_PID=$!
    VLA_BRIDGE_PGID="${VLA_BRIDGE_PID}"

    log "  waiting for VLA bridge PUB bind marker (up to 180s)..."
    if ! wait_for_log_marker "${VLA_BRIDGE_LOG}" "${VLA_BRIDGE_PID}" \
            "pose PUB bound on" 180 "vla-bridge"; then
        err "VLA bridge never bound its ZMQ PUB. Likely causes:"
        err "  - model load failed (check ${VLA_BRIDGE_LOG} for the traceback)"
        err "  - VLA_BRIDGE_PYTHON resolves the wrong env (no torch/transformers/GR00T)"
        err "  - port :${BODY_POSE_PORT} got stolen between preflight and now"
        exit 1
    fi
    log "  VLA bridge PUB live (pid=${VLA_BRIDGE_PID}); spawning recorder …"

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
        --teleop-only
    )
    if [[ "${APPLY_CURL_COMP}" -eq 1 ]]; then
        RECORDER_ARGS+=(--apply-curl-compensation)
    fi
    if [[ "${APPLY_OPPOSE_COMP}" -eq 1 ]]; then
        RECORDER_ARGS+=(--apply-oppose-compensation)
    fi
    if [[ -n "${ROBOCASA_SCENE_XML}" ]]; then
        RECORDER_ARGS+=(
            --robocasa-env "${ROBOCASA_ENV}"
            --scene-xml-path "${ROBOCASA_SCENE_XML}"
            --scene-state-sub-host localhost
            --scene-state-sub-port "${SCENE_STATE_PORT}"
            --scene-reset-pub-host '*'
            --scene-reset-pub-port "${SCENE_RESET_PORT}"
        )
    fi
    # --no-idle-publish stays available as an explicit knob (see the
    # CLI help on --no-idle-publish in record_x2_dataset.py) but we do
    # NOT enable it under --vla-no-policy: the bridge IS publishing
    # idle_stand frames, the recorder forwards them, the deploy tracks
    # them, robot stays upright. Suppressing the recorder's idle would
    # only matter if the bridge were also silent (--silent-wire), which
    # is now an explicit opt-in for falsification tests rather than
    # default behaviour.

    log "Step 3/4 — spawning record_x2_dataset (subscribe, teleop-only) -> ${RECORDER_LOG}"
    setsid "${PYTHON}" "${RECORDER_ARGS[@]}" >"${RECORDER_LOG}" 2>&1 < /dev/null &
    RECORDER_PID=$!
    RECORDER_PGID="${RECORDER_PID}"

    log "  waiting for recorder 'first body_pose received' marker (up to 60s)..."
    if ! wait_for_log_marker "${RECORDER_LOG}" "${RECORDER_PID}" \
            "first body_pose received" 60 "recorder"; then
        err "Recorder never received body_pose. Likely causes:"
        err "  - VLA bridge died after step 3 (check ${VLA_BRIDGE_LOG})"
        err "  - port :${BODY_POSE_PORT} bound by something else"
        err "  - topic mismatch (expect topic=${BODY_POSE_TOPIC})"
        exit 1
    fi
    log "  recorder READY (pid=${RECORDER_PID}); merge+publish loop active"

    if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
        log "Step 4/4 — spawning deploy_x2.sh sim --vla -> ${DEPLOY_LOG}"
        "${DEPLOY_SH}" "${DEPLOY_ARGS[@]}" >"${DEPLOY_LOG}" 2>&1 &
        DEPLOY_PID=$!

        log "  waiting for deploy 'Launching ...' marker (up to 180s)..."
        if ! wait_for_log_marker "${DEPLOY_LOG}" "${DEPLOY_PID}" "Launching ..." 180 "deploy"; then
            exit 1
        fi
        log "  deploy READY (pid=${DEPLOY_PID}); sim SUB on :${POSE_PORT} sees live pose feed"
        sleep 2.0
    fi
fi

# --------------------------------------------------------------------------
# Run loop — wait for the planner (which has --duration-s baked in) OR
# any child to die, whichever comes first. Prints periodic health.
# --------------------------------------------------------------------------

if [[ "${VLA_MODE}" -eq 1 ]]; then
cat <<EOF

${C_GREEN}=== Stack is LIVE (VLA closed-loop). Logs streaming under ${LOG_DIR}/ ===${C_RESET}
  Tail any of:
    tail -f ${LOG_DIR}/deploy.log
    tail -f ${LOG_DIR}/manager.log
    tail -f ${LOG_DIR}/recorder.log
    tail -f ${LOG_DIR}/vla_bridge.log

VLA bridge events streamed below as [vla] (inference cadence, pub ticks).
Deploy events streamed below as [deploy] (CONTROL ticks, grav_z, tilt watchdog).
Ctrl-C to shut down.
EOF
else
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
fi

MANAGER_TAIL_PID=""
RECORDER_TAIL_PID=""
VLA_TAIL_PID=""
DEPLOY_TAIL_PID=""

if [[ "${VLA_MODE}" -eq 0 ]]; then
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
else
    # VLA mode: mirror the bridge's pub-tick / inference timing lines
    # and the deploy's CONTROL / grav_z / tilt lines, both filtered to
    # avoid drowning the operator. Same line-style as run_live_vla_demo
    # .sh so muscle memory transfers ('pub tick', 'inference', 'video:'
    # for the bridge; 'CONTROL', 'grav_z', 'tilt' for the deploy).
    ( tail -n 0 -F "${VLA_BRIDGE_LOG}" 2>/dev/null \
        | stdbuf -oL grep -E 'pub tick|inference|video:|deploy_alive|render error|video render warn|video thread done' \
        | sed -u 's/^/[vla] /' ) &
    VLA_TAIL_PID=$!
    ( tail -n 0 -F "${DEPLOY_LOG}" 2>/dev/null \
        | stdbuf -oL grep -E 'CONTROL|grav_z|HANDOFF|POLICY|band release|tilt|fall|deploy: stopping|max-duration' \
        | sed -u 's/^/[deploy] /' ) &
    DEPLOY_TAIL_PID=$!
fi

cleanup_tails() {
    kill_pid_quiet "${MANAGER_TAIL_PID}"  "manager-tail"
    kill_pid_quiet "${RECORDER_TAIL_PID}" "recorder-tail"
    kill_pid_quiet "${VLA_TAIL_PID}"      "vla-tail"
    kill_pid_quiet "${DEPLOY_TAIL_PID}"   "deploy-tail"
}
trap 'cleanup_children; cleanup_tails; exit 130' INT TERM
trap 'cleanup_children; cleanup_tails' EXIT

# Run-loop child-died watchlist depends on which mode is active. In
# VLA mode we watch deploy + manager + recorder + VLA bridge. In
# planner mode the VLA bridge is unset and the planner trio is alive.
if [[ "${VLA_MODE}" -eq 1 ]]; then
    PIDS_TO_WATCH=("DEPLOY_PID" "MANAGER_PID" "RECORDER_PID" "VLA_BRIDGE_PID")
else
    PIDS_TO_WATCH=("DEPLOY_PID" "PLANNER_PID" "MANAGER_PID" "RECORDER_PID")
fi

START_TS=$(date +%s)
LAST_HEARTBEAT_TS=${START_TS}
while :; do
    # Any child died -> bail (cleanup trap fires).
    for pid_var in "${PIDS_TO_WATCH[@]}"; do
        pid="${!pid_var}"
        [[ -z "${pid}" ]] && continue
        if ! kill -0 "${pid}" 2>/dev/null; then
            err "${pid_var} (pid=${pid}) exited prematurely. Tail of its log:"
            case "${pid_var}" in
                DEPLOY_PID)     tail -n 30 "${DEPLOY_LOG:-/dev/null}"     >&2 || true ;;
                PLANNER_PID)    tail -n 30 "${PLANNER_LOG}"               >&2 || true ;;
                MANAGER_PID)    tail -n 30 "${MANAGER_LOG}"               >&2 || true ;;
                RECORDER_PID)   tail -n 30 "${RECORDER_LOG}"              >&2 || true ;;
                VLA_BRIDGE_PID) tail -n 30 "${VLA_BRIDGE_LOG:-/dev/null}" >&2 || true ;;
            esac
            exit 1
        fi
    done

    NOW_TS=$(date +%s)
    ELAPSED=$((NOW_TS - START_TS))
    if (( DURATION_S > 0 )) && (( ELAPSED >= DURATION_S )); then
        log "duration ${DURATION_S}s reached; shutting down stack ..."
        exit 0
    fi
    if (( NOW_TS - LAST_HEARTBEAT_TS >= 30 )); then
        LAST_HEARTBEAT_TS=${NOW_TS}
        if (( DURATION_S > 0 )); then
            DURATION_TAG="${ELAPSED}s/${DURATION_S}s"
        else
            DURATION_TAG="${ELAPSED}s (no limit)"
        fi
        if [[ "${VLA_MODE}" -eq 1 ]]; then
            log "alive: t=${DURATION_TAG}  pids[deploy=${DEPLOY_PID:-skipped} manager=${MANAGER_PID} recorder=${RECORDER_PID} vla-bridge=${VLA_BRIDGE_PID}]"
        else
            log "alive: t=${DURATION_TAG}  pids[deploy=${DEPLOY_PID:-skipped} planner=${PLANNER_PID} manager=${MANAGER_PID} recorder=${RECORDER_PID}]"
        fi
    fi
    sleep 1
done
