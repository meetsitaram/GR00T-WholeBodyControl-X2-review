#!/usr/bin/env bash
# Co-launch the X2 MuJoCo deploy in VLA mode and the Quest 3 recorder.
# The recorder embeds its own Quest 3 WebSocket server, so you don't
# need to run ``run_quest3_server.sh`` separately.
#
# Usage (record dataset):
#   bash gear_sonic/scripts/record_x2_dataset.sh \
#       --output-dir /path/to/dataset_v0 \
#       --task "pick up the red block" \
#       --sonic-checkpoint logs_rl/.../model_step_025000.pt
#
# Usage (live VR teleop, no dataset writes):
#   bash gear_sonic/scripts/record_x2_dataset.sh \
#       --teleop-only \
#       --sonic-checkpoint /path/to/model_step_025000.pt
#
# Wrapper-level flags:
#   --teleop-only               Skip exporter + ego renderer + dataset writes.
#                               Quest 3 still drives the policy live.
#   --sim-viewer                Open MuJoCo passive viewer (default ON).
#   --no-sim-viewer             Run the deploy bridge headless.
#   --no-vla                    Run the deploy in motion-replay mode (the
#                               policy tracks StandStillReference for the
#                               body); the recorder still publishes IK arm
#                               targets + finger setpoints, but the deploy
#                               ignores the ZMQ pose stream. Use this when
#                               the live-VLA reference produces OOD policy
#                               actions and you just want a stable robot
#                               to debug VR input / finger teleop.
#   --sim-omnihand              [opt-in, default OFF] Load the OmniHand-
#                               augmented MJCF (24+ finger DOFs replace
#                               the wrist stubs) so the bridge can drive
#                               fingers from VR. Requires (a) a checkpoint
#                               trained against the OmniHand-augmented
#                               dynamics -- the 25k *_g1 checkpoint goes
#                               OOD on this MJCF and triggers QACC NaN
#                               within 0.5 s -- and (b) pyzmq installed in
#                               the bridge's python env, otherwise fingers
#                               just sit at MJCF rest pose. Default OFF
#                               keeps you on the bare x2_ultra.xml the
#                               25k checkpoint was trained on.
#   --robocasa-env ENV          {none, X2PickPlaceCube, X2PickPlaceBowl}
#                               Default: none. When != none, switches
#                               both the deploy and the recorder into
#                               "robocasa scene" mode (G1 architecture):
#                               the deploy is launched with --sim-mjcf
#                               pointing at the pre-built robocasa scene
#                               XML, and the recorder loads the same XML
#                               so its ego renderer + RobocasaTaskMirror
#                               (per-tick success/reward/subtask labels)
#                               see the same table + objects. Forwards
#                               --robocasa-env to record_x2_dataset.py
#                               so it can resolve the matching scene
#                               metadata. Implies --sim-omnihand because
#                               every bundled scene is built on top of
#                               the OmniHand-augmented robot. Build
#                               scenes via
#                               gear_sonic/scripts/build_x2_robocasa_scene_xml.py.
#   --wrist-bypass MODE         {off, ik} (default ik). When 'ik', the C++
#                               deploy overwrites target_pos_mj for the 4
#                               broken wrist DOFs (left/right wrist_pitch
#                               + wrist_roll) with the IK reference from
#                               the recorder's ZMQ pose feed BEFORE the
#                               safety stack. wrist_yaw stays under SONIC.
#                               Empirical: SONIC pins wrist_pitch/roll at
#                               its trained comfort pose for both the 2k
#                               and 25k checkpoints, so an honest VR teleop
#                               session needs this override on. Set 'off'
#                               for sim-to-real fidelity probes where you
#                               want every joint to follow the policy. See
#                               x2_deploy_onnx_ref.cpp::CliArgs::WristBypass.
#   --deploy-model-dir DIR      Override the ONNX model dir
#                               (default: dirname of --sonic-checkpoint).
#   --sim-duration SECONDS      Auto-stop the deploy after N seconds (default 3600).
#
# Pass-through flags (forwarded verbatim to record_x2_dataset.py):
#   --tokenizer-device cuda
#   --hand-input grip|trigger|max
#   --no-omnihand               (debug only -- M5/M6 trained with omnihand on)
#   --quest3-no-ssl             (only on a trusted local network)
#   --rate 50
#
# Sim deploy spawns the robot upright (``--sim-init-pose default``,
# pelvis_z~0.95m) and lets the bridge auto-release the elastic band 1.0s
# after the first deploy command (the bridge's default ``--band-release-
# after-s 1.0``). The band is just a brief safety net during the policy's
# 2 s alpha ramp-up; once it auto-drops, the robot stands on its feet
# under full policy authority and is ready to walk / teleop arms.
#
# We deliberately do NOT use ``--sim-profile gantry`` -- that boots the
# bridge in a bent-knee crouch (pelvis_z=0.665m) which is OOD for the
# 25k tracking policy and triggers extreme |action_il| -> sim QACC NaN
# within a few hundred ms. We also do NOT pass
# ``--sim-band-release-after-s -1`` (band-on-forever) because that
# leaves the robot suspended in mid-air and forces the operator to hit
# key 9 to drop it -- not what we want for pure-sim teleop.
#
# The OmniHand-augmented MJCF is loaded via ``--sim-with-omnihand`` so
# the finger meshes match the trained visual configuration.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR=""
TASK=""
SONIC_CHECKPOINT=""
DEPLOY_MODEL_DIR=""
EXTRA_ARGS=()
SIM_DURATION="${SIM_DURATION:-3600}"
SIM_VIEWER="${SIM_VIEWER:-true}"
TELEOP_ONLY=false
USE_VLA=true
SIM_OMNIHAND=false
ROBOCASA_ENV="none"
# Default to 'ik' so VR teleop / dataset recording produces honest wrist
# tracking out of the box. Operators can pass --wrist-bypass off to fall
# back to pure SONIC for sim-to-real fidelity probes. Bridge safety:
# this only takes effect when --vla is also set, which the wrapper does
# unless --no-vla is passed (in which case the bypass is a no-op anyway).
WRIST_BYPASS="ik"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)         OUTPUT_DIR="$2"; shift 2 ;;
        --task)               TASK="$2"; shift 2 ;;
        --sonic-checkpoint)   SONIC_CHECKPOINT="$2"; shift 2 ;;
        --deploy-model-dir)   DEPLOY_MODEL_DIR="$2"; shift 2 ;;
        --sim-duration)       SIM_DURATION="$2"; shift 2 ;;
        --sim-viewer)         SIM_VIEWER=true; shift ;;
        --no-sim-viewer)      SIM_VIEWER=false; shift ;;
        --no-vla)             USE_VLA=false; shift ;;
        --sim-omnihand)       SIM_OMNIHAND=true; shift ;;
        --robocasa-env)       ROBOCASA_ENV="$2"; shift 2 ;;
        --wrist-bypass)       WRIST_BYPASS="$2"; shift 2 ;;
        --teleop-only)        TELEOP_ONLY=true; shift ;;
        -h|--help)
            grep -E "^# " "$0" | sed -e 's/^# //; s/^#//'
            exit 0
            ;;
        *)                    EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "${WRIST_BYPASS}" in
    off|ik) ;;
    *)
        echo "Error: --wrist-bypass must be 'off' or 'ik' (got '${WRIST_BYPASS}')" >&2
        exit 1
        ;;
esac

ROBOCASA_SCENE_XML=""
case "${ROBOCASA_ENV}" in
    none) ;;
    X2PickPlaceCube|X2PickPlaceBowl)
        ROBOCASA_SCENE_XML="${REPO_ROOT}/gear_sonic/data/assets/robocasa_scenes/${ROBOCASA_ENV}.xml"
        if [[ ! -f "${ROBOCASA_SCENE_XML}" ]]; then
            echo "Error: scene MJCF for --robocasa-env ${ROBOCASA_ENV}" >&2
            echo "       not found at ${ROBOCASA_SCENE_XML}." >&2
            echo "       Build it via:" >&2
            echo "         python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env ${ROBOCASA_ENV}" >&2
            exit 1
        fi
        # Robocasa scenes are always built on top of the OmniHand-
        # augmented robot, so the deploy bridge MUST load them with
        # --sim-with-omnihand or its hand controller plumbing will
        # never bind to the finger actuators.
        SIM_OMNIHAND=true
        ;;
    *)
        echo "Error: --robocasa-env must be one of 'none', 'X2PickPlaceCube', 'X2PickPlaceBowl' (got '${ROBOCASA_ENV}')" >&2
        exit 1
        ;;
esac

if ! ${TELEOP_ONLY}; then
    if [[ -z "${OUTPUT_DIR}" ]]; then
        echo "Error: --output-dir is required (omit by passing --teleop-only)" >&2
        grep -E "^# " "$0" | sed -e 's/^# //; s/^#//'
        exit 1
    fi
    if [[ -z "${TASK}" && "${ROBOCASA_ENV}" == "none" ]]; then
        # Robocasa mode auto-fills TASK from the scene metadata via
        # RobocasaTaskMirror.task_string, so the operator can omit it
        # there. Outside robocasa mode we still demand an explicit
        # instruction so the LeRobot ``task`` column isn't empty.
        echo "Error: --task is required outside --robocasa-env mode" >&2
        echo "       (omit --task only when --robocasa-env != none, or pass --teleop-only)." >&2
        grep -E "^# " "$0" | sed -e 's/^# //; s/^#//'
        exit 1
    fi
fi

# The recorder + live loop don't need the SONIC .pt checkpoint -- the
# deploy's tracking policy is loaded from the .onnx bundle, and we no
# longer run the FSQ encoder online. We keep --sonic-checkpoint as a
# convenient way to derive the deploy's --model dir; if the user
# passes --deploy-model-dir directly we don't need the .pt at all.
if [[ -z "${SONIC_CHECKPOINT}" && -z "${DEPLOY_MODEL_DIR}" ]]; then
    echo "Error: pass --sonic-checkpoint <model_step_NNNNN.pt> OR --deploy-model-dir <dir>." >&2
    grep -E "^# " "$0" | sed -e 's/^# //; s/^#//'
    exit 1
fi
if [[ -n "${SONIC_CHECKPOINT}" && ! -f "${SONIC_CHECKPOINT}" ]]; then
    echo "Error: SONIC checkpoint not found: ${SONIC_CHECKPOINT}" >&2
    exit 1
fi
if [[ -z "${DEPLOY_MODEL_DIR}" ]]; then
    DEPLOY_MODEL_DIR="$(dirname "${SONIC_CHECKPOINT}")"
fi

# The C++ deploy's --model flag wants an .onnx file path, not the run
# directory. Auto-resolve <run-dir>/exported/*.onnx if --deploy-model is
# unset, mirroring how showcase / sample_commands.md resolve it.
if [[ -z "${DEPLOY_MODEL:-}" ]]; then
    if [[ -f "${DEPLOY_MODEL_DIR}" && "${DEPLOY_MODEL_DIR}" == *.onnx ]]; then
        DEPLOY_MODEL="${DEPLOY_MODEL_DIR}"
    elif [[ -d "${DEPLOY_MODEL_DIR}/exported" ]]; then
        DEPLOY_MODEL="$(ls "${DEPLOY_MODEL_DIR}/exported"/*.onnx 2>/dev/null | head -n 1 || true)"
        if [[ -z "${DEPLOY_MODEL}" || ! -f "${DEPLOY_MODEL}" ]]; then
            echo "Error: no .onnx file found in ${DEPLOY_MODEL_DIR}/exported/" >&2
            echo "       Re-export the SONIC checkpoint to ONNX (see gear_sonic export tool) "  >&2
            echo "       or pass --deploy-model-dir <dir-containing-exported/*.onnx>" >&2
            exit 1
        fi
    else
        echo "Error: deploy_model_dir does not contain an exported/*.onnx bundle:" >&2
        echo "       ${DEPLOY_MODEL_DIR}" >&2
        exit 1
    fi
fi

DEPLOY_SH="${REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh"
if [[ ! -x "${DEPLOY_SH}" ]]; then
    echo "Error: deploy_x2.sh not found or not executable: ${DEPLOY_SH}" >&2
    exit 1
fi

if ! ${TELEOP_ONLY}; then
    mkdir -p "${OUTPUT_DIR}"
fi

VENV_PY=".venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
    echo "Error: ${VENV_PY} not found. Activate / create the GR00T venv first." >&2
    exit 1
fi

DEPLOY_LOG="$(mktemp -t deploy_x2_record_XXXXXX.log)"

# ----------------------------------------------------------------------
# Pre-flight cleanup
# ----------------------------------------------------------------------
# A previous deploy run can leave behind a ``docker_x2-x2sim-run-*``
# container (or a host bridge process) holding the bridge's ZMQ ports:
#   5557 -> x2_debug PUB (the recorder's deploy-alive heartbeat)
#   5559 -> scene_state PUB (RobocasaTaskMirror's per-tick observation)
#   5570 -> robot_pose PUB
# When that happens, the new bridge silently fails the bind on the
# colliding port (visible in the deploy log as
# ``[ERROR] ... x2_debug PUB failed to bind on port 5557: Address
# already in use``) but keeps running with a degraded socket set. The
# recorder then sees ``deploy_alive=False`` for the entire session and
# the operator is left teleoping into the void.
#
# The wrapper's EXIT trap normally tears the deploy container down, but
# a SIGKILL (e.g. a previous ``docker kill --filter ancestor=x2sim``
# from ``run_live_vla_demo.sh``) bypasses the trap, so we cannot
# assume the host is clean. Sweep here.
preflight_cleanup() {
    if command -v docker >/dev/null 2>&1; then
        local stale_containers
        stale_containers="$(docker ps --filter ancestor=x2sim --format '{{.Names}}' 2>/dev/null || true)"
        if [[ -n "${stale_containers}" ]]; then
            echo "[record_x2_dataset.sh] WARNING: stale x2sim container(s) detected; stopping:"
            while IFS= read -r c; do
                [[ -z "$c" ]] && continue
                echo "  - ${c}"
                docker stop --timeout 5 "${c}" >/dev/null 2>&1 || true
            done <<< "${stale_containers}"
        fi
    fi
    # Sanity-check the ZMQ ports the bridge needs to bind. ``ss`` is in
    # iproute2 on every modern linux; if it's missing we silently skip.
    if command -v ss >/dev/null 2>&1; then
        local busy
        busy="$(ss -tln 'sport = :5557 or sport = :5559 or sport = :5570' 2>/dev/null \
                | awk 'NR>1 {print $4}' \
                | sed 's/.*://g' \
                | sort -u | tr '\n' ' ')"
        if [[ -n "${busy// /}" ]]; then
            echo "Error: bridge ZMQ ports still bound by another process: ${busy}" >&2
            echo "       Free them (e.g. ``docker ps`` then ``docker stop <name>``;" >&2
            echo "       or ``lsof -i :5557 -i :5559 -i :5570``) and retry." >&2
            exit 1
        fi
    fi
}
preflight_cleanup

cleanup() {
    local rc=$?
    # 1) SIGINT the host-side bash that runs ``deploy_x2.sh`` so its
    #    own cleanup paths (e.g. RAMP_OUT, MuJoCo viewer teardown) get
    #    a chance to fire.
    if [[ -n "${DEPLOY_PID:-}" ]] && kill -0 "${DEPLOY_PID}" 2>/dev/null; then
        echo "[record_x2_dataset.sh] stopping deploy (pid=${DEPLOY_PID})"
        kill -INT "${DEPLOY_PID}" 2>/dev/null || true
        wait "${DEPLOY_PID}" 2>/dev/null || true
    fi
    # 2) ``deploy_x2.sh`` auto-relaunches inside the ``docker_x2-x2sim``
    #    container (via ``docker compose run``). Signals from the host
    #    bash do *not* reliably propagate into the container -- the
    #    compose-run shell can race with our SIGINT and exit before
    #    delivering the signal to PID 1 inside the container, leaking
    #    the deploy + bridge as a host-visible orphan that holds
    #    ports 5556/5557 forever (and won't accept ``pkill`` from the
    #    invoking UID because the container processes run as root).
    #    Find the container by parsing the ``Container <name> Creating``
    #    line that ``docker compose`` writes to our deploy log on bring-
    #    up, then ``docker stop`` it explicitly. This succeeds even if
    #    the host-side bash already exited.
    if command -v docker >/dev/null 2>&1; then
        local deploy_container
        deploy_container="$(grep -oE 'docker_x2-x2sim-run-[a-f0-9]+' "${DEPLOY_LOG}" 2>/dev/null | tail -1 || true)"
        if [[ -n "${deploy_container}" ]] && \
           docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${deploy_container}"; then
            echo "[record_x2_dataset.sh] stopping deploy container ${deploy_container}"
            docker stop --timeout 5 "${deploy_container}" >/dev/null 2>&1 || true
        fi
    fi
    echo "[record_x2_dataset.sh] deploy log preserved at: ${DEPLOY_LOG}"
    exit "${rc}"
}
trap cleanup EXIT INT TERM

if ${USE_VLA}; then
    DEPLOY_MODE_LABEL="VLA mode (recorder drives reference via ZMQ)"
else
    DEPLOY_MODE_LABEL="motion-replay mode (StandStillReference; --no-vla)"
fi
if ${SIM_OMNIHAND}; then
    DEPLOY_MJCF_LABEL="OmniHand-augmented MJCF (--sim-omnihand)"
else
    DEPLOY_MJCF_LABEL="bare x2_ultra.xml (no OmniHand fingers in sim)"
fi
if [[ "${ROBOCASA_ENV}" != "none" ]]; then
    DEPLOY_MJCF_LABEL="robocasa scene ${ROBOCASA_ENV}.xml (${ROBOCASA_SCENE_XML})"
fi
case "${WRIST_BYPASS}" in
    ik)  WRIST_BYPASS_LABEL="ik (deploy honours IK wrist refs; SONIC drives the rest)" ;;
    off) WRIST_BYPASS_LABEL="off (pure SONIC; expect wrist_pitch/roll to be pinned)" ;;
esac

echo "──────────────────────────────────────────────────────────────"
if ${TELEOP_ONLY}; then
    echo "  X2 VR Teleop + MuJoCo Deploy"
else
    echo "  X2 Dataset Recorder + MuJoCo Deploy"
fi
echo "    deploy mode    : ${DEPLOY_MODE_LABEL}"
echo "    deploy MJCF    : ${DEPLOY_MJCF_LABEL}"
echo "    wrist bypass   : ${WRIST_BYPASS_LABEL}"
echo "──────────────────────────────────────────────────────────────"
if ! ${TELEOP_ONLY}; then
    echo "  output_dir        : ${OUTPUT_DIR}"
    echo "  task              : ${TASK}"
fi
echo "  sonic_checkpoint  : ${SONIC_CHECKPOINT}"
echo "  deploy_model_dir  : ${DEPLOY_MODEL_DIR}"
echo "  deploy_model_onnx : ${DEPLOY_MODEL}"
echo "  sim_duration      : ${SIM_DURATION}s"
echo "  sim_viewer        : ${SIM_VIEWER}"
echo "  deploy_log        : ${DEPLOY_LOG}"
IP="$(hostname -I | awk '{print $1}')"
echo "  Quest 3 WebXR URL : https://${IP}:8443"
echo "──────────────────────────────────────────────────────────────"

DEPLOY_VIEWER_ARGS=()
if ${SIM_VIEWER}; then
    DEPLOY_VIEWER_ARGS+=("--sim-viewer")
fi

DEPLOY_MODE_ARGS=()
if ${USE_VLA}; then
    DEPLOY_MODE_ARGS+=("--vla")
fi

DEPLOY_OMNIHAND_ARGS=()
if ${SIM_OMNIHAND}; then
    DEPLOY_OMNIHAND_ARGS+=("--sim-with-omnihand")
fi

# Robocasa scene mode forwards --sim-mjcf to deploy_x2.sh so the bridge
# loads the same static scene XML the recorder will mirror via the
# RobocasaTaskMirror. The bridge auto-discovers the .json sidecar next
# to the .xml and uses it to resolve scene-object freejoint qpos
# addresses for its scene_state PUB / scene_reset SUB plumbing.
DEPLOY_SCENE_ARGS=()
if [[ -n "${ROBOCASA_SCENE_XML}" ]]; then
    DEPLOY_SCENE_ARGS+=("--sim-mjcf" "${ROBOCASA_SCENE_XML}")
fi

# Wrist bypass: only meaningful when the deploy is in VLA mode (the
# bypass reads from the ZMQ pose source). In --no-vla mode the override
# would be a no-op AND the C++ binary would refuse to start because of
# the "needs --input-type=zmq" guard, so suppress the flag entirely.
DEPLOY_WRIST_BYPASS_ARGS=()
if ${USE_VLA}; then
    DEPLOY_WRIST_BYPASS_ARGS+=("--wrist-bypass" "${WRIST_BYPASS}")
fi

echo "[record_x2_dataset.sh] starting deploy in background …"
"${DEPLOY_SH}" sim \
    "${DEPLOY_MODE_ARGS[@]}" \
    --sim-init-pose default \
    "${DEPLOY_OMNIHAND_ARGS[@]}" \
    "${DEPLOY_SCENE_ARGS[@]}" \
    "${DEPLOY_WRIST_BYPASS_ARGS[@]}" \
    --no-confirm \
    --autostart-after 0 \
    "${DEPLOY_VIEWER_ARGS[@]}" \
    --model "${DEPLOY_MODEL}" \
    --max-duration "${SIM_DURATION}" \
    < /dev/null \
    >"${DEPLOY_LOG}" 2>&1 &
DEPLOY_PID=$!
echo "[record_x2_dataset.sh] deploy pid=${DEPLOY_PID}; tailing log:"

# Throttle the deploy log on the way to stdout. Two reasons:
#  1) The C++ deploy publishes a "[INFO] ... CONTROL tick=N policy_t=..."
#     heartbeat at ~1 Hz. On a 5 minute teleop session that's 300 lines
#     of nearly-identical chatter that drown out the throttled recorder
#     status and the truly interesting events (scene_reset queued,
#     scene_reset applied, deploy_alive flips, etc).
#  2) Operators have asked for "one line per 5 s" on the screen; the
#     awk filter below keeps every non-tick line verbatim and admits
#     a CONTROL tick line only if at least 5 s have elapsed since the
#     last one. Full unmodified log still lives at $DEPLOY_LOG for
#     postmortem.
DEPLOY_LOG_THROTTLE_S="${DEPLOY_LOG_THROTTLE_S:-5}"
(
    tail -F "${DEPLOY_LOG}" 2>/dev/null \
    | awk -v throttle="${DEPLOY_LOG_THROTTLE_S}" '
        /CONTROL tick=/ {
            now = systime();
            if (now - last_tick >= throttle) {
                print "[deploy] " $0;
                last_tick = now;
            }
            next;
        }
        { print "[deploy] " $0 }
    '
) &
TAIL_PID=$!

# Wait for the deploy to actually be online and stepping, with a hard
# cap. Docker bring-up + ros2 startup can easily take 30-60 s the first
# time.
#
# IMPORTANT: do NOT grep for ``Launching ...``. ``deploy_x2.sh`` prints
# that string from its own banner BEFORE forking the ros2 / bridge /
# MuJoCo process, so a match on iteration 0 would short-circuit the
# wait and hand off to the recorder while docker is still spinning.
# We instead wait for the C++ deploy's first ``CONTROL tick=N`` line,
# which is only printed after:
#   (1) the bridge has bound all of its PUB sockets,
#   (2) the bridge has handed PD authority over to the deploy,
#   (3) the policy is actually running and emitting commands.
# That is the earliest moment at which the recorder will see a fresh
# x2_debug heartbeat (``deploy_alive=True``).
#
# We also abort on any ``failed to bind`` line in the bridge log,
# because a port collision means the recorder will silently get a
# half-degraded deploy (e.g. scene_state up, x2_debug down) -- exactly
# the failure mode that left the operator teleoping into the void.
DEPLOY_BOOT_TIMEOUT_S="${DEPLOY_BOOT_TIMEOUT_S:-180}"
echo "[record_x2_dataset.sh] waiting up to ${DEPLOY_BOOT_TIMEOUT_S}s for deploy to come online …"
DEPLOY_READY=false
for ((i = 0; i < DEPLOY_BOOT_TIMEOUT_S; i++)); do
    if ! kill -0 "${DEPLOY_PID}" 2>/dev/null; then
        echo "Error: deploy died during bring-up. See ${DEPLOY_LOG}" >&2
        kill "${TAIL_PID}" 2>/dev/null || true
        exit 1
    fi
    if grep -qE "failed to bind on port|PUB failed to bind|Address already in use" "${DEPLOY_LOG}" 2>/dev/null; then
        echo "" >&2
        echo "Error: bridge failed to bind one of its ZMQ PUB sockets." >&2
        echo "       This usually means a previous deploy/container is still" >&2
        echo "       holding the port. Offending lines from ${DEPLOY_LOG}:" >&2
        grep -nE "failed to bind on port|PUB failed to bind|Address already in use" "${DEPLOY_LOG}" >&2 | head -5
        echo "" >&2
        echo "       Stop the offender (``docker ps`` then ``docker stop <name>``)" >&2
        echo "       and rerun. The wrapper's preflight should normally catch this." >&2
        kill -INT "${DEPLOY_PID}" 2>/dev/null || true
        kill "${TAIL_PID}" 2>/dev/null || true
        exit 1
    fi
    if grep -q "CONTROL tick=" "${DEPLOY_LOG}" 2>/dev/null; then
        DEPLOY_READY=true
        break
    fi
    sleep 1
done

if ! ${DEPLOY_READY}; then
    echo "Error: deploy didn't reach its first CONTROL tick in ${DEPLOY_BOOT_TIMEOUT_S}s. See ${DEPLOY_LOG}" >&2
    kill -INT "${DEPLOY_PID}" 2>/dev/null || true
    kill "${TAIL_PID}" 2>/dev/null || true
    exit 1
fi

# Brief settle for the ZMQ pose SUB to bind before we start blasting.
sleep 2

# ----------------------------------------------------------------------
# Deploy watchdog
# ----------------------------------------------------------------------
# The recorder doesn't have a handle on the deploy PID, so when the
# deploy crashes mid-session (segfault, mujoco viewer close-button,
# external ``docker kill``, ``--max-duration`` trip, etc) the recorder
# just keeps publishing IK targets into a dead socket and the operator
# stares at a frozen viewer wondering if Quest 3 is broken. Spawn a
# small watcher in the background that polls ``kill -0`` on the deploy
# PID once a second and, the first time it disappears, prints a loud
# red banner on stderr AND signals the recorder to shut down so the
# operator gets immediate feedback instead of a silent dead loop.
deploy_watchdog() {
    local pid="$1"
    local recorder_pgid="$2"
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 1
    done
    # Use raw ANSI here -- the recorder's own banner uses the same.
    local R=$'\033[1;31m'; local NC=$'\033[0m'
    {
        echo ""
        echo "${R}╔══════════════════════════════════════════════════════════════════════╗${NC}"
        echo "${R}║  [deploy-watch] DEPLOY PROCESS EXITED                                ║${NC}"
        echo "${R}║  pid=${pid}  log=${DEPLOY_LOG}${NC}"
        echo "${R}║  Last 20 lines of the deploy log:                                    ║${NC}"
        echo "${R}╚══════════════════════════════════════════════════════════════════════╝${NC}"
        tail -n 20 "${DEPLOY_LOG}" 2>/dev/null | sed "s/^/[deploy-watch] /"
        echo ""
    } >&2
    # SIGINT the recorder process group so its own clean shutdown path
    # (which flushes the LeRobot dataset) gets a chance to fire.
    kill -INT -- "-${recorder_pgid}" 2>/dev/null || true
}

echo "[record_x2_dataset.sh] launching recorder …"
RECORDER_ARGS=()
if [[ -n "${SONIC_CHECKPOINT}" ]]; then
    RECORDER_ARGS+=("--sonic-checkpoint" "${SONIC_CHECKPOINT}")
fi
if ${TELEOP_ONLY}; then
    RECORDER_ARGS+=("--teleop-only")
else
    RECORDER_ARGS+=("--output-dir" "${OUTPUT_DIR}")
    if [[ -n "${TASK}" ]]; then
        RECORDER_ARGS+=("--task" "${TASK}")
    fi
fi
if [[ "${ROBOCASA_ENV}" != "none" ]]; then
    # Recorder resolves the scene XML from this name (same lookup as
    # the bridge wiring above) and pulls the canonical task instruction
    # from the .json sidecar, so the LeRobot task column matches the
    # scene the deploy is rendering.
    RECORDER_ARGS+=("--robocasa-env" "${ROBOCASA_ENV}")
fi

# Run the recorder in its own session so the deploy watchdog can
# target the whole subtree with a single ``kill -INT -- -pgid``
# without risking that we INT our own shell. We deliberately do NOT
# pass ``setsid -w``: with -w setsid stays alive as a parent and
# ``$!`` would be the setsid PID, not the recorder. Without -w,
# setsid execs into the recorder, so ``$!`` IS the recorder PID
# AND the new session leader (== PGID).
#
# We also forward the recorder's stdin from the controlling tty so
# Quest 3 isn't the only input source -- the operator can still
# Ctrl-C this wrapper and SIGINT propagates via the EXIT trap.
set +e
setsid "${VENV_PY}" -m gear_sonic.scripts.record_x2_dataset \
    "${RECORDER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" </dev/null &
RECORDER_PID=$!
RECORDER_PGID="${RECORDER_PID}"

deploy_watchdog "${DEPLOY_PID}" "${RECORDER_PGID}" &
WATCHDOG_PID=$!

wait "${RECORDER_PID}"
RC=$?
set -e

# Stop the watchdog so it doesn't fire after a clean recorder exit.
kill "${WATCHDOG_PID}" 2>/dev/null || true
kill "${TAIL_PID}" 2>/dev/null || true

echo "[record_x2_dataset.sh] recorder exited with rc=${RC}"
exit "${RC}"
