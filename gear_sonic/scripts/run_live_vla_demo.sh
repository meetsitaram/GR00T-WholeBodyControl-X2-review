#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Live VLA → SONIC closed-loop demo launcher.
#
# Spawns three processes in parallel and tails each into its own log file
# under ``$RUN_DIR``:
#
#   1. The live VLA bridge   (host, conda env_isaaclab + GPU)
#        - subs to deploy x2_debug on tcp://localhost:5557
#        - pubs motion-token poses on tcp://localhost:5556
#        - records ego_view + third_person_front videos at $VIDEO_FPS
#
#   2. The x2_debug dumper    (host, .venv)
#        - JSON-Lines telemetry trace for offline analysis
#
#   3. The deploy_x2.sh sim   (docker x2sim, opens MuJoCo passive viewer)
#        - --sim-viewer brings up the live physics window so you can SEE
#          the actual floating-base motion (feet on floor, tipping, etc.)
#        - the bridge's recorded videos still freeze the pelvis at the
#          nominal stand pose (no base_pos in x2_debug); use the viewer
#          for ground truth.
#
# Usage:
#   ./gear_sonic/scripts/run_live_vla_demo.sh                           # defaults
#   RUN_DIR=/tmp/my_run ./gear_sonic/scripts/run_live_vla_demo.sh
#   PROMPT="play piano" MODEL_DIR=/tmp/x2_n17_finetune_v1 ./gear_sonic/scripts/run_live_vla_demo.sh
#
# Stop everything:
#   ./gear_sonic/scripts/run_live_vla_demo.sh stop
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ----- knobs (override via env) -------------------------------------------
: "${RUN_DIR:=/tmp/c5_demo_live}"
: "${MODEL_DIR:=/tmp/x2_n17_finetune_v1}"
: "${PROMPT:=play minecraft music on piano}"
: "${ONNX:=/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx}"
: "${MAX_DURATION:=30}"
: "${AUTOSTART_AFTER:=3}"
: "${VIDEO_FPS:=25}"
: "${RENDER_W:=640}"
: "${RENDER_H:=480}"
: "${FRONT_W:=1280}"
: "${FRONT_H:=720}"
: "${SIM_VIEWER:=true}"   # --sim-viewer; set to "" to disable
# OmniHand augmented MJCF + finger-command ZMQ subscriber. Default ON
# so the live viewer matches what gets recorded in the rendered videos
# (which always use compose_x2_with_omnihand). Set WITH_OMNIHAND=""
# (empty string) to revert to the bare X2 MJCF -- useful when comparing
# against M2/M5 acceptance gates that pin the dummy-wrist model.
: "${WITH_OMNIHAND:=true}"
: "${DUMP_DURATION:=60}"

# Sim bring-up profile. Mirrors the real-robot handoff:
#   * ``handoff`` (default) -- start in ``gantry_hang`` (bent-knee crouch,
#     pelvis_z=0.665 m), elastic band engaged at ~88 % body support, band
#     auto-releases ``ramp_seconds + 2.0 s`` after the first deploy command.
#     Policy ramps in, then has to hold the body upright on its own once the
#     band lets go. Closest to the real X2 bring-up where the operator
#     loosens the gantry strap once the policy is stable.
#   * ``gantry`` -- same start pose but the band NEVER auto-releases;
#     gantry-supported powered run (Phase 7-9 hardware test).
#   * ``parity`` -- bridge starts with full-band proxy through the soft-
#     start ramp + buffer; useful when you want to compare against a
#     bit-exact Python sim2sim eval baseline.
#   * "" (empty) -- bridge default. Robot starts in DEFAULT_DOF upright
#     stand at pelvis_z=0.85 m with the band pulling UP. Releases after 1 s.
#     Robot ends up "floating then dropping" because the band lifts the
#     pelvis ~5 cm above the standing pose. Quick smoke only.
: "${SIM_PROFILE:=handoff}"

# Per-joint hard clamp on |target - default_angles|, in radians. Routes
# directly to deploy_x2.sh's --max-target-dev. With an undertrained VLA
# this is the difference between "robot stays anchored, arms gesture
# coarsely" and "policy outputs spike, joints accumulate hundreds of
# radians, base launches into orbit". Recommended values:
#   0.05  = 3 deg  -- first powered bring-up (matches deploy_x2.sh hint)
#   0.30  = 17 deg -- undertrained-VLA piano test (visible arm gesture,
#                    no joint runaway, base stays put)
#   ""    = disabled (legacy default; lets a divergent policy launch
#                    body_q to 700+ rad and the base 25 m forward, which
#                    is what we accidentally observed on 2026-05-08).
: "${MAX_TARGET_DEV:=0.10}"

# First-order EMA cutoff (Hz) on the published joint targets, applied
# AFTER --max-target-dev. The publisher's chunk-reset cycle (~1.25 Hz
# under L3 fix INFERENCE_MIN_PERIOD_S=0.8) generates a saw-tooth target
# whenever the model output is noisy: the C++ deploy clamps each chunk
# to default ± MAX_TARGET_DEV, so the published target flips between
# default+dev and default-dev every chunk -- O(2*dev) of joint motion
# happens at the chunk boundary, even with the clamp doing its job.
# A 4 Hz LPF rolls off the saw-tooth without damaging the < 2 Hz piano-
# gesture content. Set to "" (empty) to bypass (binary default = 0 = no
# filter, matches sim2sim parity baseline).
: "${TARGET_LPF_HZ:=4.0}"

# Diagnostic: dump every Nth full action chunk (40 steps × all heads) as
# .npz to this directory. Used to verify whether the model predicts a
# coherent multi-step gesture or just spits out near-constant frames.
# Empty = disabled (default; no probe overhead during normal runs).
: "${DUMP_CHUNKS_DIR:=}"
: "${DUMP_CHUNKS_EVERY:=5}"

# Lower bound on time between successive VLA inferences (seconds). The
# publisher always advances at --rate (=50 Hz) and resets chunk_step to 0
# whenever a fresh chunk arrives, so to actually consume the *full*
# 40-step horizon the inference cadence must match horizon × pub_period.
# With horizon=40 and pub_period=20 ms the natural cadence is exactly
# 0.8 s -- inference produces a chunk, publisher walks all 40 steps over
# 0.8 s, new chunk arrives just as the previous one is exhausted.
#
# Older default of 0.4 s caused the publisher to consume only steps
# 0..19 of every chunk, throw away steps 20..39, and visibly "snap back
# to the start of the gesture" 2.5x per second -- the spike pattern the
# user observed in the screencast on 2026-05-08. See diagnosis in
# /tmp/c5_demo_probe/ chunk-dump analysis.
: "${INFERENCE_MIN_PERIOD_S:=0.8}"

# Pin the conda env that owns torch / transformers / GR00T. Resolving the
# python by absolute path (instead of going through ``conda run … python``)
# makes the launcher immune to having ``.venv`` activated in the parent
# shell -- previously a ``(.venv)`` prompt would leak its python onto PATH
# and ``conda run`` would still pick it up, producing
# ``ModuleNotFoundError: No module named 'transformers'``.
: "${CONDA_ENV_BRIDGE:=env_isaaclab}"
: "${CONDA_PREFIX_BASE:=$HOME/miniconda3}"
BRIDGE_PY="${CONDA_PREFIX_BASE}/envs/${CONDA_ENV_BRIDGE}/bin/python"

# Detach from any active virtualenv so its ``python`` / ``PATH`` /
# ``PYTHONPATH`` don't bleed into the conda subprocesses we're about to
# spawn. ``deactivate`` is a function defined by the venv's activate
# script, so it only exists when a venv is active -- guard the call so
# ``set -e`` doesn't trip when no venv is active.
if [[ -n "${VIRTUAL_ENV:-}" ]] && declare -F deactivate > /dev/null 2>&1; then
    echo -e "[runner] detected active virtualenv (${VIRTUAL_ENV}); deactivating before spawn"
    deactivate || true
    # Also strip any residual PATH hits from /home/$USER/.local/share/uv
    # which uv-managed venvs prepend.
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        unset VIRTUAL_ENV
    fi
fi
# Even after deactivate, conda's own activation might be partially in
# effect. We don't need it -- we'll call BRIDGE_PY by absolute path.
unset PYTHONHOME 2>/dev/null || true

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

stop_all() {
    echo -e "${YELLOW}[runner] stopping live demo …${NC}"
    pkill -f "live_vla_publish_motion_token" 2>/dev/null || true
    pkill -f "dump_x2_debug" 2>/dev/null || true
    pkill -f "deploy_x2.sh" 2>/dev/null || true
    pkill -f "x2_deploy_onnx_ref" 2>/dev/null || true
    # Deploy runs as root inside the x2sim container; pkill from host can't
    # reach it. Kill the container by image filter.
    #
    # ⚠️  CROSS-PROCESS HAZARD ⚠️
    # ``--filter ancestor=x2sim`` matches **every** running container built
    # from the x2sim image -- including ones owned by other workflows
    # (e.g. ``gear_sonic/scripts/record_x2_dataset.sh`` while it is in the
    # middle of a recording session). ``docker kill`` is SIGKILL; the
    # victim container's MuJoCo viewer will vanish instantly with no log
    # flush, which from the operator's seat looks identical to a viewer
    # crash. Symptoms in the *other* terminal: viewer window disappears,
    # deploy log goes silent mid-CONTROL-tick line with no goodbye, and
    # (since v1.7) the recorder prints a red ``[recorder] !! deploy went
    # silent`` warning citing this script by name.
    #
    # If you only want to clean up *your own* run, prefer:
    #   docker stop "$(cat "$RUN_DIR/.deploy_container" 2>/dev/null)"
    # i.e. write the container name to a sidecar on ``start`` and target
    # it explicitly on ``stop``. The broad filter below is kept for the
    # legacy "nuke everything" reset when nothing else is running.
    local cid
    cid="$(docker ps -q --filter ancestor=x2sim 2>/dev/null || true)"
    [[ -n "$cid" ]] && docker kill $cid 2>/dev/null || true
    sleep 4
    echo -e "${GREEN}[runner] done.${NC}"
}

if [[ "${1:-start}" == "stop" ]]; then
    stop_all
    exit 0
fi

# ----- pre-flight ---------------------------------------------------------
mkdir -p "$RUN_DIR"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo -e "${RED}[runner] MODEL_DIR=$MODEL_DIR does not exist${NC}" >&2
    exit 1
fi
if [[ ! -f "$ONNX" ]]; then
    echo -e "${RED}[runner] ONNX=$ONNX does not exist${NC}" >&2
    exit 1
fi
if [[ ! -x "$BRIDGE_PY" ]]; then
    echo -e "${RED}[runner] BRIDGE_PY=$BRIDGE_PY not found / not executable${NC}" >&2
    echo -e "${YELLOW}    (set CONDA_PREFIX_BASE / CONDA_ENV_BRIDGE if conda lives elsewhere)${NC}" >&2
    exit 1
fi
# Hard-fail early if env_isaaclab is missing key deps -- much friendlier
# than the bridge dying 60s into model load with a stack trace.
if ! "$BRIDGE_PY" -c "import transformers, torch, zmq, mujoco" 2>/dev/null; then
    echo -e "${RED}[runner] $BRIDGE_PY is missing one of: transformers / torch / pyzmq / mujoco${NC}" >&2
    "$BRIDGE_PY" -c "import transformers, torch, zmq, mujoco" 2>&1 | sed 's/^/    /' >&2 || true
    exit 1
fi

# Anything already running on the ZMQ ports is going to cause the deploy
# bind to fail. Clean up.
if ss -tln 2>/dev/null | grep -qE ':555[6-7] '; then
    echo -e "${YELLOW}[runner] port 5556/5557 in use; killing prior runs first${NC}"
    stop_all
    sleep 4
fi

echo -e "${CYAN}=== live VLA demo ===${NC}"
echo "  RUN_DIR        = $RUN_DIR"
echo "  MODEL_DIR      = $MODEL_DIR"
echo "  PROMPT         = $PROMPT"
echo "  ONNX           = $ONNX"
echo "  MAX_DURATION   = ${MAX_DURATION}s"
echo "  SIM_VIEWER     = $SIM_VIEWER"
echo "  SIM_PROFILE    = ${SIM_PROFILE:-(bridge default)}"
echo "  MAX_TARGET_DEV = ${MAX_TARGET_DEV:-(disabled)}"
echo "  TARGET_LPF_HZ  = ${TARGET_LPF_HZ:-(bypass)}"
echo "  INF_MIN_PER_S  = ${INFERENCE_MIN_PERIOD_S}"
echo "  BRIDGE_PY      = $BRIDGE_PY"
echo

# ----- 1. bridge (live VLA + recorder) ------------------------------------
echo -e "${CYAN}[runner] starting live VLA bridge (model load takes ~60-90s) …${NC}"
PYTHONPATH="${REPO_ROOT}/external_dependencies/Isaac-GR00T:${REPO_ROOT}" \
MUJOCO_GL=egl \
nohup "$BRIDGE_PY" -m gear_sonic.scripts.live_vla_publish_motion_token \
    --model-path "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --device cuda:0 \
    --pub-host '*' --pub-port 5556 --pub-topic pose \
    --sub-host localhost --sub-port 5557 --sub-topic x2_debug \
    --rate 50 \
    --duration 0 \
    --inference-min-period-s "${INFERENCE_MIN_PERIOD_S:-0.8}" \
    --render-width "$RENDER_W" --render-height "$RENDER_H" \
    --print-every 50 \
    --video-out "$RUN_DIR/ego_view.mp4" \
    --video-front-out "$RUN_DIR/front_view.mp4" \
    --video-front-camera third_person_front \
    --video-front-width "$FRONT_W" --video-front-height "$FRONT_H" \
    --video-fps "$VIDEO_FPS" \
    ${DUMP_CHUNKS_DIR:+--dump-chunks-dir "$DUMP_CHUNKS_DIR" --dump-chunks-every "$DUMP_CHUNKS_EVERY"} \
    > "$RUN_DIR/bridge.log" 2>&1 &
echo $! > "$RUN_DIR/bridge.pid"
echo "  bridge.pid = $(cat "$RUN_DIR/bridge.pid")    log = $RUN_DIR/bridge.log"

# Wait for the model to finish loading (port 5556 only binds after weights
# are on the GPU; before that the deploy can't connect).
echo -ne "${CYAN}[runner] waiting for model load${NC}"
for i in $(seq 1 90); do
    if ss -tln 2>/dev/null | grep -q ':5556 '; then
        echo -e " ${GREEN}ready${NC} (after ${i}s)"
        break
    fi
    echo -n "."; sleep 1
    if [[ $i == 90 ]]; then
        echo -e "\n${RED}[runner] bridge never bound port 5556 -- check $RUN_DIR/bridge.log${NC}" >&2
        exit 1
    fi
done

# ----- 2. dump_x2_debug ---------------------------------------------------
echo -e "${CYAN}[runner] starting x2_debug dump (offline JSONL trace) …${NC}"
nohup .venv/bin/python -m gear_sonic.scripts.dump_x2_debug \
    --host localhost --port 5557 --topic x2_debug \
    --duration "$DUMP_DURATION" \
    --json-out "$RUN_DIR/dump.jsonl" \
    --csv-out "$RUN_DIR/dump.csv" \
    --print-every 100 \
    > "$RUN_DIR/dump.log" 2>&1 &
echo $! > "$RUN_DIR/dump.pid"
echo "  dump.pid   = $(cat "$RUN_DIR/dump.pid")    log = $RUN_DIR/dump.log"

# ----- 3. deploy + MuJoCo passive viewer ----------------------------------
DEPLOY_FLAGS=(sim --vla
    --model "$ONNX"
    --autostart-after "$AUTOSTART_AFTER"
    --max-duration "$MAX_DURATION"
    --no-confirm)
if [[ "$SIM_VIEWER" == "true" ]]; then
    DEPLOY_FLAGS+=(--sim-viewer)
fi
if [[ -n "${SIM_PROFILE:-}" ]]; then
    DEPLOY_FLAGS+=(--sim-profile "$SIM_PROFILE")
fi
if [[ "$WITH_OMNIHAND" == "true" ]]; then
    # The bridge SUBs to the live VLA bridge's pose stream so the
    # OmniHand fingers move in step with the policy. Topic / port
    # match the live-bridge defaults (see gear_sonic/scripts/
    # live_vla_publish_motion_token.py: --pub-port=5556, --pub-topic=pose).
    DEPLOY_FLAGS+=(--sim-with-omnihand)
fi
if [[ -n "${MAX_TARGET_DEV:-}" ]]; then
    DEPLOY_FLAGS+=(--max-target-dev "$MAX_TARGET_DEV")
fi
if [[ -n "${TARGET_LPF_HZ:-}" ]]; then
    DEPLOY_FLAGS+=(--target-lpf-hz "$TARGET_LPF_HZ")
fi
echo -e "${CYAN}[runner] starting deploy_x2.sh sim (--sim-viewer=$SIM_VIEWER, --with-omnihand=$WITH_OMNIHAND) …${NC}"
echo "  $ bash gear_sonic_deploy/deploy_x2.sh ${DEPLOY_FLAGS[*]}"
nohup bash gear_sonic_deploy/deploy_x2.sh "${DEPLOY_FLAGS[@]}" \
    > "$RUN_DIR/deploy.log" 2>&1 &
echo $! > "$RUN_DIR/deploy.pid"
echo "  deploy.pid = $(cat "$RUN_DIR/deploy.pid")    log = $RUN_DIR/deploy.log"

DEPLOY_PID="$(cat "$RUN_DIR/deploy.pid")"
BRIDGE_PID="$(cat "$RUN_DIR/bridge.pid")"

cat <<EOF

${GREEN}=== launched ===${NC}

  The MuJoCo passive viewer pops up only AFTER the docker container has
  booted (image cache hot: ~5-10s; cold: 30-90s while it builds the
  ROS workspace). If you run with SIM_PROFILE=handoff (the default) the
  robot starts in a bent-knee crouch supported by the elastic band, the
  policy ramps in over ~1s, then the band releases ~3s after first
  contact -- so the policy must hold the body upright on its own.

  The deploy will auto-shutdown after MAX_DURATION=${MAX_DURATION}s of
  control (this is the --max-duration flag, not a crash). The bridge
  notices the deploy went quiet, flushes both MP4 files, and exits.

  The launcher will stay foreground from here, tailing the deploy
  + bridge logs filtered to the interesting lines until the deploy
  exits, then auto-stop the bridge so videos finalise. Ctrl-C will
  cleanly tear everything down.

  Files written this run:
      $RUN_DIR/bridge.log     VLA inference + token norms + video frame counter
      $RUN_DIR/deploy.log     CONTROL ticks + grav_z + tilt watchdog
      $RUN_DIR/dump.jsonl     offline x2_debug JSONL trace
      $RUN_DIR/ego_view.mp4   1st-person camera (head sensor frame, no tilt)
      $RUN_DIR/front_view.mp4 3rd-person front view (with tilt + omnihand)

EOF

# Forward Ctrl-C to a graceful stop.
trap 'echo; echo -e "${YELLOW}[runner] Ctrl-C received, tearing down …${NC}"; stop_all; exit 130' INT TERM

echo -e "${CYAN}[runner] streaming status (deploy_pid=$DEPLOY_PID bridge_pid=$BRIDGE_PID) …${NC}"
echo -e "${CYAN}    'pub tick' lines = VLA→SONIC publish rate (50 Hz)${NC}"
echo -e "${CYAN}    'video:' lines   = recorder frame counters (per output file)${NC}"
echo -e "${CYAN}    'CONTROL'        = SONIC ONNX is firing on the ZMQ pose stream${NC}"
echo -e "${CYAN}    'grav_z='        = body upright score (-1.0 = perfectly upright)${NC}"
echo

# Stream both logs side-by-side, filtered to the interesting lines, until
# the deploy exits. ``tail -F --pid`` makes ``tail`` self-terminate when
# ``$DEPLOY_PID`` dies, which is exactly when MAX_DURATION elapses (or
# the deploy crashes).
( tail -n 0 -F --pid="$DEPLOY_PID" "$RUN_DIR/bridge.log" 2>/dev/null \
    | stdbuf -oL grep -E 'pub tick|inference|video:|deploy_alive|render error|video render warn|video thread done' \
    | sed -u 's/^/[bridge] /' ) &
TAIL_BRIDGE_PID=$!
( tail -n 0 -F --pid="$DEPLOY_PID" "$RUN_DIR/deploy.log" 2>/dev/null \
    | stdbuf -oL grep -E 'CONTROL|grav_z|HANDOFF|POLICY|band release|tilt|fall|deploy: stopping|max-duration' \
    | sed -u 's/^/[deploy] /' ) &
TAIL_DEPLOY_PID=$!

# Block until deploy exits, then clean up.
wait "$DEPLOY_PID" 2>/dev/null || true

echo
echo -e "${YELLOW}[runner] deploy exited (max-duration elapsed or crash); flushing bridge …${NC}"
# Give the bridge ~3s to notice the deploy is gone (the freshness
# watchdog triggers at DEPLOY_ALIVE_STALE_THRESHOLD_S=1.0s, plus a
# render+encode window). Then hard-stop if it's still alive.
for _ in $(seq 1 6); do
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done
if kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo -e "${YELLOW}[runner] bridge still alive after 6s, sending SIGTERM …${NC}"
    kill "$BRIDGE_PID" 2>/dev/null || true
    sleep 2
fi

# Stop the tail helpers before they exit on their own.
kill "$TAIL_BRIDGE_PID" "$TAIL_DEPLOY_PID" 2>/dev/null || true
wait "$TAIL_BRIDGE_PID" "$TAIL_DEPLOY_PID" 2>/dev/null || true

stop_all

# Quick sanity report so the user doesn't have to ffprobe by hand.
echo
echo -e "${CYAN}=== run summary ===${NC}"
for f in "$RUN_DIR/ego_view.mp4" "$RUN_DIR/front_view.mp4"; do
    if [[ -f "$f" ]]; then
        info=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=width,height,nb_frames,duration \
            -of default=nokey=1:noprint_wrappers=1 "$f" 2>/dev/null \
            | tr '\n' ' ')
        printf "  %-25s %s\n" "$(basename "$f")" "$info"
    else
        printf "  %-25s ${RED}MISSING${NC}\n" "$(basename "$f")"
    fi
done
echo
echo -e "${GREEN}[runner] done. Logs + videos in $RUN_DIR${NC}"
