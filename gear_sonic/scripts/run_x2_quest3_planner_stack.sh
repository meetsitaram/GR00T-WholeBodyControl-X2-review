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
#       [--planner {kplanner,heuristic}]
#       [--kplanner-vqvae-ckpt PATH] [--kplanner-pose-ckpt PATH]
#       [--kplanner-root-ckpt PATH] [--kplanner-warmup-qpos PATH]
#       [--kplanner-device DEV] [--kplanner-replan-threshold-frames N]
#       [--kplanner-python PATH]    # e.g. ~/miniconda3/envs/env_isaaclab/bin/python
#                                   # for Blackwell (RTX 5090) sm_120 support
#       [--kplanner-stick-shape-exp FLOAT]
#                                   # locomotion/continuous stick power-curve
#                                   # exponent (>0). 1.0=linear (default),
#                                   # 0.5=push-stick-a-little-and-walk-normal,
#                                   # 2.0=more deadzone-feel. Tunes the
#                                   # Quest3 L-stick analog response without
#                                   # changing the peak velocity.
#       [--no-pose-feedback | --with-pose-feedback]
#       [--pose-feedback-host H] [--pose-feedback-port P]
#       [--pose-feedback-topic T] [--pose-feedback-max-age-s S]
#                                   # Default ON: kplanner SUBs robot_pose
#                                   # from the bridge (127.0.0.1:5570) and
#                                   # re-anchors current_root_wxyz from
#                                   # measured pelvis yaw every IDLE tick.
#                                   # Stops SONIC from twisting the body
#                                   # back to a stale yaw after fall
#                                   # recovery or sit/stand gesture
#                                   # sequences. Pass --no-pose-feedback to
#                                   # revert to open-loop diagnostic mode.
#       [--pose-ref-watchdog {auto,on,off}]
#                                   # auto (default): off in local sim,
#                                   # on for --remote-deploy / --no-deploy.
#                                   # The deploy's 0.5 s SAFE_IDLE watchdog
#                                   # is a wifi-safety guard; in local sim
#                                   # there is no wire to drop, the cold-
#                                   # start gap reliably trips it, and the
#                                   # 4x kd snap to default_angles collapses
#                                   # the robot. Pass ``on`` to exercise the
#                                   # split-topology safety path in sim.
#       [--rate FLOAT] [--log-dir PATH] [--cleanup-only] [--validate-only]
#       [--vla-bridge MODEL_DIR --vla-prompt STR
#        [--vla-device DEV] [--vla-rate FLOAT] [--vla-inference-period-s S]
#        [--vla-python PATH] [--vla-max-target-dev RAD] [--vla-target-lpf-hz HZ]]
#       [--pc2-host HOST]    # recommended one-arg split-topology
#       [--remote-deploy HOST [--resume-pub-port PORT]
#                              [--motor-monitor-port PORT]]   # legacy alias
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
#   # Split-topology / remote-deploy mode: laptop runs only the
#   # operator-side stack (manager + planner + recorder); the C++
#   # deploy + hand bridge + motor monitor are already running on PC2
#   # via gear_sonic_deploy/scripts/x2_pc2_daemons.sh start. The
#   # recorder's x2_debug SUB is auto-redirected to PC2; the manager
#   # binds the resume PUB on :5566 and SUBs the motor monitor at
#   # tcp://PC2_HOST:5567.
#   ./run_x2_quest3_planner_stack.sh --duration 0 \
#       --remote-deploy 10.0.1.41
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
# The kplanner can need a different python from the rest of the stack
# because its torch must support the local GPU's compute capability.
# Specifically, the base ``.venv`` torch (2.6+cu124) crashes on
# Blackwell (RTX 5090, sm_120); ``env_isaaclab`` ships torch 2.7+cu128
# which works. Override with --kplanner-python or env var KPLANNER_PYTHON
# to route just the kplanner subprocess to a different interpreter.
KPLANNER_PYTHON="${KPLANNER_PYTHON:-${PYTHON}}"

# PID files we need to clean up (planner writes its own; we own the rest).
# Resolved post-CLI based on --planner (heuristic vs kplanner). Both
# planner kinds use ``/tmp/x2_*_planner.pid`` so cleanup_stale can find
# whichever one was left behind by a previous run.
PLANNER_PID_FILE="/tmp/x2_kplanner.pid"
ALT_PLANNER_PID_FILE="/tmp/x2_heuristic_planner.pid"

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
# Gesture playback (live PKL takeover): recorder SUB.binds, transient
# play_gesture script PUB.connects. See
# gear_sonic/utils/teleop/gesture_session.py and the
# "Play a gesture mid-VR-session" section in clip_motion_commands.md.
GESTURE_CMD_PORT=5568   # play_gesture PUB -> recorder SUB
GESTURE_CMD_TOPIC="gesture_cmd"
GESTURE_CATALOG="${REPO_ROOT}/gear_sonic/data/motions/gestures/gestures_v1.yaml"

# --------------------------------------------------------------------------
# CLI defaults
# --------------------------------------------------------------------------

DURATION_S=0  # 0 = unlimited (run until Ctrl-C). Pass --duration N for a fixed N-sec cap.
WITH_RECORD=0
OUTPUT_DIR=""
TASK=""

# ── PC2 head-camera bridge (Orbbec + IMX900 stereo pair) ──────────────
# When ``--head-cameras`` is set, the wrapper auto-runs
# ``gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve`` against
# ``CAMERA_HOST:CAMERA_PORT`` before spawning the recorder, then
# forwards ``--head-cameras --camera-host --camera-port`` to
# record_x2_dataset so the three real head streams
# (observation.images.{head_front,stereo_left,stereo_right}) land in
# the LeRobot parquets alongside the synthetic ego_view. Only
# meaningful with ``--with-record``; ignored in --teleop-only.
# CAMERA_HOST defaults to PC2_HOST when ``--pc2-host`` is set so the
# operator can omit it on the typical real-robot invocation.
HEAD_CAMERAS=0
CAMERA_HOST=""
CAMERA_PORT="5555"
CAMERA_AUTOSTART=1
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
# --wrist-bypass {off,ik,ik-arms} forwarded to the deploy binary.
# 'ik' (default) overrides 4 wrist DOFs with the IK reference; 'ik-arms'
# extends that to the full 14-DOF arm so VR IK drives both arms direct
# to the motors while SONIC still controls legs+waist+head for balance.
# See docs/source/user_guide/milestones/2026-06-08_arm_bypass_v1.md.
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

# Planner kind. Default to ``kplanner`` (neural kinematic planner --
# MotionBricks VQVAE/pose/root checkpoints producing 31-DOF refs from a
# velocity intent). Set to ``heuristic`` to fall back to the
# primitives + bins state machine
# (``gear_sonic/scripts/x2_heuristic_planner.py``). Both planners use
# ``--sim-profile parity`` by default but bake from different sources:
# heuristic from ``planner.current_anchor_frame()`` (idle_stand[0] in
# primitives) and kplanner from its warmup quiet-stand qpos (the
# 38-D vector in ``x2_kplanner._build_default_warmup_qpos`` or the
# user-supplied ``--kplanner-warmup-qpos`` PKL). Operators can pass
# ``--sim-profile manual`` to opt out of parity, but be aware manual
# spawns the pelvis at z=0.85m on an elastic band -- the robot drops
# if no pose ref arrives during the cold-start window.
PLANNER_KIND="kplanner"
# Optional ckpt overrides for the kplanner. Empty -> defaults baked
# into x2_kplanner.py (pinned step checkpoints in each
# motionbricks_*_x2/version_1; see ``X2PlannerPaths.default()`` in
# motionbricks/.../load_x2_planner.py for the source of truth).
KPLANNER_VQVAE_CKPT=""
KPLANNER_POSE_CKPT=""
KPLANNER_ROOT_CKPT=""
KPLANNER_WARMUP_QPOS=""
KPLANNER_DEVICE="cuda"
# Worker thread refills the kplanner's ring buffer when frames_remaining
# < this value. Lower => fewer chained replans => fewer prediction-
# feedback links => less compounding yaw drift. The PKL replay diagnostic
# matrix (see "Deploy-integration diagnostics" in
# motionbricks/docs/x2_kplanner_evaluation.md) showed thresh=2 + a
# non-oscillating (continuous, not bucketed) intent stream raises
# forward tracking ~19% -> ~72%; the same lever applies here because
# Quest 3 with --enable-continuous-locomotion is the analog of the PKL
# replay's mean-intent mode (no per-tick velocity flapping). 16 was
# the original default; 2 is the empirically-validated shippable value.
KPLANNER_REPLAN_THRESHOLD_FRAMES="2"

# Runtime velocity tuning passed straight through to ``x2_kplanner.py``.
# Default empty -> let the daemon use its own defaults (yaw-lock OFF
# == epsilon 0.0 rad/s; all direction scales 1.0). Set
# KPLANNER_YAW_LOCK_EPSILON=0.05 to enable the legacy yaw-pin behaviour
# (pin commanded yaw to the persisted root quat whenever |yaw_rate| <
# 0.05 rad/s; useful when the model has yaw drift in forward-only walks
# but warning: it can also stop the robot from turning at all if set
# higher than the operator's stick yaw rate). Override via wrapper
# flags or environment variables when compensating for the model's L/R
# asymmetry or quelling yaw drift more / less aggressively.
KPLANNER_YAW_LOCK_EPSILON="${KPLANNER_YAW_LOCK_EPSILON:-}"
KPLANNER_TURN_LEFT_SCALE="${KPLANNER_TURN_LEFT_SCALE:-}"
KPLANNER_TURN_RIGHT_SCALE="${KPLANNER_TURN_RIGHT_SCALE:-}"
KPLANNER_FORWARD_SCALE="${KPLANNER_FORWARD_SCALE:-}"
KPLANNER_BACKWARD_SCALE="${KPLANNER_BACKWARD_SCALE:-}"
KPLANNER_LATERAL_SCALE="${KPLANNER_LATERAL_SCALE:-}"
# Continuous-locomotion stick shaping exponent (power curve). Empty ->
# x2_kplanner.py default (1.0 = linear). <1.0 = closer to bucketed feel
# (push stick a little -> robot already at moderate speed); >1.0 = more
# deadzone-feel (need to push the stick hard to move). Set via env
# ``KPLANNER_STICK_SHAPE_EXP=0.5`` for a faster walking feel.
KPLANNER_STICK_SHAPE_EXP="${KPLANNER_STICK_SHAPE_EXP:-}"
# Cold-start velocity ramp time constant (s). Smooths (yaw_rate, vel_x,
# vel_z) on every idle -> playing transition so the model's implied
# 2.13 s target doesn't jump 1 m+ ahead while the context buffer still
# holds 4 frames of static stand pose -- the "torso bends forward, no
# step on cold start" failure mode reported on 2026-05-30. Default
# empty -> x2_kplanner.py default (0.20 s, ~3-replan ramp at thresh=2).
# Set ``KPLANNER_COLD_START_RAMP_TAU_S=0`` to disable the ramp and
# reproduce the pre-fix behaviour.
KPLANNER_COLD_START_RAMP_TAU_S="${KPLANNER_COLD_START_RAMP_TAU_S:-}"
# Yaw-rate ceiling for continuous-locomotion R-stick turns (rad/s at
# full deflection). Default empty -> daemon default 0.75 rad/s (~43
# deg/s, a 90-deg turn in ~2.1 s). The legacy bucketed path stays at
# 1.5 rad/s for sharp pivots; this knob only affects analog R-stick
# turns. Halve it for even gentler turns; double to roughly match the
# bucketed feel. Tied to model training distribution -- raising past
# ~1.0 rad/s starts to overdrive the current X2 root model.
KPLANNER_CONTINUOUS_TURN_MAX_RAD_S="${KPLANNER_CONTINUOUS_TURN_MAX_RAD_S:-}"
# Forward-velocity floor (m/s) for continuous-locomotion. Default empty
# -> daemon default 0.30 m/s = lift any non-zero forward stick into
# the SONIC X2 root model's in-distribution forward-walk band on
# engagement. Below ~0.30 m/s the corpus has essentially no training
# samples; the model wiggles its hips without stepping and snaps into
# an unstable stride near full stick (operator-reported 2026-05-31).
# Only the analog forward channel is touched -- backward / lateral /
# yaw, bucketed forward intents, and PKL replay are unaffected.
# Override with ``KPLANNER_CONTINUOUS_FORWARD_MIN_MPS=0.40`` for a
# more aggressive lift-off, or ``=0`` to revert to the legacy raw
# proportional behaviour and accept the sub-0.3 m/s hip-wiggle.
KPLANNER_CONTINUOUS_FORWARD_MIN_MPS="${KPLANNER_CONTINUOUS_FORWARD_MIN_MPS:-}"
# Reference-step smoother knobs (publisher-side, step-detection driven).
# The smoother watches per-tick deltas on the lower-body joint reference
# (legs + waist by default; arms / head / fingers untouched so
# manipulation tasks driven by the same body_pose stream are unaffected)
# and arms a half-cosine ramp whenever the delta exceeds the trigger.
# Eliminates the audible motor click on stick push and release reported
# on real hardware 2026-05-31 (drivetrain backlash absorbing the torque
# step that high lower-body kp turns the reference step into; MuJoCo has
# no backlash so the symptom is silent in sim).
#
# Step-detection-driven (NOT FSM-driven): fires on any source of
# reference step -- stick push, stick release, future mode flips, future
# MC-handoff piping -- without needing to know which.
#
# KPLANNER_REF_SMOOTHER_MS: ramp duration in milliseconds. Default empty
# -> daemon default 300 ms (~natural period of the leg PD loop at
# deployed kp). Tune down to 150-200 ms if the ramp feels sluggish; up
# to 500 ms if residual clicks remain. Set 0 to disable entirely
# (passthrough).
KPLANNER_REF_SMOOTHER_MS="${KPLANNER_REF_SMOOTHER_MS:-}"
# KPLANNER_REF_SMOOTHER_TRIGGER_RAD: per-tick reference delta (rad) on
# any blended-channel joint that arms the ramp. Default empty -> daemon
# default 0.05 rad ~= 3 deg. Cleanly separates the multi-degree jumps a
# stick push/release creates from the per-tick neural-buffer motion
# under steady walking. Set 0 to make every non-zero delta a candidate
# (debug only; will fire ramps continuously during smooth walking).
KPLANNER_REF_SMOOTHER_TRIGGER_RAD="${KPLANNER_REF_SMOOTHER_TRIGGER_RAD:-}"
# KPLANNER_REF_SMOOTHER_SHAPE: halfcos | linear | off. Default empty ->
# daemon default 'halfcos' (C^1 smooth at both ramp endpoints; the
# piano-work proven shape). 'linear' is for A/B; 'off' is the
# single-flag revert (byte-equivalent passthrough).
KPLANNER_REF_SMOOTHER_SHAPE="${KPLANNER_REF_SMOOTHER_SHAPE:-}"
# KPLANNER_REF_SMOOTHER_JOINTS: lower_body | legs_only | all. Default
# empty -> daemon default 'lower_body' (legs 0-11 + waist 12-14).
# 'lower_body' is intentionally chosen to leave arms (15-28) and head
# (29-30) BYTE-EQUIVALENT to today's publisher output so manipulation
# tasks (Quest 3 hand IK, future bimanual reach) are unaffected by the
# ramp. 'legs_only' skips waist for max waist-twist responsiveness;
# 'all' blends all 31 DoFs (debug A/B). Fingers are on a separate
# command path and are never in this stream regardless of preset.
KPLANNER_REF_SMOOTHER_JOINTS="${KPLANNER_REF_SMOOTHER_JOINTS:-}"

# --------------------------------------------------------------------------
# Closed-loop pose feedback (default ON). The kplanner subscribes to the
# sim bridge's robot_pose:5570 PUB (real-robot bridge publishes the same
# topic on the deploy side) and uses the latest pelvis quat to re-anchor
# its IDLE_LOOP-published ``current_root_wxyz`` every tick. Without this
# the kplanner publishes whatever yaw it last persisted from a PLAYING
# segment, and SONIC's policy actively twists the real robot back to
# that stale heading -- visible as
#   * "robot snaps back after fall recovery"
#   * "robot rotates back to the original direction after a sit+stand
#      gesture sequence"
# See gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/tokenizer_obs.cpp
# (rel = inv(measured) * reference) for why the published yaw is what
# the policy drives toward.
#
# Pass --no-pose-feedback to opt back into open-loop (regression /
# diagnostic baseline). MAX_AGE_S guards against a frozen bridge:
# observations older than this are ignored on each tick, falling back
# to the previously-persisted yaw.
WITH_POSE_FEEDBACK="${WITH_POSE_FEEDBACK:-1}"
POSE_FEEDBACK_HOST="${POSE_FEEDBACK_HOST:-127.0.0.1}"
POSE_FEEDBACK_PORT="${POSE_FEEDBACK_PORT:-5570}"
POSE_FEEDBACK_TOPIC="${POSE_FEEDBACK_TOPIC:-robot_pose}"
POSE_FEEDBACK_MAX_AGE_S="${POSE_FEEDBACK_MAX_AGE_S:-0.5}"

# --------------------------------------------------------------------------
# x2_debug -> robot_pose bridge (real-robot only by default). On the real
# robot, the laptop has NO native ``robot_pose`` publisher -- that topic is
# only emitted by ``x2_mujoco_ros_bridge.py`` in sim. Without a republisher
# the kplanner pose-feedback SUB above is a no-op and ``current_root_wxyz``
# stays pinned at its R_z(0) warmup default. The first frame the kplanner
# publishes then hands the C++ deploy an identity-yaw reference and the
# policy actively twists the body back to world +X (the
# "robot turns back to default orientation as soon as I start the VR
# planner stack" symptom).
#
# The bridge SUBs the deploy's ``x2_debug:5557`` PUB over the network
# (PC2 -> laptop), extracts the IMU ``base_quat`` field, and re-PUBs it as
# JSON ``robot_pose`` on ``localhost:5570`` -- identical wire format to the
# MuJoCo bridge so the kplanner consumes both the same way. Auto-enabled
# whenever the stack is launched in --no-deploy mode (split-topology /
# remote deploy), since that's the only configuration where the symptom
# applies; sim runs have the MuJoCo bridge supplying robot_pose directly.
# Pass --x2-debug-bridge-host to override, --no-x2-debug-bridge to opt out.
WITH_X2_DEBUG_BRIDGE="${WITH_X2_DEBUG_BRIDGE:-auto}"
# Deliberately NOT defaulting from any ambient PC2_HOST env-var: a
# stale shell export silently pointing the bridge at the wrong robot
# is the kind of failure mode that's easy to miss in a banner and
# catastrophic on real hardware (kplanner happily seeds yaw from
# whatever IMU answers and then twists the live robot toward a
# different machine's heading). Make the host explicit per-invocation:
# pass --x2-debug-bridge-host, --remote-deploy, or --no-x2-debug-bridge.
X2_DEBUG_BRIDGE_HOST=""
X2_DEBUG_BRIDGE_PORT="${X2_DEBUG_BRIDGE_PORT:-5557}"
X2_DEBUG_BRIDGE_TOPIC="${X2_DEBUG_BRIDGE_TOPIC:-x2_debug}"
X2_DEBUG_BRIDGE_RATE_CAP_HZ="${X2_DEBUG_BRIDGE_RATE_CAP_HZ:-200.0}"

# --------------------------------------------------------------------------
# Split-topology / remote-deploy mode. When --remote-deploy HOST is set
# we treat the laptop as the operator-side stack ONLY (manager + planner
# + recorder); the C++ deploy + hand bridge + motor monitor are assumed
# to be already running on PC2 (the robot's Jetson Orin NX) under
# x2_pc2_daemons.sh start. The wrapper flips three things to make this
# work over the wire:
#
#   1. WITH_DEPLOY is forced to 0 (no local sim docker spawn).
#   2. The recorder's x2_debug SUB is pointed at tcp://${REMOTE_DEPLOY_HOST}
#      :${DEBUG_PORT} so it consumes the deploy's telemetry from PC2.
#   3. The manager binds the resume PUB on tcp://0.0.0.0:${RESUME_PUB_PORT}
#      (so the deploy on PC2 can SUB the A+B chord) and SUBs the motor
#      monitor at tcp://${REMOTE_DEPLOY_HOST}:${MOTOR_MONITOR_PORT} so
#      every JSONL frame the on-bot monitor publishes lands in the
#      manager_sidecar.jsonl on the laptop.
#
# Use it like:
#
#   ./run_x2_quest3_planner_stack.sh --remote-deploy 10.0.1.41 --duration 0
#
# The --robocasa-env path is ignored in remote-deploy mode (the real
# robot has no robocasa scene).
# --------------------------------------------------------------------------
REMOTE_DEPLOY_HOST=""
RESUME_PUB_PORT=5566
MOTOR_MONITOR_PORT=5567
RESUME_PUB_TOPIC="pose_resume"
MOTOR_MONITOR_TOPIC="motor_monitor"

# --------------------------------------------------------------------------
# Canonical "where is the X2 robot" flag. One arg, everything else
# implied. Specifically, setting --pc2-host HOST is equivalent to:
#   * --no-deploy               (the deploy is on HOST, not laptop)
#   * --remote-deploy HOST      (recorder x2_debug SUB redirect, manager
#                               motor_monitor SUB, manager resume PUB)
#   * --x2-debug-bridge-host HOST  (kplanner pose-feedback bridge)
# unless those flags were already passed explicitly (in which case the
# explicit value wins -- useful for the rare case where the bridge needs
# to talk to a different host than the deploy, e.g. multi-robot test
# rigs). The older flags remain accepted so commands.md /
# gesture_commands.md / sample_commands.md don't break, but --pc2-host
# is the recommended path going forward and matches the
# x2_pc2_daemons.sh / pc2_bringup.sh vocabulary.
#
# Deliberately NOT defaulted from any ambient ${PC2_HOST} env-var: a
# stale shell export silently pointing the planner stack at the wrong
# robot is the kind of failure mode that's easy to miss in a banner and
# catastrophic on real hardware (kplanner happily seeds yaw from
# whatever IMU answers and then twists the live robot toward a
# different machine's heading). Make the host explicit per-invocation
# even though x2_pc2_daemons.sh does inherit from the env -- the
# planner stack actively drives the robot whereas the daemons script
# is purely an ssh wrapper, so a mismatch here has a much higher cost.
PC2_HOST=""

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

# --------------------------------------------------------------------------
# Pose-ref starvation watchdog (deploy-side, since 2026-05-16). The C++
# deploy ships a watchdog that flips CONTROL -> SAFE_IDLE when no pose-ref
# frame has been received for ``--pose-ref-stale-s`` seconds (default 0.5
# s) and holds ``default_angles`` with 4x kd until BOTH (a) fresh frames
# resume for 1 s AND (b) the operator sends the resume chord on
# ``pose_resume``. The watchdog is essential for the real-robot
# split-topology path (laptop -> wifi -> PC2): if the operator wire
# stalls mid-task, the watchdog catches the freeze-while-leaning failure
# mode and the proxy daemon (x2_pc2_daemons.sh -> x2_pose_proxy.py) keeps
# the wire flowing with idle frames so the watchdog only ever fires on a
# real fault.
#
# In LOCAL SIM the entire stack is localhost: there's no wire to lose,
# the wrapper's cleanup trap kills the whole tree if any process dies,
# and the cold-start ordering (Step 1 deploy -> Step 4 recorder) opens a
# multi-second gap where ``pose:5556`` has zero subscribers AND zero
# publishers. The watchdog trips ~0.5 s after CONTROL with no upstream,
# pulls every DOF to ``default_angles`` with 4x kd, and (depending on
# the spawn pose <-> default_angles delta) tips the robot. Net: in
# local sim the watchdog protects against nothing and reliably collapses
# the robot during cold-start.
#
# Default is ``auto`` -> on for split-topology (--remote-deploy / VLA),
# off for local sim. Override with --pose-ref-watchdog {on,off,auto}.
# When effectively off, ``--disable-pose-ref-watchdog`` is forwarded to
# the deploy binary via --deploy-extra-arg (deploy_x2.sh doesn't expose
# the flag as a first-class wrapper arg; x2_pc2_daemons.sh uses the
# same passthrough trick).
# --------------------------------------------------------------------------
POSE_REF_WATCHDOG="auto"

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
        --head-cameras) HEAD_CAMERAS=1; shift ;;
        --no-head-cameras) HEAD_CAMERAS=0; shift ;;
        --camera-host) CAMERA_HOST="$2"; shift 2 ;;
        --camera-port) CAMERA_PORT="$2"; shift 2 ;;
        --no-camera-autostart) CAMERA_AUTOSTART=0; shift ;;
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
        --planner) PLANNER_KIND="$2"; shift 2 ;;
        --kplanner-vqvae-ckpt) KPLANNER_VQVAE_CKPT="$2"; shift 2 ;;
        --kplanner-pose-ckpt)  KPLANNER_POSE_CKPT="$2";  shift 2 ;;
        --kplanner-root-ckpt)  KPLANNER_ROOT_CKPT="$2";  shift 2 ;;
        --kplanner-warmup-qpos) KPLANNER_WARMUP_QPOS="$2"; shift 2 ;;
        --kplanner-device) KPLANNER_DEVICE="$2"; shift 2 ;;
        --kplanner-replan-threshold-frames) KPLANNER_REPLAN_THRESHOLD_FRAMES="$2"; shift 2 ;;
        --kplanner-python) KPLANNER_PYTHON="$2"; shift 2 ;;
        --kplanner-yaw-lock-epsilon) KPLANNER_YAW_LOCK_EPSILON="$2"; shift 2 ;;
        --kplanner-turn-left-scale) KPLANNER_TURN_LEFT_SCALE="$2"; shift 2 ;;
        --kplanner-turn-right-scale) KPLANNER_TURN_RIGHT_SCALE="$2"; shift 2 ;;
        --kplanner-forward-scale) KPLANNER_FORWARD_SCALE="$2"; shift 2 ;;
        --kplanner-backward-scale) KPLANNER_BACKWARD_SCALE="$2"; shift 2 ;;
        --kplanner-lateral-scale) KPLANNER_LATERAL_SCALE="$2"; shift 2 ;;
        --kplanner-stick-shape-exp) KPLANNER_STICK_SHAPE_EXP="$2"; shift 2 ;;
        --kplanner-cold-start-ramp-tau-s) KPLANNER_COLD_START_RAMP_TAU_S="$2"; shift 2 ;;
        --kplanner-continuous-turn-max-rad-s) KPLANNER_CONTINUOUS_TURN_MAX_RAD_S="$2"; shift 2 ;;
        --kplanner-continuous-forward-min-mps) KPLANNER_CONTINUOUS_FORWARD_MIN_MPS="$2"; shift 2 ;;
        --kplanner-ref-smoother-ms) KPLANNER_REF_SMOOTHER_MS="$2"; shift 2 ;;
        --kplanner-ref-smoother-trigger-rad) KPLANNER_REF_SMOOTHER_TRIGGER_RAD="$2"; shift 2 ;;
        --kplanner-ref-smoother-shape) KPLANNER_REF_SMOOTHER_SHAPE="$2"; shift 2 ;;
        --kplanner-ref-smoother-joints) KPLANNER_REF_SMOOTHER_JOINTS="$2"; shift 2 ;;
        --no-pose-feedback) WITH_POSE_FEEDBACK=0; shift ;;
        --with-pose-feedback) WITH_POSE_FEEDBACK=1; shift ;;
        --pose-feedback-host) POSE_FEEDBACK_HOST="$2"; WITH_POSE_FEEDBACK=1; shift 2 ;;
        --pose-feedback-port) POSE_FEEDBACK_PORT="$2"; WITH_POSE_FEEDBACK=1; shift 2 ;;
        --pose-feedback-topic) POSE_FEEDBACK_TOPIC="$2"; WITH_POSE_FEEDBACK=1; shift 2 ;;
        --pose-feedback-max-age-s) POSE_FEEDBACK_MAX_AGE_S="$2"; WITH_POSE_FEEDBACK=1; shift 2 ;;
        --no-x2-debug-bridge) WITH_X2_DEBUG_BRIDGE=0; shift ;;
        --with-x2-debug-bridge) WITH_X2_DEBUG_BRIDGE=1; shift ;;
        --x2-debug-bridge-host) X2_DEBUG_BRIDGE_HOST="$2"; WITH_X2_DEBUG_BRIDGE=1; shift 2 ;;
        --x2-debug-bridge-port) X2_DEBUG_BRIDGE_PORT="$2"; WITH_X2_DEBUG_BRIDGE=1; shift 2 ;;
        --x2-debug-bridge-topic) X2_DEBUG_BRIDGE_TOPIC="$2"; WITH_X2_DEBUG_BRIDGE=1; shift 2 ;;
        --x2-debug-bridge-rate-cap-hz) X2_DEBUG_BRIDGE_RATE_CAP_HZ="$2"; WITH_X2_DEBUG_BRIDGE=1; shift 2 ;;
        --quest3-continuous-yaw-max) QUEST3_CONTINUOUS_YAW_MAX="$2"; shift 2 ;;
        --loco-decoupled-arms) LOCO_DECOUPLED_ARMS="$2"; shift 2 ;;
        --pose-ref-watchdog) POSE_REF_WATCHDOG="$2"; shift 2 ;;
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
        --pc2-host) PC2_HOST="$2"; shift 2 ;;
        --remote-deploy) REMOTE_DEPLOY_HOST="$2"; shift 2 ;;
        --resume-pub-port) RESUME_PUB_PORT="$2"; shift 2 ;;
        --motor-monitor-port) MOTOR_MONITOR_PORT="$2"; shift 2 ;;
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

# Resolve planner kind. The kplanner is the default for new runs.
case "${PLANNER_KIND}" in
    kplanner)
        PLANNER_PID_FILE="/tmp/x2_kplanner.pid"
        ALT_PLANNER_PID_FILE="/tmp/x2_heuristic_planner.pid"
        ;;
    heuristic)
        PLANNER_PID_FILE="/tmp/x2_heuristic_planner.pid"
        ALT_PLANNER_PID_FILE="/tmp/x2_kplanner.pid"
        ;;
    *)
        echo "ERROR: --planner must be one of 'kplanner' (default) or 'heuristic'; got '${PLANNER_KIND}'" >&2
        exit 1
        ;;
esac

# Parity RSI: each planner kind has its own anchor PKL since the
# heuristic derives the anchor from ``planner.current_anchor_frame()``
# (idle_stand[0] from primitives) and the kplanner derives it from
# its warmup quiet-stand qpos. Repoint SIM_RSI_PKL accordingly so the
# deploy's ``--motion <PKL>`` spawns at a pose byte-identical to what
# the upstream planner's tick 0 emits. Both planners default to
# ``--sim-profile parity`` to avoid the manual-profile mid-air spawn.
if [[ "${PLANNER_KIND}" == "kplanner" ]]; then
    SIM_RSI_PKL="${REPO_ROOT}/data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_kplanner_rsi_anchor.pkl"
fi

# --pc2-host fan-out. One arg, three internal vars: REMOTE_DEPLOY_HOST
# (recorder/manager split-topology wiring), X2_DEBUG_BRIDGE_HOST
# (kplanner pose-feedback bridge), and the implicit --no-deploy below.
# Explicit overrides win, so an operator who genuinely needs to point
# the bridge at a different host than the deploy can still do that with
# --x2-debug-bridge-host on top of --pc2-host. A mismatch between
# --pc2-host and --remote-deploy is almost certainly a typo and we
# bail out loudly rather than silently honour one and drop the other.
if [[ -n "${PC2_HOST}" ]]; then
    # Camera bridge runs on PC2; auto-target the same host when the
    # operator didn't override --camera-host explicitly. Same fanout
    # philosophy as REMOTE_DEPLOY_HOST / X2_DEBUG_BRIDGE_HOST.
    if [[ -z "${CAMERA_HOST}" ]]; then
        CAMERA_HOST="${PC2_HOST}"
    fi
    if [[ -n "${REMOTE_DEPLOY_HOST}" && "${REMOTE_DEPLOY_HOST}" != "${PC2_HOST}" ]]; then
        echo "ERROR: --pc2-host '${PC2_HOST}' contradicts --remote-deploy '${REMOTE_DEPLOY_HOST}'." >&2
        echo "       Pass one or make them match." >&2
        exit 1
    fi
    REMOTE_DEPLOY_HOST="${PC2_HOST}"
    if [[ -z "${X2_DEBUG_BRIDGE_HOST}" ]]; then
        X2_DEBUG_BRIDGE_HOST="${PC2_HOST}"
        WITH_X2_DEBUG_BRIDGE=1
    fi
fi

# Remote-deploy gating. Keep this BEFORE the --validate-only short-
# circuit so the operator can sanity-check the resolved banner with
# both --remote-deploy and --validate-only set together.
if [[ -n "${REMOTE_DEPLOY_HOST}" ]]; then
    if [[ "${VLA_MODE}" -eq 1 ]]; then
        echo "ERROR: --remote-deploy is not yet supported with --vla-bridge / --vla-no-policy" >&2
        echo "       (VLA mode runs the bridge alongside the deploy on the laptop;" >&2
        echo "        split-topology requires both to live on PC2)." >&2
        exit 1
    fi
    if [[ "${ROBOCASA_ENV}" != "none" ]]; then
        echo "ERROR: --remote-deploy is for the real robot (PC2); --robocasa-env is sim-only." >&2
        exit 1
    fi
    if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
        # Operator passed --remote-deploy without explicit --no-deploy --
        # quietly imply it.
        WITH_DEPLOY=0
    fi
fi

# An explicit --x2-debug-bridge-host means "the deploy is on that other
# host and I want its IMU yaw piped to my kplanner" -- which is exactly
# the split-topology case. Imply --no-deploy the same way --remote-deploy
# does, so the operator doesn't need to remember to pass both. Note we
# do NOT auto-populate REMOTE_DEPLOY_HOST: that flag wires additional
# split-topology surfaces (recorder x2_debug SUB redirect, manager
# motor_monitor SUB, etc.) and should stay opt-in.
if [[ -n "${X2_DEBUG_BRIDGE_HOST}" && "${WITH_DEPLOY}" -eq 1 ]]; then
    WITH_DEPLOY=0
fi

# Resolve --x2-debug-bridge=auto. We need it whenever the deploy is NOT
# running locally on the laptop -- in those cases nothing else publishes
# robot_pose:5570 and the kplanner pose-feedback would be a no-op
# without the bridge. Sim runs (WITH_DEPLOY=1) get robot_pose from the
# MuJoCo bridge directly so the x2_debug bridge is redundant.
#
# Host resolution: explicit --x2-debug-bridge-host wins; else inherit
# from --remote-deploy HOST when provided; else error out (we can't
# guess where PC2 is and silently leaving the bridge off would put us
# right back in the snap-back symptom).
case "${WITH_X2_DEBUG_BRIDGE}" in
    auto)
        if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
            WITH_X2_DEBUG_BRIDGE=0
        else
            WITH_X2_DEBUG_BRIDGE=1
        fi
        ;;
    0|1) ;;
    *)
        echo "ERROR: --with/--no-x2-debug-bridge must be 0|1 (got '${WITH_X2_DEBUG_BRIDGE}')" >&2
        exit 1
        ;;
esac
if [[ "${WITH_X2_DEBUG_BRIDGE}" -eq 1 && -z "${X2_DEBUG_BRIDGE_HOST}" ]]; then
    if [[ -n "${REMOTE_DEPLOY_HOST}" ]]; then
        X2_DEBUG_BRIDGE_HOST="${REMOTE_DEPLOY_HOST}"
    else
        echo "ERROR: x2_debug bridge is ON (split-topology) but no host was" >&2
        echo "       given on the command line. Pass one of:" >&2
        echo "         --pc2-host <PC2_IP>             # recommended (one-arg)" >&2
        echo "         --x2-debug-bridge-host <PC2_IP> # bridge-only override" >&2
        echo "         --remote-deploy <PC2_IP>        # legacy alias" >&2
        echo "         --no-x2-debug-bridge            # opt out" >&2
        echo "" >&2
        echo "       The bridge host is required per-invocation by design --" >&2
        echo "       silently inheriting it from an ambient env-var risks" >&2
        echo "       pointing the kplanner's pose-feedback at the wrong" >&2
        echo "       robot and yawing the live one toward that other" >&2
        echo "       machine's heading. Opting out with --no-x2-debug-bridge" >&2
        echo "       leaves the kplanner pose-feedback as a no-op (the C++" >&2
        echo "       bootstrap fallback still protects against starvation" >&2
        echo "       snap-back, but kplanner startup may publish a stale" >&2
        echo "       R_z(0) reference for the first few ticks)." >&2
        exit 1
    fi
fi

# Resolve --pose-ref-watchdog ``auto`` to a concrete on/off setting.
# ``auto`` -> off when the deploy runs locally (WITH_DEPLOY=1); on
# otherwise (split-topology --remote-deploy / --no-deploy paths where
# the deploy lives behind a wifi hop and the watchdog is the only thing
# protecting against a freeze-while-leaning stall). See the long
# rationale next to ``POSE_REF_WATCHDOG=`` above. Anything not in
# {on,off,auto} is a CLI typo -- fail loudly so the operator notices.
case "${POSE_REF_WATCHDOG}" in
    auto)
        if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
            POSE_REF_WATCHDOG_RESOLVED="off"
        else
            POSE_REF_WATCHDOG_RESOLVED="on"
        fi
        ;;
    on|off)
        POSE_REF_WATCHDOG_RESOLVED="${POSE_REF_WATCHDOG}"
        ;;
    *)
        echo "ERROR: --pose-ref-watchdog must be one of 'auto' (default), 'on', or 'off'; got '${POSE_REF_WATCHDOG}'" >&2
        exit 1
        ;;
esac

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
    log "cleanup_stale: PID files=${PLANNER_PID_FILE},${ALT_PLANNER_PID_FILE} ports=${POSE_PORT},${DEBUG_PORT},${PLANNER_CMD_PORT},${ARM_HANDS_PORT},${BODY_POSE_PORT}"
    local pid_path
    for pid_path in "${PLANNER_PID_FILE}" "${ALT_PLANNER_PID_FILE}"; do
        if [[ -f "${pid_path}" ]]; then
            local stale_pid
            stale_pid="$(cat "${pid_path}" 2>/dev/null || true)"
            kill_pid_quiet "${stale_pid}" "stale planner (${pid_path})"
            rm -f "${pid_path}"
        fi
    done
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
X2_DEBUG_BRIDGE_PID=""

cleanup_children() {
    log "shutting down children (reverse spawn order)..."
    if [[ -n "${RECORDER_PGID}" ]]; then
        # Recorder wrapped in setsid -> pgid==pid. Use SIGINT+grace so
        # the LeRobot writer can flush a last episode.
        kill_pgid_graceful "${RECORDER_PGID}" "recorder" 8
    fi
    kill_pid_quiet "${MANAGER_PID}"  "manager"
    kill_pid_quiet "${PLANNER_PID}"  "planner"
    # x2_debug -> robot_pose bridge: stateless republisher, single
    # SIGTERM is fine.
    kill_pid_quiet "${X2_DEBUG_BRIDGE_PID}" "x2_debug_bridge"
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
# missing curated primitives can still drive a closed-loop demo. The
# neural kplanner also doesn't consume them (it streams from trained
# checkpoints instead).
if [[ "${VLA_MODE}" -eq 0 && "${PLANNER_KIND}" == "heuristic" ]]; then
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
# kplanner preflight: verify the three MotionBricks checkpoints are
# present unless the operator overrode them explicitly via
# ``--kplanner-*-ckpt`` (in which case we trust the path and let the
# Python daemon raise a clean FileNotFoundError on boot).
if [[ "${VLA_MODE}" -eq 0 && "${PLANNER_KIND}" == "kplanner" ]]; then
    # Must mirror x2_kplanner.py's argparse defaults + load_x2_planner.py's
    # X2PlannerPaths.default(). Pinned step checkpoints (not last.ckpt) so
    # a fresh training run doesn't silently re-point inference at an
    # unverified checkpoint.
    KPL_VQVAE_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0200000.ckpt"
    KPL_POSE_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_pose_x2_v2/version_1/checkpoints/model-step=0250000.ckpt"
    KPL_ROOT_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0235000.ckpt"
    for ck in "${KPLANNER_VQVAE_CKPT:-${KPL_VQVAE_DEFAULT}}" \
              "${KPLANNER_POSE_CKPT:-${KPL_POSE_DEFAULT}}" \
              "${KPLANNER_ROOT_CKPT:-${KPL_ROOT_DEFAULT}}"; do
        if [[ ! -f "${ck}" ]]; then
            err "kplanner checkpoint not found: ${ck}"
            err "Pass --planner heuristic to use the curated primitives"
            err "fallback, or override via --kplanner-{vqvae,pose,root}-ckpt."
            exit 1
        fi
    done
    # Import-time preflight: verify the chosen --kplanner-python can
    # actually load the kplanner stack. Previously a missing motionbricks
    # install (e.g. when KPLANNER_PYTHON is routed to env_isaaclab without
    # having pip-install -e'd motionbricks/) only surfaced AFTER the
    # deploy was already running, dropping the robot mid-air when the
    # planner crashed. We now fail fast BEFORE Step 1 (deploy spawn) so
    # MuJoCo physics never starts without a publisher.
    log "kplanner preflight: probing imports under ${KPLANNER_PYTHON}..."
    KPLANNER_PREFLIGHT_PYTHONPATH="${REPO_ROOT}/motionbricks:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    if ! PYTHONPATH="${KPLANNER_PREFLIGHT_PYTHONPATH}" \
            "${KPLANNER_PYTHON}" -c "
import sys
missing = []
for mod in [
    'torch', 'numpy', 'omegaconf', 'pytorch_lightning',
    'vector_quantize_pytorch', 'adam_atan2_pytorch', 'zmq',
    'motionbricks.motion_backbone.inference.load_x2_planner',
    'motionbricks.motion_backbone.inference.neural_planner',
    'gear_sonic.utils.planner.state_machine',
    'gear_sonic.utils.teleop.zmq.zmq_planner_sender',
]:
    try:
        __import__(mod)
    except Exception as exc:
        missing.append((mod, repr(exc)))
if missing:
    for mod, exc in missing:
        print(f'MISSING {mod}: {exc}', file=sys.stderr)
    sys.exit(1)
print('all kplanner imports OK')
" >/dev/null; then
        err "kplanner python cannot import its dependencies: ${KPLANNER_PYTHON}"
        err "(this would crash the daemon AFTER deploy spawns -> robot drops)"
        err ""
        err "Fix: either"
        err "  - install missing deps: ${KPLANNER_PYTHON} -m pip install \\"
        err "        pytorch-lightning vector-quantize-pytorch adam-atan2-pytorch"
        err "  - switch interpreter: --kplanner-python ${REPO_ROOT}/.venv/bin/python"
        err "  - fall back to heuristic: --planner heuristic"
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

# Resolve the laptop's primary IPv4 addresses so the WebXR banner can
# print concrete `wss://<ip>:port` URLs instead of a literal `<host>`
# placeholder. Filter out loopback, Docker bridges (172.16/12), and
# Tailscale CGNAT (100.64/10) -- the Quest 3 needs a LAN-routable IP
# (most often the laptop WiFi address on the same /24 as the headset).
# If nothing routable is found we fall back to the legacy placeholder.
# NOTE: pattern uses `\.[0-9]+\.[0-9]+\.[0-9]+` (no `{3}`) because the
# default awk on Ubuntu (mawk) doesn't enable POSIX interval expressions.
_LOCAL_IP_CANDIDATES="$(
    hostname -I 2>/dev/null \
        | tr ' ' '\n' \
        | awk '
            /^127\./ { next }
            /^172\.(1[6-9]|2[0-9]|3[01])\./ { next }
            /^100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\./ { next }
            /^169\.254\./ { next }
            /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ { print }
        ' \
        | paste -sd ',' -
)"
if [[ -z "${_LOCAL_IP_CANDIDATES}" ]]; then
    _LOCAL_IP_CANDIDATES="<host>"
fi

# Right-pad the "Browser URL" line of the ACTION-REQUIRED ASCII box so the
# closing │ still aligns when we splice in concrete IPs. Original literal
# content width was 70 chars; the static prefix "    Browser URL:  https://"
# (26) + ":" (1) + port + ip_len + suffix-spaces = 70. Clamp at 0 so a
# very long candidate string just wraps rather than failing arithmetic.
_BROWSER_URL_PAD_LEN=$(( 70 - 26 - 1 - ${#QUEST3_HTTP_PORT} - ${#_LOCAL_IP_CANDIDATES} ))
if (( _BROWSER_URL_PAD_LEN < 0 )); then
    _BROWSER_URL_PAD_LEN=0
fi
_BROWSER_URL_PAD="$(printf '%*s' "${_BROWSER_URL_PAD_LEN}" '')"

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
  WebXR (optional) : wss://${_LOCAL_IP_CANDIDATES}:${QUEST3_WS_PORT}, https://${_LOCAL_IP_CANDIDATES}:${QUEST3_HTTP_PORT}
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
  planner kind     : ${PLANNER_KIND}$([[ "${PLANNER_KIND}" == "kplanner" ]] && echo "  (neural; INTENT_VELOCITY_MAP -> motion_inference.predict)" || echo "  (curated primitives + bins state machine)")$([[ "${PLANNER_KIND}" == "kplanner" ]] && echo "
  kplanner device  : ${KPLANNER_DEVICE}  python: ${KPLANNER_PYTHON}" || true)
  planner demo     : ${PLANNER_DEMO:-(none -- planner sits in IDLE_LOOP at startup, awaits VR planner_cmd)}
  finger comp      : curl=$([[ "${APPLY_CURL_COMP}" -eq 1 ]] && echo on || echo off)  oppose=$([[ "${APPLY_OPPOSE_COMP}" -eq 1 ]] && echo on || echo off)$([[ -n "${ROBOCASA_SCENE_XML}" ]] && echo "  (robocasa default; pass --no-apply-{curl,oppose}-compensation to override)" || echo "  (pass --apply-{curl,oppose}-compensation to enable)")
  deploy           : $([[ "${WITH_DEPLOY}" -eq 1 ]] && echo "ON  (sim --vla, profile=${SIM_PROFILE}, viewer=$([[ "${SIM_VIEWER}" -eq 1 ]] && echo on || echo off))" || echo "OFF (assume external)")
  pc2 host         : $([[ -n "${PC2_HOST}" ]] && echo "${PC2_HOST}  (single-arg fan-out: drives the two split-topology lines below)" || echo "(not set; using --remote-deploy/--x2-debug-bridge-host directly)")
  remote-deploy    : $([[ -n "${REMOTE_DEPLOY_HOST}" ]] && echo "ON  -> ${REMOTE_DEPLOY_HOST} (recorder x2_debug SUB redirected; manager resume PUB :${RESUME_PUB_PORT}; manager motor_monitor SUB tcp://${REMOTE_DEPLOY_HOST}:${MOTOR_MONITOR_PORT})" || echo "off")
  x2_debug bridge  : $([[ "${WITH_X2_DEBUG_BRIDGE}" -eq 1 ]] && echo "ON  -> SUB tcp://${X2_DEBUG_BRIDGE_HOST}:${X2_DEBUG_BRIDGE_PORT}@${X2_DEBUG_BRIDGE_TOPIC} -> PUB tcp://127.0.0.1:${POSE_FEEDBACK_PORT}@${POSE_FEEDBACK_TOPIC} (rate-cap ${X2_DEBUG_BRIDGE_RATE_CAP_HZ}Hz; feeds kplanner pose-feedback so first frame doesn't twist back to world +X)" || echo "off (sim run or --no-x2-debug-bridge; kplanner pose-feedback may be a no-op)")
  pose-feedback    : $([[ "${WITH_POSE_FEEDBACK}" -eq 1 ]] && echo "ON  scope=${KPLANNER_POSE_RESEED_SCOPE:-none} (PLAYING reseed=$([[ "${KPLANNER_POSE_RESEED_SCOPE:-none}" == "none" ]] && echo "off, integrates open-loop" || echo "${KPLANNER_POSE_RESEED_SCOPE:-none}") -- IDLE yaw refresh still runs for snap-back protection)" || echo "OFF (--no-pose-feedback; kplanner publishes stale yaw refs but is fully open-loop)")
  pose-ref watchdog: ${POSE_REF_WATCHDOG_RESOLVED} (cli=${POSE_REF_WATCHDOG}; off=--disable-pose-ref-watchdog forwarded to deploy; on=C++ default 0.5 s SAFE_IDLE trip)
  ONNX model       : ${SIM_MODEL}
  motion_token     : $([[ -n "${SONIC_CHECKPOINT}" ]] && echo "ON  (${SONIC_CHECKPOINT}, ${SONIC_TOKENIZER_DEVICE})" || echo "DISABLED (action.motion_token = zeros; dataset will NOT be VLA-trainable)")
  encoder_config   : $([[ -n "${SONIC_CHECKPOINT}" && -n "${ENCODER_CONFIG}" ]] && echo "${ENCODER_CONFIG}, modes=[retargeted_body_q], multi-frame 10x68 -> 680-D" || ([[ -n "${SONIC_CHECKPOINT}" ]] && echo "DEPRECATED freeze-pose (--encoder-config '' was passed)" || echo "(unused; tokenizer DISABLED above)"))
  operator         : ${OPERATOR_ID} (${CALIBRATION_PATH})
  WebXR endpoint   : wss://${_LOCAL_IP_CANDIDATES}:${QUEST3_WS_PORT}, https://${_LOCAL_IP_CANDIDATES}:${QUEST3_HTTP_PORT}
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

    # Parity profile needs the bridge RSI PKL so MuJoCo spawns the
    # robot ON THE FLOOR in the exact pose the policy is about to
    # track (no elastic-band drop from 0.85m, no mid-air launch). Each
    # planner has its own anchor source:
    #
    #   - heuristic: planner.current_anchor_frame() (idle_stand[0]
    #     from the curated primitives PKL).
    #   - kplanner: the daemon's warmup quiet-stand qpos
    #     (``_build_default_warmup_qpos()`` by default, or the user-
    #     supplied --kplanner-warmup-qpos PKL).
    #
    # Auto-bake on first use so a fresh checkout / training rerun
    # doesn't surface as a robot collapse.
    if [[ "${SIM_PROFILE}" == "parity" ]]; then
        if [[ "${PLANNER_KIND}" == "heuristic" ]]; then
            if [[ ! -f "${SIM_RSI_PKL}" ]]; then
                log "RSI anchor PKL not found at ${SIM_RSI_PKL}; baking now ..."
                if ! "${PYTHON}" -m gear_sonic.scripts.bake_planner_rsi_anchor \
                        --primitives-pkl "${PRIMITIVES_PKL}" \
                        --bins-yaml "${BINS_YAML}" \
                        --out "${SIM_RSI_PKL}" \
                        >>"${LOG_DIR}/rsi_anchor_bake.log" 2>&1; then
                    err "failed to bake heuristic RSI anchor; see ${LOG_DIR}/rsi_anchor_bake.log"
                    exit 1
                fi
            fi
        else
            # kplanner: bake against the same warmup qpos the daemon
            # will publish on tick 0 so the bridge RSI = wire content.
            # Always re-bake (it's <300 ms and depends only on the
            # operator's --kplanner-warmup-qpos override) so changes
            # to the override take effect without an explicit rebuild.
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
        fi
        log "parity RSI source: ${SIM_RSI_PKL}  (planner=${PLANNER_KIND})"
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

    # Pose-ref starvation watchdog. ``off`` -> forward
    # ``--disable-pose-ref-watchdog`` to the C++ binary via deploy_x2.sh's
    # generic ``--deploy-extra-arg`` passthrough (see the long rationale
    # at ``POSE_REF_WATCHDOG=`` near the top of this file). In ``auto``
    # mode this resolves to off for local sim (everything is localhost,
    # there's nothing to starve over) and on for split-topology.
    if [[ "${POSE_REF_WATCHDOG_RESOLVED}" == "off" ]]; then
        DEPLOY_ARGS+=(--deploy-extra-arg --disable-pose-ref-watchdog)
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

# --------------------------------------------------------------------------
# Step 1.5 — x2_debug -> robot_pose bridge (split-topology only).
#
# Must come BEFORE the planner so by the time x2_kplanner.py opens its
# pose-feedback SUB the bridge is already publishing on
# tcp://127.0.0.1:${POSE_FEEDBACK_PORT}. Without this, the kplanner
# pose-feedback path is a no-op on real robot (no robot_pose publisher
# on the laptop), ``current_root_wxyz`` stays at its R_z(0) warmup
# default, and the first frame the kplanner publishes hands the deploy
# a stale identity-yaw reference -- which the policy faithfully twists
# back to. See the WITH_X2_DEBUG_BRIDGE comment block up top.
# --------------------------------------------------------------------------
if [[ "${WITH_X2_DEBUG_BRIDGE}" -eq 1 ]]; then
    X2_DEBUG_BRIDGE_LOG="${LOG_DIR}/x2_debug_bridge.log"
    log "Step 1.5/4 — spawning x2_debug -> robot_pose bridge"
    log "  upstream  SUB tcp://${X2_DEBUG_BRIDGE_HOST}:${X2_DEBUG_BRIDGE_PORT} topic=${X2_DEBUG_BRIDGE_TOPIC}"
    log "  downstream PUB tcp://127.0.0.1:${POSE_FEEDBACK_PORT} topic=${POSE_FEEDBACK_TOPIC}"
    log "  log -> ${X2_DEBUG_BRIDGE_LOG}"
    "${PYTHON}" -u -m gear_sonic_deploy.scripts.x2_debug_to_robot_pose_bridge \
        --x2-debug-host "${X2_DEBUG_BRIDGE_HOST}" \
        --x2-debug-port "${X2_DEBUG_BRIDGE_PORT}" \
        --x2-debug-topic "${X2_DEBUG_BRIDGE_TOPIC}" \
        --robot-pose-bind '*' \
        --robot-pose-port "${POSE_FEEDBACK_PORT}" \
        --rate-cap-hz "${X2_DEBUG_BRIDGE_RATE_CAP_HZ}" \
        >"${X2_DEBUG_BRIDGE_LOG}" 2>&1 &
    X2_DEBUG_BRIDGE_PID=$!
    log "  bridge spawned (pid=${X2_DEBUG_BRIDGE_PID}); not blocking on" \
        "first frame -- kplanner pose-feedback has its own max-age gate" \
        "and the bridge will catch up within ~5ms once a wifi packet arrives"
    sleep 0.3
    if ! kill -0 "${X2_DEBUG_BRIDGE_PID}" 2>/dev/null; then
        log "ERROR: x2_debug bridge died on startup; tail of log:"
        tail -n 40 "${X2_DEBUG_BRIDGE_LOG}" | sed 's/^/  /'
        exit 1
    fi
else
    log "Step 1.5/4 — x2_debug -> robot_pose bridge SKIPPED (sim run or --no-x2-debug-bridge)."
fi

if [[ "${VLA_MODE}" -eq 0 ]]; then

# --------------------------------------------------------------------------
# Step 2 — Spawn planner (SECOND). Configure for Phase 0 wire:
#   - PUB body_pose@5565 (instead of legacy direct-to-deploy pose@5556)
#   - SUB planner_cmd@5563 (manager drives state machine via Quest3 inputs)
# Wait for the "Phase 0 mode" log line so we know body_pose PUB is live.
# --------------------------------------------------------------------------

PLANNER_LOG="${LOG_DIR}/planner.log"

if [[ -n "${REMOTE_DEPLOY_HOST}" ]]; then
    # body_pose still goes to the local recorder, but bind on '*' so
    # someone debugging from a third host can SUB if needed. Recorder
    # is on the laptop, so localhost connects back through 127.0.0.1.
    PLANNER_PUB_HOST='*'
else
    PLANNER_PUB_HOST=127.0.0.1
fi

if [[ "${PLANNER_KIND}" == "heuristic" ]]; then
    PLANNER_ARGS=(
        -m gear_sonic.scripts.x2_heuristic_planner
        --primitives "${PRIMITIVES_PKL}"
        --bins "${BINS_YAML}"
        --pub-host "${PLANNER_PUB_HOST}"
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
else
    # kplanner mode (default). The neural daemon's command surface is
    # a strict subset of the heuristic's, so we forward the same VR
    # / scripted-demo / pub-host knobs and just swap the entry-point.
    PLANNER_ARGS=(
        -m gear_sonic.scripts.x2_kplanner
        --pub-host "${PLANNER_PUB_HOST}"
        --body-pose-port "${BODY_POSE_PORT}"
        --zmq-cmd-host localhost
        --zmq-cmd-port "${PLANNER_CMD_PORT}"
        --zmq-cmd-topic "${PLANNER_CMD_TOPIC}"
        --warmup-quiet-stand-s "${WARMUP_QUIET_STAND_S}"
        --pid-file "${PLANNER_PID_FILE}"
        --duration-s "${DURATION_S}"
        --device "${KPLANNER_DEVICE}"
        --replan-threshold-frames "${KPLANNER_REPLAN_THRESHOLD_FRAMES}"
    )
    if [[ -n "${PLANNER_DEMO}" ]]; then
        PLANNER_ARGS+=(--demo "${PLANNER_DEMO}")
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
    if [[ -n "${KPLANNER_YAW_LOCK_EPSILON}" ]]; then
        PLANNER_ARGS+=(--yaw-lock-epsilon "${KPLANNER_YAW_LOCK_EPSILON}")
    fi
    if [[ -n "${KPLANNER_TURN_LEFT_SCALE}" ]]; then
        PLANNER_ARGS+=(--turn-left-scale "${KPLANNER_TURN_LEFT_SCALE}")
    fi
    if [[ -n "${KPLANNER_TURN_RIGHT_SCALE}" ]]; then
        PLANNER_ARGS+=(--turn-right-scale "${KPLANNER_TURN_RIGHT_SCALE}")
    fi
    if [[ -n "${KPLANNER_FORWARD_SCALE}" ]]; then
        PLANNER_ARGS+=(--forward-scale "${KPLANNER_FORWARD_SCALE}")
    fi
    if [[ -n "${KPLANNER_BACKWARD_SCALE}" ]]; then
        PLANNER_ARGS+=(--backward-scale "${KPLANNER_BACKWARD_SCALE}")
    fi
    if [[ -n "${KPLANNER_LATERAL_SCALE}" ]]; then
        PLANNER_ARGS+=(--lateral-scale "${KPLANNER_LATERAL_SCALE}")
    fi
    if [[ -n "${KPLANNER_COLD_START_RAMP_TAU_S}" ]]; then
        PLANNER_ARGS+=(--cold-start-ramp-tau-s "${KPLANNER_COLD_START_RAMP_TAU_S}")
    fi
    if [[ -n "${KPLANNER_CONTINUOUS_TURN_MAX_RAD_S}" ]]; then
        PLANNER_ARGS+=(--continuous-turn-max-rad-s "${KPLANNER_CONTINUOUS_TURN_MAX_RAD_S}")
    fi
    if [[ -n "${KPLANNER_CONTINUOUS_FORWARD_MIN_MPS}" ]]; then
        PLANNER_ARGS+=(--continuous-forward-min-mps "${KPLANNER_CONTINUOUS_FORWARD_MIN_MPS}")
    fi
    if [[ -n "${KPLANNER_REF_SMOOTHER_MS}" ]]; then
        PLANNER_ARGS+=(--ref-smoother-ms "${KPLANNER_REF_SMOOTHER_MS}")
    fi
    if [[ -n "${KPLANNER_REF_SMOOTHER_TRIGGER_RAD}" ]]; then
        PLANNER_ARGS+=(--ref-smoother-trigger-rad "${KPLANNER_REF_SMOOTHER_TRIGGER_RAD}")
    fi
    if [[ -n "${KPLANNER_REF_SMOOTHER_SHAPE}" ]]; then
        PLANNER_ARGS+=(--ref-smoother-shape "${KPLANNER_REF_SMOOTHER_SHAPE}")
    fi
    if [[ -n "${KPLANNER_REF_SMOOTHER_JOINTS}" ]]; then
        PLANNER_ARGS+=(--ref-smoother-joints "${KPLANNER_REF_SMOOTHER_JOINTS}")
    fi
    if [[ -n "${KPLANNER_STICK_SHAPE_EXP}" ]]; then
        PLANNER_ARGS+=(--stick-shape-exp "${KPLANNER_STICK_SHAPE_EXP}")
    fi
    if [[ "${WITH_POSE_FEEDBACK}" -eq 1 ]]; then
        # Closed-loop yaw refresh: kplanner's IDLE_LOOP branch reads the
        # latest robot_pose pelvis quat each tick and overwrites the
        # persisted ``current_root_wxyz`` with R_z(measured_yaw) so SONIC
        # stops twisting the body back to a stale heading after fall
        # recovery / gesture overrides. See the "Closed-loop pose
        # feedback" comment near the defaults section for context.
        PLANNER_ARGS+=(
            --pose-feedback-host "${POSE_FEEDBACK_HOST}"
            --pose-feedback-port "${POSE_FEEDBACK_PORT}"
            --pose-feedback-topic "${POSE_FEEDBACK_TOPIC}"
            --pose-feedback-max-age-s "${POSE_FEEDBACK_MAX_AGE_S}"
        )
        # Reseed-scope hygiene. ``full_root`` (the kplanner default)
        # overwrites slots [0:7] of the planner's context buffer (xyz +
        # quat) with each reseed sample. Every scope other than
        # ``none`` regresses something in this stack:
        #
        #   * ``full_root``  -- on the real-robot x2_debug bridge the
        #                       publisher has no position sensor and
        #                       sends xy=z=0, teleporting the planner's
        #                       xy history to the origin every reseed
        #                       tick -> catastrophic instability during
        #                       walk / turn. AND on the local-sim path
        #                       the MuJoCo bridge's qpos is the
        #                       SONIC-tracked pose, which lags the
        #                       planner reference; feeding that lagged
        #                       pose back as "ground truth" each replan
        #                       collapses the planner's commanded motion
        #                       into the lag (operator-verified 2026-06:
        #                       robot crawls / barely turns on full
        #                       stick vs. responsive when the reseed is
        #                       off).
        #   * ``quat_only`` -- model integrates yaw open-loop and the
        #                       physical robot lags the reference
        #                       (inertia / tracking dynamics); pinning
        #                       the planner's neural-buffer quat to the
        #                       lagging measured yaw means the planner
        #                       can never get more than one replan-tick
        #                       ahead of the robot, and commanded turns
        #                       under-rotate.
        #   * ``none``      -- PLAYING reseed disabled entirely. The
        #                       model integrates open-loop during
        #                       PLAYING (turns work as commanded). The
        #                       pose_deque still feeds the IDLE_LOOP
        #                       yaw refresh, the startup yaw seed, and
        #                       the IDLE -> PLAYING transition seed --
        #                       all yaw-only writes to current_root_wxyz
        #                       only, no neural-buffer mutation. Net:
        #                       snap-back protection without the
        #                       PLAYING-side correctness penalty.
        #
        # Conclusion: ``scope=none`` is the right default for every path
        # we currently ship (real-bridge AND local-sim). Override with
        # ``KPLANNER_POSE_RESEED_SCOPE=full_root`` if you're A/B testing
        # against the old behaviour.
        POSE_RESEED_SCOPE_DEFAULT="none"
        POSE_RESEED_SCOPE="${KPLANNER_POSE_RESEED_SCOPE:-${POSE_RESEED_SCOPE_DEFAULT}}"
        PLANNER_ARGS+=(--pose-reseed-scope "${POSE_RESEED_SCOPE}")
        if [[ "${WITH_X2_DEBUG_BRIDGE}" -eq 1 ]]; then
            log "  closed-loop pose feedback: SUB tcp://${POSE_FEEDBACK_HOST}:${POSE_FEEDBACK_PORT} topic=${POSE_FEEDBACK_TOPIC} (max_age=${POSE_FEEDBACK_MAX_AGE_S}s, scope=${POSE_RESEED_SCOPE} -- IMU-only bridge; snap-back guarded by IDLE_LOOP + IDLE->PLAYING + startup yaw refreshes)"
        else
            log "  closed-loop pose feedback: SUB tcp://${POSE_FEEDBACK_HOST}:${POSE_FEEDBACK_PORT} topic=${POSE_FEEDBACK_TOPIC} (max_age=${POSE_FEEDBACK_MAX_AGE_S}s, scope=${POSE_RESEED_SCOPE} -- local-sim MuJoCo bridge; full_root regressed responsiveness, default flipped to none 2026-06)"
        fi
    else
        log "  closed-loop pose feedback: DISABLED (--no-pose-feedback); kplanner will publish stale yaw refs"
    fi

    log "Step 2/4 — spawning x2_kplanner -> ${PLANNER_LOG}"
    log "  using python: ${KPLANNER_PYTHON}"
    # The kplanner imports ``motionbricks.motion_backbone.*`` and
    # ``gear_sonic.*``. The default ``.venv`` has motionbricks installed
    # editable (pip install -e motionbricks/), but alternate pythons
    # routed via ``--kplanner-python`` (e.g. env_isaaclab for sm_120
    # support) typically don't. Inject PYTHONPATH explicitly so import
    # resolves regardless of which interpreter is selected; this is
    # idempotent on .venv runs because the editable install masks it.
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
# Remote-deploy split-topology flags. When --remote-deploy HOST is set
# we (a) enable the resume PUB so the operator's A+B chord on the
# Quest 3 reaches the deploy on PC2, and (b) point the motor-monitor
# SUB at PC2's :5567 so every JSONL frame the on-bot monitor publishes
# lands in the manager_sidecar.jsonl on the laptop.
if [[ -n "${REMOTE_DEPLOY_HOST}" ]]; then
    MANAGER_ARGS+=(
        --resume-pub-enabled
        --resume-pub-host '*'
        --resume-pub-port "${RESUME_PUB_PORT}"
        --resume-pub-topic "${RESUME_PUB_TOPIC}"
        --motor-monitor-host "${REMOTE_DEPLOY_HOST}"
        --motor-monitor-port "${MOTOR_MONITOR_PORT}"
        --motor-monitor-topic "${MOTOR_MONITOR_TOPIC}"
    )
fi
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
# Continuous-locomotion: forwards raw Quest3 stick deflections as
# ``locomotion / continuous`` so the kplanner gets analog control
# instead of binary on/off intent buckets. The heuristic planner has
# no consumer for this intent, so we only enable it in ``kplanner``
# mode. Override via env var KPLANNER_CONTINUOUS_LOCOMOTION={1,0}.
KPLANNER_CONTINUOUS_LOCOMOTION="${KPLANNER_CONTINUOUS_LOCOMOTION:-1}"
if [[ "${PLANNER_KIND}" == "kplanner" && "${KPLANNER_CONTINUOUS_LOCOMOTION}" -eq 1 ]]; then
    MANAGER_ARGS+=(--enable-continuous-locomotion)
fi
# Maximum R-stick X amplitude forwarded as stick_yaw in continuous-
# locomotion mode. Defaults to 0.5 (= half deflection) which the
# operator validated on 2026-05-30 as "turns look smooth now" against
# the current X2 root-model checkpoint. Combined with the kplanner's
# 0.75 rad/s yaw ceiling that gives a 0.375 rad/s (~21 deg/s) full-
# stick turn -- inside what the policy can track without commit-and-
# overshoot from brief R-stick bursts.
#
# Lower for gentler demo turns, 1.0 to restore the legacy "full stick
# = full planner ceiling" mapping for A/B comparison. Lives on the
# teleop side so it shapes operator feel without changing the
# planner's physical yaw-rate ceiling
# (KPLANNER_CONTINUOUS_TURN_MAX_RAD_S handles that downstream concern
# separately).
QUEST3_CONTINUOUS_YAW_MAX="${QUEST3_CONTINUOUS_YAW_MAX:-0.5}"
MANAGER_ARGS+=(--continuous-yaw-max "${QUEST3_CONTINUOUS_YAW_MAX}")
# LOCO_DECOUPLED_ARMS=1 (default) -> manager publishes its frozen arm
# pose to the recorder during LOCOMOTION and the recorder overrides
# the planner's predicted arms. Required for the ARM_MANIPULATION ->
# LOCOMOTION arm-hold workflow (operator positions arms in ARM_MAN,
# toggles to LOCOMOTION to walk to a new spot, arms STAY LOCKED at
# the manipulation pose). DO NOT flip this default without
# coordinating with manipulation operators.
#
# LOCO_DECOUPLED_ARMS=0 -> opt-in for whole-body locomotion: in
# LOCOMOTION the manager sets passthrough_arm_targets=True in the
# arm_targets payload, the recorder nulls its cached arm pose, and
# the merge step falls through to planner-predicted arms (natural
# gait-coupled swing from the x2_ultra_locowalk training corpus).
# ARM_MAN and OFF are unaffected.
#
# Run two Quest 3 sessions with this env var = 1 then = 0 to A/B test
# whether the static-arms override hurts forward-walking quality.
LOCO_DECOUPLED_ARMS="${LOCO_DECOUPLED_ARMS:-1}"
if [[ "${LOCO_DECOUPLED_ARMS}" == "1" ]]; then
    MANAGER_ARGS+=(--loco-decoupled-arms)
elif [[ "${LOCO_DECOUPLED_ARMS}" == "0" ]]; then
    MANAGER_ARGS+=(--no-loco-decoupled-arms)
else
    log "WARN: LOCO_DECOUPLED_ARMS must be 0 or 1; got '${LOCO_DECOUPLED_ARMS}'. Falling back to default (1)."
    MANAGER_ARGS+=(--loco-decoupled-arms)
fi

# Quest3 raw capture sidecar -- replayable input-smoothing fixture.
# Set QUEST3_RECORD_TO=/path/to/quest3_raw.jsonl to dump every manager
# tick that consumed a Quest sample (post-invert axes + buttons +
# 3pt-pose + hand curls). Default empty -> no capture. Consumed by
# the Quest3Replayer (Part 2) so one live operator session becomes a
# reusable fixture for input-smoothing knob sweeps without re-donning
# the rig. See docs/source/references/x2_quest3_stick_smoothing.md.
QUEST3_RECORD_TO="${QUEST3_RECORD_TO:-}"
if [[ -n "${QUEST3_RECORD_TO}" ]]; then
    # mkdir -p the parent so the manager doesn't fail on first write
    # if the operator pointed at a fresh out/ subtree.
    mkdir -p "$(dirname "${QUEST3_RECORD_TO}")"
    MANAGER_ARGS+=(--quest3-record-to "${QUEST3_RECORD_TO}")
    log "  Quest3 raw capture ENABLED -> ${QUEST3_RECORD_TO}"
fi

# --- VR stick smoothing (StickFilter) ----------------------------------
# Bring the live VR p99 |d(vel_z)/dt| into the kplanner's training band
# (~3 m/s^2) so the operator's forward-walk inputs don't lurch the
# robot. Tuned via offline analyzer
# (scripts/analyze_planner_cmd_jsonl.py) against the captured live
# fixture; see docs/source/references/x2_quest3_stick_smoothing.md for
# the methodology and per-channel rationale.
#
# Env vars (all optional; unset preserves legacy unfiltered path):
#   QUEST3_STICK_LPF_TAU    -- first-order LPF time constant (s).
#                              0.10 is the tuned default; 0.0 disables.
#   QUEST3_STICK_SLEW_MAX   -- per-channel slew cap (stick-units/s).
#                              ``inf`` (default) disables slew clamp.
#   QUEST3_STICK_RETURN_TAU -- optional asymmetric release LPF tau (s).
#                              0.0 disables -> symmetric engage/release.
QUEST3_STICK_LPF_TAU="${QUEST3_STICK_LPF_TAU:-}"
QUEST3_STICK_SLEW_MAX="${QUEST3_STICK_SLEW_MAX:-}"
QUEST3_STICK_RETURN_TAU="${QUEST3_STICK_RETURN_TAU:-}"
if [[ -n "${QUEST3_STICK_LPF_TAU}" ]]; then
    MANAGER_ARGS+=(--stick-lpf-tau "${QUEST3_STICK_LPF_TAU}")
    log "  StickFilter LPF tau: ${QUEST3_STICK_LPF_TAU} s"
fi
if [[ -n "${QUEST3_STICK_SLEW_MAX}" ]]; then
    MANAGER_ARGS+=(--stick-slew-max "${QUEST3_STICK_SLEW_MAX}")
    log "  StickFilter slew max: ${QUEST3_STICK_SLEW_MAX} stick-units/s"
fi
if [[ -n "${QUEST3_STICK_RETURN_TAU}" ]]; then
    MANAGER_ARGS+=(--stick-return-tau "${QUEST3_STICK_RETURN_TAU}")
    log "  StickFilter release tau: ${QUEST3_STICK_RETURN_TAU} s"
fi

# --- VR wrist orientation offset (operator-side, "controller mount
# calibration" stop-gap until vr_operator_calibrate is re-run) -------
# Three floats per side: roll pitch yaw in DEGREES, intrinsic XYZ
# Tait-Bryan, applied in the operator's wrist-local frame BEFORE the
# calibration alignment. Defaults empty -> no flag passed -> manager's
# (0, 0, 0) default -> today's behaviour. Pass space-separated values
# via env var, e.g.:
#     QUEST3_LEFT_WRIST_OFFSET_RPY_DEG="0 0 -30"  ./run_*.sh
# rotates the LEFT operator-wrist quat by -30deg about the wrist normal
# axis (yaw) before the calibration applies, compensating a left
# controller mounted ~30deg outward on the operator's cuff.
QUEST3_LEFT_WRIST_OFFSET_RPY_DEG="${QUEST3_LEFT_WRIST_OFFSET_RPY_DEG:-}"
QUEST3_RIGHT_WRIST_OFFSET_RPY_DEG="${QUEST3_RIGHT_WRIST_OFFSET_RPY_DEG:-}"
if [[ -n "${QUEST3_LEFT_WRIST_OFFSET_RPY_DEG}" ]]; then
    # shellcheck disable=SC2206  # intentional word splitting for nargs=3
    _LEFT_RPY=( ${QUEST3_LEFT_WRIST_OFFSET_RPY_DEG} )
    if [[ "${#_LEFT_RPY[@]}" -ne 3 ]]; then
        err "QUEST3_LEFT_WRIST_OFFSET_RPY_DEG must be 3 floats (roll pitch yaw, deg); got: '${QUEST3_LEFT_WRIST_OFFSET_RPY_DEG}'"
        exit 2
    fi
    MANAGER_ARGS+=(--left-wrist-offset-rpy-deg "${_LEFT_RPY[@]}")
    log "  LEFT wrist op-quat offset: rpy_deg=(${_LEFT_RPY[0]}, ${_LEFT_RPY[1]}, ${_LEFT_RPY[2]})"
fi
if [[ -n "${QUEST3_RIGHT_WRIST_OFFSET_RPY_DEG}" ]]; then
    # shellcheck disable=SC2206
    _RIGHT_RPY=( ${QUEST3_RIGHT_WRIST_OFFSET_RPY_DEG} )
    if [[ "${#_RIGHT_RPY[@]}" -ne 3 ]]; then
        err "QUEST3_RIGHT_WRIST_OFFSET_RPY_DEG must be 3 floats (roll pitch yaw, deg); got: '${QUEST3_RIGHT_WRIST_OFFSET_RPY_DEG}'"
        exit 2
    fi
    MANAGER_ARGS+=(--right-wrist-offset-rpy-deg "${_RIGHT_RPY[@]}")
    log "  RIGHT wrist op-quat offset: rpy_deg=(${_RIGHT_RPY[0]}, ${_RIGHT_RPY[1]}, ${_RIGHT_RPY[2]})"
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
│    Browser URL:  https://${_LOCAL_IP_CANDIDATES}:${QUEST3_HTTP_PORT}${_BROWSER_URL_PAD}│
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
│  Continuous waist (v7.4; position-mapped, slewed 60deg/s,           │
│  yaw-priority cone on R-stick, roll-priority cone on L-stick):       │
│    R stick fwd / back  bidirectional lean (clamp +/-20deg)           │
│    R stick L/R soft    torso twist L/R       (clamp +/-40deg)        │
│    (yaw-priority cone: |rx| dominating |ry| past 0.4 suppresses     │
│     pitch so a near-pure twist doesn't accidentally lean the body.)  │
│  v7.4: R-stick lean+twist active in BOTH LOCOMOTION + ARM_MAN.       │
│  ARM_MANIPULATION L-stick (v7.4; ignored in LOCOMOTION):             │
│    L stick L/R         lateral lean / roll      (clamp +/-10deg)     │
│    L stick fwd         continuous SQUAT (default ~9cm down)          │
│    L stick back        continuous STAND-UP (default ~4cm up)         │
│    (roll-priority cone: |lx| dominating |ly| past 0.4 suppresses    │
│     hip-height so a pure roll doesn't accidentally squat. Hip       │
│     height is kplanner-only -- the heuristic planner ignores it.)   │
│  Note: the planner-log "command" appears flipped vs. the operator    │
│  intent because the curated bins were authored in a body frame       │
│  rotated 180 deg from the bridge's RSI init. End-to-end behaviour    │
│  is correct (push the stick the way you want the robot to move).    │
│  ARM_MANIPULATION (arms track VR + recorder buttons):                │
│    A press           engage / disengage arm IK                       │
│    X press           start episode (--with-record only)              │
│    Y press           stop & save episode (--with-record only)        │
│    R stick           SAME lean / twist as in LOCOMOTION (v7.2)       │
│    L stick           roll (lx) + squat / stand (ly) (v7.4)           │
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

if [[ -n "${REMOTE_DEPLOY_HOST}" ]]; then
    RECORDER_DEBUG_SUB_HOST="${REMOTE_DEPLOY_HOST}"
else
    RECORDER_DEBUG_SUB_HOST="localhost"
fi

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
    --sub-host "${RECORDER_DEBUG_SUB_HOST}"
    --sub-port "${DEBUG_PORT}"
    --sub-topic "${DEBUG_TOPIC}"
    --gesture-cmd-host '*'
    --gesture-cmd-port "${GESTURE_CMD_PORT}"
    --gesture-cmd-topic "${GESTURE_CMD_TOPIC}"
    --gesture-catalog "${GESTURE_CATALOG}"
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

# Head-camera ingestion: only meaningful in --with-record (no parquet
# means no place to write the frames). Auto-launch the PC2 bridge over
# SSH unless the operator passed --no-camera-autostart, then forward
# the three camera CLI flags to record_x2_dataset.
if [[ "${HEAD_CAMERAS}" -eq 1 && "${WITH_RECORD}" -eq 1 ]]; then
    if [[ -z "${CAMERA_HOST}" ]]; then
        err "--head-cameras requires --camera-host (or --pc2-host) so we know" \
            "which X2 PC2 the camera bridge should bind on."
        exit 1
    fi
    if [[ "${CAMERA_AUTOSTART}" -eq 1 ]]; then
        PC2_CAM_SH="${REPO_ROOT}/gear_sonic_deploy/scripts/x2_pc2_cameras.sh"
        if [[ -x "${PC2_CAM_SH}" ]]; then
            log "Camera bridge: launching x2_pc2_cameras.sh serve" \
                "--host ${CAMERA_HOST} --port ${CAMERA_PORT} …"
            X2_PC2_HOST="${CAMERA_HOST}" X2_PC2_CAM_PORT="${CAMERA_PORT}" \
                "${PC2_CAM_SH}" serve --host "${CAMERA_HOST}" \
                                       --port "${CAMERA_PORT}" \
                || { err "PC2 camera bridge failed to start. Run" \
                         "\`${PC2_CAM_SH} status --host ${CAMERA_HOST}\` to diagnose," \
                         "then rerun with --no-camera-autostart once it's up."
                     exit 1; }
        else
            err "${PC2_CAM_SH} not executable; cannot auto-start cameras."
            exit 1
        fi
    else
        log "Camera bridge: --no-camera-autostart set; assuming bridge already" \
            "running on ${CAMERA_HOST}:${CAMERA_PORT}"
    fi
    RECORDER_ARGS+=(
        --head-cameras
        --camera-host "${CAMERA_HOST}"
        --camera-port "${CAMERA_PORT}"
    )
elif [[ "${HEAD_CAMERAS}" -eq 1 && "${WITH_RECORD}" -eq 0 ]]; then
    warn "--head-cameras has no effect in teleop-only mode (no parquet to" \
         "write into); ignoring."
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
    # See the matching block in the non-VLA branch (~line 2049) for the
    # rationale. Re-checked here so VLA-mode runs can also produce the
    # raw-input fixture if the operator opts in via env.
    QUEST3_RECORD_TO="${QUEST3_RECORD_TO:-}"
    if [[ -n "${QUEST3_RECORD_TO}" ]]; then
        mkdir -p "$(dirname "${QUEST3_RECORD_TO}")"
        MANAGER_ARGS+=(--quest3-record-to "${QUEST3_RECORD_TO}")
        log "  Quest3 raw capture ENABLED -> ${QUEST3_RECORD_TO}"
    fi
    # Same StickFilter env vars as the non-VLA branch (see ~line 2065
    # for the description). VLA bridge runs benefit from the smoother
    # stick inputs equally; operator-feel is identical.
    QUEST3_STICK_LPF_TAU="${QUEST3_STICK_LPF_TAU:-}"
    QUEST3_STICK_SLEW_MAX="${QUEST3_STICK_SLEW_MAX:-}"
    QUEST3_STICK_RETURN_TAU="${QUEST3_STICK_RETURN_TAU:-}"
    if [[ -n "${QUEST3_STICK_LPF_TAU}" ]]; then
        MANAGER_ARGS+=(--stick-lpf-tau "${QUEST3_STICK_LPF_TAU}")
        log "  StickFilter LPF tau: ${QUEST3_STICK_LPF_TAU} s"
    fi
    if [[ -n "${QUEST3_STICK_SLEW_MAX}" ]]; then
        MANAGER_ARGS+=(--stick-slew-max "${QUEST3_STICK_SLEW_MAX}")
        log "  StickFilter slew max: ${QUEST3_STICK_SLEW_MAX} stick-units/s"
    fi
    if [[ -n "${QUEST3_STICK_RETURN_TAU}" ]]; then
        MANAGER_ARGS+=(--stick-return-tau "${QUEST3_STICK_RETURN_TAU}")
        log "  StickFilter release tau: ${QUEST3_STICK_RETURN_TAU} s"
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
        --gesture-cmd-host '*'
        --gesture-cmd-port "${GESTURE_CMD_PORT}"
        --gesture-cmd-topic "${GESTURE_CMD_TOPIC}"
        --gesture-catalog "${GESTURE_CATALOG}"
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
