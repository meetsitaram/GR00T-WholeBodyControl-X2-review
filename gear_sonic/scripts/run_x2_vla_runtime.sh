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
# ---- Manual-takeover (vla_control) plumbing (2026-06-10 milestone) ----
# When VLA_CONTROL_PORT > 0, the bridge subscribes to a remote
# vla_control PUB (the x2_pose_proxy on PC2) and cold-restarts on
# the override_released edge so the operator can teleop-nudge the
# arm out of a stuck pose without restarting any process. Default
# disabled so existing autonomous-only runs are byte-for-byte
# unchanged. Set both VLA_CONTROL_PORT and (typically) VLA_CONTROL_HOST
# to the proxy's bind address.
: "${VLA_CONTROL_HOST:=}"
: "${VLA_CONTROL_PORT:=-1}"
: "${VLA_CONTROL_TOPIC:=vla_control}"
: "${VLA_COLD_RESTART_HOLD_TICKS:=25}"
# Smooth-handoff guard: maximum ticks the bridge will keep the wire at
# the operator hand-off pose AFTER the minimum hold above, while
# waiting for the first eligible decoded chunk (chunk_id > cold-restart
# baseline). The minimum (above) is the proxy HOLD ladder bridge; this
# is the safety cap on the await-first-chunk wait that prevents an
# abrupt snap to the idle_stand clip when the inference cadence
# (~15 Hz) trails the wire (50 Hz). Default 200 @ 50 Hz = 4 s. Set to
# 0 to disable the guard (legacy 2026-06-10 behaviour). MUST be >=
# VLA_COLD_RESTART_HOLD_TICKS or live_vla_publish_motion_token.py
# refuses to start.
: "${VLA_HANDOFF_MAX_HOLD_TICKS:=200}"

# Post-handoff slow-step window for the wire's per-element rate clamp.
# Defaults reflect 2026-06-10 follow-up 6: ``handoff_max_wire_step``
# (0.012 rad/tick = ~36 deg/s/joint) applies for the first
# ``handoff_step_ramp_ticks`` (250 = 5 s @ 50 Hz) after the cold-restart
# hold releases, then linearly ramps back to ``vla_max_wire_step``
# (0.035 rad/tick = ~100 deg/s/joint). Motivation: VLA's first decoded
# chunk after a long teleop window is often ~3.7 rad (L_inf) from the
# operator's pose -- the existing 75-tick LPF + 0.035 rad/tick rate
# clamp lets the wire move all 31 joints at once at 1.75 rad/s
# coordinated, which the operator visually reports as a slam even
# though no single joint exceeds the limit (terminal 2 of session
# 73f3d2a2 at tick 16000: raw_Δ=3.720 rad, body_Δ=0.247 rad sustained
# over the ramp). Set HANDOFF_MAX_WIRE_STEP == VLA_MAX_WIRE_STEP to
# disable the slow window; set HANDOFF_STEP_RAMP_TICKS=0 to apply the
# slow step indefinitely (rarely useful -- operator usually wants the
# wire back to normal speed once VLA stabilises).
: "${VLA_HANDOFF_MAX_WIRE_STEP:=0.012}"
: "${VLA_HANDOFF_STEP_RAMP_TICKS:=250}"

# ---- Tracking feedback (2026-06-10 follow-up 11, closed-loop wire) -----
# Closed-loop per-joint tracking feedback on the wire step cap. When
# enabled, the bridge reads x2_debug's measured arm-joint positions
# and velocities each tick and per-joint-throttles the per-tick step
# so the wire never outpaces the actuator's actual response.
# Eliminates the open-loop sensitivity to inference jitter / battery
# sag / motor temp drift that drove the 2026-06-10 PM oscillation
# incident.
#
# Step 1 rollout: default DISABLED (export VLA_TRACKING_FEEDBACK=1 or
# pass --vla-tracking-feedback). v3 static defaults (LPF/blend/step-
# cap) remain in place; feedback is additive belt-and-suspenders so
# any regression on real robot can be isolated by flipping the flag.
# Step 2 (separate commit) flips the default to ON and relaxes the
# static defaults once feedback is validated.
: "${VLA_TRACKING_FEEDBACK:=0}"
: "${VLA_TRACKING_SOFT_RAD:=0.15}"
: "${VLA_TRACKING_HARD_RAD:=0.40}"
: "${VLA_TRACKING_VELOCITY_MARGIN:=1.5}"
: "${VLA_TRACKING_VELOCITY_FLOOR_RAD_TICK:=0.01}"
: "${VLA_TRACKING_STALE_MS:=100}"

# ---- Sim-mode pose proxy plumbing (2026-06-10 milestone, sim path) -----
# When SIM_MODE=1 AND (VLA_CONTROL_PORT > 0 OR POSE_PROXY_OVERRIDE_PORT > 0),
# this launcher spawns a LOCAL x2_pose_proxy on loopback between the
# bridge and the sim deploy so the manual-takeover loop works in pure
# sim without any PC2 daemons. The bridge keeps publishing to
# LAPTOP_POSE_PORT (5556); the proxy SUBs there, arbitrates against
# the override SUB, and PUBs the merged wire to POSE_PROXY_DOWNSTREAM_PORT
# (5558) where the sim deploy reads from.
#
# Defaults match x2_pc2_daemons.sh so the same operator runbook works
# in sim and real-robot mode. Override via env vars.
: "${POSE_PROXY_DOWNSTREAM_HOST:=127.0.0.1}"
: "${POSE_PROXY_DOWNSTREAM_PORT:=5558}"
: "${POSE_PROXY_OVERRIDE_HOST:=127.0.0.1}"
: "${POSE_PROXY_OVERRIDE_PORT:=-1}"   # default disabled
: "${POSE_PROXY_OVERRIDE_TOPIC:=pose}"
: "${POSE_PROXY_OVERRIDE_STALE_MS:=200}"
# Frozen-frame release: the Quest3 manager publishes the FROZEN last
# commanded pose every tick in OFF/LOCOMOTION mode, so the override
# SUB never goes silent across an A+B+X+Y disengage gesture. Frame-
# equality detection in the proxy catches this and fires
# override_released exactly once after N consecutive identical
# frames. Default 10 ticks @ 50Hz = 200ms (matches stale-ms). Set
# POSE_PROXY_OVERRIDE_FROZEN_TICKS=0 to disable and fall back to
# silence-only release (legacy behaviour, only fires when the
# operator Ctrl-C's the entire teleop stack).
: "${POSE_PROXY_OVERRIDE_FROZEN_TICKS:=10}"
# Bumped from 1e-4 on 2026-06-10 after observing repeated single-tick
# engage/release cycles in sim from sub-degree controller-rest drift
# while the manager was in OFF (each cycle fires a heavy VLA cold-
# restart). 5e-3 rad ~ 0.3 deg total joint-space motion is well above
# resting jitter and well below intentional teleop motion. Lower to
# 1e-4 only for strict bytes-match detection.
: "${POSE_PROXY_OVERRIDE_FROZEN_L2_TOL:=5e-3}"
# Symmetric engage-side hysteresis: require N consecutive override
# frames with joint-space delta ABOVE --override-frozen-l2-tol before
# firing override_engaged. Same default as frozen-ticks (10 = 200ms
# @ 50Hz). Together with the higher tolerance above this prevents
# brief controller jitter from spurious engage / release / cold-restart
# cycles (each cycle is heavy: bridge wipes chunk + LPF + ramp state).
# Set to 0 for legacy single-frame-engage behaviour (used only by
# older smoke tests, not the operator runbook).
: "${POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS:=10}"
# Engagement slow-step ramp (2026-06-10 follow-up 9). Symmetric to
# the bridge's --vla-handoff-max-wire-step / --vla-handoff-step-ramp-
# ticks: when the proxy fires LIVE -> OVERRIDE the operator's first
# OVERRIDE frame can be ~3 rad (L_inf) away from VLA's last
# commanded body pose, and forwarding it verbatim made the deploy
# slam the body across the delta in one tick. The proxy now clamps
# the operator's joint_pos_mj per-element from the last forwarded
# (VLA) pose, linearly relaxing the clamp back to the steady-state
# step over the ramp window. Defaults match the bridge's handoff
# defaults so the two takeover directions feel symmetric.
: "${POSE_PROXY_ENGAGEMENT_MAX_WIRE_STEP:=0.012}"
: "${POSE_PROXY_ENGAGEMENT_STEADY_WIRE_STEP:=0.035}"
: "${POSE_PROXY_ENGAGEMENT_STEP_RAMP_TICKS:=250}"
# Operator-mode SUB (2026-06-10 follow-up). The Quest3 manager
# publishes its current ``stream_mode`` ("OFF" | "LOCOMOTION" |
# "ARM_MANIPULATION") on the recorder PUB every tick. When this
# port is set, the proxy uses mode != "OFF" as the STRICT engage
# gate -- motion-hysteresis / frozen-detection are bypassed and the
# operator holding the controller still no longer flicker-releases
# the wire (the bug the user hit on 2026-06-10). 5564 matches the
# manager's --recorder-pub-port default; override here when the
# manager binds elsewhere.
: "${POSE_PROXY_TELEOP_MODE_HOST:=127.0.0.1}"
: "${POSE_PROXY_TELEOP_MODE_PORT:=5564}"
: "${POSE_PROXY_TELEOP_MODE_TOPIC:=stream_mode}"
: "${POSE_PROXY_TELEOP_MODE_STALE_MS:=1000}"
: "${POSE_PROXY_IDLE_STALE_MS:=300}"
: "${POSE_PROXY_IDLE_MODE:=blend}"
: "${POSE_PROXY_HOLD_LAST_SECS:=10.0}"
: "${POSE_PROXY_BLEND_SECS:=3.0}"
# Baked idle clip the proxy falls back to. Repo ships
# gear_sonic_deploy/data/idle_stand.x2m2 (regenerate via
# `python -m gear_sonic_deploy.scripts.bake_idle_stand_x2m2`).
: "${POSE_PROXY_IDLE_X2M2:=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/gear_sonic_deploy/data/idle_stand.x2m2}"
# Python interpreter for the proxy. pyzmq + numpy are the only deps;
# the bridge venv has both. We default to the bridge's interpreter
# so we don't introduce a second venv requirement.
: "${POSE_PROXY_PY:=}"
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
# When set, --vla-raw zeros every wire-shaping knob added on top of
# the policy output -- LPFs, per-tick clamps, chunk blends, and the
# dev-from-body cap. The startup ramp (VLA_RAMP_IN_TICKS) and deploy
# stabilization delay (VLA_DECODE_DELAY_TICKS) are intentionally LEFT
# ON because they only affect the first ~0.5-3 s of the run and exist
# to prevent a step-input shove onto the SONIC tracker at deploy
# hand-off. CRITICAL: VLA_MAX_ACTION_IL stays at its 8.0 default --
# it is NOT a wire shaper, it clips the policy's own ``last_action``
# proprio echo to match the training ``action_clip_value=20``
# headroom. Disabling it creates a positive-feedback loop (proprio's
# last_action grows unboundedly -> policy predicts ever-larger
# action_il -> repeat) that diverges within a few chunks and ends
# with the wire commanding joints at 90+ rad. PC2's
# --max-target-dev guard on the SONIC tracker remains active
# regardless and is the last line of defense for runaway poses.
: "${VLA_RAW:=0}"
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
# Wrist bypass forwarded to the C++ deploy binary via --wrist-bypass {off,ik}.
# Default ``ik`` so the operator's wrist gestures AND VLA wrist tokens
# actually move the robot's wrist (per wrist_bypass.hpp, the SONIC
# tracker pins ``*_wrist_pitch``/``*_wrist_roll`` at a comfort pose
# regardless of what the IK reference says -- the operator on
# 2026-06-10 13:12 confirmed "wrist not responding. we need the wrist
# ik enabled i think"). Wrist slam mitigation lives ENTIRELY on the
# bridge side (see follow-up 8 below + follow-up 6's slow-step ramp):
# the wire's per-element rate clamp (``--vla-max-wire-step``, default
# 0.035 rad/tick) implicitly rate-limits the bypassed wrist target
# because ``wrist_bypass=ik`` force-writes ``target_pos_mj`` from the
# wire's ``joint_pos_mj``. Right after a handoff, the slow-step ramp
# (``--vla-handoff-max-wire-step`` default 0.012 rad/tick for the
# first 5 s) bounds the transition further. To run without the
# bypass (SONIC pins wrist pitch/roll at comfort pose; gestures
# unresponsive), pass ``--wrist-bypass off``.
: "${WRIST_BYPASS:=ik}"
# DO NOT auto-pair with ``--max-target-dev``. The 13:21 run on
# 2026-06-10 proved why: ``--max-target-dev`` is a GLOBAL absolute
# clamp ``|target - default_angles| <= N`` applied to ALL joint
# groups (leg + waist + arm + head per the deploy startup log),
# NOT a per-tick rate limit. Setting 0.05 pinned every joint to
# +/-2.9 deg of default_angles -- the robot couldn't bend its
# knees enough to stand and collapsed forward onto the table
# (act_clip_ticks=916/1000 = 92% of policy outputs clamped while
# the safety stack fought the policy on every tick).
#
# Wrist slam mitigation now lives entirely on the bridge side:
# the wire's per-element rate clamp (``--vla-max-wire-step``,
# default 0.035 rad/tick) limits how fast the wire's wrist values
# can change, and follow-up 6's slow-step ramp
# (``--vla-handoff-max-wire-step``, default 0.012 rad/tick for the
# first 5 s after a handoff) bounds the post-takeover transition.
# Because ``wrist_bypass=ik`` force-writes ``target_pos_mj`` from
# the wire's ``joint_pos_mj``, those bridge-side clamps implicitly
# rate-limit the bypassed wrist target without needing any
# deploy-side clamp.
#
# Operators who still want a deploy-side wrist-specific clamp
# should pass per-group overrides (``--max-target-dev-arm``)
# via ``--deploy-extra-arg`` -- but note that "arm" covers MJ
# joints 15..28 = shoulder/elbow/wrist_yaw too, so 0.05 rad
# would also break shoulder/elbow tracking. There's no
# ``--max-target-dev-wrist`` group in the C++ deploy today;
# adding one is a separate fix tracked in the next-session TODO.
# Robocasa scene override for sim mode. Empty -> deploy default (x2_ultra.xml,
# empty world). Set to a short name like "X2PickPlaceApple" and the launcher
# resolves it to gear_sonic/data/assets/robocasa_scenes/<name>.xml and passes
# it to the sim deploy via --sim-mjcf. Ignored on real-robot runs.
: "${ROBOCASA_ENV:=}"
ROBOCASA_SCENE_DIR="${REPO_ROOT}/gear_sonic/data/assets/robocasa_scenes"
ROBOCASA_SCENE_XML=""
: "${RENDER_WIDTH:=640}"
: "${RENDER_HEIGHT:=480}"
: "${VIDEO_OUT:=}"
: "${VIDEO_FRONT_OUT:=}"

# ── --with-record knobs (mirror the teleop launcher's recording surface).
# When WITH_RECORD=1 the launcher spawns gear_sonic/scripts/record_x2_dataset
# in "vla" subscribe mode (one-run = one-episode auto-save) AFTER the bridge
# logs ``policy ready``. The recorder lives on the laptop alongside the
# bridge; it subscribes to the bridge's :5556 pose PUB for body + hands +
# token, the deploy's :5557 x2_debug for proprio, and (in real-robot mode)
# the PC2 :5555 camera bridge for head_front + stereo_left + stereo_right
# RGB tracks. ``observation.images.ego_view`` is the MuJoCo render the
# recorder builds inline; head cams default ON in real-robot mode (we have
# them) and OFF in sim (no PC2 to fetch from -- pass --head-cameras + a
# camera host to record sim stereo from a separate publisher if you ever
# need that).
: "${WITH_RECORD:=0}"
: "${OUTPUT_DIR:=}"
: "${TASK:=}"
: "${HEAD_CAMERAS:=}"   # default resolved after SIM_MODE is known
: "${CAMERA_HOST:=}"    # default to PC2_HOST in real mode
: "${CAMERA_PORT:=${PC2_CAMERAS_PORT:-5555}}"
: "${ENCODER_CONFIG:=gear_sonic/data/encoder/x2_observation_config.yaml}"
: "${SONIC_TOKENIZER_DEVICE:=cpu}"
: "${RECORDER_PY:=${REPO_ROOT}/.venv/bin/python}"

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
  --robocasa-env NAME          Load a Robocasa scene in sim instead of the empty
                               default. NAME is a short id (e.g. X2PickPlaceApple)
                               resolved against gear_sonic/data/assets/robocasa_scenes/.
                               Sim-only; rejected when --pc2-host is set.
  --sim-viewer                 Enable MuJoCo passive viewer in sim (default).
  --no-sim-viewer              Disable MuJoCo passive viewer (sim).
  --sim-with-omnihand          Compose X2 + OmniHand in sim (default).
  --no-sim-with-omnihand       Use bare X2 (no OmniHand) in sim.
  --wrist-bypass MODE          {off, ik} -- forwarded to the C++ deploy as
                               --wrist-bypass. Default 'ik'. With 'ik',
                               the deploy force-writes target_pos_mj for
                               MJ indices {20,21,27,28} (left + right
                               wrist pitch + roll) to the wire's
                               joint_pos_mj BEFORE the safety stack, so
                               operator wrist gestures and VLA wrist
                               tokens actually move the wrist. SONIC
                               otherwise pins those 4 DOFs at a comfort
                               pose regardless of the IK reference (see
                               wrist_bypass.hpp). The launcher does NOT
                               auto-pair this with --max-target-dev:
                               that flag is a GLOBAL deviation clamp on
                               ALL joint groups, not a wrist-specific
                               rate clamp, and pinning legs+waist+arm to
                               +/-2.9 deg of default makes the robot
                               collapse (2026-06-10 follow-up 8). Wrist
                               slam mitigation lives on the bridge side
                               via --vla-max-wire-step (steady-state
                               rate clamp) and --vla-handoff-max-wire-
                               step (slow-step ramp right after handoff).
                               Set 'off' for SONIC-pinned wrist (wrist
                               will NOT respond to operator/VLA wrist
                               commands).
  --vla-ramp-in-ticks N        Bridge ramp-in ticks (default: 75)
  --vla-target-lpf-hz HZ       Body wire LPF (default: 2.0)
  --vla-future-lpf-hz HZ       Future-window LPF (default: 2.0)
  --vla-hand-lpf-hz HZ         Hand wire LPF (default: 1.0)
  --vla-hand-chunk-blend-ticks N Hand inter-chunk blend (default: 30)
  --vla-max-hand-step R          Max per-tick hand step (default: 0.08)
  --vla-max-wire-dev-from-body R Max wire pose deviation from body_q (default: 0.18)
  --vla-max-wire-step R          Max per-tick wire joint step (default: 0.035)
  --vla-decode-delay-ticks N     Idle ticks before VLA decode (default: 150)
  --vla-raw                      Disable EVERY bridge-side wire-shaping
                                 knob: LPFs, per-tick clamps, chunk
                                 blends, dev-from-body cap, and action-il
                                 clamp are all zeroed. The startup ramp +
                                 decode delay stay on (they fire once at
                                 hand-off, not in steady state). PC2's
                                 --max-target-dev guard on the SONIC
                                 tracker remains active. Useful for
                                 visualising raw policy chunks; expect
                                 jerky motion and use the E-stop.
  --body-mode MODE               manipulation | locomotion (VR analog)
  --mode-control-file PATH       Runtime mode switch file (one word per line)
  --freeze-body-groups LIST      Override manipulation freeze groups (default legs,waist)
  --render-width / --render-height   Ghost camera resolution (sim)
  --video-out PATH             Record ego_view MP4 (sim, ghost mode)
  --video-front-out PATH       Record third-person MP4 (sim)
  --with-record                Spawn record_x2_dataset alongside the bridge
                               (one-run = one-episode auto-save). Requires
                               --output-dir and --task.
  --output-dir PATH            LeRobot v2.1 dataset directory (required with
                               --with-record). Convention:
                               data/lerobot/vla_run_<task>_<checkpoint>/
  --task TEXT                  Language instruction for every episode in this
                               dataset (required with --with-record).
  --head-cameras / --no-head-cameras
                               Force-enable / force-disable head-camera ingest
                               by the recorder. Default: ON in real-robot mode
                               (PC2 :5555 head_front + stereo_left + stereo_right),
                               OFF in sim mode (no PC2 to fetch from).
  --camera-host HOST           PC2 host for the camera SUB. Defaults to
                               --pc2-host in real mode.
  --camera-port PORT           PC2 camera ZMQ port (default 5555).
  --encoder-config PATH        Encoder observation YAML for the inline SONIC
                               tokenizer (default gear_sonic/data/encoder/
                               x2_observation_config.yaml). Pass '' to fall
                               back to the deprecated freeze-pose path.
  --sonic-tokenizer-device DEV cuda:0|cuda:N|cpu for the inline tokenizer
                               (default cpu so the GR00T policy keeps cuda:0).
  --recorder-py PATH           Python interpreter for record_x2_dataset
                               (default .venv/bin/python so the LeRobot deps
                               install path is hit, not env_isaaclab).

  --- Manual-takeover plumbing (2026-06-10 milestone) ---
  Enables the operator to teleop-nudge the arm out of a stuck VLA pose
  without restarting the bridge. CLI flags override matching env vars.
  In sim mode, setting either of the two PORTs > 0 also spawns a local
  x2_pose_proxy on loopback so the same loop works without PC2.

  --vla-control-port PORT      Bridge SUB port for vla_control events
                               (proxy emits override_engaged/released
                               edges here). Default -1 = disabled.
  --vla-control-host HOST      Host for the vla_control SUB. Defaults to
                               --pc2-host in real mode, 127.0.0.1 in sim.
  --vla-control-topic TOPIC    Topic prefix on the vla_control SUB
                               (default 'vla_control').
  --vla-cold-restart-hold-ticks N
                               Bridge hold-at-measured-pose duration on
                               override_released (default 25 = 500 ms).
                               This is the MINIMUM dwell.
  --vla-handoff-max-hold-ticks N
                               Safety cap on the smooth-handoff guard
                               (default 200 = 4 s). The bridge keeps
                               the wire at the operator hand-off pose
                               between --vla-cold-restart-hold-ticks
                               and this cap, releasing as soon as the
                               first eligible decoded chunk arrives.
                               Without this guard, the post-hold tick
                               can land before the next chunk decodes
                               (inference cadence ~15 Hz vs wire
                               50 Hz) and the wire snaps from the
                               operator pose to the idle_stand clip.
                               MUST be >= --vla-cold-restart-hold-ticks.
                               Set 0 to disable.
  --vla-handoff-max-wire-step RAD
                               Per-element max joint-position step on
                               the wire during the post-handoff slow
                               window (default 0.012 rad/tick = ~36
                               deg/s/joint at 50 Hz, vs the steady-
                               state --vla-max-wire-step default of
                               0.035 rad/tick = ~100 deg/s/joint).
                               Linearly ramps back to --vla-max-wire-
                               step over --vla-handoff-step-ramp-ticks
                               ticks after the cold-restart hold
                               releases. Set equal to --vla-max-wire-
                               step to disable the slow window.
                               2026-06-10 follow-up 6.
  --vla-handoff-step-ramp-ticks N
                               Ticks over which to ramp the wire step
                               from --vla-handoff-max-wire-step (slow,
                               applied right after handoff) back to
                               --vla-max-wire-step (normal). Default
                               250 @ 50 Hz = 5 s. Set 0 to skip the
                               slow window entirely.
  --vla-tracking-feedback      Enable closed-loop per-joint tracking
                               feedback on the wire step cap (2026-06-10
                               follow-up 11). When ON, the bridge reads
                               x2_debug's measured arm positions +
                               velocities and per-joint-throttles the
                               per-tick wire step so it never outpaces
                               the actuator's actual response. Default
                               OFF (Step 1 belt-and-suspenders rollout:
                               v3 LPF/blend/step-cap remain in place;
                               feedback is additive). Opt in via this
                               flag or VLA_TRACKING_FEEDBACK=1 env.
  --no-vla-tracking-feedback   Force-disable tracking feedback (mid-
                               session A/B against the scalar clamp).
  --vla-tracking-soft-rad RAD  Position error below which the per-
                               joint step cap stays at its base value
                               (default 0.15 rad ~= 8.6 deg).
  --vla-tracking-hard-rad RAD  Position error above which the per-
                               joint step cap drops to 0 (joint frozen
                               until actuator catches up). Default 0.40
                               rad ~= 23 deg. Linear ramp between soft
                               and hard.
  --vla-tracking-velocity-margin SCALE
                               Per-tick cap is also bounded by SCALE *
                               |measured_dq| * dt (default 1.5 = wire
                               can move at most 50% faster than the
                               actuator is currently moving). Lower =
                               more conservative.
  --vla-tracking-velocity-floor-rad-tick RAD
                               Minimum velocity-cap floor (default
                               0.01 rad/tick = 0.5 rad/s @ 50 Hz). Lets
                               the wire start from rest even when
                               measured_dq is zero. Set 0 to require
                               non-zero measured velocity (motion-only).
  --vla-tracking-stale-ms MS   Tracking-feedback proprio staleness
                               threshold. If the x2_debug snapshot is
                               older than this, the bridge falls back
                               to the scalar clamp for that tick.
                               Default 100 ms = 5 publish ticks at
                               50 Hz; covers a single packet drop.
  --pose-proxy-override-port PORT
                               Sim-only: proxy override SUB port (where
                               the teleop recorder PUBs). Default -1 =
                               disabled. Real-robot mode uses the env var
                               consumed by x2_pc2_daemons.sh instead.
  --pose-proxy-override-host HOST
                               Sim-only: override SUB host (default
                               127.0.0.1).
  --pose-proxy-override-topic TOPIC
                               Sim-only: override SUB topic prefix
                               (default 'pose').
  --pose-proxy-override-stale-ms MS
                               Sim-only: silence debounce before the
                               proxy fires override_released (default
                               200 ms). Only fires if the operator
                               kills the teleop stack, since the
                               Quest3 manager keeps publishing frozen
                               frames in OFF/LOCOMOTION. See
                               --pose-proxy-override-frozen-ticks for
                               the gesture-friendly release path.
  --pose-proxy-override-frozen-ticks N
                               Sim-only: fire override_released after N
                               consecutive identical override frames
                               (within --frozen-l2-tol). Default 10 =
                               200ms @ 50Hz. Catches the manager's
                               freeze-on-disengage pattern so A+B+X+Y
                               actually releases. Set to 0 to disable
                               (fall back to silence-only).
  --pose-proxy-override-frozen-l2-tol R
                               Sim-only: L2 distance tolerance (rad)
                               for two override frames to count as
                               frozen (default 5e-3 ~ 0.3 deg total
                               joint motion; bumped from 1e-4 on
                               2026-06-10 to absorb controller-rest
                               jitter that was causing repeated
                               single-frame engage/release cycles).
                               Lower to 1e-4 for strict bytes-match
                               detection only.
  --pose-proxy-override-engage-motion-ticks N
                               Sim-only: symmetric engage-side
                               hysteresis. Require N consecutive
                               override frames with joint-space delta
                               ABOVE --override-frozen-l2-tol before
                               firing override_engaged (default 10 =
                               200ms @ 50Hz, mirrors frozen-ticks).
                               Prevents brief jitter from spurious
                               engage/release cold-restart cycles.
                               Set to 0 for the legacy single-frame
                               engage used by older smoke tests. IGNORED
                               when --pose-proxy-teleop-mode-port > 0
                               (mode-gated engagement bypasses motion
                               hysteresis entirely).
  --pose-proxy-engagement-max-wire-step R
                               Sim-only: per-element max joint step
                               (rad) applied at the LIVE -> OVERRIDE
                               edge so the operator's first override
                               frame doesn't slam the body across the
                               full VLA -> operator delta in one tick.
                               Default 0.012 rad/tick (~36 deg/s @
                               50 Hz; matches the bridge's --vla-
                               handoff-max-wire-step). The proxy
                               clamps each forwarded override frame
                               relative to the previously forwarded
                               pose, linearly relaxing back to --pose-
                               proxy-engagement-steady-wire-step over
                               --pose-proxy-engagement-step-ramp-ticks
                               ticks. Set to 0 (or equal to the steady
                               value) to disable the engagement ramp.
  --pose-proxy-engagement-steady-wire-step R
                               Sim-only: per-element steady-state
                               max joint step (rad) the engagement
                               ramp converges to. Default 0.035
                               rad/tick (matches bridge --vla-max-
                               wire-step). After the ramp completes
                               the proxy stops clamping override
                               frames entirely.
  --pose-proxy-engagement-step-ramp-ticks N
                               Sim-only: ticks to linearly ramp the
                               engagement clamp from slow -> steady
                               (default 250 @ 50Hz = 5.0 s, matches
                               bridge --vla-handoff-step-ramp-ticks).
                               Set to 0 to disable engagement
                               clamping (forwards verbatim from the
                               very first OVERRIDE tick -- pre-2026-
                               06-10 behaviour, produces the slam
                               this guard prevents).
  --pose-proxy-teleop-mode-host HOST
                               Sim-only: host of the Quest3 manager's
                               stream_mode PUB (default 127.0.0.1).
                               When manager runs on a different machine
                               point this at it.
  --pose-proxy-teleop-mode-port PORT
                               Sim-only: port of the Quest3 manager's
                               stream_mode PUB (default 5564, matches
                               the manager's --recorder-pub-port).
                               When > 0, the proxy gates engagement
                               STRICTLY on the manager's broadcast
                               mode (mode != "OFF") and BYPASSES the
                               motion-hysteresis / frozen-detection
                               heuristics. Set to -1 to fall back to
                               the legacy heuristic path (will flicker
                               if operator holds controller still in
                               ARM_MANIPULATION; pre-2026-06-10
                               behaviour).
  --pose-proxy-teleop-mode-topic TOPIC
                               Sim-only: ZMQ topic prefix for the
                               manager's mode PUB (default
                               'stream_mode').
  --pose-proxy-teleop-mode-stale-ms MS
                               Sim-only: treat mode signal as gone
                               after this many ms of silence (default
                               1000). When stale, strict mode BLOCKS
                               engagement -- a dead manager fails
                               closed within ~1 s.
  --pose-proxy-downstream-port PORT
                               Sim-only: port the proxy PUBs to and the
                               sim deploy SUBs from (default 5558).
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
        --robocasa-env)     ROBOCASA_ENV="$2"; shift 2 ;;
        --sim-viewer)       SIM_VIEWER=1; shift ;;
        --no-sim-viewer)    SIM_VIEWER=0; shift ;;
        --sim-with-omnihand)    SIM_WITH_OMNIHAND=1; shift ;;
        --no-sim-with-omnihand) SIM_WITH_OMNIHAND=0; shift ;;
        --wrist-bypass)
            case "$2" in
                off|ik) ;;
                *)
                    log "[FATAL] --wrist-bypass must be {off, ik}; got '$2'"
                    exit 2
                    ;;
            esac
            WRIST_BYPASS="$2"
            shift 2
            ;;
        --autostart-after)  SIM_AUTOSTART_AFTER="$2"; shift 2 ;;
        --max-target-dev)   SIM_MAX_TARGET_DEV="$2"; shift 2 ;;
        --deploy-target-lpf-hz) SIM_DEPLOY_TARGET_LPF_HZ="$2"; shift 2 ;;
        --vla-ramp-in-ticks) VLA_RAMP_IN_TICKS="$2"; shift 2 ;;
        --vla-target-lpf-hz) VLA_TARGET_LPF_HZ="$2"; VLA_TARGET_LPF_HZ_SET=1; shift 2 ;;
        --vla-future-lpf-hz) VLA_FUTURE_LPF_HZ="$2"; VLA_FUTURE_LPF_HZ_SET=1; shift 2 ;;
        --vla-hand-lpf-hz) VLA_HAND_LPF_HZ="$2"; VLA_HAND_LPF_HZ_SET=1; shift 2 ;;
        --vla-hand-chunk-blend-ticks) VLA_HAND_CHUNK_BLEND_TICKS="$2"; VLA_HAND_CHUNK_BLEND_TICKS_SET=1; shift 2 ;;
        --vla-max-hand-step) VLA_MAX_HAND_STEP="$2"; VLA_MAX_HAND_STEP_SET=1; shift 2 ;;
        --vla-max-wire-dev-from-body) VLA_MAX_WIRE_DEV_FROM_BODY="$2"; VLA_MAX_WIRE_DEV_FROM_BODY_SET=1; shift 2 ;;
        --vla-max-wire-step) VLA_MAX_WIRE_STEP="$2"; VLA_MAX_WIRE_STEP_SET=1; shift 2 ;;
        --vla-chunk-blend-ticks) VLA_CHUNK_BLEND_TICKS="$2"; VLA_CHUNK_BLEND_TICKS_SET=1; shift 2 ;;
        --vla-max-action-il) VLA_MAX_ACTION_IL="$2"; VLA_MAX_ACTION_IL_SET=1; shift 2 ;;
        --vla-decode-delay-ticks) VLA_DECODE_DELAY_TICKS="$2"; shift 2 ;;
        --vla-raw) VLA_RAW=1; shift ;;
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
        --with-record)      WITH_RECORD=1; shift ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --task)             TASK="$2"; shift 2 ;;
        --head-cameras)     HEAD_CAMERAS=1; shift ;;
        --no-head-cameras)  HEAD_CAMERAS=0; shift ;;
        --camera-host)      CAMERA_HOST="$2"; shift 2 ;;
        --camera-port)      CAMERA_PORT="$2"; shift 2 ;;
        --encoder-config)   ENCODER_CONFIG="$2"; shift 2 ;;
        --sonic-tokenizer-device) SONIC_TOKENIZER_DEVICE="$2"; shift 2 ;;
        --recorder-py)      RECORDER_PY="$2"; shift 2 ;;
        # --- manual-takeover plumbing (2026-06-10 milestone) -----------
        # These CLI flags take precedence over the matching env vars,
        # which remain as fallbacks so the existing x2_pc2_daemons.sh
        # tmux-env propagation path keeps working.
        --vla-control-port) VLA_CONTROL_PORT="$2"; shift 2 ;;
        --vla-control-host) VLA_CONTROL_HOST="$2"; shift 2 ;;
        --vla-control-topic) VLA_CONTROL_TOPIC="$2"; shift 2 ;;
        --vla-cold-restart-hold-ticks) VLA_COLD_RESTART_HOLD_TICKS="$2"; shift 2 ;;
        --vla-handoff-max-hold-ticks) VLA_HANDOFF_MAX_HOLD_TICKS="$2"; shift 2 ;;
        --vla-handoff-max-wire-step) VLA_HANDOFF_MAX_WIRE_STEP="$2"; shift 2 ;;
        --vla-handoff-step-ramp-ticks) VLA_HANDOFF_STEP_RAMP_TICKS="$2"; shift 2 ;;
        # --- tracking feedback (2026-06-10 follow-up 11) ---------------
        # Closed-loop wire step cap; default OFF for Step 1 belt-and-
        # suspenders rollout. See the VLA_TRACKING_* env block above.
        --vla-tracking-feedback) VLA_TRACKING_FEEDBACK=1; shift ;;
        --no-vla-tracking-feedback) VLA_TRACKING_FEEDBACK=0; shift ;;
        --vla-tracking-soft-rad) VLA_TRACKING_SOFT_RAD="$2"; shift 2 ;;
        --vla-tracking-hard-rad) VLA_TRACKING_HARD_RAD="$2"; shift 2 ;;
        --vla-tracking-velocity-margin) VLA_TRACKING_VELOCITY_MARGIN="$2"; shift 2 ;;
        --vla-tracking-velocity-floor-rad-tick) VLA_TRACKING_VELOCITY_FLOOR_RAD_TICK="$2"; shift 2 ;;
        --vla-tracking-stale-ms) VLA_TRACKING_STALE_MS="$2"; shift 2 ;;
        --pose-proxy-override-port) POSE_PROXY_OVERRIDE_PORT="$2"; shift 2 ;;
        --pose-proxy-override-host) POSE_PROXY_OVERRIDE_HOST="$2"; shift 2 ;;
        --pose-proxy-override-topic) POSE_PROXY_OVERRIDE_TOPIC="$2"; shift 2 ;;
        --pose-proxy-override-stale-ms) POSE_PROXY_OVERRIDE_STALE_MS="$2"; shift 2 ;;
        --pose-proxy-override-frozen-ticks) POSE_PROXY_OVERRIDE_FROZEN_TICKS="$2"; shift 2 ;;
        --pose-proxy-override-frozen-l2-tol) POSE_PROXY_OVERRIDE_FROZEN_L2_TOL="$2"; shift 2 ;;
        --pose-proxy-override-engage-motion-ticks) POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS="$2"; shift 2 ;;
        --pose-proxy-engagement-max-wire-step) POSE_PROXY_ENGAGEMENT_MAX_WIRE_STEP="$2"; shift 2 ;;
        --pose-proxy-engagement-steady-wire-step) POSE_PROXY_ENGAGEMENT_STEADY_WIRE_STEP="$2"; shift 2 ;;
        --pose-proxy-engagement-step-ramp-ticks) POSE_PROXY_ENGAGEMENT_STEP_RAMP_TICKS="$2"; shift 2 ;;
        --pose-proxy-teleop-mode-host) POSE_PROXY_TELEOP_MODE_HOST="$2"; shift 2 ;;
        --pose-proxy-teleop-mode-port) POSE_PROXY_TELEOP_MODE_PORT="$2"; shift 2 ;;
        --pose-proxy-teleop-mode-topic) POSE_PROXY_TELEOP_MODE_TOPIC="$2"; shift 2 ;;
        --pose-proxy-teleop-mode-stale-ms) POSE_PROXY_TELEOP_MODE_STALE_MS="$2"; shift 2 ;;
        --pose-proxy-downstream-port) POSE_PROXY_DOWNSTREAM_PORT="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  ARGS+=("$1"); shift ;;
    esac
done

# ----- defaults derived after parsing -----------------------------------
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="/tmp/x2_vla_runtime-$(date +%Y%m%d_%H%M%S)"
fi

# --vla-raw overrides every wire-shaping knob to its "disabled" value
# (the bridge accepts 0 as "off" for each of these per its argparse
# help text). The startup ramp + decode delay stay on. Operator-set
# overrides for the same knobs win silently in the loose sense but
# emit a warn so the operator knows their intent might be in tension
# with --vla-raw. We snapshot ``*_SET`` markers in the arg-parser
# above whenever an operator explicitly types one of these flags so
# the warning fires only on real conflicts, not on env-default
# values that happen to differ from the raw=0 target.
if [[ "$VLA_RAW" -eq 1 ]]; then
    declare -A _VLA_RAW_CONFLICTS=()
    # NOTE: VLA_MAX_ACTION_IL is NOT in this list and NOT zeroed below.
    # It clips the policy's own ``last_action`` proprio echo (not the
    # wire) and disabling it triggers a proprio-feedback runaway that
    # blows the wire up to 90+ rad within a few chunks (observed in
    # /tmp/x2_vla_runtime-20260609_090317). Operators who really want
    # to disable it must pass --vla-max-action-il 0 explicitly.
    for knob in \
        VLA_TARGET_LPF_HZ VLA_FUTURE_LPF_HZ VLA_HAND_LPF_HZ \
        VLA_HAND_CHUNK_BLEND_TICKS VLA_MAX_HAND_STEP \
        VLA_MAX_WIRE_DEV_FROM_BODY VLA_MAX_WIRE_STEP \
        VLA_CHUNK_BLEND_TICKS; do
        setvar="${knob}_SET"
        if [[ "${!setvar:-0}" -eq 1 ]]; then
            _VLA_RAW_CONFLICTS["$knob"]="${!knob}"
        fi
    done
    if [[ ${#_VLA_RAW_CONFLICTS[@]} -gt 0 ]]; then
        echo -e "${YELLOW}[vla-runtime] --vla-raw overrides these operator-set knobs:${NC}" >&2
        for k in "${!_VLA_RAW_CONFLICTS[@]}"; do
            echo -e "${YELLOW}    ${k}=${_VLA_RAW_CONFLICTS[$k]}  ->  0 (disabled by --vla-raw)${NC}" >&2
        done
    fi
    VLA_TARGET_LPF_HZ=0
    VLA_FUTURE_LPF_HZ=0
    VLA_HAND_LPF_HZ=0
    VLA_HAND_CHUNK_BLEND_TICKS=0
    VLA_MAX_HAND_STEP=0
    VLA_MAX_WIRE_DEV_FROM_BODY=0
    VLA_MAX_WIRE_STEP=0
    VLA_CHUNK_BLEND_TICKS=0
    # 2026-06-10 follow-up 11: tracking feedback is wire-shaping too
    # (closed-loop variant of max_wire_step). --vla-raw means "no
    # wire shaping" by contract, so disable it here. Operators who
    # really want feedback active under --vla-raw must opt back in
    # explicitly with --vla-tracking-feedback AFTER --vla-raw.
    if [[ "${VLA_TRACKING_FEEDBACK}" == "1" ]]; then
        echo -e "${YELLOW}[vla-runtime] --vla-raw disables VLA_TRACKING_FEEDBACK (closed-loop wire shaping)${NC}" >&2
    fi
    VLA_TRACKING_FEEDBACK=0
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

# --robocasa-env validation (sim-only). Resolve the short name to an absolute
# MJCF path under gear_sonic/data/assets/robocasa_scenes/<NAME>.xml and fail
# loud if the scene isn't pre-built. On real-robot runs the flag makes no
# sense (scene comes from physics, not MuJoCo), so reject early.
if [[ -n "$ROBOCASA_ENV" ]]; then
    if [[ "$SIM_MODE" -ne 1 ]]; then
        echo -e "${RED}[vla-runtime] --robocasa-env is sim-only; remove it or drop --pc2-host${NC}" >&2
        exit 2
    fi
    ROBOCASA_SCENE_XML="${ROBOCASA_SCENE_DIR}/${ROBOCASA_ENV}.xml"
    if [[ ! -f "$ROBOCASA_SCENE_XML" ]]; then
        echo -e "${RED}[vla-runtime] no robocasa scene at ${ROBOCASA_SCENE_XML}${NC}" >&2
        echo -e "${YELLOW}  available envs:${NC}" >&2
        for xml in "${ROBOCASA_SCENE_DIR}"/*.xml; do
            [[ -e "$xml" ]] || continue
            local_name="$(basename "$xml" .xml)"
            echo -e "${YELLOW}    --robocasa-env ${local_name}${NC}" >&2
        done
        echo -e "${YELLOW}  (re-build assets via: python gear_sonic/scripts/build_x2_robocasa_assets.py)${NC}" >&2
        exit 2
    fi
fi

PID_FILE_BRIDGE="${RUN_DIR}/bridge.pid"
LOG_FILE_BRIDGE="${RUN_DIR}/bridge.log"
PID_FILE_DEPLOY="${RUN_DIR}/deploy.pid"
LOG_FILE_DEPLOY="${RUN_DIR}/deploy.log"
PID_FILE_RECORDER="${RUN_DIR}/recorder.pid"
LOG_FILE_RECORDER="${RUN_DIR}/recorder.log"
PID_FILE_SIM_PROXY="${RUN_DIR}/sim_proxy.pid"
LOG_FILE_SIM_PROXY="${RUN_DIR}/sim_proxy.log"

# Sim-mode pose proxy is in the loop iff the operator opted into
# manual-takeover by enabling either VLA_CONTROL_PORT (edge events
# for the bridge cold restart) or POSE_PROXY_OVERRIDE_PORT (dual-
# source arbitration). Either alone is a valid use case, so we OR
# them. Only takes effect when SIM_MODE=1; real-robot runs always
# rely on the PC2-side proxy spawned by x2_pc2_daemons.sh and
# SIM_PROXY_ENABLED stays 0 so the launcher doesn't try to bind
# loopback sockets that conflict with the laptop's bridge PUB.
SIM_PROXY_ENABLED=0
if [[ "${SIM_MODE}" -eq 1 ]]; then
    if [[ "${VLA_CONTROL_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${VLA_CONTROL_PORT}" -gt 0 ]]; then
        SIM_PROXY_ENABLED=1
    fi
    if [[ "${POSE_PROXY_OVERRIDE_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${POSE_PROXY_OVERRIDE_PORT}" -gt 0 ]]; then
        SIM_PROXY_ENABLED=1
    fi
fi
# Bridge<->recorder ready-file handshake. Always defined so the bridge
# CLI is stable, but only wired into BRIDGE_ARGS / recorder_args below
# when WITH_RECORD=1 (otherwise the bridge starts inference immediately
# the moment policy loads, matching legacy behaviour).
RECORDER_READY_FILE="${RUN_DIR}/recorder_ready"

# ── --with-record validation + default resolution ─────────────────────────
# Validate the flag combination NOW (before any subprocess spawns) so a
# bad command-line fails fast instead of after the bridge is half-up.
if [[ "$WITH_RECORD" -eq 1 ]]; then
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo -e "${RED}[vla-runtime] --with-record requires --output-dir PATH${NC}" >&2
        echo -e "${YELLOW}  e.g. --output-dir data/lerobot/vla_run_x2_pick_and_place_soda_can_n17_50k_v1${NC}" >&2
        exit 2
    fi
    if [[ -z "$TASK" ]]; then
        echo -e "${RED}[vla-runtime] --with-record requires --task TEXT${NC}" >&2
        echo -e "${YELLOW}  e.g. --task \"pick up the mini soda can with your left hand and place it in the open black container on the right\"${NC}" >&2
        exit 2
    fi
    if [[ -z "$MOTION_TOKEN_DECODER" ]]; then
        echo -e "${RED}[vla-runtime] --with-record requires --motion-token-decoder PATH${NC}" >&2
        echo -e "${YELLOW}  (the recorder runs the SONIC inline tokenizer to compute action.motion_token labels; without it the column is zeros and the dataset is NOT VLA-trainable)${NC}" >&2
        exit 2
    fi
fi
# Head-camera default: ON in real-robot mode (we have PC2 stereo), OFF
# in sim (no PC2). Explicit --head-cameras / --no-head-cameras always
# wins. Camera host defaults to PC2_HOST in real mode.
if [[ -z "$HEAD_CAMERAS" ]]; then
    if [[ "$SIM_MODE" -eq 1 ]]; then
        HEAD_CAMERAS=0
    else
        HEAD_CAMERAS=1
    fi
fi
if [[ -z "$CAMERA_HOST" && -n "$PC2_HOST" ]]; then
    CAMERA_HOST="$PC2_HOST"
fi
if [[ "$WITH_RECORD" -eq 1 && "$HEAD_CAMERAS" -eq 1 && -z "$CAMERA_HOST" ]]; then
    echo -e "${RED}[vla-runtime] --with-record + head cameras requires --camera-host HOST (or --pc2-host)${NC}" >&2
    exit 2
fi

# ----- helpers ----------------------------------------------------------

log()   { echo -e "${CYAN}[vla-runtime]${NC} $*"; }
ok()    { echo -e "${GREEN}[vla-runtime]${NC} $*"; }
warn()  { echo -e "${YELLOW}[vla-runtime]${NC} $*" >&2; }
err()   { echo -e "${RED}[vla-runtime]${NC} $*" >&2; }

# ----- Tracking feedback config log (2026-06-10 follow-up 11) ---------
# Emit the operator-facing log line EARLY (right after the log()
# helper is defined, but BEFORE preflight) so the state is visible
# even on failed preflights. Actual BRIDGE_ARGS threading happens
# later in the spawn block.
if [[ "${VLA_TRACKING_FEEDBACK}" == "1" ]]; then
    log "tracking feedback ENABLED: soft=${VLA_TRACKING_SOFT_RAD}rad hard=${VLA_TRACKING_HARD_RAD}rad vel_margin=${VLA_TRACKING_VELOCITY_MARGIN} vel_floor=${VLA_TRACKING_VELOCITY_FLOOR_RAD_TICK}rad/tick stale=${VLA_TRACKING_STALE_MS}ms (closed-loop arm-joint cap; falls back to scalar clamp when proprio stale)"
else
    log "tracking feedback DISABLED (set --vla-tracking-feedback or VLA_TRACKING_FEEDBACK=1 to enable closed-loop wire step cap)"
fi

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

stop_sim_proxy() {
    # Tear down the sim-mode pose proxy if it was spawned. Order
    # matters in stop_all: the sim deploy SUBs from the proxy's
    # downstream port, so stopping deploy FIRST avoids the deploy
    # logging "input went silent" right before we kill it anyway.
    # The proxy holds no operator state (no parquet flush, no model
    # weights), so a 2 s SIGTERM window is plenty.
    if [[ -f "$PID_FILE_SIM_PROXY" ]]; then
        local ppid
        ppid="$(cat "$PID_FILE_SIM_PROXY")"
        if kill -0 "$ppid" 2>/dev/null; then
            log "SIGTERM sim pose proxy pid=$ppid …"
            kill -TERM "$ppid" 2>/dev/null || true
            sleep 2
            kill -KILL "$ppid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE_SIM_PROXY"
    fi
}

stop_recorder() {
    # Drain ordering matters: SIGTERM the recorder FIRST so the
    # finally-block in _run_subscribe_mode auto-saves the open episode
    # (auto-started on first body_pose). We wait up to 30 s because
    # finalising a multi-minute LeRobot episode includes computing
    # per-track video stats + flushing parquet shards.
    if [[ ! -f "$PID_FILE_RECORDER" ]]; then
        return 0
    fi
    local rpid
    rpid="$(cat "$PID_FILE_RECORDER")"
    if ! kill -0 "$rpid" 2>/dev/null; then
        log "recorder pid $rpid already dead; cleaning up pidfile."
        rm -f "$PID_FILE_RECORDER"
        return 0
    fi
    log "SIGTERM recorder pid=$rpid; waiting up to 30s for episode save …"
    kill -TERM "$rpid" 2>/dev/null || true
    for _ in $(seq 1 30); do
        sleep 1
        if ! kill -0 "$rpid" 2>/dev/null; then
            ok "recorder stopped cleanly (episode saved)."
            rm -f "$PID_FILE_RECORDER"
            return 0
        fi
    done
    warn "recorder pid=$rpid still alive after 30s; SIGKILL (episode may be partial)"
    kill -KILL "$rpid" 2>/dev/null || true
    sleep 1
    rm -f "$PID_FILE_RECORDER"
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
    # Sim-mode pose proxy spawned by spawn_sim_proxy. Match the full
    # script path to avoid killing a PC2-side proxy that may be
    # running under a different python in a remote session.
    pkill -TERM -f "gear_sonic_deploy/scripts/x2_pose_proxy.py" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "live_vla_publish_motion_token" 2>/dev/null || true
    pkill -KILL -f "gear_sonic.scripts.dump_x2_debug" 2>/dev/null || true
    pkill -KILL -f "deploy_x2.sh.*sim" 2>/dev/null || true
    pkill -KILL -f "gear_sonic_deploy/scripts/x2_pose_proxy.py" 2>/dev/null || true
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
    # Build a list of ports the proxy + bridge + deploy may have left
    # bound. Override + control entries only included when the operator
    # opted into them (else they're -1 and would expand to nonsense).
    local cleanup_ports=("${LAPTOP_POSE_PORT}" "${PC2_DEBUG_PORT}")
    if [[ "${SIM_PROXY_ENABLED:-0}" -eq 1 ]]; then
        cleanup_ports+=("${POSE_PROXY_DOWNSTREAM_PORT}")
        if [[ "${POSE_PROXY_OVERRIDE_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${POSE_PROXY_OVERRIDE_PORT}" -gt 0 ]]; then
            cleanup_ports+=("${POSE_PROXY_OVERRIDE_PORT}")
        fi
        if [[ "${VLA_CONTROL_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${VLA_CONTROL_PORT}" -gt 0 ]]; then
            cleanup_ports+=("${VLA_CONTROL_PORT}")
        fi
    fi
    local port_re=""
    local port
    for port in "${cleanup_ports[@]}"; do
        [[ -z "${port_re}" ]] && port_re=":${port}\b" || port_re+="|:${port}\b"
    done
    if ss -tln 2>/dev/null | grep -qE "${port_re}"; then
        if command -v fuser >/dev/null 2>&1; then
            warn "ZMQ ports still bound; fuser -k on ${cleanup_ports[*]}"
            for port in "${cleanup_ports[@]}"; do
                fuser -k "${port}/tcp" 2>/dev/null || true
            done
            sleep 2
        fi
    fi
}

stop_all() {
    # Recorder FIRST so its finally-block can drain + save the open
    # episode before the bridge / deploy disappear out from under it.
    # The bridge still has time to receive recorder Ctrl-C, finish its
    # current chunk, and exit cleanly because we wait for the recorder
    # in stop_recorder.
    stop_recorder
    stop_bridge
    stop_deploy
    # Sim proxy LAST: bridge + deploy both touch its sockets, so we
    # let them exit cleanly first. No-op when SIM_PROXY_ENABLED=0.
    stop_sim_proxy
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

spawn_recorder() {
    # Spawn gear_sonic.scripts.record_x2_dataset in VLA subscribe mode.
    # Pre-condition: the bridge has logged "policy ready" (model loaded,
    # ramp-in clock started). On real-robot mode we also assume the PC2
    # head-camera bridge is up because the launcher's preflight verified
    # the :5555 PUB stream is alive.
    #
    # Episode lifecycle: one VLA run = one episode. The recorder
    # auto-starts on first body_pose (no manager publishing
    # ``recorder_cmd``) and the finally-block in _run_subscribe_mode
    # auto-saves on SIGTERM. stop_recorder waits up to 30 s for the
    # parquet/video flush to complete before SIGKILL.
    if [[ "$WITH_RECORD" -ne 1 ]]; then
        return 0
    fi
    if [[ ! -x "$RECORDER_PY" ]]; then
        err "recorder python ${RECORDER_PY} not executable; cannot spawn"
        err "  set --recorder-py to a venv with datasets / av / lerobot"
        return 1
    fi

    local recorder_args=(
        -m gear_sonic.scripts.record_x2_dataset
        --body-pose-source vla
        --arm-targets-source vla
        --body-pose-sub-host localhost
        --body-pose-sub-port "$LAPTOP_POSE_PORT"
        --body-pose-sub-topic pose
        --sub-host "$DEBUG_SUB_HOST"
        --sub-port "$PC2_DEBUG_PORT"
        --sub-topic x2_debug
        --pub-host '*'
        --pub-port "$LAPTOP_POSE_PORT"
        --pub-topic pose
        --rate "$RATE"
        --output-dir "$OUTPUT_DIR"
        --task "$TASK"
        --sonic-checkpoint "$MOTION_TOKEN_DECODER"
        --sonic-tokenizer-device "$SONIC_TOKENIZER_DEVICE"
        --ready-file "$RECORDER_READY_FILE"
    )
    if [[ -n "$ENCODER_CONFIG" ]]; then
        recorder_args+=(--encoder-config "$ENCODER_CONFIG")
    fi
    if [[ "$HEAD_CAMERAS" -eq 1 ]]; then
        recorder_args+=(
            --head-cameras
            --camera-host "$CAMERA_HOST"
            --camera-port "$CAMERA_PORT"
        )
    fi

    log "spawning recorder -> ${LOG_FILE_RECORDER}"
    log "  output_dir = ${OUTPUT_DIR}"
    log "  task       = ${TASK}"
    log "  head cams  = $([ "$HEAD_CAMERAS" -eq 1 ] && echo "ON (${CAMERA_HOST}:${CAMERA_PORT})" || echo OFF)"
    log "  encoder    = ${ENCODER_CONFIG:-<deprecated freeze-pose>}"
    log "  ready-file = ${RECORDER_READY_FILE}"
    log "    (bridge is holding idle stand via --wait-for-ready-file"
    log "     until the recorder touches this file; arm rise will land"
    log "     inside the recorded episode instead of pre-recording)"
    log "  CMD: ${RECORDER_PY} ${recorder_args[*]}"

    # ``-u`` forces stdout / stderr to be line-buffered so the recorder
    # log shows progress in real time. Without it Python defaults to
    # block-buffered when stdout is a file (nohup redirect), and slow
    # init steps (Pinocchio import, OnlineSonicTokenizer ~400 MB load)
    # look like a silent hang -- the operator sees only the first
    # ``flush=True`` print and assumes the process died.
    # ``PYTHONUNBUFFERED=1`` is belt + suspenders for any subprocess
    # the recorder spawns later (e.g. ffmpeg via lerobot's
    # video encoder).
    PYTHONPATH="${REPO_ROOT}" \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    TQDM_DISABLE=1 \
    nohup "$RECORDER_PY" -u "${recorder_args[@]}" \
        > "$LOG_FILE_RECORDER" 2>&1 &
    local rpid=$!
    echo "$rpid" > "$PID_FILE_RECORDER"
    log "recorder.pid = ${rpid}"

    # Wait for the recorder to either:
    #   (a) bind its camera SUB and log "first body_pose" once data flows
    #       (success path), or
    #   (b) crash during the writer-chain pip install / preflight.
    # We give it up to 60 s -- pip-installing datasets/av/lerobot on a
    # cold venv can take 30+ s, and the camera warmup itself is bounded
    # at 8 s -- then carry on. If the recorder is still alive but
    # hasn't logged either marker we assume it's working and just
    # haven't seen body_pose yet.
    sleep 1
    if ! kill -0 "$rpid" 2>/dev/null; then
        err "recorder died immediately; tail of log:"
        tail -n 40 "$LOG_FILE_RECORDER" >&2 || true
        return 1
    fi
    ok "recorder up (episode will auto-start on first VLA body_pose)"
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

spawn_sim_proxy() {
    # Spawn the pose proxy on loopback so the sim deploy gets dual-
    # source arbitration + the same fallback ladder the PC2-side proxy
    # provides on the real robot. The bridge keeps publishing to
    # LAPTOP_POSE_PORT; the proxy SUBs there and republishes to
    # POSE_PROXY_DOWNSTREAM_PORT (where the sim deploy reads from).
    # No-op when SIM_PROXY_ENABLED=0.
    local proxy_py="$POSE_PROXY_PY"
    if [[ -z "$proxy_py" ]]; then
        proxy_py="$BRIDGE_PY"
    fi
    local proxy_script="${REPO_ROOT}/gear_sonic_deploy/scripts/x2_pose_proxy.py"
    if [[ ! -f "$proxy_script" ]]; then
        err "sim pose proxy script missing: $proxy_script"
        return 1
    fi
    if [[ ! -f "$POSE_PROXY_IDLE_X2M2" ]]; then
        err "POSE_PROXY_IDLE_X2M2 missing: $POSE_PROXY_IDLE_X2M2"
        err "  Regenerate via: python -m gear_sonic_deploy.scripts.bake_idle_stand_x2m2"
        return 1
    fi
    local proxy_args=(
        "$proxy_script"
        --upstream-host 127.0.0.1
        --upstream-port "$LAPTOP_POSE_PORT"
        --upstream-topic pose
        --downstream-host "$POSE_PROXY_DOWNSTREAM_HOST"
        --downstream-port "$POSE_PROXY_DOWNSTREAM_PORT"
        --downstream-topic pose
        --idle-x2m2 "$POSE_PROXY_IDLE_X2M2"
        --idle-stale-ms "$POSE_PROXY_IDLE_STALE_MS"
        --idle-mode "$POSE_PROXY_IDLE_MODE"
        --hold-last-secs "$POSE_PROXY_HOLD_LAST_SECS"
        --blend-secs "$POSE_PROXY_BLEND_SECS"
        # Sim has no x2_debug PUB on the deploy side until the deploy
        # boots and binds :5557. Yaw rebase is a real-robot affordance
        # (IMU pelvis quat). Disable so the proxy doesn't spam decode
        # warnings during the deploy warmup window.
        --no-x2-debug-yaw-track
    )
    if [[ "${POSE_PROXY_OVERRIDE_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${POSE_PROXY_OVERRIDE_PORT}" -gt 0 ]]; then
        proxy_args+=(
            --override-host "$POSE_PROXY_OVERRIDE_HOST"
            --override-port "$POSE_PROXY_OVERRIDE_PORT"
            --override-topic "$POSE_PROXY_OVERRIDE_TOPIC"
            --override-stale-ms "$POSE_PROXY_OVERRIDE_STALE_MS"
            --override-frozen-ticks "$POSE_PROXY_OVERRIDE_FROZEN_TICKS"
            --override-frozen-l2-tol "$POSE_PROXY_OVERRIDE_FROZEN_L2_TOL"
            --override-engage-motion-ticks "$POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS"
            --engagement-max-wire-step "$POSE_PROXY_ENGAGEMENT_MAX_WIRE_STEP"
            --engagement-steady-wire-step "$POSE_PROXY_ENGAGEMENT_STEADY_WIRE_STEP"
            --engagement-step-ramp-ticks "$POSE_PROXY_ENGAGEMENT_STEP_RAMP_TICKS"
        )
        # Mode-gated engagement (2026-06-10). When the manager's
        # stream_mode PUB is reachable, hand the proxy its address so
        # engagement keys on operator button presses instead of pose
        # deltas. Setting POSE_PROXY_TELEOP_MODE_PORT <= 0 (or
        # --pose-proxy-teleop-mode-port -1) falls back to the legacy
        # motion-hysteresis path -- only useful for replay smoke
        # tests where no manager is running.
        if [[ "${POSE_PROXY_TELEOP_MODE_PORT}" =~ ^-?[0-9]+$ ]] && \
           [[ "${POSE_PROXY_TELEOP_MODE_PORT}" -gt 0 ]]; then
            proxy_args+=(
                --teleop-mode-host "$POSE_PROXY_TELEOP_MODE_HOST"
                --teleop-mode-port "$POSE_PROXY_TELEOP_MODE_PORT"
                --teleop-mode-topic "$POSE_PROXY_TELEOP_MODE_TOPIC"
                --teleop-mode-stale-ms "$POSE_PROXY_TELEOP_MODE_STALE_MS"
            )
        fi
    fi
    if [[ "${VLA_CONTROL_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${VLA_CONTROL_PORT}" -gt 0 ]]; then
        proxy_args+=(
            # In sim we bind on loopback only; the bridge SUBs from
            # 127.0.0.1 via the VLA_CONTROL_HOST_RESOLVED fallback.
            --vla-control-bind-host 127.0.0.1
            --vla-control-port "$VLA_CONTROL_PORT"
            --vla-control-topic "$VLA_CONTROL_TOPIC"
        )
    fi
    log "spawning sim pose proxy -> ${LOG_FILE_SIM_PROXY}"
    log "  CMD: ${proxy_py} ${proxy_args[*]}"
    nohup "$proxy_py" -u "${proxy_args[@]}" \
        > "$LOG_FILE_SIM_PROXY" 2>&1 &
    echo $! > "$PID_FILE_SIM_PROXY"
    ok "sim_proxy.pid = $(cat "$PID_FILE_SIM_PROXY")"
    # Brief settle so the bind completes before the sim deploy SUBs.
    # PUB/SUB on loopback is forgiving but a SUB attaching mid-bind
    # silently drops the first few frames.
    sleep 0.5
}

spawn_sim_deploy() {
    # When the sim proxy is in the loop, point the deploy at the
    # proxy's downstream port instead of LAPTOP_POSE_PORT. The bridge
    # still publishes to LAPTOP_POSE_PORT; the proxy bridges the two.
    local deploy_pose_host="127.0.0.1"
    local deploy_pose_port="$LAPTOP_POSE_PORT"
    if [[ "${SIM_PROXY_ENABLED:-0}" -eq 1 ]]; then
        deploy_pose_host="$POSE_PROXY_DOWNSTREAM_HOST"
        deploy_pose_port="$POSE_PROXY_DOWNSTREAM_PORT"
    fi
    local deploy_args=(
        sim --vla
        --model "$SIM_MODEL"
        --autostart-after "$SIM_AUTOSTART_AFTER"
        --max-duration "$MAX_DURATION"
        --no-confirm
        --vla-zmq-host "$deploy_pose_host"
        --vla-zmq-port "$deploy_pose_port"
        --vla-zmq-topic pose
        --vla-debug-port "$PC2_DEBUG_PORT"
        --vla-debug-topic x2_debug
        --sim-profile "$SIM_PROFILE"
        # Pose-ref watchdog OFF for local sim: bridge cold-start + docker
        # SUB wiring can exceed the 0.5 s SAFE_IDLE trip (quest3/pkl paths
        # do the same). Without this the robot collapses before SONIC loads.
        --deploy-extra-arg --disable-pose-ref-watchdog
    )
    # Wrist bypass: forwarded verbatim to the C++ deploy binary via
    # ``deploy_x2.sh --deploy-extra-arg``. Two extras because the C++
    # CLI is ``--wrist-bypass <mode>`` (value separated). With ``ik``
    # the deploy overwrites ``target_pos_mj[{20,21,27,28}]`` (left +
    # right wrist pitch + roll) with the IK reference from the latest
    # ZMQ pose frame; the policy outputs for those slots are
    # discarded. ``wrist_yaw`` is left under SONIC because v2
    # telemetry shows it tracks the reference cleanly (corr ~0.8).
    # See ``gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/
    # wrist_bypass.hpp`` for the full empirical justification and the
    # unit test that pins the MJ indices against
    # ``policy_parameters.hpp::mujoco_joint_names`` ordering.
    if [[ -n "${WRIST_BYPASS:-}" ]] && [[ "${WRIST_BYPASS}" != "off" ]]; then
        deploy_args+=(
            --deploy-extra-arg --wrist-bypass
            --deploy-extra-arg "$WRIST_BYPASS"
        )
        # 2026-06-10 follow-up 8: DO NOT auto-pair with
        # ``--max-target-dev``. The 13:21 run proved it the hard
        # way -- ``--max-target-dev`` is a GLOBAL absolute clamp
        # ``|target - default_angles| <= N rad`` applied to ALL
        # joint groups (leg/waist/arm/head), NOT a per-tick step
        # clamp. The deploy startup log spells it out:
        #
        #   SAFETY: per-joint target clamp ENABLED. Effective per-group
        #   |target - default| limits: leg=0.050 rad (2.9 deg),
        #   waist=0.050 rad (2.9 deg), arm=0.050 rad (2.9 deg),
        #   head=0.050 rad (2.9 deg) (global default --max-target-dev=
        #   0.050 rad (2.9 deg); per-group overrides win when > 0)
        #
        # 0.05 rad pinned leg + waist + arm + head joints to +/-2.9
        # deg of default_angles -- the robot couldn't bend its
        # knees more than 2.9 deg to stand, and it collapsed
        # forward onto the table. act_clip_ticks=916/1000 = 92%
        # of policy outputs clamped because the safety clamp was
        # fighting the policy on every tick.
        #
        # The deploy DOES have per-group overrides (per the log,
        # "per-group overrides win when > 0"), but until we wire
        # them through deploy_x2.sh + verify the override names
        # match what the C++ binary expects, leave the auto-pair
        # OFF. wrist_bypass=ik without max-target-dev is the
        # condition that produced the 12:26 wrist slam; operators
        # who want both safe wrist tracking AND standing legs
        # need a wrist-only clamp. Documented as a follow-up.
        :
    fi
    if [[ "$SIM_PROFILE" == "parity" ]]; then
        deploy_args+=(--motion "$SIM_RSI_PKL")
    fi
    if [[ "$SIM_VIEWER" -eq 1 ]]; then
        deploy_args+=(--sim-viewer)
    fi
    if [[ "$SIM_WITH_OMNIHAND" -eq 1 ]]; then
        deploy_args+=(--sim-with-omnihand)
        # When the manual-takeover proxy is in the wire, the OmniHand
        # ZMQ subscriber MUST go through the same proxy as the body
        # joints -- otherwise the override path looks like this:
        #
        #   Operator A+B+X+Y -> recorder :5560 -> proxy :5558
        #       -> sim deploy body SUB  -> body joints applied  ✓
        #       -> OmniHand SUB still pinned at :5556 (bridge)  ✗
        #
        # The bridge keeps publishing VLA hand chunks throughout, so
        # without this redirect the operator's finger commands are
        # silently dropped and only VLA controls the fingers
        # regardless of override engagement state. This is the
        # 2026-06-10 late-afternoon "fingers still not responding"
        # symptom even after the stream_mode gate landed. Routing
        # the OmniHand SUB through the proxy means the same
        # arbitration (mode-gated when teleop_mode is configured)
        # decides body and fingers together -- they can never
        # disagree about which source is driving.
        if [[ "${SIM_PROXY_ENABLED:-0}" -eq 1 ]]; then
            deploy_args+=(
                --sim-hand-zmq-host "$deploy_pose_host"
                --sim-hand-zmq-port "$deploy_pose_port"
            )
        fi
    fi
    if [[ -n "$ROBOCASA_SCENE_XML" ]]; then
        deploy_args+=(--sim-mjcf "$ROBOCASA_SCENE_XML")
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
  ${BOLD}Sim scene       ${NC}: $([[ -n "$ROBOCASA_SCENE_XML" ]] && echo "robocasa: ${ROBOCASA_ENV} (${ROBOCASA_SCENE_XML})" || echo "default x2_ultra.xml (empty world)")
  ${BOLD}x2_debug SUB    ${NC}: tcp://localhost:${PC2_DEBUG_PORT}
  ${BOLD}Cameras         ${NC}: ghost MuJoCo renderer (modality-driven stereo keys)
EOF
    if [[ "${SIM_PROXY_ENABLED:-0}" -eq 1 ]]; then
        cat <<EOF
  ${BOLD}Sim pose proxy  ${NC}: ON (loopback) bridge :${LAPTOP_POSE_PORT} -> proxy -> deploy :${POSE_PROXY_DOWNSTREAM_PORT}
$( [[ "${POSE_PROXY_OVERRIDE_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${POSE_PROXY_OVERRIDE_PORT}" -gt 0 ]] && echo "  ${BOLD}Override SUB    ${NC}: tcp://${POSE_PROXY_OVERRIDE_HOST}:${POSE_PROXY_OVERRIDE_PORT} topic=${POSE_PROXY_OVERRIDE_TOPIC} stale=${POSE_PROXY_OVERRIDE_STALE_MS}ms" || echo "  ${BOLD}Override SUB    ${NC}: disabled (set POSE_PROXY_OVERRIDE_PORT > 0)" )
$( [[ "${VLA_CONTROL_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${VLA_CONTROL_PORT}" -gt 0 ]] && echo "  ${BOLD}vla_control PUB ${NC}: tcp://127.0.0.1:${VLA_CONTROL_PORT} topic=${VLA_CONTROL_TOPIC} (bridge SUBs here)" || echo "  ${BOLD}vla_control PUB ${NC}: disabled (set VLA_CONTROL_PORT > 0)" )
  ${BOLD}Handoff guard   ${NC}: cold_restart_hold=${VLA_COLD_RESTART_HOLD_TICKS}t max_hold=${VLA_HANDOFF_MAX_HOLD_TICKS}t slow_step=${VLA_HANDOFF_MAX_WIRE_STEP}rad/t for ${VLA_HANDOFF_STEP_RAMP_TICKS}t (wire stays at operator pose until first eligible chunk, then ramps via slow-step to normal max_wire_step)
EOF
    fi
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
  ${BOLD}Wire safety     ${NC}: ramp ${VLA_RAMP_IN_TICKS}t bodyLPF ${VLA_TARGET_LPF_HZ}Hz futLPF ${VLA_FUTURE_LPF_HZ}Hz handLPF ${VLA_HAND_LPF_HZ}Hz handBlend ${VLA_HAND_CHUNK_BLEND_TICKS}t handStep≤${VLA_MAX_HAND_STEP} bodyStep≤${VLA_MAX_WIRE_STEP} body≤${VLA_MAX_WIRE_DEV_FROM_BODY} blend ${VLA_CHUNK_BLEND_TICKS}t actionIL≤${VLA_MAX_ACTION_IL}$( [[ "$VLA_RAW" -eq 1 ]] && echo "  ${RED}${BOLD}[--vla-raw: WIRE FILTERS OFF; action-IL clamp KEPT to prevent proprio runaway]${NC}" )
  ${BOLD}Tracking feedback${NC}: $( [[ "${VLA_TRACKING_FEEDBACK}" == "1" ]] && echo "${BOLD}ON${NC}  soft=${VLA_TRACKING_SOFT_RAD}rad hard=${VLA_TRACKING_HARD_RAD}rad vel_margin=${VLA_TRACKING_VELOCITY_MARGIN} vel_floor=${VLA_TRACKING_VELOCITY_FLOOR_RAD_TICK}rad/tick stale=${VLA_TRACKING_STALE_MS}ms (closed-loop per-arm-joint wire cap; falls back to scalar clamp when proprio stale)" || echo "OFF (legacy scalar clamp; opt in via --vla-tracking-feedback)" )
  ${BOLD}Deploy clamp    ${NC}: max-target-dev ${SIM_MAX_TARGET_DEV:-off} deploy-LPF ${SIM_DEPLOY_TARGET_LPF_HZ:-off} Hz (sim)
  ${BOLD}Wrist bypass    ${NC}: ${WRIST_BYPASS:-off}$( [[ "${WRIST_BYPASS:-off}" == "ik" ]] && echo "  (deploy force-writes target_pos_mj[{20,21,27,28}] to wire IK; rate-limited via bridge max_wire_step + follow-up 6 slow-step ramp; NO global max-target-dev auto-pair after the 13:21 leg-collapse regression)" || echo "  ${RED}WARNING: wrist pitch/roll pinned at SONIC comfort -- will NOT respond to operator/VLA wrist commands. Set --wrist-bypass ik to enable.${NC}" )
  ${BOLD}Rate            ${NC}: ${RATE} Hz publisher, ${INFERENCE_MIN_PERIOD_S}s min inference period
  ${BOLD}Max duration    ${NC}: ${MAX_DURATION} s
EOF
if [[ "$SIM_MODE" -eq 0 ]]; then
    echo "  ${BOLD}Camera staleness${NC}: ${CAMERAS_STALENESS_S} s"
fi
if [[ "$WITH_RECORD" -eq 1 ]]; then
    cat <<EOF
  ${BOLD}Recording       ${NC}: ON  -> ${OUTPUT_DIR}
  ${BOLD}  task          ${NC}: "${TASK}"
  ${BOLD}  head cams     ${NC}: $([[ "$HEAD_CAMERAS" -eq 1 ]] && echo "ON (${CAMERA_HOST}:${CAMERA_PORT})" || echo "OFF (ego_view + arm/hand proprio only)")
  ${BOLD}  encoder       ${NC}: ${ENCODER_CONFIG:-DISABLED (deprecated freeze-pose; action.motion_token = zeros)}
  ${BOLD}  one-episode   ${NC}: auto-start on first VLA body_pose, auto-save on Ctrl-C
EOF
else
    echo "  ${BOLD}Recording       ${NC}: OFF (pass --with-record + --output-dir + --task to capture)"
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
# Recorder ready-file handshake: only when --with-record. Holds the
# inference thread at idle stand AFTER 'policy ready' until the
# recorder has subscribed + ingested its first body_pose, so the
# arm rise from idle is captured in the recording (instead of falling
# into the ~8 s warm-up window the recorder needs for X2 model,
# cameras, clock-skew detection, and MuJoCo renderer init).
if [[ "$WITH_RECORD" -eq 1 ]]; then
    # Remove any stale sentinel from a previous run in the same
    # RUN_DIR -- the bridge can't tell a fresh failure from a
    # leftover, and would immediately bypass the wait.
    rm -f "$RECORDER_READY_FILE"
    BRIDGE_ARGS+=(
        --wait-for-ready-file "$RECORDER_READY_FILE"
        --wait-for-ready-file-timeout-s 120
    )
fi
# ----- Manual-takeover (vla_control) wiring ---------------------------
# Opt-in: only forward the vla_control SUB args when an operator has
# explicitly set VLA_CONTROL_PORT > 0 in the environment. We resolve
# the host default LATE (after PC2_HOST has been validated) so the
# common "proxy runs on PC2, bridge on laptop" topology Just Works
# without an extra env var. Pass VLA_CONTROL_HOST explicitly to point
# the bridge at a different vla_control PUB (e.g. a proxy running on
# a third box).
if [[ "${VLA_CONTROL_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${VLA_CONTROL_PORT}" -gt 0 ]]; then
    if [[ -z "${VLA_CONTROL_HOST}" ]]; then
        if [[ -n "${PC2_HOST}" ]]; then
            VLA_CONTROL_HOST_RESOLVED="${PC2_HOST}"
        else
            VLA_CONTROL_HOST_RESOLVED="127.0.0.1"
        fi
    else
        VLA_CONTROL_HOST_RESOLVED="${VLA_CONTROL_HOST}"
    fi
    BRIDGE_ARGS+=(
        --vla-control-host "${VLA_CONTROL_HOST_RESOLVED}"
        --vla-control-port "${VLA_CONTROL_PORT}"
        --vla-control-topic "${VLA_CONTROL_TOPIC}"
        --vla-cold-restart-hold-ticks "${VLA_COLD_RESTART_HOLD_TICKS}"
        --vla-handoff-max-hold-ticks "${VLA_HANDOFF_MAX_HOLD_TICKS}"
        --vla-handoff-max-wire-step "${VLA_HANDOFF_MAX_WIRE_STEP}"
        --vla-handoff-step-ramp-ticks "${VLA_HANDOFF_STEP_RAMP_TICKS}"
    )
    log "vla_control SUB enabled: tcp://${VLA_CONTROL_HOST_RESOLVED}:${VLA_CONTROL_PORT} topic=${VLA_CONTROL_TOPIC} (cold_restart_hold_ticks=${VLA_COLD_RESTART_HOLD_TICKS} handoff_max_hold_ticks=${VLA_HANDOFF_MAX_HOLD_TICKS} handoff_max_wire_step=${VLA_HANDOFF_MAX_WIRE_STEP} handoff_step_ramp_ticks=${VLA_HANDOFF_STEP_RAMP_TICKS})"
else
    log "vla_control SUB disabled (set VLA_CONTROL_PORT > 0 to enable manual-takeover cold restarts)"
fi

# ----- Tracking feedback (2026-06-10 follow-up 11) -------------------
# Closed-loop per-joint feedback on the wire step cap. Default OFF
# for Step 1 belt-and-suspenders rollout; opt in via env var
# (VLA_TRACKING_FEEDBACK=1) or --vla-tracking-feedback CLI flag.
# When ON the bridge reads x2_debug's measured arm-joint positions
# and velocities each tick and throttles the per-joint per-tick
# wire step based on tracking error + measured velocity. The static
# LPF / chunk-blend / scalar step cap stay in place (Step 1) so any
# regression is attributable.
#
# Note: the log line was already emitted right after env-default
# resolution (see the "tracking feedback ENABLED/DISABLED" line
# above the preflight section) so the operator sees the config
# state even if preflight fails. Here we only thread the actual
# CLI args into BRIDGE_ARGS for the spawned bridge.
if [[ "${VLA_TRACKING_FEEDBACK}" == "1" ]]; then
    BRIDGE_ARGS+=(
        --vla-tracking-feedback
        --vla-tracking-soft-rad "${VLA_TRACKING_SOFT_RAD}"
        --vla-tracking-hard-rad "${VLA_TRACKING_HARD_RAD}"
        --vla-tracking-velocity-margin "${VLA_TRACKING_VELOCITY_MARGIN}"
        --vla-tracking-velocity-floor-rad-tick "${VLA_TRACKING_VELOCITY_FLOOR_RAD_TICK}"
        --vla-tracking-stale-ms "${VLA_TRACKING_STALE_MS}"
    )
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
PYTHONUNBUFFERED=1 \
MUJOCO_GL=egl \
TQDM_DISABLE=1 \
nohup "$BRIDGE_PY" -u "${BRIDGE_ARGS[@]}" \
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

    # Manual-takeover plumbing (sim path): spawn the local pose proxy
    # BEFORE the deploy so :5558 is already bound when the deploy SUBs.
    # No-op when SIM_PROXY_ENABLED=0; legacy autonomous-only sim runs
    # remain byte-for-byte unchanged (bridge ↔ deploy direct on :5556).
    if [[ "${SIM_PROXY_ENABLED:-0}" -eq 1 ]]; then
        log "sim manual-takeover plumbing ON: bridge :${LAPTOP_POSE_PORT} -> proxy -> deploy :${POSE_PROXY_DOWNSTREAM_PORT}"
        if ! spawn_sim_proxy; then
            err "sim pose proxy failed to spawn — see $LOG_FILE_SIM_PROXY"
            stop_all
            exit 1
        fi
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
elif [[ "$WITH_RECORD" -eq 1 ]]; then
    # Real-robot recording: we don't spawn a deploy here (PC2 already
    # runs SONIC), but we still need the bridge's policy to be loaded
    # before the recorder starts ingesting -- otherwise the dataset's
    # first ~tens of frames are bootstrap idle_stand rather than VLA
    # decode output, and the action.motion_token labels would be zeros.
    log "waiting for bridge model load (policy ready, ≤180s) …"
    if ! wait_for_log_marker "$LOG_FILE_BRIDGE" "$BRIDGE_PID" \
            "policy ready" 180 "bridge"; then
        err "bridge never finished GR00T load — check $LOG_FILE_BRIDGE"
        stop_all
        exit 1
    fi
    ok "bridge policy ready; spawning recorder"
fi

# Recording: spawn the LeRobot writer AFTER bridge model load. Sim and
# real modes share this path; the function is a no-op when WITH_RECORD=0.
if ! spawn_recorder; then
    err "recorder failed to spawn — check $LOG_FILE_RECORDER"
    stop_all
    exit 1
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
    # ``-n +1`` makes tail emit the full file from line 1 before following
    # new appends. Without it, tail defaults to the last 10 lines, which
    # skips the entire bridge bootstrap (preflight + model load + the
    # recorder-ready wait + pub ticks 0..~850) -- operators see their first
    # ``pub tick=`` line at ~900 and wrongly assume the publisher started
    # late. Showing the full log makes the handshake timeline obvious in
    # the live terminal too.
    tail -n +1 -F --pid="$BRIDGE_PID" "$LOG_FILE_BRIDGE" &
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
if [[ "$WITH_RECORD" -eq 1 ]]; then
    log "recorder log    : ${LOG_FILE_RECORDER}"
    if [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" ]]; then
        n_episodes="$(find "${OUTPUT_DIR}/data" -name 'episode_*.parquet' 2>/dev/null | wc -l || true)"
        log "dataset         : ${OUTPUT_DIR} (${n_episodes:-0} episode parquet(s) on disk)"
        log "replay with     : ./gear_sonic/scripts/view_x2_recorded_dataset.sh --dataset $(basename "$OUTPUT_DIR") --episode <N>"
    else
        warn "dataset dir ${OUTPUT_DIR} missing; recorder may have failed before first save"
    fi
fi
exit "$EXIT_CODE"
