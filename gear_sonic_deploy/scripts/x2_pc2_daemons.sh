#!/usr/bin/env bash
# X2 split-topology PC2 daemon lifecycle wrapper.
#
# Spawns up to four tmux sessions on the robot's PC2 (Jetson Orin NX) and
# gives you a single command to start / stop / inspect them:
#
#     x2_pose_proxy     -- the python idle-fallback pose proxy
#                          (gear_sonic_deploy/scripts/x2_pose_proxy.py).
#                          SUBs to the laptop's pose stream on
#                          tcp://<laptop>:5556 and re-PUBs to
#                          tcp://localhost:${PC2_POSE_PROXY_PORT}
#                          (default 5558). When the upstream wire is
#                          flowing, frames are forwarded byte-for-byte.
#                          When the laptop is silent > 300 ms (wifi
#                          drop, laptop crash, planner stack not yet
#                          started), the proxy runs the staged
#                          fallback ladder (HOLD -> BLEND -> IDLE_CLIP)
#                          keyed by POSE_PROXY_IDLE_MODE -- by default
#                          it re-publishes the LAST forwarded upstream
#                          frame for POSE_PROXY_HOLD_LAST_SECS, then
#                          lerps into the baked idle_stand clip from
#                          ${PC2_PREFIX}/data/idle_stand.x2m2 over
#                          POSE_PROXY_BLEND_SECS. This keeps the
#                          deploy's wire alive (so its pose-ref
#                          starvation watchdog stays disabled in
#                          split-topology mode) WITHOUT stepping the
#                          commanded reference -- the pre-2026-06-08
#                          behaviour (immediate snap to default-stand)
#                          slammed arms into tables on every WiFi
#                          hiccup.
#
#     x2_deploy         -- the agi_x2_deploy_onnx_ref process, launched
#                          via ``deploy_x2.sh onbot`` so the operator
#                          gets the full safety scaffold (Y/n safety
#                          gate, MC stop_app/start_app via PC1 EM HTTP
#                          or aima em CLI fallback, sentinel handoff,
#                          RAMP_OUT -> HOLD_FOR_MC graceful shutdown,
#                          restart_mc_on_exit trap) rather than a bare
#                          ``ros2 run``. With the pose proxy enabled
#                          (default) the deploy SUBs to
#                          tcp://localhost:${PC2_POSE_PROXY_PORT}
#                          instead of the laptop directly, publishes
#                          ``x2_debug`` on tcp://0.0.0.0:5557, and
#                          subscribes to ``pose_resume`` from the
#                          laptop on tcp://<laptop>:5566 for SAFE_IDLE
#                          recovery.
#
#     x2_hand_bridge    -- the python AimDK bridge that translates the
#                          laptop's ``hand_finger_cmd`` ZMQ stream into
#                          /aima/hal/joint/{left_hand,right_hand}/command
#                          ROS topics on PC2.
#
#     x2_motor_monitor  -- the python continuous-motor-state monitor
#                          (writes ${PC2_PREFIX}/log/motor_monitor_*.jsonl
#                          on PC2 and PUBs a compact summary on
#                          tcp://0.0.0.0:5567 for the laptop manager
#                          to subscribe).
#
# All PC2-side artifacts (ONNX runtime, python venv, colcon overlay, logs,
# policies) live under ${PC2_PREFIX} (default /home/run/getsolo) so the
# system Python and /opt are left untouched. Run `pc2_bringup.sh` once on
# a fresh PC2 to lay everything out under that prefix.
#
# Why tmux: the children must outlive any laptop SSH session. When the
# WiFi bridge between the laptop and the robot blinks, the laptop-side
# `ssh -t` would drop and (without tmux) take the deploy down with it,
# dropping the robot mid-policy. With tmux the children keep running on
# PC2 and the pose proxy keeps the wire flowing with local
# idle_stand frames; the robot stays continuously tracked by Sonic
# (instead of latching the last reference and tilting under gravity
# while waiting for resume).
#
# Commands
# --------
#
#     x2_pc2_daemons.sh start        # start all three (idempotent)
#     x2_pc2_daemons.sh stop         # stop all three (clean MC restart)
#     x2_pc2_daemons.sh restart      # stop then start
#     x2_pc2_daemons.sh status       # tmux + service status, recent log tails
#     x2_pc2_daemons.sh logs deploy  # follow-tail one daemon (ctrl-C to exit)
#     x2_pc2_daemons.sh logs hand
#     x2_pc2_daemons.sh logs monitor
#     x2_pc2_daemons.sh logs all
#     x2_pc2_daemons.sh attach deploy  # raw ``tmux attach`` (for typing
#                                        # in the deploy CLI; rarely needed)
#     x2_pc2_daemons.sh postmortem [--center-ts ISO] [--window-s SEC]
#                                    # rsync logs back, run x2_freeze_postmortem.py
#     x2_pc2_daemons.sh print-env    # show resolved PC1/PC2/LAPTOP IPs +
#                                    # ZMQ ports + ping probe. Useful
#                                    # before `start` when switching
#                                    # between wired and WiFi setups.
#
# To DISCOVER the wired + WiFi IPs of both laptop and PC2 in one shot
# (and write `~/.x2/env.wired` + `~/.x2/env.wifi` env files), run the
# companion script `x2_discover_network.sh`. Do this once on the wire
# before your first WiFi session.
#
# Required env / flags
# --------------------
#
#     PC2_USER=run                       # default 'run'
#     PC2_HOST=10.0.1.41                 # default '10.0.1.41' (SDK ethernet)
#     LAPTOP_HOST=10.0.1.50              # default $(hostname -I)
#     PC2_PREFIX=/home/run/getsolo       # root for ws/, venv/, onnxruntime/, log/, policies/
#     PC2_WS=${PC2_PREFIX}/ws            # colcon overlay (auto-derived from PC2_PREFIX)
#     X2_MODEL=/.../model_step_NNNN_g1.onnx   # ONNX path on PC2 (required)
#     X2_TUNING=/.../expressive.yaml          # tuning preset on LAPTOP (optional)
#
# (Or pass --pc2-host / --pc2-user / --laptop-host / --prefix / --pc2-ws /
# --model / --tuning on the CLI.)
#
# We DO NOT stage a `.x2m2` reference-motion file on PC2. The C++ deploy
# runs with `--input-type zmq` here, so its tokenizer reference window is
# fed by ZmqPoseInputSource (Quest 3 manager -> recorder PUB -> PC2 SUB)
# rather than PklMotionReference. SAFE_IDLE uses the hardcoded
# `default_angles` constant in policy_parameters.hpp, so no motion file
# is needed for the recoverable stand-hold either. The only deploy path
# that consumes `--motion <x2m2>` is `--input-type=motion_file` (offline
# replay), which split-topology doesn't use.
#
# Safety contract
# ---------------
#
# `start` spawns the deploy via ``deploy_x2.sh onbot``. The wrapper
# blocks on the Y/n safety gate inside its tmux pane, so MC is NOT
# stopped until the operator confirms; the deploy itself sits in
# STANDBY (writer suppressed) until the start-trigger sentinel fires.
# Pair with ``--attach`` to land directly on the prompt, or
# ``attach deploy`` after the fact. Pass ``--no-confirm`` via
# ``--extra-deploy-arg --no-confirm`` for unattended CI runs.
#
# `stop` SIGINTs the deploy session. deploy_x2.sh's restart_mc_on_exit
# trap then handles the graceful RAMP_OUT -> HOLD_FOR_MC ->
# start_app(mc) handoff inside the tmux pane on PC2, so MC comes back
# into STAND_DEFAULT without any laptop-side intervention (works even
# if WiFi is fully down). The legacy fallback below (curl PC1 EM
# directly) only fires when --no-confirm-stop is set, for unattended
# emergency drops.
#
# postmortem rsync's the deploy CSV log dir + motor_monitor JSONL back
# to the laptop, then runs x2_freeze_postmortem.py with whatever
# manager_sidecar JSONL is already on the laptop.

set -u
set -o pipefail

# -------------------------------------------------------------------------
# Defaults (override via env or CLI flags below).
# -------------------------------------------------------------------------

PC2_USER="${PC2_USER:-run}"
PC2_HOST="${PC2_HOST:-10.0.1.41}"
PC1_HOST="${PC1_HOST:-10.0.1.40}"
PC1_EM_PORT="${PC1_EM_PORT:-50080}"
LAPTOP_HOST="${LAPTOP_HOST:-}"
# All PC2-side artifacts live under PC2_PREFIX. pc2_bringup.sh stages
# onnxruntime/, venv/, ws/, policies/, log/ underneath it. Everything
# else here is derived from this one root, so flipping --prefix is the
# only knob you need to relocate a full PC2 install.
PC2_PREFIX="${PC2_PREFIX:-/home/run/getsolo}"
PC2_WS="${PC2_WS:-${PC2_PREFIX}/ws}"
PC2_LOG_ROOT="${PC2_LOG_ROOT:-${PC2_PREFIX}/log}"
PC2_VENV="${PC2_VENV:-${PC2_PREFIX}/venv}"
PC2_ONNXRUNTIME="${PC2_ONNXRUNTIME:-${PC2_PREFIX}/onnxruntime}"
# aimdk_msgs lives in the AgiBot system tree; only one path on PC2 carries
# the full cmake+lib+python bundle, so we add it to AMENT_PREFIX_PATH at
# launch time.
PC2_AIMDK_PREFIX="${PC2_AIMDK_PREFIX:-/agibot/software/housekeeper/bin/aimdk_msgs}"
PC2_DEPLOY_BIN="${PC2_DEPLOY_BIN:-x2_deploy_onnx_ref}"
PC2_PKG="${PC2_PKG:-agi_x2_deploy_onnx_ref}"
# Source path inside the workspace (must match pc2_bringup.sh layout:
# the C++ package CMakeLists does `add_subdirectory(../../common)` so it
# requires nesting under <ws>/src/x2/).
PC2_PKG_SRC_REL="${PC2_PKG_SRC_REL:-x2/${PC2_PKG}}"
# deploy_x2.sh on PC2 (staged by pc2_bringup.sh step 7). The deploy
# session shells through this wrapper rather than calling `ros2 run`
# directly so the operator gets the full safety scaffold (preflight,
# Y/n gate, MC stop_app/start_app, sentinel handoff, RAMP_OUT trap).
PC2_DEPLOY_SH="${PC2_DEPLOY_SH:-${PC2_PREFIX}/gear_sonic_deploy/deploy_x2.sh}"
# When --no-confirm is set, deploy_x2.sh skips its Y/n safety gate
# (used for unattended runs / CI smoke tests; default OFF so the
# operator is always prompted).
DEPLOY_NO_CONFIRM=0
ROS_DISTRO="${ROS_DISTRO:-humble}"

X2_MODEL="${X2_MODEL:-}"
X2_TUNING="${X2_TUNING:-}"
# Wrist-bypass mode (C++ binary --wrist-bypass): 'ik' overwrites the
# policy's wrist commands with the laptop-computed inverse-kinematics
# targets when --input-type=zmq, which is what every Quest-3-driven
# split-topology run wants. Set to 'off' to fall back to pure policy.
X2_WRIST_BYPASS="${X2_WRIST_BYPASS:-ik}"
# When --lock-head-straight is set, pass --max-target-dev-head to the
# deploy so the policy head target is clamped near the trained default
# (yaw=0, pitch=0). Must be > 0: 0.0 disables the safety clamp.
LOCK_HEAD_STRAIGHT_RAD="${LOCK_HEAD_STRAIGHT_RAD:-0.01}"

# Friendly warn if the operator still has X2_MOTION exported from a
# pre-2026-05 setup; the deploy is now driven by --input-type=zmq so
# any motion clip on PC2 would be silently ignored (the previous
# version of this script required --motion).
if [[ -n "${X2_MOTION:-}" ]]; then
    printf '\e[33m[pc2 WARN]\e[0m X2_MOTION=%q is ignored in split-topology\n' "${X2_MOTION}" >&2
    printf '\e[33m[pc2 WARN]\e[0m   (deploy runs --input-type=zmq; reference window comes from the laptop pose stream)\n' >&2
fi

# ZMQ ports across the laptop <-> PC2 wire.
LAPTOP_POSE_PORT="${LAPTOP_POSE_PORT:-5556}"      # recorder PUB on laptop
LAPTOP_RESUME_PORT="${LAPTOP_RESUME_PORT:-5566}"  # manager PUB on laptop
PC2_DEBUG_PORT="${PC2_DEBUG_PORT:-5557}"          # deploy PUB on PC2
PC2_MONITOR_PORT="${PC2_MONITOR_PORT:-5567}"      # motor-monitor PUB on PC2
LAPTOP_HAND_PORT="${LAPTOP_HAND_PORT:-5564}"      # manager arm/hands on laptop
# PC2-local pose proxy: SUBs to LAPTOP_POSE_PORT on laptop, PUBs to this
# port on PC2 loopback. The C++ deploy SUBs here (not directly to the
# laptop) so the proxy can swap in idle_stand frames whenever the upstream
# wire goes silent (>POSE_PROXY_STALE_MS) without the deploy noticing.
PC2_POSE_PROXY_PORT="${PC2_POSE_PROXY_PORT:-5558}"
# Default 300 ms (was 100 ms before 2026-06-02). The laptop<->PC2 wifi link
# in this lab routinely hits ~100 ms RTT with 23 ms stddev (measured via
# 500-packet ping bursts), so a 100 ms threshold flipped the proxy
# between live-forwarding and idle-fallback every few seconds. Each flip
# is a step change in joint_pos_mj reference -> waist-yaw motor unlocks
# and locks -> audible click. 300 ms is 3x the worst observed wifi gap
# (absorbs every realistic blip) while still falling back to the safe
# idle pose within ~0.3 s of a real outage (laptop crash / VR stack
# shutdown), which keeps the deploy publishing constant 50 Hz pose --
# SONIC's tracking policy expects an uninterrupted reference stream.
POSE_PROXY_STALE_MS="${POSE_PROXY_STALE_MS:-300}"
POSE_PROXY_IDLE_X2M2="${POSE_PROXY_IDLE_X2M2:-${PC2_PREFIX}/data/idle_stand.x2m2}"
# Upstream-silent fallback ladder (see x2_pose_proxy.py --idle-mode).
#
# The 2026-06-08 default is 'blend': HOLD the last forwarded upstream
# frame for POSE_PROXY_HOLD_LAST_SECS (default 10s), then lerp into the
# baked idle clip over POSE_PROXY_BLEND_SECS (default 3s). This soaks
# up WiFi blips / laptop GC stalls without changing the commanded
# reference at all, and only glides toward default-stand if the wire
# stays silent for genuinely long periods (laptop crash, operator
# walked away). The pre-2026-06-08 behaviour (immediate snap to
# default-stand on the first stale tick -- which slammed arms into
# tables during WiFi hiccups) is still available via
# POSE_PROXY_IDLE_MODE=idle-stand for diagnostics / regression
# baselines. POSE_PROXY_IDLE_MODE=hold-last is the operator-
# responsibility mode (HOLD forever; cut power to recover).
POSE_PROXY_IDLE_MODE="${POSE_PROXY_IDLE_MODE:-blend}"
POSE_PROXY_HOLD_LAST_SECS="${POSE_PROXY_HOLD_LAST_SECS:-10.0}"
POSE_PROXY_BLEND_SECS="${POSE_PROXY_BLEND_SECS:-3.0}"
# ---- Manual-takeover dual-source arbitration (2026-06-10 milestone) ----
# When POSE_PROXY_OVERRIDE_PORT > 0, the proxy SUBs to a second pose
# stream (typically the teleop recorder publishing the operator's wire
# on :5560) and prefers it over the primary VLA stream whenever the
# operator is active. POSE_PROXY_CONTROL_PORT > 0 makes the proxy
# emit override_engaged / override_released edge events on a control
# PUB that the VLA bridge SUBs to (vla_control), driving its cold-
# restart on operator release. Both default disabled so existing
# autonomous deployments are byte-for-byte unchanged.
#
# Conventional ports:
#   override   = LAPTOP_HOST:5560 (teleop recorder PUB)
#   control    = 0.0.0.0:5559 (proxy PUB; bridge SUBs from laptop)
POSE_PROXY_OVERRIDE_HOST="${POSE_PROXY_OVERRIDE_HOST:-${LAPTOP_HOST}}"
POSE_PROXY_OVERRIDE_PORT="${POSE_PROXY_OVERRIDE_PORT:--1}"
POSE_PROXY_OVERRIDE_TOPIC="${POSE_PROXY_OVERRIDE_TOPIC:-pose}"
POSE_PROXY_OVERRIDE_STALE_MS="${POSE_PROXY_OVERRIDE_STALE_MS:-200}"
# Frozen-frame release (2026-06-10 follow-up). The Quest3 manager
# publishes the frozen last commanded pose every tick when teleop
# mode is OFF or LOCOMOTION, so the override SUB never goes silent
# across an A+B+X+Y disengage gesture. Frame-equality detection in
# the proxy catches this and fires override_released exactly once
# after N consecutive identical frames. Default 10 ticks @ 50Hz =
# 200ms (matches stale-ms semantics). Set to 0 to disable and
# rely on silence-only release (legacy, only fires on full Ctrl-C
# of the teleop stack).
POSE_PROXY_OVERRIDE_FROZEN_TICKS="${POSE_PROXY_OVERRIDE_FROZEN_TICKS:-10}"
# Bumped from 1e-4 on 2026-06-10 after sim repros showed repeated
# single-frame engage/release cycles from sub-degree controller-rest
# drift while the manager was in OFF (each cycle fires a heavy bridge
# cold-restart). 5e-3 rad ~ 0.3 deg total joint-space motion is well
# above resting jitter and well below intentional teleop motion. Set
# to 1e-4 only for paranoid bytes-match detection.
POSE_PROXY_OVERRIDE_FROZEN_L2_TOL="${POSE_PROXY_OVERRIDE_FROZEN_L2_TOL:-5e-3}"
# Symmetric engage-side hysteresis: require N consecutive override
# frames with joint-space delta ABOVE --override-frozen-l2-tol before
# firing override_engaged. Same default as frozen-ticks (10 = 200ms
# @ 50Hz). Together with the higher tolerance above this prevents
# brief controller jitter from spurious engage/release cycles. Set
# to 0 for the legacy single-frame engage behaviour (only used by
# older smoke tests, not the operator runbook).
POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS="${POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS:-10}"
# Operator-mode SUB (2026-06-10 follow-up). When the laptop's Quest3
# manager is on a known host:port, the proxy gates engagement on the
# manager's stream_mode broadcast (mode != "OFF") and BYPASSES
# motion-hysteresis. Default port matches the manager's
# --recorder-pub-port default (5564). Set TELEOP_MODE_PORT to -1 to
# disable -- the legacy heuristic path is the only one available in
# that case and WILL flicker if the operator holds the controller
# still in ARM_MANIPULATION. Real-robot operators MUST set
# TELEOP_MODE_HOST to the laptop's address (PC2 cannot reach
# 127.0.0.1's manager).
POSE_PROXY_TELEOP_MODE_HOST="${POSE_PROXY_TELEOP_MODE_HOST:-127.0.0.1}"
POSE_PROXY_TELEOP_MODE_PORT="${POSE_PROXY_TELEOP_MODE_PORT:-5564}"
POSE_PROXY_TELEOP_MODE_TOPIC="${POSE_PROXY_TELEOP_MODE_TOPIC:-stream_mode}"
POSE_PROXY_TELEOP_MODE_STALE_MS="${POSE_PROXY_TELEOP_MODE_STALE_MS:-1000}"
POSE_PROXY_CONTROL_BIND="${POSE_PROXY_CONTROL_BIND:-0.0.0.0}"
POSE_PROXY_CONTROL_PORT="${POSE_PROXY_CONTROL_PORT:--1}"
POSE_PROXY_CONTROL_TOPIC="${POSE_PROXY_CONTROL_TOPIC:-vla_control}"
# Auto-disable the proxy if the X2M2 file isn't staged (operators on an
# older bringup; failure mode is just "no idle fallback, behave like before").
NO_POSE_PROXY=0

DEPLOY_SESSION="${DEPLOY_SESSION:-x2_deploy}"
HAND_SESSION="${HAND_SESSION:-x2_hand_bridge}"
MONITOR_SESSION="${MONITOR_SESSION:-x2_motor_monitor}"
POSE_PROXY_SESSION="${POSE_PROXY_SESSION:-x2_pose_proxy}"

POSTMORTEM_OUT="${POSTMORTEM_OUT:-./postmortem_out}"

# Laptop-side repo root, derived from this script's location. Used by the
# just-in-time rsync that ships the --tuning YAML to PC2 at deploy start
# (see cmd_start). Without this, every YAML edit would silently require a
# separate ``pc2_bringup.sh`` run before the new value is observed by the
# deploy binary -- pc2_bringup.sh step 7 is the only other code path that
# stages tuning configs onto PC2.
SCRIPT_DIR_DAEMONS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAPTOP_REPO_ROOT="$(cd "${SCRIPT_DIR_DAEMONS}/../.." && pwd)"

# Set to 1 with --no-sync-tuning to skip the JIT rsync. Useful if the
# operator is intentionally testing a PC2-only override of a tuning preset
# (rare; the default of "always sync" matches the principle of least
# surprise -- the YAML that the deploy reads should be the one the
# operator just edited on the laptop).
NO_SYNC_TUNING=0

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_DIM=$'\e[2m'; C_BLUE=$'\e[34m'; C_RESET=$'\e[0m'
log()  { printf '%s[pc2 %s]%s %s\n' "${C_GREEN}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
warn() { printf '%s[pc2 %s WARN]%s %s\n' "${C_YELLOW}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
err()  { printf '%s[pc2 %s ERROR]%s %s\n' "${C_RED}" "$(date +%H:%M:%S)" "${C_RESET}" "$*" >&2; }
info() { printf '%s[pc2]%s %s\n' "${C_BLUE}" "${C_RESET}" "$*"; }

usage() {
    awk '/^# X2 split-topology/,/^# postmortem/{ sub(/^# ?/, ""); print }' "$0" >&2
    cat >&2 <<USAGE_TAIL

Per-command flags (after the subcommand):

    start    [--pc2-host H] [--pc2-user U] [--laptop-host H]
             [--pc1-host H] [--pc1-em-port N]
             [--prefix DIR] [--pc2-ws W] [--pc2-venv V]
             [--pc2-onnxruntime O] [--pc2-deploy-sh PATH] [--aimdk-prefix A]
             [--model PATH] [--tuning PATH] [--no-sync-tuning]
             [--wrist-bypass {ik,off}]
             [--lock-head-straight]
             [--no-confirm] [--no-monitor] [--no-hand] [--no-pose-proxy]
             [--pose-proxy-port N] [--pose-proxy-stale-ms MS]
             [--pose-proxy-idle-x2m2 PATH]
             [--pose-proxy-idle-mode {blend,hold-last,idle-stand}]
             [--pose-proxy-hold-last-secs SEC]
             [--pose-proxy-blend-secs SEC]
             [--attach] [--attach-settle-seconds N]
             [--extra-deploy-arg ARG]...
    stop     [--pc2-host H] [--pc2-user U]
             [--pc1-host H] [--pc1-em-port N]
             [--no-mc-restart] [--keep-logs] [--yes|-y]
             # stop is interactive by default: prints a banner, runs a
             # short Ctrl-C-able countdown, and then requires typing
             # 'y' + Enter. Use --yes to skip the prompt in scripts.
    status   [--pc2-host H] [--pc2-user U]
    logs     [--pc2-host H] [--pc2-user U] {deploy|hand|monitor|proxy|all}
    attach   [--pc2-host H] [--pc2-user U] {deploy|hand|monitor|proxy}
    postmortem [--pc2-host H] [--pc2-user U]
               [--center-ts ISO] [--window-s SEC] [--out-dir PATH]
    print-env  [--pc2-host H] [--pc2-user U] [--laptop-host H]
               [--pc1-host H] [--pc1-em-port N]

All IP / port flags also accept environment overrides (PC2_HOST,
PC2_USER, LAPTOP_HOST, PC1_HOST, PC1_EM_PORT, LAPTOP_POSE_PORT,
LAPTOP_RESUME_PORT, PC2_DEBUG_PORT, PC2_MONITOR_PORT, LAPTOP_HAND_PORT).
CLI flags win over env. Useful for wired vs WiFi setups:

    # ~/.x2/env.wired
    export PC2_HOST=10.0.1.41
    export LAPTOP_HOST=10.0.1.50

    # ~/.x2/env.wifi
    export PC2_HOST=192.168.1.41
    export LAPTOP_HOST=192.168.1.50

    source ~/.x2/env.wifi && ./x2_pc2_daemons.sh print-env
USAGE_TAIL
    exit 1
}

# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then usage; fi

SUBCMD="$1"; shift

EXTRA_DEPLOY_ARGS=()
LOCK_HEAD_STRAIGHT=0
NO_MONITOR=0
NO_HAND=0
NO_MC_RESTART=0
KEEP_LOGS=0
ATTACH_AFTER_START=0
# Bypass the cmd_stop confirmation prompt. Default 0 ("always ask"); set
# to 1 via --yes for scripted invocations (CI / postmortem automation /
# x2_freeze_postmortem.py-style tooling).
STOP_YES=0
ATTACH_SETTLE_SECONDS="${ATTACH_SETTLE_SECONDS:-2}"
CENTER_TS=""
WINDOW_S=30
LOGS_WHICH="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pc2-host) PC2_HOST="$2"; shift 2 ;;
        --pc2-user) PC2_USER="$2"; shift 2 ;;
        --laptop-host) LAPTOP_HOST="$2"; shift 2 ;;
        --pc1-host) PC1_HOST="$2"; shift 2 ;;
        --pc1-em-port) PC1_EM_PORT="$2"; shift 2 ;;
        --prefix)
            PC2_PREFIX="$2"
            # Re-derive any unset-by-env downstream paths so --prefix
            # actually relocates the whole install in one shot. We only
            # override values that look like our default-derived ones
            # (callers who passed an explicit --pc2-ws, --pc2-venv etc.
            # via env or future flags shouldn't have those silently
            # clobbered).
            PC2_WS="${PC2_PREFIX}/ws"
            PC2_LOG_ROOT="${PC2_PREFIX}/log"
            PC2_VENV="${PC2_PREFIX}/venv"
            PC2_ONNXRUNTIME="${PC2_PREFIX}/onnxruntime"
            PC2_DEPLOY_SH="${PC2_PREFIX}/gear_sonic_deploy/deploy_x2.sh"
            POSE_PROXY_IDLE_X2M2="${PC2_PREFIX}/data/idle_stand.x2m2"
            shift 2 ;;
        --pc2-ws) PC2_WS="$2"; shift 2 ;;
        --pc2-venv) PC2_VENV="$2"; shift 2 ;;
        --pc2-onnxruntime) PC2_ONNXRUNTIME="$2"; shift 2 ;;
        --pc2-deploy-sh) PC2_DEPLOY_SH="$2"; shift 2 ;;
        --aimdk-prefix) PC2_AIMDK_PREFIX="$2"; shift 2 ;;
        --model) X2_MODEL="$2"; shift 2 ;;
        --tuning) X2_TUNING="$2"; shift 2 ;;
        --no-sync-tuning) NO_SYNC_TUNING=1; shift ;;
        --wrist-bypass) X2_WRIST_BYPASS="$2"; shift 2 ;;
        --lock-head-straight) LOCK_HEAD_STRAIGHT=1; shift ;;
        --no-confirm) DEPLOY_NO_CONFIRM=1; shift ;;
        --no-monitor) NO_MONITOR=1; shift ;;
        --no-hand) NO_HAND=1; shift ;;
        --no-pose-proxy) NO_POSE_PROXY=1; shift ;;
        --pose-proxy-idle-x2m2) POSE_PROXY_IDLE_X2M2="$2"; shift 2 ;;
        --pose-proxy-stale-ms) POSE_PROXY_STALE_MS="$2"; shift 2 ;;
        --pose-proxy-port) PC2_POSE_PROXY_PORT="$2"; shift 2 ;;
        --pose-proxy-idle-mode) POSE_PROXY_IDLE_MODE="$2"; shift 2 ;;
        --pose-proxy-hold-last-secs) POSE_PROXY_HOLD_LAST_SECS="$2"; shift 2 ;;
        --pose-proxy-blend-secs) POSE_PROXY_BLEND_SECS="$2"; shift 2 ;;
        --no-mc-restart) NO_MC_RESTART=1; shift ;;
        --keep-logs) KEEP_LOGS=1; shift ;;
        --yes|-y) STOP_YES=1; shift ;;
        --attach) ATTACH_AFTER_START=1; shift ;;
        --attach-settle-seconds) ATTACH_SETTLE_SECONDS="$2"; shift 2 ;;
        --extra-deploy-arg) EXTRA_DEPLOY_ARGS+=("$2"); shift 2 ;;
        --center-ts) CENTER_TS="$2"; shift 2 ;;
        --window-s) WINDOW_S="$2"; shift 2 ;;
        --out-dir) POSTMORTEM_OUT="$2"; shift 2 ;;
        deploy|hand|monitor|proxy|all) LOGS_WHICH="$1"; shift ;;
        -h|--help) usage ;;
        *) err "unknown flag: $1"; usage ;;
    esac
done

# Resolve LAPTOP_HOST late so the --laptop-host CLI override wins.
# LAPTOP_HOST_SOURCE records where the final value came from so
# print-env can report it honestly (and so users debugging "wrong IP"
# can tell whether their override actually landed).
if [[ -n "${LAPTOP_HOST}" ]]; then
    LAPTOP_HOST_SOURCE="(--laptop-host / LAPTOP_HOST env)"
else
    # Pick the first non-loopback IPv4 the laptop knows about. The deploy on
    # PC2 will SUB pose+resume from this address, so it must be reachable
    # over WiFi (or whatever bridge the operator is using).
    LAPTOP_HOST="$(hostname -I 2>/dev/null | awk '{ for (i=1; i<=NF; i++) if ($i !~ /^127\./) { print $i; exit } }')"
    LAPTOP_HOST_SOURCE="(auto-detected via hostname -I; first non-loopback)"
    if [[ -z "${LAPTOP_HOST}" ]]; then
        err "could not auto-detect LAPTOP_HOST. Pass --laptop-host or export LAPTOP_HOST."
        exit 1
    fi
fi

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

ssh_pc2() {
    # Wrap ssh with sane non-interactive defaults.
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
        "${PC2_USER}@${PC2_HOST}" "$@"
}

scp_pc2() {
    scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$@"
}

tmux_session_exists() {
    local name="$1"
    ssh_pc2 "tmux has-session -t ${name} 2>/dev/null" >/dev/null 2>&1
}

tmux_start_session() {
    # Spawn a detached tmux session that runs ${cmd}.
    #
    # We previously embedded ${cmd} inline inside a triple-nested quote
    # chain (ssh "tmux new-session -d -s NAME 'bash -lc \"...\"'"). That
    # was brittle: laptop bash, PC2 sshd, tmux, and the inner bash -lc
    # each consumed one layer of quoting, and any `${VAR:-}` in the cmd
    # (e.g. ${PYTHONPATH:-} we need expanded INSIDE the tmux pane AFTER
    # source /opt/ros/humble/setup.bash) was being expanded too early
    # and silently lost. The symptom was the python daemons exiting at
    # startup with `No module named 'rclpy'` and tmux killing the
    # session as soon as bash returned, leaving "no server running" by
    # the time --attach tried to attach.
    #
    # The robust fix is to write the cmd to a script file on PC2 first
    # and have tmux just run that script. No nested quoting, and the
    # script stays on disk so the operator can `cat` it to debug what
    # exactly the daemon was trying to run. We also trap the script
    # body with `exec bash -l` on failure so the tmux pane stays
    # attachable -- otherwise an early crash takes the whole tmux
    # session down before the operator can see the traceback.
    local name="$1"; shift
    local cmd="$*"
    if tmux_session_exists "${name}"; then
        # The _keep_alive trap keeps the tmux pane attachable after the
        # daemon body crashes (via `exec bash -l`), which is great for
        # post-mortem debugging but TERRIBLE for the start-skip-if-
        # exists logic below: a zombie pane (daemon already exited,
        # interactive shell still attached) looks identical to a
        # healthy session from `tmux has-session`'s point of view, so
        # we'd silently re-use the corpse instead of launching the
        # new code that pc2_bringup.sh just rsynced. This is exactly
        # how 2026-06-08's stale-proxy debugging session ate an hour:
        # every `start` was a no-op because the proxy had crashed on
        # an argparse mismatch and left its pane behind as a bash -l.
        #
        # Detect the post-mortem state by looking for the _keep_alive
        # banner in the pane's recent scrollback and force-kill it if
        # found. Healthy sessions never print this line.
        local pane_tail=""
        pane_tail="$(ssh_pc2 "tmux capture-pane -p -t ${name} -S -200" 2>/dev/null || true)"
        if grep -q '\[tmux-launch\] cmd exited with status=' <<<"${pane_tail}"; then
            warn "  tmux session ${name}: existing pane is a post-mortem (daemon already exited)"
            warn "    killing and re-launching so the new code actually runs"
            ssh_pc2 "tmux kill-session -t ${name} 2>/dev/null || true"
        else
            log "  tmux session ${name}: already exists -- skipping"
            return 0
        fi
    fi
    local script_path="${PC2_LOG_ROOT}/start_${name}.sh"
    # Build a self-contained script: print banner, run cmd, then via an
    # EXIT trap drop the operator into an interactive shell with the
    # same env so the failure scrollback survives `tmux attach -t NAME`.
    #
    # Quirks worth noting:
    #   - NO `set -u` here. /opt/ros/humble/setup.bash references unbound
    #     ament internals (e.g. AMENT_TRACE_SETUP_FILES) and the source
    #     would abort instantly under strict mode. The inner daemon
    #     scripts (deploy / hand / monitor) set their own strict mode.
    #   - The trap fires on ANY exit (normal, error, signal). That is
    #     what keeps the tmux pane attachable when the daemon body
    #     errors out -- without it, the pane process exits, tmux
    #     reaps the session, and `start --attach` sees "no sessions".
    #   - `exec bash -l` replaces the shell so tmux still sees an
    #     active foreground process; Ctrl-D in the attached pane
    #     cleanly takes the session down.
    local script_body
    printf -v script_body '%s\n' \
        '#!/usr/bin/env bash' \
        "# autogenerated by x2_pc2_daemons.sh for tmux session: ${name}" \
        "# regenerated on every 'start'; edit the daemon script, not this file" \
        '_keep_alive() {' \
        '    local rc=$?' \
        '    echo' \
        '    echo "[tmux-launch] cmd exited with status=${rc}"' \
        '    if [[ "${rc}" -ne 0 ]]; then' \
        '        echo "[tmux-launch] NON-ZERO EXIT -- inspect scrollback above for the real error."' \
        '    fi' \
        '    echo "[tmux-launch] keeping pane alive (read scrollback above; Ctrl-D to close)"' \
        '    exec bash -l' \
        '}' \
        'trap _keep_alive EXIT' \
        'echo "[tmux-launch] session=${TMUX_PANE:-?} pid=$$"' \
        'echo "[tmux-launch] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"' \
        'echo "[tmux-launch] ---- daemon body ----"' \
        "${cmd}"
    log "  tmux session ${name}: writing ${script_path}"
    # Stream the script body to PC2 via ssh stdin (no quoting layers).
    if ! ssh_pc2 "mkdir -p $(dirname "${script_path}") && cat > ${script_path} && chmod +x ${script_path}" <<<"${script_body}"; then
        err "  tmux session ${name}: failed to write launch script to PC2"
        return 1
    fi
    log "  tmux session ${name}: starting"
    # Single layer of quoting now -- tmux just invokes the script file.
    ssh_pc2 "tmux new-session -d -s ${name} ${script_path}"
}

tmux_kill_session() {
    local name="$1"
    if ! tmux_session_exists "${name}"; then
        log "  tmux session ${name}: not running"
        return 0
    fi
    log "  tmux session ${name}: SIGINT then 5s grace, then kill-session"
    # Send Ctrl-C into the pane so the python / ros2 shutdown hooks can
    # flush. The tmux send-keys C-c lands as SIGINT to the foreground
    # process inside the pane.
    ssh_pc2 "tmux send-keys -t ${name} C-c 2>/dev/null || true"
    sleep 5
    ssh_pc2 "tmux kill-session -t ${name} 2>/dev/null || true"
}

tail_session_log() {
    local name="$1"
    local label="$2"
    info "follow-tailing ${name} (${label}) (Ctrl-C to exit)..."
    ssh_pc2 -t "tmux capture-pane -p -t ${name} -S -200 2>/dev/null || echo '[no pane found]'; tmux pipe-pane -t ${name} 'cat >&2' 2>/dev/null; sleep 2147483647" || true
}

# -------------------------------------------------------------------------
# start
# -------------------------------------------------------------------------

cmd_start() {
    if [[ -z "${X2_MODEL}" ]]; then
        err "--model PATH (or X2_MODEL env) is required for start."
        exit 1
    fi

    log "starting X2 split-topology daemons on ${PC2_USER}@${PC2_HOST}"
    log "  prefix=${PC2_PREFIX}  laptop_host=${LAPTOP_HOST}"
    log "  ws=${PC2_WS}  venv=${PC2_VENV}  onnxruntime=${PC2_ONNXRUNTIME}"
    log "  model=${X2_MODEL}"
    log "  wrist_bypass=${X2_WRIST_BYPASS}"
    [[ "${LOCK_HEAD_STRAIGHT}" -eq 1 ]] \
        && log "  lock_head_straight=1 (max_target_dev_head=${LOCK_HEAD_STRAIGHT_RAD})"
    [[ -n "${X2_TUNING}" ]] && log "  tuning=${X2_TUNING}"

    log "  ensuring ${PC2_LOG_ROOT} exists on PC2 (user-owned, no sudo)"
    ssh_pc2 "mkdir -p ${PC2_LOG_ROOT}"

    local now_tag
    now_tag="$(date +%Y%m%d_%H%M%S)"
    local deploy_log_dir="${PC2_LOG_ROOT}/deploy_${now_tag}"
    log "  per-tick CSVs will land in ${deploy_log_dir}/"

    # Start deploy session.
    #
    # We shell through ``deploy_x2.sh onbot`` (PC2-native mode, staged
    # by pc2_bringup.sh at ${PC2_DEPLOY_SH}) rather than calling
    # `ros2 run ${PC2_PKG} ${PC2_DEPLOY_BIN}` directly. The wrapper
    # owns the full safety scaffold that the bare ros2 run path was
    # missing:
    #   - Y/n safety gate (operator confirms BEFORE MC is stopped)
    #   - mc_em_post stop_app (HTTP first, then `aima em stop-app mc`
    #     CLI fallback) -- works on both old and new PC1 firmware
    #   - sentinel-driven STANDBY -> CONTROL handoff
    #   - RAMP_OUT trap + HOLD_FOR_MC + start_app(mc) graceful
    #     shutdown on Ctrl-C, deploy exit, or stop subcommand
    #   - restart_mc_on_exit trap so MC is ALWAYS brought back even
    #     if the deploy crashes mid-run
    # All of this runs ON PC2 in the deploy tmux pane, so it survives
    # a laptop-side WiFi disconnect (the orchestration shell never
    # depends on the laptop being reachable).
    #
    # Note: the deploy_x2.sh wrapper exposes a richer flag surface
    # (--vla, --vla-zmq-*, --vla-resume-*, --tuning-config) plus a
    # --deploy-extra-arg passthrough; we lean on those rather than
    # building the native C++ CLI by hand.
    # Decide whether the pose proxy is in play. When it is, the deploy
    # SUBs to localhost (proxy output) instead of the laptop directly,
    # and the C++ pose-ref starvation watchdog is disabled (the proxy
    # guarantees the wire never goes silent -- it falls back to
    # idle_stand.x2m2 frames whenever the upstream from the laptop is
    # quiet for > POSE_PROXY_STALE_MS).
    local pose_proxy_enabled=0
    if [[ "${NO_POSE_PROXY}" -eq 0 ]]; then
        if ssh_pc2 "test -f '${POSE_PROXY_IDLE_X2M2}'" >/dev/null 2>&1; then
            pose_proxy_enabled=1
            log "  pose proxy: ENABLED"
            log "    upstream    = tcp://${LAPTOP_HOST}:${LAPTOP_POSE_PORT}"
            log "    downstream  = tcp://localhost:${PC2_POSE_PROXY_PORT} (deploy SUBs here)"
            log "    idle x2m2   = ${POSE_PROXY_IDLE_X2M2}"
            log "    stale_ms    = ${POSE_PROXY_STALE_MS}"
            case "${POSE_PROXY_IDLE_MODE}" in
                blend)
                    log "    idle_mode   = blend (HOLD ${POSE_PROXY_HOLD_LAST_SECS}s, BLEND ${POSE_PROXY_BLEND_SECS}s)"
                    ;;
                hold-last)
                    log "    idle_mode   = hold-last (republish last upstream frame indefinitely)"
                    ;;
                idle-stand)
                    warn "    idle_mode   = idle-stand (LEGACY; arms snap to default on first stale tick)"
                    ;;
                *)
                    warn "    idle_mode   = ${POSE_PROXY_IDLE_MODE} (unrecognised; proxy will reject and exit)"
                    ;;
            esac
        else
            warn "pose proxy idle X2M2 missing on PC2: ${POSE_PROXY_IDLE_X2M2}"
            warn "  -> proxy DISABLED; deploy will SUB directly to laptop."
            warn "  Run pc2_bringup.sh to stage data/idle_stand.x2m2, then retry."
        fi
    else
        log "  pose proxy: DISABLED (--no-pose-proxy)"
    fi

    local deploy_vla_host="${LAPTOP_HOST}"
    local deploy_vla_port="${LAPTOP_POSE_PORT}"
    if [[ "${pose_proxy_enabled}" -eq 1 ]]; then
        deploy_vla_host="localhost"
        deploy_vla_port="${PC2_POSE_PROXY_PORT}"
    fi

    local deploy_args=(
        onbot
        --no-docker
        --vla
        --vla-zmq-host "${deploy_vla_host}"
        --vla-zmq-port "${deploy_vla_port}"
        --vla-zmq-topic "pose"
        --vla-debug-port "${PC2_DEBUG_PORT}"
        --vla-debug-topic "x2_debug"
        --vla-resume-host "${LAPTOP_HOST}"
        --vla-resume-port "${LAPTOP_RESUME_PORT}"
        --vla-resume-topic "pose_resume"
        --wrist-bypass "${X2_WRIST_BYPASS}"
        --model "${X2_MODEL}"
        --log-dir "${deploy_log_dir}"
        --onbot-prefix "${PC2_PREFIX}"
        --onbot-ws "${PC2_WS}"
        --onbot-venv "${PC2_VENV}"
        --onbot-onnxruntime "${PC2_ONNXRUNTIME}"
        --onbot-aimdk-prefix "${PC2_AIMDK_PREFIX}"
    )
    if [[ "${pose_proxy_enabled}" -eq 1 ]]; then
        # Proxy keeps the wire continuously flowing -> SAFE_IDLE is no
        # longer needed (and previously caused a hard PD step to
        # default_angles with 4x kd that whirred the motors). Pass the
        # disable flag via --deploy-extra-arg since deploy_x2.sh doesn't
        # expose --disable-pose-ref-watchdog as a first-class flag.
        deploy_args+=(--deploy-extra-arg --disable-pose-ref-watchdog)
    fi
    if [[ -n "${X2_TUNING}" ]]; then
        # --tuning-config resolves relative to deploy_x2.sh's SCRIPT_DIR
        # on PC2 (i.e. /home/run/getsolo/gear_sonic_deploy/), so absolute
        # paths or paths relative to that root both work; users typically
        # pass `gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml`
        # on the laptop and we map it onto the staged PC2 copy.
        #
        # In addition to the path remap below, we JIT-rsync the laptop
        # YAML to its PC2 destination so an edit on the laptop is
        # automatically reflected in the next deploy start. Without this,
        # the only code path that ships tuning YAMLs to PC2 is
        # ``pc2_bringup.sh`` step 7 -- and people forget to re-run it,
        # leading to the very confusing situation where the deploy reads
        # a STALE preset that doesn't match what's open in the editor.
        # The rsync is a few KB and runs in <100 ms, so we always do it
        # unless ``--no-sync-tuning`` is set (intentional PC2-only test).
        local resolved_tuning="${X2_TUNING}"
        local laptop_tuning_src=""
        case "${X2_TUNING}" in
            /*)
                # Absolute path. If it sits under the laptop repo root,
                # rsync it; otherwise pass through and trust the operator.
                if [[ -f "${X2_TUNING}" && "${X2_TUNING}" == "${LAPTOP_REPO_ROOT}/"* ]]; then
                    laptop_tuning_src="${X2_TUNING}"
                fi
                ;;
            gear_sonic_deploy/*)
                resolved_tuning="${PC2_PREFIX}/${X2_TUNING}"
                laptop_tuning_src="${LAPTOP_REPO_ROOT}/${X2_TUNING}"
                ;;
            *)
                # Treat as basename under configs/real_deploy_tuning/.
                resolved_tuning="${PC2_PREFIX}/gear_sonic_deploy/configs/real_deploy_tuning/${X2_TUNING}"
                laptop_tuning_src="${LAPTOP_REPO_ROOT}/gear_sonic_deploy/configs/real_deploy_tuning/${X2_TUNING}"
                ;;
        esac
        if [[ "${NO_SYNC_TUNING}" -eq 0 && -n "${laptop_tuning_src}" ]]; then
            if [[ -f "${laptop_tuning_src}" ]]; then
                log "  rsync tuning YAML (laptop -> PC2):"
                log "    src = ${laptop_tuning_src}"
                log "    dst = ${PC2_USER}@${PC2_HOST}:${resolved_tuning}"
                # --inplace keeps the destination's inode stable so any
                # daemon that's already running (and reading the file)
                # sees a consistent view. --checksum forces a content
                # compare instead of mtime, since the just-edited file
                # often has a brand-new mtime even when it didn't change.
                if ! rsync -av --inplace --checksum --info=stats0,progress0 \
                        -e "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5" \
                        "${laptop_tuning_src}" \
                        "${PC2_USER}@${PC2_HOST}:${resolved_tuning}"; then
                    err "tuning YAML rsync to PC2 failed."
                    err "  Fix the connection (or pass --no-sync-tuning) and retry."
                    exit 1
                fi
            else
                warn "tuning YAML not found on laptop at ${laptop_tuning_src};"
                warn "  trusting whatever's staged on PC2. (Pass --no-sync-tuning"
                warn "  to silence this warning if the absence is intentional.)"
            fi
        fi
        deploy_args+=(--tuning-config "${resolved_tuning}")
    fi
    if [[ "${DEPLOY_NO_CONFIRM}" -eq 1 ]]; then
        deploy_args+=(--no-confirm)
    fi
    if [[ "${LOCK_HEAD_STRAIGHT}" -eq 1 ]]; then
        deploy_args+=(--max-target-dev-head "${LOCK_HEAD_STRAIGHT_RAD}")
    fi
    for a in "${EXTRA_DEPLOY_ARGS[@]:-}"; do
        deploy_args+=("$a")
    done

    # Sanity: deploy_x2.sh + dependencies must already be staged on PC2.
    if ! ssh_pc2 "test -x ${PC2_DEPLOY_SH}"; then
        err "deploy_x2.sh not found at ${PC2_DEPLOY_SH} on PC2."
        err "  Run pc2_bringup.sh from the laptop first to stage it:"
        err "    ./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host ${PC2_HOST}"
        exit 1
    fi

    # Build the bash one-liner the deploy tmux pane will run. We do NOT
    # source ROS / aimdk / colcon / onnxruntime here -- deploy_x2.sh
    # onbot does all of that internally (see its `--onbot-*` plumbing),
    # so this stays a thin wrapper. cd into the gear_sonic_deploy/
    # tree on PC2 so SCRIPT_DIR-relative lookups (configs/, scripts/,
    # x2_preflight.py, tuning_config_to_args.py, x2_mc_escalator.py)
    # resolve against the staged copy and not the user's pwd.
    local deploy_cmd="cd ${PC2_PREFIX}/gear_sonic_deploy && \
        ${PC2_DEPLOY_SH} ${deploy_args[*]}"

    # Shared env prelude for the python daemons (pose proxy, hand bridge,
    # motor monitor). Every child needs:
    #   - ROS humble setup (rclpy / rclcpp / ros2 launcher)
    #   - aimdk_msgs on AMENT_PREFIX_PATH + LD_LIBRARY_PATH (its cmake config,
    #     libs, and msg defs live under PC2_AIMDK_PREFIX; not exported by
    #     /opt/ros/humble/setup.bash)
    #   - our colcon overlay's install/setup.bash (registers
    #     agi_x2_deploy_onnx_ref, adds its lib dir to LD_LIBRARY_PATH)
    #   - prebuilt ONNX Runtime on LD_LIBRARY_PATH (the deploy binary
    #     dynamically links libonnxruntime.so from PC2_ONNXRUNTIME/lib;
    #     not strictly required for the python daemons but harmless and
    #     keeps both sessions byte-identical)
    #
    # The deploy session does NOT use this prelude any more -- it shells
    # through deploy_x2.sh onbot, which sources the equivalent env
    # itself (so changing it in two places stays unnecessary).
    #
    # NB: do NOT single-quote the values containing \$AMENT_PREFIX_PATH /
    # \$LD_LIBRARY_PATH -- single quotes suppress PC2-side expansion and a
    # literal '${AMENT_PREFIX_PATH:-}' would otherwise end up on the path,
    # breaking ament_index_get_resources at deploy startup.
    local env_prelude="source /opt/ros/${ROS_DISTRO}/setup.bash && \
        export AMENT_PREFIX_PATH=${PC2_AIMDK_PREFIX}:\$AMENT_PREFIX_PATH && \
        export LD_LIBRARY_PATH=${PC2_AIMDK_PREFIX}/lib:\$LD_LIBRARY_PATH && \
        source ${PC2_WS}/install/setup.bash && \
        export LD_LIBRARY_PATH=${PC2_ONNXRUNTIME}/lib:\$LD_LIBRARY_PATH"

    # Python daemons run under the prefix venv (--system-site-packages so
    # numpy comes from system, pyzmq from the venv). PYTHONPATH needs three
    # things:
    #   - aimdk_msgs python bindings (no env hook ships them in PYTHONPATH;
    #     they live in <aimdk_prefix>/local/lib/python3.10/dist-packages/)
    #   - PC2_WS/src so the hand bridge can resolve `from gear_sonic.utils.
    #     teleop.zmq.zmq_packed_message_decoder import ...` from the rsynced
    #     compat tree at ws/src/gear_sonic/utils/teleop/zmq/
    local python_env="${env_prelude} && \
        export PYTHONPATH=${PC2_AIMDK_PREFIX}/local/lib/python3.10/dist-packages:${PC2_WS}/src:\${PYTHONPATH:-}"
    local python_bin="${PC2_VENV}/bin/python3"

    # Start pose proxy BEFORE the deploy. The deploy SUBs to
    # localhost:PC2_POSE_PROXY_PORT and would otherwise see "no upstream"
    # and trip its starvation watchdog (unless --disable-pose-ref-watchdog
    # is passed, which we do). Bringing the proxy up first means by the
    # time the operator hits "go" in the deploy tmux pane, the wire is
    # already flowing (idle frames if the laptop publisher isn't running
    # yet; live frames once it does).
    if [[ "${pose_proxy_enabled}" -eq 1 ]]; then
        local proxy_log="${PC2_LOG_ROOT}/pose_proxy_${now_tag}.log"
        local proxy_script="${PC2_PREFIX}/gear_sonic_deploy/scripts/x2_pose_proxy.py"
        # Optional manual-takeover args. Only forwarded when the
        # operator has opted in by setting POSE_PROXY_OVERRIDE_PORT
        # and/or POSE_PROXY_CONTROL_PORT to a positive integer in
        # the environment / via systemd unit overrides.
        local proxy_takeover_args=""
        if [[ "${POSE_PROXY_OVERRIDE_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${POSE_PROXY_OVERRIDE_PORT}" -gt 0 ]]; then
            proxy_takeover_args+=" --override-host ${POSE_PROXY_OVERRIDE_HOST}"
            proxy_takeover_args+=" --override-port ${POSE_PROXY_OVERRIDE_PORT}"
            proxy_takeover_args+=" --override-topic ${POSE_PROXY_OVERRIDE_TOPIC}"
            proxy_takeover_args+=" --override-stale-ms ${POSE_PROXY_OVERRIDE_STALE_MS}"
            proxy_takeover_args+=" --override-frozen-ticks ${POSE_PROXY_OVERRIDE_FROZEN_TICKS}"
            proxy_takeover_args+=" --override-frozen-l2-tol ${POSE_PROXY_OVERRIDE_FROZEN_L2_TOL}"
            proxy_takeover_args+=" --override-engage-motion-ticks ${POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS}"
            if [[ "${POSE_PROXY_TELEOP_MODE_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${POSE_PROXY_TELEOP_MODE_PORT}" -gt 0 ]]; then
                proxy_takeover_args+=" --teleop-mode-host ${POSE_PROXY_TELEOP_MODE_HOST}"
                proxy_takeover_args+=" --teleop-mode-port ${POSE_PROXY_TELEOP_MODE_PORT}"
                proxy_takeover_args+=" --teleop-mode-topic ${POSE_PROXY_TELEOP_MODE_TOPIC}"
                proxy_takeover_args+=" --teleop-mode-stale-ms ${POSE_PROXY_TELEOP_MODE_STALE_MS}"
                log "  pose proxy: override SUB enabled tcp://${POSE_PROXY_OVERRIDE_HOST}:${POSE_PROXY_OVERRIDE_PORT} stale_ms=${POSE_PROXY_OVERRIDE_STALE_MS} frozen_ticks=${POSE_PROXY_OVERRIDE_FROZEN_TICKS} frozen_l2_tol=${POSE_PROXY_OVERRIDE_FROZEN_L2_TOL} engage_motion_ticks=${POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS} (engage gate: STRICT stream_mode tcp://${POSE_PROXY_TELEOP_MODE_HOST}:${POSE_PROXY_TELEOP_MODE_PORT} topic=${POSE_PROXY_TELEOP_MODE_TOPIC} stale_ms=${POSE_PROXY_TELEOP_MODE_STALE_MS}; motion-hysteresis IGNORED)"
            else
                log "  pose proxy: override SUB enabled tcp://${POSE_PROXY_OVERRIDE_HOST}:${POSE_PROXY_OVERRIDE_PORT} stale_ms=${POSE_PROXY_OVERRIDE_STALE_MS} frozen_ticks=${POSE_PROXY_OVERRIDE_FROZEN_TICKS} frozen_l2_tol=${POSE_PROXY_OVERRIDE_FROZEN_L2_TOL} engage_motion_ticks=${POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS} (engage gate: LEGACY motion-hysteresis; will flicker if operator holds controller still -- set POSE_PROXY_TELEOP_MODE_PORT to enable strict mode)"
            fi
        fi
        if [[ "${POSE_PROXY_CONTROL_PORT}" =~ ^-?[0-9]+$ ]] && [[ "${POSE_PROXY_CONTROL_PORT}" -gt 0 ]]; then
            proxy_takeover_args+=" --vla-control-bind-host ${POSE_PROXY_CONTROL_BIND}"
            proxy_takeover_args+=" --vla-control-port ${POSE_PROXY_CONTROL_PORT}"
            proxy_takeover_args+=" --vla-control-topic ${POSE_PROXY_CONTROL_TOPIC}"
            log "  pose proxy: vla_control PUB enabled tcp://${POSE_PROXY_CONTROL_BIND}:${POSE_PROXY_CONTROL_PORT} topic=${POSE_PROXY_CONTROL_TOPIC}"
        fi
        local proxy_cmd="${python_env} && \
            ${python_bin} ${proxy_script} \
                --upstream-host ${LAPTOP_HOST} \
                --upstream-port ${LAPTOP_POSE_PORT} \
                --upstream-topic pose \
                --downstream-port ${PC2_POSE_PROXY_PORT} \
                --downstream-topic pose \
                --idle-x2m2 ${POSE_PROXY_IDLE_X2M2} \
                --idle-stale-ms ${POSE_PROXY_STALE_MS} \
                --idle-mode ${POSE_PROXY_IDLE_MODE} \
                --hold-last-secs ${POSE_PROXY_HOLD_LAST_SECS} \
                --blend-secs ${POSE_PROXY_BLEND_SECS}${proxy_takeover_args} \
                2>&1 | tee -a ${proxy_log}"
        tmux_start_session "${POSE_PROXY_SESSION}" "${proxy_cmd}"
    fi

    tmux_start_session "${DEPLOY_SESSION}" "${deploy_cmd}"

    if [[ "${NO_HAND}" -eq 0 ]]; then
        local hand_log="${PC2_LOG_ROOT}/hand_bridge_${now_tag}.log"
        local hand_script="${PC2_WS}/src/${PC2_PKG_SRC_REL}/scripts/x2_hand_zmq_to_aimdk_bridge.py"
        # Hand bridge subscribes to the SAME pose stream as the deploy:
        # the laptop's planner-stack publish_motion_token message carries
        # finger joint angles inside the pose payload, NOT on a separate
        # hand-only topic. This mirrors deploy_x2.sh local-mode (which
        # uses ${VLA_ZMQ_HOST}/${VLA_ZMQ_PORT}/${VLA_ZMQ_TOPIC} with
        # default topic "pose") and gives the bridge the same idle-stand
        # fallback the deploy gets when upstream goes silent (proxy
        # injects idle frames that include neutral finger joints).
        # NOTE: LAPTOP_HAND_PORT (5564) on quest3_manager_x2 carries
        # separate `arm_targets` / `hand_finger_cmd` / `stream_mode`
        # topics in a DIFFERENT message format -- the bridge expects
        # pose-format messages and would silently drop everything from
        # 5564 (which is what happened pre-2026-05-17).
        local hand_cmd="${python_env} && \
            ${python_bin} ${hand_script} \
                --zmq-host ${deploy_vla_host} \
                --zmq-port ${deploy_vla_port} \
                --zmq-topic pose \
                2>&1 | tee -a ${hand_log}"
        tmux_start_session "${HAND_SESSION}" "${hand_cmd}"
    else
        log "  hand bridge: SKIPPED (--no-hand)"
    fi

    if [[ "${NO_MONITOR}" -eq 0 ]]; then
        local monitor_log="${PC2_LOG_ROOT}/motor_monitor_${now_tag}.jsonl"
        local monitor_script="${PC2_WS}/src/${PC2_PKG_SRC_REL}/scripts/x2_motor_monitor.py"
        local monitor_cmd="${python_env} && \
            ${python_bin} ${monitor_script} \
                --jsonl ${monitor_log} \
                --zmq-port ${PC2_MONITOR_PORT} \
                --zmq-topic motor_monitor"
        tmux_start_session "${MONITOR_SESSION}" "${monitor_cmd}"
    else
        log "  motor monitor: SKIPPED (--no-monitor)"
    fi

    log "started; check status with: $0 status"

    # If --attach was passed, give the deploy a moment to settle into
    # WAIT_FOR_CONTROL (load ONNX, transition STANDBY -> INIT) before
    # we exec into the tmux pane. The deploy's stdin prompt for `go`
    # is what we want the operator to see; landing too early just
    # shows the boot scrollback, which is also fine. We use `exec`
    # so the laptop-side wrapper is replaced by ssh -- detaching
    # the pane with Ctrl-B d returns the operator straight to the
    # laptop shell with no nested process to clean up.
    if [[ "${ATTACH_AFTER_START}" -eq 1 ]]; then
        log "  --attach: sleeping ${ATTACH_SETTLE_SECONDS}s for deploy to reach WAIT_FOR_CONTROL, then attaching..."
        sleep "${ATTACH_SETTLE_SECONDS}"
        info "attaching to ${DEPLOY_SESSION} on PC2 (Ctrl-B d to detach without killing; type 'go' + Enter to begin CONTROL)"
        exec ssh -t "${PC2_USER}@${PC2_HOST}" "tmux attach -t ${DEPLOY_SESSION}"
    fi
}

# -------------------------------------------------------------------------
# stop
# -------------------------------------------------------------------------

cmd_stop() {
    # Stop is destructive and operator-only -- a stray ``stop`` after
    # several hours of teleop kills the policy mid-session and forces a
    # full pc2_bringup or daemons start --attach restart that also has
    # to wait through the WAIT_FOR_CONTROL handshake. We've eaten that
    # cost at least once. Default policy is therefore "explicit
    # acknowledgement required", with --yes as the documented bypass
    # for scripted invocations (CI / postmortem automation).

    # Surface which sessions are actually about to die. Capturing this
    # also serves as a "did you mean to do this?" sanity check -- if
    # nothing is running, we can short-circuit the prompt entirely
    # rather than annoying the operator with a confirm-to-kill-nothing.
    local sessions_running=()
    for s in "${DEPLOY_SESSION}" "${POSE_PROXY_SESSION}" "${HAND_SESSION}" "${MONITOR_SESSION}"; do
        if tmux_session_exists "${s}"; then
            sessions_running+=("${s}")
        fi
    done

    if [[ "${STOP_YES}" -ne 1 ]]; then
        if [[ ${#sessions_running[@]} -eq 0 ]]; then
            log "nothing to stop on ${PC2_USER}@${PC2_HOST} (no daemon sessions running)."
            if [[ "${NO_MC_RESTART}" -eq 1 ]]; then
                return 0
            fi
            # Fall through to the MC restart so operators who just want
            # to re-arm MC after a crash can still use this path. But
            # still gate it on confirmation.
        fi

        # Banner. Yellow because this is "you sure?" -- the actual kill
        # below is the red action.
        printf '\n'
        printf '%s========================================================%s\n' "${C_YELLOW}" "${C_RESET}"
        printf '%s ABOUT TO STOP X2 DAEMONS on %s%s%s%s\n' \
            "${C_YELLOW}" "${C_RESET}" "${C_BLUE}" "${PC2_USER}@${PC2_HOST}" "${C_RESET}"
        printf '%s========================================================%s\n' "${C_YELLOW}" "${C_RESET}"
        if [[ ${#sessions_running[@]} -gt 0 ]]; then
            printf '   tmux sessions queued for kill:\n'
            for s in "${sessions_running[@]}"; do
                printf '     - %s\n' "${s}"
            done
        fi
        if [[ "${NO_MC_RESTART}" -eq 0 ]]; then
            printf '   then will re-arm MC via EM HTTP on %s:%s\n' "${PC1_HOST}" "${PC1_EM_PORT}"
        else
            printf '   --no-mc-restart: MC will NOT be re-armed\n'
        fi
        printf '%s========================================================%s\n' "${C_YELLOW}" "${C_RESET}"
        printf '   Pass --yes to skip this confirmation in scripted use.\n'
        printf '\n'

        # Countdown. Purely visual -- the kill does NOT auto-fire when
        # the timer hits zero; it's a "give the operator 3 seconds to
        # Ctrl-C if they're already regretting it" gate. The Enter
        # prompt below is the actual gate.
        local countdown_s=3
        local i
        for ((i=countdown_s; i>=1; i--)); do
            printf '\r   %sCtrl-C to abort... %d %s' "${C_YELLOW}" "${i}" "${C_RESET}"
            sleep 1
        done
        printf '\r                                  \r'

        # Explicit-Enter gate. read with a -p prompt + -r (no backslash
        # munging) catches "I pressed return on muscle memory" as well
        # as accidental copy-paste. We DON'T accept just bare Enter --
        # the operator has to type 'y' (or 'yes') first. That keeps a
        # stray Enter on the terminal from triggering the kill.
        local reply=""
        printf '   Type %sy%s + Enter to confirm stop, anything else to abort: ' \
            "${C_GREEN}" "${C_RESET}"
        # In case stdin is non-interactive (e.g., piped from a script
        # that didn't pass --yes), read will return immediately with
        # an empty reply, which falls into the abort branch below --
        # that's the safe default.
        read -r reply || true
        printf '\n'
        case "${reply}" in
            y|Y|yes|YES)
                : ;;  # proceed
            *)
                warn "stop aborted (reply=${reply:-<empty>}); daemons untouched."
                return 0
                ;;
        esac
    fi

    log "stopping X2 daemons on ${PC2_USER}@${PC2_HOST}"
    # Order: deploy first (its restart_mc_on_exit trap brings MC back up
    # via the wire-still-flowing proxy), then proxy, then hand bridge +
    # motor monitor. Killing the proxy before the deploy would yank the
    # pose wire out from under deploy's RAMP_OUT and tip MC over a
    # silent input source mid-handoff.
    tmux_kill_session "${DEPLOY_SESSION}"
    tmux_kill_session "${POSE_PROXY_SESSION}"
    tmux_kill_session "${HAND_SESSION}"
    tmux_kill_session "${MONITOR_SESSION}"

    if [[ "${NO_MC_RESTART}" -eq 0 ]]; then
        log "  damping MC (passive) via ssh -> PC2 -> EM HTTP"
        ssh_pc2 "curl -s -m 5 --request POST \
            'http://${PC1_HOST}:${PC1_EM_PORT}/x2/em/start_app?app=mc' \
            -H 'Content-Type: application/json' -d '{}' \
            > /dev/null 2>&1 || true"
    else
        log "  --no-mc-restart: skipping MC damp/restart"
    fi
    log "stopped"
}

# -------------------------------------------------------------------------
# status
# -------------------------------------------------------------------------

cmd_status() {
    log "X2 daemon status (${PC2_USER}@${PC2_HOST})"
    for s in "${POSE_PROXY_SESSION}" "${DEPLOY_SESSION}" "${HAND_SESSION}" "${MONITOR_SESSION}"; do
        if tmux_session_exists "${s}"; then
            local last_lines
            last_lines="$(ssh_pc2 "tmux capture-pane -p -t ${s} -S -10" 2>/dev/null || true)"
            printf '  %s%-22s%s %sRUNNING%s\n' "${C_BLUE}" "${s}" "${C_RESET}" "${C_GREEN}" "${C_RESET}"
            if [[ -n "${last_lines}" ]]; then
                printf '%s\n' "${last_lines}" | sed 's/^/      | /'
            fi
        else
            printf '  %s%-22s%s %sSTOPPED%s\n' "${C_BLUE}" "${s}" "${C_RESET}" "${C_RED}" "${C_RESET}"
        fi
    done
}

# -------------------------------------------------------------------------
# logs
# -------------------------------------------------------------------------

cmd_logs() {
    case "${LOGS_WHICH}" in
        deploy)  tail_session_log "${DEPLOY_SESSION}"     "C++ deploy" ;;
        hand)    tail_session_log "${HAND_SESSION}"       "hand bridge" ;;
        monitor) tail_session_log "${MONITOR_SESSION}"    "motor monitor" ;;
        proxy)   tail_session_log "${POSE_PROXY_SESSION}" "pose proxy" ;;
        all)
            warn "showing the LAST 80 lines of each (use 'logs deploy|hand|monitor|proxy' to follow-tail)"
            for s in "${POSE_PROXY_SESSION}" "${DEPLOY_SESSION}" "${HAND_SESSION}" "${MONITOR_SESSION}"; do
                printf '\n%s== %s ==%s\n' "${C_BLUE}" "${s}" "${C_RESET}"
                ssh_pc2 "tmux capture-pane -p -t ${s} -S -80" 2>/dev/null || warn "  ${s}: not running"
            done
            ;;
        *) err "unknown logs target: ${LOGS_WHICH}"; usage ;;
    esac
}

# -------------------------------------------------------------------------
# attach
# -------------------------------------------------------------------------

cmd_attach() {
    local target="${LOGS_WHICH}"
    if [[ "${target}" == "all" ]]; then
        err "attach requires deploy|hand|monitor|proxy"
        exit 1
    fi
    local s
    case "${target}" in
        deploy)  s="${DEPLOY_SESSION}" ;;
        hand)    s="${HAND_SESSION}" ;;
        monitor) s="${MONITOR_SESSION}" ;;
        proxy)   s="${POSE_PROXY_SESSION}" ;;
        *) err "unknown attach target: ${target}"; usage ;;
    esac
    info "attaching to ${s} on PC2 (Ctrl-b d to detach without killing)"
    exec ssh -t "${PC2_USER}@${PC2_HOST}" "tmux attach -t ${s}"
}

# -------------------------------------------------------------------------
# postmortem
# -------------------------------------------------------------------------

cmd_postmortem() {
    log "postmortem: pulling logs from PC2 -> laptop"
    mkdir -p "${POSTMORTEM_OUT}"
    local pulled_dir="${POSTMORTEM_OUT}/pc2_logs"
    mkdir -p "${pulled_dir}"

    log "  rsyncing ${PC2_LOG_ROOT} -> ${pulled_dir}/"
    rsync -av --inplace --info=progress2 \
        "${PC2_USER}@${PC2_HOST}:${PC2_LOG_ROOT}/" \
        "${pulled_dir}/" || warn "rsync had non-zero exit; continuing"

    # Find the most recent deploy log dir + motor monitor JSONL.
    local last_deploy_dir
    last_deploy_dir="$(ls -td "${pulled_dir}"/deploy_* 2>/dev/null | head -1 || true)"
    local last_monitor
    last_monitor="$(ls -t "${pulled_dir}"/motor_monitor*.jsonl 2>/dev/null | head -1 || true)"

    if [[ -z "${last_deploy_dir}" && -z "${last_monitor}" ]]; then
        warn "no deploy CSV dir or monitor JSONL found under ${pulled_dir}"
    fi

    # Locate manager sidecar JSONL on the laptop. Convention: most-recent
    # /tmp/x2_quest3_planner_stack-* with manager_sidecar.jsonl.
    local sidecar=""
    sidecar="$(ls -t /tmp/x2_quest3_planner_stack-*/manager_sidecar.jsonl 2>/dev/null | head -1 || true)"
    if [[ -z "${sidecar}" ]]; then
        warn "no manager_sidecar.jsonl found under /tmp/x2_quest3_planner_stack-*"
    fi

    local pm_args=(--out-dir "${POSTMORTEM_OUT}/timeline")
    [[ -n "${last_deploy_dir}" ]] && pm_args+=(--deploy-log-dir "${last_deploy_dir}")
    [[ -n "${last_monitor}" ]]    && pm_args+=(--motor-monitor "${last_monitor}")
    [[ -n "${sidecar}" ]]         && pm_args+=(--manager-sidecar "${sidecar}")
    [[ -n "${CENTER_TS}" ]]       && pm_args+=(--center-ts "${CENTER_TS}" --window-s "${WINDOW_S}")

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    log "  running ${script_dir}/x2_freeze_postmortem.py ${pm_args[*]}"
    python3 "${script_dir}/x2_freeze_postmortem.py" "${pm_args[@]}"
}

# -------------------------------------------------------------------------
# print-env
# -------------------------------------------------------------------------

cmd_print_env() {
    printf '\n'
    printf '%s== x2_pc2_daemons.sh resolved environment ==%s\n' "${C_BLUE}" "${C_RESET}"
    printf '\n'
    printf '  %-22s %s\n' "PC2_HOST"           "${PC2_HOST}"
    printf '  %-22s %s\n' "PC2_USER"           "${PC2_USER}"
    printf '  %-22s %s\n' "PC2_PREFIX"         "${PC2_PREFIX}"
    printf '  %-22s %s\n' "PC2_WS"             "${PC2_WS}"
    printf '  %-22s %s\n' "PC2_VENV"           "${PC2_VENV}"
    printf '  %-22s %s\n' "PC2_ONNXRUNTIME"    "${PC2_ONNXRUNTIME}"
    printf '  %-22s %s\n' "PC2_AIMDK_PREFIX"   "${PC2_AIMDK_PREFIX}"
    printf '  %-22s %s\n' "PC2_DEPLOY_SH"      "${PC2_DEPLOY_SH}"
    printf '  %-22s %s\n' "PC2_LOG_ROOT"       "${PC2_LOG_ROOT}"
    printf '  %-22s %s\n' "PC1_HOST"           "${PC1_HOST}"
    printf '  %-22s %s\n' "PC1_EM_PORT"        "${PC1_EM_PORT}"
    printf '  %-22s %s  %s\n' \
        "LAPTOP_HOST" "${LAPTOP_HOST}" "${LAPTOP_HOST_SOURCE}"
    printf '\n'
    printf '  %-22s %s\n' "LAPTOP_POSE_PORT"   "${LAPTOP_POSE_PORT}"
    printf '  %-22s %s\n' "LAPTOP_RESUME_PORT" "${LAPTOP_RESUME_PORT}"
    printf '  %-22s %s\n' "LAPTOP_HAND_PORT"   "${LAPTOP_HAND_PORT}"
    printf '  %-22s %s\n' "PC2_DEBUG_PORT"     "${PC2_DEBUG_PORT}"
    printf '  %-22s %s\n' "PC2_MONITOR_PORT"   "${PC2_MONITOR_PORT}"
    printf '\n'
    printf '  %-22s %s\n' "X2_MODEL"           "${X2_MODEL:-<unset>}"
    printf '  %-22s %s\n' "X2_TUNING"          "${X2_TUNING:-<unset>}"
    printf '  %-22s %s\n' "X2_WRIST_BYPASS"    "${X2_WRIST_BYPASS}"
    printf '  %-22s %s\n' "LOCK_HEAD_STRAIGHT" "${LOCK_HEAD_STRAIGHT} (rad=${LOCK_HEAD_STRAIGHT_RAD})"
    printf '\n'

    # Light-touch reachability probe (no daemons touched).
    info "reachability probes:"
    local ping_count=1
    if ping -c "${ping_count}" -W 1 -q "${PC2_HOST}" >/dev/null 2>&1; then
        printf '    %s[ok]%s   ping %s\n' "${C_GREEN}" "${C_RESET}" "${PC2_HOST}"
    else
        printf '    %s[fail]%s ping %s (PC2 unreachable on this network)\n' "${C_RED}" "${C_RESET}" "${PC2_HOST}"
    fi
    if ping -c "${ping_count}" -W 1 -q "${PC1_HOST}" >/dev/null 2>&1; then
        printf '    %s[ok]%s   ping %s (PC1)\n' "${C_GREEN}" "${C_RESET}" "${PC1_HOST}"
    else
        printf '    %s[warn]%s ping %s (PC1 -- expected; PC1 is reached from PC2 over SDK ethernet, not from laptop)\n' "${C_YELLOW}" "${C_RESET}" "${PC1_HOST}"
    fi
}

# -------------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------------

case "${SUBCMD}" in
    start)      cmd_start ;;
    stop)       cmd_stop ;;
    # restart already signals "yes, kill and bring back up" -- adding a
    # second confirmation prompt would just train operators to mash
    # Enter, which defeats the gate we just added on plain ``stop``.
    restart)    STOP_YES=1; cmd_stop && cmd_start ;;
    status)     cmd_status ;;
    logs)       cmd_logs ;;
    attach)     cmd_attach ;;
    postmortem) cmd_postmortem ;;
    print-env)  cmd_print_env ;;
    *) err "unknown subcommand: ${SUBCMD}"; usage ;;
esac
