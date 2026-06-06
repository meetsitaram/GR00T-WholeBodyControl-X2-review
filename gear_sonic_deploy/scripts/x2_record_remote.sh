#!/usr/bin/env bash
# x2_record_remote.sh -- record an X2 dataset capture on PC2 over SSH.
#
# Sibling to x2_record.sh (which auto-relaunches inside the docker_x2/x2sim
# container on the laptop). This one instead SSHes into PC2 and runs
# x2_record_real_run.py natively on the robot's Jetson, then optionally
# rsyncs the .npz back to the laptop. Use this when:
#
#   * The laptop doesn't have a working GPU driver / docker_x2 container
#     (e.g. fresh kernel + nvidia DKMS not rebuilt yet).
#   * You want the recorder process to survive a laptop wifi blip.
#   * You're driving the robot through stock MC routines (stand from lying,
#     sit, walk, dance, etc.) and don't need any deploy / Quest3 stack -- the
#     recorder is a pure passive ROS subscriber on AIMDK joint + IMU buses.
#
# What it captures (per x2_record_real_run.py):
#   /aima/hal/joint/{leg,waist,arm,head}/{state,command}  -- 31 dof body
#   /aima/hal/imu/torso/state                             -- IMU
#   GetMcAction polled at 5 Hz                            -- MC mode timeline
#
# Hand finger joints are intentionally NOT captured.
#
# Usage
# -----
#
#   ./x2_record_remote.sh TAKE_NAME --pc2-host IP [options]
#
# Examples
# --------
#
#   # Run until Ctrl-C, rsync back automatically.
#   ./gear_sonic_deploy/scripts/x2_record_remote.sh \
#       02_stand_from_lay_take1 \
#       --pc2-host 192.168.86.26 \
#       --task "MC stand-from-lay routine, gantry-supported, take 1"
#
#   # Fixed 30 s window.
#   ./gear_sonic_deploy/scripts/x2_record_remote.sh \
#       05_walk_fwd_take1 \
#       --pc2-host 192.168.86.26 \
#       --duration 30 \
#       --task "walking forward 5 steps"
#
#   # Keep file on PC2 only (no rsync back).
#   ./gear_sonic_deploy/scripts/x2_record_remote.sh \
#       99_long_burnin --pc2-host 192.168.86.26 --no-rsync --duration 0
#
# To finalize an in-progress recording cleanly: hit Ctrl-C ONCE in this
# terminal. ssh -t forwards SIGINT to the python process on PC2, which
# catches it, flushes the .npz, prints the summary, and exits. Then the
# rsync runs (unless --no-rsync).

set -uo pipefail

# Defaults
PC2_HOST=""
PC2_USER="${PC2_USER:-run}"
PC2_PREFIX="${PC2_PREFIX:-/home/run/getsolo}"
SESSION_NAME="${X2_SESSION_NAME:-move_library_$(date -u +%Y%m%d)}"
DURATION="0"
TASK=""
NO_RSYNC=0
NO_TRACK_MC=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_OUT_ROOT="${REPO_ROOT}/scratch/runs"

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_BLUE=$'\e[34m'; C_DIM=$'\e[2m'; C_RESET=$'\e[0m'
log()  { printf '%s[record_remote]%s %s\n' "${C_BLUE}" "${C_RESET}" "$*"; }
warn() { printf '%s[record_remote WARN]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
err()  { printf '%s[record_remote ERROR]%s %s\n' "${C_RED}" "${C_RESET}" "$*" >&2; }

usage() {
    awk '/^# x2_record_remote.sh/,/^set -uo pipefail/{ sub(/^# ?/, ""); print }' "$0" >&2
    cat >&2 <<'USAGE_TAIL'

Required:
  TAKE_NAME              Take filename (without .npz), e.g.
                         01_stand_default_baseline or 02_stand_from_lay_take1
                         No slashes or spaces.
  --pc2-host HOST        PC2 IP / hostname. The wifi DHCP lease changes
                         session-to-session, so we deliberately require this
                         every invocation. (Check it with:
                             arp -a | grep -v incomplete
                         or `ssh run@<ip> hostname` if you know the candidate.)

Options:
  --task "text"          Free-form task description saved into the .npz
                         meta_json (stored as meta.note in the file --
                         the underlying recorder API still calls it `note`,
                         but the wrapper exposes it as `--task` to match
                         LeRobot dataset-collection vocabulary). Highly
                         recommended -- this is your only sidecar label.
  --duration SECS        Fixed duration in seconds (0 = until Ctrl-C).
                         Default: 0.
  --pc2-user USER        SSH user on PC2. Default: run.
  --pc2-prefix DIR       PC2 install prefix that contains gear_sonic_deploy/
                         and log/. Default: /home/run/getsolo.
  --session-name NAME    Subdir under PC2 ${prefix}/log/ AND under
                         <repo>/scratch/runs/. Default:
                         move_library_<UTC YYYYMMDD>.
  --no-rsync             Skip pulling the .npz back to the laptop.
  --no-track-mc-mode     Skip polling GetMcAction (saves 5 Hz of cross-host
                         service traffic; you lose the MC mode timeline).
  --local-out-root DIR   Where to land rsynced files. Default:
                         <repo>/scratch/runs/.
  -h, --help             Show this help.

Environment overrides (CLI wins):
  PC2_USER, PC2_PREFIX, X2_SESSION_NAME, ROS_DOMAIN_ID, ROS_LOCALHOST_ONLY
USAGE_TAIL
    exit 1
}

if [[ $# -lt 1 ]]; then usage; fi
case "$1" in -h|--help) usage ;; esac

TAKE="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pc2-host) PC2_HOST="$2"; shift 2 ;;
        --pc2-user) PC2_USER="$2"; shift 2 ;;
        --pc2-prefix) PC2_PREFIX="$2"; shift 2 ;;
        --session-name) SESSION_NAME="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --no-rsync) NO_RSYNC=1; shift ;;
        --no-track-mc-mode) NO_TRACK_MC=1; shift ;;
        --local-out-root) LOCAL_OUT_ROOT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) err "unknown argument: $1"; usage ;;
    esac
done

# Validate TAKE.
TAKE="${TAKE%.npz}"
case "$TAKE" in
    */*) err "TAKE_NAME must not contain '/' (got '${TAKE}')"; exit 1 ;;
    *' '*) err "TAKE_NAME must not contain spaces (got '${TAKE}')"; exit 1 ;;
    "") err "TAKE_NAME is required"; usage ;;
esac

if [[ -z "${PC2_HOST}" ]]; then
    err "--pc2-host is required (DHCP-assigned, varies per session)."
    exit 1
fi

if ! [[ "${DURATION}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    err "--duration must be a number (got '${DURATION}')."
    exit 1
fi

# Reachability + key check before we go any further.
log "ssh probe ${PC2_USER}@${PC2_HOST} ..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=3 \
        -o StrictHostKeyChecking=accept-new \
        "${PC2_USER}@${PC2_HOST}" true 2>/dev/null; then
    err "cannot ssh to ${PC2_USER}@${PC2_HOST} non-interactively."
    err "  fix: ssh-copy-id ${PC2_USER}@${PC2_HOST}"
    exit 1
fi

REMOTE_SESSION_DIR="${PC2_PREFIX}/log/${SESSION_NAME}"
REMOTE_NPZ="${REMOTE_SESSION_DIR}/${TAKE}.npz"
REMOTE_RECORDER="${PC2_PREFIX}/gear_sonic_deploy/scripts/x2_record_real_run.py"

# Stage the recorder script if PC2 doesn't already have it. pc2_bringup.sh
# does not include x2_record_real_run.py in its manifest as of 2026-05-26,
# so we ship it lazily.
if ! ssh "${PC2_USER}@${PC2_HOST}" "test -x '${REMOTE_RECORDER}'" 2>/dev/null; then
    log "recorder missing on PC2; scp'ing ${SCRIPT_DIR}/x2_record_real_run.py -> ${REMOTE_RECORDER}"
    if ! ssh "${PC2_USER}@${PC2_HOST}" "mkdir -p '$(dirname "${REMOTE_RECORDER}")'"; then
        err "could not create $(dirname "${REMOTE_RECORDER}") on PC2"
        exit 1
    fi
    if ! scp "${SCRIPT_DIR}/x2_record_real_run.py" \
            "${PC2_USER}@${PC2_HOST}:${REMOTE_RECORDER}" >/dev/null; then
        err "scp of recorder failed"
        exit 1
    fi
    ssh "${PC2_USER}@${PC2_HOST}" "chmod +x '${REMOTE_RECORDER}'"
fi

# Refuse to clobber an existing take by default. The Recorder doesn't
# noclobber on its own, so if you accidentally re-use a TAKE_NAME it'd
# silently overwrite. Tell the operator and bail.
if ssh "${PC2_USER}@${PC2_HOST}" "test -f '${REMOTE_NPZ}'" 2>/dev/null; then
    err "remote .npz already exists: ${REMOTE_NPZ}"
    err "  pick a new TAKE_NAME, or delete the file first:"
    err "    ssh ${PC2_USER}@${PC2_HOST} rm '${REMOTE_NPZ}'"
    exit 1
fi

# Make the session dir.
ssh "${PC2_USER}@${PC2_HOST}" "mkdir -p '${REMOTE_SESSION_DIR}'"

# Build the recorder flag set.
TRACK_FLAG="--track-mc-mode"
[[ "${NO_TRACK_MC}" -eq 1 ]] && TRACK_FLAG="--no-track-mc-mode"

# Escape the task text for one round of remote shell parsing. printf %q
# gives us something safe to drop into a single-shell-parse context (which
# is what ssh's remote command is). We pass it to the recorder's --note
# flag because that's the underlying API; the wrapper exposes it as --task
# to match LeRobot dataset-collection vocabulary.
TASK_QUOTED=$(printf '%q' "${TASK}")

# Recap before we dive in.
printf '\n'
printf '%s== record_remote: starting take ==%s\n' "${C_GREEN}" "${C_RESET}"
printf '  %-20s %s\n' "PC2"           "${PC2_USER}@${PC2_HOST}"
printf '  %-20s %s\n' "session dir"   "${SESSION_NAME}"
printf '  %-20s %s\n' "take name"     "${TAKE}"
printf '  %-20s %s\n' "remote npz"    "${REMOTE_NPZ}"
printf '  %-20s %s\n' "duration"      "${DURATION}s ($([[ "${DURATION}" = "0" ]] && echo 'until Ctrl-C' || echo 'fixed'))"
printf '  %-20s %s\n' "track MC mode" "$([[ "${NO_TRACK_MC}" -eq 1 ]] && echo 'OFF' || echo 'ON (5 Hz)')"
printf '  %-20s %s\n' "rsync back"    "$([[ "${NO_RSYNC}" -eq 1 ]] && echo 'OFF' || echo "ON -> ${LOCAL_OUT_ROOT}/${SESSION_NAME}/")"
printf '  %-20s %s\n' "task"          "${TASK:-(none)}"
printf '\n'
printf '%sHit Ctrl-C ONCE to finalize the recording cleanly.%s\n' "${C_DIM}" "${C_RESET}"
printf '\n'

# Run the recorder on PC2.
#
# Why ssh -t: we want Ctrl-C in this terminal to land as SIGINT on the
# python process on PC2 so the recorder's signal handler can flush the
# buffers, write the .npz, and print the summary. Without -t, ssh just
# eats the Ctrl-C locally and the remote process is left orphaned (which
# eventually exits cleanly via its --duration deadline, but only if you
# set one).
#
# Quoting: the remote shell sees a single string. Local expansion fills
# in ${REMOTE_RECORDER} / ${REMOTE_NPZ} / ${DURATION} / ${TRACK_FLAG} /
# ${TASK_QUOTED}; literal \${ envs expand on the remote side at run time
# (with empty defaults so unset PATH-like vars don't trip set -u).
ssh -t "${PC2_USER}@${PC2_HOST}" "
    source /opt/ros/humble/setup.bash \
    && export AMENT_PREFIX_PATH=/agibot/software/housekeeper/bin/aimdk_msgs:\${AMENT_PREFIX_PATH:-} \
    && export LD_LIBRARY_PATH=/agibot/software/housekeeper/bin/aimdk_msgs/lib:\${LD_LIBRARY_PATH:-} \
    && export PYTHONPATH=/agibot/software/housekeeper/bin/aimdk_msgs/local/lib/python3.10/dist-packages:\${PYTHONPATH:-} \
    && export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0} \
    && export ROS_LOCALHOST_ONLY=\${ROS_LOCALHOST_ONLY:-0} \
    && exec python3 '${REMOTE_RECORDER}' \
        --out '${REMOTE_NPZ}' \
        --duration '${DURATION}' \
        ${TRACK_FLAG} \
        --note ${TASK_QUOTED}
" || REMOTE_RC=$?
REMOTE_RC="${REMOTE_RC:-0}"

if [[ "${REMOTE_RC}" -ne 0 && "${REMOTE_RC}" -ne 130 ]]; then
    # 130 = SIGINT from Ctrl-C, expected. Anything else is suspect.
    warn "remote recorder exited rc=${REMOTE_RC}; file may or may not exist."
fi

# Confirm the file exists on PC2 before we try to rsync.
if ! ssh "${PC2_USER}@${PC2_HOST}" "test -f '${REMOTE_NPZ}'" 2>/dev/null; then
    err "no .npz produced at ${REMOTE_NPZ}."
    err "  Check the recorder's stderr above. Common causes: aimdk_msgs"
    err "  import error (-> ssh in and re-source env), MC service unreachable"
    err "  (-> check ros2 service list), or Ctrl-C'd before any messages landed."
    exit 1
fi

# Quick size check so the operator knows whether the take is meaningful.
REMOTE_SIZE=$(ssh "${PC2_USER}@${PC2_HOST}" "stat -c %s '${REMOTE_NPZ}'" 2>/dev/null || echo 0)
log "remote file size: $(numfmt --to=iec-i --suffix=B "${REMOTE_SIZE}" 2>/dev/null || echo "${REMOTE_SIZE} B")"

# Rsync back.
if [[ "${NO_RSYNC}" -eq 0 ]]; then
    LOCAL_SESSION_DIR="${LOCAL_OUT_ROOT}/${SESSION_NAME}"
    mkdir -p "${LOCAL_SESSION_DIR}"
    log "rsync ${PC2_USER}@${PC2_HOST}:${REMOTE_NPZ} -> ${LOCAL_SESSION_DIR}/"
    if rsync -a --inplace \
            "${PC2_USER}@${PC2_HOST}:${REMOTE_NPZ}" \
            "${LOCAL_SESSION_DIR}/"; then
        LOCAL_NPZ="${LOCAL_SESSION_DIR}/${TAKE}.npz"
        printf '\n%ssaved -> %s%s\n' "${C_GREEN}" "${LOCAL_NPZ}" "${C_RESET}"
    else
        warn "rsync failed; file remains at ${PC2_USER}@${PC2_HOST}:${REMOTE_NPZ}"
        exit 1
    fi
else
    printf '\n%sfile remains on PC2:%s %s@%s:%s\n' \
        "${C_GREEN}" "${C_RESET}" "${PC2_USER}" "${PC2_HOST}" "${REMOTE_NPZ}"
fi
