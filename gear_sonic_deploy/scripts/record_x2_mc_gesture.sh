#!/usr/bin/env bash
# record_x2_mc_gesture.sh -- one-button capture of a single mobile-app-
# triggered MC gesture (wave, handshake, ...) into a SOMA-byte-compatible
# motion_lib PKL.
#
# Reuses the existing pipeline end-to-end:
#
#   x2_record_remote.sh         passive ROS recorder on PC2
#     -> NPZ under gear_sonic/data/motions/x2_recorded/mc_gestures_npz/
#                  <session>/<take>.npz   (tracked via Git LFS)
#   convert_x2_record_to_motion_lib.py  NPZ -> SOMA-schema PKL
#     -> gear_sonic/data/motions/x2_recorded/mc_gestures/<take>.pkl
#                                            (tracked via Git LFS)
#   play_x2_motion_mujoco.py    (optional) MuJoCo kinematic replay
#
# Operator UX
# -----------
#
#   1. Make sure MC is running on PC1 and the mobile app is paired.
#      (See docs/source/getting_started/pc2_jetson_bringup.md for the
#       start_app curl line if MC isn't up yet.)
#   2. Run this script with the gesture name + take number. Don't tap the
#      gesture on the mobile app yet -- the recorder needs ~1 s to attach
#      to the HAL bus first.
#   3. Wait for the green "READY -- trigger the gesture now" banner, then
#      tap the gesture on the mobile app.
#   4. After the gesture completes (robot returns to rest), hit Ctrl-C
#      ONCE in this terminal. The remote recorder finalises the NPZ,
#      rsyncs it back, and the converter runs automatically.
#   5. On success the PKL path is printed in green. Pass --view to also
#      pop a MuJoCo viewer for sanity-check.
#
# Usage
# -----
#
#   ./record_x2_mc_gesture.sh GESTURE TAKE_NUM --pc2-host IP [options]
#
# Examples
# --------
#
#   # First wave take of the day (session auto = mc_gestures_<UTC date>).
#   ./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh wave 001 \
#       --pc2-host 192.168.86.32
#
#   # Handshake take with MuJoCo replay popped at the end.
#   ./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh handshake 002 \
#       --pc2-host 192.168.86.32 --view
#
#   # Capture MC's commanded reference instead of the executed state.
#   ./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh wave 001 \
#       --pc2-host 192.168.86.32 --cmd-source
#
#   # Wider trim window for a slow gesture (3 s before, 2 s after).
#   ./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh dance 001 \
#       --pc2-host 192.168.86.32 --trim-start 3.0 --trim-end 2.0
#
#   # Record only; convert later by re-running with --convert-only.
#   ./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh wave 003 \
#       --pc2-host 192.168.86.32 --no-convert
#
#   ./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh wave 003 \
#       --pc2-host 192.168.86.32 --convert-only

set -uo pipefail

# Defaults
PC2_HOST="${PC2_HOST:-}"
PC2_USER="${PC2_USER:-run}"
SESSION_NAME="${X2_MC_SESSION:-mc_gestures_$(date -u +%Y%m%d)}"
DURATION="0"
SOURCE_KIND="state"
TRIM_START="0.5"
TRIM_END="0.5"
FPS="30"
ROOT_ROT="foot-flat"  # MC gestures are in-place; foot-flat derives
                      # pelvis rotation from leg kinematics so the
                      # anchor foot stays planted at its idle (flat)
                      # orientation. The torso IMU disagrees with the
                      # foot-flat constraint on the hug (IMU reads a
                      # forward tilt that would physically tip the
                      # robot); foot-flat gives the correct backward
                      # counter-balance pitch instead. Override with
                      # --root-rot torso-imu for non-stationary takes
                      # (walking, large COM excursions away from feet).
FLOOR_ANCHOR="lower-foot"
ANCHOR_XY=1  # MC gestures (hug, bow, dance, ...) shift weight; pinning
             # pelvis XY at (0,0) makes the feet visually slide under a
             # locked pelvis. Default ON; --no-anchor-xy to opt out.
DO_CONVERT=1
CONVERT_ONLY=0
DO_VIEW=0
OVERRIDE=0
PYTHON_BIN="${PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# NPZ output root. Lives under the LFS-tracked motion-data tree so
# captures are durable + co-located with the derived PKLs they produce.
# (Previously this was scratch/runs/, which is fine for throwaway deploy
# probes but wrong for MC gesture sources -- those are the irreplaceable
# raw recordings the PKLs are re-derivable from.)
LOCAL_OUT_ROOT="${REPO_ROOT}/gear_sonic/data/motions/x2_recorded/mc_gestures_npz"
PKL_OUT_ROOT="${REPO_ROOT}/gear_sonic/data/motions/x2_recorded/mc_gestures"
REMOTE_HELPER="${SCRIPT_DIR}/x2_record_remote.sh"
CONVERTER_MODULE="gear_sonic.data_process.convert_x2_record_to_motion_lib"
VIEWER_SCRIPT="${REPO_ROOT}/gear_sonic/scripts/play_x2_motion_mujoco.py"

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'
C_BLUE=$'\e[34m'; C_DIM=$'\e[2m'; C_BOLD=$'\e[1m'; C_RESET=$'\e[0m'
log()  { printf '%s[mc_gesture]%s %s\n' "${C_BLUE}" "${C_RESET}" "$*"; }
warn() { printf '%s[mc_gesture WARN]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
err()  { printf '%s[mc_gesture ERROR]%s %s\n' "${C_RED}" "${C_RESET}" "$*" >&2; }

usage() {
    awk '/^# record_x2_mc_gesture.sh/,/^set -uo pipefail/{ sub(/^# ?/, ""); print }' "$0" >&2
    cat >&2 <<'USAGE_TAIL'

Required positional:
  GESTURE                Short name, e.g. wave, handshake, bow, dance.
                         Lowercase, no spaces / slashes.
  TAKE_NUM               Three-digit zero-padded take number, e.g. 001.
                         Free-form is allowed but the convention keeps the
                         on-disk listing sorted.

Required flag (or env):
  --pc2-host HOST        PC2 IP / hostname. Forward to x2_record_remote.sh.
                         Env override: PC2_HOST.

Optional:
  --pc2-user USER        SSH user on PC2 (default: run; env: PC2_USER).
  --session NAME         Session dir name (under
                         gear_sonic/data/motions/x2_recorded/mc_gestures_npz/
                         locally, and PC2's log/ remotely).
                         Default: mc_gestures_<UTC YYYYMMDD>.
                         Env override: X2_MC_SESSION.
  --duration SECS        Fixed recorder duration (default 0 = until Ctrl-C).
  --cmd-source           Convert from MC's commanded angles instead of the
                         executed state. Default: state (sim-to-real
                         training data wants what physically happened).
  --root-rot MODE        identity | torso-imu | torso-imu-raw |
                         foot-flat. Default: foot-flat (derives pelvis
                         rotation from leg kinematics so the anchor
                         foot stays at its idle/flat orientation --
                         physically-correct counter-balance pitch for
                         in-place gestures). Use torso-imu for non-
                         stationary takes.
  --floor-anchor MODE    lower-foot | left-foot | right-foot | none.
                         Default: lower-foot (per-frame pelvis Z from
                         foot FK; XY also tracked unless --no-anchor-xy).
  --no-anchor-xy         Disable the pelvis-XY foot lock (revert to the
                         legacy XY=(0,0) pin). Default: XY lock is ON
                         for MC gestures so weight-shift motions
                         (hug, bow, squat, dance) read correctly --
                         without it the feet appear to slide under a
                         pelvis that's stuck at the origin.
  --fps N                Resample target rate. Default 30 (matches the
                         SOMA bones-seed PKL corpus).
  --trim-start S         Trim seconds at start (default 0.5).
  --trim-end S           Trim seconds at end (default 0.5).
  --no-convert           Record only; skip the converter. PKL won't be
                         produced. Re-run later with --convert-only.
  --convert-only         Skip the recorder; just (re)convert an existing
                         NPZ. Useful for tweaking trim / source flags
                         without re-recording.
  --view                 After conversion, launch play_x2_motion_mujoco.py
                         against the freshly-written PKL.
  --override             Overwrite an existing PKL at the output path.
                         Default: refuse and exit.
  -h, --help             Show this help.

Outputs (both LFS-tracked, durable):
  gear_sonic/data/motions/x2_recorded/mc_gestures_npz/<session>/<take>.npz
                                              (raw HAL recording rsynced
                                               from PC2; source of truth)
  gear_sonic/data/motions/x2_recorded/mc_gestures/<take>.pkl
                                              (single-key motion_lib PKL,
                                               byte-compatible with SOMA;
                                               re-derivable from the NPZ
                                               via --convert-only)
USAGE_TAIL
    exit 1
}

# Arg parsing accepts positionals (GESTURE TAKE_NUM) anywhere in the arg
# list, before or after / interleaved with the optional flags. We collect
# non-flag tokens into POSITIONAL[] and resolve them at the end.
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pc2-host) PC2_HOST="$2"; shift 2 ;;
        --pc2-user) PC2_USER="$2"; shift 2 ;;
        --session) SESSION_NAME="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --cmd-source) SOURCE_KIND="cmd"; shift ;;
        --root-rot) ROOT_ROT="$2"; shift 2 ;;
        --floor-anchor) FLOOR_ANCHOR="$2"; shift 2 ;;
        --no-anchor-xy) ANCHOR_XY=0; shift ;;
        --anchor-xy) ANCHOR_XY=1; shift ;;
        --fps) FPS="$2"; shift 2 ;;
        --trim-start) TRIM_START="$2"; shift 2 ;;
        --trim-end) TRIM_END="$2"; shift 2 ;;
        --no-convert) DO_CONVERT=0; shift ;;
        --convert-only) CONVERT_ONLY=1; shift ;;
        --view) DO_VIEW=1; shift ;;
        --override) OVERRIDE=1; shift ;;
        -h|--help) usage ;;
        --) shift; POSITIONAL+=("$@"); break ;;
        -*) err "unknown flag: $1"; usage ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL[@]} -lt 2 ]]; then
    err "GESTURE and TAKE_NUM are required positional args (got ${#POSITIONAL[@]})."
    usage
fi
if [[ ${#POSITIONAL[@]} -gt 2 ]]; then
    err "too many positional args (got ${#POSITIONAL[@]}: ${POSITIONAL[*]}; expected 2: GESTURE TAKE_NUM)."
    usage
fi
GESTURE="${POSITIONAL[0]}"
TAKE_NUM="${POSITIONAL[1]}"

# Validate names.
case "${GESTURE}" in
    "") err "GESTURE is required"; usage ;;
    */*) err "GESTURE must not contain '/' (got '${GESTURE}')"; exit 1 ;;
    *' '*) err "GESTURE must not contain spaces (got '${GESTURE}')"; exit 1 ;;
esac
case "${TAKE_NUM}" in
    "") err "TAKE_NUM is required"; usage ;;
    */*) err "TAKE_NUM must not contain '/' (got '${TAKE_NUM}')"; exit 1 ;;
    *' '*) err "TAKE_NUM must not contain spaces (got '${TAKE_NUM}')"; exit 1 ;;
esac

if [[ "${CONVERT_ONLY}" -eq 0 && -z "${PC2_HOST}" ]]; then
    err "--pc2-host (or PC2_HOST env) is required unless --convert-only."
    exit 1
fi

if [[ "${DO_CONVERT}" -eq 0 && "${CONVERT_ONLY}" -eq 1 ]]; then
    err "--no-convert and --convert-only are mutually exclusive."
    exit 1
fi

# Sanity-check helper scripts exist before we kick off anything.
if [[ "${CONVERT_ONLY}" -eq 0 && ! -x "${REMOTE_HELPER}" ]]; then
    err "missing or non-executable: ${REMOTE_HELPER}"
    exit 1
fi
if [[ "${DO_CONVERT}" -eq 1 ]]; then
    if [[ ! -d "${REPO_ROOT}/gear_sonic/data_process" ]]; then
        err "converter package not found at ${REPO_ROOT}/gear_sonic/data_process"
        exit 1
    fi
fi
if [[ "${DO_VIEW}" -eq 1 && ! -f "${VIEWER_SCRIPT}" ]]; then
    warn "viewer not found at ${VIEWER_SCRIPT}; --view will be skipped."
    DO_VIEW=0
fi

# Computed paths.
TAKE_NAME="${GESTURE}_${TAKE_NUM}"
LOCAL_SESSION_DIR="${LOCAL_OUT_ROOT}/${SESSION_NAME}"
LOCAL_NPZ="${LOCAL_SESSION_DIR}/${TAKE_NAME}.npz"
PKL_PATH="${PKL_OUT_ROOT}/${TAKE_NAME}.pkl"
TASK_LABEL="MC gesture: ${GESTURE}"

# Refuse to clobber the PKL early -- before doing 10 minutes of recording.
if [[ "${DO_CONVERT}" -eq 1 && -f "${PKL_PATH}" && "${OVERRIDE}" -eq 0 ]]; then
    err "PKL already exists: ${PKL_PATH}"
    err "  pass --override to overwrite, or pick a different TAKE_NUM."
    exit 1
fi

# Recap.
printf '\n'
printf '%s== record_x2_mc_gesture: %s ==%s\n' "${C_BOLD}${C_GREEN}" "${TAKE_NAME}" "${C_RESET}"
printf '  %-18s %s\n' "gesture"        "${GESTURE}"
printf '  %-18s %s\n' "take #"         "${TAKE_NUM}"
printf '  %-18s %s\n' "session"        "${SESSION_NAME}"
if [[ "${CONVERT_ONLY}" -eq 0 ]]; then
    printf '  %-18s %s\n' "PC2"        "${PC2_USER}@${PC2_HOST}"
    printf '  %-18s %s\n' "duration"   "${DURATION}s ($([[ "${DURATION}" = "0" ]] && echo 'until Ctrl-C' || echo 'fixed'))"
fi
printf '  %-18s %s\n' "NPZ (local)"    "${LOCAL_NPZ}"
if [[ "${DO_CONVERT}" -eq 1 ]]; then
    printf '  %-18s %s\n' "PKL out"    "${PKL_PATH}"
    printf '  %-18s source=%s root_rot=%s floor=%s%s fps=%s trim=%s/%s\n' \
        "converter"                    "${SOURCE_KIND}" "${ROOT_ROT}" \
                                        "${FLOOR_ANCHOR}" \
                                        "$([[ "${ANCHOR_XY}" -eq 1 ]] && echo '+xy' || echo '')" \
                                        "${FPS}" \
                                        "${TRIM_START}" "${TRIM_END}"
fi
printf '  %-18s %s\n' "post-view"      "$([[ "${DO_VIEW}" -eq 1 ]] && echo 'ON (MuJoCo)' || echo 'off (--view to enable)')"
printf '\n'

# =============================================================================
# Step 1: record on PC2 (unless --convert-only).
# =============================================================================
if [[ "${CONVERT_ONLY}" -eq 0 ]]; then
    printf '%s[1/2] Recording on PC2.%s\n\n' "${C_BOLD}" "${C_RESET}"

    # Loud, impossible-to-miss WAIT prelude. The recorder takes ~5-10 s to
    # come up on the Jetson (Python startup + ROS imports + FastDDS peer
    # discovery against PC1's HAL endpoints) before any topic data flows.
    # During that quiet window the operator must NOT tap the gesture, or
    # the capture will miss the start of the motion. The READY banner
    # fires the moment the recorder is actually live; THAT is the only
    # valid tap cue.
    C_BG_YELLOW=$'\e[43m'; C_BG_GREEN=$'\e[42m'; C_FG_BLACK=$'\e[30m'
    printf '%s%s%s                                                              %s\n' "${C_BOLD}" "${C_BG_YELLOW}" "${C_FG_BLACK}" "${C_RESET}"
    printf '%s%s%s  ===  DO NOT TAP THE GESTURE YET  ===                        %s\n' "${C_BOLD}" "${C_BG_YELLOW}" "${C_FG_BLACK}" "${C_RESET}"
    printf '%s%s%s                                                              %s\n' "${C_BOLD}" "${C_BG_YELLOW}" "${C_FG_BLACK}" "${C_RESET}"
    printf '%s%s%s  Recorder is starting on PC2 (5-10 s on the Jetson). You    %s\n' "${C_BOLD}" "${C_BG_YELLOW}" "${C_FG_BLACK}" "${C_RESET}"
    printf '%s%s%s  will see ssh-probe + setup lines first, then a BIG GREEN   %s\n' "${C_BOLD}" "${C_BG_YELLOW}" "${C_FG_BLACK}" "${C_RESET}"
    printf '%s%s%s  "READY -- trigger gesture NOW" banner. ONLY THEN tap the   %s\n' "${C_BOLD}" "${C_BG_YELLOW}" "${C_FG_BLACK}" "${C_RESET}"
    printf '%s%s%s  gesture on the mobile app. Hit Ctrl-C ONCE when done.      %s\n' "${C_BOLD}" "${C_BG_YELLOW}" "${C_FG_BLACK}" "${C_RESET}"
    printf '%s%s%s                                                              %s\n\n' "${C_BOLD}" "${C_BG_YELLOW}" "${C_FG_BLACK}" "${C_RESET}"

    # FastDDS discovery emits ~one warning per peer matched/unmatched event.
    # On the X2 with dozens of HAL endpoints this floods the terminal and
    # buries the recorder's own startup/status lines. Filter them out --
    # they're informational and irrelevant to the capture.
    #
    # We also inject a big READY banner the first time the recorder prints
    # either its startup line ("Recording -> ...") or its 1 Hz status line
    # (pattern: `[<elapsed>s] leg:s<N> c<N> ...`, see
    # x2_record_real_run.py:_on_status). The 1 Hz status line only fires
    # once topics are subscribed, which is the operator's go-signal.
    #
    # Why python -u for the filter (and stdbuf on the upstream): we need
    # both line-buffered writes AND line-by-line input reads. gawk's
    # default 4 KB stdin buffering hides the recorder's slow trickle of
    # status lines, so the banner only fires after ~4 KB has accumulated
    # in the pipe (often the entire run). Python with -u and
    # iter(readline, "") drains the pipe one line at a time and flushes
    # immediately. stdbuf on x2_record_remote.sh / ssh forces THEIR
    # stdout to line-buffer so individual lines reach the pipe promptly
    # in the first place.
    if ! command -v stdbuf >/dev/null 2>&1; then
        warn "stdbuf not found (coreutils); recorder output may stall in"
        warn "pipe buffers. Install coreutils for prompt READY banner."
        STDBUF_PREFIX=()
    else
        STDBUF_PREFIX=(stdbuf -oL -eL)
    fi

    # --override covers both ends of the pipeline. The local PKL gate was
    # checked above (pre-flight); now also clear any stale NPZ left on PC2
    # from a previous take with the same name, since x2_record_remote.sh
    # noclobbers on its own (it has no --override knob).
    REMOTE_SESSION_DIR_DEFAULT="/home/run/getsolo/log/${SESSION_NAME}"
    REMOTE_NPZ_DEFAULT="${REMOTE_SESSION_DIR_DEFAULT}/${TAKE_NAME}.npz"
    if [[ "${OVERRIDE}" -eq 1 ]]; then
        log "override: clearing stale local + remote NPZ for ${TAKE_NAME}"
        rm -f "${LOCAL_NPZ}"
        # Best-effort remote delete. Failure is non-fatal -- if the file
        # isn't there, x2_record_remote.sh will proceed anyway; if the ssh
        # itself fails, the downstream ssh probe will surface a clearer
        # error than we can.
        ssh -o BatchMode=yes -o ConnectTimeout=3 \
            "${PC2_USER}@${PC2_HOST}" \
            "rm -f '${REMOTE_NPZ_DEFAULT}'" 2>/dev/null \
            || warn "could not clear remote NPZ (continuing anyway): ${PC2_USER}@${PC2_HOST}:${REMOTE_NPZ_DEFAULT}"
    fi

    # Visible checkpoint right before the pipe -- if the READY banner ever
    # scrolls off, the operator can search backwards for this line and the
    # banner will be just below it.
    printf '%s>>> launching recorder on PC2; READY banner appears in ~5-10 s <<<%s\n\n' "${C_DIM}" "${C_RESET}"

    # Defer SIGINT in this shell so x2_record_remote.sh + ssh -t can finalize
    # cleanly. Without this, bash sends SIGINT to the script too and we'd
    # exit before the rsync finishes.
    trap '' INT
    "${STDBUF_PREFIX[@]}" "${REMOTE_HELPER}" "${TAKE_NAME}" \
        --pc2-host "${PC2_HOST}" \
        --pc2-user "${PC2_USER}" \
        --session-name "${SESSION_NAME}" \
        --task "${TASK_LABEL}" \
        --duration "${DURATION}" \
        --local-out-root "${LOCAL_OUT_ROOT}" 2>&1 \
      | "${PYTHON_BIN}" -u -c "$(cat <<'PYFILTER'
import re, sys

RTPS_RE = re.compile(
    r"\[Warning\] \[\d+\] \[(?:RTPS_|SECURITY|DOMAIN|TRANSPORT|XML|PARTICIPANT|DISCOVERY)"
)
READY_RE = re.compile(r"^Recording -> |^\[ *\d+\.\d+s\] ")
banner_done = False

# iter(readline, "") drains the pipe one line at a time; we never touch
# the file iterator's internal 8 KB read-ahead buffer that `for line in
# sys.stdin` would use, so each line is flushed as soon as it lands.
for line in iter(sys.stdin.readline, ""):
    if RTPS_RE.search(line):
        continue
    if not banner_done and READY_RE.search(line):
        banner_done = True
        sys.stdout.write("\n\x1b[1;32m============================================================\x1b[0m\n")
        sys.stdout.write("\x1b[1;32m  READY -- trigger the gesture on the MOBILE APP now\x1b[0m\n")
        sys.stdout.write("\x1b[1;32m  Hit Ctrl-C ONCE after the robot returns to rest\x1b[0m\n")
        sys.stdout.write("\x1b[1;32m============================================================\x1b[0m\n\n")
        sys.stdout.flush()
    sys.stdout.write(line)
    sys.stdout.flush()
PYFILTER
)"
    REC_RC=${PIPESTATUS[0]}
    trap - INT

    if [[ "${REC_RC}" -ne 0 ]]; then
        err "x2_record_remote.sh exited rc=${REC_RC}; aborting before convert."
        exit "${REC_RC}"
    fi
fi

# =============================================================================
# Step 2: convert NPZ -> PKL (unless --no-convert).
# =============================================================================
if [[ "${DO_CONVERT}" -eq 0 ]]; then
    printf '\n%sRecording complete. --no-convert specified; skipping conversion.%s\n' "${C_DIM}" "${C_RESET}"
    printf '  NPZ at: %s\n' "${LOCAL_NPZ}"
    printf '  Re-run with --convert-only when ready.\n'
    exit 0
fi

if [[ ! -f "${LOCAL_NPZ}" ]]; then
    err "expected NPZ not found: ${LOCAL_NPZ}"
    err "  Was the recorder interrupted before the rsync? Check"
    err "  gear_sonic/data/motions/x2_recorded/mc_gestures_npz/${SESSION_NAME}/"
    err "  or the PC2 log dir."
    exit 1
fi

mkdir -p "${PKL_OUT_ROOT}"

# Re-check overwrite gate (may have changed since pre-flight if user
# concurrently produced the file; cheap belt+braces).
if [[ -f "${PKL_PATH}" && "${OVERRIDE}" -eq 0 ]]; then
    err "PKL already exists: ${PKL_PATH}"
    err "  pass --override to overwrite."
    exit 1
fi

printf '\n%s[2/2] Converting NPZ -> PKL%s\n' "${C_BOLD}" "${C_RESET}"

# Runs the converter, captures combined stdout+stderr (so we can sniff
# for the scipy euler-shape bug), and tees the output to the terminal so
# the operator sees it in real time. We can't use bash process
# substitution + pipefail cleanly here without losing the rc, so we use
# a tempfile pattern.
run_converter() {
    local _rot="$1"
    local _tmp
    _tmp=$(mktemp)
    local _xy_args=()
    if [[ "${ANCHOR_XY}" -eq 1 ]]; then
        _xy_args=(--anchor-xy)
    fi
    (
        cd "${REPO_ROOT}"
        "${PYTHON_BIN}" -m "${CONVERTER_MODULE}" \
            "${LOCAL_NPZ}" \
            --output "${PKL_PATH}" \
            --source "${SOURCE_KIND}" \
            --root-rot "${_rot}" \
            --floor-anchor "${FLOOR_ANCHOR}" \
            "${_xy_args[@]}" \
            --fps "${FPS}" \
            --trim-start "${TRIM_START}" \
            --trim-end "${TRIM_END}" \
            --name "${TAKE_NAME}" 2>&1 | tee "${_tmp}"
    )
    local _rc=${PIPESTATUS[0]}
    CONV_LOG=$(cat "${_tmp}")
    rm -f "${_tmp}"
    return ${_rc}
}

log "${PYTHON_BIN} -m ${CONVERTER_MODULE} ${LOCAL_NPZ} -> ${PKL_PATH} (root-rot=${ROOT_ROT})"
run_converter "${ROOT_ROT}"
CONV_RC=$?

# Auto-fallback for the known scipy compat bug in the torso-imu chain:
# `Rotation.from_euler("z", waist_yaw)` with waist_yaw.shape=(T,) raises
# "Expected last dimension of `angles` to match number of sequence axes
# specified, got <T>" on scipy versions that require an explicit (T, 1)
# shape. Identity rotation drops pelvis pitch/roll fidelity (acceptable
# for in-place gestures) but produces a valid PKL.
if [[ "${CONV_RC}" -ne 0 && "${ROOT_ROT}" == torso-imu* ]] \
        && grep -q "Expected last dimension of .angles." <<<"${CONV_LOG}"; then
    warn "torso-imu hit a scipy euler shape mismatch in the converter."
    warn "Retrying with --root-rot identity (pelvis pitch/roll is dropped;"
    warn "arm + waist joint motion is preserved -- fine for standing gestures)."
    warn "Follow-up: patch convert_x2_record_to_motion_lib.py _pelvis_rot_from_torso_imu"
    warn "to pass waist_yaw[:, None] etc. to Rotation.from_euler."
    log "${PYTHON_BIN} -m ${CONVERTER_MODULE} ${LOCAL_NPZ} -> ${PKL_PATH} (root-rot=identity, fallback)"
    run_converter "identity"
    CONV_RC=$?
fi

if [[ "${CONV_RC}" -ne 0 ]]; then
    err "converter failed (rc=${CONV_RC}). NPZ kept at ${LOCAL_NPZ}."
    exit "${CONV_RC}"
fi

if [[ ! -f "${PKL_PATH}" ]]; then
    err "converter reported success but PKL is missing: ${PKL_PATH}"
    exit 1
fi

printf '\n%ssaved -> %s%s\n' "${C_GREEN}" "${PKL_PATH}" "${C_RESET}"

# =============================================================================
# Step 3: optional kinematic replay for sanity-check.
# =============================================================================
if [[ "${DO_VIEW}" -eq 1 ]]; then
    printf '\n%sLaunching MuJoCo viewer for %s ...%s\n' "${C_DIM}" "${TAKE_NAME}" "${C_RESET}"
    printf '%s  (Close the viewer window or Ctrl-C here to exit.)%s\n' "${C_DIM}" "${C_RESET}"
    (
        cd "${REPO_ROOT}"
        "${PYTHON_BIN}" "${VIEWER_SCRIPT}" \
            --motion "${PKL_PATH}" \
            --motion-key "${TAKE_NAME}"
    )
fi

exit 0
