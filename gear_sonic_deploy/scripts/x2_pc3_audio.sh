#!/usr/bin/env bash
# x2_pc3_audio.sh -- mute / unmute / inspect the speakers on the X2's PC3
# (interaction unit, default 10.0.1.42).
#
# PC3 ships with a pure ALSA stack (no PulseAudio / PipeWire) and the
# real onboard audio is a Rockchip ES8388 codec exposed as ALSA card 1
# (card 0 is the Loopback virtual card, card 2 is the ES7210 mic
# array). The Speaker control is a `pswitch` (mute toggle, no volume),
# Headphone is a separate pswitch, and PCM is the playback volume.
#
# Usage:
#
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh                # mute (default)
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh mute
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh unmute
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh status
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh volume 60      # set PCM%
#
# Flags (all subcommands):
#
#   --host HOST        PC3 hostname/IP. Default: ${X2_PC3_HOST:-10.0.1.42}.
#   --user USER        SSH user. Default: ${X2_PC3_USER:-agi}.
#   --password PASS    Password for sshpass / sudo. Default: env
#                      X2_PC3_PASSWORD or "1" (the X2 SDK convention,
#                      same as gear_sonic_deploy/scripts/x2_mc_input_source_probe.sh).
#                      Pass --password "" to force key-only auth.
#   --pc2-host HOST    If set, tunnel through PC2 at HOST (e.g. a WiFi IP
#                      like 192.168.86.32) instead of going direct to PC3.
#                      Needed when the wired SDK link to PC3 is down but
#                      PC2 is reachable on another network and PC2 in
#                      turn can reach PC3 over the robot's internal net.
#                      The PC2 leg reuses --user and --password.
#                      Default: env X2_PC2_HOST or empty (= direct).
#   --card N           ALSA card index for the ES8388 codec. Default: 1.
#   --no-persist       Skip `alsactl store` (mute/volume becomes in-memory
#                      only and is lost on reboot).
#   -h, --help         Show this help.
#
# Examples:
#
#   # One-shot mute before a noisy demo:
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh mute
#
#   # Bring it back to ~60% PCM and unmuted:
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh unmute
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh volume 60
#
#   # Just check what state PC3 is in:
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh status
#
#   # When the wired SDK cable is unplugged and PC3 is only reachable
#   # via PC2 over WiFi:
#   ./gear_sonic_deploy/scripts/x2_pc3_audio.sh volume 70 \
#       --pc2-host 192.168.86.32
#
# Exit status: 0 on success, non-zero on SSH/ALSA failure.

set -euo pipefail

NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'

# ────────────────────────────────────────────────────────────────────────
# Defaults / CLI parsing
# ────────────────────────────────────────────────────────────────────────
HOST="${X2_PC3_HOST:-10.0.1.42}"
USER_NAME="${X2_PC3_USER:-agi}"
PASSWORD="${X2_PC3_PASSWORD-1}"   # `-1` so explicit empty means "no password"
PC2_HOST="${X2_PC2_HOST:-}"       # if set, route PC3 ssh through PC2 (bastion)
CARD="${X2_PC3_CARD:-1}"
NO_PERSIST=0
ACTION=""
VOLUME_PCT=""

print_help() {
    sed -n '/^# x2_pc3_audio.sh/,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '$d' -e 's/^# \{0,1\}//'
}

while (( $# > 0 )); do
    case "$1" in
        mute|unmute|status)
            ACTION="$1"; shift ;;
        volume)
            ACTION="volume"; shift
            if (( $# == 0 )); then
                echo "[x2_pc3_audio] 'volume' requires a percentage (0-100)" >&2
                exit 2
            fi
            VOLUME_PCT="$1"; shift ;;
        --host)        HOST="$2"; shift 2 ;;
        --user)        USER_NAME="$2"; shift 2 ;;
        --password)    PASSWORD="$2"; shift 2 ;;
        --pc2-host)    PC2_HOST="$2"; shift 2 ;;
        --card)        CARD="$2"; shift 2 ;;
        --no-persist)  NO_PERSIST=1; shift ;;
        -h|--help)     print_help; exit 0 ;;
        *)
            echo "[x2_pc3_audio] unknown arg: $1" >&2
            print_help >&2
            exit 2 ;;
    esac
done

ACTION="${ACTION:-mute}"

if [[ "$ACTION" == "volume" ]]; then
    if ! [[ "$VOLUME_PCT" =~ ^[0-9]+$ ]] || (( VOLUME_PCT < 0 || VOLUME_PCT > 100 )); then
        echo "[x2_pc3_audio] volume must be an integer 0..100, got: $VOLUME_PCT" >&2
        exit 2
    fi
fi

# ────────────────────────────────────────────────────────────────────────
# SSH helpers
# ────────────────────────────────────────────────────────────────────────
SSH_OPTS=(
    -o ConnectTimeout=5
    -o StrictHostKeyChecking=accept-new
    -o LogLevel=ERROR
)

# Pick auth method: if PASSWORD is set, try password auth via sshpass; else key.
USE_SSHPASS=0
if [[ -n "$PASSWORD" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
        echo -e "${YELLOW}[x2_pc3_audio] sshpass not installed; falling back to key auth.${NC}" >&2
        echo -e "${YELLOW}                         Install with: sudo apt-get install -y sshpass${NC}" >&2
    else
        USE_SSHPASS=1
        SSH_OPTS+=( -o PreferredAuthentications=password -o PubkeyAuthentication=no )
    fi
fi

run_remote() {
    # Run the bash snippet (stdin from caller) on PC3, passing "$@" as
    # positional args. When --pc2-host is set we tunnel via PC2 by
    # nesting a second sshpass+ssh inside the outer hop's command --
    # OpenSSH's ProxyJump is cleaner but only one leg can be password-
    # authed via sshpass, so we hand-roll the two-hop here.
    if (( USE_SSHPASS )); then
        if [[ -n "$PC2_HOST" ]]; then
            local inner_opts="-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o PreferredAuthentications=password -o PubkeyAuthentication=no"
            local hop_cmd="sshpass -p '${PASSWORD//\'/\'\\\'\'}' ssh ${inner_opts} ${USER_NAME}@${HOST} bash -s"
            local arg
            for arg in "$@"; do
                hop_cmd+=" $(printf '%q' "$arg")"
            done
            sshpass -p "$PASSWORD" ssh "${SSH_OPTS[@]}" "${USER_NAME}@${PC2_HOST}" "$hop_cmd"
        else
            sshpass -p "$PASSWORD" ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "bash -s" "$@"
        fi
    else
        local jump_opt=()
        [[ -n "$PC2_HOST" ]] && jump_opt+=( -o "ProxyJump=${USER_NAME}@${PC2_HOST}" )
        ssh "${SSH_OPTS[@]}" "${jump_opt[@]}" "${USER_NAME}@${HOST}" "bash -s" "$@"
    fi
}

# ────────────────────────────────────────────────────────────────────────
# Remote payloads (driven from stdin so we don't have to escape twice)
# ────────────────────────────────────────────────────────────────────────
remote_status() {
    run_remote "$CARD" <<'REMOTE'
set -e
CARD="$1"
echo "=== /proc/asound/cards ==="
cat /proc/asound/cards
echo
echo "=== card $CARD Speaker ==="
amixer -c "$CARD" sget Speaker  2>/dev/null | tail -5 || echo "(no Speaker control on card $CARD)"
echo "=== card $CARD Headphone ==="
amixer -c "$CARD" sget Headphone 2>/dev/null | tail -5 || echo "(no Headphone control on card $CARD)"
echo "=== card $CARD PCM ==="
amixer -c "$CARD" sget PCM 2>/dev/null | tail -5 || echo "(no PCM control on card $CARD)"
REMOTE
}

remote_set() {
    # $1 = "mute" | "unmute" | "volume:NN"
    local mode="$1"
    run_remote "$CARD" "$mode" "$NO_PERSIST" "$PASSWORD" <<'REMOTE'
set -e
CARD="$1"; MODE="$2"; NO_PERSIST="$3"; PASS="$4"

case "$MODE" in
    mute)
        amixer -c "$CARD" sset Speaker  mute   >/dev/null
        amixer -c "$CARD" sset Headphone mute  >/dev/null
        # Also drop PCM to 0% so even an unmuted output upstream is silent.
        amixer -c "$CARD" sset PCM 0% >/dev/null 2>&1 || true
        ;;
    unmute)
        amixer -c "$CARD" sset Speaker  unmute >/dev/null
        # Leave Headphone alone so unplugged-headphone behaviour stays sane;
        # if the operator wants headphones on, run `unmute` then plug them in
        # and run `volume <pct>` (Headphone tracks PCM on this codec).
        # Restore a sensible PCM level if the previous state was 0%.
        cur_pcm="$(amixer -c "$CARD" sget PCM 2>/dev/null \
                    | awk -F'[][]' '/Front Left:|Mono:/ {print $2; exit}' \
                    | tr -d '%')"
        if [[ -z "$cur_pcm" || "$cur_pcm" == "0" ]]; then
            amixer -c "$CARD" sset PCM 80% >/dev/null 2>&1 || true
        fi
        ;;
    volume:*)
        pct="${MODE#volume:}"
        amixer -c "$CARD" sset PCM "${pct}%" >/dev/null
        ;;
    *)
        echo "[remote] unknown mode: $MODE" >&2; exit 2 ;;
esac

if [[ "$NO_PERSIST" != "1" ]]; then
    if [[ -n "$PASS" ]]; then
        echo "$PASS" | sudo -S -p "" alsactl store "$CARD" 2>/dev/null \
            && echo "[x2_pc3_audio:remote] persisted via alsactl store $CARD" \
            || echo "[x2_pc3_audio:remote] WARN: alsactl store $CARD failed (in-memory only)"
    else
        sudo -n alsactl store "$CARD" 2>/dev/null \
            && echo "[x2_pc3_audio:remote] persisted via alsactl store $CARD" \
            || echo "[x2_pc3_audio:remote] WARN: passwordless sudo unavailable; mute is in-memory only"
    fi
fi

# Echo final state so the caller can verify.
echo "=== card $CARD final state ==="
amixer -c "$CARD" sget Speaker  2>/dev/null | tail -4 || true
amixer -c "$CARD" sget Headphone 2>/dev/null | tail -4 || true
amixer -c "$CARD" sget PCM       2>/dev/null | tail -4 || true
REMOTE
}

# ────────────────────────────────────────────────────────────────────────
# Dispatch
# ────────────────────────────────────────────────────────────────────────
if [[ -n "$PC2_HOST" ]]; then
    echo -e "${BLUE}[x2_pc3_audio]${NC} ${BOLD}${ACTION}${NC} on ${USER_NAME}@${HOST} via ${USER_NAME}@${PC2_HOST} (card ${CARD})"
else
    echo -e "${BLUE}[x2_pc3_audio]${NC} ${BOLD}${ACTION}${NC} on ${USER_NAME}@${HOST} (card ${CARD})"
fi

case "$ACTION" in
    status)
        remote_status ;;
    mute)
        remote_set mute ;;
    unmute)
        remote_set unmute ;;
    volume)
        remote_set "volume:${VOLUME_PCT}" ;;
    *)
        echo "[x2_pc3_audio] internal error: unhandled action '$ACTION'" >&2
        exit 2 ;;
esac

echo -e "${GREEN}[x2_pc3_audio] done.${NC}"
