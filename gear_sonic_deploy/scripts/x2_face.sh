#!/usr/bin/env bash
# x2_face.sh -- take over the X2's face display, or hand it back.
#
# Designed to be called from PC2 (e.g. from the sonic start daemon or a pad
# chord). Pure curl against the flutter-pi face UI on PC3 -- no Python, no ssh,
# no state file.
#
#   ./x2_face.sh on                 # loop the partner-logo reel
#   ./x2_face.sh off                # hand back to the robot's own idle face
#   ./x2_face.sh toggle             # flip, based on what is ACTUALLY playing
#   ./x2_face.sh status             # what is on screen right now
#   ./x2_face.sh media <path>       # loop an arbitrary mp4 (path is ON PC3)
#
# NOTHING IS SHUT DOWN. The face UI arbitrates by priority: a request below the
# currently-playing priority is silently rejected (code 1100). We post above the
# vendor's levels to take over, and PlayDefaultEmoji to give it back. If this
# script never runs, or dies midway, the robot's normal face is unaffected.
#
# Env:
#   X2_PC3_HOST   PC3 address            (default 10.0.1.42)
#   X2_FACE_MEDIA default mp4 for `on`   (default the logo reel)
#   X2_FACE_PRIO  takeover priority      (default 100; MUST exceed 60, which is
#                                         what TTS raises the face to while the
#                                         robot is speaking)
set -uo pipefail

HOST="${X2_PC3_HOST:-10.0.1.42}"
MEDIA="${X2_FACE_MEDIA:-/opt/x2_interact/media/logos_loop.mp4}"
PRIO="${X2_FACE_PRIO:-100}"
BASE="http://${HOST}:18080"
CURL=(curl -s --max-time 5)
MODE_LOOP=2

# Marks our content apart from vendor assets (which live under /agibot/...).
OURS_MATCH="/opt/x2_interact/"

die() { echo "[x2_face] $*" >&2; exit 1; }

now_playing() {   # -> e_path, empty on failure
    "${CURL[@]}" "${BASE}/status" 2>/dev/null \
        | sed -n 's/.*"e_path":"\([^"]*\)".*/\1/p'
}

face_on() {
    local path="$1"
    local resp
    resp="$("${CURL[@]}" -X POST "${BASE}/PlayVideo" \
        -H 'Content-Type: application/json' \
        -d "{\"video_path\":\"${path}\",\"mode\":${MODE_LOOP},\"priority\":${PRIO}}")" \
        || die "cannot reach face UI at ${BASE}"
    case "$resp" in
        *'"success":true'*) echo "[x2_face] ON  -> ${path}" ;;
        *1100*) die "rejected: something higher-priority is playing. Raise X2_FACE_PRIO. ($resp)" ;;
        *) die "PlayVideo failed: $resp" ;;
    esac
}

face_off() {
    local resp
    resp="$("${CURL[@]}" "${BASE}/PlayDefaultEmoji")" \
        || die "cannot reach face UI at ${BASE}"
    case "$resp" in
        *'"success":true'*) echo "[x2_face] OFF -> vendor idle face restored" ;;
        *) die "PlayDefaultEmoji failed: $resp" ;;
    esac
}

case "${1:-toggle}" in
    on)     face_on "$MEDIA" ;;
    off)    face_off ;;
    media)  [ $# -ge 2 ] || die "media needs an mp4 path (on PC3)"; face_on "$2" ;;
    status)
        p="$(now_playing)"
        [ -n "$p" ] || die "no response from ${BASE}"
        case "$p" in
            *"$OURS_MATCH"*) echo "ours: $p" ;;
            *)               echo "vendor: $p" ;;
        esac ;;
    toggle)
        # Read real state rather than tracking our own -- the vendor can change
        # the face underneath us at any time.
        p="$(now_playing)"
        [ -n "$p" ] || die "no response from ${BASE}"
        case "$p" in
            *"$OURS_MATCH"*) face_off ;;
            *)               face_on "$MEDIA" ;;
        esac ;;
    -h|--help)
        sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    *)  die "unknown command: $1 (want: on|off|toggle|status|media <path>)" ;;
esac
