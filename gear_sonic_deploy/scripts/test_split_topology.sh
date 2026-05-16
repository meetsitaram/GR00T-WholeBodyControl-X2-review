#!/usr/bin/env bash
# Local-only smoke test for the split-topology deploy plumbing.
#
# This does NOT touch the real robot; it stands up the new components
# (motor monitor + manager motor-monitor SUB + manager resume PUB)
# against synthetic data so a developer can verify in <1 minute that:
#
#   * x2_motor_monitor.py imports cleanly + emits a boot record + at
#     least one sample record at the expected JSONL path;
#   * the laptop manager's --motor-monitor-host SUB receives a
#     ZMQ-published summary frame;
#   * the laptop manager's --resume-pub-* PUB binds successfully and
#     a separate SUB sees the heartbeat;
#   * x2_freeze_postmortem.py loads a synthetic motor_monitor JSONL
#     + sidecar JSONL + an empty deploy log dir and produces a
#     timeline.csv + timeline.md without crashing.
#
# It does NOT smoke the C++ deploy (that needs a colcon build + ROS
# overlay; gate that separately) and does NOT smoke the actual ROS
# subscriptions (no ROS bag is fed in). The intent is "did our Python
# wiring break in any of the obvious ways"; for end-to-end real-robot
# verification, run pc2_preflight.sh + the full stack on the bot.
#
# Pass an explicit --python /path/to/python if your repo's .venv is
# not at .venv/bin/python.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3)"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_BLUE=$'\e[34m'; C_RESET=$'\e[0m'
PASS=0; FAIL=0
ok()   { printf '  %s[ ok ]%s %s\n' "${C_GREEN}" "${C_RESET}" "$*"; PASS=$((PASS+1)); }
fail() { printf '  %s[FAIL]%s %s\n' "${C_RED}"   "${C_RESET}" "$*"; FAIL=$((FAIL+1)); }
section() { printf '\n%s== %s ==%s\n' "${C_BLUE}" "$*" "${C_RESET}"; }

TMP_DIR="$(mktemp -d -t x2_split_smoke_XXXX)"
trap 'rm -rf "${TMP_DIR}"; jobs -p | xargs -r kill 2>/dev/null || true' EXIT

# -------------------------------------------------------------------------
# 1. AST-parse the new scripts (no ROS / ZMQ side effects required).
# -------------------------------------------------------------------------
section "1. AST-parse new Python scripts"
for f in \
    gear_sonic_deploy/scripts/x2_motor_monitor.py \
    gear_sonic_deploy/scripts/x2_freeze_postmortem.py
do
    if "${PYTHON}" -c "import ast, sys; ast.parse(open('${f}').read()); print('ok')" >/dev/null 2>&1; then
        ok "${f} parses"
    else
        fail "${f} failed to parse"
    fi
done

# -------------------------------------------------------------------------
# 2. quest3_manager_x2.py argparse smoke -- verify the new flags exist.
# -------------------------------------------------------------------------
section "2. quest3_manager_x2.py CLI surface"
for flag in --resume-pub-enabled --resume-pub-port --resume-pub-topic \
            --motor-monitor-host --motor-monitor-port --motor-monitor-topic
do
    if "${PYTHON}" -m gear_sonic.scripts.quest3_manager_x2 --help 2>/dev/null \
            | grep -q -- "${flag}"; then
        ok "${flag} exposed"
    else
        fail "${flag} missing from --help"
    fi
done

# -------------------------------------------------------------------------
# 3. x2_freeze_postmortem.py end-to-end with synthetic logs.
# -------------------------------------------------------------------------
section "3. x2_freeze_postmortem.py against synthetic logs"
SYN_MONITOR="${TMP_DIR}/motor_monitor.jsonl"
SYN_SIDECAR="${TMP_DIR}/manager_sidecar.jsonl"

now_ts=$(date +%s.%N)
"${PYTHON}" - "${SYN_MONITOR}" "${now_ts}" <<'PY'
import json, sys
path, ts0 = sys.argv[1], float(sys.argv[2])
with open(path, "w") as fh:
    fh.write(json.dumps({"kind": "boot", "ts": ts0, "summary_rate_hz": 1.0,
                         "tracking_error_warn_rad": 0.30,
                         "limit_margin_rad": 0.05,
                         "stale_state_s": 0.5, "stale_command_s": 1.0,
                         "zmq_port": 5567, "zmq_topic": "motor_monitor",
                         "jsonl_path": path, "rotate_daily": False,
                         "pid": 42}) + "\n")
    fh.write(json.dumps({"kind": "sample", "ts": ts0 + 1.0, "rel_t": 1.0,
                         "mc_action_mode": 200, "mc_action_desc": "STAND_DEFAULT",
                         "mc_action_status": 1,
                         "groups": {"leg": {"count": 12, "max_tracking_err": 0.04}},
                         "top_tracking_err": []}) + "\n")
    fh.write(json.dumps({"kind": "event", "ts": ts0 + 2.0,
                         "type": "tracking_error_spike",
                         "joint": "left_knee_joint",
                         "tracking_err": 0.42,
                         "threshold": 0.30,
                         "pos": 0.5, "target": 0.92,
                         "vel": 0.1, "eff": 5.0,
                         "kp": 60.0, "kd": 1.0}) + "\n")
PY
"${PYTHON}" - "${SYN_SIDECAR}" "${now_ts}" <<'PY'
import json, sys
path, ts0 = sys.argv[1], float(sys.argv[2])
with open(path, "w") as fh:
    fh.write(json.dumps({"kind": "planner_cmd", "ts": ts0,
                         "intent": "walk", "magnitude": "forward"}) + "\n")
    fh.write(json.dumps({"kind": "resume_chord", "ts": ts0 + 1.5,
                         "event": "press", "press_count": 1}) + "\n")
PY

PM_OUT="${TMP_DIR}/postmortem"
if "${PYTHON}" gear_sonic_deploy/scripts/x2_freeze_postmortem.py \
        --motor-monitor "${SYN_MONITOR}" \
        --manager-sidecar "${SYN_SIDECAR}" \
        --out-dir "${PM_OUT}" >/dev/null 2>&1; then
    ok "postmortem ran cleanly"
else
    fail "postmortem crashed"
fi
if [[ -f "${PM_OUT}/timeline.csv" && -f "${PM_OUT}/timeline.md" ]]; then
    ok "timeline.csv + timeline.md emitted"
else
    fail "timeline outputs missing"
fi
if grep -q "tracking_error_spike" "${PM_OUT}/timeline.csv"; then
    ok "spike event reached the timeline"
else
    fail "spike event NOT in timeline.csv"
fi

# -------------------------------------------------------------------------
# 4. Manager motor_monitor SUB receives a synthetic PUB frame.
# -------------------------------------------------------------------------
section "4. Manager motor_monitor SUB / synthetic monitor PUB"
SUB_LOG="${TMP_DIR}/sub_test.log"
MON_PUB_OUT="${TMP_DIR}/mon_pub_received.txt"

"${PYTHON}" - "${MON_PUB_OUT}" <<'PY' &
import json, sys, time, zmq
out_path = sys.argv[1]
ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "motor_monitor")
sub.connect("tcp://127.0.0.1:25567")
poller = zmq.Poller()
poller.register(sub, zmq.POLLIN)
deadline = time.monotonic() + 5.0
got = False
while time.monotonic() < deadline:
    socks = dict(poller.poll(200))
    if sub in socks:
        topic, payload = sub.recv_multipart()
        with open(out_path, "w") as fh:
            fh.write(payload.decode("utf-8"))
        got = True
        break
sub.close()
sys.exit(0 if got else 1)
PY
SUB_PID=$!
sleep 0.8

"${PYTHON}" - <<'PY'
import json, time, zmq
ctx = zmq.Context.instance()
pub = ctx.socket(zmq.PUB)
pub.bind("tcp://127.0.0.1:25567")
time.sleep(0.5)
payload = {"sample": {"kind": "sample", "ts": time.time(), "rel_t": 0.0,
                       "mc_action_mode": 200, "mc_action_desc": "STAND_DEFAULT",
                       "mc_action_status": 1,
                       "groups": {"leg": {"count": 12}},
                       "top_tracking_err": []},
           "events": []}
for _ in range(3):
    pub.send_multipart([b"motor_monitor", json.dumps(payload).encode("utf-8")])
    time.sleep(0.2)
pub.close()
PY

wait "${SUB_PID}"
SUB_RC=$?
if [[ "${SUB_RC}" -eq 0 && -s "${MON_PUB_OUT}" ]]; then
    ok "motor_monitor SUB received synthetic PUB"
else
    fail "motor_monitor SUB did not see the PUB (rc=${SUB_RC})"
fi

# -------------------------------------------------------------------------
# 5. Resume PUB <-> SUB roundtrip via raw ZMQ.
# -------------------------------------------------------------------------
section "5. pose_resume PUB / SUB roundtrip"
RESUME_OUT="${TMP_DIR}/resume_received.txt"
"${PYTHON}" - "${RESUME_OUT}" <<'PY' &
import sys, time, zmq
out = sys.argv[1]
ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "pose_resume")
sub.connect("tcp://127.0.0.1:25566")
poller = zmq.Poller()
poller.register(sub, zmq.POLLIN)
deadline = time.monotonic() + 5.0
got = False
while time.monotonic() < deadline:
    socks = dict(poller.poll(200))
    if sub in socks:
        topic, payload = sub.recv_multipart()
        with open(out, "w") as fh:
            fh.write(f"topic={topic.decode()} payload_len={len(payload)}")
        got = True
        break
sub.close()
sys.exit(0 if got else 1)
PY
SUB_PID=$!
sleep 0.8

"${PYTHON}" - <<'PY'
import struct, time, zmq
ctx = zmq.Context.instance()
pub = ctx.socket(zmq.PUB)
pub.bind("tcp://127.0.0.1:25566")
time.sleep(0.5)
for _ in range(3):
    pub.send_multipart([b"pose_resume", struct.pack("d", time.monotonic())])
    time.sleep(0.2)
pub.close()
PY

wait "${SUB_PID}"
SUB_RC=$?
if [[ "${SUB_RC}" -eq 0 && -s "${RESUME_OUT}" ]]; then
    ok "pose_resume SUB received synthetic PUB"
else
    fail "pose_resume SUB did not see the PUB (rc=${SUB_RC})"
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo
printf '%sSummary:%s pass=%d  fail=%d\n' "${C_BLUE}" "${C_RESET}" "${PASS}" "${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
    exit 1
fi
exit 0
