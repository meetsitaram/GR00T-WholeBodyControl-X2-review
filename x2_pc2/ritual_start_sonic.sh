#!/bin/bash
# PC2-local sonic start (GATED). Plumbing first; deploy ONLY after the
# pose stream is verified flowing on the deploy SUB wire (127.0.0.1:5558).
# Sonic must NEVER take MC without its constant idle reference.
LOG=/home/run/getsolo/log/ritual_fired.log
echo "$(date +%F_%T) RITUAL START" >> $LOG
start_one() {
  S=$1; SCRIPT=/home/run/getsolo/log/start_${S}.sh
  [ -f "$SCRIPT" ] || { echo "$(date +%F_%T) MISSING $SCRIPT" >> $LOG; return 1; }
  if tmux has-session -t $S 2>/dev/null; then
    if tmux capture-pane -p -t $S -S -200 2>/dev/null | grep -q "cmd exited with status="; then
      tmux kill-session -t $S 2>/dev/null
      echo "$(date +%F_%T) $S: cleared post-mortem pane" >> $LOG
    else
      echo "$(date +%F_%T) $S: already running -- skipped" >> $LOG
      return 0
    fi
  fi
  tmux new-session -d -s $S "bash $SCRIPT"
  echo "$(date +%F_%T) $S: STARTED" >> $LOG
}
for S in x2_pose_watchdog x2_hand_bridge x2_motor_monitor; do
  start_one $S
done
# ---- GATE: pose frames must flow before deploy may exist ----
/home/run/getsolo/venv/bin/python - <<PYEOF
import zmq, sys, time
ctx = zmq.Context(); sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
sub.connect("tcp://127.0.0.1:5558")
sub.RCVTIMEO = 12000
t0 = time.time()
try:
    sub.recv_multipart()
    print(f"pose stream OK after {time.time()-t0:.1f}s")
    sys.exit(0)
except zmq.Again:
    print("NO POSE STREAM within 12s")
    sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then
  echo "$(date +%F_%T) GATE FAILED: no pose stream -- DEPLOY NOT STARTED" >> $LOG
  exit 1
fi
echo "$(date +%F_%T) GATE PASSED: pose stream flowing" >> $LOG
start_one x2_deploy
