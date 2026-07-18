#!/bin/bash
# LAPTOP-FREE demo ignition. Chain: local-upstream watchdog -> plumbing ->
# pc2 planner runtime (ONNX) -> pose-stream gate -> deploy -> pad bridge.
LOG=/home/run/getsolo/log/ritual_fired.log
PS=/home/run/getsolo/planner_stack
PY=/home/run/getsolo/venv/bin/python
echo "$(date +%F_%T) DEMO RITUAL START" >> $LOG
start_tmux() {  # name, command
  tmux has-session -t $1 2>/dev/null && { echo "$(date +%F_%T) $1: running -- skipped" >> $LOG; return 0; }
  tmux new-session -d -s $1 "$2"
  echo "$(date +%F_%T) $1: STARTED" >> $LOG
}
start_tmux x2_pose_watchdog "bash /home/run/getsolo/start_x2_pose_watchdog_local.sh"
start_tmux x2_hand_bridge   "bash /home/run/getsolo/log/start_x2_hand_bridge.sh"
start_tmux x2_motor_monitor "bash /home/run/getsolo/log/start_x2_motor_monitor.sh"
start_tmux pc2_kplanner "PYTHONPATH=$PS:$PS/motionbricks $PY /home/run/getsolo/pc2_kplanner_onnx.py \
  --onnx $PS/models/planner_onnx/x2_planner_template.onnx --planner-mode slow_walk \
  --warmup-qpos $PS/models/kplanner_idle_anchor_g1teleop_v3.pkl \
  --dances-dir $PS/models/dances_x2m2 2>&1 | tee -a /home/run/getsolo/log/pc2_kplanner.log"
sleep 3
start_tmux pad_bridge "PYTHONPATH=$PS/gear_sonic $PY /home/run/getsolo/pad_locomotion_bridge.py \
  --bind --source zmq --pad-host 127.0.0.1 --lock-speed --deadman left \
  --clip-pkl $PS/models/dances_x2m2 --clip-keys \"$(ls $PS/models/dances_x2m2 | sed s/.x2m2$// | paste -sd,)\" \
  2>&1 | tee -a /home/run/getsolo/log/pad_bridge.log"
# gate: pose frames must flow (watchdog downstream :5558) before deploy exists
$PY - <<PYEOF
import zmq, sys, time
ctx = zmq.Context(); sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
sub.connect("tcp://127.0.0.1:5558"); sub.RCVTIMEO = 15000
try:
    m = sub.recv_multipart(); print(f"pose stream OK ({len(m[-1])}b)"); sys.exit(0)
except zmq.Again:
    print("NO POSE STREAM"); sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then
  echo "$(date +%F_%T) GATE FAILED -- DEPLOY NOT STARTED" >> $LOG; exit 1
fi
echo "$(date +%F_%T) GATE PASSED" >> $LOG
tmux has-session -t x2_deploy 2>/dev/null || tmux new-session -d -s x2_deploy "bash /home/run/getsolo/log/start_x2_deploy.sh"
echo "$(date +%F_%T) x2_deploy: STARTED" >> $LOG

# Partner-logo reel on the face display. Cosmetic only: runs AFTER the pose gate
# and after deploy start, and swallows every failure, so it can never delay or
# block ignition. Toggle live with L1/R1 chord -> interact/x2_face.sh toggle.
( /home/run/getsolo/interact/x2_face.sh on >/dev/null 2>&1 \
    && echo "$(date +%F_%T) face: logo reel ON" >> $LOG \
    || echo "$(date +%F_%T) face: logo reel FAILED (ignored)" >> $LOG ) &
