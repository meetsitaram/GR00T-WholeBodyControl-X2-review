#!/bin/bash
# LAPTOP-FREE demo ignition. Chain: local-upstream watchdog -> plumbing ->
# pc2 planner runtime (ONNX) -> pose-stream gate -> deploy -> pad bridge.
LOG=/home/run/getsolo/log/ritual_fired.log
PS=/home/run/getsolo/planner_stack
PY=/home/run/getsolo/venv/bin/python
echo "$(date +%F_%T) DEMO RITUAL START" >> $LOG
start_tmux() {  # name, command
  tmux has-session -t $1 2>/dev/null && { echo "$(date +%F_%T) $1: running -- skipped" >> $LOG; return 0; }
  # `; rc=$?` tail: record the child exit status in the ritual log so a
  # killed session is distinguishable from a crashed one. Without this the
  # only trace was the tmux pane, which dies with the session.
  tmux new-session -d -s $1 "$2 ; rc=\$?; echo \"$(date +%F_%T) $1: EXITED rc=\$rc\" >> /home/run/getsolo/log/ritual_fired.log"
  echo "$(date +%F_%T) $1: STARTED" >> $LOG
}
start_tmux x2_pose_watchdog "bash /home/run/getsolo/start_x2_pose_watchdog_local.sh"
start_tmux x2_hand_bridge   "bash /home/run/getsolo/log/start_x2_hand_bridge.sh"
start_tmux x2_motor_monitor "bash /home/run/getsolo/log/start_x2_motor_monitor.sh"
# Visualiser + scan pipeline, then the obstacle guard, both BEFORE kplanner
# so the scan is up before kplanner. The guard now starts AFTER kplanner:
# its ZMQ SUB reconnects when the publisher appears, so the clamp arms a few
# on /scan and the guard read inf. kplanner subscribes to ZMQ 5571 and zeroes
# forward/lateral while blocked, latched until the deadman is released.
# Toggle for the forward-obstacle guard (lumi scan pipeline + scan_guard_pub).
# 0 = OFF: kplanner's clamp is FAIL-OPEN, so with no publisher it never
# clamps — operator deadman is the only stop. Set to 1 to re-arm.
ENABLE_SCAN_GUARD=0

if [ "$ENABLE_SCAN_GUARD" = "1" ]; then
bash /home/run/getsolo/lumi.sh >> /home/run/getsolo/log/scan_guard.log 2>&1
sleep 8
fi

# Kplanner kitchen-teleop tuning 2026-08-02 (see gear_sonic_deploy/configs/
# kplanner_tuning_history.md for the sweep evidence + old/new comparison):
# FWD 0.5->0.4 (template floor ~0.45; 0.4 = smoothest), ARC_TURN default
# 0.55->0.70 (walking-turn radius ~1.2m -> 0.6-0.85m), replan threshold
# 32->48 (turn response at PC2 latency 0.68s -> 0.22s). Standing turn
# stays 1.0 (July sweep, robot-verified).
start_tmux pc2_kplanner "PYTHONPATH=$PS:$PS/motionbricks KPLANNER_FIXED_TURN_RAD_S=1.0 KPLANNER_FIXED_FWD_MPS=0.4 KPLANNER_FIXED_ARC_TURN_RAD_S=0.70 stdbuf -oL -eL $PY /home/run/getsolo/pc2_kplanner_onnx.py \
  --onnx $PS/models/planner_onnx/x2_planner_template.onnx --planner-mode slow_walk --cmd-bind --replan-threshold-frames 48 \
  --warmup-qpos $PS/models/kplanner_idle_anchor_g1teleop_v3.pkl \
  --dances-dir $PS/models/dances_x2m2 --ort-gpu --playing-yaw-resync-dps 10 2>&1 | tee -a /home/run/getsolo/log/pc2_kplanner.log"
sleep 3

# SYSTEM python with absolute ROS paths: the gear_sonic venv has no rclpy and
# a fresh post-reboot shell has no ROS env, so without these spelled out the
# guard dies silently and the robot drives unguarded.
if [ "$ENABLE_SCAN_GUARD" = "1" ]; then
start_tmux scan_guard "LD_LIBRARY_PATH=/agibot/software/common/lib:/opt/ros/humble/lib \
  AMENT_PREFIX_PATH=/agibot/software/common:/opt/ros/humble \
  PYTHONPATH=/agibot/software/common/local/lib/python3.10/dist-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages \
  stdbuf -oL -eL python3 /home/run/getsolo/scan_guard_pub.py \
  2>&1 | tee -a /home/run/getsolo/log/scan_guard.log"
sleep 4
fi

# Curated dance banks (was: every clip in dances_x2m2, 14 of them).
#   L1+Y / L1+A  -> EASY
#   L1+X / L1+B  -> MEDIUM
#   L1+R1        -> STOP  (moved off B, which is now MEDIUM-previous)
# dance_freedom_wheels_001__A465 is deliberately EXCLUDED: it falls in sim
# under BOTH .pt and ONNX (pelvis_z 0.395 at t=5.52s, identical for each), and
# the operator saw the same backward tilt on the robot.
EASY_DANCES="dance_party_hips_003__A467,dance_party_hips_003__A464,dance_party_hips_003__A465"
# Bank B is now SHADOW BOXING (was MEDIUM_DANCES). These are the PRE-G1-execution
# retargets (x2_upgraded_demo/pkl_motions), not the shadow_boxing_executed/ ones:
# passing through G1-SONIC execution damped the punches to ~55% of the source arm
# motion. Both are 120 fps -- the clip player handles that (float phase advancing
# fps/OUTPUT_FPS per tick, pc2_kplanner_onnx.py:1393). ~1.3 m of travel each.
COMBAT="shadow_boxing_R_003__A359_M,shadow_boxing_R_003__A359"

# One bank per face button: Y=easy dances  X=combat  A=gestures  B=medium dances.
MEDIUM="egipt_dance_R_001__A438,dance_hiphop_stick_n_roll_dancehall_R_loop_003__A324,dance_distraction_dance_001__A466"

# Bank G: gestures on L1+A (cycles forward). Taking L1+A for this means bank A
# no longer has a "previous" -- L1+Y just advances through the dances.
# bow_001 REMOVED: benchmark_motions_mujoco vs softland_4800 -> FELL at 2.80s
# (pelvis_z 0.39). The other 7 all survived the full 12 s.
# RIGHT-STICK 4-way (deadman RELEASED). Order: LEFT,RIGHT,UP,DOWN.
# LEFT/RIGHT are in-place turns (~0.09 m travel) and fire on entry -- no dwell.
# UP/DOWN travel 3.5 m / 5.1 m so they need a 2 s hold.
# NOTE the sign convention: __A056 is the RIGHT turn, __A056_M is the LEFT.
TURNS="locowalk__idle_turn_270_002__A056_M,locowalk__idle_turn_270_002__A056,relaxed_walk_forward,walk_circle_001"

GESTURES="right_wave_001,right_kiss_001,right_five_001,right_shake_001,turn_wave_right_001,turn_wave_left_001"

# PYTHONPATH/LD_LIBRARY_PATH include ROS humble dist-packages (rclpy for the
# bridge's ROS side) and getsolo/ itself; --cmd-bind on the planner means the
# bridge CONNECTS for planner_cmd (no --bind here). Back-ported 2026-07-29
# from the robot-verified PC2 copy that had drifted ahead of the repo.
start_tmux pad_bridge "PYTHONPATH=/home/run/getsolo:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:$PS/gear_sonic LD_LIBRARY_PATH=/opt/ros/humble/lib:\$LD_LIBRARY_PATH stdbuf -oL -eL $PY /home/run/getsolo/pad_locomotion_bridge.py \
  --source zmq --pad-host 127.0.0.1 --lock-speed --deadman left \
  --clip-pkl $PS/models/dances_x2m2 \\
  --clip-keys \"$EASY_DANCES\" --clip-keys-b \"$COMBAT\" \
  --clip-keys-g \"$GESTURES\" --clip-keys-m \"$MEDIUM\" \
  --clip-keys-turn \"$TURNS\" \
  --clip-key-dpad-up walk_circle_001 \\
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
# PC2-RESIDENT ritual launcher (checked into repo, --no-confirm baked in),
# NEVER log/start_x2_deploy.sh: that body is regenerated by every laptop
# x2_pc2_daemons.sh start with that session's flags, and one without
# --no-confirm leaves the deploy stuck at a y/N prompt no gamepad can
# answer (2026-07-29 blocked-ignition incident).
tmux has-session -t x2_deploy 2>/dev/null || tmux new-session -d -s x2_deploy "bash /home/run/getsolo/start_x2_deploy_ritual.sh"
echo "$(date +%F_%T) x2_deploy: STARTED" >> $LOG

# Partner-logo reel on the face display. Cosmetic only: runs AFTER the pose gate
# and after deploy start, and swallows every failure, so it can never delay or
# block ignition. Toggle live with L1/R1 chord -> interact/x2_face.sh toggle.
( /home/run/getsolo/interact/x2_face.sh on >/dev/null 2>&1 \
    && echo "$(date +%F_%T) face: logo reel ON" >> $LOG \
    || echo "$(date +%F_%T) face: logo reel FAILED (ignored)" >> $LOG ) &
