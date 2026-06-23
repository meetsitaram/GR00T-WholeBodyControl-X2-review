# MC Gesture Capture Commands

### Action
Capture mobile-app-triggered MC gestures (wave, handshake, ...) from the
real X2 robot into SOMA-byte-compatible motion-lib PKLs, ready to drop
into sonic training alongside the bones-seed corpus.

=============== do not auto edit this section ===============

### replay gestures in sim
gear_sonic/scripts/run_x2_quest3_planner_stack.sh

python -m gear_sonic.scripts.play_gesture     --pkl gear_sonic/data/motions/x2_recorded/mc_gestures/left_kiss_001.pkl

=============== end of pure manual notes section ===============

### prereq: MC running + mobile app paired
# If `pc2_preflight.sh` reports `publishers=0`, MC is not running. Start
# it from the laptop (PC1 = motion-control unit on the wired SDK link):
curl -X POST 'http://10.0.1.40:50080/x2/em/start_app?app=mc' \
    -d '{}' -H 'Content-Type: application/json'
# Then pair the mobile app as usual.

### capture one take (record + auto-convert)
# Defaults: source=state, root_rot=foot-flat, floor-anchor=lower-foot
# (+xy lock), fps=30, trim 0.5s/0.5s, session=mc_gestures_<UTC date>.
#
# Defaults explained:
#   * root_rot=foot-flat -- pelvis rotation is derived from leg-chain
#     FK such that the anchor foot stays at its frame-0 ("idle, flat
#     on the floor") world orientation. Produces physically-correct
#     counter-balance pitch (pelvis tilts BACK when arms reach
#     forward). torso-imu was the prior default but reads a forward
#     torso tilt on hug-style gestures, which would tip the robot.
#   * floor-anchor=lower-foot + xy lock -- anchor foot ankle holds
#     its frame-0 world XY (pass --no-anchor-xy for legacy XY=(0,0)).
# Override per-take with --root-rot torso-imu for walking / large-COM
# excursion takes where leg FK alone can't track pelvis pose.
# Workflow:
#   1. Run the command below.
#   2. The wrapper filters FastDDS RTPS noise and pops a big green
#      "READY -- trigger the gesture on the MOBILE APP now" banner the
#      moment the recorder's first 1Hz status line lands (~1-3 s after
#      launch). Wait for that banner BEFORE tapping.
#   3. Tap the gesture on the mobile app. Wait for the robot to settle.
#   4. Hit Ctrl-C ONCE. The recorder finalises, rsyncs back, converts,
#      and (if --view) pops MuJoCo.
./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh wave 001 \
    --pc2-host 192.168.86.32

./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh handshake 001 \
    --pc2-host 192.168.86.32

### capture + auto-preview in MuJoCo (recommended for the first few takes)
./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh wave 001 \
    --pc2-host 192.168.86.32 --view

### capture MC's commanded reference instead of executed state
# Use when MC's tracking error is making the captured motion look noisy.
# Default `state` is preferred for sim-to-real fine-tuning.
./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh dance 001 \
    --pc2-host 192.168.86.32 --cmd-source

### wider trim window for a slow gesture (e.g. 3 s leading, 2 s trailing)
./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh stretch 001 \
    --pc2-host 192.168.86.32 --trim-start 3.0 --trim-end 2.0

### record only; convert later (e.g. you want to A/B trim flags)
./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh wave 002 \
    --pc2-host 192.168.86.32 --no-convert

./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh wave 002 \
    --convert-only --trim-start 1.0

### verify a finished PKL in MuJoCo
.venv/bin/python gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_recorded/mc_gestures/wave_001.pkl \
    --motion-key wave_001

### inspect the PKL schema from python (sanity check before training)
python3 -c "
import joblib
m = joblib.load('gear_sonic/data/motions/x2_recorded/mc_gestures/wave_001.pkl')
for k, v in m.items():
    shapes = {kk: getattr(vv, 'shape', vv) for kk, vv in v.items() if kk in
              ('root_trans_offset','pose_aa','dof','root_rot','smpl_joints','fps')}
    print(k, shapes)
"
# Expected per entry:
#   root_trans_offset (T, 3)   pose_aa (T, 32, 3)   dof (T, 31)
#   root_rot (T, 4) xyzw       smpl_joints (T, 24, 3) zeros
#   fps = 30

### re-summarize an NPZ without rebuilding the PKL
# Lets you spot zero-message captures (e.g. DDS partitioned, MC down).
.venv/bin/python gear_sonic_deploy/scripts/x2_record_real_run.py \
    --summarize gear_sonic/data/motions/x2_recorded/mc_gestures_npz/mc_gestures_<UTC date>/wave_001.npz

### list captured takes so far this session
ls -lh gear_sonic/data/motions/x2_recorded/mc_gestures_npz/mc_gestures_$(date -u +%Y%m%d)/ \
    gear_sonic/data/motions/x2_recorded/mc_gestures/

### suggested gesture names (lowercase, no spaces)
# wave, handshake, bow, nod, salute, point, clap, dance, stretch,
# thumbs_up, fist_bump, hi_five, hands_on_hips, idle_breathe, ...



### outputs
# NPZ (rsynced from PC2):
#   scratch/runs/mc_gestures_<UTC date>/<gesture>_<take>.npz
# PKL (per take, ready for sonic training):
#   gear_sonic/data/motions/x2_recorded/mc_gestures/<gesture>_<take>.pkl
# PKL is byte-compatible with x2_ultra_bones_seed.pkl entries -- drop into
# the training corpus the same way SOMA-retargeted clips are wired in.

### troubleshooting
# - "torso-imu hit a scipy euler shape mismatch": FIXED in
#   _pelvis_rot_from_torso_imu (waist_*[:, None] when calling
#   scipy.spatial.transform.Rotation.from_euler). The wrapper still
#   carries an auto-fallback to --root-rot identity in case the bug
#   reappears on a different scipy build.
# - "no .npz produced": the recorder saw zero HAL messages. Check that
#   PC2 can reach PC1's DDS (Robogym WiFi can partition DDS); confirm
#   with `pc2_preflight.sh` or `ros2 topic hz /aima/hal/joint/arm/state`
#   on PC2. MC must be in an active mode (STAND_DEFAULT, not OFF).
# - "RTPS_READER / RTPS_WRITER warning spam": the wrapper filters them
#   (FastDDS discovery noise). If you bypass the wrapper and call
#   x2_record_remote.sh directly, the noise comes back.
# - "no READY banner appears": the recorder's first 1Hz status line
#   never printed -- subscriptions failed. Same root cause as
#   "no .npz produced" above. Check `ros2 topic list` on PC2.
# - "mc_mode_str stays constant during the take": expected for mobile-app
#   gestures. They run inside MC without flipping the parent action.
#   Capture still works; the mode timeline is informational only.
# - "PKL already exists": pass --override, or pick a fresh TAKE_NUM.
# - "feet slide / rock on a single sphere / robot leans toward
#   tip-over": FIXED via the default --root-rot foot-flat plus
#   --anchor-xy. foot-flat derives pelvis rotation from leg-chain FK
#   so the anchor foot stays rigidly at its frame-0 orientation;
#   anchor-xy keeps the anchor foot ankle at its frame-0 world XY.
#   Together: anchor foot is fully pinned (orientation + position),
#   the pelvis pitches/translates whatever the leg kinematics
#   require, body counter-balances correctly. To re-fix any take
#   captured before this change, reconvert:
#     ./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh --convert-only \
#         --override <gesture> <take>
#   For a single-foot in-place gesture (e.g. one-leg balance) force
#   the dominant foot to eliminate the ~1-2mm anchor-swap jitter:
#     --floor-anchor left-foot   (or right-foot)
#   For walking / large-COM excursions where leg FK alone can't
#   track the pelvis (foot transfer would cause a pelvis jump),
#   use --root-rot torso-imu instead.
# - "robot strides through the floor during replay": the take included
#   stepping. Pelvis-XY foot lock can't recover real world translation
#   from joints+IMU alone; you'd need the SLAM odometry path (currently
#   disabled for MC gestures since they're nominally in-place).
# - "MuJoCo viewer fails to launch": run from the env that has mujoco
#   installed, e.g. `conda run -n env_isaaclab --no-capture-output \
#   python gear_sonic/scripts/play_x2_motion_mujoco.py ...`.
