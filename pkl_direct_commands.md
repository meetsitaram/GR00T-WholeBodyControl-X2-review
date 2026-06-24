# Direct PKL Playback Commands

### Action
Play a motion-clip PKL straight into SONIC with no kplanner / manager /
VLA in the loop. Sibling to the teleop and VLA stacks; recorder fans
in over a single `motion_clip_cmd` wire from a separate
`play_gesture` / `play_locomotion` shell.

Useful when:
- the kplanner's velocity-to-pose mapping is degrading a clip you
  know SONIC can already track,
- you want to A/B SONIC tracking on a SOMA clip without standing up
  a VR headset,
- you want to smoke a freshly retargeted clip end-to-end through the
  policy + actuators before wiring it into the kplanner library.

The recorder publishes the same `pose:5556` envelope the kplanner
would (current `joint_pos_mj` + 9-slot future window), so the C++
deploy needs zero special configuration. See
[gear_sonic/utils/teleop/motion_clip_session.py](gear_sonic/utils/teleop/motion_clip_session.py)
for the wire shape and the gesture / locomotion `kind` discriminator.

Cross-references:
- [mc_gesture_capture_commands.md](mc_gesture_capture_commands.md) --
  capture + corpus pipeline for MC gestures (the catalog source).
- [clip_motion_commands.md](clip_motion_commands.md) -- end-to-end
  MuJoCo recipe for the in-place gesture path through the full
  Quest3 stack (more moving parts than this one).

=============== do not auto edit this section ===============

### sim: bring up direct-PKL stack
```sh
gear_sonic/scripts/run_x2_pkl_direct_stack.sh
```
Spawns `deploy_x2.sh sim --vla` + recorder. No body_pose upstream, no
manager, no parquet. Recorder publishes idle stand on every no-clip
tick. Ready when the wrapper logs `recorder READY`.

### sim: play a catalog gesture (in-place; yaw-rebased to robot heading)
```sh
python -m gear_sonic.scripts.play_gesture --list
python -m gear_sonic.scripts.play_gesture sit_stand_sit_A538
```

### sim: play an ad-hoc gesture PKL
```sh
python -m gear_sonic.scripts.play_gesture \
    --pkl gear_sonic/data/motions/x2_recorded/mc_gestures/hug_001.pkl
```

### sim: play a locomotion clip (walks / turns; authored yaw preserved)
```sh
# Stitched home loop (walk forward, turn, walk back).
python -m gear_sonic.scripts.play_locomotion \
    --pkl gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl

# Single-cycle forward walk.
python -m gear_sonic.scripts.play_locomotion \
    --pkl ~/locomotion_smoke/Loop_Forward_Walk_001__A018.pkl

# Specific motion-key inside a multi-clip PKL.
python -m gear_sonic.scripts.play_locomotion \
    --pkl gear_sonic/data/motions/x2_ultra_locowalk.pkl \
    --motion-key Loop_Forward_Walk_001__A018
```

### sim: abort an in-flight clip (from another terminal)
```sh
python -m gear_sonic.scripts.play_locomotion --release
# or play_gesture --release; same wire, same effect.
```

### real robot: bring up direct-PKL stack
##### start sonic on PC2 (Robogym wifi)
```sh
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight
```

##### bring up laptop-side recorder
```sh
./gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
    --pc2-host 192.168.86.32
```
PC2's `x2_pose_proxy` already SUBs `<LAPTOP_IP>:5556` over wifi; the
wrapper binds the recorder's pose PUB on `*` only when `--pc2-host`
is set so the wire isn't broadcast to PC2 in sim mode.

##### play clip onto the real robot
```sh
python -m gear_sonic.scripts.play_gesture <name>
python -m gear_sonic.scripts.play_locomotion --pkl <path>
```

##### stop sonic
```sh
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32
```

=============== end of pure manual notes section ===============

### gesture vs locomotion (which CLI?)

| trait | play_gesture | play_locomotion |
|---|---|---|
| target clips | in-place (wave, sit, bow, handshake) | walks / turns / sidesteps |
| yaw handling | PKL frame-0 yaw rebased onto robot heading | authored yaw kept verbatim |
| trigger surface | catalog name OR `--pkl <path>` | `--pkl <path>` only (no catalog) |
| hold-after pose | supported (catalog or `--hold` flag) | not supported (locomotion ends in idle stand) |
| wire `kind` | `"gesture"` | `"locomotion"` |
| use the wrong CLI? | walk clip starts at robot's current yaw, drifts off-axis | gesture clip plays at whatever world-frame yaw the PKL was authored at; robot does not rotate to match |

Both publish on the same `motion_clip_cmd` topic
([motion_clip_session.MOTION_CLIP_CMD_DEFAULT_PORT](gear_sonic/utils/teleop/motion_clip_session.py)
= 5568); the recorder dispatches on the `kind` field. You can A/B by
flipping the CLI without touching the stack.

### loop-seam pitfalls (locomotion)

When the C++ deploy wraps a single-cycle PKL (`PklMotionReference::Sample`
in
[reference_motion.cpp](gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/reference_motion.cpp)),
joint angles snap from `dof[N-1]` back to `dof[0]` in one frame. A
clip whose joint seam is >~5deg will visibly jolt SONIC every wrap.
Quick check before playing:

```sh
python3 -c "
import joblib, numpy as np
m = joblib.load('<path-to-pkl>')
m = next(iter(m.values()))
print('N=%d fps=%.1f dur=%.2fs' % (m['dof'].shape[0], m['fps'], m['dof'].shape[0]/m['fps']))
print('joint seam (max abs): %.4f rad' % np.max(np.abs(m['dof'][0]-m['dof'][-1])))
"
```

Stitched home loops (`gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl`
and siblings) are authored to close (seam <1deg). Single-cycle SOMA
clips like `Loop_Forward_Walk_001__A018` are NOT closed loops despite
the name (seam ~14deg) -- play them with `play_locomotion` to see
the wrap jolt for yourself.

### cleanup orphans / freed ports
```sh
./gear_sonic/scripts/run_x2_pkl_direct_stack.sh --cleanup-only
```
