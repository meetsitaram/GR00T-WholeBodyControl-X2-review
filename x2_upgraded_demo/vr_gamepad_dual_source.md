# Gamepad + Quest 3 VR dual-source control (pad initiates, VR toggles in)

Goal: keep today's gamepad-initiated, PC2-resident pipeline exactly as is, and
let the operator toggle control to a Quest 3 (locomotion + arm/hand
manipulation) that connects to the robot over WiFi / the robot AP
(`aima nm start-ap` -> SSID X2-ROBOT). Verified in sim first.

## Why wiring was needed at all

Everything else already existed (quest3_manager_x2, WebXR server :8443/:8765,
arm IK + finger retarget, the `planner_cmd` contract shared with the pad
bridge). The one real gap: pad and VR could not COEXIST on :5563 —
quest3_manager_x2 PUB-**binds** :5563, `pad_locomotion_bridge --bind` also
binds, and the bridge's "connect" fallback was PUB->PUB (silently dropped by
ZMQ). Two publishers need the SUBSCRIBER to own the bind.

## What changed (2026-07-24, sim-verified wiring)

| File | Change |
|---|---|
| `gear_sonic/scripts/pc2_kplanner_onnx.py` | new `--cmd-bind`: planner_cmd SUB binds `tcp://*:5563` instead of connecting |
| `gear_sonic/scripts/x2_kplanner.py` | same as `--zmq-cmd-bind` (torch runtime parity) |
| `gear_sonic/scripts/quest3_manager_x2.py` | new `--planner-cmd-connect`: planner_cmd PUB connects instead of binding |
| `gear_sonic/scripts/run_x2_quest3_planner_stack.sh` | new `--pad-and-vr` mode: spawns quest3_manager_x2 AND a pad-bridge sidecar (PUB-connect, own log `pad_bridge.log`, watched pid), passes the bind flag to whichever kplanner runtime is active |
| `gear_sonic/scripts/sim_onnx_planner.sh` | new `--vr` flag -> stack runs `--pad-and-vr` instead of `--pad-only` |

`pad_locomotion_bridge.py` needed no change — its default mode was already
PUB-connect to 127.0.0.1:5563.

Arbitration is behavioral, no arbiter process: the pad publishes only while
its deadman is held (one idle cmd on release); the manager publishes only when
engaged via the A+B+X+Y chord. Last-writer-wins; one active source at a time
by operator discipline. Pad E-stop chord (play_xbox_controller) is a separate
wire and unaffected.

## Sim verification (laptop, deploy-parity ONNX runtime)

```bash
cd ~/Projects/GR00T-WholeBodyControl
# pad visible to pygame BEFORE launch; env_isaaclab conda env for the kplanner
ALLOW_MISMATCH=1 \
SONIC_MODEL=~/x2_cloud_checkpoints/g1teleop_overnight/sonic/softland_173528/exported/softland_4800_g1.onnx \
PLANNER_MODEL=~/x2_cloud_checkpoints/planner_onnx_p500k_stance \
./gear_sonic/scripts/sim_onnx_planner.sh --pc2-host 192.168.86.32 --vr
```

Checklist:
1. Pad drives as before (L2 deadman, sticks) — proves the bridge sidecar path.
2. Quest 3 browser -> `https://<laptop-ip>:8443` (same WiFi; ufw allow
   8443/tcp 8765/tcp), accept cert, Connect WS, Start VR.
3. A+B+X+Y on controllers -> LOCOMOTION; sticks drive the robot in MuJoCo.
   Release everything; hold L2 on the pad -> pad drives again (the toggle).
4. B -> ARM_MANIPULATION, A -> engage arm IK -> arms + fingers track in the
   viewer. (In this tethered/sim stack, record_x2_dataset does the
   body_pose+arm_targets -> final_pose merge — already wired.)
5. kplanner log should show `SUB bind` on :5563; manager log
   `planner_cmd PUB connected`; pad log `PUB connect`.

## Robot-side phase (untethered PC2) — NOT yet implemented

In untethered mode there is NO recorder/merger: `pc2_kplanner_onnx.py`
publishes `pose` :5556 straight to the watchdog -> deploy. To get VR arms on
the robot:

1. **Arm/hand merge in pc2_kplanner_onnx.py** (the deploy-parity move — same
   file sim exercises): SUB the manager's msgpack topics on :5564
   (`arm_targets` [7+7], `hand_finger_cmd` [10+10], `stream_mode`) and, while
   `is_engaged`, override the 14 arm DOFs in outgoing pose frames and fill
   `left/right_hand_joints` (currently zero-filled, wire already carries
   them). Short engage/disengage blend like record_x2_dataset does. Physical
   fingers additionally ride the existing x2_hand_bridge, which can SUB
   127.0.0.1:5564 once the manager is local to PC2.
2. **quest3_manager_x2 on PC2** as a new tmux session in
   `x2_pc2/ritual_start_demo.sh` (start with the watchdog group; passive until
   headset engages). Flags: `--planner-cmd-connect`; certs auto-generate via
   openssl on first run.
3. **PC2 venv deps** (add to pc2_bringup.sh): `websockets msgpack scipy`
   (manager chain needs nothing heavier; arm IK is numpy-only; mujoco import
   is lazy/optional FK). Files land under /home/run/getsolo/ per convention.
4. **pad_bridge ritual line**: drop `--bind`, and start pc2_kplanner_onnx with
   `--cmd-bind` (one-line changes in ritual_start_demo.sh).
5. **Network**: headset joins robot AP (X2-ROBOT / x2demo2026), browser to
   `https://<pc2-ap-ip>:8443`. Check PC2 firewall for 8443/8765.
6. Ignition unchanged: pad chord L1+R1+L2+R2 3s + Y still gates everything;
   VR is a passive extra source until its own engage chord.

Open items: kplanner intents are a strict subset of the heuristic planner's —
manager squat/torso-height intents may be partial no-ops; Orin headroom for
the manager next to GPU replans (expected light, verify thermals); decide
bucketed vs continuous VR locomotion (`intent_enable_continuous_locomotion`,
default OFF = bucketed steps).
