# Gesture Commands

=============== do not auto edit this section ===============
### start sonic on PC2 : Robogym Wifi
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

### start local stack
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pc2-host 192.168.86.32

### play gesture to sit and stand
python -m gear_sonic.scripts.play_gesture sit_down_A540

python -m gear_sonic.scripts.play_gesture stand_up_A540 --delay 5

### stop sonic
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32


=============== end of pure manual notes section ===============


## List

```sh
python -m gear_sonic.scripts.play_gesture --list
```

## MuJoCo

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh

python -m gear_sonic.scripts.play_gesture sit_stand_sit_A540
```

## Real robot

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pc2-host 192.168.86.32

python -m gear_sonic.scripts.play_gesture sit_stand_sit_A540
```

## Split sit + hold + stand

```sh
python -m gear_sonic.scripts.play_gesture sit_down_A540

python -m gear_sonic.scripts.play_gesture stand_up_A540 --delay 5
```

## Release a held pose

```sh
python -m gear_sonic.scripts.play_gesture --release
```

---

## Why we add `--lock-head-straight` to the SONIC start

### The observation

With SONIC running, you cannot move the head by hand — it feels stiff.
Confirmed:

* Head yaw motor works (free movement with SONIC down, and via the AgiBot app)
* SONIC was actively holding head yaw at ~+20° (commanded +0.50 rad,
  measured +0.345 rad)
* Head pitch is not actuated in firmware regardless

### Root cause

The deploy publishes full PD commands (kp≈16.8, kd≈1.07) on
`/aima/hal/joint/head/command` at 50 Hz. The policy's head target was
drifting off-center, and the deploy held it there stiffly.

Sending head joints through the ZMQ pose stream (VR / VLA path) does
**not** override this — the policy decides head motor targets, and
there is no head bypass (unlike the wrist bypass).

### The fix

`--max-target-dev-head` already exists in the C++ safety stack: it
clamps the policy's head target to within ±N radians of the trained
default (yaw=0, pitch=0). Setting it to a small positive value
(`0.01` ≈ 0.6°) effectively locks the head straight ahead. The
trained default *is* straight-ahead, so clamping there just keeps the
policy from drifting off-center.

### What we shipped

A convenience flag on `x2_pc2_daemons.sh`:

```sh
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start ... --lock-head-straight
```

It expands to `--max-target-dev-head 0.01` on `deploy_x2.sh`, which
overrides the tuning YAML's default of `0.50` (~29° sweep). Override
the clamp value via env var `LOCK_HEAD_STRAIGHT_RAD` if needed (e.g.,
loosen to `0.05` ≈ 3°).

### What it does and doesn't do

| Effect | Result |
|---|---|
| Locks head yaw near 0 (straight) during SONIC | yes |
| Removes head stiffness | no — head is still PD-held, just at center |
| Enables operator head control during teleop | no — would need a head bypass like wrists |
| Fixes head pitch (up/down) | no — firmware limitation |

