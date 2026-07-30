# Forward obstacle guard

Stops the X2 walking into things. A LiDAR-derived forward wedge is watched for
returns closer than a threshold; while blocked, forward and lateral velocity are
zeroed at the `planner_cmd` ingest so the gait controller walks the robot to a
controlled stand. Yaw stays live.

Built because this unit has **no SLAM/mapping module installed** — the vendor
`pnc` nav app needs a saved map plus relocalization, and `MappingService` /
`RelocalizationService` are advertised over the AimRT→ROS2 bridge but no binary
serves them on any of the three boards. Full nav2 was not an option; a reactive
guard was.

---

## Design

```
LiDAR PointCloud2 (10 Hz)
  → pointcloud_to_laserscan  → /scan (LaserScan)
    → scan_guard.py          → blocked / distance
      → scan_guard_pub.py    → ZMQ PUB tcp://127.0.0.1:5571
        → pc2_kplanner_onnx.py  clamps stick_fwd / stick_side
```

### Why the clamp lives in kplanner

kplanner **binds** `planner_cmd` on 5563 (`--cmd-bind`). Every input source —
pad bridge, VR, anything else — publishes into that one socket. Clamping at the
ingest in `_zmq_command_thread` covers all of them; clamping in the pad bridge
would have covered only the pad.

The clamp handles both payload shapes: stick values *and* `direct_velocity`,
which VR-style inputs send instead of sticks.

### Why the guard is a separate process

kplanner and `pad_locomotion_bridge.py` run in the `gear_sonic` venv, which has
no `rclpy`. Importing ROS into the gait process was not worth it — and an early
attempt at a hard `import scan_guard` at module scope killed the bridge outright
on startup. The guard runs under system python and ships a boolean over ZMQ.

### Fail-open, deliberately

If the guard process dies or its data goes stale (>0.5 s), `_guard_blocked()`
returns `False` and the planner behaves exactly as before. A guard that freezes
a walking biped because its own sensor died is worse than no guard — the
operator deadman is the real stop. The tradeoff is that a silently dead guard
means unguarded driving, which is why it is started by the ritual and logged.

### Latching

Once tripped, forward stays dead until the operator releases the deadman (the
bridge sends an all-zero frame, which resets the latch). Auto-release on a clear
path would let the robot resume walking without a human deciding to.

---

## Files

| File | Role |
|---|---|
| `scan_guard.py` | `/scan` subscriber; `blocked()` / `distance()` |
| `scan_guard_pub.py` | Wraps the above, PUBs on 5571, speaks the warning |
| `watch_guard.py` | Terminal monitor for the ZMQ feed |
| `obstacle.wav` | espeak-ng rendering of "Object in front of me" |
| `pc2_kplanner_onnx.py` | Clamp + latch at `planner_cmd` ingest |
| `ritual_start_demo.sh` | Starts lumi (scan) → kplanner → guard |
| `lumi.sh` | Owns the PointCloud2 → LaserScan converter |

---

## Tuning

```python
STOP_M         = 1.6     # block below this
CLEAR_M        = 1.6     # release at/above this (raise for hysteresis)
ARC_DEG        = 25.0    # half-angle of the forward wedge
ARC_CENTER_DEG = -58.0   # MEASURED, not assumed — see below
MIN_HITS       = 4       # beams needed; rejects single stray returns
SELF_M         = 0.25    # closer than this is the robot's own body
```

### ARC_CENTER_DEG is measured, not derived

The scan is produced in `base_link`, and "forward" does **not** land at 0°. A
person standing 0.6 m directly in front read **-58°**. The whole usable scan
spans roughly -90° to -20°; everything else is `inf`.

Re-measure after any change to the converter's `target_frame`, and verify from
a second robot heading — the offset was captured in one position and a guard
pointed 58° off to the side is worse than none, because you would trust it.

```bash
cd /home/run/getsolo && python3 - <<'EOF'
import math, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
q = QoSProfile(depth=5); q.reliability = ReliabilityPolicy.BEST_EFFORT
rclpy.init(); n = Node("dbg"); got = []
n.create_subscription(LaserScan, "/scan", lambda m: got.append(m), q)
while not got: rclpy.spin_once(n, timeout_sec=1)
m = got[-1]
best = min(((r, math.degrees(m.angle_min + i*m.angle_increment))
            for i, r in enumerate(m.ranges) if math.isfinite(r)),
           key=lambda t: t[0])
print(f"nearest return: {best[0]:.2f}m at {best[1]:.0f}deg")
EOF
```

Stand directly in front while it runs; the reported bearing **is** forward.

### Arc width

`ARC_DEG = 180` was tried and reverted: it pulls in side returns and the
robot's own arms, so reported distance collapsed from ~2.2 m to ~0.8 m with
nothing ahead. Forward-only is also correct behaviour — an obstacle 90° to the
side should not stop forward walking.

An early `ARC_DEG = 40` latched permanently on a stray 0.75 m return at -15°,
probably an arm in the scan plane. `MIN_HITS` and a narrower arc fixed it.

---

## Speech

The onboard TTS is a **Chinese-only** Volcano (ByteDance) voice. English is read
phonetically — "Hi" comes out as "hei". None of the following changed that:

- `voice_id` 0–7 (`SetTtsParameters`) — all the same engine
- `voice_id` in `PlayTtsRequest` — commented out of the `.msg`
- `SetRobotLanguage("en")` — returns success, no effect
- `voice_type` in **both** copies of `tts_config.conf`
  (`agent/bin/cfg/` and `agent_bin/cfg/`) set to `en_female_anna_mars_bigtts`,
  caches cleared, agent restarted — still Chinese

So the warning is a pre-rendered WAV played through `PlayAudioFile`:

```bash
espeak-ng -w obstacle.wav "Object in front of me"
scp obstacle.wav run@10.0.1.42:/agibot/data/home/agi/audio/
```

The file must live on **SOC2 (10.0.1.42)**, which owns the audio hardware —
playing a path that exists only on SOC1 returns `SUCCESS` and is silent. Avoid
`/tmp`; it clears on reboot and the speech dies quietly.

Spoken on the clear→blocked edge only, with a cooldown, so standing in front of
the robot does not loop it.

---

## Gotchas

**Environment.** `rclpy` on this robot lives in
`/opt/ros/humble/local/lib/python3.10/dist-packages` (not `site-packages`), and
`rpyutils` lives in `site-packages`. Both are needed. A fresh post-reboot login
shell has no ROS env at all, so paths are spelled out absolutely in the ritual
rather than appended to `$PYTHONPATH`.

**tmux session reuse.** `start_tmux` skips a session that already exists. After
editing any of these files, `tmux kill-session -t <name>` or the old process
keeps running and the change appears to do nothing. This wasted real time during
development — the bridge ran unpatched for several test cycles.

**`--once` and lazy subscription.** `pointcloud_to_laserscan` only subscribes to
the cloud once something subscribes to `/scan`. `ros2 topic echo /scan --once`
disconnects before the first real scan, so it reports all-`inf` and looks broken
when it is fine. Use a persistent subscriber to check.

**QoS.** Sensor topics are `BEST_EFFORT`. RViz defaults to `RELIABLE` and shows
"0 points from 0 messages" until Reliability is switched under the Topic
dropdown.

**Two TF trees.** `robot_state_publisher` (URDF) and the vendor `joint_tf` both
publish. Static transforms carry timestamp 0, so a lookup at the cloud's stamp
can fail; `transform_tolerance:=1.0` covers it.

---

## Verify

```bash
tmux ls                                    # want: scan_guard
grep -ai "scan guard" log/pc2_kplanner.log # want: SUB tcp://127.0.0.1:5571
python3 watch_guard.py                     # want: a real distance, not inf
```

Then, stationary, with an obstacle in place: hold the deadman, push forward, and
watch for `OBSTACLE ... LATCHED` in `log/pc2_kplanner.log` with no motion.
Release the deadman to reset.

Test against something soft before anything else. The clamp stops the *command*;
the gait still finishes its current step, so there is overshoot at speed. The
1.6 m threshold absorbs it with room to spare.
