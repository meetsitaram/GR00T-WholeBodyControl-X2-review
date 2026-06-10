# 2026-06-08 — Pose-proxy fallback ladder: freeze arms on upstream stall

> **Session focus.** Stop the "violent arm reset to default-stand"
> safety bug observed during VR teleoperation on the real X2. A WiFi
> hiccup, a laptop GC pause, or a Cursor reload mid-session would cause
> the robot's arms to swing from the operator's commanded pose (arms
> up, in front of the table) to the SONIC default stand pose (arms
> hanging straight down at the shoulders) in ~200 ms. Strong enough to
> slam any object on the table on the way down. The fix is a staged
> `LIVE -> HOLD -> BLEND -> IDLE_CLIP` fallback ladder in the PC2
> pose proxy; the C++ deploy is untouched.

---

## TL;DR

| Symptom (before) | Cause | Fix |
|---|---|---|
| During VR teleop on real X2, a brief WiFi blip or laptop stall causes the robot's arms to swing through their full ROM to default-stand in ~200 ms, slamming tables/objects on the way down. | The PC2 [pose proxy](../../../../gear_sonic_deploy/scripts/x2_pose_proxy.py) immediately publishes baked `idle_stand` frames the moment the upstream wire goes silent past `--idle-stale-ms` (300 ms). The deploy's `target_lpf_hz=8 Hz` + `max_target_dev_arm=1.5 rad` clamps cannot absorb a multi-radian step in the commanded reference, so the arms swing through their full ROM in ~200 ms. The deploy's own "hold last good frame" safety net is bypassed because `--disable-pose-ref-watchdog` is forwarded whenever the proxy is enabled. | Replace the binary `LIVE/IDLE` fallback with a staged ladder: HOLD the last forwarded upstream frame byte-for-byte for `--hold-last-secs` (default 10 s), then BLEND its `joint_pos_mj` toward the baked idle clip over `--blend-secs` (default 3 s), then publish baked idle indefinitely. New `--idle-mode {blend,hold-last,idle-stand}` CLI knob (default `blend`). |
| No way to soak up brief upstream stalls (WiFi blip, laptop GC, Cursor reload) without changing the commanded reference at all. | Proxy had no notion of "this might be a transient gap, hold the wire alive without changing what you publish". | HOLD state re-publishes the LAST forwarded upstream BYTES verbatim. The deploy sees identical `joint_pos_mj` -> `jvel = 0` -> zero kinematic surprise. Soaks up `hold_last_secs` of upstream silence with no observable effect on the robot. |
| Operators have no quick way to revert to the legacy behaviour for diagnostics or to fall back to operator-responsibility mode (HOLD forever). | Behaviour was hard-coded. | `POSE_PROXY_IDLE_MODE` env var on [x2_pc2_daemons.sh](../../../../gear_sonic_deploy/scripts/x2_pc2_daemons.sh): `blend` (new default), `hold-last` (HOLD forever; operator owns recovery), `idle-stand` (pre-2026-06-08 behaviour; regression escape only). |

---

## State machine

```{mermaid}
stateDiagram-v2
    [*] --> COLDIDLE: never received upstream
    COLDIDLE --> LIVE: first upstream frame
    LIVE --> GAP: upstream silent &le; idle_stale_ms
    GAP --> LIVE: upstream resumes
    LIVE --> HOLD: upstream silent &gt; idle_stale_ms
    HOLD --> LIVE: upstream resumes (snap; no rollback)
    HOLD --> BLEND: still silent past hold_last_secs
    BLEND --> LIVE: upstream resumes (snap; no rollback)
    BLEND --> IDLECLIP: blend_secs elapsed
    IDLECLIP --> LIVE: upstream resumes
```

- **GAP** (`age <= idle_stale_ms`): proxy publishes NOTHING. The deploy's `ZmqPoseInputSource::Sample()` runs at 500 Hz and returns the cached last frame, so one missing 20 ms slice is invisible to the policy. Prevents bouncing between LIVE and fallback on every WiFi RTT jitter.
- **HOLD** (`idle_stale_ms < age <= idle_stale_ms + hold_last_secs`): re-publish the last forwarded upstream BYTES verbatim. Identical bytes -> identical `joint_pos_mj` -> deploy's finite-diff `jvel = 0`. No kinematic surprise; robot holds the operator's last pose.
- **BLEND** (`age <= idle_stale_ms + hold_last_secs + blend_secs`): lerp `joint_pos_mj` from cached upstream toward the baked idle clip. `alpha = (age - idle_stale_ms - hold_last_secs) / blend_secs`. Reuses the existing yaw-rebase + future-window machinery in [`build_idle_frame_msg`](../../../../gear_sonic_deploy/scripts/x2_pose_proxy.py) via a new `joint_pos_mj_override` kwarg. Future-window slots stay on the idle clip — they are an advisory horizon, not the immediate command.
- **IDLE_CLIP**: publish baked `idle_stand` frames indefinitely (legacy destination behaviour, reached gradually rather than in one tick).

---

## What landed

### Primary edit — [`gear_sonic_deploy/scripts/x2_pose_proxy.py`](../../../../gear_sonic_deploy/scripts/x2_pose_proxy.py)

1. **`decode_pose_joint_pos_mj`** — new tolerant decoder that extracts `joint_pos_mj` (f32, shape `(NUM_BODY_DOFS,)`) from a packed pose frame. Mirrors `decode_x2_debug_base_quat`: returns `None` on any decode failure rather than raising, so a malformed cached frame doesn't wedge the publish thread.

2. **`build_idle_frame_msg(..., joint_pos_mj_override=None)`** — optional override for the CURRENT frame's `joint_pos_mj`. Used by the BLEND path to substitute a lerp anchor while reusing the rest of the frame (root_quat with yaw rebase, motion token zeros, hand zeros, future window). Legacy callers (`override=None`) get bit-identical output, pinned by [`test_build_idle_frame_msg_override_none_preserves_legacy`](../../../../tests/test_x2_pose_proxy_fallback_ladder.py) and the existing [`test_build_idle_frame_msg_without_rebase_emits_baked_yaw`](../../../../tests/test_x2_pose_proxy_yaw_rebase.py).

3. **`decide_fallback_state(...)`** — pure decision function returning `(target_state, blend_alpha)`. Split out so the state-machine logic is unit-testable without ZMQ sockets or real-time clocks. The "fallback clock" starts at `stale_s`: HOLD runs for `hold_last_secs`, then BLEND for `blend_secs`, then IDLE_CLIP forever.

4. **Main loop** — replaces the binary `in_idle` flag with the state machine above. Caches `last_upstream_msg` (bytes, for HOLD re-publish) and `last_upstream_jpos` (decoded array, for BLEND lerp anchor) on every fresh-forward. State transitions emit one-line log markers operators can grep for:

   ```
   [pose_proxy] state: LIVE -> HOLD (re-publishing last upstream frame; will hold for 10.0s)
   [pose_proxy] state: HOLD -> BLEND (lerping cached -> idle_stand over 3.0s)
   [pose_proxy] state: HOLD -> LIVE (upstream pose frames flowing again after 4231 ms gap)
   ```

5. **CLI knobs**: `--idle-mode {blend,hold-last,idle-stand}` (default `blend`), `--hold-last-secs FLOAT` (default `10.0`), `--blend-secs FLOAT` (default `3.0`).

6. **Status print**: extended to show `HOLD t=4.2/10.0s` or `BLEND alpha=0.41` so operators can read the fallback timer at a glance from the periodic line. Adds `hold` / `blend` / `gap_skip` counters next to the existing `fwd` / `idle` counters.

### Secondary edit — [`gear_sonic_deploy/scripts/x2_pc2_daemons.sh`](../../../../gear_sonic_deploy/scripts/x2_pc2_daemons.sh)

Three new env vars with sane defaults (override-able from the start-line via `--pose-proxy-idle-mode`, `--pose-proxy-hold-last-secs`, `--pose-proxy-blend-secs`):

```sh
POSE_PROXY_IDLE_MODE="${POSE_PROXY_IDLE_MODE:-blend}"
POSE_PROXY_HOLD_LAST_SECS="${POSE_PROXY_HOLD_LAST_SECS:-10.0}"
POSE_PROXY_BLEND_SECS="${POSE_PROXY_BLEND_SECS:-3.0}"
```

The `pose proxy: ENABLED` banner the wrapper prints on every `start` now shows the active mode:

```text
[pc2 09:01:22] pose proxy: ENABLED
[pc2 09:01:22]     upstream    = tcp://192.168.86.22:5556
[pc2 09:01:22]     downstream  = tcp://localhost:5558 (deploy SUBs here)
[pc2 09:01:22]     idle x2m2   = /home/run/getsolo/data/idle_stand.x2m2
[pc2 09:01:22]     stale_ms    = 300
[pc2 09:01:22]     idle_mode   = blend (HOLD 10.0s, BLEND 3.0s)
```

`idle-stand` mode prints a WARN-level line in the banner so the regression escape is visible in every session log.

### Tests — [`tests/test_x2_pose_proxy_fallback_ladder.py`](../../../../tests/test_x2_pose_proxy_fallback_ladder.py)

20 new tests covering:

- `decode_pose_joint_pos_mj` round-trip and tolerance (`wrong topic`, `truncated`, `wrong shape`).
- `decide_fallback_state` for all six states (COLD_IDLE, GAP, HOLD, BLEND, IDLE_CLIP, plus LIVE via the caller). Includes `hold-last` mode (HOLD forever), `idle-stand` mode (pre-2026-06-08 regression escape), edge cases (`hold_last_secs=0`, `blend_secs=0`, alpha clamping).
- `build_idle_frame_msg(joint_pos_mj_override=...)` round-trip: override appears in the published frame's `joint_pos_mj` bit-for-bit, monotonic lerp from cached anchor to idle, legacy `override=None` is byte-identical to default call.
- HOLD path byte-identical regression test (catches any future refactor that accidentally round-trips the cached frame through decode/re-pack).

All 38 proxy tests pass (20 new + 18 existing yaw-rebase tests, no regressions).

---

## How to revert / harden

| Goal | Setting |
|---|---|
| New 2026-06-08 default (HOLD 10 s + BLEND 3 s + IDLE_CLIP). | `POSE_PROXY_IDLE_MODE=blend` (the default). |
| Operator-responsibility mode: HOLD the last upstream frame forever. Robot stays in commanded pose until operator cuts power or upstream returns. Use when you know upstream WILL come back (e.g. live VR teleop where a stale wire == operator already walked away from the headset). | `POSE_PROXY_IDLE_MODE=hold-last` |
| Pre-2026-06-08 behaviour. Snap to default-stand on the first stale tick. KNOWN to slam arms into tables during WiFi hiccups. Regression escape only. | `POSE_PROXY_IDLE_MODE=idle-stand` |
| Longer hold window for laptops with long GC pauses. | `POSE_PROXY_HOLD_LAST_SECS=30.0` |
| Snappier blend for less startled-looking glide. | `POSE_PROXY_BLEND_SECS=1.5` |

Pass any of these as env vars to `x2_pc2_daemons.sh start` (no flag changes needed), or use the new `--pose-proxy-idle-mode` / `--pose-proxy-hold-last-secs` / `--pose-proxy-blend-secs` flags.

---

## How to verify on the robot

1. Start the daemons with the new default:
   ```sh
   ./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
       --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
       --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
       --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
       --lock-head-straight
   ```
   Confirm in the banner that `idle_mode = blend (HOLD 10.0s, BLEND 3.0s)`.

2. Start a normal VR teleop session, move the arms into a non-default pose (e.g. arms up in front of the table).

3. Simulate a WiFi blip by suspending the laptop publisher for ~1 s:
   ```sh
   # In the planner stack tmux pane, hit Ctrl-Z to SIGSTOP, then `fg` to resume.
   ```
   In the pose proxy log (`logs/pc2/.../pose_proxy_*.log`), expect:
   ```
   [pose_proxy] state: LIVE -> GAP (upstream silent < 300 ms; holding deploy cache)
   [pose_proxy] state: GAP -> LIVE (upstream pose frames flowing again after ... ms gap)
   ```
   The arms should NOT move during this window — the deploy is still tracking the cached frame from its 500 Hz Sample() loop.

4. Simulate a longer outage (~3-5 s):
   ```sh
   # SIGSTOP the publisher, wait 3-5 s, then `fg` to resume.
   ```
   Expect:
   ```
   [pose_proxy] state: LIVE -> HOLD (re-publishing last upstream frame; will hold for 10.0s)
   [pose_proxy] state: HOLD -> LIVE (upstream pose frames flowing again after ... ms gap)
   ```
   The arms should remain frozen in the operator's last pose for the entire outage.

5. Simulate a full upstream loss (>13 s, hits HOLD then BLEND then IDLE_CLIP):
   ```sh
   # Kill the laptop publisher entirely.
   ```
   Expect a slow drift to default-stand over ~3 s, NOT a slam:
   ```
   [pose_proxy] state: LIVE -> HOLD ...
   [pose_proxy] state: HOLD -> BLEND (lerping cached -> idle_stand over 3.0s)
   [pose_proxy] state: BLEND -> IDLE_CLIP ...
   ```

---

## Out of scope (intentionally)

- **Closed-loop IK feedback from measured joints** — the VR IK on the laptop is still open-loop (uses its own previous output as the seed instead of the robot's measured joint positions). A separate, much bigger change. This milestone only fixes the snap-to-default symptom.
- **PD gain rework** (`kp_scale_shoulder/elbow/wrist`) — deferred per earlier convo. The arms can still under-track aggressive operator IK on the way to the held pose; this milestone just prevents the snap-back.
- **C++ deploy changes** — the deploy already does the right thing when its watchdog is disabled (holds last good frame). The fix lives entirely in the Python wire layer that sits in front of it.
- **Recorder idle-publish path** — `x2_dataset_recorder.py` has its own `idle_publish_enabled` fallback, but it only fires BEFORE the first body_pose arrives at startup, so it's not in play for mid-session hiccups.

## What to revisit next

- **Tune `hold_last_secs` from telemetry**. 10 s is the conservative default; on a stable WiFi link the proxy will almost never need >2 s of HOLD. Worth instrumenting "HOLD entry / exit" pairs in `x2_freeze_postmortem.py` so we can compute the actual hold-time distribution and tighten the default to the 99th percentile observed in production.
- **Slerp the root_quat during BLEND**. Currently BLEND lerps `joint_pos_mj` but lets `build_idle_frame_msg` rebuild `root_quat` from the idle clip's R_z(0) + measured yaw. Over 3 s with a static measured yaw this is harmless, but if the robot is leaning during the blend the small pitch/roll mismatch could induce a one-time tracking jolt at BLEND -> IDLE_CLIP boundary. Slerping the cached quat -> idle quat would smooth that out; cost is a small scipy or hand-written slerp helper. Defer until we actually see the symptom.
