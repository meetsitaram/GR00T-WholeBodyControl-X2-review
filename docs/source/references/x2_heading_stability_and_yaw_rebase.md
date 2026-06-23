# X2 heading stability & `root_quat` yaw rebase — subsystem reference

How the X2 stack keeps the robot facing the **physically-measured**
heading instead of "world +X" (= the heading the C++ deploy boots
with), what goes wrong when it doesn't, and the patterns the codebase
uses to keep each laptop-side wire publisher honest.

This page is the long-form companion to a recurring class of bugs the
operator perceives as:

> The robot snaps back to its boot orientation. On VLA start it
> rotates to spawn heading; mid-session it resists manual rotation;
> after a wifi blip it twists back. Sometimes there's a loud
> waist-yaw click at startup.

The symptom keeps coming back because the underlying contract is
subtle and four different processes implement it independently —
three on the laptop and one on PC2 — each with its own freshness
gate, fallback policy, and degree of robustness. This doc inventories
all four, the patterns they use, the deploy-side bootstrap escape
hatch, the diagnostic tooling, and the historical timeline of bugs
and fixes. Read this **before** patching any of them again.

Related references:

- [`X2 Deploy ZMQ Wire Protocol`](x2_zmq_protocol.md) — the `pose` /
  `x2_debug` envelopes and the `root_quat_xyzw` / `base_quat` fields
  this page revolves around.
- [`X2 VLA motion_token Decoder`](x2_vla_motion_token_decoder.md) —
  the bridge-side wire publisher that this page's most recent fix
  modified.
- [`X2 SONIC Runtime Architecture`](x2_sonic_runtime_architecture.md)
  — which laptop process publishes `pose` in each mode (teleop /
  record / VLA).
- [`X2 Split-topology PC2 daemons`](x2_split_deploy_pc2.md) — the
  watchdog / proxy daemons on PC2 that consume the laptop's wire.

Milestones with point-in-time wrap-ups:

- [2026-06-07 — VLA bridge heading stability + head-lock](../user_guide/milestones/2026-06-07_vla_bridge_heading_stability_and_head_lock.md)
- [2026-06-23 — VLA bridge holds last-good live `root_quat` across `x2_debug` stalls](../user_guide/milestones/2026-06-23_vla_bridge_yaw_hold_last_good.md)

---

## 1. TL;DR

* The wire's `root_quat_xyzw` (in the `pose` envelope, port 5556) is
  the **orientation reference** the C++ deploy's tokenizer compares
  the measured IMU pose against. If the wire ships
  `R_z(measured_yaw)` (= the IMU yaw projected onto world Z), the
  policy sees zero orientation error and holds heading. If the wire
  ships identity (`(0,0,0,1)` = world +X = the C++ deploy's boot
  frame), the policy sees a real error and **commands the body
  toward yaw=0** — snap-back.
* The IMU source is `x2_debug:5557` (PUB on PC2, packed-binary,
  contains `base_quat`). Every laptop-side wire publisher subscribes
  to it directly OR through the
  [`x2_debug_to_robot_pose_bridge.py`](#5-the-shared-piece-x2_debug_to_robot_pose_bridge)
  republisher.
* The right pattern is **hold last good**: once an `x2_debug` frame
  has ever arrived, keep using its `base_quat` (or, more precisely,
  the cached yaw extracted from it) across arbitrarily long stalls
  rather than reverting to identity. A stale cached yaw is strictly
  better than identity because identity is a **known-wrong**
  orientation that drags the body back to world +X on every tick;
  a stale yaw is **possibly-stale** but usually still close enough
  for the few hundred milliseconds before the stream recovers.
* The codebase uses **two architectural patterns** to implement
  this (`§6`): hold-last-good (kplanner, VLA bridge as of
  2026-06-23) and cache-with-max-age (PC2 watchdog). The recorder
  still uses an older identity-fallback pattern and has a
  documented dormant bug.
* If you see a regression, start with the diagnostic
  toolkit in [§8](#8-diagnostics) — `x2_yaw_click_sniffer.py` and
  the bridge log STALE / RECOVERED edges will tell you which
  publisher is misbehaving in <30 s.

---

## 2. The orientation reference contract on the wire

The `pose` envelope (full schema in
[`x2_zmq_protocol.md`](x2_zmq_protocol.md)) carries an xyzw
`root_quat_xyzw` field that the C++ deploy treats as the
**SONIC tracker's orientation reference** for the body. Internally
the deploy's `BuildTokenizerObs` (`x2_deploy_onnx_ref/src/tokenizer_obs.cpp`)
computes

```
rel_quat = inverse(measured_base_quat) * root_quat_xyzw_from_wire
```

and feeds `rel_quat` (plus the rest of proprio) into the SONIC
encoder. The policy then produces actions that drive the body to
make `rel_quat` approach identity — i.e., to align the **measured**
heading with the **commanded** reference.

The convention used everywhere on the laptop side is **yaw-only**:
each publisher extracts a single yaw scalar from the IMU
`base_quat`, drops pitch / roll (so a transient lean from a fall
recovery doesn't bleed into the wire), and ships
`R_z(yaw) = (0, 0, sin(yaw/2), cos(yaw/2))` as `root_quat_xyzw`.

### 2.1 What "identity" means and why it's catastrophic

`(0, 0, 0, 1)` represents the C++ deploy's boot frame, which is
"robot facing world +X at the moment SONIC started." The deploy
never resets this — across teleop / VLA / handoff cycles, the
boot frame is a fixed direction in the room. If the wire ships
identity while the robot is physically facing some other heading
Y, the tracker sees a `(0 - Y)` orientation error and twists the
body to Y = 0 via `waist_yaw_joint` (MJ slot 12, the dominant
heading-correction effector). Visible as:

- a sharp turn at startup,
- a sustained ~33° steady-state yaw drift in `manipulation` mode
  (the idle-clip waist_yaw offset, see
  [§3.3](#33-the-waist_yaw-pin-as-a-mitigation-not-a-fix)),
- an audible **waist-yaw click** at gesture transitions
  (the `x2_yaw_click_sniffer.py` script was built specifically to
  catch this).

### 2.2 Why the laptop has to know the IMU yaw at all

Conceptually the C++ deploy already has the IMU — couldn't it
just substitute the measured quat for the wire's
`root_quat_xyzw` when it sees identity? It does, but only at
boot: the bootstrap-safe override in
`x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp:2608-2625` fires
when `ZmqPoseInputSource::LastReceivedMonotonicS() < 0` (no real
frame received yet) and uses the measured `base_quat` as the
reference. The escape hatch closes as soon as the laptop sends
its first real frame — every subsequent identity frame is taken
at face value. **The deploy cannot tell "stale identity from a
laptop bug" apart from "operator intentionally commanded
yaw=0".** That's why the laptop side has to ship the right
reference every tick.

---

## 3. The IMU source and what stalls look like

### 3.1 `x2_debug:5557`

The C++ deploy publishes `x2_debug` at 50 Hz on PC2 with topic
`x2_debug` (packed-binary, 1280 B JSON header + binary fields).
The fields relevant to heading stability are:

| Field | Type | Used by |
|---|---|---|
| `base_quat` | f64 × 4 (wxyz) | All four yaw-rebase consumers below |
| `body_q` | f64 × 31 (MuJoCo order) | Bridge's waist_yaw pin (`body_q[12]`) |
| `body_dq` | f64 × 31 | Bridge's tracking feedback (unrelated) |
| `base_ang_vel` | f64 × 3 | Bridge's proprio buffer (unrelated) |

Decoder: `gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder.unpack_message`.

### 3.2 What a stall looks like

`x2_debug` can go silent on the laptop SUB for many reasons:

- WiFi packet drops between PC2 and the laptop (~5–100 ms blips
  are routine; >1 s blips happen on busy 2.4 GHz channels).
- PC2 RT-priority preemption (long GC, RT-kernel preemption,
  storage flush) → the C++ deploy publishes nothing for a
  burst window.
- Bridge-side GPU stalls — the VLA bridge's inference thread can
  hold the GIL during a slow inference and starve the SUB
  drainer.
- The deploy is restarted / killed → `x2_debug` goes silent
  permanently until the next `x2_pc2_daemons.sh start`.

Each laptop-side subscriber has to decide what to put on the
wire while `x2_debug` is silent. **That decision is the entire
substance of this page.**

### 3.3 The waist_yaw pin as a mitigation, not a fix

Even when the wire `root_quat_xyzw` is correct, the bridge's
`manipulation` body mode freezes the legs + waist joint targets
to the `idle_stand` clip (this is intentional — the clip's
jitter is what the SONIC tracker was trained against; remove it
and the body sags 25° under gravity within seconds). But the
clip's frame-0 `waist_yaw` is **~33° off `DEFAULT_STAND_POSE`**,
so freezing waist_yaw to the clip drives a ~33° steady-state
heading drift on top of any rebase error. The bridge ships a
**surgical waist_yaw pin** (`live_vla_publish_motion_token.py`
Section F, ~line 3345) that overwrites only slot 12 with the
live measured value, leaving the other frozen DOFs on the clip.

The waist_yaw pin shares the same `x2_debug` freshness gate as
the yaw-rebase (and therefore the same hold-last-good fix — see
[§4.2](#42-vla-bridge-live_vla_publish_motion_tokenpy)).

---

## 4. The four laptop / PC2 consumers

Whichever mode the operator is running, exactly one of these four
processes is responsible for the wire's `root_quat_xyzw` at any
given tick. They implement the contract independently; the table
below is the canonical inventory.

| # | Process | Mode it owns | Source of `base_quat` | Pattern | Stall behaviour | Status |
|---|---|---|---|---|---|---|
| 1 | `gear_sonic/scripts/x2_kplanner.py` | VR teleop, recording | `pose_deque` from `robot_pose:5570` (republished from `x2_debug` by the bridge daemon below) | **Hold last good** | `current_root_wxyz` is left at last value; refresh is skipped when `age > pose_feedback_max_age_s` (default 0.5 s) | ✅ Correct since 2026-06-01 |
| 2 | `gear_sonic/scripts/live_vla_publish_motion_token.py` | Autonomous VLA | Direct SUB on `x2_debug:5557` via `_LatestState` | **Hold last good** (was identity-fallback before 2026-06-23) | `last_known_base_quat_wxyz` cache; rebase + waist_yaw pin keep firing across arbitrary stalls | ✅ Correct since 2026-06-23 |
| 3 | `gear_sonic/utils/teleop/x2_dataset_recorder.py` | Recording (idle frames) | Direct SUB on `x2_debug:5557` via `_LatestState` | **Identity fallback** (returns `None` from `_compute_idle_root_quat_xyzw` → `_publish_pose` uses identity) | Wire reverts to identity within 1 s of `x2_debug` silence | ⚠️ Dormant bug (`§7`) |
| 4 | `gear_sonic_deploy/scripts/x2_pose_watchdog.py` | PC2 idle-clip fallback | Local SUB on `x2_debug:5557` (loopback on PC2) | **Cache with max-age** (configurable `--x2-debug-max-age-s`, default **0.5 s**) | Reverts to identity if cache > 0.5 s old; tight default but acceptable here because the watchdog is itself the fallback ladder (`§4.4`) | ✅ Acceptable for its role |

The next subsections describe each in detail.

### 4.1 Kplanner — `gear_sonic/scripts/x2_kplanner.py`

The kplanner is the wire publisher during VR teleop and recording.
It maintains a `current_root_wxyz` quaternion that represents
"where the body is facing right now from the kplanner's POV" and
republishes it on every wire tick.

The pattern is **three layered yaw refreshes**, all gated on
`pose_feedback_max_age_s` (default 0.5 s, set via
`--pose-feedback-max-age-s`):

1. **Startup yaw seed** (one-shot, after the warmup PKL):
   ```
   if age_s <= float(pose_feedback_max_age_s):
       current_root_wxyz = _yaw_only_wxyz_from_pelvis(
           latest_seed_obs.pelvis_qpos_wxyz
       )
   ```
   Else: `current_root_wxyz` stays at the warmup PKL's frame-0 quat
   (= identity in practice). This window is bounded by the warmup
   length (`warmup_quiet_stand_s * OUTPUT_FPS` frames) and is the
   only place where the kplanner could publish identity.
2. **IDLE_LOOP yaw refresh** (every tick while in `IDLE_LOOP`).
3. **IDLE→PLAYING transition seed** (single shot at the state edge,
   ensures a clean handoff into walking).

In all three: the refresh is **skipped** if `age_s > max_age_s`;
`current_root_wxyz` retains its previous value. The publish
loop then writes `current_root_wxyz` to `qpos[3:7]` on every
wire frame, so a stale `pose_deque` keeps the wire heading
honest indefinitely. The comment at `x2_kplanner.py:1046-1048`
calls this out:

> snap-back protection still comes from the IDLE_LOOP yaw refresh and
> the new IDLE → PLAYING transition seed (both yaw-only, both writing
> to `current_root_wxyz` only, never to the model's neural buffer).

**This is the canonical hold-last-good shape.** It's what the
VLA bridge fix mirrors.

### 4.2 VLA bridge — `gear_sonic/scripts/live_vla_publish_motion_token.py`

The VLA bridge subscribes directly to `x2_debug:5557` (no
`robot_pose` middleman) into `_LatestState`. The pattern (since
2026-06-23) is:

1. **Cache**: every tick that reports `deploy_fresh = True`
   (alive check, 1 s `DEPLOY_ALIVE_STALE_THRESHOLD_S`), copy
   `base_quat_wxyz` + `body_q_mj` into module-local
   `last_known_*` variables.
2. **Resolve**: a pure helper `_resolve_wire_rebase_source`
   returns a `_WireRebaseSource(source ∈ {"live", "cached",
   "none"}, base_quat_wxyz, body_q_mj, cache_age_s)`. Prefers
   the live snapshot when fresh, falls back to the cache when
   not, returns `"none"` only when `x2_debug` has never arrived.
3. **Apply**: Section H yaw-rebase and Section F waist_yaw pin
   both gate on `rebase_source.base_quat_wxyz is not None`
   (rather than `deploy_fresh`), so they keep firing across
   arbitrary stalls.
4. **Log**: `_log_rebase_source_transition` emits one line per
   edge:
   ```
   [live-VLA] root_quat yaw-rebase ACTIVE: ... (yaw=+45.0deg)
   [live-VLA] root_quat yaw-rebase STALE: x2_debug silent >1.0s; holding cached base_quat (cache age=2503ms). Wire will NOT revert to identity (no snap-back to spawn heading).
   [live-VLA] root_quat yaw-rebase RECOVERED: x2_debug back online; resuming live yaw tracking.
   ```
   Same-state ticks are silent so a 50 Hz publish loop never
   spams.

A separate **bootstrap-safe publish gate** (~line 3572) keys on
`state.received_any` and withholds the first publish until any
`x2_debug` frame has arrived. While the gate is closed, the
C++ deploy's own measured-quat bootstrap override carries the
reference (no laptop-side identity-quat phantom can be
published).

Tests pinning the contract: `tests/test_live_vla_bridge_yaw_hold_last_good.py`.

### 4.3 Recorder — `gear_sonic/utils/teleop/x2_dataset_recorder.py`

The recorder publishes wire frames during recording, but in
**relay mode** (passing through the kplanner's `body_pose:5565`
with merged arms / hands) for the vast majority of ticks. Its
own `_compute_idle_root_quat_xyzw` (`x2_dataset_recorder.py:3354-3438`)
only generates a `root_quat_xyzw` when the recorder is producing
**idle frames** itself — gesture gaps, idle-stand fallback,
gestures with no kplanner upstream.

The pattern is the **legacy identity-fallback shape**: gates on
`_LatestState.snapshot()`'s `alive` bool and returns `None`
(which `_publish_pose` translates to identity) when not alive.
Two log lines (`ACTIVE` / `stale`) are gated by
`_idle_yaw_rebase_logged_active` / `_idle_yaw_rebase_logged_fallback`
booleans so the 50 Hz loop doesn't spam.

This is the **dormant bug** noted in [§7](#7-known-dormant-issues).
It hasn't been operator-visible in a long time because during
normal VR teleop the kplanner is always publishing
`body_pose:5565`, so the recorder is in relay mode and the
idle-frame path is cold. If the recorder ever becomes the sole
wire publisher again during a long session with `x2_debug`
stalls, the snap-back returns.

Tests pinning the current (buggy-but-shipping) shape:
`tests/test_recorder_idle_yaw_rebase.py`.

### 4.4 PC2 watchdog — `gear_sonic_deploy/scripts/x2_pose_watchdog.py`

The PC2 watchdog is the **fallback ladder** that sits between
the laptop's wire and the C++ deploy: `bridge → mux → watchdog
→ deploy`. When the upstream wire is alive, it forwards bytes
verbatim. When the upstream stalls past `--idle-stale-ms`, it
walks a staged ladder: `LIVE → HOLD (last bytes) → BLEND →
IDLE_CLIP (idle_stand baseline)`. See
[`2026-06-08_arm_freeze_on_upstream_stall.md`](../user_guide/milestones/2026-06-08_arm_freeze_on_upstream_stall.md)
for the full ladder design.

The watchdog has its OWN `x2_debug` SUB (local on loopback, since
the watchdog runs on PC2 next to the deploy) and rebases the
**idle clip's** `root_quat_xyzw` to the live measured yaw via
the cache-with-max-age pattern at lines 460-615:

```python
yaw_max_age_s = float(args.x2_debug_max_age_s)
last_measured_yaw_rad = 0.0
last_measured_yaw_s = -1.0
...
# On every x2_debug arrival:
last_measured_yaw_rad = yaw_from_quat_wxyz(base_quat_wxyz)
last_measured_yaw_s = now
...
# When generating an idle frame:
if (yaw_track_enabled
    and last_measured_yaw_s >= 0
    and (now - last_measured_yaw_s) <= yaw_max_age_s):
    yaw_rebase = last_measured_yaw_rad
```

Default `--x2-debug-max-age-s = 0.5 s`. Tighter than the
kplanner / bridge's hold-last-good shape, which means a >0.5 s
stall reverts the watchdog's idle clip to identity yaw.

**Why that's acceptable here:** the watchdog is **already** in
the fallback ladder by the time its idle frames matter. If the
upstream is healthy, the watchdog forwards bytes (and the yaw
rebase only applies to its `IDLE_CLIP` / `BLEND` states). If
the upstream is gone for 10+ seconds, the wire heading is
already a secondary concern — the operator should be looking at
the deploy's pose-ref starvation watchdog (`PoseRefStarvationWatchdog`,
trips to `SAFE_IDLE` at `--pose-ref-stale-s`, default 0.5 s)
rather than expecting clean rebase. The 0.5 s default is a
deliberate "fail loud" stance.

Killing the yaw track entirely is also a documented operator
choice via `--no-x2-debug-yaw-track` (the watchdog's CLI flag),
useful when debugging whether the rebase itself is at fault.

---

## 5. The shared piece — `x2_debug_to_robot_pose_bridge.py`

The kplanner consumes pose feedback via the `robot_pose` topic on
`localhost:5570`. On the **real robot**, that topic doesn't exist
natively on the laptop — `robot_pose` is normally produced by the
MuJoCo sim bridge (`x2_mj_bridge.py`) which writes full pelvis
pose including XY/Z. On real robot, the laptop has no XY/Z source
and there's no native `robot_pose` producer.

The republisher daemon
[`gear_sonic_deploy/scripts/x2_debug_to_robot_pose_bridge.py`](../../../gear_sonic_deploy/scripts/x2_debug_to_robot_pose_bridge.py)
plugs this gap. It SUBs PC2's `x2_debug:5557` over the network,
extracts `base_quat`, and re-PUBs as `robot_pose` JSON on
`localhost:5570` (XY/Z = 0 since the IMU has no position
measurement; downstream consumers only use the quat for yaw).

The daemon's docstring captures the original symptom that
motivated it:

> Without this bridge, real-robot deployments have NO measured-yaw
> source on the laptop, so the kplanner boots
> `current_root_wxyz = R_z(0)` from its warmup PKL and the very
> first frame it publishes hands the C++ deploy a stale
> identity-yaw reference. The deploy then twists the body back
> to world +X. This is the **"robot turns back to default
> orientation as soon as I start the VR planner stack"** symptom.

The VLA bridge does NOT use this daemon — it has its own direct
`x2_debug` SUB. The launchers (`run_x2_quest3_planner_stack.sh`
and `run_x2_vla_runtime.sh`) spawn the daemon conditionally
based on whether the laptop needs `robot_pose` for kplanner /
recorder consumption.

Port note: `robot_pose:5570` collides with VLA's takeover-mode
internal port (the bridge auto-shifts to `5571` when
`--enable-takeover` is set); see
[`2026-06-11_pose_mux_split.md`](../user_guide/milestones/2026-06-11_pose_mux_split.md)
for the topology decision.

---

## 6. The two architectural patterns

| Pattern | Where it lives | Bound | Pros | Cons | Right call when |
|---|---|---|---|---|---|
| **Hold last good (indefinite)** | Kplanner, VLA bridge (since 2026-06-23) | Cache is one-way sticky from the first `x2_debug` frame; no max-age cutoff | Wire never reverts to identity once we've ever measured a yaw. Survives arbitrary stalls. Identity is a known-wrong reference; cached is at worst stale. | If the IMU is genuinely broken (frozen quat), we'd never know from this path alone. Mitigation: rely on the `PoseRefStarvationWatchdog` in the C++ deploy to catch a frozen upstream. | The publisher is the **live consumer** of yaw (kplanner integrates yaw to compose `current_root_wxyz`; VLA tokenizer compares to measured every tick). Stale measured is strictly better than identity here. |
| **Cache with max-age** | PC2 watchdog (`--x2-debug-max-age-s`, default 0.5 s) | Cache is used only if `(now - last_arrival) <= max_age_s`; otherwise revert to identity | Fails loud — operator sees a clear "I'm not rebasing anymore" boundary | A 0.5 s default reproduces the snap-back bug on a wifi blip | The publisher is itself **already a fallback** (the watchdog ladder fires only after the upstream is gone) AND a parallel safety mechanism (the C++ deploy's starvation watchdog) is monitoring health |

These are not mutually exclusive — the codebase uses both
deliberately, scoped to their respective roles. The mistake we
made in the VLA bridge before 2026-06-23 was using a third,
implicit pattern: **identity-fallback** (revert to a baked
identity quat on stale). That pattern has no defensible use case
on the laptop side because identity is a known-wrong reference,
and it's what the recorder still uses for its dormant-bug path.

---

## 7. Known dormant issues

### Recorder identity-fallback (`§4.3`)

The recorder's `_compute_idle_root_quat_xyzw` reverts to identity
within 1 s of `x2_debug` silence. Not currently operator-visible
because:

1. During VR teleop, the kplanner is the wire publisher; the
   recorder is in relay mode.
2. During VLA, the recorder is not the wire publisher at all.
3. The recorder's own idle-frame path (`_publish_idle`) only fires
   in gesture gaps / idle-stand fallback, which haven't been hit
   in long enough sessions to expose the bug.

Fix shape would be identical to the VLA bridge: cache
`last_known_base_quat_wxyz` and use it instead of `None` on
`not alive`. The two-log-line `_idle_yaw_rebase_logged_active` /
`_idle_yaw_rebase_logged_fallback` gating could be promoted to
a three-edge transition logger mirroring `_log_rebase_source_transition`.

Filed for the next session that touches the recorder; tests in
`tests/test_recorder_idle_yaw_rebase.py` currently pin the
identity-fallback behaviour and would need updating.

### Watchdog's tight 0.5 s default

The PC2 watchdog defaults to `--x2-debug-max-age-s 0.5`. A
WiFi blip on the IMU stream **inside PC2** (loopback,
unlikely but not impossible) within the watchdog's
`IDLE_CLIP` window would revert the idle frame's yaw to
identity. Not currently a reported issue. If it becomes one,
the right fix is probably to switch the watchdog to the
hold-last-good pattern as well — its parallel safety mechanism
(`PoseRefStarvationWatchdog`) already catches the "IMU truly
dead" case independently, so the watchdog's identity-fallback
isn't earning its keep.

---

## 8. Diagnostics

### 8.1 What to grep for in the bridge log

After the 2026-06-23 fix, the VLA bridge emits one line per
yaw-rebase source transition:

| Pattern | Meaning |
|---|---|
| `[live-VLA] withholding pose publish until first x2_debug frame arrives ...` | Bootstrap gate is closed (no `x2_debug` ever). Expected for the first second or two after bridge launch. If it persists, the bridge isn't receiving `x2_debug` — check the SUB topology in `run_x2_vla_runtime.sh`. |
| `[live-VLA] first pose publish (tick=N); x2_debug seen, root_quat now tracks live heading.` | Bootstrap gate opened. Expected once per session. |
| `[live-VLA] root_quat yaw-rebase ACTIVE: ... (yaw=±Y.Ydeg)` | First-time activation. Y should match the operator's perceived heading at startup. |
| `[live-VLA] root_quat yaw-rebase STALE: x2_debug silent >1.0s; holding cached base_quat (cache age=Nms). Wire will NOT revert to identity (no snap-back to spawn heading).` | Mid-session stall; cache is being used. **Body must NOT snap-back at this moment.** Cache age tells you how long the stream has been silent at the transition. |
| `[live-VLA] root_quat yaw-rebase RECOVERED: x2_debug back online; resuming live yaw tracking.` | Stall ended. |

If the body snaps back to spawn heading and you do NOT see a
`STALE` line in the bridge log, the snap-back is coming from
somewhere other than the bridge yaw-rebase — check the
recorder's wire path or PC2 watchdog state.

### 8.2 `x2_yaw_click_sniffer.py`

[`gear_sonic_deploy/scripts/x2_yaw_click_sniffer.py`](../../../gear_sonic_deploy/scripts/x2_yaw_click_sniffer.py)
is the dedicated diagnostic tool. It SUBs the three orientation-
carrying streams simultaneously:

| Stream | Port | Producer |
|---|---|---|
| `body_pose` | `127.0.0.1:5565` | kplanner |
| `pose` | `127.0.0.1:5556` | recorder (merge of body_pose + arms/hands), or VLA bridge |
| `robot_pose` | `127.0.0.1:5570` | `x2_debug_to_robot_pose_bridge` (real) or `x2_mj_bridge` (sim) |

For each stream it tracks `root_yaw_deg` (world-Z rotation of
`root_quat_xyzw`) and `waist_yaw_rad` (`joint_pos_mj[12]`),
differencing each scalar against the previous frame on the
same stream and flagging discontinuities above the per-channel
spike threshold. Output:

```
.venv/bin/python -m gear_sonic_deploy.scripts.x2_yaw_click_sniffer
```

Legend:

- `[TICK]` normal frame (rate-limited; once per second per source)
- `[SPIKE]` delta above threshold — prints prev/curr + cross-stream context
- `[GAP]` gap > 100 ms between two frames on the same stream
- `[HOLD]` stream went silent for > 1 s
- `[SUMMARY]` once per second: rate + max-|delta| per channel per stream

Use the SPIKE lines to read off who agreed with whom and who
jumped: if `body_pose` and `robot_pose` agree but `pose` jumped,
the bug is between the kplanner / VLA-bridge and the wire
(usually a stale identity quat on `pose`). If all three streams
jumped in sync, the bug is upstream of the kplanner.

### 8.3 Manual reproduction

The reliable way to reproduce a stall mid-session:

```sh
# In a separate shell, drop a few seconds of x2_debug:
ssh ubuntu@<PC2_HOST> "sudo iptables -I OUTPUT -p tcp --sport 5557 -j DROP"
sleep 3
ssh ubuntu@<PC2_HOST> "sudo iptables -D OUTPUT -p tcp --sport 5557 -j DROP"
```

After the fix the body should not move during the drop. Before
the fix the body would have snapped to spawn heading.

---

## 9. Historical timeline

| Date | Surface | Symptom | Diagnosis | Fix | Milestone |
|---|---|---|---|---|---|
| Pre-2026-06 | Kplanner | "Robot turns back to default orientation as soon as I start the VR planner stack" on real robot | Kplanner boots `current_root_wxyz = R_z(0)` from warmup PKL; first frame ships identity to the deploy. | Added `x2_debug_to_robot_pose_bridge.py` republisher + kplanner three-layer yaw refresh (startup seed, IDLE_LOOP, IDLE→PLAYING transition) keyed on `pose_feedback_max_age_s`. | (kplanner README) |
| 2026-06-01 | Recorder | Waist-yaw click at startup on real robot (root_quat=identity for the recorder's idle frames before the kplanner takes over) | Recorder's `_compute_idle_root_quat_xyzw` returned identity always. | Added live `x2_debug`-based yaw rebase to the recorder; still identity-fallback on stale (the dormant bug). | — |
| 2026-06-07 | VLA bridge | First VLA start of a SONIC session yanks the robot toward yaw=0 (the heading at SONIC boot). | Bridge publishes `root_quat_xyzw = identity` during the bootstrap window between PUB-bind and the first `x2_debug` arrival. | Bootstrap-safe publish gate (`state.received_any`) + live yaw-rebase on `root_quat_xyzw` + surgical `waist_yaw` (slot 12) pin to measured during legs/waist freeze. | [2026-06-07 — VLA bridge heading stability + head-lock](../user_guide/milestones/2026-06-07_vla_bridge_heading_stability_and_head_lock.md) |
| 2026-06-11 | Pose mux split | (Re-routing) | `robot_pose:5570` collision when the new `x2_pose_mux` runs alongside the existing `x2_debug_to_robot_pose_bridge`. | Bridge `--pub-port` auto-shifts to `5571` when `--enable-takeover` is set so the mux can bind `5556`. | [2026-06-11 — Pose mux split](../user_guide/milestones/2026-06-11_pose_mux_split.md) |
| 2026-06-23 | LAN isolation | Sim runs unintentionally drove the real robot via PC2's always-on `x2_pose_proxy`. | Pose PUB bound on `*:5556` unconditionally; PC2 SUB attached over wifi. | New `PUB_BIND_HOST` env var derived from `SIM_MODE` (runtime) / `PC2_HOST` (replay); sim → `127.0.0.1`, real → `*`. Banner echoes the resolved bind. | [2026-06-23 — Pose PUB LAN isolation](../user_guide/milestones/2026-06-23_pose_pub_lan_isolation.md) |
| 2026-06-23 | VLA bridge (regression) | Robot snaps back to boot heading on every nudge / wifi blip even though `x2_debug` is being subscribed to. | Yaw-rebase and waist_yaw pin gated on `deploy_fresh` (1 s freshness); on the `not deploy_fresh` branch the wire shipped baked-identity quat. Quest3 was immune because the kplanner uses hold-last-good on `current_root_wxyz`. | Lifted the kplanner's hold-last-good pattern into the bridge: `last_known_base_quat_wxyz` cache + `_resolve_wire_rebase_source` + `_log_rebase_source_transition`. | [2026-06-23 — VLA bridge holds last-good live `root_quat` across `x2_debug` stalls](../user_guide/milestones/2026-06-23_vla_bridge_yaw_hold_last_good.md) |

---

## 10. Future direction

If this comes back, here's where to look (in priority order):

1. **Run `x2_yaw_click_sniffer.py`** alongside the failing
   session. The three-stream cross-check will tell you which
   publisher is jumping in <30 s of failing run.
2. **Grep `bridge.log` for the four sentinel lines in [§8.1](#81-what-to-grep-for-in-the-bridge-log).**
   If the STALE line is absent during a snap-back, the bridge
   isn't the publisher at fault; check the recorder or watchdog.
3. **Check whether a new publisher was introduced** that doesn't
   use either hold-last-good or cache-with-max-age. The
   identity-fallback pattern is what bites us. The integration
   tests in `tests/test_live_vla_bridge_yaw_hold_last_good.py`
   pin the bridge's contract; analogous tests should exist for
   any new wire publisher.
4. **Consider lifting the hold-last-good pattern into a shared
   helper.** Today each consumer reimplements it (`_resolve_wire_rebase_source`
   in the bridge, the three-layer refresh in the kplanner, the
   cache-with-max-age in the watchdog). A shared
   `gear_sonic/utils/heading/yaw_rebase.py` module that
   exports the cache + resolver + transition-logger would
   eliminate the per-process drift. The reason this hasn't
   happened yet is that each consumer has slightly different
   inputs (the kplanner uses a `pose_deque` of timestamped
   observations; the bridge uses a thread-shared `_LatestState`;
   the watchdog uses local-tick polling), so the helper would
   need a clean abstraction over the source.
5. **Recorder fix when next touching that file.** Lift the
   hold-last-good pattern into `_compute_idle_root_quat_xyzw`;
   update the tests in `tests/test_recorder_idle_yaw_rebase.py`
   to pin the new contract.

The deploy-side `BuildTokenizerObs` bootstrap-safe override
(`x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp:2608-2625`) is
NOT a replacement for laptop-side hold-last-good — it only fires
when `LastReceivedMonotonicS() < 0`, i.e. **never received a
real frame**. The moment the laptop ships its first frame, the
deploy locks onto that reference and the escape hatch closes.
Any laptop-side bug after the first frame is on the laptop side
to fix.

---

## 11. Code map

| File | Role | Lines of interest |
|---|---|---|
| `gear_sonic/scripts/live_vla_publish_motion_token.py` | VLA bridge wire publisher | `_WireRebaseSource` + `_resolve_wire_rebase_source` + `_log_rebase_source_transition` (~lines 230-400); rebase application (Section H, ~line 3470); waist_yaw pin (Section F, ~line 3345); bootstrap-safe publish gate (~line 3580) |
| `gear_sonic/scripts/x2_kplanner.py` | Teleop / record wire publisher | Startup yaw seed (~line 2540); IDLE_LOOP yaw refresh (~line 2810); IDLE→PLAYING transition seed (~line 2730); `pose_feedback_max_age_s` default 0.5 s (line 2090) |
| `gear_sonic/utils/teleop/x2_dataset_recorder.py` | Recorder idle-frame producer | `_compute_idle_root_quat_xyzw` (`3354-3438`); `_idle_yaw_rebase_logged_active` / `_fallback` flags (lines 1647-1648) |
| `gear_sonic_deploy/scripts/x2_pose_watchdog.py` | PC2 fallback ladder | `--x2-debug-max-age-s` (line 318); yaw-rebase cache (lines 460-615) |
| `gear_sonic_deploy/scripts/x2_debug_to_robot_pose_bridge.py` | Laptop-side `x2_debug` → `robot_pose` republisher | Whole file |
| `gear_sonic_deploy/scripts/x2_yaw_click_sniffer.py` | Three-stream yaw discontinuity sniffer | Whole file |
| `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp` | C++ deploy bootstrap-safe override | `BuildTokenizerObs` measured-quat substitution (lines 2608-2625) |
| `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/safety.hpp` | C++ deploy pose-ref starvation watchdog | `PoseRefStarvationWatchdog` (lines 120-193); trips to SAFE_IDLE on stall, complements (does not replace) laptop-side hold-last-good |
| `tests/test_live_vla_bridge_yaw_hold_last_good.py` | Bridge contract tests | 25 tests covering source resolution, cache semantics, waist_yaw pin during stall, state-transition logger |
| `tests/test_recorder_idle_yaw_rebase.py` | Recorder contract tests | 17 tests — currently pin the identity-fallback behaviour (will need updating when the recorder is fixed) |
