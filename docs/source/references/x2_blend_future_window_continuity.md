# X2 BLEND future-window continuity — subsystem reference

Why the policy snaps when *any* of the wire fields it integrates flips
in one tick (even if `joint_pos_mj` is continuous), the cache-and-lerp
pattern the pose watchdog uses to prevent that, and the historical
timeline of related "arms snap to default after a stall" fixes.

This page is the long-form companion to a recurring class of bugs the
operator perceives as:

> The arms hold the in-flight pose for a few seconds after the wire
> goes quiet, then snap to a new pose, then smoothly settle to
> default-stand. The wrists look smooth throughout, the
> shoulders / elbows are abrupt at the snap. Same shape on Ctrl+C of
> a VLA run, on a wifi blip mid-teleop, and on a planner crash.

The symptom shape is misleading: the smooth tail makes it look like
the snap was a hand-off from a "fast bad ramp" to a "slow good ramp",
but in reality it's the single-tick discontinuity in the
**future-window + motion_token + hands** part of the wire frame at
the HOLD → BLEND boundary. The current `joint_pos_mj` was always
continuous; the policy is what reshapes its planning horizon in one
tick when the rest of the frame switches.

Related references:

- [`X2 Deploy ZMQ Wire Protocol`](x2_zmq_protocol.md) — the v4/v5
  pose envelope, all the fields this page revolves around.
- [`X2 Heading Stability & root_quat Yaw Rebase`](x2_heading_stability_and_yaw_rebase.md)
  — the orientation half of "what flows on the wire". The
  yaw-rebase fix landed 2026-06-23 is a prerequisite for this fix
  (the cached upstream futures need to be in the live-heading frame
  before BLEND can lerp them).
- [`X2 VLA motion_token Decoder`](x2_vla_motion_token_decoder.md) —
  what the 64-d `motion_token` actually encodes (the bridge's
  SONIC encoder output) and why the deploy reads it but doesn't
  decode it.
- [`X2 Split-topology PC2 daemons`](x2_split_deploy_pc2.md) — where
  the pose watchdog runs (PC2 on real robot, laptop in sim).

Milestones with point-in-time wrap-ups on this surface:

- [2026-06-08 — Pose-proxy fallback ladder: freeze arms on upstream stall](../user_guide/milestones/2026-06-08_arm_freeze_on_upstream_stall.md)
  (the original HOLD → BLEND → IDLE_CLIP state machine; `joint_pos_mj`-only BLEND lerp)
- [2026-06-23 — BLEND future-window continuity + faster HOLD timeout](../user_guide/milestones/2026-06-23_blend_future_window_continuity.md)
  (extends BLEND lerp to the full upstream frame; reduces HOLD default 10 → 5 s)

---

## 1. TL;DR

* The C++ deploy's tokenizer integrates more than just `joint_pos_mj`
  from the wire. It also reads `joint_pos_mj_future` (9 future slots),
  `root_quat_xyzw_future`, `joint_vel_mj_future`, `motion_token`
  (64-d intent vector), and `left_hand_joints` /
  `right_hand_joints`. The SONIC policy then plans across the full
  future window with the latched motion_token as a side input.
  *Any* one-tick flip in those fields produces a discontinuity in
  the policy's planning horizon, which SONIC propagates to a step
  change in its commanded body targets — even if `joint_pos_mj`
  itself is continuous.
* The pose watchdog's 2026-06-08 fallback ladder lerps only
  `joint_pos_mj` across BLEND. Everything else — the future window,
  the motion_token, the hands — switches in one tick at the HOLD →
  BLEND boundary from the cached VLA values to the idle clip's
  values (and `ZERO_MOTION_TOKEN` / `ZERO_HAND`). That one-tick
  flip *is* the snap operators see.
* The wrists are usually exempt from the snap because
  `--wrist-bypass ik` in the C++ deploy copies wire
  `joint_pos_mj[wrist]` directly into `target_pos_mj[wrist]`,
  bypassing SONIC. Wrists only see the smooth BLEND lerp on the
  current frame, never the future-window discontinuity. That's
  what makes the symptom shape ("wrists smooth, arms snap")
  diagnostic: it points squarely at the wire fields SONIC reads
  that wrist-bypass doesn't.
* The right fix is **cache+lerp on the full upstream-derived
  frame**: snapshot the future window + motion_token at every
  fresh upstream tick, and lerp all of them across BLEND with the
  same `alpha` that drives the current `joint_pos_mj` lerp. Hands
  are deliberately out of scope today (operator confirmed they
  don't matter for the current failure modes) but the same
  pattern applies.
* The codebase uses **two architectural patterns** here:
  cache-with-fallback (the watchdog's BLEND branch as of 2026-06-23
  — see [§ 5](#5-cache--lerp-pattern)) and the existing
  hold-last-good pattern that the kplanner and VLA bridge use for
  `root_quat_xyzw` (documented in
  [`x2_heading_stability_and_yaw_rebase.md`](x2_heading_stability_and_yaw_rebase.md)).
  Both patterns share the same underlying philosophy: *a stale
  cached value is strictly better than a known-wrong default*.

---

## 2. The wire-fields-the-policy-integrates contract

The C++ deploy's `ZmqPoseInputSource` decodes the following fields
from every pose frame (see
[gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/zmq_pose_input_source.cpp](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/zmq_pose_input_source.cpp)
lines 218-347):

| Wire field | Shape | What the deploy does with it |
|---|---|---|
| `joint_pos_mj` | `(31,)` f32 | Current body reference. Goes into `latest_frame_.joint_pos_mj`; SONIC tokenizer obs reads it as `window[0]`. |
| `root_quat_xyzw` | `(4,)` f32 | Current root orientation. Goes into `latest_frame_.root_quat_xyzw`; tokenizer reads as `window[0]`. |
| `joint_pos_mj_future` | `(9, 31)` f32 | The 9 future slots SONIC plans against. Each slot is `+0.1 s` apart by default. Goes into `latest_window_[1..9].joint_pos_mj`. |
| `root_quat_xyzw_future` | `(9, 4)` f32 | Per-slot future orientation. Goes into `latest_window_[1..9].root_quat_xyzw`. |
| `joint_vel_mj_future` | `(9, 31)` f32 (optional) | Per-slot future joint velocities. If absent, deploy back-finite-diffs from `joint_pos_mj_future`. |
| `frame_index_future` | `(9,)` i64 | Per-slot frame index (advisory). |
| `future_dt_s` | `(1,)` f32 | Sample spacing between consecutive future slots; default `0.1 s` (`DT_FUTURE_REF`). |
| `motion_token` | `(64,)` f32 | SONIC encoder output for the in-flight VLA chunk. Latched into `latest_motion_token_`; tokenizer obs reads it as the **intent** side input. |
| `left_hand_joints` / `right_hand_joints` | `(10,)` f32 each | Finger DOFs. Routed to the hand bridge / AIMDK driver via a separate side channel. Not part of the body SONIC input. |
| `frame_index` | `(1,)` i64 | Current frame index (advisory). |

The SONIC policy reads `window[0..9]` (current + 9 future) and
`motion_token` as a single tokenizer input on every control tick.
That's why a one-tick discontinuity in the **future window** alone
— with `joint_pos_mj` (`window[0].joint_pos_mj`) untouched —
produces a visible step in the policy's output. The 9 future slots
span ~0.9 s of planning horizon at the default `future_dt_s=0.1 s`,
which is exactly the horizon SONIC was trained against.

`window` promotion semantics (lines 320-347 of the cpp): the deploy
only promotes `latest_window_` from the message when **both**
`joint_pos_mj_future` AND `root_quat_xyzw_future` are present. A
partial v5 message keeps the previous window in place — but our
watchdog always produces complete frames, so this only matters for
legacy v4 producers (heuristic planner, mock VLA) that never set
the future window at all.

---

## 3. The HOLD → BLEND boundary problem

The 2026-06-08 pose-watchdog ladder runs three states once upstream
goes silent:

```{mermaid}
stateDiagram-v2
    LIVE --> HOLD: silent > idle_stale_ms
    HOLD --> BLEND: silent past hold_last_secs
    BLEND --> IDLE_CLIP: silent past hold_last_secs + blend_secs
```

* **HOLD** re-publishes the *last forwarded upstream bytes verbatim*.
  Every wire field (current jpos, future window, motion_token,
  hands, root quat) is byte-identical to the last VLA frame.
  Deploy sees `jvel ≈ 0`. No kinematic surprise.
* **BLEND** switches to `build_idle_frame_msg(..., joint_pos_mj_override=lerp)`
  where `lerp = (1 - alpha) * cached_jpos + alpha * idle_clip_jpos`.
  The pre-2026-06-23 implementation only lerped `joint_pos_mj`.
  Everything else came from `build_idle_frame_msg`'s defaults — i.e.
  the idle clip's future window, `ZERO_MOTION_TOKEN`,
  `ZERO_HAND` × 2.

The discontinuity at the HOLD → BLEND boundary, pre-2026-06-23,
looks like this on a field-by-field basis:

| Wire field | Last HOLD tick | First BLEND tick (alpha=0) | Continuous? |
|---|---|---|---|
| `joint_pos_mj` | Cached VLA jpos | `(1-0)*cached + 0*idle = cached` | ✅ |
| `root_quat_xyzw` (current) | Cached VLA quat | Idle clip quat with yaw rebase to live heading | ≈ ✅ (small mismatch if pitch / roll differ) |
| `joint_pos_mj_future` (×9) | Cached VLA future | Idle clip future window | ❌ **snap** |
| `root_quat_xyzw_future` (×9) | Cached VLA quat future | Idle clip quat future (yaw-rebased) | ❌ **snap** |
| `joint_vel_mj_future` (×9) | Cached VLA jvel future | `ZERO_QVEL_FUTURE` | ❌ **snap** |
| `motion_token` (64-d) | Cached VLA motion_token | `ZERO_MOTION_TOKEN` | ❌ **snap** |
| `left_hand_joints` | Cached VLA hand | `ZERO_HAND` | ❌ **snap** |
| `right_hand_joints` | Cached VLA hand | `ZERO_HAND` | ❌ **snap** |
| `frame_index` / `frame_index_future` / `future_dt_s` | Cached VLA | Idle clip indices, `0.1 s` | ❌ (advisory; doesn't drive policy) |

The four ❌ rows in the middle (futures + token) are what produces
the policy step-change. SONIC re-tokenizes against the new horizon
and emits commands appropriate for "start playing idle_stand
clip", not "continue the in-flight chunk". The mid-loop step is
amplified on the body joints SONIC actually owns (shoulders, elbows,
torso, hips) while wrist-bypass keeps the wrists smooth.

Total visible operator timeline (defaults pre-2026-06-23):

```{mermaid}
gantt
    title HOLD -> BLEND -> IDLE_CLIP, default tuning pre-2026-06-23
    dateFormat X
    axisFormat %s s

    section Wire state
    GAP (300 ms)    :gap, 0, 300
    HOLD (10 s)     :hold, after gap, 10000
    BLEND (3 s)     :blend, after hold, 3000
    IDLE_CLIP       :idle, after blend, 1500

    section What operator sees
    Frozen in VLA pose      :hold_obs, 0, 10300
    SUDDEN SNAP             :crit, snap_obs, 10300, 100
    Smooth glide to default :blend_obs, 10400, 2900
    Held at default-stand   :idle_obs, after blend_obs, 1500
```

Post-2026-06-23 (full-frame BLEND lerp + HOLD default 10 → 5 s):

```{mermaid}
gantt
    title HOLD -> BLEND -> IDLE_CLIP, default tuning post-2026-06-23
    dateFormat X
    axisFormat %s s

    section Wire state
    GAP (300 ms)    :gap, 0, 300
    HOLD (5 s)      :hold, after gap, 5000
    BLEND (3 s)     :blend, after hold, 3000
    IDLE_CLIP       :idle, after blend, 1500

    section What operator sees
    Frozen in VLA pose      :hold_obs, 0, 5300
    Smooth glide to default :blend_obs, 5300, 3000
    Held at default-stand   :idle_obs, after blend_obs, 1500
```

The 5 s HOLD is still long enough to soak up typical WiFi blips (<1 s
in practice) and planner restarts (<2 s); it's now half-as-long for
intentional Ctrl+C while still being a generous safety margin for
unexpected stalls.

---

## 4. Why the wrists look smooth

Wrist DOFs 20, 21, 27, 28 (left/right wrist pitch + roll) are
routed through `--wrist-bypass ik` in the C++ deploy (see
[gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp)
lines 63-72):

```cpp
for (const int mj : kBypassedWristMjDofs) {
  target_pos_mj[mj] = ref.joint_pos_mj[mj];
}
```

This copies wire `joint_pos_mj[wrist]` directly into the policy
target *after* SONIC has run, bypassing the policy's per-joint
output. The 8 Hz target LPF on real robot then smooths the motor
command. Net effect: wrists track the wire `joint_pos_mj` (which IS
being smoothly lerped across BLEND) with a soft first-order lag.
They never see the future-window discontinuity because the bypass
doesn't read the future window at all.

Wrist yaw (MJ slots 19, 26) and shoulder / elbow / torso joints go
through SONIC normally and therefore *do* see the discontinuity —
which is why those are the joints that snap.

This explains the diagnostic symptom shape (wrists smooth, arms
snap) and is a useful sanity check when triaging a future
regression: if you see a "snap" that affects the wrists, the
cause is *not* the future-window discontinuity — it's something
downstream (deploy clamp, motor controller, hand bridge).

---

## 5. Cache + lerp pattern

The fix in
[gear_sonic_deploy/scripts/x2_pose_watchdog.py](../../../gear_sonic_deploy/scripts/x2_pose_watchdog.py)
and
[gear_sonic/utils/pose_pipeline/fallback.py](../../../gear_sonic/utils/pose_pipeline/fallback.py)
mirrors the well-established "cache the freshest upstream value,
lerp it toward a default during fallback" pattern used elsewhere in
the stack (kplanner's `current_root_wxyz` hold-last-good for yaw —
see [heading-stability reference](x2_heading_stability_and_yaw_rebase.md)).

### Anatomy

1. **Cache on every fresh upstream tick.** At every successful
   upstream `recv`, the watchdog decodes and caches:

   * `last_upstream_jpos` (already there pre-2026-06-23)
   * `last_upstream_jpos_future` (NEW, decoder
     `decode_pose_joint_pos_mj_future`)
   * `last_upstream_quat_future` (NEW, decoder
     `decode_pose_root_quat_xyzw_future`)
   * `last_upstream_jvel_future` (NEW, decoder
     `decode_pose_joint_vel_mj_future`)
   * `last_upstream_motion_token` (NEW, decoder
     `decode_pose_motion_token`)

   Decoders return `None` on absent / malformed fields. The cache
   keeps the previous value when a decode returns `None` — a
   slightly stale cache is strictly better than no cache for the
   BLEND continuity invariant.

2. **Pre-rebase the idle-clip side BEFORE lerping.** The cached
   upstream `root_quat_xyzw_future` is already in the live-heading
   frame (the VLA bridge applies yaw rebase pre-publish as of
   2026-06-23). The idle clip's future quats are not yaw-rebased
   by default. The BLEND branch rebases the idle clip side first so
   both lerp endpoints share a frame, then nlerps.

3. **Lerp everything with the same alpha** that drives the current
   `joint_pos_mj` lerp. For each cached field that exists:

   * `joint_pos_mj_future_override`: per-element f32 linear lerp.
   * `root_quat_xyzw_future_override`: per-slot
     `nlerp_quat_arrays_xyzw` with dot-sign correction.
   * `joint_vel_mj_future_override`: per-element f32 linear lerp
     toward the idle clip's `ZERO_QVEL_FUTURE`.
   * `motion_token_override`: linear decay toward zero
     (`(1 - alpha) * cached_token`). The idle endpoint is
     deliberately zero — the policy gets no extra intent during
     idle.

4. **Graceful fallback when a cache is None.** If the upstream
   producer never set a given field (legacy v4 heuristic planner,
   mock VLA, replay before 2026-06-22), the corresponding
   `*_override` kwarg is simply not passed. `build_idle_frame_msg`
   then defaults to the idle clip's value (or `ZERO_MOTION_TOKEN`)
   for that field only — the *other* fields still get the cached
   upstream lerp. Net effect: no regression for non-VLA upstreams
   on the fields they don't set, full continuity on the fields
   they do set.

### `build_idle_frame_msg` API

The override kwargs are all optional and additive:

```python
fallback.build_idle_frame_msg(
    replay, tick, topic,
    *,
    yaw_rebase_rad=None,
    joint_pos_mj_override=None,             # existing (current jpos)
    joint_pos_mj_future_override=None,      # NEW
    root_quat_xyzw_future_override=None,    # NEW
    joint_vel_mj_future_override=None,      # NEW
    motion_token_override=None,             # NEW
) -> bytes
```

Validation rules:

* Each `*_override` is shape-checked (raises `ValueError` on
  mismatch) and cast to `float32`.
* `root_quat_xyzw_future_override` is assumed to be **already in the
  live-heading frame**: `build_idle_frame_msg` skips the second
  yaw rebase that would otherwise apply via `yaw_rebase_rad`.
  The watchdog is responsible for pre-rebasing the idle-clip side
  before lerping so both endpoints share a frame.
* No kwargs = no behaviour change. The pre-2026-06-23 path is
  preserved exactly.

---

## 6. Quat lerp: nlerp, not slerp

`gear_sonic/utils/pose_pipeline/wire.py::nlerp_quat_arrays_xyzw`
does **normalized linear interpolation with dot-sign correction**:

```python
def nlerp_quat_arrays_xyzw(q_from, q_to, alpha):
    dot = sum(q_from * q_to, axis=1)
    q_to_aligned = where(dot < 0, -q_to, q_to)   # short path
    lerp = (1 - alpha) * q_from + alpha * q_to_aligned
    return lerp / norm(lerp)
```

The watchdog runs in a scipy-free venv on PC2 (numpy + pyzmq +
stdlib budget, see comment near the top of
[wire.py](../../../gear_sonic/utils/pose_pipeline/wire.py)), so a
full slerp implementation (`scipy.spatial.transform.Rotation` or a
hand-rolled `sin`/`atan2` path) would be a costly dependency
addition. For the per-tick alpha deltas we see across BLEND:

* `blend_secs = 3.0 s`, `rate_hz = 50 Hz` → 150 ticks
* `alpha` increment per tick ≈ 0.0067
* Typical angle between cached upstream future and idle clip
  future for VLA tasks: ~10-30° on the dominant yaw axis, ~0° on
  pitch/roll
* nlerp vs slerp error at the worst-case (alpha = 0.5, angle = 30°):
  ~0.5°

That's well below the deploy's `target_lpf_hz = 8 Hz` output
bandwidth — nlerp is indistinguishable from slerp at the motor
target after the LPF runs. Pin tests in
[tests/test_pose_pipeline_blend_future_continuity.py](../../../tests/test_pose_pipeline_blend_future_continuity.py)
verify:

* Alpha = 0 → returns `q_from` exactly.
* Alpha = 1 → returns `q_to` (up to overall sign).
* Sign correction: antipodal endpoints (`q_to = -q_from`,
  representing the same rotation) lerp to `q_from` at alpha = 0.5
  instead of the degenerate origin.
* Unit norm preserved across the entire ramp.

---

## 7. Yaw-rebase frame consistency

This fix depends on the cached upstream `root_quat_xyzw_future`
being in the live-heading frame already. That's only true since
the [2026-06-23 VLA bridge yaw hold-last-good fix](../user_guide/milestones/2026-06-23_vla_bridge_yaw_hold_last_good.md)
made the bridge cache `last_known_base_quat_wxyz` and apply yaw
rebase even across `x2_debug` stalls. If you ever revert that
patch (or run with `--no-x2-debug-yaw-track`), the watchdog's
BLEND lerp will mix two different reference frames — the cached
upstream side will be in the *historical* heading, the idle clip
side will be in the *current measured* heading. The lerp will
"glide through" intermediate orientations that don't correspond
to any single physical heading. Operators will see a heading
twist during BLEND in addition to the (now-smooth) joint glide.

The watchdog explicitly documents this in the BLEND branch
comments. The
[heading-stability reference](x2_heading_stability_and_yaw_rebase.md)
covers the full publisher-side picture (which laptop processes own
`root_quat_xyzw`, what pattern each uses, and how a stale yaw is
handled at each).

---

## 8. Hand-joint handling is deliberately out of scope

Per operator decision on 2026-06-23, hand joints continue to snap
from cached VLA grip to `ZERO_HAND` at the HOLD → BLEND boundary.
The mechanism is unchanged from the original 2026-06-08
implementation: `build_idle_frame_msg` always writes `ZERO_HAND`
into both hand fields regardless of any override kwargs.

If we revisit, the same cache+lerp pattern applies. Concretely we'd
add:

* `last_upstream_left_hand` / `last_upstream_right_hand` caches in
  the watchdog using the existing
  [`decode_pose_left_hand`](../../../gear_sonic/utils/pose_pipeline/wire.py) /
  [`decode_pose_right_hand`](../../../gear_sonic/utils/pose_pipeline/wire.py)
  decoders (which already exist for the laptop mux's takeover
  detector).
* `left_hand_override` / `right_hand_override` kwargs on
  `build_idle_frame_msg` mirroring the four future-window
  overrides.
* Linear lerp from cached hand pose toward `ZERO_HAND` over the
  same blend window.

The fingers do not go through SONIC at all — they're handled by
the separate `x2_hand_zmq_to_aimdk_bridge.py` daemon on PC2 — so
their behaviour during BLEND is independent of the policy snap
question. The current "snap to fully-open" behaviour matches
operator expectation for the
"force-quit the run, robot should release any grip" failure mode.

---

## 9. What this fix does NOT address

Explicit non-goals so the next operator knows where the fix's
authority ends:

* **The 10 → 5 s HOLD reduction shortens the dead-time but doesn't
  eliminate it.** Intentional Ctrl+C still produces ~5 s of
  frozen VLA pose before the smooth BLEND ramp begins. The
  bridge-side graceful-exit option (laptop intercepts SIGINT,
  publishes its own N-second lerp before exiting) would eliminate
  the wait entirely; deferred.
* **C++ deploy RAMP_OUT (`--enable-soft-shutdown` /
  `--return-seconds`)** is orthogonal and not exercised on real
  robot today (the launcher's signal protocol only SIGTERMs
  laptop processes; PC2 deploy keeps running). Could be added
  later for "shut down PC2 daemons safely" workflows.
* **The SONIC policy's own internal smoothing.** SONIC has its
  own output LPF (`target_lpf_hz`) and per-DOF deviation clamp
  (`max_target_dev`) on the motor side. These help with
  small-amplitude noise but cannot absorb a multi-radian
  one-tick step on the policy's *input* — which is what the
  pre-fix BLEND boundary was producing.
* **WiFi-blip recovery semantics.** If upstream comes back during
  BLEND, the wire snaps back to upstream bytes on the next tick
  (this is intentional — the watchdog's job is to keep the
  deploy fed, not to ramp every transition). Operators who'd
  like a "smooth BLEND → LIVE handoff" would need a more
  intricate state machine; deferred until we actually see a
  failure mode from it.

---

## 10. Historical timeline

Chronological log of related fixes on this surface. Each entry
links the original milestone for the full session-level wrap-up.

| Date | What landed | Why it matters here |
|---|---|---|
| [2026-06-08](../user_guide/milestones/2026-06-08_arm_freeze_on_upstream_stall.md) | Introduced the HOLD → BLEND → IDLE_CLIP fallback ladder. Replaced the pre-2026-06-08 binary `LIVE/IDLE` fallback that slammed arms to default-stand in ~200 ms on every WiFi blip. BLEND lerps `joint_pos_mj` only. | First fix on this surface. Solved the "instant slam" problem but left the "snap at HOLD → BLEND" problem that 2026-06-23 fixes. |
| [2026-06-22](../user_guide/milestones/2026-06-22_dataset_replay_v5_wire.md) | Dataset replay starts publishing the full v5 future-window (`joint_pos_mj_future`, `root_quat_xyzw_future`, `joint_vel_mj_future`, `frame_index_future`, `future_dt_s`). Confirms — via root-cause investigation of "replay's fingers move but body doesn't" — that the C++ deploy ignores `motion_token` on the wire and re-tokenizes from the future window. | Proves the future window matters: a missing future window causes the deploy to back-fill with `default_angles`, which is exactly the same shape as the pre-2026-06-23 BLEND snap. The replay fix and this BLEND-continuity fix are two views of the same underlying truth. |
| [2026-06-23 (yaw hold-last-good)](../user_guide/milestones/2026-06-23_vla_bridge_yaw_hold_last_good.md) | VLA bridge caches `last_known_base_quat_wxyz` so the wire's `root_quat_xyzw` stays in the live heading frame even across `x2_debug` stalls. | Prerequisite for this fix: the cached upstream `root_quat_xyzw_future` must be in the live heading frame for the BLEND quat lerp to make sense. If we ever revert this, the BLEND quat lerp will mix two different reference frames. |
| [2026-06-23 (this fix)](../user_guide/milestones/2026-06-23_blend_future_window_continuity.md) | Watchdog snapshots the full upstream future window + motion_token; BLEND branch lerps everything; HOLD default 10 → 5 s. | Eliminates the snap at the HOLD → BLEND boundary that operators observed on intentional Ctrl+C and that was almost certainly also present (but unreported) on long WiFi blips. |

---

## 11. Diagnostics

If the snap reappears (regression, or a new failure mode that
matches the same symptom shape), the fastest triage path is:

1. **Capture watchdog state-transition logs.** The watchdog logs
   one line at each state transition (see
   [x2_pose_watchdog.py](../../../gear_sonic_deploy/scripts/x2_pose_watchdog.py)
   lines 689+). Look for the `HOLD -> BLEND` line and the time
   between `HOLD entered` and `BLEND entered`. Confirm the timing
   matches your configured `POSE_PROXY_HOLD_LAST_SECS`.

2. **Confirm the watchdog is running the new code.** The banner
   at startup includes `idle mode: blend (HOLD <X>s, BLEND <Y>s)`.
   If `<X>` is still `10.0` on a fresh install, either
   `POSE_PROXY_HOLD_LAST_SECS` is set in your environment or the
   PC2 sync didn't include the new defaults.

3. **Check whether the upstream is setting the future window.**
   Add a one-shot log in the watchdog's upstream-decode block
   that prints which of the four new caches are non-`None`
   after the first 50 frames. A legacy v4 upstream (heuristic
   planner, mock VLA) will have `None` for all four; a v5
   upstream (VLA bridge, dataset replay post-2026-06-22) will
   have all four set.

4. **Decode published BLEND bytes off the wire.** With
   `tcpdump` / `ngrep` on PC2 or a tiny ZMQ subscriber, snapshot
   the first BLEND-tick frame and the last HOLD-tick frame.
   Decode both with the
   [decode_pose_*](../../../gear_sonic/utils/pose_pipeline/wire.py)
   helpers and diff per-field. The only fields that should
   differ at the alpha=0 boundary are advisory: `frame_index`,
   `frame_index_future`, `future_dt_s` (if upstream uses a
   non-default), and `root_quat_xyzw_future` *only if* the
   upstream-side yaw-rebase frame and the watchdog's idle-side
   yaw-rebase frame have drifted (see [§ 7](#7-yaw-rebase-frame-consistency)).

5. **Run the unit tests.** All boundary continuity invariants are
   pinned in
   [tests/test_pose_pipeline_blend_future_continuity.py](../../../tests/test_pose_pipeline_blend_future_continuity.py).
   If the fix code is intact but the symptom is back, one of those
   tests would catch a regression in the watchdog / fallback
   arithmetic. If they all pass and the symptom is still there,
   the regression is likely upstream (the cache decoders are
   getting `None` from the upstream frame for some reason).

---

## 12. Related references

* [`X2 Deploy ZMQ Wire Protocol`](x2_zmq_protocol.md) — the v4/v5
  envelope structure all the fields above sit in.
* [`X2 Heading Stability & root_quat Yaw Rebase`](x2_heading_stability_and_yaw_rebase.md)
  — orientation half of "what flows on the wire", and the
  prerequisite for the BLEND quat lerp to mix matched reference
  frames.
* [`X2 VLA motion_token Decoder`](x2_vla_motion_token_decoder.md) —
  what the 64-d `motion_token` actually encodes (SONIC's
  bridge-side closure of the body-motion loop).
* [`X2 Split-topology PC2 daemons`](x2_split_deploy_pc2.md) —
  where the watchdog runs and how it gets restarted.
* [`X2 SONIC Runtime Architecture`](x2_sonic_runtime_architecture.md)
  — which laptop process produces the wire in each mode (teleop /
  record / VLA) and therefore which producers' caches feed the
  watchdog's BLEND lerp.
