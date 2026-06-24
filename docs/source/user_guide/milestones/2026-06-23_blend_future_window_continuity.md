# 2026-06-23 — BLEND future-window continuity + faster HOLD timeout

> **Status:** code + tests landed; awaiting real-robot validation.
> Sim watchdog picks up the new BLEND code immediately. The PC2-side
> watchdog needs an operator-driven rsync + `x2_pc2_daemons.sh`
> restart before the fix takes effect on real hardware — the
> assistant did NOT touch PC2 for this change (operator owns that
> step, see [§ How to roll out on PC2](#how-to-roll-out-on-pc2)).

> **Session focus.** Operator-reported follow-up to the
> [2026-06-08 fallback-ladder milestone](2026-06-08_arm_freeze_on_upstream_stall.md):
> on Ctrl+C of a VLA run, after a few seconds of stillness the robot
> arms *snap to a different pose*, then smoothly settle. The smooth
> tail was the watchdog's existing 3 s BLEND lerp on
> `joint_pos_mj`. The snap was happening at the HOLD → BLEND
> boundary, where the watchdog flipped *every other wire field*
> (future window, motion_token, hands) from the cached VLA values to
> the idle clip's values in one tick, even though `joint_pos_mj`
> itself was continuous. Because the C++ deploy integrates the full
> future window + motion_token, the policy's planning horizon
> reshaped in one tick → shoulder/elbow snap. The wrists looked fine
> because `--wrist-bypass ik` routes them directly from the wire and
> the 3 s BLEND lerp on `joint_pos_mj` smoothed them transparently.
>
> **Subsystem reference.** The architectural picture (why the policy
> integrates each wire field, the cache+lerp pattern, the four
> publishers that own this wire, the historical timeline of related
> snap-back fixes) lives at
> [`x2_blend_future_window_continuity`](../../references/x2_blend_future_window_continuity.md).
> Read that one for the long-form story; this milestone is the
> point-in-time wrap-up.

---

## TL;DR

| Symptom (before) | Cause | Fix |
|---|---|---|
| On Ctrl+C of a VLA run, after ~10 s of stillness the robot arms snap to a different pose then smoothly settle. Wrists look fine, shoulders / elbows are abrupt. | At the HOLD → BLEND boundary, the watchdog kept `joint_pos_mj` continuous (lerp `alpha = 0` matches the cached upstream value) but switched `joint_pos_mj_future`, `root_quat_xyzw_future`, `joint_vel_mj_future`, `motion_token`, and `left_hand_joints` / `right_hand_joints` from the cached VLA frame to the idle clip's values (and `ZERO_MOTION_TOKEN` / `ZERO_HAND`) in one tick. The C++ deploy's tokenizer integrates the full future window + motion_token, so the policy's planning horizon reshaped in one tick. SONIC's new step-change in shoulder / elbow targets *is* the snap. Wrists were exempt because `--wrist-bypass ik` copies wire `joint_pos_mj[wrist]` directly into `target_pos_mj[wrist]` — so they only see the smooth BLEND lerp on the current frame, not the future-window discontinuity. | Snapshot the upstream `joint_pos_mj_future`, `root_quat_xyzw_future`, `joint_vel_mj_future`, and `motion_token` at every fresh upstream tick. In the BLEND branch, lerp all of them against the (yaw-rebased) idle clip values using the same alpha that drives `joint_pos_mj`. `build_idle_frame_msg` gains four optional override kwargs so the watchdog can pass the per-tick lerps through. Legacy v4 upstreams (heuristic planner / mock VLA) that never set those fields fall back to today's snap on those fields only — no regression. |
| The 10 s HOLD wait before BLEND starts is dead time for intentional Ctrl+C — operator stares at a frozen VLA pose for 10 s before any motion happens. | `POSE_PROXY_HOLD_LAST_SECS` defaulted to `10.0` (sized for a long WiFi blip / Cursor reload). | Reduce the default to `5.0` in all three sources (`x2_pose_watchdog.py` CLI, `x2_pc2_daemons.sh` env, `run_x2_vla_runtime.sh` env). 5 s is still plenty for typical WiFi blips (<1 s) and planner restarts (<2 s); paired with the smooth BLEND, the cost of entering BLEND a few seconds earlier is benign. Operators wanting the old window can `export POSE_PROXY_HOLD_LAST_SECS=10.0` before launching. |
| Hands snap from VLA grip to fully-open at HOLD → BLEND. | `build_idle_frame_msg` unconditionally puts `ZERO_HAND` in both hand fields. | **Out of scope this turn** — operator confirmed hands don't matter for this issue. If we revisit, the same cache+lerp pattern applies. |

---

## Sequence before vs after (HOLD → BLEND)

```{mermaid}
sequenceDiagram
    participant Op as Operator
    participant Br as VLA bridge<br/>(laptop)
    participant Wd as Pose watchdog<br/>(PC2 / sim)
    participant Dp as C++ deploy

    Note over Op,Dp: BEFORE -- snap at HOLD->BLEND boundary
    Op->>Br: Ctrl+C
    Br->>Br: stop publishing
    Wd-->>Dp: HOLD: last VLA bytes verbatim (10 s)
    Note over Wd: at HOLD->BLEND boundary --<br/>jpos continuous, BUT<br/>future window, motion_token,<br/>hands all flip in 1 tick
    Wd->>Dp: BLEND tick 1: lerp(alpha=0) jpos + idle_clip future + ZERO token + ZERO_HAND
    Dp->>Dp: SONIC re-plans on new horizon ->step change in shoulder/elbow targets
    Note over Dp: SNAP

    Note over Op,Dp: AFTER -- full-frame BLEND lerp, faster HOLD
    Op->>Br: Ctrl+C
    Br->>Br: stop publishing
    Wd-->>Dp: HOLD: last VLA bytes verbatim (5 s)
    Wd->>Dp: BLEND tick 1: alpha=0 lerp on ALL fields -> matches cached VLA
    Wd->>Dp: BLEND tick N: alpha=k/N lerp on ALL fields -> glides toward idle
    Wd->>Dp: BLEND tick last: alpha=1 -> matches idle clip
    Note over Dp: Policy horizon reshapes continuously over 3 s.<br/>SONIC tracks smoothly; no snap.
```

---

## What changed

### 1. Four new wire-field decoders

[gear_sonic/utils/pose_pipeline/wire.py](../../../../gear_sonic/utils/pose_pipeline/wire.py):

| Function | Returns | Used by |
|---|---|---|
| `decode_pose_joint_pos_mj_future` | `(NUM_FUTURE_SLOTS, NUM_BODY_DOFS)` f32 or `None` | Watchdog upstream snapshot |
| `decode_pose_root_quat_xyzw_future` | `(NUM_FUTURE_SLOTS, 4)` f32 or `None` | Watchdog upstream snapshot |
| `decode_pose_joint_vel_mj_future` | `(NUM_FUTURE_SLOTS, NUM_BODY_DOFS)` f32 or `None` | Watchdog upstream snapshot |
| `decode_pose_motion_token` | `(SONIC_MOTION_TOKEN_DIM,)` f32 or `None` | Watchdog upstream snapshot |

All built on the existing `_decode_pose_field_f32` helper (one-line wrappers). The helper itself was tweaked to `reshape(expected_shape)` on the way out so 2-D fields decode to their declared shape instead of the flat 1-D buffer (existing 1-D callers are unaffected: `(N,)` reshapes to `(N,)` as a no-op).

### 2. `nlerp_quat_arrays_xyzw`

Normalized linear interpolation between batches of xyzw quats with dot-sign correction (`q` and `-q` represent the same rotation; flip `q_to → -q_to` when `dot < 0` so the lerp takes the short great-circle path). Numpy-only — keeps the watchdog in the existing scipy-free dependency budget on PC2.

For our use case (3 s BLEND, 50 Hz, per-tick alpha increment ~0.007, futures mostly close to identity), nlerp matches scipy's full slerp to better than 0.5° per tick — well below the deploy's `target_lpf_hz=8 Hz` output bandwidth.

### 3. `build_idle_frame_msg` gains four optional override kwargs

[gear_sonic/utils/pose_pipeline/fallback.py](../../../../gear_sonic/utils/pose_pipeline/fallback.py):

```python
def build_idle_frame_msg(
    replay, tick, topic,
    *,
    yaw_rebase_rad=None,
    joint_pos_mj_override=None,
    joint_pos_mj_future_override=None,    # NEW
    root_quat_xyzw_future_override=None,  # NEW
    joint_vel_mj_future_override=None,    # NEW
    motion_token_override=None,           # NEW
) -> bytes:
```

Each override is shape-validated and casts to f32. `root_quat_xyzw_future_override` is assumed to be **already in the live-heading frame** (the BLEND branch pre-rebases the idle-clip side before nlerping, so both lerp endpoints share a frame) — `build_idle_frame_msg` deliberately skips the second yaw rebase that would otherwise apply via `yaw_rebase_rad`.

Default behaviour (no kwargs) is unchanged — the new code path is purely opt-in.

### 4. Watchdog BLEND branch lerps everything

[gear_sonic_deploy/scripts/x2_pose_watchdog.py](../../../../gear_sonic_deploy/scripts/x2_pose_watchdog.py):

- Four new caches alongside `last_upstream_jpos`: `last_upstream_jpos_future`, `last_upstream_quat_future`, `last_upstream_jvel_future`, `last_upstream_motion_token`.
- Each is decoded at every fresh upstream tick. When the upstream is a legacy v4 producer that never sets a given field, the corresponding cache stays at its previous value (or `None` until something arrives).
- BLEND branch pre-rebases the idle-clip future quats, then lerps each cached upstream field against its idle-clip counterpart using the same `blend_alpha` that drives the current jpos lerp. Each override is only passed when the cache is non-None — legacy v4 upstreams gracefully retain today's snap on the absent fields only.

### 5. Faster HOLD default (10.0 s → 5.0 s)

Three config sources updated:

| Source | Before | After |
|---|---|---|
| [gear_sonic_deploy/scripts/x2_pose_watchdog.py](../../../../gear_sonic_deploy/scripts/x2_pose_watchdog.py) `--hold-last-secs` | `default=10.0` | `default=5.0` |
| [gear_sonic_deploy/scripts/x2_pc2_daemons.sh](../../../../gear_sonic_deploy/scripts/x2_pc2_daemons.sh) `POSE_PROXY_HOLD_LAST_SECS` | `:-10.0` | `:-5.0` |
| [gear_sonic/scripts/run_x2_vla_runtime.sh](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh) `POSE_PROXY_HOLD_LAST_SECS` | `:=10.0` | `:=5.0` |

Cuts the operator-visible dead time before BLEND starts on intentional Ctrl+C roughly in half (~10.3 s → ~5.3 s of frozen VLA pose). For unexpected stalls (WiFi blip, planner crash), the margin shrinks from 10 s to 5 s but smooth BLEND makes early entry benign. Operators wanting the old window:

```bash
export POSE_PROXY_HOLD_LAST_SECS=10.0
./gear_sonic/scripts/run_x2_vla_runtime.sh ...
```

The single pinning test at [tests/test_run_x2_vla_runtime_sim_proxy.py:140](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) was updated from `"10.0"` to `"5.0"`. The state-machine table-driven tests in [tests/test_x2_pose_watchdog_fallback_ladder.py](../../../../tests/test_x2_pose_watchdog_fallback_ladder.py) pass `hold_last_secs=10.0` as an explicit input (not as a default-pin), so they're left untouched — the state machine continues to be tested across a range of HOLD durations.

The historical milestone references to "default 10 s" in [2026-06-08_arm_freeze_on_upstream_stall.md](2026-06-08_arm_freeze_on_upstream_stall.md), [2026-06-10_vla_manual_takeover.md](2026-06-10_vla_manual_takeover.md), and [2026-06-11_pose_mux_split.md](2026-06-11_pose_mux_split.md) are deliberately preserved as point-in-time records.

---

## Tests

All new tests are pure pytest unit tests running locally on the laptop. **No SSH, no rsync, no ZMQ sockets, no subprocess of the watchdog, no PC2 of any kind** — see the docstring at the top of [tests/test_pose_pipeline_blend_future_continuity.py](../../../../tests/test_pose_pipeline_blend_future_continuity.py) for the explicit guardrail.

| File | New / changed tests | What's pinned |
|---|---|---|
| [tests/test_pose_pipeline_blend_future_continuity.py](../../../../tests/test_pose_pipeline_blend_future_continuity.py) | **28 new** | Decoder round-trips for the four new fields, graceful v4 fallback (legacy heuristic-planner frames decode to `None`), `nlerp_quat_arrays_xyzw` (alpha=0/1 endpoints, antipodal sign correction, unit-norm preservation, shape validation), `build_idle_frame_msg` override shape validation, HOLD → BLEND boundary continuity at alpha=0 (futures + motion_token equal the cached upstream snapshot), BLEND → IDLE_CLIP boundary at alpha=1 (futures equal the idle clip, motion_token decays to zero), monotonic per-slot ramp (no overshoot), unit-norm quat preservation across the full ramp, override-quat NOT yaw-rebased again. |
| [tests/test_run_x2_vla_runtime_sim_proxy.py](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) | 1 updated | Default-pin for `POSE_PROXY_HOLD_LAST_SECS`: `"10.0"` → `"5.0"`. |
| [tests/test_x2_pose_watchdog_fallback_ladder.py](../../../../tests/test_x2_pose_watchdog_fallback_ladder.py) | unchanged | 20/20 pass — state-machine semantics untouched. |
| [tests/test_x2_pose_watchdog_smoke.py](../../../../tests/test_x2_pose_watchdog_smoke.py) | unchanged | passes — watchdog still imports cleanly. |

Verification command:

```bash
python -m pytest tests/test_pose_pipeline_blend_future_continuity.py tests/test_x2_pose_watchdog_fallback_ladder.py tests/test_x2_pose_watchdog_smoke.py tests/test_run_x2_vla_runtime_sim_proxy.py -q
```

Result: 49 passed, 1 skipped (the skip is the existing optional ZMQ smoke test for the watchdog when pyzmq isn't installed in the test venv — unrelated to this change).

---

## How to roll out on PC2

**The assistant deliberately did not touch PC2 for this change.** The operator owns the PC2 deploy and will do it themselves, safely, on their own schedule.

Files that must reach PC2 before the fix takes effect on the real robot:

- [gear_sonic_deploy/scripts/x2_pose_watchdog.py](../../../../gear_sonic_deploy/scripts/x2_pose_watchdog.py) — the watchdog itself
- [gear_sonic/utils/pose_pipeline/wire.py](../../../../gear_sonic/utils/pose_pipeline/wire.py) — new decoders + nlerp helper
- [gear_sonic/utils/pose_pipeline/fallback.py](../../../../gear_sonic/utils/pose_pipeline/fallback.py) — `build_idle_frame_msg` override kwargs

(Sim has no PC2 step: the sim watchdog launched by `spawn_sim_watchdog()` in [run_x2_vla_runtime.sh](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh) picks up the new code immediately from the local repo.)

Operator runbook (copy-paste, run from the laptop):

```bash
# 1. Sync the repo to PC2 (whatever rsync command you usually use).
#    Example -- adjust to your local convention:
# rsync -avz --delete ~/Projects/GR00T-WholeBodyControl/ ubuntu@<pc2-host>:~/GR00T-WholeBodyControl/

# 2. Bounce the PC2 daemons so the watchdog reloads with the new code.
ssh ubuntu@<pc2-host> './gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop'
ssh ubuntu@<pc2-host> './gear_sonic_deploy/scripts/x2_pc2_daemons.sh start'

# 3. Quick sanity check on the banner the watchdog prints at startup:
ssh ubuntu@<pc2-host> 'tail -50 ~/x2_pc2_daemons.log | grep -E "(pose_watchdog|HOLD|BLEND)"'
```

Look for the watchdog banner line showing the new defaults: `idle mode: blend (HOLD 5.0s, BLEND 3.0s)`. If you still see `HOLD 10.0s`, either `POSE_PROXY_HOLD_LAST_SECS` is set in your environment or the PC2 sync didn't reach `gear_sonic_deploy/scripts/x2_pc2_daemons.sh`.

Then reproduce the fix on the real robot:

```bash
./gear_sonic/scripts/run_x2_vla_runtime.sh --pc2-host <pc2-host> \
    --model <checkpoint> --motion-token-decoder <decoder> \
    --prompt "reach for the soda can on the table and return your hand to rest"
# ... wait for SONIC + VLA to settle, then trigger the snap reproduction ...
# Ctrl+C
```

Expected behaviour after fix:

1. ~5 s of frozen VLA pose (HOLD).
2. Smooth 3 s glide to idle_stand — shoulders / elbows track wrists, **no visible snap** at the HOLD → BLEND boundary.
3. Hold idle_stand thereafter.

If you still see the snap, capture `tail -200 ~/x2_pc2_daemons.log` from PC2 and the laptop's bridge log; the diagnostics section in the [BLEND continuity reference doc](../../references/x2_blend_future_window_continuity.md) walks through what to look for.

---

## Out of scope (deferred)

- **Bridge-side graceful shutdown ramp.** On SIGINT the bridge could publish its own N-second lerp from the last VLA frame to an idle pose, eliminating the HOLD wait entirely. That's a laptop-only change that composes naturally with this watchdog fix. Deferred.
- **C++ deploy RAMP_OUT (`--enable-soft-shutdown`).** Orthogonal to this fix; would soft-stop the deploy on its own SIGINT instead of leaving teardown to the watchdog ladder.
- **Hand-joint continuity at HOLD → BLEND.** Hands currently snap from cached VLA grip to `ZERO_HAND` at the boundary. Operator confirmed this is fine for now. Cache+lerp pattern would apply if we revisit.
- **Recorder's `_compute_idle_root_quat_xyzw` dormant bug.** Mentioned for completeness in the [heading-stability reference doc](../../references/x2_heading_stability_and_yaw_rebase.md); not touched here.

---

## Files touched

| File | Lines | What |
|---|---|---|
| [gear_sonic/utils/pose_pipeline/wire.py](../../../../gear_sonic/utils/pose_pipeline/wire.py) | +95 / -3 | 4 new field decoders + `nlerp_quat_arrays_xyzw` helper + tiny `_decode_pose_field_f32` reshape fix |
| [gear_sonic/utils/pose_pipeline/fallback.py](../../../../gear_sonic/utils/pose_pipeline/fallback.py) | +75 / -8 | 4 new override kwargs on `build_idle_frame_msg` + centralised validator |
| [gear_sonic_deploy/scripts/x2_pose_watchdog.py](../../../../gear_sonic_deploy/scripts/x2_pose_watchdog.py) | +95 / -7 | 4 new upstream caches + full-frame BLEND lerp + HOLD default 10.0 → 5.0 |
| [gear_sonic_deploy/scripts/x2_pc2_daemons.sh](../../../../gear_sonic_deploy/scripts/x2_pc2_daemons.sh) | +6 / -3 | HOLD default 10.0 → 5.0 with comment update |
| [gear_sonic/scripts/run_x2_vla_runtime.sh](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh) | +6 / -1 | HOLD default 10.0 → 5.0 with comment |
| [tests/test_pose_pipeline_blend_future_continuity.py](../../../../tests/test_pose_pipeline_blend_future_continuity.py) | +505 (new) | 28 new pure-pytest unit tests |
| [tests/test_run_x2_vla_runtime_sim_proxy.py](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) | +6 / -1 | Default-pin update with explanatory comment |
| [docs/source/user_guide/milestones/2026-06-23_blend_future_window_continuity.md](2026-06-23_blend_future_window_continuity.md) | +250 (new) | This milestone |
| [docs/source/user_guide/milestones/2026-06-08_arm_freeze_on_upstream_stall.md](2026-06-08_arm_freeze_on_upstream_stall.md) | +5 | Forward-pointer to this milestone in the "follow-ups" section |
| [docs/source/user_guide/milestones/index.md](index.md) | +1 toctree + 1 table row | Index entry |
| [docs/source/references/x2_blend_future_window_continuity.md](../../references/x2_blend_future_window_continuity.md) | new | Subsystem reference (long-form) |
| [docs/source/references/index.md](../../references/index.md) | +1 line | Reference index entry |
