# Closed-loop tracking feedback on the VLA wire (2026-06-10 follow-up 11)

**Surface**: Real X2 + SONIC + N1.7 VLA bridge (`live_vla_publish_motion_token.py`).
**Outcome**: Bridge can now per-arm-joint-throttle the wire's per-tick
step using `x2_debug` proprio feedback, eliminating the open-loop
sensitivity to inference jitter / battery sag / motor temperature
drift that drove the 2026-06-10 PM oscillation incident. Default OFF
for Step 1 belt-and-suspenders rollout (the existing v3 LPF / blend /
scalar step-cap stack stays in place; feedback is additive); Step 2
in a separate commit will flip the default ON and relax the static
defaults once real-robot validation proves it out.

## Root cause: the open-loop wire is fragile

Recap from the 2026-06-09 wire-tuning iter milestone and the 2026-06-10
PM oscillation incident:

- The bridge is **fully open-loop**. It reads camera frames, runs the
  policy, decodes a chunk via the SONIC pose decoder, and writes the
  resulting joint targets to the wire at 50 Hz.
- It does NOT observe the robot's actual joint positions when shaping
  the wire. The only feedback path is the chunk-level rate cap
  (`--vla-max-wire-step`) which is a **scalar** that scales the
  whole delta vector by `max_step / peak`.
- This means: if SONIC's PID is fighting the wire (battery sag,
  motor temperature drift, contact, joint-limit lockup, etc.) the
  bridge has no way to know. It keeps pushing the wire forward at
  its static rate cap, the actuator can't keep up, and the
  difference accumulates as a tracking error that SONIC tries to
  correct -- producing the 1-1.5 Hz post-SONIC oscillations the
  operator saw on 2026-06-10 even though the v3 defaults
  (`target_lpf_hz=5`, `chunk_blend_ticks=40`, `max_wire_step=0.07`)
  were unchanged from the previous stable run.

The smoking gun was the parquet capture:

- **Pre-SONIC wire** (`action.body_q_mj`): smooth ~1 Hz oscillation,
  amplitude ~0.09 rad on `R_sh_r`. Already not great but not the
  source of the operator-visible vibration.
- **Post-SONIC measured** (`observation.state` -- echoing
  `body_q_mj` since real-robot lacks true proprio): same frequency,
  amplitude ~0.37 rad on `R_sh_r`. **~4x amplification by SONIC**.

SONIC has its own loop that the bridge doesn't model. Static tuning
("dial in the LPF / blend / step-cap once and never touch them") is
necessarily a compromise: too tight and the wire is sluggish during
fast motions; too loose and SONIC amplifies any residual wobble.

## Fix: closed-loop per-joint tracking feedback

Add a feedback mechanism that **observes** the measured joint state
(`body_q_mj`, `body_dq_mj` from `x2_debug`) and per-joint-throttles
the wire's per-tick step in real time:

1. **Position backoff**: if `|target - measured| > soft_rad` for a
   joint, scale that joint's step cap linearly from `base` (at soft)
   down to 0 (at `hard_rad`). Joints whose actuators are saturating
   get frozen until they catch up.
2. **Velocity cap**: cap each joint's per-tick step by `vel_margin *
   |measured_dq| * dt` (with a small floor `vel_floor_rad_tick` so
   cold starts from rest still work). Prevents the wire from
   commanding faster than the actuator is currently moving.

The two caps combine via `min`: the joint moves at the smaller of
"how much position error allows" and "how fast the actuator is
already moving". When proprio is fresh and the actuator is tracking
well, both caps are well above the base step and the law is a no-op.
When the actuator falls behind, the law throttles the wire **only
for the affected joint**, not the whole vector.

### Why per-joint, not scalar

The existing `_clamp_vector_step` is a scalar clamp: it scales the
entire delta vector by `max_step / peak`, preserving the wire's
per-tick direction. The new helper `_clamp_vector_step_per_joint`
clamps each joint independently. The trade-off:

- **Scalar** preserves wire direction but couples joints: when one
  joint saturates, the whole arm slows down.
- **Per-joint** preserves per-DOF responsiveness but the wire's
  per-tick direction may shift slightly when one joint is
  throttled. This shift is exactly what a per-joint actuator with
  PID feedback would produce anyway, so it matches the physical
  reality the policy will encounter.

Per-joint is the right choice for tracking feedback because the
common failure mode is "one joint hits a contact / limit / temp
issue while the others are fine". With the scalar clamp those other
joints would also slow down for no reason.

### Joint mask: arms only

Tracking feedback only applies to the **14 arm joints** (MJ 15..28
= 7 left arm + 7 right arm). Legs (0..11), waist (12..14), and head
(29..30) pass through with the base step:

- **Legs + waist**: SONIC owns the balance loop; tracking feedback
  would fight it and produce stuck legs or a body that can't
  recover from a load disturbance.
- **Head**: locked straight by `--lock-head-straight` on the PC2
  daemons; feedback would interact badly with that pin.

### Fallback paths

The feedback is **gated on data freshness**. If `x2_debug` is stale
(`> --vla-tracking-stale-ms`, default 100 ms = 5 publish ticks), the
bridge transparently falls back to the existing scalar clamp for
that tick. Same for missing proprio (no q or dq snapshot yet) and
for shape mismatches (schema drift). The contract: **disabled or
stale = byte-identical to legacy scalar clamp**. The unit tests pin
all four fallback paths.

## Step 1 rollout: belt-and-suspenders

The plan calls for a **two-step** rollout to keep regression
attributable:

### Step 1 (this commit)

- Default OFF: operator opts in via `--vla-tracking-feedback` or
  `VLA_TRACKING_FEEDBACK=1`.
- All existing v3 defaults stay in place
  (`target_lpf_hz=5`, `future_lpf_hz=5`, `chunk_blend_ticks=40`,
  `max_wire_step=0.07`). Feedback is **additive** -- when enabled
  it composes with the existing wire-shaping stack, never replaces
  it.
- This way, the operator can validate that feedback ON produces
  smoother motion than feedback OFF using IDENTICAL static
  defaults, isolating the feedback's contribution.

### Step 2 (separate commit, after 2+ successful real-robot runs)

- Default flips to ON.
- Static defaults relax: `target_lpf_hz 5 -> 8`,
  `future_lpf_hz 5 -> 8`, `chunk_blend_ticks 40 -> 10`,
  `max_wire_step 0.07 -> 0.05`. The wire becomes less defensive
  and more responsive, relying on the closed loop to handle the
  exceptional cases the static defaults were padding against.
- Separate milestone so any regression in real-robot smoothness
  is attributable to the right cause (relaxed statics vs.
  feedback law itself).

## Operator runbook

### Enable feedback (Step 1)

Add `--vla-tracking-feedback` to your existing launcher command. Or
export `VLA_TRACKING_FEEDBACK=1` in your shell rc. Everything else
stays the same.

```bash
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --pc2-host pc2.local \
    --vla-tracking-feedback                                # <-- new
```

### Read the telemetry

When feedback is enabled, the pub-tick log line gains a
`tf_throttle=N/14` field:

```
[live-VLA] pub tick= 12345  chunk_id= 234 step= 5/40  |token|=0.234  |left|=0.123  VLA-pose raw_Δ=0.080rad wire_Δ=0.045rad body_Δ=0.030rad tf_throttle=3/14  deploy_alive=True
```

- `tf_throttle=0/14`: all 14 arm joints are tracking cleanly;
  feedback is essentially a no-op.
- `tf_throttle=N/14` for some N: that many arm joints are being
  actively protected (their per-tick step is below 50% of base).
  Brief excursions during fast motion are normal.
- Sustained `tf_throttle=14/14`: the actuator is saturating across
  the board (battery near dead, severe contact, etc.). Operator
  should consider stopping the run.

### Disable feedback (mid-session A/B)

`--no-vla-tracking-feedback` overrides the env var, so an operator
who has `VLA_TRACKING_FEEDBACK=1` in their persistent rc can A/B
against the scalar clamp without changing their shell environment.

`--vla-raw` (the existing "disable all wire shaping" flag) now also
disables tracking feedback for consistency. The launcher warns if
the operator passes `--vla-tracking-feedback` together with
`--vla-raw`.

### Tunables

Defaults are conservative (tested in sim, tuned by the law's
geometry). The operator-facing knobs:

| Flag | Default | Purpose |
|------|---------|---------|
| `--vla-tracking-soft-rad` | 0.15 rad | Position error below which no backoff |
| `--vla-tracking-hard-rad` | 0.40 rad | Position error above which joint frozen |
| `--vla-tracking-velocity-margin` | 1.5 | Max wire speed = `margin * actuator speed` |
| `--vla-tracking-velocity-floor-rad-tick` | 0.01 rad/tick | Floor for cold-start motion |
| `--vla-tracking-stale-ms` | 100 ms | Proprio staleness threshold |

If you see sustained throttling that you don't think is warranted
(e.g., actuator clearly responsive, tracking error visually small),
loosen `soft_rad` first. If the wire feels jerky despite throttling
correctly, tighten `velocity_margin` first.

## Files touched

### Bridge

- `gear_sonic/scripts/live_vla_publish_motion_token.py`
  - New `_clamp_vector_step_per_joint` helper (per-joint element-wise
    clamp; sibling to existing scalar `_clamp_vector_step`).
  - New `_apply_tracking_feedback` helper (the feedback law).
  - New `_ARM_JOINT_INDICES` constant (the joint mask).
  - New publisher params:
    `tracking_feedback_enabled`, `tracking_soft_rad`,
    `tracking_hard_rad`, `tracking_vel_margin`,
    `tracking_vel_floor_rad_tick`, `tracking_stale_ms`. All default
    OFF / conservative.
  - Hot-loop integration at the existing two scalar-clamp sites
    (decoder-succeeded branch + idle/decoder-skipped branch). When
    feedback is enabled AND proprio is fresh, the scalar
    `effective_max_step` becomes the per-arm-joint UPPER bound and
    each joint's individual cap is throttled by the feedback law.
    Falls back to the scalar clamp transparently otherwise.
  - Pub-tick log line gains `tf_throttle=N/14` (only when feedback
    enabled; legacy logs unchanged).
  - 6 new CLI flags + `--no-vla-tracking-feedback` counterpart.

### Launcher

- `gear_sonic/scripts/run_x2_vla_runtime.sh`
  - 6 new env-var defaults (`VLA_TRACKING_*`) + matching CLI parsing.
  - Early `log` call announces ENABLED / DISABLED state so the
    operator can confirm feedback state even on failed preflights.
  - `--vla-raw` also disables tracking feedback (with operator warn
    if conflict).
  - Banner row added.
  - `--help` heredoc extended with all 6 flag descriptions.

### Tests

- `tests/test_vla_tracking_feedback.py` (NEW, 30 tests)
  - 5 tests on the per-joint clamp semantics (cap=0 freezes,
    negative=no-cap, shape mismatch raises, etc.).
  - 5 tests on the feedback law's fallback paths (no prev, no q,
    no dq, base=0, shape mismatch).
  - 5 tests on the position backoff (zero error, below soft, at
    soft, midpoint, above hard).
  - 5 tests on the velocity cap (floor at rest, floor=0 freezes
    at rest, high velocity lifts cap, pos vs vel binding, negative
    velocity).
  - 3 tests on the joint mask (only arms throttled, mask constant
    pinned at 14 DOFs, custom mask).
  - 3 tests on the throttle counter (zero / partial / full).
  - 3 end-to-end integration tests (clean motion, lagging joint
    isolated, 50-tick convergence sim).
  - 1 divergence pin (scalar scales direction, per-joint clamps
    element-wise).

- `tests/test_run_x2_vla_runtime_sim_proxy.py` (EXTENDED, 5 new
  tests)
  - All 7 new CLI flags surfaced in `--help`.
  - Launcher accepts `--vla-tracking-feedback` and the 4 threshold
    flags without falling through to the catch-all.
  - Default-OFF state shows up in the early log line.
  - `--no-vla-tracking-feedback` overrides env `VLA_TRACKING_FEEDBACK=1`.
  - `--vla-raw` disables tracking feedback with operator warn.
  - `BRIDGE_ARGS+=` block for tracking feedback is materialised
    (greps the launcher source).

## What to validate on real robot (Step 1)

1. **Smoke**: same recipe as previous successful runs, with
   `--vla-tracking-feedback` added. Verify the bridge logs
   "tracking feedback ENABLED" at start and the pub-tick line
   gains the `tf_throttle=N/14` field after first chunk decode.
2. **A/B against scalar clamp**: run the same task with and
   without `--vla-tracking-feedback` (same model, same defaults
   otherwise). Compare:
   - Pre-SONIC vs post-SONIC oscillation amplitude on the same
     joint (`R_sh_r` is the canary).
   - Hand smoothness during the grasp.
   - `tf_throttle` time-series (should be mostly 0, brief
     excursions to 1-3 during fast motion).
3. **Failure modes** to watch for:
   - **Stuck arm**: feedback could over-throttle if `soft_rad`
     is too small. Symptom: wire freezes for many ticks at a
     time. Fix: loosen `--vla-tracking-soft-rad`.
   - **Sluggish wire**: feedback could over-throttle the
     velocity cap. Symptom: wire moves but slowly even when
     actuator is responsive. Fix: increase
     `--vla-tracking-velocity-margin`.

## What's deferred to Step 2

- Flipping the default to ON.
- Relaxing the v3 static defaults:
  - `target_lpf_hz 5 -> 8` (more responsive low-pass)
  - `future_lpf_hz 5 -> 8`
  - `chunk_blend_ticks 40 -> 10` (less inter-chunk smoothing)
  - `max_wire_step 0.07 -> 0.05` (slightly tighter scalar cap
    as a backstop)
- A new milestone documenting the Step 2 results.

The whole point of Step 1 is that we can attribute any real-robot
regression to either (a) the feedback law itself (if Step 1 is bad)
or (b) the relaxed statics (if Step 1 is good but Step 2 regresses).
