# X2 Safety Incident Log

Running record of **unsafe events on real hardware** — loss of control, collapse, or
uncommanded motion. Separate from the deploy/regression docs on purpose: those record
whether a build is *good*, this records when the robot became *dangerous*.

Rules for this log:
- Every event gets an entry, including near-misses that a human caught. A collapse the
  operator happened to be holding is the same defect as one they weren't.
- Record what was **actually on the robot** at the time (models + code), not what we
  intended. Several entries below turned on that distinction.
- Record the **root cause when known** and **"unknown"** when it isn't. Do not
  retroactively assume a fix worked.

Related: `deploy_visual_regression_checklist.md` (pre-deploy gate),
`kplanner_sonic_handoff_g1_parity.md` (handoff fix).

---

## Cross-cutting root cause: SONIC cannot hold a frame

Most entries below share one mechanism. SONIC is a finite-bandwidth **tracking** policy,
not a stabilising controller. It computes actions from the delta between measured state
and a *moving* reference. Therefore:

- a **frozen** reference starves it → collapse
- a **gap** in the reference starves it → collapse
- a reference that **teleports** cannot be chased → collapse
- a reference that **rotates with the robot** is positive feedback → runaway

Any fallback, hold, pause, or transition must keep a *continuous, bounded, non-self-
referential* reference stream. "Hold the last pose" is never a safe fallback.

---

## INC-2026-07-17-A — Collapse on killing a live deploy

| | |
|---|---|
| **Date** | 2026-07-17 |
| **Severity** | HIGH — uncontrolled collapse, robot fell into operator's hands |
| **Injury/damage** | None; operator was supporting the robot |
| **Root cause** | CONFIRMED |

**What happened.** A live deploy was killed while the robot was standing. It collapsed
immediately and was caught by the operator.

**Root cause.** The assistant assumed "robot is lying down = safe to kill" while telemetry
showed `grav_z = -1.00` (upright). Killing the deploy removed the reference stream, and
SONIC collapses without one.

**Corrective actions.**
- Never kill a live deploy without an MC handoff.
- Read posture from **telemetry**, never from assumption or from what a log implies.
- Recorded as a standing rule.

---

## INC-2026-07-17-B — Collapse from missing idle stream at startup

| | |
|---|---|
| **Date** | 2026-07-17 |
| **Severity** | HIGH — collapse shortly after bring-up |
| **Injury/damage** | None |
| **Root cause** | CONFIRMED |

**What happened.** SONIC was brought up without a pose stream already publishing;
`pose_ref_age = -1.000s`. The robot collapsed.

**Root cause.** A `:5558` bind race — the pose proxy beat the watchdog to the port, so no
idle stream reached the policy. SONIC started with nothing to track.

**Corrective actions.**
- Watchdog starts **first**; pose proxy dropped from that path.
- A ZMQ pose-stream gate now blocks deploy start until frames are confirmed flowing.
- Rule: never start SONIC without a constant idle stream already publishing.

---

## INC-2026-07-18-A — Idle stream stalls, robot nearly collapsed (×2)

| | |
|---|---|
| **Date** | 2026-07-18, ~16:10 |
| **Severity** | HIGH — two near-collapses in one session |
| **Injury/damage** | None; operator was holding the robot both times |
| **Root cause** | CONFIRMED (two independent defects) |

**What happened.** During normal driving the idle/reference stream stopped a couple of
times and the robot was about to go down. The operator was standing next to it and caught
it each time.

**Evidence.**
```
16:10:34 WARNING loop fell behind by 452ms; resyncing
16:10:35 WARNING loop fell behind by 399ms; resyncing
...      328ms, 367ms, 387ms, 432ms, 503ms, 548ms
```
At 50 Hz a 400 ms stall is **20 missed pose frames**.

**Root cause 1 — publisher starvation (the trigger).** `OnnxPlannerBackend.replan()` ran
the full ONNX inference *inside* `replan_lock`. The 50 Hz publish loop takes that same lock
to read every frame, so each replan blocked the pose stream for its entire 300–500 ms
inference. The "async" replan worker was effectively serialised against publishing.

**Root cause 2 — the safety net was itself unsafe (why it wasn't caught).** The watchdog
*did* fire (`--idle-stale-ms 300`), but its first fallback rung is **HOLD**, which
re-publishes the last frame byte-for-byte — a **frozen reference**, for
`--hold-last-secs 5.0`. The only rung that keeps SONIC alive is the *looping* `IDLE_CLIP`,
which sat behind HOLD (5 s) + BLEND (3 s) = **8.3 s**.

Ladder as configured at the time of the incident:

| t | state | reference fed to SONIC |
|---|---|---|
| 0–300 ms | LIVE | stalled — nothing arriving |
| 300 ms | HOLD | **frozen frame, 5 s** |
| 5.3 s | BLEND | glide toward idle |
| 8.3 s | IDLE_CLIP | looping — the only safe rung |

HOLD's design assumption ("freezing the pose is harmless, it soaks up WiFi blips") was
written for the laptop-tethered era and is **false for a tracking policy**.

**Corrective actions.**
- Split `replan()` into `replan_prepare` / `replan_infer` / `replan_commit`; the lock is
  held only for the two cheap phases, inference runs unlocked. Verified motion-equivalent
  (0.35 vs 0.39 m/s; max frame jump 0.151 vs 0.160 rad).
- Tightened the watchdog ladder to reach a **looping** reference in ~300 ms:
  `--idle-stale-ms 100 --hold-last-secs 0.05 --blend-secs 0.15`.

**Open risk.** A 150 ms blend is inside the regime the watchdog's own docstring warns
about ("slamming through their full ROM in 200 ms"). If a stall occurs while the robot is
in a pose far from idle (mid-dance, arms extended), the *recovery* becomes a violent
150 ms whole-arm transition. Watch for arm snap on idle transitions; lengthen `--blend-secs`
if seen, independently of `--hold-last-secs`.

---

## INC-2026-07-18-B — Runaway spin, emergency battery pull

| | |
|---|---|
| **Date** | 2026-07-18, ~16:27 |
| **Severity** | **CRITICAL** — total loss of control; stopped only by removing the battery |
| **Injury/damage** | None reported |
| **Root cause** | CONFIRMED — self-inflicted, introduced same session |

**What happened.** On a right-stick input the robot began turning, then continued turning
after the command ended, accelerating out of control. The operator performed an emergency
stop by **pulling the battery**.

**Evidence** (`incident_20260718_spin/pc2_kplanner.log`, intent layout
`(yaw_rate, vel_x, vel_z, hip)`):
```
16:27:14 intent applied ... -> target=(-0.3, 0.0, 0.0, 0.687)      # right stick = yaw right
16:27:14 state: IDLE_LOOP -> PLAYING (intent=(-0.3, ...))
16:27:14 intent applied ... -> target=(0.0, 0.0, 0.0, 0.687)       # stick released
16:27:14 state: PLAYING -> IDLE_LOOP (intent back to idle); freezing at ...
```
The log tail is NUL padding — the filesystem had not flushed when power was cut, so the
final seconds are lost. What survives is sufficient.

**This exonerates intent-latching:** the command cleared correctly, and the planner
correctly entered IDLE_LOOP and froze its internal root quaternion. **The robot kept
spinning anyway.**

**Root cause — yaw positive feedback.** Earlier the same session, `_track_idle_yaw()` was
added so idle would stop fighting hand-nudges: it updated `yaw_offset` from the measured
IMU yaw on **every idle tick**. Previously that offset was captured **once at ignition**.

The planner froze its quat, but `_reb1()` then rotated that frozen quat by a `yaw_offset`
being continuously updated from the robot's *still-rotating* heading:

> robot turns → `yaw_offset` tracks new heading → published reference rotates further →
> robot turns more → repeat

A feedback path with **no restoring term and no rate limit**. The planner believed it was
holding still while the reference rotated with the robot.

**Why it was not caught.**
1. The change was reasoned as safe because "PLAYING freezes the offset." The actual hazard
   is the opposite state — **idle while physically rotating**, which is precisely the moment
   after a turn command ends. That state was never considered.
2. It is **not testable in sim**: the feature is gated on `measured_yaw`, sourced from
   `x2_debug`, which only the real deploy publishes. In sim `yaw_offset` stays `None` and
   the code is a no-op. This was known and flagged, and it went to hardware anyway.
3. It shipped on a night that already had three other first-on-hardware changes (softland
   4800 sonic, v3 idle anchor, tightened watchdog ladder), so there was no clean attribution
   if anything went wrong.

**Corrective actions.**
- `_track_idle_yaw()` call **disabled**; deployed and verified (0 active call sites). The
  function and a full explanation are retained in-source as a warning.
- The replan lock split (INC-2026-07-18-A) was **kept** — it is independent and verified.

**Requirements before any retry of idle-yaw tracking:**
- Hard guard: refuse to track while `is_playing`.
- **Slew limit** (rad/s cap) so a feedback loop cannot run away *even if the guard is wrong*.
- **Bounded total deviation** from the ignition heading.
- Changed **alone**, with the robot supported, with a stop plan agreed in advance.

---

## INC-2026-07-18-C — Violent turn to a different orientation

| | |
|---|---|
| **Date** | 2026-07-18 |
| **Severity** | MEDIUM-HIGH — abrupt uncommanded reorientation |
| **Injury/damage** | None |
| **Root cause** | **CONFIRMED — world coordinates baked into the idle stand reference** |

**What happened.** The robot performed a violent turn to a different orientation
(an orientation "snap" rather than a commanded turn).

**Root cause.** The idle anchor bakes an **absolute world pose**, and the loader applies it:

```
kplanner_idle_anchor_g1teleop_v2.pkl / _v3.pkl
  root_rot(xyzw) = [0, 0, 0, 1]      <- IDENTITY = world +X heading
  root_trans     = [0, 0, 0.665]     <- world ORIGIN
```

`pc2_kplanner_onnx._qpos_from_deploy_pkl_frame()` copies all of it into the published
reference, not just the joints:

```python
qpos[0:3] = f0_trans        # world origin
qpos[3:7] = f0_rot (wxyz)   # identity -> world +X
qpos[7:38] = f0_dof         # joints
```

So an unrebased idle publish **commands the robot to face world +X, at the world origin**,
regardless of where it actually is or which way it is pointing. Measured heading at the time
was `+81.7 deg` (from `x2_debug` in the watchdog log) — an ~82 deg orientation error that
SONIC closes as fast as it can. That is the violent turn.

**Naming trap.** The v2 clip's own key is **`idle_anchor_v2_joints_only`** — the name says
joints-only, but the loader takes `root_trans` and `root_rot` as well. Intent and behaviour
disagree, which is why this survived review.

**This reframes the entire yaw-rebase subsystem.** `MeasuredYaw` / `yaw_offset` / `_reb1()`
exist *solely to compensate for this defect* — its own docstring says so: rebase the published
root quats to the robot's actual heading, "else SONIC twists the body to world +X -- the
orientation snap". Consequences:

- The violent turn is what happens **whenever the patch is not armed**: `x2_debug` late or
  absent, the ignition-capture fail-safe timeout firing, or any path that publishes the raw
  anchor. The fail-safe explicitly "proceeds unrebased" — i.e. it fails *toward* the snap.
- INC-2026-07-18-B was a modification to this **patch layer** rather than to the defect
  beneath it. Editing a compensator for a world-frame bug is inherently fragile, and that is
  a large part of why a plausible-looking change produced a runaway.

**Correct fix (not yet implemented).** The idle reference must be **heading- and
position-relative**, never absolute:
- Ignore `root_trans` / `root_rot` from the anchor entirely — it is a *pose* asset, not a
  world placement — and seed the root from the robot's measured state.
- Then the rebase patch becomes unnecessary rather than load-bearing, and there is no
  unrebased path that can command world +X.
- Until then, treat any change to the rebase layer as safety-critical.

**Interim mitigation.** Do not rely on the ignition-capture fail-safe timeout: if `x2_debug`
never appears, the current behaviour proceeds **unrebased**, which is precisely the snap
condition. It should refuse to publish instead.

---

## INC-2026-07-18-D — Tilted / asymmetric idle stance

| | |
|---|---|
| **Date** | 2026-07-18 (observed; anchor predates this) |
| **Severity** | LOW-MEDIUM — no fall observed, but a degraded standing posture |
| **Root cause** | CONFIRMED (asymmetry); **partially fixed** |

**What happened.** At idle the robot stood with one leg planted well forward and that foot
turned out, and "always wants to stay in that odd position."

**Root cause.** The deployed idle anchor `kplanner_idle_anchor_g1teleop_v2.pkl` is
asymmetric (values verbatim in MuJoCo joint order, as the runtime loads them):

| joint | left | right | asymmetry |
|---|---|---|---|
| **hip_pitch** | +0.108 | **-0.284** | **0.392** |
| **hip_yaw** | -0.008 | **-0.362** | **0.370** |
| **ankle_pitch** | -0.199 | +0.087 | **0.286** |
| knee | +0.096 | +0.098 | 0.002 |

0.392 rad of hip_pitch stagger is the forward-planted leg; 0.370 rad of hip_yaw is the
turned-out foot.

**Corrective action.** Switched to `v3` (graft: symmetric legs, v2 arms), max asymmetry
0.044 rad, hip_pitch stagger 0.003 rad. Deployed and referenced by
`ritual_start_demo.sh`, `sim_onnx_planner.sh`, and the `pc2_kplanner_onnx.py` default.

**Open risk — NOT fixed by v3.** Both anchors stand at `knee ≈ 0.097` while the training
default is `0.669` and all six direction clips walk at **0.29–0.39**. So idle commands a
near-straight-legged stance far outside the trained operating point, and outside where the
gait lives. This is a plausible contributor to instability and to "wants to stay in that odd
position." A `v4` bringing knees to ~0.3 has been discussed but **not built**.

**Caveat on v3.** It is a composite pose that no captured frame ever produced. It is
symmetric and geometrically sane, but "SONIC can settle into it and hold it" is **untested**
— it went to hardware without a sim run of the full ritual.

---

## Patterns worth acting on

1. **Every entry above is a reference-stream defect**, not a policy-quality defect. No
   amount of model improvement addresses them. The reference contract — continuous,
   bounded, never frozen, never self-referential — is the safety-critical interface.
2. **The safety net was itself a hazard** (INC-2026-07-18-A). A fallback ladder whose first
   rung freezes the pose cannot protect a tracking policy. Fallbacks need testing against
   the actual failure they claim to cover.
3. **Feedback paths need rate limits, not just correctness arguments** (INC-2026-07-18-B). A
   guard believed correct is not a substitute for a bound that holds when it isn't.
4. **Untestable-in-sim changes are a distinct risk class.** `x2_debug`-gated behaviour cannot
   be exercised locally. Such changes must go to hardware alone, supported, with a stop plan
   — never stacked with other firsts.
5. **Near-misses are hits.** Three of these were survivable only because a human was holding
   the robot. That is not a control.

## Pre-flight rules distilled from these incidents

- Never start SONIC without a continuous idle stream already publishing.
- Never kill a live deploy without an MC handoff; read posture from telemetry.
- Never introduce a hold/freeze as a fallback; fallbacks loop.
- Never ship an `x2_debug`-gated change together with other first-on-hardware changes.
- Any feedback path gets a slew limit and a bounded total deviation.
- Support the robot physically for the first stand after **any** anchor, watchdog, or
  reference-path change.
