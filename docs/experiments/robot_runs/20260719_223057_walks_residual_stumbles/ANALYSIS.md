# Residual-stumble session — census + attribution (post-rewind-fix)

Run: `20260719_223057_walks_residual_stumbles` (daemon md5 ca294af85046, all
recorders live). Operator report: "a few more walks... there are still some
stumbles." Includes **back walks** and one **in-place turn attempt**.

Clock: robot telemetry → tape clock offset **5354.72 s** (hip-pitch
cross-correlation, not single-anchor). All times below are tape-clock seconds.

## Session contents (16 command segments)

- 11 forward walks (94.3, then 464.5–585.3 block), all `fwd=0.3`
- 4 back walks: 452.5–458.9 (6.3 s) and three short bursts 543.9–552.0
- 1 in-place turn attempt: 488.3–490.2 (`yaw=0.3`, fwd=0)
- One 120 ms mid-walk yaw tap at 483.54 (inside walk #3)

## Deploy-fix health check (the previous defects stayed fixed)

- 0 starvation events; wire continuously paced; `commit_ff` fired at every
  seam (10.8–20.4 frames).
- Every chunk generated *while walking* (fwd or back) was vigorous:
  hip-pitch std 0.13–0.27, zero stills across ~100 walking chunks.
- Liveness gate: 4 firings, **all during the in-place turn attempt**
  (hip std 0.029–0.040) — first live catch of the still-chunk emission; 3
  re-rolls + 1 exhaustion commit, right as the stick was released. Known
  architecture limitation (shared with stock G1), not a deploy defect.

## Stumble census (12 leg-vel spike episodes > 10 rad/s, full-session scan)

| tape t | context | peak rad/s | tilt° | attribution |
|---|---|---|---|---|
| 455.5 | back walk 1 | 13.1 | 10.9 | back-gait tracking |
| 456.6 | back walk 1 | 10.2 | 2.9 | back-gait tracking |
| 458.3 | back walk 1 | 12.1 | 5.9 | back-gait tracking |
| 458.9 | back walk 1 end | 13.9 | **15.6** | back-gait + stop |
| 484.8 | walk 3, 1.2 s after yaw tap | 11.1 | 11.0 | course-change seam (wire bend 0.074 rad/tick, ~2× normal) |
| 492.4 | idle after turn attempt | 12.2 | 4.5 | robot self-recovery (wire perfectly still) |
| 494.8 | idle | 10.5 | 5.4 | robot self-recovery (wire still) |
| 502.6 | walk 4 STOP | 12.1 | 10.3 | stop-blend transition |
| 517.7 | idle | 10.3 | 5.7 | settling wobble (wire still) |
| 520.7 | walk 6 mid, seam −0.18 s | 12.2 | 6.0 | mild seam wobble |
| 545.6 | back walk 2, seam −0.03 s | 14.3 | 13.3 | back-gait + seam |
| 584.9 | walk 10 STOP | **17.0** | 13.8 | stop-blend transition (worst event; wire jump 0.084 rad/tick) |

Caveat: 10 rad/s threshold — milder stumbles the operator felt may sit below
it (operator disputes "9 of 11 forward walks clean"); rescan at lower
threshold pending.

## Failure families (ranked)

1. **Backward-gait tracking (5/12 episodes, tilts to 15.6°).** Planner
   reference healthy throughout (chunks vigorous, wire continuous) — the
   divergence is SONIC-side. Backward gait is rare in tracker training data.
2. **Stop transitions (2 episodes, incl. the worst).** Stop-blend to anchor
   produces the largest wire discontinuities of the session (0.084 rad/tick).
   Was a "polish item" (0.48 s anchor gap); now the top forward-walk offender.
3. **Course-change seams (1).** Mid-walk intent change → forced replan with a
   sharper reference bend than the 8-tick blend absorbs.
4. **Idle settling (3, mild).** Wire provably still; robot catching its own
   balance. Not a planner event.
5. **In-place turns.** Architecture limitation; gate now contains it.

## Reference viewing

Video (in this dir): `overlay_planner_vs_robot_450_590.mp4` — ghost blue =
planner wire, solid = robot measured, shared root translation (no odometry in
telemetry; judge limbs/tilt, not XY), IMU yaw-aligned at window start.

Interactive:

```bash
python gear_sonic/scripts/overlay_run_mujoco.py \
  docs/experiments/robot_runs/20260719_223057_walks_residual_stumbles \
  --window 452 460 --clock-offset 5354.72        # back walk 1
# 583 587 = worst stop-transition; 526 532 = clean walk contrast
```

Landmarks inside the 450–590 window: back walk 1 at +2–9 s, yaw-tap walk at
+33–36 s, turn attempt +38–40 s, walk-4 stop +52 s, short back walks
+93–102 s, worst stop event +134–135 s.

## Isolation campaign (2026-07-20): models vs plumbing vs PC2 vs hardware

Question: are the residual stumbles the models, the planner→SONIC plumbing,
PC2 resource starvation, or a sim-to-real / joint-hardware gap?

**T0 — SONIC commands vs measured joints (existing capture, free).**
At every stumble episode SONIC's own commanded action rate spikes to 118–192
(clean-walk baseline: median 20, p99 57) while measured joint velocity only
reaches 10–17 rad/s. The motors LOW-PASS SONIC's commands; the joints are not
misbehaving. Lead/lag: in 10/12 episodes gyro/joint disturbance precedes the
command spike (policy reacting), 584.9 is command-first. **Joint hardware
exonerated.**

**T1 — captured wire replayed through the deployed policy in MuJoCo**
(`wire_*.pkl` sliced from frame tape; `softland_4800` via its source .pt,
PT↔ONNX parity ~1e-7; `record_x2_eval_mujoco.py --traj-csv`, now with
action/joint logging + `--auto-reset`):

| window | robot | sim (same reference) |
|---|---|---|
| back walk (452–460) | 4 episodes, tilt 15.6° | same escalating spikes at same tape times, tilt 32.8° |
| clean fwd walk (526–532) | clean | clean (tilt 4.3°) — control validates method |
| walk-10 stop (581–587) | 17 rad/s, tilt 13.8° | fell (window includes walk onset) |

**The failure follows the reference into a clean sim on a fast machine →
not PC2 resources, not zmq plumbing, not hardware.**

**T3 — ideal-plumbing reference** (`replay_intent_tape.py --sim-latency-ms 0`,
recorded seeds; root seam-sawtooth artifact of the replay tool clamped;
windows re-cut to start mid-walk after the replay's idle-at-origin quirk):

| window | recorded wire | ideal reference |
|---|---|---|
| back walk (mid-walk start) | tilt 19.9° | tilt 32.3° |
| clean fwd walk | 6.6° | 6.9° |
| walk→stop incl. stop blend | 6.1° | 14.6° |

Ideal plumbing does NOT rescue the hard content (backward gait stays bad) and
the recorded wire is not worse than ideal — **the deploy pipeline's seams are
not the driver of the residual stumbles**. Caveat: the ideal wire is a
different rollout (seed-schedule drift), so treat magnitudes loosely.

Mid-walk starts never fall; windows that begin at the idle→walk boundary
fall or spike hard in BOTH references → **walk-onset transitions are a real
stress point** (matches the robot's early-walk episode cluster and the known
0.48 s anchor-gap item).

**Full-session robot-parallel rollout** (140 s, auto-reset): 14 falls vs
robot's 0 — sim amplifies. First-fall sites align with content classes:
back-walk onset (452.6), first stop transition (469.5), in-place-turn still
chunks (488.8), post-stop settle (517.4). Post-reset falls chain (fresh
zero-history proprioception), so per-episode matching beyond the first fall
of each cluster is not meaningful.

### Verdict

| axis | verdict |
|---|---|
| joint hardware / sim-to-real | **exonerated** (T0 + T1) |
| PC2 resources / starvation | **exonerated for these stumbles** (reproduce with zero latency, fast machine) |
| planner→SONIC plumbing (seams) | **not the driver** (T3); walk-onset anchor gap remains a real transition stress |
| models: SONIC × reference content | **confirmed**: backward gait, walk-onset and stop transitions, in-place-turn stills destabilize the tracker; steady forward walking is robust |

### RE-REVISION (operator-caught #2): the "sliding" was a TAPE FRAME BUG,
### and the skate numbers below are inflated by it

The operator asked why forward-only commands produced sideways root motion.
Chasing it found a **daemon publish bug**: the root QUAT is rebased by the
ignition yaw offset (`_reb1`) but the root XY was published in the raw
planner frame — position and orientation disagreed by the ignition heading,
so forward walks looked like world-frame crab-walks (lat −1.5..−2.6 m vs fwd
0.4..1.3 m per walk) in every tape consumer (overlay, gait metrics, this
doc's skate table). Proof of mechanism: the same ONNX graph replayed offline
(self-consistent frames) walks straight (fwd +1.8..+3.0 m, |lat| ≤ 0.3 m);
the pose stream's feet swing forward (+0.9..1.1 m/s) even in the buggy tape.

**Impact scope: analysis-only.** SONIC's tokenizer obs = joints + relative
orientation 6D; reference root XY never enters the obs, on the robot or in
the sim harness — so no robot behavior and no sim result was caused by this
bug. FIXED in `pc2_kplanner_onnx.py` (`_reb_xy` applied at all three publish
sites; internal planner-frame state untouched); needs ship to PC2 + ritual
restart.

True reference foot-skate, recomputed on properly-framed replay output:
0.20–0.31 m/s (vs 0.115 known-good, 0.085 floor) — elevated ~2×, a real but
moderate root↔pose consistency gap, NOT the 4–6× catastrophe below. The
overspeed observation (root ~0.5 m/s vs 0.3 command, visible in the clean
replay too) stands as a genuine model-side calibration flag.

### SUPERSEDED by the above — kept for the record: the original skate table
### (computed on the mis-framed tape; numbers inflated)

The operator noticed world-frame sliding in the full-session replay. FK on
the wire confirms (frame_series foot-slide, stance-foot planar speed while
in contact; known-good clip walk_circle_001 ≈ 0.115 m/s; idle noise floor
0.085):

| segment | slide mean m/s | root speed m/s |
|---|---|---|
| FWD walks 4/7/9 | 0.73 / 0.68 / 0.56 | 0.57 / 0.59 / 0.46 |
| BACK walks 1/2 | 0.44 / 0.39 | 0.35 / 0.32 |

**The planner's published references skate at 4–6× a good clip — stance feet
glide at ~root speed. The reference walks kinematically but is dynamically
impossible.** This revises the verdict: SONIC is not failing a fair
reference; it absorbs the skate in steady forward gait and tips over where
extra difficulty stacks (backward gait, onsets, stops). Also consistent with
sim (different contact model) falling more than the robot on identical
references. Root cause direction: root-head vs pose-head consistency is only
learned, never enforced (same family as the in-place-turn
root-spins-legs-don't diagnosis). OPEN FLAG: wire root speed is 1.6–1.9× the
0.3 command — re-verify the deployed 30→50 Hz resample path (fps-handoff
investigation) before trusting absolute speeds.

Implication for next steps: planner-side root↔pose consistency (and a
foot-skate metric in eval gates / per-chunk deploy telemetry) is
higher-leverage than SONIC retraining alone.

### FINAL VERDICT (supersedes the matrix above): harness-matched sim
### tracks the ENTIRE session cleanly — the earlier sim falls were
### harness artifacts, and the "models confirmed" verdict is OVERTURNED

The operator rejected the sim collapses ("the real robot never collapsed —
artifacts somewhere") and was right. The harness was missing the deploy's
target conditioning: `walking_soft_kp.yaml` gain scales (hip kd 0.476×,
ankle kd 2.0×...), max-target-dev clamps (0.7 rad legs) and the 16 Hz
target LPF. Wired in as `record_x2_eval_mujoco.py --tuning-yaml` (deploy
twin: clamp → LPF → scaled PD).

Result on the full 140 s session wire, matched conditioning:
**0 falls (was 14), 0 spike episodes >10 rad/s (robot had 12), tilt max
9.1°.** Back walk tilt 5.5°, stop 8.0°, turn attempt 3.3°.

Reading:
- The planner's reference content — back walks, onsets, stops, turn stills —
  **is trackable** by the deployed SONIC under deploy conditioning. The
  earlier content-class falls were the untuned harness overreacting at the
  content stress points (the correlation was real, the failures were not).
- The robot's 12 real stumble episodes do NOT reproduce in matched sim →
  they live in the **real-world layer** sim doesn't model: contact/surface
  friction, onboard state estimation and IMU noise, actuator
  friction/backlash, thermals — interacting with thin margins at the same
  content stress points (which is why they cluster there).
- The deploy conditioning stack is not incidental protection — it is
  integral to closed-loop stability. Any sim evaluation of SONIC without
  `--tuning-yaml` is measuring a different controller.
- Reference-side flags that remain real (from the clean offline replay):
  moderate foot-skate 0.20–0.31 m/s (~2× a known-good clip) and root
  overspeed (~0.5 m/s at 0.3 command). Margin-wideners, not proven causes.

### Heading-adherence investigation (operator observation #3)

Operator: idle allows gentle manual turning (idle yaw resync), but nudging
~15° MID-WALK makes joints "click" and the robot fights back to the
commanded world heading. Confirmed and quantified:

- PLAYING publishes the planner's open-loop world-frame heading verbatim
  (`_resync_idle_yaw_from_measured` is IDLE-only by design). SONIC's obs is
  relative-orientation 6D → any reference↔measured yaw gap is error it
  corrects, hard: measured corrections of 10–20° inside 0.25–0.5 s.
- Reference-vs-IMU heading mismatch grows in EVERY walk (resynced to 0 at
  each idle, wanders to 6–33° by walk end — planner root yaw drift is
  open-loop). Worst stumble (584.85, 17 rad/s) was REFERENCE-LED: wire yaw
  wandered −7° in 0.5 s near the stop, then the correction whipped the robot
  +12° opposite → oscillation, tilt 13.8°. Back-walk stumble (455.5) was
  ROBOT-LED (slip +17°), then harshly pulled back in 0.5 s.
- IMU exonerated: yaw-rate bias +0.09 °/min (quietest hands-off window);
  the 20–30 °/s "idle" readings elsewhere are the operator physically
  turning the robot between walks.

Direction (NOT yet designed): PLAYING-scope yaw resync — assignment-form
like idle (never multiply: 2026-07-18 runaway spin), slew-limited
(~10 °/s) with a small deadband, so the reference gently follows the
robot's true heading mid-walk instead of fighting it. Prior art warning:
"revert yaw feedback" (commit a235e65) — whatever is tried must be
evaluated in the instrumented sim stack (pose-feedback bridge + MuJoCo
push tooling) before hardware.

### Concrete next steps (revised per final verdict)

1. Adopt `--tuning-yaml` as MANDATORY for every SONIC sim eval; the tuned
   full-session rollout is now the regression baseline (0 falls, 0 spikes,
   tilt ≤9.1°). Ship the daemon `_reb_xy` fix to PC2 (tape-only bug, but
   every future capture depends on it).
2. Margin-wideners at the reference stress points (cheap, already scoped):
   onset pre-roll through the 0.48 s anchor gap, longer stop blend, optional
   backward-speed cap. These attack where real-world margins are thinnest.
3. Chase the real-world layer directly: next robot session, capture with the
   frame-tape fix in place and diff sim-tuned vs robot at the SAME episodes —
   candidates are estimation/IMU noise at transitions, foot-contact events,
   and actuator friction/thermals. Reference skate/overspeed calibration
   (model-side) widens margins too.
4. SONIC retraining on backward/transition-dense data is DEPRIORITIZED — the
   tuned sim shows the current policy tracks this content.
