# kplanner unlocked — GPU inference on PC2, in-place 360° turns, verdict chain

**Milestone (2026-07-20): first fully-working kplanner deploy.** Smooth
forward walks, back walks, stop transitions, and — for the first time ever —
**full 360° standing in-place turns in BOTH directions** on hardware
(session `robot_runs/20260720_080239_gpu_first_session_360turns`).

## What shipped

1. **Orin GPU inference** (`--ort-gpu`, in the ritual): the venv already
   carried a Jetson CUDA build of onnxruntime (masked by a stray CPU build —
   uninstalling the CPU dist unmasked it; no network needed). Replan latency
   **270–620 ms → 72 ms median / 105 max**, seam fast-forward 10.8–20.4 →
   **1.8 frames**, walk-onset anchor gap ~0.48 s → ~0.1 s, zero starvation.
   PC2 timing now equals the workstation sim-stack that always walked smooth.
2. **Wire-frame fix** (`_reb_xy`): root XY is now rebased by the same
   ignition yaw offset as the quat. Tape-only bug (SONIC never consumes root
   XY) but it made every capture show forward walks as world-frame
   crab-walks until the operator caught it.
3. **PLAYING yaw resync** (`--playing-yaw-resync-dps 10`, in the ritual):
   mid-walk heading compliance — assignment-form, slew-limited (10°/s),
   deadbanded (2°), gated off while a turn is commanded, trim clamped ±34°.
   Extends the IDLE compliance so nudges and the root head's open-loop yaw
   wander (6–33°/walk; reference-led whip at the worst 2026-07-19 stumble)
   bleed off instead of accumulating into violent SONIC corrections.
4. **Deploy-twin sim harness** (`record_x2_eval_mujoco.py --tuning-yaml`):
   the robot's real conditioning (soft-kp gain scales, max-target-dev
   clamps, target LPF) applied in sim. MANDATORY for any SONIC eval — with
   it, the full 140 s stumble-session wire tracks with 0 falls / 0 spikes
   (without: 14 fake falls). Plus `--auto-reset`, action/joint CSV logging,
   and a prototype `--wholebody-rate-limit` (L2-norm budget on the aggregate
   target step — the whole-body complement to per-joint clamps; calibrate
   before any deploy port).
5. **Overlay + capture tooling**: `overlay_run_mujoco.py` (ghost planner
   reference vs solid robot telemetry in one scene, `--save-video`),
   frame-consistent tapes, chunk dumps.

## Why the 360° turn works (mechanism, from the tape)

The pose head STILL cannot generate turn stepping — during the 360s the
liveness gate re-rolled 203 times and mostly committed standing chunks
(hip-std ~0.023–0.030). The unlock is the **root heading stream + SONIC's
own turning skill**: the root head rotates the reference heading correctly
(1.03× certified), and SONIC — trained on turning references — does its own
footwork to follow a smoothly rotating orientation target. GPU latency is
what made the heading stream smooth (72 ms updates vs 0.3–0.6 s lurches
that previously triggered violent corrections instead of turns).

Consequences: the "in-place turns are an architecture limitation" verdict
stands at the MODEL level but is **deploy-irrelevant**; the canned-primer
plan and the fullmask training bet are superseded at the deploy layer.
Efficiency note: during yaw-only intents the gate burns 3× inference
re-rolling chunks whose stillness doesn't matter — candidate exemption.

## The verdict chain that got here (full detail in
`robot_runs/20260719_223057_walks_residual_stumbles/ANALYSIS.md`)

Three deploy-runtime defects (double-replan race, ring starvation, seam
rewind — see `kplanner_replan_seam_rewind_fix_20260720.md`) → first clean
walk. Residual stumbles then isolated by test matrix: joints exonerated
(motors low-pass SONIC's commands), PC2 plumbing exonerated (zero-latency
replay no better), reference content exonerated (deploy-tuned sim tracks
everything), leaving the real-world layer at content stress points — and
the two levers above (latency, heading compliance) attack exactly those.
Three operator catches drove the corrections: windowed metrics hiding
freezes, sim falls being harness artifacts, and the crab-walk tape bug.

## Still open

- Reference quality flags (model-side, moderate): foot-skate 0.20–0.31 m/s
  (~2× a good clip) and root overspeed (~0.5 m/s at 0.3 command) — measured
  on clean offline replay; margin-wideners for a future training pass.
- Resync hardware validation (nudge test) — first session pending.
- Whole-body rate limit: calibrate budget on known-good recoveries in sim.
- p500k planner promotion test via the instrumented sim-stack flow.
- Gate exemption for yaw-only intents (3× inference waste).
