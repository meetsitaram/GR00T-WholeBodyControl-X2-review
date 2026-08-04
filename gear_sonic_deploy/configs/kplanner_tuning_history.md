# Kplanner locomotion tuning history

Side-by-side record of the deployed kplanner intent/cadence knobs so config
tuning never loses track of what changed, when, and on what evidence.
Knobs land in `x2_pc2/ritual_start_demo.sh` (envs + planner flags); sweep
tooling: `out/kplanner_maneuver_sweep/maneuver_sweep.py` (reference-level
maneuver metrics vs the SHIPPED ONNX graph) and
`gear_sonic/scripts/kplanner_turnrate_sweep.py` (in-place rates).

## 2026-08-02 — kitchen-teleop tuning (VR hop-in era)

| Knob | Old | New | Evidence (maneuver sweep, template graph) |
|---|---|---|---|
| `KPLANNER_FIXED_FWD_MPS` | 0.50 | **0.40** | slow_walk template has a ~0.45 m/s gait floor: cmd 0.3 walks 0.43–0.45 with speed-hunting jerk (accP95 3.0–4.9); cmd 0.4 walks ~0.48 at accP95 ~1.7 (smoothest); cmd 0.5 walks 0.50–0.55 |
| `KPLANNER_FIXED_ARC_TURN_RAD_S` | 0.55 (default) | **0.70** | walking-turn radius at 0.4 m/s: ~1.2 m → 0.6–0.85 m, heading-step p95 1.3–1.6°; 0.90 gives R 0.4–0.6 m but p95 2.2–2.5° (choppy) — candidate next step |
| `--replan-threshold-frames` | 32 (default) | **48** | turn-cmd response at 0.45 s PC2 inference: 0.68 s → 0.22 s (3×); at 0.6 s: 0.82 → 0.40 s; seam cost unchanged (4 repeat frames ≈ 80 ms) |
| `KPLANNER_FIXED_TURN_RAD_S` | 1.0 | 1.0 (kept) | July 2026-07-21 turnrate sweep: clean to 1.3 rad/s, robot-verified at 1.0 |
| `KPLANNER_ARC_FWD_BOOST` | 1.0 | 1.0 (kept) | reference-level: boost only widens arcs (R×1.3) and broke 90° captures (58–68°); the under-translation it targets is tracked-level — revisit only if hardware arcs under-travel |

Validated: 36 s scripted deploy-sim run (straight, wide/tight arcs, 90°,
U-turn, in-place, stop) with the new combo — upright throughout, zero
SAFE_IDLE trips, zero ring starvation.

Known model-level limits (knobs cannot fix; training work):
- True ≤0.35 m/s walking (template speed floor; slow-walk dead-band)
- In-place-turn quality / turn-in-walk crispness → kplanner retrain
  (root/vqvae/pose) with more in-place-turn clips

### Why raising `--replan-threshold-frames` increases responsiveness
(verified against the MotionBricks SIGGRAPH 2026 paper + code, 2026-08-03)

The paper's reference design has TWO replan triggers (Alg. 1: "C changed
OR |B| running low"; Appendix C: instant replan on command change, τ only
3–9 frames; Orin deploy: "10 Hz or whenever commands change") — commands
never queue there. Our PC2 port (`pc2_kplanner_onnx.py`) deliberately
drops the instant-on-change trigger: CPU inference is 0.3–0.6 s and VR/pad
sticks stream continuously-varying commands, so change-triggered replans
would fire nonstop. A mid-walk command therefore waits until the ring
drains to the threshold before the next replan reads it (only IDLE→PLAYING
forces an immediate replan). The threshold is thus a refill-trigger level
and the ONLY mid-walk responsiveness lever: 32→48 fires each replan 16
model frames (~0.53 s at 30 fps) earlier, which is the measured 0.68 s →
0.22 s turn response above. Floor: threshold must cover inference time in
frames or the ring starves (16 starved at 0.53 s CPU inference, tape
20260719 — hence 32; with `--ort-gpu` on Jetson, ~tens-of-ms inference
would let it drop back toward 16). Deviation is now documented in the
runtime's "Known deviations" header.

## 2026-07-28 — pre-tuning baseline (for reference)

`KPLANNER_FIXED_FWD_MPS=0.5`, `KPLANNER_FIXED_TURN_RAD_S=1.0` (ritual),
`ARC_TURN 0.55` / `threshold 32` / `boost 1.0` (code defaults).
