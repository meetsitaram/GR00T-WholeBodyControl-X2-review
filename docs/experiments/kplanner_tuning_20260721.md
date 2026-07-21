# kplanner tuning day — turn rates, templates, hover verdict, fwd 0.5 on HW

Follow-on to `kplanner_gpu_unlock_20260720.md`. Operator verdict at day end,
robot session `robot_runs/20260721_075434_p500k_fwd05_stability`:
**"amazing walk stability"** at fwd 0.5 m/s, dual-rate turns, p500k graphs,
GPU inference (84 ms median replan), yaw-resync active.

## Shipped (validated)

1. **Turn-rate discovery** (`kplanner_turnrate_sweep.py`): the root head
   scales to 1.3 rad/s with NO saturation, and **the idle attractor
   vanishes at >= 0.8 rad/s** — strong conditioning makes the pose head
   generate real turn stepping. The standing-collapse we spent days on is a
   weak-conditioning phenomenon. Right turns systematically ~15-20% weaker
   than left (corpus imbalance).
2. **Dual-rate turn setpoints** (operator-driven): standing turns
   `KPLANNER_FIXED_TURN_RAD_S` (ritual: 1.0), walking arcs
   `KPLANNER_FIXED_ARC_TURN_RAD_S` (default 0.55) — full-rate arcs turned
   more than they walked. Sim-validated (deploy-tuned harness: 360s at
   46-53 deg/s, no falls), then HW.
3. **Forward setpoint env-tunable + launcher clobber fix**:
   `sim_onnx_planner.sh` hard-exported `KPLANNER_FIXED_FWD_MPS=0.3` over
   any caller value ("no matter what fwd mps I set, it walks the same").
   Now `:-0.3` default. Model tracks commanded speed ~1:1 up to 0.8 m/s;
   **skate explodes at 1.0** (0.57-0.78 m/s) — treat 0.8 as ceiling.
   Ritual runs 0.5.
4. **Sim GPU by default**: workstation env_isaaclab got a cu12
   onnxruntime-gpu (1.22.0 + `ort.preload_dlls()`, now called in the
   daemon's --ort-gpu path; ORT 1.27 needs CUDA 13, don't upgrade);
   `run_x2_quest3_planner_stack.sh` passes `--ort-gpu` unless
   `KPLANNER_ORT_GPU=0`. Matching provider = matching sampling (CPU/GPU
   draws differ; a CPU-sim reference walked ~30% slower than the robot's
   GPU reference at the same command).
5. **p500k velocity graph synced to PC2** (template was already p500k since
   2026-07-20; the stale velocity graph was what tripped the sim identity
   gate MISMATCH). Robot == sim == local, md5-verified.
6. **`KPLANNER_ARC_FWD_BOOST`** (default 1.0 = inert): opt-in over-command
   of forward during arcs. Kept dormant by operator decision — tracked
   arcs under-translate because SONIC's obs carries no reference root
   translation (forward is implied by gait joints; heading error is
   explicit), but hard-coding compensation would poison future models.

## Template A/B/C (mode clips) — negative results, kept for the record

Question: replan target keyframes sample a uniform-random 4-frame window of
the mode clip; wide gait-clip windows put a MID-SWING anchor on 34%
(slow_walk) / 54% (walk) of draws -> improvised landings ("bumpy steps").
Three graphs, identical p500k weights (initializer-hash-verified: 410/414
tensors byte-equal, only `core._clip_library.*` differ):

- **control** (g1teleop wide windows) — baseline.
- **g1style** (stock-G1 recipe: ONE neutral-idle keyframe for all modes) —
  **2x the foot-skate of control on every walking cell** (0.33-0.40 vs
  0.14-0.19): the idle anchor fights translation. Works for G1's backbone,
  not ours. RETIRED.
- **stance** (`--modes g1teleop_stance`, windows cut to longest
  fully-grounded runs) — equal to control on walking metrics with every
  anchor grounded; operator: less bumpy. BUT standing turns regressed in
  live context (wire heading ~0.6 deg/s vs 50+ expected) — UNRESOLVED, so
  control stays deployed. `build_x2_planner_clips.py` gained the
  `g1teleop_stance` table; `kplanner_template_sweep.py` scores
  graph x command x speed.

Packaging note: clip frames are baked into the exported ONNX as constants
(torch-free PC2 runtime). Template iteration = rebuild library + re-export.
Design improvement for later: make the 4-frame target window a GRAPH INPUT
sampled daemon-side — clip changes then need no export.

## The right-foot "tap-tap" — investigated to ground, deploy fixes DISCARDED

Wire FK: right arcs showed inside-foot hovers (600 ms - 2.7 s airborne) that
the tracker converts into rapid touch-retouch taps. Findings:

- **Per-FOOT bias, not per-direction** (right foot floats more in all four
  graph x direction cells) — killed the sagittal-mirror idea (transform was
  built and validated to 0.00 mm FK error; parked for data augmentation).
- **No command pocket**: hover persists across all rate x speed combos.
- **Context-contagious**: once the 4-frame context holds an airborne foot,
  ~75% of fresh-seed generations EXTEND the hover (seed diversity is
  irrelevant; context dominates). Entering arcs from straight walking is
  the clean regime; arcs from standing/turning start hovers.
- **Hover gate v1 (per-chunk FK budget) failed**: hovers continue across
  seams, each chunk individually legal. **v2 (cross-seam, buffer-tail
  prepended) made it WORSE**: re-rolls can't escape the context and the
  3x inference delay lengthened the floats (933-1258 ms means, max 3.9 s).
  Both versions REMOVED per pre-agreed discard contract.
- **Verdict: model-generation defect; the fix is training data** (right-
  foot-swing / right-turn enrichment) — now the top item for the next
  training pass, with quantified symptoms at three levels (turn rate,
  stillness, hover). Interim: prefer left arcs in choreography; enter arcs
  from a walk.

## Ritual/config state after today (robot, verified)

`--ort-gpu --playing-yaw-resync-dps 10`, env `KPLANNER_FIXED_TURN_RAD_S=1.0
KPLANNER_FIXED_FWD_MPS=0.5`; daemon md5 c2b98f4f...; planner graphs
template+velocity = local p500k (backups kept beside them).
