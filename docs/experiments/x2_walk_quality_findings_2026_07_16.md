# X2 Walk-Quality Findings — G1-teleop corpus, 3-way eval (2026-07-16)

Overnight fine-tune round on the 61 newly-recorded G1-teleop clips (slow_walk /
turns / back / walk / run, 10 s each). SONIC tracker fine-tuned from
`dance_v3_3k` on the v4 corpus (243 targets); kplanner chain (vqvae→root→pose)
fine-tuned from the g1ret models with the windowed clips prioritized @30%.

Artifacts: `docs/experiments/x2_walk_quality_3way_20260716/`
(WALK_QUALITY.md full tables, COMPARISON.md feasibility, per-clip JSON, scorer,
MuJoCo pass tally). Eval ran locally (RTX 5090, IsaacLab im_eval, 61 clips,
G1-stock vs base-3k vs FT-iter1144).

## 1. Feasibility saturates on walks — walk quality is the discriminator

All three policies: 60/61 survival (only `run_004`, the fastest clip, fails on
X2). Every walking category is 100% feasible for every policy, so fall-rate
cannot distinguish base from fine-tuned. Root/pose *tracking* can:

| metric (58-61 clips) | G1-stock | base-3k | FT-1144 |
|---|---|---|---|
| root-rel MPJPE (mm) | **13.5** | 27.7 | 26.8 |
| — upper body (mm) | 9.5 | 28.6 | 28.2 |
| net-displacement err (m) | 0.226 | 0.241 | **0.204** |
| mean root pos err (m) | 0.127 | 0.143 | **0.118** |
| heading err (deg) | 2.3 | 3.3 | 2.6 |

**FT-1144 beats base-3k on every category** (pose + root) after only ~1.1k
fine-tune iterations. Heading is near G1 parity. Remaining gaps: upper-body
pose (~3×, see §2) and runs (20% under-run, §4).

## 2. THE ELBOW OFFSET ISSUE (root cause of the "upper-body gap")

Per-body breakdown shows the gap is concentrated in the **elbow: ~54 mm error
vs G1's 10 mm** — the single worst body on the robot, worse than the
balance-loaded ankles.

Mechanism analysis (amplitude ratio / cross-correlation lag / static-dynamic
decomposition of the elbow trajectory):

- swing **rhythm tracks well**: correlation 0.77–0.99, lag ≈ 0–40 ms,
  amplitude 1.0–1.5× (it *over*-swings — so NOT torque/velocity saturation,
  despite elbow + ankle_roll sharing the 24 N·m actuator family; that
  correlation is a red herring — walking arm-swing needs ≪ 24 N·m);
- the error is a **~50 mm CONSTANT offset** (dynamic residual only 20 mm;
  G1's static offset is 6 mm). The arm is carried ~5 cm off the reference
  pose — visible in the viewer as hands tucked in front of the torso.

**Likely cause:** the arm-tracking reward weights were deliberately moderated
in the arm-dynamics/dance config lineage (strong arm rewards destabilized
training). With a weak pull toward the reference arm pose, gravity + the
default-pose regularizer settle the elbow a few cm off-target. Diagnostic
tell: fine-tuning improves knees (38.8→33.5 mm) but leaves the elbow untouched
(53.7→53.9 mm) — no reward gradient, so more data/steps cannot fix it.

**Fix candidates (next round, deliberately not tonight):**
1. raise arm-joint tracking-reward weight (carefully — the moderation exists
   for a reason; A/B against stability);
2. gravity feedforward on arm joints in the PD (removes the sag without
   touching rewards);
3. check whether the retargeted reference elbow poses sit in an awkward region
   of X2's range.

Impact: cosmetic for locomotion (balance/footwork unaffected) but it is the
main term in the 28 mm vs 9.5 mm upper-body fidelity gap to G1.

## 3. 0.2-speed dead-band (the "robot resists moving" clips)

Displacement shortfall vs reference (FT-1144): `slow_walk_0.2_*` 22–38 %,
**`slow_walk_back_0.2_*` 69–78 %** (near-immobile), everything ≥0.4 speed
≤12 %. Two stacked causes:
1. the known low-speed reward dead-band (see slow_manip_focus work);
2. **the G1 source executions themselves understep at 0.2** (stride
   0.20–0.65) — the weakness is partially baked into the retarget targets.

Fine-tune already helps (back_0.3: 58→31 %, 0.2_002: 98→22 %) but cannot fully
overcome weak references. Corpus fix candidates: re-record 0.2 clips after a
dead-band fix, or floor commands at ~0.3.

## 4. Runs
Progress on `run_004` (fastest clip, ~0.74 m/s): base-3k dies at 6 % progress,
FT-iter1144 at 11 % (fell 0.86 s into the MuJoCo pass), **FT-iter2082 completes
the full 10 s pass in MuJoCo (61/61 clips, zero falls)**. G1-stock reaches 66 %
in IsaacLab. Run category still under-runs ~20 % on path length — the next
milestones should keep closing this.

## 5. Eval mechanics gotchas (cost hours — do not rediscover)
- `IM_EVAL_DUMP_TRAJ=<dir>` enables per-clip trajectory dumps in the X2
  im_eval; **the dump field names are swapped** (`pred_pos` = reference,
  `gt_pos` = robot). Verified by net-travel match to the motion PKLs.
- G1 CSV recorder (`g1_onnx_policy_shim`) needs `num_envs ≥ #clips` or it
  device-asserts.
- Marginal run-clip feasibility is **env-batch dependent** (4 vs 32 vs 64 envs
  differ); judge run clips with small dedicated batches.
- kplanner evals: FT-vqvae + old pose = mismatched tokenizer pair → trajectory
  results are noise. Only evaluate matched vq⟷pose pairs.
- The SONIC runner maintains a rolling `last.pt` only (occasional
  `model_step_*.pt`, unreliable cadence) — snapshot `last.pt` with an
  iter-stamp when milestones matter.
