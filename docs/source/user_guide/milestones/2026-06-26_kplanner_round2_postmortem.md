# 2026-06-26 — kplanner Round 2 postmortem (and Round 2.1 / 2.2 fine-tunes)

> **Status:** All training is **off the cloud**, all final checkpoints
> pulled to local and SHA-verified, Nebius 8× H100 instance is
> **stopped** (boot disk preserved). The disappointment is the
> deliverable.

> **Surface:** kinematic planner (`x2_kplanner.py`) only.
> **Not** SONIC actor, **not** VR teleop, **not** real-robot
> deploy. The downstream stack still works; this milestone is about
> *what the planner produces* being mixed-to-bad despite a large
> training spend.

## TL;DR

We spent **~$600 of 8× H100 SXM time and ~25.5 hr of wall-clock**
retraining the X2 kinematic planner triple (VQVAE + Pose + Root)
from scratch on a 10× larger corpus with 8× larger effective
batches than Round 1. **Training losses look healthy** on paper.
**Closed-loop sim and on-robot behavior are still mixed-to-bad.**
Two fine-tunes (FT1, FT2) followed and traded one regression for
another without fixing the underlying behavior. The long-form
analysis lives at
[`kplanner_training_runs.md`](../kplanner_training_runs.md);
this milestone is the operator-facing wrap-up.

## What got measured

| Test                                                    | R1 (legacy) | R2 (300K)    | R2.1 / FT1 (315K) |
|---------------------------------------------------------|-------------|--------------|-------------------|
| Open-loop replay, fwd dist on `Loop_Forward_Walk_001__A018` (30 s) | not measured | 9.29 m       | 7.62 m            |
| Yaw drift over 30 s of straight-fwd intent              | not measured | **+46.5°**   | +34.0°            |
| Lateral drift over 30 s                                 | not measured | +0.28 m      | **+0.65 m**       |
| Joint-pose RMS error                                    | not measured | 0.20         | 0.18              |
| Closed-loop sim (`run_x2_pkl_planner_stack.sh`)         | "barely works" | "noticeable but not stable" | similar           |
| Quest3 + real X2 with sticks at 0.4–0.8 m/s             | "stiff, narrow envelope" | "side L/R aggressive, fwd intermittent, occasional back-walk-on-fwd-intent" | not retested      |

For reference, a clean forward-walk over 30 s at 0.8 m/s should
travel ≈ 24 m, not 9 m, and accumulate ≈ 0° of yaw drift, not 46°.

## Four root causes

(Detail and supporting numbers in [`kplanner_training_runs.md` → Round 2
postmortem](../kplanner_training_runs.md#round-2-postmortem-why-25-hours-of-h100s-still-did-not-give-a-smooth-vr-walk).)

1. **Compute deficit vs the G1 reference.** X2 R2 trained Root for
   **300K steps × batch 256 ≈ 77M samples** on a **~35K-clip corpus**.
   The G1 MotionBricks reference trained for **2M steps × batch 2048
   ≈ 4.1B samples** on a **~350K-clip corpus**. That's **53× fewer
   samples** end-to-end. G1 spent **90 %** of its training post-
   keyframe-curriculum warmup; X2 R2 spent **33 %**. We are nowhere
   near the regime where the recipe is *known* to produce a smooth
   planner.
2. **Corpus narrowing.** The Round-2 `chain_matched_v2` PKL has
   49,790 clips at the top but **only 499 (3.4 %) are pure
   forward-walk, all from a single base motion**. ~33 % are
   manipulation false-positives (regex hits "leg" / "body" body-part
   words in arm-and-object motions). Turns dominate the genuine
   locomotion slice at 46 %. The 0.5× halfspeed merge introduced
   **velocity bimodality** (clusters at 0.5× and 1.0× of natural
   walking speed, nothing in between, nothing at very-slow VR-stick
   regime). The model learned "forward axis = turning + manipulation",
   and the inference-time forward intent drifts accordingly.
3. **Loss imbalance.** ~98 % of the Root training-loss magnitude was
   consumed by the **`num_token` classification** auxiliary task
   (which token-count the keyframe curriculum is asking for). The
   actual continuous root-trajectory reconstruction losses
   (global-root recon ~0.002, local-root recon ~0.025) are orders of
   magnitude smaller. With G1's 10× post-warmup step budget this
   auxiliary task saturates and the trajectory losses start moving;
   with ours, the curriculum is still warming up when we stop.
4. **Inference-time mode mismatch.** The Root backbone is an
   **in-betweener** at training time (it sees start + target body
   pose + target root). At inference time
   `motionbricks/motion_backbone/inference/neural_planner.py`
   explicitly **masks off the target body pose** (`has_local_poses[:, -NUM_FT:] = False`)
   and asks the model to invent the target pose from velocity intent
   alone — an out-of-distribution use of the model.
   `motionbricks/scripts/probe_root_constraint_modes.py` already
   exists and previously reported a ~3 % tracking-slope improvement
   when the target body pose is supplied; the probe has **not yet
   been re-run on the R2 / FT1 / FT2 checkpoints**, so we don't
   actually know whether pose templates would help more on our
   specific failure mode.

## What changed in code (this commit)

| File                                                                  | Change                                                                                                                |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `motionbricks/scripts/train_root_x2.py`                               | New `--init-from <ckpt>` flag for weights-only init (orthogonal to `--resume`). New `--peak-lr / --warmup-steps / --final-lr` overrides for hparams.yaml LR knobs. New `--ckpt-subdir <name>` to write FT checkpoints to a separate dir. |
| `motionbricks/motionbricks/data/x2_loco_filters.py`                   | Replaced Python `\b` word-boundaries (which treat `_` as a word char and fail on snake_case keys) with letter-only boundaries `(?<![A-Za-z])X(?![A-Za-z])` so `walk` matches `Loop_Forward_Walk_001__A018` but `walker` still doesn't. |
| `motionbricks/motionbricks/data/x2_bones_seed_dataset.py`             | When cache HITs, apply `include_patterns` / `exclude_patterns` / `max_clips` to the manifest keys instead of silently returning the full manifest. Fixes "I asked for `--filter loco` but got 49,790 clips anyway".                       |
| `motionbricks/motionbricks/motion_backbone/inference/load_x2_planner.py` | Default checkpoint paths bumped to Round-2 finals (VQVAE 500K, Pose 500K, Root 300K) and the `motionbricks_pose_x2_v2/` symlink-hop retired.                                                                                  |
| `gear_sonic/scripts/x2_kplanner.py`                                   | CLI default ckpts bumped to match (VQVAE 500K, Pose 500K, Root 300K).                                                  |
| `gear_sonic/scripts/run_x2_pkl_planner_stack.sh`                      | Preflight default ckpt paths bumped to Round-2 finals.                                                                |
| `gear_sonic/scripts/run_x2_quest3_planner_stack.sh`                   | Preflight default ckpt paths bumped to Round-2 finals.                                                                |
| `motionbricks/scripts/cloud/run_root_finetune_v2.sh`                  | New cloud launcher for FT2 (weights-only init from FT1, fresh optimizer, peak LR 1e-5, warmup 1K, final 5e-7, 30K steps, batch 32 × 8 GPU). Documented one-off in `cloud/` directory convention. |
| `motionbricks/scripts/cloud/eval_root_finetune.sh`                    | New cloud side-eval polling loop — watches `checkpoints_ft2/` for new `model-step=*.ckpt` files and runs `replay_pkl_through_kplanner.py` on each, appending to `~/root_finetune_v2_eval.csv`. |
| `docs/source/user_guide/kplanner_training_runs.md`                    | New Round 2.1 / 2.2 sections + long-form postmortem with G1-vs-X2 compute table, loss-imbalance breakdown, and "what we would try next" priority list. |

The 25 FT2 intermediate ckpts (steps 6K-30K) stay on the **stopped**
Nebius boot disk — not committed, not pulled bulk. Only the FT2 final
(`checkpoints_ft2/model-step=0030000.ckpt`, 391 MB, sha256 `c0e84bff…`)
is local. The Round 2 finals (VQVAE 500K, Pose 500K, Root 300K) were
already pulled in earlier sessions and SHA-verified against cloud the
night the instance was stopped.

## What we would try next (if we paid this $600 again)

In priority order — cheapest informative experiment first:

1. **Re-run `probe_root_constraint_modes.py`** on R2 / FT1 / FT2 with
   the actual failure clip. 10 min, zero cloud cost. Tells us
   whether the inference-time mode mismatch is a small effect (skip)
   or a large effect (plumb 7 X2 canonical pose templates into
   `_predict_with_velocity` behind a `--use-pose-templates` flag).
2. **Train a small specialist Root** on the existing X2 PKL corpus
   only, at fixed hip height + fixed body-pose-template family, for
   ~50K steps on 1× H100. **~$10.** Not aiming to outperform the
   generalist Root — aiming for a model that was *trained* on the
   inference distribution we deploy in.
3. **Only if 1 + 2 conclusively fail:** another 8× H100 run with a
   smarter corpus (heavy upweight on pure straight-line walks,
   continuous speed jitter instead of bimodal halfspeed, no
   manipulation false-positives) and a longer schedule (~600K Root
   steps to spend more budget post-keyframe-warmup). **~$1000.**

Path 1 is essentially free. Path 2 is ~1 % of Path 3's cost.
Path 3 should only be attempted if 1 + 2 give a clear signal
that we need a fundamentally better *generalist* Root model.

## Lessons captured

- Training loss going down ≠ the model will work in deploy. Always
  pair training with an **open-loop replay metric** on a
  representative clip from the deploy envelope (we now do this via
  `motionbricks/scripts/cloud/eval_root_finetune.sh` for fine-tunes,
  but it should run during the main training too).
- Cosine schedules with `final_lr ≈ 2e-6` have a "dead tail". Use
  `--init-from` (weights-only) + fresh `--peak-lr/--warmup-steps/--final-lr`
  if you need to continue past it — full-state `--resume` will waste
  GPU hours doing no gradient work.
- "Bigger corpus" ≠ "better corpus". Audit any new corpus by
  gait class **before** spending H100 hours on it. A 2.8× corpus
  growth with 0× growth in pure-forward-walk signal density is the
  worst kind of cost growth.
- Auxiliary classification losses (e.g. `num_token`) can dominate
  the gradient budget and starve the metric you actually care about.
  Decompose loss panels in W&B *during* training, not after.
- 8× H100 is the wrong instance size for fine-tunes / experiments.
  Use 1× H100 (or local 5090) for FT1 / FT2 / specialist runs;
  reserve multi-GPU for from-scratch ≥ 500K-step training.

## What's still open

- A/B FT2-30K vs R2-300K vs R2.1-315K on the **single** forward-walk
  PKL (`Loop_Forward_Walk_001__A018`) — the only clip where our
  trajectory summarizer gives clean numbers. Multi-clip eval on the
  18K-key locowalk PKL is bimodal-noisy and useless for picking a
  step.
- The `probe_root_constraint_modes.py` re-run on R2 / FT1 / FT2.
- The decision on Path 2 (small specialist Root) — gated on the
  probe re-run.

The Nebius node (`computeinstance-e00cn8h67tdq00t2ys`,
`195.242.31.46`) is **stopped, not deleted**. Boot disk preserved
so the 25 FT2 intermediate ckpts and the W&B
`root_x2_ft2_lr1e5_from315k` run state remain reachable if we want
to do a multi-step sweep without re-pulling 9.4 GB of ckpts. Restart
the instance via Nebius console if needed.
