# kplanner fixed-corpus from-scratch retrain — campaign record (2026-07-19)

Continuation and resolution of
[kplanner_investigation_handoff_20260719.md](kplanner_investigation_handoff_20260719.md)
(see its addendum for the corpus-corruption root cause). This doc records the
corpus rebuild, the retrain, the certification evidence, and the in-place-turn
deep dive that ended at an architecture-level discovery and a novel training fix.

## 1. Corpus rebuild (uniform pin-root dialect)

- All 33,206 feasible clips re-retargeted G1→X2 with the **pin-root** config —
  16 H100 shards across two Nebius nodes, 33 min (vs 3h15 for the original
  local run). Driver: `agibot-x2-references/soma-retargeter/batched_g1x2_driver.py`
  (now with a post-solve `sanity_check` that quarantines wrong-basin output).
- Kinematic gate (`gear_sonic/data_process/filter_kinematic_continuity.py`,
  seed-differential so genuine floor content survives): dropped 1,263 clips
  (3.8%) vs 2,493 (7.5%) under the old plain-config dialect — pin-root halves
  the manufactured corruption. **Final corpus: 31,943 clips**
  (`x2_ultra_bones_seed_g1_retarget_feasible_fixed.pkl`, built on node A;
  drop log `x2_ultra_bones_seed_g1_retarget_feasible_kinclean.dropped.txt`).
- Gate lesson: jump-counting undercounts — clips fully inside a wrong IK basin
  show zero jumps. Residency (root tilt / hip-pitch beyond limits) + a
  |yaw-rate| > 6 rad/s gate are both required.

## 2. Training (W&B `TRL_X2Ultra_Planner`, from scratch, uniform sampling)

| model | run | steps | wall | note |
|---|---|---|---|---|
| vqvae | `vqvae_fixed_scratch_300k` | 300k | 4h05 | frozen for all pose work |
| root | `root_fixed_scratch_500k` | 450k saved (300k certified) | — | small/fast model (391 MB ckpt) |
| pose | `pose_fixed_scratch_500k` | 320k at session stop | — | resumable via `last.ckpt` |

Fresh normalization stats over the fixed corpus (one canonical
`stats/motion/{mean,std}.npy` shared by all three models — pull with
`rsync -aL`, the pose/root copies are symlinks). Checkpoints archived on each
node's 1 TB `/mnt/ckpt` disk and `s3://kplanner-checkpoints/<hostname>/`.

## 3. Certification evidence (all probes reproducible from local bundles at `~/x2_cloud_checkpoints/fixed_scratch/`)

- **Root**: yaw tracking 1.03× (was ~3×); **conditioning-equivalence** — GT
  vqvae tokens decoded under root-predicted external_cond vs GT cond: foot
  lift 43 vs 44 mm (identical) → root output is in-distribution conditioning.
- **VQVAE ceiling test**: reference in-place turns encode→decode with 80–95%
  of foot lift, full leg articulation, exact yaw → the tokens carry stepping;
  nothing downstream is information-limited.
- **Behavioral map** (foot-lift metric — datum-free per-foot height range —
  belongs beside the yaw ratio in every future eval):

| maneuver | result |
|---|---|
| straight walk | healthy (pose@250k: 0.37–0.44 m in-body travel) |
| arc turn (vx 0.4 + yaw 0.4, walk seed) | **54/57 mm lift @ pose@250k** — above reference range |
| continue in-place turn (mid-turn seed) | 39 mm — works |
| initiate in-place turn (standing seed) | 6–8 mm, both directions, all checkpoints — broken |
| stop-then-turn from walk | broken (passes through standing) |

## 4. In-place-turn initiation: exhaustive elimination

Flat at 6–8 mm across: uniform 70k→250k; blended turn-priority fine-tune
(+35k); **pure overfit on 1,137 turn-initiation windows (~3,500 epochs — loss
0.62→0.31, fits under teacher forcing)**; argmax→gumbel sampling; 1→10
MaskGIT refinement steps. Conclusion: the model *fits* initiations and
*continues* turns but cannot *generate* them from scratch.

**Stock G1 has the same flaw**: standing-seed probe on stock G1 weights =
7/10 mm lift, 1.9 rad leg motion, 1.36× cold yaw — quantitatively identical
to X2. Standing-start in-place turns were never in this architecture family's
generated repertoire; G1's reputation came from walking-context behavior.
X2 is therefore at **true stock parity including the hidden flaw**.

Root cause of unlearnability: training masks at most
`masked_token_ratio`=0.8 of focused tokens — **≥20% ground-truth anchors are
visible in every training sample**, so the 100%-masked regime that generation
actually runs in is never trained; ambiguous-conditioning motions (idle vs
turn) collapse to the idle mode.

## 5. The fix (novel over stock): `fully_masked_sample_prob`

`motionbricks/motion_backbone/models/pose_model.py::_get_token_masks` now
supports `fully_masked_sample_prob` (hparams key next to `masked_token_ratio`;
default 0.0 = bit-identical legacy behavior): with that probability per
sample, every token is masked, directly training the from-scratch prior.

Status at session end: warm-start fine-tunes from pose@240k —
`pose_fullmask_ft` (0.25/priority 0.3) flat at +25k but training-loss
elevation (0.644 vs 0.591) confirms the regime engages;
**`pose_fullmask_hot` (0.5/priority 0.5 → ~25% fully-masked turn windows)
launched and stopped with the instances before its first checkpoint — verdict
pending on resume.** If hot stays flat by +25–50k, warm-starts cannot unlearn
the anchored prior → decide on a from-scratch pose run with `fully_masked_sample_prob`
enabled from step 0.

## 6. Deploy recipe available now (no training required)

- Never command yaw at vx=0: floor vx ≥ ~0.25 m/s while turning (pad-bridge
  one-liner). Demo choreography = walk-out arc + loop back (the
  `walk_circle_001` pattern already proven on hardware).
- Optional true pirouette: canned initiation primer (~0.5–1 s of a reference
  turn opening via the existing clip/anchor streaming) then planner takeover —
  continuation is proven at 39 mm.
- Best export chain today: **vqvae@300k + root@300k + pose@250k+** via
  `motionbricks/scripts/export_x2_planner_onnx.py`.

## 7. Resume notes

- Nodes were stopped by operator; IPs change on restart. Run scripts in each
  home dir (`run_kplanner_fixed_A_vqvae_pose.sh`, `run_kplanner_B_fullmask_ft.sh`
  etc.); earlier experiment arms preserved in
  `out_fixed_scratch/motionbricks_pose_x2/version_1/checkpoints_{turnft_v2,turnval,fullmask25}/`.
- Feature-cache gotcha: cache subdir is keyed to the exact `--pkl` list; union
  caches must symlink with **absolute** targets.
- Retarget gotcha: pre-warm the newton asset cache once per node before
  launching parallel shards (thundering-herd git clone crash).
- **Rename gotcha (resume-critical)**: the flag was renamed
  `full_mask_prob` → `fully_masked_sample_prob` after the nodes were stopped.
  Node B's `out_fixed_scratch/motionbricks_pose_x2/version_1/hparams.yaml`
  still carries the OLD key, which the renamed code silently ignores
  (defaults to 0.0). On resume: ship the current `pose_model.py` AND rename
  the key in B's hparams before relaunching the fullmask runs.
- Flag verified end-to-end post-rename: real `_get_token_masks` with
  DictConfig args produces 0% / 26% / 53% fully-masked samples at
  prob 0 / 0.25 / 0.5, legacy max mask fraction 0.75 unchanged, and the
  hparams key resolves into `model.args`.
