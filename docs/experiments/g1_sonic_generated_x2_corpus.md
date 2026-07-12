# Experiment: G1-SONIC-Generated (Feasibility-Filtered) X2 Training Corpus

**Status:** pilot mechanism VALIDATED (2026-07-11) · **Author idea:** user · **Drafted:** 2026-07-11
**Goal:** cut X2 SONIC training compute by training only on *physically-achievable,
dynamically-consistent* motions, using the stock **G1 tracking policy (SONIC) as a
feasibility oracle**, then quantify how much compute we'd save.

> **SCOPE (user, 2026-07-11): this experiment is G1-ONLY.** Phase 1 (this doc) = run the
> stock G1 policy over the bones-seed G1 corpus, produce the **feasibility filter (Metric 1)
> + deviation (Metric 2) + estimated compute savings**, and dump the feasible executed G1
> motions. **Phase 2 (LATER, out of scope here): retarget the feasible G1 motions → X2 and
> train X2 SONIC** — deferred until Phase 1 is fully complete. So steps (e)–(g) below are
> Phase 2; the *experiment* ends at the G1 metrics + the dumped feasible G1 corpus.
>
> **MECHANISM DRY-RUN (2026-07-11) — X2 assets, NOT part of this experiment:** to prove the
> vectorized `im_eval` sweep works, we ran it in **34 parallel envs, ~1.5 min, no new code**
> (on convenient X2 stand-in assets) → `metrics_eval.json` emitted per-motion
> `progress`/`terminated`/`failed_keys` = **Metric 1 for free**. This only validates the
> *mechanism + metric keys*; **the experiment redoes it as G1 policy on G1 references (no X2).**
> Two gaps to close for the real G1 run: (i) enable **Metric 2 (MPJPE deviation)** output in
> `im_eval` (needs a `log_keys`/config flag or small extension); (ii) **dump executed
> trajectories** (the callback already has `all_body_pos_pred`).
> Also confirmed: the **stock G1 policy is framework-native** (encoder/decoder ONNX,
> `policy/release/{model_encoder,model_decoder}.onnx`, obs config in this framework's
> vocabulary — `token_state`, `his_body_joint_positions_*`, `motion_joint_positions_*`,
> `vr_3point_local_target`), so it plugs into `im_eval` with the G1 robot config — **no obs
> adapter needed** (STEP 0 path (a) is clean; disregard the batched-MuJoCo fallback unless
> obs surprises appear).

### Pilot findings — HOW SUCCESS IS MEASURED + how to visualize (2026-07-11)
**Success metric (= Metric 1, feasibility) — confirmed by reading the eval loop + pilot output:**
- **SUCCESS ⇔ the episode was NOT `terminated`** — i.e. the robot completed the whole clip
  without hitting any env termination condition (fall / tracking-divergence / anchor-orientation
  / foot-position / etc.). **`progress`** = fraction of the clip reached before termination
  (`1.0` = full clip).
- Pilot proof (34 X2 teleop clips, fine-tuned model): 33/34 SUCCESS (all `terminated=False`,
  `progress=1.0`); the single FAIL `slow_walk_mid_keyboard_003` (idx 15) had
  **`terminated=True, progress=0.65`** — it fell ~65% through. `success_rate=0.9706`,
  `progress_rate=0.9897`.
- **For the G1 sweep this is exactly the filter:** `terminated=False` ⇒ **feasible** (keep +
  record the executed motion); `terminated=True` ⇒ **infeasible** (drop). `progress` gives a
  graded "how far did it get" signal for the FAIL bucket.
- **`metrics_eval.json` keys** (written to `eval_output_dir`): `eval/success/success_rate`,
  `eval/success/progress_rate`, `eval/all_metrics_dict.{terminated, progress, motion_keys,
  sampling_prob}`, `failed_keys`, `failed_idxes`.
- ⚠️ **The `terminated` flag is only as good as the env's termination conditions.** Inspect /
  calibrate them for the G1 env (height, `anchor_ori_full`, `foot_pos_xyz`, tracking-error
  thresholds) so "infeasible" matches intent — a too-lenient termination keeps bad motions;
  too-strict drops learnable ones. `terminated` is the oracle; know what trips it.
- **Metric 2 (MPJPE deviation) is computed internally** (`im_eval_callback.py` tracks
  `self.mpjpe` = `env.dif_global_body_pos.norm()*1000` mm) **but was NOT saved in the pilot
  JSON** — enable it via `log_keys` / a small callback tweak so per-motion MPJPE lands in
  `metrics_eval.json` (this is the "tracked-but-poor" signal for survivors).

**Visualizing generated (executed) vs input (reference) in a kinematic MuJoCo viewer:**
`im_eval` does **not** dump executed trajectories today (only metrics), so there is nothing to
view yet — this is the **same trajectory-dump gap** as Metric 2 / the retarget data. To enable:
1. Extend `im_eval_callback.py` to **save each env's executed root+dof trajectory** to a motion
   PKL (the callback already has the predicted body state via `self.pred_pos` /
   `env.dif_global_body_pos`; add the root+joint qpos capture + a `joblib.dump`).
2. Then view side-by-side: **`gear_sonic/scripts/play_x2_motion_mujoco.py --motion <executed.pkl>`**
   in one window and `--motion <input.pkl>` in another (pure `mj_forward` kinematic, no policy),
   or the soma-retargeter **`scripts/view_g1_x2.py`** for a merged multi-robot view. For G1 use
   the G1-MJCF kinematic viewer equivalent.
3. One-at-a-time spot check (no dump needed, does NOT scale): play a clip through the deploy
   SONIC stack + `record_motion_to_pkl.py`, then view the recorded PKL beside the input.
> The executed-trajectory dump is the single highest-value next code change: it unlocks
> Metric 2, the visualization, AND the Phase-2 retarget data all at once.

---

## 1. Hypothesis

Training X2 SONIC on **raw human-MoCap → X2 kinematic retargets** wastes large amounts
of compute: many retargeted human motions are **physically unachievable** (self-collision,
over-extension, dynamically unstable poses, sub-threshold micro-motions). The RL tracker
burns gradient fighting to reach references it can never hit, which slows convergence and
injects instability (we saw this: `anchor_ori_full` terminations, and the policy "giving
up" — standing instead of stepping — on infeasible slow gaits).

**Proposal:** pre-filter the corpus to feasible motions by:
1. Running each **G1 reference** through the **stock G1 SONIC tracking policy in sim**.
2. **Dropping** motions the G1 robot can't execute (falls / diverges).
3. **Recording the robot's *executed* trajectory** (not the human reference) for the ones
   it can.
4. **Retargeting the recorded G1 motion → X2** (proven G1→X2 pipeline, pin-root config).
5. Training X2 SONIC on that clean, feasible, dynamically-consistent corpus.

**Expected result:** materially faster convergence + higher final tracking quality + lower
GPU-hours, because ~100% of gradient goes to trackable targets.

### Why this is principled, not just a hunch
- The repo **already** does a crude version: the `v3_recovery` config drops clips with
  `peak_pelvis_tilt > 20°` / `peak_pelvis_angvel > 3 rad/s` because those human poses
  generate nearly all the training failures. **This experiment replaces that hand-tuned
  *kinematic* filter with a *dynamic* one** (a policy actually trying to do the motion).
- Two elegant properties:
  1. **The rollout is the feasibility oracle** — no hand-tuned thresholds; the policy's
     success/failure auto-curates the corpus.
  2. **The recorded motion is dynamically consistent** — real contact timing, weight
     shifts, momentum — strictly better reference data than a kinematic retarget.
- **G1 as the teacher (not filtering the weaker current X2):** G1 SONIC is the *stronger*
  policy, so its feasible set is broader and cleaner. Use the good policy to define the
  achievable manifold, transfer via the proven G1→X2 retarget. (Teacher→student with a
  cross-embodiment hop.)

---

## 2. Deliverables / Metrics (what the experiment must produce)

### Metric 1 — Feasibility filter (coverage): "how many hard motions got dropped?"
For **every** G1 reference clip, run it through G1 SONIC and classify:
- **SUCCESS** — completed the clip, upright, tracking within tolerance.
- **FAIL-FALL** — robot collapsed (pelvis height < ~0.4 m, or sustained |tilt| > ~45°).
- **FAIL-DIVERGE** — stayed up but tracking error blew past a threshold (lost the motion).
- **FAIL-EARLY-TERM** — episode terminated before clip end for any reason.

**Output:** `feasibility_report.csv`, one row per clip:
`clip, category, is_mirror, class, frames_survived, frames_total, fall_frame,
peak_pelvis_tilt_deg, peak_root_drift_m`.
**Aggregate:** per-category and overall **% failed** + failure-class breakdown →
*this is the "how many difficult motions were filtered out" number.*

### Metric 2 — Tracking deviation (quality): "how badly did the survivors struggle?"
For **SUCCESS** clips (didn't fall), measure how far the **executed** motion deviates from
the **reference** (suspicion: most won't collapse, but many track poorly due to physical
challenge — e.g. understepping):
- `joint_pos_mae_deg` — mean per-joint abs deviation (executed vs reference DOF).
- `root_xy_drift_m` — executed root-XY path vs reference root-XY path (mean + final).
- `root_z_dev_m` — height deviation.
- `body_pos_mae_m` — FK body-position deviation (all 14/N bodies).
- `stride_ratio` — executed foot displacement / reference foot displacement (**catches
  understepping** — the failure mode we saw on slow gaits).

**Output:** distributions (percentiles/histogram) per category, and a flag for
**"tracked-but-poor"** clips (e.g. top-quartile deviation) — the survived-but-struggled set.

### Metric 3 (headline) — Estimated compute savings
Combine the above into a rough estimate for the *next* full training run:
- `F` = fraction of clips filtered out (Metric 1) → that fraction of gradient was pure waste.
- Deviation stats (Metric 2) → the kept set is dynamically clean; quantify vs the raw
  retarget's implied difficulty.
- **Convergence A/B (from the pilot, §4):** iters-to-reach-a-target-metric on the
  G1-recorded corpus vs the raw-retarget corpus, same category.
- **Headline:** *"~X% of motions infeasible (filtered) + ~Y% faster convergence on the
  feasible set → ~Z% GPU-hour reduction on the next run."*

---

## 3. Pipeline (step by step)

```
STEP 0 (prerequisite): secure a G1 SONIC policy that runs in the IsaacLab framework
bones-seed G1 reference (already a motion-lib PKL, or convert G1 CSV → PKL)
  └─(a) VECTORIZED im_eval sweep: G1 SONIC tracks every motion across N parallel envs
      ├─(b) per-motion `progress` (clip-completion) ................ Metric 1 (feasibility)
      ├─(c) per-motion predicted-vs-GT body/joint error ............ Metric 2 (deviation)
      └─(d) dump each env's EXECUTED trajectory (root+dof) .......... feasible motion data
          └─(e) [SUCCESS only] retarget executed G1 motion → X2 (pin-root config)
              └─(f) build X2 training PKL (feasible corpus)
                  └─(g) train X2 SONIC → A/B vs raw-retarget baseline
```

### ⚠️ SCALE: do NOT run one MuJoCo sim per motion. Use the vectorized eval framework.
39K clips × several seconds each is impossible one-at-a-time. The **same IsaacLab
infrastructure that runs SONIC *training* with thousands of parallel envs** is the vehicle —
just in **eval mode** (frozen policy, no learning). The `im_eval` callback already does the
heavy lifting; this is minutes-to-hours on one GPU, not days.

**PRIMARY mechanism — vectorized `im_eval` sweep (`gear_sonic/trl/callbacks/im_eval_callback.py`):**
- Runs the policy over **all motions across N parallel envs, multi-GPU** ("Supports multigpu").
- Already computes **per-motion `progress`** (fraction of the clip completed before
  termination → **Metric 1**; `progress == 1.0` ⇒ SUCCESS, `< 1.0` ⇒ FAIL) and
  **predicted-vs-GT body tracking error** (→ **Metric 2**). Writes `metrics_eval.json`
  (per-motion) via `save_metrics_eval()` into `eval_output_dir`.
- Already stacks the executed body states (`all_body_pos_pred` / `self.pred_pos`), so
  **step (d) — dumping executed trajectories (root_trans/root_rot/dof per frame) for
  retargeting — is a SMALL EXTENSION to the callback, not a new system.**
- Invoked via `eval_agent_trl.py` with `++run_eval_loop=true ++eval_callbacks=im_eval
  ++headless=True +checkpoint=<G1 SONIC .pt> ++num_envs=<large>
  ++eval_output_dir=<dir>` (see `dump_isaaclab_step0.py` for the invocation pattern).
- Ensure **systematic full coverage** (every motion evaluated once), not adaptive/random
  sampling — check the eval path uses deterministic per-motion assignment
  (`motion_lib_base.py` has `uniform_sampling_rate` + a `sample_motions`/coverage path).

### STEP 0 — the gating decision: run the STOCK UNITREE G1 ONNX at scale (DECIDED)
**Oracle = the stock Unitree G1 ONNX** (user decision, 2026-07-11) — the same policy the G1
deploy loads. There is **no G1 SONIC framework `.pt`**, and we will NOT train one. The
consequence: the stock ONNX's *native* environment is the **deploy's MuJoCo with Unitree's
obs layout**, not the IsaacLab framework. So the vectorized sweep needs the stock ONNX
driven over many parallel envs. Two ways to do it — **this is the main engineering fork:**

- **(a) Wrap the stock ONNX inside the IsaacLab framework** (reuse `im_eval`'s vectorization).
  Feed the IsaacLab G1 env's per-env observations through `onnxruntime` (batched over envs)
  and apply the actions. **Blocker:** the IsaacLab G1 env obs must be reconciled to *exactly*
  the obs layout the stock ONNX expects (order, scaling, history, reference encoding) — i.e.
  build an **obs adapter**. If the mismatch is small, this is the cheapest path (im_eval gives
  Metrics 1&2 + coverage for free). Verify the G1 deploy's obs spec vs `robots/g1.py`.
- **(b) Batched MuJoCo (MJX / GPU) harness with the stock ONNX** — the faithful path.
  Reproduce the deploy's obs construction + control loop in a **vectorized MuJoCo** rollout
  (MJX runs thousands of G1 sims on GPU), stepping the stock ONNX natively (no obs adapter
  risk). **Cost:** reimplement the deploy's obs+step logic (currently C++ in
  `gear_sonic_deploy/`) in the batched harness, plus your own per-motion progress/error
  logging (you don't get `im_eval` for free here).

> Recommendation: first **diff the stock G1 ONNX's expected obs against the IsaacLab G1 env
> obs**. If they're close → path (a) (cheapest). If they diverge badly (likely, since Unitree's
> obs ≠ this framework's) → path (b). Either way the metric/record/retarget steps below are
> identical. NOTE: the framework `robots/g1.py` + G1 configs still matter for path (a).

### Fallback (debug only, NOT for scale): G1 deploy in MuJoCo, one clip at a time
`deploy.sh sim --motion-data <ref>` + `record_motion_to_pkl.py` (proven tools) — useful to
**sanity-check a handful of clips** and calibrate failure thresholds against visual
judgment, but it does NOT scale to 39K. Do not build the batch pipeline on this.

---

## 4. Pilot first (prove it in ~a day, not a cloud campaign)

Do **one category** end-to-end before scaling:
- Pick **`locowalk`** (walks — cleanest, most likely feasible, easiest to judge).
- Run steps (a)–(f) on locowalk only → produce `feasibility_report.csv` + the X2 feasible PKL.
- **A/B fine-tune:** train X2 SONIC on (i) the G1-recorded→X2 locowalk corpus vs (ii) the
  same locowalk clips from the raw human retarget (`x2_ultra_locowalk_chain_matched.pkl`),
  same warm-start / iters / envs. Compare **iters-to-target tracking metric** and final
  quality.
- **Decision gate:** if the feasible corpus converges materially faster / cleaner → scale to
  all 4 categories. If not → the hypothesis didn't hold; document why.

---

## 5. Key paths, tools, models (everything a fresh session needs)

### Inputs (all local, no download)
- **G1 references (source):** `agibot-x2-references/bones-seed/g1/csv/<date>/*.csv`
  — pre-retargeted human→G1, **120 fps**, soma format
  (`Frame, root_translate[cm], root_rotate[euler xyz deg], 29 joints[deg]`), ~142k files.
- **Categories & manifest:** `agibot-x2-references/bones-seed/metadata/seed_metadata_v004.csv`
  (`move_name, move_duration_frames@120fps, category, is_mirror, ...`). The 4 planner
  categories: locowalk / locomanip / locopost / locobal.
- **Raw-retarget X2 baselines (for the A/B):** per-category combined PKLs at
  `gear_sonic/data/motions/x2_ultra_<cat>_chain_matched.pkl` (chain_matched recipe = the
  training recipe). Also `x2_uniform_h14` / legacy `x2` variants under
  `agibot-x2-references/bones-seed/retargeted/`.

### Tools that already exist (built/used in the 2026-07-11 session)
- **`gear_sonic/scripts/record_motion_to_pkl.py`** — captures the robot's *executed* motion
  → motion PKL **with world coords**; `--robot g1 --g1-csv <path>` also emits a
  retargeter-ready G1 soma CSV. Needs the sim's `robot_pose` ZMQ PUB:
  set `GEAR_SONIC_ROBOT_POSE_ZMQ_PORT=5570` on the G1 sim bridge
  (`gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`, opt-in). Captures at ~50 fps.
- **G1→X2 retarget:** `agibot-x2-references/soma-retargeter` (its own git repo, branch
  `g1-to-x2`). Run `app/g1_csv_to_x2_csv.py --g1-dir <dir> --out-dir <dir>
  --config soma_retargeter/configs/agibot_x2_ultra/g1_to_x2_ultra_pinroot_retargeter_config.json`
  (**use the pin-root config** — default `r_weight=2` flips the pelvis on heavy-turning
  clips; pin-root fixes it, regression-safe). ~7 s/clip on RTX 5090, no walk-speed trim.
- **X2 CSV → motion PKL entry:** `_x2_csv_to_entry()` in
  `gear_sonic/scripts/g1_captures_to_x2_motion_pkl.py` (also does full-folder batch build).
- **PKL → G1 deploy reference CSVs (for Option A):**
  `gear_sonic/scripts/training_pkl_to_deploy_csv.py` (Phase-2 shim, runs under `.venv_sim`;
  MuJoCo FK → `reference/<clip>/<motion>/` CSVs). Then `deploy.sh sim --motion-data <ref>`.
- **soma CSV ↔ motion PKL, robot-agnostic:** `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`
  (`--robot g1` / `x2_ultra`, `--fps 30 --fps-source 120`; supports individual output).

### Tools to BUILD (main new work)
- **Headless batch feasibility runner** (Option A) or **G1 eval rollout dumper** (Option B):
  loop all clips, run G1 SONIC, record executed motion, emit `feasibility_report.csv`
  (Metric 1) + deviation stats (Metric 2), auto-classify SUCCESS/FAIL. This is the crux.
- **Metric-2 deviation scorer:** executed-vs-reference joint/root/body/stride errors.
- **Compute-savings summarizer:** aggregate Metrics 1+2 + the pilot A/B into the headline.

### Training (same stack as the v1 teleop fine-tune)
- **Env:** conda `env_isaaclab` (`/home/stickbot/miniconda3/envs/env_isaaclab/bin/python`;
  isaaclab + torch 2.7/cu128 + accelerate). RTX 5090 local, or 8-GPU cloud for scale.
- **Launcher (parameterized):**
  `NUM_PROCESSES=1 NUM_ENVS=4096 NUM_ITERS=<N> USE_WANDB=True MOTION_FILE=<feasible pkl>
   EXP_NAME=sonic_x2_ultra_bones_seed_chain_matched_v3_recovery
   EXTRA_FLAGS="+checkpoint=<warm-start .pt>" bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh`
- **Warm-start base:** `~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/model_step_001376.pt`.
- **Config sets `target_fps: 50`** (`gear_sonic/config/manager_env/commands/terms/motion.yaml`);
  the loader resamples each clip from its own `fps` → 50 (`motion_lib_base.py:1877`). Store
  the feasible PKL at the recorded fps (≈50) — no need to downsample to 30.
- **ONNX export for deploy eval:** `gear_sonic/scripts/reexport_x2_g1_onnx.py --run-dir <RUNDIR>
   --checkpoint <pt INSIDE the run dir> --output <onnx> --force`
  (offline parity needs `dump_isaaclab_step0.py`; `--force` + onnxruntime sanity is fine).
- **Live eval:** `gear_sonic/scripts/run_x2_pkl_direct_stack.sh --model <onnx>` (USER's own
  terminal — the docker sim viewer needs the host GPU display) + `play_locomotion.py --pkl`.

---

## 6. Open design questions (resolve in the pilot)
1. **Option A (deploy MuJoCo) vs B (IsaacLab eval)** for the batched G1 rollout — pick by
   scalability. Verify whether a G1 SONIC *eval* path exists in the training framework.
2. **What exactly is "G1 SONIC"?** Confirm the stock G1 policy used for reference *tracking*
   (the release ONNX the G1 deploy loads), and that a pure reference-playback/tracking mode
   exists (`deploy.sh sim --motion-data`). The G1 planner is stock Unitree; make sure we're
   using the *tracker*, not the kplanner.
3. **fps handling:** G1 refs are 120 fps, the recorder captures ~50 fps, the deploy runs at
   its control rate. Confirm the executed recording faithfully represents the motion and
   downstream training (`target_fps=50`) is consistent.
4. **Failure thresholds** for Metric 1 (fall height/tilt, divergence tolerance) — calibrate
   on the pilot so the classifier matches visual judgment.

---

## 6b. ⚠️ CRITICAL: policy-failure vs motion-infeasibility (the false-negative problem)
**The central methodological risk (user insight, 2026-07-11):** `terminated` (our SUCCESS
signal) does NOT purely mean "infeasible." It conflates:
- **(a) genuine infeasibility** — the G1 reference is physically impossible → the robot falls /
  diverges hard → correctly filtered OUT.
- **(b) policy false-negative** — the reference IS feasible (e.g. a motion a robot has actually
  executed) but the *oracle policy* can't track it → terminates on a threshold → **wrongly
  filtered out.**

If we filter naïvely on `terminated`, we throw away **learnable, feasible G1 motions** the
oracle merely struggles with — the opposite of what we want.

**Implications for the G1 experiment:**
1. The filter's **false-negative rate = the stock G1 policy's tracking-failure rate on
   feasible motions.** A strong oracle (the point of using stock G1 SONIC) minimizes this, but
   do NOT assume zero — verify it.
2. **Distinguish the two failure modes — don't drop on `terminated` alone:**
   - **Genuine infeasible** ⇒ hard fall: large pelvis tilt / height collapse / runaway tracking
     error near the fail frame.
   - **Policy false-negative** ⇒ clean reference (low ref tilt, normal kinematics), moderate
     deviation, terminates on a soft threshold. **Keep these** — feasible, just hard (exactly
     the learnable clips a downstream training wants).
   - Classify using **Metric 2 (deviation) + fall-severity at the fail frame + the reference's
     own kinematics**. Consider: only DROP on catastrophic fall; for soft-threshold terminations,
     KEEP (or flag for review).
3. **Sanity-check the filter on known-feasible G1 motions:** feed the G1 sweep motions we KNOW
   are executable (e.g. G1-recorded / teleoped clips **on G1**) — they should nearly all pass.
   A high false-negative rate there means the **oracle is too weak OR the G1 env termination
   conditions are miscalibrated** (§6 open Q4) → fix before trusting the filter.
4. **Eval start-state matters:** training uses RSI (random start); a deterministic frame-0 eval
   can hit a mid-clip hard segment from a worse state than training ever did. Evaluate each
   motion from a few start offsets and take the best/most-common outcome so one unlucky rollout
   doesn't false-fail a feasible clip.

> NOTE: this insight surfaced from an **X2-only mechanism dry-run** of `im_eval` (a clean,
> robot-recorded reference got terminated by the policy despite being feasible). The dry-run is
> **not part of this experiment** — it only validated that the vectorized `im_eval` sweep runs
> and emits the feasibility metrics. The experiment itself is **G1 policy on G1 references,
> X2 entirely excluded** until Phase 2.

## 7. Risks / caveats (be honest in the writeup)
- **G1-feasible ≠ X2-feasible.** Different height (G1 1.70 vs X2 1.40), leg length, wrist
  range (we hit X2's narrower wrists). A motion G1 executes can become infeasible *again*
  after the G1→X2 retarget (shorter legs → recorded stride may not fit). **Mitigation:** a
  cheap second pass — run the retargeted X2 refs through X2 SONIC and drop the ones that
  broke in transfer (a second feasibility filter, on the target embodiment).
- **You inherit G1 SONIC's imperfections.** If G1 understeps / has its own floor, the
  recording has it, and you distill it into X2 — capping X2 at "G1's behavior." Fine for a
  strong fast baseline; know the ceiling.
- **Data-gen compute:** one G1 rollout per clip (~35k). Cheap (a forward-pass rollout) vs
  PPO, but non-zero; Option B amortizes it into an eval pass.

---

## 8. Success criteria
- **Metric 1 + 2 produced** for at least the pilot category (ideally all 4).
- **Pilot A/B** shows the feasible corpus reaches a target tracking metric in **fewer iters**
  (or higher final quality at equal iters) than the raw-retarget corpus.
- A **defensible headline compute-savings estimate** for the next full run.
- Decision: **scale** (all categories, mix into the big corpus) or **stop** (hypothesis
  didn't hold), with the data to justify it.

---

## 9. How this connects to prior work (context for the fresh session)
- The **G1→X2 retarget** (soma-retargeter `g1-to-x2` branch, pin-root config) was built and
  validated 2026-07-11; the **teleop capture→retarget→fine-tune loop** was proven the same
  day (v1 teleop fine-tune improved medium-walk foot-slip on the real robot). This
  experiment generalizes that loop from *hand-teleoped* clips to the *whole bones-seed corpus*,
  using G1 SONIC as an automatic feasibility filter instead of a human operator.
- Go-forward from v1: mix feasible clips into the big bones-seed corpus for the real run;
  this experiment produces exactly that clean, feasible corpus + the compute-savings case.
- See project memory `project_g1_pipeline_status.md` for the full retarget/fine-tune history.
