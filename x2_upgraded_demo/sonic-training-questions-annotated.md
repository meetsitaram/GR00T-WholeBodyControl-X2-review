# SONIC / kplanner questions — annotated against the papers

Sources:
- **SONIC**: "SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control" (`~/Downloads/gear_sonic.pdf`, 37 pp)
- **MotionBricks** (kplanner): SIGGRAPH 2026, ACM TOG 45(4) (`~/Downloads/motionbricks_siggraph_2026.pdf`, 22 pp)

Legend: ✅ answered by paper · 🟡 partially answered · ❌ paper is silent → **ask NVIDIA**

---

## kplanner (MotionBricks)

### 1. Trained only on the 145k boneseed, or other datasets too?
✅ **Answered — and it corrects our premise.** The flagship model is trained on the full **proprietary 350k-clip / 700-hour** mocap dataset (9,300 skills, 36 categories, 163 performers; p.11 §7.1). **BONES-SEED (~143,792 clips) is the released open-source *subset* (~40% of motions, 17/36 categories), not the training set** (Appendix E, p.18). They also retrain separately on HumanML3D and LaFAN1-G1 for benchmarking (Table 2, p.10); no public data is mixed into the flagship model. A "Bones-70k" variant exists for the scaling study (Table 4, p.13).

> Implication for us: training on BONES-SEED alone is already a ~2.4x data reduction vs their flagship. Worth asking what quality delta they saw on the Bones-70k / BONES-SEED-scale variants.

### 2. Discard motions >200 frames? Split or discard?
✅ **Resolved by code archaeology (2026-07-24): the >200-frame discard is OURS, not NVIDIA's.** The paper has no length-discard rule — training samples **windows of 12–64 frames (steps of 4) at 30 fps** from clips of any length (p.5 §3–4); long public clips are segmented, not dropped (LaFAN1 cut into 6 s clips, p.11).

Code provenance:
- NVIDIA's preview release (vendored at commit `b9b634b`) ships **no real-data loader at all** — `scripts/train_vqvae.py` / `train_root.py` / `train_pose.py` use `SyntheticMotionDataset`, where `min_frames=80, max_frames=200` (root: 200/400) is just the **random length range for generated placeholder data**, not a filter. Their `docs/adding_your_own_dataset.md` says the real training code comes later via GR00T-WholeBodyControl and gives no length-cap guidance.
- The actual hard skip (`if n_eff < min_frames or n_eff > max_frames: continue`) lives in **our** `motionbricks/motionbricks/data/x2_bones_seed_dataset.py:161`, added in our commit `a070e64` (2026-05-28), with `--max_frames default=200` in our `train_*_x2.py` — the 200 was copied from upstream's synthetic placeholder range.

**Action item (ours, not a question for NVIDIA):** replace the discard with windowing — sample 12–64-frame windows from long clips like the paper does. Discarding drops every clip >6.7 s at 30 fps, which likely skews our corpus toward short/simple motions.
**Optional ask for NVIDIA:** confirm the forthcoming real dataloader windows arbitrary-length clips (no length cap).

### 3. 1M steps on 32 GPUs — does 8 GPUs get ~1/4 the quality?
✅ **Answered, and better than feared.** Actual budget: **2M updates, batch 256/GPU, 32 GPUs (4×8), lr 5e-5 cosine** (p.15). Wall-clock on H100: **7d tokenizer + 3d root + 7d pose**; "16 GPUs yields similar visual quality"; L40S ≈ 2× slower. GPU-scaling ablation (Appendix A, p.18; Fig 16 p.19) ran 1/2/4/8/16/32/64 GPUs: "more GPUs lead to faster convergence and modestly better asymptotic performance… gap primarily in tokenizer quality… only minor perceptual changes. **Training with fewer GPUs remains viable without significant degradation in visual quality.**" 8-GPU curves sit close to 32/64; only 1 GPU is visibly worse. It is *not* a 1/4th-quality situation — mostly just longer wall-clock.

### 4. If VQVAE is trained further, must the Pose model restart from scratch?
🟡 **Dependency is explicit; the resume question is not.** Training is strictly sequential: tokenizer first, then root + pose (p.4 §3; separate 7d/3d/7d runs, p.15). The pose module's output heads are **softmax distributions over the VQVAE codebook indices** (Eq. 5, p.7), so any change to codebook contents (running-mean codebook updates continue during extended VQVAE training, p.6 §4.3) invalidates the pose model's targets → retraining pose is architecturally required. The **root module is codebook-independent** (operates on root trajectories/keyframes, p.6) and should survive a VQVAE change.
**Ask NVIDIA:** after extending VQVAE training, does warm-starting the pose model work in practice, or is from-scratch needed? Same for root.

### 5. In-place turns underperform for manipulation — known issue? How to prioritize during fine-tuning?
❌ **Not answered — keep in questionnaire.** No turning-in-place analysis. Turning is ~4% of activities in the data distribution (Fig 18, p.19). The paper acknowledges rare-motion coverage as a limitation ("only one clip for vaulting at 0.5m", p.15 §8.1). Crucially, the paper's stance is **zero fine-tuning** — a fixed pre-trained backbone with no further tuning or tagging (p.3) — so there is no oversampling/fine-tuning recipe at all.
**Ask NVIDIA:** recommended recipe for category oversampling / fine-tuning for in-place turns; also whether the root module's "control dead zone" handling (p.7-8 §6.1) interacts badly with near-zero-translation turning commands.

### 6. How do we know kplanner training has converged (to save compute)?
🟡 **Partial.** They use a fixed 2M-update budget; "Figure 13 suggests performance plateaus earlier" (p.15). Metrics to watch: tokenizer **validation reconstruction loss** and pose-module **validation token cross-entropy** (Fig 16 p.19, Fig 13 p.14), plus keyframe joint position error. ⚠️ **FID is a misleading stop signal** — it *worsens* with more data due to small-data overfitting artifacts (p.14); keyframe precision is the metric that improves monotonically.
**Ask NVIDIA:** what validation-loss plateau thresholds they use in practice per component (tokenizer/root/pose) to call a run done.

### kplanner bonus facts worth having
- Architecture: multi-head VQVAE 23.5M params (recommend **128–256 tokens/head**, ~10^6–10^7 total capacity sweet spot, p.13); root module 50M; pose module 150M (p.15).
- All data at 30 fps; 0–10 random keyframe constraints per sample.
- Deployment: replan every 3–9 frames + instant replan on command change (App. C, p.18); Jetson Orin ~5 ms/inference at 10 Hz replanning.

---

## SONIC

Global context: Unitree G1 only (29 DoF), Isaac Lab, **128 GPUs × 7 days = 21,000 GPU-h, 50k iterations, 4096 envs/GPU**, 317,189 clips / 611 h at 50 Hz (Table 2 p.12, Table S2 p.31). All real deployments use the 42M-param model. **Cross-embodiment (X2) fine-tuning is never addressed — that's our #1 question.**

### 1. Fast arm tracking (combat); will an added arm-velocity reward upset balance?
🟡 **Partial.** No arm-specific velocity reward exists — whole-body link lin/ang-velocity tracking terms (weights 1.0/1.0, σ=1.0/3.14) cover arms implicitly (Table S3, p.33). Combat is huge in their data: **50,162 combat clips** (Table 2, p.12), with sword lunge / roundhouse kick / interactive boxing demos working. ⚠️ Note there's an **anti-shake penalty on wrist+head angular velocity above 1.5 rad/s** (weight −5e-3) that mildly fights very fast arm moves — check whether our port kept it and whether it clips combat speeds.
**Ask NVIDIA:** have they needed an explicit arm-velocity term; does our added one risk fighting the anti-shake penalty.

### 2. Style tuning: fluid dance vs firm manipulation?
✅ **Answered: there is no style parameter.** One policy, one reward set, identical hyperparameters for everything (p.11). Style comes entirely from the **kinematic planner side** — style-specific motion clips, "one representative motion clip and no retraining" (p.20). Dance = 9,689 clips in training. So: get fluid dance / steady manipulation by feeding the right reference motions, not by tuning tracker rewards.

### 3. Arm payload / arms drooping / falling forward under load?
🟡 **Partial — training has NO arm payload.** No load in arms during training, no weight-holding config. Closest thing: **base-CoM offset randomization** (Δx±0.075, Δy±0.1, Δz±0.1 m) and push perturbations (Table S4, p.33). Robustness demos (11 kg dropped on the robot, box carry in VLA task) are deploy-time emergent, not trained (p.29-32).
**Ask NVIDIA:** recommended way to train for sustained arm payload — end-effector mass randomization? CoM randomization range scaling? (This matches our observed fall-forward-with-load failure; the paper's CoM DR is the obvious knob to extend.)

### 4. How to conclude SONIC training has converged?
✅ **Answered.** Fixed **50k iterations "when performance usually plateaus"** (p.5), across all compute scales. Watch success rate (termination: root/EE height dev >0.25 m or root ori >1 rad) and MPJPE. 128 GPUs × 7 days; ablations at 16/32/128 GPUs all ran the same 50k iterations.

### 5. Adaptive sampling: do impossible motions eat compute? Are infeasible motions discarded?
✅ **Answered on both halves.** (a) Bins are **1 s fixed-duration**, sampling weighted by **failure rate capped at β=200**, blended α=0.1 with uniform (p.16, Table S2 p.31) — the cap exists precisely to bound compute spent on never-succeeding motions. (b) **Yes, they pre-filter**: "we filter out physically implausible motions (e.g., stair climbing, seated activities) that cannot be executed on the target robot," cutting ~700 h → 611 h (p.12 §3.1). So dropping ladder-climb-type motions from boneseed is exactly their own practice. Residual impossible motions still remain (zombie crawl, cross-legged sit fail, p.29).

### 6. 8-GPU node: reasonable convergence, just slower?
🟡 **Partial — and the answer is unfavorable.** Their ablation (16/32/128 GPUs, all to 50k iters): "more GPUs yield better **asymptotic** performance at the same iteration count, as larger batch sizes improve optimization stability" (p.5, Fig 2c). I.e. fewer GPUs plateaus *lower*, not just later — 98.0%/27.7 mm (smallest) vs 99.6%/23.8 mm (largest). They did **not** test whether running small-GPU jobs longer closes the gap.
**Ask NVIDIA:** on 8 GPUs, does running more iterations (or gradient accumulation to emulate the 128-GPU batch) recover the asymptote?

### 7. Train on only ~40K locomotion+manipulation clips — what do we lose?
✅ **Answered.** Data ablation at 20k/50k/110k/310k clips (4M/10M/22M/100M frames), subsets sampled uniformly across sub-categories (p.5, Fig 2a). **What degrades is OOD generalization** — test-content success ~99.6% → ~98.8% going 100M → 4M frames — while test-repetition (seen skill types) stays ~99.6-99.8%. So a 40k-clip run keeps trained skills but loses robustness to novel motions. Their advice embedded in method: preserve **category diversity** when subsampling.

### 8. Pre-filtering boneseed via stock G1 model — reasonable?
❌ **No precedent in the paper — keep in questionnaire.** Their only filtering is a-priori physical-plausibility filtering post-retargeting (p.12). Adaptive sampling *down-weights* persistent failures online rather than discarding.
**Ask NVIDIA:** is policy-based pre-filtering sensible, or does it remove hard-but-learnable motions that adaptive sampling would have eventually cracked? (Risk: stock-G1 failures ≠ X2 failures.)

### 9. Soft-step / landing reward?
🟡 **Partial.** No contact-force/soft-landing reward. They have a **feet joint-acceleration penalty** (Σ ankle q̈², weight −2.5e-6, "to encourage smooth foot contacts", p.14 + Table S3) and undesired-contact penalty (−0.1, force >1 N, non-ankle/wrist bodies). Nothing scaled by actual landing impact force.
**Ask NVIDIA:** did they try a contact-force-based landing penalty, and does one destabilize the tracking-reward balance?

#### Follow-up: does the feet-acceleration penalty hurt fast combat/dance leg moves?
Analysis (2026-07-24) — **mostly no**, verified against our own port ([terms/feet_acc.yaml](../gear_sonic/config/manager_env/rewards/terms/feet_acc.yaml) = `joint_acc_l2` on `.*ankle.*`; [sonic_x2_ultra.yaml](../gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra.yaml) overrides weight to −2.5e-6, matching the paper):
1. **Ankle-joint-space only.** Kicks and fast steps are driven by hip/knee, which the penalty doesn't touch; the ankle *joint angle* can stay smooth while the leg swings fast. It is not a Cartesian foot-motion penalty.
2. **Magnitudes only bite at impacts.** 2.5e-6 × q̈²: a brisk ankle at 100–200 rad/s² costs 0.025–0.1 reward (noise vs the ~6.5 total tracking weight); an impulsive ground strike at ~2000 rad/s² costs ~10 (dominant). The term is effectively inert except at hard contact transients — hence the paper's "encourage smooth foot contacts."
3. **Tracking gradient wins.** Link lin/ang velocity tracking (weights 1.0 each) pays for matching fast reference legs; slowing down to dodge the tiny acceleration penalty costs far more. Empirically NVIDIA trained 50k combat + 9.7k dance clips at this exact weight to >99% success.

Real interaction: sharp stomps / heel strikes get slightly softened — which for us is a feature (it's an indirect soft-step term, as the `sonic_x2_ultra_dance_softland.yaml` comment notes). ⚠️ Config gotcha: the term file's base default is **−2.5e-7** (10× weaker than paper); only configs that override it (e.g. x2_ultra) match the paper's −2.5e-6 — verify what the big-run config resolves to.

Sharper framing for the questionnaire: our robot lands hard *despite* this term being active at paper weight → points at sim-to-real (paper's own feet MPJPE gap is largest: 53.7 mm real vs 29.0 mm sim), not a missing reward.

### 10. Pause mid-run, evaluate, tweak, resume?
❌ **Not addressed — keep in questionnaire.** Single-stage end-to-end training, no curriculum, identical hyperparameters for all runs (p.11, p.16). Built on HF Accelerate + TRL (checkpoint/resume implied by tooling but never discussed). Note: we already have working `+resume=true` mechanics in our fork — the real question for NVIDIA is whether *changing rewards/settings mid-run* is something they've done safely (PPO value function / adaptive-sampling stats going stale).

### 11. Motor torque specs — follow manufacturer exactly or adjust for Isaac Lab?
❌ **Paper is silent — keep as a top question.** No effort limits, velocity limits, torque limits, action scaling, or actuator modeling anywhere; PD gains "follow prior art" (p.14); and notably the DR table (S4) has **no motor-strength/torque randomization at all**. Directly relevant to our waist-torque (48→36) and wrist velocity_limit (4.188 vs 20.9 hw) findings.
**Ask NVIDIA:** how do they set effort/velocity limits in sim vs datasheet; any actuator-strength DR recommended for a robot (X2) with less well-characterized motors.

### 12. Robot-encoder-only training; robot encoder for lower body + IK arms for teleop?
✅ **Mostly answered.** Three encoders (robot / SMPL-human / hybrid) → shared FSQ token space, jointly trained with cycle-consistency losses; dropping consistency losses → 8× cross-encoder divergence (p.14-15, p.19-20). Per-encoder: robot 99.6% SR, human 99.6%, hybrid 99.2%. Robot-encoder-only training is not ablated but nothing architecturally forbids it — you lose human-data and VLA-mixing paths.
**Our teleop plan has direct precedent**: the **hybrid encoder is exactly this** — sparse upper-body keypoints (head+hands) + lower-body robot motion, used for 3-point VR teleop (p.9 §2.4, p.36 §S8). Difference: keypoints feed the encoder directly, not an IK solver.
**Ask NVIDIA:** compute saved by robot-only (or robot+hybrid, skipping SMPL) training; any gotcha training hybrid without the human encoder as an anchor.

### 13. Wrist training / joint min-max limits?
🟡 **Partial.** Wrists get special treatment: anti-shake angular-velocity penalty (>1.5 rad/s, −5e-3), excluded from undesired-contact penalty, and joint-limit violation penalty is heavy (−10.0/joint) (Table S3, p.33). **No sim joint velocity limits are specified anywhere** — which is exactly where our X2 wrist-twist bug lived (velocity_limit 5× too low).
**Ask NVIDIA:** what velocity/effort limits they set on G1 wrists in Isaac Lab, and whether they saw limit-induced wrist artifacts.

---

## Refined questionnaire for the NVIDIA SONIC team

Questions the papers already answer are dropped; the remaining list, roughly priority-ordered for our X2 run:

1. **Embodiment transfer (new, most important):** SONIC and MotionBricks are G1-only in the papers. For a new robot (Agibot X2, 29 DoF-class), do you recommend from-scratch training on retargeted data, or is there any warm-start/fine-tune path from G1 checkpoints? Any embodiment-transfer experiments internally?
2. **Actuator modeling:** How do you set sim effort/velocity limits vs manufacturer datasheets? Table S4 shows no motor-strength DR — deliberate? For a robot with less-certain motor specs, would you add actuator-strength randomization? (Context: we root-caused a deploy wrist artifact to a 5×-too-low sim velocity_limit, and had to adjust waist effort limits.)
3. **8-GPU compute:** Fig 2c shows fewer GPUs plateau *lower* at fixed 50k iterations. On 8 GPUs, does running longer — or gradient-accumulating to the 128-GPU effective batch — recover the asymptote, or is the gap fundamental?
4. **Arm payload:** Training includes no arm load, and we see fall-forward + arm droop when carrying weight in teleop. Recommended recipe — end-effector mass randomization, extended CoM DR, payload curriculum?
5. **Policy-based data filtering:** To cut costs we're filtering boneseed by running clips through the stock G1 SONIC model and dropping failures. Reasonable, or does this discard hard-but-learnable motions that adaptive sampling (β=200 cap) would have handled? Any smarter feasibility pre-filter you use beyond the plausibility pass in §3.1?
6. **Mid-run tweaks:** Have you paused a run, changed rewards/DR/sampling settings, and resumed? Any known failure modes (stale value function, adaptive-sampling stats)?
7. **Added rewards:** (a) arm-velocity tracking term for fast combat arms — does it fight the wrist anti-shake penalty? (b) contact-force-based soft-landing penalty — tried it? Impact on reward balance?
8. **Encoders scope:** Compute cost of training robot-encoder-only (or robot+hybrid, skipping the SMPL encoder)? For 3-point teleop we plan hybrid-style upper keypoints + robot lower body — any issues training the hybrid encoder without the human encoder present?
9. **Wrist specifics:** What velocity/effort limits do you use for G1 wrists in Isaac Lab; any joint-limit-induced wrist artifacts observed?
10. **kplanner resume:** After extending VQVAE training, does warm-starting the pose module work in practice, or is from-scratch retraining needed? Does the root module survive a codebook change untouched?
11. **kplanner in-place turns:** We see weak in-place-turn execution for manipulation positioning. Known? Recommended oversampling/fine-tuning recipe (the paper is zero-fine-tune by design), and does the root module's control-dead-zone handling interact with near-zero-translation turn commands?
12. **kplanner convergence thresholds:** What validation-loss plateau criteria do you use per component (tokenizer recon loss / pose token CE / keyframe error) to end a run early of the 2M-update budget? (We know FID is misleading with dataset scale.)
13. **kplanner data windowing:** Will the full open-sourced dataloader (GR00T-WholeBodyControl release) window arbitrary-length clips into 12–64-frame samples with no length cap? (Resolved on our side: the >200-frame discard was our own code in `x2_bones_seed_dataset.py`, copied from the preview release's synthetic placeholder range — we plan to switch to windowing.)

---

## Original questions (verbatim, for review)

- kplanner - was it trained only on the 145k boneseed or includes other datasets mentioned in the paper as well
- kplanner - training discards any motions more than 200 frames (what is exact number). are bigger motion files broken down into smaller ones or discarded?
- kplanner - when trained for 1M steps on 32 gpus, will training 1M steps on 8 gpus achieve approx. 1/4th success of the full run?
- kplanner - if we want to resume training for another 1M steps, since pose model is dependent on vqvae, does pose has to be started from scratch after vqvae runs for another 1M steps?
- kplanner - i found it hard to get the model to  execute in-place turns for manipulation tasks. have you observed similar under performance for such moves? or is there somewthing we can do to prioritize such motions during fine-turning at later stages
- kplanner - in order to save compute costs, how can we conclude that a training run has finally converged and it is time to stop further training.

- sonic - i observed that the model doesn't track fast moving arms well - especialy for combat moves. I have added extra reward to track arm velocity. will it impact overall balance across all rewards
- sonic - if we want to fine tune for specific style of motions - fluid dance moves - is there any specific parameter that can be tuned? v.s. if we want it for manipulation tasks - anything to tune to have it more steardy and firm positioning?
- sonic - the arms go down when we try to do pick ad place teleop for manipultions tasks. does the existing training incluce the max load capacity of the arms? or is there any config that can be added to improve overall weight holdign capacity of the arms. i observed the robot gets unbalanced and keeps falling forward where there is weight in the arms
- sonic - how can we conclude that the training run has finally converged
- sonic - adaptive sampling bins - if there are really hard motions in the dataset that the robot can never successed - will such moves take up a big portion of the overall compute used for the training? can we drop such moves to cut costs. there are motions in the boneseed that are impossible for the robot to train on - like climbing a ladder. are such moves discarded during trainging?
- sonic - we have planning to use a 8 gpu nodes for this training. will the convergance be reasonable and we just expect it to take much longer comparedc to a 128 node run?
- sonic - how much of the model degradation can occur if we just choose to traing  on roughly 40K locomotion and manipulatino tasks? what kind of abilities we will lose by such reduction
- sonic - in order to cut costs, we are runnign the 145K boneseed motions on the g1 stock model to filter out any motios that the g1 sonic model already fails. is this a reasonable?
- sonic - additional reward for soft steps - we observed our robot lands with hard steps during walks.  have you explored adding any reward for soft landing feet? if we add such reward, would it impact the overall balance of rewards?
- sonic - instead of one big trianing run, can we pause in the middle, check how the model runs, tweek settings and resume for there? 
- sonic - torque settings of motor - we had some challenges with waist torque settings for our robot. should the training follow exactly the specs provided by the manufacturer? or adjust something to make it work better in isaaclac?
- sonic - encoders - can we limit the training to just robot encoder and skip smpl/human/hubrid encoders to limit scope and cut down compute costs? for manipulation tasks that require teleop, can we use the robot encoder for lower body control, and ik for arms with 3 point teleop?
- sonic - we also noticed our wrists are trained well due to min-max limits of the robot wrist joint. have you observed any such challenges during training? or any tips on improving the wrist control?
- sonic - do you have any upcoming PRs for the repo for supporting new robot embodiments? right now we have written a ton of extra code to support our robot. wondering if there is awy to just pass in a urdf and get the trianing run ready for any new robot.
