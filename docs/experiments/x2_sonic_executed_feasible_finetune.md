# X2 SONIC Fine-Tune on the G1-SONIC Executed-Feasible Corpus (Nebius run plan)

**Goal:** fine-tune the X2 Ultra SONIC tracking policy on the new
`x2_sonic_executed_feasible.pkl` corpus (35,974 clips) on an 8-GPU Nebius node,
for **30k iterations**, warm-starting from the established iter-1376 base
checkpoint (`model_step_001376.pt`).

This doc is the run plan. It is **self-contained for the specific decisions and
commands of THIS run** and defers all cloud boilerplate (provisioning, Isaac Lab
install, bootstrap, gotchas) to the canonical guide:
[`docs/source/user_guide/train-on-cloud.md`](../source/user_guide/train-on-cloud.md).
Read that guide's Appendix A (Nebius) + Appendix B (gotchas) alongside this.

---

## 0. What this corpus is (why we're training on it)

`gear_sonic/data/motions/x2_sonic_executed_feasible.pkl` — **35,974 clips @ 50fps, dof=31, 4.1 GB, ~85.6 h of motion:**

- **35,940** G1-SONIC **executed** (dynamically-consistent) bones-seed motions
  that survived the G1-SONIC feasibility filter (the ~10% infeasible + ~28%
  degraded were dropped), then retargeted G1→X2 with the **pin-root / +4.4 cm
  foot-offset** config (single normal config — the floor-clamp variant measured
  *worse* for ground poses; see the retarget notes in
  `project_g1_pipeline_status` memory).
- **34** teleop walk clips (slow / medium / regular; VR + keyboard), merged from
  `x2_teleop_g1recorded_v1.pkl` at the user's request.

The defining property: every clip is a trajectory the **G1 robot actually
executed in sim**, so the corpus is implicitly feasibility-curated. This is the
compute-saving alternative to training on the raw bones-seed retarget and
letting the policy waste capacity on motions no robot can hold.

Provenance of the build: retarget via
`agibot-x2-references/soma-retargeter/batched_g1x2_driver.py` (4 sharded
workers), PKL via `gear_sonic/scripts/build_x2_motion_pkl_from_csvs.py`
(`--merge-pkl x2_teleop_g1recorded_v1.pkl`).

---

## 1. Ready-to-use artifacts (already created in this repo)

| Artifact | Path | Status |
|---|---|---|
| Training corpus | `gear_sonic/data/motions/x2_sonic_executed_feasible.pkl` | ✅ built + validated (no-NaN, 200-sample check) |
| Experiment config | `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_executed_feasible.yaml` | ✅ dry-composes clean |
| 8-GPU launcher | `gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh` | ✅ 30k iters, warm-start baked in |
| Warm-start checkpoint (`.pt`) | `~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/model_step_001376.pt` | ✅ on workstation (383 MB) |

> **Use the `.pt`, not the ONNX.** The warm-start must be the trainable torch
> checkpoint `model_step_001376.pt`. The `exported/model_step_001376_g1.onnx`
> beside it is the frozen deploy-inference graph exported *from* that same
> `.pt`; `+checkpoint=` cannot resume training from it. Same iter-1376 model,
> just the trainable form — the launcher already points at the `.pt`.

The config inherits the chain_matched-free **`sonic_x2_ultra_bones_seed_sphere_feet`**
base and re-applies the exact env/obs/reward recipe the iter-1376 checkpoint was
trained under (`level0_4_pd` KP/KD DR + `local_dir_hist_v2` obs noise +
orientation-reward boosts) so the observation normalization + reward balance
carry over cleanly. Those are recipe settings (DR / obs-noise / reward weights),
independent of any retargeting method. It overrides **only** the corpus, names,
LR ceiling, and iteration budget — no "chain_matched" appears in this run's
identity (`project_name: TRL_X2Ultra_ExecutedFeasible`, `exp_var:
executed_feasible_v1`).

---

## 2. Decisions to confirm before launch (with recommendations)

| Decision | Options | Recommendation |
|---|---|---|
| **Warm-start** | the iter-1376 base `.pt` (`model_step_001376.pt`) — **DECIDED** | Confirmed. Launcher defaults to it. (Not the ONNX — see §1 note.) |
| **Corpus scope** | (a) this corpus standalone · (b) mix in more bones-seed to anchor general capability | **(a)** — the executed corpus is already broad (35.9k clips, all categories) and dynamically feasible. Revisit (b) only if you see forgetting on non-executed motions. |
| **Iteration budget** | **30k — DECIDED** | Numbered checkpoints land every 2000; pull mid-run + eval, and early-stop if the metrics below plateau well before 30k (a warm-start often does). |
| **Learning rate** | KL-adaptive (current) vs cosine decay — see §2.1 | Run **KL-adaptive** as the baseline (with the capped ceiling). A/B a cosine run only if it stalls/oscillates. |
| **Reward weights (caveat)** | keep orientation boosts · or neutralize them | Keep them, but **watch `anchor_ori_full` terminations** — the executed corpus keeps some ground/lying motions (e.g. `faint_stand_up`) an upright-pelvis curriculum would drop, and the upright-pelvis reward prior can fight those. Neutralizing = one-line edit in the yaml (documented in its header). |

### 2.1 Learning rate — what's active, and the cosine option

**The active LR mechanism is KL-adaptive, not fixed and not cosine.** The
config resolves to:

```
schedule: adaptive          desired_kl: 0.01
actor_learning_rate: 2.0e-5 (start)     critic_learning_rate: 1.0e-3
adaptive_lr_min: 1.0e-5     adaptive_lr_max: 1.0e-4   (← capped for this run; base default 2e-4)
```

Each PPO update, `ppo_trainer._adjust_learning_rate_based_on_kl` reads the
measured policy KL and nudges the actor LR to hold it near `desired_kl`:

- KL > `2·desired_kl` (>0.02) → `lr = max(1e-5, lr/1.5)` (shrink — policy moved too much)
- KL < `desired_kl/2` (<0.005) → `lr = min(adaptive_lr_max, lr·1.5)` (grow — room to push)
- else → unchanged

then writes `lr` into every optimizer param-group. This is the rsl_rl-style
controller SONIC has always used; it self-regulates step size from the actual
policy-change magnitude, which is exactly what you want for a **warm-start**
(it auto-throttles if the new corpus initially disagrees with the base policy).
For this 30k run we lower `adaptive_lr_max` 2e-4 → **1e-4** so a long run can't
ramp the LR high enough to clobber the good base policy.

**Cosine decay is available in the codebase but is NOT a config flip.**
`gear_sonic/trl/utils/scheduler.py` has a `WarmupCosineScheduler`
(linear warmup → cosine anneal to `final_lr`), and `lr_scheduler_type` exists in
the config (currently `constant`). BUT the KL-adaptive function above
**overwrites `param_group["lr"]` every update**, so it would clobber any cosine
schedule unless you first disable it. To actually run cosine decay:

1. Disable KL-adaptive: `++algo.config.desired_kl=null` (the adjust fn early-returns when `desired_kl is None`).
2. Wire `WarmupCosineScheduler` into the PPO actor optimizer and step it each iter — **verify this path is actually constructed from `lr_scheduler_type`** (the constant default is effectively inert for the actor; the cosine wiring may need a small code change in `ppo_trainer.py`), with e.g. `num_warmup_steps≈500`, `num_training_steps=30000`, `final_lr≈1e-6`.

**Recommendation:** run the KL-adaptive baseline (capped ceiling) first — it's
proven for this stack and ideal for warm-start. Treat cosine as a *follow-up
A/B experiment* if the baseline oscillates or plateaus early, not a
prerequisite. If you want the exploration in this first run, the cleanest
low-risk variant is to keep KL-adaptive but tighten it further
(`++algo.config.adaptive_lr_max=5e-5`) rather than switch controllers.

---

## 3. Node

Provision (or reuse) an 8-GPU Nebius node per train-on-cloud.md Appendix A;
keep its `$INSTANCE_ID` and `$PUBLIC_IP` handy (from the console or CLI). Confirm
SSH + `nvidia-smi` show 8 GPUs before proceeding. If it's a fresh boot disk, run
`bootstrap_fresh_node.sh` (§4 below); if you're reusing a node that already has
`env_isaaclab`, skip to §5.

---

## 3b. Credentials on the node — do this FIRST (before clone/bootstrap)

The bootstrap clones the **private** repo and the run logs to W&B, so set both
credentials up before §4 (see train-on-cloud.md §2b for the full rationale):

```bash
# GitHub (private repo → HTTPS clone needs a 'repo'-scope token, else it hangs)
TOKEN=$(gh auth token)
printf 'https://<your-gh-user>:%s@github.com\n' "$TOKEN" | \
  ssh ubuntu@$PUBLIC_IP 'umask 077; cat > ~/.git-credentials; \
    git config --global credential.helper store'
ssh ubuntu@$PUBLIC_IP 'git ls-remote https://github.com/<fork>/GR00T-WholeBodyControl.git HEAD'
#   ^ a 40-char SHA => auth works; a hang/401 => fix the token before bootstrap

# W&B (required for USE_WANDB=True)
scp ~/.netrc ubuntu@$PUBLIC_IP:~/.netrc && ssh ubuntu@$PUBLIC_IP 'chmod 600 ~/.netrc'
```

---

## 4. Bootstrap (fresh node only)

Per train-on-cloud.md §3 (and after §3b credentials are in place). On your
workstation:

```bash
# push code first (see §5 — the config + launcher are new tracked files)
scp gear_sonic/scripts/cloud/bootstrap_fresh_node.sh ubuntu@$PUBLIC_IP:~/
```

On the node (13 idempotent phases, ~15 min, ends at a passing Hydra dry-compose):

```bash
REPO_URL=https://github.com/<your-fork>/GR00T-WholeBodyControl.git \
REPO_BRANCH=<branch-with-the-new-config-and-launcher> \
  tmux new -d -s bootstrap "bash ~/bootstrap_fresh_node.sh 2>&1 | tee ~/bootstrap.log"
tail -f ~/bootstrap.log
```

> **Branch footgun (train-on-cloud.md §1):** the new config + launcher are
> tracked files. If they aren't pushed to `REPO_BRANCH`, the cloud clone won't
> have them and you'll either fall back to the bundle copy or run stale code.
> `git status && git log @{u}..HEAD` before bootstrap.

---

## 5. Local pre-flight — commit/push + build the side-channel bundle

The PKL is gitignored (`data/`); the config + launcher are new tracked files.
Two clean options:

**Option A (recommended): commit + push the code, bundle only the PKL.**

```bash
cd <repo root>
git add gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_executed_feasible.yaml \
        gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh \
        gear_sonic/scripts/build_x2_motion_pkl_from_csvs.py
git commit -m "feat(x2-finetune): executed-feasible corpus config + 8gpu launcher"
git push    # to the branch you'll set as REPO_BRANCH

# bundle = the gitignored PKL only
tar -czf /tmp/x2_executed_feasible_bundle.tar.gz \
    gear_sonic/data/motions/x2_sonic_executed_feasible.pkl
ls -lh /tmp/x2_executed_feasible_bundle.tar.gz   # ~4 GB (already compressed PKL; tar adds little)
sha256sum /tmp/x2_executed_feasible_bundle.tar.gz
```

**Option B (no push): bundle the PKL + config + launcher together.**

```bash
tar -czf /tmp/x2_executed_feasible_bundle.tar.gz \
    gear_sonic/data/motions/x2_sonic_executed_feasible.pkl \
    gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_executed_feasible.yaml \
    gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh \
    gear_sonic/scripts/build_x2_motion_pkl_from_csvs.py
```

---

## 6. Transfer the bundle + the warm-start checkpoint

```bash
# corpus bundle (~4 GB; a few min on home wifi)
scp /tmp/x2_executed_feasible_bundle.tar.gz ubuntu@$PUBLIC_IP:~/

# warm-start checkpoint (383 MB) — into the exact path the launcher expects
ssh ubuntu@$PUBLIC_IP 'mkdir -p ~/x2_cloud_checkpoints/chain_matched_v3_iter_001376'
scp ~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/model_step_001376.pt \
    ubuntu@$PUBLIC_IP:~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/
```

On the node, extract from the repo root so paths land correctly:

```bash
cd ~/GR00T-WholeBodyControl
sha256sum ~/x2_executed_feasible_bundle.tar.gz     # match local
tar -xzf ~/x2_executed_feasible_bundle.tar.gz
git pull                                           # if you used Option A
```

---

## 7. Verify data + config (no GPU)

```bash
cd ~/GR00T-WholeBodyControl
# PKL sanity
python -c "
import joblib
d = joblib.load('gear_sonic/data/motions/x2_sonic_executed_feasible.pkl')
print(len(d), 'clips; fields:', sorted(next(iter(d.values())).keys()))
"
# Expect: 35974 clips; fields: ['dof','fps','pose_aa','root_rot','root_trans_offset','smpl_joints']

# warm-start present
ls -lh ~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/model_step_001376.pt

# Hydra dry-compose resolves the new config -> the unpacked PKL
python gear_sonic/train_agent_trl.py --config-name=base \
  +exp=manager/universal_token/all_modes/sonic_x2_ultra_executed_feasible \
  --cfg job 2>&1 | grep -E "motion_file|num_learning_iter|project_name"
# Expect:
#   motion_file: gear_sonic/data/motions/x2_sonic_executed_feasible.pkl
#   num_learning_iterations: 30000
#   project_name: TRL_X2Ultra_ExecutedFeasible
```

---

## 8. Smoke test (~3 min, ~$2) — do NOT skip

```bash
cd ~/GR00T-WholeBodyControl
NUM_ITERS=10 USE_WANDB=False LOG_FILE=$HOME/ef_smoke.log \
  bash gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh
```

Pass = it reaches the PPO loop, prints a few `Learning iteration` lines with a
finite reward, and exits clean. The launcher's pre-flight already hard-fails if
the PKL or warm-start `.pt` is missing.

---

## 9. Full run

```bash
tmux new -d -s ef "bash gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh"
tmux a -t ef        # Ctrl-b d to detach
tail -f ~/executed_feasible.log
```

Defaults: `NUM_PROCESSES=8`, `NUM_ENVS=12288`, `NUM_ITERS=30000`,
`USE_WANDB=True`, warm-start = iter-1376 `.pt`. W&B project
`TRL_X2Ultra_ExecutedFeasible`. Override any via env vars (see launcher header).

**Per-GPU `NUM_ENVS`** (from train-on-cloud.md §8b): 12288 is the H100-80GB safe
default; on H200-141GB you can push 16384 for ~+33% throughput. If you see
multi-hour memory creep, drop one tier.

---

## 10. Monitor — what to watch (this run's success signals)

Health snapshot (train-on-cloud.md §9a / B.0):

```bash
grep -nE "Learning iteration" ~/executed_feasible.log | tail -3
grep -cE "Traceback|Fatal|OOM|OutOfMemory" ~/executed_feasible.log   # 0
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

Run-specific metrics (W&B), vs the warm-start baseline:

| Metric | Expect |
|---|---|
| `Env/Metrics/motion/error_body_pos`, `error_joint_pos` | **drop** (tracking sharpens on the feasible corpus) |
| `Env/Episode_Termination/time_out` (success) | **rise** |
| `Env/Episode_Termination/anchor_ori_full` | **stay low / flat** — if it *climbs*, the ground/lying clips are fighting the orientation reward prior (see §2 caveat) |
| `Env/Episode_Termination/foot_pos_xyz` | watch — the v1 teleop run saw this rise; the executed corpus should behave better since it's dynamically consistent |
| mean reward | rise then plateau; early-stop candidate |

The repo writes numbered `model_step_NNNNNN.pt` every 2000 iters + a rolling
`last.pt` every 50. Grab a mid-run snapshot by copying `last.pt`.

---

## 11. Pull, export ONNX, eval

```bash
# (local) pull the run dir
RUN_DIR=$(ssh ubuntu@$PUBLIC_IP \
  'ls -td ~/GR00T-WholeBodyControl/logs_rl/TRL_X2Ultra_ExecutedFeasible/manager/universal_token/all_modes/* | head -1')
mkdir -p ~/x2_cloud_checkpoints/executed_feasible_v1
rsync -avz --partial ubuntu@$PUBLIC_IP:"$RUN_DIR/" \
  ~/x2_cloud_checkpoints/executed_feasible_v1/$(basename "$RUN_DIR")/
```

Export to deploy ONNX with the **standard** tool (NOT `export_wandb_run.py`):

```bash
python gear_sonic/scripts/reexport_x2_g1_onnx.py \
  --run-dir <RUN_DIR> \
  --checkpoint <RUN_DIR>/model_step_NNNNNN.pt \
  --output <RUN_DIR>/exported/model_step_NNNNNN_g1.onnx --force
```
> Gotcha: `--checkpoint` must be a `.pt` **inside** the run dir (config
> resolution sets `experiment_dir = checkpoint.parent`, needs the sibling
> `config.yaml`). A durable-copy path fails.

**Live eval** (user's desktop terminal — the docker MuJoCo viewer needs the host
display; see the `run_x2_pkl_direct_stack.sh` NV-GLX gotcha in the memory):

```bash
bash gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
  --model <exported onnx> &
python gear_sonic/scripts/play_locomotion.py \
  --pkl gear_sonic/data/motions/g1_recorded_x2/slow_walk_slow_keyboard_001.pkl
```
Designated validation motion = `slow_walk_slow_keyboard_001` (the sub-0.3 m/s
stepping floor the v1 run couldn't crack — see if this bigger corpus does).
Also spot-check IsaacLab `im_eval` feasibility across a corpus sample.

---

## 12. Budget + cost discipline

8× H100 SXM, `NUM_ENVS=12288`, ~9 s/iter at this corpus scale:

| Iters | Wall clock | Cost @ ~$22/hr (Nebius H100) |
|---|---|---|
| 10 (smoke) | ~3 min | ~$1 |
| 4000 (early checkpoint) | ~10 h | ~$220 |
| 8000 (often "good enough") | ~20 h | ~$440 |
| 30000 (full budget) | ~75 h | ~$1,650 |

> Pull + eval numbered checkpoints as they land (every 2000 iters) and
> **early-stop** the moment the §10 metrics plateau — the 30k cap is a ceiling,
> not a target. On H200-141GB, `NUM_ENVS=16384` cuts wall-clock ~25%.

**Stop the instance whenever not actively training** (Nebius bills per second;
disk-only is cheap):

```bash
nebius compute instance stop --id $INSTANCE_ID
```

---

## 13. One-glance command block (node already bootstrapped)

```bash
# workstation
git add -A && git commit -m "feat(x2-finetune): executed-feasible config+launcher" && git push
tar -czf /tmp/x2_ef_bundle.tar.gz gear_sonic/data/motions/x2_sonic_executed_feasible.pkl
scp /tmp/x2_ef_bundle.tar.gz ubuntu@$PUBLIC_IP:~/
ssh ubuntu@$PUBLIC_IP 'mkdir -p ~/x2_cloud_checkpoints/chain_matched_v3_iter_001376'
scp ~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/model_step_001376.pt \
    ubuntu@$PUBLIC_IP:~/x2_cloud_checkpoints/chain_matched_v3_iter_001376/

# node
cd ~/GR00T-WholeBodyControl && git pull && tar -xzf ~/x2_ef_bundle.tar.gz
NUM_ITERS=10 USE_WANDB=False bash gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh   # smoke
tmux new -d -s ef "bash gear_sonic/scripts/cloud/run_executed_feasible_8gpu.sh"            # real
```
