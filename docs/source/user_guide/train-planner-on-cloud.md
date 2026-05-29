# Training the X2 Kinematic Planner on a Cloud GPU Node

This guide walks through training the MotionBricks-based **kinematic planner**
for the Agibot X2 Ultra on a multi-GPU cloud node. The planner consumes
high-level Quest 3 commands (yaw / forward / strafe) and emits 50 Hz
whole-body `pose` messages over ZMQ — replacing the heuristic planner that
ships with the deploy stack today.

It is the **planner counterpart** of [`train-on-cloud.md`](train-on-cloud.md)
(SONIC RL policy on Isaac Lab). The two flows share the same Nebius account,
the same BONES-SEED motion library, and the same side-channel-bundle pattern,
but the cloud-side stack is much lighter: **no Isaac Lab / Isaac Sim**, no
Vulkan ICD pin, no Omniverse EULA chase. MotionBricks is plain PyTorch +
PyTorch Lightning + MuJoCo (for forward kinematics only, headless).

> Reference docs: see [`train-on-cloud.md`](train-on-cloud.md) for the
> SONIC variant — the Nebius CLI walkthrough (Appendix A) and the Nebius
> capacity / husk advice (Appendix B.1–B.5) carry over verbatim, only the
> bootstrap and launcher are different.

## Architecture

```
Local workstation                                  Cloud node (8x GPU)
────────────────────                               ──────────────────────
  scp x2_planner_bundle.tar.gz  ─────────────►    ~/x2_planner_bundle.tar.gz
                                                          │
  Git remote  ─── git clone (current HEAD) ───►   GR00T-WholeBodyControl/
                                                          │  tar -xzf
                                                          ▼
                                                   gear_sonic/data/motions/*.pkl
                                                          │
                                                   build_x2_skeleton_assets.py
                                                          │
                                                   train_vqvae_x2.py  (DDP, N GPUs)
                                                          ↓
                                                   train_pose_x2.py   (DDP, N GPUs)
                                                          ↓
                                                   train_root_x2.py   (DDP, N GPUs)
                                                          │
                                                          ▼
                                                   motionbricks/out/
                                                      motionbricks_vqvae_x2/.../*.ckpt
                                                      motionbricks_pose_x2/.../*.ckpt
                                                      motionbricks_root_x2/.../*.ckpt
                                                   (+ optional W&B)
```

The pipeline trains three model components in sequence:

| Stage | Script | Depends on | Typical steps | Wall-clock (8x H200) |
|---|---|---|---|---|
| 1. VQVAE (motion tokenizer) | `train_vqvae_x2.py` | – | 500K | ~6 hr |
| 2. Pose backbone | `train_pose_x2.py` | VQVAE checkpoint | 200K | ~5 hr |
| 3. Root backbone | `train_root_x2.py` | – | 200K | ~3 hr |

Pose strictly requires a trained VQVAE; root is independent. Total wall-clock
on 8x H200 SXM is ~14 hr, ~$420 at $30/hr.

## 1. Pre-flight on your local workstation

> **Commit + push your code changes BEFORE building the bundle.** The
> cloud-side workflow is `git clone <REPO_URL> --branch <REPO_BRANCH>` →
> `tar -xzf bundle.tar.gz` on top. Anything you've only edited locally
> (Python, configs, shell scripts) but haven't pushed will silently be
> the *old* version on the cloud node. Run `git status && git log
> @{u}..HEAD` immediately before launching bootstrap. Same footgun
> warning as `train-on-cloud.md` §1.

The cloud node needs the gitignored motion library; the rest (skeleton
classes, train scripts, MJCF + meshes) lives in git and arrives via
`git clone` + `git lfs pull`.

For the X2 planner training, the bundle contains:

| Path | Purpose | Why not in git |
|---|---|---|
| `gear_sonic/data/motions/x2_ultra_bones_seed.pkl` | 37,968-clip training corpus (~3 GB) — full BONES-SEED minus dances, curated into 4 tiers (`locowalk`, `locopost`, `locomanip`, `locobal`) by `agibot-x2-references/bones-seed/scripts/curate_x2_planner.py` | gitignored (`data/`) |
| `gear_sonic/data/motions/x2_ultra_planner_smoke.pkl` | 30-clip smoke PKL (~3 MB; optional but recommended) | gitignored |
| `motionbricks/out/motionbricks_vqvae_x2/version_1/feature_cache/` | Pre-computed MuJoCo FK + motion-rep features (one `.pt` per clip + `manifest.json`) — built locally by `motionbricks/scripts/build_feature_cache_x2.py` to avoid burning H200 time on CPU FK | gitignored (large & deterministic) |

Build the bundle from the repo root with the helper:

```bash
cd <path/to/GR00T-WholeBodyControl>
bash motionbricks/scripts/cloud/build_planner_bundle.sh
ls -lh /tmp/x2_planner_bundle.tar.gz
sha256sum /tmp/x2_planner_bundle.tar.gz   # capture for cloud-side verify
```

The script tar-czips the BONES-SEED PKL plus the smoke PKL (auto-built if
missing) and prints both the size and sha256 hash. Override `OUT_TAR=` or
`INCLUDE_SMOKE=0` for variants.

## 2. Cloud node prerequisites

Same hardware envelope as SONIC training, but lighter on disk:

| Requirement | Recommended |
|---|---|
| OS | Ubuntu 22.04+ (Nebius `ubuntu24.04-cuda13.0` is ideal) |
| GPU | 8x NVIDIA GPUs, >= 24 GB VRAM each (planner is far less memory-hungry than SONIC RL — 24 GB is plenty) |
| CUDA | 12.x or 13.x driver + runtime |
| Disk | ~80 GB free (much smaller than SONIC's 200 GB; no IsaacLab + no IsaacSim wheels) |
| Network | SSH (for `scp`); outbound HTTPS for pip wheels (+ W&B if used) |

> **Tip — check Nebius capacity *before* `compute instance create`.**
> Same `nebius_gpu_scan.py` advice as `train-on-cloud.md` Appendix A —
> reuse the existing helper:
>
> ```bash
> python gear_sonic/scripts/cloud/nebius_gpu_scan.py --gpus 8 --min-on-demand 1
> ```
>
> The husk-detection pattern from `train-on-cloud.md` B.4 also applies.
> Allocate a 1×H200 first to validate the path if you're cold-starting on
> a new region.

## 3. Provision and bootstrap the node

The Nebius CLI walkthrough is identical to `train-on-cloud.md` Appendix A
(A.1 CLI auth → A.2 instance type → A.3 boot disk + cloud-init →
`ssh ubuntu@$PUBLIC_IP`). When you land on the node, run the
**planner-specific** bootstrap script instead of `bootstrap_fresh_node.sh`:

```bash
# (workstation) push the bootstrap script
scp motionbricks/scripts/cloud/bootstrap_planner_node.sh ubuntu@$PUBLIC_IP:~/

# (cloud node) SSH in and bootstrap
ssh ubuntu@$PUBLIC_IP
REPO_URL=https://github.com/<fork>/GR00T-WholeBodyControl.git \
REPO_BRANCH=<your-branch> \
  tmux new -d -s bootstrap "bash ~/bootstrap_planner_node.sh 2>&1 | tee ~/bootstrap.log"
# Detach-safe; poll with: tail -f ~/bootstrap.log
```

The script runs ~6 phases vs SONIC's 13:

| Phase | What it does |
|---|---|
| 0 | `nvidia-smi` pre-flight (assumes the boot image already has the driver) |
| 1 | OS packages: `git`, `git-lfs`, `tmux`, `htop`, `rsync`, `libgl1`, `libglu1-mesa`, `libegl1`, `libosmesa6` |
| 2 | Miniconda |
| 3 | Conda ToS accept (B.6 from `train-on-cloud.md`) |
| 4 | `conda create -n motionbricks python=3.10` |
| 5 | `git clone` GR00T repo + `git lfs pull` for X2 MJCF/meshes |
| 6 | `pip install -e motionbricks/` + `wandb` + `joblib` |
| 7 | Validation: torch CUDA visibility, `motionbricks` importable, `train_vqvae_x2.py --help` |

Wall-clock from a clean `ubuntu24.04-cuda13.0` Nebius node: **~5 min**
(vs ~12–15 min for SONIC). The dominant cost is the torch wheel pull
(~600 MB) and `git lfs pull` of the X2 meshes (~110 MB).

The script is **idempotent** — re-running on a partially set-up node skips
already-completed phases.

## 4. Transfer and unpack the bundle

From your workstation:

```bash
scp /tmp/x2_planner_bundle.tar.gz ubuntu@$PUBLIC_IP:~/
```

On the cloud node — extract from the **repo root** so paths land in their
final locations:

```bash
cd ~/GR00T-WholeBodyControl
sha256sum ~/x2_planner_bundle.tar.gz          # match the local hash
tar -xzf ~/x2_planner_bundle.tar.gz

# spot-check
ls -lh gear_sonic/data/motions/*.pkl
```

## 5. One-time setup: skeleton assets + stats + feature cache

MotionBricks needs a per-skeleton bundle of: kinematic-tree definition,
canonical T-pose, motion-feature normalization stats. For X2 these live
under `motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/`:

```bash
conda activate motionbricks
cd ~/GR00T-WholeBodyControl
python motionbricks/scripts/build_x2_skeleton_assets.py
# ~30 s; reads ~50 clips from the BONES-SEED PKL to estimate
# per-feature mean/std, writes to motionbricks/out/.../version_1/{skeleton,stats,hparams.yaml}
```

If you ever change the X2 skeleton class or the motion-feature pipeline,
re-run this. Otherwise it's strictly one-time per node.

### 5a. Pre-compute the FK feature cache (run **locally**, not on the H200)

Each clip's MuJoCo forward kinematics + motion-rep tensor is CPU-bound
(no GPU needed). Computing it serially inside `X2MotionDataset.__init__`
on the H200 wastes ~30+ minutes of GPU time per training launch. Instead,
run the parallel builder once locally and ship the cache in the bundle:

```bash
python motionbricks/scripts/build_feature_cache_x2.py \
  --pkl gear_sonic/data/motions/x2_ultra_bones_seed.pkl \
  --out-dir motionbricks/out/motionbricks_vqvae_x2/version_1/feature_cache \
  --workers 24 \
  --recompute
# 38k clips on a 32-core workstation: ~10-15 min
```

The bundle script (`build_planner_bundle.sh`) auto-detects this cache and
ships it; on the cloud node, training will see `manifest.json` in
`feature_cache/` and skip the FK extraction entirely.

> **DDP-safety note.** The skeleton-assets step is single-process and runs
> outside DDP. The training launcher (`run_planner_train_8gpu.sh`)
> auto-runs it if missing, so you can skip step 5 manually if you want;
> step 5a should still be run locally to save cloud GPU time.

## 6. Configure W&B (optional)

Once per cloud node:

```bash
wandb login                                  # paste your API key
```

To enable, pass `USE_WANDB=1` to either run script. The W&B project defaults
to `TRL_X2Ultra_Planner`; override with `WANDB_PROJECT=...`.

## 7. Launch training

### 7a. Smoke test on all 8 GPUs (~3 min, ~$2)

Before committing to the multi-hour run, verify the full pipeline (FK
extraction → DDP all-reduce → VQVAE → pose → root) end-to-end with the
30-clip planner-smoke PKL:

```bash
# launch on all 8 GPUs in tmux (W&B off, smoke step counts)
tmux new -d -s smoke "bash motionbricks/scripts/cloud/run_planner_smoke_8gpu.sh"
tmux a -t smoke               # attach to watch, Ctrl-b d to detach
tail -f ~/plan_smoke.log      # ...or follow the log file
```

The smoke runs 1000 VQVAE + 500 pose + 500 root steps on the smoke PKL.
Total wall-clock: ~3 min on 8x H200 SXM.

Override knobs (env vars; same vocabulary as the full launcher):
`NUM_GPUS`, `PKL`, `BATCH_PER_GPU`, `NUM_WORKERS`, `USE_WANDB`,
`VQVAE_STEPS`, `POSE_STEPS`, `ROOT_STEPS`, `SAVE_EVERY`, `LOG_FILE`.

If the smoke completes without crashing and you see VQVAE perplexity moving
in the log, you're cleared for the full run.

### 7b. Full training run

The same launcher script runs the real thing — just override the env vars:

```bash
tmux new -d -s plan_train "
  NUM_GPUS=8 \
  PKL=gear_sonic/data/motions/x2_ultra_bones_seed.pkl \
  VQVAE_STEPS=500000 \
  POSE_STEPS=200000 \
  ROOT_STEPS=200000 \
  BATCH_PER_GPU=4 \
  USE_WANDB=1 \
  LOG_FILE=\$HOME/plan_train.log \
  bash motionbricks/scripts/cloud/run_planner_train_8gpu.sh
"
tmux a -t plan_train     # attach; Ctrl-b d to detach
```

The launcher runs the three stages **sequentially**. After VQVAE finishes,
the pose stage automatically picks up
`motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt`.
Override `VQVAE_CKPT=` to point pose at a different VQVAE.

**Per-GPU `BATCH_PER_GPU` tuning** (measured on H200 SXM 144 GB):

| Stage | BATCH_PER_GPU | Mem used | Step time | Notes |
|---|---|---|---|---|
| VQVAE | 4 | ~14 GB | ~0.4 s | safe baseline |
| VQVAE | 8 | ~26 GB | ~0.6 s | +33% throughput |
| Pose | 2 | ~28 GB | ~0.7 s | safe baseline |
| Pose | 4 | ~52 GB | ~1.1 s | recommended on H200 |
| Root | 4 | ~12 GB | ~0.3 s | safe baseline |

Pose has the largest backbone (transformer w/ ~1.6 GB ckpt at G1 scale) so
it bottlenecks first. If pose OOMs at `BATCH_PER_GPU=4`, drop to 2.

**Budget envelope** (8x H200 SXM, full target step counts):

| Stage | Steps | Wall clock | Cost @ ~$30/hr |
|---|---|---|---|
| VQVAE | 500K | ~6 hr | ~$180 |
| Pose | 200K | ~5 hr | ~$150 |
| Root | 200K | ~3 hr | ~$90 |
| **Total** | | **~14 hr** | **~$420** |

The default is `FILTER=none`, which uses the full 37,968-clip curated
corpus. This is the recommended setup — the dataset is already curated at
the metadata level by `curate_x2_planner.py` (4 tiers: `locowalk`,
`locopost`, `locomanip`, `locobal`; dances explicitly excluded). Setting
`FILTER=loco` further narrows to the regex include/exclude patterns in
`motionbricks/data/x2_loco_filters.py`, which is rarely useful given the
metadata curation already done upstream.

```bash
# default — train on all 37,968 curated clips
bash motionbricks/scripts/cloud/run_planner_train_8gpu.sh

# narrow further (rarely needed)
FILTER=loco \
  bash motionbricks/scripts/cloud/run_planner_train_8gpu.sh
```

**Always launch inside `tmux`** (the helper scripts already use it) so an
SSH drop does not kill the multi-stage run.

## 8. Monitor

### 8a. While it's running (cloud-side)

```bash
ssh ubuntu@$PUBLIC_IP

# the live training session
tmux a -t plan_train     # Ctrl-b d to detach without killing
tail -f ~/plan_train.log

# per-stage log (most recent stage only — VQVAE log overwritten by pose etc.):
tail -f ~/plan_train_logs/vqvae.log    # while stage 1 is running
tail -f ~/plan_train_logs/pose.log     # while stage 2 is running
tail -f ~/plan_train_logs/root.log     # while stage 3 is running

# quick health snapshot
grep -E "Epoch|step=|loss" ~/plan_train.log | tail -n 6
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
grep -cE "Traceback|Fatal|^Error|FAIL|OOM" ~/plan_train.log    # should stay 0
```

The PyTorch Lightning trainer writes `last.ckpt` (rolling) and
`model-{step:07d}.ckpt` (numbered) under
`motionbricks/out/motionbricks_<stage>_x2/version_1/checkpoints/`. The
`SAVE_EVERY` env var controls cadence (default 5,000 steps for full runs,
500 for smoke).

A 500K VQVAE run produces 100 numbered checkpoints + 1 rolling = ~30 GB
under `motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/`. Drop
`--save-top-k` to a small positive number in the launcher if you want to
auto-prune; default is `-1` (keep all).

### 8b. Cloud → local round-trip (validating the trained planner)

The fastest way to sanity-check the trained planner without paying cloud
GPU time for visualization: pull the freshest checkpoints to your
workstation and run the local visualizer.

```bash
# (local) one-shot pull of all three stage out-dirs (~3 GB total)
mkdir -p ~/x2_cloud_planner_ckpts
for stage in vqvae pose root; do
  rsync -avz --partial \
    ubuntu@$PUBLIC_IP:"~/GR00T-WholeBodyControl/motionbricks/out/motionbricks_${stage}_x2/" \
    ~/x2_cloud_planner_ckpts/motionbricks_${stage}_x2/
done

# (local) run the VQVAE-reconstruction visualizer
DISPLAY=:0 conda run -n motionbricks --no-capture-output python \
  motionbricks/scripts/visualize_x2_vqvae_reconstruction.py \
  --motion-key Relaxed_walk_forward \
  --ckpt ~/x2_cloud_planner_ckpts/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt
```

Press `M` in the MuJoCo viewer to toggle between ground-truth and the
trained-VQVAE reconstruction. If the reconstruction tracks the GT closely
(jitter < ~5 cm on the pelvis, recognizable foot strikes), VQVAE training
converged — proceed to evaluating the pose+root planner.

## 9. Resume after interruption

The same Lightning checkpoint resume pattern as SONIC, but per-stage:

| Use case | Env var | Notes |
|---|---|---|
| VQVAE crashed at step N, pick up from `last.ckpt` | `RESUME_VQVAE=motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt` | preserves optimizer + LR scheduler |
| Skip VQVAE entirely (already trained), retrain pose | `RUN_VQVAE=0 VQVAE_CKPT=<path>` | pose loads VQVAE weights directly |
| Skip VQVAE+pose, retrain root only | `RUN_VQVAE=0 RUN_POSE=0` | root is independent |
| Retrain pose with a different VQVAE | `RUN_VQVAE=0 VQVAE_CKPT=<other-vqvae>.ckpt RESUME_POSE=<pose-ckpt>` | useful for VQVAE A/B tests |

Example — resume a crashed pose stage:

```bash
RUN_VQVAE=0 RUN_ROOT=0 \
RESUME_POSE=motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/last.ckpt \
  bash motionbricks/scripts/cloud/run_planner_train_8gpu.sh
```

## 10. Adapting this guide to a different embodiment / dataset

Mirroring `train-on-cloud.md` "Adapting this guide" — only the bundle
contents and the skeleton-assets step change:

1. Build the new motion-lib PKL with
   [`gear_sonic/data_process/build_x2_bones_seed_motion_lib.py`](../../../gear_sonic/data_process/build_x2_bones_seed_motion_lib.py)
   (or the embodiment-specific wrapper for your robot).
2. Add a new skeleton class under
   [`motionbricks/motionbricks/motionlib/core/skeletons/`](../../../motionbricks/motionbricks/motionlib/core/skeletons/)
   (use `x2.py` as a template) and a matching MJCF.
3. Add an asset-builder under `motionbricks/scripts/build_<robot>_skeleton_assets.py`
   that points at the new MJCF + skeleton class.
4. Update the bundle's file list in `build_planner_bundle.sh` (or set
   `OUT_TAR` / `INCLUDE_SMOKE` env vars). Everything else stays the same.

## Appendix — pointer to SONIC's Nebius reference

The Nebius CLI auth (A.1), instance-type table (A.2), boot disk + cloud-init
recipe (A.3), and Lessons-Learned (B.1–B.5) all live in
[`train-on-cloud.md`](train-on-cloud.md) and are 100% reusable for the
planner. The planner-specific bootstrap (`bootstrap_planner_node.sh`)
intentionally **skips** the heavy SONIC-only fixes that don't apply to a
pure-PyTorch workload:

| SONIC fix | Why the planner can skip it |
|---|---|
| B.7 isaacsim wheel pin | MotionBricks doesn't use IsaacSim |
| B.8 Omniverse EULA env | No EULA gate without IsaacSim |
| B.9 setuptools / flatdict / `--no-build-isolation` | Pulled in by IsaacLab; not needed |
| B.10b open3d / tensordict / vector-quantize | Already in `motionbricks/setup.py` |
| B.10c libGLU (only matters for Iray) | Not needed for headless training; we still install it for parity |
| B.10d NVIDIA Vulkan ICD pin | Vulkan is only needed by IsaacSim's renderer; PyTorch CUDA does not use Vulkan |

This is why the planner bootstrap finishes in ~5 min vs ~12–15 min for
SONIC. The flip side: if you ever want to evaluate the trained planner
**inside Isaac Lab** (e.g. against the SONIC RL policy in IsaacLab's
sim2sim sweep), you'll need to bootstrap the SONIC stack too — they
coexist on the same node without conflict (separate conda envs).
