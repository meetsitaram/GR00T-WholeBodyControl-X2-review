# X2 kplanner Round 2 — Nebius training summary

> Consolidated reference for how the currently-deployed X2 kplanner
> (VQVAE 500K + Pose 500K + Root 315K) was trained on Nebius cloud
> between 2026-06-24 and 2026-06-26. This is a *snapshot* summary
> distilled from the primary sources below; when they disagree, the
> primary sources win.
>
> **Primary sources**
> - `docs/source/user_guide/kplanner_training_runs.md` — definitive
>   training history (all runs, W&B IDs, checkpoints)
> - `docs/source/user_guide/train-planner-on-cloud.md` — procedural
>   pipeline (bootstrap, corpus build, launch, pull-back)
> - `docs/source/user_guide/milestones/2026-06-26_kplanner_round2_postmortem.md`
>   — outcome analysis and what still didn't work

Deployed today: **VQVAE 500K + Pose 500K + Root 315K** (Root is an FT1
fine-tune from the 300K base). Defaults are pinned in
`gear_sonic/scripts/x2_kplanner.py` and mirrored in
`motionbricks/motionbricks/motion_backbone/inference/load_x2_planner.py`.

---

## 1. Nebius cloud instance

| Item | Value |
|------|-------|
| Instance | `computeinstance-e00cn8h67tdq00t2ys` at `195.242.31.46` |
| GPUs | **8× NVIDIA H100 SXM, 80 GB VRAM each** (not H200) |
| Host | 128 vCPU, 1.5 TB RAM, 1.3 TB boot disk |
| Image | `ubuntu24.04-cuda13.0` (no custom Docker) |
| Bootstrap | `motionbricks/scripts/cloud/bootstrap_planner_node.sh` — conda env `motionbricks`, Python 3.10, `pip install -e motionbricks/`, `wandb`, `git clone` + LFS |
| Cost / time | ~$600 for ~25.5 hr wall-clock |
| Now | Stopped; boot disk preserved for FT2 intermediates |

Capacity-planning helper: `gear_sonic/scripts/cloud/nebius_gpu_scan.py`.

**Data movement** (cloud-side build, not pre-built bundle):

| Direction | Method | What |
|-----------|--------|------|
| Cloud ← HuggingFace | HF download | `bones-studio/seed` → `soma_uniform.tar.gz` (~42 GB, ~55 min) |
| Cloud ← git | `git clone` in bootstrap | Code, MJCF via LFS |
| Workstation ← cloud | `rsync` | PKLs via `gear_sonic/scripts/cloud/pull_chain_matched_pkls.sh` |
| Workstation ← cloud | `rsync` | Skeleton / stats / hparams via `gear_sonic/scripts/cloud/pull_motionbricks_assets.sh` |
| Workstation ← cloud | `rsync` (manual) | Checkpoints (~2.3 GB total) — pattern documented in `train-planner-on-cloud.md` §8b |

---

## 2. Data pipeline (upstream of training)

Round 2 built the corpus on the cloud instance rather than shipping a
pre-built bundle.

1. **Raw** — HuggingFace `bones-studio/seed` (`soma_uniform.tar.gz`,
   ~42 GB).
2. **Curation** — planner-specific `curate_x2_planner.py` splits into
   4 tiers → **37,968** BVH clips (`locowalk` 18,036 / `locopost` 8,752
   / `locomanip` 9,712 / `locobal` 1,468). Dances excluded.
3. **Retargeting** — Soma retargeter with
   `soma_to_x2_ultra_chain_matched_retargeter_config.json` (the fix for
   the legacy `uniform_h14` "Groucho crouch": pelvis Z ~0.51 m →
   chain_matched ~0.63-0.68 m). 8× H100 parallel with per-shard
   `CUDA_VISIBLE_DEVICES` round-robin, **24 shards / 3 per GPU**,
   ~46 min for all 37,968 CSVs (~16 GB).
4. **CSV → PKL** —
   `gear_sonic/data_process/build_x2_bones_seed_motion_lib.py
   --out-suffix _chain_matched --subsets locowalk locopost locomanip
   locobal` → `x2_ultra_bones_seed_chain_matched.pkl` (**37,968
   clips**, used for VQVAE).
5. **Halfspeed** — same builder over `locowalk` only with
   `--fps-source 60 --min-mean-speed 0.20 --min-mean-yaw-rate 0.15` →
   `x2_ultra_bones_seed_chain_matched_halfspeed.pkl` (**11,822 clips**,
   ~65% of locowalk kept via OR-filter on mean speed / yaw rate).
6. **Merge** — `motionbricks/scripts/cloud/merge_halfspeed_pkl.py` →
   `x2_ultra_bones_seed_chain_matched_v2.pkl` (**49,790 clips**, used
   for Pose + Root).
7. **Skeleton + stats** —
   `motionbricks/scripts/build_x2_skeleton_assets.py` on the base PKL,
   ~102 min, writes `hparams.yaml`, `skeleton/`, `stats/`. VQVAE's
   feature cache is then symlinked for Pose/Root to save ~60 min per
   stage.

**Design intent**: VQVAE stays on the base 1× PKL so the codebook
doesn't double-count halfspeed duplicates. Pose + Root learn on v2 to
broaden low-speed coverage for VR-driven walking.

### Clip counts at each stage

| Artifact | Clips | locowalk | locopost | locomanip | locobal |
|----------|-------|----------|----------|-----------|---------|
| Base `*_chain_matched.pkl` (VQVAE) | **37,968** | 18,036 | 8,752 | 9,712 | 1,468 |
| `*_halfspeed.pkl` | **11,822** | 11,822 | — | — | — |
| Merged `*_v2.pkl` (Pose+Root) | **49,790** | 29,858 | 8,752 | 9,712 | 1,468 |
| After skeleton `[60,400]` filter | **34,232** | — | — | — | — |
| Feature cache base (`[80,300]`) | **31,329** cached | — | — | — | — |
| Feature cache v2 | **35,654** cached | — | — | — | — |

---

## 3. Training invocations (the three models)

All three stages use **PyTorch Lightning DDP**, **fp32**,
**AdamAtan2** + WarmupCosineScheduler, `MIN_FRAMES=80 MAX_FRAMES=200`,
launched through `motionbricks/scripts/cloud/run_planner_train.sh`. No
DeepSpeed. `NUM_WORKERS=8` per rank.

| Stage | PKL | Per-GPU batch | Effective batch (8 GPU) | Steps | LR (peak → final) | Wall-clock | W&B run |
|-------|-----|---------------|-------------------------|-------|-------------------|------------|---------|
| VQVAE | base | 16 | **128** | **500K** | 2e-4 → 4e-6 | ~6h 50m | `v32rja0e` |
| Pose | v2 | 32 | **256** | **500K** | 1e-4 → 2e-6 | (see below) | `wt3549mj` |
| Root R2 | v2 | 32 | **256** | **300K** | 1e-4 → 2e-6 | ~4h 40m | `itv2b4jq` |
| Root FT1 → **315K** | v2, loco filter | 32 | 256 | +15K | cosine tail | ~50 min | `u1ju8uer` |

`SAVE_EVERY=5000` originally, later `2500` after the GPU-1 HBM DBE.
Keyframe warmup: 200K steps for both Pose and Root.

### Stage 1 — VQVAE (base PKL)

```bash
tmux new -d -s plan_train "
  NUM_GPUS=8 \
  PKL=gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched.pkl \
  VQVAE_BATCH_PER_GPU=16 VQVAE_STEPS=500000 \
  RUN_POSE=0 RUN_ROOT=0 \
  NUM_WORKERS=8 MIN_FRAMES=80 MAX_FRAMES=200 \
  SAVE_EVERY=5000 USE_WANDB=1 \
  WANDB_PROJECT=TRL_X2Ultra_Planner \
  LOG_FILE=\$HOME/plan_train.log \
  bash motionbricks/scripts/cloud/run_planner_train.sh
"
```

Equivalent direct invocation:

```bash
python motionbricks/scripts/train_vqvae_x2.py \
  --pkl gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched.pkl \
  --filter none \
  --max_steps 500000 \
  --batch_size 16 \
  --num_workers 8 \
  --min_frames 80 --max_frames 200 \
  --save-every-n-steps 5000 \
  --devices 8 --num-nodes 1 \
  --use-wandb --wandb-project TRL_X2Ultra_Planner
```

- Final loss: 0.0275; perplexity 8.44
- W&B: [`v32rja0e`](https://wandb.ai/meetsitaram/TRL_X2Ultra_Planner/runs/v32rja0e)

### Stage 2 — Pose (v2 PKL, needs VQVAE ckpt)

```bash
tmux new -d -s plan_train "
  NUM_GPUS=8 \
  PKL=gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2.pkl \
  RUN_VQVAE=0 \
  VQVAE_CKPT=motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt \
  POSE_BATCH_PER_GPU=32 POSE_STEPS=500000 \
  RUN_ROOT=0 \
  NUM_WORKERS=8 MIN_FRAMES=80 MAX_FRAMES=200 \
  SAVE_EVERY=2500 USE_WANDB=1 \
  LOG_FILE=\$HOME/plan_train.log \
  bash motionbricks/scripts/cloud/run_planner_train.sh
"
```

Operational blip: **GPU 1 HBM double-bit error** crashed Pose at
~95K steps. Recovered via 7-GPU resume (`CUDA_VISIBLE_DEVICES=0,2..7`,
`MASTER_PORT=29502`, `RESUME_POSE=…/checkpoints/last.ckpt`), then a
node reboot and 8-GPU resume from the same `last.ckpt`. Three failed /
partial W&B runs (`81vn6xcs`, `zokuuyes`) are retained for audit;
`wt3549mj` is the clean run.

Resume command (documented in the training-history doc):

```bash
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
export NUM_GPUS=7
export MASTER_PORT=29502

RUN_VQVAE=0 RUN_POSE=1 RUN_ROOT=0 \
  PKL=gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2.pkl \
  RESUME_POSE=motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/last.ckpt \
  POSE_STEPS=500000 POSE_BATCH_PER_GPU=32 NUM_WORKERS=8 \
  SAVE_EVERY=2500 USE_WANDB=1 \
  bash motionbricks/scripts/cloud/run_planner_train.sh
```

Final loss: 0.586.

### Stage 3 — Root (v2 PKL, independent of VQVAE)

```bash
tmux new -d -s plan_train "
  NUM_GPUS=8 \
  PKL=gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2.pkl \
  RUN_VQVAE=0 RUN_POSE=0 \
  ROOT_BATCH_PER_GPU=32 ROOT_STEPS=300000 \
  NUM_WORKERS=8 MIN_FRAMES=80 MAX_FRAMES=200 \
  SAVE_EVERY=5000 USE_WANDB=1 \
  LOG_FILE=\$HOME/plan_train.log \
  bash motionbricks/scripts/cloud/run_planner_train.sh
"
```

- Total loss: 1.503; global-root recon: 0.002; top-5 token acc: **84.73%**
- W&B: [`itv2b4jq`](https://wandb.ai/meetsitaram/TRL_X2Ultra_Planner/runs/itv2b4jq)

### Stage 3b — Root FT1 → 315K (deployed Root)

After R2 finished at 300K, closed-loop sim eval on
`Loop_Forward_Walk_001__A018` showed **+46.5° yaw drift** over 30 s.
FT1 is a cheap continuation: filter the corpus to loco clips only,
resume with full state from 300K, cosine-tail for +15K steps.

```bash
RUN_VQVAE=0 RUN_POSE=0 RUN_ROOT=1 \
  PKL=gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2.pkl \
  FILTER=loco \
  RESUME_ROOT=motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0300000.ckpt \
  ROOT_STEPS=315000 ROOT_BATCH_PER_GPU=32 \
  NUM_WORKERS=8 SAVE_EVERY=1000 USE_WANDB=1 \
  bash motionbricks/scripts/cloud/run_planner_train.sh
```

- Yaw drift 46.5° → 34° (**−27%**), but forward distance
  9.29 m → 7.62 m (**−18%**) and lateral drift **+132%**.
- W&B: `root_x2_d8_b32_s315000_resume300k` (id `u1ju8uer`)

FT2 (30K steps from 315K, fresh optimizer) was tried later but did
not clearly beat FT1 on multi-clip eval, so **315K remained the deploy
default**.

---

## 4. Checkpoint outputs

Final artifacts land under `motionbricks/out/motionbricks_<stage>_x2/version_1/checkpoints/`.

| Stage | Step | Size | File | Used by |
|-------|------|------|------|---------|
| VQVAE | **500K** | 273 MB | `motionbricks_vqvae_x2/…/model-step=0500000.ckpt` | `load_x2_planner.py`, `x2_kplanner.py` |
| Pose | **500K** | 1.6 GB | `motionbricks_pose_x2/…/model-step=0500000.ckpt` | same |
| Root R2 base | 300K | 391 MB | `motionbricks_root_x2/…/model-step=0300000.ckpt` | (still available; wrapper default) |
| Root FT1 (**deployed**) | **315K** | 391 MB | `motionbricks_root_x2/…/model-step=0315000.ckpt` | `load_x2_planner.py`, `x2_kplanner.py` |

Pull-back from cloud to laptop is manual `rsync`:

```bash
for stage in vqvae pose root; do
  rsync -avz --partial \
    ubuntu@195.242.31.46:~/GR00T-WholeBodyControl/motionbricks/out/motionbricks_${stage}_x2/ \
    motionbricks/out/motionbricks_${stage}_x2/
done
```

**Deploy defaults confusion note**: `gear_sonic/scripts/x2_kplanner.py`
pins Root to **315K**. `run_x2_quest3_planner_stack.sh` still lists
300K in its own defaults; the daemon's default wins in the live stack
because the wrapper only overrides ckpt paths when passed explicitly.

---

## 5. Round 1 vs Round 2 (why cloud was worth it)

| Aspect | Round 1 (local, May 28 – Jun 2) | Round 2 (Nebius, Jun 24-26) |
|--------|--------------------------------|-----------------------------|
| Hardware | 1× RTX 5090 (~24 GB) | 8× H100 SXM 80 GB |
| Corpus | `x2_ultra_locowalk.pkl` ~18K, `uniform_h14` | 37,968 base / 49,790 v2, `chain_matched` + halfspeed |
| Retargeting | `uniform_h14` (Groucho crouch) | `chain_matched` (pelvis/knee fix) |
| Slow-VR coverage | none | 0.5× locowalk merge for Pose+Root |
| VQVAE steps / batch | ~200K, batch 4 | **500K**, batch 128 |
| Pose steps / batch | ~250K, batch 4 | **500K**, batch 256 |
| Root steps / batch | ~240K, batch 512 (1 GPU) | **300K** (+ FT1 → 315K), batch 256 |
| Wall-clock | ~5 days (intermittent) | ~25.5 hr |
| Cost | electricity | ~$600 |
| Outcome | "barely working" VR walk, stiff | losses healthy; closed-loop still mixed |

**Postmortem headline** (see
`2026-06-26_kplanner_round2_postmortem.md`): even after Round 2 the
training set is still ~53× smaller than the G1 MotionBricks reference,
corpus growth didn't concentrate signal on pure forward walking, Root
loss is dominated by the `num_token` auxiliary head (~98%), and
inference masks out the target body pose (OOD vs training). Those are
the levers on the table if a Round 3 ever runs.

---

## Quick citation block

> **X2 kplanner Round 2** (2026-06-24 – 2026-06-26): Nebius
> `computeinstance-e00cn8h67tdq00t2ys`, 8× H100 SXM 80 GB, bootstrap
> via `bootstrap_planner_node.sh`. Corpus from HF `bones-studio/seed`,
> chain_matched retarget of 37,968 clips, halfspeed merge to
> 49,790-clip v2 PKL. Trained VQVAE 500K (base PKL, batch 128), Pose
> 500K (v2, batch 256), Root 300K (v2, batch 256) via
> `run_planner_train.sh` + PyTorch Lightning DDP, fp32, AdamAtan2.
> W&B: `v32rja0e` / `wt3549mj` / `itv2b4jq`. Deployed: VQVAE 500K +
> Pose 500K + **Root 315K** (FT1 loco fine-tune). Total ~25.5 hr,
> ~$600. Full history: `docs/source/user_guide/kplanner_training_runs.md`.
