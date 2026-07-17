# Kplanner fine-tune on G1→X2 retarget corpus — TOMORROW RUNBOOK

Prepared overnight 2026-07-14/15. Everything below is staged locally and smoke-validated.
Goal: fine-tune the 3 kplanner models (VQVAE, Pose, Root) from their 24h base checkpoints
on the better **G1→X2-retargeted, 30 fps** corpus, on the Nebius 8×H100 node.
W&B project **TRL_X2Ultra_Planner** (same as Round 2, for comparison).

## Why this corpus (the fps saga, resolved)
- The sonic corpus `x2_sonic_executed_feasible.pkl` is **50 fps** (executed rollouts). The kplanner
  pipeline is **30 fps** and does **NO reframing** — and it takes **velocity** as input + encodes
  **4 frames → 1 VQVAE token** (`down_t=2`), so ANY fps error breaks generation. 50→30 is a messy
  non-integer 1.67× resample.
- FIX: rebuild from the **G1 bones-seed source (120 fps)** via the already-existing G1→X2 retarget
  `agibot-x2-references/bones-seed/retargeted/x2_from_g1/` (37,968 CSVs, 4 tiers, EXACT match to
  Round 2's clip set) → clean **120→30 integer 4× decimation**. Better than chain_matched (human→X2).
- Then filtered to the **sonic feasible subset** → **33,206 clips** (aligns kplanner w/ what the
  sonic tracker executes).

## Artifacts already built (local)
- **Corpus:** `gear_sonic/data/motions/x2_ultra_bones_seed_g1_retarget_feasible.pkl`
  (33,206 clips @ 30 fps, median 186f, [80,200]=57.4%, 2.36 GB). Matches Round2 distribution (184f/57.8%).
  (Also full unfiltered `x2_ultra_bones_seed_g1_retarget.pkl` = 37,968, + per-tier PKLs.)
- **Fine-tune result dir:** `motionbricks/out_g1ft/motionbricks_{vqvae,pose,root}_x2/version_1/`
  with base `hparams.yaml + skeleton/ + stats/` COPIED in (reuse base normalization → consistent
  with warm-start). Full **feature cache** building here now (bg task, ~1-2h; check
  `out_g1ft/.../feature_cache/x2_ultra_bones_seed_g1_retarget_feasible/`).
- **Code:** `--init-from` (weights-only warm-start) ADDED to `train_vqvae_x2.py` + `train_pose_x2.py`
  (mirrors Root's existing flag). All 3 parse OK. **These edits are NOT on the node** (node cloned
  main) → must rsync tomorrow.
- **Smoke-validated** end-to-end on 300-clip subset, 1 GPU: all 3 load base weights via --init-from
  (clean, no key mismatch), Pose consumes VQVAE ckpt, all train + save to out_g1ft. No hurdles.

## TOMORROW — steps
### 0. Resume node + get new IP
Nebius instance `computeinstance-e00ckg14vp99z0hgnm` (project `project-e00yn4n7pr00m8t2x6cdqa`).
IP was 195.242.13.202 but **CHANGES on resume** — get new IP from console or
`nebius compute instance get --id computeinstance-e00ckg14vp99z0hgnm | grep -A1 public_ip`.
`ssh ubuntu@<newIP>`. Bootstrap persists on the boot disk (conda motionbricks, repo, LFS).

### 1. rsync to node (from laptop repo root)
```
IP=<newIP>; R=/home/ubuntu/GR00T-WholeBodyControl
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=10"
# corpus
rsync -aP -e "$SSH" gear_sonic/data/motions/x2_ultra_bones_seed_g1_retarget_feasible.pkl ubuntu@$IP:$R/gear_sonic/data/motions/
# FT dir (base stats + full feature cache) -- BIG, the cache is ~few GB
rsync -aP -e "$SSH" motionbricks/out_g1ft/ ubuntu@$IP:$R/motionbricks/out_g1ft/
# base checkpoints (for --init-from) -- ~2.3GB
rsync -aP -e "$SSH" motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0500000.ckpt \
  motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/model-step=0500000.ckpt \
  motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0315000.ckpt \
  ubuntu@$IP:$R/motionbricks/out/  # (into matching subpaths -- use -R or per-file)
# edited trainer scripts (--init-from) -- node cloned main WITHOUT these
rsync -aP -e "$SSH" motionbricks/scripts/train_vqvae_x2.py motionbricks/scripts/train_pose_x2.py \
  ubuntu@$IP:$R/motionbricks/scripts/
```
(Clean the smoke checkpoints first: `rm out_g1ft/motionbricks_*_x2/version_1/checkpoints/*.ckpt` locally
before rsync, OR they'll be overwritten by the real run anyway — --init-from ignores them.)

### 2. Symlink the feature cache for Pose + Root (model-agnostic features; saves ~1-2h each)
On node, after rsync:
```
cd $R; CACHE=x2_ultra_bones_seed_g1_retarget_feasible
for m in pose root; do
  mkdir -p motionbricks/out_g1ft/motionbricks_${m}_x2/version_1/feature_cache
  ln -sfn ../../../motionbricks_vqvae_x2/version_1/feature_cache/$CACHE \
     motionbricks/out_g1ft/motionbricks_${m}_x2/version_1/feature_cache/$CACHE
done
```

### 3. Launch (in tmux, motionbricks env; call scripts DIRECTLY — wrapper lacks --init-from)
```
conda activate motionbricks; cd $R
PKL=gear_sonic/data/motions/x2_ultra_bones_seed_g1_retarget_feasible.pkl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
**VQVAE (250K, batch 128 = 16×8):**
```
python motionbricks/scripts/train_vqvae_x2.py --result_dir motionbricks/out_g1ft --pkl $PKL \
  --init-from motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0500000.ckpt \
  --max_steps 250000 --batch_size 16 --num_workers 8 --devices 8 --num-nodes 1 \
  --min_frames 80 --max_frames 200 --save-every-n-steps 5000 \
  --use-wandb --wandb-project TRL_X2Ultra_Planner --wandb-name vqvae_g1ret_250k
```
**Pose (250K, batch 256 = 32×8; AFTER VQVAE done, point --vqvae-ckpt at the NEW vqvae last.ckpt):**
```
python motionbricks/scripts/train_pose_x2.py --result_dir motionbricks/out_g1ft --pkl $PKL \
  --init-from motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/model-step=0500000.ckpt \
  --vqvae-ckpt motionbricks/out_g1ft/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt \
  --max_steps 250000 --batch_size 32 --num_workers 8 --devices 8 --num-nodes 1 \
  --min_frames 80 --max_frames 200 --save-every-n-steps 2500 \
  --use-wandb --wandb-project TRL_X2Ultra_Planner --wandb-name pose_g1ret_250k
```
**Root (250K, batch 256; independent of VQVAE — can run right after VQVAE or interleave):**
```
python motionbricks/scripts/train_root_x2.py --result_dir motionbricks/out_g1ft --pkl $PKL \
  --init-from motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0315000.ckpt \
  --max_steps 250000 --batch_size 32 --num_workers 8 --devices 8 --num-nodes 1 \
  --min_frames 80 --max_frames 200 --save-every-n-steps 5000 \
  --use-wandb --wandb-project TRL_X2Ultra_Planner --wandb-name root_g1ret_250k
```
Sequential VQVAE→Pose→Root on the 8 GPUs, ~250K each ≈ ~3.5-4h each ≈ **~11-13h total**.

## WATCH / gotchas
- `--init-from` under 8-GPU DDP: validated only on 1 GPU in smoke. Each rank runs the load_state_dict
  in main() before fit — should be consistent, but eyeball the first VQVAE launch for "loaded" on rank 0
  + no state_dict errors.
- Pose prints "Found N modules in eval mode" = the frozen VQVAE sub-net. Benign.
- Pull-back after: `rsync ubuntu@$IP:$R/motionbricks/out_g1ft/motionbricks_{vqvae,pose,root}_x2/` → local.
- Sonic v3 run is on the OTHER node (unaffected).
