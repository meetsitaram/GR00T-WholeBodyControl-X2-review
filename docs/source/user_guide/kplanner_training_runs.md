# X2 kplanner — Training Runs History

A running log of kinematic-planner training rounds (VQVAE + Pose + Root) and
their corpus / hardware / wall-clock / outcome metadata. Each entry captures
one *round* end-to-end so we don't lose institutional knowledge between
multi-month gaps in training cadence.

> **Companion docs.**
> - [`train-planner-on-cloud.md`](train-planner-on-cloud.md) — procedural
>   how-to (bootstrap, env vars, launcher mechanics). Stays generic.
> - [`x2_kplanner.md`](../references/x2_kplanner.md) — runtime / inference
>   architecture (how the trained checkpoints get consumed by
>   `x2_kplanner.py` on the robot).
> - This doc — what we actually ran, how long it took, what broke, and
>   what we learned. Append a new section every time a fresh training
>   round lands.

---

## Quick reference — all rounds at a glance

| Round | Date           | Hardware              | Corpus                                        | Clips       | Total wall-clock | Outcome                                  |
|-------|----------------|-----------------------|-----------------------------------------------|-------------|------------------|------------------------------------------|
| 1     | 2026-05-28 → 06-02 | Local 1× RTX 5090 (~24 GB) | `x2_ultra_locowalk.pkl` (legacy retarget)     | ~18,000     | ~5 days (off-and-on) | "Barely working" — robot walks under VR, but stiff, noisy, narrow velocity range. Reason: locowalk-only corpus, single GPU, undertrained VQVAE + Pose. |
| 2     | 2026-06-24 → 06-26 | Nebius 8× H100 SXM (cloud, ~1.5 TB RAM) | `x2_ultra_bones_seed_chain_matched.pkl` (VQVAE) + `_v2` w/ 0.5× locowalk merge (Pose+Root) | 37,968 base / 49,790 v2 | ~25.5 hr (VQVAE 6h50m, Pose ~13.5 hr incl. GPU 1 fault recovery, Root 4h40m) | All 3 stages converged on paper (VQVAE 0.028, Pose 0.586, Root 1.50 / top-5 token 85 %). **Closed-loop sim still bad**: forward walk yaw drift +46.5°/30 s, slow / sideways / back gaits worse than Round 1 in places. See [Round 2 postmortem](#round-2-postmortem-why-25-hours-of-h100s-still-did-not-give-a-smooth-vr-walk). |
| 2.1   | 2026-06-26     | Same Nebius node, 8× H100 | Same v2 PKL, `loco` filter (locowalk + locopost + locobal subset only) | ~14,400 (filter on 49,790) | ~50 min (15K steps from 300K) | **FT1**: cosine-tail continuation from R2 step 300K. Yaw bias 46.5° → 34° (−27 %) but fwd throughput regressed 9.29 m → 7.62 m (−18 %) and lateral drift grew 0.28 m → 0.65 m. Trade, not a win. Cosine schedule was bottomed-out at LR ~2e-6 by step 300K — couldn't be pushed further on the same recipe. |
| 2.2   | 2026-06-26     | Same Nebius node, 8× H100 | Same v2 PKL, `loco` filter | ~14,400 | ~28 min (30K steps, weights-only resume from FT1 315K) | **FT2**: weights-only init + fresh optimizer + fresh cosine (peak LR 1e-5, warmup 1K, final 5e-7). Open-loop multi-clip eval was too noisy to pick a single "best" step. Final 30K checkpoint pulled locally; A/B vs 300K/315K not yet run on hardware. **Net effect on closed-loop sim still likely small** given the inference-mode mismatch documented in the postmortem. |

---

## Round 2 — 2026-06-24: full chain_matched corpus on 8× H100

> **Status:** in progress. Skeleton + stats done; feature cache building; VQVAE
> + Pose + Root training queued. Updates will be appended below as stages
> complete.

### Why this round

Round 1 had three known problems we explicitly addressed here:

1. **Single-GPU undertraining.** Round 1's VQVAE saw ~200 K steps on a 5090
   with `batch_size=4`. On 8 H100s with the same per-GPU batch the effective
   batch is 8× larger, so 500 K steps should give materially better
   convergence in 1/3 the wall-clock.
2. **Corpus too narrow.** Round 1 used only `locowalk` (~18 K clips). Round 2
   uses the full curated BONES-SEED corpus (37,968 clips: locowalk +
   locopost + locomanip + locobal). The VQVAE codebook now sees posture,
   manipulation, and balance motions during training.
3. **Stiff / jumpy VR control at low velocity.** Round 1 motions were all at
   natural human velocity; VR thumbsticks at 50% deflection wanted poses at
   half-speed that the VQVAE had never seen. Round 2 adds **0.5× playback
   variants of locowalk clips** (with a `--min-mean-speed 0.20 m/s` /
   `--min-mean-yaw-rate 0.15 rad/s` OR-filter so stationary clips don't
   pollute the slow-walk distribution). The result is the `_v2` PKL with
   11,822 `__speed_0.5` clips appended to the base 37,968 (49,790 total)
   used for Pose + Root training. **VQVAE trains on the base PKL only** —
   tokenizing the same poses twice (at 1× and 0.5×) would inflate codebook
   pressure without adding new pose content.

### Retargeting upgrade

Switched the entire 37,968-clip corpus from legacy (`uniform_h14`)
retargeting to **`chain_matched`** to fix the "Groucho-Marx deep-crouch" bug
(pelvis Z ~0.51 m, knees 70°). Chain-matched config (committed separately to
`soma-retargeter` as `soma_to_x2_ultra_chain_matched_retargeter_config.json`)
lifts pelvis Z to 0.63-0.68 m at idle and knee flex stays ≤23°. This is a
strictly better signal for the planner — same number of clips, more
physically plausible.

### Hardware & cost

| Item                     | Value                                                                                |
|--------------------------|--------------------------------------------------------------------------------------|
| Cloud node               | Nebius `computeinstance-e00cn8h67tdq00t2ys`, 8× H100 SXM, 128 vCPU, 1.5 TB RAM, 1.3 TB disk |
| Public IP                | `195.242.31.46` (ssh `ubuntu@...`)                                                   |
| Provisioned              | 2026-06-24                                                                           |
| List price (Nebius H100) | ~$2.50-3.00 / GPU·hr → ~$20-24 / node·hr                                             |
| Expected total           | ~25-30 hr (VQVAE 7h + Pose 14-17h + Root 4-5h) × ~$22/hr → **~$600**                |
| W&B project              | `TRL_X2Ultra_Planner` (user `meetsitaram`)                                           |

### Stage-by-stage wall-clock (cloud)

> Update each row as it completes. Times are wall-clock on this node unless noted.

| #  | Stage                                                        | Started (UTC)   | Finished (UTC)  | Wall-clock | Notes                                                                                       |
|----|--------------------------------------------------------------|-----------------|-----------------|------------|---------------------------------------------------------------------------------------------|
| 0a | Cloud node bootstrap (conda, uv, repo clone, motionbricks env) | 2026-06-24 16:30 | 2026-06-24 17:37 | ~1h 07m   | Done in parallel with HF dataset download. `bootstrap_planner_node.sh` ran clean.           |
| 0b | HF dataset download (`bones-studio/seed` soma_uniform.tar.gz, ~42 GB) | 2026-06-24 16:30 | 2026-06-24 17:25 | ~55 min    | In parallel `tmux` with bootstrap.                                                          |
| 0c | BVH extraction + curation (37,968 clips into 4 tiers)         | 2026-06-24 17:30 | 2026-06-24 17:55 | ~25 min    | Single-process tar + `curate_x2_planner.py`.                                                |
| 1  | **chain_matched retarget** (BVH → CSV)                       | 2026-06-24 18:30 | 2026-06-24 19:16 | **~46 min** | 8× H100, 3 shards/GPU = 24 shards in parallel. Patched `retarget_x2_parallel.py` for `CUDA_VISIBLE_DEVICES` round-robin pinning. All 37,968 CSVs (~16 GB). |
| 2  | Build motion-lib PKLs (base + per-tier + halfspeed + v2)     | 2026-06-24 19:50 | 2026-06-24 19:59 | ~9 min     | `build_x2_bones_seed_motion_lib.py --out-suffix _chain_matched` + halfspeed pass + merge.   |
| 3  | **Skeleton + stats + hparams** (`build_x2_skeleton_assets.py --max-clips-stats 0`) | 2026-06-24 20:59 | 2026-06-24 22:41 | **~102 min** | Single-process, 770% CPU via torch/MKL threading. dim=418, frames=6,432,530, clips=34,232 (3,736 dropped by `[60,400]` frame filter). |
| 4  | Feature cache (base PKL, VQVAE input)                        | 2026-06-24 22:59 | 2026-06-24 23:01 | **~2 min** (after thread-cap fix) | 21 workers × 6 OMP threads. Output: 31,329 / 37,968 clips cached (6,639 out_of_band by `[80, 300]` frame filter), 8.7 GB. See issue 8 below — first attempt thrashed for 15 min with 0 progress before thread caps were applied. |
| 5  | Feature cache (v2 PKL, Pose+Root input)                      | 2026-06-24 23:01 |                 |            | 20 workers × 6 OMP threads (auto-scaled from 21 by memory budget). v2 PKL is 3.8 GB → est. ~5,500 new .pt files for halfspeed keys (base PKL files reused via `ok_cached`). |
| 6  | **VQVAE training** (8 GPU DDP, 500K steps)                   | 2026-06-24 23:18 | 2026-06-25 06:08 | **~6h 50m** | `batch_per_gpu=16` → effective batch 128. ~20 steps/sec sustained. Loss 0.82 → 0.024 (-97%), pose recon 0.62 → 0.011, perplexity stable at 8.4 (codebook healthy). W&B `jnft5d6l`. |
| 7  | **Pose training** (8 GPU DDP, 500K steps, needs VQVAE ckpt)  | 2026-06-25 06:20 |                 |            | `batch_per_gpu=32` → effective batch 256. `NUM_WORKERS=8`. Cache symlinked from VQVAE's cache (see issue 9 below). GPU util 55-76% (vs VQVAE's 38-55%) — bigger batch fills H100s better. |
| 8  | **Root training** (8 GPU DDP, 300K steps, independent)       |                 |                 |            | Sequential after Pose. `batch_per_gpu=32`, `NUM_WORKERS=8`. Cache also symlinked (proactive).                                                              |

### Corpus statistics

After PKL build + halfspeed filtering, before the `[60, 400]` frame filter that
the skeleton-build applies:

| PKL                                                  | Clips  | Halfspeed | locowalk | locopost | locomanip | locobal |
|------------------------------------------------------|--------|-----------|----------|----------|-----------|---------|
| `x2_ultra_bones_seed_chain_matched.pkl` (base)       | 37,968 | 0         | 18,036   | 8,752    | 9,712     | 1,468   |
| `x2_ultra_bones_seed_chain_matched_halfspeed.pkl`    | 11,822 | locowalk  | 11,822   | —        | —         | —       |
| `x2_ultra_bones_seed_chain_matched_v2.pkl` (merged)  | 49,790 | 11,822    | 29,858   | 8,752    | 9,712     | 1,468   |

Halfspeed filter retained 11,822 / 18,036 = 65 % of locowalk clips (the
remaining 35 % were ≤ 0.20 m/s mean linear speed AND ≤ 0.15 rad/s net yaw
rate — i.e. effectively stationary clips that contribute no slow-velocity
signal).

After the skeleton-build's `[60, 400]` frame filter on the base PKL: **34,232
clips × 6,432,530 frames** used for stats. The frame-count filter drops
~3,736 short or pathologically-long clips.

### Issues encountered (and how they were resolved)

1. **`pull_chain_matched_csvs.sh` watcher syntax error** — parsing `done_flag`
   from `grep -c` left whitespace that broke an `if` test. Stripped whitespace
   in the parser. Local CSV pull then ran clean.

2. **`uv: command not found` in non-interactive shells** — uv installs into
   `~/.local/bin/` which isn't in the default `PATH` for `bash -c` invocations.
   Prepended `export PATH="$HOME/.local/bin:$PATH"` before every cloud-side
   `uv run` call (including inside `tmux` command strings). Fixed.

3. **`ModuleNotFoundError: No module named 'pandas'`** in cloud PKL build —
   `motionbricks` conda env was missing pandas. `pip install pandas` in the
   `motionbricks` env. (Should be added to `bootstrap_planner_node.sh`.)

4. **Parallel retarget loading all shards onto `cuda:0`** — default GPU
   placement put every Python process on GPU 0 → 8× memory pressure and ≤1
   GPU's worth of throughput. Patched `retarget_x2_parallel.py` to set
   `CUDA_VISIBLE_DEVICES=<global_shard_idx % NUM_GPUS>` per shard. After fix:
   even load across all 8 H100s, 46 min for the full 37,968-clip corpus.

5. **`FileNotFoundError: Missing hparams.yaml`** during early feature-cache
   attempts — feature cache needs `hparams.yaml` produced by
   `build_x2_skeleton_assets.py`, but the cloud-side training launcher we'd
   set up went straight to cache. Reordered the post-retarget pipeline so
   skeleton-build runs before cache. Captured this dependency in the docs.

6. **Skeleton-build wall-clock surprise — doc said "~30 s", reality was 102
   min** on the full 37,968-clip corpus with `--max-clips-stats 0`. The
   docs claim assumed the smoke-default of ~50 clips. Updated
   `train-planner-on-cloud.md §5` with the actual cost breakdown by corpus
   size. See follow-up plan:
   [`.cursor/plans/incremental_kplanner_stats_cache_f43b9bab.plan.md`](../../../.cursor/plans/incremental_kplanner_stats_cache_f43b9bab.plan.md)
   to refactor stats into per-clip aggregates so adding even one clip
   doesn't trigger a full 100-min recompute.

7. **Duplicate feature-cache builders racing on the same PKL** — when the
   skeleton-build task finished, it auto-launched its own `plan_train` tmux
   session running the full downstream pipeline. A separate `auto_train`
   watcher I'd set up at the same time fired independently. Both started
   `build_feature_cache_x2.py` with overlapping worker pools writing into
   the same `manifest.json`. Killed the duplicate (`auto_train`), kept
   `plan_train` since it had more workers and started 30 s earlier. **Lesson
   for future rounds:** only ever set up *one* downstream auto-launcher per
   pipeline trigger.

8. **Catastrophic thread oversubscription in `build_feature_cache_x2.py` on
   the 128-core node** — first feature-cache launch ran for 15 min with **zero
   completed clips** despite all 21 workers at 500-700 % CPU. Diagnosis via
   `py-spy` showed workers deep in `dual_rep`/`compute_motion_features` (real
   work), but `/proc/<pid>/task | wc -l` revealed each worker had spawned **64
   threads** (torch's default = `nproc` = 128 / 2 hyperthreading-aware).
   21 workers × 64 threads = **1,344 threads competing for 128 cores**.
   `vmstat` confirmed 140 K context switches/sec → kernel-scheduler thrashing.

   **Fix:** restart with explicit thread caps in the pipeline env:

   ```bash
   export OMP_NUM_THREADS=6
   export MKL_NUM_THREADS=6
   export OPENBLAS_NUM_THREADS=6
   export NUMEXPR_NUM_THREADS=6
   ```

   21 workers × 6 threads = 126 threads ≈ 128 cores → clean 1:1 mapping.
   `cs/sec` dropped 140K → 13K (10×), load avg 256 → 50, and **the next 21K
   clips finished in 2 min instead of 0 in 15 min** — ~150× per-clip
   throughput jump.

   The skeleton-assets step (single-process, 64 threads on 128 cores) didn't
   show this because there was no inter-process scheduler contention. The
   bug only manifests with multi-process pools on high-core-count cloud
   nodes. The local 5090 box (32 cores) wouldn't trip it either.

   **Action item:** add `export OMP_NUM_THREADS=<num_threads_per_worker>` to
   `motionbricks/scripts/cloud/run_planner_train.sh` and to the bundle
   bootstrap. See follow-up plan
   [`.cursor/plans/incremental_kplanner_stats_cache_f43b9bab.plan.md`](../../../.cursor/plans/incremental_kplanner_stats_cache_f43b9bab.plan.md)
   — same root cause applies to a future per-clip parallel stats build.

9. **Per-model feature-cache duplication (Pose + Root each rebuild from
   scratch)** — when VQVAE finished and Pose started, Pose began building
   *its own* feature cache from the v2 PKL inside
   `motionbricks/out/motionbricks_pose_x2/version_1/feature_cache/`, even
   though a bit-identical cache already existed at
   `motionbricks/out/motionbricks_vqvae_x2/version_1/feature_cache/`. Root
   would have done the same thing afterward. The data-loader looks up its
   cache via `default_cache_dir_for(version_dir, pkls)` where `version_dir`
   is derived from the *training script's* output directory — so each of
   the three stages (VQVAE / Pose / Root) has its own cache directory by
   default, and the cache content is identical (same skeleton, same stats,
   same PKL, same `[80, 300]` frame filter).

   **Symptoms.** Pose process at 152 % CPU but 0 % GPU util for 10+ min
   after launch. `strace` showed continuous `openat(... .pt, O_WRONLY |
   O_CREAT | O_TRUNC)` calls into the Pose `feature_cache` dir. The
   `manifest.json` was missing from the Pose dir → cache loader concluded
   "no cache exists, must rebuild." Single-process build rate was ~15
   .pt/sec, so the 35,654-clip v2 cache would take **~30-40 min** before
   training could start. Root would burn the same time again.

   **Fix (manual symlink, no code change yet).** Kill the in-progress
   Pose, symlink the already-built VQVAE cache into the Pose + Root cache
   locations, restart Pose:

   ```bash
   VQVAE_CACHE=~/GR00T-WholeBodyControl/motionbricks/out/motionbricks_vqvae_x2/version_1/feature_cache/x2_ultra_bones_seed_chain_matched_v2
   for STAGE in pose root; do
       STAGE_CACHE_DIR=~/GR00T-WholeBodyControl/motionbricks/out/motionbricks_${STAGE}_x2/version_1/feature_cache
       mkdir -p "$STAGE_CACHE_DIR"
       rm -rf "$STAGE_CACHE_DIR/x2_ultra_bones_seed_chain_matched_v2"
       ln -sfn "$VQVAE_CACHE" "$STAGE_CACHE_DIR/x2_ultra_bones_seed_chain_matched_v2"
   done
   ```

   Verified by `md5sum` that VQVAE-built and Pose-built `.pt` files were
   bit-identical, so the symlink is safe. After the symlink, Pose loaded
   the cache instantly (`cache_dir: ... (HIT)`, 35,654 clips) and DDP
   training started within ~30 sec instead of 30 min — **~30 min saved
   per downstream stage, ~60 min total for this round.**

   **Action item / follow-up.** The proper fix is to make the cache
   location independent of the calling training script — store it under
   a shared `motionbricks/out/feature_cache/<pkl_name>/` (or even just
   under the PKL directory itself) and have all three trainers look there
   first. Captured in the same caching follow-up plan:
   [`.cursor/plans/incremental_kplanner_stats_cache_f43b9bab.plan.md`](../../../.cursor/plans/incremental_kplanner_stats_cache_f43b9bab.plan.md)
   — extend the per-clip stats refactor with a "shared feature cache root"
   section so VQVAE / Pose / Root all populate and consume the same
   directory. Until then, keep the symlink workaround in the cloud
   dispatcher for any sequential training run.

10. **GPU 1 HBM3 hardware fault — Pose crash at step 95K** — overnight, the
    Pose training process died at step 95,000 (~6 h 40 min into the 8-GPU
    run) with:

    ```
    torch.AcceleratorError: CUDA error: Invalid access of peer GPU memory
    over nvlink or a hardware error
    ```

    `dmesg` showed no OOM, but Nebius's per-node Xid stream (visible in
    the cloud console) revealed the actual cause on PCI `0000:91:00`
    (= GPU 1) at 01:04 UTC:

    | Xid | Meaning                                                                   |
    |-----|----------------------------------------------------------------------------|
    | 48  | Uncorrectable double-bit ECC (DBE) in framebuffer at `physAddr 0x10e0ed320` |
    | 171 | Uncorrectable DRAM error in HBM, FBPA 16 subpartition 3 (same address)     |
    | 63  | Row Remapper marked the row for retirement (needs GPU reset to activate)   |
    | 94  | "Contained" SM fault propagated to PyTorch CUDA channels 0x0a–0x11         |
    | 154 | Driver issued `Drain and Reset` on GPU 1                                   |

    Sequence: GPU 1 hit a hard HBM bit failure at 01:04 UTC; the driver
    self-reset the GPU and row-remapped the bad address; training came
    back up automatically; **6 h 57 min later (at 08:01 UTC) the same GPU
    failed again** during the DDP backward pass — the row-remap
    band-aid didn't survive sustained heavy load.

    **Symptoms.** `nvidia-smi` showed GPU 1 with `ECC=1` (one
    uncorrectable error logged) even after the GPU "looked healthy"
    (idle, 0 MiB allocated, NVLink at 26.5 GB/s on every link). The
    persistent ECC counter is the give-away — without a node reboot
    that counter doesn't clear, and on H100 it implies a physical
    cell failure rather than a transient.

    **Diagnosis (key finding: the remap was pending, not applied).**
    Inspect the row-remapper state to confirm the diagnosis:

    ```bash
    nvidia-smi -i 1 -q -d ROW_REMAPPER | \
      grep -E "Uncorrectable Error|Pending|Remapping Failure"
    # Expected on the bad GPU after a fault:
    #   Uncorrectable Error : 1
    #   Pending             : Yes        ← remap requested but not committed
    #   Remapping Failure   : No
    ```

    `Pending: Yes` is the smoking gun. The 01:04 UTC `Xid 154` "Drain
    and Reset" was a *software* reset — it re-initialised CUDA contexts
    but did **not** commit the row remap to the GPU's onboard EEPROM.
    The bad row was still live, which is exactly why the GPU failed
    again 7 hr later under sustained load.

    **Fix A — immediate fallback (recommended if you're mid-deadline).**
    Drop the bad GPU and resume on 7 GPUs:

    ```bash
    export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
    export NUM_GPUS=7
    export MASTER_PORT=29502   # avoid stale PG state from dead 8-GPU run

    RUN_VQVAE=0 RUN_POSE=1 RUN_ROOT=0 \
      RESUME_POSE=motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/last.ckpt \
      POSE_STEPS=500000 POSE_BATCH_PER_GPU=32 NUM_WORKERS=8 \
      SAVE_EVERY=2500 USE_WANDB=1 \
      bash motionbricks/scripts/cloud/run_planner_train.sh
    ```

    Effective batch drops from 256 → 224 (negligible quality impact),
    step rate ~13/s vs ~15/s (~12 % slower), total wall-clock cost ~+1
    hr.

    **Fix B — commit the pending remap (only works via full node reboot).**
    On H100 SXM nodes with NVSwitch fabric (i.e., every Nebius 8× H100
    box), per-GPU reset is **not supported** because the NVSwitch
    permanently locks each GPU to the fabric topology. We verified this
    the hard way:

    ```bash
    # All of these fail with "GPU 00000000:91:00.0: In use by another client"
    sudo nvidia-smi -i 1 -r
    sudo systemctl stop nvidia-fabricmanager nvidia-dcgm nvidia-persistenced
    sudo nvidia-smi -i 1 -r   # still blocked — NVSwitch holds GPU 1
    ```

    The only path that activates a pending remap on these nodes is a
    full node reboot. The remap is then applied during driver init on
    next boot and persists in EEPROM forever:

    ```bash
    sudo reboot                              # ~90 sec on Nebius H100 box
    # ...after reconnect...
    nvidia-smi -i 1 -q -d ROW_REMAPPER | grep -E "Pending|Remapping Failure"
    #   Pending             : No            ← committed!
    #   Remapping Failure   : No
    ```

    Cost: ~10 min total disruption (~90 sec reboot + DDP relaunch +
    feature-cache warmup). On a 14-hr Pose run that saves ~100 min by
    going from 7 GPUs back to 8 GPUs — clearly worth it if you have
    >1 hr of training left. **Always re-check `Pending`/`Remapping
    Failure Occurred` after the reboot before re-introducing the GPU**;
    if `Remapping Failure Occurred: Yes`, the chip is dead — fall back
    to Fix A and ticket Nebius.

    Also **lowered `SAVE_EVERY` from 5000 → 2500** going forward so any
    future crash loses ≤ 2.5 min of training, not 5 min.

    **Action items.**
    - Open a Nebius support ticket against this instance to flag GPU 1
      for replacement at end-of-task. Even with a successful remap,
      the chip has shown one DBE and could fail again in another row.
    - Add `SAVE_EVERY=2500` as the default in
      `motionbricks/scripts/cloud/run_planner_train.sh` for any multi-hour
      training; 2.5K-step ckpts are ~1.6 GB each, still cheap.
    - For future cloud rounds, **pre-flight every fresh node** with:

      ```bash
      nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total \
        --format=csv
      for i in $(seq 0 7); do
        nvidia-smi -i $i -q -d ROW_REMAPPER | \
          grep -E "Pending|Remapping Failure" | \
          sed "s/^/  gpu $i: /"
      done
      ```

      If any GPU shows `Pending: Yes` or `Remapping Failure: Yes` at
      provisioning time, reboot the node *before* starting training
      (or ticket Nebius for replacement). Cheap insurance against
      7-hour-in crashes.

### Round-1 → Round-2 design changes (one-glance)

| Aspect                       | Round 1                                  | Round 2                                                |
|------------------------------|------------------------------------------|--------------------------------------------------------|
| Compute                      | 1× RTX 5090 (~24 GB)                     | 8× H100 SXM 80 GB                                      |
| VQVAE batch (per GPU × DDP world) | 4 × 1 = 4                            | 16 × 8 = **128**  (32× larger effective batch)         |
| Pose batch (per GPU × DDP world)  | 4 × 1 = 4                            | 32 × 8 = **256**  (64× larger effective batch)         |
| Root batch (per GPU × DDP world)  | 512 × 1 = 512                        | 32 × 8 = **256**  (same order; smaller per-GPU but DDP-averaged) |
| DataLoader workers           | 8 (single-rank)                          | 8 per rank × 8 ranks = 64 (NUM_WORKERS=8)              |
| Precision                    | fp32                                     | fp32 (bf16 considered, deferred for stability)         |
| VQVAE steps                  | ~200 K                                   | **500 K**                                              |
| Pose steps                   | ~250 K (resumed)                          | **500 K** from scratch (Pose is the biggest model + most undertrained in round 1) |
| Root steps                   | ~240 K (resumed from 100 K, then +100 K) | **300 K** from scratch (round-1 was already converged at 240 K, modest bump) |
| Corpus                       | `x2_ultra_locowalk.pkl` only             | full curated chain_matched corpus (locowalk + post + manip + bal) |
| Retargeting method           | legacy `uniform_h14`                     | `chain_matched` (pelvis-Z + knee-flex correct)         |
| Slow-velocity coverage       | none (natural human speeds only)         | 0.5× locowalk variants merged for Pose + Root          |
| W&B                          | enabled                                  | enabled, project `TRL_X2Ultra_Planner`                 |
| Skeleton-stats compute       | ~30 s (locowalk-only corpus)             | ~102 min (full corpus, `--max-clips-stats 0`)          |
| Feature cache (per stage)    | ~15 min, locowalk only                   | ~2 min (after thread-cap fix); Pose + Root reuse VQVAE cache via symlink (see issue 9) |

### Local artifact archival (for the next round)

The skeleton/stats/hparams from this round are **gitignored but reusable**:
~10 KB total, valid for any future round that stays on chain_matched
retargeting + the same X2 MJCF.
[`gear_sonic/scripts/cloud/pull_motionbricks_assets.sh`](../../../gear_sonic/scripts/cloud/pull_motionbricks_assets.sh)
rsyncs them back to local under
`motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/`. Watcher ran
automatically when `hparams.yaml` landed on cloud (15:43 PT on 2026-06-24).

When can these be reused? See the table in
[`train-planner-on-cloud.md §5`](train-planner-on-cloud.md). TL;DR: any
resume / fine-tune / corpus-tweak on the same retarget config + same MJCF
is free; switching retargeting or skeleton forces a rebuild.

### Outcome

All three stages completed successfully. Round 2 finished 2026-06-26 00:48
UTC — total wall-clock from VQVAE start to Root finish was ~25 hr 30 min
(within the 25–30 hr cost estimate).

**Final loss / accuracy** (from W&B summary at last step):

| Stage  | Final step | Final train loss | Notable secondary metrics                                                                                                                                 |
|--------|------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| VQVAE  | 500,000    | **0.0275**       | pose recon (local) 0.011, joint vel 0.004, skate contact 0.026, codebook perplexity **8.44** (stable throughout, no codebook collapse)                    |
| Pose   | 500,000    | **0.586**        | started resume @ 0.816 → 0.586 = ~28 % loss reduction over the 400 K post-resume steps; smooth monotone-decreasing curve through 500 K                    |
| Root   | 300,000    | **1.503** total  | global-root recon **0.002** (excellent continuous-trajectory accuracy), local-root recon 0.025, token top-1 **42.65 %**, top-3 **72.90 %**, top-5 **84.73 %** |

**W&B run IDs (Round-2 final, finished/successful):**

- VQVAE: `vqvae_x2_d8_b16_s500000` — [`v32rja0e`](https://wandb.ai/meetsitaram/TRL_X2Ultra_Planner/runs/v32rja0e)
- Pose: `pose_x2_d8_b32_s500000` (post-reboot resume) — [`wt3549mj`](https://wandb.ai/meetsitaram/TRL_X2Ultra_Planner/runs/wt3549mj)
- Root: `root_x2_d8_b32_s300000` — [`itv2b4jq`](https://wandb.ai/meetsitaram/TRL_X2Ultra_Planner/runs/itv2b4jq)

(Two earlier Pose runs ended with `state=crashed`: `81vn6xcs` is the
initial 8-GPU launch that hit the GPU 1 HBM DBE at step 95K; `zokuuyes`
is the 7-GPU resume that was killed manually for the node reboot. Both
contributed steps 0 → 97,500 of the Pose curve and are kept on W&B for
the audit trail.)

**Final checkpoint paths.** All three are on local with md5-verified
integrity (rsync'd from Nebius after Root finished):

```
motionbricks/out/
├── motionbricks_vqvae_x2/version_1/checkpoints/model-step=0500000.ckpt   273 MB   md5 1a02347c83e3b9a13bf61b85b21dfd06
├── motionbricks_pose_x2/version_1/checkpoints/model-step=0500000.ckpt    1.6 GB   md5 1257d76f8f209790b7f0b63dd070ff94
└── motionbricks_root_x2/version_1/checkpoints/model-step=0300000.ckpt    391 MB   md5 acef34db836cc54f12e164204789f208
```

On cloud the same three files live at `~/GR00T-WholeBodyControl/motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/checkpoints/model-step=0{500000,500000,300000}.ckpt`.
Total artifact size: ~2.3 GB.

**Stability through the run.** Despite the GPU 1 HBM3 DBE that crashed
Pose at step 95K (see Issue #10), the post-reboot relaunch ran for
~10 hr 40 min through Pose 97.5K → 500K and the full Root 0 → 300K
without a single new Xid on any GPU. Row remap held.

**Subjective VR-driving feel vs Round 1.** Mixed-to-bad. Detailed in the
[Round 2 postmortem](#round-2-postmortem-why-25-hours-of-h100s-still-did-not-give-a-smooth-vr-walk).
Headline: forward walk drifts +46.5° yaw over 30 s of straight-forward
intent; sideways / backward gaits feel worse than Round 1 in several
regimes; turns are more aggressive than R1 (robot nearly fell once).
Pose + VQVAE replacement on top of an R1 Root checkpoint produced a
"noticeable walk" but not a stable one. Open-loop replay against
`x2_ultra_locowalk_chain_matched.pkl` shows the Root model significantly
undershoots forward translation, which matches the observed
on-robot symptoms.

---

## Round 2.1 — 2026-06-26: FT1, cosine-tail continuation on locomotion-only slice

> **Status:** finished 2026-06-26 03:50 UTC. **Verdict:** trade, not a
> win — improved yaw bias but regressed forward throughput and lateral
> stability. Useful only as a diagnostic of the cosine-tail problem.

### Why this round

The Round 2 Root model significantly undershoots forward translation
and introduces yaw bias on a straight-forward-walk PKL replay. Our
working hypothesis: the corpus that Root saw (49,790-clip v2 PKL)
was dominated by turning / manipulation / postural clips, so the model
under-weights "just walk forward" trajectories. Filtering the corpus
to a locomotion-only subset (`loco` filter — see
`motionbricks/motionbricks/data/x2_loco_filters.py`) and continuing
training was the cheapest experiment to run.

The `loco` filter narrows the v2 PKL from 49,790 → ~14,400 clips by
include-matching `walk|stride|forward|backward|sideway|turn|pivot|idle|stand`
and excluding `run|sprint|crawl|sit|jump|dance|pick|carry|chair|...`.

> **Filter bug fix that this round depended on.** The original `\b`
> word-boundary regex did not match snake_case keys like
> `Loop_Forward_Walk_001__A018`, so a naive `--filter loco` returned
> zero clips. Patched to letter-only boundaries
> (`(?<![A-Za-z])X(?![A-Za-z])`) which match across underscores and
> digits but still reject partial-word hits like `walker`. See diff
> in this commit.

### Hardware & cost

| Item                 | Value                                                      |
|----------------------|------------------------------------------------------------|
| Cloud node           | Same Nebius `computeinstance-e00cn8h67tdq00t2ys` as Round 2 |
| Wall-clock           | ~50 min (15K steps from 300K → 315K)                       |
| Cost                 | ~$22/hr × 50 min ≈ $18                                     |
| W&B run              | `root_x2_d8_b32_s315000_resume300k` — `u1ju8uer` (resumed) |

### What changed vs Round 2

| Knob                | Round 2 (Root)                                  | Round 2.1 (FT1)                                                        |
|---------------------|-------------------------------------------------|------------------------------------------------------------------------|
| Corpus              | full v2 (49,790 clips, all gait classes)        | `loco`-filtered (~14,400 clips, walks + turns + idles)                 |
| Init                | from scratch                                    | full-state resume from R2 step 300K (weights + optimizer + LR sched)   |
| LR schedule         | WarmupCosine, peak 1e-4 → final 2e-6 over 300K  | **same schedule continued** — by step 300K LR was at ~2e-6 ("dead tail") |
| Steps               | 300K total                                      | +15K (300K → 315K)                                                     |
| Save cadence        | every 5K                                        | every 1K (to find a sweet spot before LR fully decayed)                |

### Outcome — eval on `Loop_Forward_Walk_001__A018` (30 s open-loop replay)

| Metric                         | R2 step 300K | R2.1 step 315K | Delta                          |
|--------------------------------|--------------|----------------|--------------------------------|
| Forward distance (m, 30 s)     | 9.29         | 7.62           | **−18 % (regression)**         |
| Yaw drift (deg over 30 s)      | +46.5        | +34.0          | **−27 % (improvement)**        |
| Lateral drift (m over 30 s)    | +0.28        | +0.65          | **+132 % (regression)**        |
| Joint-pose RMS error           | 0.20         | 0.18           | ~−10 %                         |

**Diagnosis:** the cosine schedule had bottomed out at LR ~2e-6 by step
300K. The 15K extra steps at LR < 2e-6 were doing minimal gradient
work — net change to the model was small, dominated by whichever
gradient happened to push hardest in those last steps. The result
was a different trade-off curve, not a strictly better model.

This is the experiment that made it clear: any further fine-tune
needs a **fresh** optimizer + LR schedule, not a cosine-tail
continuation. → FT2.

---

## Round 2.2 — 2026-06-26: FT2, weights-only resume + fresh optimizer

> **Status:** finished 2026-06-26 06:36 UTC. **Verdict:** ran cleanly,
> 25 checkpoints saved, eval signal too noisy on the multi-clip suite
> to pick a single best step. Final step-30K ckpt pulled locally; A/B
> against R2 / FT1 on hardware not yet run. **Expected impact small**
> because the binding constraint is now the inference-mode mismatch,
> not the Root model weights.

### Why this round

FT1 showed the cosine-tail problem. The fix is to load the FT1
weights into a fresh optimizer with a fresh LR schedule, so the model
can actually move. This is implemented in this commit as a new
`--init-from <ckpt>` flag on
[`train_root_x2.py`](../../../motionbricks/scripts/train_root_x2.py)
(distinct from `--resume`, which restores full Lightning state).

### Hardware & cost

| Item                 | Value                                                             |
|----------------------|-------------------------------------------------------------------|
| Cloud node           | Same Nebius node (`computeinstance-e00cn8h67tdq00t2ys`)            |
| Wall-clock           | ~28 min (30K steps, 8× H100 DDP, batch 32 / GPU)                  |
| Cost                 | ~$22/hr × 28 min ≈ $10                                            |
| W&B run              | `root_x2_ft2_lr1e5_from315k` — [`jkf8oa7q`](https://wandb.ai/meetsitaram/TRL_X2Ultra_Planner/runs/jkf8oa7q) |
| Ckpts written        | 25 ckpts every 1K under `version_1/checkpoints_ft2/` (steps 6K-30K) |

### What changed vs Round 2.1

| Knob                | Round 2.1 (FT1)                                 | Round 2.2 (FT2)                                            |
|---------------------|-------------------------------------------------|------------------------------------------------------------|
| Init                | full-state resume (incl. dead-tail cosine)      | **weights-only** `--init-from <FT1-315K>` + fresh opt + fresh sched |
| LR peak             | inherited from R2 (1e-4, but already decayed)   | **1e-5** (10× smaller than R2 peak)                        |
| LR warmup           | inherited (already past)                        | 1K steps                                                   |
| LR final            | inherited (2e-6)                                | 5e-7                                                       |
| Steps               | +15K                                            | +30K (fresh counter starts at 0)                           |
| Ckpt dir            | `checkpoints/`                                  | `checkpoints_ft2/` (separate, avoids collision)            |

The new flags on `train_root_x2.py` are kept generic so future fine-tunes
can reuse them without hardcoding magic numbers in the script:

```
--init-from <ckpt>   weights-only init (mutually exclusive with --resume)
--peak-lr <float>    override hparams.yaml model.optimizer.lr
--warmup-steps <int> override hparams.yaml model.scheduler.num_warmup_steps
--final-lr <float>   override hparams.yaml model.scheduler.final_lr
--ckpt-subdir <name> override 'checkpoints/' so FT runs don't shadow base ckpts
```

The cloud launcher is committed as
[`motionbricks/scripts/cloud/run_root_finetune_v2.sh`](../../../motionbricks/scripts/cloud/run_root_finetune_v2.sh)
and the side-eval polling loop as
[`motionbricks/scripts/cloud/eval_root_finetune.sh`](../../../motionbricks/scripts/cloud/eval_root_finetune.sh).

### Outcome — eval CSV (`~/root_finetune_v2_eval.csv`, last 5 rows)

```
step    joint_rms  joint_l2   dyaw     dx        dy       col8     status
26000   0.193      0.413      +28.27   +0.401    +0.250   -0.004   PARTIAL
27000   0.198      0.416      +37.49   +0.411    +0.239   -0.004   PARTIAL
28000   0.179      0.396      −4.71    −0.014    −0.852   −0.007   PARTIAL
29000   0.178      0.420      +0.08    +0.026    −0.852   −0.006   PARTIAL
30000   0.183      0.405      −3.41    +0.258    −1.209   −0.005   PARTIAL
```

`status=PARTIAL` throughout because the multi-clip replay
(`replay_pkl_through_kplanner.py` over the locowalk PKL's 18K-key fan-out)
breaks the trajectory summarizer on non-forward gaits — yaw values
swing tens of degrees between adjacent ckpts due to mixing forward /
side / back / turn clips, not real model variance. Useless for picking
a single "best" step.

Final checkpoint pulled locally and SHA-verified:

```
motionbricks/out/motionbricks_root_x2/version_1/checkpoints_ft2/
└── model-step=0030000.ckpt   391 MB   sha256 c0e84bff36d2a17e39648021650a8122125a3a1ad57f98dac0a8ce41a6f83991
```

A/B against R2-300K and FT1-315K on the **single forward-walk PKL**
`Loop_Forward_Walk_001__A018` is the only metric we trust at this point;
not yet run.

---

## Round 2 postmortem — why 25 hours of H100s still did not give a smooth VR walk

> **Bottom line.** We spent ~$600 of 8× H100 SXM time and ~25.5 hr of
> wall-clock to retrain the X2 kinematic planner from scratch on a much
> larger corpus with much bigger batches than Round 1. Final training
> losses look healthy on paper (VQVAE 0.028, Pose 0.586, Root 1.50
> with top-5 token acc 85 %, global-root recon 0.002). **The on-robot
> result is mixed-to-bad**: forward walks drift +46.5° yaw over 30 s,
> several Round-1 gaits are worse, and the only way to get a "smooth"
> demo is to stay within a very narrow VR thumbstick envelope.
> Two follow-up fine-tunes (FT1 and FT2) trade one regression for
> another but don't fix the underlying behavior.

This section captures **why** so we don't repeat the same money for the
same disappointment in Round 3.

### How we measured the disappointment

| Test                                                    | Round 1 (legacy) | Round 2 (R2) | Round 2.1 (FT1) | Notes                                  |
|---------------------------------------------------------|------------------|--------------|-----------------|----------------------------------------|
| Open-loop replay, fwd dist on `Loop_Forward_Walk_001__A018` (30 s) | not measured     | 9.29 m       | 7.62 m          | Expected ≈ 24 m at 0.8 m/s × 30 s     |
| Open-loop replay, yaw drift over 30 s                   | not measured     | +46.5°       | +34.0°          | Expected ~0° on a straight-fwd clip   |
| Closed-loop sim (`run_x2_pkl_planner_stack.sh`)         | "barely works"   | "noticeable but not stable" | similar         | Robot steps in place; back walk is the only clean direction |
| Closed-loop on real X2 + Quest3 (sticks at 0.4-0.8 m/s) | "stiff, narrow envelope" | "side L/R aggressive, fwd intermittent, sometimes back-walks instead of fwd" | not retested    | Round-1 was at least predictable; R2 is "less predictable" |

Round 2 traded "uniformly stiff" (R1) for "wider envelope but unreliable
direction-tracking" (R2). That's not a clear win.

### Root cause 1: compute deficit vs the G1 reference

The original MotionBricks paper trained the G1 planner on a corpus and
schedule we can compare to ours like-for-like:

| Quantity                          | G1 reference         | X2 Round 2          | Ratio (X2 / G1) |
|-----------------------------------|----------------------|---------------------|-----------------|
| Total training steps (Root)       | 2,000,000            | 300,000             | **0.15× (6.6× fewer)** |
| Effective batch (DDP world)       | 16 GPUs × 128 = 2048 | 8 GPUs × 32 = 256   | **0.125× (8× smaller)** |
| Total samples processed (steps × batch) | ~4.1 B          | ~77 M               | **0.019× (53× fewer)** |
| Corpus size (clips)               | ~350,000             | ~35,000             | **0.10× (10× smaller)** |
| Steps after keyframe-curriculum warmup ends (200K) | 1.8M of 2M = 90 % | 100K of 300K = 33 % | — |

X2 Round 2 saw **53× fewer samples** than the G1 reference, spent only
**1/3 of training** post-keyframe-warmup (G1 spent 9/10), and trained
on **a 10× smaller corpus**. We are nowhere near the regime where the
MotionBricks recipe is known to give a smooth planner. "Training loss
went down" is not the same as "we paid the right amount of training".

### Root cause 2: corpus narrowing made the locowalk slice too narrow

The Round-2 v2 PKL has 49,790 clips at the top, but a corpus audit
(see `/tmp/audit_loco_corpus.py` from the postmortem session) showed:

- Only **499 clips (3.4 %)** are pure forward-walk, and **all from
  one base motion** (`Loop_Forward_Walk_001__A018`).
- About **33 % (7,452 clips)** are manipulation false-positives — the
  `locomanip` / `locopost` regex includes them because the body-part
  word (e.g. "leg") shows up in the file name even when the motion is
  arm/object work.
- Of the genuinely-locomotion clips, **turning motions dominate
  (46 %)**. Pure straight-line walks are minoritized.
- The 0.5× halfspeed merge intended to fix the low-velocity envelope
  created **velocity bimodality**: clusters at 0.5× and 1.0× of the
  natural human walking speed, almost nothing in between, and nothing
  at the very-slow VR-stick regime we actually deploy in.

So even though the corpus *count* grew 2.8× vs Round 1 (49.8K vs 18K),
the forward-walk *signal density* did not. The model learned "the
forward axis is dominated by turning + manipulation"; predictably, the
inference-time forward intent drifts.

### Root cause 3: training loss is dominated by an auxiliary task

A loss-decomposition check on the Root model's wandb panels shows:

- **`num_token` classification loss** (which token-count the keyframe
  curriculum is asking for) dominates the total loss at roughly
  **~98 %** of magnitude across training.
- The actual **continuous root-trajectory reconstruction losses**
  (global-root recon, local-root recon) are at ~0.002 and ~0.025
  respectively — orders of magnitude smaller.

This is not "wrong" per se — the keyframe curriculum is designed to
spend most of its loss budget on the discrete token-count head until
the curriculum is fully unrolled. But it means **the optimizer was
spending the bulk of its gradient budget reducing the curriculum loss**,
not improving the trajectory we actually use at inference. With G1's
training budget (10× the steps post-warmup), this auxiliary task
saturates and the trajectory losses start moving. With our budget,
the curriculum is still warming up when we stop.

### Root cause 4: inference-time mode mismatch (we use Root in OOD)

This is the most under-appreciated one. The Root backbone is an
**in-betweener**: at training time it sees a start pose, a target
**body pose**, and a target root position/velocity, and learns to fill
in the trajectory between them. The `has_local_poses` constraint mask
is True on both ends of the 8-frame window during training.

At inference time, `motionbricks/motion_backbone/inference/neural_planner.py`
explicitly **masks `has_local_poses[:, -NUM_FT:] = False`**. The model
is asked to predict a trajectory toward a target the user described
with **velocity intent + implied target xy + nothing else for the body
pose**. This is an out-of-distribution use of the model relative to
training.

This isn't a new finding for this codebase — a probe script
[`motionbricks/scripts/probe_root_constraint_modes.py`](../../../motionbricks/scripts/probe_root_constraint_modes.py)
already exists and compares three modes:
`velocity_only` / `velocity_plus_target_pos` / `demo_full`. On the
G1 + X2 reference checkpoints, the probe found that adding the
target keyframe body pose (`demo_full`) only improves per-token
forward-tracking slope by ~3 % over `velocity_plus_target_pos`. The
comment in `neural_planner.py` calls this out and concludes "not
worth the coupling" for shipping a robot-specific stand-pose
template.

But that probe ran on tracking, not on the smoothness criterion we
actually care about for deploy, and it ran on the Round-1 X2 + the
G1 reference checkpoints — not on our specific Round-2 failure mode.
It is plausible (not proven) that pose templates would help more on
our specific brokenness. If we run that probe locally against the
current R2 + FT1 + FT2 ckpts and the gap is still ~3 %, then pose
templates are not the lever. If it's noticeably bigger, they are.

### What we would try next (if we paid this $600 again)

In priority order:

1. **Re-run `probe_root_constraint_modes.py` on R2 / FT1 / FT2** with
   our actual failure clip (`Loop_Forward_Walk_001__A018`). 10 min of
   local-GPU work, zero cloud cost. If the `velocity_plus_target_pos`
   → `demo_full` delta is meaningfully bigger than 3 %, plumb 7 X2
   canonical pose templates (one per discrete VR gait command) into
   `_predict_with_velocity` behind a `--use-pose-templates` flag, and
   A/B in sim. **Likely cost: 1 evening.**
2. **A small specialist Root model** trained on the existing X2 PKL
   corpus only, at fixed hip height, fixed body-pose-template family,
   for ~50K steps on 1 H100. **Estimated cost: ~$10**. The point is
   not to outperform the generalist Root, it's to give us a Root model
   that was *trained* on the inference distribution we deploy in
   (narrow VR velocity envelope, 7 discrete gait poses, locked
   pelvis height). The generalist Root is the wrong tool for our
   actual operating envelope.
3. **Only if 1 + 2 both fail to give a smooth walk**: another
   8× H100 run with **a smarter corpus** — heavily upweighted pure
   straight-line walks, no halfspeed bimodality (use continuous speed
   jitter instead), no manipulation false-positives. Target ~600K
   steps (not 300K), to spend more of the budget post-keyframe-warmup.
   **Estimated cost: ~$1000.**

Path 1 is essentially free. Path 2 is ~1 % of Path 3's cost. Path 3
should only be undertaken if 1 + 2 conclusively diagnose that we need
a fundamentally better generalist Root model, not a better inference
mode or a specialist.

### Lessons we are taking forward

| Lesson                                                                                                       | Action                                                                                                                   |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Training loss going down is not the same as "the model will work in deploy"                                  | Always pair training with an **open-loop replay metric** on a representative clip from the deploy envelope               |
| Cosine schedules with `final_lr` near 2e-6 have a "dead tail"; continuing past that is wasted compute        | `--init-from` (weights-only) + fresh `--peak-lr/--warmup-steps/--final-lr` is now a supported pattern in `train_root_x2.py` |
| "Bigger corpus" is not the same as "better corpus"                                                           | Audit any new corpus by gait class **before** spending H100 hours on it                                                  |
| Auxiliary classification losses can dominate the gradient budget and starve the metric you actually care about | Decompose loss panels in W&B during training, not after; surface an early warning if `num_token_loss / total_loss > 0.8` |
| Inference may not match training (in-betweener mode vs velocity-only mode)                                   | Run `probe_root_constraint_modes.py` on every new Root checkpoint before declaring it "trained"                          |
| 8× H100 is the wrong instance size for fine-tuning / experiments                                             | Use 1× H100 (or local 5090) for FT1 / FT2 / specialist runs; reserve multi-GPU for from-scratch ≥ 500K-step training     |

---

## Round 1 — 2026-05-28 → 06-02: single-GPU locowalk-only

> **Reconstruction.** This round predates this history doc; details are
> reconstructed from local artifacts under
> `motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/` and W&B
> metadata files. Not all numbers are exact.

### Hardware & cost

| Item                | Value                                                                                |
|---------------------|--------------------------------------------------------------------------------------|
| Compute             | Local workstation, 1× NVIDIA RTX 5090 (~24 GB)                                       |
| Wall-clock          | ~5 days off-and-on (with resumes / pauses)                                           |
| Cost                | electricity only                                                                     |
| W&B project         | `TRL_X2Ultra_Planner` (user `meetsitaram`)                                           |
| W&B runs            | `ef9jjfi3` (VQVAE), `xogxwh4z` (Pose), `u1ju8uer` (Root resume from 100K → 240K)    |

### Corpus

| PKL                                  | Clips   | Notes                                |
|--------------------------------------|---------|--------------------------------------|
| `x2_ultra_locowalk.pkl` (legacy retarget) | ~18,000 | locowalk only; legacy uniform_h14 retargeting |

### Stage wall-clock (best-effort reconstruction)

| Stage                          | Steps   | Wall-clock | Notes                                                                          |
|--------------------------------|---------|------------|--------------------------------------------------------------------------------|
| Skeleton + stats + hparams     | —       | ~30 s      | Doc-quoted figure; small corpus, fast.                                         |
| Feature cache (locowalk)       | —       | ~15 min    | Local CPU, ~24 workers.                                                        |
| VQVAE                          | 200 K   | ~28 hr     | Single-GPU, batch_size=4. Slow but reached step 200 K.                          |
| Pose                           | 250 K   | ~30 hr     | Single-GPU, batch_size=4. Resumed several times.                                |
| Root                           | 100 K → 240 K | ~30 hr | Initially 100 K, then resumed for another ~140 K steps with `--resume`.   |

### Outcome

- **Working but stiff.** Robot walked under Quest 3 VR teleop using the
  trained kplanner triple, but motions were noticeably stiff, jittery at
  low velocities, and noisy on direction changes.
- **Narrow velocity envelope.** Anything below ~0.4 m/s thumbstick
  deflection produced near-frozen poses; clips at that speed weren't in the
  training distribution.
- **Posture/manipulation/balance unconditioned.** Because the corpus was
  locowalk only, the model never saw arm-rich, sit, or balance-recovery
  clips. Out-of-distribution at runtime → fallback behaviors.

These three issues motivated the round-2 design decisions above.

### Local artifacts (still on disk)

```
motionbricks/out/motionbricks_vqvae_x2/version_1/
├── checkpoints/last.ckpt                                  # 285 MB
├── stats/motion/{mean,std}.npy                            # built on locowalk-only corpus
└── ...

motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/
└── model-step=0100000.ckpt + last.ckpt                    # 1.6 GB each

motionbricks/out/motionbricks_root_x2/version_1/checkpoints/
└── model-step=0240000.ckpt + last.ckpt                    # 410 MB each
```

These checkpoints are **not** valid starting points for round 2 (different
corpus + retargeting + stats), but are kept for:
1. A/B comparison ("how much better did round 2 actually get?").
2. Fallback in case round 2 surfaces a regression.

---

## How to add the next entry

1. While the round is running, append a `## Round N — YYYY-MM-DD: <one-liner>`
   section above this one (most recent first).
2. Fill the stage table as stages complete — wall-clock timestamps,
   notes, and any issues encountered.
3. After training finishes, fill in the **Outcome** subsection: W&B
   run IDs, final checkpoint paths, codebook utilization, val-loss
   shape, and subjective behavior assessment.
4. Update the **Quick reference** table at the top with a one-line
   summary row.
5. If the round introduced new failure modes or surprising costs,
   update the appropriate companion doc:
   - Procedural issues → `train-planner-on-cloud.md`
   - Runtime/inference issues → `x2_kplanner.md`
   - Pipeline refactor opportunities → `.cursor/plans/*.plan.md`
