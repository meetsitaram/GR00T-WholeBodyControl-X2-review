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
| 2     | 2026-06-24 → in progress | Nebius 8× H100 SXM (cloud, ~1.5 TB RAM) | `x2_ultra_bones_seed_chain_matched.pkl` (VQVAE) + `_v2` w/ 0.5× locowalk merge (Pose+Root) | 37,968 base / 49,790 v2 | (in progress; ETA below)        | TBD                                        |

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
| Expected total           | ~14-18 hr training × ~$22/hr → **~$350**                                             |
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
| 6  | **VQVAE training** (8 GPU DDP, 500K steps)                   |                 |                 |            | `batch_per_gpu=4` → effective batch 32. W&B on.                                              |
| 7  | **Pose training** (8 GPU DDP, 200K steps, needs VQVAE ckpt)  |                 |                 |            |                                                                                              |
| 8  | **Root training** (8 GPU DDP, 200K steps, independent)       |                 |                 |            | Sequential after Pose to avoid GPU contention.                                              |

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

### Round-1 → Round-2 design changes (one-glance)

| Aspect                       | Round 1                                  | Round 2                                                |
|------------------------------|------------------------------------------|--------------------------------------------------------|
| Compute                      | 1× RTX 5090 (~24 GB)                     | 8× H100 SXM 80 GB                                      |
| Effective batch              | 4                                        | 32 (4 / GPU × 8 GPUs DDP)                              |
| VQVAE steps                  | ~200 K                                   | 500 K                                                  |
| Pose steps                   | ~250 K (resumed)                          | 200 K from scratch                                     |
| Root steps                   | ~240 K (resumed from 100 K, then +100 K) | 200 K from scratch                                     |
| Corpus                       | `x2_ultra_locowalk.pkl` only             | full curated chain_matched corpus (locowalk + post + manip + bal) |
| Retargeting method           | legacy `uniform_h14`                     | `chain_matched` (pelvis-Z + knee-flex correct)         |
| Slow-velocity coverage       | none (natural human speeds only)         | 0.5× locowalk variants merged for Pose + Root          |
| W&B                          | enabled                                  | enabled, project `TRL_X2Ultra_Planner`                 |
| Skeleton-stats compute       | ~30 s (locowalk-only corpus)             | ~102 min (full corpus, `--max-clips-stats 0`)          |

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

### Outcome (TBD)

Will fill in after training completes:

- VQVAE final codebook utilization (target: >80 %)
- Pose val-loss curve shape (target: monotone-decreasing through 200 K)
- Root val-loss curve
- W&B run IDs (3 — one per stage)
- Final checkpoint paths on cloud + local
- Subjective VR-driving feel vs round 1 (smoothness, velocity range,
  posture realism, manipulation reach)

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
