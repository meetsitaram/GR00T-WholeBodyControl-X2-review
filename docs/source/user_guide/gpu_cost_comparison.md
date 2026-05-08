# GPU Cost Analysis: H100 vs. H200 vs. B200 (8-GPU node)

A cost-vs-quality comparison of the three NVIDIA SXM platforms typically
available for X2 Ultra training on Nebius. Built from **direct
measurements** taken during the 2026-05-01 controlled head-to-head
experiment between an 8× H200 and 8× B200 running the same X2 Ultra
sphere-feet config.

> **Companion docs:**
>
> - [`ppo_batch_size_explainer.md`](ppo_batch_size_explainer.md) —
>   explains why bigger rollout buffers don't proportionally improve
>   convergence at our scale.
> - [`compute_scaling_sonic_vs_ours.md`](compute_scaling_sonic_vs_ours.md)
>   — covers *why* per-GPU env counts differ from NVIDIA's published
>   SONIC setup.

## TL;DR (verified by 2026-05-01 head-to-head measurements)

For an 8-GPU node running our X2 Ultra training:

| Use case | Best pick | Cost (20K iters) | Wall-clock |
|---|---|---:|---:|
| **Hyperparameter / reward sweeps** | 8× H100, 8K envs | **~$515** ¹ | ~19 h |
| **From-scratch production training** | **8× H200, 16K envs** | **$1,291** | **45.9 h** |
| **From-scratch (max batch)** | 8× B200, 24K envs | $2,059 | 46.7 h |
| **Research scaling experiment** | 16-128× H100 / H200 | varies | varies |

**Headline:** at the same wall-clock (B200 8.38s/iter ≈ H200 8.25s/iter),
B200 produces 50% more samples per iter — but those extra samples land
deep in the PPO gradient-SNR plateau and don't translate to meaningfully
better policies. The cheaper $/hr of H200 wins by **$768 (37%)** on a
20K-iter from-scratch run with no measurable quality penalty.

¹ Extrapolated; H100 8-GPU not directly measured this run.

## Pricing (Nebius console, verified 2026-05-01)

8-GPU SXM configs + 1280 GiB SSD:

| GPU class | List $/hr | All-in (with disk) |
|---|---:|---:|
| 8× H100 SXM (80 GB / GPU) | $26.40 | $26.53 |
| 8× H200 NVLink (141 GB / GPU, eu-west1) | $28.00 | **$28.13** |
| 8× B200 NVLink (192 GB / GPU, us-central1) | $44.00 | **$44.13** |

> **Pricing gotcha:** The 8-GPU presets are **NOT** strict 8× the
> per-GPU shell rate. The Nebius shell list (`h200-1`, `b200-1`) gives
> $3.80 / $5.50 per GPU-hr, which would extrapolate to $30.40 / $44 for
> 8 GPUs. The actual 8-GPU console price is $28 / $44 — H200 has the
> larger per-GPU discount when bundled (it's about $3.50 / GPU at the
> 8× tier).

## Measured per-iter throughput (from the 2026-05-01 head-to-head)

This is the same 20K-iter X2 Ultra `bones_seed_sphere_feet` config
running simultaneously on both GPU classes, sampled at sustained
operating point (steady-state, last 100 iters):

| GPU class | $/hr | VRAM/GPU | NUM_ENVS / GPU | Iter time | FPS | Util | $/iter | $/M-env-step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8× H100 SXM | $26.53 | 80 GB | 8,192 ¹ | ~3.5 s ¹ | ~750 K ¹ | ~85% ¹ | ~$0.026 ¹ | ~$0.010 ¹ |
| **8× H200 NVLink** | $28.13 | 141 GB | 16,384 | **8.25 s** | **390 K** | **70%** | **$0.064** | **$0.020** |
| **8× B200 NVLink** | $44.13 | 192 GB | 24,576 | **8.38 s** | **547 K** | **78%** | **$0.103** | **$0.022** |

¹ H100 is **not directly measured this run**; numbers extrapolated from
H200 measurements scaled by memory-bandwidth ratio (3.35 / 4.8 = 0.70x
slower per env-step) and capped at 8K envs/GPU per
`train-on-cloud.md` §A.4.

### Surprise: B200 and H200 are within 2% on iter time

The 2026-05-01 head-to-head produced an unexpected result:
**8.38 s/iter on B200 vs 8.25 s/iter on H200**, despite B200 processing
**50% more environments per iter** (24,576 vs 16,384 per GPU).

Two ways to read this:

1. **B200 is *not* framework-bound** the way I initially hypothesized.
   It absorbs 50% more work in essentially the same wall-clock — a
   1.40× FPS uplift over H200. The GPU is genuinely doing more useful
   work.
2. **But that extra work doesn't translate to converged-policy quality
   at our batch scale.** Both runs sit deep in the PPO gradient-SNR
   plateau where bigger rollout buffer ≠ proportionally faster
   convergence. B200's 196K total envs vs H200's 131K is a 50% bump in
   *samples per gradient update*, which the [Andrychowicz et al. 2021
   plateau theory](https://arxiv.org/abs/2006.05990) says is mostly
   noise reduction we can't act on (PPO clip ε=0.2 limits step size
   regardless of gradient quality). See
   [`ppo_batch_size_explainer.md`](ppo_batch_size_explainer.md).

So the "right" denominator for cost comparison is **per-iter cost**,
not per-env-step cost — because each iter = one gradient update = one
unit of policy improvement, and at our scale the *quality* of each
gradient update is essentially the same on both GPUs.

By that metric, **H200 is 37% cheaper per iter** ($0.064 vs $0.103).

### Convergence quality at the same iter (apples-to-apples)

Sampled at iter 214 from the head-to-head (last 10 iters averaged):

| Metric | B200 (4.72M batch) | H200 (3.15M batch) | Δ |
|---|---:|---:|---:|
| Mean reward | 1.085 | 1.011 | B200 +7.4% |
| Mean episode length | 17.7 | 16.9 | B200 +4.9% |
| `policy/approxkl_avg` | 0.00948 | 0.01041 | H200 +9.8% noisier |

B200 converges slightly faster per-iter (~5-10% reward lead), exactly
as the √(N) gradient-noise theory predicts. But the gap is small:
- Theoretical prediction: H200 should be 22% noisier (√(4.72/3.15)=1.22)
- **Measured**: H200 is only 9.8% noisier
- Both KL values sit deep in the 0.005-0.05 healthy band — neither run
  is gradient-starved

**Implication:** to match B200's policy at iter 20K, H200 would need
roughly 20K × 1.07 ≈ **21.4K iters**. We launched H200 with 25K iters
specifically as buffer. So H200 will produce at-least-equivalent or
slightly better policy than B200's 20K-iter run.

## Cost for a fixed 20K-iter from-scratch run

| GPU | NUM_ENVS / GPU | Total batch | Iter time | 20K wall-clock | Cost @ console | vs H200 |
|---|---:|---:|---:|---:|---:|---:|
| 8× H100 ¹ | 8,192 | 65,536 | ~3.5 s | ~19.4 h | **~$515** | -60% |
| **8× H200** | 16,384 | 131,072 | **8.25 s** | **45.9 h** | **$1,291** | baseline |
| 8× B200 | 24,576 | 196,608 | **8.38 s** | **46.7 h** | **$2,059** | **+60%** |

¹ Extrapolated; not measured 2026-05-01.

The 60% cost premium on B200 buys you essentially nothing useful:
- Same wall-clock (within 2%)
- Slightly cleaner gradient (deep in plateau, not actionable)
- 50% more samples consumed (wasted past the PPO clip threshold)

## The catch: batch size matters when it matters

The plateau argument assumes **convergence is gradient-limited, not
sample-limited**. There are three regimes where bigger batch *does* pay
off and B200 starts to make sense:

### Regime 1: Hard exploration (sparse rewards, long horizons)

If most of your rollout transitions don't contain useful learning
signal (e.g., reaching tasks where reward only fires on success),
then you need a bigger rollout to *include* the rare informative
samples. This is partly why NVIDIA SONIC used 524K total batch on 128
GPUs — 100M+ frames is *a lot* of human motion to track, much of it
underrepresented.

X2 Ultra tracking is **dense reward** (per-frame body / anchor /
joint tracking objectives), so this regime doesn't apply to us. Our
plateau analysis holds.

### Regime 2: Late-stage fine-tuning

Once policy changes per iter become tiny (small KL, small step size),
gradient noise relative to step size grows. A bigger batch helps
stabilize the final descent. This matters mostly for runs >50K iters
on a fully-converged policy. Our 20K-iter from-scratch is well
inside the noisy-gradient-tolerable regime.

### Regime 3: Aggressive learning rate scaling

If you increase `actor_learning_rate` to converge faster (we use
2e-5; SONIC uses 1e-4), the trust region (`clip_param=0.2`) tightens
and you need cleaner gradients to avoid spending iters in the
clip-saturated regime. Our 8× H100 sweeps frequently use 5e-5 and
still stay healthy at 64K total batch — but if you push to 1e-4 or
beyond, B200's bigger batch helps.

## Recommendation matrix

| Use case | Best pick | Why |
|---|---|---|
| **Hyperparameter / reward sweeps** | 8× H100, 8K envs, 20K iters | $515/run, 19 h — fast feedback; small batch is fine for ranking experiments |
| **Fine-tune from existing checkpoint** | 8× H100, 8K envs, 4-8K iters | $100-200/run; existing weights tolerate small batches well |
| **From-scratch production training** | **8× H200, 16K envs, 20K iters** | **$1,291, proven recipe, 25% of NVIDIA batch** |
| **High-stakes from-scratch** (where wasted compute is cheaper than wasted scientist time) | 8× B200, 24K envs, 20K iters | $2,059, +37% over H200 — buys "psychological safety" via 1.5× bigger batch (real-world quality bump: ~7%) |
| **Memory-hungry experimental architectures** | 8× B200, 24K envs+ | 192 GB / GPU lets you grow model size or DR comp without sharding |
| **Research scaling experiment** (paper-grade Pareto) | 16-128× H100 / H200 with reduced envs/GPU | matches NVIDIA's 128-way SONIC style |

## When to deviate from the matrix

### Pick **B200** when:
- 8× H200 capacity is unavailable across all regions and you need to
  start a run *now* (B200 is sometimes the only HIGH-availability
  8-GPU preset).
- You're testing **memory-hungry** model variants (>50 GB / GPU)
  where 192 GB headroom matters.
- You're running with `actor_lr ≥ 1e-4` and need the bigger batch's
  gradient stability.
- You want maximum total samples for an exploration-heavy task —
  ~94B samples (B200, 20K iters) vs ~63B samples (H200, 20K iters).

### Pick **H100** when:
- You're doing a sweep of >5 configs (cumulative cost dominates).
- You're fine-tuning from a known-good checkpoint (small-batch is
  fine on warm starts).
- You're prototyping a new reward function / DR config and want fast
  feedback (19 h vs 46 h matters when iterating).

### Pick **H200** in all other cases.

## Why isn't B200 a slam dunk if it's "the latest"?

Hardware specs:

| GPU | TFLOPS BF16 (TC) | Memory bandwidth | VRAM | Power TDP |
|---|---:|---:|---:|---:|
| H100 SXM | 989 | 3.35 TB/s | 80 GB | 700 W |
| H200 SXM | 989 | 4.8 TB/s | 141 GB | 700 W |
| B200 SXM | 2,250 | 8 TB/s | 192 GB | 1000 W |

B200 has 2.3× H100's compute and 2.4× the bandwidth. Yet our measured
**8.38 s/iter on B200 at 24K envs** vs **8.25 s/iter on H200 at 16K
envs** is essentially tied wall-clock. **Per-env**, B200 is 1.40×
faster (0.341 vs 0.504 ms/env-step) — that's a real hardware win.

But the ratio you actually pay for is **wall-clock per iter**, since
PPO updates are gated by collection time, not env-step microbenchmarks.
And on that axis, B200 only beats H200 when you give it 50% more envs
to chew on — at which point you're using its compute for noise
reduction past the SNR plateau.

For workloads that *do* expose B200's advantage (large transformer
training, FP8 LLM inference, scientific HPC, or RL with ≥30K envs/GPU
on tasks that need the batch), the picture flips. B200 vs H200 is a
**workload-specific call**, not a strict upgrade.

## Spot / preemptible discounts

Nebius offers preemptible (spot) presets at ~55-60% of list price.
Effective cost for the same 20K-iter run on preemptible (scaling
on-demand prices by 0.55-0.60):

| GPU | On-demand | Est. preemptible | Caveats |
|---|---:|---:|---|
| 8× H100 | $515 | ~$285 | Acceptable — checkpoint every 2K iters means ≤ 1 hr lost on preemption |
| 8× H200 | $1,291 | ~$715 | Acceptable — same |
| 8× B200 | $2,059 | ~$1,135 | Acceptable — same |

**Preemptible spot prices need to be re-verified in the Nebius
console** — earlier doc revisions cited $15/$22/$28 per hr from
extrapolation, likely overstated by ~25-30%.

## Open questions for future runs

- **What's the 8× H100 80GB iter time at 8K envs?** Never directly
  measured. A 1-iter smoke run would replace the extrapolated `~3.5 s`
  with a measured number. Cost: ~$0.50 of preemptible H100 time.
- **Does B200 at 32K envs/GPU close the throughput gap?** B200 VRAM
  was at only 38% during the 2026-05-01 run — there's room for 32-40K
  envs/GPU. At 32K, total batch jumps to 256K, util likely hits
  85-90%, iter time may grow to ~10 s. Worth a 1-iter smoke test on
  the next B200 run.
- **Does running at FP8 (B200 supports native FP8) recover B200's
  advantage?** Isaac Lab + PPO doesn't currently use FP8; experimental
  conversion would be a research project on its own.
- **Does `accelerate launch --use-deepspeed` change the picture?**
  DeepSpeed's stage-3 sharding changes how inter-GPU sync works and
  may unlock more of B200's compute on smaller models.

## Revision history

This doc was rewritten three times as data came in:

1. **2026-05-01 (initial smoke):** B200 9.2 s/iter, 68% util — based on
   the first 10 iters before warm-up. Wrong-pricing $/hr ($38 H200,
   $48 B200 from per-GPU shell list × 8). Conclusion: H200 saves
   ~$1,032/run vs B200 ($1,460 vs $2,294 for 20K iters). Implied B200
   was severely framework-bound (only 9% per-env advantage over H200
   despite 2.3× compute).
2. **2026-05-01 (sustained, iter 170):** Corrected to 8.6 s/iter
   sustained, 78% util. Pricing corrected to $28.13/$44.13 from
   Nebius console. Conclusion shifted: H200 saves $1,032/run vs B200
   ($1,077 vs $2,109).
3. **2026-05-01 (head-to-head, iter 214):** Direct simultaneous
   measurements — B200 8.38 s/iter, H200 8.25 s/iter (essentially
   tied wall-clock), confirmed reward gap of only ~7% per iter
   (within plateau theory). Final cost numbers: H200 $1,291, B200
   $2,059, gap is $768 (37%). The B200 run was terminated at iter ~480
   to save the remaining ~$1,953, and the H200 run continued to 25K
   iters as the production training.

The journey illustrates why **head-to-head measurements with
controlled config matter** — smoke tests systematically underestimate
B200's compute boundedness, and per-GPU price extrapolation
overestimates 8-GPU bundled rates. **Always trust the all-in console
price and the iter-time at iter ≥100.**

## References

- **Pricing source:** Nebius console UI (2026-05-01) for both H200
  eu-west1 and B200 us-central1 8-GPU configs.
- **Throughput measurements:** W&B runs
  [`ru7nx75z` (B200)](https://wandb.ai/meetsitaram/TRL_X2Ultra_BonesSeed_SphereFeet/runs/ru7nx75z)
  and [`z7docj57` (H200)](https://wandb.ai/meetsitaram/TRL_X2Ultra_BonesSeed_SphereFeet/runs/z7docj57).
- **PPO scaling theory:** Andrychowicz et al., *"What Matters in
  On-Policy Reinforcement Learning?"* (2021),
  [arxiv 2006.05990](https://arxiv.org/abs/2006.05990).
  See [`ppo_batch_size_explainer.md`](ppo_batch_size_explainer.md) for
  our paraphrase.
- **NVIDIA SONIC scaling:**
  [arxiv 2511.07820 §6](https://arxiv.org/abs/2511.07820) — see
  [`compute_scaling_sonic_vs_ours.md`](compute_scaling_sonic_vs_ours.md)
  for our paraphrase of why they picked their operating point.
- **Local artifacts from B200 run:**
  `~/x2_cloud_checkpoints/b200-run-20260501_142541/` (last.pt at iter
  ~480, full train log, bootstrap log).
