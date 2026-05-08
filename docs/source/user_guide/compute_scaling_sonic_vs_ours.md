# Compute Scaling: NVIDIA SONIC vs. Our X2 Ultra Training

A side-by-side comparison of the compute footprint used by NVIDIA's released
SONIC training run on the Unitree G1 (paper
[arxiv 2511.07820](https://arxiv.org/abs/2511.07820), code
[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)) and our X2 Ultra
training runs on cloud nodes. The goal is to understand *why* the two setups
chose very different (envs/GPU, GPU count, batch size) operating points, so
we can make informed scaling decisions when we re-run on different cluster
sizes in the future.

## TL;DR

| Metric | NVIDIA SONIC (G1) | Ours (X2 Ultra, 2026-05-01) |
|---|---|---|
| **GPUs** | **128** | **8** |
| **GPU-hours** (released model) | **~9,000** | **~240** (estimated) |
| **Wall-clock** | ~70 h ≈ 3 days | ~30 h ≈ 1.25 days (est.) |
| **Hardware (per-GPU)** | H100 (80 GB SXM, presumed) | B200 (192 GB SXM) |
| **Envs / GPU** | **4,096** | **24,576** |
| **Total parallel envs** | **524,288** (128 × 4,096) | **196,608** (8 × 24,576) |
| **Effective gradient batch (samples/step)** | ~3.1 M | ~1.18 M |
| **Model size** | 42 M params | smaller (token-conditioned) |
| **Motion data** | 100 M+ frames (700 h mocap) | 2,550 motions (~50-80 h equiv) |

Same algorithm family (PPO + motion tracking), wildly different operating
points, and *both* are sensible choices given each side's constraints. This
doc explains why.

> Some scaling-experiment runs in the SONIC paper went up to **21K-32K
> GPU-hours** (parameter / data sweeps for the Pareto plots in §6 of the
> paper). The 9K figure above is for the *released* checkpoint.

## NVIDIA's choices, explained

NVIDIA explicitly designed for a 128-GPU cluster running a 42 M-parameter
model. Three constraints dominated their per-GPU env choice:

### 1. Memory budget per GPU (≤ 80 GB on H100)

A 42 M-parameter policy plus optimizer state plus IsaacSim physics scratch
plus replay buffers eats VRAM fast. With ~4,096 envs/GPU they have
comfortable headroom for memory creep across 70 h of continuous training.
A larger per-GPU env count would have pushed them closer to OOM during
multi-day runs — and an OOM at iter 60K is *very* expensive at $30+/hr × 128
GPUs.

### 2. Communication overhead at 128-way data-parallel

PPO is on-policy: every gradient step requires an all-reduce across all 128
GPUs. The cost of that all-reduce is roughly proportional to the model size
and the network topology, **not to the per-GPU env count**. So the optimal
strategy at 128 GPUs is to keep the *per-step compute time* low (so the
all-reduce overlap is good) rather than maximizing samples per GPU.

This is why scaling is sublinear past ~32 GPUs for PPO without specialized
collectives — and why frameworks like rsl_rl, IsaacLab's RL stack, and
SONIC's training code default to 4-8 K envs/GPU, not 32 K.

### 3. The PPO gradient-SNR plateau

At 128 × 4,096 = 524,288 envs and rollout length ~24, NVIDIA's effective
gradient batch is ~3.1 M samples. This is **already past the
gradient-SNR knee** — past this point, additional samples add information
to each gradient *very* slowly. So increasing per-GPU envs would burn
extra compute for vanishing benefit.

Reference: Andrychowicz et al., *"What Matters in On-Policy
Reinforcement Learning?"* (2021). Their finding: PPO's minibatch size is
one of the *least* sensitive hyperparameters once you compensate with
learning rate.

### Why so much *total* compute (9,000 GPU-hours)?

Not because they *needed* it for convergence, but because they were after
the **scaling Pareto curve** — they wanted to show how performance
improves with parameters and data volume. That requires multiple sweeps,
each ~1-3 K GPU-hours. The released model is the upper-right point of that
curve: 42 M params, 100 M+ frames, ~9 K GPU-hours.

## Our X2 Ultra setup, explained

We have the opposite resource shape: **fewer, beefier GPUs in a single node**
(8x B200 SXM 192 GB, NVLinked, no inter-node fabric). The optimal
operating point is a mirror image:

### 1. Concentrate envs per GPU because we have memory headroom

192 GB on B200 vs 80 GB on H100 = **2.4× more VRAM/GPU**. Plus our model is
substantially smaller than 42 M params. Net: we can host ~3-4× the envs/GPU
that NVIDIA could without OOM risk.

### 2. No communication overhead worth optimizing

8 GPUs all on a single B200 SXM tray means all-reduce happens entirely over
NVLink/NVSwitch — microsecond-scale latency, near-bandwidth-saturated
throughput. Increasing per-GPU compute time has *zero* communication cost
in our regime. So the "keep per-step compute low" pressure that drove
NVIDIA's 4 K choice doesn't apply to us.

### 3. We want enough total batch to be near (but below) the SNR plateau

If we used 4,096 envs/GPU like NVIDIA, our **total batch is just 32 K** —
*way* below the gradient-SNR plateau and well into the regime where PPO
is sample-inefficient (tiny gradients with high variance).

By scaling to 24,576/GPU we get **196 K total envs**, ~2.7× smaller than
NVIDIA's batch but firmly in the healthy regime. Effective gradient batch
~1.18 M samples is well-behaved per Andrychowicz.

### 4. We chose 24,576, not 32,768 — why not max out?

Past ~24 K envs/GPU on any modern GPU, **IsaacSim physics scaling is
sublinear**. The env-update kernel becomes memory-bandwidth-bound rather
than compute-bound, so doubling envs gives ~1.5x throughput, not 2x. The
returns are diminishing.

24,576 is also a clean power-of-two-friendly batch size that aligns
nicely with rollout buffer size and minibatch chunking.

## Trade-off table: their setup vs ours

| Axis | NVIDIA (128 × H100, 4 K envs) | Ours (8 × B200, 24 K envs) |
|---|---|---|
| Per-GPU memory pressure | High (80 GB tight) | Low (192 GB roomy) |
| Inter-GPU communication | Major bottleneck (128-way all-reduce) | Negligible (NVLink intra-node) |
| Per-step compute time | Want low (overlap with all-reduce) | Free to be high |
| Total batch size target | 524 K (past SNR plateau) | 196 K (well-behaved regime) |
| Iters per wall-clock hour | High (~700/h) | Lower (~600/h, est.) |
| Iters per GPU-hour | ~5.5 | ~75 (us)  *measured at H200; revise* |
| Model + data scale | 42M params, 100M frames | smaller, 2,550 motions |
| Total $$ (rough) | $250-300K @ $30/h × 9000 h | ~$1,400 @ $48/h × 30 h |

## What we'd do differently with 32+ GPUs

If we had access to 32 × H100 or 32 × B200, the right move would be:

1. **Drop envs/GPU to ~8,192 - 12,288** to balance per-step compute against
   the now-non-trivial all-reduce cost.
2. **Increase total envs to ~256 K - 393 K**, much closer to NVIDIA's
   524 K, exploiting the linear regime of PPO scaling up to that point.
3. **Watch all-reduce overlap carefully** — at 32+ GPUs, PyTorch DDP +
   accelerate's default settings start showing 5-10% communication overhead
   that didn't exist at 8-GPU. Could justify torch.distributed.fsdp or
   ZeRO-style sharding.

## What we'd do differently with 1-2 GPUs

If we had only a 1-2 GPU dev node, the right move flips the other way:

1. **Stay at 4 K envs/GPU** — going higher gives diminishing single-GPU
   throughput once memory bandwidth saturates.
2. **Total batch ~4-8 K is *too small* for healthy PPO**. Need to either
   (a) compensate with smaller learning rate + more iters, or (b) accept
   that 1-2 GPU runs are for *prototyping*, not for final policies.
3. **Iterate on reward shaping / DR / motion data** at this scale, then
   move to 8 GPUs for the actual training runs.

## Open questions for future runs

- **How much sample efficiency do we lose with 24 K vs 4 K envs/GPU at
  fixed total envs?** Worth a controlled ablation: train two policies, one
  with `(8 GPU × 24 K)` and one with `(64 GPU × 3 K)`, both at 196 K total
  envs, compare convergence per gradient step and per wall-clock hour.
- **Does B200 compute throughput at 24 K envs really beat H100 at 4 K
  envs by 6×?** Need to measure iter time once the current run hits
  steady-state at iter 200+. Update this doc with the measured
  iters/hr figure.
- **Where does our gradient-SNR knee actually sit?** NVIDIA's empirical
  knee (per their scaling plots) is somewhere between 256 K and 1 M envs.
  Ours might be lower because of the smaller model. If we ever find that
  64 K and 196 K envs converge similarly, we should cut envs to free up
  GPUs for parallel ablations.

## References

- Paper: [arxiv 2511.07820](https://arxiv.org/abs/2511.07820) — "SONIC:
  Supersizing Motion Tracking for Natural Humanoid Whole-Body Control".
  See §6 (scaling experiments) for the per-GPU env count and total
  compute breakdown.
- Project page: [nvlabs.github.io/SONIC](https://nvlabs.github.io/SONIC/)
- Released checkpoint: [huggingface.co/nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)
- Andrychowicz et al., *"What Matters in On-Policy Reinforcement
  Learning? A Large-Scale Empirical Study"* (2021), arxiv 2006.05990 —
  the canonical reference on PPO hyperparameter sensitivity, including
  minibatch size scaling.
- Our run: see [`training_runs.md`](training_runs.md) for the
  per-iter metric history of `bones_seed_sphere_feet-20260501`.
- Our cloud setup: see
  [`train-on-cloud.md`](train-on-cloud.md) for the bootstrap recipe
  used to get the 8x B200 node ready for training.

## Appendix — measured iter throughput across our hardware

Updated as we collect data. Goal: empirically validate the scaling
arguments above with our actual numbers.

| Hardware | NUM_ENVS / GPU | Iter time | Util | VRAM used | Source |
|---|---|---:|---:|---:|---|
| 8 × H200 SXM 144 GB | 16,384 | 6.9 s | 87% | 48 GB | `train-on-cloud.md` §A.4 |
| 8 × A100 SXM 80 GB | 3,072 | ? | ? | ? | `training_runs.md` (sonic_x2_ultra_bones_seed-20260420) |
| 8 × B200 SXM 192 GB | 24,576 | TBD | TBD | TBD | run `bones_seed_sphere_feet-20260501` (this run) |
| 1 × H200 SXM 144 GB | 8,192 | TBD | TBD | TBD | typical 1-GPU sphere-feet fine-tune |

Once the current run reaches steady state (iter ~500), refresh the B200
row above with measured numbers.
