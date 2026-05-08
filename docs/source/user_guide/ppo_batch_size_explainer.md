# PPO Batch Size, Sample Efficiency, and Compute Trade-offs

A pedagogical reference for understanding how rollout buffer size,
mini-batch size, and number of parallel envs interact in our X2 Ultra
PPO training, and what the W&B metrics tell you about whether your
batch is too small / too big / just right.

> **Audience:** anyone deciding `NUM_ENVS` and `NUM_PROCESSES` for a new
> training run, or trying to interpret why two runs at different batch
> sizes converge differently.
>
> **Companion docs:**
> - [`gpu_cost_comparison.md`](gpu_cost_comparison.md) — picks GPU
>   class for a fixed 8-GPU node.
> - [`compute_scaling_sonic_vs_ours.md`](compute_scaling_sonic_vs_ours.md)
>   — compares us with NVIDIA's 128-GPU SONIC scaling.

## TL;DR

- "Batch size" in PPO is **three numbers** that interact: parallel envs,
  rollout buffer, and mini-batch.
- Bigger rollout buffer → cleaner gradient → fewer iters to converge,
  but with **strong diminishing returns** above ~1M samples per update.
- Both our 8× H200 (3.15M) and 8× B200 (4.72M) configs sit **deep in
  the gradient-SNR plateau** — convergence quality is essentially
  identical; the gap is wall-clock and per-iter throughput.
- In W&B, watch **`Loss/approx_kl`** and **`Loss/value_loss`** — those
  are the early-warning signals if a batch is too small for the
  current learning rate.

## 1. The three batch sizes in PPO

In supervised learning "batch size" is one number. In on-policy RL it's
three:

```
For each PPO iteration:
  ┌─ COLLECTION ─────────────────────────────────────┐
  │  Spin up N_envs parallel environments.            │
  │  Each rolls out for T steps using current policy. │
  │  → "rollout buffer" of N_envs × T transitions     │
  └──────────────────────────────────────────────────┘
  ┌─ LEARNING ───────────────────────────────────────┐
  │  Repeat K_epochs times:                          │
  │    Shuffle rollout buffer.                       │
  │    Split into N_minibatches mini-batches.        │
  │    For each: compute gradient, clip, step Adam.  │
  └──────────────────────────────────────────────────┘
```

| Knob | What it controls | Affects |
|---|---|---|
| **`num_envs × num_processes`** (parallel envs) | how many sims run simultaneously | parallelism, GPU mem, throughput |
| **`num_envs × num_processes × num_steps_per_env`** (rollout buffer) | total fresh samples per gradient update | gradient noise, KL stability |
| **`rollout_buffer / num_mini_batches`** (mini-batch size) | size of each SGD step | optimization stability, peak GPU mem |
| **`num_learning_epochs`** | how many times each transition is reused | sample efficiency, off-policyness |

Our X2 Ultra config (`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml`):

```yaml
num_steps_per_env: 24       # T
num_learning_epochs: 5      # K
num_mini_batches: 4         # M
clip_param: 0.2             # PPO clip ε
actor_learning_rate: 2e-5   # Adam lr (actor head)
critic_learning_rate: 1e-3  # Adam lr (critic head)
```

For the two runs currently active:

| Run | `num_envs/GPU` × 8 GPUs | × T=24 | Rollout buffer | Mini-batch (÷ 4) |
|---|---:|---:|---:|---:|
| 8× H200 | 16,384 × 8 = 131,072 | × 24 | **3,145,728** | 786,432 |
| 8× B200 | 24,576 × 8 = 196,608 | × 24 | **4,718,592** | 1,179,648 |

**B200's rollout buffer is 50% larger than H200's** (4.72M vs 3.15M).

## 2. Why bigger rollout ≠ proportionally better

Each transition is a noisy estimate of the true policy gradient. The
sample mean of N noisy estimates has variance $\sigma^2 / N$, so
**gradient noise scales as $1 / \sqrt{N}$**.

So doubling your rollout buffer:

- ✓ reduces gradient noise by **1.41×** (not 2×)
- ✗ also doubles wall-clock per iter (collection time is roughly linear
  in N_envs above the bottleneck threshold)
- ✗ doubles GPU memory pressure (rollout buffer must fit alongside model)

The break-even point — where bigger batch stops paying for itself — is
called the **gradient signal-to-noise ratio plateau**.

### The empirical plateau (Andrychowicz et al. 2021)

[Andrychowicz, Raichuk, Stańczyk, et al. 2021,
"What Matters in On-Policy Reinforcement Learning?"](https://arxiv.org/abs/2006.05990)
ran 250,000+ training runs across MuJoCo, Atari, and continuous control
benchmarks varying batch size, learning rate, clip range, etc.

Their headline finding for **batch size scaling** in PPO:

| Regime | Samples per gradient update | Behavior |
|---|---|---|
| **Undersampled** | < 64K | Noisy gradients → high variance, may diverge with normal lr |
| **Healthy zone** | 64K – 1M | Sweet spot; cleaner gradients without much overhead |
| **Plateau** | 1M – 10M | Same final policy quality, just wasted compute |
| **Wasted** | > 10M | No measurable benefit from more samples |

The plateau exists because **at some point the gradient direction is
already known precisely** — adding more samples just refines a number
that was already accurate to 4 decimal places. The optimizer can't act
on more precision than the policy update step (clipped at $\epsilon=0.2$)
permits.

### Where do we sit?

```
     │←── undersampled ──→│←──── healthy ────→│←──── plateau ────→│ wasted
     │                    │                    │                    │
  10K              100K               1M                10M               100M
                                                                                    samples
                                                                                  per update
                                       ↑                ↑
                                       H200 (3.15M) ───┘
                                       B200 (4.72M) ───────┘
```

**Both our runs are deep in the plateau.** The H200 run isn't
under-batched; it's just less wastefully over-batched. Convergence
quality between 3.15M and 4.72M is essentially indistinguishable in
the published Andrychowicz curves.

## 3. The trade-off you're really making

There are two competing axes you optimize over when picking
`num_envs`:

```
Wall-clock per iter ─────────►  bigger batch = slower per iter
                                 (more env steps to collect)
        ↕ trade-off
Quality per iter   ─────────►  bigger batch = cleaner gradient
                                (less noise in policy update)
```

For a fixed compute budget (e.g. "20K iters total"), the optimal
batch size is **just above the plateau knee** — about 1-2M samples
per update. Past that, you're paying wall-clock for noise reduction
you can't use.

### Sample efficiency vs throughput

Two ways to measure compute efficiency:

| Metric | Definition | What it tells you |
|---|---|---|
| **Throughput** | env-steps / sec | How fast can I generate experience? |
| **Sample efficiency** | reward / env-step | How much policy improvement per experience? |

Bigger batches help **sample efficiency** (cleaner gradient → larger
effective lr → more progress per iter), but only up to the SNR plateau.

Bigger batches hurt **throughput** (linear in N_envs above the
bottleneck threshold) so you're trading wall-clock for cleaner
gradients you don't need.

### When to actually want a bigger batch

The plateau argument breaks down in three cases:

1. **Hard exploration problems** (sparse rewards, multi-task) — the
   policy gradient itself is high-variance because few transitions
   contain useful learning signal. A larger batch doesn't reduce per-
   sample noise but does ensure you *include* the rare informative
   transitions. SONIC's 524K batch on 100M+ frames is partly explained
   by this — many motion clips are unusual / underrepresented.
2. **Late-stage fine-tuning** — when policy changes per iter are
   already tiny (small KL), gradient noise relative to step size is
   high. A bigger batch helps stabilize the final descent.
3. **Brittle learning rate** — if you're using a high lr (e.g. for
   faster convergence), the trust region (PPO clip) is tight and you
   need cleaner gradients to avoid clipping every step.

For our X2 Ultra setup, we're not hitting any of these. Both batches
are way past the plateau knee; the bigger lever is probably **lr
schedule**, not batch size.

## 4. KL divergence and the trust region

PPO's "trust region" is enforced by the clip mechanism:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t \hat{A}_t,\, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$ is the
importance ratio.

The clip is a **soft trust region**. Two failure modes show up in
metrics:

### Symptom: `approx_kl` spikes

$$\text{approx\_kl} = \mathbb{E}[\log \pi_{\text{old}} - \log \pi_\theta]$$

This measures how far the new policy moved from the rollout policy.

- **Normal:** 0.005 – 0.05 per iter
- **Concerning:** 0.05 – 0.1 per iter sustained
- **Diverging:** > 0.1 spikes

What it means: your gradient noise is high enough that the optimizer
is taking large steps in random directions. The PPO clip is preventing
catastrophic collapse, but you're not making real progress — you're
just oscillating in policy space.

**Cause:** rollout buffer too small for current lr. Either
- **(a)** increase batch size (or num_envs), or
- **(b)** decrease lr, or
- **(c)** decrease `clip_param` (tighter trust region)

### Symptom: `value_loss` spikes

The critic head learns a value estimator $V(s)$ from rollout returns.
With a noisy rollout buffer, the critic targets are themselves noisy,
and the critic loss curve becomes spiky.

- **Normal:** smooth monotonic decrease (with occasional bumps when
  the policy explores new state regions)
- **Concerning:** sawtooth pattern, never settles
- **Diverging:** value loss grows over time

**Cause:** same as above — noisy rollouts. Critic typically suffers
*before* actor because critic is trained directly on rollouts (no
trust region clip).

### Symptom: `entropy_loss` increases

Mean policy entropy normally decays smoothly as the policy commits to
better actions:

- **Normal:** smooth decay from initial $\sim$0 (Gaussian std=1) toward
  more negative values (sharper policies) over training
- **Concerning:** entropy oscillates or *grows*
- **Diverging:** entropy keeps growing → policy unable to commit

**Cause:** the optimizer is fighting noise. Each gradient step pushes
the policy toward random actions to "explore around" the noisy
gradient signal. Indicates batch-size / lr mismatch.

## 5. W&B metrics cheat sheet — what to watch

For our two runs, here's what to compare side-by-side in W&B:

| Metric (in W&B) | Healthy band | What B200 should look like | What H200 might look like |
|---|---|---|---|
| `Loss/approx_kl` | 0.005 – 0.05 | smooth, possibly slightly lower | possibly slightly higher (smaller batch = noisier gradient) |
| `Loss/value_loss` | smooth monotonic ↓ | smooth | possibly bumpier early, smoothing by iter 200+ |
| `Loss/entropy_loss` | smooth monotonic decay | smooth | smooth |
| `Loss/policy_loss` | small magnitude, oscillating around 0 | tight oscillation | wider oscillation |
| `Loss/clip_fraction` | 0.1 – 0.3 | ~0.15 | possibly ~0.2 (more clipping = more noise) |
| `Train/Mean_rewards` | smooth ascent | smooth | smooth (after iter ~100 warmup) |
| `Train/Mean_episode_length` | smooth ascent | smooth | smooth |
| `Env/Episode_Termination/foot_pos_xyz` | should ↓ over training | smooth ↓ | smooth ↓ |

The **smaller batch → noisier curves** prediction is mild but
measurable. If the H200 curves look essentially identical to B200,
that's confirmation we're solidly in the SNR plateau and the choice
of 16K vs 24K envs/GPU doesn't materially affect convergence quality.

If the H200 curves look noticeably worse (e.g. KL > 0.05 sustained,
or rewards oscillate), that's a sign 3.15M is uncomfortably close to
the plateau knee for our specific reward structure (multi-task tracking
with adaptive sampling), and we should consider:

- bumping `num_envs/GPU` up toward 24K on H200 (we have memory headroom)
- decreasing `actor_learning_rate` (currently 2e-5)
- decreasing `clip_param` (currently 0.2)

## 6. The 3 levers, ranked by effect on convergence

If you have to pick *one* knob to tune:

### Most impactful: `actor_learning_rate`
The lr × clip-param × batch-size product determines effective trust-
region size per iter. Halving lr is roughly equivalent to 4× batch
size in terms of update stability.

### Medium: `num_envs` (rollout buffer size)
Diminishing returns above ~1M samples/update, but bigger does help
*if* you're in or below the healthy zone.

### Smallest: `num_mini_batches` and `num_learning_epochs`
These control sample reuse within an iter. PPO is robust to a wide
range here (M=4-16, K=4-10). Don't tune unless you have a specific
hypothesis.

## 7. Practical takeaways for our setup

1. **Both current runs (3.15M vs 4.72M) are in the plateau.** The H200
   isn't under-batched and the B200 isn't getting a meaningful quality
   boost from the bigger rollout.
2. **Compare W&B `approx_kl` curves directly** to confirm. If both
   stay in 0.005-0.05 band, the batch difference is
   convergence-irrelevant.
3. **For future runs:** if you want to push batch higher, do it on the
   B200 (memory budget) and target ~6-8M samples per update — that's
   the sweet spot for "comfortably plateau, not wasteful."
4. **If you want to push batch lower** (e.g. to fit more concurrent
   sweep configs on a budget), don't go below ~1M samples per update.
   That's `num_envs/GPU × 8 × 24 ≥ 1M` → `num_envs/GPU ≥ 5,200`.
5. **Wall-clock matters more than batch size** at our scale. The
   B200's bigger batch is essentially "free" in compute terms (came
   along with bigger memory budget); the wall-clock cost of *also*
   running 50% more envs is the actual trade-off.

## References

- **Andrychowicz et al. 2021**, *"What Matters in On-Policy
  Reinforcement Learning? A Large-Scale Empirical Study"*,
  arxiv: [2006.05990](https://arxiv.org/abs/2006.05990).
  - §6.4: Batch size and number of parallel environments.
  - §6.5: Number of epochs / mini-batches.
- **Schulman et al. 2017**, *"Proximal Policy Optimization
  Algorithms"*, arxiv: [1707.06347](https://arxiv.org/abs/1707.06347)
  — the original PPO paper, defines clip and KL terms.
- **McCandlish et al. 2018**, *"An Empirical Model of Large-Batch
  Training"*, arxiv: [1812.06162](https://arxiv.org/abs/1812.06162)
  — derives the gradient-SNR plateau theoretically (originally for
  supervised learning, but the math carries over to on-policy RL).
- **NVIDIA SONIC**, arxiv: [2511.07820](https://arxiv.org/abs/2511.07820)
  — uses 524K batch on 128 GPUs for 9K GPU-hours; see
  [`compute_scaling_sonic_vs_ours.md`](compute_scaling_sonic_vs_ours.md)
  for our paraphrase of why they picked that operating point.
- **Our config:**
  `gear_sonic/config/algo/ppo_im_phc.yaml` (PPO hyperparams),
  `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml`
  (rollout/env config).
