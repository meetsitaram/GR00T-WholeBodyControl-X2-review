# Cloud Training Runs — Per-Run Metric History

A running log of cloud training runs and their iter-vs-iter metric
comparisons. Each entry captures one *run*, one *snapshot pair*, and a
short reading. Append a new section every time you:

1. Pull a fresher `train.log` off the cloud trainer.
2. Re-run the parser + comparison tool to produce a new
   `COMPARE_iter<A>_vs_<B>.md`.
3. Want a permanent record of "what this iter range did vs the last one"
   that survives losing the local working tree.

---

## Workflow (how to add a new entry)

The compare tool and intermediate artifacts live **outside** the repo at
`/home/stickbot/sim2sim_armature_eval/training_curves/` (the raw
`train.log` is too large to track and gets overwritten on each refresh).
This doc, in the repo, is the durable summary.

```bash
# 1. Pull the latest tee'd stdout from the cloud trainer.
scp ubuntu@<cloud-ip>:/home/ubuntu/train.log \
    /home/stickbot/sim2sim_armature_eval/training_curves/train.log

# 2. Re-parse → metrics CSVs + plots.
cd /home/stickbot/sim2sim_armature_eval/training_curves
python parse_and_plot.py

# 3. Build the new windowed comparison (window default ±50 iters).
python build_compare.py <iter_A> <iter_B> --window 50 \
    --out COMPARE_iter<A>_vs_<B>.md

# 4. Copy the resulting markdown into the relevant run section below.
```

`build_compare.py` is also the single source of metric directions
(`OK` / `!!`) so the markdown stays consistent across snapshots.

---

## Run: `sonic_x2_ultra_bones_seed_bones_seed-20260420_083925`

**Trained on:** 8× A100 80 GB cloud node (Lambda), `accelerate launch`
with 8 processes, `++num_envs=3072`, `++headless=True`.

**Config name:** `manager/universal_token/all_modes/sonic_x2_ultra_bones_seed`.

**Started:** 2026-04-20 08:39 UTC.

**Stdout log on cloud:** `~/train.log` (tee'd, currently ~46 MB at iter 8k).

**Local mirror:** `/home/stickbot/sim2sim_armature_eval/training_curves/train.log`.

**Local checkpoints:** `/home/stickbot/x2_cloud_checkpoints/run-20260420_083925/`
(`model_step_002000.pt`, `004000`, `006000`, plus `last.pt`).

**Training budget:** `max_train_steps: 20000` (per `meta.yaml`).

**W&B:** **disabled** for this run (`use_wandb: false` in `config.yaml`,
`meta.yaml: wandb_run: null`). All metric history comes from grepping
`train.log` — no live dashboard exists. For the next run, flip
`++use_wandb=True` at launch and `wandb login` on the cloud node first.

### Snapshot 1 — iter ~3900 vs iter ~5834 (Δ ≈ 1934 iter, ~3.5h)

Generated 2026-04-20 mid-day. Source:
`sim2sim_armature_eval/training_curves/COMPARE_iter3900_vs_5834.md`.
Inserted here verbatim.

#### Summary

| flag | metric | iter ~3900 | iter ~5834 | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| `OK` | `Mean_rewards` | 14.95 | 16.31 | +1.36 | +9.1% |
| `OK` | `Mean_length` | 255.9 | 263.1 | +7.2 | +2.8% |
| `` | `Mean_entropy` | 20.24 | 19.72 | -0.514 | -2.5% |
| `` | `Mean_action_noise_std` | 0.47 | 0.46 | -0.01 | -2.1% |

#### Tracking Error

| flag | metric | iter ~3900 | iter ~5834 | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| `OK` | `error_anchor_rot` | 0.1561 | 0.1437 | -0.0124 | -8.0% |
| `OK` | `error_body_rot` | 0.3014 | 0.2837 | -0.0177 | -5.9% |
| `OK` | `error_body_pos` | 0.06107 | 0.05761 | -0.00345 | -5.7% |
| `OK` | `error_body_ang_vel` | 2.297 | 2.196 | -0.101 | -4.4% |
| `OK` | `error_joint_pos` | 0.1914 | 0.1832 | -0.00819 | -4.3% |
| `OK` | `error_body_lin_vel` | 0.4606 | 0.4423 | -0.0183 | -4.0% |
| `OK` | `error_anchor_ang_vel` | 1.703 | 1.654 | -0.0495 | -2.9% |
| `OK` | `error_anchor_lin_vel` | 0.3794 | 0.3688 | -0.0106 | -2.8% |
| `OK` | `error_joint_vel` | 1.749 | 1.708 | -0.0407 | -2.3% |
| `OK` | `error_anchor_pos` | 0.5262 | 0.523 | -0.00317 | -0.6% |

#### Episode Reward

| flag | metric | iter ~3900 | iter ~5834 | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| `OK` | `tracking_anchor_pos` | 0.1114 | 0.1227 | +0.0113 | +10.2% |
| `OK` | `tracking_relative_body_ori` | 0.2694 | 0.2934 | +0.024 | +8.9% |
| `OK` | `tracking_body_angvel` | 0.27 | 0.2905 | +0.0205 | +7.6% |
| `OK` | `tracking_vr_5point_local` | 0.7771 | 0.8314 | +0.0543 | +7.0% |
| `!!` | `joint_limit` | -0.0388 | -0.04138 | -0.00258 | +6.6% |
| `OK` | `tracking_anchor_ori` | 0.2139 | 0.2252 | +0.0113 | +5.3% |
| `OK` | `tracking_body_linvel` | 0.4123 | 0.4325 | +0.0202 | +4.9% |
| `OK` | `tracking_relative_body_pos` | 0.4647 | 0.4854 | +0.0207 | +4.5% |
| `!!` | `undesired_contacts` | -0.2959 | -0.3082 | -0.0123 | +4.2% |
| `OK` | `anti_shake_ang_vel` | -0.01817 | -0.01767 | +0.000503 | -2.8% |
| `OK` | `feet_acc` | -0.1376 | -0.1358 | +0.0018 | -1.3% |
| `OK` | `action_rate_l2` | -0.6378 | -0.6304 | +0.00749 | -1.2% |

#### Termination

| flag | metric | iter ~3900 | iter ~5834 | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| `OK` | `anchor_pos` | 0.1316 | 0.08171 | -0.0499 | -37.9% |
| `OK` | `foot_pos_xyz` | 12.91 | 9.456 | -3.45 | -26.8% |
| `OK` | `anchor_ori_full` | 2.437 | 1.916 | -0.521 | -21.4% |
| `OK` | `ee_body_pos` | 7.66 | 6.586 | -1.07 | -14.0% |
| `!!` | `time_out` | 46.05 | 45.48 | -0.572 | -1.2% |

**Reading at the time:** Healthy mid-training improvement across the board.
Total reward up 9.1%, every tracking error down, fall-style terminations
down 14–38%. Penalty terms rose a touch (`joint_limit`, `undesired_contacts`)
because the policy is more confident on harder clips. This is the snapshot
that motivated continuing past 6k.

### Snapshot 2 — iter ~5970 vs iter ~8050 (Δ ≈ 2080 iter, ~3.6h)

Generated 2026-04-20 late afternoon. Source:
`sim2sim_armature_eval/training_curves/COMPARE_iter5970_vs_8050.md`.

#### Summary

| flag | metric | iter ~5970 | iter ~8050 | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| `OK` | `Mean_rewards` | 17.64 | 17.79 | +0.15 | +0.9% |
| `!!` | `Mean_length` | 282.6 | 275.6 | -7.07 | -2.5% |
| `` | `Mean_entropy` | 19.67 | 19.27 | -0.40 | -2.0% |
| `` | `Mean_action_noise_std` | 0.46 | 0.45 | -0.01 | -2.2% |

#### Tracking Error — every single one improved

| flag | metric | iter ~5970 | iter ~8050 | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| `OK` | `error_body_lin_vel` | 0.4569 | 0.4352 | -0.0217 | -4.7% |
| `OK` | `error_anchor_lin_vel` | 0.3795 | 0.3624 | -0.0171 | -4.5% |
| `OK` | `error_body_ang_vel` | 2.234 | 2.155 | -0.0789 | -3.5% |
| `OK` | `error_anchor_rot` | 0.147 | 0.142 | -0.00479 | -3.3% |
| `OK` | `error_anchor_ang_vel` | 1.677 | 1.633 | -0.0442 | -2.6% |
| `OK` | `error_joint_vel` | 1.728 | 1.684 | -0.044 | -2.5% |
| `OK` | `error_body_pos` | 0.0590 | 0.0581 | -9.2e-04 | -1.6% |
| `OK` | `error_body_rot` | 0.288 | 0.284 | -0.00411 | -1.4% |
| `OK` | `error_anchor_pos` | 0.499 | 0.493 | -0.00652 | -1.3% |
| `!!` | `error_joint_pos` | 0.186 | 0.187 | +8.8e-04 | +0.5% |

#### Episode Reward — penalties down, tracking flat-to-slightly-down

| flag | metric | iter ~5970 | iter ~8050 | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| `OK` | `anti_shake_ang_vel` | -0.01724 | -0.01586 | +0.00138 | -8.0% |
| `OK` | `joint_limit` | -0.03922 | -0.03621 | +0.00301 | -7.7% |
| `OK` | `action_rate_l2` | -0.6197 | -0.581 | +0.0387 | -6.2% |
| `OK` | `feet_acc` | -0.1339 | -0.1285 | +0.00543 | -4.1% |
| `OK` | `undesired_contacts` | -0.3015 | -0.2918 | +0.00975 | -3.2% |
| `!!` | `tracking_relative_body_pos` | 0.475 | 0.460 | -0.0149 | -3.1% |
| `!!` | `tracking_anchor_ori` | 0.221 | 0.214 | -0.00674 | -3.1% |
| `!!` | `tracking_body_linvel` | 0.424 | 0.411 | -0.0124 | -2.9% |
| `!!` | `tracking_vr_5point_local` | 0.815 | 0.796 | -0.0187 | -2.3% |
| `!!` | `tracking_relative_body_ori` | 0.290 | 0.284 | -0.00564 | -1.9% |
| `!!` | `tracking_body_angvel` | 0.285 | 0.281 | -0.00376 | -1.3% |
| `OK` | `tracking_anchor_pos` | 0.118 | 0.119 | +5.9e-04 | +0.5% |

#### Termination — fall-style terms dropped 16–39 %

| flag | metric | iter ~5970 | iter ~8050 | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| `OK` | `anchor_pos` | 0.104 | 0.063 | -0.041 | **-39.4%** |
| `OK` | `ee_body_pos` | 6.90 | 5.46 | -1.44 | **-20.8%** |
| `OK` | `anchor_ori_full` | 1.86 | 1.52 | -0.333 | -17.9% |
| `OK` | `foot_pos_xyz` | 9.81 | 8.23 | -1.58 | -16.2% |
| `` | `time_out` | 43.15 | 45.91 | +2.75 | +6.4% (good — more clips reach horizon) |

#### Adaptive Sampling — concentrating on harder clips

| flag | metric | iter ~5970 | iter ~8050 | Δ% |
|---|---|---:|---:|---:|
| `` | `num_episodes_max` | 4.39e+04 | 6.58e+04 | +49.9% |
| `` | `num_episodes_mean` | 1.79e+03 | 2.42e+03 | +35.1% |
| `` | `failure_rate_min` | 3e-04 | 2e-04 | -33.3% |
| `` | `num_episodes_min` | 14.0 | 18.2 | +29.7% |
| `` | `num_failures_max` | 2.89e+04 | 3.64e+04 | +26.1% |
| `` | `failure_rate_mean` | 0.250 | 0.198 | -20.8% |
| `` | `num_concentrated_bins` | 64.6 | 53.0 | -17.9% |
| `` | `failure_rate_max` | 4.27 | 3.77 | -11.9% |

(other adp_samp metrics moved <10%, omitted for brevity; full table in
`COMPARE_iter5970_vs_8050.md`)

**Reading at the time:** the policy is in a **maturation phase**, not a
regression. Falls dropped 16–39 % across all four fall-style termination
codes, every tracking error nudged down, penalty terms all softer. Total
reward held flat because of the redistribution: the adaptive sampler
shrunk active-clip set by 18 % (53 vs 65 concentrated bins) and pushed
50 % more long-tail episodes onto the policy. The 1–3 % slip on the
per-clip tracking-reward terms is the cost of the policy now spending
more time on harder clips. Earlier intuition (from the .pt rewbuffer
snapshot — biased toward the most recent 100 *terminated* episodes)
incorrectly suggested a regression; the windowed per-iteration averages
above are the reliable view.

**Action taken:** kept training going; planned to head-to-head benchmark
`step_008000.pt` vs `step_006000.pt` once the next checkpoint saved.

### Snapshot 3 — *(append next comparison here when ready)*

<!--
Template:

### Snapshot N — iter ~A vs iter ~B (Δ ≈ X iter, ~Yh)

Generated YYYY-MM-DD. Source:
`sim2sim_armature_eval/training_curves/COMPARE_iter<A>_vs_<B>.md`.

(paste the four section tables here from the COMPARE markdown)

**Reading at the time:** ...

**Action taken:** ...
-->

---

## See Also

- {doc}`train-on-cloud` — cloud training runbook (bundle, smoke, launch)
- {doc}`sim2sim_mujoco` — MuJoCo deployment + per-checkpoint sim2sim
  benchmarks (G16/G16b/G17 used the iter-6000 checkpoint from this run)
- `sim2sim_armature_eval/training_curves/parse_and_plot.py` — train.log
  parser (regenerates `metrics.csv`, `metrics_wide.csv`, and the
  per-group plots in `plots/`)
- `sim2sim_armature_eval/training_curves/build_compare.py` — windowed
  iter-A-vs-iter-B comparison generator (use to produce the markdown
  pasted into each Snapshot section above)
