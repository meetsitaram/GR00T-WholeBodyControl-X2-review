# Fine-tune SONIC X2 on a New Motion Corpus

This guide walks through fine-tuning a pre-trained SONIC X2 Ultra policy on
a **new set of motions** — for example, a demo-specific motion playlist, a
fresh capture of MC gestures, a set of retargeted dances, or any
combination. It covers both the **first-time** fine-tune from a public
warm-start checkpoint and the **iteration** pattern of resuming from a
previous fine-tune with a wider domain-randomization envelope.

The guide assumes you have:

- A trained SONIC X2 warm-start checkpoint on disk (e.g. the public H200
  25k sphere-feet checkpoint at `~/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt`).
- IsaacLab installed and `env_isaaclab` conda env activated. See
  [`installation_training.md`](../getting_started/installation_training.md)
  for the canonical setup.
- A GPU with ≥ 32 GB VRAM (RTX 5090 is the reference local box; H200 for
  cloud).

> **Related docs**
> - [`training.md`](training.md) — upstream SONIC training architecture and
>   the original 30k-iter from-scratch recipe.
> - [`train-on-cloud.md`](train-on-cloud.md) — multi-GPU cloud workflow if
>   your corpus is too big for a single-GPU fine-tune.
> - [`training_data.md`](training_data.md) — BONES-SEED motion library and
>   retargeting pipeline (where many of your source motions come from).
> - [`x2_sonic_deploy_real.md`](x2_sonic_deploy_real.md) — what to do with
>   the trained checkpoint (ONNX export, sim parity, real-robot deploy).
>
> **Operator-side cheat sheets** (concrete commands, in-repo):
> - `gear_sonic/data/motions/demo_v1_sources/SKILL.md` — step-by-step
>   playbook for the first-time fine-tune.
> - `gear_sonic/data/motions/demo_v1_sources/ITERATE.md` — iteration
>   playbook (resume from previous checkpoint + DR ramp).

## When to fine-tune (vs. train from scratch)

| Scenario | Fine-tune | Train from scratch |
|---|:---:|:---:|
| Demo-specific motion corpus (50–500 entries) | ✓ | |
| Adding a few new motion categories to an existing skillset | ✓ | |
| Bigger DR / sim2real ramp on an existing policy | ✓ | |
| New robot URDF or actuator config | | ✓ |
| New reward function or termination rules | | ✓ |
| First time training on this robot embodiment | | ✓ |

For X2 Ultra, the public warm-start checkpoint is trained on 2,550 BONES-SEED
motions for 25k iters on 8×H200. Fine-tuning preserves that broad skillset
and biases the policy toward your demo motions in **another 4k iters on a
single GPU** (~3.5 h wall-clock). Training from scratch on a 100-motion
corpus is the wrong tool — it would converge to a much narrower policy and
take 10× longer.

## Pipeline at a glance

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Stage motion │ -> │ Merge into   │ -> │ Author exp   │ -> │ Launch       │
│ source PKLs  │    │ training PKL │    │ yaml +       │    │ training     │
│ as symlinks  │    │ (auto-prefix │    │ launcher     │    │ (~3.5 h on   │
│              │    │  + validate) │    │              │    │  RTX 5090)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
                                                            ┌──────────────┐
                                                            │ Eval +       │
                                                            │ ONNX export  │
                                                            │ + deploy     │
                                                            └──────────────┘
```

Each stage is one shell command. The whole loop is ~10 min of operator
work to launch + 3.5 h training + ~5 min ONNX export.

## Stage 1 — Collect your motion sources

SONIC fine-tuning consumes a single **motion-lib PKL** (a Python pickle
holding `{motion_name: motion_dict}`). Each motion_dict has at minimum:

```python
{
    "dof":               np.ndarray,  # (T, 31) joint targets, X2 Ultra MJ order
    "root_trans_offset": np.ndarray,  # (T, 3)  world-frame xyz
    "root_rot":          np.ndarray,  # (T, 4)  scipy xyzw quaternion
    "fps":               float,       # 30.0 for SONIC
    # Optional (auto-synthesized by the merger if missing):
    "pose_aa":           np.ndarray,  # (T, J, 3) axis-angle per joint
    "smpl_joints":       np.ndarray,  # (T, K, 3) FK joint positions
}
```

Typical sources:

| Source type | How to produce |
|---|---|
| **BONES-SEED retargeted motion** | [`training_data.md`](training_data.md) covers the SOMA retargeter end-to-end. Each clip lands as a `.pkl` under `gear_sonic/data/motions/` already in motion-lib format. |
| **MC gesture captures** (real X2) | `gear_sonic/scripts/teleop_x2_kinematic.py` records gestures; per-take PKLs land under `gear_sonic/data/motions/mc_gestures/`. |
| **Stitched warehouse motion** (closed-loop walks etc.) | `gear_sonic/scripts/make_warehouse_motion.py` from a playlist YAML — see `gear_sonic/data/motions/playlists/` for examples. |
| **SOMA chain-matched CSVs** | Pre-convert with `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` (downsamples 120→30 fps). Output is a per-clip PKL. |
| **Aggregate PKLs** (e.g. `x2_ultra_dances.pkl` with 34 entries) | Drop in as-is; the merger respects each entry inside. |

You don't need to pre-merge — that's Stage 2's job. You just need to know
where the source PKLs live.

## Stage 2 — Stage and merge into a training PKL

The repo uses a **staging-directory** convention: create one subdir per
motion category and symlink each source PKL into the matching subdir. The
merger walks the staging dir, prefixes every motion key with the subdir
name (e.g. `dances/x2_ultra_dances.pkl[foo]` → `dance__foo`), and writes
one consolidated PKL.

```bash
STAGE=gear_sonic/data/motions/<CORPUS>_sources
mkdir -p "$STAGE"/{mc_gestures,retargeted,combat_chain_matched,dances,body_check}

# Example: stage 6 walks
for f in /home/.../*.pkl; do
    ln -sf "$(realpath "$f")" "$STAGE/retargeted/$(basename "$f")"
done

# Build the merged training PKL
conda run -n env_isaaclab --no-capture-output python \
  gear_sonic/data_process/build_x2_demo_motion_lib.py \
    --stage-dir "$STAGE" \
    --out gear_sonic/data/motions/x2_ultra_<CORPUS>.pkl
```

The merger:
- Validates every source PKL has the required schema.
- Asserts no key collisions after prefixing (catches duplicates between
  bundle PKLs and per-clip PKLs).
- Auto-synthesizes `pose_aa` and `smpl_joints` for entries missing them
  (lossless remap of `dof + root_rot`, no FK).

See `gear_sonic/data/motions/demo_v1_sources/SKILL.md` Step 2 for the
full subdir→prefix table and the SOMA CSV one-liner.

## Stage 3 — Author the experiment yaml and launcher

The whole experiment config is a single Hydra yaml that inherits from
`sonic_x2_ultra_bones_seed_sphere_feet`. For the common case you only edit
4 fields: corpus name, wandb project, motion file path, iteration count.

```bash
EXP_DIR=gear_sonic/config/exp/manager/universal_token/all_modes
cp $EXP_DIR/sonic_x2_ultra_demo_v1.yaml $EXP_DIR/sonic_x2_ultra_<CORPUS>.yaml
# edit exp_var, project_name, motion_file, num_learning_iterations
```

And clone the local launcher script:

```bash
cp gear_sonic/scripts/run_local_finetune_demo_v1.sh \
   gear_sonic/scripts/run_local_finetune_<CORPUS>.sh
# edit MOTION_FILE, EXP_NAME, LOG_FILE env vars
```

Both files have inline comments explaining every editable field. Default
sizing is RTX 5090 (32 GB VRAM, 3072 envs, 4000 iters).

## Stage 4 — Verify the config resolves, then launch

A 1-second Hydra dry-compose catches typos before you commit to a 3.5 h
training run:

```bash
conda run -n env_isaaclab --no-capture-output python \
  gear_sonic/train_agent_trl.py \
    --config-name=base \
    "+exp=manager/universal_token/all_modes/sonic_x2_ultra_<CORPUS>" \
    "num_envs=4" "use_wandb=False" \
    --cfg job 2>&1 | grep -E 'motion_file|num_learning_iterations|project_name'
```

If the resolved `motion_file` points at your new PKL and exists on disk,
launch:

```bash
setsid nohup bash gear_sonic/scripts/run_local_finetune_<CORPUS>.sh \
    </dev/null >/dev/null 2>&1 &
echo $! > ~/sonic_<CORPUS>.pid
tail -f ~/sonic_<CORPUS>.log
```

The first PPO iter lands ~90 s after launch once Isaac Sim is initialized.
Checkpoints save every 2000 iters; a 4000-iter run produces
`model_step_002000.pt`, `model_step_004000.pt`, and a rolling `last.pt`.

## Stage 5 — Evaluate and export

Three checks before considering the run shippable to a real robot:

```bash
RUN=logs_rl/TRL_X2Ultra_<TitleCase>/.../sonic_x2_ultra_<CORPUS>_<exp_var>-<TS>

# (a) Demo-corpus benchmark — does the policy track the motions we trained on?
python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint "$RUN/model_step_004000.pt" \
  --motion gear_sonic/data/motions/x2_ultra_<CORPUS>.pkl

# (b) Regression check — has the broader skillset (e.g. walking) degraded?
python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint "$RUN/model_step_004000.pt" \
  --motion gear_sonic/data/motions/x2_ultra_bones_seed.pkl --sample 50

# (c) ONNX export + sim parity validation
bash gear_sonic_deploy/scripts/reexport_x2_onnx.sh "$RUN" \
    "$RUN/exported/model_step_004000_g1.onnx"
```

The ONNX exporter validates that `max|onnx − pt| < 1e-3 rad` before
overwriting the deploy artifact — if it fails the validation, it refuses
to promote and leaves the previous ONNX intact. See
[`x2_sonic_deploy_real.md`](x2_sonic_deploy_real.md) for the real-robot
deploy chain after that.

---

## Iterating: continue from a previous fine-tune

The base flow above produces a "v1" checkpoint. In practice you'll
typically want to iterate — drop bad motions, add more domain
randomization, or just run more iterations. The repo supports this as a
first-class pattern with a 4-axis decomposition:

```
┌─────────────────────┬────────────────────────────────────────────────────┐
│ Axis                │ Where you change it                                │
├─────────────────────┼────────────────────────────────────────────────────┤
│ Corpus              │ Filter the v1 PKL or re-merge the staging dir      │
│ Checkpoint          │ EXTRA_FLAGS=+checkpoint=... in the launcher        │
│ Domain randomization│ events/tracking/*.yaml + observations/policy/*.yaml│
│ Iteration count     │ algo.config.num_learning_iterations in exp yaml    │
└─────────────────────┴────────────────────────────────────────────────────┘
```

### When to iterate vs. start fresh

Iterate when the policy network output stays meaningful for the new env:
< 30 % corpus delta, same reward/termination, same robot config. Start
fresh when env semantics shift (new reward, new URDF, new obs space).

### Domain randomization knobs (most useful axis for real-robot prep)

The default `level0_4` events preset (in
`gear_sonic/config/manager_env/events/tracking/level0_4.yaml`) already
applies friction, mass, CoM, joint-bias, and push randomization. The
default policy observation config (`local_dir_hist.yaml`) already applies
additive uniform noise to `gravity_dir`, `base_ang_vel`, `joint_pos`,
`joint_vel`.

To **widen observation noise**: clone `local_dir_hist.yaml` with bigger
`n_min`/`n_max` values, then override in your exp yaml:

```yaml
defaults:
  - /exp/manager/universal_token/all_modes/sonic_x2_ultra_bones_seed_sphere_feet
  - override /manager_env/observations/policy: local_dir_hist_v2
  - _self_
```

To **add KP/KD motor-gain randomization**: IsaacLab provides
`randomize_actuator_gains` out of the box. Drop a 12-line yaml term at
`events/terms/randomize_actuator_gains.yaml`, wire it into a new events
preset (`events/tracking/level0_4_pd.yaml`), and override in the exp yaml.
**One catch**: the `EventCfg` class in `gear_sonic/envs/manager_env/mdp/events.py`
has hardcoded field slots — you must add `randomize_actuator_gains = None`
to the class, otherwise Hydra fails with:

```
TypeError: EventCfg.__init__() got an unexpected keyword argument 'randomize_actuator_gains'
```

Same gotcha applies to any **new** event term you add via this preset
mechanism (the existing ones in `level0_4` are already declared).

### DR scale rule of thumb

| Goal | Obs noise scale vs `level0_4` | Actuator KP/KD scale |
|---|---|---|
| Conservative (lowest regression risk) | 1.5× | ±10 % |
| Sim2real moderate (recommended) | 2× | ±15 % |
| Sim2real aggressive | 3× | ±25 % |
| Maximum robustness | 4–5× | ±35 % |

Aggressive DR usually needs **more iterations** to converge — budget at
least 1.5× the iter count of the previous tighter run.

### Iteration-count behavior on resume

The iter counter **resets to 0** when you `+checkpoint=...` resume — only
the weights and obs-normalizer stats persist. A 4k-iter run resumed from a
prior 4k-iter run will emit `model_step_002000.pt` and
`model_step_004000.pt` in the new run dir, representing cumulative iters
6000 and 8000. The adaptive sampler state does NOT persist across resume,
but it rediscovers hard motions within ~200 iters.

### Worked example: demo_v1 → demo_v2 (2026-06-24)

Trigger: at end of demo_v1 training the adaptive sampler was concentrating
40× on the hardest motion (`prob_max_over_uniform=40.2`) across 18
concentrated bins. Inspection showed it was the 6 sit-on-chair motions —
the IsaacLab training env has no chair geometry, so those motions trained
against gravity-only support and could never be learned. Result: ~40 % of
training episodes were on impossible motions.

Demo_v2 changes:

| Knob | demo_v1 | demo_v2 |
|---|---|---|
| Corpus | 101 entries | 95 entries (sit-stand dropped) |
| Starting checkpoint | `model_step_025000.pt` (warm-start) | `model_step_004000.pt` (demo_v1 final) |
| Obs noise | level0_4 defaults | 2× across `gravity_dir / base_ang_vel / joint_pos / joint_vel` |
| Actuator KP/KD scale | fixed | 0.85–1.15× per episode |
| Events preset | `level0_4` | `level0_4_pd` (new) |
| Iterations | 4000 | 4000 (cumulative 8k since warm-start) |

End-to-end operator work: **~10 min** (filter PKL, author 4 yamls, edit
launcher, patch `EventCfg`, dry-compose, launch). Training itself was
identical wall-clock (~3.5 h, 3.0 s/iter — no slowdown from the wider DR).

For the full step-by-step commands, including the `EventCfg` patch and the
4 new yaml files, see `gear_sonic/data/motions/demo_v1_sources/ITERATE.md`.

---

## Common issues

| Symptom | Root cause | Fix |
|---|---|---|
| `KeyError: 'pose_aa'` in motion_lib loading | Source PKL is missing `pose_aa`/`smpl_joints` (typical for warehouse-stitched motions) | Run your sources through `build_x2_demo_motion_lib.py` — it auto-synthesizes both. |
| Merger `RuntimeError: key collision on '<prefix>__<name>'` | Two staged PKLs contribute the same fully-prefixed key | Drop one source. Common with aggregate PKLs whose entries duplicate per-clip PKLs in a sibling subdir. |
| Adaptive sampler `prob_max_over_uniform > 25` at end of training | Some motion in the corpus is untrainable in the current env (e.g. sit-on-chair without a chair, jumping without enough air time) | Inspect which motion is failing, filter it out, iterate. See demo_v2 worked example above. |
| `OutOfMemoryError` early in Isaac init | `num_envs` too high for the GPU | Drop 25–50 %. Heuristic: 3072 fits 32 GB, 16384 fits 80 GB. |
| `TypeError: EventCfg.__init__() got an unexpected keyword argument '<term>'` | Added a new event term without declaring its slot in `EventCfg` | Add `<term_name> = None` to the `EventCfg` configclass in `gear_sonic/envs/manager_env/mdp/events.py`. |
| Log has every line doubled | Both nohup redirect and the launcher's internal `tee` wrote to the same file | Don't add `>>$LOG` to the nohup line — `run_local_finetune_*.sh` handles logging via its own `tee`. |
| Trained policy regresses on walking after a demo-corpus fine-tune | The new corpus had no walking motions — catastrophic forgetting on the broader skillset | Add a few representative walks to the corpus before re-running, or accept the regression if walking isn't a demo requirement. The regression eval in Stage 5 (b) quantifies it. |
| `NCCL WARN Cuda failure 'invalid argument'` at the first DDP collective on a ≥4-GPU cloud run | CUDA driver / bundled-NCCL minor-version mismatch interacting with IsaacSim CUDA-context init | Already fixed in-tree via an NCCL prime-barrier in `train_agent_trl.py`; if the patch is missing, see [`train-on-cloud.md` §B.15](train-on-cloud.md). Does not affect single-GPU local fine-tunes. |

## Compute and cost ballpark

| Setup | Per-run wall-clock (4k iters, 100-entry corpus) | Cost |
|---|---|---|
| RTX 5090 (local) | ~3.5 h | electricity only |
| Single H200 (cloud) | ~45 min | $5–10 |
| 8×H200 (cloud) | ~10 min | $40–80 |

The 8×H200 cloud setup pays off when training a from-scratch policy
(30k+ iters on the full 142k BONES-SEED library — see
[`train-on-cloud.md`](train-on-cloud.md)). For a 4k-iter fine-tune on a
100-motion corpus, the cloud bootstrap overhead (~15 min) eats most of the
8× throughput win, so local is usually the right call.

## Next steps

After a successful fine-tune you'll typically want to:

1. **Sim-test the new policy** — `gear_sonic_deploy/deploy_x2.sh sim
   --model <new.onnx> --motion <pkl>` for each demo motion. See
   `play_pkl_motions_commands.md` in the repo root for the canonical
   command pattern.
2. **Build a stitched demo playlist** — chain your confirmed-working
   motions into a single playlist YAML
   (`gear_sonic/data/motions/playlists/<corpus>_winners.yaml`) for an
   end-to-end demo bake.
3. **Real-robot deploy** — follow
   [`x2_sonic_deploy_real.md`](x2_sonic_deploy_real.md) from the ONNX
   exported in Stage 5 (c).
4. **Iterate** — drop motions that failed, add domain randomization, run
   another 4k iters. See the "Iterating" section above and the
   `ITERATE.md` operator playbook for the full pattern.
