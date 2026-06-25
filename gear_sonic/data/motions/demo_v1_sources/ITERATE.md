# Iterate on a previous SONIC X2 fine-tune (demo_v1 → demo_v2 pattern)

Companion doc to [`SKILL.md`](./SKILL.md) in this same directory. The SKILL
covers the **base** fine-tune workflow (stage → merge → exp yaml → launch).
This doc covers what to do **after that first run is done** and you want to
iterate: resume from the previous checkpoint, drop some motions, ramp up
domain randomization, etc.

If you've never run a fine-tune before, go read `SKILL.md` first — most of
this doc assumes you already have a `model_step_NNNN.pt` from a prior run
and a working corpus PKL.

> **Project-wide reference**: a higher-level walkthrough lives at
> `docs/source/user_guide/finetune-x2-on-new-corpus.md` (registered in the
> Sphinx Training toctree). That doc covers both base + iteration flows
> from a "zero context" perspective; this file is the operator's hands-on
> playbook with the exact commands.

## When to iterate vs fresh fine-tune

| Trigger | Iterate (resume + tweak) | Fresh fine-tune |
|---|---|---|
| Corpus add/remove < 30 % of entries | ✓ | |
| Corpus add/remove > 30 % of entries | | ✓ |
| Wider DR / more obs noise | ✓ | |
| Reward function changes | | ✓ |
| Termination criteria change | | ✓ |
| Robot URDF / actuator config change | | ✓ |
| Adaptive sampler over-concentrating on impossible motions (see eval) | ✓ (filter the impossible motions) | |

Quick rule: if the policy network output stays meaningful for the new env,
resume; if env semantics shift, restart.

## The 4 axes you can iterate on

```
┌─────────────────────┬────────────────────────────────────────────────────┐
│ Axis                │ Where you change it                                │
├─────────────────────┼────────────────────────────────────────────────────┤
│ Corpus              │ filter PKL or re-merge staging                     │
│ Checkpoint          │ EXTRA_FLAGS=+checkpoint=... in launcher            │
│ Domain randomization│ events/tracking/*.yaml + observations/policy/*.yaml│
│ Iteration count     │ algo.config.num_learning_iterations in exp yaml    │
└─────────────────────┴────────────────────────────────────────────────────┘
```

The rest of this doc walks each axis with the demo_v2 worked example.

---

## Axis 1 — Filter the corpus

If your prior run flagged motions as untrainable (see "Diagnosing a bad
corpus" below), don't re-stage the whole pipeline — just filter the
existing merged PKL into a smaller one. It's faster and the staging dir
stays as canonical documentation of "what we ever considered".

```python
# conda run -n env_isaaclab --no-capture-output python -
import joblib
src = 'gear_sonic/data/motions/x2_ultra_demo_v1.pkl'
dst = 'gear_sonic/data/motions/x2_ultra_demo_v2.pkl'
DROP_PREFIXES = ('sitstand__',)        # whatever your bad-cluster prefix is
d = joblib.load(src)
filtered = {k: v for k, v in d.items() if not k.startswith(DROP_PREFIXES)}
joblib.dump(filtered, dst)
print(f'src: {len(d)} entries -> dst: {len(filtered)} entries (dropped {len(d)-len(filtered)})')
```

For finer-grained filtering (per-key, by duration, by failure rate),
substitute the predicate. The merger's prefix scheme (see SKILL.md → "Step
1 → Subdir → merger prefix" table) makes prefix-level filtering one-liners.

### Diagnosing a bad corpus (decide what to drop)

The adaptive sampler's end-of-run stats are the cleanest signal:

```bash
grep -E 'failure_rate_max|prob_max_over_uniform|num_concentrated_bins' \
    ~/sonic_<CORPUS>.log | tail -3
```

| Stat | Healthy (after 4k iters) | Suspicious |
|---|---|---|
| `failure_rate_max` | 2–4× | > 5× → some motion can't be learned |
| `prob_max_over_uniform` | 5–15× | > 25× → sampler is stuck on impossible motions |
| `num_concentrated_bins` | proportional to corpus difficulty | > 25 % of corpus → too many failing |

For demo_v1: `failure_rate_max=6.0`, `prob_max_over_uniform=40.2`,
`num_concentrated_bins=18/124`. The sit-on-chair motions had no chair in
the IsaacLab training env — sampler over-concentrated on impossible
motions → 6 entries dropped → demo_v2 corpus.

---

## Axis 2 — Pick the continuation checkpoint

```bash
# In your run_local_finetune_<CORPUS>.sh:
export EXTRA_FLAGS="+checkpoint=$REPO_ROOT/logs_rl/TRL_X2Ultra_<PREV>/.../model_step_004000.pt"
```

**Two valid choices**:

| Choice | Use when |
|---|---|
| Previous fine-tune's final `model_step_NNNN.pt` (e.g. demo_v1 → demo_v2) | You want to **continue building** on the corpus-specific learning. Faster path to demo-ready behavior. |
| Original warm-start (e.g. `h200-iter-25000-sphere-feet-20260501/model_step_025000.pt`) | You want a **clean baseline** that never saw the dropped motions. Costs you 4k iters of corpus learning but produces a "no contamination" reference. |

**The iter counter resets to 0 on resume** — checkpoint inputs are the
weights only, not the optimizer/sampler/iter state. Plan for it:

- A 4k-iter demo_v2 run that resumes from demo_v1's 4k checkpoint will
  emit `model_step_002000.pt` and `model_step_004000.pt` in the v2 run
  dir, representing iters 6000 and 8000 of cumulative training.
- The adaptive sampler **does not** persist across resume — it rediscovers
  hard motions in the first few hundred iters. This is fine; it converges
  fast.
- The obs RunningMeanStd **does** persist (it's in the checkpoint).

---

## Axis 3 — Domain randomization ramp

This is the most useful axis when prepping for real-robot deploy. Both
**observation noise** and **actuator gain randomization** are pure config
changes — no Python code unless you add a NEW DR term that doesn't exist
in IsaacLab's `mdp.events` module.

### What DR is already active in `level0_4` (the default for X2)

| Term | Mode | Ranges (defaults) |
|---|---|---|
| `physics_material` | startup | static_fric 0.3–1.6, dynamic_fric 0.3–1.2, restitution 0–0.5 |
| `randomize_rigid_body_mass` (wrist_yaw + torso) | startup | scale 0.8×–2.5× |
| `base_com` (torso) | startup | x±2.5 cm, y±5 cm, z±5 cm |
| `add_joint_default_pos` (all joints) | startup | ±0.01 rad bias |
| `push_robot` | interval (every 4–6 s) | vel ±0.5 m/s lin, ±30° rot, ±45° yaw |

Defined in `gear_sonic/config/manager_env/events/tracking/level0_4.yaml`.
The terms themselves live one level down in `events/terms/*.yaml`.

### Observation noise — already wired, just widen

Defined in `gear_sonic/config/manager_env/observations/policy/local_dir_hist.yaml`
(applied because `enable_corruption: True`):

| Obs | level0_4 default | "2× sim2real" preset (`local_dir_hist_v2.yaml`) |
|---|---|---|
| `gravity_dir` | ±0.05 | ±0.10 |
| `base_ang_vel` | ±0.2 rad/s | ±0.4 rad/s |
| `joint_pos` | ±0.01 rad (0.57°) | ±0.02 rad (1.15°) |
| `joint_vel` | ±0.5 rad/s | ±1.0 rad/s |

To widen further: clone `local_dir_hist_v2.yaml` to `local_dir_hist_v3.yaml`
with bigger numbers; override in your exp yaml via
`- override /manager_env/observations/policy: local_dir_hist_v3`.

### Actuator (KP/KD) gain randomization — IsaacLab built-in, NOT in level0_4

IsaacLab ships `isaaclab.envs.mdp.events:randomize_actuator_gains` out of
the box. To wire it in: drop a yaml term, add it to a new events preset.

`gear_sonic/config/manager_env/events/terms/randomize_actuator_gains.yaml`:

```yaml
randomize_actuator_gains:
  _target_: isaaclab.managers.EventTermCfg
  func: isaaclab.envs.mdp.events:randomize_actuator_gains
  mode: "reset"     # per-episode resample (matches mass randomization cadence)
  params:
    asset_cfg:
      _target_: isaaclab.managers.SceneEntityCfg
      name: "robot"
      joint_names: [".*"]
    stiffness_distribution_params: [0.85, 1.15]
    damping_distribution_params:   [0.85, 1.15]
    operation: "scale"
    distribution: "uniform"
```

Then create a new events preset that layers it on top of level0_4 —
`gear_sonic/config/manager_env/events/tracking/level0_4_pd.yaml`:

```yaml
defaults:
  - terms/physics_material@_here_
  - terms/add_joint_default_pos@_here_
  - terms/base_com@_here_
  - terms/push_robot@_here_
  - terms/randomize_rigid_body_mass@_here_
  - terms/randomize_actuator_gains@_here_

_target_: gear_sonic.envs.manager_env.mdp.events.EventCfg

# (override randomize_rigid_body_mass and push_robot blocks here — copy them
#  verbatim from level0_4.yaml unless you also want to widen those ranges)
```

**Critical gotcha — `EventCfg` strict fields.** `EventCfg` in
`gear_sonic/envs/manager_env/mdp/events.py` is an `@configclass` with
**hardcoded slots**. Adding a new term to the yaml without declaring a slot
fails with:

```
TypeError: EventCfg.__init__() got an unexpected keyword argument 'randomize_actuator_gains'
```

Fix is a 1-line addition to `EventCfg`:

```python
@configclass
class EventCfg:
    physics_material = None
    add_joint_default_pos = None
    add_hand_joint_default_pos = None
    base_com = None
    push_robot = None
    randomize_rigid_body_mass = None
    randomize_actuator_gains = None   # ← add slot for any new event term
```

Then wire the preset into your exp yaml via
`- override /manager_env/events: tracking/level0_4_pd`.

### DR scale rule of thumb

| Goal | Obs noise scale vs `level0_4` | PD scale |
|---|---|---|
| Conservative (lowest regression risk) | 1.5× | ±10 % |
| Sim2real moderate (Lee et al. 2020 / Margolis et al. 2022) | **2×** | **±15 %** |
| Sim2real aggressive | 3× | ±25 % |
| Maximum robustness (research) | 4–5× | ±35 % |

Aggressive DR usually needs **more iterations** to converge — budget at
least 1.5× the iter count of the previous tighter run.

---

## Axis 4 — Iteration count

```yaml
algo:
  config:
    num_learning_iterations: 4000     # additional iters on top of the resumed checkpoint
```

Heuristic on RTX 5090 (3072 envs, 30 fps env, 3 s/iter):

| iters | wall-clock | use case |
|---|---|---|
| 2,000 | ~1.7 h | quick A/B / DR sensitivity test |
| **4,000** | **~3.5 h** | **standard fine-tune iteration (recommended default)** |
| 8,000 | ~7 h | overnight; bigger DR jump |
| 16,000 | ~14 h | major corpus expansion + bigger DR; long overnight |
| 30,000 | ~26 h | full from-scratch on broader corpus (not really "iteration") |

---

## End-to-end: demo_v2 worked example (built 2026-06-24)

What was done to go from `demo_v1` (101 entries, default DR) →
`demo_v2` (95 entries, 2× obs noise + ±15 % PD).

| Step | Time | Output |
|---|---|---|
| Filter PKL (drop 6 `sitstand__*`) | 1 s | `x2_ultra_demo_v2.pkl` (95 entries, 23.9 MB) |
| Author `events/terms/randomize_actuator_gains.yaml` | 2 min | 30-line yaml |
| Author `events/tracking/level0_4_pd.yaml` (level0_4 + actuator_gains) | 1 min | 40-line yaml |
| Author `observations/policy/local_dir_hist_v2.yaml` (2× noise) | 2 min | clone of `local_dir_hist.yaml` with 4 noise blocks doubled |
| Author `sonic_x2_ultra_demo_v2.yaml` exp yaml | 1 min | inherits demo_v1, overrides events + observations + motion_file + project_name + exp_var |
| Author `run_local_finetune_demo_v2.sh` launcher | 1 min | clone of demo_v1; points `CHECKPOINT` at v1's `model_step_004000.pt` |
| Add `randomize_actuator_gains = None` slot to `EventCfg` | 30 s | 3 lines in `events.py` |
| Hydra dry-compose | 3 s | verified motion_file, KP/KD range, doubled noise values |
| Launch | 10 s | training PID + wandb run |
| First iter lands | ~3 min | "Iteration time: 3.0 s, ETA 11857 s" |

End-to-end: **~10 min** of operator work, **identical** to the demo_v1
spin-up time. The DR ramp adds **zero wall-clock** to training itself
(iter speed stayed at 3.0 s/iter).

### Files added for demo_v2 (full list)

```
gear_sonic/
├── config/
│   ├── exp/manager/universal_token/all_modes/
│   │   └── sonic_x2_ultra_demo_v2.yaml                       NEW
│   └── manager_env/
│       ├── events/
│       │   ├── terms/randomize_actuator_gains.yaml           NEW
│       │   └── tracking/level0_4_pd.yaml                     NEW
│       └── observations/policy/local_dir_hist_v2.yaml        NEW
├── data/motions/x2_ultra_demo_v2.pkl                         NEW (filtered from v1)
├── envs/manager_env/mdp/events.py                            EDITED (+3 lines)
└── scripts/run_local_finetune_demo_v2.sh                     NEW
```

Resulting wandb run: https://wandb.ai/meetsitaram/TRL_X2Ultra_DemoV2

---

## Common gotchas (beyond SKILL.md's table)

| Symptom | Root cause | Fix |
|---|---|---|
| `TypeError: EventCfg.__init__() got an unexpected keyword argument '<term>'` | Added a new event term to a tracking preset without declaring its slot in `EventCfg` | Add `<term_name> = None` to the `EventCfg` configclass in `gear_sonic/envs/manager_env/mdp/events.py`. See "EventCfg strict fields" above. |
| Resume run reports `iter 0` (not `iter <previous_final>`) | `+checkpoint=...` loads weights but does NOT preserve the iter counter | Expected. The new run's wandb history will look like a fresh start; cumulative iter count = previous_run_iters + new_run_iters. |
| Adaptive sampler is uniform at iter 0 of resume | Sampler state isn't serialized in the checkpoint | Expected. It rediscovers hard motions within ~200 iters. |
| Reward drops sharply after resume even with no DR change | RunningMeanStd is in the checkpoint, but the env reset distribution shifted slightly with the new corpus (different motion mix → different start poses) | Wait ~500 iters; it usually recovers. If not, the new corpus may be too different — restart from the warm-start instead. |
| Wider DR causes reward to never recover to baseline | DR was too aggressive for the iter budget | Drop one DR tier (e.g. `±15 %` → `±10 %` PD, `2×` → `1.5×` obs noise) OR double the iter count. |
| `level0_4_pd` doesn't apply — log shows old events | Hydra didn't pick up the new preset | Confirm `- override /manager_env/events: tracking/level0_4_pd` is in your exp yaml's `defaults:` block (under `- /exp/...`, before `- _self_`). Run the dry-compose check from SKILL.md to verify. |

---

## When to graduate to a NEW corpus (demo_v3, etc.)

The iteration pattern (this doc) is good for ~3–5 successive runs on the
same base corpus. Beyond that, the staging dir and exp yaml chain get
unwieldy. Signals to start fresh:

- > 30 % of motions have been added or removed since the original
- Reward plateaued for 2 consecutive iterations
- Robot URDF or actuator config changed (start fresh; old checkpoint is stale)
- New eval mode (e.g. teleop / VLA) added that needs a different obs space

To start a new corpus: go back to `SKILL.md` Step 1 with a new `<CORPUS>`
name. Don't fork `demo_v1_sources/` — make a new top-level
`<NEW_CORPUS>_sources/` and stage from scratch.
