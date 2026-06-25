---
name: spin-sonic-x2-finetune
description: End-to-end playbook for spinning a new X2 Ultra SONIC fine-tune from a set of motion files. Use when the user wants to fine-tune the SONIC X2 policy on a new motion corpus (e.g. demo_v2, prod_v1, a new gesture pack, new retargeted clips, or fresh teleop captures). Covers staging, merging into motion_lib PKL, authoring the Hydra exp yaml, launching local or cloud training, monitoring, and post-train evaluation.
---

# Spin a new SONIC X2 fine-tune from a motion corpus

This is the repeatable workflow used to build `x2_ultra_demo_v1` (see
`README.md` next to this file for the snapshot it produced). Everything below
is parameterized by a single `<CORPUS>` name (e.g. `demo_v1`, `demo_v2`,
`gesture_pack_jun`) — substitute throughout.

> **Already have a fine-tune finished and want to keep building on it?** See
> the companion doc [`ITERATE.md`](./ITERATE.md) next to this file. It
> covers the iteration pattern: resuming from a previous checkpoint,
> filtering the corpus, ramping domain randomization (observation noise +
> KP/KD), and the `EventCfg` strict-fields gotcha that bites when adding
> new event terms. Worked example: `demo_v1 → demo_v2` (2026-06-24).

## Quick reference

```
1. Stage         → gear_sonic/data/motions/<CORPUS>_sources/<subdir>/*.pkl  (symlinks)
2. Merge         → python gear_sonic/data_process/build_x2_demo_motion_lib.py
                       --stage-dir gear_sonic/data/motions/<CORPUS>_sources
                       --out      gear_sonic/data/motions/x2_ultra_<CORPUS>.pkl
3. Exp yaml      → gear_sonic/config/exp/manager/universal_token/all_modes/
                       sonic_x2_ultra_<CORPUS>.yaml   (copy demo_v1, edit 4 fields)
4. Launcher      → gear_sonic/scripts/run_local_finetune_<CORPUS>.sh
                       (copy demo_v1, edit 3 env vars)
5. Hydra check   → 1-shot compose to confirm yaml resolves + PKL exists
6. Launch        → setsid nohup bash gear_sonic/scripts/run_local_finetune_<CORPUS>.sh \
                       </dev/null >/dev/null 2>&1 &
7. Monitor       → tail -f ~/sonic_<CORPUS>.log  +  nvidia-smi -l 30
8. Eval          → MuJoCo benchmark on corpus + regression eval + ONNX export
```

## Inputs you need before starting

- `<CORPUS>` name (snake_case, e.g. `demo_v2`)
- A set of motion sources, each one of:
  - **Motion-lib PKL** — single-entry `{key: entry}` or multi-entry bundle. Schema: `{dof, fps, pose_aa, root_rot, root_trans_offset, smpl_joints}`. `pose_aa` + `smpl_joints` are auto-synthesized by the merger if missing.
  - **SOMA chain-matched CSV** — flat `*.csv` at 120 fps; pre-convert into a PKL via the one-liner below.
  - **Kinematic-teleop NPZ** — from `gear_sonic/scripts/teleop_x2_kinematic.py` `--output-dir <dir>` (writes `debug/teleop_episode_*.npz`). Needs the `convert_kinematic_teleop_to_motion_lib.py` converter (TBD — see "Open gaps" below).
- A warm-start checkpoint (most demos start from `~/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt`).

## Step 1 — Stage motions as PKL symlinks

Staging is **uniform PKL symlinks only**, one symlink per motion source,
grouped by category subdir. Subdir name drives the merger's key prefix
(see the prefix table below). Empty subdirs are silently skipped.

```bash
STAGE=gear_sonic/data/motions/<CORPUS>_sources
mkdir -p "$STAGE"/{mc_gestures,retargeted,combat_chain_matched,fighting_chain_matched,sitstand_chain_matched,dances,body_check,teleop_kinematic}

# Symlink each canonical PKL into the matching subdir
ln -sf "$(realpath path/to/canonical.pkl)" "$STAGE/<subdir>/<name>.pkl"
```

**Subdir → merger prefix** (defined in `build_x2_demo_motion_lib.py`):

| Subdir | Prefix | Typical content |
|---|---|---|
| `body_check/` | `bodycheck` | Joint-range / DoF-sweep sanity sequences |
| `combat_chain_matched/` | `combat` | Shadow-boxing, punches |
| `dances/` | `dance` | Bundle PKLs of diverse dances |
| `fighting_chain_matched/` | `fighting` | Latino kicks etc. *(superseded by `dances/` in demo_v1)* |
| `mc_gestures/` | `gesture` | MC-mode gestures captured from real X2 |
| `retargeted/` | `walk` | Stitched-loop walks from `make_warehouse_motion.py` |
| `sitstand_chain_matched/` | `sitstand` | Sit-down / sit-loop / stand-up |
| `teleop_kinematic/` | `teleop` | Quest3 kinematic-teleop captures |

To add a new category, edit `SUBDIR_PREFIX` in
`gear_sonic/data_process/build_x2_demo_motion_lib.py` and pick a short prefix.

### Pre-convert SOMA CSVs → PKL (only if you have CSVs)

CSVs aren't staged directly — they're pre-converted into per-clip PKLs under
`gear_sonic/data/motions/x2_recorded/soma_csvs_converted/<subdir>/<stem>.pkl`,
then symlinked into staging. Run from repo root inside `env_isaaclab`:

```python
# conda run -n env_isaaclab --no-capture-output python -
import sys, joblib
sys.path.insert(0, '.')
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (
    set_robot, load_bones_csv, convert_sequence, downsample_sequence,
)
set_robot("x2_ultra")
for csv_path, out_path in [
    ("agibot-x2-references/.../FOO__x2_chain_matched.csv",
     "gear_sonic/data/motions/x2_recorded/soma_csvs_converted/<subdir>/FOO__x2_chain_matched.pkl"),
    # ... more pairs ...
]:
    seq = load_bones_csv(csv_path)
    entry = downsample_sequence(convert_sequence(seq, 120), 120, 30)
    stem = csv_path.split("/")[-1].removesuffix(".csv")
    joblib.dump({stem: entry}, out_path, compress=True)
```

Bones-SEED CSVs are 120 fps; SONIC consumes 30 fps — always downsample.

## Step 2 — Merge into the training PKL

The merger walks every staged PKL, prefixes keys by subdir, asserts no
post-prefix key collisions, and auto-synthesizes any missing `pose_aa` /
`smpl_joints` (lossless remap of `dof + root_rot[xyzw]`, no FK).

```bash
conda run -n env_isaaclab --no-capture-output python \
  gear_sonic/data_process/build_x2_demo_motion_lib.py \
    --stage-dir gear_sonic/data/motions/<CORPUS>_sources \
    --out       gear_sonic/data/motions/x2_ultra_<CORPUS>.pkl
```

Expect output like (this is the demo_v1 run):

```
body_check                    pkls= 1  entries=  1  frames=  866  (  28.9 s @ 30fps)
combat_chain_matched          pkls= 3  entries=  3  frames=  865  (  28.8 s @ 30fps)
dances                        pkls= 1  entries= 34  frames= 5424  ( 180.8 s @ 30fps)
mc_gestures                   pkls=51  entries= 51  frames=13508  ( 450.3 s @ 30fps)
retargeted                    pkls= 6  entries=  6  frames= 9662  ( 322.1 s @ 30fps)
sitstand_chain_matched        pkls= 1  entries=  6  frames= 1487  (  49.6 s @ 30fps)
TOTAL                         pkls=63  entries=101  frames=31812  (1060.4 s @ 30fps)
(synthesized pose_aa+smpl_joints for 6 entries that lacked them — pure remap of dof+root_rot, no FK)
```

If any source PKL is malformed (missing `dof`/`root_rot`/`root_trans_offset`/
`fps`) the merger fails loudly with the offending path + key. Fix and re-run;
it's idempotent.

## Step 3 — Author the Hydra exp yaml

Copy `sonic_x2_ultra_demo_v1.yaml` as the template; edit 4 fields. The
inheritance chain (`bones_seed_sphere_feet` → `bones_seed` → `smoke`) handles
everything else.

```bash
EXP_DIR=gear_sonic/config/exp/manager/universal_token/all_modes
cp $EXP_DIR/sonic_x2_ultra_demo_v1.yaml $EXP_DIR/sonic_x2_ultra_<CORPUS>.yaml
```

Then edit (only these 4 fields matter for the common case):

```yaml
exp_var: <CORPUS>                          # e.g. demo_v2
project_name: TRL_X2Ultra_<TitleCase>      # wandb project; e.g. TRL_X2Ultra_DemoV2
num_envs: 3072                             # 3072 single-RTX-5090 (32GB); 16384 single-H200 (80GB)

manager_env:
  commands:
    motion:
      motion_lib_cfg:
        motion_file: gear_sonic/data/motions/x2_ultra_<CORPUS>.pkl

algo:
  config:
    num_learning_iterations: 4000          # fine-tune budget; 30000 for from-scratch
```

`robot.foot: sphere` is inherited — don't override unless deliberately
deploying against mesh feet.

### Hydra dry-compose (always do this before launching)

Catch typos / missing files in 1 s without launching the full stack:

```bash
conda run -n env_isaaclab --no-capture-output python - <<'PY'
import os, sys
sys.path.insert(0, '.')
from hydra import compose, initialize_config_dir
cfg_dir = os.path.abspath('gear_sonic/config')
with initialize_config_dir(version_base=None, config_dir=cfg_dir):
    cfg = compose(
        config_name='base',
        overrides=['+exp=manager/universal_token/all_modes/sonic_x2_ultra_<CORPUS>',
                   '++num_envs=3072', '++headless=True', '++use_wandb=False'],
    )
mf = cfg.manager_env.commands.motion.motion_lib_cfg.motion_file
print('exp_var               :', cfg.exp_var)
print('project_name          :', cfg.project_name)
print('motion_file           :', mf, '  exists:', os.path.exists(mf))
print('robot.foot            :', cfg.manager_env.config.robot.foot)
print('num_learning_iterations:', cfg.algo.config.num_learning_iterations)
PY
```

## Step 4 — Author the local launcher

Copy `gear_sonic/scripts/run_local_finetune_demo_v1.sh`; edit 3 env vars.

```bash
cp gear_sonic/scripts/run_local_finetune_demo_v1.sh \
   gear_sonic/scripts/run_local_finetune_<CORPUS>.sh
```

Then edit the `export ...` block:

```bash
export MOTION_FILE="gear_sonic/data/motions/x2_ultra_<CORPUS>.pkl"
export EXP_NAME="sonic_x2_ultra_<CORPUS>"
export LOG_FILE="$HOME/sonic_<CORPUS>_${LAUNCH_TS}.log"

# Knobs you may tune:
export NUM_PROCESSES=1       # 1 local single-GPU; 8 for 8xH200 cloud
export NUM_ENVS=3072         # 3072 RTX 5090 (32GB); 16384 H200 (80GB)
export NUM_ITERS=4000        # fine-tune; 30000 from scratch
export USE_WANDB=True        # set False to skip wandb
export EXTRA_FLAGS="+checkpoint=$HOME/x2_cloud_checkpoints/.../model_step_NNNNNN.pt"
# Drop EXTRA_FLAGS for from-scratch training.
# For a full optimizer-state resume mid-run, add: +resume=True
```

Both launchers delegate to `gear_sonic/scripts/cloud/run_smoke_8gpu.sh` —
that's the single Hydra+accelerate-launch entrypoint for local and cloud
alike. The "smoke" name is historical; it's a fully general launcher.

## Step 5 — Launch (detached, no tmux required)

```bash
setsid nohup bash gear_sonic/scripts/run_local_finetune_<CORPUS>.sh \
    </dev/null >/dev/null 2>&1 &
echo $! > ~/sonic_<CORPUS>.pid
```

The launcher script's `exec > >(tee -a "$LOG_FILE") 2>&1` handles logging
internally — do **not** add an outer `>>$LOG` to the nohup line (it
double-tees every line into the log file).

**Verify within ~10 s** that something is alive:

```bash
ps -ef | grep -E 'train_agent_trl|accelerate' | grep -v grep
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

Expect a `python gear_sonic/train_agent_trl.py` process at ≥50% CPU and GPU
util climbing as Isaac Sim initializes (~60-90 s). First PPO iter log lands
~90-120 s after launch.

## Step 6 — Monitor

```bash
tail -f ~/sonic_<CORPUS>.log                  # live trainer output
nvidia-smi -l 30                              # GPU pressure
ls logs_rl/TRL_X2Ultra_<TitleCase>/.../*.pt   # checkpoints as they save
kill $(cat ~/sonic_<CORPUS>.pid)              # stop the run cleanly
```

**Healthy signals** (per iter log):
- `Iteration time` ~3 s on RTX 5090 with NUM_ENVS=3072
- `Env/Metrics/motion/error_body_pos` trending down over hundreds of iters
- `Env/Episode_Termination/time_out` rising relative to other terminations
- No `KeyError` / `Traceback` / `OutOfMemoryError`

**Save cadence**: every **2000 iters** (`gear_sonic/config/callbacks/model_save.yaml`).
A 4000-iter fine-tune produces `model_step_002000.pt`, `model_step_004000.pt`,
plus a rolling `last.pt` in
`logs_rl/TRL_X2Ultra_<TitleCase>/.../sonic_x2_ultra_<CORPUS>_<exp_var>-<TS>/`.

## Step 7 — Post-train eval

Three checks before considering the run shippable:

```bash
# (a) Demo-corpus benchmark — does the new policy actually do better on the
# motions we fine-tuned on?
conda run -n env_isaaclab --no-capture-output python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint logs_rl/TRL_X2Ultra_<TitleCase>/.../model_step_004000.pt \
  --motion gear_sonic/data/motions/x2_ultra_<CORPUS>.pkl

# (b) Regression check — has the broader bones_seed skillset degraded?
conda run -n env_isaaclab --no-capture-output python gear_sonic/scripts/eval_x2_mujoco.py \
  --checkpoint logs_rl/TRL_X2Ultra_<TitleCase>/.../model_step_004000.pt \
  --motion gear_sonic/data/motions/x2_ultra_bones_seed.pkl --sample 50

# (c) ONNX export + sim validation before any real-robot deploy
conda run -n env_isaaclab --no-capture-output python gear_sonic/scripts/reexport_x2_g1_onnx.py \
  --checkpoint logs_rl/TRL_X2Ultra_<TitleCase>/.../model_step_004000.pt
bash gear_sonic_deploy/deploy_x2.sh sim     # smoke against the new ONNX
```

## Cloud variant (only if local GPU is too small)

Use the same launcher script with cloud knobs. Provision 8×H200 first per
`docs/source/user_guide/train-on-cloud.md`:

```bash
# On the cloud node, from the repo root:
MOTION_FILE=gear_sonic/data/motions/x2_ultra_<CORPUS>.pkl \
EXP_NAME=sonic_x2_ultra_<CORPUS> \
NUM_PROCESSES=8 NUM_ENVS=16384 NUM_ITERS=4000 \
USE_WANDB=True \
EXTRA_FLAGS="+checkpoint=$HOME/model_step_025000.pt" \
LOG_FILE=$HOME/sonic_<CORPUS>.log \
bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
```

Heuristic: stay local for ≤2k motion entries + ≤4k iters; go cloud beyond
that. Cloud overhead (~15 min bootstrap + scp + commit/push + rsync-back)
eats most of the 8× throughput win for short runs on small corpora.

## Common errors and fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `KeyError: 'pose_aa'` inside `motion_lib_base.py:1769` | Source PKL missing `pose_aa` (e.g. PKLs built by `make_warehouse_motion.py`) | Already handled by the merger's auto-synth. If you hit it, you bypassed the merger — re-merge through `build_x2_demo_motion_lib.py`. |
| Merger `RuntimeError: key collision on '<prefix>__<name>'` | Two staged PKLs contribute the same fully-prefixed key (often a bundle PKL whose entries duplicate per-clip PKLs in a sibling subdir) | Drop one source. Per demo_v1: dropping `fighting_chain_matched/` because `x2_ultra_dances.pkl` already contained those 2 latino kicks. |
| `OutOfMemoryError` early in Isaac init | NUM_ENVS too high for the GPU | Drop NUM_ENVS by 25-50%. Heuristic: 3072 fits 32 GB, 16384 fits 80 GB. |
| Log file has every line **doubled** | Both the outer nohup redirect AND the script's internal `tee` are writing to LOG_FILE | Don't add `>>$LOG` to the nohup line — the script's `exec > >(tee -a ...)` handles it. |
| `tmux: command not found` | tmux not installed on this host | Use `setsid nohup bash ... </dev/null >/dev/null 2>&1 &` instead — same detachment, no install needed. |
| Hydra `InterpolationResolutionError: HydraConfig was not set` when inspecting `experiment_dir` | Reading interpolated fields outside `@hydra.main` | Read the saved `<run_dir>/config.yaml` instead — Hydra writes a fully-resolved snapshot there at launch. |

## Files this skill touches

```
gear_sonic/
├── config/exp/manager/universal_token/all_modes/
│   ├── sonic_x2_ultra_bones_seed_sphere_feet.yaml   # base for fine-tunes (sphere feet)
│   ├── sonic_x2_ultra_demo_v1.yaml                  # template — copy + edit 4 fields
│   └── sonic_x2_ultra_<CORPUS>.yaml                 # NEW, per corpus
├── data/motions/
│   ├── demo_v1_sources/                             # template staging dir + this SKILL
│   ├── <CORPUS>_sources/                            # NEW, per corpus
│   ├── x2_ultra_<CORPUS>.pkl                        # NEW, merger output
│   └── x2_recorded/soma_csvs_converted/             # pre-converted SOMA CSVs
├── data_process/
│   ├── build_x2_demo_motion_lib.py                  # merger (handles every corpus)
│   └── convert_soma_csv_to_motion_lib.py            # CSV→PKL helpers
└── scripts/
    ├── cloud/run_smoke_8gpu.sh                      # the actual launcher (local & cloud)
    ├── run_local_finetune_demo_v1.sh                # template — copy + edit 3 env vars
    └── run_local_finetune_<CORPUS>.sh               # NEW, per corpus
```

## Open gaps (things still manual today)

- **No CLI wrapper for SOMA CSV→PKL**. Today the conversion is inline Python
  (see step 1). Worth codifying as `convert_soma_csvs_to_pkls.py` taking
  `--csvs <list> --out-dir <dir> --robot x2_ultra` when next touched.
- **No `convert_kinematic_teleop_to_motion_lib.py`**. Stub exists in the
  plan; needs ~100 LOC to convert
  `gear_sonic/scripts/teleop_x2_kinematic.py`'s
  `debug/teleop_episode_*.npz` files into motion_lib PKLs.
- **Per-corpus exp yaml + launcher are still manual copies**. A Hydra-only
  launch path (`++project_name=... ++motion_file=...` on CLI, no new yaml)
  would remove step 3 + 4 for the common case.

## See also

- [`ITERATE.md`](./ITERATE.md) — How to iterate on a previous fine-tune
  (resume + corpus filter + DR ramp). Worked example: demo_v1 → demo_v2.
- [`README.md`](./README.md) — Snapshot of what's in `demo_v1_sources/`
  (staged motion entries by category).
- `docs/source/user_guide/finetune-x2-on-new-corpus.md` — User-guide
  version of this skill, written for a reader without chat context (good
  for onboarding new operators).

## Worked example — demo_v1 (built 2026-06-23)

What was actually done to produce `x2_ultra_demo_v1.pkl` + the running
4k-iter fine-tune. Times are wall-clock end-to-end.

| Step | Time | Output |
|---|---|---|
| Stage 63 PKL symlinks across 6 subdirs (incl. pre-converting 5 SOMA CSVs) | ~5 min | `demo_v1_sources/` (101 motion entries) |
| Run merger | ~2 s | `x2_ultra_demo_v1.pkl` (5.81 MB, 101 entries, 6 walks pose_aa-synth'd) |
| Copy + edit exp yaml | ~1 min | `sonic_x2_ultra_demo_v1.yaml` (4 lines changed from base) |
| Hydra dry-compose | ~3 s | confirms exp_var/project_name/motion_file/num_envs/iters/foot |
| Copy + edit launcher | ~1 min | `run_local_finetune_demo_v1.sh` |
| Launch | ~10 s | training PID + log symlink |
| Monitor until first iter | ~90 s | "Iteration time: 3.1s, ETA 12,373s" |

End-to-end: **~10 min** of operator work to go from "I have a list of motion
files" to "training is running and emitting iter logs". The 4k-iter run then
takes ~3.4 h wall-clock on the RTX 5090.

Resulting wandb run: https://wandb.ai/meetsitaram/TRL_X2Ultra_DemoV1
