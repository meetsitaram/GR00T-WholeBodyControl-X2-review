# Training on a Cloud GPU Node

This guide walks through training the Agibot X2 Ultra (or any new embodiment)
on a multi-GPU cloud node, using the BONES-SEED retargeted motion library as
the worked example.

The recipe is deliberately **side-channel friendly**: nothing has to be pushed
upstream first. You bundle every git-untracked / gitignored artifact your
cloud node needs into a single tarball, `scp` it over after `git clone`, and
launch.

> Reference docs: see [`installation_training.md`](../getting_started/installation_training.md)
> for the canonical Isaac Lab install instructions, and
> [`training.md`](training.md) for the upstream G1 training recipe.

## Architecture

```
Local workstation                                  Cloud node (8x GPU)
────────────────────                               ──────────────────────
  scp x2_cloud_bundle.tar.gz  ───────────────►    ~/x2_cloud_bundle.tar.gz
                                                          │
  Git remote  ─── git clone (current HEAD) ───►   GR00T-WholeBodyControl/
                                                          │  tar -xzf
                                                          ▼
                                                   gear_sonic/data/motions/*.pkl
                                                   gear_sonic/config/.../<exp>.yaml
                                                   gear_sonic/scripts/*.py
                                                          │
                                                   accelerate launch x N
                                                          │
                                                          ▼
                                                   logs_rl/  +  W&B
```

## 1. Pre-flight on your local workstation

> **Commit + push your code changes BEFORE building the bundle.** The
> cloud-side workflow is `git clone <REPO_URL> --branch <REPO_BRANCH>` →
> `tar -xzf bundle.tar.gz` on top. Anything you've only edited locally
> (Python, configs, shell scripts) but haven't pushed will silently be
> the *old* version on the cloud node. The bundle is only for gitignored
> artifacts; tracked-but-uncommitted files are a footgun. Run
> `git status` and either `git commit && git push` or accept that the
> cloud node will run stale code. (We learned this the slow way — see
> commit `8fb1ca6` for the symptom.)

The cloud node needs a handful of files that aren't in git: the motion
libraries (gitignored under `data/`) and any new experiment yaml / helper
scripts that haven't been committed yet. Bundle them all into one tarball.

For the X2 Ultra BONES-SEED training, the bundle contains:

| Path | Purpose | Why not in git |
|---|---|---|
| `gear_sonic/data/motions/x2_ultra_bones_seed.pkl` | 2,550-motion training library | gitignored (`data/`) |
| `gear_sonic/data/motions/x2_ultra_body_check.pkl` | Single-clip PKL for replay sanity check | gitignored |
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_bones_seed.yaml` | Hydra entry point for the new run | typically untracked while iterating |
| `gear_sonic/data_process/build_x2_bones_seed_motion_lib.py` | Rebuild the PKL from retargeted CSVs | typically untracked |
| `gear_sonic/scripts/play_x2_motion_mujoco.py` | Local MuJoCo kinematic replay | optional on a headless cloud |

For the **sphere-foot sim2sim-gap experiment** (G18 follow-up; see §11), add:

| Path | Purpose | Why not in git |
|---|---|---|
| `gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra_sphere_feet.urdf` | 24-sphere foot URDF mirroring the MJCF foot collider | gitignored (`data/`) |

The new Hydra entry `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_bones_seed_sphere_feet.yaml`,
the `make_x2_ultra_cfg(...)` factory in `gear_sonic/envs/manager_env/robots/x2_ultra.py`,
and the `++robot.foot` plumbing in `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`
are tracked in git — commit + `git pull` on the cloud node and they'll be there.

Build the bundle from the **repo root** so the tar paths match the cloud
layout exactly:

```bash
cd <path/to/GR00T-WholeBodyControl>
tar -czf /tmp/x2_cloud_bundle.tar.gz \
    gear_sonic/data/motions/x2_ultra_bones_seed.pkl \
    gear_sonic/data/motions/x2_ultra_body_check.pkl \
    gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_bones_seed.yaml \
    gear_sonic/data_process/build_x2_bones_seed_motion_lib.py \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra_sphere_feet.urdf

ls -lh /tmp/x2_cloud_bundle.tar.gz                # ~200 MB compressed
tar -tzf /tmp/x2_cloud_bundle.tar.gz              # sanity: 6 paths
sha256sum /tmp/x2_cloud_bundle.tar.gz             # capture for cloud-side verify
```

Drop `x2_ultra_sphere_feet.urdf` from the list if you're not running the
sphere-foot sim2sim experiment (§11). All other paths are needed by the
core BONES-SEED training.

Add or remove files as needed for your embodiment. The PKLs always need a
side-channel because the source motion data is licensed; the configs and
scripts can also be committed upstream and skipped from the bundle once the
recipe is stable.

## 2. Cloud node prerequisites

| Requirement | Recommended |
|---|---|
| OS | Ubuntu 22.04+ |
| GPU | 8x NVIDIA GPUs, >= 32 GB VRAM each (A100 / L40 / H100 / 5090). 80 GB cards let you push `num_envs` higher. |
| CUDA | 12.x driver + runtime |
| Disk | ~200 GB free for repo + Isaac Lab install + checkpoints |
| Network | SSH (for `scp`); outbound HTTPS for W&B and pip wheels |

> **Provider quickstarts.** If you don't already have a node, see the
> appendices for turnkey provisioning recipes:
> [Appendix A — Nebius](#appendix-a-provisioning-on-nebius).

> **Tip — check Nebius capacity *before* `compute instance create`.** The
> create call doesn't fail fast on no-capacity; it sits in the scheduler
> queue for ~5 min and only then surfaces `NotEnoughResources`. Run
> [`gear_sonic/scripts/cloud/nebius_gpu_scan.py`](../../../gear_sonic/scripts/cloud/nebius_gpu_scan.py)
> first to pick a (region, platform) tuple with non-zero on-demand inventory:
>
> ```bash
> python gear_sonic/scripts/cloud/nebius_gpu_scan.py --gpus 8 --min-on-demand 1
> ```
>
> Output ends with a `Recommended:` line pointing at the highest-availability
> (region, platform, preset) tuple that matches your filter. Use that
> region/platform in the `compute instance create` call below — it's the
> difference between "node ready in 2 min" and "scheduler timeout in 5 min".
>
> **Capacity advice is `* STALE` and frequently lies for 8-GPU presets.**
> Every row in the scanner output is marked stale (Nebius's own caveat),
> and we've seen "4/4 HIGH on-demand" 8x H200 advice produce a *husk*
> instance: `state: STOPPED`, no public IP, `resource_version: 1`,
> never actually scheduled (see B.4). 1-GPU presets allocate reliably.
> **If you're cold-starting on a new region or after a long away period,
> create a 1×H200 first** (cheap, ~30 s create) to validate the whole
> path (image family, subnet, SSH key, bootstrap), then scale up to 8x
> with confidence.

## 3. Install Isaac Lab and create the conda env

> **Tip — one-shot bootstrap.** If you're starting from a fresh boot disk
> (deleted the previous one, new tenant, etc.), skip this section and §4
> entirely and run
> [`gear_sonic/scripts/cloud/bootstrap_fresh_node.sh`](../../../gear_sonic/scripts/cloud/bootstrap_fresh_node.sh)
> instead. It bundles every fix in Appendix B (B.6 conda ToS, B.7 isaacsim
> pin, B.8 EULA env, B.9 setuptools/flatdict, B.10b/c/d, B.11 git-lfs)
> into 13 idempotent phases and ends at a passing Hydra dry-compose. ~15
> min on a fresh `ubuntu24.04-cuda13.0` Nebius node (cuda12 retired in
> April 2026 — see B.2):
>
> **Required env vars** — without these the script hard-fails at Phase 11:
>
> | Variable | Purpose | Example |
> |---|---|---|
> | `REPO_URL` | git URL to clone (HTTPS works without SSH key on the cloud node) | `https://github.com/<fork>/GR00T-WholeBodyControl.git` |
> | `REPO_BRANCH` | branch to check out (default `main`) | `wip/mujoco-experiments-20260420` |
>
> ```bash
> # On your workstation, scp the script to the new node:
> scp gear_sonic/scripts/cloud/bootstrap_fresh_node.sh ubuntu@$PUBLIC_IP:~/
>
> # On the cloud node:
> ssh ubuntu@$PUBLIC_IP
> REPO_URL=https://github.com/<fork>/GR00T-WholeBodyControl.git \
> REPO_BRANCH=<your-branch> \
>   tmux new -d -s bootstrap "bash ~/bootstrap_fresh_node.sh 2>&1 | tee ~/bootstrap.log"
> # Detach-safe; poll with: tail -f ~/bootstrap.log
> ```
>
> Picking the branch wrong is the most common silent failure: if the
> branch you specify doesn't have your latest commits (you forgot to
> `git push`), training will run with stale code and you won't notice
> until a behavior regresses. Always `git status && git log @{u}..HEAD`
> on your workstation immediately before launching bootstrap.
>
> When it finishes, jump straight to §5 (`scp` bundle). The rest of §3 + §4
> below is the manual walkthrough the script automates — read it for
> background or for adapting to a non-Nebius cloud.

Mirror the local development env (`env_isaaclab`). Follow the
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
# ...run the IsaacLab installer per their docs (isaaclab.sh -i)
python -c "import isaaclab; print(isaaclab.__version__)"
```

## 4. Clone the repo and install training extras

The current `main` (or whichever branch holds your trained-and-validated
training stack) is fine — the bundle in the next step supplies everything
that's not in git.

```bash
sudo apt-get install -y git-lfs libglu1-mesa
git lfs install --skip-repo

git clone <your-fork-url> GR00T-WholeBodyControl
cd GR00T-WholeBodyControl
git lfs pull --include "gear_sonic/data/assets/robot_description/**"   # ~110 MB of robot meshes

pip install -e "gear_sonic/[training]"
pip install "open3d==0.19.0" "tensordict==0.12.1" "vector-quantize-pytorch==1.28.1"   # missing from [training] extra; see B.10b
python check_environment.py --training      # repo's pre-flight
```

## 5. Transfer and unpack the bundle

From your local workstation:

```bash
scp /tmp/x2_cloud_bundle.tar.gz user@cloud:~/
```

On the cloud node — extract from the **repo root** so paths land in their
final locations:

```bash
cd ~/GR00T-WholeBodyControl
sha256sum ~/x2_cloud_bundle.tar.gz          # match the local hash
tar -xzf ~/x2_cloud_bundle.tar.gz

# spot-check
ls -lh gear_sonic/data/motions/*.pkl
ls -lh gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_bones_seed.yaml
```

## 6. Verify the data and the new Hydra config

Pure-Python PKL check (no GPU needed):

```bash
python -c "
import joblib
d = joblib.load('gear_sonic/data/motions/x2_ultra_bones_seed.pkl')
print(f'{len(d)} motions, fields: {list(next(iter(d.values())).keys())}')
"
# Expect:
#   2550 motions, fields: ['root_trans_offset','pose_aa','dof','root_rot','smpl_joints','fps']
```

Hydra dry-compose, to confirm the new yaml resolves and points at the
unpacked PKL:

```bash
python gear_sonic/train_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/all_modes/sonic_x2_ultra_bones_seed \
  --cfg job 2>&1 | grep -E "motion_file|num_envs|num_learning_iterations|project_name"
# Expect:
#   motion_file: gear_sonic/data/motions/x2_ultra_bones_seed.pkl
#   num_envs: 4096
#   num_learning_iterations: 30000
#   project_name: TRL_X2Ultra_BonesSeed
```

If either check fails, re-extract and verify the sha256 of the bundle —
silent transfer corruption is the most common cause.

## 7. Configure W&B (optional)

Once per cloud node:

```bash
wandb login                                  # paste your API key
```

To skip W&B entirely, append `++use_wandb=False` to the launch command in
the next step.

Runs land in the W&B project named by `project_name` in the experiment yaml
(for the X2 BONES-SEED run that's `TRL_X2Ultra_BonesSeed`).

## 8. Launch training

The repo's recommended distributed launcher is `accelerate` (see the
multi-GPU section of [`training.md`](training.md)).

### 8a. Smoke test on all 8 GPUs (~3 min, ~$2)

Before committing to the multi-hour run, verify the full pipeline end-to-end
with a tiny stand-still motion library and 200 PPO iterations. Two helper
scripts under [`gear_sonic/scripts/cloud/`](../../../gear_sonic/scripts/cloud/)
make this a one-liner:

```bash
# one-time: carve a stand-still smoke PKL out of the BONES-SEED bundle
python gear_sonic/scripts/cloud/build_stand_idle_smoke.py
# -> gear_sonic/data/motions/x2_ultra_stand_idle_smoke.pkl  (~1.6 MB)

# launch on all 8 GPUs in tmux (W&B off, num_envs=4096, 200 iters by default)
tmux new -d -s smoke "bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh"
tmux a -t smoke               # attach to watch, Ctrl-b d to detach
tail -f ~/smoke.log           # ...or follow the log file
```

Override knobs on the launcher (env vars): `NUM_PROCESSES`, `NUM_ENVS`,
`NUM_ITERS`, `MOTION_FILE`, `USE_WANDB`. See the script header for defaults.

If the smoke completes without crashing and you see episode rewards moving in
the log, you're cleared for the full run.

### 8b. Full training run

The same `run_smoke_8gpu.sh` helper runs the real thing — just override the
env vars. Recommended invocation, validated on 8x H200 SXM (April 2026):

```bash
tmux new -d -s train "
  NUM_ENVS=16384 \
  NUM_ITERS=20000 \
  MOTION_FILE=gear_sonic/data/motions/x2_ultra_bones_seed.pkl \
  USE_WANDB=False \
  LOG_FILE=\$HOME/train.log \
  bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
"
tmux a -t train     # attach; Ctrl-b d to detach
```

**Per-GPU `NUM_ENVS` tuning** (measured throughput at iter ~30):

| Card / mem | NUM_ENVS | Iter time | Samples/sec | GPU util | Mem used | Notes |
|---|---|---|---|---|---|---|
| H200 SXM (144 GB) | 8192 | 4.6s | 342K | ~70% | 29 GB | safe baseline |
| H200 SXM (144 GB) | **16384** | **6.9s** | **456K** | **~87%** | **48 GB** | **+33% throughput, recommended** |
| H100 SXM (80 GB) | 4096 | ~3.4s* | ~290K* | ~60% | ~17 GB | *extrapolated |
| H100 SXM (80 GB) | 8192 | ~5.5s* | ~370K* | ~85% | ~32 GB | *extrapolated |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the EULA env vars are
set inside the helper script, so you don't need to pass them again.

**Tuning rationale**

- More envs/proc trade per-iter wall-clock for **higher samples/sec** (better
  amortization of NCCL all-reduce + Python overhead).
- The PPO trust region tolerates much larger batches than the repo defaults
  assume; we measured no quality regression at 16K envs/GPU vs 8K.
- If you see slow memory creep over many hours, drop one tier — the OOM-late
  failure mode is far more expensive than a 33% throughput loss.

**Budget envelope** (8x H200, full 20K iter run):

| Iters | Wall clock | Cost @ ~$30/hr |
|---|---|---|
| 5K (early checkpoint) | ~10 hrs | ~$300 |
| 10K (often "good enough") | ~19 hrs | ~$570 |
| 20K (config default cap) | ~38 hrs | ~$1,140 |

**Always launch inside `tmux`** (the helper scripts already use it) so an
SSH drop does not kill the run.

## 9. Monitor

### 9a. While it's running (cloud-side)

```bash
ssh ubuntu@$PUBLIC_IP

# the live training session
tmux a -t train          # Ctrl-b d to detach without killing it
tail -f ~/train.log      # alternative: just follow the log

# quick health snapshot (no need to attach tmux)
grep -E "Learning iteration|Iteration time|ETA:" ~/train.log | tail -n 6
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
grep -cE "Traceback|Fatal|^Error|FAIL|OOM" ~/train.log    # should stay 0
```

The repo writes two flavors of checkpoint:

- **`model_step_NNNNNN.pt`** every 2,000 iters (numbered, kept forever)
- **`last.pt`** every 50 iters (overwritten, always the freshest)

So a 20K run produces ~10 numbered + 1 rolling = ~4-5 GB total under
`logs_rl/TRL_X2Ultra_BonesSeed/<run-dir>/`.

### 9b. Cloud → local eval round-trip

The fastest way to sanity-check policy quality without paying cloud GPU
time for visualization: pull the freshest checkpoint to your workstation
and run `eval_agent_trl.py` locally with the render recorder.

```bash
# (local) one-shot pull of the latest run dir (~400 MB, ~50 s on home wifi)
RUN_DIR=$(ssh ubuntu@$PUBLIC_IP \
  'ls -td ~/GR00T-WholeBodyControl/logs_rl/TRL_X2Ultra_BonesSeed/manager/universal_token/all_modes/* | head -1')
mkdir -p ~/x2_cloud_checkpoints
rsync -avz --partial \
  ubuntu@$PUBLIC_IP:"$RUN_DIR/" \
  ~/x2_cloud_checkpoints/$(basename "$RUN_DIR")/

# (local) render 4 envs for 300 steps; videos land in /tmp/x2_eval_renders/
cd ~/path/to/GR00T-WholeBodyControl
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
mkdir -p /tmp/x2_eval_renders
conda run -n env_isaaclab --no-capture-output python gear_sonic/eval_agent_trl.py \
  +checkpoint=$HOME/x2_cloud_checkpoints/$(basename "$RUN_DIR")/last.pt \
  +headless=True \
  ++num_envs=4 \
  ++run_once=True \
  ++max_render_steps=300 \
  ++manager_env.config.render_results=True \
  "++manager_env.config.save_rendering_dir=/tmp/x2_eval_renders" \
  ++manager_env.config.env_spacing=10.0 \
  "~manager_env/recorders=empty" "+manager_env/recorders=render"
```

This works on any local CUDA box with the same Isaac Lab env (we tested on
RTX 5090 + IsaacLab 2.2.0). Avoid `eval_callbacks=im_eval` locally unless
you also `pip install smpl_sim` — without it the metrics callback crashes
(the render path above sidesteps it).

#### Live viewer (real-time, multiple robots)

For a "watch the policy in action" sanity check rather than recorded
videos, run headed:

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y DISPLAY=:1
conda run -n env_isaaclab --no-capture-output python gear_sonic/eval_agent_trl.py \
  +checkpoint=$HOME/x2_cloud_checkpoints/<run-dir>/last.pt \
  +headless=False \
  ++num_envs=8 \
  ++run_once=False \
  ++manager_env.config.env_spacing=4.0 \
  ++sim.render_interval=4
```

Two key flags:

- `+headless=False` opens the Isaac Sim viewer window.
- `++sim.render_interval=4` renders every 4 physics ticks → viewer runs
  at the policy's 50 Hz instead of the 200 Hz physics rate (fixes
  slow-motion playback).

Use the viewer's WASD + right-click drag to fly the camera. Press `F` to
focus on the active robot. Bump `++num_envs` up if your GPU has headroom;
bump `++manager_env.config.env_spacing` up when locomotion clips make
robots from different envs visually run into each other (no physical
collision between envs — purely cosmetic). With `num_envs=N`, env *i*
gets the *i*-th motion in the library, so 8 envs naturally show 8
different clips side-by-side. Camera focus
toggles between "free" and "track robot" via the viewer's `F` shortcut
(handled by `manager_env_wrapper.focusing_viewer`).

Cadence we found useful for an X2 BONES-SEED run:

| When | What to look for |
|---|---|
| iter ~2,000 (~hr 4) | robot doesn't immediately fall; episodes survive past ~2 s |
| iter ~5,000 (~hr 10) | partial tracking on simple `loco__` / `standing__` motions |
| iter ~10,000 (~hr 19) | most idle/loco motions track recognizably; if quality plateaus, **early-stop and save ~$500** |
| iter 20,000 (~hr 38) | full converged policy — body-check / aggressive motions only land here |

## 10. Resume after interruption

We have two resume modes; the right choice depends on whether you're keeping
the same training schedule or changing it.

| Use case | Flags | Notes |
|---|---|---|
| Same hyperparameters, picking up where you left off | `++resume=True ++experiment_dir=<full path to run dir>` | Reuses optimizer + LR scheduler state. Required when continuing the same run. |
| Changing the iteration budget, the LR schedule, or finetuning into a different config | `+checkpoint=<path/to/model_step_NNNNNN.pt>` | Loads weights only, fresh optimizer/scheduler. Avoids the LR-discontinuity regression we hit when resuming with a changed iteration budget. |

## 11. Sim2sim-gap experiment — fine-tune (or retrain) on the 24-sphere foot

Background: G18 in [`sim2sim_mujoco.md`](sim2sim_mujoco.md) showed that the
mesh-foot policy collapses the moment IsaacLab is moved to the 24-sphere
foot collider that MuJoCo deploys against (progress 0.49, terminations
0.67 at the 16k checkpoint vs 1.00 / 0.00 with mesh feet). The fix is to
let the policy see the deployment-time contact geometry during training.

> **Why not "true" per-episode mesh/sphere DR?** IsaacLab spawns one
> `ArticulationCfg`, clones it across all envs (`replicate_physics=True`),
> and can't swap the underlying collider per episode without significant
> per-env-asset surgery. The practical equivalent — used here — is to
> train *all* envs on the sphere URDF and let the existing per-env
> friction DR (`physics_material.yaml`,
> `static_friction_range: [0.3, 1.6]`) fan out the contact distribution
> on top of the new geometry.

The new Hydra entry that does this is
`sonic_x2_ultra_bones_seed_sphere_feet.yaml`. It inherits everything
from `sonic_x2_ultra_bones_seed` and only overrides
`manager_env.config.robot.foot=sphere`. That single override routes the
env spawn through `make_x2_ultra_cfg(foot="sphere")` (in
`gear_sonic/envs/manager_env/robots/x2_ultra.py`) and loads
`x2_ultra_sphere_feet.urdf`.

### 11a. Pre-flight (cloud node)

The repo-tracked changes (factory + Hydra plumbing + new config) come in
via `git pull`. The gitignored URDF asset comes in via the bundle (you
already added it to the tarball in §1):

```bash
cd ~/GR00T-WholeBodyControl

# Make sure you're on the branch with the make_x2_ultra_cfg factory.
# (If you haven't pushed your local commits, this `git pull` will silently
# leave the cloud on stale code — see §1's commit-and-push warning.)
git pull

# Re-extract the bundle if you haven't yet.
tar -xzf ~/x2_cloud_bundle.tar.gz

# Sanity-check the new asset and config landed.
ls -lh gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra_sphere_feet.urdf
ls -lh gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra_bones_seed_sphere_feet.yaml

# Hydra dry-compose (confirms the override resolves).
python gear_sonic/train_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/all_modes/sonic_x2_ultra_bones_seed_sphere_feet \
  --cfg job 2>&1 | grep -E "motion_file|num_envs|robot|project_name"
# Expect:
#   project_name: TRL_X2Ultra_BonesSeed_SphereFeet
#   robot: { type: x2_ultra, foot: sphere }
#   motion_file: gear_sonic/data/motions/x2_ultra_bones_seed.pkl
```

### 11b. Variant 1 — fine-tune the existing 16k mesh-trained checkpoint (recommended first)

Fastest path to an answer: load the 16k mesh-trained weights with
`+checkpoint=` (weight-only load, fresh optimizer / LR scheduler — see
§10), and let the policy adapt to the sphere contact regime for a few
thousand more iters.

```bash
# (cloud) Make sure the 16k mesh checkpoint is on the node. If you don't
# already have it, scp it over from your workstation:
#   scp ~/x2_cloud_checkpoints/run-20260420_083925/model_step_016000.pt \
#       ubuntu@$PUBLIC_IP:~/x2_cloud_checkpoints/run-20260420_083925/
ls -lh ~/x2_cloud_checkpoints/run-20260420_083925/model_step_016000.pt

tmux new -d -s sphere_ft "
  NUM_ENVS=8192 \
  NUM_ITERS=4000 \
  EXP_NAME=sonic_x2_ultra_bones_seed_sphere_feet \
  EXTRA_FLAGS='+checkpoint=$HOME/x2_cloud_checkpoints/run-20260420_083925/model_step_016000.pt' \
  USE_WANDB=False \
  LOG_FILE=\$HOME/sphere_ft.log \
  bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
"
tmux a -t sphere_ft     # attach; Ctrl-b d to detach
```

> `EXP_NAME` and `EXTRA_FLAGS` are forwarded into the Hydra invocation
> by `run_smoke_8gpu.sh` (see the script's header comment for the full
> override list). `EXTRA_FLAGS` is appended raw, so multi-flag values
> like `EXTRA_FLAGS='+checkpoint=... ++algo.config.lr=5e-5'` are fine.

Budget envelope (8x H200 SXM, `NUM_ENVS=8192`, ~4-5 s/iter):

| Iters | Wall clock | Cost @ ~$30/hr |
|---|---|---|
| 2,000 (early signal) | ~2-3 hrs | ~$60-90 |
| 4,000 (likely converged delta) | ~5-6 hrs | ~$150-180 |

### 11c. Variant 2 — train from scratch on spheres (only if Variant 1 stalls)

Same launcher pattern, no `+checkpoint=`, full 20-30k iter budget. Use
this only after Variant 1 has shown the gap is closing — Variant 2
costs ~$1k+ on H200 SXM and should not be the first attempt.

```bash
tmux new -d -s sphere_full "
  NUM_ENVS=16384 \
  NUM_ITERS=20000 \
  EXP_NAME=sonic_x2_ultra_bones_seed_sphere_feet \
  USE_WANDB=False \
  LOG_FILE=\$HOME/sphere_full.log \
  bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
"
```

### 11d. Validating the result (after pulling the new checkpoint locally)

```bash
# (local) IsaacLab eval on spheres — confirms training transferred
conda run -n env_isaaclab --no-capture-output python \
  gear_sonic/scripts/sweep_isaac_mujoco_mirror.py \
  --rows A0_isaac_stock A3_sphere_feet A5_full_mirror \
  --checkpoint-root ~/x2_cloud_checkpoints/<sphere-feet-run-dir> \
  --checkpoints <step>

# (local) MuJoCo benchmark — the actual sim2sim measurement
conda run -n env_mujoco --no-capture-output python \
  gear_sonic/scripts/benchmark_motions_mujoco.py \
  --checkpoint ~/x2_cloud_checkpoints/<sphere-feet-run-dir>/model_step_<step>.pt \
  --motion-file gear_sonic/data/motions/x2_ultra_standing_only.pkl \
  --num-motions 50 --seed 0 --max-seconds 6.0
```

What success looks like:

- `A3_sphere_feet` row drops to <0.30 termination (was 0.67 with the
  mesh-trained 16k policy).
- MuJoCo `bench_step*.csv` mean survival on the standing benchmark
  comes back up toward 4-6 s and **stops degrading** as iters increase
  (MuJoCo currently goes 2.98s @ 2k → 2.12s @ 16k with the mesh-trained
  policy; the sphere-trained policy should reverse that direction).

If both checks pass, the sim2sim gap on the standing benchmark is closed
and Open Work #1 in [`sim2sim_mujoco.md`](sim2sim_mujoco.md) graduates
from "diagnosed" to "fixed". Walking motions (G17) are a separate bucket
and likely still need the joint-DR pass from Open Work #2.

## Adapting this guide to a different embodiment / dataset

Everything in this document generalizes — only the bundle contents change.
For a new robot or new motion library:

1. Build the new motion-lib PKL with
   [`gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`](../../../gear_sonic/data_process/convert_soma_csv_to_motion_lib.py)
   (or the embodiment-specific wrapper, e.g.
   [`build_x2_bones_seed_motion_lib.py`](../../../gear_sonic/data_process/build_x2_bones_seed_motion_lib.py)).
2. Add an experiment yaml under `gear_sonic/config/exp/...` that inherits
   from the closest working base config and overrides
   `manager_env.commands.motion.motion_lib_cfg.motion_file`,
   `project_name`, `num_envs`, and `num_learning_iterations`.
3. Update the bundle's file list (Step 1) to include the new PKL(s) and the
   new yaml. Everything else in this guide stays the same.

## Appendix A — Provisioning on Nebius

Nebius AI Cloud (https://nebius.com) is a good fit for this workload: it
ships pre-built Ubuntu + CUDA images, bills per second, and offers 8x H100
SXM nodes at the lower end of the multi-GPU price range.

This appendix replaces sections 2 and the SSH parts of section 5 with a
Nebius-specific recipe. Once your instance is up and you've SSHed in,
return to **Step 3** and follow the rest of the guide unchanged.

### A.1. One-time CLI + auth setup (local workstation)

```bash
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="$HOME/.nebius/bin:$PATH"            # also added to ~/.bashrc by installer
nebius version --full

nebius profile create                            # interactive: browser SSO + region pick
nebius iam whoami                                # confirm tenant/project

PROJECT_ID=$(nebius iam project list --format json | jq -r '.items[0].metadata.id')
echo $PROJECT_ID                                 # save this; used below
```

> **SSH keys note.** This CLI version (0.12.x) does not have a top-level
> `iam ssh-key` resource. Public keys are injected via the cloud-init
> `users:` block on the instance itself — see the cloud-init template in
> A.3. Keep `~/.ssh/id_ed25519.pub` ready.

### A.2. Pick an instance type

Verified offerings (April 2026):

| Platform | Preset | GPUs | vCPU / RAM | List price | Region(s) |
|---|---|---|---|---|---|
| `gpu-h100-sxm` | `8gpu-128vcpu-1600gb` | 8x H100 80 GB SXM | 128 / 1600 GiB | ~$28/hr | `eu-north1` |
| `gpu-h200-sxm` | `8gpu-128vcpu-1600gb` | 8x H200 141 GB SXM | 128 / 1600 GiB | ~$36-40/hr | `eu-north1`, `eu-north2`, `eu-west1`, `us-central1` |
| `gpu-h200-sxm` | `1gpu-16vcpu-200gb` | 1x H200 141 GB SXM | 16 / 200 GiB | ~$5/hr | same regions; allocates reliably |

**Default recommendation: 8x H100 SXM in `eu-north1`** — cheapest 8-GPU
config that comfortably supports `++num_envs=4096-6144` per GPU for the X2
BONES-SEED config. Pick the H200 preset if you need more VRAM headroom or a
US/west-EU region for `scp` latency.

**Use the 1×H200 preset as a path-validation step.** When 8-GPU presets
keep husking (B.4), allocate a 1-GPU H200 first (~30 s create, real
public IP, real SSH access), run the bootstrap end-to-end on it, and
either fine-tune small or hold the disk while you keep retrying for an
8x slot in another region. This was the only path that worked on
2026-04-28 when 8xH200 in `eu-west1` was advised "4/4 HIGH" but kept
producing husks.

The boot-disk image to use is **`ubuntu24.04-cuda13.0`** — Ubuntu 24.04
with NVIDIA driver 580.x and CUDA 13 runtime already installed, which
lets you skip driver install in Step 3. (The older `ubuntu24.04-cuda12`
family was retired from `project-e00public-images` in April 2026 — see
B.2. Our `torch 2.7.0+cu128` from `isaacsim==5.1.0.0` runs fine under
the cuda13 driver via forward-compat.)

### A.3. Create the boot disk and instance

The CLI requires creating the boot disk separately and then referencing it
when launching the instance. We also bake the SSH key into a cloud-init
`users:` block.

```bash
PROJECT_ID=<from A.1>
SUBNET_ID=$(nebius vpc subnet list --parent-id $PROJECT_ID --format json \
  | jq -r '.items[0].metadata.id')
PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"

# 1. Provision the boot disk from the public Ubuntu 24.04 + CUDA 13 image.
#    (cuda12 family was retired April 2026; cuda13.0 is the current default.)
nebius compute disk create \
  --parent-id $PROJECT_ID \
  --name x2-train-h100-boot \
  --type network_ssd \
  --size-gibibytes 500 \
  --source-image-family-image-family ubuntu24.04-cuda13.0 \
  --source-image-family-parent-id project-e00public-images
DISK_ID=$(nebius compute disk get-by-name --parent-id $PROJECT_ID \
  --name x2-train-h100-boot --format json | jq -r '.metadata.id')

# 2. Cloud-init: install helper packages and inject the SSH pubkey.
cat > /tmp/cloud-init.yaml <<EOF
#cloud-config
package_update: true
packages:
  - tmux
  - htop
  - rsync
  - jq
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - ${PUBKEY}
runcmd:
  - [ bash, -lc, "ulimit -n 65536" ]
EOF

# 3. Launch the 8x H100 SXM instance.
nebius compute instance create \
  --parent-id $PROJECT_ID \
  --name x2-train-h100 \
  --resources-platform gpu-h100-sxm \
  --resources-preset 8gpu-128vcpu-1600gb \
  --boot-disk-existing-disk-id $DISK_ID \
  --boot-disk-attach-mode read_write \
  --network-interfaces '[{"name":"eth0","subnet_id":"'"$SUBNET_ID"'","ip_address":{},"public_ip_address":{}}]' \
  --cloud-init-user-data "$(cat /tmp/cloud-init.yaml)"
```

Notes on the flags:

- `--source-image-family-parent-id project-e00public-images`: that's
  Nebius's catalog of public stock images. The `ubuntu24.04-cuda13.0`
  family ships with NVIDIA driver 580.x and CUDA 13 already installed.
  If you see `SourceImageFamily: Image family "..." not found`, the
  family was renamed in the catalog — list current families with
  `nebius compute image list --parent-id project-e00public-images
  --format json | jq -r '.items[].spec.image_family'`, or inspect an
  existing-disk's source via `nebius compute disk get --id <id>`.
- `--network-interfaces '...public_ip_address...'`: assigns a public IP
  for `ssh`/`scp`. Drop the `public_ip_address` key if you have a
  bastion or VPN.
- `--cloud-init-user-data`: cloud-init runs once on first boot, installs
  `tmux/htop/rsync` and provisions the `ubuntu` user with your SSH key.

After ~3 minutes the instance is ready. Grab the IP and SSH in:

```bash
INSTANCE_ID=$(nebius compute instance list --parent-id $PROJECT_ID --format json \
  | jq -r '.items[] | select(.metadata.name=="x2-train-h100") | .metadata.id')
PUBLIC_IP=$(nebius compute instance get --id $INSTANCE_ID --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address')
echo $PUBLIC_IP

ssh ubuntu@$PUBLIC_IP
nvidia-smi                                       # should show 8x H100, driver 580.x, CUDA 13
```

> **Husk-detection check.** Even after `state: RUNNING` and a public IP
> are reported, the GPU node may not actually exist. Run `bash -c
> '</dev/tcp/$PUBLIC_IP/22 && echo open'` from your workstation; a real
> instance opens port 22 within ~30 s of `RUNNING`, while a husk leaves
> port 22 closed indefinitely. See B.4 for the full pattern.

### A.4. Adjustments to the main flow

When running the rest of this guide on a Nebius node:

| Step in main guide | Nebius adjustment |
|---|---|
| Step 3 (Install Isaac Lab) | Skip the NVIDIA driver install — the `ubuntu24.04-cuda13.0` image already has it. The conda + Isaac Lab bits are unchanged. |
| Step 5 (`scp` bundle) | `scp /tmp/x2_cloud_bundle.tar.gz ubuntu@$PUBLIC_IP:~/` — at ~200 MB this finishes in 30-60 s on a typical home connection. |
| Step 8 (Smoke test) | Defaults (`NUM_ENVS=4096`, 200 iters) work on every card we tested. |
| Step 8b (Full run) | On H200 SXM (144 GB) we run `NUM_ENVS=16384` and measure 6.9 s/iter, 87% util, 48 GB used. On 80 GB H100 SXM, top out at `NUM_ENVS=8192` to keep some headroom for memory creep over multi-day runs. See §8b for a measured tuning table. |

### A.5. Cost discipline

Nebius bills per second, so **stop the instance whenever you're not actively
training** — disk-only charges are a small fraction of GPU-hour cost.

```bash
# Stop (no GPU charges, only disk):
nebius compute instance stop --id $INSTANCE_ID

# Resume — the public IP usually changes:
nebius compute instance start --id $INSTANCE_ID
PUBLIC_IP=$(nebius compute instance get --id $INSTANCE_ID --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address')

# Fully delete when done (releases disk too):
nebius compute instance delete --id $INSTANCE_ID
```

The Step 8a smoke test (200 iterations) costs roughly $5 on 8x H200 SXM
(~12 min wall-clock at $30/hr) — cheap insurance before kicking off the
multi-hour real run.

## Appendix B — Lessons learned (gotchas + fixes)

These are real issues we hit the first time we ran this end-to-end on a
fresh Nebius account. Pre-empting them on the next run saves ~30-60 min.

### B.0. Cheat sheet — coming back to a running training job

If you SSH'd back in after a break and just want to know "is it still
healthy", paste this:

```bash
ssh ubuntu@$PUBLIC_IP 'echo "--- iter ---"
grep -nE "Learning iteration" ~/train.log | tail -n 3
echo "--- timing ---"
grep -nE "Iteration time|ETA:" ~/train.log | tail -n 4
echo "--- gpu ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "--- errors (should be 0) ---"
grep -cE "Traceback|Fatal|^Error|FAIL|OOM|OutOfMemory" ~/train.log
echo "--- checkpoints saved so far ---"
ls -lh ~/GR00T-WholeBodyControl/logs_rl/TRL_X2Ultra_BonesSeed/manager/universal_token/all_modes/*/model_step_*.pt 2>/dev/null | tail -n 5'
```

Healthy looks like: monotonically increasing iter number, iter time stable,
GPU util 70-90%, GPU memory growing slowly but well under 144 GB, error
count zero. If memory has grown >2 GB/hr since the last check, plan to
kill the run at the next checkpoint and resume at a lower `NUM_ENVS`.

When training finishes (or you decide to stop early):

```bash
# (local) pull everything down
RUN_DIR=$(ssh ubuntu@$PUBLIC_IP \
  'ls -td ~/GR00T-WholeBodyControl/logs_rl/TRL_X2Ultra_BonesSeed/manager/universal_token/all_modes/* | head -1')
mkdir -p ~/x2_cloud_checkpoints
rsync -avz --partial \
  ubuntu@$PUBLIC_IP:"$RUN_DIR/" \
  ~/x2_cloud_checkpoints/$(basename "$RUN_DIR")/

# stop the meter (disk-only charges from here)
nebius compute instance stop --id $INSTANCE_ID

# ...or fully release the disk too when you're sure you're done
# nebius compute instance delete --id $INSTANCE_ID
```

### B.1. Nebius CLI surface

The CLI version we tested (`0.12.204`) has a few mismatches with the
"obvious" syntax someone might guess from other clouds:

| Wrong | Right | Why |
|---|---|---|
| `nebius --version` | `nebius version --full` | `--version` flag does not exist |
| `nebius iam ssh-key create ...` | inject the key via cloud-init `users:` block (see A.3) | no top-level `iam ssh-key` resource exists |
| `nebius compute instance create --ssh-keys ...` | same — use cloud-init | no `--ssh-keys` flag |
| `--boot-disk "size=...,source-image-family=..."` (single composite flag) | `nebius compute disk create` first, then attach with `--boot-disk-existing-disk-id` | no composite `--boot-disk` flag exists |

### B.2. Stock images live in `project-e00public-images`

When creating a disk, the `--source-image-family-parent-id` is **not** your
project — it's `project-e00public-images`. Current stock families
(verified 2026-04-28): `ubuntu24.04-cuda13.0` and
`ubuntu24.04-driverless`. The `ubuntu24.04-cuda12` family was retired
in April 2026; if a tutorial or older script references it, swap in
`ubuntu24.04-cuda13.0`. (Our `torch 2.7.0+cu128` from
`isaacsim==5.1.0.0` runs on either CUDA 12 or CUDA 13 driver via
forward-compat, so the runtime difference is invisible.) The
`mk8s-worker-node-...` families are for managed-Kubernetes nodes and
shouldn't be used for VMs — even though their names mention `cuda12.8`,
they are not drop-in replacements for the standalone `ubuntu24.04-*`
families.

To list the current catalog at any time:

```bash
nebius compute image list --parent-id project-e00public-images --format json \
  | jq -r '.items[] | "\(.spec.image_family)\t\(.metadata.name)"' | sort -u
```

### B.3. New-account permissions and billing

A freshly created tenant returns `PermissionDenied` on every
`compute disk create` / `compute instance create` until billing is
attached. Add a card in the web console **before** the first CLI attempt
or you'll spend 5 min debugging IAM that's actually a billing block.

### B.4. Web form vs. CLI for the actual `instance create` — and the husk pattern

`gpu-h100-sxm` and `gpu-h200-sxm` (8 GPUs) are routinely fully booked in
`eu-north1`. The CLI surfaces this as `NotEnoughResources` after a 5-min
scheduler timeout. **Curiously, the web console sometimes finds capacity
the same minute the CLI fails** — the two paths poll different scheduling
queues. If the CLI keeps timing out:

1. Try the web console once.
2. If both fail, switch to `us-central1` — different region, separate
   capacity pool, and the H200 SXM fleet there is usually freer.

When you do use the web form, **double-check the "Public IPv4" toggle** —
the default in some flows is *no public IP*, leaving you with only the
`10.x.x.x` internal address. Without it you can't `ssh`/`scp` from your
workstation.

**The "husk" failure mode (worse than `NotEnoughResources`).** Sometimes
`instance create` returns success, the API reports `state: STOPPED`,
and the disk binds to the instance — but the underlying GPU node was
never actually scheduled. Calling `instance start` even bumps the state
to `STARTING` → `RUNNING` and assigns a public IP, but TCP/22 stays
closed forever. Telltale signs of a husk in `nebius compute instance
get --id <id> --format json`:

| Field | Husk | Real |
|---|---|---|
| `metadata.resource_version` | `1` (never advanced past create) | `>= 2` after a successful start |
| `status.network_interfaces[0].public_ip_address` | `{}` (empty) on STOPPED, may "fill in" on STARTING but stays unreachable | populated and reachable on TCP/22 within ~30 s |
| TCP probe `</dev/tcp/$IP/22` from workstation | times out indefinitely | opens within ~30 s |
| `nvidia-smi` over SSH | never reachable | works |

There's no charge for husks (Nebius bills only when the GPU is truly
allocated), but they waste time. The fastest discriminator is the TCP/22
probe — if it doesn't open within 60 s of `RUNNING`, it's a husk; stop
the instance, optionally try a different region/preset, and don't
bother running the bootstrap until SSH succeeds. The husk objects can
sit indefinitely without cost; clean them up with `compute instance
delete` when you're sure you don't want to keep retrying that slot.

### B.5. The web form's username vs. SSH key comment

The Linux user that gets created on the VM is *not* the trailing
`<user>@<host>` comment in your SSH key. For `ubuntu24.04-cuda13.0` the
default user is `ubuntu`. Set the username to `ubuntu` in the web form
even if your key ends with `stickbot@laptop`.

### B.6. Conda Terms-of-Service prompt

Modern miniconda refuses `conda create` until you accept ToS for both
Anaconda channels. Run once per new node:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### B.7. Pin `isaacsim==5.1.0.0` (not 4.5)

`isaacsim==4.5.0` only ships wheels for Python 3.10. With Python 3.11
(which is what Isaac Lab v2.2.x wants), pip will say
`Could not find a version that satisfies the requirement isaacsim==4.5.0`.
Use `isaacsim[all,extscache]==5.1.0.0` to mirror the local dev env. The
matching IsaacLab git tag is `v2.2.0`.

### B.8. Omniverse EULA is interactive — pre-accept it

`./isaaclab.sh -i` will hang on
`Do you accept the EULA? (Yes/No):` if stdin is empty (which it is in any
non-tty automation). Export these **before** running it:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
```

If you forgot, just send "Yes" to the running tmux session:

```bash
ssh ubuntu@<ip> 'tmux send-keys -t setup "Yes" Enter'
```

### B.9. `flatdict` / setuptools 82 / build-isolation

This is the most painful one. The chain:

- `isaaclab` → depends on `flatdict==4.0.1`
- `flatdict==4.0.1`'s `setup.py` does `import pkg_resources`
- `setuptools` 81+ removed the public `pkg_resources` module
- pip 25's build isolation fetches the **latest** setuptools into the
  isolated build env, ignoring whatever is in your active env
- → `ModuleNotFoundError: No module named 'pkg_resources'`, every time

The fix that actually works:

```bash
pip install "setuptools==80.9.0" wheel
pip install --no-build-isolation "flatdict==4.0.1"
```

The `--no-build-isolation` is the load-bearing flag — pinning setuptools
in the env alone is not enough.

### B.10. `import isaaclab_assets` outside SimulationApp always fails

Don't waste time debugging
`ModuleNotFoundError: No module named 'carb'` from a smoke
`python -c "import isaaclab_assets"` check. `carb` is a native module
that lives inside Isaac Sim and is only on `sys.path` after
`SimulationApp(...)` has been instantiated. The same import fails on the
local dev box too. The real verification is to launch
`gear_sonic/train_agent_trl.py` — if Hydra resolves and the env builds,
everything is wired up.

### B.10b. Missing Python deps in `gear_sonic[training]`

Two packages are imported by `gear_sonic` but **not declared** in
`gear_sonic/pyproject.toml`'s `[training]` extra. On the local dev
machine they're present from prior installs, so the gap is invisible
there; on a clean cloud node `train_agent_trl.py` crashes mid-startup
with `ModuleNotFoundError`:

| Module | Imported by | Failure point |
|---|---|---|
| `open3d` | `gear_sonic/utils/motion_lib/torch_humanoid_batch.py` | env config import (very early) |
| `tensordict` | `gear_sonic/trl/modules/actor_critic_modules.py` | policy instantiate (after env build) |
| `vector_quantize_pytorch` | Hydra-instantiated by `config/actor_critic/quantizers/fsq.yaml` | policy instantiate |

Pin to the local env's versions and install all of them up-front:

```bash
pip install "open3d==0.19.0" "tensordict==0.12.1" "vector-quantize-pytorch==1.28.1"
```

All three should also be added to `gear_sonic/pyproject.toml`'s
`[training]` extra so future cloud spins don't trip on them.

### B.10c. `libGLU.so.1` missing on `ubuntu24.04-cuda12/cuda13.0`

Isaac Sim's optional Iray renderer plugin fails to load with
`libGLU.so.1: cannot open shared object file`. By itself this is a
warning (we run headless), but it's also a marker that the image is
slim on graphics libs. Install once per node:

```bash
sudo apt-get install -y libglu1-mesa
```

### B.10d. NVIDIA Vulkan ICD missing — Isaac Sim sees zero GPUs

This is the **biggest** Nebius gotcha and the one that masquerades as a
multi-GPU bug. Both `ubuntu24.04-cuda12` (retired April 2026) and the
current `ubuntu24.04-cuda13.0` images are *compute-only* NVIDIA
builds: `libnvidia-compute-580` is installed but `libnvidia-gl-580` (which
ships `nvidia_icd.json` for Vulkan plus `libGLX_nvidia` / `libEGL_nvidia`)
is **deliberately blocked** by an apt pin in
`/etc/apt/preferences.d/nvidia-lock`:

```
Package: libnvidia-gl*
Pin: release *
Pin-Priority: -1
```

Without the Vulkan ICD, Isaac Sim's Kit renderer can't enumerate any GPU
even in headless mode. You'll see:

```
[Error] [gpu.foundation.plugin] No device could be created.
[Error] [gpu.foundation.plugin] The chosen activeGpu index N is higher than the available GPUs.
[Error] [omni.kit.renderer.plugin] Graphics plugins not available
[Fatal] [omni.usd] attempted member lookup on NULL TfRefPtr<UsdStage>
| Driver Version: 0  | Graphics API: Vulkan      <-- empty GPU table
```

…even though `nvidia-smi` shows all 8 cards. Diagnose with `vulkaninfo
--summary`: a healthy box lists each GPU under `driverName = NVIDIA`. A
broken box lists nothing (or only `lvp` / `llvmpipe`).

Fix — override the pin and install the matching `libnvidia-gl` package:

```bash
# Driver version on the box (e.g. 580.126.09):
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRV_MAJOR=${DRV%%.*}
DRV_PIN=$(apt-cache madison libnvidia-gl-${DRV_MAJOR} \
  | awk -F'|' -v want="$DRV" 'index($2,want){gsub(/ /,"",$2); print $2; exit}')

sudo tee /etc/apt/preferences.d/allow-libnvidia-gl >/dev/null <<EOF
Package: libnvidia-gl-${DRV_MAJOR}
Pin: version ${DRV_PIN}
Pin-Priority: 700
EOF

sudo apt-get update
sudo apt-get install -y "libnvidia-gl-${DRV_MAJOR}=${DRV_PIN}" vulkan-tools

vulkaninfo --summary | grep -E "deviceName|driverName"   # sanity: one row per GPU
```

The version pin is critical — `libnvidia-gl-X.Y.Z` must match the running
driver exactly or Vulkan refuses to load it.

### B.11. `git-lfs` is required — robot meshes are LFS-tracked

`check_environment.py --training` flags this as a soft warning, but it's
**load-bearing** for training. The X2/G1 STL meshes referenced by the
URDF live in Git LFS (1,236 files repo-wide, ~110 MB just for x2_ultra
meshes). A `git clone` without LFS leaves only 132-byte pointer files
in their place. Isaac Sim's URDF importer then "succeeds" silently and
produces empty USD files (~492 bytes each), and you crash later with:

```
[Error] [omni.usd] Failed to open layer @/tmp/IsaacLab/usd_<ts>_<rnd>/configuration/pelvis.tmp.usd@
[Fatal] [omni.usd] attempted member lookup on NULL TfRefPtr<UsdStage>
```

Fix — install LFS **before** doing anything robot-related, then pull:

```bash
sudo apt-get install -y git-lfs
git lfs install --skip-repo

cd ~/GR00T-WholeBodyControl
git lfs pull --include "gear_sonic/data/assets/robot_description/**"
# spot-check: a real mesh, not a 132-byte pointer
ls -lh gear_sonic/data/assets/robot_description/urdf/x2_ultra/meshes/pelvis.STL
# expect: ~2.8 MB (a healthy STL), not 132 bytes
```

If you've already run training with the bad meshes, also clear the bad
USD cache so the next launch reconverts cleanly:

```bash
rm -rf /tmp/IsaacLab/usd_*
```

### B.12. Benign pip "incompatible version" warnings

After installing `gear_sonic[training]` you'll see:

```
isaacsim-kernel 5.1.0.0 requires click==8.1.7, but you have click 8.3.2
isaacsim-kernel 5.1.0.0 requires numpy==1.26.0, but you have numpy 1.26.4
isaacsim-kernel 5.1.0.0 requires Pillow==11.3.0, but you have pillow 11.2.1
```

These are over-tight pins inside `isaacsim-kernel`'s metadata. The local
env has the same warnings and trains fine. Ignore.

### B.13. Realistic timing (8x H200 SXM, fast network)

Bare timings, validated 2026-04-28 on a 1×H200 in `eu-west1` running
the bootstrap script unattended. (8×H200 numbers are essentially
identical for everything except `scp` bandwidth — the bottleneck is the
torch wheel + IsaacLab clone, not GPU count.)

| Bootstrap phase | Wall-clock | Notes |
|---|---|---|
| 0–1 (pre-flight + apt) | ~30 s | hits Ubuntu mirrors |
| 2 (NVIDIA Vulkan ICD pin override) | ~50 s | B.10d fix |
| 3 (EULA env vars in `~/.bashrc`) | <1 s | |
| 4 (Miniconda install) | ~10 s | |
| 5 (conda ToS) | ~2 s | B.6 fix |
| 6 (env_isaaclab create) | ~10 s | |
| 7 (setuptools 80.9.0 + flatdict) | ~2 s | B.9 fix |
| 8 (`isaacsim==5.1.0.0` + torch 2.7.0+cu128) | ~6 min | dominated by 7 GB wheel download |
| 9 (IsaacLab v2.2.0 clone + install) | ~3 min | |
| 10 (git-lfs init) | <1 s | |
| 11 (GR00T repo clone + LFS pull meshes) | ~30 s | depends on REPO_BRANCH size |
| 12 (`pip install -e gear_sonic[training]` + extras) | ~30 s | |
| 13 (validation: vulkaninfo + Hydra dry-compose) | ~30 s | |
| **Total bootstrap** | **~12–15 min** | from clean cuda13.0 disk |

Add to that:

| Step | Wall-clock |
|---|---|
| Local bundle build (incl. 211 MB bones_seed.pkl) | <5 s |
| `scp` 211 MB bundle (home wifi → eu-west1) | ~50 s |
| `scp` 380 MB checkpoint (when fine-tuning, §11b) | ~50 s |
| Nebius `instance create` → SSH-able | ~30 s (1×H200) / ~2 min (8×H200, when capacity is real) |
| Husk-detection wasted time when 8-GPU advice lies (B.4) | 5–60 min lost per husk; mitigate by 1×H200 first |

Total: **~15–20 min from `nebius profile create` to a green smoke test**,
assuming GPU capacity is available on the first try.

### B.14. Reusable artifacts

Two things on the cloud node are worth keeping across instances:

- **The boot disk** (`x2-train-h100-boot`). When you `delete` an instance,
  the disk persists. Re-attach it to a new instance and you skip the
  entire 15-min installer.
- **`~/x2_cloud_bundle.tar.gz`** lives on that disk. As long as the
  bundle's contents (the PKLs, the new yaml, the helper scripts) haven't
  changed locally, you don't need to re-`scp` it.

If you're going to delete the instance and recreate it, **stop** rather
than `delete` whenever possible — stop releases the GPU charges but keeps
the boot disk attached.
