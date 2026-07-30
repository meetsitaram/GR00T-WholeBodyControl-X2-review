# X2 Lean-Cut Setup (from scratch)

Validated end-to-end on 2026-07-28 with a fresh venv on Ubuntu (kernel 6.17),
Python 3.10.19, CPU-only torch. Every command below was actually run; expected
outputs are quoted from that run.

## TL;DR — one command

```bash
git clone <this-repo> && cd <repo> && bash install_scripts/setup_x2.sh
```

`install_scripts/setup_x2.sh` runs every manual step below in order (LFS pull,
venv, the validated pip sequence, model download into the cache, verification)
and is idempotent — safe to re-run. Optional flags: `--with-docker` builds the
`gear_sonic_deploy/docker_x2` sim deploy image (fully self-contained,
AgiBot's `aimdk_msgs` fetched automatically from their official SDK
artifact (see gear_sonic_deploy/thirdparty/aimdk_msgs/README.md); skipped
gracefully if docker is absent);
`--skip-models` skips the checkpoint download. The manual steps below remain
the reference for what the script does and for debugging a failed step.

## Prerequisites

- **Python 3.10** (`python3.10`). The packages pin `numpy==1.26.4` and require
  `python>=3.10`; 3.10 is the validated interpreter. Not present on newer
  Ubuntu (24.04 ships 3.12) — install via the deadsnakes PPA:
  `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install
  python3.10 python3.10-venv` (macOS: `brew install python@3.10`).
- **git-lfs**. Motion PKLs, MJCF meshes, and other binary data are LFS-tracked
  (~1300 files). On a fresh clone run:

  ```bash
  git lfs install
  git lfs pull
  ```

  Quick check that data is real and not a pointer stub:
  `ls -la gear_sonic/data/motions/x2_dances_easy.pkl` → ~10.6 MB (a pointer
  stub would be ~130 bytes).
- **Disk**: ~3 GB for the venv with CPU torch (CUDA torch is ~10 GB larger),
  plus ~0.5 GB for the model checkpoints below.
- No GPU, no docker, and no robot are needed for anything on this page.

## 1. Create the venv

```bash
cd <repo-root>
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
```

Note: `gear_sonic/scripts/run_x2_quest3_planner_stack.sh` auto-discovers
`<repo-root>/.venv/bin/python`, so creating the venv at the repo root (not
elsewhere) is the path of least resistance.

## 2. Install packages

Order matters — install the CPU torch wheel **first** so `gear_sonic`'s
`torch>=2.4.0` dependency is already satisfied and pip does not pull the
multi-GB CUDA wheel:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e "./gear_sonic[sim]"
.venv/bin/pip install onnxruntime
.venv/bin/pip install -e ./motionbricks
```

(GPU users: skip the `--index-url` and let pip pick the CUDA wheel.)

That is the complete list of manual installs. The root `pyproject.toml` is
tooling-config only — the installable packages are `gear_sonic/` (PEP 621
pyproject) and `motionbricks/` (setup.py). The `[sim]` extra brings mujoco,
tyro, pin, pyyaml, pyzmq, msgpack(-numpy), opencv-python on top of the base
deps (numpy 1.26.4, scipy 1.15.3, joblib, tqdm, easydict, loguru);
motionbricks brings hydra-core, omegaconf, pytorch-lightning, transformers,
pynput, matplotlib, vector-quantize-pytorch, colorlog, adam-atan2-pytorch.

`onnxruntime` is **not** declared in any package extra (known gap) and must be
installed explicitly — `eval_x2_mujoco_onnx.py` and the ONNX deploy path need
it.

See `requirements-x2.txt` for the same sequence in requirements form.
Validated versions: torch 2.13.0+cpu, mujoco 3.11.0, onnxruntime 1.23.2.

Sanity check:

```bash
.venv/bin/python -c "import gear_sonic, motionbricks, mujoco, onnxruntime, torch; print('ok')"
```

## 3. Download models

Model checkpoints are hosted on Hugging Face, **not** in the repo:
`tinkerbuggy/sonic-x2` (private at the moment — request access, then
`hf auth login`). They live in the **SONIC model cache** — a fixed,
multi-embodiment location the stack scripts probe automatically:

- root: `$SONIC_HOME`, default `~/.cache/sonic`
- X2 subtree: `$SONIC_HOME/x2` (override just this via `$SONIC_X2_MODELS`)
- G1 subtree: `$SONIC_HOME/g1`; future embodiments (e.g. asimov) slot in as
  `$SONIC_HOME/asimov`

One downloader serves every embodiment (`--robot x2` was added to the
upstream G1 script):

```bash
.venv/bin/pip install -U "huggingface_hub[cli]"
.venv/bin/python download_from_hf.py --robot x2   # tinkerbuggy/sonic-x2 -> ~/.cache/sonic/x2
.venv/bin/python download_from_hf.py --robot g1   # nvidia/GEAR-SONIC    -> ~/.cache/sonic/g1
```

Expected X2 layout (`~/.cache/sonic/x2/` used throughout this page):

```
~/.cache/sonic/x2/
├── sonic_policy/
│   ├── x2_sonic_policy.onnx      # 58 MB  — deploy/eval policy (obs 1670 → action 31)
│   └── x2_sonic_policy.pt        # 400 MB — training checkpoint, parity reference
├── kplanner_onnx/
│   ├── x2_kplanner_template.onnx # fused kinematic-planner graph
│   └── parity_report.json
└── kplanner_torch/
    ├── vqvae/  x2_kplanner_vqvae.ckpt + hparams.yaml + skeleton/ + stats/
    ├── pose/   x2_kplanner_pose.ckpt  + hparams.yaml + skeleton/ + stats/
    ├── root/   x2_kplanner_root.ckpt  + hparams.yaml + skeleton/ + stats/
    ├── x2_clip.ckpt              # pose-template library (planner modes)
    └── x2_clip.modes.json
```

## 4. Smoke tests

All commands run from the repo root with the venv python. Headless — no
display needed.

### 4.1 Policy eval (ONNX, MuJoCo, 20 sim-seconds)

```bash
.venv/bin/python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
    --motion gear_sonic/data/motions/x2_dances_easy.pkl \
    --no-viewer --total-sim-seconds 20
```

`--onnx` is omitted: the script auto-resolves the policy from the model cache
(`$SONIC_X2_MODELS` > `$SONIC_HOME/x2`) and prints the resolved path. Pass
`--onnx /path/to/other.onnx` to override.

Expected: loads the 743-frame @120 fps dance clip (6.2 s), RSI-inits from
frame 0, and runs ~3 full episodes. Every episode should end with
`reason: motion_end` (i.e. the robot survives the whole clip — a fall prints
`pelvis_z` / gravity reset reasons instead), pelvis height ~0.58–0.59 m,
finishing with:

```
[end] cumulative sim time 20.02s >= --total-sim-seconds=20.0s, exiting.
```

### 4.2 ONNX ↔ PT parity

```bash
.venv/bin/python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
    --motion gear_sonic/data/motions/x2_dances_easy.pkl \
    --no-viewer --total-sim-seconds 20 \
    --compare-pt ~/.cache/sonic/x2/sonic_policy/x2_sonic_policy.pt \
    --max-episode 10
```

The .pt drives the sim; the ONNX runs as a passive observer and per-step
action deltas are compared. Expected (validated run, 998 samples):

```
Max  |a_pt - a_onnx|_inf:  4.578e-05
Mean |a_pt - a_onnx|_inf:  1.183e-05
Threshold:             1.000e-04
Verdict:               PASS
```

Anything at the ~1e-5 scale is normal float32 export noise; the script fails
loudly above 1e-4. A CSV of per-step deltas lands in
`logs/x2/parity_pt_vs_onnx.csv`.

### 4.3 Stack scripts parse + preflight (no docker, no robot)

```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --help   # exit 0
bash gear_sonic_deploy/deploy_x2.sh --help                      # exit 0
```

Full argument-parse + preflight without spawning anything, via
`--validate-only` (heuristic planner uses the in-repo curated primitives, so
no checkpoints are needed):

```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --validate-only --no-deploy --no-record --no-x2-debug-bridge --planner heuristic
```

Expected: config banner, then
`validate-only: pre-flight passed; exiting before any spawn.` (exit 0).

The kplanner variant additionally probes that motionbricks imports under the
venv (with the model cache populated, both the fused planner graph and the
torch checkpoint tier auto-resolve — no paths needed):

```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --validate-only --no-deploy --no-record --no-x2-debug-bridge \
    --planner kplanner --kplanner-device cpu
```

(Explicit `--kplanner-{vqvae,pose,root}-ckpt PATH` overrides still win, as
does an exported `KPLANNER_ONNX=` graph path.)

Notes:

- `--no-x2-debug-bridge` is required for any run without a robot/PC2 host —
  the script refuses to guess the bridge host by design.
- If preflight reports a port in use (5556/5557/5563/5564/5565), something
  else on the machine holds it. `--pose-port <n>` moves the pose port.
  `--cleanup-only` force-frees the ports but **kills whatever holds them** —
  don't use it on a machine running another stack.
- With the model cache populated (step 3) the script's `--model` and SONIC
  tokenizer `.pt` default straight to the cached
  `sonic_policy/x2_sonic_policy.{onnx,pt}` — a full (non-validate) run needs
  no model flags. Resolution order: explicit `--model` / `X2_PLANNER_SMOKE_MODEL`
  env > `$SONIC_X2_MODELS` > `$SONIC_HOME/x2` > legacy machine-local paths.

## What is NOT covered here

- `deploy_x2.sh` beyond `--help`: local/sim modes need docker + the ROS 2
  deploy image; `onbot` needs the robot's PC2. Untestable without hardware.
- Launching the actual Quest3 stack (spawns manager/planner/recorder and,
  with deploy on, a docker sim container).
- GPU training / IsaacLab — out of scope for the lean cut's sim smoke.

## Known gaps found during validation (2026-07-28)

1. `onnxruntime` missing from every `gear_sonic` extra — manual install
   required (would fit naturally in `[sim]` or a new `[deploy]` extra;
   `setup_x2.sh` installs it for you).
2. ~~`download_from_hf.py` is G1-only~~ — FIXED: `--robot x2` downloads the
   `tinkerbuggy/sonic-x2` snapshot into the model cache.
3. ~~stack scripts default to machine-specific checkpoint paths~~ — FIXED:
   all stack entrypoints and `eval_x2_mujoco_onnx.py` now resolve models from
   the cache (`$SONIC_X2_MODELS` > `$SONIC_HOME/x2`) before the legacy paths.
4. Root `pyproject.toml` mentions `decoupled_wbc` extras in its header
   comment but the X2 sim path only needs `gear_sonic` + `motionbricks`.

No source files were missing from the lean cut — all three smokes ran from
the checkout as-is.

## VLA pipeline (record / replay / train / inference)

The X2 VLA workflow depends on **NVIDIA Isaac-GR00T as a pinned upstream
dependency** — no local modifications:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T external_dependencies/Isaac-GR00T
git -C external_dependencies/Isaac-GR00T checkout 3df8b38   # pinned; newer main likely fine
```
The train/inference scripts set `PYTHONPATH=external_dependencies/Isaac-GR00T`.

External models/data (not in this repo): base VLM `nvidia/Cosmos-Reason2-2B`
(HF, pulled by the finetune launcher), your recorded LeRobot-v2.1 dataset
(produced by `record_x2_dataset.py --with-record`; `features_x2_vla.py`
generates its `meta/modality.json`), and the SONIC motion-token decoder
checkpoint for runtime.

Entry points: record `record_x2_dataset.py`, replay
`run_x2_replay_stack.sh`, train `train_groot_vla.sh` (wraps
`launch_finetune_x2.py`), inference `run_x2_vla_runtime.sh`.

Known gap: `x2_pose_proxy.py` referenced by the replay/runtime stacks is a
PC2-side component not present in this repository — real-robot replay
requires the PC2 provisioning (see `pc2_bringup.sh`).
