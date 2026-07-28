# X2 Lean-Cut Setup (from scratch)

Validated end-to-end on 2026-07-28 with a fresh venv on Ubuntu (kernel 6.17),
Python 3.10.19, CPU-only torch. Every command below was actually run; expected
outputs are quoted from that run.

## Prerequisites

- **Python 3.10** (`python3.10`). The packages pin `numpy==1.26.4` and require
  `python>=3.10`; 3.10 is the validated interpreter.
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
`tinkerbuggy/sonic-x2` (private at the moment — request access). The repo's
`download_from_hf.py` only fetches the G1 models from `nvidia/GEAR-SONIC`;
there is no X2 downloader script yet (known gap). Fetch with:

```bash
hf download tinkerbuggy/sonic-x2 --local-dir ~/x2_hf_staging
```

Expected local layout (`~/x2_hf_staging/` used throughout this page):

```
x2_hf_staging/
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
    --onnx ~/x2_hf_staging/sonic_policy/x2_sonic_policy.onnx \
    --motion gear_sonic/data/motions/x2_dances_easy.pkl \
    --no-viewer --total-sim-seconds 20
```

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
    --onnx ~/x2_hf_staging/sonic_policy/x2_sonic_policy.onnx \
    --motion gear_sonic/data/motions/x2_dances_easy.pkl \
    --no-viewer --total-sim-seconds 20 \
    --compare-pt ~/x2_hf_staging/sonic_policy/x2_sonic_policy.pt \
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
    --validate-only --no-deploy --no-x2-debug-bridge --planner heuristic
```

Expected: config banner, then
`validate-only: pre-flight passed; exiting before any spawn.` (exit 0).

The kplanner variant additionally probes that motionbricks imports under the
venv (checkpoint paths point at the HF layout from step 3):

```bash
bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --validate-only --no-deploy --no-x2-debug-bridge \
    --planner kplanner --kplanner-device cpu \
    --kplanner-vqvae-ckpt ~/x2_hf_staging/kplanner_torch/vqvae/x2_kplanner_vqvae.ckpt \
    --kplanner-pose-ckpt  ~/x2_hf_staging/kplanner_torch/pose/x2_kplanner_pose.ckpt \
    --kplanner-root-ckpt  ~/x2_hf_staging/kplanner_torch/root/x2_kplanner_root.ckpt
```

Notes:

- `--no-x2-debug-bridge` is required for any run without a robot/PC2 host —
  the script refuses to guess the bridge host by design.
- If preflight reports a port in use (5556/5557/5563/5564/5565), something
  else on the machine holds it. `--pose-port <n>` moves the pose port.
  `--cleanup-only` force-frees the ports but **kills whatever holds them** —
  don't use it on a machine running another stack.
- The script's default `--model` / SONIC tokenizer checkpoint paths point at
  machine-local checkpoint directories (known gap); a full (non-validate)
  run needs explicit `--model <onnx>` and `--sonic-checkpoint <pt>`.

## What is NOT covered here

- `deploy_x2.sh` beyond `--help`: local/sim modes need docker + the ROS 2
  deploy image; `onbot` needs the robot's PC2. Untestable without hardware.
- Launching the actual Quest3 stack (spawns manager/planner/recorder and,
  with deploy on, a docker sim container).
- GPU training / IsaacLab — out of scope for the lean cut's sim smoke.

## Known gaps found during validation (2026-07-28)

1. `onnxruntime` missing from every `gear_sonic` extra — manual install
   required (would fit naturally in `[sim]` or a new `[deploy]` extra).
2. `download_from_hf.py` is G1-only (`nvidia/GEAR-SONIC`); no downloader for
   the X2 checkpoints (`tinkerbuggy/sonic-x2`), layout documented above.
3. `run_x2_quest3_planner_stack.sh` defaults for `--model` and the SONIC
   tokenizer point at non-repo, machine-specific paths
   (`~/x2_cloud_checkpoints/...`); fresh installs must always pass them
   explicitly.
4. Root `pyproject.toml` mentions `decoupled_wbc` extras in its header
   comment but the X2 sim path only needs `gear_sonic` + `motionbricks`.

No source files were missing from the lean cut — all three smokes ran from
the checkout as-is.
