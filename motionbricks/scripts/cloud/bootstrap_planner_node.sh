#!/usr/bin/env bash
# Bootstrap a fresh Nebius (or any Ubuntu 24.04 + CUDA 13) GPU node into a
# state where ``run_planner_train.sh`` can train the MotionBricks-based
# X2 kinematic planner.
#
# This is the planner counterpart of
# ``gear_sonic/scripts/cloud/bootstrap_fresh_node.sh``. It is much simpler
# (~6 phases, ~5 min, no Isaac Lab / no Vulkan ICD pin / no EULA chase)
# because MotionBricks training is plain PyTorch + Lightning + MuJoCo — no
# IsaacSim involvement on the cloud node.
#
# What this script does NOT do:
#   - Install the NVIDIA driver or CUDA toolkit (assumes the boot image
#     already shipped them — Nebius ``ubuntu24.04-cuda13.0`` does).
#   - Pull the gitignored side-channel bundle (the BONES-SEED motion PKL).
#     That's an ``scp`` from your workstation; see
#     docs/source/user_guide/train-planner-on-cloud.md §1 + §5.
#
# Usage (on the cloud node, fresh ssh session):
#
#   wget https://raw.githubusercontent.com/<your-fork>/<branch>/motionbricks/scripts/cloud/bootstrap_planner_node.sh
#   # or scp from your workstation:
#   #   scp motionbricks/scripts/cloud/bootstrap_planner_node.sh ubuntu@$IP:~/
#
#   chmod +x bootstrap_planner_node.sh
#   REPO_URL=https://github.com/<your-fork>/GR00T-WholeBodyControl.git \
#     REPO_BRANCH=<your-branch> \
#     bash ~/bootstrap_planner_node.sh 2>&1 | tee ~/bootstrap.log
#
# Override knobs (all optional):
#   REPO_URL         git URL to clone the GR00T repo from              (required if repo not yet cloned)
#   REPO_BRANCH      branch to check out                               (default: main)
#   REPO_DIR         where to clone it                                 (default: $HOME/GR00T-WholeBodyControl)
#   CONDA_ENV        conda env name                                    (default: motionbricks)
#   PYTHON_VERSION   python version inside conda env                   (default: 3.10)
#   GH_TOKEN         GitHub PAT, used for cloning private repos        (default: empty)
#                    On your workstation: GH_TOKEN=$(gh auth token)
#                    Forward via ssh:    ssh ... GH_TOKEN=$GH_TOKEN bash bootstrap...
#                    Tokens are scrubbed from the on-disk remote URL after the
#                    clone so they don't show up in `git remote -v` later.

set -euo pipefail

REPO_URL=${REPO_URL:-}
REPO_BRANCH=${REPO_BRANCH:-main}
REPO_DIR=${REPO_DIR:-$HOME/GR00T-WholeBodyControl}
CONDA_ENV=${CONDA_ENV:-motionbricks}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
GH_TOKEN=${GH_TOKEN:-}

CONDA_PREFIX_DIR=${CONDA_PREFIX_DIR:-$HOME/miniconda3}

log()  { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
step() { printf '  - %s\n' "$*"; }
skip() { printf '  ~ skip: %s\n' "$*"; }

#-------------------------------------------------------------------------------
# Phase 0 — pre-flight
#-------------------------------------------------------------------------------
log "Phase 0: pre-flight"

# Phase 0a: install the NVIDIA kernel driver on the boot image if missing.
# As of May 2026 Nebius retired the standalone 'ubuntu24.04-cuda13.0' image
# family across most regions, leaving 'ubuntu24.04-driverless' (eu-north1 only)
# as the only workable public family. On a driverless image we have to bring
# the driver ourselves; PyTorch wheels ship the CUDA *runtime*, so we only
# need the kernel module + libcuda, not the full toolkit.
if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1; then
  step "NVIDIA driver already present:"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
else
  log "Phase 0a: NVIDIA driver missing -> installing (driverless image path)"
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "linux-headers-$(uname -r)" build-essential ca-certificates curl wget gnupg

  # NVIDIA CUDA keyring + repo (Ubuntu 24.04). Pinning to the keyring package
  # bootstraps the right apt source automatically.
  if [[ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]]; then
    KEYRING=/tmp/cuda-keyring_1.1-1_all.deb
    wget -q -O "$KEYRING" \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i "$KEYRING"
    sudo apt-get update -q
  fi

  step "installing cuda-drivers (just the driver, not the full toolkit)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers

  # Try to load the module without rebooting; cuda-drivers usually leaves the
  # kernel ready immediately, but if it fails we fall back to a reboot hint.
  sudo modprobe nvidia 2>/dev/null || true
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi still not working after install; a reboot may be required." >&2
    echo "      Run: sudo reboot, then re-execute this bootstrap script." >&2
    exit 1
  fi
  step "post-install nvidia-smi:"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
fi

#-------------------------------------------------------------------------------
# Phase 1 — OS packages
#-------------------------------------------------------------------------------
log "Phase 1: OS packages"
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  tmux htop rsync jq git git-lfs \
  build-essential ca-certificates curl wget \
  libgl1 libglu1-mesa libegl1 libosmesa6

#-------------------------------------------------------------------------------
# Phase 2 — Miniconda
#-------------------------------------------------------------------------------
log "Phase 2: Miniconda"
if [[ -d "${CONDA_PREFIX_DIR}" && -x "${CONDA_PREFIX_DIR}/bin/conda" ]]; then
  skip "${CONDA_PREFIX_DIR} already exists"
else
  step "downloading miniconda installer"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_PREFIX_DIR}"
  "${CONDA_PREFIX_DIR}/bin/conda" init bash >/dev/null
fi

# shellcheck disable=SC1091
source "${CONDA_PREFIX_DIR}/etc/profile.d/conda.sh"

#-------------------------------------------------------------------------------
# Phase 3 — conda ToS (B.6 from train-on-cloud.md)
#-------------------------------------------------------------------------------
log "Phase 3: conda ToS"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
step "ToS accept attempted for main + r channels"

#-------------------------------------------------------------------------------
# Phase 4 — conda env motionbricks
#-------------------------------------------------------------------------------
log "Phase 4: conda env '${CONDA_ENV}' (python=${PYTHON_VERSION})"
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  skip "env '${CONDA_ENV}' already exists"
else
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"
python -V

#-------------------------------------------------------------------------------
# Phase 5 — git-lfs init + clone the GR00T repo
#-------------------------------------------------------------------------------
log "Phase 5: GR00T-WholeBodyControl repo"
git lfs install --skip-repo
step "git-lfs hooks installed at user level"

if [[ -d "${REPO_DIR}/.git" ]]; then
  step "existing repo at ${REPO_DIR}; running git pull on '${REPO_BRANCH}'"
  (cd "${REPO_DIR}" && git fetch --all --quiet && git checkout "${REPO_BRANCH}" && git pull --ff-only)
else
  if [[ -z "${REPO_URL}" ]]; then
    echo "FATAL: REPO_DIR=${REPO_DIR} doesn't exist and REPO_URL is unset." >&2
    echo "       Re-run with: REPO_URL=https://github.com/<fork>/GR00T-WholeBodyControl.git $0" >&2
    exit 1
  fi
  step "cloning ${REPO_URL} (branch ${REPO_BRANCH}) into ${REPO_DIR}"
  # Inject the PAT into the URL if the repo is private. We strip it back out
  # after the clone so the remote URL stored in .git/config doesn't carry
  # secrets across the rest of the node lifetime.
  CLONE_URL="${REPO_URL}"
  if [[ -n "${GH_TOKEN}" && "${REPO_URL}" =~ ^https://github\.com/ ]]; then
    CLONE_URL="${REPO_URL/https:\/\//https://x-access-token:${GH_TOKEN}@}"
    step "GH_TOKEN provided -> using authenticated clone URL"
  fi
  # --depth 1 --single-branch: this monorepo has ~3GB of history (large
  # binaries committed directly to git, not LFS). We only need the latest
  # snapshot of the planner branch for training. GIT_LFS_SKIP_SMUDGE=1 keeps
  # the clone from auto-pulling every LFS object; we explicitly fetch only
  # the X2 robot description below.
  GIT_LFS_SKIP_SMUDGE=1 git clone \
    --depth 1 --single-branch \
    --branch "${REPO_BRANCH}" \
    "${CLONE_URL}" \
    "${REPO_DIR}"
fi

# Pull the X2 MJCF + meshes (~110 MB). The MotionBricks pipeline needs the
# x2_ultra.xml and its referenced STL meshes for FK extraction.
# NOTE: we deliberately do this BEFORE scrubbing the token from the remote URL,
# because git-lfs follows the remote and will hit the same auth wall on private
# repos otherwise.
step "git lfs pull (X2 MJCF + meshes only)"
(cd "${REPO_DIR}" && git lfs pull --include "gear_sonic/data/assets/robot_description/**")

if [[ -d "${REPO_DIR}/.git" && -n "${GH_TOKEN}" ]]; then
  (cd "${REPO_DIR}" && git remote set-url origin "${REPO_URL}")
  step "scrubbed token from .git/config (remote.origin.url reset to public form)"
fi

mesh_check="${REPO_DIR}/gear_sonic/data/assets/robot_description/urdf/x2_ultra/meshes/pelvis.STL"
if [[ -f "${mesh_check}" ]]; then
  size=$(stat -c%s "${mesh_check}")
  if [[ "${size}" -lt 1000 ]]; then
    echo "FATAL: ${mesh_check} is only ${size} bytes — LFS pull failed." >&2
    echo "       Try: cd ${REPO_DIR} && git lfs pull" >&2
    exit 1
  fi
  step "pelvis.STL = ${size} bytes (looks healthy)"
fi

#-------------------------------------------------------------------------------
# Phase 6 — pip install -e motionbricks/  + W&B
#-------------------------------------------------------------------------------
log "Phase 6: motionbricks editable install + extras"
cd "${REPO_DIR}"
pip install --quiet --upgrade pip
pip install --quiet -e "motionbricks/"

# wandb is optional but cheap; install so --use-wandb works without re-pip.
pip install --quiet "wandb"

# joblib for the smoke PKL builder; mujoco for FK; pytorch_lightning for trainer.
# All three should already be pulled in by motionbricks/setup.py, but pin
# wandb/joblib explicitly for reproducibility:
pip install --quiet "joblib"

step "installed motionbricks (editable) + wandb + joblib"

#-------------------------------------------------------------------------------
# Phase 7 — Validation
#-------------------------------------------------------------------------------
log "Phase 7: validation"

step "torch + lightning CUDA visibility:"
python - <<'PY'
import torch, pytorch_lightning as pl
print(f"  torch              : {torch.__version__}")
print(f"  cuda available     : {torch.cuda.is_available()}")
print(f"  cuda devices       : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"    [{i}] {torch.cuda.get_device_name(i)}")
print(f"  pytorch_lightning  : {pl.__version__}")
PY

step "motionbricks importable:"
python - <<'PY'
import motionbricks
from motionbricks.motionlib.core.skeletons import x2 as x2_skel  # noqa: F401
from motionbricks.data.x2_pkl_to_motion import default_x2_mjcf_path
print(f"  motionbricks       : {motionbricks.__file__}")
print(f"  X2 MJCF default    : {default_x2_mjcf_path()}")
PY

step "(optional) train_vqvae_x2.py --help:"
(cd "${REPO_DIR}" && python motionbricks/scripts/train_vqvae_x2.py --help \
  | grep -E "devices|num-nodes|strategy|resume|use-wandb" | head -10) \
  || echo "  (--help failed; check the install)"

cat <<EOF

================================================================================
Bootstrap complete.
================================================================================
Next steps:
  1. From your workstation, scp the side-channel bundle (BONES-SEED PKL +
     optional smoke PKL):
       bash motionbricks/scripts/cloud/build_planner_bundle.sh
       scp /tmp/x2_planner_bundle.tar.gz ubuntu@<this node>:~/
  2. Extract from the repo root on the cloud node:
       cd ${REPO_DIR}
       tar -xzf ~/x2_planner_bundle.tar.gz
  3. (One-time) build skeleton assets + stats:
       conda activate ${CONDA_ENV}
       python motionbricks/scripts/build_x2_skeleton_assets.py
  4. Smoke test (~3 min, ~\$2 on 8x H200):
       tmux new -d -s smoke "bash motionbricks/scripts/cloud/run_planner_smoke_8gpu.sh"
       tmux a -t smoke
  5. Full training run:
       tmux new -d -s plan_train "bash motionbricks/scripts/cloud/run_planner_train.sh"
       tmux a -t plan_train
================================================================================
EOF
