#!/usr/bin/env bash
# Bootstrap a fresh Nebius node (or any Ubuntu 24.04 + GPU) into a state ready
# to run the BONES-SEED -> X2 retargeting pipeline.
#
# Output of this script is a node where you can:
#   1. scp our newly-generated filename lists + bones-seed/scripts dir over
#   2. Download the 43GB soma_uniform.tar.gz from HuggingFace
#   3. Run extract_bones.py + retarget_x2_parallel.py + build_x2_bones_seed_motion_lib.py
#
# Designed to live alongside ``bootstrap_planner_node.sh`` but slimmer:
#   - Uses ``uv`` instead of conda (the soma-retargeter requires Python 3.12)
#   - Skips Lightning / wandb / motionbricks deps (this node only retargets)
#   - Installs joblib + scipy + numpy via the same ``uv`` venv to satisfy the
#     PKL build script ``gear_sonic/data_process/build_x2_bones_seed_motion_lib.py``.
#
# Usage (cloud-init or manual ssh):
#
#   GH_TOKEN=$(gh auth token) HF_TOKEN=$(cat ~/.cache/huggingface/token) \
#     bash bootstrap_retarget_node.sh 2>&1 | tee ~/bootstrap_retarget.log
#
# Env knobs (all optional unless noted):
#   GH_TOKEN          GitHub PAT for cloning private GR00T repo (required for private)
#   HF_TOKEN          HuggingFace token for downloading bones-studio/seed
#   REPO_URL          GR00T monorepo URL                              (default: meetsitaram fork)
#   REPO_BRANCH       branch                                          (default: planner-train)
#   REPO_DIR          local clone path                                (default: $HOME/GR00T-WholeBodyControl)
#   RETARGETER_URL    soma-retargeter URL (PUBLIC, no auth needed)    (default: meetsitaram/soma-retargeter agibot-x2 branch)
#   RETARGETER_DIR    where to clone retargeter                       (default: $HOME/agibot-x2-references/soma-retargeter)
#   BONES_SEED_DIR    where to put bones-seed data                    (default: $HOME/agibot-x2-references/bones-seed)

set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/meetsitaram/GR00T-WholeBodyControl.git}
REPO_BRANCH=${REPO_BRANCH:-planner-train}
REPO_DIR=${REPO_DIR:-$HOME/GR00T-WholeBodyControl}
RETARGETER_URL=${RETARGETER_URL:-https://github.com/meetsitaram/soma-retargeter.git}
RETARGETER_BRANCH=${RETARGETER_BRANCH:-agibot-x2}
RETARGETER_DIR=${RETARGETER_DIR:-$HOME/agibot-x2-references/soma-retargeter}
BONES_SEED_DIR=${BONES_SEED_DIR:-$HOME/agibot-x2-references/bones-seed}
GH_TOKEN=${GH_TOKEN:-}
HF_TOKEN=${HF_TOKEN:-}

log()  { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
step() { printf '  - %s\n' "$*"; }
skip() { printf '  ~ skip: %s\n' "$*"; }

#-------------------------------------------------------------------------------
# Phase 0 — pre-flight (NVIDIA driver if missing)
#-------------------------------------------------------------------------------
log "Phase 0: pre-flight"
if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1; then
  step "NVIDIA driver already present:"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
else
  log "Phase 0a: NVIDIA driver missing -> installing (driverless image path)"
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "linux-headers-$(uname -r)" build-essential ca-certificates curl wget gnupg
  if [[ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]]; then
    KEYRING=/tmp/cuda-keyring_1.1-1_all.deb
    wget -q -O "$KEYRING" \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i "$KEYRING"
    sudo apt-get update -q
  fi
  step "installing cuda-drivers"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers
  sudo modprobe nvidia 2>/dev/null || true
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi still not working after install; a reboot may be required." >&2
    exit 1
  fi
  step "post-install nvidia-smi:"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
fi

#-------------------------------------------------------------------------------
# Phase 1 — OS packages (slim, retargeter doesn't need OpenGL libs in null mode)
#-------------------------------------------------------------------------------
log "Phase 1: OS packages"
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  tmux htop rsync jq git git-lfs \
  build-essential ca-certificates curl wget python3-pip \
  libgl1 libglu1-mesa libegl1

#-------------------------------------------------------------------------------
# Phase 2 — uv (handles Python 3.12 download + venv creation for retargeter)
#-------------------------------------------------------------------------------
log "Phase 2: uv installation"
if command -v uv >/dev/null 2>&1; then
  skip "uv already installed: $(uv --version)"
else
  step "installing uv via official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Make uv visible in this script's PATH (installer puts it at ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
uv --version

#-------------------------------------------------------------------------------
# Phase 3 — clone soma-retargeter (PUBLIC, no PAT needed) + uv sync
#-------------------------------------------------------------------------------
log "Phase 3: soma-retargeter clone + venv"
mkdir -p "$(dirname "${RETARGETER_DIR}")"
if [[ -d "${RETARGETER_DIR}/.git" ]]; then
  step "existing retargeter at ${RETARGETER_DIR}; pulling ${RETARGETER_BRANCH}"
  (cd "${RETARGETER_DIR}" && git fetch --all --quiet && git checkout "${RETARGETER_BRANCH}" && git pull --ff-only)
else
  step "cloning ${RETARGETER_URL} (branch ${RETARGETER_BRANCH})"
  git clone --depth 1 --single-branch \
    --branch "${RETARGETER_BRANCH}" \
    "${RETARGETER_URL}" \
    "${RETARGETER_DIR}"
fi

step "uv sync — provisioning Python 3.12 + retargeter deps"
(cd "${RETARGETER_DIR}" && uv sync)

# Add joblib + scipy + numpy + huggingface_hub + hf_transfer + pandas + pyarrow
# for the PKL builder + bones-seed download + curate scripts (all share the
# same retargeter venv to keep it simple).
step "installing extra deps for PKL build + HF download + curation"
(cd "${RETARGETER_DIR}" && uv pip install --quiet \
   joblib pandas pyarrow huggingface_hub hf_transfer)

#-------------------------------------------------------------------------------
# Phase 4 — clone GR00T-WholeBodyControl (private) for build script + convert helpers
#-------------------------------------------------------------------------------
log "Phase 4: GR00T-WholeBodyControl repo"
git lfs install --skip-repo

if [[ -d "${REPO_DIR}/.git" ]]; then
  step "existing repo at ${REPO_DIR}; pulling ${REPO_BRANCH}"
  (cd "${REPO_DIR}" && git fetch --all --quiet && git checkout "${REPO_BRANCH}" && git pull --ff-only)
else
  CLONE_URL="${REPO_URL}"
  if [[ -n "${GH_TOKEN}" && "${REPO_URL}" =~ ^https://github\.com/ ]]; then
    CLONE_URL="${REPO_URL/https:\/\//https://x-access-token:${GH_TOKEN}@}"
    step "GH_TOKEN provided -> using authenticated clone URL"
  fi
  step "cloning ${REPO_URL} (branch ${REPO_BRANCH}) into ${REPO_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 git clone \
    --depth 1 --single-branch \
    --branch "${REPO_BRANCH}" \
    "${CLONE_URL}" \
    "${REPO_DIR}"
fi
# We don't need any LFS objects on this node — retargeting only touches
# CSV/BVH/parquet, never the .STL meshes. Skip ``git lfs pull``.
if [[ -n "${GH_TOKEN}" ]]; then
  (cd "${REPO_DIR}" && git remote set-url origin "${REPO_URL}")
  step "scrubbed token from .git/config"
fi

#-------------------------------------------------------------------------------
# Phase 5 — HF auth + download bones-seed (43GB)
#-------------------------------------------------------------------------------
log "Phase 5: bones-seed download"
mkdir -p "${BONES_SEED_DIR}"

if [[ -n "${HF_TOKEN}" ]]; then
  step "writing HF token to ~/.cache/huggingface/token"
  mkdir -p ~/.cache/huggingface
  printf '%s' "${HF_TOKEN}" > ~/.cache/huggingface/token
  chmod 600 ~/.cache/huggingface/token
fi

# Download script lives in bones-seed/scripts/ — but those are gitignored. We
# rsync them from the workstation in a follow-up step (run_after_bootstrap.sh).
# For now just kick off the HF download via inline python so we're truly hands-off.
step "downloading soma_uniform.tar.gz + metadata via HF (uses hf_transfer)"
(cd "${RETARGETER_DIR}" && uv run python - "${BONES_SEED_DIR}" <<'PY'
import os, sys
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
from huggingface_hub import snapshot_download
dest = sys.argv[1]
print(f'Downloading bones-studio/seed -> {dest}')
snapshot_download(
    repo_id='bones-studio/seed',
    repo_type='dataset',
    local_dir=dest,
    allow_patterns=[
        'README.md',
        'LICENSE.md',
        'metadata/*',
        'soma_shapes/*',
        'soma_uniform.tar.gz',
    ],
    max_workers=8,
)
print('Done.')
PY
)

#-------------------------------------------------------------------------------
# Phase 6 — Validation
#-------------------------------------------------------------------------------
log "Phase 6: validation"

step "retargeter import check + warp/newton init:"
(cd "${RETARGETER_DIR}" && uv run python - <<'PY' || true
import warp as wp
print(f"  warp.__version__: {wp.__version__}")
wp.init()
print(f"  warp devices    : {[d.name for d in wp.get_devices()]}")
import newton
print(f"  newton          : ok")
PY
)

step "bones-seed download summary:"
du -sh "${BONES_SEED_DIR}" 2>/dev/null || true
ls -la "${BONES_SEED_DIR}/"
[[ -f "${BONES_SEED_DIR}/soma_uniform.tar.gz" ]] && \
  step "soma_uniform.tar.gz: $(du -h "${BONES_SEED_DIR}/soma_uniform.tar.gz" | awk '{print $1}')"

cat <<EOF

================================================================================
Bootstrap complete (retargeter node).
================================================================================

Next steps (run from your workstation):

  1. Sync bones-seed scripts + filename lists to this node:
     rsync -av agibot-x2-references/bones-seed/scripts/ ubuntu@\$IP:${BONES_SEED_DIR}/scripts/
     rsync -av agibot-x2-references/bones-seed/x2-*-filenames.txt ubuntu@\$IP:${BONES_SEED_DIR}/

  2. SSH in and run the retarget pipeline:
     ssh ubuntu@\$IP
     cd ${BONES_SEED_DIR}
     # Extract the 4 curated subsets (~5 min):
     uv run --project ${RETARGETER_DIR} python scripts/extract_bones.py

     # Retarget with N parallel shards (need to add subset names to script first):
     # PARALLEL_SHARDS=32 RETARGET_SUBSETS=locowalk,locopost,locomanip,locobal \\
     #   uv run --project ${RETARGETER_DIR} python scripts/retarget_x2_parallel.py

     # Build merged PKL:
     uv run --project ${RETARGETER_DIR} python \\
       ${REPO_DIR}/gear_sonic/data_process/build_x2_bones_seed_motion_lib.py \\
       --subsets locowalk locopost locomanip locobal

  3. scp the resulting PKL back:
     scp ubuntu@\$IP:${REPO_DIR}/gear_sonic/data/motions/x2_ultra_bones_seed.pkl ./
================================================================================
EOF
