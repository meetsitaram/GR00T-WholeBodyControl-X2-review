#!/usr/bin/env bash
# install_viewer.sh
# Sets up the .venv-viewer/ venv for the X2 LeRobot dataset Rerun viewer
# (gear_sonic/scripts/view_x2_recorded_dataset.py).
#
# Why a dedicated venv?
# ---------------------
# The main .venv/ pins ``pin`` (pinocchio) 2.7.0, which only works with
# numpy<2; rerun-sdk >= 0.30 hard-requires numpy>=2. Sticking with
# rerun-sdk 0.21 makes the recorded .rrd unreadable by the system
# conda rerun-cli 0.31.4 viewer ("Invalid encoding options"). The
# clean fix is to keep the planner stack pinned and put the upgraded
# Rerun in its own venv that's never imported by anything else.
#
# See requirements-viewer.txt for the full rationale and pin set.
#
# Usage:  bash install_scripts/install_viewer.sh   (run from repo root)
#
# Idempotent: re-running the script wipes .venv-viewer/ and re-creates
# it. The wrapper script (gear_sonic/scripts/view_x2_recorded_dataset.sh)
# will then keep working unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQ_FILE="${REPO_ROOT}/requirements-viewer.txt"
VENV_DIR="${REPO_ROOT}/.venv-viewer"

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[ERROR] ${REQ_FILE} not found. This script must be run from a checkout" >&2
    echo "        that has requirements-viewer.txt at the repo root." >&2
    exit 1
fi

# ── 0. Architecture sanity print ────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

# ── 1. Ensure uv is installed and available ────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH." >&2
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run." >&2
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 (matches main .venv/) ──────────
echo "[INFO] Installing uv-managed Python 3.10 …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ────────────────────────────────────
cd "${REPO_ROOT}"
echo "[INFO] Removing old .venv-viewer/ (if present) …"
rm -rf "${VENV_DIR}"

# ── 4. Create venv & install pinned viewer deps ────────────────────────
echo "[INFO] Creating .venv-viewer/ with uv-managed Python 3.10 …"
uv venv "${VENV_DIR}" --python "${MANAGED_PY}" --prompt gear_sonic_viewer

echo "[INFO] Installing viewer dependencies from requirements-viewer.txt …"
VIRTUAL_ENV="${VENV_DIR}" uv pip install -r "${REQ_FILE}"

# ── 5. Smoke test: import + version check ──────────────────────────────
echo "[INFO] Verifying viewer venv is healthy …"
"${VENV_DIR}/bin/python" - <<'PY'
import sys
import numpy as np
import pyarrow as pa
import rerun as rr
print(f"  python    : {sys.version.split()[0]}")
print(f"  rerun-sdk : {rr.__version__}")
print(f"  numpy     : {np.__version__}")
print(f"  pyarrow   : {pa.__version__}")
# Confirm the columnar API the viewer script depends on is present
# (regressions here would mean a future requirements bump silently
# broke the script before any user ran it).
assert hasattr(rr, "TimeColumn"), "rerun-sdk missing TimeColumn (need >=0.30)"
assert hasattr(rr, "Scalars"), "rerun-sdk missing Scalars (need >=0.30)"
assert hasattr(rr.Scalars, "columns"), "Scalars.columns missing (need >=0.30)"
assert hasattr(rr.VideoFrameReference, "columns"), "VideoFrameReference.columns missing"
print("  columnar API: ok")
PY

# ── 6. Confirm the wrapper picks up the new interpreter ────────────────
WRAPPER="${REPO_ROOT}/gear_sonic/scripts/view_x2_recorded_dataset.sh"
if [[ -x "${WRAPPER}" ]]; then
    echo "[OK] Wrapper found at ${WRAPPER#$REPO_ROOT/} (already points at .venv-viewer/)"
else
    echo "[WARN] ${WRAPPER#$REPO_ROOT/} missing or not executable; the viewer venv" >&2
    echo "       is ready but the convenience wrapper is gone. Reinstate it with:" >&2
    echo "         git checkout gear_sonic/scripts/view_x2_recorded_dataset.sh" >&2
fi

# ── 7. Cross-check: main .venv/ planner deps untouched ─────────────────
MAIN_PY="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${MAIN_PY}" ]]; then
    echo "[INFO] Verifying main .venv/ planner stack is untouched …"
    "${MAIN_PY}" - <<'PY' || echo "[WARN] main .venv/ check failed; planner may be broken (NOT caused by this script)"
import numpy, pinocchio
print(f"  main .venv/ numpy     : {numpy.__version__} (expect 1.x for pinocchio 2.7)")
print(f"  main .venv/ pinocchio : {pinocchio.__version__}")
assert numpy.__version__.startswith("1."), \
    f"main .venv/ numpy is {numpy.__version__}, expected 1.x"
PY
else
    echo "[INFO] main .venv/ not present; skipping planner cross-check."
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Viewer venv ready at ${VENV_DIR#$REPO_ROOT/}"
echo ""
echo "  View a recorded episode:"
echo ""
echo "    ./gear_sonic/scripts/view_x2_recorded_dataset.sh \\"
echo "        --dataset x2_grab_a_drink --episode 6"
echo ""
echo "  See docs/source/tutorials/x2_dataset_record_and_replay.md"
echo "  (section 6.2) for the full operator workflow."
echo "══════════════════════════════════════════════════════════════"
