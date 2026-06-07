#!/usr/bin/env bash
# Wrapper that runs the Rerun viewer for a recorded X2 LeRobot v2.1
# dataset using the isolated ``.venv-viewer/`` interpreter.
#
# Why this exists
# ---------------
# The planner stack in ``.venv/`` pins ``rerun-sdk==0.21.0`` because
# the ``pin`` (pinocchio) 2.7.0 cmeel wheel requires ``numpy<2``, and
# ``rerun-sdk >= 0.30`` hard-requires ``numpy >= 2``. Trying to view
# H.264 mp4s from the conda-shipped ``rerun-cli 0.31.4`` against
# 0.21 SDK output fails silently (the conda viewer rejects the wire
# format with "Codec error: Invalid encoding options"). To keep the
# planner pin AND get a working viewer, we ship a second interpreter
# at ``.venv-viewer/`` with the upgraded stack:
#
#     .venv-viewer/  ← rerun-sdk 0.31.4 + numpy>=2 + pyarrow + pillow
#     .venv/         ← planner stack, untouched
#
# This wrapper forwards every CLI argument straight to the python
# script so existing muscle memory keeps working. See the script
# docstring at ``gear_sonic/scripts/view_x2_recorded_dataset.py``
# for the full argument list.
#
# Examples
# --------
#
#     ./gear_sonic/scripts/view_x2_recorded_dataset.sh \\
#         --dataset x2_grab_a_drink --episode 6
#
#     ./gear_sonic/scripts/view_x2_recorded_dataset.sh \\
#         --root data/lerobot/x2_pick_place_cams_v1 --episode 3 \\
#         --skip-scalars

set -euo pipefail

# Resolve repo root from this script's location so the wrapper still
# works if invoked from a different CWD (e.g. from another script).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIEWER_PY="${REPO_ROOT}/.venv-viewer/bin/python"

if [[ ! -x "${VIEWER_PY}" ]]; then
    echo "Error: viewer interpreter ${VIEWER_PY} not found." >&2
    echo "" >&2
    echo "Recreate the viewer venv with the idempotent installer:" >&2
    echo "    cd ${REPO_ROOT}" >&2
    echo "    bash install_scripts/install_viewer.sh" >&2
    echo "" >&2
    echo "Pinned dependencies live in requirements-viewer.txt at the" >&2
    echo "repo root; bump the rerun-sdk pin there if you upgrade the" >&2
    echo "system conda rerun-cli viewer." >&2
    exit 2
fi

cd "${REPO_ROOT}"
exec "${VIEWER_PY}" -m gear_sonic.scripts.view_x2_recorded_dataset "$@"
