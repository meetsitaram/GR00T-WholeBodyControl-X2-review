#!/usr/bin/env bash
# Bundle the gitignored / untracked artifacts the cloud planner-training node
# needs into a single tarball at /tmp/x2_planner_bundle.tar.gz.
#
# Mirrors gear_sonic/scripts/cloud/build_stand_idle_smoke.py + the doc
# instructions in docs/source/user_guide/train-on-cloud.md §1, but for the
# MotionBricks-based planner instead of the SONIC RL policy.
#
# What goes in the bundle:
#   - gear_sonic/data/motions/x2_ultra_bones_seed.pkl  (~210 MB; required)
#   - x2_ultra_planner_smoke.pkl                        (~3 MB; optional smoke)
#   - any local-only changes to motionbricks/scripts/build_x2_skeleton_assets.py
#     and x2_pkl_to_motion.py (source-of-truth on cloud comes from `git pull`,
#     but we tar them as a safety net for off-branch experimentation)
#
# What does NOT go in the bundle (lives in git, comes via `git pull`):
#   - The MJCF (gear_sonic/data/assets/.../x2_ultra.xml) — comes via git LFS.
#   - The X2Skeleton34 class, pkl_to_motion converter, train scripts —
#     all in motionbricks/, all tracked in git.
#
# Run from the repo root:
#
#   bash motionbricks/scripts/cloud/build_planner_bundle.sh
#
# Override knobs:
#   OUT_TAR        path to write the bundle      (default: /tmp/x2_planner_bundle.tar.gz)
#   INCLUDE_SMOKE  bundle the smoke PKL too      (default: 1; set 0 to skip)

set -euo pipefail

OUT_TAR=${OUT_TAR:-/tmp/x2_planner_bundle.tar.gz}
INCLUDE_SMOKE=${INCLUDE_SMOKE:-1}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BONES_PKL="gear_sonic/data/motions/x2_ultra_bones_seed.pkl"
SMOKE_PKL="gear_sonic/data/motions/x2_ultra_planner_smoke.pkl"

if [[ ! -f "$BONES_PKL" ]]; then
  echo "FATAL: $BONES_PKL not found." >&2
  echo "       Build it first with gear_sonic/data_process/build_x2_bones_seed_motion_lib.py" >&2
  echo "       (or the existing SONIC pipeline)." >&2
  exit 1
fi

paths=(
  "$BONES_PKL"
)

if [[ "$INCLUDE_SMOKE" == "1" ]]; then
  if [[ ! -f "$SMOKE_PKL" ]]; then
    echo "INFO: $SMOKE_PKL not found; building it now."
    if [[ -n "${MOTIONBRICKS_PYTHON:-}" ]]; then
      "$MOTIONBRICKS_PYTHON" motionbricks/scripts/cloud/build_planner_smoke_pkl.py
    elif command -v conda >/dev/null; then
      conda run -n motionbricks --no-capture-output python \
        motionbricks/scripts/cloud/build_planner_smoke_pkl.py
    else
      python motionbricks/scripts/cloud/build_planner_smoke_pkl.py
    fi
  fi
  if [[ -f "$SMOKE_PKL" ]]; then
    paths+=("$SMOKE_PKL")
  fi
fi

echo "Bundling ${#paths[@]} path(s) into $OUT_TAR ..."
for p in "${paths[@]}"; do
  size=$(du -h "$p" | awk '{print $1}')
  echo "  + $p ($size)"
done

tar -czf "$OUT_TAR" "${paths[@]}"

ls -lh "$OUT_TAR"
sha=$(sha256sum "$OUT_TAR" | awk '{print $1}')
echo "  sha256: $sha"
echo
echo "Next:"
echo "  scp $OUT_TAR ubuntu@<cloud-ip>:~/"
echo "  # then on the cloud node, from the repo root:"
echo "  cd ~/GR00T-WholeBodyControl && tar -xzf ~/$(basename "$OUT_TAR")"
