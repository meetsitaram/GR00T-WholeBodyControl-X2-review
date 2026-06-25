#!/usr/bin/env bash
# Pull chain_matched motion_lib PKL files from the Nebius cloud node to local disk.
#
# Round-2 corpus PKLs needed for SONIC fine-tuning and future kplanner re-runs:
#   x2_ultra_bones_seed_chain_matched.pkl         (37,968 clips, base)
#   x2_ultra_bones_seed_chain_matched_halfspeed.pkl  (~12K locowalk @ 0.5x)
#   x2_ultra_bones_seed_chain_matched_v2.pkl       (merged base + halfspeed)
#   x2_ultra_locowalk_chain_matched.pkl            (per-tier)
#   x2_ultra_locowalk_chain_matched_halfspeed.pkl  (per-tier halfspeed)
#   x2_ultra_locomanip_chain_matched.pkl
#   x2_ultra_locopost_chain_matched.pkl
#   x2_ultra_locobal_chain_matched.pkl
#
# Usage:
#   bash gear_sonic/scripts/cloud/pull_chain_matched_pkls.sh
#
# Env:
#   CLOUD_HOST   SSH target (default: ubuntu@195.242.31.46)
#   CLOUD_SRC    remote motions dir
#   LOCAL_DST    local motions dir

set -euo pipefail

CLOUD_HOST="${CLOUD_HOST:-ubuntu@195.242.31.46}"
CLOUD_SRC="${CLOUD_SRC:-~/GR00T-WholeBodyControl/gear_sonic/data/motions}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_DST="${LOCAL_DST:-$REPO_ROOT/gear_sonic/data/motions}"

mkdir -p "$LOCAL_DST"

echo "=== rsync chain_matched PKLs ==="
echo "  from: ${CLOUD_HOST}:${CLOUD_SRC}/"
echo "  to:   ${LOCAL_DST}/"
echo

# Pull only the bones_seed + per-tier chain_matched PKLs (NOT the dance/sitstand
# variants which live in subdirs and are managed separately).
# -a archive, -z compress, --partial + --append-verify for resumability.
time rsync -avz --partial --append-verify --info=progress2 \
  --include='x2_ultra_bones_seed_chain_matched*.pkl' \
  --include='x2_ultra_locowalk_chain_matched*.pkl' \
  --include='x2_ultra_locomanip_chain_matched*.pkl' \
  --include='x2_ultra_locopost_chain_matched*.pkl' \
  --include='x2_ultra_locobal_chain_matched*.pkl' \
  --exclude='*' \
  "${CLOUD_HOST}:${CLOUD_SRC}/" \
  "${LOCAL_DST}/"

echo
echo "=== verify local PKLs ==="
ls -lh "${LOCAL_DST}/"*chain_matched*.pkl 2>/dev/null | grep -E "bones_seed|locowalk|locomanip|locopost|locobal" || echo "(no matching PKLs)"
echo
total_bytes=$(du -sb "${LOCAL_DST}"/*chain_matched*.pkl 2>/dev/null | awk '{sum+=$1} END {print sum}')
echo "  total: $(numfmt --to=iec ${total_bytes:-0})"
echo "Done."
