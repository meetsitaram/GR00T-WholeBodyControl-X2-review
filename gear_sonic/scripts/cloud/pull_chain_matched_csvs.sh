#!/usr/bin/env bash
# Pull chain_matched retargeted CSVs from the Nebius cloud node to local disk.
#
# Default destination mirrors the bones-seed layout used by build_x2_bones_seed_motion_lib.py:
#   agibot-x2-references/bones-seed/retargeted/x2_chain_matched/{locowalk,locopost,...}/
#
# Usage:
#   # one-shot pull (assumes retarget is already complete):
#   bash gear_sonic/scripts/cloud/pull_chain_matched_csvs.sh
#
#   # wait until cloud has all 37,968 CSVs, then pull:
#   WAIT=1 bash gear_sonic/scripts/cloud/pull_chain_matched_csvs.sh
#
# Env:
#   CLOUD_HOST   SSH target (default: ubuntu@195.242.31.46)
#   CLOUD_SRC    remote retargeted dir
#   LOCAL_DST    local destination root
#   EXPECTED     total CSV count gate for WAIT=1 (default: 37968)
#   POLL_SEC     poll interval when WAIT=1 (default: 120)

set -euo pipefail

CLOUD_HOST="${CLOUD_HOST:-ubuntu@195.242.31.46}"
CLOUD_SRC="${CLOUD_SRC:-~/x2_retarget/bones-seed/retargeted/x2_chain_matched}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_DST="${LOCAL_DST:-$REPO_ROOT/agibot-x2-references/bones-seed/retargeted/x2_chain_matched}"
EXPECTED="${EXPECTED:-37968}"
POLL_SEC="${POLL_SEC:-120}"
WAIT="${WAIT:-0}"

remote_csv_count() {
  ssh "$CLOUD_HOST" "find ${CLOUD_SRC} -name '*.csv' -type f 2>/dev/null | wc -l" 2>/dev/null || echo 0
}

if [[ "$WAIT" == "1" ]]; then
  echo "=== waiting for cloud retarget to reach ${EXPECTED} CSVs (poll every ${POLL_SEC}s) ==="
  while true; do
    n="$(remote_csv_count)"
    ts="$(date +%H:%M:%S)"
    echo "[$ts] cloud CSVs: ${n} / ${EXPECTED}"
    if [[ "$n" -ge "$EXPECTED" ]]; then
      echo "Retarget complete — starting rsync."
      break
    fi
    # Also accept explicit done marker in retarget.log
    done_flag="$(ssh "$CLOUD_HOST" "grep -c 'PHASE 2 DENSE DONE' ~/retarget.log 2>/dev/null || true" | tr -d '[:space:]')"
    done_flag="${done_flag:-0}"
    if [[ "$done_flag" -ge 1 && "$n" -ge $((EXPECTED - 100)) ]]; then
      echo "Retarget log shows DONE with ${n} CSVs — starting rsync."
      break
    fi
    sleep "$POLL_SEC"
  done
fi

mkdir -p "$LOCAL_DST"
echo "=== rsync chain_matched CSVs ==="
echo "  from: ${CLOUD_HOST}:${CLOUD_SRC}/"
echo "  to:   ${LOCAL_DST}/"
echo

# -a archive, -z compress, --info=progress2 for throughput visibility
# --partial + --append-verify for resumability on flaky links
time rsync -az --partial --append-verify --info=progress2 \
  "${CLOUD_HOST}:${CLOUD_SRC}/" \
  "${LOCAL_DST}/"

echo
echo "=== verify local counts ==="
for tier in locowalk locopost locomanip locobal; do
  n="$(find "${LOCAL_DST}/${tier}" -name '*.csv' -type f 2>/dev/null | wc -l)"
  echo "  ${tier}: ${n} CSVs"
done
total="$(find "${LOCAL_DST}" -name '*.csv' -type f 2>/dev/null | wc -l)"
echo "  TOTAL: ${total} CSVs"
du -sh "$LOCAL_DST"
echo "Done."
