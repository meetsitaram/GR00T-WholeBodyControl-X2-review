#!/bin/bash
# Polls the cloud node for the round-2 X2 skeleton/stats/hparams artifacts,
# then rsyncs them to local under motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/.
#
# Idempotent: re-running just re-rsyncs (fast no-op if already complete).
# Usage:
#   bash gear_sonic/scripts/cloud/pull_motionbricks_assets.sh
#   WAIT=1 bash gear_sonic/scripts/cloud/pull_motionbricks_assets.sh   # poll until ready
set -euo pipefail

HOST=${HOST:-ubuntu@195.242.31.46}
REMOTE_BASE=${REMOTE_BASE:-/home/ubuntu/GR00T-WholeBodyControl}
LOCAL_BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WAIT=${WAIT:-1}
POLL_SEC=${POLL_SEC:-60}

VQVAE_HP="motionbricks/out/motionbricks_vqvae_x2/version_1/hparams.yaml"
POSE_HP="motionbricks/out/motionbricks_pose_x2/version_1/hparams.yaml"
ROOT_HP="motionbricks/out/motionbricks_root_x2/version_1/hparams.yaml"

ts() { date +%H:%M:%S; }
log(){ echo "[$(ts)] $*"; }

if [[ "$WAIT" == "1" ]]; then
  log "polling $HOST every ${POLL_SEC}s for $VQVAE_HP ..."
  while true; do
    if ssh "$HOST" "test -f $REMOTE_BASE/$VQVAE_HP && test -f $REMOTE_BASE/$POSE_HP && test -f $REMOTE_BASE/$ROOT_HP"; then
      log "all three hparams.yaml present on cloud"
      break
    fi
    # bail out if skeleton process died and hparams missing
    if ! ssh "$HOST" "pgrep -f build_x2_skeleton_assets >/dev/null"; then
      if ! ssh "$HOST" "test -f $REMOTE_BASE/$VQVAE_HP"; then
        log "ERROR: build_x2_skeleton_assets died and hparams.yaml missing on cloud"
        exit 2
      fi
    fi
    sleep "$POLL_SEC"
  done
fi

log "rsync skeleton/stats/hparams to local ..."
for variant in vqvae pose root; do
  sub="motionbricks/out/motionbricks_${variant}_x2/version_1"
  mkdir -p "$LOCAL_BASE/$sub/skeleton" "$LOCAL_BASE/$sub/stats/motion"
  # hparams.yaml (small file)
  rsync -av "$HOST:$REMOTE_BASE/$sub/hparams.yaml" "$LOCAL_BASE/$sub/hparams.yaml"
done

# skeleton + stats live under vqvae_x2/version_1 (other variants symlink)
sub="motionbricks/out/motionbricks_vqvae_x2/version_1"
rsync -av --no-links "$HOST:$REMOTE_BASE/$sub/skeleton/"      "$LOCAL_BASE/$sub/skeleton/"
rsync -av --no-links "$HOST:$REMOTE_BASE/$sub/stats/motion/"  "$LOCAL_BASE/$sub/stats/motion/"

log "done. local artifacts:"
ls -la "$LOCAL_BASE/$sub/hparams.yaml" \
       "$LOCAL_BASE/$sub/skeleton/" \
       "$LOCAL_BASE/$sub/stats/motion/"
