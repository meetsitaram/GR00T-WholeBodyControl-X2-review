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
#   - motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/
#       hparams.yaml, skeleton/, stats/                 (~60 KB total)
#     These are the deterministic outputs of build_x2_skeleton_assets.py;
#     shipping them prebuilt saves a 5-8 minute CPU run on every cloud node.
#   - motionbricks/out/motionbricks_vqvae_x2/version_1/feature_cache/
#       *.pt + manifest.json                            (~500 MB; optional)
#     MuJoCo FK extraction + frame-band filter output. Pure CPU work, no GPU
#     benefit, so we run it locally once and ship. Without this, every cloud
#     node burns ~5 min of GPU-priced CPU on first dataset-load. The cache is
#     keyed by (motion_rep config, frame band) and silently bypasses the
#     per-clip pipeline when present. INCLUDE_FEATURE_CACHE=0 to skip
#     (e.g. when changing the loco filter set or feature dim).
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
#   OUT_TAR                path to write the bundle             (default: /tmp/x2_planner_bundle.tar.gz)
#   PKL_GLOB               glob of PKLs to bundle                (default: gear_sonic/data/motions/x2_ultra_*.pkl
#                          - matches per-tier PKLs locowalk/locopost/locomanip/locobal AND merged
#                          bones_seed PKL. Set to a single path to ship just one.)
#   INCLUDE_SMOKE          bundle the smoke PKL too             (default: 1; set 0 to skip)
#   INCLUDE_ASSETS         bundle prebuilt skeleton/stats/hparams (default: 1; set 0 to skip
#                          and force a fresh build_x2_skeleton_assets.py run on the cloud node)
#   INCLUDE_FEATURE_CACHE  bundle prebuilt MuJoCo FK feature cache (default: 1; set 0 to skip
#                          and force a fresh per-clip FK extraction on the cloud node, e.g.
#                          when you've changed the loco filter set or motion_rep feat_dim)

set -euo pipefail

OUT_TAR=${OUT_TAR:-/tmp/x2_planner_bundle.tar.gz}
PKL_GLOB=${PKL_GLOB:-gear_sonic/data/motions/x2_ultra_*.pkl}
INCLUDE_SMOKE=${INCLUDE_SMOKE:-1}
INCLUDE_ASSETS=${INCLUDE_ASSETS:-1}
INCLUDE_FEATURE_CACHE=${INCLUDE_FEATURE_CACHE:-1}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SMOKE_PKL="gear_sonic/data/motions/x2_ultra_planner_smoke.pkl"

# Resolve the PKL glob (nullglob so missing matches expand to nothing instead
# of the literal pattern).
shopt -s nullglob
pkl_paths=( $PKL_GLOB )
shopt -u nullglob

if [[ ${#pkl_paths[@]} -eq 0 ]]; then
  echo "FATAL: no PKLs matched $PKL_GLOB." >&2
  echo "       Build them first with gear_sonic/data_process/build_x2_bones_seed_motion_lib.py" >&2
  exit 1
fi

paths=( "${pkl_paths[@]}" )

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

# Skeleton + stats + hparams artifacts (output of build_x2_skeleton_assets.py).
# We only bundle hparams.yaml, skeleton/, stats/ — explicitly excluding
# checkpoints/ and feature_cache/ which can be many GB and aren't needed for
# bootstrapping a fresh training run.
if [[ "$INCLUDE_ASSETS" == "1" ]]; then
  asset_root="motionbricks/out"
  for variant in vqvae pose root; do
    vd="${asset_root}/motionbricks_${variant}_x2/version_1"
    if [[ ! -d "$vd" ]]; then
      echo "WARN: $vd missing -> skipping prebuilt asset bundling for $variant" >&2
      echo "      cloud node will fall back to build_x2_skeleton_assets.py" >&2
      continue
    fi
    for sub in hparams.yaml skeleton stats; do
      [[ -e "$vd/$sub" ]] && paths+=("$vd/$sub")
    done
  done
fi

# Feature cache (MuJoCo FK extraction output). The dataset class checks
# manifest.json + cached *.pt files and short-circuits if present, skipping
# the 5-8 min FK pass over BONES-SEED. Only the VQVAE variant builds the
# cache directly; pose/root re-use it via shared cache_dir, but the trainer
# scripts each look at their own variant dir, so we ship 3 copies (deduped
# by tar's hard-link detection — see -h flag below).
if [[ "$INCLUDE_FEATURE_CACHE" == "1" ]]; then
  vqvae_cache_root="motionbricks/out/motionbricks_vqvae_x2/version_1/feature_cache"
  if [[ -d "$vqvae_cache_root" ]]; then
    # Per-PKL convention: feature_cache/<pkl_stem>/manifest.json.
    # Ship every populated subdir so any of the PKLs (locowalk, locopost, ...)
    # in the bundle can find its matching cache. Cross-contamination is
    # impossible because each cache is in its own dir.
    shipped=0
    for sub in "$vqvae_cache_root"/*/; do
      [[ -d "$sub" ]] || continue
      manifest="$sub/manifest.json"
      if [[ -f "$manifest" ]]; then
        cache_count=$(ls "$sub"/*.pt 2>/dev/null | wc -l)
        echo "INFO: shipping feature_cache: $sub  ($cache_count clips, $(du -sh "$sub" | awk '{print $1}'))"
        paths+=("$sub")
        shipped=$((shipped + 1))
      fi
    done
    # Legacy flat layout: feature_cache/manifest.json (no subdir). Still ship
    # so older bundles work, but warn since this is unsafe for multi-PKL setups.
    if [[ -f "$vqvae_cache_root/manifest.json" ]]; then
      echo "WARN: legacy flat feature_cache/ detected -> shipping but recommend rebuilding under per-PKL convention" >&2
      paths+=("$vqvae_cache_root/manifest.json")
      paths+=($(ls "$vqvae_cache_root"/*.pt 2>/dev/null))
      shipped=$((shipped + 1))
    fi
    if [[ "$shipped" -eq 0 ]]; then
      echo "WARN: no populated feature_cache/<pkl>/ subdir found under $vqvae_cache_root" >&2
      echo "      run motionbricks/scripts/build_feature_cache_x2.py first" >&2
    fi
  else
    echo "WARN: $vqvae_cache_root missing -> skipping feature_cache bundling" >&2
  fi
fi

echo "Bundling ${#paths[@]} path(s) into $OUT_TAR ..."
for p in "${paths[@]}"; do
  size=$(du -shL "$p" 2>/dev/null | awk '{print $1}')
  echo "  + $p ($size)"
done

# -h follows symlinks (skeleton/ in pose_x2 and root_x2 is a symlink to the
# vqvae one) so the tarball contains real files, not dangling links the cloud
# node can't dereference.
tar -czhf "$OUT_TAR" "${paths[@]}"

ls -lh "$OUT_TAR"
sha=$(sha256sum "$OUT_TAR" | awk '{print $1}')
echo "  sha256: $sha"
echo
echo "Next:"
echo "  scp $OUT_TAR ubuntu@<cloud-ip>:~/"
echo "  # then on the cloud node, from the repo root:"
echo "  cd ~/GR00T-WholeBodyControl && tar -xzf ~/$(basename "$OUT_TAR")"
if [[ "$INCLUDE_FEATURE_CACHE" == "1" ]]; then
  echo "  # symlink pose+root feature_cache subdirs to the vqvae copy"
  echo "  # (per-PKL layout — preserves the safe-naming convention)"
  echo "  for v in pose root; do"
  echo "    src=\"\$PWD/motionbricks/out/motionbricks_vqvae_x2/version_1/feature_cache\""
  echo "    dst=\"motionbricks/out/motionbricks_\${v}_x2/version_1/feature_cache\""
  echo "    rm -rf \"\$dst\" && ln -sf \"\$src\" \"\$dst\""
  echo "  done"
fi
