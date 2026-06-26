#!/bin/bash
# Side-eval loop for the Root fine-tune (PR #?).
#
# Watches motionbricks/out/motionbricks_root_x2/version_1/checkpoints/ for new
# ``model-step=*.ckpt`` files written by the running fine-tune, and for each
# new one runs ``replay_pkl_through_kplanner.py`` against a canonical
# forward-walk clip. Parses out the four headline metrics from the verdict
# block and appends to a CSV that can be ``tail -f``'d.
#
# This is the actual go/no-go signal for the fine-tune: train loss going down
# tells us nothing if the Root model's forward-walk trajectory undershoot and
# yaw drift are not improving.
#
# Usage (on the cloud node, in its own tmux pane):
#
#   tmux new -d -s root_eval "bash motionbricks/scripts/cloud/eval_root_finetune.sh"
#   tail -f ~/root_finetune_eval.csv
#
# Override knobs (env vars):
#   POLL_SEC          how often to look for new ckpts                 (default: 30)
#   START_STEP        ignore ckpts at or before this step              (default: 300000)
#   VQVAE_CKPT        VQVAE checkpoint (Round-2 frozen)                (default: 500K)
#   POSE_CKPT         Pose checkpoint (Round-2 frozen)                 (default: 500K)
#   ROOT_CKPT_GLOB    glob for Root checkpoints written by the run    (default: model-step=*.ckpt)
#   CLIP_PKL          motion lib                                       (default: x2_ultra_locowalk.pkl)
#   CLIP_KEY          clip key to replay                               (default: first forward-walk)
#   CSV_OUT           output CSV path                                  (default: ~/root_finetune_eval.csv)
#   STABLE_SEC        seconds a ckpt mtime must be stable before eval (default: 8)

set -euo pipefail

POLL_SEC=${POLL_SEC:-30}
START_STEP=${START_STEP:-300000}
VQVAE_CKPT=${VQVAE_CKPT:-motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0500000.ckpt}
POSE_CKPT=${POSE_CKPT:-motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/model-step=0500000.ckpt}
ROOT_CKPT_DIR=${ROOT_CKPT_DIR:-motionbricks/out/motionbricks_root_x2/version_1/checkpoints}
CLIP_PKL=${CLIP_PKL:-gear_sonic/data/motions/x2_ultra_locowalk_chain_matched.pkl}
# Pin the canonical forward-walk clip we used for the Layer-2 baseline
# diagnostic (Loop_Forward_Walk_001__A018, mirror=off). This makes
# step-to-step comparison meaningful: the same clip every time.
CLIP_KEY=${CLIP_KEY:-Loop_Forward_Walk_001__A018}
CSV_OUT=${CSV_OUT:-$HOME/root_finetune_eval.csv}
STABLE_SEC=${STABLE_SEC:-8}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Activate the same conda env the training uses.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate motionbricks
export PYTHONPATH="${REPO_ROOT}/motionbricks:${REPO_ROOT}"

# Pre-flight: confirm everything we need exists. Failing here is much better
# than failing inside the polling loop and leaving the user staring at an empty
# CSV.
for p in "$VQVAE_CKPT" "$POSE_CKPT" "$CLIP_PKL"; do
  if [[ ! -f "$p" ]]; then
    echo "ERROR: required file not found: $p" >&2
    exit 1
  fi
done
if [[ ! -d "$ROOT_CKPT_DIR" ]]; then
  echo "ERROR: ROOT_CKPT_DIR not a directory: $ROOT_CKPT_DIR" >&2
  exit 1
fi

# Build the CLI args used for every replay call. The replay script honors
# ``--root-ckpt`` for the model under test and falls back to defaults for the
# other two — but we pin VQVAE/Pose explicitly because the on-disk default
# paths in ``load_x2_planner.py`` may be out-of-date relative to the
# Round-2 trained checkpoints.
CLIP_FLAGS=()
[[ -n "$CLIP_KEY" ]] && CLIP_FLAGS+=(--clip-key "$CLIP_KEY")

# Headers; only written if CSV doesn't already exist.
if [[ ! -f "$CSV_OUT" ]]; then
  echo "timestamp,step,joint_rms,worst_joint,dyaw_err_deg,dx_err_m,dy_err_m,hip_err_m,verdict" > "$CSV_OUT"
fi

echo "=== Root fine-tune eval loop ==="
echo "  poll_sec     : $POLL_SEC"
echo "  start_step   : $START_STEP"
echo "  vqvae_ckpt   : $VQVAE_CKPT"
echo "  pose_ckpt    : $POSE_CKPT"
echo "  root_ckpt_dir: $ROOT_CKPT_DIR"
echo "  clip_pkl     : $CLIP_PKL"
echo "  clip_key     : ${CLIP_KEY:-<auto first forward-walk>}"
echo "  csv_out      : $CSV_OUT"
echo "  stable_sec   : $STABLE_SEC"
echo ""

seen_steps_file=$(mktemp)
trap 'rm -f "$seen_steps_file"' EXIT

# Pre-seed seen with anything already in the CSV so re-runs skip past work.
if [[ -s "$CSV_OUT" ]]; then
  tail -n +2 "$CSV_OUT" | awk -F, '{print $2}' > "$seen_steps_file"
fi

while true; do
  # Find any model-step ckpt whose step is > START_STEP and not yet evaluated,
  # whose mtime is older than STABLE_SEC (i.e. write has settled).
  candidates=()
  while IFS= read -r f; do
    base="$(basename "$f")"
    [[ "$base" =~ model-step=([0-9]+)\.ckpt$ ]] || continue
    step=$((10#${BASH_REMATCH[1]}))
    if (( step <= START_STEP )); then continue; fi
    if grep -qxF "$step" "$seen_steps_file"; then continue; fi
    # Skip if the file is still being written. mtime within STABLE_SEC =
    # probably still being torch.save()'d.
    mtime=$(stat -c %Y "$f")
    now=$(date +%s)
    if (( now - mtime < STABLE_SEC )); then continue; fi
    candidates+=("$step:$f")
  done < <(ls -1 "$ROOT_CKPT_DIR"/model-step=*.ckpt 2>/dev/null)

  if (( ${#candidates[@]} == 0 )); then
    sleep "$POLL_SEC"
    continue
  fi

  # Sort by step to evaluate in order.
  IFS=$'\n' sorted=($(printf '%s\n' "${candidates[@]}" | sort -t: -k1,1n))
  unset IFS

  for entry in "${sorted[@]}"; do
    step="${entry%%:*}"
    ckpt="${entry#*:}"
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "[$ts] step=$step  evaluating $ckpt"

    raw=$(python motionbricks/scripts/replay_pkl_through_kplanner.py \
          --vqvae-ckpt "$VQVAE_CKPT" \
          --pose-ckpt "$POSE_CKPT" \
          --root-ckpt "$ckpt" \
          --motion-lib-pkl "$CLIP_PKL" \
          "${CLIP_FLAGS[@]}" \
          2>&1 || true)

    # Tail of the raw output for debugging if parse fails.
    last_block=$(echo "$raw" | tail -40)

    # Parse the four headline numbers + verdict. ``|| true`` so a parse miss
    # logs blanks rather than killing the loop.
    joint_rms=$(echo "$raw" | grep -oP 'overall RMS\s+=\s+\K[\d.]+' | head -1)
    worst=$(echo "$raw" | grep -oP 'worst joint RMS\s+=\s+\K[\d.]+' | head -1)
    dyaw=$(echo "$raw" | grep -E 'actual dyaw' | grep -oP 'err\s+=\s+\K[+-]?[0-9]+\.[0-9]+' | head -1)
    dx=$(echo "$raw" | grep -E 'actual dx_m' | grep -oP 'err\s+=\s+\K[+-]?[0-9]+\.[0-9]+' | head -1)
    dy=$(echo "$raw" | grep -E 'actual dy_m' | grep -oP 'err\s+=\s+\K[+-]?[0-9]+\.[0-9]+' | head -1)
    hip=$(echo "$raw" | grep -E 'actual hip_z' | grep -oP 'err\s+=\s+\K[+-]?[0-9]+\.[0-9]+' | head -1)
    verdict=$(echo "$raw" | grep -E '^\s+(PASS|PARTIAL|FAIL)\b' | head -1 | awk '{print $1}')

    if [[ -z "$joint_rms" ]]; then
      echo "[$ts] step=$step  WARN parse miss; tail of replay output:"
      echo "$last_block" | sed 's/^/    /'
    fi

    printf '%s,%d,%s,%s,%s,%s,%s,%s,%s\n' \
      "$ts" "$step" \
      "${joint_rms:-}" "${worst:-}" "${dyaw:-}" "${dx:-}" "${dy:-}" "${hip:-}" \
      "${verdict:-}" >> "$CSV_OUT"

    echo "[$ts] step=$step  joint_rms=${joint_rms:-?}  dy_err=${dy:-?}  dyaw_err=${dyaw:-?}deg  dx_err=${dx:-?}  verdict=${verdict:-?}"
    echo "$step" >> "$seen_steps_file"
  done
done
