#!/bin/bash
# ============================================================================
# DEPLOY VISUAL REGRESSION SUITE  (run in SIM before shipping any new sonic
# model or kplanner code/model to the robot)
#
# Two parts -- see docs/experiments/deploy_visual_regression_checklist.md for
# the per-clip pass criteria:
#   PART A (kplanner + handoff): pad-drive slow walk + walk-with-turns through
#           the live stack. Tests the planner and the 30->50Hz resample+blend.
#   PART B (sonic tracking): this script -- sonic tracks the curated clip set
#           (2 walks/turns, 2 easy dances, 2 medium dances, 1 combat/boxing).
#
# *** PC2 IDENTITY GATE (required) ***
# The suite ALWAYS verifies that what you are regression-checking is byte-identical
# to what is actually deployed on the robot -- sonic model, planner graphs, and the
# handoff-fixed runtime. Without this you can green-light model A locally while the
# robot runs model B. Pass --pc2 <ip> (or --no-pc2 to deliberately skip, e.g. when
# validating a candidate BEFORE it is deployed -- the mismatch is then expected).
#
# Usage (run in YOUR terminal -- the MuJoCo viewer needs the host display):
#   ./gear_sonic/scripts/deploy_regression_check.sh --pc2 192.168.86.32
#   ./gear_sonic/scripts/deploy_regression_check.sh --pc2 10.0.1.41 <model.pt|_g1.onnx>
#   ./gear_sonic/scripts/deploy_regression_check.sh --no-pc2 <candidate.pt>   # pre-deploy
#
#   .pt  -> eval_x2_mujoco.py --motions : ALL 7 clips in one window, press N to step.
#   .onnx-> eval_x2_mujoco_onnx.py takes ONE motion PKL, so we loop the 7 clips;
#           CLOSE each viewer window to advance.
#
# Viewer: SPACE=pause, N=next clip (.pt mode), R=reset, ,/. = speed.
# A deploy PASSES only if every clip is visually clean AND the PC2 gate matched.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

PC2_IP=""; SKIP_PC2=0; SONIC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pc2)    PC2_IP="$2"; shift 2 ;;
    --no-pc2) SKIP_PC2=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *)        SONIC="$1"; shift ;;
  esac
done

if [[ -z "$PC2_IP" && "$SKIP_PC2" -eq 0 ]]; then
  echo "ERROR: --pc2 <ip> is required (identity gate: proves you are testing what the robot runs)."
  echo "       use --no-pc2 only when validating a candidate BEFORE deploying it."
  echo "usage: $0 --pc2 <ip> [model.pt|_g1.onnx]"
  exit 1
fi

DEFAULT_SONIC="$HOME/x2_cloud_checkpoints/g1teleop_overnight/sonic/snapshots/ft_2082.pt"  # = robot's deployed ft_2082_g1.onnx
SONIC="${SONIC:-$DEFAULT_SONIC}"
[[ -f "$SONIC" ]] || { echo "model not found: $SONIC"; exit 1; }

PY=/home/stickbot/miniconda3/envs/env_isaaclab/bin/python
SUITE=gear_sonic/data/motions/deploy_regression_suite.pkl
LOCAL_PLANNER_DIR="$HOME/x2_cloud_checkpoints/planner_onnx_ft"

# ---------------------------------------------------------------------------
# PC2 identity gate
# ---------------------------------------------------------------------------
GATE="SKIPPED"
if [[ "$SKIP_PC2" -eq 0 ]]; then
  echo "=== PC2 IDENTITY GATE ($PC2_IP) ==="
  # sonic: compare the ONNX actually deployed vs the ONNX form of the model under test
  case "$SONIC" in
    *.onnx) LOCAL_SONIC_ONNX="$SONIC" ;;
    *)      # .pt -> the exported ONNX must correspond to THIS checkpoint by name
            # (<stem>_g1.onnx). Never glob-and-guess: picking an unrelated export
            # would silently certify the wrong model.
            _stem=$(basename "$SONIC" .pt)
            LOCAL_SONIC_ONNX="$(dirname "$SONIC")/exported/${_stem}_g1.onnx"
            if [[ ! -f "$LOCAL_SONIC_ONNX" ]]; then
              # try the checkpoint's own dir too (some runs export beside the .pt)
              _alt="$(dirname "$SONIC")/${_stem}_g1.onnx"
              [[ -f "$_alt" ]] && LOCAL_SONIC_ONNX="$_alt" || LOCAL_SONIC_ONNX=""
            fi ;;
  esac
  REMOTE_SONIC_MD5=$(ssh -o ConnectTimeout=8 "run@$PC2_IP" \
      'md5sum /home/run/getsolo/policies/agibot_x2_sonic.onnx 2>/dev/null | cut -d" " -f1' || echo "")
  if [[ -n "$LOCAL_SONIC_ONNX" && -f "$LOCAL_SONIC_ONNX" ]]; then
    LOCAL_SONIC_MD5=$(md5sum "$LOCAL_SONIC_ONNX" | cut -d' ' -f1)
    if [[ "$LOCAL_SONIC_MD5" == "$REMOTE_SONIC_MD5" ]]; then
      echo "  sonic     : MATCH   ($(basename "$LOCAL_SONIC_ONNX"))"; SONIC_OK=1
    else
      echo "  sonic     : *** MISMATCH ***  local=${LOCAL_SONIC_MD5:0:12} robot=${REMOTE_SONIC_MD5:0:12}"
      echo "              ($(basename "$LOCAL_SONIC_ONNX") is NOT what the robot runs)"; SONIC_OK=0
    fi
  else
    echo "  sonic     : UNKNOWN (no exported *_g1.onnx beside $(basename "$SONIC") to compare)"; SONIC_OK=0
  fi
  # planner graphs
  PLAN_OK=1
  for g in x2_planner_template x2_planner_velocity; do
    R=$(ssh -o ConnectTimeout=8 "run@$PC2_IP" \
        "md5sum /home/run/getsolo/planner_stack/models/planner_onnx/$g.onnx 2>/dev/null | cut -d' ' -f1" || echo "")
    L=$([[ -f "$LOCAL_PLANNER_DIR/$g.onnx" ]] && md5sum "$LOCAL_PLANNER_DIR/$g.onnx" | cut -d' ' -f1 || echo "")
    if [[ -n "$L" && "$L" == "$R" ]]; then echo "  $g: MATCH"
    else echo "  $g: MISMATCH  local=${L:0:12} robot=${R:0:12}"; PLAN_OK=0; fi
  done
  # handoff-fixed runtime on the robot
  HF=$(ssh -o ConnectTimeout=8 "run@$PC2_IP" \
      'grep -c "get_next_frame_resampled" /home/run/getsolo/pc2_kplanner_onnx.py 2>/dev/null' || echo 0)
  if [[ "${HF:-0}" -gt 0 ]]; then echo "  handoff fix: PRESENT on robot runtime"; HF_OK=1
  else echo "  handoff fix: *** ABSENT *** (robot runtime lacks 30->50Hz resample!)"; HF_OK=0; fi

  if [[ "${SONIC_OK:-0}" -eq 1 && "$PLAN_OK" -eq 1 && "${HF_OK:-0}" -eq 1 ]]; then
    GATE="MATCHED"; echo "  => GATE MATCHED: you are testing exactly what the robot runs."
  else
    GATE="MISMATCHED"
    echo "  => GATE MISMATCHED. Results do NOT certify the robot's current build."
    echo "     (expected if you are validating a candidate pre-deploy; otherwise sync first.)"
  fi
  echo
fi

echo "=== DEPLOY REGRESSION (Part B: sonic tracking) ==="
echo "  model: $SONIC"
echo "  pc2 gate: $GATE"
echo "  clips: 2 walk/turn | 2 easy dance | 2 medium dance | 1 combat/boxing"
echo "  judge each against docs/experiments/deploy_visual_regression_checklist.md"
echo

case "$SONIC" in
  *.onnx)
    echo "  ONNX mode: one clip per window -- CLOSE each window to advance."; echo
    TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT
    mapfile -t KEYS < <("$PY" - "$SUITE" <<'PYEOF'
import joblib, sys
for k in joblib.load(sys.argv[1]): print(k)
PYEOF
)
    i=0
    for k in "${KEYS[@]}"; do
      i=$((i+1)); echo "--- [$i/${#KEYS[@]}] $k ---"
      "$PY" - "$SUITE" "$k" "$TMPD/clip.pkl" <<'PYEOF'
import joblib, sys
d = joblib.load(sys.argv[1]); joblib.dump({sys.argv[2]: d[sys.argv[2]]}, sys.argv[3])
PYEOF
      .venv/bin/python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
        --onnx "$SONIC" --motion "$TMPD/clip.pkl" || true
    done
    echo "=== suite complete ($i clips), pc2 gate: $GATE ==="
    ;;
  *)
    exec .venv/bin/python gear_sonic/scripts/eval_x2_mujoco.py \
      --checkpoint "$SONIC" --wrist-ref --motions "$SUITE"
    ;;
esac
