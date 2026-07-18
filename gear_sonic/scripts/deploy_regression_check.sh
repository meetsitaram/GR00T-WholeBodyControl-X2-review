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
#   .onnx (DEFAULT, and what the robot actually runs) -> eval_x2_mujoco_onnx.py takes
#           ONE motion PKL, so we loop the 7 clips; CLOSE each window to advance.
#   .pt   -> debugging convenience only (one window, N to step). NOT the shipped
#           artifact -- never certify a deploy from a .pt run.
#
# Viewer: SPACE=pause, N=next clip (.pt mode), R=reset, ,/. = speed.
# A deploy PASSES only if every clip is visually clean AND the PC2 gate matched.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

PC2_IP=""; SKIP_PC2=0; SONIC=""; PLANNER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pc2)    PC2_IP="$2"; shift 2 ;;
    --no-pc2) SKIP_PC2=1; shift ;;
    --planner) PLANNER="$2"; shift 2 ;;
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

# THE ROBOT RUNS ONNX -- so the regression tests ONNX by default. Testing a .pt
# would validate an artifact that never ships and could mask an export defect.
DEFAULT_SONIC="$HOME/x2_cloud_checkpoints/g1teleop_overnight/sonic/snapshots/exported/ft_2082_g1.onnx"
SONIC="${SONIC:-$DEFAULT_SONIC}"
[[ -f "$SONIC" ]] || { echo "model not found: $SONIC"; exit 1; }

PY=/home/stickbot/miniconda3/envs/env_isaaclab/bin/python
SUITE=gear_sonic/data/motions/deploy_regression_suite.pkl
# data/motions is gitignored -> regenerate the curated suite on a fresh clone.
[[ -f "$SUITE" ]] || "$PY" gear_sonic/scripts/build_deploy_regression_suite.py
# Planner ONNX under test -- a deploy changes BOTH models, so one run validates both.
PLANNER="${PLANNER:-$HOME/x2_cloud_checkpoints/planner_onnx_ft/x2_planner_template.onnx}"
LOCAL_PLANNER_DIR="$(dirname "$PLANNER")"

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

NCLIPS=$("$PY" - "$SUITE" <<'PYEOF'
import joblib, sys; print(len(joblib.load(sys.argv[1])))
PYEOF
)
# --- PART A (automated): drive the PLANNER ONNX under test, then play its own
# --- output through the sonic ONNX under test => validates the deployed pipeline.
GEND=$(mktemp -d); trap 'rm -rf "$GEND"' EXIT
if [[ -f "$PLANNER" ]]; then
  echo "  generating kplanner clips from $(basename "$PLANNER") ..."
  "$PY" gear_sonic/scripts/gen_kplanner_clip.py --planner-onnx "$PLANNER" \
      --out "$GEND/kp_walk.pkl" --seconds 8 --vel-z 0.5 --name kplanner__walk_0.5 2>/dev/null | tail -1
  "$PY" gear_sonic/scripts/gen_kplanner_clip.py --planner-onnx "$PLANNER" \
      --out "$GEND/kp_turn.pkl" --seconds 8 --vel-z 0.4 --yaw-rate 0.4 --name kplanner__walk_turn 2>/dev/null | tail -1
  "$PY" gear_sonic/scripts/merge_regression_clips.py "$SUITE" "$GEND/kp_walk.pkl" "$GEND/kp_turn.pkl" "$GEND/combined.pkl" \
      && SUITE="$GEND/combined.pkl"
else
  echo "  WARNING: planner ONNX not found ($PLANNER) -- kplanner stage SKIPPED"
fi

START_TS=$(date +%Y-%m-%dT%H:%M:%S)
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
      i=$((i+1))
      # Big, unmissable banner + an explicit gate: the viewer window carries no clip
      # name, so without this you cannot tell which clip you are judging.
      echo
      echo "==============================================================="
      printf '  CLIP [%d/%d]  %s\n' "$i" "${#KEYS[@]}" "$k"
      echo "==============================================================="
      if [ -t 0 ] || [ -e /dev/tty ]; then
        read -r -p "  press ENTER to play this clip (or Ctrl-C to stop) ..." _ < /dev/tty || true
      fi
      "$PY" - "$SUITE" "$k" "$TMPD/clip.pkl" <<'PYEOF'
import joblib, sys
d = joblib.load(sys.argv[1]); joblib.dump({sys.argv[2]: d[sys.argv[2]]}, sys.argv[3])
PYEOF
      .venv/bin/python gear_sonic/scripts/eval_x2_mujoco_onnx.py \
        --onnx "$SONIC" --motion "$TMPD/clip.pkl" || true
    done
    ;;
  *)
    # NOTE: not exec'd -- we must survive the viewer to print/log the run stats.
    .venv/bin/python gear_sonic/scripts/eval_x2_mujoco.py \
      --checkpoint "$SONIC" --wrist-ref --motions "$SUITE" || true
    ;;
esac

# ---------------------------------------------------------------------------
# Run stats: printed AND appended to a log, so every deploy decision has a record.
# ---------------------------------------------------------------------------
END_TS=$(date +%Y-%m-%dT%H:%M:%S)
LOGDIR=logs/deploy_regression; mkdir -p "$LOGDIR"
LOGF="$LOGDIR/$(date +%Y%m%d_%H%M%S)_$(basename "$SONIC").log"
SONIC_MD5_SHORT="${LOCAL_SONIC_MD5:-n/a}"; SONIC_MD5_SHORT="${SONIC_MD5_SHORT:0:12}"
{
  echo "===================== DEPLOY REGRESSION RUN STATS ====================="
  echo "  started / ended : $START_TS  ->  $END_TS"
  echo "  model under test: $SONIC"
  echo "  model onnx form : ${LOCAL_SONIC_ONNX:-n/a}"
  echo "  model md5       : $SONIC_MD5_SHORT"
  echo "  clips shown     : $NCLIPS  (2 walk/turn | 2 easy | 2 medium | 1 combat)"
  echo "  planner onnx    : $PLANNER"
  echo "  planner md5     : $([[ -f "$PLANNER" ]] && md5sum "$PLANNER" | cut -c1-12 || echo n/a)"
  echo "  suite           : $SUITE"
  echo "  ---- PC2 identity gate ----"
  echo "  pc2             : ${PC2_IP:-<skipped>}"
  echo "  gate verdict    : $GATE"
  echo "  robot sonic md5 : ${REMOTE_SONIC_MD5:0:12}"
  echo "  planner graphs  : $([[ "${PLAN_OK:-0}" -eq 1 ]] && echo MATCH || echo MISMATCH/unchecked)"
  echo "  handoff fix     : $([[ "${HF_OK:-0}" -eq 1 ]] && echo PRESENT || echo ABSENT/unchecked)"
  echo "  ---- verdict (fill in) ----"
  echo "  visual result   : GO / NO-GO   <- record per docs/experiments/deploy_visual_regression_checklist.md"
  echo "  notes           :"
  echo "======================================================================"
} | tee "$LOGF"
echo "logged -> $LOGF"
if [[ "$GATE" != "MATCHED" ]]; then
  echo "REMINDER: gate was $GATE -- this run does NOT certify the robot's current build."
fi
