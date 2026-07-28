#!/bin/bash
# Standard SIM launch, gated against the robot: ONNX planner + MuJoCo sim +
# local gamepad. SIM-ONLY -- no --pc2-host, so the pose wire binds 127.0.0.1
# and the robot cannot subscribe.
#
#   ./gear_sonic/scripts/sim_onnx_planner.sh <PC2_IP>
#
# The PC2 IP is the ONLY argument, and it is required. Before launching, this
# md5-compares every model you are about to play with against the ones actually
# deployed on the robot, and reports each file's last-modified time on the robot
# so a stale artifact is visible. It then auto-selects the local sonic ONNX whose
# md5 matches the robot's, so "what you test" IS "what the robot runs" by
# construction rather than by hope.
#
# Why the gate exists: it once caught the robot running ft_2082_g1.onnx while
# every sim launcher used walkft_3065_g1.onnx -- i.e. sim had been validating a
# different policy than the robot. A model can pass every numeric and visual
# check locally and still not be the model on the robot.
#
# ONNX-ONLY, to match the robot: PC2 has no .pt files at all. The .pt the stack
# would otherwise auto-resolve is the *recorder's* VLA tokenizer, not the deploy
# policy, and the robot never runs the recorder -- so opting out is the
# robot-faithful configuration.
#
# Overrides (env, for pre-deploy candidates). Either one intentionally trips the
# gate, and a tripped gate now ABORTS unless ALLOW_MISMATCH=1 is also set --
# testing a candidate must be deliberate, never something you drift into.
#   SONIC_MODEL=<path_g1.onnx>   candidate sonic policy       (alias: MODEL=)
#   PLANNER_MODEL=<dir|file>     candidate planner graphs (dir holding
#                                x2_planner_{template,velocity}.onnx, or one of them)
#   ALLOW_MISMATCH=1             acknowledge the gate and run anyway
#   PLANNER=velocity             use the velocity graph (default: template)
#   SONIC_CKPT=<path.pt>         ONLY when deliberately recording a VLA dataset
#
# Both models are gated independently, so you can vary one and hold the other at
# the robot's version -- which is what makes a candidate result attributable:
#   ALLOW_MISMATCH=1 SONIC_MODEL=<cand>   ./sim_onnx_planner.sh <ip>   # sonic only
#   ALLOW_MISMATCH=1 PLANNER_MODEL=<dir>  ./sim_onnx_planner.sh <ip>   # planner only
#
# Controls: L2 = deadman drive (locked 0.3), L1+Y/A = cycle dances, L1+B = stop.
# Needs a pad visible to pygame BEFORE launch (--pad-only tears the stack down
# otherwise) and the env_isaaclab conda env.
#
# --vr: dual-source input (pad + Quest 3). Spawns quest3_manager_x2 alongside
# the pad bridge (stack --pad-and-vr); the kplanner SUB binds :5563 and both
# sources PUB-connect. Open https://<laptop-ip>:8443 in the Quest 3 browser,
# A+B+X+Y to engage; the pad keeps working whenever its deadman is held.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Accept a bare IP or the --pc2-host/--pc2 flag form: the flag form is what the
# demo command sheet documents, and silently treating "--pc2-host" as a hostname
# produces a baffling "cannot reach run@--pc2-host".
PC2_IP=""
VR_MODE=0
ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pc2-host|--pc2) PC2_IP="${2:-}"; shift 2 ;;
    --vr)             VR_MODE=1; shift ;;
    --*)              echo "unknown option: $1" >&2; shift ;;
    *)                PC2_IP="$1"; shift ;;
  esac
done
if [[ -z "$PC2_IP" ]]; then
  echo "usage: $0 <PC2_IP>            # e.g. $0 192.168.86.32" >&2
  echo "       $0 --pc2-host <PC2_IP> [--vr]" >&2
  echo "  the robot's IP is required: it is what makes this run certifiable." >&2
  exit 2
fi

CKPT_ROOT="${CKPT_ROOT:-$HOME/x2_cloud_checkpoints}"
GRAPH="${PLANNER:-template}"   # template | velocity
DANCES="$CKPT_ROOT/dances_x2m2"
LOCAL_PLANNER_DIR="$CKPT_ROOT/planner_onnx"
R_SONIC=/home/run/gear-sonic/policies/agibot_x2_sonic.onnx
R_PLAN=/home/run/gear-sonic/planner_stack/models/planner_onnx
R_RUNTIME=/home/run/gear-sonic/pc2_kplanner_onnx.py

# ---------------------------------------------------------------------------
# Fetch md5 + mtime for every robot-side artifact in ONE ssh round trip.
# ---------------------------------------------------------------------------
echo "=== PC2 IDENTITY GATE ($PC2_IP) ==="
REMOTE=$(ssh -o ConnectTimeout=8 "run@$PC2_IP" "
  for f in $R_SONIC $R_PLAN/x2_planner_template.onnx $R_PLAN/x2_planner_velocity.onnx; do
    if [ -f \"\$f\" ]; then
      echo \"\$f \$(md5sum \"\$f\" | cut -d' ' -f1) \$(date -r \"\$f\" '+%Y-%m-%d_%H:%M:%S')\"
    else
      echo \"\$f MISSING -\"
    fi
  done
  echo \"handoff \$(grep -c get_next_frame_resampled $R_RUNTIME 2>/dev/null || echo 0) \$(date -r $R_RUNTIME '+%Y-%m-%d_%H:%M:%S' 2>/dev/null || echo -)\"
" 2>/dev/null) && REMOTE_RC=0 || REMOTE_RC=$?
if [[ $REMOTE_RC -ne 0 ]]; then
  # PC2 unreachable. A sim-only validation run must NOT require the robot to be
  # online -- the whole point is to test a candidate BEFORE deploying. So:
  #   ALLOW_MISMATCH=1 -> warn and proceed with NO identity comparison.
  #   otherwise        -> abort (a certification run must confirm the robot).
  if [[ "${ALLOW_MISMATCH:-0}" == "1" ]]; then
    echo "  WARN: cannot reach run@$PC2_IP -- no robot identity comparison possible."
    echo "        Proceeding validates the CANDIDATE with NO check against the robot."
    # A manual keypress, not just the env var: skipping the robot check is a
    # deliberate act each run, never a silent default. Read from the terminal
    # (/dev/tty) so a piped stdin can't auto-answer; no tty -> treat as 'no'.
    printf "  Proceed OFFLINE without the robot check? [y/N] "
    read -r _off_ans </dev/tty 2>/dev/null || _off_ans=""
    if [[ "$_off_ans" != "y" && "$_off_ans" != "Y" ]]; then
      echo "  aborted -- no offline confirmation." >&2
      exit 3
    fi
    echo "  confirmed -- running OFFLINE, candidate results only."
    REMOTE=""
    GATE_OFFLINE=1
  else
    echo "  ERROR: cannot reach run@$PC2_IP" >&2
    echo "         (set ALLOW_MISMATCH=1 to validate a candidate without the robot)" >&2
    exit 3
  fi
fi

r_field() { echo "$REMOTE" | awk -v k="$1" '$1 ~ k {print $2; exit}'; }
r_time()  { echo "$REMOTE" | awk -v k="$1" '$1 ~ k {print $3; exit}'; }

REMOTE_SONIC_MD5=$(r_field "agibot_x2_sonic.onnx"); REMOTE_SONIC_T=$(r_time "agibot_x2_sonic.onnx")
HF=$(r_field "^handoff$");                          HF_T=$(r_time "^handoff$")

# --- sonic: prefer the local ONNX that MATCHES the robot (unless overridden) ---
# SONIC_MODEL is the preferred name; MODEL kept as a backward-compatible alias.
SONIC_ONNX="${SONIC_MODEL:-${MODEL:-}}"
AUTO_NOTE=""
if [[ -z "$SONIC_ONNX" ]]; then
  # Find the local export whose bytes equal the robot's. This is the whole point:
  # select by identity, never by a hardcoded path that can silently drift.
  while IFS= read -r f; do
    if [[ "$(md5sum "$f" | cut -d' ' -f1)" == "$REMOTE_SONIC_MD5" ]]; then
      SONIC_ONNX="$f"; AUTO_NOTE=" (auto-selected: matches robot)"; break
    fi
  done < <(find "$CKPT_ROOT" -name '*_g1.onnx' -type f 2>/dev/null)
fi

SONIC_OK=0
if [[ -n "$SONIC_ONNX" && -f "$SONIC_ONNX" ]]; then
  LOCAL_SONIC_MD5=$(md5sum "$SONIC_ONNX" | cut -d' ' -f1)
  if [[ "$LOCAL_SONIC_MD5" == "$REMOTE_SONIC_MD5" ]]; then
    echo "  sonic     : MATCH    $(basename "$SONIC_ONNX")  [robot mtime $REMOTE_SONIC_T]"
    SONIC_OK=1
  else
    echo "  sonic     : *** MISMATCH ***  local=${LOCAL_SONIC_MD5:0:12} robot=${REMOTE_SONIC_MD5:0:12}"
    echo "              testing $(basename "$SONIC_ONNX") -- NOT what the robot runs"
    echo "              [robot artifact mtime $REMOTE_SONIC_T]"
  fi
else
  echo "  sonic     : *** NO LOCAL MATCH *** for robot md5 ${REMOTE_SONIC_MD5:0:12}"
  echo "              [robot mtime $REMOTE_SONIC_T] -- robot runs something not present locally."
  echo "              Pass MODEL=<path_g1.onnx> to choose explicitly."
  exit 4
fi

# --- planner graphs: select the local dir by IDENTITY, not by a fixed path ---
# A hardcoded dir drifts the moment new graphs are shipped to the robot. This
# once left every sim run driving planner_onnx/ while the robot ran the FT trio
# in planner_onnx_ft/ -- so sim was validating a different planner than deploy.
# Prefer whichever local dir actually matches the robot.
R_TMPL_MD5=$(r_field "x2_planner_template")
PLANNER_NOTE=""
if [[ -n "${PLANNER_MODEL:-}" ]]; then
  # Explicit candidate planner. Accept either the directory holding the graphs
  # or one of the graph files (we need the dir, since both graphs are checked).
  if [[ -d "$PLANNER_MODEL" ]]; then
    LOCAL_PLANNER_DIR="$PLANNER_MODEL"
  elif [[ -f "$PLANNER_MODEL" ]]; then
    LOCAL_PLANNER_DIR="$(dirname "$PLANNER_MODEL")"
  else
    echo "  ERROR: PLANNER_MODEL=$PLANNER_MODEL is neither a file nor a directory" >&2
    exit 7
  fi
  PLANNER_NOTE=" (explicit PLANNER_MODEL)"
  echo "  planner dir: $(basename "$LOCAL_PLANNER_DIR")/ -- explicit override, not auto-selected"
else
  for d in "$CKPT_ROOT"/planner_onnx*; do
    [[ -d "$d" ]] || continue
    c="$d/x2_planner_template.onnx"; [[ -f "$c" ]] || continue
    if [[ "$(md5sum "$c" | cut -d' ' -f1)" == "$R_TMPL_MD5" ]]; then
      if [[ "$d" != "$LOCAL_PLANNER_DIR" ]]; then
        echo "  planner dir: using $(basename "$d")/ (matches robot; $(basename "$LOCAL_PLANNER_DIR")/ is stale)"
      fi
      LOCAL_PLANNER_DIR="$d"; break
    fi
  done
fi

PLAN_OK=1
declare -A PLAN_T=()
for g in x2_planner_template x2_planner_velocity; do
  R=$(r_field "$g"); T=$(r_time "$g"); PLAN_T[$g]="$T"
  L=$([[ -f "$LOCAL_PLANNER_DIR/$g.onnx" ]] && md5sum "$LOCAL_PLANNER_DIR/$g.onnx" | cut -d' ' -f1 || echo "")
  if [[ -n "$L" && "$L" == "$R" ]]; then
    echo "  $g: MATCH    [robot mtime $T]"
  else
    echo "  $g: MISMATCH local=${L:0:12} robot=${R:0:12}  [robot mtime $T]"; PLAN_OK=0
  fi
done

# --- handoff fix present in the robot's runtime? ---
if [[ "${HF:-0}" -gt 0 ]]; then
  echo "  handoff fix: PRESENT ($HF markers)  [robot mtime $HF_T]"; HF_OK=1
else
  echo "  handoff fix: *** ABSENT *** -- robot runtime lacks the 30->50Hz resample!"
  echo "               [robot mtime $HF_T]"; HF_OK=0
fi

if [[ "$SONIC_OK" -eq 1 && "$PLAN_OK" -eq 1 && "${HF_OK:-0}" -eq 1 ]]; then
  GATE="MATCHED"; echo "  => GATE MATCHED: you are testing exactly what the robot runs."
else
  GATE="MISMATCHED"
  echo "  => GATE MISMATCHED. This run does NOT certify the robot's current build."
  # FAIL CLOSED. A gate you can drive past is not a gate: the whole failure mode
  # it exists to prevent is someone eyeballing a clean sim run and concluding the
  # robot is good, when sim was driving different artifacts. Testing a candidate
  # pre-deploy is legitimate, but it must be a deliberate act, never the default.
  if [[ "${ALLOW_MISMATCH:-0}" != "1" ]]; then
    echo
    echo "  ABORTING: refusing to launch against artifacts the robot does not run."
    echo "  Whatever you saw in the viewer would NOT be evidence about the robot."
    echo
    echo "  If this is deliberate (validating a candidate before deploying it):"
    echo "      ALLOW_MISMATCH=1 $0 ${ORIG_ARGS[*]}"
    echo "  Otherwise sync the mismatched artifact(s) to the robot and re-run."
    exit 6
  fi
  echo "  ALLOW_MISMATCH=1 set -- proceeding as an explicit CANDIDATE test."
  echo "  Results describe the candidate ONLY. Do not record them as a deploy GO."
fi
echo

PLANNER_ONNX="$LOCAL_PLANNER_DIR/x2_planner_${GRAPH}.onnx"

# ---------------------------------------------------------------------------
# Auto-cleanup of a previous LOCAL sim, so stale ports never block a launch.
#
# *** THIS MUST NEVER BE ABLE TO TOUCH THE ROBOT. ***
# Killing a live sonic drops the robot -- it cannot hold a frame, so losing its
# control process means it collapses. Safety rests on three facts, asserted here
# rather than assumed:
#   1. run_x2_quest3_planner_stack.sh --cleanup-only resolves victims ONLY via
#      lsof/fuser on LOCAL tcp ports and kills LOCAL pids. It contains no ssh or
#      scp in any kill path (verified).
#   2. This launcher is SIM-ONLY: it never passes --pc2-host, so the pose wire
#      binds 127.0.0.1 and the robot cannot even subscribe.
#   3. The PC2 IP above is used for the gate ONLY, over read-only ssh
#      (md5sum / date / grep). Nothing in this script writes to or signals PC2.
# The one way local cleanup could escape this machine is a remote Docker daemon,
# so refuse to run if DOCKER_HOST points off-box.
# ---------------------------------------------------------------------------
if [[ -n "${DOCKER_HOST:-}" && "$DOCKER_HOST" != unix://* ]]; then
  echo "REFUSING to auto-clean: DOCKER_HOST=$DOCKER_HOST is not a local socket." >&2
  echo "Cleanup stops containers; against a remote daemon that could reach the robot." >&2
  exit 5
fi
echo "=== local sim cleanup (frees stale ports; robot is NOT touched) ==="
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --cleanup-only 2>&1 | sed 's/^/  /' || true
echo

# ---------------------------------------------------------------------------
# End-of-run summary. Printed on ANY exit so the models under test are always
# recorded next to the verdict -- a visual result with no model provenance is
# not evidence of anything.
# ---------------------------------------------------------------------------
summary() {
  echo
  echo "==================== SIM RUN SUMMARY ===================="
  echo "  pc2                : $PC2_IP"
  echo "  gate verdict       : $GATE"
  echo "  --- models used ----------------------------------------"
  printf '  sonic              : %s\n' "$(basename "$SONIC_ONNX")$AUTO_NOTE"
  printf '                       md5 %s  robot mtime %s\n' "${LOCAL_SONIC_MD5:0:12}" "$REMOTE_SONIC_T"
  printf '  planner graph      : %s (%s)%s\n' "$(basename "$PLANNER_ONNX")" "$GRAPH" "$PLANNER_NOTE"
  printf '                       dir %s\n' "$LOCAL_PLANNER_DIR"
  printf '                       md5 %s  robot mtime %s\n' \
      "$([[ -f "$PLANNER_ONNX" ]] && md5sum "$PLANNER_ONNX" | cut -c1-12 || echo n/a)" \
      "${PLAN_T[x2_planner_$GRAPH]:-n/a}"
  printf '  planner runtime    : handoff fix %s  robot mtime %s\n' \
      "$([[ "${HF:-0}" -gt 0 ]] && echo PRESENT || echo ABSENT)" "$HF_T"
  printf '  dances             : %s\n' "$DANCES"
  printf '  warmup qpos        : kplanner_idle_anchor_g1teleop_v3.pkl\n'
  printf '  sonic .pt          : none (ONNX-only, matches robot)\n'
  printf '  input mode         : %s\n' "$([[ "$VR_MODE" -eq 1 ]] && echo 'pad + Quest 3 VR (dual-source)' || echo 'pad only')"
  echo "  --------------------------------------------------------"
  if [[ "$GATE" != "MATCHED" ]]; then
    echo "  REMINDER: gate was $GATE -- results do NOT certify the robot's build."
  fi
  echo "========================================================="
}
trap summary EXIT

export KPLANNER_ONNX="$PLANNER_ONNX"
export KPLANNER_DANCES_DIR="$DANCES"
# Respect a caller-provided speed (was a hard export that silently clobbered
# any KPLANNER_FIXED_FWD_MPS the operator set -- "no matter what fwd mps I
# set, it walks the same", 2026-07-21).
export KPLANNER_FIXED_FWD_MPS="${KPLANNER_FIXED_FWD_MPS:-0.3}"
export PAD_LOCK_SPEED=1
export PAD_DEADMAN=left
export PAD_CLIP_PKL=gear_sonic/data/motions/x2_dances_easy.pkl
# Same four banks as the robot ritual -- do NOT glob the dances dir: that put
# gestures, turn clips and combat all on one button and made sim behaviour
# diverge from the robot.
export PAD_CLIP_KEYS="dance_party_hips_003__A467,dance_party_hips_003__A464,dance_party_hips_003__A465"
export PAD_CLIP_KEYS_X="shadow_boxing_R_003__A359_M,shadow_boxing_R_003__A359"
export PAD_CLIP_KEYS_M="egipt_dance_R_001__A438,dance_hiphop_stick_n_roll_dancehall_R_loop_003__A324,dance_distraction_dance_001__A466"
export PAD_CLIP_KEYS_G="right_wave_001,both_heart_002,right_kiss_001,right_five_001,right_shake_001,turn_wave_right_001,turn_wave_left_001"
export PAD_CLIP_KEYS_TURN="locowalk__idle_turn_270_002__A056_M,locowalk__idle_turn_270_002__A056,relaxed_walk_forward,walk_circle_001"

MODE_FLAG=()
[[ "$GRAPH" == "template" ]] && MODE_FLAG=(--kplanner-planner-mode "${MODE:-slow_walk}")
SONIC_CKPT_FLAG=(--no-sonic-checkpoint)
[[ -n "${SONIC_CKPT:-}" ]] && SONIC_CKPT_FLAG=(--sonic-checkpoint "$SONIC_CKPT")

# --vr swaps --pad-only for --pad-and-vr: quest3_manager_x2 + pad bridge both
# feed the kplanner's bound planner_cmd SUB (dual-source; see stack script).
INPUT_FLAG=(--pad-only)
[[ "$VR_MODE" -eq 1 ]] && INPUT_FLAG=(--pad-and-vr)

# NOT exec'd: we must survive the stack to print the summary trap.
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
  --duration 0 "${INPUT_FLAG[@]}" "${MODE_FLAG[@]}" "${SONIC_CKPT_FLAG[@]}" \
  --kplanner-warmup-qpos gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl \
  --model "$SONIC_ONNX" \
  --kplanner-python "$HOME/miniconda3/envs/env_isaaclab/bin/python" || true
