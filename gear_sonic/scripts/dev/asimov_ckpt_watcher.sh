#!/bin/bash
# Overnight checkpoint watcher for the Asimov bring-up run.
# Every CHECK_INTERVAL: if a new model_step_*.pt appeared in RUN_DIR, run
# im_eval on the held-out eval pkl and append a summary line to STATUS_FILE.
# Start with:  setsid nohup bash gear_sonic/scripts/dev/asimov_ckpt_watcher.sh <RUN_DIR> &
set -u
RUN_DIR="$1"
REPO=~/Projects/GR00T-WholeBodyControl
PY=~/miniconda3/envs/env_isaaclab/bin/python
EVAL_PKL="gear_sonic/data/motions/asimov_eval8.pkl"
STATUS_FILE="$RUN_DIR/asimov_watch_status.log"
CHECK_INTERVAL=900   # 15 min
LAST=""

cd "$REPO"
echo "$(date '+%F %T') watcher started on $RUN_DIR" >> "$STATUS_FILE"
while true; do
  CKPT=$(ls -t "$RUN_DIR"/model_step_*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ] && [ "$CKPT" != "$LAST" ]; then
    LAST="$CKPT"
    STEP=$(basename "$CKPT" | grep -oE '[0-9]+')
    OUT="$RUN_DIR/eval_step_${STEP}"
    echo "$(date '+%F %T') evaluating $CKPT" >> "$STATUS_FILE"
    timeout 1200 $PY gear_sonic/eval_agent_trl.py \
        +exp=manager/universal_token/all_modes/sonic_asimov_loco \
        +checkpoint="$CKPT" ++headless=True ++num_envs=4 \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file="$EVAL_PKL" \
        ++experiment_dir="$OUT" \
        > "$OUT.log" 2>&1
    RC=$?
    METRICS=$(find "$OUT" -name metrics_eval.json 2>/dev/null | head -1)
    if [ -n "$METRICS" ]; then
      SUMMARY=$($PY - "$METRICS" <<'EOF'
import json, sys, statistics as st
m = json.load(open(sys.argv[1]))
rows = list(m.values()) if isinstance(m, dict) else m
def col(k):
    v = [r[k] for r in rows if isinstance(r, dict) and k in r]
    return v
succ = col("success"); mp = col("mpjpe_l"); prog = col("progress")
out = []
if succ: out.append(f"success {sum(bool(s) for s in succ)}/{len(succ)}")
if prog: out.append(f"progress median {st.median(prog):.2f}")
if mp: out.append(f"mpjpe_l median {st.median(mp)*1000:.1f}mm")
print(", ".join(out) if out else "metrics keys: " + ",".join(sorted({k for r in rows if isinstance(r,dict) for k in r})[:8]))
EOF
)
      echo "$(date '+%F %T') step $STEP: $SUMMARY" >> "$STATUS_FILE"
    else
      echo "$(date '+%F %T') step $STEP: eval rc=$RC, no metrics (see $OUT.log)" >> "$STATUS_FILE"
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
