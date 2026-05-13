#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Wrapper around external_dependencies/Isaac-GR00T's launch_finetune.py.
#
# Why a wrapper?
#   1. Activate ``env_isaaclab`` correctly (deactivate any active venv first;
#      we kept hitting "transformers not installed" in env that DID have it
#      because a stale .venv was masking the conda env).
#   2. Pin the working batch math for the local 32 GB RTX 5090. The trainer
#      maps ``--global-batch-size G --num-gpus N`` to per_device = G/N, so to
#      get per_device=2 (the maximum that fits N1.7-3B + LoRA-only on this
#      card) you MUST set ``--global-batch-size 2``, NOT ``--global-batch-size
#      <effective>``. Effective batch is per_device * num_gpus * grad_accum.
#   3. Fail fast on the "tune_llm freezing didn't apply" silent corruption.
#      The base ``nvidia/GR00T-N1.7-3B`` HuggingFace config ships with
#      ``tune_llm=True`` baked in. Even when the user explicitly passes
#      ``--no-tune-llm``, the override has been observed not to take effect
#      in some configurations (likely a transformers >=5.0 / accelerate
#      interaction; under investigation). When the override fails, ALL
#      ~3.45 B parameters become trainable, which crashes Adam at the
#      first optimizer step with a 41 GB memory request -- 90 s into the
#      run, after the user has already moved on to something else. This
#      wrapper greps the trainer's "Trainable parameters: N (X.YZ%)" line
#      and aborts the run if X is unreasonably high (default >25%, which
#      catches the 100% bug while leaving headroom for legitimate larger
#      finetunes that include vlln, mask_token, etc.).
#
# Usage:
#   ./gear_sonic/scripts/train_groot_vla.sh <dataset_path> <output_dir>
#   ./gear_sonic/scripts/train_groot_vla.sh /tmp/x2_piano_gentle_v1 \
#       /tmp/x2_n17_finetune_v2_gentle
#
# Env-var knobs (override defaults):
#   MAX_STEPS=3000               training steps
#   MICROBATCH=2                 per-device microbatch (== --global-batch-size
#                                with --num-gpus 1)
#   GRAD_ACCUM=2                 gradient accumulation steps
#                                (effective batch = MICROBATCH*GRAD_ACCUM)
#   LR=0.0001                    learning rate
#   SAVE_STEPS=1000              checkpoint cadence
#   BASE_MODEL=nvidia/GR00T-N1.7-3B   HF model id or local path
#   MODALITY=gear_sonic/data/x2_modality_config_10dof.py   side-loader path
#   EMBODIMENT_TAG=NEW_EMBODIMENT
#   MAX_TRAINABLE_PCT=70         abort if trainable% exceeds this. Catches
#                                the 100% freezing bug while leaving room
#                                for the legitimate freeze-mode recipe
#                                (LLM+visual frozen + action head fully
#                                trainable = ~56% on GR00T-N1.7-3B). Set
#                                to 100 to disable the safety check (e.g.
#                                when you actually DO want to train the
#                                LLM end-to-end).
#   TUNE_LLM=false               --tune-llm / --no-tune-llm (default: false)
#   TUNE_VISUAL=false            --tune-visual / --no-tune-visual
#   TUNE_PROJECTOR=true          --tune-projector / --no-tune-projector
#   TUNE_DIFFUSION=true          --tune-diffusion-model / --no-tune-...
#   GRAD_CHECKPOINT=true         --gradient-checkpointing toggle
#   USE_WANDB=false              --use-wandb / --no-use-wandb
#   EXTRA_ARGS=""                extra positional args appended verbatim
#
# Outputs:
#   $OUTPUT_DIR                  trainer artefacts (checkpoint-*, processor/,
#                                experiment_cfg/, model.safetensors, ...)
#   $OUTPUT_DIR/../<basename>_run/
#       finetune.log             full trainer stdout+stderr (long, includes
#                                HF download progress bars; grep through it)
#       finetune.pid             setsid wrapper pid for cleanup / monitoring
#       trainable_pct.txt        the parsed "X.YZ%" the wrapper saw at startup
#                                (whether it passed the safety threshold)
#
# Exit codes:
#   0   trainer exited 0
#   1   trainer crashed (non-zero exit)
#   2   pre-flight failure (dataset missing, conda env missing, etc.)
#   3   trainable-% check failed (likely the freezing bug -- investigate
#       before retrying. The wrapper killed the trainer to free the GPU.)
# ---------------------------------------------------------------------------
set -euo pipefail

# ----- colours (only when stdout is a tty) ---------------------------------
if [[ -t 1 ]]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'
    BLUE=$'\033[0;34m'; CYAN=$'\033[0;36m';   NC=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""; NC=""
fi

ts() { date '+%H:%M:%S'; }

# ----- args ----------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    sed -n '/^# Usage:/,/^# Outputs:/p' "$0" | sed 's/^# \?//'
    exit 2
fi
DATASET_PATH="$1"
OUTPUT_DIR="$2"
shift 2
EXTRA_POS_ARGS=("$@")

# ----- defaults ------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${MAX_STEPS:=3000}"
: "${MICROBATCH:=2}"
: "${GRAD_ACCUM:=2}"
: "${LR:=0.0001}"
: "${WARMUP_RATIO:=0.05}"
: "${SAVE_STEPS:=1000}"
: "${SHARD_SIZE:=1024}"
: "${EPISODE_SAMPLING_RATE:=1.0}"
: "${BASE_MODEL:=nvidia/GR00T-N1.7-3B}"
: "${MODALITY:=gear_sonic/data/x2_modality_config_10dof.py}"
: "${EMBODIMENT_TAG:=NEW_EMBODIMENT}"
: "${TUNE_LLM:=false}"
: "${TUNE_VISUAL:=false}"
: "${TUNE_PROJECTOR:=true}"
: "${TUNE_DIFFUSION:=true}"
: "${GRAD_CHECKPOINT:=true}"
: "${USE_WANDB:=false}"
: "${MAX_TRAINABLE_PCT:=70}"
: "${EXTRA_ARGS:=}"

CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
CONDA_PROFILE="${CONDA_PROFILE:-/home/stickbot/miniconda3/etc/profile.d/conda.sh}"

EFFECTIVE_BATCH=$((MICROBATCH * GRAD_ACCUM))

RUN_DIR="${OUTPUT_DIR%/}_run"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/finetune.log"
PID_FILE="$RUN_DIR/finetune.pid"
PCT_FILE="$RUN_DIR/trainable_pct.txt"

# ----- pre-flight ----------------------------------------------------------
echo -e "${CYAN}[$(ts) train_groot_vla] pre-flight checks …${NC}"

if [[ ! -d "$DATASET_PATH" ]]; then
    echo -e "${RED}[FATAL] dataset path not found: $DATASET_PATH${NC}" >&2
    exit 2
fi
if [[ ! -f "$DATASET_PATH/meta/info.json" ]]; then
    echo -e "${RED}[FATAL] dataset is not a LeRobot v2.1 dir (missing meta/info.json): $DATASET_PATH${NC}" >&2
    exit 2
fi
if [[ ! -f "$REPO_ROOT/$MODALITY" ]]; then
    echo -e "${RED}[FATAL] modality config not found: $REPO_ROOT/$MODALITY${NC}" >&2
    exit 2
fi
if [[ ! -f "$REPO_ROOT/gear_sonic/scripts/launch_finetune_x2.py" ]]; then
    echo -e "${RED}[FATAL] gear_sonic/scripts/launch_finetune_x2.py not found.${NC}" >&2
    exit 2
fi
if [[ ! -f "$REPO_ROOT/external_dependencies/Isaac-GR00T/gr00t/experiment/launch_finetune.py" ]]; then
    echo -e "${RED}[FATAL] upstream Isaac-GR00T launch_finetune.py not found under external_dependencies/${NC}" >&2
    echo -e "${YELLOW}        (we don't run it directly, but our X2 launcher imports its config classes.)${NC}" >&2
    exit 2
fi
if [[ ! -f "$CONDA_PROFILE" ]]; then
    echo -e "${RED}[FATAL] conda profile not found: $CONDA_PROFILE (export CONDA_PROFILE=)${NC}" >&2
    exit 2
fi

# GPU sanity: at least 20 GB free is required for N1.7-3B + LoRA.
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ -n "$GPU_FREE_MIB" ]] && [[ "$GPU_FREE_MIB" -lt 20000 ]]; then
        echo -e "${RED}[FATAL] GPU 0 has only ${GPU_FREE_MIB} MiB free; N1.7-3B finetune needs >=20000 MiB.${NC}" >&2
        echo -e "${YELLOW}        Free up GPU memory (close any live demos, kill stale python procs) and retry.${NC}" >&2
        exit 2
    fi
    echo -e "${GREEN}[ok]${NC} GPU 0 free=${GPU_FREE_MIB} MiB"
fi

cat <<EOF
${CYAN}[train_groot_vla] resolved config:${NC}
  dataset            = $DATASET_PATH
  output_dir         = $OUTPUT_DIR
  run_dir            = $RUN_DIR
  base_model         = $BASE_MODEL
  modality           = $MODALITY
  embodiment_tag     = $EMBODIMENT_TAG
  conda_env          = $CONDA_ENV
  microbatch         = $MICROBATCH      (per-device, == --global-batch-size with --num-gpus 1)
  grad_accum         = $GRAD_ACCUM
  effective_batch    = $EFFECTIVE_BATCH
  max_steps          = $MAX_STEPS
  learning_rate      = $LR
  warmup_ratio       = $WARMUP_RATIO
  save_steps         = $SAVE_STEPS
  tune_llm           = $TUNE_LLM
  tune_visual        = $TUNE_VISUAL
  tune_projector     = $TUNE_PROJECTOR
  tune_diffusion     = $TUNE_DIFFUSION
  gradient_ckpt      = $GRAD_CHECKPOINT
  max_trainable_pct  = $MAX_TRAINABLE_PCT  (safety: abort if trainer reports trainable% above this)
  extra_args         = ${EXTRA_ARGS}
  extra_pos_args     = ${EXTRA_POS_ARGS[*]:-}
EOF

# ----- assemble launch args ------------------------------------------------
TRAIN_ARGS=(
    --base-model-path "$BASE_MODEL"
    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY"
    --num-gpus 1
    --output-dir "$OUTPUT_DIR"
    --shard-size "$SHARD_SIZE"
    --episode-sampling-rate "$EPISODE_SAMPLING_RATE"
    --max-steps "$MAX_STEPS"
    --global-batch-size "$MICROBATCH"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --learning-rate "$LR"
    --warmup-ratio "$WARMUP_RATIO"
    --save-steps "$SAVE_STEPS"
)
[[ "$GRAD_CHECKPOINT" == "true" ]] && TRAIN_ARGS+=(--gradient-checkpointing)
[[ "$USE_WANDB"       == "true" ]] && TRAIN_ARGS+=(--use-wandb) || TRAIN_ARGS+=(--no-use-wandb)
[[ "$TUNE_LLM"        == "true" ]] && TRAIN_ARGS+=(--tune-llm)        || TRAIN_ARGS+=(--no-tune-llm)
[[ "$TUNE_VISUAL"     == "true" ]] && TRAIN_ARGS+=(--tune-visual)     || TRAIN_ARGS+=(--no-tune-visual)
[[ "$TUNE_PROJECTOR"  == "true" ]] && TRAIN_ARGS+=(--tune-projector)  || TRAIN_ARGS+=(--no-tune-projector)
[[ "$TUNE_DIFFUSION"  == "true" ]] && TRAIN_ARGS+=(--tune-diffusion-model) \
                                   || TRAIN_ARGS+=(--no-tune-diffusion-model)
# shellcheck disable=SC2086
read -r -a EXTRA_FLAGS <<< "${EXTRA_ARGS}"
TRAIN_ARGS+=("${EXTRA_FLAGS[@]}")
TRAIN_ARGS+=("${EXTRA_POS_ARGS[@]}")

# ----- launch in background -----------------------------------------------
echo -e "${CYAN}[$(ts) train_groot_vla] launching trainer in background …${NC}"
echo "  log  = $LOG"
echo "  pid  = $PID_FILE"

# Drop any active virtualenv before activating conda (the .venv that this
# repo uses for non-trainer work shadows env_isaaclab when both are
# resolvable on PATH; we hit ``ModuleNotFoundError: transformers`` from
# this exact failure mode several times before we caught it).
nohup setsid bash -c '
set -euo pipefail
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate 2>/dev/null || true
    unset VIRTUAL_ENV
fi
source '"$CONDA_PROFILE"'
conda activate '"$CONDA_ENV"'
cd '"$REPO_ROOT"'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[finetune] starting at $(date -Iseconds)"
echo "[finetune] python: $(which python)  $(python -V)"
echo "[finetune] torch:  $(python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" 2>&1)"
echo "[finetune] cmd:    python gear_sonic/scripts/launch_finetune_x2.py '"${TRAIN_ARGS[*]}"'"

PYTHONPATH=external_dependencies/Isaac-GR00T:. python \
    gear_sonic/scripts/launch_finetune_x2.py \
    '"${TRAIN_ARGS[*]}"'

echo "[finetune] exit_code=$? at $(date -Iseconds)"
' > "$LOG" 2>&1 < /dev/null &

WRAPPER_PID=$!
echo "$WRAPPER_PID" > "$PID_FILE"

# We need to track the actual python pid (a child of the setsid bash) so
# we can kill it if the trainable-pct check trips. Wait briefly for it to
# spawn under PYTHONPATH=... python.
sleep 6
PY_PID="$(pgrep -P "$WRAPPER_PID" -f "launch_finetune.py" || true)"
if [[ -z "$PY_PID" ]]; then
    # python may be a grandchild via conda's bash wrapper; widen the search.
    PY_PID="$(pgrep -f "launch_finetune.py" | head -1 || true)"
fi
echo "  python_pid = ${PY_PID:-unknown}"

# ----- watch the log for the freezing-bug signal --------------------------
# The trainer prints "Trainable parameters: N (X.YZ%)" once per process,
# right after the model loads (before the first optimizer step). This is
# exactly the canary moment to abort if freezing didn't take effect: at
# this point only ~6 GB of model weights are on the GPU; killing here
# avoids the 90 s walk to the inevitable Adam-OOM at step 0.
echo -e "${CYAN}[$(ts) train_groot_vla] waiting for 'Trainable parameters' line (timeout 240 s) …${NC}"
DEADLINE=$(( $(date +%s) + 240 ))
PCT_LINE=""
while [[ -z "$PCT_LINE" ]]; do
    if (( $(date +%s) > DEADLINE )); then
        break
    fi
    if ! kill -0 "$WRAPPER_PID" 2>/dev/null; then
        # Trainer wrapper exited before printing the trainable-param line.
        # That's a fatal startup error (probably argparse rejection or
        # missing dependency). Surface the tail and bail out.
        echo -e "${RED}[FATAL] trainer wrapper exited before model load. Tail of log:${NC}" >&2
        grep -vE "HTTP Request|^$|Loading checkpoint shards" "$LOG" 2>/dev/null | tail -20 >&2 || true
        exit 1
    fi
    PCT_LINE="$(grep -E 'Trainable parameters:' "$LOG" 2>/dev/null | tail -1 || true)"
    [[ -z "$PCT_LINE" ]] && sleep 3
done

if [[ -z "$PCT_LINE" ]]; then
    echo -e "${YELLOW}[warn] no 'Trainable parameters' line within 240 s -- trainer may still be downloading the base model on first run.${NC}"
    echo -e "${YELLOW}        Continuing anyway; tail $LOG to monitor progress.${NC}"
else
    # Line format: "... Trainable parameters: 3,455,180,928 (100.00%)"
    PCT="$(printf '%s\n' "$PCT_LINE" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')"
    printf '%s\n' "$PCT" > "$PCT_FILE"
    echo -e "${BLUE}[$(ts) train_groot_vla] trainer reports: ${PCT_LINE#*INFO - }${NC}"
    # Numeric comparison via awk (bash can't do float >).
    OVER=$(awk -v p="$PCT" -v lim="$MAX_TRAINABLE_PCT" 'BEGIN{print (p>lim)?1:0}')
    if [[ "$OVER" == "1" ]]; then
        echo -e "${RED}[FATAL] trainable% = ${PCT} exceeds MAX_TRAINABLE_PCT=${MAX_TRAINABLE_PCT}.${NC}" >&2
        echo -e "${RED}        This usually means the LLM/visual freeze did NOT take effect.${NC}" >&2
        echo -e "${RED}        Killing trainer (pid $WRAPPER_PID${PY_PID:+, python $PY_PID}) before it OOMs at the optimizer step.${NC}" >&2
        # kill the entire process group so HF dataloader workers go too
        kill -TERM -- -"$WRAPPER_PID" 2>/dev/null || true
        [[ -n "$PY_PID" ]] && kill -TERM "$PY_PID" 2>/dev/null || true
        sleep 3
        kill -KILL -- -"$WRAPPER_PID" 2>/dev/null || true
        [[ -n "$PY_PID" ]] && kill -KILL "$PY_PID" 2>/dev/null || true
        exit 3
    fi
    echo -e "${GREEN}[ok]${NC} trainable% within budget (<=${MAX_TRAINABLE_PCT}). Streaming progress …"
fi

# ----- foreground stream + cleanup ----------------------------------------
trap 'echo -e "${YELLOW}[$(ts) train_groot_vla] received signal -- killing trainer …${NC}"; kill -TERM -- -"$WRAPPER_PID" 2>/dev/null || true; exit 130' INT TERM

# Tail useful progress lines; stop when the wrapper exits.
( tail -n 0 -F --pid="$WRAPPER_PID" "$LOG" 2>/dev/null \
    | stdbuf -oL grep -E 'train_loss|loss=|Step|Trainable parameters|exit_code|OutOfMemory|RuntimeError|epoch|saving model|Saved' \
    | sed -u "s/^/[$(ts) trainer] /" ) &
TAIL_PID=$!

wait "$WRAPPER_PID" 2>/dev/null
EXIT_CODE=$?
kill "$TAIL_PID" 2>/dev/null || true

echo
if [[ "$EXIT_CODE" == "0" ]]; then
    echo -e "${GREEN}[$(ts) train_groot_vla] trainer exited 0 -- artefacts in $OUTPUT_DIR${NC}"
    ls -la "$OUTPUT_DIR" 2>/dev/null | head -15
    exit 0
else
    echo -e "${RED}[$(ts) train_groot_vla] trainer exited $EXIT_CODE${NC}" >&2
    echo -e "${YELLOW}    last informative log lines:${NC}" >&2
    grep -vE "HTTP Request|^$|Loading checkpoint shards" "$LOG" | tail -20 >&2 || true
    exit 1
fi
