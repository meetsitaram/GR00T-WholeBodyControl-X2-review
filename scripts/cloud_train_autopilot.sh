#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Autopilot for cloud GR00T VLA training:
#   1. Watches the cloud trainer until it exits.
#   2. On success, rsync's the requested checkpoints to the local laptop.
#   3. On all-rsync-success, STOP's (not deletes) the Nebius instance so the
#      compute charge stops but the disk stays around for review.
#   4. On any failure, halts and leaves the instance running so the user can
#      ssh in and inspect.
#
# Designed to be `nohup`'d so the laptop can disconnect/sleep. All progress
# is appended to ${STATUS_LOG}.
# ---------------------------------------------------------------------------
set -u
set -o pipefail

# -------- knobs (override via env) -----------------------------------------
INSTANCE_ID="${INSTANCE_ID:-computeinstance-e00hp7pf0r4tkze90f}"
HOST="${HOST:-ubuntu@195.242.31.119}"
# REMOTE_CKPT_DIR is the --output-dir passed to the trainer. The HF Trainer
# then creates <REMOTE_CKPT_DIR>/<experiment_name>/checkpoint-N/ underneath,
# so phase 2 auto-descends one level if it finds exactly one subdir there.
REMOTE_CKPT_DIR="${REMOTE_CKPT_DIR:-/home/ubuntu/GR00T-WholeBodyControl/data/checkpoints/x2_reach_and_retract_v1_stereo_30k_v1_b16}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-${REMOTE_CKPT_DIR}_run}"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-/home/stickbot/x2_cloud_checkpoints/reach_and_retract_v1_b16_30k}"
STATUS_LOG="${STATUS_LOG:-/home/stickbot/logs/cloud_train_autopilot.log}"
PID_FILE="${PID_FILE:-/home/stickbot/logs/cloud_train_autopilot.pid}"
# checkpoints we _want_ pulled. If the trainer rotated some away
# (save_total_limit), we WARN and pull the ones that survived rather than
# failing the whole pipeline.
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-10000 20000 25000 27500 30000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
HEARTBEAT_EVERY="${HEARTBEAT_EVERY:-300}"   # 5 min between progress prints when no event

# -------- helpers ----------------------------------------------------------
log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${STATUS_LOG}"
}
fail() {
    log "FATAL: $*"
    log "Instance LEFT RUNNING (id=${INSTANCE_ID}) so you can inspect. Stop manually when ready."
    exit 1
}

mkdir -p "$(dirname "${STATUS_LOG}")" "$(dirname "${PID_FILE}")" "${LOCAL_CKPT_DIR}"
echo $$ > "${PID_FILE}"

# -------- preamble ---------------------------------------------------------
{
    echo
    echo "================================================================================"
    echo "cloud_train_autopilot.sh starting at $(date -Iseconds)"
    echo "  pid:               $$"
    echo "  instance_id:       ${INSTANCE_ID}"
    echo "  host:              ${HOST}"
    echo "  remote_ckpt_dir:   ${REMOTE_CKPT_DIR}"
    echo "  local_ckpt_dir:    ${LOCAL_CKPT_DIR}"
    echo "  checkpoint_steps:  ${CHECKPOINT_STEPS}"
    echo "  poll_seconds:      ${POLL_SECONDS}"
    echo "  pid_file:          ${PID_FILE}"
    echo "================================================================================"
} | tee -a "${STATUS_LOG}"

# -------- phase 1: wait for training to finish -----------------------------
log "Phase 1: polling cloud trainer until exit ..."
LAST_STATUS_PRINT=0
while true; do
    NOW=$(date +%s)
    # Single SSH check pulls: is trainer alive + last progress line + exit_code marker
    REMOTE=$(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 "${HOST}" bash <<RSH 2>/dev/null
        alive=\$(pgrep -af launch_finetune_x2.py | head -1 | wc -l)
        last_step=\$(tail -c 4000 ${REMOTE_RUN_DIR}/finetune.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/30000' | tail -1)
        exit_marker=\$(grep -E '\\[finetune\\] exit_code=' ${REMOTE_RUN_DIR}/finetune.log 2>/dev/null | tail -1)
        echo "alive=\$alive"
        echo "last_step=\$last_step"
        echo "exit_marker=\$exit_marker"
RSH
    )
    SSH_RC=$?
    if [[ $SSH_RC -ne 0 ]]; then
        log "WARN: ssh check failed (rc=$SSH_RC). Retrying in ${POLL_SECONDS}s."
        sleep "${POLL_SECONDS}"
        continue
    fi

    ALIVE=$(echo "$REMOTE" | grep -oE 'alive=[0-9]+' | cut -d= -f2)
    LAST_STEP=$(echo "$REMOTE" | grep -oE 'last_step=[0-9]+/[0-9]+' | cut -d= -f2)
    EXIT_MARKER=$(echo "$REMOTE" | grep '^exit_marker=' | cut -d= -f2-)

    if [[ -n "${EXIT_MARKER}" ]]; then
        log "Trainer wrote exit line: ${EXIT_MARKER}"
        # Parse exit code from the marker line "[finetune] exit_code=N at <iso>"
        EXIT_CODE=$(echo "${EXIT_MARKER}" | sed -E 's/.*exit_code=([0-9]+).*/\1/')
        if [[ "${EXIT_CODE}" != "0" ]]; then
            fail "trainer exited with code ${EXIT_CODE}. Last 30 lines of finetune.log:
$(ssh "${HOST}" "tail -30 ${REMOTE_RUN_DIR}/finetune.log" 2>&1)"
        fi
        log "Trainer SUCCESS (exit_code=0)."
        break
    fi
    if [[ "${ALIVE}" == "0" ]]; then
        log "Trainer process is GONE but no exit_code marker found. Tail of log:"
        ssh "${HOST}" "tail -30 ${REMOTE_RUN_DIR}/finetune.log" 2>&1 | tee -a "${STATUS_LOG}"
        fail "trainer disappeared without writing exit_code line (likely killed by OOM or signal)."
    fi

    # heartbeat
    if (( NOW - LAST_STATUS_PRINT >= HEARTBEAT_EVERY )); then
        log "heartbeat: trainer ALIVE, last_step=${LAST_STEP:-unknown}"
        LAST_STATUS_PRINT=$NOW
    fi
    sleep "${POLL_SECONDS}"
done

# -------- phase 2: resolve real checkpoint dir + see which steps survived ---
log "Phase 2: resolving real checkpoint dir and inventorying surviving checkpoints ..."

# HF Trainer writes to <output_dir>/<experiment_name>/checkpoint-N. If no
# checkpoint-* dirs are found at REMOTE_CKPT_DIR but exactly one subdir
# beneath it contains them, descend into that subdir.
RESOLVED_CKPT_DIR="${REMOTE_CKPT_DIR}"
DESCENT=$(ssh "${HOST}" bash <<RSH 2>/dev/null
    if compgen -G "${REMOTE_CKPT_DIR}/checkpoint-*" > /dev/null; then
        echo "${REMOTE_CKPT_DIR}"
    else
        # look one level deeper for a unique experiment subdir
        cands=()
        for d in ${REMOTE_CKPT_DIR}/*/; do
            [ -d "\$d" ] || continue
            if compgen -G "\${d}checkpoint-*" > /dev/null; then
                cands+=("\${d%/}")
            fi
        done
        if [ \${#cands[@]} -eq 1 ]; then
            echo "\${cands[0]}"
        else
            echo "AMBIGUOUS:\${cands[*]}"
        fi
    fi
RSH
)
if [[ "${DESCENT}" == AMBIGUOUS:* ]]; then
    fail "could not uniquely resolve checkpoint dir under ${REMOTE_CKPT_DIR}: ${DESCENT#AMBIGUOUS:}"
fi
if [[ -n "${DESCENT}" && "${DESCENT}" != "${REMOTE_CKPT_DIR}" ]]; then
    log "  auto-descended into nested experiment dir: ${DESCENT}"
    RESOLVED_CKPT_DIR="${DESCENT}"
fi

# enumerate every checkpoint-N that actually exists with weights (single ssh).
AVAILABLE=$(ssh "${HOST}" bash <<RSH 2>/dev/null
    out=""
    for d in ${RESOLVED_CKPT_DIR}/checkpoint-*/; do
        [ -d "\$d" ] || continue
        step=\$(basename "\${d%/}" | sed 's/^checkpoint-//')
        has_weights=0
        for f in model.safetensors pytorch_model.bin model-00001-of-00002.safetensors; do
            if [ -f "\${d}\${f}" ]; then has_weights=1; break; fi
        done
        if [ \$has_weights -eq 1 ]; then
            echo "\$step"
        fi
    done
RSH
)
AVAILABLE=$(echo "${AVAILABLE}" | tr '\n' ' ' | xargs)
log "  available checkpoints on cloud: ${AVAILABLE:-<none>}"

if [[ -z "${AVAILABLE}" ]]; then
    fail "no usable checkpoints found under ${RESOLVED_CKPT_DIR}"
fi

# intersect wanted vs available
TO_PULL=""
ROTATED_AWAY=""
for step in ${CHECKPOINT_STEPS}; do
    if echo " ${AVAILABLE} " | grep -q " ${step} "; then
        TO_PULL="${TO_PULL} ${step}"
    else
        ROTATED_AWAY="${ROTATED_AWAY} ${step}"
    fi
done
TO_PULL=$(echo "${TO_PULL}" | xargs)
ROTATED_AWAY=$(echo "${ROTATED_AWAY}" | xargs)
if [[ -n "${ROTATED_AWAY}" ]]; then
    log "  WARN: requested checkpoint(s) [${ROTATED_AWAY}] were rotated away by save_total_limit; skipping them"
fi
if [[ -z "${TO_PULL}" ]]; then
    fail "none of the requested checkpoints (${CHECKPOINT_STEPS}) survived. Available: ${AVAILABLE}"
fi
log "  will pull: ${TO_PULL}"

# -------- phase 3: rsync each surviving checkpoint --------------------------
log "Phase 3: rsync ${TO_PULL} to ${LOCAL_CKPT_DIR}/ ..."
RSYNC_FAILED=""
for step in ${TO_PULL}; do
    DST="${LOCAL_CKPT_DIR}/checkpoint-${step}"
    mkdir -p "${DST}"
    log "  rsync checkpoint-${step} ..."
    if rsync -a --compress --info=stats2 \
        "${HOST}:${RESOLVED_CKPT_DIR}/checkpoint-${step}/" "${DST}/" >> "${STATUS_LOG}" 2>&1; then
        SIZE=$(du -sh "${DST}" | awk '{print $1}')
        log "    ok: checkpoint-${step} pulled (${SIZE})"
    else
        log "    FAIL: rsync of checkpoint-${step} failed"
        RSYNC_FAILED="${RSYNC_FAILED} ${step}"
    fi
done

# also pull the shared per-experiment processor/ + experiment_cfg/ (needed
# alongside any checkpoint for inference) -- small dirs.
log "Phase 3a: pulling shared processor/ + experiment_cfg/ ..."
rsync -a --compress \
    "${HOST}:${RESOLVED_CKPT_DIR}/processor/" "${LOCAL_CKPT_DIR}/processor/" \
    >> "${STATUS_LOG}" 2>&1 \
    && log "  ok: processor/ pulled" \
    || log "  WARN: processor/ pull failed (non-fatal)"
rsync -a --compress \
    "${HOST}:${RESOLVED_CKPT_DIR}/experiment_cfg/" "${LOCAL_CKPT_DIR}/experiment_cfg/" \
    >> "${STATUS_LOG}" 2>&1 \
    && log "  ok: experiment_cfg/ pulled" \
    || log "  WARN: experiment_cfg/ pull failed (non-fatal)"

# also pull the run dir (finetune.log + trainable_pct.txt + pid file -- tiny)
log "Phase 3b: pulling small run-metadata bundle ..."
RUN_DST="${LOCAL_CKPT_DIR}_run"
mkdir -p "${RUN_DST}"
rsync -a --compress \
    --exclude='*.bin' --exclude='*.safetensors' --exclude='wandb' \
    "${HOST}:${REMOTE_RUN_DIR}/" "${RUN_DST}/" >> "${STATUS_LOG}" 2>&1 \
    && log "  ok: run-metadata pulled to ${RUN_DST}/" \
    || log "  WARN: run-metadata pull failed (non-fatal)"

if [[ -n "${RSYNC_FAILED}" ]]; then
    fail "one or more checkpoint rsyncs failed: ${RSYNC_FAILED}"
fi

# -------- phase 4: stop (not delete) Nebius instance -----------------------
log "Phase 4: stopping Nebius instance (preserves disk) ..."
STOP_OUT=$(nebius compute instance stop --id "${INSTANCE_ID}" 2>&1)
STOP_RC=$?
log "  nebius stop output: ${STOP_OUT}"
if [[ ${STOP_RC} -ne 0 ]]; then
    fail "nebius stop failed (rc=${STOP_RC}). Stop the VM manually to halt billing."
fi

# verify
sleep 5
STATUS=$(nebius compute instance get --id "${INSTANCE_ID}" --format=json 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',{}).get('state','?'))" 2>/dev/null)
log "  instance status now: ${STATUS}"

# -------- summary ----------------------------------------------------------
log "DONE. Summary:"
log "  local checkpoints: ${LOCAL_CKPT_DIR}/"
log "  local run-metadata: ${RUN_DST}/"
log "  instance ${INSTANCE_ID} is ${STATUS} (disk preserved)."
log "  to restart later: nebius compute instance start --id ${INSTANCE_ID}"
log "  to permanently delete: nebius compute instance delete --id ${INSTANCE_ID}"

rm -f "${PID_FILE}"
exit 0
