#!/bin/bash
# Wait until the letter override and the argmax trajectory release both GPUs,
# then measure image-key blocking. Does not kill unrelated processes and does
# not start a second copy of a live job.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
PY="$ROOT/.venv/bin/python3"
SCRIPT="$ROOT/experiments/unmask_order/image_key_block.py"
TRAJ_SCRIPT="$ROOT/experiments/unmask_order/letter_argmax_traj.py"
OVR_SCRIPT="$ROOT/experiments/unmask_order/letter_second_override.py"
OVR="$ROOT/outputs/letter_second_override_20261010"
TRAJ="$ROOT/outputs/letter_argmax_traj_20261010"
CONTRAST="$ROOT/outputs/image_key_block_20261010"
DELTA="$ROOT/outputs/image_key_blocked_delta2_20261010"
LOGDIR="$ROOT/logs"
ORCH="$LOGDIR/image_key_orchestrator_20261010.log"
LOCK="$LOGDIR/image_key_orchestrator_20261010.lock"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$ROOT/hf-cache"
export URD_PROJECT="$ROOT"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ORCH_LOCK_HELD $(date -Iseconds)" >> "$ORCH"
  exit 0
fi

log() {
  echo "$*" >> "$ORCH"
  echo "$*"
}

nlines() {
  if [ -f "$1" ]; then
    wc -l < "$1" | tr -d " "
  else
    echo 0
  fi
}

alive() {
  pgrep -f "$1" >/dev/null 2>&1
}

gpu_mem() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " " | paste -sd, -
}

compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
}

gpu_slot_free() {
  local idx="$1"
  local mem apps
  mem="$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")"
  apps="$(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")"
  [ -n "$mem" ] && [ "$mem" -lt 2000 ] && [ -z "$apps" ]
}

both_gpus_free() {
  gpu_slot_free 0 && gpu_slot_free 1
}

hard_fail() {
  local log_path="$1"
  [ -f "$log_path" ] || return 1
  python3 - "$log_path" << 'PY'
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
chunk = text.split("START ")[-1]
needles = (
    "FD_CHECK_FAIL",
    "finite-difference check failed",
    "reconstruction argmax",
    "did not change logits",
    "not one contiguous",
    "refusing to write",
    "no image tokens",
    "outside the image",
    "non-finite",
    "unexpected ff_out",
)
sys.exit(0 if any(needle in chunk for needle in needles) else 1)
PY
}

quote_log() {
  local log_path="$1"
  log "----- tail $log_path -----"
  tail -n 80 "$log_path" >> "$ORCH" 2>/dev/null || true
  log "----- end tail -----"
}

launch_job() {
  local backend="$1"
  local cuda="$2"
  local outdir="$3"
  local log_path="$4"
  local script="$5"
  local part_args="$6"
  local needle="$7"
  if alive "$needle"; then
    log "ALREADY_RUNNING $backend $needle"
    return 0
  fi
  mkdir -p "$outdir"
  local out="$outdir/${backend}.jsonl"
  local prog="$outdir/${backend}.progress.json"
  local n
  n="$(nlines "$out")"
  if [ "$n" -ge 2017 ]; then
    log "ALREADY_DONE $backend n=$n $outdir"
    return 0
  fi
  log "LAUNCH backend=$backend cuda=$cuda n=$n script=$(basename "$script") $(date -Iseconds)"
  setsid bash -c "
    set -u
    cd \"$ROOT\"
    export PYTHONUNBUFFERED=1 WANDB_MODE=disabled WANDB_DISABLED=true
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME=\"$ROOT/hf-cache\" URD_PROJECT=\"$ROOT\"
    echo START \$(date -Iseconds) backend=$backend cuda=$cuda >> \"$log_path\"
    exec \"$PY\" -u \"$script\" $part_args --backend \"$backend\" --cuda \"$cuda\" --output \"$out\" --progress \"$prog\"
  " >> "$log_path" 2>&1 < /dev/null &
  log "LAUNCHED pid=$! backend=$backend"
}

log "ORCH_START $(date -Iseconds) pid=$$"
log "override_match $(pgrep -af 'letter_second_override.py --backend' | awk '{print $1}' | paste -sd, - || true)"
log "waiter_match $(pgrep -af 'wait_letter_argmax_traj_20261010.sh' | awk '{print $1}' | paste -sd, - || true)"

override_resumes=0
traj_resumes=0
while true; do
  o1="$(nlines "$OVR/lavida.jsonl")"
  o2="$(nlines "$OVR/mmada.jsonl")"
  t1="$(nlines "$TRAJ/lavida.jsonl")"
  t2="$(nlines "$TRAJ/mmada.jsonl")"
  log "POLL override=$o1,$o2 traj=$t1,$t2 gpu=$(gpu_mem) $(date -Iseconds)"
  log "COMPUTE $(compute_apps | tr '\n' '; ')"
  if alive "letter_second_override.py --backend" || alive "wait_letter_argmax_traj_20261010.sh" || alive "letter_argmax_traj.py --backend"; then
    sleep 60
    continue
  fi
  if [ "$o1" -lt 2017 ] || [ "$o2" -lt 2017 ]; then
    if [ "$override_resumes" -ge 1 ]; then
      log "OVERRIDE_INCOMPLETE lavida=$o1 mmada=$o2"
      quote_log "$LOGDIR/letter_second_override_lavida_20261010.log"
      quote_log "$LOGDIR/letter_second_override_mmada_20261010.log"
      exit 2
    fi
    if ! both_gpus_free; then
      log "OVERRIDE_RESUME_WAIT_GPU $(gpu_mem)"
      sleep 60
      continue
    fi
    log "OVERRIDE_PARTIAL_RESUME lavida=$o1 mmada=$o2"
    if [ "$o1" -lt 2017 ]; then
      launch_job lavida 0 "$OVR" "$LOGDIR/letter_second_override_lavida_20261010.log" "$OVR_SCRIPT" "" "letter_second_override.py --backend lavida"
    fi
    if [ "$o2" -lt 2017 ]; then
      launch_job mmada 1 "$OVR" "$LOGDIR/letter_second_override_mmada_20261010.log" "$OVR_SCRIPT" "" "letter_second_override.py --backend mmada"
    fi
    override_resumes=$((override_resumes + 1))
    sleep 60
    continue
  fi
  if [ "$t1" -lt 2017 ] || [ "$t2" -lt 2017 ]; then
    if [ "$traj_resumes" -ge 1 ]; then
      log "TRAJ_INCOMPLETE lavida=$t1 mmada=$t2"
      quote_log "$LOGDIR/letter_argmax_traj_lavida_20261010.log"
      quote_log "$LOGDIR/letter_argmax_traj_mmada_20261010.log"
      exit 3
    fi
    if ! both_gpus_free; then
      log "TRAJ_RESUME_WAIT_GPU $(gpu_mem)"
      sleep 60
      continue
    fi
    log "TRAJ_PARTIAL_RESUME lavida=$t1 mmada=$t2"
    if [ "$t1" -lt 2017 ]; then
      launch_job lavida 0 "$TRAJ" "$LOGDIR/letter_argmax_traj_lavida_20261010.log" "$TRAJ_SCRIPT" "" "letter_argmax_traj.py --backend lavida"
    fi
    if [ "$t2" -lt 2017 ]; then
      launch_job mmada 1 "$TRAJ" "$LOGDIR/letter_argmax_traj_mmada_20261010.log" "$TRAJ_SCRIPT" "" "letter_argmax_traj.py --backend mmada"
    fi
    traj_resumes=$((traj_resumes + 1))
    sleep 60
    continue
  fi
  if ! both_gpus_free; then
    log "GPU_STILL_BUSY $(gpu_mem)"
    sleep 60
    continue
  fi
  log "READY_FOR_PART_A $(date -Iseconds)"
  break
done

run_measurement() {
  local part="$1"
  local outdir="$2"
  local launches=0
  while true; do
    local n1 n2
    n1="$(nlines "$outdir/lavida.jsonl")"
    n2="$(nlines "$outdir/mmada.jsonl")"
    local live1=0 live2=0
    if alive "image_key_block.py --part $part --backend lavida"; then
      live1=1
    fi
    if alive "image_key_block.py --part $part --backend mmada"; then
      live2=1
    fi
    log "POLL $part lavida=$n1 live=$live1 mmada=$n2 live=$live2 gpu=$(gpu_mem) $(date -Iseconds)"
    if [ "$n1" -ge 2017 ] && [ "$n2" -ge 2017 ] && [ "$live1" -eq 0 ] && [ "$live2" -eq 0 ]; then
      log "PART_DONE $part $(date -Iseconds)"
      return 0
    fi
    local log1 log2
    if [ "$part" = "contrast" ]; then
      log1="$LOGDIR/image_key_block_lavida_20261010.log"
      log2="$LOGDIR/image_key_block_mmada_20261010.log"
    else
      log1="$LOGDIR/image_key_blocked_delta2_lavida_20261010.log"
      log2="$LOGDIR/image_key_blocked_delta2_mmada_20261010.log"
    fi
    if [ "$live1" -eq 0 ] && [ "$n1" -lt 2017 ] && hard_fail "$log1"; then
      log "HARD_FAIL $part lavida"
      quote_log "$log1"
      return 1
    fi
    if [ "$live2" -eq 0 ] && [ "$n2" -lt 2017 ] && hard_fail "$log2"; then
      log "HARD_FAIL $part mmada"
      quote_log "$log2"
      return 1
    fi
    local started=0
    if [ "$live1" -eq 0 ] && [ "$n1" -lt 2017 ] && gpu_slot_free 0; then
      if [ "$launches" -ge 4 ]; then
        log "LAUNCH_CAP $part lavida"
        quote_log "$log1"
        return 1
      fi
      launch_job lavida 0 "$outdir" "$log1" "$SCRIPT" "--part $part" "image_key_block.py --part $part --backend lavida"
      started=1
    fi
    if [ "$live2" -eq 0 ] && [ "$n2" -lt 2017 ] && gpu_slot_free 1; then
      if [ "$launches" -ge 4 ]; then
        log "LAUNCH_CAP $part mmada"
        quote_log "$log2"
        return 1
      fi
      launch_job mmada 1 "$outdir" "$log2" "$SCRIPT" "--part $part" "image_key_block.py --part $part --backend mmada"
      started=1
    fi
    if [ "$started" -eq 1 ]; then
      launches=$((launches + 1))
    fi
    sleep 60
  done
}

if ! run_measurement contrast "$CONTRAST"; then
  log "PART_A_FAILED $(date -Iseconds)"
  exit 4
fi

if ! run_measurement delta2 "$DELTA"; then
  log "PART_B_FAILED $(date -Iseconds)"
  exit 5
fi

log "SUMMARIZE $(date -Iseconds)"
"$PY" -u "$SCRIPT" --summarize-only >> "$ORCH" 2>&1
log "ORCH_DONE $(date -Iseconds)"
