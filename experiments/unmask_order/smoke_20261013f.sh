#!/usr/bin/env bash
# P6 smoke: --limit 5 for each arm into outputs/p6_causal_20261013f/smoke2 (GPU picked by pick_gpu), then analysis.
# Usage: nohup bash smoke_20261013f.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/p6_causal_20261013f/smoke2
LOGD=$ROOT/logs/p6_20261013f
mkdir -p "$OUT" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
export GUEST_LOG=$LOGD/smoke2_queue.log
source "$S/gpu_guest_20261013d.sh"
PY=$ROOT/.venv/bin/python3
SCRIPT=$S/p6_causal_20261013f.py
Q=$LOGD/smoke2_queue.log
echo "SMOKE START $(date -Iseconds) pid=$$" >> "$Q"
run_job() {
  name=$1; shift
  gpu=$(pick_gpu); log_gpu_state "$Q"
  echo "LAUNCH $name gpu=$gpu $(date -Iseconds)" >> "$Q"
  $PY "$SCRIPT" --cuda "$gpu" --out-dir "$OUT" "$@" >> "$LOGD/smoke2_$name.log" 2>&1 < /dev/null
  rc=$?
  echo "EXIT $name rc=$rc $(date -Iseconds) $(grep -h 'BETA0_EQUALS_STOCK' "$LOGD/smoke2_$name.log" | tail -1)" >> "$Q"
}
run_job steer_writer_heads --arm steer --group writer_heads --limit 5
run_job steer_random_heads --arm steer --group random_heads --limit 5
run_job steer_writer_comps --arm steer --group writer_comps --limit 5
run_job patch --arm patch --limit 5 --pool 60
run_job canvas --arm canvas --limit 5
$PY "$S/p6_analysis_20261013f.py" --out-dir "$OUT" >> "$LOGD/smoke2_analysis.log" 2>&1
echo "SMOKE END rc_analysis=$? $(date -Iseconds)" >> "$Q"
