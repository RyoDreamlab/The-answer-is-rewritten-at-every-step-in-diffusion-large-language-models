#!/usr/bin/env bash
# Upgrade 1 queue: attractor decodes (stock + 12 step-0 perturbations + 4 calibration placements + teacher-forced dose)
# on the first 400 ScienceQA-IMG items, LaViDa then MMaDA, then the CPU report.  One GPU job at a time, GPU picked
# before each launch (guest etiquette, gpu_guest_20261013d.sh).
# Usage: nohup bash run_attractor_20261013d.sh [all|smoke|lavida|mmada|report] > /dev/null 2>&1 &
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/attractor_20261013d
LOGD=$ROOT/logs/attractor_20261013d
mkdir -p "$OUT" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
export GUEST_LOG=$LOGD/queue.log
source "$S/gpu_guest_20261013d.sh"
PY=$ROOT/.venv/bin/python3
SCRIPT=$S/attractor_20261013d.py
Q=$LOGD/queue.log
STAGE=${1:-all}
echo "QUEUE START stage=$STAGE $(date -Iseconds)" >> "$Q"

run_job() {
  name=$1; shift
  gpu=$(pick_gpu)
  log_gpu_state "$Q"
  echo "LAUNCH $name gpu=$gpu $(date -Iseconds)" >> "$Q"
  $PY "$SCRIPT" --cuda "$gpu" "$@" >> "$LOGD/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "EXIT $name rc=$rc $(date -Iseconds)" >> "$Q"
  return $rc
}

if [ "$STAGE" = all ] || [ "$STAGE" = smoke ]; then
  # 5 items, no shortcut: every variant decoded in full; stock / δ=0 / calib_every must match the existing files
  run_job smoke_lavida --mode decode --backend lavida --limit 5 --no-shortcut --out-dir "$OUT/smoke" || exit 1
  run_job smoke_mmada  --mode decode --backend mmada  --limit 5 --no-shortcut --out-dir "$OUT/smoke" || exit 1
  $PY "$SCRIPT" --mode report --out-dir "$OUT/smoke" >> "$LOGD/smoke_report.log" 2>&1
fi
if [ "$STAGE" = all ] || [ "$STAGE" = lavida ]; then
  run_job lavida --mode decode --backend lavida --limit 400 || exit 1
fi
if [ "$STAGE" = all ] || [ "$STAGE" = mmada ]; then
  run_job mmada --mode decode --backend mmada --limit 400 || exit 1
fi
$PY "$SCRIPT" --mode report >> "$LOGD/report.log" 2>&1
echo "QUEUE END $(date -Iseconds)" >> "$Q"
