#!/usr/bin/env bash
# LaViDa causal write-site test: head discovery (200 held-out MMStar last-step items) -> steer (next 200) -> analyze.
# New outputs only under outputs/lavida_causal_20261019/. Does not touch other outputs, E lanes, or other GPUs.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT TOKENIZERS_PARALLELISM=false
PY=$ROOT/.venv/bin/python3
SCRIPT=$ROOT/experiments/unmask_order/lavida_causal_20261019.py
BACKEND=${BACKEND:-lavida}
LOG=$ROOT/logs/${BACKEND}_causal_20261019
mkdir -p "$LOG"
CUDA=${CUDA:-0}
Q=$LOG/queue.log
echo "LC START $(date -Is) pid=$$ cuda=$CUDA backend=$BACKEND" >> "$Q"

run_step() {
  local name=$1; shift
  echo "LC LAUNCH $name $(date -Is)" >> "$Q"
  "$PY" "$SCRIPT" "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "LC RETRY $name rc=$rc $(date -Is)" >> "$Q"
    "$PY" "$SCRIPT" "$@" >> "$LOG/$name.log" 2>&1
    rc=$?
  fi
  echo "LC DONE $name rc=$rc $(date -Is)" >> "$Q"
  return $rc
}

run_step dla   --mode dla   --backend "$BACKEND" --cuda "$CUDA" || { echo "LC ABORT dla $(date -Is)" >> "$Q"; exit 1; }
run_step steer --mode steer --backend "$BACKEND" --cuda "$CUDA" || { echo "LC ABORT steer $(date -Is)" >> "$Q"; exit 1; }
run_step analyze --mode analyze --backend "$BACKEND"
echo "LC END $(date -Is)" >> "$Q"
