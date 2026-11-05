#!/usr/bin/env bash
# Follow-up to run_dla_20261013d.sh: MMaDA full decode with ALL THREE blank-input attention writers zero-ablated
# (L32.attn + L30.attn + L31.attn; together they flatten the blank prior to KL 0.008 at n=2), then re-report.
# Waits for the master queue (queue_20261013d.sh) to exit first so at most one GPU job runs at a time.
# Usage: nohup bash run_dla_extra_20261013d.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
LOGD=$ROOT/logs/dla_20261013d
Q=$LOGD/queue.log
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
export GUEST_LOG=$Q
source "$S/gpu_guest_20261013d.sh"
PY=$ROOT/.venv/bin/python3
SCRIPT=$S/dla_circuit_20261013d.py
echo "EXTRA START $(date -Iseconds) waiting for master queue" >> "$Q"
while pgrep -f "queue_20261013d.sh" > /dev/null; do sleep 120; done
echo "EXTRA master queue finished $(date -Iseconds)" >> "$Q"
gpu=$(pick_gpu); log_gpu_state "$Q"
echo "LAUNCH decode_mmada_zero3 gpu=$gpu $(date -Iseconds)" >> "$Q"
$PY "$SCRIPT" --cuda "$gpu" --mode decode --backend mmada --limit 300 --components L32.attn L30.attn L31.attn \
  --ablation zero --tag writer3_L32.attn-L30.attn-L31.attn_zero >> "$LOGD/decode_mmada_zero3.log" 2>&1 < /dev/null
echo "EXIT decode_mmada_zero3 rc=$? $(date -Iseconds)" >> "$Q"
$PY "$SCRIPT" --mode report >> "$LOGD/report.log" 2>&1
echo "EXTRA END $(date -Iseconds)" >> "$Q"
