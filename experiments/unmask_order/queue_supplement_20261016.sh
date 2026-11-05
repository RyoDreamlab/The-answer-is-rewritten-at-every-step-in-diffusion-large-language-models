#!/usr/bin/env bash
# Supplement 20261016. Does not touch the paused E-package lanes and does not start the Dream-VL accuracy queue.
# GPU 0: MMaDA factorial (symbol / position / len8 / len32) on the first 100 ScienceQA items.
# GPU 1: Dream-VL letter-commit check on the first 100 ScienceQA items, then MMaDA vision on the first 200 MMStar items.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/supplement_20261016
LOGD=$ROOT/logs/supplement_20261016
PY=$ROOT/.venv/bin/python3
mkdir -p "$OUT" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
Q=$LOGD/queue.log
echo "QUEUE START $(date -Iseconds) pid=$$" >> "$Q"

$PY "$S/supplement_20261016.py" --part factor --cuda 0 --n-sciqa 100 --output "$OUT/factor.jsonl" \
  >> "$LOGD/factor.log" 2>&1 &
FPID=$!
echo "factor pid=$FPID cuda=0" >> "$Q"

$PY "$S/dreamvl_commit_20261016.py" --cuda 1 --limit 100 --output "$OUT/dreamvl_sciqa_stock.jsonl" \
  >> "$LOGD/dreamvl.log" 2>&1
echo "dreamvl exit=$? $(date -Iseconds)" >> "$Q"
$PY "$S/supplement_20261016.py" --part vision --cuda 1 --n-mmstar 200 --output "$OUT/vision.jsonl" \
  >> "$LOGD/vision.log" 2>&1
echo "vision exit=$? $(date -Iseconds)" >> "$Q"
$PY "$S/supplement_20261016.py" --part analyze --output "$OUT/vision.jsonl" >> "$LOGD/analyze_vision.log" 2>&1 || true

wait "$FPID"
echo "factor exit=$? $(date -Iseconds)" >> "$Q"
$PY "$S/supplement_20261016.py" --part analyze --output "$OUT/factor.jsonl" >> "$LOGD/analyze_factor.log" 2>&1 || true
echo "QUEUE END $(date -Iseconds)" >> "$Q"
