#!/usr/bin/env bash
# B2: gated (1 - p_min/p_max) content-free calibration, FULL ScienceQA-IMG test decode (2017, gen 16), one model.
# Usage: run_b2_sqa_gated_20261013.sh <lavida|mmada> <cuda>. Refuses to double-launch. Resumable jsonl.
set -u
model=$1; cuda=$2
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
OUT=$ROOT/outputs/calib_gated_decode_20261013
mkdir -p "$OUT" "$ROOT/logs/week1_20261013"
out=$OUT/${model}.jsonl
log=$ROOT/logs/week1_20261013/b2_sqa_gated_${model}.log
if pgrep -f "calib_gated_decode_20261013.py --backend $model .*--output $out" >/dev/null; then
  echo "ALREADY_RUNNING $model" >> "$log"; exit 1
fi
echo "LAUNCH $model cuda=$cuda $(date -Iseconds)" >> "$log"
echo "CMD .venv/bin/python3 experiments/unmask_order/calib_gated_decode_20261013.py --backend $model --cuda $cuda --output $out --progress $OUT/${model}.progress.json" >> "$log"
.venv/bin/python3 experiments/unmask_order/calib_gated_decode_20261013.py --backend "$model" --cuda "$cuda" \
  --output "$out" --progress "$OUT/${model}.progress.json" >> "$log" 2>&1 < /dev/null &
echo $! > "$OUT/${model}.pid"
wait $!
echo "EXIT $model rc=$? n=$(grep -c . "$out") $(date -Iseconds)" >> "$log"
