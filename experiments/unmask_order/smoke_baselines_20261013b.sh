#!/usr/bin/env bash
# 5-item smoke tests of the baseline wrappers (GPU 0), one model per invocation; outputs under .../smoke/.
# Usage: nohup bash smoke_baselines_20261013b.sh <lavida|mmada> > logs/baselines_20261013b/smoke_<model>.nohup 2>&1 &
set -u
model=$1
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
O=$ROOT/outputs/calib_baselines_20261013b/smoke
L=$ROOT/logs/baselines_20261013b
mkdir -p "$O" "$L"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
slog=$L/smoke_${model}.log
echo "SMOKE START $model $(date -Iseconds)" >> "$slog"
for rule in none atabuzzaman kumar; do
  tag=${model}_sqa_${rule}
  .venv/bin/python3 "$S/calib_baselines_decode_20261013b.py" --rule $rule --backend "$model" --cuda 0 --limit 5 \
    --output "$O/$tag.jsonl" --progress "$O/$tag.progress.json" >> "$L/smoke_$tag.log" 2>&1 < /dev/null
  echo "SQA $rule rc=$? n=$(grep -c . "$O/$tag.jsonl" 2>/dev/null) $(date -Iseconds)" >> "$slog"
done
for rule in none atabuzzaman; do
  tag=${model}_m3cot_${rule}
  if [ "$rule" = none ]; then extra="--mode stock"; else extra="--mode calib --prior-file $ROOT/outputs/m3cot_20261012/cf_${model}.jsonl"; fi
  .venv/bin/python3 "$S/calib_baselines_decode_m3cot_20261013b.py" --rule $rule --backend "$model" --cuda 0 --limit 5 $extra \
    --output "$O/$tag.jsonl" --progress "$O/$tag.progress.json" >> "$L/smoke_$tag.log" 2>&1 < /dev/null
  echo "M3COT $rule rc=$? n=$(grep -c . "$O/$tag.jsonl" 2>/dev/null) $(date -Iseconds)" >> "$slog"
done
echo "SMOKE END $model $(date -Iseconds)" >> "$slog"
