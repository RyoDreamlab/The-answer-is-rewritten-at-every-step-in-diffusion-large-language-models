#!/usr/bin/env bash
# Week-1 second queue for one model/GPU: waits for week1_queue_20261013.sh <model> to finish, then
#   B3 layer-wise logit lens (all 2017 items, real + content-free blank)
#   B4 answer-slot dose control (M{k} vs Trand{k}/Tlast{k}, k = 1,2,4,8,15)
# Usage: week1_queue2_20261013.sh <lavida|mmada> <cuda> [b4_max_samples] [nowait]
set -u
model=$1; cuda=$2; b4max=${3:-0}; nowait=${4:-}
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
L=$ROOT/logs/week1_20261013; mkdir -p "$L"
qlog=$L/queue2_${model}.log
echo "QUEUE2 START $model cuda=$cuda b4max=$b4max $(date -Iseconds)" >> "$qlog"
if [ "$nowait" != "nowait" ]; then
  while pgrep -f "week1_queue_20261013.sh $model" >/dev/null; do sleep 60; done
fi
echo "QUEUE1 finished; starting B3 $(date -Iseconds)" >> "$qlog"
O3=$ROOT/outputs/layer_logit_lens_20261013; mkdir -p "$O3"
echo "CMD .venv/bin/python3 $S/layer_logit_lens_20261013.py --backend $model --cuda $cuda --output $O3/${model}.jsonl --progress $O3/${model}.progress.json" >> "$qlog"
.venv/bin/python3 "$S/layer_logit_lens_20261013.py" --backend "$model" --cuda "$cuda" \
  --output "$O3/${model}.jsonl" --progress "$O3/${model}.progress.json" >> "$L/b3_lens_${model}.log" 2>&1 < /dev/null
echo "B3 rc=$? n=$(grep -c . "$O3/${model}.jsonl") $(date -Iseconds)" >> "$qlog"
O4=$ROOT/outputs/slot_dose_20261013; mkdir -p "$O4"
echo "CMD .venv/bin/python3 $S/slot_dose_20261013.py --backend $model --cuda $cuda --max-samples $b4max --output $O4/${model}.jsonl --progress $O4/${model}.progress.json" >> "$qlog"
.venv/bin/python3 "$S/slot_dose_20261013.py" --backend "$model" --cuda "$cuda" --max-samples "$b4max" \
  --output "$O4/${model}.jsonl" --progress "$O4/${model}.progress.json" >> "$L/b4_dose_${model}.log" 2>&1 < /dev/null
echo "B4 rc=$? n=$(grep -c . "$O4/${model}.jsonl") $(date -Iseconds)" >> "$qlog"
echo "QUEUE2 END $model $(date -Iseconds)" >> "$qlog"
