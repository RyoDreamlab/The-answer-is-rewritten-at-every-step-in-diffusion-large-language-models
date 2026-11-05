#!/usr/bin/env bash
# Full 2017-item calibrated decode (gen 16) for one model. Refuses to double-launch.
set -u
model=$1
cuda=$2
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
OUT=$ROOT/outputs/calib_decode_20261012
out=$OUT/${model}.jsonl
log=$ROOT/logs/calib_decode_${model}_20261012.log
if pgrep -f "^[^ ]*python3 [^ ]*calib_decode.py --backend $model .*--output $out" >/dev/null; then
  echo "ALREADY_RUNNING $model" >> "$log"; exit 1
fi
echo "LAUNCH $model cuda=$cuda $(date -Iseconds)" >> "$log"
.venv/bin/python3 experiments/unmask_order/calib_decode.py --backend "$model" --cuda "$cuda" \
  --output "$out" --progress "$OUT/${model}.progress.json" >> "$log" 2>&1 < /dev/null &
echo $! > "$OUT/${model}.pid"
wait $!
echo "EXIT $model rc=$? n=$(grep -c . "$out") $(date -Iseconds)" >> "$log"
