#!/usr/bin/env bash
# B1: M3CoT val (1108, gen 16) calibrated decode, one model, one alpha rule.
# Usage: run_b1_m3cot_20261013.sh <lavida|mmada> <cuda> <spread|full>
#   spread = gated rule alpha_n = 1 - p_min/p_max ; full = alpha 1 (Zhao-2021 full-strength calibration)
set -u
model=$1; cuda=$2; rule=$3
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
OUT=$ROOT/outputs/m3cot_20261013
mkdir -p "$OUT" "$ROOT/logs/week1_20261013"
out=$OUT/${model}_calib_${rule}.jsonl
log=$ROOT/logs/week1_20261013/b1_m3cot_${model}_${rule}.log
prior=$ROOT/outputs/m3cot_20261012/cf_${model}.jsonl
if pgrep -f "calib_decode_m3cot_20261013.py --backend $model .*--output $out" >/dev/null; then
  echo "ALREADY_RUNNING $model $rule" >> "$log"; exit 1
fi
echo "LAUNCH $model rule=$rule cuda=$cuda $(date -Iseconds)" >> "$log"
echo "CMD .venv/bin/python3 experiments/unmask_order/calib_decode_m3cot_20261013.py --backend $model --cuda $cuda --mode calib --alpha-rule $rule --prior-file $prior --output $out --progress $OUT/${model}_calib_${rule}.progress.json" >> "$log"
.venv/bin/python3 experiments/unmask_order/calib_decode_m3cot_20261013.py --backend "$model" --cuda "$cuda" \
  --mode calib --alpha-rule "$rule" --prior-file "$prior" \
  --output "$out" --progress "$OUT/${model}_calib_${rule}.progress.json" >> "$log" 2>&1 < /dev/null &
echo $! > "$OUT/${model}_calib_${rule}.pid"
wait $!
echo "EXIT $model rule=$rule rc=$? n=$(grep -c . "$out") $(date -Iseconds)" >> "$log"
