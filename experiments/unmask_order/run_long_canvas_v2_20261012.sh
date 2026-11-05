#!/usr/bin/env bash
# Sequential C32 then C64 stock decode for one model. No retries; stops on failure.
set -u
model=$1
cuda=$2
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
OUT=$ROOT/outputs/long_canvas_20261012
CHAIN=$ROOT/logs/long_canvas_${model}_chain_20261012.log
echo "CHAIN_START $(date -Iseconds) pid=$$ model=$model cuda=$cuda" >> "$CHAIN"
for gen in 32 64; do
  tag=c$gen
  out=$OUT/${model}_${tag}.jsonl
  log=$ROOT/logs/long_canvas_${model}_${tag}_20261012.log
  n=$(grep -c . "$out" 2>/dev/null); n=${n:-0}
  if [ "$n" -ge 2017 ]; then
    echo "ALREADY_DONE $tag n=$n" >> "$CHAIN"
    continue
  fi
  pat="^[^ ]*python3 [^ ]*long_canvas_decode.py --backend $model .*--output $out"
  running=$(pgrep -f "$pat" | head -n 1)
  if [ -n "$running" ]; then
    echo "ALREADY_RUNNING $tag pid=$running: $(ps -o args= -p "$running"); waiting" >> "$CHAIN"
    while kill -0 "$running" 2>/dev/null; do sleep 30; done
    n=$(grep -c . "$out" 2>/dev/null); n=${n:-0}
    echo "EXIT_OBSERVED $tag pid=$running n=$n $(date -Iseconds)" >> "$CHAIN"
    if [ "$n" -lt 2017 ]; then
      echo "CHAIN_STOP $tag incomplete" >> "$CHAIN"
      exit 1
    fi
    continue
  fi
  echo "LAUNCH $tag n=$n $(date -Iseconds)" >> "$CHAIN"
  .venv/bin/python3 experiments/unmask_order/long_canvas_decode.py --backend "$model" --cuda "$cuda" \
    --gen "$gen" --output "$out" --progress "$OUT/${model}_${tag}.progress.json" >> "$log" 2>&1 < /dev/null &
  echo $! > "$OUT/${model}_${tag}.pid"
  wait $!
  rc=$?
  n=$(grep -c . "$out" 2>/dev/null); n=${n:-0}
  echo "EXIT $tag rc=$rc n=$n $(date -Iseconds)" >> "$CHAIN"
  if [ "$rc" -ne 0 ] || [ "$n" -lt 2017 ]; then
    echo "CHAIN_STOP $tag failed" >> "$CHAIN"
    exit 1
  fi
done
echo "CHAIN_DONE $(date -Iseconds)" >> "$CHAIN"
