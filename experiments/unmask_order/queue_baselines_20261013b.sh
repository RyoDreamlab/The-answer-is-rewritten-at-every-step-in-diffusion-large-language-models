#!/usr/bin/env bash
# Prior-work calibration baselines queue (2026-10-13b). GPU 0 ONLY.
# Two independent chains (one per model), each sequential:
#   B1 Atabuzzaman SQA -> B1 Atabuzzaman M3CoT -> B2 Kumar SQA
# Every GPU job is launched with nohup ... & and waited; one automatic retry (decodes resume).
# Before each launch waits until GPU-0 used memory < MAXMEM MiB. When both chains finish, runs the analysis
# script and writes a final line to logs/baselines_20261013b/queue.log.
# Usage: nohup bash queue_baselines_20261013b.sh > logs/baselines_20261013b/queue.nohup 2>&1 &
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
O=$ROOT/outputs/calib_baselines_20261013b
L=$ROOT/logs/baselines_20261013b
CUDA=0
MAXMEM=60000
mkdir -p "$O" "$L"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
qlog=$L/queue.log
echo "QUEUE START $(date -Iseconds) pid=$$ cuda=$CUDA" >> "$qlog"

wait_mem() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt "$MAXMEM" ]; then break; fi
    echo "WAIT gpu$CUDA used=${used}MiB >= $MAXMEM $(date -Iseconds)" >> "$qlog"
    sleep 30
  done
}

run_job() {  # run_job <tag> <cmd...>
  tag=$1; shift
  for attempt in 1 2; do
    wait_mem
    echo "LAUNCH $tag attempt=$attempt gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $CUDA) $(date -Iseconds)" >> "$qlog"
    echo "CMD $*" >> "$qlog"
    nohup "$@" >> "$L/$tag.log" 2>&1 < /dev/null &
    pid=$!
    echo $pid > "$O/$tag.pid"
    wait $pid
    rc=$?
    n=$(grep -c . "$O/$tag.jsonl" 2>/dev/null || echo 0)
    echo "EXIT $tag attempt=$attempt rc=$rc n=$n $(date -Iseconds)" >> "$qlog"
    [ "$rc" -eq 0 ] && return 0
    sleep 60
  done
  return 1
}

run_sqa() {  # run_sqa <model> <rule>
  model=$1; rule=$2; tag=${model}_sqa_${rule}
  run_job "$tag" .venv/bin/python3 "$S/calib_baselines_decode_20261013b.py" --rule "$rule" --backend "$model" --cuda $CUDA \
    --output "$O/$tag.jsonl" --progress "$O/$tag.progress.json"
}

run_m3cot() {  # run_m3cot <model> <rule>
  model=$1; rule=$2; tag=${model}_m3cot_${rule}
  run_job "$tag" .venv/bin/python3 "$S/calib_baselines_decode_m3cot_20261013b.py" --rule "$rule" --backend "$model" --cuda $CUDA \
    --mode calib --prior-file "$ROOT/outputs/m3cot_20261012/cf_${model}.jsonl" \
    --output "$O/$tag.jsonl" --progress "$O/$tag.progress.json"
}

chain() {  # chain <model>
  model=$1
  run_sqa "$model" atabuzzaman
  run_m3cot "$model" atabuzzaman
  run_sqa "$model" kumar
  echo "CHAIN END $model $(date -Iseconds)" >> "$qlog"
}

chain lavida &
c1=$!
sleep 120   # stagger the two model loads
chain mmada &
c2=$!
wait $c1 $c2
echo "ALL DECODES DONE $(date -Iseconds)" >> "$qlog"
.venv/bin/python3 "$S/baselines_analysis_20261013b.py" > "$L/analysis.log" 2>&1
echo "ANALYSIS rc=$? $(date -Iseconds)" >> "$qlog"
echo "QUEUE END $(date -Iseconds) outputs=$O tables=$O/tables.md summary=$O/summary.json" >> "$qlog"
