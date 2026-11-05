#!/usr/bin/env bash
# Dream-VL-7B smoke (2026-10-13g): content-free prior (real cf file) + 5-item ScienceQA decodes stock (with the official
# sampler on the same 5 items), gated alpha-scale 0 (ALPHA0_EQUALS_STOCK) and gated. Serial, one GPU job at a time, GPU
# picked by gpu_guest_20261013d.sh:pick_gpu (>= 30 GB free, else poll 5 min). Never touches other queues / outputs.
# Usage: cd $ROOT && SMOKE_TAG=smoke nohup bash experiments/unmask_order/smoke_dreamvl_20261013g.sh > /dev/null 2>&1 < /dev/null &
#   SMOKE_TAG (default smoke) names the output sub-directory outputs/dreamvl_20261013g/$SMOKE_TAG and the log prefix;
#   outputs are never deleted -- a re-run after a code fix uses a new tag (smoke2, ...).
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/dreamvl_20261013g
TAG=${SMOKE_TAG:-smoke}
SMK=$OUT/$TAG
LOGD=$ROOT/logs/dreamvl_20261013g
mkdir -p "$OUT" "$SMK" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
Q=$LOGD/${TAG}_queue.log
export GUEST_LOG=$Q
source "$S/gpu_guest_20261013d.sh"
PY=$ROOT/.venv/bin/python3
DEC=$S/calib_decode_dreamvl_20261013g.py
N=${N:-5}

run_job() {  # run_job NAME CMD...  (polls nvidia-smi for the GPU's peak used memory, all processes)
  local name=$1; shift
  local gpu; gpu=$(pick_gpu)
  log_gpu_state "$Q"
  echo "LAUNCH $name gpu=$gpu $(date -Iseconds)" >> "$Q"
  echo "CMD $* --cuda $gpu" >> "$Q"
  local t0; t0=$(date +%s)
  local before; before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  "$@" --cuda "$gpu" >> "$LOGD/${TAG}_$name.log" 2>&1 < /dev/null &
  local pid=$! peak=0 u
  while kill -0 "$pid" 2>/dev/null; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    [ "${u:-0}" -gt "$peak" ] && peak=$u
    sleep 5
  done
  wait "$pid"; local rc=$?
  echo "EXIT $name rc=$rc sec=$(( $(date +%s) - t0 )) gpu${gpu}_used_before=${before}MiB peak_used_all_procs=${peak}MiB $(grep -h 'PEAK_GPU_MEM' "$LOGD/${TAG}_$name.log" | tail -1) $(date -Iseconds)" >> "$Q"
  return $rc
}

echo "SMOKE START backend=dreamvl tag=$TAG n=$N $(date -Iseconds) pid=$$" >> "$Q"
# content-free prior for the MCQ sets (deterministic; the real file the queue will use; skipped if present)
[ -f "$OUT/cf_dreamvl.json" ] || run_job prior_mcq "$PY" "$DEC" --dataset sciqa --mode prior --output "$OUT/cf_dreamvl.json" || exit 1
# ScienceQA 5-item stock (+ official sampler on the same items)
run_job sciqa_stock "$PY" "$DEC" --dataset sciqa --mode stock --limit $N --official-check $N \
  --output "$SMK/smoke_sciqa_stock_n$N.jsonl" --progress "$SMK/progress_smoke_sciqa_stock_n$N.json" || exit 1
run_job sciqa_gated_a0 "$PY" "$DEC" --dataset sciqa --mode gated --alpha-scale 0 --prior-file "$OUT/cf_dreamvl.json" --limit $N \
  --output "$SMK/smoke_sciqa_gated_a0_n$N.jsonl" --progress "$SMK/progress_smoke_sciqa_gated_a0_n$N.json" || exit 1
run_job sciqa_gated "$PY" "$DEC" --dataset sciqa --mode gated --prior-file "$OUT/cf_dreamvl.json" --limit $N \
  --output "$SMK/smoke_sciqa_gated_n$N.jsonl" --progress "$SMK/progress_smoke_sciqa_gated_n$N.json" || exit 1
# sanity report (CPU)
"$PY" "$S/analysis_dreamvl_20261013g.py" --smoke --smoke-dir "$SMK" >> "$Q" 2>&1
echo "SMOKE END backend=dreamvl $(date -Iseconds)" >> "$Q"
