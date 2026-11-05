#!/usr/bin/env bash
# Follow-on CVPR decodes. Waits until the 100/200 supplement queue prints QUEUE END, then uses both GPUs.
# Does not resume the paused E lanes and does not start the Dream-VL gated accuracy queue.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/supplement_20261016
LOGD=$ROOT/logs/supplement_20261016
PY=$ROOT/.venv/bin/python3
Q=$LOGD/queue.log
mkdir -p "$OUT" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT

echo "REST WAIT $(date -Iseconds) pid=$$" >> "$Q"
while ! grep -q "QUEUE END" "$Q"; do
  sleep 30
done
echo "REST START $(date -Iseconds)" >> "$Q"

run() {
  local name=$1; shift
  echo "REST LAUNCH $name $(date -Iseconds)" >> "$Q"
  if ! "$PY" "$@" >> "$LOGD/${name}.log" 2>&1; then
    echo "REST FAIL $name $(date -Iseconds) retry" >> "$Q"
    "$PY" "$@" >> "$LOGD/${name}.log" 2>&1 || echo "REST GIVEUP $name $(date -Iseconds)" >> "$Q"
  fi
  echo "REST DONE $name $(date -Iseconds)" >> "$Q"
}

# GPU 0: widen the factorial to the P6 400, then the full MMStar causal/image table.
run factor400 "$S/supplement_20261016.py" --part factor --cuda 0 --n-sciqa 400 --output "$OUT/factor.jsonl" &
P0=$!
# GPU 1: Dream-VL letter-commit (the 100-item run crashed before any record), then LaViDa MMStar,
# then prompt wording and the commit-step image swap.
run dreamvl_sciqa "$S/dreamvl_commit_20261016.py" --cuda 1 --dataset sciqa --limit 0 \
  --output "$OUT/dreamvl_sciqa_stock.jsonl"
run dreamvl_mmstar "$S/dreamvl_commit_20261016.py" --cuda 1 --dataset mmstar --limit 0 \
  --output "$OUT/dreamvl_mmstar_stock.jsonl"
run lavida_mmstar "$S/cvpr_rest_20261016.py" --part lavida --cuda 1 --output "$OUT/lavida_mmstar_stock.jsonl"
run prompt "$S/cvpr_rest_20261016.py" --part prompt --cuda 1 --n-sciqa 400 --output "$OUT/prompt.jsonl"
run commit "$S/cvpr_rest_20261016.py" --part commit --cuda 1 --n-sciqa 400 --output "$OUT/commit_swap.jsonl"
wait "$P0"
echo "REST factor400 exit $? $(date -Iseconds)" >> "$Q"
run mmstar_full "$S/cvpr_rest_20261016.py" --part mmstar --cuda 0 --output "$OUT/mmstar_full.jsonl"
echo "REST END $(date -Iseconds)" >> "$Q"
