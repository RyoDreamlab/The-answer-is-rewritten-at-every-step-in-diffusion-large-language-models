#!/usr/bin/env bash
# Full stock letter-commit for the cells that were not in the 2026-10-16 supplement.
# Does not resume the paused E lanes and does not start gated decodes.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/gap_20261019
LOGD=$ROOT/logs/gap_20261019
PY=$ROOT/.venv/bin/python3
Q=$LOGD/queue.log
mkdir -p "$OUT" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT

echo "GAP START $(date -Iseconds) pid=$$" >> "$Q"

run() {
  local name=$1; shift
  echo "GAP LAUNCH $name $(date -Iseconds)" >> "$Q"
  if ! "$PY" "$@" >> "$LOGD/${name}.log" 2>&1; then
    echo "GAP FAIL $name $(date -Iseconds) retry" >> "$Q"
    "$PY" "$@" >> "$LOGD/${name}.log" 2>&1 || echo "GAP GIVEUP $name $(date -Iseconds)" >> "$Q"
  fi
  echo "GAP DONE $name $(date -Iseconds)" >> "$Q"
}

# GPU 0: LLaDA-V, the missing fourth model. ScienceQA and M3CoT stock already exist.
run lladav_mmbench "$S/commit_gap_20261019.py" --backend lladav --dataset mmbench --cuda 0 \
  --output "$OUT/lladav_mmbench_stock.jsonl" &
P0=$!
# GPU 1: Dream-VL on the three datasets it does not have, then POPE for MMaDA and LaViDa.
run dreamvl_mmbench "$S/commit_gap_20261019.py" --backend dreamvl --dataset mmbench --cuda 1 \
  --output "$OUT/dreamvl_mmbench_stock.jsonl"
run dreamvl_m3cot "$S/commit_gap_20261019.py" --backend dreamvl --dataset m3cot --cuda 1 \
  --output "$OUT/dreamvl_m3cot_stock.jsonl"
run dreamvl_pope "$S/commit_gap_20261019.py" --backend dreamvl --dataset pope --cuda 1 \
  --output "$OUT/dreamvl_pope_stock.jsonl"
run mmada_pope "$S/commit_gap_20261019.py" --backend mmada --dataset pope --cuda 1 \
  --output "$OUT/mmada_pope_stock.jsonl"
run lavida_pope "$S/commit_gap_20261019.py" --backend lavida --dataset pope --cuda 1 \
  --output "$OUT/lavida_pope_stock.jsonl"
wait "$P0"
echo "GAP lladav_mmbench exit $? $(date -Iseconds)" >> "$Q"
run lladav_mmstar "$S/commit_gap_20261019.py" --backend lladav --dataset mmstar --cuda 0 \
  --output "$OUT/lladav_mmstar_stock.jsonl"
run lladav_pope "$S/commit_gap_20261019.py" --backend lladav --dataset pope --cuda 0 \
  --output "$OUT/lladav_pope_stock.jsonl"
echo "GAP END $(date -Iseconds)" >> "$Q"
