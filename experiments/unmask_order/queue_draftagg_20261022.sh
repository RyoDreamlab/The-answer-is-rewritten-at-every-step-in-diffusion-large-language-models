#!/usr/bin/env bash
# Draft aggregation (draftagg_20261022.py) lanes.
#   gpu0: lavida 2017 -> dreamvl 2017 -> mmada 2017 (full runs, resume-safe)
#   gpu1: lladav pilot n=200 (all conds); if any agg cond has delta >= 0 vs stock, continue the same dir to 2017.
# Usage: bash queue_draftagg_20261022.sh gpu0|gpu1
set -u
cd /mnt/pfs/peng_gao/visual-hysteresis || exit 1
export PYTHONUNBUFFERED=1 HF_HOME=$PWD/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=.venv/bin/python3
SCRIPT=experiments/unmask_order/draftagg_20261022.py
LANE=${1:?lane gpu0|gpu1}
echo "QUEUE $LANE start $(date)"
if [ "$LANE" = "gpu0" ]; then
  for B in lavida dreamvl mmada; do
    echo "QUEUE $B start $(date)"
    $PY $SCRIPT --backend $B --cuda 0 --n 2017
    echo "QUEUE $B rc=$? $(date)"
  done
elif [ "$LANE" = "gpu1" ]; then
  OUT=outputs/draftagg_20261022/lladav
  $PY $SCRIPT --backend lladav --cuda 1 --n 200 --out-dir "$OUT"
  rc=$?
  echo "QUEUE lladav pilot rc=$rc $(date)"
  if [ "$rc" -ne 0 ]; then echo "QUEUE lladav pilot failed; stop"; exit "$rc"; fi
  if $PY $SCRIPT --mode gate --backend lladav --out-dir "$OUT"; then
    echo "QUEUE lladav gate passed; continue to 2017 $(date)"
    $PY $SCRIPT --backend lladav --cuda 1 --n 2017 --out-dir "$OUT"
    echo "QUEUE lladav full rc=$? $(date)"
  else
    echo "QUEUE lladav gate failed (no agg cond with delta >= 0 on the 200 pilot); lane stopped $(date)"
  fi
else
  echo "unknown lane $LANE"; exit 2
fi
echo "QUEUE $LANE end $(date)"
