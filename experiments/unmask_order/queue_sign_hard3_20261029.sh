#!/bin/bash
# The three cells whose answer slot never wrote the correct letter.
# Small sign-flipped every-step prior on the free GPU. New files only.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
PY=$ROOT/.venv/bin/python3
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HOME=$ROOT/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export URD_PROJECT=$ROOT
LOG=$ROOT/logs/hyst9_20261029/gpu0_sign.log
V=experiments/unmask_order/calib_decode_vstar_mme_20261022.py
DV=experiments/unmask_order/calib_decode_vstar_mme_dreamvl_20261022.py
MMEOUT=$ROOT/outputs/sign010_now_20261029/mme_20261022
DREAMOUT=$ROOT/outputs/sign010_now_20261029/dreamvm_20261022
mkdir -p "$ROOT/logs/hyst9_20261029" "$MMEOUT" "$DREAMOUT"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] QUEUE START gpu=0 signflip=-0.1 hard3"

run() {
  local name=$1; shift
  echo "[$(date -Is)] LAUNCH $name"
  if ! "$PY" "$@"; then
    echo "[$(date -Is)] FAIL $name rc=$?"
    exit 1
  fi
  echo "[$(date -Is)] EXIT $name rc=0"
}

score() {
  local name=$1 stock=$2 new=$3
  echo "[$(date -Is)] SCORE $name"
  "$PY" - "$name" "$stock" "$new" << 'PY'
import json, sys
name, stock, new = sys.argv[1:]
def load(p):
    d = {}
    with open(p) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[r["id"]] = r.get("finished_letter") == r.get("gold")
    return d
a, b = load(stock), load(new)
ids = sorted(set(a) & set(b))
sa, sb = sum(a[i] for i in ids), sum(b[i] for i in ids)
n = len(ids)
if n == 0:
    raise SystemExit("RESULT %s n=0" % name)
print("RESULT %s n=%d stock=%d (%.2f%%) sign010=%d (%.2f%%) delta=%d (%+.2fpp)" % (
    name, n, sa, sa / n * 100, sb, sb / n * 100, sb - sa, (sb - sa) / n * 100), flush=True)
PY
}

run lavida_mme "$V" --backend lavida --dataset mme --mode gated --alpha-rule spread --alpha-scale=-0.1 \
  --prior-file outputs/mme_20261022/cf_mme_lavida.json --cuda 0 \
  --output "$MMEOUT/lavida_mme.jsonl" --progress "$MMEOUT/lavida_mme.progress.json"
score lavida_mme outputs/mme_20261022/lavida_stock.jsonl "$MMEOUT/lavida_mme.jsonl"

run mmada_mme "$V" --backend mmada --dataset mme --mode gated --alpha-rule spread --alpha-scale=-0.1 \
  --prior-file outputs/mme_20261022/cf_mme_mmada.json --cuda 0 \
  --output "$MMEOUT/mmada_mme.jsonl" --progress "$MMEOUT/mmada_mme.progress.json"
score mmada_mme outputs/mme_20261022/mmada_stock.jsonl "$MMEOUT/mmada_mme.jsonl"

run dream_vstar "$DV" --dataset vstar --mode gated --alpha-rule spread --alpha-scale=-0.1 \
  --prior-file outputs/vstar_20261022/cf_vstar_dreamvl.json --cuda 0 \
  --output "$DREAMOUT/dream_vstar.jsonl" --progress "$DREAMOUT/dream_vstar.progress.json"
score dream_vstar outputs/vstar_20261022/dreamvl_stock.jsonl "$DREAMOUT/dream_vstar.jsonl"
echo "[$(date -Is)] QUEUE END"
