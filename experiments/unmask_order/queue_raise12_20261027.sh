#!/bin/bash
# Raise the cells that are still at or below stock.
# Dream-VL MMStar (priortie +3) stays frozen, as do older cells that already beat stock.
#
# Two commit-step settings are the ones the saved unmask logits say are strictly up:
#   Dream-VL M3CoT: shift only the surviving write when margin < 0.05, alpha-scale 0.1
#   MMaDA MMStar:   shift only the surviving write when its argmax is still the prior, alpha-scale 2
# The other nine get every-step prior subtraction at alpha-scale 0.25 (full scale already failed).
#
# gpu0: LaViDa POPE, MMStar, MME; MMaDA MME; Dream-VL ScienceQA, MMBench
# gpu1: Dream-VL M3CoT commit, MMaDA MMStar commit, Dream-VL MME, Dream-VL V*, LLaDA-V POPE
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
PY=$ROOT/.venv/bin/python3
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HOME=$ROOT/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export URD_PROJECT=$ROOT
GPU=${1:?gpu}
LOG=$ROOT/logs/raise12_20261027/gpu${GPU}.log
EOUT=$ROOT/outputs/weakscale025_20261027/e_20261013e
MMEOUT=$ROOT/outputs/weakscale025_20261027/mme_20261022
LLADAVOUT=$ROOT/outputs/weakscale025_20261027/lladav_20261022
DREAMOUT=$ROOT/outputs/dreamvl_20261013g/weakscale025_20261027
DREAMVMOUT=$ROOT/outputs/weakscale025_20261027/dreamvm_20261022
COMMITOUT=$ROOT/outputs/dreamvl_20261013g/commitmargin_20261027
MMADAOUT=$ROOT/outputs/commitmargin_20261027/e_20261013e
mkdir -p "$ROOT/logs/raise12_20261027" "$EOUT" "$MMEOUT" "$LLADAVOUT" "$DREAMOUT" "$DREAMVMOUT" "$COMMITOUT" "$MMADAOUT"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] QUEUE START gpu=$GPU"
echo "[$(date -Is)] FROZEN dream_mmstar and every cell that already beat stock"

need() {
  if [[ ! -f $1 ]]; then
    echo "[$(date -Is)] MISSING $1"
    exit 1
  fi
}

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
  local name=$1 stock=$2 new=$3 tag=$4
  echo "[$(date -Is)] SCORE $name"
  "$PY" - "$name" "$stock" "$new" "$tag" << 'PY'
import json, sys
name, stock, new, tag = sys.argv[1:]
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
    raise SystemExit("RESULT %s n=0 empty intersection" % name)
print("RESULT %s n=%d stock=%d (%.2f%%) %s=%d (%.2f%%) delta=%d (%+.2fpp)" % (
    name, n, sa, sa / n * 100, tag, sb, sb / n * 100, sb - sa, (sb - sa) / n * 100), flush=True)
PY
}

E=experiments/unmask_order/calib_decode_multi_20261013e.py
V=experiments/unmask_order/calib_decode_vstar_mme_20261022.py
D=experiments/unmask_order/calib_decode_dreamvl_20261013g.py
DV=experiments/unmask_order/calib_decode_vstar_mme_dreamvl_20261022.py
L=experiments/unmask_order/calib_decode_lladav_e_20261022.py
C=experiments/unmask_order/run_commitmargin_20261027.py

if [[ "$GPU" == "0" ]]; then
  need outputs/pope_20261013e/cf_pope_lavida.json
  need outputs/pope_20261013e/lavida_stock.jsonl
  need outputs/mmstar_20261013e/cf_lavida.json
  need outputs/mmstar_20261013e/lavida_stock.jsonl
  need outputs/mme_20261022/cf_mme_lavida.json
  need outputs/mme_20261022/lavida_stock.jsonl
  need outputs/mme_20261022/cf_mme_mmada.json
  need outputs/mme_20261022/mmada_stock.jsonl
  need outputs/dreamvl_20261013g/cf_dreamvl.json
  need outputs/dreamvl_20261013g/sciqa_20261022/dreamvl_stock.jsonl
  need outputs/dreamvl_20261013g/mmbench_20261022/dreamvl_stock.jsonl
  run smoke_lavida_pope "$E" --backend lavida --dataset pope --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 --limit 5 \
    --output "$EOUT/smoke_lavida_pope.jsonl" --progress "$EOUT/smoke_lavida_pope.progress.json"
  echo "[$(date -Is)] SMOKE PASS smoke_lavida_pope"
  run lavida_pope "$E" --backend lavida --dataset pope --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 \
    --output "$EOUT/lavida_pope_gated.jsonl" --progress "$EOUT/lavida_pope.progress.json"
  score lavida_pope outputs/pope_20261013e/lavida_stock.jsonl "$EOUT/lavida_pope_gated.jsonl" weak025
  run lavida_mmstar "$E" --backend lavida --dataset mmstar --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/mmstar_20261013e/cf_lavida.json --cuda 0 \
    --output "$EOUT/lavida_mmstar_gated.jsonl" --progress "$EOUT/lavida_mmstar.progress.json"
  score lavida_mmstar outputs/mmstar_20261013e/lavida_stock.jsonl "$EOUT/lavida_mmstar_gated.jsonl" weak025
  run lavida_mme "$V" --backend lavida --dataset mme --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/mme_20261022/cf_mme_lavida.json --cuda 0 \
    --output "$MMEOUT/lavida_mme_gated.jsonl" --progress "$MMEOUT/lavida_mme.progress.json"
  score lavida_mme outputs/mme_20261022/lavida_stock.jsonl "$MMEOUT/lavida_mme_gated.jsonl" weak025
  run mmada_mme "$V" --backend mmada --dataset mme --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/mme_20261022/cf_mme_mmada.json --cuda 0 \
    --output "$MMEOUT/mmada_mme_gated.jsonl" --progress "$MMEOUT/mmada_mme.progress.json"
  score mmada_mme outputs/mme_20261022/mmada_stock.jsonl "$MMEOUT/mmada_mme_gated.jsonl" weak025
  run dream_sciqa "$D" --dataset sciqa --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 \
    --output "$DREAMOUT/sciqa_gated.jsonl" --progress "$DREAMOUT/sciqa.progress.json"
  score dream_sciqa outputs/dreamvl_20261013g/sciqa_20261022/dreamvl_stock.jsonl "$DREAMOUT/sciqa_gated.jsonl" weak025
  run dream_mmbench "$D" --dataset mmbench --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 \
    --output "$DREAMOUT/mmbench_gated.jsonl" --progress "$DREAMOUT/mmbench.progress.json"
  score dream_mmbench outputs/dreamvl_20261013g/mmbench_20261022/dreamvl_stock.jsonl "$DREAMOUT/mmbench_gated.jsonl" weak025
elif [[ "$GPU" == "1" ]]; then
  need outputs/dreamvl_20261013g/cf_dreamvl.json
  need outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl
  need outputs/mmstar_20261013e/cf_mmada.json
  need outputs/mmstar_20261013e/mmada_stock.jsonl
  need outputs/mme_20261022/cf_mme_dreamvl.json
  need outputs/mme_20261022/dreamvl_stock.jsonl
  need outputs/vstar_20261022/cf_vstar_dreamvl.json
  need outputs/vstar_20261022/dreamvl_stock.jsonl
  need outputs/pope_lladav_20261022/cf_pope_lladav.json
  need outputs/pope_lladav_20261022/lladav_stock.jsonl
  export COMMIT_MODE=margin_only COMMIT_TAU=0.05
  run smoke_dream_m3cot "$C" dream \
    --dataset m3cot --mode gated --alpha-rule spread --alpha-scale 0.1 \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 1 --limit 5 \
    --output "$COMMITOUT/smoke_m3cot.jsonl" --progress "$COMMITOUT/smoke_m3cot.progress.json"
  echo "[$(date -Is)] SMOKE PASS smoke_dream_m3cot"
  run dream_m3cot "$C" dream \
    --dataset m3cot --mode gated --alpha-rule spread --alpha-scale 0.1 \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 1 \
    --output "$COMMITOUT/m3cot_gated.jsonl" --progress "$COMMITOUT/m3cot.progress.json"
  score dream_m3cot outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl "$COMMITOUT/m3cot_gated.jsonl" commit
  export COMMIT_MODE=prior_and_margin COMMIT_TAU=1.01
  run mmada_mmstar "$C" e \
    --backend mmada --dataset mmstar --mode gated --alpha-rule spread --alpha-scale 2 \
    --prior-file outputs/mmstar_20261013e/cf_mmada.json --cuda 1 \
    --output "$MMADAOUT/mmada_mmstar_gated.jsonl" --progress "$MMADAOUT/mmada_mmstar.progress.json"
  score mmada_mmstar outputs/mmstar_20261013e/mmada_stock.jsonl "$MMADAOUT/mmada_mmstar_gated.jsonl" commit
  run dream_mme "$DV" --dataset mme --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/mme_20261022/cf_mme_dreamvl.json --cuda 1 \
    --output "$DREAMVMOUT/mme_gated.jsonl" --progress "$DREAMVMOUT/mme.progress.json"
  score dream_mme outputs/mme_20261022/dreamvl_stock.jsonl "$DREAMVMOUT/mme_gated.jsonl" weak025
  run dream_vstar "$DV" --dataset vstar --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/vstar_20261022/cf_vstar_dreamvl.json --cuda 1 \
    --output "$DREAMVMOUT/vstar_gated.jsonl" --progress "$DREAMVMOUT/vstar.progress.json"
  score dream_vstar outputs/vstar_20261022/dreamvl_stock.jsonl "$DREAMVMOUT/vstar_gated.jsonl" weak025
  run lladav_pope "$L" --backend lladav --dataset pope --mode gated --alpha-rule spread --alpha-scale 0.25 \
    --prior-file outputs/pope_lladav_20261022/cf_pope_lladav.json --cuda 1 \
    --output "$LLADAVOUT/lladav_pope_gated.jsonl" --progress "$LLADAVOUT/lladav_pope.progress.json"
  score lladav_pope outputs/pope_lladav_20261022/lladav_stock.jsonl "$LLADAVOUT/lladav_pope_gated.jsonl" weak025
else
  echo "gpu must be 0 or 1"; exit 1
fi
echo "[$(date -Is)] QUEUE END gpu=$GPU"
