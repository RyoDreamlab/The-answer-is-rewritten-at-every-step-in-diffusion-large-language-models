#!/bin/bash
# Prior-tie 0.10 reruns. New files only (suffix priortie10_20261026).
# Shift -alpha*log p_cf only when BOTH: option argmax is still the prior letter
# (argmax of p_cf) AND the top-two softmax margin is < 0.10. The unmask step is
# included. A confident write stays unchanged.
# Frozen: Dream-VL MMStar (priortie +3) and every older cell that already beat stock.
# These 11 are still not strictly above stock.
# gpu0: LaViDa POPE, MMStar, MME; MMaDA MMStar, MME; Dream-VL MMBench, ScienceQA, M3CoT
# gpu1: Dream-VL MME, V*; LLaDA-V POPE
# MME is split across GPUs: LaViDa MME and MMaDA MME on gpu0, Dream-VL MME on gpu1.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
PY=$ROOT/.venv/bin/python3
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HOME=$ROOT/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export URD_PROJECT=$ROOT
GPU=${1:?gpu}
LOG=$ROOT/logs/priortie10_20261026/gpu${GPU}.log
EOUT=$ROOT/outputs/priortie10_20261026/e_20261013e
MMEOUT=$ROOT/outputs/priortie10_20261026/mme_20261022
LLADAVOUT=$ROOT/outputs/priortie10_20261026/lladav_20261022
DREAMOUT=$ROOT/outputs/dreamvl_20261013g/priortie10_20261026
DREAMVMOUT=$ROOT/outputs/priortie10_20261026/dreamvm_20261022
mkdir -p "$ROOT/logs/priortie10_20261026" "$EOUT" "$MMEOUT" "$LLADAVOUT" "$DREAMOUT" "$DREAMVMOUT"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] QUEUE START gpu=$GPU rule=priortie10 tau=0.10"
echo "[$(date -Is)] FROZEN dream_mmstar and every cell that already beat stock; not in this queue"

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
    raise SystemExit("RESULT %s n=0 empty intersection" % name)
print("RESULT %s n=%d stock=%d (%.2f%%) priortie10=%d (%.2f%%) delta=%d (%+.2fpp)" % (
    name, n, sa, sa / n * 100, sb, sb / n * 100, sb - sa, (sb - sa) / n * 100), flush=True)
PY
}

REL=(experiments/unmask_order/run_priortie10_20261026.py)
if [[ "$GPU" == "0" ]]; then
  need outputs/pope_20261013e/cf_pope_lavida.json
  need outputs/pope_20261013e/lavida_stock.jsonl
  need outputs/mmstar_20261013e/cf_lavida.json
  need outputs/mmstar_20261013e/lavida_stock.jsonl
  need outputs/mme_20261022/cf_mme_lavida.json
  need outputs/mme_20261022/lavida_stock.jsonl
  need outputs/mmstar_20261013e/cf_mmada.json
  need outputs/mmstar_20261013e/mmada_stock.jsonl
  need outputs/mme_20261022/cf_mme_mmada.json
  need outputs/mme_20261022/mmada_stock.jsonl
  need outputs/dreamvl_20261013g/cf_dreamvl.json
  need outputs/dreamvl_20261013g/mmbench_20261022/dreamvl_stock.jsonl
  need outputs/dreamvl_20261013g/sciqa_20261022/dreamvl_stock.jsonl
  need outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl
  run smoke_lavida_pope "${REL[@]}" e --backend lavida --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 --limit 5 \
    --output "$EOUT/smoke_lavida_pope.jsonl" \
    --progress "$EOUT/smoke_lavida_pope.progress.json"
  echo "[$(date -Is)] SMOKE PASS smoke_lavida_pope"
  run smoke_mmada_mmstar "${REL[@]}" e --backend mmada --dataset mmstar --mode gated --alpha-rule spread \
    --prior-file outputs/mmstar_20261013e/cf_mmada.json --cuda 0 --limit 5 \
    --output "$EOUT/smoke_mmada_mmstar.jsonl" \
    --progress "$EOUT/smoke_mmada_mmstar.progress.json"
  echo "[$(date -Is)] SMOKE PASS smoke_mmada_mmstar"
  run smoke_dream_mmbench "${REL[@]}" dream --dataset mmbench --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 --limit 5 \
    --output "$DREAMOUT/smoke_mmbench.jsonl" \
    --progress "$DREAMOUT/smoke_mmbench.progress.json"
  echo "[$(date -Is)] SMOKE PASS smoke_dream_mmbench"
  run lavida_pope "${REL[@]}" e --backend lavida --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 \
    --output "$EOUT/lavida_pope_gated.jsonl" \
    --progress "$EOUT/lavida_pope.progress.json"
  score lavida_pope outputs/pope_20261013e/lavida_stock.jsonl "$EOUT/lavida_pope_gated.jsonl"
  run lavida_mmstar "${REL[@]}" e --backend lavida --dataset mmstar --mode gated --alpha-rule spread \
    --prior-file outputs/mmstar_20261013e/cf_lavida.json --cuda 0 \
    --output "$EOUT/lavida_mmstar_gated.jsonl" \
    --progress "$EOUT/lavida_mmstar.progress.json"
  score lavida_mmstar outputs/mmstar_20261013e/lavida_stock.jsonl "$EOUT/lavida_mmstar_gated.jsonl"
  run lavida_mme "${REL[@]}" vstar --backend lavida --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_lavida.json --cuda 0 \
    --output "$MMEOUT/lavida_mme_gated.jsonl" \
    --progress "$MMEOUT/lavida_mme.progress.json"
  score lavida_mme outputs/mme_20261022/lavida_stock.jsonl "$MMEOUT/lavida_mme_gated.jsonl"
  run mmada_mmstar "${REL[@]}" e --backend mmada --dataset mmstar --mode gated --alpha-rule spread \
    --prior-file outputs/mmstar_20261013e/cf_mmada.json --cuda 0 \
    --output "$EOUT/mmada_mmstar_gated.jsonl" \
    --progress "$EOUT/mmada_mmstar.progress.json"
  score mmada_mmstar outputs/mmstar_20261013e/mmada_stock.jsonl "$EOUT/mmada_mmstar_gated.jsonl"
  run mmada_mme "${REL[@]}" vstar --backend mmada --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_mmada.json --cuda 0 \
    --output "$MMEOUT/mmada_mme_gated.jsonl" \
    --progress "$MMEOUT/mmada_mme.progress.json"
  score mmada_mme outputs/mme_20261022/mmada_stock.jsonl "$MMEOUT/mmada_mme_gated.jsonl"
  run dream_mmbench "${REL[@]}" dream --dataset mmbench --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 \
    --output "$DREAMOUT/mmbench_gated.jsonl" \
    --progress "$DREAMOUT/mmbench.progress.json"
  score dream_mmbench outputs/dreamvl_20261013g/mmbench_20261022/dreamvl_stock.jsonl "$DREAMOUT/mmbench_gated.jsonl"
  run dream_sciqa "${REL[@]}" dream --dataset sciqa --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 \
    --output "$DREAMOUT/sciqa_gated.jsonl" \
    --progress "$DREAMOUT/sciqa.progress.json"
  score dream_sciqa outputs/dreamvl_20261013g/sciqa_20261022/dreamvl_stock.jsonl "$DREAMOUT/sciqa_gated.jsonl"
  run dream_m3cot "${REL[@]}" dream --dataset m3cot --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 \
    --output "$DREAMOUT/m3cot_gated.jsonl" \
    --progress "$DREAMOUT/m3cot.progress.json"
  score dream_m3cot outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl "$DREAMOUT/m3cot_gated.jsonl"
elif [[ "$GPU" == "1" ]]; then
  need outputs/mme_20261022/cf_mme_dreamvl.json
  need outputs/mme_20261022/dreamvl_stock.jsonl
  need outputs/vstar_20261022/cf_vstar_dreamvl.json
  need outputs/vstar_20261022/dreamvl_stock.jsonl
  need outputs/pope_lladav_20261022/cf_pope_lladav.json
  need outputs/pope_lladav_20261022/lladav_stock.jsonl
  run smoke_dream_mme "${REL[@]}" dreamvm --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_dreamvl.json --cuda 1 --limit 5 \
    --output "$DREAMVMOUT/smoke_mme.jsonl" \
    --progress "$DREAMVMOUT/smoke_mme.progress.json"
  echo "[$(date -Is)] SMOKE PASS smoke_dream_mme"
  run smoke_lladav_pope "${REL[@]}" lladav --backend lladav --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_lladav_20261022/cf_pope_lladav.json --cuda 1 --limit 5 \
    --output "$LLADAVOUT/smoke_lladav_pope.jsonl" \
    --progress "$LLADAVOUT/smoke_lladav_pope.progress.json"
  echo "[$(date -Is)] SMOKE PASS smoke_lladav_pope"
  run dream_mme "${REL[@]}" dreamvm --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_dreamvl.json --cuda 1 \
    --output "$DREAMVMOUT/mme_gated.jsonl" \
    --progress "$DREAMVMOUT/mme.progress.json"
  score dream_mme outputs/mme_20261022/dreamvl_stock.jsonl "$DREAMVMOUT/mme_gated.jsonl"
  run dream_vstar "${REL[@]}" dreamvm --dataset vstar --mode gated --alpha-rule spread \
    --prior-file outputs/vstar_20261022/cf_vstar_dreamvl.json --cuda 1 \
    --output "$DREAMVMOUT/vstar_gated.jsonl" \
    --progress "$DREAMVMOUT/vstar.progress.json"
  score dream_vstar outputs/vstar_20261022/dreamvl_stock.jsonl "$DREAMVMOUT/vstar_gated.jsonl"
  run lladav_pope "${REL[@]}" lladav --backend lladav --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_lladav_20261022/cf_pope_lladav.json --cuda 1 \
    --output "$LLADAVOUT/lladav_pope_gated.jsonl" \
    --progress "$LLADAVOUT/lladav_pope.progress.json"
  score lladav_pope outputs/pope_lladav_20261022/lladav_stock.jsonl "$LLADAVOUT/lladav_pope_gated.jsonl"
else
  echo "gpu must be 0 or 1"; exit 1
fi
echo "[$(date -Is)] QUEUE END gpu=$GPU"
