#!/bin/bash
# Prior-tie reruns: shift -alpha*log p_cf only when the option argmax is still the
# prior letter AND the softmax margin of the top two option letters is < 0.05.
# Frozen cells that already beat stock are not in this queue. These 12 are still <= stock.
# gpu0: LaViDa POPE, MMStar, MME; MMaDA MMStar, MME; Dream-VL MMBench, ScienceQA
# gpu1: Dream-VL MME, MMStar, M3CoT, V*; LLaDA-V POPE
# The three MME jobs are not stacked on one GPU: LaViDa MME and MMaDA MME on gpu0,
# Dream-VL MME on gpu1.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
PY=$ROOT/.venv/bin/python3
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HOME=$ROOT/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export URD_PROJECT=$ROOT
GPU=${1:?gpu}
LOG=$ROOT/logs/priortie_20261026/gpu${GPU}.log
mkdir -p "$ROOT/logs/priortie_20261026" \
  "$ROOT/outputs/priortie_20261013e" \
  "$ROOT/outputs/priortie_mme_20261022" \
  "$ROOT/outputs/priortie_pope_lladav_20261022" \
  "$ROOT/outputs/dreamvl_20261013g/priortie_20261026" \
  "$ROOT/outputs/dreamvl_priortie_20261022"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] QUEUE START gpu=$GPU rule=priortie tau=0.05"

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
print(f"RESULT {name} n={n} stock={sa} ({sa/n*100:.2f}%) priortie={sb} ({sb/n*100:.2f}%) delta={sb-sa} ({(sb-sa)/n*100:+.2f}pp)", flush=True)
PY
}

REL=(experiments/unmask_order/run_priortie_20261026.py)
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
  run smoke_lavida_pope "${REL[@]}" e --backend lavida --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 --limit 5 \
    --output outputs/priortie_20261013e/smoke_lavida_pope.jsonl \
    --progress outputs/priortie_20261013e/smoke_lavida_pope.progress.json
  run lavida_pope "${REL[@]}" e --backend lavida --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 \
    --output outputs/priortie_20261013e/lavida_pope_gated.jsonl \
    --progress outputs/priortie_20261013e/lavida_pope.progress.json
  score lavida_pope outputs/pope_20261013e/lavida_stock.jsonl outputs/priortie_20261013e/lavida_pope_gated.jsonl
  run lavida_mmstar "${REL[@]}" e --backend lavida --dataset mmstar --mode gated --alpha-rule spread \
    --prior-file outputs/mmstar_20261013e/cf_lavida.json --cuda 0 \
    --output outputs/priortie_20261013e/lavida_mmstar_gated.jsonl \
    --progress outputs/priortie_20261013e/lavida_mmstar.progress.json
  score lavida_mmstar outputs/mmstar_20261013e/lavida_stock.jsonl outputs/priortie_20261013e/lavida_mmstar_gated.jsonl
  run lavida_mme "${REL[@]}" vstar --backend lavida --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_lavida.json --cuda 0 \
    --output outputs/priortie_mme_20261022/lavida_mme_gated.jsonl \
    --progress outputs/priortie_mme_20261022/lavida_mme.progress.json
  score lavida_mme outputs/mme_20261022/lavida_stock.jsonl outputs/priortie_mme_20261022/lavida_mme_gated.jsonl
  run smoke_mmada_mmstar "${REL[@]}" e --backend mmada --dataset mmstar --mode gated --alpha-rule spread \
    --prior-file outputs/mmstar_20261013e/cf_mmada.json --cuda 0 --limit 5 \
    --output outputs/priortie_20261013e/smoke_mmada_mmstar.jsonl \
    --progress outputs/priortie_20261013e/smoke_mmada_mmstar.progress.json
  run mmada_mmstar "${REL[@]}" e --backend mmada --dataset mmstar --mode gated --alpha-rule spread \
    --prior-file outputs/mmstar_20261013e/cf_mmada.json --cuda 0 \
    --output outputs/priortie_20261013e/mmada_mmstar_gated.jsonl \
    --progress outputs/priortie_20261013e/mmada_mmstar.progress.json
  score mmada_mmstar outputs/mmstar_20261013e/mmada_stock.jsonl outputs/priortie_20261013e/mmada_mmstar_gated.jsonl
  run mmada_mme "${REL[@]}" vstar --backend mmada --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_mmada.json --cuda 0 \
    --output outputs/priortie_mme_20261022/mmada_mme_gated.jsonl \
    --progress outputs/priortie_mme_20261022/mmada_mme.progress.json
  score mmada_mme outputs/mme_20261022/mmada_stock.jsonl outputs/priortie_mme_20261022/mmada_mme_gated.jsonl
  run smoke_dream_mmbench "${REL[@]}" dream --dataset mmbench --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 --limit 5 \
    --output outputs/dreamvl_20261013g/priortie_20261026/smoke_mmbench.jsonl \
    --progress outputs/dreamvl_20261013g/priortie_20261026/smoke_mmbench.progress.json
  run dream_mmbench "${REL[@]}" dream --dataset mmbench --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 \
    --output outputs/dreamvl_20261013g/priortie_20261026/mmbench_gated.jsonl \
    --progress outputs/dreamvl_20261013g/priortie_20261026/mmbench.progress.json
  score dream_mmbench outputs/dreamvl_20261013g/mmbench_20261022/dreamvl_stock.jsonl outputs/dreamvl_20261013g/priortie_20261026/mmbench_gated.jsonl
  run dream_sciqa "${REL[@]}" dream --dataset sciqa --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 0 \
    --output outputs/dreamvl_20261013g/priortie_20261026/sciqa_gated.jsonl \
    --progress outputs/dreamvl_20261013g/priortie_20261026/sciqa.progress.json
  score dream_sciqa outputs/dreamvl_20261013g/sciqa_20261022/dreamvl_stock.jsonl outputs/dreamvl_20261013g/priortie_20261026/sciqa_gated.jsonl
elif [[ "$GPU" == "1" ]]; then
  need outputs/mme_20261022/cf_mme_dreamvl.json
  need outputs/mme_20261022/dreamvl_stock.jsonl
  need outputs/dreamvl_20261013g/cf_dreamvl.json
  need outputs/dreamvl_20261013g/mmstar_20261022/dreamvl_stock.jsonl
  need outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl
  need outputs/vstar_20261022/cf_vstar_dreamvl.json
  need outputs/vstar_20261022/dreamvl_stock.jsonl
  need outputs/pope_lladav_20261022/cf_pope_lladav.json
  need outputs/pope_lladav_20261022/lladav_stock.jsonl
  run smoke_dream_mme "${REL[@]}" dreamvm --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_dreamvl.json --cuda 1 --limit 5 \
    --output outputs/dreamvl_priortie_20261022/smoke_mme.jsonl \
    --progress outputs/dreamvl_priortie_20261022/smoke_mme.progress.json
  run smoke_lladav_pope "${REL[@]}" lladav --backend lladav --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_lladav_20261022/cf_pope_lladav.json --cuda 1 --limit 5 \
    --output outputs/priortie_pope_lladav_20261022/smoke_lladav_pope.jsonl \
    --progress outputs/priortie_pope_lladav_20261022/smoke_lladav_pope.progress.json
  run dream_mme "${REL[@]}" dreamvm --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_dreamvl.json --cuda 1 \
    --output outputs/dreamvl_priortie_20261022/mme_gated.jsonl \
    --progress outputs/dreamvl_priortie_20261022/mme.progress.json
  score dream_mme outputs/mme_20261022/dreamvl_stock.jsonl outputs/dreamvl_priortie_20261022/mme_gated.jsonl
  run dream_mmstar "${REL[@]}" dream --dataset mmstar --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 1 \
    --output outputs/dreamvl_20261013g/priortie_20261026/mmstar_gated.jsonl \
    --progress outputs/dreamvl_20261013g/priortie_20261026/mmstar.progress.json
  score dream_mmstar outputs/dreamvl_20261013g/mmstar_20261022/dreamvl_stock.jsonl outputs/dreamvl_20261013g/priortie_20261026/mmstar_gated.jsonl
  run dream_m3cot "${REL[@]}" dream --dataset m3cot --mode gated --alpha-rule spread \
    --prior-file outputs/dreamvl_20261013g/cf_dreamvl.json --cuda 1 \
    --output outputs/dreamvl_20261013g/priortie_20261026/m3cot_gated.jsonl \
    --progress outputs/dreamvl_20261013g/priortie_20261026/m3cot.progress.json
  score dream_m3cot outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl outputs/dreamvl_20261013g/priortie_20261026/m3cot_gated.jsonl
  run dream_vstar "${REL[@]}" dreamvm --dataset vstar --mode gated --alpha-rule spread \
    --prior-file outputs/vstar_20261022/cf_vstar_dreamvl.json --cuda 1 \
    --output outputs/dreamvl_priortie_20261022/vstar_gated.jsonl \
    --progress outputs/dreamvl_priortie_20261022/vstar.progress.json
  score dream_vstar outputs/vstar_20261022/dreamvl_stock.jsonl outputs/dreamvl_priortie_20261022/vstar_gated.jsonl
  run lladav_pope "${REL[@]}" lladav --backend lladav --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_lladav_20261022/cf_pope_lladav.json --cuda 1 \
    --output outputs/priortie_pope_lladav_20261022/lladav_pope_gated.jsonl \
    --progress outputs/priortie_pope_lladav_20261022/lladav_pope.progress.json
  score lladav_pope outputs/pope_lladav_20261022/lladav_stock.jsonl outputs/priortie_pope_lladav_20261022/lladav_pope_gated.jsonl
else
  echo "gpu must be 0 or 1"; exit 1
fi
echo "[$(date -Is)] QUEUE END gpu=$GPU"
