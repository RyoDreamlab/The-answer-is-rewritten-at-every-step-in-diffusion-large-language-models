#!/bin/bash
# Resume only unfinished prior-hold cells. Completed cells are not relaunched.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
PY=$ROOT/.venv/bin/python3
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HOME=$ROOT/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export URD_PROJECT=$ROOT
GPU=${1:?gpu}
LOG=$ROOT/logs/priorhold_20261026/resume_gpu${GPU}.log
mkdir -p "$ROOT/logs/priorhold_20261026" "$ROOT/outputs/priorhold_pope_lladav_20261022"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] RESUME START gpu=$GPU"
REL=(experiments/unmask_order/run_priorhold_20261026.py)
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
print(f"RESULT {name} n={n} stock={sa} ({sa/n*100:.2f}%) priorhold={sb} ({sb/n*100:.2f}%) delta={sb-sa} ({(sb-sa)/n*100:+.2f}pp)", flush=True)
PY
}
if [[ "$GPU" == "0" ]]; then
  run dream_mme "${REL[@]}" dreamvm --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_dreamvl.json --cuda 0 \
    --output outputs/dreamvl_priorhold_20261022/mme_gated.jsonl \
    --progress outputs/dreamvl_priorhold_20261022/mme.progress.json
  score dream_mme outputs/mme_20261022/dreamvl_stock.jsonl outputs/dreamvl_priorhold_20261022/mme_gated.jsonl
elif [[ "$GPU" == "1" ]]; then
  run dream_vstar "${REL[@]}" dreamvm --dataset vstar --mode gated --alpha-rule spread \
    --prior-file outputs/vstar_20261022/cf_vstar_dreamvl.json --cuda 1 \
    --output outputs/dreamvl_priorhold_20261022/vstar_gated.jsonl \
    --progress outputs/dreamvl_priorhold_20261022/vstar.progress.json
  score dream_vstar outputs/vstar_20261022/dreamvl_stock.jsonl outputs/dreamvl_priorhold_20261022/vstar_gated.jsonl
  run lladav_pope "${REL[@]}" lladav --backend lladav --dataset pope --mode gated --alpha-rule spread \
    --prior-file outputs/pope_lladav_20261022/cf_pope_lladav.json --cuda 1 \
    --output outputs/priorhold_pope_lladav_20261022/lladav_pope_gated.jsonl \
    --progress outputs/priorhold_pope_lladav_20261022/lladav_pope.progress.json
  score lladav_pope outputs/pope_lladav_20261022/lladav_stock.jsonl outputs/priorhold_pope_lladav_20261022/lladav_pope_gated.jsonl
else
  echo "gpu must be 0 or 1"; exit 1
fi
echo "[$(date -Is)] RESUME END gpu=$GPU"
