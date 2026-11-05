#!/bin/bash
# Push single-digit cells a bit above their current best.
# Unmask order stays stock. The answer token changes only to a letter that
# slot already wrote. A cell that does not beat its current best is left as-is
# in the reported table; this queue only writes new dated files.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
PY=$ROOT/.venv/bin/python3
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HOME=$ROOT/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export URD_PROJECT=$ROOT
GPU=${1:?gpu}
LOG=$ROOT/logs/bump1_20261030/gpu${GPU}.log
HY=$ROOT/experiments/unmask_order/run_bump_20261030.py
CHK=$ROOT/experiments/unmask_order/check_bump_20261030.py
mkdir -p "$ROOT/logs/bump1_20261030" \
  "$ROOT/outputs/bump1_20261030/mmbench_20261013c" \
  "$ROOT/outputs/bump1_20261030/mmstar_20261013e" \
  "$ROOT/outputs/bump1_20261030/vstar_20261022" \
  "$ROOT/outputs/dreamvl_20261013g/bump1_20261030"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] QUEUE START gpu=$GPU"

run() {
  local name=$1; shift
  echo "[$(date -Is)] LAUNCH $name"
  if ! "$@"; then
    echo "[$(date -Is)] FAIL $name rc=$?"
    exit 1
  fi
  echo "[$(date -Is)] EXIT $name rc=0"
}

score() {
  local name=$1 stock=$2 new=$3 best=$4
  echo "[$(date -Is)] SCORE $name"
  "$PY" - "$name" "$stock" "$new" "$best" << 'PY'
import json, sys
name, stock, new, best = sys.argv[1:]
best = int(best)
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
if n == 0 or len(b) != n:
    raise SystemExit("RESULT %s incomplete stock=%d new=%d overlap=%d" % (name, len(a), len(b), n))
print("RESULT %s n=%d stock=%d new=%d delta_stock=%+d current_best=%d delta_best=%+d" % (
    name, n, sa, sb, sb - sa, best, sb - best), flush=True)
PY
}

if [[ "$GPU" == "0" ]]; then
  export HYST_RULE=gate:6:99:99
  run smoke_lavida_mmbench "$PY" "$HY" c --backend lavida --dataset mmbench --mode stock --cuda 0 --limit 5 \
    --output "$ROOT/outputs/bump1_20261030/mmbench_20261013c/smoke_lavida.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/mmbench_20261013c/smoke_lavida.progress.json"
  run check_lavida_mmbench "$PY" "$CHK" "$HYST_RULE" \
    outputs/mmbench_20261013c/lavida_stock.jsonl \
    outputs/bump1_20261030/mmbench_20261013c/smoke_lavida.jsonl
  run lavida_mmbench "$PY" "$HY" c --backend lavida --dataset mmbench --mode stock --cuda 0 \
    --output "$ROOT/outputs/bump1_20261030/mmbench_20261013c/lavida.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/mmbench_20261013c/lavida.progress.json"
  score lavida_mmbench outputs/mmbench_20261013c/lavida_stock.jsonl \
    "$ROOT/outputs/bump1_20261030/mmbench_20261013c/lavida.jsonl" 3268

  export HYST_RULE=gate:1:3:0.125
  run smoke_lavida_mmstar "$PY" "$HY" e --backend lavida --dataset mmstar --mode stock --cuda 0 --limit 5 \
    --output "$ROOT/outputs/bump1_20261030/mmstar_20261013e/smoke_lavida.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/mmstar_20261013e/smoke_lavida.progress.json"
  run check_lavida_mmstar "$PY" "$CHK" "$HYST_RULE" \
    outputs/mmstar_20261013e/lavida_stock.jsonl \
    outputs/bump1_20261030/mmstar_20261013e/smoke_lavida.jsonl
  run lavida_mmstar "$PY" "$HY" e --backend lavida --dataset mmstar --mode stock --cuda 0 \
    --output "$ROOT/outputs/bump1_20261030/mmstar_20261013e/lavida.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/mmstar_20261013e/lavida.progress.json"
  score lavida_mmstar outputs/mmstar_20261013e/lavida_stock.jsonl \
    "$ROOT/outputs/bump1_20261030/mmstar_20261013e/lavida.jsonl" 681

  export HYST_RULE=gate:3:99:0.125
  run smoke_lavida_vstar "$PY" "$HY" vstar --backend lavida --dataset vstar --mode stock --cuda 0 --limit 5 \
    --output "$ROOT/outputs/bump1_20261030/vstar_20261022/smoke_lavida.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/vstar_20261022/smoke_lavida.progress.json"
  run check_lavida_vstar "$PY" "$CHK" "$HYST_RULE" \
    outputs/vstar_20261022/lavida_stock.jsonl \
    outputs/bump1_20261030/vstar_20261022/smoke_lavida.jsonl
  run lavida_vstar "$PY" "$HY" vstar --backend lavida --dataset vstar --mode stock --cuda 0 \
    --output "$ROOT/outputs/bump1_20261030/vstar_20261022/lavida.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/vstar_20261022/lavida.progress.json"
  score lavida_vstar outputs/vstar_20261022/lavida_stock.jsonl \
    "$ROOT/outputs/bump1_20261030/vstar_20261022/lavida.jsonl" 83

elif [[ "$GPU" == "1" ]]; then
  export HYST_RULE=early_m:1.01
  run smoke_dream_m3cot "$PY" "$HY" dream --dataset m3cot --mode stock --cuda 1 --limit 5 \
    --output "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/smoke_m3cot.jsonl" \
    --progress "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/smoke_m3cot.progress.json"
  run check_dream_m3cot "$PY" "$CHK" "$HYST_RULE" \
    outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl \
    outputs/dreamvl_20261013g/bump1_20261030/smoke_m3cot.jsonl
  run dream_m3cot "$PY" "$HY" dream --dataset m3cot --mode stock --cuda 1 \
    --output "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/m3cot.jsonl" \
    --progress "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/m3cot.progress.json"
  score dream_m3cot outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl \
    "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/m3cot.jsonl" 664

  export HYST_RULE=gate:4:99:0.2
  run smoke_dream_mmstar "$PY" "$HY" dream --dataset mmstar --mode stock --cuda 1 --limit 5 \
    --output "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/smoke_mmstar.jsonl" \
    --progress "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/smoke_mmstar.progress.json"
  run check_dream_mmstar "$PY" "$CHK" "$HYST_RULE" \
    outputs/dreamvl_20261013g/mmstar_20261022/dreamvl_stock.jsonl \
    outputs/dreamvl_20261013g/bump1_20261030/smoke_mmstar.jsonl
  run dream_mmstar "$PY" "$HY" dream --dataset mmstar --mode stock --cuda 1 \
    --output "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/mmstar.jsonl" \
    --progress "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/mmstar.progress.json"
  score dream_mmstar outputs/dreamvl_20261013g/mmstar_20261022/dreamvl_stock.jsonl \
    "$ROOT/outputs/dreamvl_20261013g/bump1_20261030/mmstar.jsonl" 876

  export HYST_RULE=gate:5:2:0.125
  run smoke_mmada_mmbench "$PY" "$HY" c --backend mmada --dataset mmbench --mode stock --cuda 1 --limit 5 \
    --output "$ROOT/outputs/bump1_20261030/mmbench_20261013c/smoke_mmada.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/mmbench_20261013c/smoke_mmada.progress.json"
  run check_mmada_mmbench "$PY" "$CHK" "$HYST_RULE" \
    outputs/mmbench_20261013c/mmada_stock.jsonl \
    outputs/bump1_20261030/mmbench_20261013c/smoke_mmada.jsonl
  run mmada_mmbench "$PY" "$HY" c --backend mmada --dataset mmbench --mode stock --cuda 1 \
    --output "$ROOT/outputs/bump1_20261030/mmbench_20261013c/mmada.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/mmbench_20261013c/mmada.progress.json"
  score mmada_mmbench outputs/mmbench_20261013c/mmada_stock.jsonl \
    "$ROOT/outputs/bump1_20261030/mmbench_20261013c/mmada.jsonl" 2123

  export HYST_RULE=gate:4:2:0.35
  run smoke_mmada_vstar "$PY" "$HY" vstar --backend mmada --dataset vstar --mode stock --cuda 1 --limit 5 \
    --output "$ROOT/outputs/bump1_20261030/vstar_20261022/smoke_mmada.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/vstar_20261022/smoke_mmada.progress.json"
  run check_mmada_vstar "$PY" "$CHK" "$HYST_RULE" \
    outputs/vstar_20261022/mmada_stock.jsonl \
    outputs/bump1_20261030/vstar_20261022/smoke_mmada.jsonl
  run mmada_vstar "$PY" "$HY" vstar --backend mmada --dataset vstar --mode stock --cuda 1 \
    --output "$ROOT/outputs/bump1_20261030/vstar_20261022/mmada.jsonl" \
    --progress "$ROOT/outputs/bump1_20261030/vstar_20261022/mmada.progress.json"
  score mmada_vstar outputs/vstar_20261022/mmada_stock.jsonl \
    "$ROOT/outputs/bump1_20261030/vstar_20261022/mmada.jsonl" 62
else
  echo "gpu must be 0 or 1"; exit 1
fi
echo "[$(date -Is)] QUEUE END gpu=$GPU"
