#!/bin/bash
# Nine cells still at or below stock.
# Six use the letter the answer slot already wrote on an earlier step.
# Three never wrote the right letter into the slot; those get a small sign-flipped
# every-step prior after the six, and only keep going while the cell is not up.
# Dream-VL MMStar, Dream-VL M3CoT, and MMaDA MMStar are already up and are not here.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
PY=$ROOT/.venv/bin/python3
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HOME=$ROOT/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export URD_PROJECT=$ROOT
GPU=${1:?gpu}
LOG=$ROOT/logs/hyst9_20261029/gpu${GPU}.log
HY=$ROOT/experiments/unmask_order/run_hyst_20261029.py
E=experiments/unmask_order/calib_decode_multi_20261013e.py
V=experiments/unmask_order/calib_decode_vstar_mme_20261022.py
DV=experiments/unmask_order/calib_decode_vstar_mme_dreamvl_20261022.py
mkdir -p "$ROOT/logs/hyst9_20261029" \
  "$ROOT/outputs/hyst_20261029/e_20261013e" \
  "$ROOT/outputs/dreamvl_20261013g/hyst_20261029" \
  "$ROOT/outputs/hyst_20261029/dreamvm_20261022" \
  "$ROOT/outputs/hyst_20261029/lladav_20261022" \
  "$ROOT/outputs/sign010_20261029/mme_20261022" \
  "$ROOT/outputs/sign010_20261029/dreamvm_20261022" \
  "$ROOT/outputs/weak005_20261029/mme_20261022" \
  "$ROOT/outputs/weak005_20261029/dreamvm_20261022" \
  "$ROOT/outputs/sign005_20261029/mme_20261022" \
  "$ROOT/outputs/sign005_20261029/dreamvm_20261022"
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
    raise SystemExit("RESULT %s n=0 empty" % name)
delta = sb - sa
print("RESULT %s n=%d stock=%d (%.2f%%) %s=%d (%.2f%%) delta=%d (%+.2fpp)" % (
    name, n, sa, sa / n * 100, tag, sb, sb / n * 100, delta, delta / n * 100), flush=True)
open("/tmp/hyst_last_delta_%s" % name, "w").write(str(delta))
PY
}

# scale out_parent script args...
sign_cell() {
  local name=$1 stock=$2 script=$3
  shift 3
  local base_args=("$@")
  local scales=("-0.1:sign010" "0.05:weak005" "-0.05:sign005")
  local s tag parent out
  for spec in "${scales[@]}"; do
    s=${spec%%:*}; tag=${spec##*:}
    if [[ "$name" == dream_vstar || "$name" == dream_mme ]]; then
      parent=$ROOT/outputs/${tag}_20261029/dreamvm_20261022
    else
      parent=$ROOT/outputs/${tag}_20261029/mme_20261022
    fi
    mkdir -p "$parent"
    out=$parent/${name}.jsonl
    run "$name"_"$tag" "$PY" "$script" "${base_args[@]}" --alpha-scale="$s" --cuda "$GPU" \
      --output "$out" --progress "$parent/${name}.progress.json"
    score "$name" "$stock" "$out" "$tag"
    local delta
    delta=$(cat /tmp/hyst_last_delta_$name)
    if [[ "$delta" -gt 0 ]]; then
      echo "[$(date -Is)] UP $name delta=$delta"
      break
    fi
  done
}

if [[ "$GPU" == "0" ]]; then
  export HYST_RULE=back:8
  run smoke_lavida_pope "$PY" "$HY" e --backend lavida --dataset pope --mode stock --cuda 0 --limit 5 \
    --output "$ROOT/outputs/hyst_20261029/e_20261013e/smoke_lavida_pope.jsonl" \
    --progress "$ROOT/outputs/hyst_20261029/e_20261013e/smoke_lavida_pope.progress.json"
  run check_lavida_pope "$PY" "$ROOT/experiments/unmask_order/check_hyst_20261029.py" \
    back:8 outputs/pope_20261013e/lavida_stock.jsonl \
    outputs/hyst_20261029/e_20261013e/smoke_lavida_pope.jsonl
  run lavida_pope "$PY" "$HY" e --backend lavida --dataset pope --mode stock --cuda 0 \
    --output "$ROOT/outputs/hyst_20261029/e_20261013e/lavida_pope.jsonl" \
    --progress "$ROOT/outputs/hyst_20261029/e_20261013e/lavida_pope.progress.json"
  score lavida_pope outputs/pope_20261013e/lavida_stock.jsonl \
    "$ROOT/outputs/hyst_20261029/e_20261013e/lavida_pope.jsonl" hyst

  export HYST_RULE=back:2
  run lavida_mmstar "$PY" "$HY" e --backend lavida --dataset mmstar --mode stock --cuda 0 \
    --output "$ROOT/outputs/hyst_20261029/e_20261013e/lavida_mmstar.jsonl" \
    --progress "$ROOT/outputs/hyst_20261029/e_20261013e/lavida_mmstar.progress.json"
  score lavida_mmstar outputs/mmstar_20261013e/lavida_stock.jsonl \
    "$ROOT/outputs/hyst_20261029/e_20261013e/lavida_mmstar.jsonl" hyst

  export HYST_RULE=early_m:0.2
  run dream_sciqa "$PY" "$HY" dream --dataset sciqa --mode stock --cuda 0 \
    --output "$ROOT/outputs/dreamvl_20261013g/hyst_20261029/sciqa.jsonl" \
    --progress "$ROOT/outputs/dreamvl_20261013g/hyst_20261029/sciqa.progress.json"
  score dream_sciqa outputs/dreamvl_20261013g/sciqa_20261022/dreamvl_stock.jsonl \
    "$ROOT/outputs/dreamvl_20261013g/hyst_20261029/sciqa.jsonl" hyst
  run dream_mmbench "$PY" "$HY" dream --dataset mmbench --mode stock --cuda 0 \
    --output "$ROOT/outputs/dreamvl_20261013g/hyst_20261029/mmbench.jsonl" \
    --progress "$ROOT/outputs/dreamvl_20261013g/hyst_20261029/mmbench.progress.json"
  score dream_mmbench outputs/dreamvl_20261013g/mmbench_20261022/dreamvl_stock.jsonl \
    "$ROOT/outputs/dreamvl_20261013g/hyst_20261029/mmbench.jsonl" hyst

elif [[ "$GPU" == "1" ]]; then
  export HYST_RULE=run:6
  run smoke_dream_mme "$PY" "$HY" dreamvm --dataset mme --mode stock --cuda 1 --limit 5 \
    --output "$ROOT/outputs/hyst_20261029/dreamvm_20261022/smoke_mme.jsonl" \
    --progress "$ROOT/outputs/hyst_20261029/dreamvm_20261022/smoke_mme.progress.json"
  run check_dream_mme "$PY" "$ROOT/experiments/unmask_order/check_hyst_20261029.py" \
    run:6 outputs/mme_20261022/dreamvl_stock.jsonl \
    outputs/hyst_20261029/dreamvm_20261022/smoke_mme.jsonl
  run dream_mme "$PY" "$HY" dreamvm --dataset mme --mode stock --cuda 1 \
    --output "$ROOT/outputs/hyst_20261029/dreamvm_20261022/mme.jsonl" \
    --progress "$ROOT/outputs/hyst_20261029/dreamvm_20261022/mme.progress.json"
  score dream_mme outputs/mme_20261022/dreamvl_stock.jsonl \
    "$ROOT/outputs/hyst_20261029/dreamvm_20261022/mme.jsonl" hyst

  export HYST_RULE=run:4
  run lladav_pope "$PY" "$HY" lladav --backend lladav --dataset pope --mode stock --cuda 1 \
    --output "$ROOT/outputs/hyst_20261029/lladav_20261022/pope.jsonl" \
    --progress "$ROOT/outputs/hyst_20261029/lladav_20261022/pope.progress.json"
  score lladav_pope outputs/pope_lladav_20261022/lladav_stock.jsonl \
    "$ROOT/outputs/hyst_20261029/lladav_20261022/pope.jsonl" hyst

  sign_cell lavida_mme outputs/mme_20261022/lavida_stock.jsonl "$V" \
    --backend lavida --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_lavida.json
  sign_cell mmada_mme outputs/mme_20261022/mmada_stock.jsonl "$V" \
    --backend mmada --dataset mme --mode gated --alpha-rule spread \
    --prior-file outputs/mme_20261022/cf_mme_mmada.json
  sign_cell dream_vstar outputs/vstar_20261022/dreamvl_stock.jsonl "$DV" \
    --dataset vstar --mode gated --alpha-rule spread \
    --prior-file outputs/vstar_20261022/cf_vstar_dreamvl.json
else
  echo "gpu must be 0 or 1"; exit 1
fi
echo "[$(date -Is)] QUEUE END gpu=$GPU"
