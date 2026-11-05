#!/usr/bin/env bash
# GPU1: hold-slot LLaDA-V sciqa. Pilot n=200; if >=1 lenient letter changed, continue to the full 2017 in the same dir.
set -u
cd /mnt/pfs/peng_gao/visual-hysteresis || exit 1
export PYTHONUNBUFFERED=1 HF_HOME=$PWD/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=.venv/bin/python3
OUT=outputs/holdslot_lladav_20261021
echo "QUEUE start $(date) pilot n=200 cuda=1"
$PY experiments/unmask_order/holdslot_lladav_20261021.py --cuda 1 --n 200 --out-dir "$OUT"
rc=$?
echo "QUEUE pilot rc=$rc $(date)"
if [ "$rc" -ne 0 ]; then echo "QUEUE pilot failed; stop"; exit "$rc"; fi
CHANGED=$($PY - <<'EOF'
import json
print(json.load(open("outputs/holdslot_lladav_20261021/summary.json"))["letters_changed"])
EOF
)
echo "QUEUE pilot letters_changed=$CHANGED"
if [ "$CHANGED" -ge 1 ]; then
  echo "QUEUE continue to full 2017 $(date)"
  $PY experiments/unmask_order/holdslot_lladav_20261021.py --cuda 1 --n 2017 --out-dir "$OUT"
  echo "QUEUE full rc=$? $(date)"
else
  echo "QUEUE 0 letters changed in pilot; not running the full set"
fi
echo "QUEUE end $(date)"
