#!/bin/bash
# Intervention 3 chain: force-unmask vs hidden-state write at step 0.  lavida -> dreamvl -> lladav on one GPU.
# usage: bash run_forceunmask_20261021.sh <cuda>
set -u
cd /mnt/pfs/peng_gao/visual-hysteresis
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=$PWD/hf-cache URD_PROJECT=$PWD TOKENIZERS_PARALLELISM=false
CUDA=${1:-0}
PY=.venv/bin/python3
S=experiments/unmask_order/forceunmask_20261021.py
LOG=logs/forceunmask_20261021
mkdir -p "$LOG"
run() {
  local name=$1; shift
  echo "START $name $(date -Is)" | tee -a "$LOG/chain.log"
  "$PY" "$S" "$@" >> "$LOG/${name}.log" 2>&1
  echo "EXIT $name code=$? $(date -Is)" | tee -a "$LOG/chain.log"
}
run lavida  --mode run --backend lavida  --cuda "$CUDA" --out-dir $PWD/outputs/forceunmask_20261021/lavida
run dreamvl --mode run --backend dreamvl --cuda "$CUDA" --out-dir $PWD/outputs/forceunmask_20261021/dreamvl
run lladav  --mode run --backend lladav  --cuda "$CUDA" --out-dir $PWD/outputs/forceunmask_20261021/lladav
echo "CHAIN DONE $(date -Is)" | tee -a "$LOG/chain.log"
