#!/usr/bin/env bash
# GPU0: wait until the MMaDA commit job (PID 2975415) is gone and GPU0 memory < 2000 MiB, then run mmada_grayswap.
set -u
cd /mnt/pfs/peng_gao/visual-hysteresis || exit 1
export PYTHONUNBUFFERED=1 HF_HOME=$PWD/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
WAIT_PID=2975415
echo "WAITER start $(date) waiting for pid $WAIT_PID and GPU0 mem < 2000 MiB"
while true; do
  alive=$(ps -eo pid | grep -w "$WAIT_PID" | wc -l)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "WAITER $(date +%H:%M:%S) pid_alive=$alive gpu0_mem=${mem}MiB"
  if [ "$alive" -eq 0 ] && [ "${mem:-99999}" -lt 2000 ]; then break; fi
  sleep 60
done
echo "WAITER go $(date)"
.venv/bin/python3 experiments/unmask_order/mmada_grayswap_20261021.py --cuda 0 --n 1500 --out-dir outputs/mmada_grayswap_20261021/mmada
echo "WAITER job rc=$? $(date)"
