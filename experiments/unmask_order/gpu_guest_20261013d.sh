#!/usr/bin/env bash
# Guest-GPU etiquette helpers (sourced by the 20261013d queue scripts).
#   pick_gpu  -> echoes the index of the GPU with more free memory; blocks (polling every 5 min) while both GPUs
#                have < 30 GB free.  Writes a wait note to $GUEST_LOG if set.
MIN_FREE_MIB=${MIN_FREE_MIB:-30720}

gpu_free_mib() {  # $1 = index
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" | awk -F', ' '{print $1-$2}'
}

pick_gpu() {
  while true; do
    f0=$(gpu_free_mib 0); f1=$(gpu_free_mib 1)
    if [ "$f0" -ge "$f1" ]; then best=0; bestfree=$f0; else best=1; bestfree=$f1; fi
    if [ "$bestfree" -ge "$MIN_FREE_MIB" ]; then
      echo "$best"; return 0
    fi
    [ -n "${GUEST_LOG:-}" ] && echo "WAIT $(date -Iseconds) free0=${f0}MiB free1=${f1}MiB (<${MIN_FREE_MIB}) sleeping 300s" >> "$GUEST_LOG"
    sleep 300
  done
}

log_gpu_state() {  # $1 = log file
  echo "GPU $(date -Iseconds) $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> "$1"
}
