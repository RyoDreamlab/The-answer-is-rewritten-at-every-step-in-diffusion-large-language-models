#!/usr/bin/env bash
# Shared helpers for the 2026-10-13c GPU-1 queues (sourced; not executed directly).
# GPU 1 only. Every GPU job is started with nohup ... & and waited on, so the queue survives disconnects
# when the queue script itself is started with nohup/setsid.

PROJECT="${PROJECT:-/mnt/pfs/peng_gao/visual-hysteresis}"
PY="${PROJECT}/.venv/bin/python3"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${PROJECT}/hf-cache" URD_PROJECT="${PROJECT}"
GPU=1
LOGDIR="${PROJECT}/logs/gpu1_20261013c"
QLOG="${LOGDIR}/queue.log"
DEC="${PROJECT}/experiments/unmask_order/calib_decode_multi_20261013c.py"
MMB="${PROJECT}/outputs/mmbench_20261013c"
LLV="${PROJECT}/outputs/lladav_20261013c"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
mkdir -p "$LOGDIR" "$MMB" "$LLV"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
qlog() { echo "[$(ts)] $*" | tee -a "$QLOG"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' '; }
used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' '; }

wait_for_memory() {  # wait until GPU 1 has >= MIN_FREE_MIB free
  local need="${1:-$MIN_FREE_MIB}" n=0
  while [ "$(free_mib)" -lt "$need" ]; do
    [ $((n % 30)) -eq 0 ] && qlog "waiting for GPU${GPU} memory: free=$(free_mib) MiB need=${need}"
    sleep 20; n=$((n + 1))
  done
}

# run_job NAME CMD...   : nohup CMD in background, poll peak GPU-1 memory, wait, log START/END + rc. Returns rc.
run_job() {
  local name="$1"; shift
  local log="${LOGDIR}/${name}.log"
  qlog "START ${name} :: $*"
  nohup "$@" >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "${LOGDIR}/${name}.pid"
  local peak=0 u
  while kill -0 "$pid" 2>/dev/null; do
    u="$(used_mib)"; [ "${u:-0}" -gt "$peak" ] && peak="$u"
    sleep 15
  done
  wait "$pid"; local rc=$?
  qlog "END   ${name} rc=${rc} gpu${GPU}_peak_used_mib(all procs)=${peak} log=${log}"
  return $rc
}

# launch_job NAME CMD... : nohup CMD in background WITHOUT waiting; prints pid. Used for concurrent lanes.
launch_job() {
  local name="$1"; shift
  local log="${LOGDIR}/${name}.log"
  qlog "START ${name} :: $*"
  nohup "$@" >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "${LOGDIR}/${name}.pid"
  echo "$pid"
}

decode() {  # decode NAME BACKEND DATASET MODE OUTDIR [extra args...]
  local name="$1" backend="$2" dataset="$3" mode="$4" outdir="$5"; shift 5
  run_job "$name" "$PY" "$DEC" --backend "$backend" --dataset "$dataset" --mode "$mode" --cuda "$GPU" \
    --output "${outdir}/${name}.jsonl" --progress "${outdir}/progress_${name}.json" "$@"
}
