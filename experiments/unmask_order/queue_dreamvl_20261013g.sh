#!/usr/bin/env bash
# Dream-VL-7B full queue (2026-10-13g). NOT auto-launched: start it by hand once the smoke is accepted.
#   1. wait until logs/p6_20261013f/queue.log contains "QUEUE END" (P6 has GPU priority; if the file does not exist
#      yet, keep polling every 60 s -- no timeout: this queue must never pre-empt P6);
#   2. content-free priors (cf_dreamvl.json for the MCQ sets, cf_pope_dreamvl.json for POPE; skipped if present);
#   3. strictly serial decodes, each on the GPU picked by gpu_guest_20261013d.sh:pick_gpu (>= 30 GB free, freer GPU
#      first, otherwise poll every 5 min), each resumable (jsonl append + id skip), failed jobs retried once after 5 min:
#        sciqa stock -> sciqa gated -> m3cot stock -> m3cot gated -> mmbench stock -> mmbench gated
#        -> mmstar stock -> mmstar gated -> mmstar gray -> pope stock -> pope gated
#   4. analysis (CPU, paired bootstrap 1000 draws seed 20261013) -> outputs/dreamvl_20261013g/{summary.json,tables.md};
#      interim analysis every 30 min while the queue runs.
# Budget (smoke2, 2026-10-13 18:27-18:32, GPU 0 shared with 3-4 other processes at 99% util): 2.95 s/decode stock,
#   2.37 s/decode gated, ~18-19 GB per job; 21408 decodes -> ~17.5 GPU-h at the contended rate (+13 model loads of
#   ~15 s); an exclusive GPU should be 2-4x faster (~4-8 GPU-h).
# Only writes under outputs/dreamvl_20261013g/ and logs/dreamvl_20261013g/. Never touches other queues or outputs.
# Start: cd $ROOT && nohup bash experiments/unmask_order/queue_dreamvl_20261013g.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/dreamvl_20261013g
LOGD=$ROOT/logs/dreamvl_20261013g
P6Q=$ROOT/logs/p6_20261013f/queue.log
mkdir -p "$OUT" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
Q=$LOGD/queue.log
export GUEST_LOG=$Q
source "$S/gpu_guest_20261013d.sh"   # gpu_free_mib, pick_gpu, log_gpu_state, MIN_FREE_MIB (30720)
PY=$ROOT/.venv/bin/python3
DEC=$S/calib_decode_dreamvl_20261013g.py
ANA=$S/analysis_dreamvl_20261013g.py
CF=$OUT/cf_dreamvl.json
CFP=$OUT/cf_pope_dreamvl.json

qlog() { echo "[$(date -Iseconds)] $*" >> "$Q"; }

for f in "$DEC" "$ANA" "$S/backend_dreamvl_20261013g.py" "$ROOT/checkpoints/Dream-VL-7B/config.json" \
         "$ROOT/datasets/sciqa_img_standard_v1/manifest.json" "$ROOT/datasets/mmbench_dev_v1/manifest.json" \
         "$ROOT/datasets/mmstar_20261013e/manifest.json" "$ROOT/datasets/pope_20261013e/manifest.json"; do
  [ -f "$f" ] || { qlog "missing prerequisite $f"; exit 2; }
done
qlog "===== QUEUE START pid=$$ ====="

# ---------------------------------------------------------------- 1. P6 has priority: wait for its QUEUE END
n=0
while true; do
  if [ -f "$P6Q" ] && grep -q "QUEUE END" "$P6Q"; then
    qlog "P6 queue ended ($(grep 'QUEUE END' "$P6Q" | tail -1))"; break
  fi
  [ $((n % 30)) -eq 0 ] && qlog "waiting for P6 (QUEUE END in $P6Q; exists=$([ -f "$P6Q" ] && echo yes || echo no))"
  sleep 60; n=$((n + 1))
done

# ---------------------------------------------------------------- 2./3. jobs
# run_job NAME CMD... : pick a GPU (>= 30 GB free, else poll), run CMD --cuda G, log rc + wall time; retry once.
run_job() {
  local name=$1; shift
  local attempt=1 rc gpu t0
  while [ $attempt -le 2 ]; do
    gpu=$(pick_gpu)
    log_gpu_state "$Q"
    qlog "LAUNCH $name attempt=$attempt gpu=$gpu free=$(gpu_free_mib "$gpu")MiB"
    qlog "CMD $* --cuda $gpu"
    t0=$(date +%s)
    "$@" --cuda "$gpu" >> "$LOGD/$name.log" 2>&1 < /dev/null
    rc=$?
    qlog "EXIT $name attempt=$attempt rc=$rc sec=$(( $(date +%s) - t0 )) $(grep -h 'PEAK_GPU_MEM' "$LOGD/$name.log" | tail -1)"
    [ $rc -eq 0 ] && return 0
    attempt=$((attempt + 1))
    [ $attempt -le 2 ] && { qlog "RETRY $name in 300s"; sleep 300; }
  done
  qlog "GIVEUP $name"
  return 1
}

decode() {  # decode NAME DATASET MODE [extra args...]
  local name=$1 ds=$2 mode=$3; shift 3
  run_job "$name" "$PY" "$DEC" --dataset "$ds" --mode "$mode" --output "$OUT/${ds}_${mode}.jsonl" \
    --progress "$OUT/progress_${ds}_${mode}.json" "$@"
}

interim() {  # background interim analysis every 30 min while the main chain runs
  while true; do
    sleep 1800
    "$PY" "$ANA" > "$LOGD/analysis_interim.log" 2>&1 && qlog "interim analysis ok" || qlog "interim analysis failed (see analysis_interim.log)"
  done
}
interim &
INTERIM_PID=$!

[ -f "$CF" ]  || run_job prior_mcq  "$PY" "$DEC" --dataset sciqa --mode prior --output "$CF"
[ -f "$CFP" ] || run_job prior_pope "$PY" "$DEC" --dataset pope  --mode prior --output "$CFP"
[ -f "$CF" ] && [ -f "$CFP" ] || { qlog "priors missing; abort"; kill "$INTERIM_PID" 2>/dev/null; exit 3; }

decode sciqa_stock    sciqa   stock
decode sciqa_gated    sciqa   gated --alpha-rule spread --prior-file "$CF"
decode m3cot_stock    m3cot   stock
decode m3cot_gated    m3cot   gated --alpha-rule spread --prior-file "$CF"
decode mmbench_stock  mmbench stock
decode mmbench_gated  mmbench gated --alpha-rule spread --prior-file "$CF"
decode mmstar_stock   mmstar  stock
decode mmstar_gated   mmstar  gated --alpha-rule spread --prior-file "$CF"
decode mmstar_gray    mmstar  gray
decode pope_stock     pope    stock
decode pope_gated     pope    gated --alpha-rule spread --prior-file "$CFP"

kill "$INTERIM_PID" 2>/dev/null
"$PY" "$ANA" > "$LOGD/analysis_final.log" 2>&1
qlog "FINAL analysis rc=$? log=$LOGD/analysis_final.log -> $OUT/summary.json $OUT/tables.md"
qlog "===== QUEUE END ====="
