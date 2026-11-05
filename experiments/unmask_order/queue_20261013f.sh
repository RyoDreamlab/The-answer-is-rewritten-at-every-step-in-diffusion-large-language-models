#!/usr/bin/env bash
# P6 queue (causal perturbation of the step-0 decision, MMaDA only).
#   1. wait for the E-package queue (logs/e_20261013e/queue.log must contain "QUEUE END"; if the file does not exist
#      yet, poll for it for up to 2 h and then proceed anyway);
#   2. choose the Arm-1 plan from the smoke timing (sec/decode): full = 400 items × β{-4,-2,0,2,4,8}; if the Arm-1
#      estimate (400 × 16 decodes) exceeds 8 GPU-h, reduced = 300 items × β{-2,0,2,4,8};
#   3. run, one GPU job at a time, GPU picked before every launch (guest etiquette, gpu_guest_20261013d.sh):
#      steer writer_heads -> steer random_heads -> patch -> steer writer_comps -> canvas -> analysis.
# Usage: nohup bash queue_20261013f.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/p6_causal_20261013f
LOGD=$ROOT/logs/p6_20261013f
EQ=$ROOT/logs/e_20261013e/queue.log
mkdir -p "$OUT" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
export GUEST_LOG=$LOGD/queue.log
source "$S/gpu_guest_20261013d.sh"
PY=$ROOT/.venv/bin/python3
SCRIPT=$S/p6_causal_20261013f.py
Q=$LOGD/queue.log
echo "QUEUE START $(date -Iseconds) pid=$$" >> "$Q"

# ---------------------------------------------------------------- 1. wait for the E queue
# Proceed when the E log says QUEUE END, or when its coordinator has PAUSED the E lanes in favour of P6 (17:42 marker:
# "PAUSED lanes ... user prioritised P6 over E1/E2" — waiting for QUEUE END in that state would deadlock).
t_wait0=$(date +%s)
while true; do
  if [ -f "$EQ" ]; then
    if grep -q "QUEUE END" "$EQ"; then echo "EQUEUE ended ($(grep 'QUEUE END' "$EQ" | tail -1)) $(date -Iseconds)" >> "$Q"; break; fi
    if grep -q "PAUSED lanes.*P6" "$EQ"; then echo "EQUEUE lanes paused for P6 ($(grep 'PAUSED lanes' "$EQ" | tail -1)) $(date -Iseconds)" >> "$Q"; break; fi
  else
    if [ $(( $(date +%s) - t_wait0 )) -ge 7200 ]; then echo "EQUEUE log never appeared within 2 h; proceeding $(date -Iseconds)" >> "$Q"; break; fi
  fi
  sleep 60
done
# wait for the P6 smoke (code-path validation of every arm) to finish, at most 3 h
SMK=$LOGD/smoke2_queue.log
t_wait1=$(date +%s)
while ! grep -q "SMOKE END" "$SMK" 2>/dev/null; do
  if [ $(( $(date +%s) - t_wait1 )) -ge 10800 ]; then echo "SMOKE not finished within 3 h; proceeding $(date -Iseconds)" >> "$Q"; break; fi
  sleep 60
done
echo "SMOKE state: $(grep -E 'EXIT|SMOKE END' "$SMK" 2>/dev/null | tr '\n' ';') $(date -Iseconds)" >> "$Q"

# ---------------------------------------------------------------- 2. plan from smoke timing
SPD=$($PY - "$OUT/smoke2/progress_steer_writer_heads.json" <<'EOF'
import json, sys
try:
    print(json.load(open(sys.argv[1]))["sec_per_decode"])
except Exception:
    print(3.5)
EOF
)
EST_H=$($PY -c "print(round(400*16*float('$SPD')/3600, 2))")
if $PY -c "import sys; sys.exit(0 if float('$EST_H') > 8 else 1)"; then
  LIMIT=300; BETAS="-2,0,2,4,8"; PLAN=reduced
else
  LIMIT=400; BETAS="-4,-2,0,2,4,8"; PLAN=full
fi
echo "PLAN $PLAN smoke_sec_per_decode=$SPD arm1_est_hours_full=$EST_H limit=$LIMIT betas=$BETAS $(date -Iseconds)" >> "$Q"

run_job() {  # $1 = name, rest = python args (without --cuda)
  name=$1; shift
  gpu=$(pick_gpu)
  log_gpu_state "$Q"
  echo "LAUNCH $name gpu=$gpu $(date -Iseconds)" >> "$Q"
  $PY "$SCRIPT" --cuda "$gpu" "$@" >> "$LOGD/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "EXIT $name rc=$rc $(date -Iseconds) $(grep -h 'BETA0_EQUALS_STOCK' "$LOGD/$name.log" | tail -1)" >> "$Q"
  return $rc
}

# ---------------------------------------------------------------- 3. jobs
run_job steer_writer_heads --arm steer --group writer_heads --limit "$LIMIT" "--betas=$BETAS"
run_job steer_random_heads --arm steer --group random_heads --limit "$LIMIT" "--betas=$BETAS"
run_job patch --arm patch --limit 300 --pool 400
run_job steer_writer_comps --arm steer --group writer_comps --limit "$LIMIT" "--betas=$BETAS"
run_job canvas --arm canvas --limit "$LIMIT"
$PY "$S/p6_analysis_20261013f.py" >> "$LOGD/analysis.log" 2>&1
echo "ANALYSIS rc=$? $(date -Iseconds)" >> "$Q"
echo "QUEUE END $(date -Iseconds)" >> "$Q"
