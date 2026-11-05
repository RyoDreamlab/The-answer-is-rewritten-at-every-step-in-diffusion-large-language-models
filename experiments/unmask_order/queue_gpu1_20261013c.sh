#!/usr/bin/env bash
# Phase 2 (GPU 1): full decodes in three concurrent lanes, then auto-scoring.
#   lane A: lavida MMBench stock  -> mmada MMBench stock
#   lane B: lavida MMBench gated  -> mmada MMBench gated
#   lane C: lladav ScienceQA stock -> lladav ScienceQA gated -> lladav M3CoT stock -> lladav M3CoT gated
# Every job: nohup ... & (run_job), resumable (jsonl append + id skip), progress json per job.
# Start with:  cd $PROJECT && nohup setsid bash experiments/unmask_order/queue_gpu1_20261013c.sh > logs/gpu1_20261013c/queue_main.out 2>&1 < /dev/null &
set -uo pipefail
source "$(dirname "$0")/queue_lib_20261013c.sh"
SCORE="${PROJECT}/experiments/unmask_order/score_20261013c.py"
qlog "===== MAIN QUEUE start (GPU ${GPU}) ====="
for f in "$MMB/cf_lavida.json" "$MMB/cf_mmada.json" "$LLV/cf_lladav.json" "${PROJECT}/datasets/mmbench_dev_v1/manifest.json"; do
  [ -f "$f" ] || { qlog "missing prerequisite $f"; exit 2; }
done

# a lane = sequential jobs; each job waits for free memory before loading its model
lane_a() {
  wait_for_memory; decode lavida_stock lavida mmbench stock "$MMB"
  wait_for_memory; decode mmada_stock mmada mmbench stock "$MMB"
}
lane_b() {
  sleep 120; wait_for_memory; decode lavida_gated lavida mmbench gated "$MMB" --prior-file "$MMB/cf_lavida.json" --alpha-rule spread
  sleep 60;  wait_for_memory; decode mmada_gated mmada mmbench gated "$MMB" --prior-file "$MMB/cf_mmada.json" --alpha-rule spread
}
lane_c() {
  sleep 240; wait_for_memory; decode sciqa_stock lladav sciqa stock "$LLV"
  wait_for_memory; decode sciqa_gated lladav sciqa gated "$LLV" --prior-file "$LLV/cf_lladav.json" --alpha-rule spread
  wait_for_memory; decode m3cot_stock lladav m3cot stock "$LLV"
  wait_for_memory; decode m3cot_gated lladav m3cot gated "$LLV" --prior-file "$LLV/cf_lladav.json" --alpha-rule spread
}

lane_a > "${LOGDIR}/lane_a.out" 2>&1 &
pa=$!
lane_b > "${LOGDIR}/lane_b.out" 2>&1 &
pb=$!
lane_c > "${LOGDIR}/lane_c.out" 2>&1 &
pc=$!
qlog "lanes launched pids a=${pa} b=${pb} c=${pc}"

# periodic scoring while running (cheap, read-only), every 30 min
while kill -0 "$pa" 2>/dev/null || kill -0 "$pb" 2>/dev/null || kill -0 "$pc" 2>/dev/null; do
  sleep 1800
  "$PY" "$SCORE" > "${LOGDIR}/score_interim.log" 2>&1 && qlog "interim scoring ok" || qlog "interim scoring failed (see score_interim.log)"
done
wait "$pa" "$pb" "$pc"
qlog "all lanes finished"

qlog "START scoring :: $PY $SCORE"
"$PY" "$SCORE" > "${LOGDIR}/score_final.log" 2>&1
rc=$?
qlog "END   scoring rc=${rc} log=${LOGDIR}/score_final.log"
qlog "===== MAIN QUEUE end ====="
