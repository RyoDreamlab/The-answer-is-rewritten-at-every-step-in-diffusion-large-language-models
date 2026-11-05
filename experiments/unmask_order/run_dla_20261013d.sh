#!/usr/bin/env bash
# Upgrade 2 queue: DLA circuit (blank + 200 real) for both models -> zero-ablation full decodes (300 items) -> report.
# One GPU job at a time; GPU chosen before every launch (guest etiquette, see gpu_guest_20261013d.sh).
# Usage: nohup bash run_dla_20261013d.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
OUT=$ROOT/outputs/dla_circuit_20261013d
LOGD=$ROOT/logs/dla_20261013d
mkdir -p "$OUT" "$LOGD"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/hf-cache URD_PROJECT=$ROOT
export GUEST_LOG=$LOGD/queue.log
source "$S/gpu_guest_20261013d.sh"
PY=$ROOT/.venv/bin/python3
SCRIPT=$S/dla_circuit_20261013d.py
Q=$LOGD/queue.log
STAGE=${1:-all}
echo "QUEUE START stage=$STAGE $(date -Iseconds)" >> "$Q"

run_job() {  # $1 = name, rest = python args (without --cuda)
  name=$1; shift
  gpu=$(pick_gpu)
  log_gpu_state "$Q"
  echo "LAUNCH $name gpu=$gpu $(date -Iseconds)" >> "$Q"
  $PY "$SCRIPT" --cuda "$gpu" "$@" >> "$LOGD/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "EXIT $name rc=$rc $(date -Iseconds)" >> "$Q"
  return $rc
}

if [ "$STAGE" = all ] || [ "$STAGE" = smoke ]; then
  # sanity: no-ablation decode on 5 items must reproduce stock letters (and shows peak memory)
  run_job smoke_decode_lavida --mode decode --backend lavida --limit 5 --tag noablation_smoke5 || exit 1
fi
if [ "$STAGE" = all ] || [ "$STAGE" = dla ]; then
  [ -f "$OUT/dla_lavida.json" ] || run_job dla_lavida --mode dla --backend lavida --n-real 200 || exit 1
  [ -f "$OUT/dla_mmada.json" ] || run_job dla_mmada --mode dla --backend mmada --n-real 200 || exit 1
fi
if [ "$STAGE" = all ] || [ "$STAGE" = decode ]; then
  # pick components from the blank-input DLA (mean of n=2 and n=3 contributions)
  # writer   : largest mean contribution onto the final prior direction u_n over n = 2, 3 (where MMaDA is most biased)
  # canceller: most negative mean projection onto the DEEP prior direction (embed + L1..L31 letter logits) over n = 2..5
  #            (LaViDa's final n=2 prior is ~flat, so the final direction is noise there; the deep direction is not)
  pick() { $PY - "$1" "$2" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); names = d["components"]
if sys.argv[2] == "writer":
    c = [(a + b) / 2 for a, b in zip(d["blank"]["2"]["contrib"], d["blank"]["3"]["contrib"])]
    i = max(range(1, len(c)), key=lambda k: c[k]); print(names[i])
else:
    cols = [d["blank"][n]["contrib_deep31"] for n in ("2", "3", "4", "5")]
    c = [sum(col[k] for col in cols) / 4 for k in range(len(names))]
    i = min(range(1, len(c)), key=lambda k: c[k]); print(names[i] if c[i] < 0 else "")
EOF
  }
  canc=$(pick "$OUT/dla_lavida.json" canceller)
  writ=$(pick "$OUT/dla_mmada.json" writer)
  echo "PICK lavida_canceller=$canc mmada_writer=$writ" >> "$Q"
  [ -n "$canc" ] && run_job decode_lavida_zero --mode decode --backend lavida --limit 300 --components "$canc" --ablation zero --tag "canceller_${canc}_zero"
  run_job decode_mmada_zero --mode decode --backend mmada --limit 300 --components "$writ" --ablation zero --tag "writer_${writ}_zero"
  # secondary: mean-ablation controls (real-input component mean), same items
  [ -n "$canc" ] && run_job decode_lavida_mean --mode decode --backend lavida --limit 300 --components "$canc" --ablation mean --tag "canceller_${canc}_mean"
  run_job decode_mmada_mean --mode decode --backend mmada --limit 300 --components "$writ" --ablation mean --tag "writer_${writ}_mean"
fi
$PY "$SCRIPT" --mode report >> "$LOGD/report.log" 2>&1
echo "QUEUE END $(date -Iseconds)" >> "$Q"
