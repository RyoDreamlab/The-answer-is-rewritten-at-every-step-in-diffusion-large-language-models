#!/usr/bin/env bash
# Phase 1 (GPU 1): content-free priors + 5-item smokes + "gated with alpha forced to 0 == stock" sanity.
# Start with:  cd $PROJECT && nohup setsid bash experiments/unmask_order/queue_smoke_gpu1_20261013c.sh > logs/gpu1_20261013c/queue_smoke.out 2>&1 &
set -uo pipefail
source "$(dirname "$0")/queue_lib_20261013c.sh"
qlog "===== SMOKE QUEUE start (GPU ${GPU}) ====="
fail=0

# --- LLaDA-V: smoke stock 5 ScienceQA items, prior n=2..5, gated alpha-scale 0 (must equal stock)
decode smoke_lladav_sciqa_stock_n5 lladav sciqa stock "$LLV" --limit 5 || fail=1
if [ ! -f "$LLV/cf_lladav.json" ]; then
  run_job prior_lladav "$PY" "$DEC" --backend lladav --dataset sciqa --mode prior --cuda "$GPU" --output "$LLV/cf_lladav.json" || fail=1
fi
decode smoke_lladav_sciqa_gated_a0_n5 lladav sciqa gated "$LLV" --limit 5 --alpha-scale 0 --prior-file "$LLV/cf_lladav.json" || fail=1
decode smoke_lladav_sciqa_gated_n5 lladav sciqa gated "$LLV" --limit 5 --prior-file "$LLV/cf_lladav.json" || fail=1

# --- LaViDa / MMaDA on MMBench: prior (same template as ScienceQA), stock 5, gated alpha 0 (== stock), gated 5
for model in lavida mmada; do
  if [ ! -f "$MMB/cf_${model}.json" ]; then
    run_job prior_${model}_mmbench "$PY" "$DEC" --backend "$model" --dataset mmbench --mode prior --cuda "$GPU" --output "$MMB/cf_${model}.json" || fail=1
  fi
  decode smoke_${model}_mmbench_stock_n5 "$model" mmbench stock "$MMB" --limit 5 || fail=1
  decode smoke_${model}_mmbench_gated_a0_n5 "$model" mmbench gated "$MMB" --limit 5 --alpha-scale 0 --prior-file "$MMB/cf_${model}.json" || fail=1
  decode smoke_${model}_mmbench_gated_n5 "$model" mmbench gated "$MMB" --limit 5 --prior-file "$MMB/cf_${model}.json" || fail=1
done

# --- sanity: alpha-0 gated reproduces stock letters/text; print smoke letters
"$PY" - "$LLV" "$MMB" <<'PY' 2>&1 | tee -a "$QLOG"
import json, sys
from pathlib import Path
llv, mmb = Path(sys.argv[1]), Path(sys.argv[2])
def jl(p): return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []
ok = True
for stock, a0, gated in ((llv/"smoke_lladav_sciqa_stock_n5.jsonl", llv/"smoke_lladav_sciqa_gated_a0_n5.jsonl", llv/"smoke_lladav_sciqa_gated_n5.jsonl"),
                         (mmb/"smoke_lavida_mmbench_stock_n5.jsonl", mmb/"smoke_lavida_mmbench_gated_a0_n5.jsonl", mmb/"smoke_lavida_mmbench_gated_n5.jsonl"),
                         (mmb/"smoke_mmada_mmbench_stock_n5.jsonl", mmb/"smoke_mmada_mmbench_gated_a0_n5.jsonl", mmb/"smoke_mmada_mmbench_gated_n5.jsonl")):
    s, z, g = jl(stock), jl(a0), jl(gated)
    print(f"SMOKE {stock.name}: n_stock={len(s)} n_a0={len(z)} n_gated={len(g)}")
    for r in s:
        print(f"   stock  {r['id']} gold={r['gold']} letter={r['finished_letter']} step={r['letter_unmask_step']} s0raw={r['step0_restricted_raw']} text={r['finished_text']!r} prefix_len={r.get('prefix_len')} sec={r.get('sec')}")
    for r in g:
        print(f"   gated  {r['id']} gold={r['gold']} letter={r['finished_letter']} step={r['letter_unmask_step']} s0raw={r['step0_restricted_raw']} s0cal={r['step0_restricted_cal']} alpha={r['alpha']:.4f} text={r['finished_text']!r}")
    zm = {r['id']: r for r in z}
    same = all(r['id'] in zm and zm[r['id']]['finished_text'] == r['finished_text'] and zm[r['id']]['letter_unmask_step'] == r['letter_unmask_step'] for r in s)
    print(f"   ALPHA0_EQUALS_STOCK={same} (n={len(s)})")
    ok = ok and same and len(s) == 5 and len(z) == 5 and len(g) == 5
print("SMOKE_ALL_OK" if ok else "SMOKE_FAILED")
PY

qlog "===== SMOKE QUEUE end fail=${fail} ====="
