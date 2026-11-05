#!/usr/bin/env python3
"""Prior-work calibration baselines, M3CoT validation (1108) decode, gen 16 -- thin wrapper over
calib_decode_m3cot.py (unchanged), same pipeline as calib_decode_m3cot_20261013.py. Rule definitions:
calib_baselines_common_20261013b.py.

    --rule atabuzzaman|kumar|none   (stripped before calib_decode_m3cot.py parses the rest)
    --rule none        requires --mode stock
    other rules        require --mode calib --prior-file outputs/m3cot_20261012/cf_{backend}.jsonl (a_global p_cf
                       on the M3CoT template)

Pre-pass (first item of the run, before its decode; cached next to --output):
  atabuzzaman  bctx_m3cot_{backend}.json      step-0 raw letter logits of the first ceil(10%) items (parquet order)
  kumar        cf_fullvocab_m3cot_{backend}.json  content-free full-vocab step-0 logits, n = 2..5 (M3CoT template,
                                              same construction as calib_decode_m3cot.py --mode prior)
Outputs go to outputs/calib_baselines_20261013b/.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
sys.path.insert(0, str(PROJECT / "experiments"))
import calib_baselines_common_20261013b as cb  # noqa: E402
import calib_contentfree as ccf  # noqa: E402
import calib_decode_m3cot as m  # noqa: E402
import letter_argmax_traj as traj  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "calib_baselines_20261013b").resolve()

if "--rule" not in sys.argv:
    raise SystemExit("--rule atabuzzaman|kumar|none required")
_i = sys.argv.index("--rule")
cb.S["rule"] = sys.argv[_i + 1]
del sys.argv[_i:_i + 2]
if cb.S["rule"] not in cb.RULES:
    raise SystemExit(f"--rule must be one of {cb.RULES}")
_mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else None
if cb.S["rule"] == "none" and _mode != "stock":
    raise SystemExit("--rule none requires --mode stock")
if cb.S["rule"] != "none" and _mode != "calib":
    raise SystemExit(f"--rule {cb.S['rule']} requires --mode calib --prior-file ...")
ARGS = {"limit": 0, "output": None, "prior_file": None, "backend": None}
for key, flag in (("limit", "--limit"), ("output", "--output"), ("prior_file", "--prior-file"), ("backend", "--backend")):
    if flag in sys.argv:
        ARGS[key] = sys.argv[sys.argv.index(flag) + 1]
ARGS["limit"] = int(ARGS["limit"])

_orig_lavida_decode = m.lavida_decode
_orig_mmada_decode = traj.mmada_decode
NA = {n: {"question": "N/A", "context": "", "choices": ["N/A"] * n} for n in (2, 3, 4, 5)}


def ref_letter_logit_model() -> dict[int, list[float]]:
    if not ARGS["prior_file"]:
        return {}
    recs = [json.loads(l) for l in Path(ARGS["prior_file"]).open(encoding="utf-8") if l.strip()]
    return {r["n_opt"]: r.get("letter_logit_model") for r in recs if r["cond"] == "a_global"}


def prepass(step0_of_row, step0_cf) -> None:
    """step0_of_row(row) -> step-0 dict for a real item; step0_cf(n) -> step-0 dict for the content-free input."""
    rule = cb.S["rule"]
    backend = ARGS["backend"]
    cache_dir = Path(ARGS["output"]).parent
    saved_n = m.STATE["n_opt"]
    cb.S["step0_only"] = True
    try:
        if rule == "atabuzzaman":
            bpath = cache_dir / f"bctx_m3cot_{backend}.json"
            if bpath.exists():
                cb.S["B"] = cb.load_B_table(bpath)
                cb.log(f"loaded B table {bpath}")
            else:
                rows = m.load_rows(ARGS["limit"])
                k = math.ceil(cb.CTX_FRAC * len(rows))
                items, meta = [], []
                for j, row in enumerate(rows[:k]):
                    m.STATE["n_opt"] = row["n_opt"]
                    with torch.no_grad():
                        d = step0_of_row(row)
                    items.append((row["n_opt"], d["step0_letter_raw"]))
                    meta.append({"id": row["id"], "n_opt": row["n_opt"], "step0_letter_raw": d["step0_letter_raw"]})
                    if (j + 1) % 25 == 0 or j + 1 == k:
                        cb.log(f"PREPASS bctx {j + 1}/{k}")
                p_cf_by_n = {n: [math.exp(v) for v in lp] for n, lp in m.STATE["log_pcf"].items()}
                by_n = cb.build_B(p_cf_by_n, items)
                cb.dump_B_table(bpath, backend, "m3cot_validation", by_n, meta)
                cb.S["B"] = cb.load_B_table(bpath)
                cb.log(f"wrote B table {bpath}")
            for n, B in sorted(cb.S["B"].items()):
                cb.log(f"B n={n} {[round(v, 4) for v in B]}")
        elif rule == "kumar":
            cpath = cache_dir / f"cf_fullvocab_m3cot_{backend}.json"
            if not cpath.exists():
                ref = ref_letter_logit_model()
                per_n = {}
                for n in (2, 3, 4, 5):
                    m.STATE["n_opt"] = n
                    with torch.no_grad():
                        d = step0_cf(n)
                    per_n[n] = cb.cf_fullvocab_record(d["step0_full_logits"], n, ref.get(n))
                    cb.log(f"CF n={n} vocab={per_n[n]['vocab_size']} letter_logit_model={per_n[n]['letter_logit_model']} "
                           f"max_abs_diff_vs_cf_jsonl={per_n[n]['max_abs_diff_vs_cf_jsonl_letter_logit_model']} "
                           f"letter_mass={per_n[n]['letter_mass_fullvocab']:.4f}")
                cb.dump_cf_fullvocab(cpath, backend, "m3cot_validation", per_n)
                cb.log(f"wrote {cpath}")
            cb.S["cf_full_prob"] = cb.load_cf_fullvocab(cpath)
            cb.log(f"loaded cf full-vocab probs {cpath} n={sorted(cb.S['cf_full_prob'])}")
    finally:
        cb.S["step0_only"] = False
        m.STATE["n_opt"] = saved_n
    cb.S["ready"] = True


def lavida_decode_cap(lm, model, processor, tokenizer, text, image):
    if not cb.S["ready"]:
        cb.log(f"rule={cb.S['rule']} ALPHA={cb.ALPHA} TAU={cb.TAU} ctx_frac={cb.CTX_FRAC} (lavida pre-pass)")
        prepass(
            lambda row: _orig_lavida_decode(lm, model, processor, tokenizer, m.build_text("lavida", row), m.rgb(row)),
            lambda n: _orig_lavida_decode(lm, model, processor, tokenizer, m.lavida_text(**NA[n]),
                                          ccf.gray(ccf.GLOBAL_SIZE)),
        )
    return _orig_lavida_decode(lm, model, processor, tokenizer, text, image)


def mmada_decode_cap(model, tokenizer, vq, mm, image, question, device):
    if not cb.S["ready"]:
        cb.log(f"rule={cb.S['rule']} ALPHA={cb.ALPHA} TAU={cb.TAU} ctx_frac={cb.CTX_FRAC} (mmada pre-pass)")
        prepass(
            lambda row: _orig_mmada_decode(model, tokenizer, vq, mm, m.rgb(row), m.build_text("mmada", row), device),
            lambda n: _orig_mmada_decode(model, tokenizer, vq, mm, ccf.gray(ccf.GLOBAL_SIZE), m.mmada_text(**NA[n]),
                                         device),
        )
    return _orig_mmada_decode(model, tokenizer, vq, mm, image, question, device)


m.lavida_decode = lavida_decode_cap
traj.mmada_decode = mmada_decode_cap
m.decode_canvas = cb.make_decode_canvas(m.STATE, base, delta)
m.OUT_ROOT = OUT_ROOT

if __name__ == "__main__":
    m.main()
