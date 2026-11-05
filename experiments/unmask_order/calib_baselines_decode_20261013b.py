#!/usr/bin/env python3
"""Prior-work calibration baselines, FULL ScienceQA-IMG test decode (gen 16, 16 steps, temperature 0,
low-confidence remasking) -- thin wrapper over calib_decode.py (unchanged), same pipeline as
calib_gated_decode_20261013.py. Rule definitions: calib_baselines_common_20261013b.py.

    --rule atabuzzaman|kumar|none   (stripped before calib_decode.py parses the rest)
    everything else as calib_decode.py: --backend --cuda --output --progress [--limit]

Pre-pass (inside the same process, before the decode; results cached next to --output):
  atabuzzaman  bctx_sqa_{backend}.json   step-0 raw letter logits of the first ceil(10%) items (manifest order)
                                          -> B_ctx per n; B = (B_gen + B_ctx)/2 (zero-centred each)
  kumar        cf_fullvocab_sqa_{backend}.json  full-vocab step-0 logits for the content-free input, n = 2..5
                                          (mid-gray 384x384, question N/A, n options N/A; same construction as
                                          calib_contentfree.py a_global). 4 extra forwards.
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
import calib_decode as cd  # noqa: E402
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
if cb.S["rule"] == "none" and "--no-calib" not in sys.argv:
    sys.argv.append("--no-calib")

CAP = {}
_orig_lavida_decode = traj.lavida_decode
_orig_mmada_decode = traj.mmada_decode


def lavida_decode_cap(lm, model, processor, tokenizer, row):
    CAP["lavida"] = (lm, model, processor, tokenizer)
    return _orig_lavida_decode(lm, model, processor, tokenizer, row)


def mmada_decode_cap(model, tokenizer, vq, mm, image, question, device):
    CAP["mmada"] = (model, tokenizer, vq, mm, device)
    return _orig_mmada_decode(model, tokenizer, vq, mm, image, question, device)


traj.lavida_decode = lavida_decode_cap
traj.mmada_decode = mmada_decode_cap


def cf_forward_fullvocab(backend: str, n: int) -> torch.Tensor:
    """Content-free step-0 forward (same construction as calib_contentfree.py a_global); returns pos-0 logits."""
    gen = base.GEN
    if backend == "lavida":
        from llava.constants import IMAGE_TOKEN_INDEX

        lm, model, processor, tokenizer = CAP["lavida"]
        past, plen = ccf.lavida_prefix_img(lm, model, processor, tokenizer, ccf.lavida_question_na(n),
                                           ccf.gray(ccf.GLOBAL_SIZE), IMAGE_TOKEN_INDEX)
        canvas = torch.full((1, gen), base.MASK_ID_LAVIDA, dtype=torch.long, device=model.device)
        logits = traj.lavida_forward(lm, past, canvas)
        cb.log(f"CF forward lavida n={n} prefix_len={plen}")
        return logits[0, 0]
    import sciqa_standard_eval as sq

    model, tokenizer, vq, mm, device = CAP["mmada"]
    q = sq.build_task_prompt({"question": "N/A", "choices": ["N/A"] * n, "hint": ""})
    vq_ids = mm.encode_vq(vq, ccf.gray(ccf.GLOBAL_SIZE), device)
    prompt_ids, _ = mm.build_prompt_ids(tokenizer, q, vq_ids, device)
    plen = int(prompt_ids.shape[1])
    canvas = torch.full((1, gen), mm.MASK_ID, dtype=torch.long, device=device)
    logits = base.mmada_logits(model, prompt_ids, canvas, mm.MASK_ID, None)
    if logits.shape[1] != plen + gen:
        raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
    cb.log(f"CF forward mmada n={n} prefix_len={plen}")
    return logits[0, plen]


def ref_letter_logit_model(backend: str) -> dict[int, list[float]]:
    recs = [json.loads(l) for l in (cd.CF_DIR / f"cf_{backend}.jsonl").open(encoding="utf-8") if l.strip()]
    return {r["n_opt"]: r["letter_logit_model"] for r in recs if r["cond"] == "a_global"}


def prepass(args, backend: str, rows: list[dict], fn) -> None:
    rule = cb.S["rule"]
    cache_dir = args.output.parent
    cb.S["step0_only"] = True
    try:
        if rule == "atabuzzaman":
            bpath = cache_dir / f"bctx_sqa_{backend}.json"
            if bpath.exists():
                cb.S["B"] = cb.load_B_table(bpath)
                cb.log(f"loaded B table {bpath}")
            else:
                k = math.ceil(cb.CTX_FRAC * len(rows))
                items, meta = [], []
                for j, row in enumerate(rows[:k]):
                    n = cd.n_options(row)
                    cd.STATE["n_opt"] = n
                    with torch.no_grad():
                        d = fn(row)
                    items.append((n, d["step0_letter_raw"]))
                    meta.append({"id": row["id"], "n_opt": n, "step0_letter_raw": d["step0_letter_raw"]})
                    if (j + 1) % 25 == 0 or j + 1 == k:
                        cb.log(f"PREPASS bctx {j + 1}/{k}")
                p_cf_by_n = {n: [math.exp(v) for v in lp] for n, lp in cd.STATE["log_pcf"].items()}
                by_n = cb.build_B(p_cf_by_n, items)
                cb.dump_B_table(bpath, backend, "scienceqa_img_test", by_n, meta)
                cb.S["B"] = cb.load_B_table(bpath)
                cb.log(f"wrote B table {bpath}")
            cb.S["n_prepass_items"] = len(cb.S["B"])
            for n, B in sorted(cb.S["B"].items()):
                cb.log(f"B n={n} {[round(v, 4) for v in B]}")
        elif rule == "kumar":
            cpath = cache_dir / f"cf_fullvocab_sqa_{backend}.json"
            if not cpath.exists():
                cd.STATE["n_opt"] = cd.n_options(rows[0])
                with torch.no_grad():
                    fn(rows[0])  # capture model objects only
                ref = ref_letter_logit_model(backend)
                per_n = {}
                for n in (2, 3, 4, 5):
                    with torch.no_grad():
                        lg = cf_forward_fullvocab(backend, n)
                    per_n[n] = cb.cf_fullvocab_record(lg, n, ref.get(n))
                    cb.log(f"CF n={n} vocab={per_n[n]['vocab_size']} letter_logit_model={per_n[n]['letter_logit_model']} "
                           f"max_abs_diff_vs_cf_jsonl={per_n[n]['max_abs_diff_vs_cf_jsonl_letter_logit_model']} "
                           f"letter_mass={per_n[n]['letter_mass_fullvocab']:.4f}")
                cb.dump_cf_fullvocab(cpath, backend, "scienceqa_img_test", per_n)
                cb.log(f"wrote {cpath}")
            cb.S["cf_full_prob"] = cb.load_cf_fullvocab(cpath)
            cb.log(f"loaded cf full-vocab probs {cpath} n={sorted(cb.S['cf_full_prob'])}")
    finally:
        cb.S["step0_only"] = False
    cb.S["ready"] = True


_orig_make_loop = cd.make_loop


def make_loop_b(backend: str, limit: int):
    inner = _orig_make_loop(backend, limit)

    def loop(args, rows, fn) -> None:
        use = rows[:limit] if limit else rows
        cb.log(f"rule={cb.S['rule']} n_rows={len(use)} ALPHA={cb.ALPHA} TAU={cb.TAU} ctx_frac={cb.CTX_FRAC}")
        prepass(args, backend, use, fn)
        inner(args, rows, fn)

    return loop


cd.make_loop = make_loop_b
cd.decode_canvas = cb.make_decode_canvas(cd.STATE, base, delta)
cd.OUT_ROOT = OUT_ROOT

if __name__ == "__main__":
    cd.main()
