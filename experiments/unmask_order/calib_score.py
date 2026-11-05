#!/usr/bin/env python3
"""Score content-free letter calibration on step-0 letter logits and write the combined summary.

Real step-0 letter logits: saved ln_f input (subset ITM) from outputs/hidden_shapley_split_20261010,
read out in float64 with ln_f gamma and ff_out letter rows (as hidden_shapley_offline_rules.py).
Content-free priors: outputs/calib_vote_20261012/cf_{model}.jsonl (calib_contentfree.py).

p_cf = softmax of the content-free letter logits over the item's own options.
  sub     argmax_y  logit_y - log p_cf(y)          (contextual calibration in logit space)
  diagW   argmax_y  softmax(logit)_y / p_cf(y)     (Zhao et al. 2021, W = diag(1/p_cf), b = 0)
  probsub argmax_y  softmax(logit)_y - p_cf(y)     (W = I, b = -p_cf; extra variant)
  alpha   argmax_y  logit_y - alpha log p_cf(y), alpha fit on one half, scored on the other
No gold is used by sub / diagW / probsub. Gold is used only to score and to pick alpha on the fit half.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import hidden_shapley_offline_rules as R  # noqa: E402

OUT = PROJECT / "outputs" / "calib_vote_20261012"
LETTERS = "ABCDE"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
REF = {
    "lavida": {"step0": 1448, "letter_bias_crossfit": 1463, "finished": 1460, "official_scienceqa_mc": 1454},
    "mmada": {"step0": 1064, "letter_bias_crossfit": 1070, "finished": 1059, "official_scienceqa_mc": 1059},
}


def halves_ids():
    ids = sorted(f"scienceqa_img_test_{i:04d}" for i in range(R.hss.N_TOTAL))
    perm = np.random.default_rng(0).permutation(len(ids))
    half = (len(ids) + 1) // 2
    return {"A": [ids[i] for i in perm[:half]], "B": [ids[i] for i in perm[half:]]}


def real_logits(model: str):
    records = [json.loads(l) for l in (R.SPLIT_DIR / f"{model}.jsonl").open(encoding="utf-8")]
    ids = [r["id"] for r in records]
    if ids != [f"scienceqa_img_test_{i:04d}" for i in range(R.hss.N_TOTAL)]:
        raise RuntimeError("split jsonl not in manifest order")
    arr = np.load(R.SPLIT_DIR / f"{model}_h_subsets.npy", mmap_mode="r")
    H = torch.from_numpy(np.ascontiguousarray(arr[:, R.IDX["ITM"]])).to(torch.float64)
    gamma, head = R.load_readout(model, "cpu")
    lid = R.letter_ids_from_records(records)
    if [lid[l] for l in LETTERS] != [32, 33, 34, 35, 36]:
        raise RuntimeError(f"letter ids {lid}")
    W = torch.stack([head[lid[l]] for l in LETTERS]).to(torch.float64)
    v = H * gamma * torch.rsqrt((H * H).mean(-1, keepdim=True) + R.EPS)
    L = v @ W.T
    n_opt = torch.tensor([len(r["option_letters"]) for r in records])
    valid = torch.arange(5)[None, :] < n_opt[:, None]
    gold = torch.tensor([LETTERS.index(r["gold"]) for r in records])
    return ids, L, valid, gold, n_opt


def log_softmax_masked(x, valid):
    return torch.log_softmax(x.masked_fill(~valid, float("-inf")), dim=1)


def pick(score, valid):
    return torch.argmax(score.masked_fill(~valid, float("-inf")), dim=1)


def counts(pred, gold, base, idx=None):
    if idx is not None:
        pred, gold, base = pred[idx], gold[idx], base[idx]
    r, br = pred.eq(gold), base.eq(gold)
    return {"n": int(pred.numel()), "correct": int(r.sum()), "baseline_correct": int(br.sum()),
            "changed": int(pred.ne(base).sum()), "fixed": int((r & ~br).sum()), "broke": int((~r & br).sum())}


def dist(pred):
    c = Counter(LETTERS[int(p)] for p in pred)
    return {l: c.get(l, 0) for l in LETTERS}


def run_model(model: str, halves: dict) -> dict:
    ids, L, valid, gold, n_opt = real_logits(model)
    pos = {i: k for k, i in enumerate(ids)}
    hidx = {h: torch.tensor([pos[i] for i in v]) for h, v in halves.items()}
    base = pick(L, valid)
    base_correct = int(base.eq(gold).sum())
    if base_correct != REF[model]["step0"]:
        raise RuntimeError(f"{model} step-0 restricted {base_correct} != reference {REF[model]['step0']}")

    cf = [json.loads(l) for l in (OUT / f"cf_{model}.jsonl").open(encoding="utf-8")]
    glob = {r["n_opt"]: torch.tensor(r["letter_logit64"], dtype=torch.float64) for r in cf if r["cond"] == "a_global"}
    per = {c: {} for c in ("b_imgfree", "c_txtfree", "real")}
    for r in cf:
        if r["cond"] in per:
            per[r["cond"]][r["id"]] = r
    for c in ("b_imgfree", "c_txtfree"):
        if set(per[c]) != set(ids):
            raise RuntimeError(f"{model} {c} has {len(per[c])} items")
    if sorted(glob) != [2, 3, 4, 5]:
        raise RuntimeError(f"{model} global priors {sorted(glob)}")
    for c in ("b_imgfree", "c_txtfree"):
        bad = [i for i in ids if per[c][i]["n_opt"] != int(n_opt[pos[i]])]
        if bad:
            raise RuntimeError(f"{model} {c} option count mismatch {bad[:3]}")

    real_check = []
    for i, r in per["real"].items():
        diff = (torch.tensor(r["letter_logit64"], dtype=torch.float64) - L[pos[i]]).abs()
        k = pos[i]
        same = int(pick(torch.tensor([r["letter_logit64"]], dtype=torch.float64), valid[k:k + 1])[0]) == int(base[k])
        real_check.append((float(diff.max()), same))

    cf_logits = {
        "a_global": torch.stack([glob[int(n)] for n in n_opt]),
        "b_imgfree": torch.stack([torch.tensor(per["b_imgfree"][i]["letter_logit64"], dtype=torch.float64) for i in ids]),
        "c_txtfree": torch.stack([torch.tensor(per["c_txtfree"][i]["letter_logit64"], dtype=torch.float64) for i in ids]),
    }
    logp = log_softmax_masked(L, valid)
    out = {
        "n": len(ids),
        "baseline_step0_restricted_correct": base_correct,
        "references": REF[model],
        "real_forward_readout_check": {
            "n": len(real_check),
            "max_abs_letter_logit_diff_vs_saved_ITM": max(d for d, _ in real_check),
            "n_same_restricted_argmax": sum(s for _, s in real_check),
            "note": "fresh stock-path forward vs saved masked-path ITM hidden state; real logits used for scoring are the saved ITM ones",
        },
        "gold_distribution": dist(gold),
        "step0_pred_distribution": dist(base),
        "global_priors": {f"n{n}": {LETTERS[j]: float(p) for j, p in enumerate(
            torch.softmax(glob[n][:n], dim=0))} for n in (2, 3, 4, 5)},
        "conditions": {},
    }
    for cname, C in cf_logits.items():
        logpcf = log_softmax_masked(C, valid)
        pcf = logpcf.exp()
        variants = {
            "sub": L - logpcf,
            "diagW": logp.exp() / pcf,
            "probsub": logp.exp() - pcf,
        }
        res = {}
        preds = {}
        for vname, s in variants.items():
            p = pick(s, valid)
            preds[vname] = p
            res[vname] = {**counts(p, gold, base), "pred_distribution": dist(p),
                          "halves": {h: counts(p, gold, base, ix) for h, ix in hidx.items()}}
        res["diagW_equals_sub_argmax_items"] = int(preds["sub"].eq(preds["diagW"]).sum())
        cf_top = pick(C, valid)
        res["prior_stats"] = {
            "cf_argmax_distribution": dist(cf_top),
            "mean_p_cf_by_letter_over_items": {LETTERS[j]: float(pcf[:, j][valid[:, j]].mean()) for j in range(5)},
            "n_cf_argmax_equals_step0_pred": int(cf_top.eq(base).sum()),
            "n_cf_argmax_equals_gold_diagnostic": int(cf_top.eq(gold).sum()),
        }
        if cname != "a_global":
            res["prior_stats"]["n_cf_argmax_text_is_letter"] = sum(
                per[cname][i]["argmax_text"].strip() in LETTERS for i in ids)
        # strength alpha, cross-fitted
        apreds = [pick(L - a * logpcf, valid) for a in ALPHAS]
        folds, tot = [], {"correct": 0, "baseline_correct": 0, "changed": 0, "fixed": 0, "broke": 0}
        for fit, score in (("A", "B"), ("B", "A")):
            fc = [int(p[hidx[fit]].eq(gold[hidx[fit]]).sum()) for p in apreds]
            best = max(fc)
            j = min((k for k, c in enumerate(fc) if c == best), key=lambda k: (ALPHAS[k], k))
            held = counts(apreds[j], gold, base, hidx[score])
            folds.append({"fit": fit, "score": score, "chosen_alpha": ALPHAS[j], "fit_counts": dict(zip(map(str, ALPHAS), fc)),
                          "heldout": held})
            for k in tot:
                tot[k] += held[k]
        res["alpha_crossfit"] = {"grid": ALPHAS, "tie_break": "smallest alpha", "folds": folds, "crossfit": tot}
        res["alpha_same_set_diagnostic"] = {str(a): int(p.eq(gold).sum()) for a, p in zip(ALPHAS, apreds)}
        out["conditions"][cname] = res
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT / "summary.json")
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    torch.set_grad_enabled(False)
    halves = halves_ids()
    part2 = json.loads((OUT / "vote_part2.json").read_text())
    summary = {
        "date": "2026-10-12",
        "dataset": "ScienceQA-IMG test, 2017 items",
        "split": {"method": "numpy default_rng(0).permutation of sorted ids; A = first ceil(N/2)",
                  "sizes": {h: len(v) for h, v in halves.items()}},
        "part1_contentfree_calibration": {
            "position": "letter position (canvas index 0), decode step 0, full canvas masked",
            "conditions": {
                "a_global": "mid-gray 384x384 image, question 'N/A', n options 'N/A' (n = item's count), no context",
                "b_imgfree": "item's real question/context/options, mid-gray image at the item's image size",
                "c_txtfree": "item's real image, question 'N/A', n options 'N/A', context dropped",
            },
            "prompt_formats": {
                "lavida": "Question: {q}\\nAnswer: with q = '{question}[\\nContext: ..]\\nOptions:\\nA. ..\\nWhich letter is correct?'",
                "mmada": "sciqa_standard_eval.build_task_prompt (lmms-eval sqa_doc_to_text) with question/choices/hint replaced",
            },
            "real_logits": "saved ITM ln_f input from hidden_shapley_split_20261010, float64 ln_f + ff_out letter rows",
            "variants": {
                "sub": "logit_y - log p_cf(y), restricted to options (no gold)",
                "diagW": "W = diag(1/p_cf), b = 0 on restricted probabilities; same argmax as sub by construction",
                "probsub": "p_y - p_cf(y) (W = I, b = -p_cf), extra variant (no gold)",
                "alpha": "logit_y - alpha log p_cf(y); alpha picked on one half by gold, scored on the other",
            },
            "finished_letter_calibration": (
                "not computed: the stock decode does not save letter logits at the letter's unmask step "
                "(trajectory file stores argmax text only; LaViDa 1804/2017 and MMaDA 2010/2017 letters unmask at "
                "step 15). Calibrating the finished letter needs a re-decode that saves the 5 letter logits at the "
                "unmask step; only step-0 is estimated here."),
            "models": {m: run_model(m, halves) for m in ("lavida", "mmada")},
        },
        "part2_cross_step_vote": part2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    for m, r in summary["part1_contentfree_calibration"]["models"].items():
        print(m, "step0", r["baseline_step0_restricted_correct"], "check", r["real_forward_readout_check"])
        for c, res in r["conditions"].items():
            for v in ("sub", "diagW", "probsub"):
                x = res[v]
                print(f"  {c:10s} {v:8s} correct={x['correct']} changed={x['changed']} fixed={x['fixed']} "
                      f"broke={x['broke']} dist={x['pred_distribution']}")
            a = res["alpha_crossfit"]
            print(f"  {c:10s} alpha-cf {a['crossfit']} chosen={[f['chosen_alpha'] for f in a['folds']]}")
            print(f"  {c:10s} prior {res['prior_stats']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
