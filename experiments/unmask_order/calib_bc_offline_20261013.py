#!/usr/bin/env python3
"""Candidate C for the one allowed re-selection (STORY_LOCK §5): Batch-Calibration prior grouped by option count.

Offline, ScienceQA-IMG step-0 letter logits (same inputs as calib_gated_offline.py). No model forwards.
prior_BC[n] = mean over items with n options of log_softmax(real step-0 logits) (transductive, no labels).
Cross-fit: prior estimated on half A, applied to half B, and vice versa (same seed-0 halves).
Variants:
  bc_full       score = logit - 1.0 * prior_BC
  bc_spread     alpha_n = 1 - exp(-(max-min prior_BC[n]))   (same zero-constant gate as apriori_spread, but on BC prior)
  cf_spread     reference = current rule (content-free prior, apriori_spread), recomputed here for the same ids
Writes outputs/calib_gated_reselect_20261013/summary.json (new dir; nothing overwritten).
"""
import json, math, os, sys
from pathlib import Path
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import calib_score as CS  # noqa
import calib_gated_offline as G  # noqa

OUT = PROJECT / "outputs" / "calib_gated_reselect_20261013"
OUT.mkdir(parents=True, exist_ok=True)


def bc_prior(L, valid, n_opt, idx):
    lp = CS.log_softmax_masked(L[idx], valid[idx])
    pri = {}
    for n in (2, 3, 4, 5):
        m = n_opt[idx] == n
        pri[n] = lp[m][:, :n].mean(0) if m.any() else torch.zeros(n, dtype=torch.float64)
    return pri


def prior_matrix(pri, n_opt, L):
    P = torch.zeros_like(L)
    for k, n in enumerate(n_opt.tolist()):
        P[k, :n] = pri[n]
    return P


def alpha_vec(pri, n_opt):
    a = {n: 1.0 - math.exp(-float(pri[n].max() - pri[n].min())) for n in pri}
    return torch.tensor([a[n] for n in n_opt.tolist()], dtype=torch.float64), a


def main():
    halves = CS.halves_ids()
    summary = {"date": "2026-10-13", "note": __doc__.strip(), "models": {}}
    for model in ("lavida", "mmada"):
        D = G.load(model)
        ids = D["ids"]; L = D["L"]; valid = D["valid"]; gold = D["gold"]; n_opt = D["n_opt"]
        pos = {i: k for k, i in enumerate(ids)}
        idxA = torch.tensor([pos[i] for i in halves["A"]]); idxB = torch.tensor([pos[i] for i in halves["B"]])
        base = CS.pick(L, valid)
        res = {"step0_raw_correct": int(base.eq(gold).sum())}
        preds = {"bc_full": base.clone(), "bc_spread": base.clone(), "cf_spread": base.clone()}
        alphas = {}
        for fit_idx, score_idx, tag in ((idxA, idxB, "A->B"), (idxB, idxA, "B->A")):
            pri = bc_prior(L, valid, n_opt, fit_idx)
            P = prior_matrix(pri, n_opt, L)
            preds["bc_full"][score_idx] = CS.pick((L - P)[score_idx], valid[score_idx])
            av, a_by_n = alpha_vec(pri, n_opt)
            alphas[tag] = {"bc_alpha_by_n": {str(n): round(v, 4) for n, v in a_by_n.items()},
                           "bc_prior_by_n": {str(n): [round(float(x), 4) for x in pri[n]] for n in pri}}
            preds["bc_spread"][score_idx] = CS.pick((L - av[:, None] * P)[score_idx], valid[score_idx])
        # reference: content-free prior with apriori_spread (fixed rule, no fit -> same on both halves)
        acf = G.alpha_items(D, "apriori_spread", 0)
        preds["cf_spread"] = CS.pick(L - acf[:, None] * D["logpcf"], valid)
        for name, p in preds.items():
            res[name] = {"correct": int(p.eq(gold).sum()), "delta_vs_raw": int(p.eq(gold).sum() - base.eq(gold).sum()),
                         "changed": int(p.ne(base).sum()),
                         "fixed": int((p.eq(gold) & base.ne(gold)).sum()), "broke": int((p.ne(gold) & base.eq(gold)).sum())}
        res["alphas"] = alphas
        summary["models"][model] = res
        print(model, json.dumps({k: v for k, v in res.items() if k != "alphas"}, ensure_ascii=False))
        print("   ", json.dumps(alphas, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
