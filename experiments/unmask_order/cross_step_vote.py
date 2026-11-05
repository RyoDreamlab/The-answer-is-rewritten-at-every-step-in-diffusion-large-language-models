#!/usr/bin/env python3
"""Cross-step voting on the position-0 argmax trajectory (offline, no GPU, no gold in the rules).

Input: outputs/letter_argmax_traj_20261010/{model}.jsonl; pos0_argmax[t] is the argmax text at
canvas position 0 at decode step t while it was still masked (t = 0..letter_unmask_step).
Option counts come from outputs/hidden_shapley_split_20261010/{model}.jsonl (option_letters).

Each step votes with its leading letter (measure_letter_hidden_delta.leading_letter) if that letter
is one of the item's options; other steps abstain. Ties: the finished letter if it is tied, else
the most recent tied step's letter. No votes: the finished letter.
  all      every recorded step
  last_k   the last k recorded steps, k in {2,4,8}
  from_s   recorded steps t >= s, s in {0,4,8,12}
k / s are chosen on half A and scored on half B and vice versa (seed-0 split as in
hidden_shapley_offline_rules.py); every fixed k / s is also reported on all items (same-set).
Per-step confidences are not stored, so no weighted vote is computed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import measure_letter_hidden_delta as delta  # noqa: E402

TRAJ = PROJECT / "outputs" / "letter_argmax_traj_20261010"
SPLIT = PROJECT / "outputs" / "hidden_shapley_split_20261010"
N_TOTAL = 2017
KS = (2, 4, 8)
SS = (0, 4, 8, 12)


def halves_ids():
    ids = sorted(f"scienceqa_img_test_{i:04d}" for i in range(N_TOTAL))
    perm = np.random.default_rng(0).permutation(len(ids))
    half = (len(ids) + 1) // 2
    return {"A": [ids[i] for i in perm[:half]], "B": [ids[i] for i in perm[half:]]}


def vote(letters: list, finished, options: str):
    counts = Counter(l for l in letters if l in options)
    if not counts:
        return finished
    top = max(counts.values())
    tied = {l for l, c in counts.items() if c == top}
    if len(tied) == 1:
        return next(iter(tied))
    if finished in tied:
        return finished
    for l in reversed(letters):
        if l in tied:
            return l
    raise AssertionError


def rules():
    out = {"all": lambda seq: seq}
    for k in KS:
        out[f"last_{k}"] = (lambda k: lambda seq: seq[-k:])(k)
    for s in SS:
        out[f"from_{s}"] = (lambda s: lambda seq: seq[s:])(s)
    return out


def counts(pred, rows, ids):
    c = {"n": len(ids), "correct": 0, "baseline_correct": 0, "changed": 0, "fixed": 0, "broke": 0}
    for i in ids:
        r = rows[i]
        p, b, g = pred[i], r["finished_letter"], r["gold"]
        c["correct"] += p == g
        c["baseline_correct"] += b == g
        c["changed"] += p != b
        c["fixed"] += (p == g) and (b != g)
        c["broke"] += (p != g) and (b == g)
    return c


def run_model(model: str, halves: dict) -> dict:
    rows = {}
    for line in (TRAJ / f"{model}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r
    opts = {}
    for line in (SPLIT / f"{model}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        opts[r["id"]] = r["option_letters"]
    if len(rows) != N_TOTAL or set(rows) != set(opts):
        raise RuntimeError(f"{model}: {len(rows)} trajectories, ids mismatch with option file")
    for i, r in rows.items():
        r["options"] = opts[i]
        r["leading"] = [delta.leading_letter(t) for t in r["pos0_argmax"]]
        if len(r["pos0_argmax"]) != r["letter_unmask_step"] + 1:
            raise RuntimeError(f"{i} trajectory length")
    all_ids = sorted(rows)
    fns = rules()
    preds = {name: {i: vote(fn(rows[i]["leading"]), rows[i]["finished_letter"], rows[i]["options"])
                    for i in all_ids} for name, fn in fns.items()}

    finished_correct = sum(r["finished_letter"] == r["gold"] for r in rows.values())
    last_step_letter = {i: rows[i]["leading"][-1] for i in all_ids}
    diag = {
        "finished_letter_correct": finished_correct,
        "n_finished_letter_none": sum(r["finished_letter"] is None for r in rows.values()),
        "n_last_recorded_argmax_equals_finished": sum(last_step_letter[i] == rows[i]["finished_letter"] for i in all_ids),
        "letter_unmask_step_hist": dict(sorted(Counter(r["letter_unmask_step"] for r in rows.values()).items())),
        "n_items_with_any_letter_change_in_traj": sum(len({l for l in r["leading"]}) > 1 for r in rows.values()),
        "n_items_with_nonletter_step": sum(any(l is None for l in r["leading"]) for r in rows.values()),
    }
    same_set = {name: counts(p, rows, all_ids) for name, p in preds.items()}

    def crossfit(names):
        folds, tot = [], {"correct": 0, "baseline_correct": 0, "changed": 0, "fixed": 0, "broke": 0}
        for fit, score in (("A", "B"), ("B", "A")):
            fc = {n: counts(preds[n], rows, halves[fit]) for n in names}
            best = max(c["correct"] for c in fc.values())
            cands = [n for n in names if fc[n]["correct"] == best]
            chosen = min(cands, key=lambda n: (fc[n]["changed"], names.index(n)))
            held = counts(preds[chosen], rows, halves[score])
            folds.append({"fit": fit, "score": score, "chosen": chosen, "fit_correct": best,
                          "fit_counts": {n: c["correct"] for n, c in fc.items()},
                          "n_tied_at_best": len(cands), "heldout": held})
            for k in tot:
                tot[k] += held[k]
        return {"folds": folds, "crossfit": tot}

    pred_dist = {name: dict(sorted(Counter(p.get(i) or "none" for i in all_ids).items()))
                 for name, p in [("finished", {i: rows[i]["finished_letter"] for i in all_ids}), *preds.items()]}
    return {
        "diagnostics": diag,
        "same_set_all_2017": same_set,
        "crossfit_last_k": crossfit([f"last_{k}" for k in KS]),
        "crossfit_from_s": crossfit([f"from_{s}" for s in SS]),
        "crossfit_all_families": crossfit(list(fns)),
        "pred_letter_distribution": pred_dist,
        "gold_distribution": dict(sorted(Counter(r["gold"] for r in rows.values()).items())),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    halves = halves_ids()
    res = {
        "inputs": str(TRAJ),
        "split": {"method": "numpy default_rng(0).permutation of sorted ids; A = first ceil(N/2)",
                  "sizes": {k: len(v) for k, v in halves.items()}},
        "baseline": "finished_letter from the stock trajectory decode",
        "vote_rule": "leading letter per step restricted to item options; tie -> finished letter if tied, "
                     "else most recent tied step; no votes -> finished letter",
        "crossfit_tiebreak": "fit-half ties -> fewest changes vs finished on the fit half, then list order",
        "not_done": "confidence-weighted voting: per-step confidences are not stored in the trajectory file; "
                    "needs a re-decode that saves them",
        "models": {m: run_model(m, halves) for m in ("lavida", "mmada")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for m, r in res["models"].items():
        print(m, r["diagnostics"])
        for name, c in r["same_set_all_2017"].items():
            print(f"  same-set {name:8s} {c}")
        for fam in ("crossfit_last_k", "crossfit_from_s", "crossfit_all_families"):
            print(f"  {fam} {r[fam]['crossfit']} chosen={[f['chosen'] for f in r[fam]['folds']]}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
