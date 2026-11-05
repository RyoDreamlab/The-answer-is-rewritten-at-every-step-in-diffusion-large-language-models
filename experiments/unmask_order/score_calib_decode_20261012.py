#!/usr/bin/env python3
"""Score calib_decode_20261012 outputs against stock (read-only on inputs).

Usage: score_calib_decode_20261012.py <model>
       score_calib_decode_20261012.py --sanity <model> <path>
Grading is the same as score_long_canvas_20261012.py (leading letter + graders.scienceqa_mc,
gold rows from datasets/urd_eval_full, changed/fixed/broke vs letter_argmax_traj_20261010).
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import score_long_canvas_20261012 as slc  # noqa: E402

OUT = PROJECT / "outputs" / "calib_decode_20261012"
N = slc.N


def score(model):
    rows = slc.jl(OUT / f"{model}.jsonl")
    if len(rows) != N or len({r["id"] for r in rows}) != N:
        raise SystemExit(f"{model}: {len(rows)} rows")
    if not all(r["calib"] for r in rows):
        raise SystemExit(f"{model}: uncalibrated rows present")
    g = slc.grader_rows()
    tr = {r["id"]: r for r in slc.jl(slc.TRAJ / f"{model}.jsonl")}
    letter = sum(int(r["finished_letter"] == g[r["id"]]["gold"]) for r in rows)
    off = sum(int(bool(slc.scienceqa_mc(r["finished_text"], g[r["id"]]))) for r in rows)
    changed = fixed = broke = 0
    for r in rows:
        s = tr[r["id"]]
        gold = g[r["id"]]["gold"]
        if r["finished_letter"] != s["finished_letter"]:
            changed += 1
            fixed += int(r["finished_letter"] == gold)
            broke += int(s["finished_letter"] == gold)
    stock_text = slc.stock_text(model)
    off_changed = off_fixed = off_broke = 0
    for r in rows:
        a = bool(slc.scienceqa_mc(r["finished_text"], g[r["id"]]))
        b = bool(slc.scienceqa_mc(str(stock_text[r["id"]]), g[r["id"]]))
        off_changed += int(a != b)
        off_fixed += int(a and not b)
        off_broke += int(b and not a)
    step0_raw = sum(int(r["step0_restricted_raw"] == g[r["id"]]["gold"]) for r in rows)
    step0_cal = sum(int(r["step0_restricted_cal"] == g[r["id"]]["gold"]) for r in rows)
    commit_eq_step0cal = sum(int(r["finished_letter"] == r["step0_restricted_cal"]) for r in rows)
    sl, so = slc.STOCK[model]
    res = {
        "model": model, "n": N, "letter": letter, "letter_pct": round(100 * letter / N, 2),
        "official": off, "official_pct": round(100 * off / N, 2),
        "stock_letter": sl, "stock_official": so,
        "delta_letter": letter - sl, "delta_letter_pp": round(100 * (letter - sl) / N, 2),
        "delta_official": off - so, "delta_official_pp": round(100 * (off - so) / N, 2),
        "changed_vs_stock_letter": changed, "fixed": fixed, "broke": broke,
        "official_changed_vs_stock_text": off_changed, "official_fixed": off_fixed, "official_broke": off_broke,
        "no_leading_letter": sum(int(r["finished_letter"] is None) for r in rows),
        "step0_restricted_raw_correct": step0_raw, "step0_restricted_cal_correct": step0_cal,
        "finished_letter_equals_step0_cal": commit_eq_step0cal,
        "letter_unmask_step_dist": dict(sorted(Counter(r["letter_unmask_step"] for r in rows).items())),
        "stock_letter_unmask_step_dist": dict(sorted(Counter(s["letter_unmask_step"] for s in tr.values()).items())),
        "finished_letter_dist": dict(sorted(Counter(str(r["finished_letter"]) for r in rows).items())),
        "stock_finished_letter_dist": dict(sorted(Counter(str(s["finished_letter"]) for s in tr.values()).items())),
    }
    print(f"{model}: letter={slc.pct(letter)} (stock {slc.pct(sl)}) official={slc.pct(off)} (stock {slc.pct(so)}) "
          f"changed={changed} fixed={fixed} broke={broke} step0 raw={step0_raw} cal={step0_cal}")
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "--sanity":
        sys.exit(1 if slc.sanity(sys.argv[2], sys.argv[3]) else 0)
    score(sys.argv[1])
