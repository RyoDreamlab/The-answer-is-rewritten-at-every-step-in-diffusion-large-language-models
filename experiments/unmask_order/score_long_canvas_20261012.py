#!/usr/bin/env python3
"""Score long_canvas_20261012 outputs against stock (read-only on inputs).

Usage: score_long_canvas_20261012.py <model> <tag>   e.g. lavida c32
       score_long_canvas_20261012.py --sanity <model> <path>
"""

import json
import os
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "experiments" / "caft"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
from graders import scienceqa_mc  # noqa: E402

OUT = PROJECT / "outputs" / "long_canvas_20261012"
TRAJ = PROJECT / "outputs" / "letter_argmax_traj_20261010"
STOCK_TEXT = {
    "lavida": (PROJECT / "outputs/full_eval_table1_stock_a_scienceqa_20260922/eval_arm_A_scienceqa.jsonl", "pred"),
    "mmada": (PROJECT / "outputs/sciqa_standard_baseline_mmada.jsonl", "prediction_text"),
}
STOCK = {"lavida": (1460, 1454), "mmada": (1059, 1059)}
N = 2017


def jl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def pct(k, n=N):
    return f"{k}/{n} ({100 * k / n:.2f}%)"


def grader_rows():
    rows = json.loads((PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json").read_text(encoding="utf-8"))
    return {r["id"]: {"gold": str(r["gold"]).strip(), "gold_text": str(r["gold_text"])} for r in rows}


def stock_text(model):
    path, key = STOCK_TEXT[model]
    return {r["id"]: r[key] for r in jl(path)}


def sanity(model, path):
    rows = jl(path)
    tr = {r["id"]: r for r in jl(TRAJ / f"{model}.jsonl")}
    st = stock_text(model)
    bad = 0
    for r in rows:
        t = tr[r["id"]]
        ok_letter = r["finished_letter"] == t["finished_letter"]
        ok_step = r["letter_unmask_step"] == t["letter_unmask_step"] and r["pos0_argmax"] == t["pos0_argmax"]
        ok_text = r["finished_text"].strip() == str(st[r["id"]]).strip()
        bad += int(not (ok_letter and ok_step and ok_text))
        print(f"{model} {r['id']} text={r['finished_text']!r} stock_text={st[r['id']]!r} "
              f"letter={r['finished_letter']}/{t['finished_letter']} step={r['letter_unmask_step']}/{t['letter_unmask_step']} "
              f"match_letter={ok_letter} match_traj={ok_step} match_text={ok_text}")
    print(f"SANITY {model} n={len(rows)} mismatches={bad}")
    return bad


def score(model, tag):
    rows = jl(OUT / f"{model}_{tag}.jsonl")
    if len(rows) != N or len({r["id"] for r in rows}) != N:
        raise SystemExit(f"{model}_{tag}: {len(rows)} rows")
    g = grader_rows()
    tr = {r["id"]: r for r in jl(TRAJ / f"{model}.jsonl")}
    letter = sum(int(r["finished_letter"] == g[r["id"]]["gold"]) for r in rows)
    off = sum(int(bool(scienceqa_mc(r["finished_text"], g[r["id"]]))) for r in rows)
    changed = fixed = broke = 0
    for r in rows:
        s = tr[r["id"]]
        gold = g[r["id"]]["gold"]
        if r["finished_letter"] != s["finished_letter"]:
            changed += 1
            fixed += int(r["finished_letter"] == gold)
            broke += int(s["finished_letter"] == gold)
    no_letter = sum(int(r["finished_letter"] is None) for r in rows)
    sl, so = STOCK[model]
    res = {"model": model, "config": tag, "n": N, "letter": letter, "letter_pct": round(100 * letter / N, 2),
           "official": off, "official_pct": round(100 * off / N, 2),
           "stock_letter": sl, "stock_official": so,
           "delta_letter": letter - sl, "delta_official": off - so,
           "changed_vs_stock_letter": changed, "fixed": fixed, "broke": broke, "no_leading_letter": no_letter}
    print(f"{model} {tag}: letter={pct(letter)} (stock {pct(sl)}) official={pct(off)} (stock {pct(so)}) "
          f"changed={changed} fixed={fixed} broke={broke} no_leading_letter={no_letter}")
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "--sanity":
        sys.exit(1 if sanity(sys.argv[2], sys.argv[3]) else 0)
    score(sys.argv[1], sys.argv[2])
