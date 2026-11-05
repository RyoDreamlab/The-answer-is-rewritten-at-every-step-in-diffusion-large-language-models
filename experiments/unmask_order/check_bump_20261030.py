#!/usr/bin/env python3
"""Check a bump smoke decode against the stock trajectory."""
from __future__ import annotations

import json
import math
import sys

from release_bump_20261030 import choose_index

rule, stock_path, new_path = sys.argv[1:]


def load(path):
    rows = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def margin(vals):
    if len(vals) < 2:
        return 1.0
    peak = max(vals)
    exps = [math.exp(v - peak) for v in vals]
    total = sum(exps)
    probs = sorted((e / total for e in exps), reverse=True)
    return probs[0] - probs[1]


def cf(text):
    return (text or "").casefold()


stock = load(stock_path)
new = load(new_path)
if not new:
    raise SystemExit("smoke jsonl is empty")
bad = 0
for item_id, row in new.items():
    src = stock[item_id]
    hist = src.get("pos0_argmax") or []
    raw = src.get("letter_logits_raw_at_commit") or []
    n = int(src["n_opt"])
    pick = choose_index(hist, margin(raw[:n]), rule)
    expect = hist[pick]
    got = row.get("finished_letter")
    ok = cf(expect) == cf(got)
    labelish = cf(expect) in {"a", "b", "c", "d", "e", "yes", "no"}
    if labelish and not ok:
        bad += 1
        print(f"MIS {item_id} expect={expect} got={got} text={row.get('finished_text')!r}", flush=True)
    else:
        print(f"OK {item_id} expect={expect} got={got}", flush=True)
if bad:
    raise SystemExit(f"smoke mismatch {bad}")
print(f"SMOKE MATCH n={len(new)} rule={rule}", flush=True)
