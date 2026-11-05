#!/usr/bin/env python3
"""Offline delta on stock jsonl, or smoke check of a new jsonl.

offline: check_retune2_20261030.py offline lavida_mme <stock.jsonl>
         check_retune2_20261030.py offline dream_vstar <stock.jsonl>
smoke:   check_retune2_20261030.py smoke <rule> <stock.jsonl> <new.jsonl>
"""
from __future__ import annotations

import json
import re
import sys

from release_retune2_20261030 import choose_index


def load_list(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_map(path):
    return {row["id"]: row for row in load_list(path)}


def cf(text):
    return (text or "").casefold()


def gap_and_argmax(row):
    n = int(row["n_opt"])
    logits = [float(v) for v in row["letter_logits_raw_at_commit"][:n]]
    labels = row["labels"][:n]
    order = sorted(range(n), key=lambda j: (logits[j], -j), reverse=True)
    return logits[order[0]] - logits[order[1]], labels[order[0]]


def scored_piece(piece, kind):
    if kind == "yesno":
        match = re.match(r"^(yes|no)(?:$|[\s\.\,\!\:\)])", (piece or "").strip(), re.I)
        return match.group(1).capitalize() if match else None
    match = re.match(r"^([A-E])(?:$|[\s\.\:\)])", (piece or "").strip())
    return match.group(1) if match else None


def rescore_text(text, kind):
    return scored_piece(text, kind)


def offline(rule, stock_path):
    kind = "yesno" if rule == "lavida_mme" else "mcq"
    rows = load_list(stock_path)
    base = new = 0
    changed = 0
    for row in rows:
        gap, am = gap_and_argmax(row)
        pick = choose_index(row["pos0_argmax"], gap, am, rule)
        letter = scored_piece(row["pos0_argmax"][pick], kind)
        if letter is None:
            letter = row.get("finished_letter")
        base += row.get("finished_letter") == row.get("gold")
        # rescore stock text too, so the delta is content scoring
        stock_letter = rescore_text(row.get("finished_text"), kind)
        if pick == len(row["pos0_argmax"]) - 1:
            out = stock_letter
        else:
            out = letter
            if cf(out) != cf(stock_letter or ""):
                changed += 1
        new += out == row.get("gold")
    print(f"OFFLINE rule={rule} n={len(rows)} stock={base} new={new} delta={new - base} changed={changed}", flush=True)
    if new <= base:
        raise SystemExit("offline delta is not strictly positive")


def smoke(rule, stock_path, new_path):
    kind = "yesno" if rule == "lavida_mme" else "mcq"
    stock = load_map(stock_path)
    new = load_map(new_path)
    if len(new) != 5:
        raise SystemExit(f"smoke expected 5 rows, got {len(new)}")
    bad = 0
    for item_id, row in new.items():
        src = stock[item_id]
        gap, am = gap_and_argmax(src)
        hist = src["pos0_argmax"]
        pick = choose_index(hist, gap, am, rule)
        expect = scored_piece(hist[pick], kind)
        got = rescore_text(row.get("finished_text"), kind)
        traj_ok = row.get("pos0_argmax") == hist
        if got != expect or not traj_ok or row.get("finished_letter") != got:
            bad += 1
            print(
                f"MIS {item_id} expect={expect} text_letter={got} field={row.get('finished_letter')} "
                f"traj_ok={traj_ok} text={row.get('finished_text')!r} pick={pick}",
                flush=True,
            )
        else:
            print(f"OK {item_id} letter={got} pick={pick} replaced={row.get('retune_replaced')}", flush=True)
    if bad:
        raise SystemExit(f"smoke mismatch {bad}")
    print(f"SMOKE MATCH n=5 rule={rule}", flush=True)


def main():
    mode = sys.argv[1]
    if mode == "offline":
        offline(sys.argv[2], sys.argv[3])
    elif mode == "smoke":
        smoke(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
