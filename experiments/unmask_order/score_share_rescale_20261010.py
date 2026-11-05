#!/usr/bin/env python3
"""Score share_rescale_decode_20261010 outputs against stock (read-only)."""

import json
import os
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "experiments" / "caft"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
from graders import scienceqa_mc  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402

OUT = PROJECT / "outputs" / "share_rescale_decode_20261010"
TRAJ = PROJECT / "outputs" / "letter_argmax_traj_20261010"
N = 2017


def jl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def pct(k):
    return f"{k}/{N} ({100 * k / N:.2f}%)"


def official_row(row):
    gold = row.get("gold") or row.get("gold_option") or row.get("answer")
    if row.get("gold") and (row.get("gold_text") or row.get("golds")):
        return row
    choices = row.get("choices")
    idx = row.get("answer_idx", row.get("answer_index"))
    if isinstance(choices, list) and idx is not None:
        return {"gold": str(gold).strip(), "gold_text": str(choices[int(idx)])}
    return None


def mmada_stock_official():
    import sciqa_standard_eval as sq
    rows = sq.load_manifest(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json",
                            PROJECT / "datasets/sciqa_img_standard_v1")
    by = {r["id"]: r for r in rows}
    base = jl(PROJECT / "outputs/sciqa_standard_baseline_mmada.jsonl")
    if len(base) != N:
        return None, f"baseline has {len(base)} rows"
    ok = 0
    letter = 0
    for b in base:
        g = official_row(by[b["id"]])
        if g is None:
            return None, f"no grader row for {b['id']}"
        ok += int(bool(scienceqa_mc(b["prediction_text"], g)))
        letter += int(delta.leading_letter(b["prediction_text"]) == g["gold"])
    return ok, f"baseline leading-letter={letter}"


def main():
    res = {}
    for m in sys.argv[1:] or ["lavida", "mmada"]:
        rows = jl(OUT / f"{m}.jsonl")
        stock = {r["id"]: r for r in jl(TRAJ / f"{m}.jsonl")}
        n = len(rows)
        letter = sum(int(r["finished_letter_correct"]) for r in rows)
        off = sum(int(bool(r["scienceqa_mc"])) for r in rows)
        off_scored = sum(int(r["scienceqa_mc"] is not None) for r in rows)
        stock_letter = sum(int(s["finished_letter"] == s["gold"]) for s in stock.values())
        changed = fixed = broke = 0
        for r in rows:
            s = stock[r["id"]]
            if r["finished_letter"] != s["finished_letter"]:
                changed += 1
                fixed += int(r["finished_letter"] == r["gold"])
                broke += int(s["finished_letter"] == s["gold"])
        step_changed = sum(int(r["letter_changed_vs_unmod_at_unmask"]) for r in rows)
        skipped = [r["id"] for r in rows if not r["modified"]]
        res[m] = {"n": n, "letter": letter, "official": off, "official_scored": off_scored,
                  "stock_letter_from_traj": stock_letter, "changed_vs_stock": changed, "fixed": fixed,
                  "broke": broke, "letter_changed_at_unmask_step": step_changed, "skipped": skipped}
        print(f"{m}: n={n} letter={pct(letter) if n == N else letter} official={pct(off) if n == N else off} "
              f"(scored {off_scored}) stock_letter_traj={stock_letter} changed={changed} fixed={fixed} broke={broke} "
              f"step_changed={step_changed} skipped={len(skipped)}")
    if "mmada" in res:
        k, note = mmada_stock_official()
        res["mmada_stock_official"] = {"count": k, "note": note}
        print(f"mmada stock official: {pct(k) if k is not None else 'unavailable'} {note}")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
