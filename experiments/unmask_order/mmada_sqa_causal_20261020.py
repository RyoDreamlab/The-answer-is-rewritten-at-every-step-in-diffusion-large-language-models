#!/usr/bin/env python3
"""MMaDA ScienceQA writer-head steer, same site and beta as the MMStar full table.

Per item: real stock, writer step-0 beta=+8, writer every-step beta=+8, random-head every-step beta=+8.
Heads are the P6 writer heads already used for MMaDA MMStar. New files only, under
outputs/mmada_causal_sqa_20261020/. Resumes by skipping finished (id, factor, cond) keys.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import supplement_20261016 as S  # noqa: E402

OUT = (PROJECT / "outputs" / "mmada_causal_sqa_20261020").resolve()
OUTPUT = OUT / "steer.jsonl"


def allow_out(path: Path) -> None:
    resolved = Path(path).resolve()
    if resolved == OUT or OUT in resolved.parents:
        return
    raise RuntimeError(f"refusing to write outside {OUT}: {path}")


S.check_out = allow_out


def jobs_for(ctx) -> list[dict]:
    jobs = []
    holders = {}
    for row in S.sciqa_rows(ctx, 2017):
        choices = S.choices_of(row)
        gold = S.gold_of(row)
        labels = "ABCDE"[: len(choices)]
        if gold not in labels:
            raise RuntimeError(f"bad gold {row['id']} {gold}")
        image = lambda rec=row: ctx["store"].load_rgb(rec)
        question = S.sq.build_task_prompt(row)
        common = {
            "id": row["id"], "question": question, "labels": labels, "gold": gold, "gold_mapped": gold,
            "gen": 16, "canvas_pos": 0, "prefill": None, "image": image,
        }
        stock = {**common, "factor": "real", "cond": "stock", "heads": "writer"}
        jobs.append(stock)
        holders[(row["id"], "real")] = stock
        for cond in ("step0", "every"):
            jobs.append({**common, "factor": "real", "cond": cond, "heads": "writer", "stock_holder": stock})
        jobs.append({
            **common, "factor": "real_random", "cond": "every", "heads": "random", "stock_holder": stock,
        })
    return jobs


def report(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {}
    for rec in rows:
        by.setdefault((rec["factor"], rec["cond"]), {})[rec["id"]] = rec
    stock = by.get(("real", "stock"), {})
    lines = [f"# MMaDA ScienceQA writer steer", f"n_stock={len(stock)}", ""]
    last = {i for i, rec in stock.items() if int(rec.get("last_step", 0)) == 1}
    lines.append(f"stock letter at last step: {len(last)}/{len(stock)}")
    for factor, cond in (("real", "step0"), ("real", "every"), ("real_random", "every")):
        got = by.get((factor, cond), {})
        both = [i for i in stock if i in got]
        last_ids = [i for i in both if i in last]
        changed = [i for i in both if got[i].get("changed") == 1]
        changed_last = [i for i in last_ids if got[i].get("changed") == 1]
        hit = sum(1 for i in changed if got[i].get("hit_target") == 1)
        hit_last = sum(1 for i in changed_last if got[i].get("hit_target") == 1)
        lines.append(
            f"{factor}/{cond}: changed {len(changed)}/{len(both)}, hit {hit}/{len(changed) or 0}; "
            f"last-step changed {len(changed_last)}/{len(last_ids)}, hit {hit_last}/{len(changed_last) or 0}"
        )
    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    cuda = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if cuda not in (0, 1):
        raise RuntimeError("cuda must be 0 or 1")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(cuda)
    ctx = S.load_mmada(cuda)
    S.run_jobs(ctx, jobs_for(ctx), OUTPUT)
    report(OUTPUT)


if __name__ == "__main__":
    main()
