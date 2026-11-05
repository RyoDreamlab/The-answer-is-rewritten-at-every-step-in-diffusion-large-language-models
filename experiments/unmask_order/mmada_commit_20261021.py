#!/usr/bin/env python3
"""MMaDA MMStar writer-head edit at the stock commit step only.

Same heads and beta as the MMStar full table. Per item: stock, writer commit, random commit.
New files only, under outputs/causal_commit_20261021/mmada/.
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

OUT = (PROJECT / "outputs" / "causal_commit_20261021" / "mmada").resolve()
OUTPUT = OUT / "steer.jsonl"


def allow_out(path: Path) -> None:
    resolved = Path(path).resolve()
    if resolved == OUT or OUT in resolved.parents:
        return
    raise RuntimeError(f"refusing to write outside {OUT}: {path}")


S.check_out = allow_out


def jobs_for() -> list[dict]:
    jobs = []
    for row in S.mmstar_rows(1500):
        choices = S.choices_of(row)
        gold = S.gold_of(row)
        labels = "ABCDE"[: len(choices)]
        question = S.sq.build_task_prompt(row)
        common = {
            "id": row["id"], "question": question, "labels": labels, "gold": gold, "gold_mapped": gold,
            "gen": 16, "canvas_pos": 0, "prefill": None, "image": row["_image"],
        }
        stock = {**common, "factor": "real", "cond": "stock", "heads": "writer"}
        jobs.append(stock)
        jobs.append({**common, "factor": "real", "cond": "commit", "heads": "writer", "stock_holder": stock})
        jobs.append({**common, "factor": "real_random", "cond": "commit", "heads": "random", "stock_holder": stock})
    return jobs


def report(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {}
    for rec in rows:
        by.setdefault((rec["factor"], rec["cond"]), {})[rec["id"]] = rec
    stock = by.get(("real", "stock"), {})
    last = {i for i, rec in stock.items() if int(rec.get("last_step", 0)) == 1}
    lines = [f"# MMaDA MMStar commit-step writer edit", f"n_stock={len(stock)} last_step={len(last)}", ""]
    for factor in ("real", "real_random"):
        got = by.get((factor, "commit"), {})
        both = [i for i in last if i in got]
        changed = [i for i in both if got[i].get("changed") == 1]
        hit = sum(1 for i in changed if got[i].get("hit_target") == 1)
        lines.append(f"{factor}/commit last-step: changed {len(changed)}/{len(both)}, hit {hit}/{len(changed) or 0}")
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
    S.run_jobs(ctx, jobs_for(), OUTPUT)
    report(OUTPUT)


if __name__ == "__main__":
    main()
