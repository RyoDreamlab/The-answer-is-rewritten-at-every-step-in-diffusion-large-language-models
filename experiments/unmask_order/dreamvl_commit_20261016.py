#!/usr/bin/env python3
"""Dream-VL stock decodes: does the letter unmask at the last step?

First --limit ScienceQA-IMG items, gen 16, temperature 0, one token per step.
No steering, no calibration. Writes only under outputs/supplement_20261016/.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import backend_dreamvl_20261013g as bd  # noqa: E402
import calib_decode_dreamvl_20261013g as dv  # noqa: E402
import calib_decode_multi_20261013e as ee  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "supplement_20261016").resolve()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--dataset", choices=("sciqa", "mmstar"), default="sciqa")
    p.add_argument("--limit", type=int, default=0, help="0 = full set")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.cuda not in (0, 1):
        raise RuntimeError("--cuda must be 0 or 1")
    if args.output is None:
        args.output = OUT_ROOT / f"dreamvl_{args.dataset}_stock.jsonl"
    resolved = args.output.resolve()
    if resolved != OUT_ROOT and OUT_ROOT not in resolved.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix(args.output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {args.output}") from exc
    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    rows = dv.DATASETS[args.dataset]("dreamvl", args.limit, "real")
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    be = bd.Dreamvl(argparse.Namespace(model_path=str(bd.MODEL_DIR), cuda=args.cuda))
    dv.set_labels(be, "mcq")
    ee.STATE["mode"] = "stock"
    ee.STATE["task"] = "mcq"
    ee.STATE["alpha_scale"] = 1.0
    print(f"dreamvl commit dataset={args.dataset} n={len(rows)} done={len(done)} gen=16 -> {args.output}", flush=True)
    last = 0
    t0 = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            t1 = time.time()
            ee.STATE["n_opt"] = int(row["n_opt"])
            with torch.no_grad():
                decoded = be.decode(row["text"], row["image"]())
            rec = {
                "id": row["id"], "gold": row["gold"], "n_opt": row["n_opt"], "backend": "dreamvl",
                "finished_letter": decoded["finished_letter"],
                "letter_unmask_step": decoded["letter_unmask_step"],
                "last_step": int(decoded["letter_unmask_step"] == 15),
                "finished_text": decoded["finished_text"][:240],
                "sec": round(time.time() - t1, 3),
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            last += int(rec["last_step"])
            done.add(row["id"])
            print(
                f"PROGRESS {len(done)}/{len(rows)} id={row['id']} letter={rec['finished_letter']} "
                f"unmask={rec['letter_unmask_step']} sec={rec['sec']}",
                flush=True,
            )
    n = len(done)
    print(f"DREAMVL DONE n={n} last_step={last} new_in_this_process elapsed={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
