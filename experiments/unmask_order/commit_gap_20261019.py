#!/usr/bin/env python3
"""Stock letter-commit for the cells missing from the 2026-10-16 supplement.

Writes only under outputs/gap_20261019/. Does not touch earlier jsonl files.
Gen 16, temperature 0, one token per step, no steering and no calibration.

Cells this file is meant to run (the others already have stock letter_unmask_step):
  lladav  mmstar, mmbench, pope
  dreamvl m3cot, mmbench, pope
  mmada   pope
  lavida  pope
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
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import backend_dreamvl_20261013g as bd  # noqa: E402
import calib_decode_dreamvl_20261013g as dv  # noqa: E402
import calib_decode_multi_20261013c as c  # noqa: E402
import calib_decode_multi_20261013e as ee  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "gap_20261019").resolve()
LLADAV_DIR = PROJECT / "checkpoints/LLaDA-V"


class LladavE(c.Lladav):
    """Same forward as LLaDA-V in the 20261013c backend, scored by the 20261013e canvas decoder.

    The E decoder knows both A–E and Yes/No, so POPE and the MCQ sets share one record schema.
    """

    def decode(self, text: str, image) -> dict:
        pe, plen = self.lb.prefix(self.ctx, text, image)
        x = torch.full((1, c.GEN), self.mask_id, dtype=torch.long, device=pe.device)
        decoded = ee.decode_canvas(x, self.mask_id, self.tokenizer, self.lb.make_forward(self.ctx, pe))
        decoded["prefix_len"] = plen
        return decoded


def rows_for(backend: str, dataset: str) -> tuple[list[dict], str]:
    if backend == "dreamvl":
        task = dv.TASK[dataset]
        return dv.DATASETS[dataset]("dreamvl", 0, "real"), task
    if dataset == "mmstar":
        return ee.rows_mmstar(backend, 0, "real"), "mcq"
    if dataset == "mmbench":
        return c.rows_mmbench(backend, 0), "mcq"
    if dataset == "m3cot":
        return c.rows_m3cot(backend, 0), "mcq"
    if dataset == "pope":
        return ee.rows_pope(backend, 0, "real"), "yesno"
    raise RuntimeError(f"no rows for {backend} {dataset}")


def load_backend(backend: str, cuda: int, task: str):
    torch.cuda.set_device(cuda)
    torch.set_grad_enabled(False)
    if backend == "dreamvl":
        be = bd.Dreamvl(argparse.Namespace(model_path=str(bd.MODEL_DIR), cuda=cuda))
        dv.set_labels(be, task)
        ee.STATE["mode"] = "stock"
        ee.STATE["task"] = task
        ee.STATE["alpha_scale"] = 1.0
        return be
    if backend == "lladav":
        be = LladavE(argparse.Namespace(model_path=str(LLADAV_DIR), cuda=cuda))
    else:
        be = ee.load_backend(backend, cuda)
    ee.set_labels(be, task)
    ee.STATE["mode"] = "stock"
    ee.STATE["alpha_scale"] = 1.0
    return be


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("lladav", "dreamvl", "mmada", "lavida"), required=True)
    p.add_argument("--dataset", choices=("mmstar", "mmbench", "m3cot", "pope"), required=True)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = full set; used for smoke")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.cuda not in (0, 1):
        raise RuntimeError("--cuda must be 0 or 1")
    output = args.output or (OUT_ROOT / f"{args.backend}_{args.dataset}_stock.jsonl")
    resolved = output.resolve()
    if resolved != OUT_ROOT and OUT_ROOT not in resolved.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {output}") from exc
    rows, task = rows_for(args.backend, args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    be = load_backend(args.backend, args.cuda, task)
    print(
        f"commit-gap backend={args.backend} dataset={args.dataset} task={task} "
        f"n={len(rows)} done={len(done)} -> {output}",
        flush=True,
    )
    t0 = time.time()
    n_new = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            t1 = time.time()
            ee.STATE["n_opt"] = int(row["n_opt"])
            with torch.no_grad():
                decoded = be.decode(row["text"], row["image"]())
            rec = {
                "id": row["id"],
                "gold": row["gold"],
                "n_opt": int(row["n_opt"]),
                "backend": args.backend,
                "dataset": args.dataset,
                "finished_letter": decoded["finished_letter"],
                "letter_unmask_step": int(decoded["letter_unmask_step"]),
                "last_step": int(decoded["letter_unmask_step"] == c.GEN - 1),
                "finished_text": str(decoded.get("finished_text") or "")[:240],
                "sec": round(time.time() - t1, 3),
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            n_new += 1
            done.add(row["id"])
            print(
                f"PROGRESS {len(done)}/{len(rows)} id={row['id']} letter={rec['finished_letter']} "
                f"unmask={rec['letter_unmask_step']} sec={rec['sec']} rate={n_new / (time.time() - t0):.3f}/s",
                flush=True,
            )
    print(f"GAP DONE backend={args.backend} dataset={args.dataset} n={len(done)} elapsed={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
