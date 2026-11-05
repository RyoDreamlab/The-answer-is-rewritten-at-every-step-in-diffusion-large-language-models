#!/usr/bin/env python3
"""Remaining CVPR decodes. New file. Writes only under outputs/supplement_20261016/.

Parts
  mmstar   MMaDA, all 1500 MMStar items: real stock / step0 / every (writer heads, β=+8),
           gray stock, pixel-shuffled stock, and every-step random-head steer at the same residual norm.
  prompt   MMaDA, first --n-sciqa ScienceQA items, two extra answer instructions, stock / step0 / every.
  commit   MMaDA, ScienceQA first --n-sciqa and all MMStar. Decode until the answer slot is chosen, then
           one extra forward of that same canvas with a gray image and with a pixel-shuffled image.
  lavida   LaViDa stock gen-16 on all MMStar. Records whether the letter unmasks at the last step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import calib_decode_multi_20261013e as ee  # noqa: E402
import run_unmask_order as base  # noqa: E402
import sciqa_standard_eval as sq  # noqa: E402
import supplement_20261016 as S  # noqa: E402

OUT = S.OUT_ROOT
PROMPTS = {
    "letter_only": "\nReply with the letter only.",
    "nothing_else": "\nGive the option letter and nothing else.",
}


def shuffled(image, item_id: str):
    from PIL import Image

    arr = np.asarray(image.convert("RGB")).copy()
    rng = np.random.default_rng(zlib.crc32(item_id.encode()) & 0xFFFFFFFF)
    flat = arr.reshape(-1, arr.shape[-1])
    flat[:] = flat[rng.permutation(flat.shape[0])]
    return Image.fromarray(arr)


def mmstar_jobs(ctx, lo: int = 0, hi: int = 1500) -> list[dict]:
    rows = S.mmstar_rows(1500)
    if not (0 <= lo < hi <= len(rows)):
        raise RuntimeError(f"bad mmstar slice {lo}:{hi} of {len(rows)}")
    jobs = []
    for row in rows[lo:hi]:
        ch = S.choices_of(row)
        gold = S.gold_of(row)
        question = sq.build_task_prompt(row)
        labels = "ABCDE"[: len(ch)]
        common = {
            "id": row["id"], "question": question, "labels": labels, "gold": gold, "gold_mapped": gold,
            "gen": 16, "canvas_pos": 0, "prefill": None, "heads": "writer",
        }
        stock = {**common, "factor": "real", "cond": "stock", "image": row["_image"]}
        jobs.append(stock)
        for cond in ("step0", "every"):
            jobs.append({**common, "factor": "real", "cond": cond, "image": row["_image"], "stock_from": stock})
        jobs.append({**common, "factor": "gray", "cond": "stock", "image": S.gray_image})
        jobs.append({
            **common, "factor": "shuffle", "cond": "stock",
            "image": (lambda rec=row: shuffled(rec["_image"](), rec["id"])),
        })
        jobs.append({
            **common, "factor": "real_random", "cond": "every", "heads": "random",
            "image": row["_image"], "stock_from": stock,
        })
    return jobs


def prompt_jobs(ctx, n: int) -> list[dict]:
    jobs = []
    for row in S.sciqa_rows(ctx, n):
        ch = S.choices_of(row)
        gold = S.gold_of(row)
        base_q = sq.build_task_prompt(row)
        image = lambda rec=row: ctx["store"].load_rgb(rec)
        labels = "ABCDE"[: len(ch)]
        for name, suffix in PROMPTS.items():
            stock = {
                "id": row["id"], "factor": name, "cond": "stock", "question": base_q + suffix,
                "labels": labels, "gold": gold, "gold_mapped": gold, "gen": 16, "canvas_pos": 0,
                "prefill": None, "heads": "writer", "image": image,
            }
            jobs.append(stock)
            for cond in ("step0", "every"):
                jobs.append({**stock, "cond": cond, "stock_from": stock})
    return jobs


def order_jobs(jobs: list[dict]) -> list[dict]:
    holders = {}
    ordered = []
    for job in jobs:
        if job["cond"] == "stock":
            holders[(job["id"], job["factor"])] = job
            ordered.append(job)
        else:
            holder = holders.get((job["id"], job["factor"]))
            if holder is None and job["factor"] == "real_random":
                holder = holders.get((job["id"], "real"))
            if holder is None:
                raise RuntimeError(f"stock missing for {job['id']} {job['factor']}")
            job["stock_holder"] = holder
            ordered.append(job)
    return ordered


def restricted(logits_row, ids, labels) -> str:
    vals = [float(logits_row[i]) for i in ids]
    return labels[max(range(len(vals)), key=lambda j: (vals[j], -j))]


def commit_one(ctx, item_id: str, question: str, image_real, gold: str, n_opt: int) -> dict:
    labels = "ABCDE"[:n_opt]
    ids = ctx["abc"][:n_opt]
    gen = 16
    mask_id = ctx["mask_id"]
    fwd_real, _ = S.make_forward(ctx, question, image_real, gen)
    x = torch.full((1, gen), mask_id, dtype=torch.long, device=ctx["device"])
    pre = None
    real_letter = None
    real_argmax = None
    commit_step = None
    for step in range(gen):
        logits = fwd_real(x)
        if int(x[0, 0].item()) == int(mask_id):
            real_letter = restricted(logits[0, 0], ids, labels)
            real_argmax = int(torch.argmax(logits[0, 0]).item())
        pos = base.unmask_one(x, logits, mask_id)
        if pos == 0:
            pre = x.clone()
            pre[0, 0] = mask_id
            commit_step = step
            break
    if pre is None:
        raise RuntimeError(f"{item_id} never unmasked slot 0")
    gray = S.gray_image()
    shuf = shuffled(image_real, item_id)
    out = {"id": item_id, "gold": gold, "n_opt": n_opt, "commit_step": int(commit_step),
           "real_letter": real_letter, "real_argmax": real_argmax}
    for name, image in (("gray", gray), ("shuffle", shuf)):
        fwd, _ = S.make_forward(ctx, question, image, gen)
        logits = fwd(pre)
        letter = restricted(logits[0, 0], ids, labels)
        argmax = int(torch.argmax(logits[0, 0]).item())
        out[f"{name}_letter"] = letter
        out[f"{name}_argmax"] = argmax
        out[f"{name}_letter_changed"] = int(letter != real_letter)
        out[f"{name}_argmax_changed"] = int(argmax != real_argmax)
    return out


def run_commit(ctx, n_sciqa: int, output: Path) -> None:
    S.check_out(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["set"], r["id"]))
    rows = [("sciqa", r, lambda rec=r: ctx["store"].load_rgb(rec), sq.build_task_prompt(r))
            for r in S.sciqa_rows(ctx, n_sciqa)]
    rows += [("mmstar", r, r["_image"], sq.build_task_prompt(r)) for r in S.mmstar_rows(1500)]
    print(f"commit jobs {len(rows)} done {len(done)} -> {output}", flush=True)
    t0 = time.time()
    n_new = 0
    with output.open("a", encoding="utf-8") as handle:
        for name, row, image_fn, question in rows:
            if (name, row["id"]) in done:
                continue
            t1 = time.time()
            rec = commit_one(ctx, row["id"], question, image_fn(), S.gold_of(row), len(S.choices_of(row)))
            rec["set"] = name
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            n_new += 1
            el = time.time() - t0
            print(f"PROGRESS commit {name} {row['id']} step={rec['commit_step']} real={rec['real_letter']} "
                  f"gray_changed={rec['gray_letter_changed']} shuf_changed={rec['shuffle_letter_changed']} "
                  f"sec={time.time()-t1:.1f} rate={n_new/el:.3f}/s", flush=True)


def run_lavida(output: Path, cuda: int) -> None:
    S.check_out(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    rows = ee.rows_mmstar("lavida", 0, "real")
    torch.cuda.set_device(cuda)
    torch.set_grad_enabled(False)
    args = argparse.Namespace(
        cuda=cuda, model_path=str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct"),
        vq_model="", mmada_repo="",
    )
    be = ee.LavidaE(args)
    ee.STATE["mode"] = "stock"
    ee.STATE["task"] = "mcq"
    ee.STATE["labels"] = list("ABCDE")
    ee.STATE["label_ids"] = be.letter_ids()
    ee.STATE["alpha_scale"] = 1.0
    print(f"lavida mmstar n={len(rows)} done={len(done)} -> {output}", flush=True)
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
                "id": row["id"], "gold": row["gold"], "n_opt": row["n_opt"], "backend": "lavida",
                "finished_letter": decoded.get("finished_letter"),
                "letter_unmask_step": decoded["letter_unmask_step"],
                "last_step": int(decoded["letter_unmask_step"] == 15),
                "sec": round(time.time() - t1, 3),
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            n_new += 1
            print(f"PROGRESS lavida {len(done)+n_new}/{len(rows)} id={row['id']} "
                  f"unmask={rec['letter_unmask_step']} sec={rec['sec']}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("mmstar", "prompt", "commit", "lavida"), required=True)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--n-sciqa", type=int, default=400)
    p.add_argument("--lo", type=int, default=0, help="first MMStar row index, inclusive")
    p.add_argument("--hi", type=int, default=1500, help="last MMStar row index, exclusive")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.cuda not in (0, 1):
        raise RuntimeError("--cuda must be 0 or 1")
    defaults = {
        "mmstar": OUT / "mmstar_full.jsonl",
        "prompt": OUT / "prompt.jsonl",
        "commit": OUT / "commit_swap.jsonl",
        "lavida": OUT / "lavida_mmstar_stock.jsonl",
    }
    output = args.output or defaults[args.part]
    S.check_out(output)
    if args.part == "lavida":
        run_lavida(output, args.cuda)
        return
    ctx = S.load_mmada(args.cuda)
    if args.part == "commit":
        run_commit(ctx, args.n_sciqa, output)
        return
    if args.part == "mmstar":
        jobs = mmstar_jobs(ctx, args.lo, args.hi)
    else:
        jobs = prompt_jobs(ctx, args.n_sciqa)
    S.run_jobs(ctx, order_jobs(jobs), output)


if __name__ == "__main__":
    main()
