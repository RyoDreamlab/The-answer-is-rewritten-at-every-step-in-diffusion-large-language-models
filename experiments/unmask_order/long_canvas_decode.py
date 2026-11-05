#!/usr/bin/env python3
"""Stock low-confidence decode on ScienceQA-IMG with a longer answer canvas.

Reuses letter_argmax_traj prompt/forward code and run_unmask_order.unmask_one.
gen_length == steps == block_length (one token per step, single block),
temperature 0. No edits, no overrides, no weight updates.
Per item: finished text, finished letter, position-0 unmask step, and the
position-0 argmax at every step while position 0 is masked.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import letter_argmax_traj as traj  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "long_canvas_20261012").resolve()


def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
    gen = base.GEN
    if x.shape[1] != gen:
        raise RuntimeError(f"canvas {tuple(x.shape)} != gen {gen}")
    unmask_step = [-1] * gen
    pos0_argmax = []
    for step in range(gen):
        logits = forward(x)
        if int(x[0, 0].item()) == int(mask_id):
            token_id = int(torch.argmax(logits[0, 0]).item())
            pos0_argmax.append(base.piece(tokenizer, token_id, mask_id))
        pos = base.unmask_one(x, logits, mask_id)
        unmask_step[pos] = step
    letter_step = unmask_step[0]
    if letter_step < 0:
        raise RuntimeError("letter position was never selected")
    if len(pos0_argmax) != letter_step + 1:
        raise RuntimeError(f"recorded {len(pos0_argmax)} argmax steps, letter unmasked at {letter_step}")
    if sorted(unmask_step) != list(range(gen)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return {
        "letter_unmask_step": int(letter_step),
        "pos0_argmax": pos0_argmax,
        "finished_text": pred,
        "finished_letter": delta.leading_letter(pred),
    }


def make_loop(gen: int, limit: int):
    def loop(args, rows, fn) -> None:
        if limit:
            rows = rows[:limit]
        done = base.load_done(args.output)
        finished = len(done)
        t0 = time.time()
        n_new = 0
        print(f"resume n_done={finished}/{len(rows)} gen={gen}", flush=True)
        with args.output.open("a", encoding="utf-8") as handle:
            for row in rows:
                if row["id"] in done:
                    continue
                with torch.no_grad():
                    decoded = fn(row)
                record = {
                    "id": row["id"],
                    "gold": traj.gold_of(row),
                    "backend": args.backend,
                    "gen_length": gen,
                    "steps": gen,
                    "block_length": gen,
                    "finished_text": decoded["finished_text"],
                    "finished_letter": decoded["finished_letter"],
                    "letter_unmask_step": decoded["letter_unmask_step"],
                    "pos0_argmax": decoded["pos0_argmax"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.add(record["id"])
                finished += 1
                n_new += 1
                el = time.time() - t0
                rate = n_new / el if el > 0 else 0.0
                payload = {
                    "backend": args.backend,
                    "gen_length": gen,
                    "n_done": finished,
                    "n_total": len(rows),
                    "id": record["id"],
                    "letter_unmask_step": record["letter_unmask_step"],
                    "elapsed_sec": round(el, 1),
                    "items_per_sec": round(rate, 4),
                    "eta_sec": round((len(rows) - finished) / rate, 0) if rate > 0 else None,
                }
                base.write_progress(args.progress, payload)
                print(
                    f"PROGRESS n_done={finished}/{len(rows)} id={record['id']} "
                    f"letter_unmask_step={record['letter_unmask_step']} rate={rate:.3f}/s",
                    flush=True,
                )
        if finished != len(rows):
            raise RuntimeError(f"finished {finished} of {len(rows)}")
        print(f"SUMMARY n={finished} gen={gen}", flush=True)

    return loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("lavida", "mmada"), required=True)
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--gen", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    parser.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    args = parser.parse_args()
    for p in (args.output, args.progress):
        if OUT_ROOT not in p.resolve().parents:
            raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {p}")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    base.GEN = args.gen
    traj.decode_canvas = decode_canvas
    traj.loop = make_loop(args.gen, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traj.repair_tail(args.output)
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_handle = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"decode already running for {args.output}") from exc
    torch.cuda.set_device(args.cuda)
    print(
        f"start backend={args.backend} cuda={args.cuda} gen={base.GEN} steps={base.GEN} "
        f"block={base.GEN} temperature=0 remasking=low_confidence limit={args.limit} output={args.output}",
        flush=True,
    )
    try:
        if args.backend == "lavida":
            traj.run_lavida(args)
        else:
            traj.run_mmada(args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
