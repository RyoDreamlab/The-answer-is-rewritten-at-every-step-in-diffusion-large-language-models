#!/usr/bin/env python3
"""Stock low-confidence decode (gen 16) with content-free contextual calibration at the letter position.

Prior: variant (a) global of calib_contentfree.py / calib_score.py, read from
outputs/calib_vote_20261012/cf_{model}.jsonl: one record per option count n in 2..5
(mid-gray 384x384 image, question 'N/A', n options 'N/A'), float64 letter logits at canvas
index 0, step 0. log p_cf = log_softmax over the item's n option letters.

At every step while canvas position 0 is masked, the position-0 logits of the n option
letters (token ids 32..32+n-1) are shifted by -log p_cf (strength 1) before the stock
low-confidence argmax/selection (run_unmask_order.unmask_one). All other vocab and
positions are untouched. After position 0 is committed the rest decodes as stock.
--no-calib runs the identical path without the shift (stock-match sanity).
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

OUT_ROOT = (PROJECT / "outputs" / "calib_decode_20261012").resolve()
CF_DIR = PROJECT / "outputs" / "calib_vote_20261012"
LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]

STATE = {"calib": True, "log_pcf": None, "n_opt": None}


def load_priors(backend: str) -> dict[int, list[float]]:
    recs = [json.loads(l) for l in (CF_DIR / f"cf_{backend}.jsonl").open(encoding="utf-8") if l.strip()]
    glob = {r["n_opt"]: r for r in recs if r["cond"] == "a_global"}
    if sorted(glob) != [2, 3, 4, 5]:
        raise RuntimeError(f"global priors {sorted(glob)}")
    out = {}
    for n, r in glob.items():
        if r["id"] != f"global_n{n}" or r["backend"] != backend:
            raise RuntimeError(f"bad prior record {r['id']} {r['backend']}")
        lg = torch.tensor(r["letter_logit64"][:n], dtype=torch.float64)
        out[n] = [float(v) for v in torch.log_softmax(lg, dim=0)]
    return out


def n_options(row: dict) -> int:
    c = row["choices"]
    if isinstance(c, str):
        c = eval(c)  # noqa: S307  same parsing as calib_contentfree.py
    return len(c)


def letter_vals(logits_pos0) -> list[float]:
    return [float(logits_pos0[i]) for i in LETTER_IDS]


def restricted_letter(vals: list[float], n: int) -> str:
    return LETTERS[max(range(n), key=lambda j: (vals[j], -j))]


def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
    gen = base.GEN
    if x.shape[1] != gen:
        raise RuntimeError(f"canvas {tuple(x.shape)} != gen {gen}")
    n = STATE["n_opt"]
    lp = STATE["log_pcf"][n]
    ids = LETTER_IDS[:n]
    unmask_step = [-1] * gen
    pos0_argmax, pos0_argmax_raw = [], []
    raw_at_commit = cal_at_commit = None
    step0 = None
    for step in range(gen):
        logits = forward(x)
        if int(x[0, 0].item()) == int(mask_id):
            raw = letter_vals(logits[0, 0])
            pos0_argmax_raw.append(base.piece(tokenizer, int(torch.argmax(logits[0, 0]).item()), mask_id))
            if STATE["calib"]:
                logits = logits.float()
                shift = torch.tensor(lp, dtype=torch.float32, device=logits.device)
                logits[0, 0, ids] = logits[0, 0, ids] - shift
            cal = letter_vals(logits[0, 0])
            pos0_argmax.append(base.piece(tokenizer, int(torch.argmax(logits[0, 0]).item()), mask_id))
            if step == 0:
                step0 = {"raw": restricted_letter(raw, n), "cal": restricted_letter(cal, n)}
            raw_at_commit, cal_at_commit = raw, cal
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
        "pos0_argmax_raw": pos0_argmax_raw,
        "letter_logits_raw_at_commit": raw_at_commit,
        "letter_logits_cal_at_commit": cal_at_commit,
        "step0_restricted_raw": step0["raw"],
        "step0_restricted_cal": step0["cal"],
        "finished_text": pred,
        "finished_letter": delta.leading_letter(pred),
    }


def make_loop(backend: str, limit: int):
    def loop(args, rows, fn) -> None:
        if limit:
            rows = rows[:limit]
        done = base.load_done(args.output)
        finished = len(done)
        t0 = time.time()
        n_new = 0
        print(f"resume n_done={finished}/{len(rows)} calib={STATE['calib']}", flush=True)
        with args.output.open("a", encoding="utf-8") as handle:
            for row in rows:
                if row["id"] in done:
                    continue
                STATE["n_opt"] = n_options(row)
                with torch.no_grad():
                    d = fn(row)
                record = {
                    "id": row["id"],
                    "gold": traj.gold_of(row),
                    "backend": backend,
                    "calib": STATE["calib"],
                    "n_opt": STATE["n_opt"],
                    "gen_length": base.GEN,
                    **d,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.add(record["id"])
                finished += 1
                n_new += 1
                el = time.time() - t0
                rate = n_new / el if el > 0 else 0.0
                base.write_progress(args.progress, {
                    "backend": backend, "calib": STATE["calib"], "n_done": finished, "n_total": len(rows),
                    "id": record["id"], "letter_unmask_step": record["letter_unmask_step"],
                    "elapsed_sec": round(el, 1), "items_per_sec": round(rate, 4),
                    "eta_sec": round((len(rows) - finished) / rate, 0) if rate > 0 else None,
                })
                if finished % 25 == 0 or finished == len(rows):
                    print(f"PROGRESS n_done={finished}/{len(rows)} rate={rate:.3f}/s", flush=True)
        if finished != len(rows):
            raise RuntimeError(f"finished {finished} of {len(rows)}")
        print(f"SUMMARY n={finished} calib={STATE['calib']}", flush=True)

    return loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("lavida", "mmada"), required=True)
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--no-calib", action="store_true")
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
    if base.GEN != 16:
        raise RuntimeError(f"expected gen 16, got {base.GEN}")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    STATE["calib"] = not args.no_calib
    STATE["log_pcf"] = load_priors(args.backend)
    for n, lp in sorted(STATE["log_pcf"].items()):
        p = [round(float(torch.tensor(v).exp()), 6) for v in lp]
        print(f"PRIOR backend={args.backend} n={n} p_cf={dict(zip(LETTERS, p))} log_p_cf={[round(v, 6) for v in lp]}",
              flush=True)
    traj.decode_canvas = decode_canvas
    traj.loop = make_loop(args.backend, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traj.repair_tail(args.output)
    lock_handle = args.output.with_suffix(args.output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"decode already running for {args.output}") from exc
    torch.cuda.set_device(args.cuda)
    print(f"start backend={args.backend} cuda={args.cuda} gen={base.GEN} steps={base.GEN} block={base.GEN} "
          f"temperature=0 remasking=low_confidence calib={STATE['calib']} limit={args.limit} output={args.output}",
          flush=True)
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
