#!/usr/bin/env python3
"""MMaDA MMStar: gray image at step 0 only vs gray image at the stock commit step only.

Same decode as supplement_20261016 (gen 16, 16 steps, T=0, one token per step, run_unmask_order.unmask_one), no head
steering.  The image seen by the forward pass is switched per step:
  stock        real image at every step
  gray_step0   gray image (384x384 RGB 128,128,128, supplement_20261016.gray_image) at step 0 ONLY, real image after
  gray_commit  gray image ONLY at the stock slot_unmask_step of the same item (the step at which stock wrote slot 0),
               real image at every other step; decoded to the end
  gray_full    gray image at every step (label only: vision-necessary = stock slot_text != gray_full slot_text)
slot_text is read from slot 0 of the final canvas (last step) for every condition.

Writes only under outputs/mmada_grayswap_20261021/mmada/ (grayswap.jsonl, summary.md, summary.json; smoke/ for smoke
tests).  Append + resume keyed on (id, cond).
"""

from __future__ import annotations

import argparse
import collections
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
import run_unmask_order as base  # noqa: E402
import sciqa_standard_eval as sq  # noqa: E402
import supplement_20261016 as S  # noqa: E402

OUT = (PROJECT / "outputs" / "mmada_grayswap_20261021" / "mmada").resolve()
CONDS = ("stock", "gray_step0", "gray_commit", "gray_full")
GEN = 16


def allow_out(path: Path) -> None:
    resolved = Path(path).resolve()
    if resolved == OUT or OUT in resolved.parents:
        return
    raise RuntimeError(f"refusing to write outside {OUT}: {path}")


S.check_out = allow_out


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def decode_switch(ctx, fwd_real, fwd_gray, gray_steps, labels: str, ids: list[int]) -> dict:
    """supplement_20261016.decode without steering; the forward at step s uses the gray image iff s in gray_steps."""
    mask_id = ctx["mask_id"]
    x = torch.full((1, GEN), mask_id, dtype=torch.long, device=ctx["device"])
    unmask = [-1] * GEN
    order = []
    step0_vals = None
    n_gray = 0
    for step in range(GEN):
        use_gray = step in gray_steps
        n_gray += int(use_gray)
        logits = (fwd_gray if use_gray else fwd_real)(x)
        if step == 0:
            step0_vals = [float(logits[0, 0, i]) for i in ids]
        pos = base.unmask_one(x, logits, mask_id)
        if unmask[pos] != -1:
            raise RuntimeError(f"double unmask {pos}")
        unmask[pos] = step
        order.append(pos)
    if sorted(unmask) != list(range(GEN)):
        raise RuntimeError(f"unmask {unmask}")
    slot_id = int(x[0, 0].item())
    slot_text = ctx["tokenizer"].decode([slot_id], skip_special_tokens=True).strip()[:8]
    rank = S.ranked(step0_vals)
    return {
        "step0_vals": [round(v, 5) for v in step0_vals],
        "step0_first": int(rank[0]),
        "step0_second": int(rank[1]) if len(rank) > 1 else int(rank[0]),
        "step0_restricted": labels[rank[0]],
        "slot_id": slot_id, "slot_text": slot_text,
        "slot_unmask_step": int(unmask[0]),
        "last_step": int(unmask[0] == GEN - 1),
        "unmask": unmask, "order": order,
        "final_ids": [int(v) for v in x[0].tolist()],
        "n_gray_steps": n_gray,
    }


def run(ctx, rows: list[dict], output: Path) -> None:
    allow_out(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {output}") from exc
    by = {(r["id"], r["cond"]): r for r in jsonl(output)}
    todo = [r for r in rows if any((r["id"], cnd) not in by for cnd in CONDS)]
    print(f"grayswap rows={len(rows)} records_done={len(by)} items_todo={len(todo)} -> {output}", flush=True)
    gray = S.gray_image()
    t0 = time.time(); n_new = 0; n_items = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            need = [cnd for cnd in CONDS if (row["id"], cnd) not in by]
            if not need:
                continue
            t1 = time.time()
            choices = S.choices_of(row)
            gold = S.gold_of(row)
            labels = "ABCDE"[: len(choices)]
            ids = ctx["abc"][: len(labels)]
            question = sq.build_task_prompt(row)
            image = row["_image"]()
            fwd_real, plen_r = S.make_forward(ctx, question, image, GEN)
            fwd_gray, plen_g = S.make_forward(ctx, question, gray, GEN)
            if plen_r != plen_g:
                raise RuntimeError(f"prompt len real {plen_r} != gray {plen_g} on {row['id']}")
            stock = by.get((row["id"], "stock"))
            common = {"id": row["id"], "gold": gold, "n_opt": len(labels), "labels": labels, "gen": GEN, "canvas_pos": 0,
                      "prompt_len": plen_r, "image_size": list(image.size)}
            for cond in CONDS:
                if (row["id"], cond) in by:
                    continue
                if cond == "stock":
                    gray_steps = set()
                elif cond == "gray_step0":
                    gray_steps = {0}
                elif cond == "gray_commit":
                    if stock is None:
                        raise RuntimeError(f"stock missing for {row['id']}")
                    gray_steps = {int(stock["slot_unmask_step"])}
                else:
                    gray_steps = set(range(GEN))
                dec = decode_switch(ctx, fwd_real, fwd_gray, gray_steps, labels, ids)
                rec = {**common, "cond": cond, "swap_steps": sorted(gray_steps) if cond != "gray_full" else "all", **dec}
                if cond == "gray_commit":
                    rec["edit_step"] = int(stock["slot_unmask_step"])
                if cond != "stock":
                    target = labels[int(stock["step0_second"])]
                    rec.update({"stock_slot_text": stock["slot_text"], "stock_slot_unmask_step": stock["slot_unmask_step"],
                                "stock_last_step": stock["last_step"], "stock_step0_second": stock["step0_second"],
                                "target": target, "changed": int(dec["slot_text"] != stock["slot_text"]),
                                "hit_target": int(dec["slot_text"] == target),
                                "order_same_as_stock": int(dec["order"] == stock["order"])})
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n"); handle.flush(); os.fsync(handle.fileno())
                by[(row["id"], cond)] = rec
                if cond == "stock":
                    stock = rec
                n_new += 1
            n_items += 1
            el = time.time() - t0
            g = {cnd: by[(row["id"], cnd)]["slot_text"] for cnd in CONDS}
            print(f"PROGRESS {n_items}/{len(todo)} id={row['id']} gold={gold} stock={g['stock']} step0={g['gray_step0']} "
                  f"commit={g['gray_commit']} full={g['gray_full']} c={by[(row['id'], 'stock')]['slot_unmask_step']} "
                  f"last={by[(row['id'], 'stock')]['last_step']} sec={time.time() - t1:.1f} rate={n_items / el:.3f} items/s", flush=True)
    lock.close()
    print(f"DONE items={n_items} records_new={n_new} elapsed={time.time() - t0:.0f}s", flush=True)


def report(output: Path) -> None:
    rows = jsonl(output)
    by = {}
    for r in rows:
        by.setdefault(r["cond"], {})[r["id"]] = r
    stock = by.get("stock", {})
    ids = [i for i in stock if all(i in by.get(cnd, {}) for cnd in CONDS)]
    last = [i for i in ids if int(stock[i]["last_step"]) == 1]
    gf = by.get("gray_full", {})
    need = [i for i in ids if stock[i]["slot_text"] != gf[i]["slot_text"]]
    need_last = [i for i in need if i in set(last)]
    hist = dict(sorted(collections.Counter(int(stock[i]["slot_unmask_step"]) for i in ids).items()))
    lines = [f"# MMaDA MMStar: gray image at step 0 only vs at the stock commit step only",
             f"n_items(all 4 conds)={len(ids)} stock_last_step={len(last)} vision-necessary(stock != gray_full)={len(need)} "
             f"(of which last-step {len(need_last)})",
             f"stock slot_unmask_step histogram: {hist}",
             f"stock slot == gold: {sum(stock[i]['slot_text'] == stock[i]['gold'] for i in ids)}/{len(ids)}; "
             f"gray_full slot == gold: {sum(gf[i]['slot_text'] == gf[i]['gold'] for i in ids)}/{len(ids)}", "",
             "| cond | changed/N (all) | changed/N (stock last-step) | changed/N (vision-necessary) | changed/N (vision-necessary & last-step) | hit step0_second (all changed) | order same as stock |",
             "|---|---|---|---|---|---|---|"]
    summary = {"n": len(ids), "n_last": len(last), "n_vision_necessary": len(need), "stock_hist": hist}

    def frac(sel, cnd):
        ch = sum(1 for i in sel if by[cnd][i].get("changed") == 1)
        return ch, len(sel)

    for cnd in ("gray_step0", "gray_commit", "gray_full"):
        a = frac(ids, cnd); b = frac(last, cnd); cc = frac(need, cnd); d = frac(need_last, cnd)
        hit = sum(1 for i in ids if by[cnd][i].get("changed") == 1 and by[cnd][i].get("hit_target") == 1)
        same = sum(1 for i in ids if by[cnd][i].get("order_same_as_stock") == 1)
        summary[cnd] = {"changed_all": a, "changed_last": b, "changed_vn": cc, "changed_vn_last": d, "hit": hit, "order_same": same}
        lines.append(f"| {cnd} | {a[0]}/{a[1]} | {b[0]}/{b[1]} | {cc[0]}/{cc[1]} | {d[0]}/{d[1]} | {hit}/{a[0]} | {same}/{len(ids)} |")
    # by commit-step bucket
    lines += ["", "| stock commit step | n | gray_step0 changed | gray_commit changed | gray_full changed |", "|---|---|---|---|---|"]
    for step in sorted(hist):
        sel = [i for i in ids if int(stock[i]["slot_unmask_step"]) == step]
        lines.append(f"| {step} | {len(sel)} | " + " | ".join(str(frac(sel, cnd)[0]) for cnd in ("gray_step0", "gray_commit", "gray_full")) + " |")
    text = "\n".join(lines) + "\n"
    (output.parent / "summary.md").write_text(text, encoding="utf-8")
    (output.parent / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("decode", "report"), default="decode")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()
    out_dir = args.out_dir.resolve()
    allow_out(out_dir)
    output = out_dir / "grayswap.jsonl"
    if args.mode == "report":
        report(output)
        return
    if args.cuda not in (0, 1):
        raise RuntimeError("cuda must be 0 or 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = S.mmstar_rows(1500)[: args.n]
    torch.cuda.set_device(args.cuda)
    ctx = S.load_mmada(args.cuda)
    run(ctx, rows, output)
    report(output)


if __name__ == "__main__":
    main()
