#!/usr/bin/env python3
"""Supplement 20261016. Pre-registered before the run. New file; writes only under outputs/supplement_20261016/.

Three questions, MMaDA-8B-Base, temperature 0, one token per step (run_unmask_order.unmask_one).

1. Factorial on the first --n-sciqa items of datasets/sciqa_img_standard_v1 (manifest order, a prefix of the P6 400).
   Writer heads are P6's L32.H27, L30.H28, L32.H4. The add is the P6 unit write toward the step-0 second-best
   letter, scaled by c = 8 / 0.022451302037976094 so the residual norm matches P6 β=+8. Conditions:
     stock   zero add
     step0   the add on step 0 only
     every   the add on every step
   Factors, one change at a time:
     symbol  option labels A..E rewritten to P..T. Canvas length 16, all masked, scored slot = 0.
     position  same prompt plus one sentence asking for six 'x' characters and then the letter.
               Canvas length 16, all masked. The scored slot is the first committed token in A..E
               (fallback 0 if the canvas contains none). Step-0 and every-step steering use that slot.
     len8 / len32  original A..E prompt, all masked, scored slot = 0, canvas length 8 or 32.

2. Vision on the first --n-mmstar items of datasets/mmstar_20261013e (manifest order).
     real_stock, real_step0, real_every  (same β=+8 writer add)
     gray_stock  384×384 RGB(128,128,128), same text
   A vision-necessary item is one whose real_stock slot letter differs from gray_stock. The split is applied
   only at analysis time; the decodes themselves are the four conditions above.

Reads that decide the claim:
   stock: fraction whose scored slot unmasks on the last step
   step0: fraction whose scored-slot letter differs from stock
   every: fraction that differs from stock, and among those, fraction that land on the steered letter
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
import hidden_shapley_split as hss  # noqa: E402
import p6_causal_20261013f as p6  # noqa: E402
import run_unmask_order as base  # noqa: E402
import sciqa_standard_eval as sq  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "supplement_20261016").resolve()
UNIT_SCALE = 0.022451302037976094  # P6 writer_heads unit_scale_used
BETA = 8.0
C_BETA = BETA / UNIT_SCALE
HEADS = list(p6.WRITER_HEADS)
SYMBOLS = "PQRST"
POST = "\nAnswer with the option's letter from the given choices directly."
POS_SUFFIX = "\nFirst write xxxxxx and then the option letter."
GRAY = (128, 128, 128)


def check_out(path: Path) -> None:
    resolved = path.resolve()
    if resolved != OUT_ROOT and OUT_ROOT not in resolved.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {path}")


def choices_of(row) -> list[str]:
    c = row["choices"]
    if isinstance(c, str):
        c = json.loads(c) if c.strip().startswith("[") else None
    if not isinstance(c, list):
        raise RuntimeError(f"choices {row.get('id')}")
    return [str(x) for x in c]


def gold_of(row) -> str:
    g = row.get("gold_option", row.get("gold"))
    if not g:
        raise RuntimeError(f"no gold on {row.get('id')}")
    return str(g).strip()


def prompt_labeled(row, labels: str) -> str:
    ch = choices_of(row)
    labs = labels[: len(ch)]
    context = str(row.get("hint") or "")
    if context:
        context = f"Context: {context}\n"
    lines = "\n".join(f"{lab}. {text}" for lab, text in zip(labs, ch, strict=True))
    return f"{context}{row['question']}\n{lines}{POST}"


def single_ids(tokenizer, letters: str) -> list[int]:
    try:
        got = hss.letter_ids(tokenizer, letters)
        ids = [int(got[ch]) for ch in letters]
    except Exception:
        ids = []
        for ch in letters:
            enc = tokenizer.encode(ch, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(f"{ch!r} tokenizes to {enc}")
            ids.append(int(enc[0]))
    if letters == "ABCDE" and ids != [32, 33, 34, 35, 36]:
        raise RuntimeError(f"A-E ids {ids}")
    return ids


def period_id(tokenizer) -> int:
    enc = tokenizer.encode(".", add_special_tokens=False)
    if len(enc) != 1:
        raise RuntimeError(f"period tokenizes to {enc}")
    return int(enc[0])


def load_mmada(cuda: int):
    sys.path.insert(0, str(PROJECT / "experiments"))
    sys.path.insert(0, str(PROJECT / "repos" / "MMaDA"))
    import evdlc_mmada as mm

    class Ns:
        model = str(PROJECT / "checkpoints/MMaDA-8B-Base")
        vq_model = str(PROJECT / "checkpoints/magvitv2")
        mmada_repo = str(PROJECT / "repos/MMaDA")

    torch.cuda.set_device(cuda)
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{cuda}")
    print("loading mmada", flush=True)
    model, tokenizer = mm.load_mmada(Ns, device)
    vq = mm.load_vq(Ns, device)
    model.eval()
    core = model.model
    p6.install_p6(core)
    head, gamma64, eps = hss.prepare(core, cuda)
    data_dir = PROJECT / "datasets/sciqa_img_standard_v1"
    recs = sq.load_manifest(data_dir / "manifest.json", data_dir)
    store = sq.ParquetImageStore(data_dir, recs)
    ctx = {
        "model": model, "tokenizer": tokenizer, "vq": vq, "mm": mm, "device": device, "core": core,
        "mask_id": mm.MASK_ID, "gamma64": gamma64, "eps": eps, "head": head, "store": store, "recs": recs,
        "Wout64": [b.attn_out.weight.detach().to(torch.float64).cpu() for b in core.transformer.blocks],
        "abc": single_ids(tokenizer, "ABCDE"),
        "pqr": single_ids(tokenizer, SYMBOLS),
        "period": period_id(tokenizer),
    }
    print(f"mmada ready mask={ctx['mask_id']} period={ctx['period']} PQRST={ctx['pqr']} c_beta={C_BETA:.4f}", flush=True)
    return ctx


def w64(ctx, ids: list[int]) -> torch.Tensor:
    return torch.stack([ctx["head"][i].detach().to(torch.float64).cpu() for i in ids])


def steerer_for(ctx, ids: list[int], heads=None) -> p6.Steerer:
    box = dict(ctx)
    box["W64"] = w64(ctx, ids)
    return p6.Steerer(box, "heads", heads=list(heads or HEADS))


def make_forward(ctx, question: str, image, gen: int):
    mm = ctx["mm"]
    vq_ids = mm.encode_vq(ctx["vq"], image, ctx["device"])
    prompt_ids, _ = mm.build_prompt_ids(ctx["tokenizer"], question, vq_ids, ctx["device"])
    prompt_len = int(prompt_ids.shape[1])

    def forward(canvas):
        if int(canvas.shape[1]) != gen:
            raise RuntimeError(f"canvas {tuple(canvas.shape)} != {gen}")
        logits = base.mmada_logits(ctx["model"], prompt_ids, canvas, ctx["mask_id"], None)
        if int(logits.shape[1]) != prompt_len + gen:
            raise RuntimeError(f"logits {tuple(logits.shape)}")
        return logits[:, prompt_len:, :]

    return forward, prompt_len


def ranked(vals: list[float]) -> list[int]:
    return sorted(range(len(vals)), key=lambda j: (-float(vals[j]), j))


def forward_at(item, x, setup) -> torch.Tensor:
    p6.clear_p()
    p6.P.armed = True
    p6.P.index = int(item["hook_index"])
    p6.P.n_applied = 0
    if setup is not None:
        setup()
    try:
        return item["forward"](x)
    finally:
        p6.clear_p()


def decode(ctx, item, apply_fn) -> dict:
    gen = item["gen"]
    mask_id = ctx["mask_id"]
    x = torch.full((1, gen), mask_id, dtype=torch.long, device=ctx["device"])
    prefill = item.get("prefill") or {}
    for pos, tok in prefill.items():
        x[0, int(pos)] = int(tok)
    n_mask = int(x.eq(mask_id).sum().item())
    unmask = [-1] * gen
    step0_vals = None
    for step in range(n_mask):
        def setup(step=step):
            if apply_fn is not None:
                apply_fn(step)

        logits = forward_at(item, x, setup)
        if step == 0:
            ids = item["symbol_ids"]
            step0_vals = [float(logits[0, item["canvas_pos"], i]) for i in ids]
        pos = base.unmask_one(x, logits, mask_id)
        if unmask[pos] != -1:
            raise RuntimeError(f"double unmask {pos}")
        unmask[pos] = step
    if sorted(v for v in unmask if v >= 0) != list(range(n_mask)):
        raise RuntimeError(f"unmask {unmask}")
    slot = item["canvas_pos"]
    slot_id = int(x[0, slot].item())
    slot_text = ctx["tokenizer"].decode([slot_id], skip_special_tokens=True).strip()[:8]
    order = p6.ranked(step0_vals, len(step0_vals))
    return {
        "step0_vals": [round(v, 5) for v in step0_vals],
        "step0_first": int(order[0]),
        "step0_second": int(order[1]) if len(order) > 1 else int(order[0]),
        "step0_restricted": item["labels"][order[0]],
        "slot_id": slot_id,
        "slot_text": slot_text,
        "slot_unmask_step": int(unmask[slot]),
        "n_mask": n_mask,
        "last_step": int(unmask[slot] == n_mask - 1),
        "unmask": unmask,
        "final_ids": [int(v) for v in x[0].tolist()],
    }


def apply_from(st: p6.Steerer, vec: dict, when: str, commit_step: int | None = None):
    raw = st.setup(vec, C_BETA)

    def fn(step, _raw=raw, _when=when, _commit=commit_step):
        if _when == "every" or (_when == "step0" and step == 0) or (
            _when == "commit" and _commit is not None and step == int(_commit)
        ):
            _raw(step, None)

    return fn


def load_done(path: Path) -> set[tuple]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((r["id"], r["factor"], r["cond"]))
    return done


def append(handle, rec: dict) -> None:
    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_jobs(ctx, jobs: list[dict], output: Path) -> None:
    check_out(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {output}") from exc
    done = load_done(output)
    by_key = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_key[(r["id"], r["factor"], r["cond"])] = r
    rand_heads, _excl = p6.pick_random_heads(p6.np.random.default_rng(p6.SEED))
    print(f"random heads {rand_heads}", flush=True)
    steer = {
        "writer": {"ABCDE": steerer_for(ctx, ctx["abc"]), "PQRST": steerer_for(ctx, ctx["pqr"])},
        "random": {
            "ABCDE": steerer_for(ctx, ctx["abc"], rand_heads),
            "PQRST": steerer_for(ctx, ctx["pqr"], rand_heads),
        },
    }
    print(f"jobs {len(jobs)} done {len(done)} -> {output}", flush=True)
    t0 = time.time()
    n_new = 0
    with output.open("a", encoding="utf-8") as handle:
        for job in jobs:
            key = (job["id"], job["factor"], job["cond"])
            if key in done:
                job["saved"] = by_key.get(key)
                continue
            cond = job["cond"]
            stock = None
            if cond != "stock":
                holder = job.get("stock_holder")
                stock = None if holder is None else holder.get("saved")
                if stock is None:
                    stock = by_key.get((job["id"], job["factor"], "stock"))
                if stock is None:
                    raise RuntimeError(f"stock missing for {job['id']} {job['factor']}")
            t1 = time.time()
            image = job["image"]()
            forward, prompt_len = make_forward(ctx, job["question"], image, job["gen"])
            labels = job["labels"]
            ids = ctx["pqr"][: len(labels)] if labels[0] == "P" else ctx["abc"][: len(labels)]
            canvas_pos = job["canvas_pos"]
            if job["factor"] == "position" and cond != "stock" and stock is not None:
                canvas_pos = int(stock.get("letter_pos", canvas_pos))
            item = {
                "forward": forward,
                "gen": job["gen"],
                "canvas_pos": canvas_pos,
                "hook_index": prompt_len + canvas_pos,
                "symbol_ids": ids,
                "labels": labels,
                "prefill": job.get("prefill"),
            }
            if cond == "stock":
                decoded = decode(ctx, item, None)
                vec_note = None
            else:
                family = "PQRST" if labels[0] == "P" else "ABCDE"
                st = steer[job.get("heads", "writer")][family]
                vec = st.vectors(int(stock["step0_first"]), int(stock["step0_second"]))
                commit_step = int(stock["slot_unmask_step"]) if stock is not None else None
                decoded = decode(ctx, item, apply_from(st, vec, cond, commit_step))
                vec_note = {"proj_norm": vec.get("proj_norm"), "cos": vec.get("cos_total_write_vs_v")}
            target = None if stock is None else labels[int(stock["step0_second"])]
            rec = {
                "id": job["id"], "factor": job["factor"], "cond": cond, "gold": job["gold"],
                "gold_mapped": job["gold_mapped"], "n_opt": len(labels), "labels": "".join(labels),
                "gen": job["gen"], "canvas_pos": canvas_pos, "heads": job.get("heads", "writer"),
                "beta": 0.0 if cond == "stock" else BETA,
                "c": 0.0 if cond == "stock" else round(C_BETA, 6), "vec": vec_note, **decoded,
            }
            if cond == "commit" and stock is not None:
                rec["edit_step"] = int(stock["slot_unmask_step"])
            if job["factor"] == "position" and cond == "stock":
                letter_pos = None
                for j, tid in enumerate(decoded["final_ids"]):
                    if int(tid) in ids:
                        letter_pos = j
                        break
                rec["letter_found"] = int(letter_pos is not None)
                rec["letter_pos"] = 0 if letter_pos is None else int(letter_pos)
                if letter_pos not in (None, 0):
                    tok = int(decoded["final_ids"][letter_pos])
                    rec["slot_id"] = tok
                    rec["slot_text"] = ctx["tokenizer"].decode([tok], skip_special_tokens=True).strip()[:8]
                    rec["slot_unmask_step"] = int(decoded["unmask"][letter_pos])
                    rec["canvas_pos"] = int(letter_pos)
                    rec["last_step"] = int(decoded["unmask"][letter_pos] == decoded["n_mask"] - 1)
            if stock is not None:
                rec["stock_slot_text"] = stock["slot_text"]
                rec["changed"] = int(decoded["slot_text"] != stock["slot_text"])
                rec["hit_target"] = int(decoded["slot_text"] == target)
                rec["target"] = target
            append(handle, rec)
            done.add(key)
            by_key[key] = rec
            job["saved"] = rec
            n_new += 1
            el = time.time() - t0
            print(
                f"PROGRESS {n_new} {job['factor']} {cond} id={job['id']} slot={decoded['slot_text']} "
                f"unmask={decoded['slot_unmask_step']}/{decoded['n_mask'] - 1} sec={time.time() - t1:.1f} "
                f"rate={n_new / el:.3f}/s",
                flush=True,
            )


def sciqa_rows(ctx, n: int) -> list[dict]:
    rows = ctx["recs"][:n]
    if len(rows) != n:
        raise RuntimeError(f"sciqa short {len(rows)}")
    return rows


def mmstar_rows(n: int) -> list[dict]:
    from PIL import Image

    rows = json.loads((PROJECT / "datasets/mmstar_20261013e/manifest.json").read_text(encoding="utf-8"))
    rows = rows[:n]
    if len(rows) != n:
        raise RuntimeError(f"mmstar short {len(rows)}")
    for r in rows:
        path = r["image"]
        r["_image"] = lambda p=path: Image.open(p).convert("RGB")
    return rows


def gray_image():
    from PIL import Image

    return Image.new("RGB", (384, 384), GRAY)


def build_factor_jobs(ctx, n: int) -> list[dict]:
    jobs = []
    for row in sciqa_rows(ctx, n):
        ch = choices_of(row)
        n_opt = len(ch)
        gold = gold_of(row)
        gi = "ABCDE".index(gold)
        image = lambda rec=row: ctx["store"].load_rgb(rec)
        base_q = sq.build_task_prompt(row)
        specs = [
            ("symbol", prompt_labeled(row, SYMBOLS), SYMBOLS[:n_opt], SYMBOLS[gi], 16, 0, None),
            ("position", base_q + POS_SUFFIX, "ABCDE"[:n_opt], gold, 16, 0, None),
            ("len8", base_q, "ABCDE"[:n_opt], gold, 8, 0, None),
            ("len32", base_q, "ABCDE"[:n_opt], gold, 32, 0, None),
        ]
        for factor, question, labels, gold_m, gen, slot, prefill in specs:
            stock_job = {
                "id": row["id"], "factor": factor, "cond": "stock", "question": question, "labels": labels,
                "gold": gold, "gold_mapped": gold_m, "gen": gen, "canvas_pos": slot, "prefill": prefill,
                "image": image,
            }
            jobs.append(stock_job)
            for cond in ("step0", "every"):
                jobs.append({
                    **{k: stock_job[k] for k in ("id", "factor", "question", "labels", "gold", "gold_mapped",
                                                 "gen", "canvas_pos", "prefill", "image")},
                    "cond": cond, "stock_from": stock_job,
                })
    return jobs


def build_vision_jobs(n: int) -> list[dict]:
    jobs = []
    for row in mmstar_rows(n):
        ch = choices_of(row)
        n_opt = len(ch)
        gold = gold_of(row)
        question = sq.build_task_prompt(row)
        labels = "ABCDE"[:n_opt]
        common = {
            "id": row["id"], "question": question, "labels": labels, "gold": gold, "gold_mapped": gold,
            "gen": 16, "canvas_pos": 0, "prefill": None,
        }
        stock = {**common, "factor": "real", "cond": "stock", "image": row["_image"]}
        jobs.append(stock)
        for cond in ("step0", "every"):
            jobs.append({**common, "factor": "real", "cond": cond, "image": row["_image"], "stock_from": stock})
        jobs.append({**common, "factor": "gray", "cond": "stock", "image": gray_image})
    return jobs


def link_stock(jobs: list[dict], prior: dict[tuple, dict]) -> list[dict]:
    ready = []
    for job in jobs:
        if job["cond"] == "stock":
            ready.append(job)
            continue
        src = job["stock_from"]
        saved = src.get("saved") or prior.get((src["id"], src["factor"], "stock"))
        if saved is None:
            raise RuntimeError(f"stock missing for {job['id']} {job['factor']} (stock must be queued first)")
        job = dict(job)
        job["stock"] = saved
        ready.append(job)
    return ready


def analyze(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["factor"], r["cond"]), []).append(r)
    lines = [f"# supplement 20261016", f"source: {path}", f"n_records: {len(rows)}", ""]

    def frac(rs, pred):
        if not rs:
            return "—"
        k = sum(1 for r in rs if pred(r))
        return f"{k}/{len(rs)} ({100 * k / len(rs):.1f}%)"

    for factor in ("symbol", "position", "len8", "len32", "real", "gray"):
        stock = by.get((factor, "stock"), [])
        if not stock:
            continue
        lines.append(f"## {factor} n={len(stock)}")
        lines.append(f"- stock slot unmasks last: {frac(stock, lambda r: r['last_step'] == 1)}")
        lines.append(f"- stock slot equals gold: {frac(stock, lambda r: r['slot_text'] == r['gold_mapped'])}")
        for cond in ("step0", "every"):
            rs = by.get((factor, cond), [])
            if not rs:
                continue
            changed = [r for r in rs if r.get("changed") == 1]
            hit = sum(1 for r in changed if r.get("hit_target") == 1)
            lines.append(
                f"- {cond} changed vs stock: {frac(rs, lambda r: r.get('changed') == 1)}; "
                f"among changes, hit steered letter: {hit}/{len(changed) if changed else 0}"
            )
        lines.append("")
    real = {r["id"]: r for r in by.get(("real", "stock"), [])}
    gray = {r["id"]: r for r in by.get(("gray", "stock"), [])}
    both = [i for i in real if i in gray]
    if both:
        need = [i for i in both if real[i]["slot_text"] != gray[i]["slot_text"]]
        lines.append(f"## vision-necessary (real slot letter != gray slot letter)")
        lines.append(f"- {len(need)}/{len(both)} ({100 * len(need) / len(both):.1f}%)")
        for cond in ("step0", "every"):
            rs = [r for r in by.get(("real", cond), []) if r["id"] in set(need)]
            if not rs:
                continue
            changed = [r for r in rs if r.get("changed") == 1]
            hit = sum(1 for r in changed if r.get("hit_target") == 1)
            lines.append(
                f"- on those items, real {cond} changed: {frac(rs, lambda r: r.get('changed') == 1)}; "
                f"hit steered letter: {hit}/{len(changed) if changed else 0}"
            )
        lines.append("")
    text = "\n".join(lines) + "\n"
    out = path.with_suffix(".md")
    out.write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("factor", "vision", "analyze"), required=True)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--n-sciqa", type=int, default=100)
    p.add_argument("--n-mmstar", type=int, default=200)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.part == "analyze":
        path = args.output or (OUT_ROOT / "factor.jsonl")
        check_out(path)
        analyze(path)
        return
    if args.cuda not in (0, 1):
        raise RuntimeError("--cuda must be 0 or 1")
    name = "factor.jsonl" if args.part == "factor" else "vision.jsonl"
    output = args.output or (OUT_ROOT / name)
    check_out(output)
    ctx = load_mmada(args.cuda)
    if args.part == "factor":
        jobs = build_factor_jobs(ctx, args.n_sciqa)
    else:
        jobs = build_vision_jobs(args.n_mmstar)
    holders = {}
    ordered = []
    for job in jobs:
        if job["cond"] == "stock":
            holders[(job["id"], job["factor"])] = job
            ordered.append(job)
        else:
            holder = holders.get((job["id"], job["factor"]))
            if holder is None:
                raise RuntimeError(f"stock job missing in queue for {job['id']} {job['factor']}")
            job["stock_holder"] = holder
            ordered.append(job)
    run_jobs(ctx, ordered, output)


if __name__ == "__main__":
    main()
