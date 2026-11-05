#!/usr/bin/env python3
"""Hold the answer slot masked until the final step (LLaDA-V, ScienceQA-IMG test, content scoring).

Question: LLaDA-V commits the letter at slot 0 early (steps 0-3).  If the slot is forced to stay masked while every
other position is filled by the normal confidence schedule, and is unmasked only at the very last step (with the full
rationale visible), does the letter change, and does content accuracy move?

Conditions per item (gen 16, 16 steps, T=0, one token per step, highest-confidence remasking = run_unmask_order.unmask_one):
  stock  plain decode.  letter_unmask_step = unmask step of slot 0 (the slot that receives the letter).
  hold   slot 0 is excluded from the unmask candidate set at steps 0..14 (its confidence is set to -inf before the
         top-1 pick); at step 15 it is the only masked position and is unmasked by the stock rule.  Everything else
         (argmax token, softmax confidence, canvas update) is identical to stock.

The answer slot is position 0 by construction of this prompt (same as placement_lladav_20261019 / lavida_causal_20261019:
letter_unmask_step = unmask[0]).  The record also stores letter_pos_stock = first position in the stock canvas whose token
is one of A..E (diagnostic only; the hold is applied to slot 0).

Scoring: lenient content letter = first standalone A-E in the finished text (placement_lladav_20261019.lenient_letter),
strict letter = leading letter.  content_correct = lenient letter == gold.

Writes only under outputs/holdslot_lladav_20261021/ (stock.jsonl, hold.jsonl, summary.md, summary.json; smoke/ for smoke
tests).  Append + resume (ids present in both files are skipped).  Backend / rows / prompt from calib_decode_multi_20261013c.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
sys.path.insert(0, str(PROJECT / "experiments"))
import calib_decode_multi_20261013c as c  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "holdslot_lladav_20261021").resolve()
MODEL_PATH = PROJECT / "checkpoints" / "LLaDA-V"
LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]
GEN = base.GEN
SLOT = 0
CONDS = ("stock", "hold")
BOOT_SEED, N_BOOT = 0, 1000
N_FULL = 2017


def check_out(path: Path) -> None:
    r = path.resolve()
    if r != OUT_ROOT and OUT_ROOT not in r.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {path}")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def lenient_letter(text: str) -> str | None:
    m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", str(text or ""))
    return m.group(1) if m else None


def unmask_hold(x: torch.Tensor, logits: torch.Tensor, mask_id: int, hold: int, final: bool) -> int:
    """run_unmask_order.unmask_one with position `hold` removed from the candidate set unless `final`."""
    mask_index = x.eq(mask_id)
    if int(mask_index.sum().item()) == 0:
        raise RuntimeError("unmask called on a full canvas")
    if final:
        if int(mask_index.sum().item()) != 1 or not bool(mask_index[0, hold].item()):
            raise RuntimeError(f"final step: masked positions {mask_index[0].nonzero().flatten().tolist()}")
        return base.unmask_one(x, logits, mask_id)
    chosen = torch.argmax(logits, dim=-1)
    probs = F.softmax(logits.double(), dim=-1)
    conf = torch.gather(probs, -1, chosen.unsqueeze(-1)).squeeze(-1)
    conf = torch.where(mask_index, conf, torch.full_like(conf, float("-inf")))
    conf[0, hold] = float("-inf")
    pos = int(torch.topk(conf[0], k=1).indices[0].item())
    if pos == hold or not bool(mask_index[0, pos].item()):
        raise RuntimeError(f"hold: selected position {pos}")
    x[0, pos] = chosen[0, pos]
    return pos


class Decoder:
    def __init__(self, be: c.Lladav):
        self.be = be
        self.mask_id = be.mask_id
        self.tok = be.tokenizer
        self.n_forward = 0

    def item(self, row):
        pe, plen = self.be.lb.prefix(self.be.ctx, row["text"], row["image"]())
        return {"forward": self.be.lb.make_forward(self.be.ctx, pe), "prefix_len": plen, "device": pe.device}

    def fwd(self, item, x):
        self.n_forward += 1
        return item["forward"](x)

    def fresh(self, item):
        return torch.full((1, GEN), self.mask_id, dtype=torch.long, device=item["device"])

    def run(self, item, hold: bool) -> dict:
        x = self.fresh(item)
        unmask = [-1] * GEN
        order = []
        pos0_letters = []
        for step in range(GEN):
            logits = self.fwd(item, x)
            if int(x[0, SLOT].item()) == self.mask_id:
                pos0_letters.append([float(v) for v in logits[0, SLOT, LETTER_IDS]])
            if hold:
                pos = unmask_hold(x, logits, self.mask_id, SLOT, final=step == GEN - 1)
            else:
                pos = base.unmask_one(x, logits, self.mask_id)
            if unmask[pos] != -1:
                raise RuntimeError(f"double unmask {pos}")
            unmask[pos] = step
            order.append(pos)
        if sorted(unmask) != list(range(GEN)):
            raise RuntimeError(f"unmask {unmask}")
        if int(x.eq(self.mask_id).sum().item()) != 0:
            raise RuntimeError("canvas still masked")
        final_ids = [int(v) for v in x[0].tolist()]
        text = self.tok.decode(x[0], skip_special_tokens=True).strip()
        letter_pos = next((j for j, t in enumerate(final_ids) if t in LETTER_IDS), None)
        return {"finished_text": text, "finished_letter": delta.leading_letter(text), "lenient_letter": lenient_letter(text),
                "letter_unmask_step": int(unmask[SLOT]), "unmask_step": unmask, "order": order, "final_ids": final_ids,
                "slot_token": final_ids[SLOT], "letter_pos": letter_pos, "pos0_letters": pos0_letters}


def run_item(dec: Decoder, row) -> dict[str, dict]:
    t1 = time.time()
    item = dec.item(row)
    nf0 = dec.n_forward
    meta = {"id": row["id"], "gold": row["gold"], "n_opt": int(row["n_opt"]), "backend": "lladav", "dataset": "sciqa",
            "slot": SLOT, "prefix_len": item["prefix_len"]}
    s = dec.run(item, hold=False)
    h = dec.run(item, hold=True)
    if h["letter_unmask_step"] != GEN - 1:
        raise RuntimeError(f"hold commit step {h['letter_unmask_step']}")
    sec = round(time.time() - t1, 2)
    nf = dec.n_forward - nf0
    common = {"stock_letter": s["lenient_letter"], "hold_letter": h["lenient_letter"],
              "stock_letter_strict": s["finished_letter"], "hold_letter_strict": h["finished_letter"],
              "stock_commit_step": s["letter_unmask_step"], "hold_commit_step": h["letter_unmask_step"],
              "stock_finished_text": s["finished_text"], "hold_finished_text": h["finished_text"],
              "changed": int(s["lenient_letter"] != h["lenient_letter"]),
              "changed_strict": int(s["finished_letter"] != h["finished_letter"]),
              "text_changed": int(s["finished_text"] != h["finished_text"]),
              "stock_content_correct": int(s["lenient_letter"] == row["gold"]),
              "hold_content_correct": int(h["lenient_letter"] == row["gold"]),
              "letter_pos_stock": s["letter_pos"], "letter_pos_hold": h["letter_pos"],
              "n_forwards_item": nf, "sec": sec}
    out = {}
    for cond, res in (("stock", s), ("hold", h)):
        out[cond] = {**meta, "cond": cond, **common, **{k: v for k, v in res.items() if k not in ("finished_text",)},
                     "finished_text": res["finished_text"], "content_correct": int(res["lenient_letter"] == row["gold"])}
    return out


def run_decode(args) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = (out_dir / "holdslot.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("already running") from exc
    files = {v: out_dir / f"{v}.jsonl" for v in CONDS}
    for f in files.values():
        check_out(f)
    done = set.intersection(*[{r["id"] for r in jsonl(f)} for f in files.values()])
    rows = c.rows_sciqa("lladav", 0)
    if len(rows) != N_FULL:
        raise RuntimeError(f"sciqa rows {len(rows)}")
    rows = rows[: args.n]
    todo = [r for r in rows if r["id"] not in done]
    print(f"start holdslot lladav n={len(rows)} done={len(done)} todo={len(todo)} out={out_dir}", flush=True)
    if not todo:
        report(out_dir)
        return
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    be = c.Lladav(argparse.Namespace(model_path=str(MODEL_PATH), cuda=args.cuda))
    if be.letter_ids() != LETTER_IDS:
        raise RuntimeError(f"letter ids {be.letter_ids()}")
    dec = Decoder(be)
    handles = {v: f.open("a", encoding="utf-8") for v, f in files.items()}
    t0 = time.time(); n_new = 0; n_changed = 0
    for row in rows:
        if row["id"] in done:
            continue
        res = run_item(dec, row)
        for v, rec in res.items():
            handles[v].write(json.dumps(rec, ensure_ascii=False) + "\n"); handles[v].flush(); os.fsync(handles[v].fileno())
        done.add(row["id"]); n_new += 1
        s = res["stock"]
        n_changed += s["changed"]
        el = time.time() - t0
        print(f"PROGRESS {len(done)}/{len(rows)} id={row['id']} gold={row['gold']} stock={s['stock_letter']} hold={s['hold_letter']} "
              f"c={s['stock_commit_step']} hc={s['hold_commit_step']} changed={s['changed']} n_changed={n_changed} "
              f"fwd/item={s['n_forwards_item']} sec={s['sec']} rate={n_new / el:.3f}/s", flush=True)
    for h in handles.values():
        h.close()
    print(f"DONE n={len(done)} new={n_new} changed={n_changed} elapsed={time.time() - t0:.0f}s", flush=True)
    report(out_dir)


def boot_ci(diff: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    sums = diff[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    n = len(diff)
    return {"delta": int(diff.sum()), "ci95": [int(round(lo)), int(round(hi))],
            "delta_pct": round(100 * float(diff.sum()) / n, 2), "ci95_pct": [round(100 * lo / n, 2), round(100 * hi / n, 2)]}


def report(out_dir: Path) -> None:
    data = {v: {r["id"]: r for r in jsonl(out_dir / f"{v}.jsonl")} for v in CONDS}
    ids = [i for i in data["stock"] if i in data["hold"]]
    if not ids:
        print("report: no items", flush=True)
        return
    st, ho = data["stock"], data["hold"]
    gold = {i: st[i]["gold"] for i in ids}
    s_ok = np.array([int(st[i]["lenient_letter"] == gold[i]) for i in ids])
    h_ok = np.array([int(ho[i]["lenient_letter"] == gold[i]) for i in ids])
    s_strict = np.array([int(st[i]["finished_letter"] == gold[i]) for i in ids])
    h_strict = np.array([int(ho[i]["finished_letter"] == gold[i]) for i in ids])
    ci = boot_ci(h_ok - s_ok)
    ci_strict = boot_ci(h_strict - s_strict)
    changed = int(sum(st[i]["lenient_letter"] != ho[i]["lenient_letter"] for i in ids))
    changed_strict = int(sum(st[i]["finished_letter"] != ho[i]["finished_letter"] for i in ids))
    text_changed = int(sum(st[i]["finished_text"] != ho[i]["finished_text"] for i in ids))
    hist = dict(sorted(collections.Counter(int(st[i]["letter_unmask_step"]) for i in ids).items()))
    hold_hist = dict(sorted(collections.Counter(int(ho[i]["letter_unmask_step"]) for i in ids).items()))
    n = len(ids)
    hold_not_last = sum(1 for i in ids if ho[i]["letter_unmask_step"] != GEN - 1)
    # transitions among changed items
    trans = collections.Counter((st[i]["lenient_letter"], ho[i]["lenient_letter"]) for i in ids if st[i]["lenient_letter"] != ho[i]["lenient_letter"])
    fixed = int(((s_ok == 0) & (h_ok == 1)).sum())
    broke = int(((s_ok == 1) & (h_ok == 0)).sum())
    lines = [f"# hold answer slot masked until the final step, LLaDA-V, ScienceQA-IMG test (first {n} items, standard manifest)", "",
             f"n={n}; gen={GEN}, steps={GEN}, T=0, one token per step, highest-confidence remasking; slot={SLOT}",
             f"hold commit step == {GEN - 1} on {n - hold_not_last}/{n} items (violations: {hold_not_last})", "",
             "| cond | content correct (lenient) | acc % | strict correct | strict % |", "|---|---|---|---|---|",
             f"| stock | {s_ok.sum()} | {100 * s_ok.mean():.2f} | {s_strict.sum()} | {100 * s_strict.mean():.2f} |",
             f"| hold | {h_ok.sum()} | {100 * h_ok.mean():.2f} | {h_strict.sum()} | {100 * h_strict.mean():.2f} |", "",
             f"delta (hold - stock) content: {ci['delta']:+d} items ({ci['delta_pct']:+.2f} pp), paired bootstrap 95% CI "
             f"[{ci['ci95'][0]}, {ci['ci95'][1]}] items = [{ci['ci95_pct'][0]:+.2f}, {ci['ci95_pct'][1]:+.2f}] pp "
             f"({N_BOOT} resamples, seed {BOOT_SEED}); wrong->right {fixed}, right->wrong {broke}",
             f"delta strict: {ci_strict['delta']:+d} [{ci_strict['ci95'][0]}, {ci_strict['ci95'][1]}]", "",
             f"letters changed (lenient) hold vs stock: {changed}/{n}; strict-letter changed: {changed_strict}/{n}; finished text changed: {text_changed}/{n}",
             f"changed transitions (stock->hold): {dict(trans)}", "",
             f"stock_commit_step histogram: {hist}", f"hold commit step histogram: {hold_hist}",
             f"stock first token is a letter: {sum(st[i]['finished_letter'] is not None for i in ids)}/{n}; "
             f"hold first token is a letter: {sum(ho[i]['finished_letter'] is not None for i in ids)}/{n}", "",
             "| stock commit step | n | letters changed | stock acc | hold acc |", "|---|---|---|---|---|"]
    for step in sorted(hist):
        sel = [k for k, i in enumerate(ids) if st[i]["letter_unmask_step"] == step]
        ch = sum(st[ids[k]]["lenient_letter"] != ho[ids[k]]["lenient_letter"] for k in sel)
        lines.append(f"| {step} | {len(sel)} | {ch} | {s_ok[sel].sum()} | {h_ok[sel].sum()} |")
    summary = {"n": n, "stock_content_correct": int(s_ok.sum()), "hold_content_correct": int(h_ok.sum()),
               "stock_acc": float(s_ok.mean()), "hold_acc": float(h_ok.mean()), "delta": ci, "delta_strict": ci_strict,
               "letters_changed": changed, "letters_changed_strict": changed_strict, "text_changed": text_changed,
               "stock_commit_step_hist": hist, "hold_commit_step_hist": hold_hist, "hold_not_last": hold_not_last,
               "transitions": {f"{a}->{b}": v for (a, b), v in trans.items()}}
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("decode", "report"), default="decode")
    p.add_argument("--cuda", type=int, default=1)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    args.out_dir = args.out_dir.resolve()
    check_out(args.out_dir)
    if args.mode == "decode":
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
        run_decode(args)
    else:
        report(args.out_dir)


if __name__ == "__main__":
    main()
