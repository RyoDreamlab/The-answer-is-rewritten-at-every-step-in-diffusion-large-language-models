#!/usr/bin/env python3
"""Draft aggregation at the answer slot (accuracy intervention), ScienceQA-IMG test, four masked-diffusion VLMs.

Phenomenon: the answer slot (canvas position 0) is rewritten at every denoising step while it stays masked; stock
decoding keeps only the write present at the step the slot is unmasked.  Method: accumulate the slot's letter
log-probabilities over every step during which the slot is masked (the forward of the unmask step included) and, at
the step the slot is unmasked, write the argmax of the accumulated score instead of the current-step argmax.

Conditions (gen 16, 16 steps, T=0, one token per step, highest-confidence remasking = run_unmask_order.unmask_one;
the schedule, i.e. which position is unmasked at which step, is the stock rule on the current logits for every cond)
  stock          unchanged decode.
  agg_mean       score = sum_s log_softmax(logits_s[slot])[letter], letters = option letters of the item (A..E[:n_opt]),
                 s over the steps where the slot is masked up to and including its stock unmask step.  The slot token
                 written at that step = argmax of the score.  Steps after the unmask step are re-decoded by the stock
                 rule from the modified canvas when the written token differs from stock (identical otherwise, so the
                 stock tail is reused; no forward is spent).
  agg_late       same with per-step weight (s+1)/S, S = gen.
  hold_agg_mean  (lladav only) the slot is excluded from the candidate set at steps 0..14 (holdslot_lladav_20261021.
                 unmask_hold), unmasked at step 15, written with agg_mean over all 16 steps.

Backends / rows / prompts / scoring: identical to the existing ScienceQA runs
  lavida, dreamvl  lavida_causal_20261019 (--dataset sciqa; load_backend / load_rows / text_of / load_image), scored
                   letter = leading letter of the finished text (measure_letter_hidden_delta.leading_letter).
  mmada            supplement_20261016 (load_mmada / sciqa_rows / make_forward, prompt sciqa_standard_eval.
                   build_task_prompt, gen 16, canvas_pos 0), scored letter = decoded slot token (the
                   mmada_causal_sqa_20261020 stock rule, 1059/2017).
  lladav           calib_decode_multi_20261013c.Lladav + rows_sciqa (as holdslot_lladav_20261021), scored letter =
                   lenient letter (first stand-alone A-E in the finished text).

Writes only under outputs/draftagg_20261022/ (<backend>/ or smoke_<backend>/): <cond>.jsonl append + resume by id
(ids present in every cond file are skipped), summary.md, summary.json.  check_out of the imported modules is
monkeypatched to this root; nothing else in outputs/ is written.
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
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "draftagg_20261022").resolve()
LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]
GEN = base.GEN
SLOT = 0
BOOT_SEED, N_BOOT = 0, 1000
CONDS_BY_BACKEND = {"lavida": ("stock", "agg_mean", "agg_late"), "dreamvl": ("stock", "agg_mean", "agg_late"),
                    "mmada": ("stock", "agg_mean", "agg_late"),
                    "lladav": ("stock", "agg_mean", "agg_late", "hold_agg_mean")}
LLADAV_PATH = PROJECT / "checkpoints" / "LLaDA-V"


def check_out(path: Path) -> None:
    r = Path(path).resolve()
    if r != OUT_ROOT and OUT_ROOT not in r.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {path}")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def leading_letter(text: str | None) -> str | None:
    m = re.match(r"^([A-E])(?:$|[\s\.\:\)])", (text or "").strip())
    return m.group(1) if m else None


def lenient_letter(text: str | None) -> str | None:
    m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", str(text or ""))
    return m.group(1) if m else None


def argmax_first(vals) -> int:
    """index of the max; ties -> lowest index (same tie rule as p6_causal_20261013f.ranked)."""
    return max(range(len(vals)), key=lambda j: (float(vals[j]), -j))


def unmask_hold(x: torch.Tensor, logits: torch.Tensor, mask_id: int, hold: int, final: bool) -> int:
    """holdslot_lladav_20261021.unmask_hold: stock rule with position `hold` removed from the candidates unless final."""
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


# ----------------------------------------------------------------------------- backends

class Backend:
    """name, mask_id, device, rows(), item(row) -> forward(canvas)->(1,GEN,V), text(ids), letter(text, ids, n_opt)."""

    name = ""
    n_forward = 0

    def rows(self) -> list[dict]:
        raise NotImplementedError

    def item(self, row):
        raise NotImplementedError

    def text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def letter(self, text: str, ids: list[int], n_opt: int) -> str | None:
        return leading_letter(text)


class LavidaLike(Backend):
    """lavida / dreamvl through lavida_causal_20261019 (sciqa dataset)."""

    def __init__(self, name: str, cuda: int):
        import lavida_causal_20261019 as S
        import p6_causal_20261013f as p6

        S.check_out = check_out
        S.DATASET = "sciqa"
        S.set_dims(name)
        self.S, self.p6, self.name = S, p6, name
        self.ctx = S.load_backend(name, cuda)
        self.tokenizer = self.ctx["tokenizer"]
        self.mask_id = int(self.ctx["mask_id"])
        self.device = self.ctx["device"]
        if self.ctx["gen"] != GEN:
            raise RuntimeError(f"gen {self.ctx['gen']}")

    def rows(self):
        return self.S.load_rows(self.name)

    def item(self, row):
        S = self.S
        it = self.ctx["make_item"](S.text_of(self.name, row), S.load_image(row))
        S.dla.disarm()

        def forward(x):
            self.n_forward += 1
            logits, _n = self.p6.forward_once(self.ctx, it, x, None)
            return logits

        return forward

    def text(self, ids):
        t = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        if self.name == "dreamvl":
            import backend_dreamvl_20261013g as bd

            t = bd.strip_special_text(t)
        return t


class Mmada(Backend):
    name = "mmada"

    def __init__(self, cuda: int):
        import p6_causal_20261013f as p6
        import supplement_20261016 as Sup

        Sup.check_out = check_out
        self.Sup, self.p6 = Sup, p6
        torch.cuda.set_device(cuda)
        self.ctx = Sup.load_mmada(cuda)
        self.tokenizer = self.ctx["tokenizer"]
        self.mask_id = int(self.ctx["mask_id"])
        self.device = self.ctx["device"]
        if self.ctx["abc"] != LETTER_IDS:
            raise RuntimeError(f"letter ids {self.ctx['abc']}")

    def rows(self):
        Sup, ctx = self.Sup, self.ctx
        out = []
        for row in Sup.sciqa_rows(ctx, 2017):
            choices = Sup.choices_of(row)
            gold = Sup.gold_of(row)
            if gold not in LETTERS[: len(choices)]:
                raise RuntimeError(f"bad gold {row['id']} {gold}")
            out.append({"id": row["id"], "gold": gold, "n_opt": len(choices), "choices": choices,
                        "text": Sup.sq.build_task_prompt(row), "image": (lambda rec=row: ctx["store"].load_rgb(rec))})
        return out

    def item(self, row):
        self.p6.clear_p()
        fwd, _plen = self.Sup.make_forward(self.ctx, row["text"], row["image"](), GEN)

        def forward(x):
            self.n_forward += 1
            return fwd(x)

        return forward

    def letter(self, text, ids, n_opt):
        # mmada_causal_sqa_20261020 stock rule: the decoded slot token itself (1059/2017 reference)
        slot_text = self.tokenizer.decode([int(ids[SLOT])], skip_special_tokens=True).strip()[:8]
        return slot_text if slot_text in LETTERS[:n_opt] else None


class Lladav(Backend):
    name = "lladav"

    def __init__(self, cuda: int):
        import calib_decode_multi_20261013c as c

        torch.cuda.set_device(cuda)
        torch.set_grad_enabled(False)
        self.c = c
        self.be = c.Lladav(argparse.Namespace(model_path=str(LLADAV_PATH), cuda=cuda))
        if self.be.letter_ids() != LETTER_IDS:
            raise RuntimeError(f"letter ids {self.be.letter_ids()}")
        self.tokenizer = self.be.tokenizer
        self.mask_id = int(self.be.mask_id)
        self.device = None

    def rows(self):
        rows = self.c.rows_sciqa("lladav", 0)
        if len(rows) != 2017:
            raise RuntimeError(f"sciqa rows {len(rows)}")
        return rows

    def item(self, row):
        pe, _plen = self.be.lb.prefix(self.be.ctx, row["text"], row["image"]())
        fwd = self.be.lb.make_forward(self.be.ctx, pe)
        self.device = pe.device

        def forward(x):
            self.n_forward += 1
            return fwd(x)

        return forward

    def letter(self, text, ids, n_opt):
        return lenient_letter(text)


def load_backend(name: str, cuda: int) -> Backend:
    torch.cuda.set_device(cuda)
    torch.set_grad_enabled(False)
    if name in ("lavida", "dreamvl"):
        return LavidaLike(name, cuda)
    if name == "mmada":
        return Mmada(cuda)
    return Lladav(cuda)


# ----------------------------------------------------------------------------- decode

def decode_stock(fwd, mask_id: int, device, lids: list[int], hold: bool = False) -> dict:
    """Stock (or hold) schedule; accumulates the slot's letter log-probs while it is masked and keeps the canvas as it
    was right before the slot's unmask step so the agg variants can branch from it."""
    x = torch.full((1, GEN), mask_id, dtype=torch.long, device=device)
    k = len(lids)
    acc_mean = np.zeros(k); acc_late = np.zeros(k)
    step_argmax: list[str] = []; per_step: list[list[float]] = []
    unmask = [-1] * GEN
    branch = None
    for step in range(GEN):
        logits = fwd(x)
        if int(x[0, SLOT].item()) == mask_id:
            lp = F.log_softmax(logits[0, SLOT].double(), dim=-1)[lids].cpu().numpy()
            acc_mean += lp
            acc_late += lp * (step + 1) / GEN
            step_argmax.append(LETTERS[argmax_first(lp)])
            per_step.append([round(float(v), 4) for v in lp])
        before = x.clone()
        if hold:
            pos = unmask_hold(x, logits, mask_id, SLOT, final=step == GEN - 1)
        else:
            pos = base.unmask_one(x, logits, mask_id)
        if unmask[pos] != -1:
            raise RuntimeError(f"double unmask {pos}")
        unmask[pos] = step
        if pos == SLOT:
            branch = {"step": step, "canvas_before": before, "stock_token": int(x[0, SLOT].item()),
                      "acc_mean": acc_mean.copy(), "acc_late": acc_late.copy(), "n_steps_agg": len(per_step)}
    if sorted(unmask) != list(range(GEN)) or branch is None:
        raise RuntimeError(f"unmask {unmask}")
    if int(x.eq(mask_id).sum().item()) != 0:
        raise RuntimeError("canvas still masked")
    return {"final_ids": [int(v) for v in x[0].tolist()], "unmask": unmask, "commit_step": int(unmask[SLOT]),
            "step_argmax": step_argmax, "per_step_lp": per_step, "branch": branch}


def decode_branch(fwd, mask_id: int, stock: dict, token: int) -> dict:
    """Write `token` into the slot at the stock unmask step and finish the remaining steps with the stock rule."""
    b = stock["branch"]
    if token == b["stock_token"]:
        return {"final_ids": list(stock["final_ids"]), "unmask": list(stock["unmask"]), "reused_stock_tail": 1,
                "extra_forwards": 0}
    x = b["canvas_before"].clone()
    x[0, SLOT] = int(token)
    unmask = [-1] * GEN
    for p, s in enumerate(stock["unmask"]):
        if s <= b["step"]:
            unmask[p] = s
    n_extra = 0
    for step in range(b["step"] + 1, GEN):
        logits = fwd(x); n_extra += 1
        pos = base.unmask_one(x, logits, mask_id)
        if unmask[pos] != -1:
            raise RuntimeError(f"branch double unmask {pos}")
        unmask[pos] = step
    if sorted(unmask) != list(range(GEN)) or int(x.eq(mask_id).sum().item()) != 0:
        raise RuntimeError(f"branch unmask {unmask}")
    return {"final_ids": [int(v) for v in x[0].tolist()], "unmask": unmask, "reused_stock_tail": 0, "extra_forwards": n_extra}


def run_item(be: Backend, row: dict, conds: tuple[str, ...]) -> dict[str, dict]:
    t1 = time.time()
    n_opt = int(row["n_opt"]); lids = LETTER_IDS[:n_opt]; letters = LETTERS[:n_opt]
    fwd = be.item(row)
    nf0 = be.n_forward
    gold = row["gold"]
    stock = decode_stock(fwd, be.mask_id, be.device, lids, hold=False)
    hold = decode_stock(fwd, be.mask_id, be.device, lids, hold=True) if "hold_agg_mean" in conds else None
    if hold is not None and hold["commit_step"] != GEN - 1:
        raise RuntimeError(f"hold commit step {hold['commit_step']}")
    meta = {"id": row["id"], "gold": gold, "n_opt": n_opt, "backend": be.name, "dataset": "sciqa", "slot": SLOT, "gen": GEN}

    def finish(cond: str, src: dict, final_ids: list[int], unmask: list[int], scores, extra: dict) -> dict:
        text = be.text(final_ids)
        letter = be.letter(text, final_ids, n_opt)
        slot_tok = final_ids[SLOT]
        return {**meta, "cond": cond, "finished_text": text[:200], "letter": letter, "content_correct": int(letter == gold),
                "commit_step": int(unmask[SLOT]), "stock_commit_step": stock["commit_step"],
                "slot_token": slot_tok, "slot_letter": LETTERS[LETTER_IDS.index(slot_tok)] if slot_tok in LETTER_IDS else None,
                "step_argmax": src["step_argmax"], "n_steps_agg": src["branch"]["n_steps_agg"],
                "agg_scores": None if scores is None else {l: round(float(v), 4) for l, v in zip(letters, scores)},
                "unmask_step": unmask, "final_ids": final_ids, **extra}

    out = {}
    s_rec = finish("stock", stock, stock["final_ids"], stock["unmask"], None, {})
    out["stock"] = s_rec
    cache: dict[int, dict] = {}
    for cond in conds:
        if cond == "stock":
            continue
        if cond == "hold_agg_mean":
            sc = hold["branch"]["acc_mean"]
            tok = lids[argmax_first(sc)]
            final_ids = list(hold["final_ids"]); final_ids[SLOT] = tok
            hold_rule_tok = hold["branch"]["stock_token"]
            extra = {"hold_rule_token": hold_rule_tok,
                     "hold_rule_letter": LETTERS[LETTER_IDS.index(hold_rule_tok)] if hold_rule_tok in LETTER_IDS else None,
                     "hold_agg_late_letter": letters[argmax_first(hold["branch"]["acc_late"])],
                     "reused_stock_tail": 0, "extra_forwards": 0}
            rec = finish(cond, hold, final_ids, hold["unmask"], sc, extra)
        else:
            sc = stock["branch"]["acc_mean"] if cond == "agg_mean" else stock["branch"]["acc_late"]
            tok = lids[argmax_first(sc)]
            if tok not in cache:
                cache[tok] = decode_branch(fwd, be.mask_id, stock, tok)
            br = cache[tok]
            rec = finish(cond, stock, br["final_ids"], br["unmask"], sc,
                         {"reused_stock_tail": br["reused_stock_tail"], "extra_forwards": br["extra_forwards"]})
        out[cond] = rec
    sec = round(time.time() - t1, 2)
    nf = be.n_forward - nf0
    for cond, rec in out.items():
        rec["stock_letter"] = s_rec["letter"]
        rec["changed"] = int(rec["letter"] != s_rec["letter"])
        rec["text_changed"] = int(rec["finished_text"] != s_rec["finished_text"])
        rec["n_forwards_item"] = nf
        rec["sec"] = sec
    return out


# ----------------------------------------------------------------------------- run / report

def run_decode(args) -> None:
    out_dir: Path = args.out_dir
    conds = args.conds
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = (out_dir / "draftagg.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("already running") from exc
    files = {v: out_dir / f"{v}.jsonl" for v in conds}
    for f in files.values():
        check_out(f)
    done = set.intersection(*[{r["id"] for r in jsonl(f)} for f in files.values()])
    be = load_backend(args.backend, args.cuda)
    rows = be.rows()
    n_loader = len(rows)
    rows = rows[: args.n]
    todo = [r for r in rows if r["id"] not in done]
    print(f"start draftagg {args.backend} loader_n={n_loader} n={len(rows)} done={len(done)} todo={len(todo)} conds={conds} out={out_dir}",
          flush=True)
    if not todo:
        report(out_dir, args.backend, conds)
        return
    handles = {v: f.open("a", encoding="utf-8") for v, f in files.items()}
    t0 = time.time(); n_new = 0; n_changed = collections.Counter()
    for row in rows:
        if row["id"] in done:
            continue
        res = run_item(be, row, conds)
        for v in conds:
            handles[v].write(json.dumps(res[v], ensure_ascii=False) + "\n"); handles[v].flush(); os.fsync(handles[v].fileno())
        done.add(row["id"]); n_new += 1
        for v in conds[1:]:
            n_changed[v] += res[v]["changed"]
        el = time.time() - t0
        rate = n_new / max(el, 1e-6)
        remaining = sum(1 for r in rows if r["id"] not in done)
        s = res["stock"]
        letters = " ".join(f"{v}={res[v]['letter']}" for v in conds[1:])
        print(f"PROGRESS {args.backend} {len(done)}/{len(rows)} id={row['id']} gold={row['gold']} stock={s['letter']} {letters} "
              f"c={s['commit_step']} steps={''.join(s['step_argmax'])} changed={dict(n_changed)} fwd={s['n_forwards_item']} "
              f"sec={s['sec']} rate={rate:.3f}/s eta_min={remaining / max(rate, 1e-9) / 60:.1f}", flush=True)
    for h in handles.values():
        h.close()
    print(f"DONE {args.backend} n={len(done)} new={n_new} changed={dict(n_changed)} elapsed={time.time() - t0:.0f}s", flush=True)
    report(out_dir, args.backend, conds)


def boot_ci(diff: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    sums = diff[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    n = len(diff)
    return {"delta": int(diff.sum()), "ci95": [int(round(lo)), int(round(hi))],
            "delta_pct": round(100 * float(diff.sum()) / n, 2), "ci95_pct": [round(100 * lo / n, 2), round(100 * hi / n, 2)]}


def report(out_dir: Path, backend: str, conds: tuple[str, ...]) -> None:
    data = {v: {r["id"]: r for r in jsonl(out_dir / f"{v}.jsonl")} for v in conds}
    ids = [i for i in data["stock"] if all(i in data[v] for v in conds)]
    if not ids:
        print("report: no items", flush=True)
        return
    st = data["stock"]
    n = len(ids)
    gold = {i: st[i]["gold"] for i in ids}
    ok = {v: np.array([int(data[v][i]["letter"] == gold[i]) for i in ids]) for v in conds}
    hist = dict(sorted(collections.Counter(int(st[i]["commit_step"]) for i in ids).items()))
    lines = [f"# draft aggregation at the answer slot, {backend}, ScienceQA-IMG test (first {n} items, loader order)", "",
             f"n={n}; gen={GEN}, steps={GEN}, T=0, one token per step, highest-confidence remasking; slot={SLOT}; "
             f"scores = sum over masked steps of log_softmax(slot logits)[option letters] (agg_late: weight (s+1)/{GEN})",
             f"stock commit-step histogram: {hist}", "",
             "| cond | content correct | acc % | delta vs stock | 95% CI (items) | 95% CI (pp) | letters changed | wrong->right | right->wrong | text changed | stock tail reused |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {"n": n, "backend": backend, "conds": {}, "stock_commit_step_hist": hist, "boot": {"n": N_BOOT, "seed": BOOT_SEED}}
    lines.append(f"| stock | {ok['stock'].sum()} | {100 * ok['stock'].mean():.2f} | - | - | - | - | - | - | - | - |")
    summary["conds"]["stock"] = {"content_correct": int(ok["stock"].sum()), "acc": float(ok["stock"].mean())}
    for v in conds[1:]:
        ci = boot_ci(ok[v] - ok["stock"])
        changed = int(sum(data[v][i]["letter"] != st[i]["letter"] for i in ids))
        fixed = int(((ok["stock"] == 0) & (ok[v] == 1)).sum()); broke = int(((ok["stock"] == 1) & (ok[v] == 0)).sum())
        tch = int(sum(data[v][i]["finished_text"] != st[i]["finished_text"] for i in ids))
        reused = int(sum(int(data[v][i].get("reused_stock_tail", 0)) for i in ids))
        trans = collections.Counter((st[i]["letter"], data[v][i]["letter"]) for i in ids if data[v][i]["letter"] != st[i]["letter"])
        lines.append(f"| {v} | {ok[v].sum()} | {100 * ok[v].mean():.2f} | {ci['delta']:+d} ({ci['delta_pct']:+.2f} pp) | "
                     f"[{ci['ci95'][0]}, {ci['ci95'][1]}] | [{ci['ci95_pct'][0]:+.2f}, {ci['ci95_pct'][1]:+.2f}] | {changed} | {fixed} | {broke} | {tch} | {reused} |")
        summary["conds"][v] = {"content_correct": int(ok[v].sum()), "acc": float(ok[v].mean()), "delta": ci,
                               "letters_changed": changed, "wrong_to_right": fixed, "right_to_wrong": broke,
                               "text_changed": tch, "stock_tail_reused": reused,
                               "transitions": {f"{a}->{b}": c for (a, b), c in trans.items()}}
    lines += ["", "## breakdown by stock commit step", "",
              "| stock commit step | n | " + " | ".join(f"{v} correct" for v in conds) + " | " + " | ".join(f"{v} changed" for v in conds[1:]) + " |",
              "|---|---|" + "---|" * (2 * len(conds) - 1)]
    by_step = {}
    for step in sorted(hist):
        sel = [k for k, i in enumerate(ids) if st[i]["commit_step"] == step]
        row = {v: int(ok[v][sel].sum()) for v in conds}
        row_ch = {v: int(sum(data[v][ids[k]]["letter"] != st[ids[k]]["letter"] for k in sel)) for v in conds[1:]}
        by_step[str(step)] = {"n": len(sel), "correct": row, "changed": row_ch}
        lines.append(f"| {step} | {len(sel)} | " + " | ".join(str(row[v]) for v in conds) + " | " + " | ".join(str(row_ch[v]) for v in conds[1:]) + " |")
    summary["by_stock_commit_step"] = by_step
    if "hold_agg_mean" in conds:
        h = data["hold_agg_mean"]
        hr = np.array([int(h[i].get("hold_rule_letter") == gold[i]) for i in ids])
        hl = np.array([int(h[i].get("hold_agg_late_letter") == gold[i]) for i in ids])
        lines += ["", f"hold diagnostics (slot token only, not re-scored on text): hold stock-rule token correct {hr.sum()}/{n}; "
                      f"hold agg_late token correct {hl.sum()}/{n}"]
        summary["hold_rule_token_correct"] = int(hr.sum()); summary["hold_agg_late_token_correct"] = int(hl.sum())
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines), flush=True)


def gate(out_dir: Path) -> int:
    """exit 0 if any agg cond has delta >= 0 vs stock in summary.json, else 1 (used by the lladav pilot chain)."""
    s = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    deltas = {v: d["delta"]["delta"] for v, d in s["conds"].items() if v != "stock"}
    ok = any(d >= 0 for d in deltas.values())
    print(f"GATE n={s['n']} deltas={deltas} -> {'CONTINUE' if ok else 'STOP'}", flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("decode", "report", "gate"), default="decode")
    p.add_argument("--backend", choices=("lavida", "dreamvl", "mmada", "lladav"), required=True)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--n", type=int, default=2017)
    p.add_argument("--conds", type=str, default=None, help="comma list; default per backend")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    args.out_dir = (args.out_dir or OUT_ROOT / args.backend).resolve()
    check_out(args.out_dir)
    conds = tuple(args.conds.split(",")) if args.conds else CONDS_BY_BACKEND[args.backend]
    if conds[0] != "stock" or any(c not in CONDS_BY_BACKEND[args.backend] for c in conds):
        raise RuntimeError(f"conds {conds}")
    args.conds = conds
    if args.mode == "decode":
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
        run_decode(args)
    elif args.mode == "report":
        report(args.out_dir, args.backend, conds)
    else:
        sys.exit(gate(args.out_dir))


if __name__ == "__main__":
    main()
