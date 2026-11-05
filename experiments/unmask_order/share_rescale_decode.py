#!/usr/bin/env python3
"""Stock low-confidence decode with a hidden-state source-share rescale at canvas position 0.

At every decode step while canvas position 0 is still masked:
  1. normal full forward (stock attention path) -> logits for all 16 canvas positions and
     the ln_f input h_full at position 0;
  2. one batch-of-8 forward over the key-blocking subsets of hidden_shapley_split.SUBSETS
     (I image keys, T prompt-text keys, M still-masked canvas keys; already-written canvas
     keys and MMaDA in-prompt mask ids are "other" and never blocked; every query keeps its
     own key) -> phi_I, phi_T, phi_M at position 0 by the exact 3-player Shapley formula;
  3. h' = h_full + (a - 1) * P phi_src, P = projection onto span of J^T (w_y - w_top) over the
     item's option letters y != top (hidden_shapley_offline_rules.py rule 3), top = argmax of
     the float64 readout of h_full over the option letters;
  4. position-0 logits <- ff_out(ln_f(h')) over the full vocabulary, computed with the model's
     own ln_f / ff_out on the full captured hidden tensor with row 0 replaced.
All other positions keep the normal full-forward logits. The normal unmask_one rule then
picks the position and token. Once position 0 is written it is never touched again.
Gold is not an input to decoding.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
sys.path.insert(0, str(PROJECT / "experiments" / "caft"))
import hidden_shapley_split as hss  # noqa: E402
import image_key_block as ikb  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402
from graders import scienceqa_mc  # noqa: E402

N_TOTAL = 2017
RULE = {"lavida": ("I", 0.5), "mmada": ("M", 2.0)}
STOCK_LETTER = {"lavida": 1460, "mmada": 1059}
OFFLINE_RULE_COUNT = {"lavida": 1458, "mmada": 1071}
SPLIT_DIR = PROJECT / "outputs" / "hidden_shapley_split_20261010"
TRAJ_DIR = PROJECT / "outputs" / "letter_argmax_traj_20261010"
OUT_DIR = PROJECT / "outputs" / "share_rescale_decode_20261010"
EPS = 1e-5
SUBSETS = hss.SUBSETS
SOURCES = hss.SOURCES
FULL = hss.FULL
EMPTY = hss.EMPTY
IDX = {hss.label(s): i for i, s in enumerate(SUBSETS)}


class Cap:
    armed = False
    sl = None
    h = None


def _hook(_module, args):
    if not Cap.armed:
        return
    Cap.h = args[0][:, Cap.sl].detach().clone()


def arm(sl) -> None:
    Cap.armed, Cap.sl, Cap.h = True, sl, None


def take() -> torch.Tensor:
    Cap.armed = False
    if Cap.h is None:
        raise RuntimeError("ln_f input was not captured")
    return Cap.h


# ---------------------------------------------------------------- rule 3 math (copied)

def rms_jy_batch(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = h.shape[-1]
    s = (h * h).sum(-1, keepdim=True) / d + EPS
    return (v - h * ((h * v).sum(-1, keepdim=True) / (d * s))) / torch.sqrt(s)


def readout(h: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Float64 option-letter logits of ln_f/ff_out for h (..., d); W (5, d)."""
    return ((h * gamma) @ W.T) / torch.sqrt((h * h).sum(-1, keepdim=True) / h.shape[-1] + EPS)


def projected(h_full: torch.Tensor, phi_x: torch.Tensor, gamma, W, k: int, top: int) -> torch.Tensor:
    others = [y for y in range(k) if y != top]
    if not others:
        return torch.zeros_like(phi_x)
    U = gamma[None, :] * (W[others] - W[top][None, :])
    G = rms_jy_batch(h_full[None, :].expand(len(others), -1), U)
    Q, _ = torch.linalg.qr(G.T)
    return Q @ (Q.T @ phi_x)


def shapley_checked(hs64: dict) -> tuple[dict, float]:
    phi = hss.shapley(hs64)
    total = phi["I"] + phi["T"] + phi["M"]
    target = hs64[FULL] - hs64[EMPTY]
    err = float(torch.linalg.vector_norm(total - target))
    scale = float(torch.linalg.vector_norm(target)) + float(torch.linalg.vector_norm(hs64[FULL]))
    if not (err <= 1e-9 * max(scale, 1.0)):
        raise RuntimeError(f"Shapley sum check failed err={err} scale={scale}")
    return phi, err


# ---------------------------------------------------------------- per-model plumbing

class Ctx:
    """Holds model handles and readout tensors for one backend."""

    def __init__(self, backend, core, tokenizer, mask_id, device):
        self.backend = backend
        self.core = core
        self.tokenizer = tokenizer
        self.mask_id = int(mask_id)
        self.device = device
        norm = core.transformer.ln_f
        if type(norm).__name__ != "RMSLayerNorm" or abs(float(norm.eps) - EPS) > 1e-12:
            raise RuntimeError(f"unexpected ln_f {type(norm).__name__} eps={getattr(norm, 'eps', None)}")
        if getattr(core.config, "weight_tying", None) or getattr(core.config, "scale_logits", None):
            raise RuntimeError("weight_tying/scale_logits set; ff_out readout would be wrong")
        norm.register_forward_pre_hook(_hook)
        self.ln_f = norm
        self.ff_out = core.transformer.ff_out
        self.gamma = norm.weight.detach().to(device=device, dtype=torch.float64)
        lid = hss.letter_ids(tokenizer, hss.LETTERS)
        self.letter_ids = lid
        self.W = torch.stack([self.ff_out.weight[lid[l]] for l in hss.LETTERS]).detach().to(device=device, dtype=torch.float64)

    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.ff_out(self.ln_f(hidden))


def lavida_setup(ctx, model, processor, row, image_token_index):
    past, image_cols = ikb.lavida_prefix(ctx.core, model, processor, ctx.tokenizer, row, image_token_index)
    prefix_len = int(past[0][0].shape[-2])
    image_cols = image_cols.cpu()
    is_image = torch.zeros(prefix_len, dtype=torch.bool)
    is_image[image_cols] = True
    text_cols = torch.nonzero(~is_image, as_tuple=False).flatten()
    batch = len(SUBSETS)
    past8 = [tuple(t.expand(batch, *t.shape[1:]) for t in layer) for layer in past]
    lm = ctx.core
    gen = base.GEN

    def stock(x):
        hss.Visible.additive = None
        arm(slice(0, gen))
        logits = lm(None, input_embeddings=lm.transformer.wte(x), past_key_values=past, use_cache=False).logits
        hidden = take()
        if logits.shape[1] != gen:
            raise RuntimeError(f"canvas logits {tuple(logits.shape)}")
        return logits, hidden  # hidden (1, gen, d) = full captured ln_f input

    def subsets(x):
        masked = x[0].eq(ctx.mask_id).cpu()
        n_keys = prefix_len + gen
        cols = {"I": image_cols, "T": text_cols,
                "M": prefix_len + torch.nonzero(masked, as_tuple=False).flatten()}
        row_key = torch.arange(prefix_len, prefix_len + gen)
        allow = torch.stack([hss.visibility_rows(gen, n_keys, row_key, cols, s) for s in SUBSETS])
        additive = torch.zeros(allow.shape, dtype=torch.float32)
        additive[~allow] = hss.NEG
        hss.Visible.additive = additive.unsqueeze(1).to(device=ctx.device)
        try:
            arm(slice(0, 1))
            lm(None, input_embeddings=lm.transformer.wte(x).repeat(batch, 1, 1), past_key_values=past8,
               use_cache=False)
            h8 = take()[:, 0]
        finally:
            hss.Visible.additive = None
        return h8

    return stock, subsets, 0


def mmada_setup(ctx, vq, mm, image, question):
    prompt_ids, image_cols = ikb.mmada_prompt(ctx.core, ctx.tokenizer, vq, mm, image, question, ctx.device)
    prompt_len = int(prompt_ids.shape[1])
    gen = base.GEN
    seq = prompt_len + gen
    image_cols = image_cols.cpu()
    prompt_cpu = prompt_ids[0].cpu()
    is_image = torch.zeros(prompt_len, dtype=torch.bool)
    is_image[image_cols] = True
    text_cols = torch.nonzero(~is_image & ~prompt_cpu.eq(mm.MASK_ID), as_tuple=False).flatten()
    batch = len(SUBSETS)
    model = ctx.outer

    def full_x(x):
        return torch.cat([prompt_ids, x], dim=1)

    def stock(x):
        arm(slice(0, seq))
        logits = model(full_x(x)).logits
        hidden = take()
        if logits.shape[1] != seq:
            raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
        return logits[:, prompt_len:], hidden  # hidden (1, seq, d)

    def subsets(x):
        masked = x[0].eq(ctx.mask_id).cpu()
        cols = {"I": image_cols, "T": text_cols,
                "M": prompt_len + torch.nonzero(masked, as_tuple=False).flatten()}
        allow = torch.stack([hss.visibility_rows(seq, seq, torch.arange(seq), cols, s) for s in SUBSETS])
        allow = allow.unsqueeze(1).to(device=ctx.device)
        arm(slice(prompt_len, prompt_len + 1))
        out = model(full_x(x).repeat(batch, 1), attention_bias=allow)
        del out
        return take()[:, 0]

    return stock, subsets, prompt_len


# ---------------------------------------------------------------- decode

def decode(ctx, stock, subsets, h_off: int, n_choices: int, a: float, src: str, modify: bool) -> dict:
    gen = base.GEN
    x = torch.full((1, gen), ctx.mask_id, dtype=torch.long, device=ctx.device)
    letters = hss.option_letters(n_choices)
    k = len(letters)
    if modify and k < 2:
        modify = False
        skip = f"n_option_letters={k}"
    else:
        skip = None
    unmask_step = [-1] * gen
    traj = []
    sum_err_max = 0.0
    recon_diff_max = 0.0
    step0 = None
    at_unmask = None
    for step in range(gen):
        logits, hidden = stock(x)
        pos0_masked = int(x[0, 0].item()) == ctx.mask_id
        if pos0_masked:
            unmod_row = logits[0, 0].clone()
            unmod_id = int(torch.argmax(unmod_row).item())
            entry = {"step": step, "unmod": base.piece(ctx.tokenizer, unmod_id, ctx.mask_id)}
            if modify:
                h8 = subsets(x)
                hs64 = {s: h8[i].to(dtype=torch.float64) for i, s in enumerate(SUBSETS)}
                phi, err = shapley_checked(hs64)
                sum_err_max = max(sum_err_max, err)
                h_full = hidden[0, h_off].to(dtype=torch.float64)
                lo = readout(h_full, ctx.gamma, ctx.W)[:k]
                top = int(torch.argmax(lo).item())
                p = projected(h_full, phi[src], ctx.gamma, ctx.W, k, top)
                h_new = h_full + (a - 1.0) * p
                hid2 = hidden.clone()
                hid2[0, h_off] = h_new.to(dtype=hidden.dtype)
                new_rows = ctx.head_logits(hid2)
                if a == 1.0:
                    recon_diff_max = max(recon_diff_max, float((new_rows[0, h_off].float() - unmod_row.float()).abs().max()))
                logits[0, 0] = new_rows[0, h_off].to(dtype=logits.dtype)
                del new_rows, hid2
                mod_id = int(torch.argmax(logits[0, 0]).item())
                entry["mod"] = base.piece(ctx.tokenizer, mod_id, ctx.mask_id)
                lid = [ctx.letter_ids[l] for l in letters]
                if step0 is None:
                    step0 = {
                        "top_f64": letters[top],
                        "unmod_restricted": letters[int(torch.argmax(unmod_row[lid].float()).item())],
                        "mod_restricted": letters[int(torch.argmax(logits[0, 0, lid].float()).item())],
                        "mod_restricted_f64": letters[int(torch.argmax(readout(h_new, ctx.gamma, ctx.W)[:k]).item())],
                        "phi_src_norm": float(torch.linalg.vector_norm(phi[src])),
                        "proj_norm": float(torch.linalg.vector_norm(p)),
                    }
            traj.append(entry)
        pos = base.unmask_one(x, logits, ctx.mask_id)
        if unmask_step[pos] >= 0:
            raise RuntimeError(f"position {pos} unmasked twice")
        unmask_step[pos] = step
        if pos == 0:
            written = base.piece(ctx.tokenizer, int(x[0, 0].item()), ctx.mask_id)
            at_unmask = {"unmod_argmax": traj[-1]["unmod"], "written": written,
                         "unmod_letter": delta.leading_letter(traj[-1]["unmod"]),
                         "written_letter": delta.leading_letter(written)}
    if sorted(unmask_step) != list(range(gen)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    if len(traj) != unmask_step[0] + 1:
        raise RuntimeError(f"traj {len(traj)} vs letter step {unmask_step[0]}")
    pred = ctx.tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return {
        "finished_text": pred,
        "finished_letter": delta.leading_letter(pred),
        "letter_unmask_step": int(unmask_step[0]),
        "modified": modify,
        "skip_reason": skip,
        "option_letters": letters,
        "at_unmask": at_unmask,
        "letter_changed_vs_unmod_at_unmask": (at_unmask["unmod_letter"] != at_unmask["written_letter"]) if modify else False,
        "token_changed_vs_unmod_at_unmask": (at_unmask["unmod_argmax"] != at_unmask["written"]) if modify else False,
        "pos0_traj": traj,
        "step0": step0,
        "shapley_sum_err_max": sum_err_max,
        "recon_diff_max_a1": recon_diff_max,
    }


def official_row(row: dict) -> dict | None:
    gold = ikb.gold_of(row)
    if not gold:
        return None
    if row.get("gold") and (row.get("gold_text") or row.get("golds")):
        return row
    choices = row.get("choices")
    idx = row.get("answer_idx", row.get("answer_index"))
    if isinstance(choices, list) and idx is not None:
        return {"gold": gold, "gold_text": str(choices[int(idx)])}
    return None


def n_choices_of(row) -> int:
    import ast
    choices = row.get("choices")
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    return len(choices)


# ---------------------------------------------------------------- offline rule reference

def offline_preds(ctx, backend: str, src: str, a: float) -> dict:
    records = [json.loads(l) for l in (SPLIT_DIR / f"{backend}.jsonl").open(encoding="utf-8")]
    arr = np.load(SPLIT_DIR / f"{backend}_h_subsets.npy", mmap_mode="r")
    out = {}
    correct = 0
    for n, r in enumerate(records):
        H = torch.from_numpy(np.ascontiguousarray(arr[n])).to(device=ctx.device, dtype=torch.float64)
        hs64 = {s: H[i] for i, s in enumerate(SUBSETS)}
        phi = hss.shapley(hs64)
        h_full = H[IDX["ITM"]]
        k = len(r["option_letters"])
        top = int(torch.argmax(readout(h_full, ctx.gamma, ctx.W)[:k]).item())
        p = projected(h_full, phi[src], ctx.gamma, ctx.W, k, top)
        pred = int(torch.argmax(readout(h_full + (a - 1.0) * p, ctx.gamma, ctx.W)[:k]).item())
        out[r["id"]] = {"base": hss.LETTERS[top], "rule": hss.LETTERS[pred]}
        correct += int(hss.LETTERS[pred] == r["gold"])
    return {"preds": out, "correct": correct}


def load_traj(backend):
    rows = [json.loads(l) for l in (TRAJ_DIR / f"{backend}.jsonl").open(encoding="utf-8") if l.strip()]
    return {r["id"]: r for r in rows}


def sanity(args, ctx, rows, item_fns):
    src, a = RULE[args.backend]
    off = offline_preds(ctx, args.backend, src, a)
    print(f"OFFLINE_RECOMPUTE backend={args.backend} src={src} a={a} correct={off['correct']} "
          f"expected={OFFLINE_RULE_COUNT[args.backend]}", flush=True)
    if off["correct"] != OFFLINE_RULE_COUNT[args.backend]:
        raise RuntimeError("offline rule recompute does not reproduce the summary count")
    changed = [r["id"] for r in rows if off["preds"][r["id"]]["rule"] != off["preds"][r["id"]]["base"]]
    pick = [rows[0]["id"], rows[1]["id"]] + changed[:3]
    by_id = {r["id"]: r for r in rows}
    traj = load_traj(args.backend)
    report = {"backend": args.backend, "src": src, "a": a, "offline_correct": off["correct"], "items": []}
    ok = True
    for item_id in pick:
        row = by_id[item_id]
        nc = n_choices_of(row)
        t0 = time.time()
        with torch.no_grad():
            st, sub, off_h = item_fns(row)
            plain = decode(ctx, st, sub, off_h, nc, 1.0, src, modify=False)
            a1 = decode(ctx, st, sub, off_h, nc, 1.0, src, modify=True)
            ar = decode(ctx, st, sub, off_h, nc, a, src, modify=True)
        tr = traj[item_id]
        plain_pos0 = [e["unmod"] for e in plain["pos0_traj"]]
        o = off["preds"][item_id]
        checks = {
            "a1_text_eq_stock": a1["finished_text"] == plain["finished_text"],
            "a1_step_eq_stock": a1["letter_unmask_step"] == plain["letter_unmask_step"],
            "stock_eq_traj_file": plain_pos0 == tr["pos0_argmax"] and plain["finished_letter"] == tr["finished_letter"],
            "step0_rule_letter_eq_offline": ar["step0"]["mod_restricted"] == o["rule"],
            "step0_rule_letter_f64_eq_offline": ar["step0"]["mod_restricted_f64"] == o["rule"],
            "step0_top_eq_offline_base": ar["step0"]["top_f64"] == o["base"],
        }
        hard = ("a1_text_eq_stock", "a1_step_eq_stock", "step0_rule_letter_eq_offline")
        item_ok = all(checks[c] for c in hard)
        ok = ok and item_ok
        rec = {"id": item_id, "gold": ikb.gold_of(row), "offline": o, "checks": checks, "ok": item_ok,
               "stock_text": plain["finished_text"], "a1_text": a1["finished_text"], "rule_text": ar["finished_text"],
               "stock_step": plain["letter_unmask_step"], "rule_step": ar["letter_unmask_step"],
               "a1_recon_diff_max": a1["recon_diff_max_a1"], "step0_rule": ar["step0"], "step0_a1": a1["step0"],
               "shapley_sum_err_max": ar["shapley_sum_err_max"], "sec": round(time.time() - t0, 1)}
        report["items"].append(rec)
        print("SANITY " + json.dumps(rec, ensure_ascii=False), flush=True)
    report["pass"] = ok
    args.sanity_out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"SANITY_{'PASS' if ok else 'FAIL'} backend={args.backend} wrote {args.sanity_out}", flush=True)
    if not ok:
        raise SystemExit(3)


def tally(path: Path) -> dict:
    c = {"n": 0, "letter": 0, "official": 0, "official_scored": 0, "letter_changed": 0, "skipped": 0}
    if not path.exists():
        return c
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        c["n"] += 1
        c["letter"] += int(bool(r["finished_letter_correct"]))
        c["letter_changed"] += int(bool(r["letter_changed_vs_unmod_at_unmask"]))
        c["skipped"] += int(not r["modified"])
        if r["scienceqa_mc"] is not None:
            c["official_scored"] += 1
            c["official"] += int(bool(r["scienceqa_mc"]))
    return c


def item_record(args, ctx, row, item_fns, src, a) -> dict:
    with torch.no_grad():
        st, sub, off_h = item_fns(row)
        d = decode(ctx, st, sub, off_h, n_choices_of(row), a, src, modify=True)
    if d["skip_reason"]:
        print(f"SKIP_MOD id={row['id']} reason={d['skip_reason']}", flush=True)
    gold = ikb.gold_of(row)
    graded = official_row(row)
    return {"id": row["id"], "gold": gold, "backend": args.backend, "src": src, "a": a,
            **{k: d[k] for k in ("finished_text", "finished_letter", "letter_unmask_step", "modified",
                                 "skip_reason", "option_letters", "letter_changed_vs_unmod_at_unmask",
                                 "token_changed_vs_unmod_at_unmask", "at_unmask", "step0",
                                 "shapley_sum_err_max", "pos0_traj")},
            "finished_letter_correct": d["finished_letter"] == gold,
            "scienceqa_mc": None if graded is None else bool(scienceqa_mc(d["finished_text"], graded))}


def verify(args, ctx, rows, item_fns):
    """Recompute --verify-ids from rows and require byte-identical JSON to --verify-ref; writes nothing."""
    src, a = RULE[args.backend]
    ref = {}
    for line in Path(args.verify_ref).open(encoding="utf-8"):
        if line.endswith("\n") and line.strip():
            ref[json.loads(line)["id"]] = line.rstrip("\n")
    by_id = {r["id"]: r for r in rows}
    for item_id in args.verify_ids.split(","):
        t0 = time.time()
        got = json.dumps(item_record(args, ctx, by_id[item_id], item_fns, src, a), ensure_ascii=False)
        same = got == ref[item_id]
        print(f"VERIFY id={item_id} identical={int(same)} sec={time.time() - t0:.1f}", flush=True)
        if not same:
            exp = json.loads(ref[item_id])
            cur = json.loads(got)
            print(f"VERIFY_DIFF id={item_id} keys={[k for k in exp if exp.get(k) != cur.get(k)]}", flush=True)
            raise SystemExit(4)
    print(f"VERIFY_PASS n={len(args.verify_ids.split(','))}", flush=True)


def loop(args, ctx, rows, item_fns):
    src, a = RULE[args.backend]
    done = base.load_done(args.output)
    c = tally(args.output)
    if c["n"] != len(done):
        raise RuntimeError(f"jsonl has {c['n']} records but {len(done)} ids")
    t0 = time.time()
    n0 = c["n"]
    print(f"resume n_done={c['n']}/{len(rows)} src={src} a={a}", flush=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            rec = item_record(args, ctx, row, item_fns, src, a)
            gold = rec["gold"]
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(rec["id"])
            c["n"] += 1
            c["letter"] += int(rec["finished_letter_correct"])
            c["letter_changed"] += int(rec["letter_changed_vs_unmod_at_unmask"])
            c["skipped"] += int(not rec["modified"])
            if rec["scienceqa_mc"] is not None:
                c["official_scored"] += 1
                c["official"] += int(rec["scienceqa_mc"])
            el = time.time() - t0
            rate = (c["n"] - n0) / el if el > 0 else 0.0
            payload = {"backend": args.backend, "n_done": c["n"], "n_total": len(rows), "id": rec["id"],
                       "src": src, "a": a, "letter_correct": c["letter"], "scienceqa_mc": c["official"],
                       "scienceqa_mc_scored": c["official_scored"], "letter_changed_at_unmask": c["letter_changed"],
                       "skipped": c["skipped"], "elapsed_sec": round(el, 1), "items_per_sec": round(rate, 4),
                       "eta_sec": round((len(rows) - c["n"]) / rate, 0) if rate > 0 else None}
            base.write_progress(args.progress, payload)
            print(f"PROGRESS n_done={c['n']}/{len(rows)} id={rec['id']} step={rec['letter_unmask_step']} "
                  f"letter={rec['finished_letter']} gold={gold} changed={int(rec['letter_changed_vs_unmod_at_unmask'])} "
                  f"letter_correct={c['letter']} scienceqa_mc={c['official']}/{c['official_scored']} "
                  f"rate={rate:.3f}/s", flush=True)
    if c["n"] != len(rows):
        raise RuntimeError(f"finished {c['n']} of {len(rows)}")
    print(f"SUMMARY backend={args.backend} n={c['n']} letter_correct={c['letter']} stock_letter={STOCK_LETTER[args.backend]} "
          f"scienceqa_mc={c['official']}/{c['official_scored']} letter_changed_at_unmask={c['letter_changed']} "
          f"skipped={c['skipped']}", flush=True)


# ---------------------------------------------------------------- loaders

def build_lavida(args):
    sys.path.insert(0, str(PROJECT / "repos" / "LaViDa"))
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.model.builder import load_pretrained_model

    siglip = (PROJECT / "hf-cache/hub/models--google--siglip-so400m-patch14-384/snapshots/"
              "9fdffc58afc957d1a03a25b10dba0329ab15c2a3")
    vision_kwargs = {
        "mm_vision_tower": str(siglip) if siglip.exists() else "google/siglip-so400m-patch14-384",
        "mm_resampler_type": None, "mm_projector_type": "mlp2x_gelu", "mm_hidden_size": 1152, "use_mm_proj": True,
    }
    print("loading lavida", flush=True)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, None, "llava_llada", device_map=f"cuda:{args.cuda}",
        vision_kwargs=vision_kwargs, torch_dtype="bfloat16",
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    if int(getattr(model.config, "mask_token_id", -1)) != base.MASK_ID_LAVIDA:
        raise RuntimeError("LaViDa mask id mismatch")
    lm = model.get_model()
    n_blocks = hss.install_visibility(lm)
    ctx = Ctx("lavida", lm, tokenizer, base.MASK_ID_LAVIDA, model.device)
    print(f"lavida core={type(lm).__name__} blocks={n_blocks} prefix_lm={getattr(model.config, 'prefix_lm', 'unset')} "
          f"mask={base.MASK_ID_LAVIDA} cuda={args.cuda}", flush=True)
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    return ctx, rows, lambda row: lavida_setup(ctx, model, processor, row, IMAGE_TOKEN_INDEX)


def build_mmada(args):
    sys.path.insert(0, str(PROJECT / "experiments"))
    sys.path.insert(0, str(PROJECT / "repos" / "MMaDA"))
    import evdlc_mmada as mm
    import sciqa_standard_eval as sq

    class Ns:
        model = args.model_path
        vq_model = args.vq_model
        mmada_repo = args.mmada_repo

    print("loading mmada", flush=True)
    device = torch.device(f"cuda:{args.cuda}")
    model, tokenizer = mm.load_mmada(Ns, device)
    if type(model).__name__ != "MMadaModelLM":
        raise RuntimeError(f"expected MMadaModelLM, got {type(model).__name__}")
    core = model.model
    vq = mm.load_vq(Ns, device)
    model.eval()
    ctx = Ctx("mmada", core, tokenizer, mm.MASK_ID, device)
    ctx.outer = model
    print(f"mmada outer={type(model).__name__} core={type(core).__name__} "
          f"language={type(core.transformer).__name__} mask={mm.MASK_ID} cuda={args.cuda}", flush=True)
    rows = sq.load_manifest(Path(args.manifest), Path(args.data_dir))
    loader = None
    for name in dir(sq):
        obj = getattr(sq, name)
        if isinstance(obj, type) and hasattr(obj, "load_rgb"):
            loader = obj(Path(args.data_dir), rows)
            break
    if loader is None:
        raise RuntimeError("ScienceQA image loader was not found")

    def fns(row):
        return mmada_setup(ctx, vq, mm, loader.load_rgb(row), sq.build_task_prompt(row))

    return ctx, rows, fns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("lavida", "mmada"), required=True)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--model-path", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    p.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    p.add_argument("--start", type=int, default=0, help="first manifest index (inclusive)")
    p.add_argument("--end", type=int, default=N_TOTAL, help="last manifest index (exclusive)")
    p.add_argument("--tag", default="", help="shard tag: writes <backend>_<tag>.jsonl with its own lock/progress")
    p.add_argument("--verify-ids", default="", help="comma-separated ids to recompute before the loop")
    p.add_argument("--verify-ref", default="", help="jsonl holding the reference records for --verify-ids")
    args = p.parse_args()
    if not 0 <= args.start < args.end <= N_TOTAL:
        raise RuntimeError(f"bad range [{args.start}, {args.end})")
    if args.start != 0 and not args.tag:
        raise RuntimeError("a range not starting at 0 needs --tag")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
    if base.GEN != 16 or base.MASK_ID_LAVIDA != 126336:
        raise RuntimeError("unexpected gen/mask")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.backend}_{args.tag}" if args.tag else args.backend
    args.output = OUT_DIR / f"{stem}.jsonl"
    args.progress = OUT_DIR / f"{stem}.progress.json"
    args.sanity_out = OUT_DIR / f"sanity_{args.backend}.json"
    if not args.sanity and base.load_done(args.output).__len__() >= args.end - args.start:
        print(f"ALREADY_DONE {args.output}", flush=True)
        return
    if args.sanity and args.sanity_out.exists():
        raise RuntimeError(f"refusing to overwrite {args.sanity_out}")
    lock_name = f"{args.backend}.sanity.lock" if args.sanity else f"{stem}.jsonl.lock"
    lock = (OUT_DIR / lock_name).open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running: {lock_name}") from exc
    if not args.sanity:
        ikb.repair_tail(args.output)
    torch.cuda.set_device(args.cuda)
    src, a = RULE[args.backend]
    print(f"start backend={args.backend} cuda={args.cuda} sanity={int(args.sanity)} src={src} a={a} gen={base.GEN} "
          f"steps={base.GEN} block_length={base.GEN} temperature=0 remasking=low_confidence gold_in_decode=0", flush=True)
    try:
        ctx, rows, fns = build_lavida(args) if args.backend == "lavida" else build_mmada(args)
        if len(rows) != N_TOTAL:
            raise RuntimeError(f"expected {N_TOTAL} rows, got {len(rows)}")
        if args.sanity:
            sanity(args, ctx, rows, fns)
        else:
            if args.verify_ids:
                verify(args, ctx, rows, fns)
            span = rows[args.start:args.end]
            print(f"range [{args.start}, {args.end}) first={span[0]['id']} last={span[-1]['id']} "
                  f"output={args.output}", flush=True)
            loop(args, ctx, span, fns)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
