#!/usr/bin/env python3
"""Unmask-step letter pick that must beat the current best on a single-digit cell.

The unmask order stays the stock order. Only the token written when the answer
slot is unmasked may change, and only to a token that slot already wrote.

HYST_RULE:
  early_m:TAU   first-half majority when the option margin is below TAU
  gate:R:S:T    if the final letter has held for at most R steps, the trajectory
                switched at most S times, and the option margin is at most T,
                restore the previous different letter
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import run_unmask_order as base

RULE = os.environ.get("HYST_RULE", "")


def _cf(text: str) -> str:
    return (text or "").casefold()


def choose_index(pieces: list[str], margin: float, rule: str) -> int:
    last = len(pieces) - 1
    if last < 0:
        raise RuntimeError("empty answer trajectory")
    if rule.startswith("gate:"):
        run_max_s, sw_hi_s, tau_s = rule.split(":")[1:]
        run_max, sw_hi, tau = int(run_max_s), int(sw_hi_s), float(tau_s)
        last_key = _cf(pieces[last])
        run = 0
        for piece in reversed(pieces):
            if _cf(piece) != last_key:
                break
            run += 1
        switches = sum(_cf(a) != _cf(b) for a, b in zip(pieces, pieces[1:]))
        prev = None
        for idx in range(last, -1, -1):
            if _cf(pieces[idx]) != last_key:
                prev = idx
                break
        if prev is None or run == 0 or run > run_max or switches > sw_hi or margin > tau:
            return last
        return prev
    kind, arg_s = rule.split(":")
    arg = float(arg_s)
    if kind == "early_m":
        if margin >= arg:
            return last
        half_end = max(1, len(pieces) // 2)
        half = pieces[:half_end]
        counts: dict[str, int] = {}
        for piece in half:
            key = _cf(piece)
            counts[key] = counts.get(key, 0) + 1
        best_key = None
        best_n = -1
        final_key = _cf(half[-1])
        for piece in half:
            key = _cf(piece)
            n = counts[key]
            if n > best_n or (n == best_n and key == final_key):
                best_key, best_n = key, n
        for idx in range(half_end - 1, -1, -1):
            if _cf(pieces[idx]) == best_key:
                return idx
        return last
    raise RuntimeError(f"unknown HYST_RULE {rule}")


def _option_margin(logits_row, ids, n: int) -> float:
    if n < 2:
        return 1.0
    raw = logits_row[list(ids[:n])].detach().float()
    prob = torch.softmax(raw, dim=0)
    top2 = torch.topk(prob, 2).values
    return float(top2[0] - top2[1])


def _select_pos(x, logits, mask_id: int):
    mask_index = x.eq(mask_id)
    chosen = torch.argmax(logits, dim=-1)
    probs = F.softmax(logits.double(), dim=-1)
    conf = torch.gather(probs, -1, chosen.unsqueeze(-1)).squeeze(-1)
    conf = torch.where(mask_index, conf, torch.full_like(conf, float("-inf")))
    pos = int(torch.topk(conf[0], k=1).indices[0].item())
    if not bool(mask_index[0, pos].item()):
        raise RuntimeError(f"selected unmasked position {pos}")
    return pos, chosen


def _vals(mod, logits_row, ids):
    if hasattr(mod, "label_vals"):
        return mod.label_vals(logits_row, ids)
    return mod.letter_vals(logits_row, ids)


def _restricted(mod, raw, n: int) -> str:
    if hasattr(mod, "restricted_label"):
        return mod.restricted_label(raw, n)
    return mod.restricted_letter(raw, n)


def _finished(mod, pred: str):
    if hasattr(mod, "finished_label"):
        return mod.finished_label(pred)
    import measure_letter_hidden_delta as delta

    return delta.leading_letter(pred)


def _install(mod) -> None:
    if not RULE:
        raise RuntimeError("HYST_RULE is empty")

    def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
        if x.shape[1] != mod.GEN:
            raise RuntimeError(f"canvas {tuple(x.shape)} != gen {mod.GEN}")
        n = mod.STATE["n_opt"]
        all_ids = mod.STATE["label_ids"] if "label_ids" in mod.STATE else mod.STATE["letter_ids"]
        ids = all_ids[:n]
        unmask_step = [-1] * mod.GEN
        pos0_argmax = []
        pos0_ids = []
        raw_at_commit = None
        step0 = None
        replaced = 0
        for step in range(mod.GEN):
            logits = forward(x)
            if int(x[0, 0].item()) == int(mask_id):
                raw = _vals(mod, logits[0, 0], all_ids)
                pos, chosen = _select_pos(x, logits, mask_id)
                tid = int(torch.argmax(logits[0, 0]).item())
                piece = base.piece(tokenizer, tid, mask_id)
                pos0_argmax.append(piece)
                pos0_ids.append(tid)
                if pos == 0:
                    margin = _option_margin(logits[0, 0], ids, n)
                    pick = choose_index(pos0_argmax, margin, RULE)
                    x[0, 0] = pos0_ids[pick]
                    if pick != len(pos0_argmax) - 1:
                        replaced += 1
                    raw_at_commit = raw
                else:
                    x[0, pos] = chosen[0, pos]
                if step == 0:
                    shown = _restricted(mod, raw, n)
                    step0 = {"raw": shown, "cal": shown}
                unmask_step[pos] = step
            else:
                pos = base.unmask_one(x, logits, mask_id)
                unmask_step[pos] = step
        letter_step = unmask_step[0]
        if letter_step < 0:
            raise RuntimeError("answer position was never selected")
        if len(pos0_argmax) != letter_step + 1:
            raise RuntimeError(f"recorded {len(pos0_argmax)} argmax steps, answer unmasked at {letter_step}")
        if sorted(unmask_step) != list(range(mod.GEN)):
            raise RuntimeError(f"unmask steps {unmask_step}")
        if step0 is None or raw_at_commit is None:
            raise RuntimeError("answer slot was never masked at step 0")
        pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
        return {
            "letter_unmask_step": int(letter_step),
            "pos0_argmax": pos0_argmax,
            "letter_logits_raw_at_commit": raw_at_commit,
            "letter_logits_cal_at_commit": raw_at_commit,
            "step0_restricted_raw": step0["raw"],
            "step0_restricted_cal": step0["cal"],
            "finished_text": pred,
            "finished_letter": _finished(mod, pred),
            "hyst_rule": RULE,
            "hyst_replaced": replaced,
        }

    mod.decode_canvas = decode_canvas


def install() -> None:
    import calib_decode_multi_20261013c as c
    import calib_decode_multi_20261013e as e

    _install(e)
    _install(c)
