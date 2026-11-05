#!/usr/bin/env python3
"""Commit-step prior shift.

The answer slot is rewritten every step. The letter that remains is the write
present when that slot is unmasked, and that write still reads the image.
Earlier steps stay on the unshifted logits, so the unmask order matches stock.
Only the token written into the answer slot is corrected, and only when the
option-letter margin is below COMMIT_TAU.

COMMIT_MODE=margin_only applies whenever the margin is below tau.
COMMIT_MODE=prior_and_margin also requires the current argmax to be the prior letter.
The shift is -alpha_spread * alpha_scale * log p_cf. alpha_scale comes from the CLI.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import run_unmask_order as base

TAU = float(os.environ.get("COMMIT_TAU", "0.05"))
MODE = os.environ.get("COMMIT_MODE", "margin_only")


def _prior_index(log_pcf, n: int) -> int:
    return max(range(n), key=lambda j: (float(log_pcf[j]), -j))


def _should_shift(logits_row, ids, n: int, log_pcf) -> bool:
    if n < 2:
        return False
    raw = logits_row[list(ids[:n])].detach().float()
    if MODE == "prior_and_margin":
        pred = int(torch.argmax(raw).item())
        if pred != _prior_index(log_pcf, n):
            return False
    elif MODE != "margin_only":
        raise RuntimeError(f"unknown COMMIT_MODE {MODE}")
    prob = torch.softmax(raw, dim=0)
    top2 = torch.topk(prob, 2).values
    return float(top2[0] - top2[1]) < TAU


def _select_pos(x, logits, mask_id: int) -> tuple[int, torch.Tensor]:
    mask_index = x.eq(mask_id)
    chosen = torch.argmax(logits, dim=-1)
    probs = F.softmax(logits.double(), dim=-1)
    conf = torch.gather(probs, -1, chosen.unsqueeze(-1)).squeeze(-1)
    conf = torch.where(mask_index, conf, torch.full_like(conf, float("-inf")))
    pos = int(torch.topk(conf[0], k=1).indices[0].item())
    if not bool(mask_index[0, pos].item()):
        raise RuntimeError(f"selected unmasked position {pos}")
    return pos, chosen


def _install_e() -> None:
    import calib_decode_multi_20261013e as e

    def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
        if x.shape[1] != e.GEN:
            raise RuntimeError(f"canvas {tuple(x.shape)} != gen {e.GEN}")
        n = e.STATE["n_opt"]
        all_ids = e.STATE["label_ids"]
        ids = all_ids[:n]
        calib = e.STATE["mode"] == "gated"
        shift = None
        log_pcf = None
        if calib:
            a = e.STATE["alpha"][n] * e.STATE["alpha_scale"]
            log_pcf = e.STATE["log_pcf"][n]
            shift = (a * torch.tensor(log_pcf, dtype=torch.float64)).to(torch.float32)
        unmask_step = [-1] * e.GEN
        pos0_argmax = []
        raw_at_commit = cal_at_commit = None
        step0 = None
        shifted = 0
        for step in range(e.GEN):
            logits = forward(x)
            if int(x[0, 0].item()) == int(mask_id):
                raw = e.label_vals(logits[0, 0], all_ids)
                pos, chosen = _select_pos(x, logits, mask_id)
                cal = raw
                wrote_shifted = False
                if pos == 0 and calib and _should_shift(logits[0, 0], ids, n, log_pcf):
                    logits = logits.float()
                    logits[0, 0, ids] = logits[0, 0, ids] - shift.to(logits.device)
                    cal = e.label_vals(logits[0, 0], all_ids)
                    x[0, 0] = int(torch.argmax(logits[0, 0]).item())
                    wrote_shifted = True
                    shifted += 1
                else:
                    x[0, pos] = chosen[0, pos]
                pos0_argmax.append(base.piece(tokenizer, int(x[0, 0].item()) if pos == 0 else int(torch.argmax(logits[0, 0]).item()), mask_id))
                if step == 0:
                    step0 = {"raw": e.restricted_label(raw, n), "cal": e.restricted_label(cal, n)}
                if pos == 0:
                    raw_at_commit, cal_at_commit = raw, cal
                unmask_step[pos] = step
                if wrote_shifted:
                    continue
            else:
                pos = base.unmask_one(x, logits, mask_id)
                unmask_step[pos] = step
        letter_step = unmask_step[0]
        if letter_step < 0:
            raise RuntimeError("answer position was never selected")
        if len(pos0_argmax) != letter_step + 1:
            raise RuntimeError(f"recorded {len(pos0_argmax)} argmax steps, answer unmasked at {letter_step}")
        if sorted(unmask_step) != list(range(e.GEN)):
            raise RuntimeError(f"unmask steps {unmask_step}")
        if step0 is None:
            raise RuntimeError("step 0 did not see a masked answer slot")
        pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
        return {
            "letter_unmask_step": int(letter_step),
            "pos0_argmax": pos0_argmax,
            "letter_logits_raw_at_commit": raw_at_commit,
            "letter_logits_cal_at_commit": cal_at_commit,
            "step0_restricted_raw": step0["raw"],
            "step0_restricted_cal": step0["cal"],
            "finished_text": pred,
            "finished_letter": e.finished_label(pred),
            "commit_shifted": shifted,
            "commit_tau": TAU,
            "commit_mode": MODE,
        }

    e.decode_canvas = decode_canvas


def install() -> None:
    _install_e()
