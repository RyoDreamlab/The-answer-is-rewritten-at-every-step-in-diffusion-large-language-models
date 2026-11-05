#!/usr/bin/env python3
"""Prior-tie release for the every-step prior shift.

The answer slot is rewritten every step, and the letter that remains is the write
present when the slot is unmasked. That write still reads the image. Subtracting
the content-free prior on a write that has already separated from a tie pulls the
surviving letter off the image.

This patch keeps the same shift (-alpha * log p_cf) but applies it on a step only
when both are true: the current argmax among the option letters is the prior
letter (argmax of p_cf), and the softmax margin between the top two option
letters is < 0.05. Otherwise the logits are left unchanged, including the unmask
step. A confident write stays even when it is the prior letter. A near-tie that
has already left the prior stays.
"""
from __future__ import annotations

import torch

import run_unmask_order as base

TAU = 0.05


def _prior_index(log_pcf, n: int) -> int:
    return max(range(n), key=lambda j: (float(log_pcf[j]), -j))


def _shift_now(logits_row, ids, n: int, log_pcf) -> bool:
    """True only when the option argmax is the prior and the top-two margin is a near-tie."""
    if n < 2:
        return False
    raw = logits_row[list(ids[:n])].detach().float()
    pred = int(torch.argmax(raw).item())
    if pred != _prior_index(log_pcf, n):
        return False
    prob = torch.softmax(raw, dim=0)
    top2 = torch.topk(prob, 2).values
    return float(top2[0] - top2[1]) < TAU


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
                if calib and _shift_now(logits[0, 0], ids, n, log_pcf):
                    logits = logits.float()
                    logits[0, 0, ids] = logits[0, 0, ids] - shift.to(logits.device)
                    shifted += 1
                cal = e.label_vals(logits[0, 0], all_ids)
                pos0_argmax.append(base.piece(tokenizer, int(torch.argmax(logits[0, 0]).item()), mask_id))
                if step == 0:
                    step0 = {"raw": e.restricted_label(raw, n), "cal": e.restricted_label(cal, n)}
                raw_at_commit, cal_at_commit = raw, cal
            pos = base.unmask_one(x, logits, mask_id)
            unmask_step[pos] = step
        letter_step = unmask_step[0]
        if letter_step < 0:
            raise RuntimeError("answer position was never selected")
        if len(pos0_argmax) != letter_step + 1:
            raise RuntimeError(f"recorded {len(pos0_argmax)} argmax steps, answer unmasked at {letter_step}")
        if sorted(unmask_step) != list(range(e.GEN)):
            raise RuntimeError(f"unmask steps {unmask_step}")
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
            "priortie_shifted_steps": shifted,
            "priortie_tau": TAU,
            "priortie_rule": "shift_only_when_argmax_is_prior_and_margin_lt_0.05",
        }

    e.decode_canvas = decode_canvas


def _install_c() -> None:
    import calib_decode_multi_20261013c as c

    def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
        if x.shape[1] != c.GEN:
            raise RuntimeError(f"canvas {tuple(x.shape)} != gen {c.GEN}")
        n = c.STATE["n_opt"]
        ids = c.STATE["letter_ids"][:n]
        calib = c.STATE["mode"] == "gated"
        shift = None
        log_pcf = None
        if calib:
            a = c.STATE["alpha"][n] * c.STATE["alpha_scale"]
            log_pcf = c.STATE["log_pcf"][n]
            shift = (a * torch.tensor(log_pcf, dtype=torch.float64)).to(torch.float32)
        unmask_step = [-1] * c.GEN
        pos0_argmax = []
        raw_at_commit = cal_at_commit = None
        step0 = None
        shifted = 0
        for step in range(c.GEN):
            logits = forward(x)
            if int(x[0, 0].item()) == int(mask_id):
                raw = c.letter_vals(logits[0, 0], c.STATE["letter_ids"])
                if calib and _shift_now(logits[0, 0], ids, n, log_pcf):
                    logits = logits.float()
                    logits[0, 0, ids] = logits[0, 0, ids] - shift.to(logits.device)
                    shifted += 1
                cal = c.letter_vals(logits[0, 0], c.STATE["letter_ids"])
                pos0_argmax.append(base.piece(tokenizer, int(torch.argmax(logits[0, 0]).item()), mask_id))
                if step == 0:
                    step0 = {"raw": c.restricted_letter(raw, n), "cal": c.restricted_letter(cal, n)}
                raw_at_commit, cal_at_commit = raw, cal
            pos = base.unmask_one(x, logits, mask_id)
            unmask_step[pos] = step
        letter_step = unmask_step[0]
        if letter_step < 0:
            raise RuntimeError("letter position was never selected")
        if len(pos0_argmax) != letter_step + 1:
            raise RuntimeError(f"recorded {len(pos0_argmax)} argmax steps, letter unmasked at {letter_step}")
        if sorted(unmask_step) != list(range(c.GEN)):
            raise RuntimeError(f"unmask steps {unmask_step}")
        pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
        return {
            "letter_unmask_step": int(letter_step),
            "pos0_argmax": pos0_argmax,
            "letter_logits_raw_at_commit": raw_at_commit,
            "letter_logits_cal_at_commit": cal_at_commit,
            "step0_restricted_raw": step0["raw"],
            "step0_restricted_cal": step0["cal"],
            "finished_text": pred,
            "finished_letter": c.delta.leading_letter(pred),
            "priortie_shifted_steps": shifted,
            "priortie_tau": TAU,
            "priortie_rule": "shift_only_when_argmax_is_prior_and_margin_lt_0.05",
        }

    c.decode_canvas = decode_canvas


def install() -> None:
    _install_e()
    _install_c()
