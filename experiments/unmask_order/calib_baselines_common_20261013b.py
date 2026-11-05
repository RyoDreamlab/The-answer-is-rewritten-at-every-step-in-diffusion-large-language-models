#!/usr/bin/env python3
"""Shared code for the prior-work calibration baselines (2026-10-13b). Imported by
calib_baselines_decode_20261013b.py (ScienceQA-IMG) and calib_baselines_decode_m3cot_20261013b.py (M3CoT).

Rules (all applied at canvas position 0, at every decode step while that position is still masked, to the
n option-letter logits only -- the same point where calib_decode.py / calib_decode_m3cot.py apply ours):

  atabuzzaman   Atabuzzaman et al., EMNLP 2025 ("Benchmarking and Mitigating MCQA Selection Bias of LVLMs"),
                confidence-adaptive logit correction:
                    L' = L - alpha_adp * B
                    alpha_adp = ALPHA / (1 + exp(conf - TAU)),   conf = max(L) - mean(L) over the item's n letters
                    ALPHA = 1, TAU = 2 (paper defaults)
                    B = (B_gen + B_ctx) / 2
                    B_gen = softmax(content-free letter logits over n) = our a_global p_cf, zero-centred
                    B_ctx = mean over the first CTX_FRAC (10%) of items in manifest order of
                            softmax(step-0 raw letter logits over n), zero-centred
                B is a (centred) probability vector while L is in logit units -- implemented as written.
                conf / alpha_adp are recomputed from the raw L of every step.
  kumar         Kumar, ACL 2022 ("Answer-level Calibration"), scaled content-free calibration:
                    score(y) = logit(y) - g * log p_cf(y),   g = 1 - TVD(p_real, p_cf)
                p_real / p_cf = FULL-VOCABULARY softmax at canvas position 0, step 0, for the real and for the
                content-free input with the same n. g is fixed per item at step 0 and applied at every step while
                position 0 is masked. log p_cf(y) is the restricted (over n letters) log-softmax of the content-free
                letter logits -- the same vector our gated rule uses, so the only difference to ours is the scalar.
  none          no logit change (stock path; sanity).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]
ALPHA = 1.0
TAU = 2.0
CTX_FRAC = 0.10
RULES = ("atabuzzaman", "kumar", "none")

S = {
    "rule": "none",
    "step0_only": False,      # when True decode_canvas() does one forward and returns step-0 readouts
    "ready": False,           # pre-pass finished (B / cf_full_prob available)
    "B": None,                # atabuzzaman: {n: [n floats]}
    "cf_full_prob": {},       # kumar: {n: full-vocab prob tensor (float64, on device after first use)}
    "n_prepass_items": 0,
}


# ------------------------------------------------------------------ small math helpers

def softmax_list(v: list[float]) -> list[float]:
    t = torch.log_softmax(torch.tensor(v, dtype=torch.float64), dim=0)
    return [float(x) for x in t.exp()]


def centred(v: list[float]) -> list[float]:
    m = sum(v) / len(v)
    return [x - m for x in v]


def alpha_adp(letter_raw_n: list[float]) -> tuple[float, float]:
    conf = max(letter_raw_n) - sum(letter_raw_n) / len(letter_raw_n)
    return ALPHA / (1.0 + math.exp(conf - TAU)), conf


def tvd(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p.double() - q.double()).abs().sum())


def letter_vals(logits_pos0) -> list[float]:
    return [float(logits_pos0[i]) for i in LETTER_IDS]


def restricted_letter(vals: list[float], n: int) -> str:
    return LETTERS[max(range(n), key=lambda j: (vals[j], -j))]


# ------------------------------------------------------------------ bias vector B (atabuzzaman)

def build_B(p_cf_by_n: dict[int, list[float]], ctx_items: list[tuple[int, list[float]]]) -> dict:
    """p_cf_by_n: {n: p_cf over n letters}; ctx_items: [(n_opt, raw letter logits (5)) ...] of the 10% subset."""
    by_n = {}
    for n in (2, 3, 4, 5):
        sm = [softmax_list(raw[:n]) for k, raw in ctx_items if k == n]
        b_gen = centred(list(p_cf_by_n[n]))
        if sm:
            mean_sm = [sum(col) / len(sm) for col in zip(*sm)]
            b_ctx = centred(mean_sm)
            B = [(a + b) / 2 for a, b in zip(b_gen, b_ctx)]
            fallback = False
        else:
            mean_sm, b_ctx, B, fallback = None, None, b_gen, True
        by_n[n] = {"count_ctx": len(sm), "p_cf": list(p_cf_by_n[n]), "B_gen_centred": b_gen,
                   "mean_softmax_ctx": mean_sm, "B_ctx_centred": b_ctx, "B": B,
                   "fallback_B_gen_only": fallback}
    return by_n


def dump_B_table(path: Path, backend: str, dataset: str, by_n: dict, ctx_items_meta: list[dict]) -> None:
    if path.exists():
        return
    obj = {"rule": "atabuzzaman: L' = L - alpha_adp*B, alpha_adp = ALPHA/(1+exp(conf-TAU)), conf = max(L)-mean(L)",
           "ALPHA": ALPHA, "TAU": TAU, "ctx_frac": CTX_FRAC,
           "ctx_selection": "first ceil(10%) of items in manifest order; step-0 raw letter logits (full-mask canvas)",
           "backend": backend, "dataset": dataset, "n_ctx_items": len(ctx_items_meta),
           "by_n_opt": {str(n): v for n, v in by_n.items()}, "ctx_items": ctx_items_meta}
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def load_B_table(path: Path) -> dict[int, list[float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {int(n): v["B"] for n, v in obj["by_n_opt"].items()}


# ------------------------------------------------------------------ content-free full-vocab cache (kumar)

def dump_cf_fullvocab(path: Path, backend: str, dataset: str, per_n: dict[int, dict]) -> None:
    if path.exists():
        return
    obj = {"backend": backend, "dataset": dataset, "letter_ids": LETTER_IDS,
           "note": "model (bf16->float32) logits at canvas position 0, step 0, full-mask canvas, content-free input "
                   "(mid-gray 384x384 image, question N/A, n options N/A); rounded to 4 decimals",
           "by_n_opt": {str(n): v for n, v in per_n.items()}}
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def load_cf_fullvocab(path: Path) -> dict[int, torch.Tensor]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for n, v in obj["by_n_opt"].items():
        lg = torch.tensor(v["logits"], dtype=torch.float64)
        out[int(n)] = torch.softmax(lg, dim=0)
    return out


def cf_fullvocab_record(logits_pos0: torch.Tensor, n: int, ref_letter_logit_model: list[float] | None) -> dict:
    lg = logits_pos0.detach().float().cpu()
    letter = [float(lg[i]) for i in LETTER_IDS]
    diff = max(abs(a - b) for a, b in zip(letter, ref_letter_logit_model)) if ref_letter_logit_model else None
    p = torch.softmax(lg.double(), dim=0)
    return {"n_opt": n, "vocab_size": int(lg.numel()), "letter_logit_model": letter,
            "max_abs_diff_vs_cf_jsonl_letter_logit_model": diff,
            "letter_mass_fullvocab": float(p[LETTER_IDS[:n]].sum()),
            "argmax_id": int(torch.argmax(lg).item()),
            "logits": [round(float(x), 4) for x in lg]}


# ------------------------------------------------------------------ the decode

def make_decode_canvas(state: dict, base, delta):
    """state: the wrapped script's STATE dict (keys 'n_opt', 'log_pcf'). base = run_unmask_order,
    delta = measure_letter_hidden_delta."""

    def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
        gen = base.GEN
        if x.shape[1] != gen:
            raise RuntimeError(f"canvas {tuple(x.shape)} != gen {gen}")
        if S["step0_only"]:
            logits = forward(x)
            return {"step0_letter_raw": letter_vals(logits[0, 0]), "step0_full_logits": logits[0, 0].detach().float().cpu()}
        if not S["ready"]:
            raise RuntimeError("pre-pass not finished before decode")
        rule = S["rule"]
        n = state["n_opt"]
        ids = LETTER_IDS[:n]
        lp = None
        if rule == "kumar":
            lp = torch.tensor(state["log_pcf"][n], dtype=torch.float64)
        Bn = torch.tensor(S["B"][n], dtype=torch.float64) if rule == "atabuzzaman" else None
        unmask_step = [-1] * gen
        pos0_argmax, pos0_argmax_raw = [], []
        alphas, confs = [], []
        g = tv = tv_letters = None
        raw_at_commit = cal_at_commit = None
        step0 = None
        for step in range(gen):
            logits = forward(x)
            if int(x[0, 0].item()) == int(mask_id):
                raw = letter_vals(logits[0, 0])
                pos0_argmax_raw.append(base.piece(tokenizer, int(torch.argmax(logits[0, 0]).item()), mask_id))
                if rule != "none":
                    logits = logits.float()
                    if rule == "atabuzzaman":
                        a, conf = alpha_adp(raw[:n])
                        alphas.append(a)
                        confs.append(conf)
                        shift = a * Bn
                    else:  # kumar
                        if step == 0:
                            p_real = torch.softmax(logits[0, 0].double(), dim=0)
                            p_cf = S["cf_full_prob"][n]
                            if p_cf.device != p_real.device:
                                p_cf = p_cf.to(p_real.device)
                                S["cf_full_prob"][n] = p_cf
                            if p_cf.numel() != p_real.numel():
                                raise RuntimeError(f"vocab mismatch cf={p_cf.numel()} real={p_real.numel()}")
                            tv = tvd(p_real, p_cf)
                            g = 1.0 - tv
                            pr_l = torch.softmax(torch.tensor(raw[:n], dtype=torch.float64), dim=0)
                            tv_letters = tvd(pr_l, lp.exp())
                        shift = g * lp
                    logits[0, 0, ids] = logits[0, 0, ids] - shift.to(torch.float32).to(logits.device)
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
        out = {
            "rule": rule,
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
        if rule == "atabuzzaman":
            out.update({"alpha": alphas[0], "alpha_adp_step0": alphas[0], "alpha_adp_at_commit": alphas[-1],
                        "alpha_adp_steps": alphas, "conf_steps": confs, "B": S["B"][n]})
        elif rule == "kumar":
            out.update({"alpha": g, "g": g, "tvd_fullvocab": tv, "tvd_letters_only": tv_letters,
                        "g_letters_only": 1.0 - tv_letters})
        else:
            out.update({"alpha": 0.0})
        return out

    return decode_canvas


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
