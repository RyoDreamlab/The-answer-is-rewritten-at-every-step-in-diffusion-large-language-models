#!/usr/bin/env python3
"""Upgrade 1 — the letter prior as an attractor of denoising: perturbation decay, calibration placement, teacher forcing.

All variants are 16-step, gen-16, temperature-0, low-confidence-remasking decodes (run_unmask_order.unmask_one), on the
first --limit ScienceQA-IMG manifest items.  Per item the STOCK decode is run first and cached (canvas after every step,
answer-position letter logits at every step, top non-letter logit per step, stock step-0 full logits).  Variants:

  perturb_{anti,second}_d{δ}   step 0 ONLY: letter logits at the answer position (canvas index 0) += δ·u.
        anti   : u = unit vector along −centered(log p_cf[:n])      (anti-prior)
        second : u = (e_second − e_first)/√2 of the stock step-0 restricted letter logits (toward 2nd-best letter)
        δ ∈ {0, 0.5, 1, 2, 4, 8}.  The decode is memoryless except through the canvas: if the perturbed step 0 commits
        the same (position, token) as stock, every later step is identical to stock (same canvas → same forward) and the
        stock trajectory is reused (branch=false; --no-shortcut recomputes it to verify).  Otherwise the branch is
        decoded to the end (branch=true).
  calib_step0    gated calibration  logits[letters] −= α_n·log p_cf  (α_n = 1 − p_min/p_max, calib_gated_decode_20261013)
                 applied at step 0 only.
  calib_commit   applied only at the stock commit step of the answer position (step where stock unmasks index 0); the
                 canvas up to that step is the stock canvas; the committed token is the calibrated full-vocab argmax; the
                 remaining steps (if any) are re-decoded.
  calib_every    applied at every step while index 0 is masked (= calib_gated_decode_20261013 path).
  calib_tf       teacher-forced canvas: step 0 as calib_step0; from step 1 the 15 non-answer positions carry the stock
                 FINAL tokens, so the only free variable is the letter.  Also records the raw (uncalibrated) letter under
                 the same forced canvas, and a context-dose sweep: k ∈ {4, 8, 12} stock tokens (first k stock-unmasked
                 positions) present, answer position masked, raw and calibrated letters.

Outputs: outputs/attractor_20261013d/{backend}_{variant}.jsonl, summary via --mode report (summary.json, decay_{m}.png,
placement.png).  New files only.
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
import calib_decode as cd  # noqa: E402
import dla_circuit_20261013d as dla  # noqa: E402  (model loading / prompt building, hooks stay disarmed)
import image_key_block as ikb  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "attractor_20261013d").resolve()
STOCK_DIR = PROJECT / "outputs" / "letter_argmax_traj_20261010"
GATED_DIR = PROJECT / "outputs" / "calib_gated_decode_20261013"
LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]
DELTAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
DOSES = [4, 8, 12]
SEED, N_BOOT = 20261013, 1000


def variants() -> list[str]:
    v = ["stock"]
    for u in ("anti", "second"):
        for d in DELTAS:
            v.append(f"perturb_{u}_d{d:g}")
    v += ["calib_step0", "calib_commit", "calib_every", "calib_tf"]
    return v


def gated_shift(backend: str) -> dict[int, list[float]]:
    lp = cd.load_priors(backend)
    out = {}
    for n, l in lp.items():
        alpha = 1.0 - math.exp(-(max(l) - min(l)))
        out[n] = {"alpha": alpha, "log_pcf": l, "shift": [alpha * v for v in l]}
    return out


def restricted(vals, n) -> str:
    return LETTERS[max(range(n), key=lambda j: (float(vals[j]), -j))]


def letters_of(logits_row) -> list[float]:
    return [float(v) for v in logits_row[LETTER_IDS]]


def nonletter_max(logits_row) -> tuple[float, int]:
    row = logits_row.detach().float().clone()
    row[LETTER_IDS] = float("-inf")
    i = int(torch.argmax(row).item())
    return float(row[i]), i


class Decoder:
    def __init__(self, ctx, shift_table):
        self.ctx = ctx
        self.gen = ctx["gen"]
        self.mask_id = ctx["mask_id"]
        self.tok = ctx["tokenizer"]
        self.shift_table = shift_table
        self.n_forward = 0

    def fwd(self, item, x):
        self.n_forward += 1
        return item["forward"](x)

    def run(self, item, x, start_step, logits0=None, mod=None, stop_after_commit=False):
        """Decode from canvas x (modified in place) at step start_step to the end.
        logits0: pre-computed (possibly modified) logits for start_step. mod(step, logits)->logits applied when index 0
        is masked (for every-step calibration). Returns per-step records from start_step."""
        recs = []
        for step in range(start_step, self.gen):
            masked0 = int(x[0, 0].item()) == int(self.mask_id)
            logits = logits0 if (step == start_step and logits0 is not None) else self.fwd(item, x)
            raw = letters_of(logits[0, 0])
            nl = nonletter_max(logits[0, 0])
            if mod is not None and masked0:
                logits = mod(step, logits)
            eff = letters_of(logits[0, 0])
            pos = base.unmask_one(x, logits, self.mask_id)
            recs.append({"step": step, "masked0": masked0, "raw": raw, "eff": eff, "nonletter_max": nl, "pos": pos,
                         "canvas": [int(v) for v in x[0].tolist()]})
            if stop_after_commit and pos == 0:
                break
        return recs

    def finish(self, x):
        pred = self.tok.decode(x[0], skip_special_tokens=True).strip()
        return pred, delta.leading_letter(pred)

    def fresh(self):
        return torch.full((1, self.gen), self.mask_id, dtype=torch.long, device=self.ctx["device"])

    def shift_logits(self, logits, n):
        logits = logits.float()
        s = torch.tensor(self.shift_table[n]["shift"], dtype=torch.float32, device=logits.device)
        logits[0, 0, LETTER_IDS[:n]] = logits[0, 0, LETTER_IDS[:n]] - s
        return logits


def summarize_run(recs, gen):
    unmask = [-1] * gen
    for r in recs:
        unmask[r["pos"]] = r["step"]
    return unmask


def run_item(dec: Decoder, item, row, n, shortcut=True) -> dict[str, dict]:
    gen = dec.gen
    out = {}
    # ---------------- stock
    x = dec.fresh()
    logits0 = dec.fwd(item, x)                        # keep the step-0 logits for all step-0 variants
    stock_recs = dec.run(item, x, 0, logits0=logits0.clone())
    stock_unmask = summarize_run(stock_recs, gen)
    if sorted(stock_unmask) != list(range(gen)):
        raise RuntimeError(f"stock unmask steps {stock_unmask}")
    stock_text, stock_letter = dec.finish(x)
    stock_final = [int(v) for v in x[0].tolist()]
    commit = stock_unmask[0]
    raw0 = stock_recs[0]["raw"]
    base_rec = {"id": row["id"], "gold": ikb.gold_of(row), "n_opt": n, "backend": dec.ctx["backend"]}
    out["stock"] = {**base_rec, "variant": "stock", "letter_unmask_step": commit, "unmask_step": stock_unmask,
                    "finished_text": stock_text, "finished_letter": stock_letter, "final_ids": stock_final,
                    "step0_restricted_raw": restricted(raw0, n), "step0_restricted_eff": restricted(raw0, n),
                    "pos0_letters": [r["raw"] for r in stock_recs], "pos0_masked": [r["masked0"] for r in stock_recs],
                    "nonletter_max": [r["nonletter_max"] for r in stock_recs]}

    def splice(step0_logits, variant, extra):
        """Generic step-0-modified variant: unmask with modified logits, then continue (shortcut if canvas == stock)."""
        xv = dec.fresh()
        # step 0 done manually so the canvas can be compared with stock before continuing
        eff = letters_of(step0_logits[0, 0])
        pos = base.unmask_one(xv, step0_logits, dec.mask_id)
        rec0 = {"step": 0, "masked0": True, "raw": raw0, "eff": eff, "nonletter_max": stock_recs[0]["nonletter_max"],
                "pos": pos, "canvas": [int(v) for v in xv[0].tolist()]}
        same = rec0["canvas"] == stock_recs[0]["canvas"]
        if same and shortcut:
            recs = [rec0] + [dict(r) for r in stock_recs[1:]]
            text, letter = stock_text, stock_letter
            final_ids = stock_final
            branch = False
        else:
            recs = [rec0] + dec.run(item, xv, 1)
            text, letter = dec.finish(xv)
            final_ids = [int(v) for v in xv[0].tolist()]
            branch = not same
        unmask = summarize_run(recs, gen)
        if sorted(unmask) != list(range(gen)):
            raise RuntimeError(f"{variant} unmask steps {unmask}")
        return {**base_rec, "variant": variant, "letter_unmask_step": unmask[0], "unmask_step": unmask,
                "finished_text": text, "finished_letter": letter, "final_ids": final_ids,
                "step0_restricted_raw": restricted(raw0, n), "step0_restricted_eff": restricted(eff, n),
                "step0_eff_letters": eff, "branch": branch, "same_canvas_after_step0": same,
                "committed_at_step0": pos == 0, "pos0_letters": [r["raw"] for r in recs],
                "pos0_masked": [r["masked0"] for r in recs], **extra}

    # ---------------- perturbations (step 0 only)
    lp = torch.tensor(dec.shift_table[n]["log_pcf"][:n], dtype=torch.float32)
    u_anti = -(lp - lp.mean()); u_anti = u_anti / u_anti.norm()
    order = sorted(range(n), key=lambda j: (-raw0[j], j))
    u_second = torch.zeros(n); u_second[order[1]] = 1 / math.sqrt(2); u_second[order[0]] = -1 / math.sqrt(2)
    for uname, u in (("anti", u_anti), ("second", u_second)):
        for d in DELTAS:
            lg = logits0.clone().float()
            lg[0, 0, LETTER_IDS[:n]] = lg[0, 0, LETTER_IDS[:n]] + d * u.to(lg.device)
            out[f"perturb_{uname}_d{d:g}"] = splice(lg, f"perturb_{uname}_d{d:g}",
                                                     {"direction": uname, "delta": d, "u": u.tolist(),
                                                      "flipped_step0": restricted(letters_of(lg[0, 0]), n) != restricted(raw0, n)})
    # ---------------- calibration placements
    sh = dec.shift_table[n]
    cal_extra = {"alpha": sh["alpha"], "shift": sh["shift"]}
    out["calib_step0"] = splice(dec.shift_logits(logits0.clone(), n), "calib_step0", dict(cal_extra))
    # every step: fresh decode, modifier at every step while masked (byte-identical to calib_gated_decode path)
    xe = dec.fresh()
    recs = dec.run(item, xe, 0, logits0=logits0.clone(), mod=lambda s, lg: dec.shift_logits(lg, n))
    unmask = summarize_run(recs, gen)
    text, letter = dec.finish(xe)
    out["calib_every"] = {**base_rec, "variant": "calib_every", "letter_unmask_step": unmask[0], "unmask_step": unmask,
                          "finished_text": text, "finished_letter": letter, "final_ids": [int(v) for v in xe[0].tolist()],
                          "step0_restricted_raw": restricted(raw0, n), "step0_restricted_eff": restricted(recs[0]["eff"], n),
                          "pos0_letters": [r["raw"] for r in recs], "pos0_letters_eff": [r["eff"] for r in recs],
                          "pos0_masked": [r["masked0"] for r in recs], **cal_extra}
    # commit-only: stock canvas up to the commit step; calibrated argmax at the commit step; re-decode the rest
    xc = dec.fresh()
    if commit > 0:
        xc[0] = torch.tensor(stock_recs[commit - 1]["canvas"], device=xc.device)
    lgc = logits0.clone() if commit == 0 else dec.fwd(item, xc)
    raw_c = letters_of(lgc[0, 0])
    lgc = dec.shift_logits(lgc, n)
    eff_c = letters_of(lgc[0, 0])
    cand_letter = restricted(eff_c, n)
    nlv, nli = stock_recs[commit]["nonletter_max"]
    tok = LETTER_IDS[LETTERS.index(cand_letter)] if eff_c[LETTERS.index(cand_letter)] >= nlv else nli
    xc[0, 0] = tok
    changed = tok != stock_final[0]
    if commit < gen - 1 and changed:
        rest = dec.run(item, xc, commit + 1)
        text, letter = dec.finish(xc)
        final_ids = [int(v) for v in xc[0].tolist()]
    elif commit < gen - 1:
        xc[0] = torch.tensor(stock_final, device=xc.device); text, letter = stock_text, stock_letter; final_ids = stock_final
    else:
        text, letter = dec.finish(xc); final_ids = [int(v) for v in xc[0].tolist()]
    out["calib_commit"] = {**base_rec, "variant": "calib_commit", "letter_unmask_step": commit, "unmask_step": stock_unmask,
                           "finished_text": text, "finished_letter": letter, "final_ids": final_ids,
                           "step0_restricted_raw": restricted(raw0, n),
                           "step0_restricted_eff": restricted(letters_of(dec.shift_logits(logits0.clone(), n)[0, 0]), n),
                           "commit_raw_letters": raw_c, "commit_eff_letters": eff_c,
                           "commit_raw_restricted": restricted(raw_c, n), "commit_eff_restricted": cand_letter,
                           "commit_token_changed": changed, "commit_raw_matches_stock_forward": abs(raw_c[0] - stock_recs[commit]["raw"][0]) < 1e-6,
                           **cal_extra}
    # teacher-forced: step 0 as calib_step0; then forced canvas (stock final tokens at 1..15, index 0 masked)
    rec_tf = out["calib_step0"]
    tf = {"step0_committed": rec_tf["committed_at_step0"]}
    dose_recs = {}
    for k in DOSES + ["full"]:
        xk = dec.fresh()
        for p_, s_ in enumerate(stock_unmask):
            if p_ != 0 and (k == "full" or s_ < k):
                xk[0, p_] = stock_final[p_]
        lgk = dec.fwd(item, xk)
        rawk = letters_of(lgk[0, 0])
        effk = letters_of(dec.shift_logits(lgk.clone(), n)[0, 0])
        dose_recs[str(k)] = {"n_context": int((xk[0] != dec.mask_id).sum().item()), "raw": rawk, "eff": effk,
                             "raw_restricted": restricted(rawk, n), "eff_restricted": restricted(effk, n)}
    full = dose_recs["full"]
    if rec_tf["committed_at_step0"]:
        tf_letter = rec_tf["step0_restricted_eff"]
        tf_raw_letter = restricted(raw0, n)
    else:
        tf_letter = full["eff_restricted"]
        tf_raw_letter = full["raw_restricted"]
    out["calib_tf"] = {**base_rec, "variant": "calib_tf", "letter_unmask_step": 0 if rec_tf["committed_at_step0"] else 1,
                       "finished_letter": tf_letter, "finished_letter_raw_tf": tf_raw_letter,
                       "step0_restricted_raw": restricted(raw0, n), "step0_restricted_eff": rec_tf["step0_restricted_eff"],
                       "dose": dose_recs, "stock_commit_step": commit, **tf, **cal_extra}
    return out


def run_decode(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = (out_dir / f"{args.backend}.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {args.backend} in {out_dir}") from exc
    files = {v: out_dir / f"{args.backend}_{v}.jsonl" for v in variants()}
    for f in files.values():
        ikb.repair_tail(f)
    done_sets = {v: {r["id"] for r in ikb.load_jsonl(f)} for v, f in files.items()}
    done = set.intersection(*done_sets.values()) if done_sets else set()
    rows = ikb.load_manifest(args)
    args.rows = rows
    rows = rows[: args.limit] if args.limit else rows
    ctx = dla.load_backend(args)
    ctx["backend"] = args.backend
    dec = Decoder(ctx, gated_shift(args.backend))
    for n, s in sorted(dec.shift_table.items()):
        print(f"GATE n={n} alpha={s['alpha']:.4f} shift={[round(v, 4) for v in s['shift']]}", flush=True)
    stock = {r["id"]: r for r in ikb.load_jsonl(STOCK_DIR / f"{args.backend}.jsonl")}
    gated = {r["id"]: r for r in ikb.load_jsonl(GATED_DIR / f"{args.backend}.jsonl")}
    handles = {v: f.open("a", encoding="utf-8") for v, f in files.items()}
    t0 = time.time(); n_new = 0; chk = {"stock_match": 0, "stock_cmp": 0, "every_match": 0, "every_cmp": 0, "d0_match": 0}
    print(f"start attractor backend={args.backend} n={len(rows)} done={len(done)} shortcut={not args.no_shortcut}", flush=True)
    for row in rows:
        if row["id"] in done:
            continue
        n = dla.n_choices(row)
        item = ctx["real_item"](row)
        nf0 = dec.n_forward
        res = run_item(dec, item, row, n, shortcut=not args.no_shortcut)
        nf = dec.n_forward - nf0
        s = stock.get(row["id"])
        if s is not None:
            chk["stock_cmp"] += 1; chk["stock_match"] += int(s["finished_letter"] == res["stock"]["finished_letter"] and s["letter_unmask_step"] == res["stock"]["letter_unmask_step"])
        g = gated.get(row["id"])
        if g is not None:
            chk["every_cmp"] += 1; chk["every_match"] += int(g["finished_letter"] == res["calib_every"]["finished_letter"])
        chk["d0_match"] += int(res["perturb_anti_d0"]["finished_letter"] == res["stock"]["finished_letter"] and
                               res["perturb_second_d0"]["finished_letter"] == res["stock"]["finished_letter"])
        for v, rec in res.items():
            rec["n_forwards_item"] = nf
            handles[v].write(json.dumps(rec, ensure_ascii=False) + "\n"); handles[v].flush()
        done.add(row["id"]); n_new += 1
        el = time.time() - t0
        base.write_progress(args.progress, {"backend": args.backend, "n_done": len(done), "n_total": len(rows),
                                            "elapsed_sec": round(el, 1), "items_per_sec": round(n_new / el, 4),
                                            "eta_sec": round((len(rows) - len(done)) * el / n_new), "checks": chk,
                                            "forwards_last_item": nf})
        if len(done) % 10 == 0 or len(done) == len(rows):
            print(f"PROGRESS {len(done)}/{len(rows)} rate={n_new / el:.3f}/s fwd/item={nf} checks={chk}", flush=True)
    for h in handles.values():
        h.close()
    print(f"DONE {args.backend} n={len(done)} checks={chk} elapsed={time.time() - t0:.0f}s "
          f"peak_gpu_mem={torch.cuda.max_memory_allocated(ctx['device']) / 2 ** 30:.2f} GiB", flush=True)


# ----------------------------------------------------------------------------- report

def boot_ci(diff: np.ndarray):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    sums = diff[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {"delta": int(diff.sum()), "ci95": [int(round(lo)), int(round(hi))], "boot_p_le_0": float((sums <= 0).mean())}


def hist(rows, key="finished_letter"):
    return {l: int(sum(r.get(key) == l for r in rows)) for l in LETTERS}


def fit_halflife(t: np.ndarray, y: np.ndarray):
    """Fit y = a exp(-t/tau) on points with y > 0 (least squares in log space). Returns half-life in steps."""
    m = y > 1e-6
    if m.sum() < 2:
        return {"half_life_steps": 0.0 if (len(y) > 1 and np.all(y[1:] <= 1e-6)) else None, "n_points": int(m.sum()),
                "note": "shift vanishes after step 0 (no decay curve to fit)" if m.sum() <= 1 else ""}
    A = np.vstack([np.ones(m.sum()), -t[m]]).T
    coef, *_ = np.linalg.lstsq(A, np.log(y[m]), rcond=None)
    tau = 1.0 / coef[1] if coef[1] > 0 else float("inf")
    return {"half_life_steps": float(tau * math.log(2)) if np.isfinite(tau) else None, "tau": float(tau) if np.isfinite(tau) else None,
            "a": float(math.exp(coef[0])), "n_points": int(m.sum())}


def run_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir
    summary = {"date": "2026-10-13", "dir": str(out_dir), "seed": SEED, "n_boot": N_BOOT, "models": {}}
    for m in ("lavida", "mmada"):
        data = {v: ikb.load_jsonl(out_dir / f"{m}_{v}.jsonl") for v in variants()}
        if not data["stock"]:
            continue
        ids = [r["id"] for r in data["stock"]]
        common = set(ids)
        for v, rows in data.items():
            common &= {r["id"] for r in rows}
        ids = [i for i in ids if i in common]
        by = {v: {r["id"]: r for r in rows} for v, rows in data.items()}
        st = [by["stock"][i] for i in ids]
        gold = {r["id"]: r["gold"] for r in st}
        nopt = {r["id"]: r["n_opt"] for r in st}
        stock_acc = np.array([int(r["finished_letter"] == r["gold"]) for r in st])
        step0_raw_acc = int(sum(r["step0_restricted_raw"] == r["gold"] for r in st))
        ms = {"n": len(ids), "stock": {"acc": int(stock_acc.sum()), "hist": hist(st), "gold_hist": {l: sum(g == l for g in gold.values()) for l in LETTERS},
                                       "step0_restricted_raw_acc": step0_raw_acc,
                                       "commit_step15_frac": float(np.mean([r["letter_unmask_step"] == 15 for r in st]))},
              "checks": {}, "placement": {}, "perturb": {}, "dose": {}}
        # checks vs existing stock / gated files
        ref = {r["id"]: r for r in ikb.load_jsonl(STOCK_DIR / f"{m}.jsonl")}
        ms["checks"]["stock_letter_matches_letter_argmax_traj"] = f"{sum(ref[i]['finished_letter'] == by['stock'][i]['finished_letter'] for i in ids if i in ref)}/{sum(i in ref for i in ids)}"
        refg = {r["id"]: r for r in ikb.load_jsonl(GATED_DIR / f"{m}.jsonl")}
        ms["checks"]["calib_every_letter_matches_calib_gated_decode_20261013"] = f"{sum(refg[i]['finished_letter'] == by['calib_every'][i]['finished_letter'] for i in ids if i in refg)}/{sum(i in refg for i in ids)}"
        ms["checks"]["perturb_d0_equals_stock"] = f"{sum(by['perturb_anti_d0'][i]['finished_letter'] == by['stock'][i]['finished_letter'] and by['perturb_second_d0'][i]['finished_letter'] == by['stock'][i]['finished_letter'] for i in ids)}/{len(ids)}"
        ms["checks"]["calib_commit_raw_forward_matches_stock"] = f"{sum(by['calib_commit'][i]['commit_raw_matches_stock_forward'] for i in ids)}/{len(ids)}"
        # placement
        step0_cal_acc = int(sum(by["calib_every"][i]["step0_restricted_eff"] == gold[i] for i in ids))
        gain0 = step0_cal_acc - step0_raw_acc
        ms["step0"] = {"raw_acc": step0_raw_acc, "cal_acc": step0_cal_acc, "gain": gain0}
        for v in ("calib_step0", "calib_commit", "calib_every", "calib_tf"):
            rows = [by[v][i] for i in ids]
            acc = np.array([int(r["finished_letter"] == r["gold"]) for r in rows])
            ent = {"acc": int(acc.sum()), "hist": hist(rows), "paired_vs_stock": boot_ci(acc - stock_acc),
                   "n_letter_changed_vs_stock": int(sum(r["finished_letter"] != by["stock"][r["id"]]["finished_letter"] for r in rows)),
                   "gain_retention_final_over_step0": (float((acc.sum() - stock_acc.sum()) / gain0) if gain0 else None),
                   "commit_step15_frac": float(np.mean([r["letter_unmask_step"] == 15 for r in rows])) if v != "calib_tf" else None}
            if v == "calib_step0":
                ent["n_committed_at_step0"] = int(sum(r["committed_at_step0"] for r in rows))
                ent["n_branch"] = int(sum(r["branch"] for r in rows))
            if v == "calib_commit":
                ent["n_commit_token_changed"] = int(sum(r["commit_token_changed"] for r in rows))
                ent["commit_raw_restricted_acc"] = int(sum(r["commit_raw_restricted"] == r["gold"] for r in rows))
                ent["commit_eff_restricted_acc"] = int(sum(r["commit_eff_restricted"] == r["gold"] for r in rows))
            if v == "calib_tf":
                raw_tf = np.array([int(r["finished_letter_raw_tf"] == r["gold"]) for r in rows])
                ent["raw_tf_acc"] = int(raw_tf.sum()); ent["raw_tf_hist"] = hist(rows, "finished_letter_raw_tf")
                ent["raw_tf_letter_equals_stock_final"] = int(sum(r["finished_letter_raw_tf"] == by["stock"][r["id"]]["finished_letter"] for r in rows))
                ent["tf_gain_cal_minus_raw"] = int(acc.sum() - raw_tf.sum())
                ent["tf_gain_retention_vs_step0"] = (float((acc.sum() - raw_tf.sum()) / gain0) if gain0 else None)
            ms["placement"][v] = ent
        # context dose
        lp = {int(k): v for k, v in cd.load_priors(m).items()}
        def proj(letters, n):
            u = np.array(lp[n][:n]); u = u - u.mean(); x = np.array(letters[:n]); x = x - x.mean()
            return float(x @ u / np.linalg.norm(u))
        dose = {}
        for k in ["0"] + [str(d) for d in DOSES] + ["full"]:
            raw_l, eff_l, pr = [], [], []
            for i in ids:
                r = by["calib_tf"][i]; n = nopt[i]
                if k == "0":
                    raw_l.append(r["step0_restricted_raw"]); eff_l.append(by["calib_every"][i]["step0_restricted_eff"])
                    pr.append(proj(by["stock"][i]["pos0_letters"][0], n))
                else:
                    d = r["dose"][k]; raw_l.append(d["raw_restricted"]); eff_l.append(d["eff_restricted"]); pr.append(proj(d["raw"], n))
            dose[k] = {"raw_acc": int(sum(a == gold[i] for a, i in zip(raw_l, ids))), "cal_acc": int(sum(a == gold[i] for a, i in zip(eff_l, ids))),
                       "raw_hist": {l: raw_l.count(l) for l in LETTERS}, "cal_hist": {l: eff_l.count(l) for l in LETTERS},
                       "raw_equals_stock_final": int(sum(a == by["stock"][i]["finished_letter"] for a, i in zip(raw_l, ids))),
                       "mean_prior_proj_raw": float(np.mean(pr))}
        ms["dose"] = dose
        # perturbation decay
        pert = {}
        for uname in ("anti", "second"):
            pert[uname] = {}
            for d in DELTAS:
                v = f"perturb_{uname}_d{d:g}"
                rows = [by[v][i] for i in ids]
                flips = [r for r in rows if r["flipped_step0"]]
                surv = sum(r["finished_letter"] == r["step0_restricted_eff"] for r in flips)
                committed0 = sum(r["committed_at_step0"] for r in rows)
                branch = sum(r["branch"] for r in rows)
                # induced shift curve (projection on u) over steps 0..15, normalized by delta
                curve = np.zeros(16); cnt = np.zeros(16); abs_step1 = []
                curve_nc = np.zeros(16); cnt_nc = 0; curve_c = np.zeros(16); cnt_c = 0
                for r in rows:
                    n = r["n_opt"]; u = np.array(r["u"]); s0 = by["stock"][r["id"]]["pos0_letters"]
                    item_curve = np.zeros(16)
                    for t in range(16):
                        if t == 0:
                            sh = d
                        else:
                            dv = np.array(r["pos0_letters"][t][:n]) - np.array(s0[t][:n]); dv = dv - dv.mean()
                            sh = float(dv @ u)
                            if t == 1:
                                abs_step1.append(abs(sh))
                        curve[t] += sh; cnt[t] += 1; item_curve[t] = sh
                    if r["committed_at_step0"]:
                        curve_c += item_curve; cnt_c += 1
                    else:
                        curve_nc += item_curve; cnt_nc += 1
                mean_curve = (curve / np.maximum(cnt, 1)).tolist()
                norm_curve = [(c / d if d > 0 else 0.0) for c in mean_curve]
                norm_nc = [(c / (d * cnt_nc) if d > 0 and cnt_nc else 0.0) for c in curve_nc]
                norm_c = [(c / (d * cnt_c) if d > 0 and cnt_c else 0.0) for c in curve_c]
                acc = np.array([int(r["finished_letter"] == r["gold"]) for r in rows])
                pert[uname][f"{d:g}"] = {
                    "n_flipped_step0": len(flips), "n_committed_at_step0": int(committed0), "n_branch": int(branch),
                    "n_flips_surviving_to_final": int(surv), "surv_frac": (surv / len(flips)) if flips else None,
                    "n_final_letter_changed_vs_stock": int(sum(r["finished_letter"] != by["stock"][r["id"]]["finished_letter"] for r in rows)),
                    "acc": int(acc.sum()), "paired_vs_stock": boot_ci(acc - stock_acc), "hist": hist(rows),
                    "mean_induced_shift_by_step": mean_curve, "normalized_shift_by_step": norm_curve,
                    "normalized_shift_by_step_not_committed_at_step0": norm_nc, "n_not_committed_at_step0": int(cnt_nc),
                    "normalized_shift_by_step_committed_at_step0": norm_c,
                    "max_abs_shift_steps1plus_not_committed": float(np.max(np.abs(curve_nc[1:])) / max(cnt_nc, 1)),
                    "decay_fit": fit_halflife(np.arange(16), np.array(norm_curve)) if d > 0 else None,
                    "mean_abs_shift_step1_over_delta": (float(np.mean(abs_step1)) / d) if d > 0 else None,
                }
        ms["perturb"] = pert
        ms["forwards_per_item_mean"] = float(np.mean([r["n_forwards_item"] for r in st]))
        summary["models"][m] = ms

        # ---------- decay figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for j, uname in enumerate(("anti", "second")):
            ax = axes[j]
            for ci, d in enumerate(DELTAS[1:]):
                e = pert[uname][f"{d:g}"]
                col = plt.cm.viridis(ci / max(1, len(DELTAS) - 2))
                ax.plot(range(16), e["normalized_shift_by_step_not_committed_at_step0"], "o-", ms=3, color=col,
                        label=f"δ={d:g}: letter NOT committed at step 0 (n={e['n_not_committed_at_step0']}); "
                              f"flips@0 {e['n_flipped_step0']}, commit@0 {e['n_committed_at_step0']}, survive {e['n_flips_surviving_to_final']}")
                ax.plot(range(16), e["normalized_shift_by_step"], ":", lw=1, color=col, alpha=0.7)
            ax.plot([], [], "k:", lw=1, label="dotted: all items (plateau = items whose letter was committed at step 0)")
            ax.set_xlabel("decode step"); ax.set_ylabel("induced letter-logit shift along u / δ (mean over items)")
            ax.set_title(f"{m}: step-0-only perturbation, u = {'anti-prior' if uname == 'anti' else 'toward 2nd-best letter'}", fontsize=9)
            ax.legend(fontsize=5.5); ax.set_ylim(-0.1, 1.1)
        ax = axes[2]
        dkeys = ["0"] + [str(d) for d in DOSES] + ["full"]
        ks = [0] + DOSES + [15]
        ax.plot(ks, [dose[k]["mean_prior_proj_raw"] for k in dkeys], "o-", color="tab:red", label="mean prior projection of raw letter logits")
        ax.set_xlabel("# stock context tokens present (answer position masked; 15 = all)"); ax.set_ylabel("prior projection", color="tab:red")
        ax2 = ax.twinx()
        ax2.plot(ks, [dose[k]["raw_acc"] for k in dkeys], "s--", color="tab:blue", label="raw letter acc")
        ax2.plot(ks, [dose[k]["cal_acc"] for k in dkeys], "^--", color="tab:green", label="gated-calibrated letter acc")
        ax2.axhline(ms["stock"]["acc"], color="gray", lw=0.8, ls=":", label="stock final acc")
        ax2.set_ylabel("correct (of %d)" % len(ids))
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right")
        ax.set_title(f"{m}: teacher-forced context dose (stock tokens in stock order)", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / f"decay_{m}.png", dpi=110); plt.close(fig)

    # ---------- placement figure
    models = [m for m in ("lavida", "mmada") if m in summary["models"]]
    if models:
        fig, axes = plt.subplots(2, len(models), figsize=(7 * len(models), 8), squeeze=False)
        for j, m in enumerate(models):
            ms = summary["models"][m]
            ax = axes[0, j]
            names = ["stock", "calib_step0", "calib_commit", "calib_every", "calib_tf"]
            accs = [ms["stock"]["acc"]] + [ms["placement"][v]["acc"] for v in names[1:]]
            err = [[0] + [ms["placement"][v]["acc"] - ms["stock"]["acc"] - ms["placement"][v]["paired_vs_stock"]["ci95"][0] for v in names[1:]],
                   [0] + [ms["placement"][v]["paired_vs_stock"]["ci95"][1] - (ms["placement"][v]["acc"] - ms["stock"]["acc"]) for v in names[1:]]]
            ax.bar(range(len(names)), accs, yerr=err, capsize=3, color=["gray", "tab:orange", "tab:green", "tab:blue", "tab:purple"])
            for i, a in enumerate(accs):
                lab = f"{a}" if i == 0 else f"{a} ({ms['placement'][names[i]]['paired_vs_stock']['delta']:+d})"
                ax.text(i, a, lab, ha="center", va="bottom", fontsize=8)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(["stock", "step 0 only", "commit step only", "every step", "teacher-forced"], fontsize=8)
            ax.set_ylim(min(accs) - 25, max(accs) + 25)
            ax.set_ylabel(f"correct letters (of {ms['n']})")
            ax.set_title(f"{m}: gated calibration placement (paired 95% CI vs stock; step-0 gain {ms['step0']['gain']:+d})", fontsize=9)
            ax = axes[1, j]
            xl = np.arange(5); w = 0.8 / 6
            ax.bar(xl - 0.4 + w / 2, [ms["stock"]["gold_hist"][l] for l in LETTERS], w, color="k", label="gold")
            ax.bar(xl - 0.4 + w * 1.5, [ms["stock"]["hist"][l] for l in LETTERS], w, color="gray", label="stock")
            for k, (v, c) in enumerate(zip(names[1:], ["tab:orange", "tab:green", "tab:blue", "tab:purple"])):
                ax.bar(xl - 0.4 + w * (2.5 + k), [ms["placement"][v]["hist"][l] for l in LETTERS], w, color=c, label=v)
            ax.set_xticks(xl); ax.set_xticklabels(list(LETTERS)); ax.legend(fontsize=7); ax.set_title(f"{m}: final letter histogram", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / "placement.png", dpi=110); plt.close(fig)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for m, ms in summary["models"].items():
        print(m, "n", ms["n"], "stock", ms["stock"]["acc"], "step0", ms["step0"], "checks", ms["checks"])
        for v, e in ms["placement"].items():
            print(f"  {v:13s} acc={e['acc']} {e['paired_vs_stock']} retention={e['gain_retention_final_over_step0']} hist={e['hist']}")
        for u, dd in ms["perturb"].items():
            for d, e in dd.items():
                print(f"  perturb {u} δ={d}: flips={e['n_flipped_step0']} commit0={e['n_committed_at_step0']} survive={e['n_flips_surviving_to_final']} "
                      f"norm_shift[0:4]={[round(x, 3) for x in e['normalized_shift_by_step'][:4]]} fit={e['decay_fit']}")
        print("  dose", {k: (v["raw_acc"], v["cal_acc"], round(v["mean_prior_proj_raw"], 3)) for k, v in ms["dose"].items()})
    print(f"wrote {out_dir / 'summary.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("decode", "report"), required=True)
    p.add_argument("--backend", choices=("lavida", "mmada"), default="lavida")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--no-shortcut", action="store_true")
    p.add_argument("--model-path", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    p.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    args = p.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir != OUT_ROOT and OUT_ROOT not in args.out_dir.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {args.out_dir}")
    if args.progress is None:
        args.progress = args.out_dir / f"progress_{args.backend}.json"
    if args.mode != "report":
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
        if base.GEN != 16:
            raise RuntimeError(f"unexpected gen {base.GEN}")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"decode": run_decode, "report": run_report}[args.mode](args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
