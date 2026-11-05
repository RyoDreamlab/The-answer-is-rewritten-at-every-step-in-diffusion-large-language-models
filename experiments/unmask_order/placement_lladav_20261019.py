#!/usr/bin/env python3
"""Calibration placement on LLaDA-V (third model), ScienceQA-IMG test, first --limit items of the standard manifest.

Replicates attractor_20261013d's placement variants (MECHANISM_UPGRADES_20261013d §2.3, done there for LaViDa and MMaDA)
on LLaDA-V with LLaDA-V's own gated content-free prior (outputs/lladav_20261013c/cf_lladav.json, rule
alpha_n = 1 - p_min/p_max, the prior used by outputs/lladav_20261013c/sciqa_gated.jsonl):

  stock         plain gen-16 / 16-step / T=0 low-confidence-remasking decode (run_unmask_order.unmask_one)
  calib_step0   letters at the answer slot shifted by -alpha_n*log p_cf at step 0 ONLY; later steps untouched.  The
                decode is memoryless except through the canvas: if step 0 commits the same (position, token) as stock
                the stock trajectory is reused (branch=false).
  calib_commit  shift applied only at the stock commit step of the answer slot: canvas before that step = stock canvas,
                committed token = calibrated full-vocab argmax (letter if its shifted logit >= best non-letter, else the
                non-letter), remaining steps re-decoded if the token changed.
  calib_every   shift at every step while the answer slot is masked (= the gated path of calib_decode_multi_20261013c).

Checks: stock vs outputs/lladav_20261013c/sciqa_stock.jsonl (letter + commit step), calib_every vs
outputs/lladav_20261013c/sciqa_gated.jsonl (letter).  Writes only under outputs/placement_lladav_20261019/.
Backend / prompt / prior loading come from calib_decode_multi_20261013c (imported, not modified).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
sys.path.insert(0, str(PROJECT / "experiments"))
import calib_decode_multi_20261013c as c  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "placement_lladav_20261019").resolve()
PRIOR = PROJECT / "outputs" / "lladav_20261013c" / "cf_lladav.json"
STOCK_REF = PROJECT / "outputs" / "lladav_20261013c" / "sciqa_stock.jsonl"
GATED_REF = PROJECT / "outputs" / "lladav_20261013c" / "sciqa_gated.jsonl"
MODEL_PATH = PROJECT / "checkpoints" / "LLaDA-V"
LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]
GEN = base.GEN
VARIANTS = ("stock", "calib_step0", "calib_commit", "calib_every")
SEED, N_BOOT = 20261013, 1000


def check_out(path: Path) -> None:
    r = path.resolve()
    extra = (PROJECT / "outputs" / "placement_lladav_full_20261020").resolve()
    if r != OUT_ROOT and OUT_ROOT not in r.parents and r != extra and extra not in r.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {path}")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def restricted(vals, n) -> str:
    return LETTERS[max(range(n), key=lambda j: (float(vals[j]), -j))]


def letters_of(row) -> list[float]:
    return [float(v) for v in row[LETTER_IDS]]


def nonletter_max(row) -> tuple[float, int]:
    r = row.detach().float().clone()
    r[LETTER_IDS] = float("-inf")
    i = int(torch.argmax(r).item())
    return float(r[i]), i


def lenient_letter(text: str) -> str | None:
    m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", str(text or ""))
    return m.group(1) if m else None


class Decoder:
    def __init__(self, be: c.Lladav, log_pcf: dict, alpha: dict):
        self.be = be
        self.mask_id = be.mask_id
        self.tok = be.tokenizer
        self.shift = {n: {"alpha": alpha[n], "log_pcf": log_pcf[n],
                          "shift": [alpha[n] * v for v in log_pcf[n]]} for n in log_pcf}
        self.n_forward = 0

    def item(self, row):
        pe, plen = self.be.lb.prefix(self.be.ctx, row["text"], row["image"]())
        return {"forward": self.be.lb.make_forward(self.be.ctx, pe), "prefix_len": plen, "device": pe.device}

    def fwd(self, item, x):
        self.n_forward += 1
        return item["forward"](x)

    def fresh(self, item):
        return torch.full((1, GEN), self.mask_id, dtype=torch.long, device=item["device"])

    def shift_logits(self, logits, n):
        logits = logits.float()
        s = torch.tensor(self.shift[n]["shift"], dtype=torch.float32, device=logits.device)
        logits[0, 0, LETTER_IDS[:n]] = logits[0, 0, LETTER_IDS[:n]] - s
        return logits

    def run(self, item, x, start, logits0=None, mod=None):
        recs = []
        for step in range(start, GEN):
            masked0 = int(x[0, 0].item()) == self.mask_id
            logits = logits0 if (step == start and logits0 is not None) else self.fwd(item, x)
            raw = letters_of(logits[0, 0]) if masked0 else None
            nl = nonletter_max(logits[0, 0]) if masked0 else None
            if mod is not None and masked0:
                logits = mod(logits)
            eff = letters_of(logits[0, 0]) if masked0 else None
            pos = base.unmask_one(x, logits, self.mask_id)
            recs.append({"step": step, "masked0": masked0, "raw": raw, "eff": eff, "nonletter_max": nl, "pos": pos,
                         "canvas": [int(v) for v in x[0].tolist()]})
        return recs

    def finish(self, x):
        text = self.tok.decode(x[0], skip_special_tokens=True).strip()
        return text, delta.leading_letter(text)


def unmask_of(recs) -> list[int]:
    u = [-1] * GEN
    for r in recs:
        u[r["pos"]] = r["step"]
    if sorted(u) != list(range(GEN)):
        raise RuntimeError(f"unmask steps {u}")
    return u


def run_item(dec: Decoder, row) -> dict[str, dict]:
    n = int(row["n_opt"])
    item = dec.item(row)
    nf0 = dec.n_forward
    meta = {"id": row["id"], "gold": row["gold"], "n_opt": n, "backend": "lladav", "dataset": "sciqa",
            "alpha": dec.shift[n]["alpha"], "shift": dec.shift[n]["shift"], "prefix_len": item["prefix_len"]}

    def pack(variant, recs, text, letter, final_ids, **extra):
        u = unmask_of(recs)
        return {**meta, "variant": variant, "letter_unmask_step": u[0], "unmask_step": u, "finished_text": text,
                "finished_letter": letter, "lenient_letter": lenient_letter(text), "final_ids": final_ids,
                "pos0_letters_raw": [r["raw"] for r in recs if r["raw"] is not None],
                "pos0_letters_eff": [r["eff"] for r in recs if r["eff"] is not None], **extra}

    # stock
    x = dec.fresh(item)
    logits0 = dec.fwd(item, x)
    srecs = dec.run(item, x, 0, logits0=logits0.clone())
    stext, sletter = dec.finish(x)
    sfinal = [int(v) for v in x[0].tolist()]
    sunmask = unmask_of(srecs)
    commit = sunmask[0]
    raw0 = srecs[0]["raw"]
    out = {"stock": pack("stock", srecs, stext, sletter, sfinal, step0_restricted_raw=restricted(raw0, n),
                         step0_restricted_eff=restricted(raw0, n))}

    # step 0 only
    xv = dec.fresh(item)
    lg = dec.shift_logits(logits0.clone(), n)
    eff0 = letters_of(lg[0, 0])
    pos = base.unmask_one(xv, lg, dec.mask_id)
    rec0 = {"step": 0, "masked0": True, "raw": raw0, "eff": eff0, "nonletter_max": srecs[0]["nonletter_max"], "pos": pos,
            "canvas": [int(v) for v in xv[0].tolist()]}
    same = rec0["canvas"] == srecs[0]["canvas"]
    if same:
        recs, text, letter, final = [rec0] + [dict(r) for r in srecs[1:]], stext, sletter, sfinal
    else:
        recs = [rec0] + dec.run(item, xv, 1)
        text, letter = dec.finish(xv)
        final = [int(v) for v in xv[0].tolist()]
    out["calib_step0"] = pack("calib_step0", recs, text, letter, final, step0_restricted_raw=restricted(raw0, n),
                              step0_restricted_eff=restricted(eff0, n), branch=not same, same_canvas_after_step0=same,
                              committed_at_step0=pos == 0)

    # every step
    xe = dec.fresh(item)
    erecs = dec.run(item, xe, 0, logits0=logits0.clone(), mod=lambda L: dec.shift_logits(L, n))
    etext, eletter = dec.finish(xe)
    out["calib_every"] = pack("calib_every", erecs, etext, eletter, [int(v) for v in xe[0].tolist()],
                              step0_restricted_raw=restricted(raw0, n), step0_restricted_eff=restricted(erecs[0]["eff"], n))

    # commit step only
    xc = dec.fresh(item)
    if commit > 0:
        xc[0] = torch.tensor(srecs[commit - 1]["canvas"], device=xc.device)
    lgc = logits0.clone() if commit == 0 else dec.fwd(item, xc)
    raw_c = letters_of(lgc[0, 0])
    lgc = dec.shift_logits(lgc, n)
    eff_c = letters_of(lgc[0, 0])
    cand = restricted(eff_c, n)
    nlv, nli = srecs[commit]["nonletter_max"]
    tok = LETTER_IDS[LETTERS.index(cand)] if eff_c[LETTERS.index(cand)] >= nlv else nli
    xc[0, 0] = tok
    changed = tok != sfinal[0]
    crecs = [dict(r) for r in srecs[:commit]] + [{"step": commit, "masked0": True, "raw": raw_c, "eff": eff_c,
                                                   "nonletter_max": (nlv, nli), "pos": 0, "canvas": [int(v) for v in xc[0].tolist()]}]
    if commit < GEN - 1 and changed:
        crecs += dec.run(item, xc, commit + 1)
        text, letter = dec.finish(xc)
        final = [int(v) for v in xc[0].tolist()]
    elif commit < GEN - 1:
        crecs += [dict(r) for r in srecs[commit + 1:]]
        text, letter, final = stext, sletter, sfinal
    else:
        text, letter = dec.finish(xc)
        final = [int(v) for v in xc[0].tolist()]
    out["calib_commit"] = pack("calib_commit", crecs, text, letter, final, step0_restricted_raw=restricted(raw0, n),
                               step0_restricted_eff=restricted(eff0, n), commit_raw_letters=raw_c, commit_eff_letters=eff_c,
                               commit_raw_restricted=restricted(raw_c, n), commit_eff_restricted=cand, commit_token_changed=changed,
                               commit_raw_matches_stock_forward=abs(raw_c[0] - srecs[commit]["raw"][0]) < 1e-6)
    nf = dec.n_forward - nf0
    for v in out.values():
        v["n_forwards_item"] = nf
    return out


def run_decode(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = (out_dir / "lladav.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("already running") from exc
    files = {v: out_dir / f"lladav_{v}.jsonl" for v in VARIANTS}
    for f in files.values():
        check_out(f)
    done_sets = [{r["id"] for r in jsonl(f)} for f in files.values()]
    done = set.intersection(*done_sets)
    rows = c.rows_sciqa("lladav", 0)[: args.limit]
    log_pcf, alpha, pcf = c.load_priors(PRIOR)
    for n in sorted(log_pcf):
        print(f"GATE n={n} alpha={alpha[n]:.4f} p_cf={[round(v, 4) for v in pcf[n]]} shift={[round(alpha[n] * v, 4) for v in log_pcf[n]]}", flush=True)
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    be = c.Lladav(argparse.Namespace(model_path=str(MODEL_PATH), cuda=args.cuda))
    if be.letter_ids() != LETTER_IDS:
        raise RuntimeError(f"letter ids {be.letter_ids()}")
    dec = Decoder(be, log_pcf, alpha)
    stock_ref = {r["id"]: r for r in jsonl(STOCK_REF)}
    gated_ref = {r["id"]: r for r in jsonl(GATED_REF)}
    handles = {v: f.open("a", encoding="utf-8") for v, f in files.items()}
    chk = {"stock_cmp": 0, "stock_match": 0, "every_cmp": 0, "every_match": 0}
    t0 = time.time(); n_new = 0
    print(f"start placement lladav n={len(rows)} done={len(done)}", flush=True)
    for row in rows:
        if row["id"] in done:
            continue
        res = run_item(dec, row)
        s = stock_ref.get(row["id"])
        if s is not None:
            chk["stock_cmp"] += 1
            chk["stock_match"] += int(s["finished_letter"] == res["stock"]["finished_letter"] and s["letter_unmask_step"] == res["stock"]["letter_unmask_step"])
        g = gated_ref.get(row["id"])
        if g is not None:
            chk["every_cmp"] += 1
            chk["every_match"] += int(g["finished_letter"] == res["calib_every"]["finished_letter"])
        for v, rec in res.items():
            handles[v].write(json.dumps(rec, ensure_ascii=False) + "\n"); handles[v].flush()
        done.add(row["id"]); n_new += 1
        el = time.time() - t0
        if len(done) % 10 == 0 or len(done) == len(rows):
            print(f"PROGRESS {len(done)}/{len(rows)} rate={n_new / el:.3f}/s fwd/item={res['stock']['n_forwards_item']} checks={chk} "
                  f"last: stock={res['stock']['finished_letter']} step0={res['calib_step0']['finished_letter']} "
                  f"commit={res['calib_commit']['finished_letter']} every={res['calib_every']['finished_letter']} c={res['stock']['letter_unmask_step']}",
                  flush=True)
    for h in handles.values():
        h.close()
    print(f"DONE n={len(done)} checks={chk} elapsed={time.time() - t0:.0f}s", flush=True)
    report(out_dir)


def boot_ci(diff: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    sums = diff[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {"delta": int(diff.sum()), "ci95": [int(round(lo)), int(round(hi))]}


def report(out_dir: Path) -> None:
    data = {v: {r["id"]: r for r in jsonl(out_dir / f"lladav_{v}.jsonl")} for v in VARIANTS}
    ids = sorted(set.intersection(*[set(d) for d in data.values()]), key=lambda i: list(data["stock"]).index(i))
    gold = {i: data["stock"][i]["gold"] for i in ids}
    stock = data["stock"]
    lines = [f"# calibration placement, LLaDA-V, ScienceQA-IMG first {len(ids)} items (standard manifest)", "",
             f"prior: {PRIOR.name} (alpha_n = 1 - p_min/p_max); stock commit-step histogram: "
             f"{dict(sorted(__import__('collections').Counter(stock[i]['letter_unmask_step'] for i in ids).items()))}",
             f"stock first token is a letter: {sum(stock[i]['finished_letter'] is not None for i in ids)}/{len(ids)}", "",
             "| variant | strict correct | Δ strict [CI95] | lenient correct | Δ lenient [CI95] | letter changed vs stock | A/B/C/D/E |",
             "|---|---|---|---|---|---|---|"]
    summary = {"n": len(ids)}
    s_strict = np.array([int(stock[i]["finished_letter"] == gold[i]) for i in ids])
    s_len = np.array([int(stock[i]["lenient_letter"] == gold[i]) for i in ids])
    for v in VARIANTS:
        d = data[v]
        strict = np.array([int(d[i]["finished_letter"] == gold[i]) for i in ids])
        len_ = np.array([int(d[i]["lenient_letter"] == gold[i]) for i in ids])
        ch = int(sum(d[i]["finished_letter"] != stock[i]["finished_letter"] for i in ids))
        hist = "/".join(str(sum(d[i]["finished_letter"] == L for i in ids)) for L in LETTERS)
        bs, bl = boot_ci(strict - s_strict), boot_ci(len_ - s_len)
        summary[v] = {"strict": int(strict.sum()), "lenient": int(len_.sum()), "d_strict": bs, "d_lenient": bl, "changed": ch}
        lines.append(f"| {v} | {strict.sum()} | {bs['delta']:+d} {bs['ci95']} | {len_.sum()} | {bl['delta']:+d} {bl['ci95']} | {ch} | {hist} |")
    same_ce = sum(data["calib_commit"][i]["finished_letter"] == data["calib_every"][i]["finished_letter"] for i in ids)
    same_ce_ids = sum(data["calib_commit"][i]["final_ids"] == data["calib_every"][i]["final_ids"] for i in ids)
    st0 = data["calib_step0"]
    lines += ["", f"commit-only == every (letter): {same_ce}/{len(ids)}; identical final canvases: {same_ce_ids}/{len(ids)}",
              f"step0-only: branched after step 0 on {sum(st0[i]['branch'] for i in ids)} items, letter committed at step 0 on "
              f"{sum(st0[i]['committed_at_step0'] for i in ids)} items, final letter changed on {summary['calib_step0']['changed']} items",
              f"step-0 restricted letter: raw correct {sum(stock[i]['step0_restricted_raw'] == gold[i] for i in ids)}, "
              f"calibrated correct {sum(st0[i]['step0_restricted_eff'] == gold[i] for i in ids)}"]
    # by commit-step bucket: does step0-only change anything even when the letter commits at step 1-3?
    lines += ["", "| stock commit step | n | step0-only changed | commit-only changed | every changed |", "|---|---|---|---|---|"]
    buckets = [("0", lambda s: s == 0), ("1-3", lambda s: 1 <= s <= 3), ("4-14", lambda s: 4 <= s <= 14), ("15", lambda s: s == 15)]
    for name, f in buckets:
        sel = [i for i in ids if f(stock[i]["letter_unmask_step"])]
        if not sel:
            continue
        row = [str(sum(data[v][i]["finished_letter"] != stock[i]["finished_letter"] for i in sel)) for v in ("calib_step0", "calib_commit", "calib_every")]
        lines.append(f"| {name} | {len(sel)} | " + " | ".join(row) + " |")
    summary["commit_eq_every_letter"] = same_ce
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("decode", "report"), required=True)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--limit", type=int, default=400)
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
