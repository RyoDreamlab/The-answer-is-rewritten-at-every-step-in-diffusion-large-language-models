#!/usr/bin/env python3
"""Week-1 analysis (read-only on inputs). Scores B1/B2 decodes vs stock, analyses B3 lens and B4 dose files.

Usage: week1_analysis_20261013.py [--out DIR]   (defaults to outputs/week1_analysis_20261013)
Partial decode files are scored on the finished subset (paired with stock on the same ids) and flagged partial.
Bootstrap: paired per-question resampling, 1000 draws, seed 20261013, percentile 95% CI of the summed difference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "caft"))
from graders import scienceqa_mc  # noqa: E402

LET = "ABCDE"
MODELS = ("lavida", "mmada")
N_BOOT = 1000
SEED = 20261013
OUT = PROJECT / "outputs" / "week1_analysis_20261013"
EOS_TEXTS = ("<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>")


def jl(p: Path):
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def pct(k, n):
    return round(100.0 * k / n, 2) if n else None


# ------------------------------------------------------------------ gold tables

def sqa_gold():
    rows = json.loads((PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json").read_text(encoding="utf-8"))
    import ast
    out = {}
    for r in rows:
        c = r["choices"]
        if isinstance(c, str):
            c = ast.literal_eval(c)
        out[r["id"]] = {"gold": str(r["gold"]).strip(), "gold_text": str(r["gold_text"]), "n_opt": len(c),
                        "choices": [str(x) for x in c]}
    return out


def m3cot_gold(stock_rows):
    return {r["id"]: {"gold": r["gold"], "gold_text": r["gold_text"], "n_opt": r["n_opt"]} for r in stock_rows}


# ------------------------------------------------------------------ stock predictions

def sqa_stock(model, gold):
    tr = jl(PROJECT / "outputs/letter_argmax_traj_20261010" / f"{model}.jsonl")
    if model == "lavida":
        txt = {r["id"]: r["pred"] for r in jl(PROJECT / "outputs/full_eval_table1_stock_a_scienceqa_20260922/eval_arm_A_scienceqa.jsonl")}
    else:
        txt = {r["id"]: r["prediction_text"] for r in jl(PROJECT / "outputs/sciqa_standard_baseline_mmada.jsonl")}
    return {r["id"]: {"finished_letter": r["finished_letter"], "finished_text": str(txt[r["id"]]),
                      "letter_unmask_step": r["letter_unmask_step"], "step0_restricted_raw": r["pos0_argmax"][0]}
            for r in tr}


def m3cot_stock(model):
    rows = jl(PROJECT / "outputs/m3cot_20261012" / f"{model}_stock.jsonl")
    return rows, {r["id"]: r for r in rows}


# ------------------------------------------------------------------ scoring

def letter_ok(r, g):
    return int(r.get("finished_letter") == g["gold"])


def official_ok(r, g):
    return int(bool(scienceqa_mc(str(r.get("finished_text", "")), {"gold": g["gold"], "gold_text": g["gold_text"]})))


def boot_ci(diff: np.ndarray, n_total: int):
    rng = np.random.default_rng(SEED)
    n = len(diff)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    sums = diff[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {"delta": int(diff.sum()), "ci95_count": [int(round(lo)), int(round(hi))],
            "ci95_pp": [round(100 * lo / n_total, 2), round(100 * hi / n_total, 2)],
            "delta_pp": round(100 * diff.sum() / n_total, 2), "boot_p_delta_le_0": float((sums <= 0).mean()),
            "n_boot": N_BOOT, "seed": SEED}


def confusion(rows, gold):
    m = {g: {p: 0 for p in LET + "N"} for g in LET}
    for r in rows:
        p = r.get("finished_letter") or "N"
        if p not in LET:
            p = "N"
        m[gold[r["id"]]["gold"]][p] += 1
    return m


def hist(rows, key="finished_letter"):
    c = Counter(str(r.get(key)) for r in rows)
    return {k: c[k] for k in sorted(c)}


def tv_by_n(rows, gold):
    """Total-variation distance between predicted-letter and gold-letter histograms, per option count and pooled."""
    out = {}
    pooled_num = 0
    for n in (2, 3, 4, 5):
        sub = [r for r in rows if gold[r["id"]]["n_opt"] == n]
        if not sub:
            continue
        pg = Counter(gold[r["id"]]["gold"] for r in sub)
        pp = Counter(r.get("finished_letter") for r in sub)
        tv = 0.5 * sum(abs(pg.get(l, 0) - pp.get(l, 0)) for l in LET + "None") / len(sub)
        out[str(n)] = {"n": len(sub), "tv": round(tv, 4), "gold_hist": {l: pg.get(l, 0) for l in LET[:n]},
                       "pred_hist": {l: pp.get(l, 0) for l in LET[:n]}}
        pooled_num += tv * len(sub)
    out["pooled_weighted_tv"] = round(pooled_num / len(rows), 4) if rows else None
    return out


def compare(name, method_rows, stock_map, gold, n_total):
    ids = [r["id"] for r in method_rows]
    partial = len(ids) != n_total
    st = [stock_map[i] | {"id": i} for i in ids]
    mL = np.array([letter_ok(r, gold[r["id"]]) for r in method_rows])
    sL = np.array([letter_ok(r, gold[r["id"]]) for r in st])
    mO = np.array([official_ok(r, gold[r["id"]]) for r in method_rows])
    sO = np.array([official_ok(r, gold[r["id"]]) for r in st])
    changed = fixed = broke = 0
    for a, b in zip(method_rows, st):
        if a.get("finished_letter") != b.get("finished_letter"):
            changed += 1
            fixed += int(letter_ok(a, gold[a["id"]]) and not letter_ok(b, gold[b["id"]]))
            broke += int(letter_ok(b, gold[b["id"]]) and not letter_ok(a, gold[a["id"]]))
    by_n = {}
    for n in (2, 3, 4, 5):
        sel = [k for k, i in enumerate(ids) if gold[i]["n_opt"] == n]
        if sel:
            by_n[str(n)] = {"n": len(sel), "stock_letter": int(sL[sel].sum()), "method_letter": int(mL[sel].sum()),
                            "delta": int(mL[sel].sum() - sL[sel].sum()),
                            "stock_pct": pct(int(sL[sel].sum()), len(sel)), "method_pct": pct(int(mL[sel].sum()), len(sel))}
    res = {
        "name": name, "n_scored": len(ids), "n_total": n_total, "partial": partial,
        "letter": int(mL.sum()), "letter_pct": pct(int(mL.sum()), len(ids)),
        "official": int(mO.sum()), "official_pct": pct(int(mO.sum()), len(ids)),
        "stock_letter_on_same_ids": int(sL.sum()), "stock_official_on_same_ids": int(sO.sum()),
        "changed_vs_stock": changed, "fixed": fixed, "broke": broke,
        "bootstrap_letter": boot_ci(mL - sL, len(ids)), "bootstrap_official": boot_ci(mO - sO, len(ids)),
        "by_n_opt": by_n,
        "confusion_method_gold_x_pred": confusion(method_rows, gold),
        "confusion_stock_gold_x_pred": confusion(st, gold),
        "pred_hist_method": hist(method_rows), "pred_hist_stock": hist(st),
        "gold_hist": dict(sorted(Counter(gold[i]["gold"] for i in ids).items())),
        "tv_method": tv_by_n(method_rows, gold), "tv_stock": tv_by_n(st, gold),
        "letter_unmask_step_method": dict(sorted(Counter(r["letter_unmask_step"] for r in method_rows).items())),
        "letter_unmask_step_stock": dict(sorted(Counter(r["letter_unmask_step"] for r in st).items())),
        "no_leading_letter_method": sum(r.get("finished_letter") is None for r in method_rows),
    }
    if "step0_restricted_cal" in method_rows[0]:
        s0r = sum(r["step0_restricted_raw"] == gold[r["id"]]["gold"] for r in method_rows)
        s0c = sum(r["step0_restricted_cal"] == gold[r["id"]]["gold"] for r in method_rows)
        res["step0_restricted_raw_correct"] = s0r
        res["step0_restricted_cal_correct"] = s0c
        res["step0_gain"] = s0c - s0r
        res["final_gain_vs_stock"] = int(mL.sum() - sL.sum())
        res["gain_retention_final_over_step0"] = round((mL.sum() - sL.sum()) / (s0c - s0r), 3) if s0c != s0r else None
        res["finished_equals_step0_cal"] = sum(r["finished_letter"] == r["step0_restricted_cal"] for r in method_rows)
    return res


# ------------------------------------------------------------------ B3 lens

def softmax(v):
    v = np.asarray(v, dtype=np.float64)
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def kl(p, q):
    p = np.asarray(p); q = np.asarray(q)
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))).sum())


def centered_cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else None


def prior_logp(model):
    recs = jl(PROJECT / "outputs/calib_vote_20261012" / f"cf_{model}.jsonl")
    out = {}
    for r in recs:
        if r["cond"] == "a_global":
            n = r["n_opt"]
            lg = np.asarray(r["letter_logit64"][:n], dtype=np.float64)
            out[n] = lg - (np.log(np.exp(lg - lg.max()).sum()) + lg.max())
    return out


def is_eos_text(t):
    return any(e in str(t) for e in EOS_TEXTS)


def analyze_lens(model, rows):
    logp = prior_logp(model)
    L = rows[0]["n_layers"]
    blank = {r["n_opt"]: r for r in rows if r["cond"] == "a_global"}
    real = [r for r in rows if r["cond"] == "real"]
    res = {"n_layers": L, "n_real": len(real), "n_blank": len(blank),
           "final_readout_max_abs_diff_vs_model_max": max(r["final_readout_max_abs_diff_vs_model"] for r in rows)}
    # blank per layer
    bl = {}
    for n, r in sorted(blank.items()):
        per = []
        for l in range(L):
            p = softmax(r["letter_logit64"][l][:n])
            per.append({"layer": l + 1, "kl_to_uniform": round(kl(p, np.full(n, 1 / n)), 4),
                        "max_prob": round(float(p.max()), 4), "argmax": LET[int(p.argmax())],
                        "cos_with_final_prior": centered_cos(r["letter_logit64"][l][:n], logp[n]),
                        "vocab_argmax": r["vocab_argmax_text"][l], "letter_mass": round(r["letter_mass"][l], 4)})
        bl[str(n)] = {"per_layer": per, "final_p": [round(float(x), 4) for x in softmax(r["letter_logit64"][-1][:n])],
                      "prior_p_calib_vote": [round(float(np.exp(x)), 4) for x in logp[n]]}
    res["blank"] = bl
    # real per layer
    per = []
    gold_hist = Counter(r["gold"] for r in real)
    for l in range(L):
        agree_final = agree_gold = 0
        kls, cats, lmass, top1 = [], Counter(), [], []
        pred = Counter()
        cos_by_n = defaultdict(list)
        for r in real:
            n = r["n_opt"]
            v = np.asarray(r["letter_logit64"][l][:n])
            vf = np.asarray(r["letter_logit64"][-1][:n])
            a = int(v.argmax()); af = int(vf.argmax())
            agree_final += int(a == af)
            agree_gold += int(LET[a] == r["gold"])
            kls.append(kl(softmax(vf), softmax(v)))
            pred[LET[a]] += 1
            t = r["vocab_argmax_text"][l]
            aid = r["vocab_argmax_id"][l]
            if 32 <= aid <= 36:
                cats["letter"] += 1
            elif is_eos_text(t):
                cats["eos"] += 1
            else:
                cats["other"] += 1
            lmass.append(r["letter_mass"][l]); top1.append(r["vocab_top1_prob"][l])
            cos_by_n[n].append(v)
        tv = 0.5 * sum(abs(gold_hist.get(x, 0) - pred.get(x, 0)) for x in LET) / len(real)
        cosn = {}
        for n, vs in cos_by_n.items():
            mean_v = np.mean(np.stack(vs), axis=0)
            cosn[str(n)] = centered_cos(mean_v, logp[n])
        per.append({"layer": l + 1, "agree_final_pct": pct(agree_final, len(real)), "agree_gold_pct": pct(agree_gold, len(real)),
                    "mean_kl_final_to_layer": round(float(np.mean(kls)), 4), "pred_hist": dict(pred),
                    "tv_pred_vs_gold": round(tv, 4), "mean_cos_meanlogit_vs_prior_by_n": cosn,
                    "vocab_argmax_frac": {k: pct(v, len(real)) for k, v in cats.items()},
                    "mean_letter_mass": round(float(np.mean(lmass)), 4), "mean_top1_prob": round(float(np.mean(top1)), 4)})
    res["real_per_layer"] = per
    res["real_gold_hist"] = dict(gold_hist)
    res["real_final_restricted_correct"] = int(sum(LET[int(np.argmax(r["letter_logit64"][-1][:r["n_opt"]]))] == r["gold"] for r in real))
    return res


def plot_lens(model, res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = res["n_layers"]
    x = list(range(1, L + 1))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for n, b in res["blank"].items():
        ax[0, 0].plot(x, [q["kl_to_uniform"] for q in b["per_layer"]], marker=".", label=f"blank n={n}")
    ax[0, 0].set_title(f"{model}: content-free input, KL(letter dist || uniform) per layer"); ax[0, 0].legend(); ax[0, 0].set_xlabel("layer")
    rp = res["real_per_layer"]
    ax[0, 1].plot(x, [q["agree_final_pct"] for q in rp], marker=".", label="argmax agrees with layer-32 answer")
    ax[0, 1].plot(x, [q["agree_gold_pct"] for q in rp], marker=".", label="argmax == gold")
    ax[0, 1].set_title("real input: per-layer restricted argmax agreement (%)"); ax[0, 1].legend(); ax[0, 1].set_xlabel("layer")
    ax[1, 0].plot(x, [q["tv_pred_vs_gold"] for q in rp], marker=".", label="TV(pred hist, gold hist)")
    for n in ("2", "3", "4"):
        ax[1, 0].plot(x, [q["mean_cos_meanlogit_vs_prior_by_n"].get(n) for q in rp], marker=".", label=f"cos(mean logit, log p_cf) n={n}")
    ax[1, 0].set_title("real input: letter-bias strength per layer"); ax[1, 0].legend(fontsize=8); ax[1, 0].set_xlabel("layer")
    for k in ("letter", "eos", "other"):
        ax[1, 1].plot(x, [q["vocab_argmax_frac"].get(k, 0) for q in rp], marker=".", label=f"vocab argmax = {k}")
    ax[1, 1].plot(x, [100 * q["mean_letter_mass"] for q in rp], marker=".", label="mean letter prob mass x100")
    ax[1, 1].set_title("real input: full-vocab argmax category (%)"); ax[1, 1].legend(); ax[1, 1].set_xlabel("layer")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ B4 dose

def analyze_dose(model, rows):
    logp = prior_logp(model)
    fams = defaultdict(lambda: defaultdict(list))
    for r in rows:
        n = r["n_opt"]
        full = r["conds"]["full"]
        lf = np.asarray(full["letter_logit64"][:n]); pf = softmax(lf)
        for name, c in r["conds"].items():
            if name == "full":
                continue
            lc = np.asarray(c["letter_logit64"][:n]); pc = softmax(lc)
            d = lc - lf
            fams[c["family"]][c["k"]].append({
                "kl_full_to_cond": kl(pf, pc), "tv": 0.5 * float(np.abs(pf - pc).sum()),
                "argmax_changed": int(pc.argmax() != pf.argmax()),
                "eos": int(is_eos_text(c["argmax_text"])), "letter_argmax_vocab": int(32 <= c["argmax_id"] <= 36),
                "cos_delta_vs_prior": centered_cos(d, logp[n]) if n >= 2 else None,
                "bias_shift_kl_uniform": kl(pc, np.full(n, 1 / n)) - kl(pf, np.full(n, 1 / n)),
                "mean_abs_delta_centered": float(np.abs(d - d.mean()).mean()),
            })
    res = {"n_items": len(rows), "full_eos_rate": pct(sum(is_eos_text(r["conds"]["full"]["argmax_text"]) for r in rows), len(rows)),
           "full_restricted_correct": int(sum(r["full_letter_restricted"] == r["gold"] for r in rows)), "families": {}}
    for fam, byk in fams.items():
        res["families"][fam] = {}
        for k, lst in sorted(byk.items()):
            agg = {}
            for key in lst[0]:
                vals = [v[key] for v in lst if v[key] is not None]
                agg[key] = round(float(np.mean(vals)), 4) if vals else None
            agg["n"] = len(lst)
            agg["eos_rate_pct"] = pct(sum(v["eos"] for v in lst), len(lst))
            agg["argmax_changed_pct"] = pct(sum(v["argmax_changed"] for v in lst), len(lst))
            res["families"][fam][str(k)] = agg
    return res


def plot_dose(model, res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for fam, byk in res["families"].items():
        ks = sorted(int(k) for k in byk)
        ax[0].plot(ks, [byk[str(k)]["eos_rate_pct"] for k in ks], marker="o", label=fam)
        ax[1].plot(ks, [byk[str(k)]["kl_full_to_cond"] for k in ks], marker="o", label=fam)
        ax[2].plot(ks, [byk[str(k)]["argmax_changed_pct"] for k in ks], marker="o", label=fam)
        ax[3].plot(ks, [byk[str(k)]["cos_delta_vs_prior"] for k in ks], marker="o", label=fam)
    ax[0].set_title(f"{model}: vocab argmax = EOS (%)"); ax[1].set_title("mean KL(p_full || p_blocked) over A-E")
    ax[2].set_title("restricted letter argmax changed (%)"); ax[3].set_title("cos(Δ letter logits, log p_cf)")
    for a in ax:
        a.set_xlabel("blocked keys k"); a.set_xscale("log", base=2); a.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ main

def md_compare_table(block, title):
    lines = [f"### {title}", "", "| 方法 | n | 字母 | 官方 | Δ字母 vs stock | 95% CI (题) | 95% CI (pp) | 改/对/错 | step0 raw→cal→final |", "|---|---|---|---|---|---|---|---|---|"]
    for r in block:
        b = r["bootstrap_letter"]
        s0 = f"{r.get('step0_restricted_raw_correct','—')}→{r.get('step0_restricted_cal_correct','—')}→{r['letter']}" if "step0_gain" in r else "—"
        lines.append(f"| {r['name']}{' (partial)' if r['partial'] else ''} | {r['n_scored']} | {r['letter']} ({r['letter_pct']}%) | {r['official']} ({r['official_pct']}%) | "
                     f"{b['delta']:+d} ({b['delta_pp']:+.2f}pp) | [{b['ci95_count'][0]}, {b['ci95_count'][1]}] | [{b['ci95_pp'][0]}, {b['ci95_pp'][1]}] | "
                     f"{r['changed_vs_stock']}/{r['fixed']}/{r['broke']} | {s0} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"date": "2026-10-13", "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "type": "paired per-question percentile"},
               "scienceqa": {}, "m3cot": {}, "lens": {}, "dose": {}}
    md = ["# Week-1 analysis tables (auto-generated)\n"]
    g = sqa_gold()
    for m in MODELS:
        st = sqa_stock(m, g)
        stock_rows = [st[i] | {"id": i} for i in sorted(st)]
        block = []
        base = compare("stock", stock_rows, st, g, 2017)
        block.append(base)
        for name, p in (("full calib α=1 (calib_decode_20261012)", PROJECT / "outputs/calib_decode_20261012" / f"{m}.jsonl"),
                        ("gated α=1-pmin/pmax (calib_gated_decode_20261013)", PROJECT / "outputs/calib_gated_decode_20261013" / f"{m}.jsonl")):
            rows = jl(p)
            if rows:
                block.append(compare(name, rows, st, g, 2017))
        summary["scienceqa"][m] = block
        md.append(md_compare_table(block, f"ScienceQA-IMG test — {m}"))
    for m in MODELS:
        srows, smap = m3cot_stock(m)
        gm = m3cot_gold(srows)
        block = [compare("stock", srows, smap, gm, 1108)]
        for name, p in (("gated α=1-pmin/pmax (m3cot_20261013 spread)", PROJECT / "outputs/m3cot_20261013" / f"{m}_calib_spread.jsonl"),
                        ("full calib α=1 (m3cot_20261013 full)", PROJECT / "outputs/m3cot_20261013" / f"{m}_calib_full.jsonl")):
            rows = jl(p)
            if rows:
                block.append(compare(name, rows, smap, gm, 1108))
        summary["m3cot"][m] = block
        md.append(md_compare_table(block, f"M3CoT val — {m}"))
    for m in MODELS:
        rows = jl(PROJECT / "outputs/layer_logit_lens_20261013" / f"{m}.jsonl")
        if rows and any(r["cond"] == "real" for r in rows):
            res = analyze_lens(m, rows)
            summary["lens"][m] = res
            plot_lens(m, res, args.out / f"lens_{m}.png")
            md.append(f"### Logit lens — {m} (n_real={res['n_real']})\n\n| layer | blank KL→U (n=2/3/4/5) | real agree final % | real agree gold % | TV(pred,gold) | argmax letter/eos/other % |\n|---|---|---|---|---|---|")
            for l in range(res["n_layers"]):
                bk = "/".join(str(res["blank"][n]["per_layer"][l]["kl_to_uniform"]) for n in sorted(res["blank"]))
                q = res["real_per_layer"][l]
                va = q["vocab_argmax_frac"]
                md.append(f"| {l+1} | {bk} | {q['agree_final_pct']} | {q['agree_gold_pct']} | {q['tv_pred_vs_gold']} | {va.get('letter',0)}/{va.get('eos',0)}/{va.get('other',0)} |")
            md.append("")
        rows = jl(PROJECT / "outputs/slot_dose_20261013" / f"{m}.jsonl")
        if rows:
            res = analyze_dose(m, rows)
            summary["dose"][m] = res
            plot_dose(m, res, args.out / f"dose_{m}.png")
            md.append(f"### Slot dose — {m} (n={res['n_items']}, full EOS rate {res['full_eos_rate']}%)\n\n| family | k | EOS % | argmax changed % | KL(full‖cond) | TV | cos(Δ, log p_cf) | ΔKL→U |\n|---|---|---|---|---|---|---|---|")
            for fam in ("M", "Trand", "Tlast"):
                for k, a in res["families"].get(fam, {}).items():
                    md.append(f"| {fam} | {k} | {a['eos_rate_pct']} | {a['argmax_changed_pct']} | {a['kl_full_to_cond']} | {a['tv']} | {a['cos_delta_vs_prior']} | {a['bias_shift_kl_uniform']} |")
            md.append("")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
