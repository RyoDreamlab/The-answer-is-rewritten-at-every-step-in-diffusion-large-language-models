#!/usr/bin/env python3
"""Analysis for the Dream-VL-7B package (2026-10-13g; CPU, read-only on inputs).

outputs/dreamvl_20261013g/{sciqa,m3cot,mmbench,mmstar}_{stock,gated}.jsonl (+ mmstar_gray.jsonl), pope_{stock,gated}.jsonl
    letter accuracy per mode; paired per-question bootstrap (1000 draws, seed 20261013,
    week1_analysis_20261013.boot_ci) gated-stock (and gray-stock for mmstar); prior effect
    acc(gold = prior letter) - acc(gold != prior letter); P(pred = prior letter); letter histograms; by n_opt;
    step-0 raw/cal correctness; letter unmask step histogram; pope: yes-ratio, precision/recall/F1 (Yes = positive).
Partial files are scored on the ids finished in both compared files and flagged partial.
Writes outputs/dreamvl_20261013g/summary.json + tables.md.
--smoke : sanity report on outputs/dreamvl_20261013g/smoke/ (ALPHA0_EQUALS_STOCK, official-vs-ours agreement,
          timings, peak memory from the logs, ETA for the full queue).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import week1_analysis_20261013 as w  # noqa: E402

OUT = PROJECT / "outputs/dreamvl_20261013g"
SMOKE = OUT / "smoke"
LOGD = PROJECT / "logs/dreamvl_20261013g"
N_BOOT, SEED = w.N_BOOT, w.SEED
MCQ = ("sciqa", "m3cot", "mmbench", "mmstar")
MODES = {"sciqa": ("stock", "gated"), "m3cot": ("stock", "gated"), "mmbench": ("stock", "gated"),
         "mmstar": ("stock", "gated", "gray"), "pope": ("stock", "gated")}


def n_total(ds: str) -> int:
    if ds == "mmbench":
        return len(json.loads((PROJECT / "datasets/mmbench_dev_v1/manifest.json").read_text(encoding="utf-8")))
    return {"sciqa": 2017, "m3cot": 1108, "mmstar": 1500, "pope": 1000}[ds]


def jl(p: Path):
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return rows or None


def pct(k, n):
    return round(100.0 * k / n, 2) if n else None


def ok(r):
    return int(r.get("finished_letter") == r["gold"])


def rate_boot(x: np.ndarray):
    if len(x) == 0:
        return None
    rng = np.random.default_rng(SEED)
    m = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(100 * m, [2.5, 97.5])
    return {"pct": pct(int(x.sum()), len(x)), "ci95_pct": [round(float(lo), 2), round(float(hi), 2)], "n": int(len(x))}


def mean_diff_boot(a: np.ndarray, b: np.ndarray):
    if len(a) == 0 or len(b) == 0:
        return None
    rng = np.random.default_rng(SEED)
    da = a[rng.integers(0, len(a), size=(N_BOOT, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(N_BOOT, len(b)))].mean(axis=1)
    lo, hi = np.percentile(100 * (da - db), [2.5, 97.5])
    return {"pp": round(100 * (a.mean() - b.mean()), 2), "ci95_pp": [round(float(lo), 2), round(float(hi), 2)],
            "n_a": int(len(a)), "n_b": int(len(b))}


def prior_letters(cf_path: Path):
    if not cf_path.exists():
        return None, None
    d = json.loads(cf_path.read_text(encoding="utf-8"))
    return {int(n): v["prior_label"] for n, v in d["by_n_opt"].items()}, d


def single(rows, name, ntot):
    acc = np.array([ok(r) for r in rows])
    return {"name": name, "n_scored": len(rows), "n_total": ntot, "partial": len(rows) != ntot,
            "acc": int(acc.sum()), "acc_pct": pct(int(acc.sum()), len(rows)), "acc_boot": rate_boot(acc),
            "unparsed": int(sum(r.get("finished_letter") is None for r in rows)),
            "pred_hist": dict(sorted(Counter(str(r.get("finished_letter")) for r in rows).items())),
            "gold_hist": dict(sorted(Counter(str(r["gold"]) for r in rows).items())),
            "step0_raw_acc_pct": pct(sum(r.get("step0_restricted_raw") == r["gold"] for r in rows), len(rows)),
            "step0_cal_acc_pct": pct(sum(r.get("step0_restricted_cal") == r["gold"] for r in rows), len(rows)),
            "letter_unmask_step_hist": dict(sorted(Counter(r["letter_unmask_step"] for r in rows).items())),
            "letter_unmask_step_mean": round(float(np.mean([r["letter_unmask_step"] for r in rows])), 3),
            "sec_per_item": round(float(np.mean([r.get("sec", 0) for r in rows])), 3),
            "n_image_tokens_mean": round(float(np.mean([r.get("n_image_tokens", 0) for r in rows])), 1),
            "n_image_tokens_max": int(max(r.get("n_image_tokens", 0) for r in rows)),
            "by_n_opt": {n: pct(sum(ok(r) for r in rows if r["n_opt"] == n), sum(r["n_opt"] == n for r in rows))
                         for n in sorted({r["n_opt"] for r in rows})}}


def paired(method_rows, stock_map, ntot, name):
    rows = [r for r in method_rows if r["id"] in stock_map]
    mL = np.array([ok(r) for r in rows])
    sL = np.array([ok(stock_map[r["id"]]) for r in rows])
    changed = sum(r.get("finished_letter") != stock_map[r["id"]].get("finished_letter") for r in rows)
    out = single(rows, name, ntot)
    out.update({"stock_acc_on_same_ids": int(sL.sum()), "stock_acc_pct_on_same_ids": pct(int(sL.sum()), len(rows)),
                "changed_vs_stock": int(changed), "fixed": int(((mL == 1) & (sL == 0)).sum()),
                "broke": int(((mL == 0) & (sL == 1)).sum()),
                "bootstrap_vs_stock": w.boot_ci(mL - sL, len(rows)) if len(rows) else None})
    return out


def prior_block(rows, prior_letter, name):
    if not prior_letter:
        return None
    a = np.array([ok(r) for r in rows if r["gold"] == prior_letter.get(r["n_opt"])])
    b = np.array([ok(r) for r in rows if r["gold"] != prior_letter.get(r["n_opt"])])
    pp = np.array([int(r.get("finished_letter") == prior_letter.get(r["n_opt"])) for r in rows])
    s0 = np.array([int(r.get("step0_restricted_raw") == prior_letter.get(r["n_opt"])) for r in rows])
    err = [r for r in rows if not ok(r) and r.get("finished_letter") is not None]
    ep = np.array([int(r["finished_letter"] == prior_letter.get(r["n_opt"])) for r in err]) if err else np.array([])
    return {"name": name, "acc_gold_eq_prior_pct": pct(int(a.sum()), len(a)), "acc_gold_ne_prior_pct": pct(int(b.sum()), len(b)),
            "prior_effect": mean_diff_boot(a, b), "p_pred_eq_prior": rate_boot(pp), "p_step0raw_eq_prior": rate_boot(s0),
            "errors_on_prior_letter_pct": pct(int(ep.sum()), len(ep)) if len(ep) else None,
            "p_gold_eq_prior_pct": pct(len(a), len(a) + len(b))}


def pope_stats(rows):
    tp = sum(r["gold"] == "Yes" and r.get("finished_letter") == "Yes" for r in rows)
    fp = sum(r["gold"] == "No" and r.get("finished_letter") == "Yes" for r in rows)
    fn = sum(r["gold"] == "Yes" and r.get("finished_letter") != "Yes" for r in rows)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"yes_ratio_pct": pct(sum(r.get("finished_letter") == "Yes" for r in rows), len(rows)),
            "precision_pct": round(100 * prec, 2) if prec is not None else None,
            "recall_pct": round(100 * rec, 2) if rec is not None else None,
            "f1_pct": round(100 * f1, 2) if f1 is not None else None}


def analyse(ds: str) -> dict:
    ntot = n_total(ds)
    files = {m: jl(OUT / f"{ds}_{m}.jsonl") for m in MODES[ds]}
    res = {"dataset": ds, "n_total": ntot, "files": {m: str(OUT / f"{ds}_{m}.jsonl") for m in MODES[ds]},
           "present": {m: bool(v) for m, v in files.items()}}
    if not files.get("stock"):
        return res
    stock_map = {r["id"]: r for r in files["stock"]}
    cf = OUT / ("cf_pope_dreamvl.json" if ds == "pope" else "cf_dreamvl.json")
    prior_letter, cfd = prior_letters(cf)
    res["prior"] = None if cfd is None else {"file": str(cf), "by_n_opt": cfd["by_n_opt"]}
    res["stock"] = single(files["stock"], "stock", ntot)
    if prior_letter:
        res["stock"]["prior_block"] = prior_block(files["stock"], prior_letter, "stock")
    for m in MODES[ds]:
        if m == "stock" or not files[m]:
            continue
        res[m] = paired(files[m], stock_map, ntot, m)
        if prior_letter:
            res[m]["prior_block"] = prior_block([r for r in files[m] if r["id"] in stock_map], prior_letter, m)
    if ds == "pope":
        for m in MODES[ds]:
            if files[m]:
                res[m]["pope"] = pope_stats(files[m])
    return res


def md_block(res: dict) -> str:
    ds = res["dataset"]
    lines = [f"## {ds} (Dream-VL-7B, gen 16 / 16 steps / low-confidence, n_total {res['n_total']})", ""]
    if "stock" not in res:
        lines.append("_no stock file yet_\n")
        return "\n".join(lines)
    lines += ["| mode | n | acc % | Δ vs stock (pp) | CI95 pp | changed / fixed / broke | unparsed | P(pred=prior) % | step0 raw acc % | mean letter step | sec/item |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODES[ds]:
        x = res.get(m)
        if not x:
            continue
        b = x.get("bootstrap_vs_stock")
        pb = x.get("prior_block") or {}
        pp = pb.get("p_pred_eq_prior") or {}
        lines.append(f"| {m}{' (partial)' if x['partial'] else ''} | {x['n_scored']} | {x['acc_pct']} | "
                     f"{b['delta_pp'] if b else '—'} | {b['ci95_pp'] if b else '—'} | "
                     f"{x.get('changed_vs_stock', '—')} / {x.get('fixed', '—')} / {x.get('broke', '—')} | {x['unparsed']} | "
                     f"{pp.get('pct', '—')} | {x['step0_raw_acc_pct']} | {x['letter_unmask_step_mean']} | {x['sec_per_item']} |")
    if res.get("prior"):
        lines.append("")
        lines.append("prior (content-free): " + "; ".join(f"n={n}: {v['prior_label']} p_cf={[round(p, 3) for p in v['p_cf']]} alpha={round(v['alpha_spread'], 3)}"
                                                          for n, v in res["prior"]["by_n_opt"].items()))
    st = res["stock"]
    lines.append("")
    lines.append(f"stock pred hist {st['pred_hist']} gold hist {st['gold_hist']}; by n_opt acc % {st['by_n_opt']}; "
                 f"image tokens mean {st['n_image_tokens_mean']} max {st['n_image_tokens_max']}")
    if ds == "pope":
        lines.append("pope: " + "; ".join(f"{m}: {res[m]['pope']}" for m in MODES[ds] if res.get(m)))
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ smoke

def smoke(smoke_dir: Path) -> None:
    tag = smoke_dir.name
    print(f"=== SMOKE REPORT backend=dreamvl (sciqa, 5 items) dir={smoke_dir} ===")
    files = {m: jl(smoke_dir / f"smoke_sciqa_{m}_n5.jsonl") for m in ("stock", "gated_a0", "gated")}
    official = jl(smoke_dir / "smoke_sciqa_stock_n5_official.jsonl")
    if not files["stock"]:
        print("no smoke stock file")
        return
    st = {r["id"]: r for r in files["stock"]}
    timing = {m: round(float(np.mean([r["sec"] for r in rows])), 3) for m, rows in files.items() if rows}
    a0 = files["gated_a0"]
    if a0:
        same = True
        for r in a0:
            s = st.get(r["id"])
            if s is None:
                same = False
                continue
            for k in ("finished_letter", "finished_text", "letter_unmask_step", "pos0_argmax", "step0_restricted_raw", "step0_restricted_cal"):
                if r[k] != s[k]:
                    same = False
                    print(f"  MISMATCH {r['id']} {k}: a0={r[k]!r} stock={s[k]!r}")
            if not all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6) for x, y in zip(r["letter_logits_raw_at_commit"], s["letter_logits_raw_at_commit"])):
                same = False
                print(f"  MISMATCH {r['id']} letter_logits_raw_at_commit")
        print(f"ALPHA0_EQUALS_STOCK={same} (n={len(a0)})")
    off = {r["id"]: r for r in official} if official else {}
    print("per item: id | gold | n | stock | gated | step(stock) | step0 raw->cal (gated) | img_tok | stock text | official lmms 16/16 | official readme 128/128 t0.1")
    agree = {"lmms_mcq_16_16_t0": 0, "readme_128_128_t0.1_seed0": 0}
    for i, s in st.items():
        g = next((r for r in (files["gated"] or []) if r["id"] == i), None)
        o = off.get(i, {})
        ol = {k: (o.get(k) or {}).get("letter") for k in agree}
        for k in agree:
            agree[k] += int(ol[k] == s["finished_letter"])
        print(f"  {i} | {s['gold']} | {s['n_opt']} | {s['finished_letter']} | {g['finished_letter'] if g else '—'} | {s['letter_unmask_step']} | "
              f"{g['step0_restricted_raw'] + '->' + g['step0_restricted_cal'] if g else '—'} | {s.get('n_image_tokens')} | {s['finished_text']!r} | "
              f"{ol['lmms_mcq_16_16_t0']} {(o.get('lmms_mcq_16_16_t0') or {}).get('text')!r} | "
              f"{ol['readme_128_128_t0.1_seed0']} {(o.get('readme_128_128_t0.1_seed0') or {}).get('text')!r}")
    for m, rows in files.items():
        if rows:
            print(f"{m}: acc {sum(ok(r) for r in rows)}/{len(rows)} unparsed {sum(r['finished_letter'] is None for r in rows)} "
                  f"pos0_argmax[0] {[r['pos0_argmax'][0] for r in rows]} letters {[r['finished_letter'] for r in rows]} "
                  f"steps {[r['letter_unmask_step'] for r in rows]} sec/item {timing[m]}")
    r0 = files["stock"][0]
    print(f"stock canvas of {r0['id']}: raw text {r0.get('finished_text_raw')!r} tokens {r0.get('canvas_tokens')}")
    print(f"image tokens per item {[r.get('n_image_tokens') for r in files['stock']]} prefix_len {[r.get('prefix_len') for r in files['stock']]}")
    if official:
        n = len(st)
        print(f"OFFICIAL_VS_OURS agreement: " + ", ".join(f"{k}: {v}/{n}" for k, v in agree.items()))
        for k in agree:
            accs = sum((o.get(k) or {}).get("letter") == st[i]["gold"] for i, o in off.items())
            secs = round(float(np.mean([(o.get(k) or {}).get("sec", 0) for o in off.values()])), 2)
            print(f"  official {k}: acc {accs}/{n} sec/item {secs}")
    peaks = []
    for log in sorted(LOGD.glob(f"{tag}_*.log")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"PEAK_GPU_MEM allocated_gib=([\d.]+) reserved_gib=([\d.]+)", line)
            if m:
                peaks.append((log.name, float(m.group(1)), float(m.group(2))))
    print(f"peak GPU memory (torch allocated / reserved GiB) per smoke job: {peaks}")
    print(f"timing sec/item: {timing}")
    if "stock" in timing:
        per = timing["stock"]
        sizes = {ds: n_total(ds) for ds in MODES}
        dec = sum(len(MODES[ds]) * sizes[ds] for ds in MODES)
        print(f"decodes in the full queue: {dec} (" + ", ".join(f"{ds}x{len(MODES[ds])}={sizes[ds]}" for ds in MODES) + ")")
        print(f"ETA full queue at {per} s/decode (sciqa smoke; MMBench/MMStar images may differ): {round(dec * per / 3600, 2)} GPU-h "
              f"+ ~{len(MODES) * 2 + 1} model loads x ~1-2 min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-dir", type=Path, default=SMOKE)
    args = ap.parse_args()
    if args.smoke:
        smoke(args.smoke_dir)
        return
    res = {ds: analyse(ds) for ds in MODES}
    boot = {"n_boot": N_BOOT, "seed": SEED, "type": "paired per-question percentile (vs stock); independent-group bootstrap for prior effect"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"experiment": "dreamvl_20261013g", "backend": "dreamvl", "bootstrap": boot,
                                                  "results": res}, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Dream-VL-7B stock / gated / gray decodes (2026-10-13g)\n",
          f"paired per-question bootstrap {N_BOOT} draws seed {SEED}; Δ in pp of n_scored.\n"] + [md_block(res[ds]) for ds in MODES]
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {OUT / 'summary.json'} {OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
