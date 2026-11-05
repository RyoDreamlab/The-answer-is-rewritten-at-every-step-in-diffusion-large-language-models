#!/usr/bin/env python3
"""Score the prior-work calibration baselines (outputs/calib_baselines_20261013b) vs stock AND vs our gated rule.

Reuses week1_analysis_20261013.py (compare(): letter / official accuracy, paired per-question bootstrap, 1000
draws, seed 20261013, per-n_opt breakdown, confusion, TV). Read-only on all inputs; writes summary.json and
tables.md into outputs/calib_baselines_20261013b/ (or --out). Partial decode files are scored on the finished
subset and flagged (partial).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import week1_analysis_20261013 as w  # noqa: E402

B = PROJECT / "outputs" / "calib_baselines_20261013b"
MODELS = ("lavida", "mmada")
MODEL_LABEL = {"lavida": "LaViDa", "mmada": "MMaDA"}

# (row key, display name, constants column)
METHODS = [
    ("stock", "stock", "—"),
    ("full", "full α=1 (Zhao 2021 content-free)", "α=1"),
    ("atabuzzaman", "Atabuzzaman α_adp (EMNLP 2025)", "α=1, τ=2, ctx 10%"),
    ("kumar", "Kumar g=1−TVD (ACL 2022)", "无 (g 逐题)"),
    ("gated", "ours gated α_n=1−p_min/p_max", "无"),
]


def method_path(key: str, dataset: str, model: str) -> Path | None:
    if dataset == "sqa":
        return {
            "full": PROJECT / "outputs/calib_decode_20261012" / f"{model}.jsonl",
            "atabuzzaman": B / f"{model}_sqa_atabuzzaman.jsonl",
            "kumar": B / f"{model}_sqa_kumar.jsonl",
            "gated": PROJECT / "outputs/calib_gated_decode_20261013" / f"{model}.jsonl",
        }.get(key)
    return {
        "full": PROJECT / "outputs/m3cot_20261013" / f"{model}_calib_full.jsonl",
        "atabuzzaman": B / f"{model}_m3cot_atabuzzaman.jsonl",
        "kumar": B / f"{model}_m3cot_kumar.jsonl",
        "gated": PROJECT / "outputs/m3cot_20261013" / f"{model}_calib_spread.jsonl",
    }.get(key)


def stats(vals):
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4), "std": round(float(a.std()), 4)}


def scalar_stats(key: str, rows: list[dict], gold: dict) -> dict | None:
    if key == "atabuzzaman":
        out = {"alpha_adp_step0": stats([r["alpha_adp_step0"] for r in rows]),
               "alpha_adp_at_commit": stats([r["alpha_adp_at_commit"] for r in rows]),
               "conf_step0": stats([r["conf_steps"][0] for r in rows]),
               "alpha_adp_step0_by_n_opt": {}}
        for n in (2, 3, 4, 5):
            sub = [r["alpha_adp_step0"] for r in rows if gold[r["id"]]["n_opt"] == n]
            if sub:
                out["alpha_adp_step0_by_n_opt"][str(n)] = stats(sub)
        out["max_abs_shift_step0"] = stats([max(abs(r["alpha_adp_step0"] * b) for b in r["B"]) for r in rows])
        return out
    if key == "kumar":
        out = {"g": stats([r["g"] for r in rows]), "tvd_fullvocab": stats([r["tvd_fullvocab"] for r in rows]),
               "g_letters_only_(not_used)": stats([r["g_letters_only"] for r in rows]), "g_by_n_opt": {}}
        for n in (2, 3, 4, 5):
            sub = [r["g"] for r in rows if gold[r["id"]]["n_opt"] == n]
            if sub:
                out["g_by_n_opt"][str(n)] = stats(sub)
        return out
    return None


def fmt_delta(c):
    if c is None:
        return "—"
    b = c["bootstrap_letter"]
    p = " (partial)" if c["partial"] else ""
    return f"{b['delta']:+d} ({b['delta_pp']:+.2f}pp) CI[{b['ci95_count'][0]},{b['ci95_count'][1]}]{p}"


def fmt_acc(c):
    if c is None:
        return "—"
    return f"{c['letter']}/{c['n_scored']} ({c['letter_pct']}%)"


def by_n_table(block: list[tuple[str, dict]]) -> list[str]:
    lines = ["| 方法 | n_opt=2 | n_opt=3 | n_opt=4 | n_opt=5 |", "|---|---|---|---|---|"]
    for name, c in block:
        cells = []
        for n in ("2", "3", "4", "5"):
            q = c["by_n_opt"].get(n)
            cells.append(f"{q['method_letter']}/{q['n']} ({q['delta']:+d})" if q else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=B)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"date": "2026-10-13b", "bootstrap": {"n_boot": w.N_BOOT, "seed": w.SEED,
                                                      "type": "paired per-question percentile, 1000 draws"},
               "constants": {"atabuzzaman": {"ALPHA": 1.0, "TAU": 2.0, "ctx_frac": 0.10,
                                             "ctx_selection": "first ceil(10%) items in manifest order"},
                             "kumar": {"g": "1 - TVD(full-vocab p_real, full-vocab p_cf) at step 0"}},
               "scienceqa": {}, "m3cot": {}, "cross_table": {}}
    md = ["# Prior-work calibration baselines vs stock vs ours (auto-generated, 2026-10-13b)\n",
          "字母准确率 = finished first letter == gold；Δ 为配对差（题数），CI 为 1000 次配对 bootstrap 95% 百分位区间（seed 20261013）。\n"]
    cross = {k: {} for k, _, _ in METHODS}
    g_sqa = w.sqa_gold()
    for dataset in ("sqa", "m3cot"):
        for model in MODELS:
            if dataset == "sqa":
                st = w.sqa_stock(model, g_sqa)
                stock_rows = [st[i] | {"id": i} for i in sorted(st)]
                gold, n_total, title = g_sqa, 2017, f"ScienceQA-IMG test — {MODEL_LABEL[model]}"
            else:
                stock_rows, st = w.m3cot_stock(model)
                gold, n_total, title = w.m3cot_gold(stock_rows), 1108, f"M3CoT val — {MODEL_LABEL[model]}"
            gated_rows = w.jl(method_path("gated", dataset, model))
            gated_map = {r["id"]: r for r in gated_rows} if gated_rows else None
            block_vs_stock, block_vs_gated, scal = [], [], {}
            for key, name, _ in METHODS:
                if key == "stock":
                    rows = stock_rows
                elif key == "gated":
                    rows = gated_rows
                else:
                    rows = w.jl(method_path(key, dataset, model))
                if not rows:
                    cross[key][f"{dataset}_{model}"] = None
                    continue
                c = w.compare(name, rows, st, gold, n_total)
                block_vs_stock.append(c)
                cross[key][f"{dataset}_{model}"] = {"letter": c["letter"], "n_scored": c["n_scored"],
                                                     "letter_pct": c["letter_pct"], "partial": c["partial"],
                                                     "delta_vs_stock": c["bootstrap_letter"]}
                if gated_map and key != "gated":
                    ids_ok = [r for r in rows if r["id"] in gated_map]
                    cg = w.compare(name + " vs gated", ids_ok, gated_map, gold, n_total)
                    block_vs_gated.append(cg)
                    cross[key][f"{dataset}_{model}"]["delta_vs_gated"] = cg["bootstrap_letter"]
                s = scalar_stats(key, rows, gold)
                if s:
                    scal[key] = s
            summary["scienceqa" if dataset == "sqa" else "m3cot"][model] = {
                "vs_stock": block_vs_stock, "vs_gated": block_vs_gated, "scalar_stats": scal}
            md.append(w.md_compare_table(block_vs_stock, title + " — vs stock"))
            if block_vs_gated:
                md.append(w.md_compare_table(block_vs_gated, title + " — vs ours (gated)").replace("vs stock", "vs gated"))
            md.append(f"#### {title} — per n_opt (method correct / n, Δ vs stock)\n")
            md.extend(by_n_table([(c["name"], c) for c in block_vs_stock]))
            md.append("")
            if scal:
                md.append(f"#### {title} — per-item scalars actually applied\n")
                for key, s in scal.items():
                    if key == "atabuzzaman":
                        md.append(f"- Atabuzzaman α_adp (step 0): mean {s['alpha_adp_step0']['mean']}, median {s['alpha_adp_step0']['median']}, "
                                  f"range [{s['alpha_adp_step0']['min']}, {s['alpha_adp_step0']['max']}]; at commit: mean {s['alpha_adp_at_commit']['mean']}, "
                                  f"median {s['alpha_adp_at_commit']['median']}; conf step0 mean {s['conf_step0']['mean']}; "
                                  f"max |α_adp·B| shift step0 mean {s['max_abs_shift_step0']['mean']} (logit units); "
                                  f"by n_opt mean α: " + ", ".join(f"n={n}: {v['mean']}" for n, v in s['alpha_adp_step0_by_n_opt'].items()))
                    elif key == "kumar":
                        md.append(f"- Kumar g = 1−TVD(full-vocab): mean {s['g']['mean']}, median {s['g']['median']}, range [{s['g']['min']}, {s['g']['max']}]; "
                                  f"TVD mean {s['tvd_fullvocab']['mean']}; by n_opt mean g: " + ", ".join(f"n={n}: {v['mean']}" for n, v in s['g_by_n_opt'].items()) +
                                  f"; (letters-only TVD would give g mean {s['g_letters_only_(not_used)']['mean']}, not used)")
                md.append("")
    summary["cross_table"] = cross
    md.append("## 汇总表（Δ字母 vs stock，题数；CI = 95% 配对 bootstrap）\n")
    md.append("| 方法 | 常数 | SQA LaViDa | SQA MMaDA | M3CoT LaViDa | M3CoT MMaDA | 双模型为正? (SQA / M3CoT) |")
    md.append("|---|---|---|---|---|---|---|")
    both_pos = {}
    for key, name, consts in METHODS:
        cells, pos = [], {}
        for col in ("sqa_lavida", "sqa_mmada", "m3cot_lavida", "m3cot_mmada"):
            c = cross[key].get(col)
            if c is None:
                cells.append("—")
            else:
                d = c["delta_vs_stock"]
                cells.append(f"{c['letter']} ({c['letter_pct']}%) {d['delta']:+d} [{d['ci95_count'][0]},{d['ci95_count'][1]}]"
                             + (" partial" if c["partial"] else ""))
                pos[col] = d["delta"] > 0
        verdict = []
        for ds in ("sqa", "m3cot"):
            ks = [k for k in pos if k.startswith(ds)]
            if key == "stock" or len(ks) < 2:
                verdict.append("—")
            else:
                verdict.append("是" if all(pos[k] for k in ks) else "否")
        both_pos[key] = {"sqa": verdict[0], "m3cot": verdict[1]}
        md.append(f"| {name} | {consts} | " + " | ".join(cells) + f" | {' / '.join(verdict)} |")
    summary["both_model_positive"] = both_pos
    md.append("")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
