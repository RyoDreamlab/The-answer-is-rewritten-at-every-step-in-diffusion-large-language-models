#!/usr/bin/env python3
"""Score the 2026-10-13c decodes (read-only on inputs): MMBench-dev (lavida/mmada[/lladav]) and LLaDA-V (sciqa/m3cot).

For every (dataset, model): stock vs gated letter accuracy, paired per-question bootstrap (1000 draws, seed 20261013,
week1_analysis_20261013.compare), letter histograms (gold / stock / gated / p_cf) per n_opt, by n_opt, by category and
L2 category (MMBench). Partial files are scored on the ids finished in BOTH stock and gated and flagged partial.
Writes summary.json + tables.md into outputs/mmbench_20261013c and outputs/lladav_20261013c, each with a cross table
(ScienceQA / M3CoT / MMBench x LaViDa / MMaDA / LLaDA-V) that pulls the earlier LaViDa/MMaDA numbers from
outputs/week1_analysis_20261013/summary.json.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import week1_analysis_20261013 as w  # noqa: E402

LET = "ABCDE"
MMB = PROJECT / "outputs/mmbench_20261013c"
LLV = PROJECT / "outputs/lladav_20261013c"
WEEK1 = PROJECT / "outputs/week1_analysis_20261013/summary.json"
N_TOTAL = {"sciqa": 2017, "m3cot": 1108, "mmbench": None}
LABEL = {"lavida": "LaViDa", "mmada": "MMaDA", "lladav": "LLaDA-V"}
DLABEL = {"sciqa": "ScienceQA-IMG test (2017)", "m3cot": "M3CoT val (1108)", "mmbench": "MMBench-dev EN"}


def jl(p: Path):
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return rows or None


def pcf_of(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return {int(n): v for n, v in d["by_n_opt"].items()}


def gold_mmbench():
    rows = json.loads((PROJECT / "datasets/mmbench_dev_v1/manifest.json").read_text(encoding="utf-8"))
    return {r["id"]: {"gold": r["gold"], "gold_text": r["gold_text"], "n_opt": r["n_opt"], "category": r["category"],
                      "l2_category": r["l2_category"]} for r in rows}, len(rows)


def gold_sciqa_standard():
    rows = json.loads((PROJECT / "datasets/sciqa_img_standard_v1/manifest.json").read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        c = r["choices"]
        if isinstance(c, str):
            c = ast.literal_eval(c)
        g = str(r["gold_option"]).strip()
        out[r["id"]] = {"gold": g, "gold_text": str(c[LET.index(g)]), "n_opt": len(c), "category": r.get("subject")}
    return out, 2017


def gold_from_rows(rows, meta_key="category"):
    return {r["id"]: {"gold": r["gold"], "gold_text": r["gold_text"], "n_opt": r["n_opt"],
                      "category": r.get(meta_key)} for r in rows}


def hist_block(stock, gated, gold, pcf):
    out = {}
    for n in sorted({gold[r["id"]]["n_opt"] for r in stock}):
        ids = [r["id"] for r in stock if gold[r["id"]]["n_opt"] == n]
        gs = Counter(gold[i]["gold"] for i in ids)
        ss = Counter(r.get("finished_letter") or "None" for r in stock if gold[r["id"]]["n_opt"] == n)
        gg = Counter(r.get("finished_letter") or "None" for r in gated if gold[r["id"]]["n_opt"] == n)
        k = len(ids)
        out[str(n)] = {"n": k,
                       "gold_pct": {l: w.pct(gs.get(l, 0), k) for l in LET[:n]},
                       "stock_pct": {l: w.pct(ss.get(l, 0), k) for l in LET[:n]}, "stock_none": ss.get("None", 0),
                       "gated_pct": {l: w.pct(gg.get(l, 0), k) for l in LET[:n]}, "gated_none": gg.get("None", 0),
                       "p_cf_pct": ({l: round(100 * v, 2) for l, v in zip(LET[:n], pcf[n]["p_cf"])} if pcf and n in pcf else None),
                       "alpha_spread": (round(pcf[n]["alpha_spread"], 4) if pcf and n in pcf else None)}
    return out


def by_key(stock, gated, gold, key):
    import numpy as np

    smap = {r["id"]: r for r in stock}
    groups = {}
    for r in gated:
        groups.setdefault(gold[r["id"]].get(key) or "?", []).append(r)
    out = {}
    for k, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        mL = np.array([w.letter_ok(r, gold[r["id"]]) for r in rows])
        sL = np.array([w.letter_ok(smap[r["id"]], gold[r["id"]]) for r in rows])
        out[k] = {"n": len(rows), "stock": int(sL.sum()), "gated": int(mL.sum()), "delta": int(mL.sum() - sL.sum()),
                  "stock_pct": w.pct(int(sL.sum()), len(rows)), "gated_pct": w.pct(int(mL.sum()), len(rows)),
                  "bootstrap": w.boot_ci(mL - sL, len(rows)) if len(rows) >= 2 else None}
    return out


def score(dataset, model, stock, gated, gold, n_total, pcf, out_dir):
    common = [r["id"] for r in stock if r["id"] in {g["id"] for g in gated}]
    cs = set(common)
    st = [r for r in stock if r["id"] in cs]
    gt = {r["id"]: r for r in gated}
    gt = [gt[i] for i in common]
    smap = {r["id"]: r for r in st}
    res = {"dataset": dataset, "model": model, "n_total": n_total, "n_stock_done": len(stock), "n_gated_done": len(gated),
           "n_scored": len(common), "partial": len(common) != n_total,
           "stock": w.compare("stock", st, smap, gold, n_total), "gated": w.compare("gated", gt, smap, gold, n_total),
           "hist_by_n_opt": hist_block(st, gt, gold, pcf), "by_category": by_key(st, gt, gold, "category")}
    if dataset == "mmbench":
        res["by_l2_category"] = by_key(st, gt, gold, "l2_category")
    res["timing"] = {"stock_sec_per_item": round(sum(r.get("sec", 0) for r in stock) / len(stock), 3),
                     "gated_sec_per_item": round(sum(r.get("sec", 0) for r in gated) / len(gated), 3)}
    res["files"] = {"stock": str(out_dir / f"{model}_stock.jsonl"), "gated": str(out_dir / f"{model}_gated.jsonl")}
    return res


def cell(res):
    if res is None:
        return "—"
    s, g = res["stock"], res["gated"]
    b = g["bootstrap_letter"]
    tag = f" (partial {res['n_scored']}/{res['n_total']})" if res["partial"] else ""
    return (f"stock {s['letter_pct']}% ({s['letter']}/{res['n_scored']}) → gated {g['letter_pct']}% ({g['letter']}); "
            f"Δ {b['delta']:+d} ({b['delta_pp']:+.2f}pp) CI95 [{b['ci95_count'][0]}, {b['ci95_count'][1]}] "
            f"pp [{b['ci95_pp'][0]}, {b['ci95_pp'][1]}]{tag}")


def week1_cell(dataset, model):
    if not WEEK1.exists():
        return "—"
    d = json.loads(WEEK1.read_text(encoding="utf-8"))
    key = "scienceqa" if dataset == "sciqa" else "m3cot"
    block = d.get(key, {}).get(model)
    if not block:
        return "—"
    st = block[0]
    gated = next((r for r in block if r["name"].startswith("gated")), None)
    if gated is None:
        return f"stock {st['letter_pct']}% ({st['letter']}/{st['n_scored']})"
    b = gated["bootstrap_letter"]
    return (f"stock {st['letter_pct']}% ({st['letter']}/{st['n_scored']}) → gated {gated['letter_pct']}% ({gated['letter']}); "
            f"Δ {b['delta']:+d} ({b['delta_pp']:+.2f}pp) CI95 [{b['ci95_count'][0]}, {b['ci95_count'][1]}] "
            f"pp [{b['ci95_pp'][0]}, {b['ci95_pp'][1]}]")


def md_block(res, pcf):
    lines = [f"### {DLABEL[res['dataset']]} — {LABEL[res['model']]}" + (" (partial)" if res["partial"] else ""), "",
             f"scored n={res['n_scored']} (stock done {res['n_stock_done']}, gated done {res['n_gated_done']}, total {res['n_total']})", "",
             w.md_compare_table([res["stock"], res["gated"]], "stock vs gated (letter / official grader)"),
             "| n_opt | n | gold % (A..) | stock % | gated % | p_cf % | α_spread | stock Δ vs stock by n |", "|---|---|---|---|---|---|---|---|"]
    bn = res["gated"]["by_n_opt"]
    for n, h in res["hist_by_n_opt"].items():
        f = lambda d: "/".join(f"{d[l]}" for l in sorted(d)) if d else "—"
        d = bn.get(n, {})
        lines.append(f"| {n} | {h['n']} | {f(h['gold_pct'])} | {f(h['stock_pct'])} | {f(h['gated_pct'])} | {f(h['p_cf_pct']) if h['p_cf_pct'] else '—'} | "
                     f"{h['alpha_spread']} | {d.get('stock_letter', '—')}→{d.get('method_letter', '—')} ({d.get('delta', 0):+d}) |")
    lines.append("")
    for key in ("by_category", "by_l2_category"):
        if key in res:
            lines += [f"**{key}** (letter acc, gated−stock, paired bootstrap CI95 in questions)", "", "| group | n | stock % | gated % | Δ | CI95 |", "|---|---|---|---|---|---|"]
            for k, v in res[key].items():
                ci = v["bootstrap"]["ci95_count"] if v["bootstrap"] else ["—", "—"]
                lines.append(f"| {k} | {v['n']} | {v['stock_pct']} | {v['gated_pct']} | {v['delta']:+d} | [{ci[0]}, {ci[1]}] |")
            lines.append("")
    lines.append(f"timing: stock {res['timing']['stock_sec_per_item']} s/item, gated {res['timing']['gated_sec_per_item']} s/item")
    lines.append("")
    return "\n".join(lines)


def prior_md(model, path: Path):
    if not path.exists():
        return f"(prior file missing: {path})\n"
    d = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"**content-free prior {LABEL[model]}** ({path.relative_to(PROJECT)}; letter ids {d['letter_ids']})", "",
             "| cond | n | p_cf (A..) | α = 1 − p_min/p_max | vocab argmax | prefix_len |", "|---|---|---|---|---|---|"]
    for r in d["records"]:
        lines.append(f"| {r['cond']} | {r['n_opt']} | {'/'.join(f'{100 * v:.1f}' for v in r['p_cf'])} | {r['alpha_1_minus_pmin_over_pmax']:.4f} | {r['argmax_text']!r} | {r['prefix_len']} |")
    return "\n".join(lines) + "\n"


def main():
    results = {}
    # ---- MMBench
    gold_mm, n_mm = gold_mmbench()
    mm_summary = {"dataset": "mmbench", "n_total": n_mm, "models": {}}
    mm_md = ["# MMBench-dev EN — stock vs gated (2026-10-13c)\n"]
    for model in ("lavida", "mmada", "lladav"):
        st, gt = jl(MMB / f"{model}_stock.jsonl"), jl(MMB / f"{model}_gated.jsonl")
        pcf = pcf_of(MMB / f"cf_{model}.json")
        mm_md.append(prior_md(model, MMB / f"cf_{model}.json") if (MMB / f"cf_{model}.json").exists() else "")
        if st and gt:
            res = score("mmbench", model, st, gt, gold_mm, n_mm, pcf, MMB)
            mm_summary["models"][model] = res
            results[("mmbench", model)] = res
            mm_md.append(md_block(res, pcf))
        else:
            mm_summary["models"][model] = {"status": "missing", "stock_done": len(st or []), "gated_done": len(gt or [])}
    # ---- LLaDA-V
    llv_summary = {"model": "lladav", "datasets": {}}
    llv_md = ["# LLaDA-V — stock vs gated on ScienceQA-IMG test and M3CoT val (2026-10-13c)\n", prior_md("lladav", LLV / "cf_lladav.json")]
    pcf = pcf_of(LLV / "cf_lladav.json")
    for ds in ("sciqa", "m3cot"):
        st, gt = jl(LLV / f"{ds}_stock.jsonl"), jl(LLV / f"{ds}_gated.jsonl")
        if st and gt:
            gold, n_total = gold_sciqa_standard() if ds == "sciqa" else (gold_from_rows(st), 1108)
            res = score(ds, "lladav", st, gt, gold, n_total, pcf, LLV)
            res["files"] = {"stock": str(LLV / f"{ds}_stock.jsonl"), "gated": str(LLV / f"{ds}_gated.jsonl")}
            llv_summary["datasets"][ds] = res
            results[(ds, "lladav")] = res
            llv_md.append(md_block(res, pcf))
        else:
            llv_summary["datasets"][ds] = {"status": "missing", "stock_done": len(st or []), "gated_done": len(gt or [])}
    # ---- cross table
    cross = ["## Cross table: letter accuracy, Δ gated−stock, paired bootstrap CI95 (1000, seed 20261013)", "",
             "| dataset | LaViDa | MMaDA | LLaDA-V |", "|---|---|---|---|"]
    for ds in ("sciqa", "m3cot", "mmbench"):
        cells = []
        for model in ("lavida", "mmada", "lladav"):
            r = results.get((ds, model))
            cells.append(cell(r) if r else (week1_cell(ds, model) if ds != "mmbench" and model != "lladav" else "—"))
        cross.append(f"| {DLABEL[ds]} | " + " | ".join(cells) + " |")
    cross.append("")
    cross.append("LaViDa/MMaDA ScienceQA and M3CoT cells: outputs/week1_analysis_20261013/summary.json "
                 "(calib_gated_decode_20261013, m3cot_20261013 spread). Others: this run.")
    cross_md = "\n".join(cross) + "\n"
    for out_dir, summary, md in ((MMB, mm_summary, mm_md), (LLV, llv_summary, llv_md)):
        out_dir.mkdir(parents=True, exist_ok=True)
        summary["bootstrap"] = {"n_boot": w.N_BOOT, "seed": w.SEED, "type": "paired per-question percentile"}
        summary["cross_table_md"] = cross_md
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        (out_dir / "tables.md").write_text("\n".join(md) + "\n" + cross_md, encoding="utf-8")
    print(cross_md)
    print(f"wrote {MMB / 'summary.json'} {LLV / 'summary.json'}")


if __name__ == "__main__":
    main()
