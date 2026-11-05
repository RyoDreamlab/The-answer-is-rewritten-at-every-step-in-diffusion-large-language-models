#!/usr/bin/env python3
"""P6 analysis (CPU): flip-rate-vs-β tables with Wilson CIs, writer-vs-random control comparison, unmask-order edit
distance distribution, Arm 2 2×2 tables, Arm 3 template table, and the pre-registered success flags
(STORY_LOCK_20261013.md, 追加 2026-10-13 17:15).

Reads outputs/p6_causal_20261013f/{steer_*.jsonl, patch.jsonl, canvas.jsonl, calib_steer.json};
writes summary.json and tables.md in the same directory (or --out-dir).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))

def load_jsonl(path: Path) -> list[dict]:
    """jsonl loader that allows several records per item id (one per β / condition)."""
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_no} is not json") from exc
    return rows


OUT_ROOT = (PROJECT / "outputs" / "p6_causal_20261013f").resolve()
LETTERS = "ABCDE"
GROUPS = ("writer_heads", "random_heads", "writer_comps")
PATCH_LAYERS = (24, 28, 32)
Z = 1.959963984540054


def wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "ci95": [None, None]}
    p = k / n
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return {"k": int(k), "n": int(n), "p": p, "ci95": [max(0.0, centre - half), min(1.0, centre + half)]}


def diff_ci(k1, n1, k2, n2) -> dict:
    """Newcombe (Wilson-based) CI for p1 − p2."""
    if n1 == 0 or n2 == 0:
        return {"diff": None, "ci95": [None, None]}
    w1, w2 = wilson(k1, n1), wilson(k2, n2)
    d = w1["p"] - w2["p"]
    lo = d - math.sqrt((w1["p"] - w1["ci95"][0]) ** 2 + (w2["ci95"][1] - w2["p"]) ** 2)
    hi = d + math.sqrt((w1["ci95"][1] - w1["p"]) ** 2 + (w2["p"] - w2["ci95"][0]) ** 2)
    return {"diff": d, "ci95": [lo, hi]}


def fmt(w: dict) -> str:
    if w["n"] == 0:
        return "—"
    return f"{w['k']}/{w['n']} = {100 * w['p']:.1f}% [{100 * w['ci95'][0]:.1f}, {100 * w['ci95'][1]:.1f}]"


def q(a: list, pct) -> float | None:
    return float(np.percentile(a, pct)) if a else None


def analyse_steer(out_dir: Path) -> tuple[dict, list[str]]:
    res, md = {}, []
    calib = json.loads((out_dir / "calib_steer.json").read_text()) if (out_dir / "calib_steer.json").exists() else None
    res["calibration"] = None
    if calib:
        res["calibration"] = {"n_items": calib["n_items"], "probes": calib["probes"], "random_heads": calib.get("random_heads"),
                              "groups": {g: {k: e.get(k) for k in ("unit_scale_own", "unit_scale_used", "unit_scale_source", "slope_p1_iqr",
                                                                     "linearity_median_ratio_p2_over_p1", "proj_norm_mean",
                                                                     "cos_total_write_vs_v_mean", "heads", "comps")}
                                         for g, e in calib["groups"].items()}}
        md.append(f"### β calibration ({calib['n_items']} items, probes c = {calib['probes']})\n")
        md.append("| group | heads / comps | unit_scale (logit per unit residual norm) | used | IQR | lin. p2/p1 | proj-norm | cos(write, v) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for g, e in calib["groups"].items():
            md.append(f"| {g} | {e['heads'] or e['comps']} | {e['unit_scale_own']:.4f} | {e['unit_scale_used']:.4f} ({e['unit_scale_source']}) | "
                      f"[{e['slope_p1_iqr'][0]:.4f}, {e['slope_p1_iqr'][1]:.4f}] | {e['linearity_median_ratio_p2_over_p1']:.3f} | "
                      f"{e['proj_norm_mean']:.3f} | {e['cos_total_write_vs_v_mean']:.3f} |")
        md.append("")
    data = {g: load_jsonl(out_dir / f"steer_{g}.jsonl") for g in GROUPS}
    groups_present = [g for g in GROUPS if data[g]]
    if not groups_present:
        return res, md
    # common items across the groups that exist (and complete β sets)
    by = {g: {} for g in groups_present}
    for g in groups_present:
        for r in data[g]:
            by[g].setdefault(r["id"], {})[r["beta"]] = r
    betas = sorted({b for g in groups_present for recs in by[g].values() for b in recs})
    common = set.intersection(*[{i for i, recs in by[g].items() if all(b in recs for b in betas)} for g in groups_present])
    ids = sorted(common)
    res["n_items_common"] = len(ids)
    res["betas"] = betas
    res["beta0_equals_stock_ref"] = {}
    res["groups"] = {}
    for g in groups_present:
        rows0 = [by[g][i][0.0] for i in ids]
        cmp = [r for r in rows0 if r.get("ref_stock_finished_letter") is not None]
        ok = sum(r["ref_stock_finished_letter"] == r["stock_finished_letter"] and r["ref_stock_letter_unmask_step"] == r["stock_letter_unmask_step"] for r in cmp)
        res["beta0_equals_stock_ref"][g] = f"{ok}/{len(cmp)}"
        per_beta = {}
        for b in betas:
            rows = [by[g][i][b] for i in ids]
            fl = [r for r in rows if r["flipped"]]
            ed = [r["order_edit_distance"] for r in rows]
            per_beta[f"{b:g}"] = {
                "flip": wilson(len(fl), len(rows)),
                "flip_to_second": wilson(sum(r["final_is_second"] for r in fl), len(fl)),
                "flip_to_prior_top": wilson(sum(r["final_is_prior_top"] for r in fl), len(fl)),
                "flip_to_other": wilson(sum(r["final_is_other"] for r in fl), len(fl)),
                "flip_to_null": wilson(sum(r["finished_letter"] is None for r in fl), len(fl)),
                "step0_flipped": wilson(sum(r["step0_flipped"] for r in rows), len(rows)),
                "step0_shift_second_minus_first_mean": float(np.mean([r["step0_shift_second_minus_first"] for r in rows])),
                "step0_shift_second_minus_first_median": float(np.median([r["step0_shift_second_minus_first"] for r in rows])),
                "order_edit_distance": {"median": q(ed, 50), "p75": q(ed, 75), "p90": q(ed, 90), "max": max(ed), "mean": float(np.mean(ed)),
                                        "frac_le1": float(np.mean([e <= 1 for e in ed])), "frac_identical": float(np.mean([r["order_identical"] for r in rows])),
                                        "hist": {str(k): int(sum(e == k for e in ed)) for k in sorted(set(ed))}},
                "letter_unmask_step15_frac": float(np.mean([r["letter_unmask_step"] == 15 for r in rows])),
                "acc": int(sum(r["correct"] for r in rows)), "stock_acc": int(sum(r["stock_finished_letter"] == r["gold"] for r in rows)),
                "letter_hist": {l: int(sum(r["finished_letter"] == l for r in rows)) for l in LETTERS},
                "c_raw_median": float(np.median([r["c_raw"] for r in rows])),
            }
        rates = [per_beta[f"{b:g}"]["flip"]["p"] for b in betas]
        nonneg = [b for b in betas if b >= 0]
        nn_rates = [per_beta[f"{b:g}"]["flip"]["p"] for b in nonneg]
        # the dose-response claim is about pushing toward the second letter (β ≥ 0); β < 0 pushes toward the stock
        # step-0 first letter and legitimately flips items whose stock final letter differs from it
        res["groups"][g] = {"per_beta": per_beta, "flip_rates": rates, "betas_nonneg": nonneg,
                            "monotone_nondecreasing_in_beta": all(nn_rates[i] <= nn_rates[i + 1] + 1e-12 for i in range(len(nn_rates) - 1)),
                            "monotone_nondecreasing_all_betas": all(rates[i] <= rates[i + 1] + 1e-12 for i in range(len(rates) - 1)),
                            "spearman_flip_vs_beta_nonneg": float(np.corrcoef(np.argsort(np.argsort(nonneg)), np.argsort(np.argsort(nn_rates)))[0, 1]) if len(set(nn_rates)) > 1 else None}
    # control comparison
    comp = {}
    if "writer_heads" in groups_present and "random_heads" in groups_present:
        for b in betas:
            w = res["groups"]["writer_heads"]["per_beta"][f"{b:g}"]["flip"]; r_ = res["groups"]["random_heads"]["per_beta"][f"{b:g}"]["flip"]
            comp[f"{b:g}"] = {"writer": w, "random": r_, "ratio": (w["p"] / r_["p"]) if r_["p"] else (math.inf if w["p"] else None),
                              "diff": diff_ci(w["k"], w["n"], r_["k"], r_["n"]),
                              "ratio_lower_bound_ci": (w["ci95"][0] / r_["ci95"][1]) if r_["ci95"][1] else None}
    res["writer_vs_random"] = comp
    # pre-registered success flags (Arm 1)
    flags = {}
    if "writer_heads" in groups_present:
        g = res["groups"]["writer_heads"]
        b4 = "4" if "4" in g["per_beta"] else None
        flags["monotone_in_beta"] = g["monotone_nondecreasing_in_beta"]
        flags["ratio_vs_random_at_beta4"] = comp.get("4", {}).get("ratio") if comp else None
        flags["ratio_ge_3_at_beta4"] = (flags["ratio_vs_random_at_beta4"] is not None and flags["ratio_vs_random_at_beta4"] >= 3) if comp else None
        flags["median_order_edit_distance_le1_at_beta4"] = (g["per_beta"][b4]["order_edit_distance"]["median"] <= 1) if b4 else None
        flags["median_order_edit_distance_le1_all_positive_betas"] = all(g["per_beta"][f"{b:g}"]["order_edit_distance"]["median"] <= 1 for b in betas if b > 0)
        flags["flip_rate_at_beta4"] = g["per_beta"][b4]["flip"]["p"] if b4 else None
        flags["flip_rate_at_beta8"] = g["per_beta"]["8"]["flip"]["p"] if "8" in g["per_beta"] else None
        flags["arm1_positive"] = bool(flags["monotone_in_beta"] and flags.get("ratio_ge_3_at_beta4") and flags["median_order_edit_distance_le1_at_beta4"]) if comp else None
    res["success_flags"] = flags
    # ---- tables
    md.append(f"### Arm 1 — persistent head steering, MMaDA, {len(ids)} common items (β = 0 reproduces stock reference: {res['beta0_equals_stock_ref']})\n")
    for g in groups_present:
        e = res["groups"][g]
        md.append(f"**{g}**  monotone(β ≥ 0, pre-registered) = {e['monotone_nondecreasing_in_beta']}, monotone(all β) = {e['monotone_nondecreasing_all_betas']}\n")
        md.append("| β | c (median) | step-0 Δ(l₂−l₁) mean | step-0 argmax flipped | final flipped vs stock [Wilson 95%] | →second | →prior top | →other | →null | order edit dist. median / p90 / frac≤1 / identical | letter @ step 15 | acc (stock) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for b in betas:
            p = e["per_beta"][f"{b:g}"]
            od = p["order_edit_distance"]
            md.append(f"| {b:+g} | {p['c_raw_median']:.1f} | {p['step0_shift_second_minus_first_mean']:+.2f} | {fmt(p['step0_flipped'])} | {fmt(p['flip'])} | "
                      f"{fmt(p['flip_to_second'])} | {fmt(p['flip_to_prior_top'])} | {fmt(p['flip_to_other'])} | {fmt(p['flip_to_null'])} | "
                      f"{od['median']:.0f} / {od['p90']:.0f} / {od['frac_le1']:.2f} / {od['frac_identical']:.2f} | {p['letter_unmask_step15_frac']:.2f} | {p['acc']} ({p['stock_acc']}) |")
        md.append("")
    if comp:
        md.append("**writer_heads vs random_heads (same injected residual norm)**\n")
        md.append("| β | writer flip | random flip | ratio | diff [Newcombe 95%] |")
        md.append("|---|---|---|---|---|")
        for b in betas:
            c = comp[f"{b:g}"]
            d = c["diff"]
            ratio = c["ratio"]
            ratio_s = "—" if ratio is None else ("∞" if ratio == math.inf else f"{ratio:.2f}")
            diff_s = f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d["diff"] is not None else "—"
            md.append(f"| {b:+g} | {fmt(c['writer'])} | {fmt(c['random'])} | {ratio_s} | {diff_s} |")
        md.append("")
    md.append(f"**Arm 1 pre-registered flags**: `{json.dumps(flags)}`\n")
    return res, md


def analyse_patch(out_dir: Path) -> tuple[dict, list[str]]:
    rows = load_jsonl(out_dir / "patch.jsonl")
    res, md = {}, []
    if not rows:
        return res, md
    res["n_records"] = len(rows)
    res["n_recipients"] = len({r["id"] for r in rows})
    res["stock_equals_ref"] = None
    tab = {}
    for variant in ("step0", "every"):
        tab[variant] = {}
        for L in PATCH_LAYERS:
            ent = {}
            for cond in ("differ", "same"):
                rr = [r for r in rows if r["variant"] == variant and r["layer"] == L and r["cond"] == cond]
                ent[cond] = {"final_eq_donor": wilson(sum(r["final_eq_donor_step0"] for r in rr), len(rr)),
                             "changed_vs_stock": wilson(sum(r["changed"] for r in rr), len(rr)),
                             "step0_patched_eq_donor": wilson(sum(r["step0_patched_eq_donor"] for r in rr), len(rr)),
                             "branch": wilson(sum(r["branch"] for r in rr), len(rr)),
                             "committed_at_step0": wilson(sum(r["committed_at_step0"] for r in rr), len(rr)),
                             "order_edit_distance_median": q([r["order_edit_distance"] for r in rr], 50),
                             "null_letter": wilson(sum(r["finished_letter"] is None for r in rr), len(rr)),
                             "acc": int(sum(r["correct"] for r in rr)),
                             "two_by_two": {"final_eq_donor": int(sum(r["final_eq_donor_step0"] for r in rr)),
                                            "final_ne_donor": int(sum(not r["final_eq_donor_step0"] for r in rr))}}
            d, s = ent["differ"]["final_eq_donor"], ent["same"]["final_eq_donor"]
            ent["differ_minus_same"] = diff_ci(d["k"], d["n"], s["k"], s["n"])
            ent["success_ge_0.3"] = (ent["differ_minus_same"]["diff"] is not None and ent["differ_minus_same"]["diff"] >= 0.3)
            tab[variant][str(L)] = ent
    res["table"] = tab
    res["success_flags"] = {"arm2_step0_any_layer_ge_0.3": any(e["success_ge_0.3"] for e in tab["step0"].values()),
                            "arm2_every_L24_or_L28_ge_0.3": any(tab["every"][str(L)]["success_ge_0.3"] for L in (24, 28) if str(L) in tab["every"]),
                            "arm2_every_L32_positive_control_final_eq_donor_differ": tab["every"].get("32", {}).get("differ", {}).get("final_eq_donor", {}).get("p"),
                            "arm2_every_L32_positive_control_ok_ge_0.9": ((tab["every"].get("32", {}).get("differ", {}).get("final_eq_donor", {}).get("p") or 0) >= 0.9)}
    md.append(f"### Arm 2 — step-0 activation patching at the answer position, {res['n_recipients']} recipients\n")
    md.append("Rows: 2×2 of condition (donor step-0 letter differs / same) × outcome (final letter = donor step-0 letter). "
              "`step0` = patch at step 0 only (literal spec; §B2 predicts null unless the step-0 commit changes); "
              "`every` = patch at every step while the answer slot is masked (L32 = positive control).\n")
    md.append("| variant | L | cond | n | final = donor [Wilson] | final ≠ donor | changed vs stock | step-0 argmax = donor | step-0 commit changed (branch) | order dist. median | differ − same [95%] | ≥0.3 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for variant in ("step0", "every"):
        for L in PATCH_LAYERS:
            e = tab[variant][str(L)]
            for cond in ("differ", "same"):
                c = e[cond]
                dm = e["differ_minus_same"]
                dms = f"{dm['diff']:+.3f} [{dm['ci95'][0]:+.3f}, {dm['ci95'][1]:+.3f}]" if (cond == "differ" and dm["diff"] is not None) else ""
                oed = c["order_edit_distance_median"]
                oed_s = "—" if oed is None else f"{oed:.0f}"
                flag = str(e["success_ge_0.3"]) if cond == "differ" else ""
                md.append(f"| {variant} | {L} | {cond} | {c['final_eq_donor']['n']} | {fmt(c['final_eq_donor'])} | {c['two_by_two']['final_ne_donor']} | "
                          f"{fmt(c['changed_vs_stock'])} | {fmt(c['step0_patched_eq_donor'])} | {fmt(c['branch'])} | {oed_s} | {dms} | {flag} |")
    md.append("")
    md.append(f"**Arm 2 pre-registered flags**: `{json.dumps(res['success_flags'])}`\n")
    return res, md


def analyse_canvas(out_dir: Path) -> tuple[dict, list[str]]:
    rows = load_jsonl(out_dir / "canvas.jsonl")
    res, md = {}, []
    if not rows:
        return res, md
    elig = [r for r in rows if r["eligible"]]
    res["n_items"] = len(rows); res["n_eligible_commit_step15"] = len(elig)
    res["full_stock_mismatch"] = int(sum(r.get("full_stock_mismatch", False) for r in elig))
    names = []
    for r in elig:
        for nm in r["templates"]:
            if nm not in names:
                names.append(nm)
    tab = {}
    for nm in names:
        rr = [r for r in elig if nm in r["templates"] and not r["templates"][nm].get("skipped")]
        sk = sum(1 for r in elig if nm in r["templates"] and r["templates"][nm].get("skipped"))
        if not rr:
            tab[nm] = {"n": 0, "n_skipped_identical": sk}
            continue
        t = [r["templates"][nm] for r in rr]
        tab[nm] = {"n": len(rr), "n_skipped_identical": sk,
                   "eq_stock_final": wilson(sum(x["eq_stock_final"] for x in t), len(t)),
                   "eq_step0": wilson(sum(x["eq_step0"] for x in t), len(t)),
                   "acc": wilson(sum(x["correct"] for x in t), len(t)),
                   "full_argmax_is_letter": wilson(sum(x["full_argmax_is_letter"] for x in t), len(t)),
                   "margin_mean": float(np.mean([x["margin"] for x in t if x["margin"] is not None])),
                   "margin_median": float(np.median([x["margin"] for x in t if x["margin"] is not None])),
                   "letter_hist": {l: int(sum(x["restricted"] == l for x in t)) for l in LETTERS}}
    res["table"] = tab
    res["stock_eq_step0_frac"] = float(np.mean([r["stock_finished_letter"] == r["step0_letter"] for r in elig])) if elig else None
    md.append(f"### Arm 3 — canvas counterfactual (report only), {len(elig)}/{len(rows)} items with stock letter commit at step 15; "
              f"stock final = step-0 letter on {100 * (res['stock_eq_step0_frac'] or 0):.1f}%\n")
    md.append("| template (positions 1..15; answer slot masked) | n | letter = stock final | letter = step-0 letter | acc | full argmax is a letter | margin mean / median | skipped (identical to stock) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for nm, e in tab.items():
        if e["n"] == 0:
            md.append(f"| {nm} | 0 | — | — | — | — | — | {e['n_skipped_identical']} |")
            continue
        md.append(f"| {nm} | {e['n']} | {fmt(e['eq_stock_final'])} | {fmt(e['eq_step0'])} | {fmt(e['acc'])} | {fmt(e['full_argmax_is_letter'])} | "
                  f"{e['margin_mean']:.2f} / {e['margin_median']:.2f} | {e['n_skipped_identical']} |")
    md.append("")
    return res, md


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir != OUT_ROOT and OUT_ROOT not in out_dir.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {out_dir}")
    summary = {"date": "2026-10-13", "dir": str(out_dir), "model": "mmada", "arms": {}}
    md = ["# P6 — causal perturbation of the step-0 decision (MMaDA)\n",
          f"Source: `{out_dir}`.  Wilson 95% intervals; Newcombe intervals for differences.\n"]
    s, m = analyse_steer(out_dir); summary["arms"]["steer"] = s; md += m
    s, m = analyse_patch(out_dir); summary["arms"]["patch"] = s; md += m
    s, m = analyse_canvas(out_dir); summary["arms"]["canvas"] = s; md += m
    flags = {**summary["arms"]["steer"].get("success_flags", {}), **summary["arms"]["patch"].get("success_flags", {})}
    summary["success_flags"] = flags
    md.append(f"## Pre-registered flags\n\n`{json.dumps(flags)}`\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {out_dir / 'summary.json'} and {out_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
