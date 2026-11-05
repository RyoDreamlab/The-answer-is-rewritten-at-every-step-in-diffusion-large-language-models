#!/usr/bin/env python3
"""Pre-registered commit-step rule evaluation (offline replay, CPU only).

Follows paper/drafts/prereg_commit_rules_20261102.json
(sha256 4a99815777f5b8956edd84232602361e1437344529bdfc681954b0edff2a8bd3).
"""
import glob
import json
import math
import os
import random
import re
from collections import Counter

import numpy as np

O = "outputs"
OUT = f"{O}/prereg_rules_20261102"
SEED = 20261102
YESNO = {"pope", "mme"}
MODELS = ["lavida", "mmada", "dreamvl", "lladav"]
DATASETS = ["sciqa", "m3cot", "mmbench", "mmstar", "pope", "mme", "vstar"]
CELLS = {
    ("lavida", "sciqa"): f"{O}/letter_argmax_traj_20261010/lavida.jsonl",
    ("mmada", "sciqa"): f"{O}/letter_argmax_traj_20261010/mmada.jsonl",
    ("dreamvl", "sciqa"): f"{O}/dreamvl_20261013g/sciqa_20261022/dreamvl_stock.jsonl",
    ("lladav", "sciqa"): f"{O}/lladav_20261013c/sciqa_stock.jsonl",
    ("lavida", "m3cot"): f"{O}/m3cot_20261012/lavida_stock.jsonl",
    ("mmada", "m3cot"): f"{O}/m3cot_20261012/mmada_stock.jsonl",
    ("dreamvl", "m3cot"): f"{O}/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl",
    ("lladav", "m3cot"): f"{O}/lladav_20261013c/m3cot_stock.jsonl",
    ("lavida", "mmbench"): f"{O}/mmbench_20261013c/lavida_stock.jsonl",
    ("mmada", "mmbench"): f"{O}/mmbench_20261013c/mmada_stock.jsonl",
    ("dreamvl", "mmbench"): f"{O}/dreamvl_20261013g/mmbench_20261022/dreamvl_stock.jsonl",
    ("lladav", "mmbench"): f"{O}/mmbench_lladav_20261022/lladav_stock.jsonl",
    ("lavida", "mmstar"): f"{O}/mmstar_20261013e/lavida_stock.jsonl",
    ("mmada", "mmstar"): f"{O}/mmstar_20261013e/mmada_stock.jsonl",
    ("dreamvl", "mmstar"): f"{O}/dreamvl_20261013g/mmstar_20261022/dreamvl_stock.jsonl",
    ("lladav", "mmstar"): f"{O}/mmstar_lladav_20261022/lladav_stock.jsonl",
    ("lavida", "pope"): f"{O}/pope_20261013e/lavida_stock.jsonl",
    ("mmada", "pope"): f"{O}/pope_20261013e/mmada_stock.jsonl",
    ("dreamvl", "pope"): f"{O}/dreamvl_20261013g/pope_20261022/dreamvl_stock.jsonl",
    ("lladav", "pope"): f"{O}/pope_lladav_20261022/lladav_stock.jsonl",
    ("lavida", "mme"): f"{O}/mme_20261022/lavida_stock.jsonl",
    ("mmada", "mme"): f"{O}/mme_20261022/mmada_stock.jsonl",
    ("dreamvl", "mme"): f"{O}/mme_20261022/dreamvl_stock.jsonl",
    ("lladav", "mme"): f"{O}/mme_20261022/lladav_stock.jsonl",
    ("lavida", "vstar"): f"{O}/vstar_20261022/lavida_stock.jsonl",
    ("mmada", "vstar"): f"{O}/vstar_20261022/mmada_stock.jsonl",
    ("dreamvl", "vstar"): f"{O}/vstar_20261022/dreamvl_stock.jsonl",
    ("lladav", "vstar"): f"{O}/vstar_20261022/lladav_stock.jsonl",
}
# (name, n_constants, order for tie-break)
CONFIGS = [("R0_stock", 0, 0), ("R1_vote_all", 0, 1), ("R2_back1", 0, 2), ("R5_commit_spread", 0, 3),
           ("R6_agree_gate", 0, 4), ("R3_run:2", 1, 5), ("R3_run:4", 1, 6), ("R3_run:6", 1, 7),
           ("R4_early_m:0.1", 1, 8), ("R4_early_m:0.2", 1, 9), ("R4_early_m:0.3", 1, 10)]
NAMES = [c[0] for c in CONFIGS]

P_ANS = re.compile(r"(?i:answer)\s*(?i:is|:)?\s*\(?([A-E])\b")
P_OPT = re.compile(r"(?i:option)\s*\(?([A-E])\b")
P_PAR = re.compile(r"\(([A-E])\)")
P_DOT = re.compile(r"(?:^|\s)([A-E])[.:]\s")
P_YN = re.compile(r"\b(yes|no)\b", re.I)


def cf(s):
    return (s or "").strip().casefold().rstrip(".")


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def labels_of(r, ds):
    lab = r.get("labels")
    if lab:
        return [cf(x) for x in lab]
    if ds in YESNO:
        return ["yes", "no"]
    n = int(r.get("n_opt") or 5)
    return [cf(x) for x in "ABCDE"[:n]]


def parse(r, ds, lab):
    fin = cf(r.get("finished_letter"))
    if fin in lab:
        return fin, "leading"
    text = r.get("finished_text") or ""
    if ds in YESNO:
        m = P_YN.search(text)
        return (m.group(1).casefold(), "regex") if m else (None, "unparsed")
    for pat in (P_ANS, P_OPT, P_PAR, P_DOT):
        m = pat.search(text)
        if m and cf(m.group(1)) in lab:
            return cf(m.group(1)), "regex"
    return None, "unparsed"


def prior_path(m, ds):
    if m in ("lavida", "mmada"):
        return {"sciqa": f"{O}/calib_vote_20261012/cf_{m}.jsonl", "m3cot": f"{O}/m3cot_20261012/cf_{m}.jsonl",
                "mmbench": f"{O}/mmbench_20261013c/cf_{m}.json", "mmstar": f"{O}/mmstar_20261013e/cf_{m}.json",
                "pope": f"{O}/pope_20261013e/cf_pope_{m}.json"}.get(ds, f"{O}/{ds}_20261022/cf_{ds}_{m}.json")
    if m == "dreamvl":
        return {"pope": f"{O}/dreamvl_20261013g/pope_20261022/cf_pope_dreamvl.json"}.get(
            ds, f"{O}/{ds}_20261022/cf_{ds}_dreamvl.json" if ds in ("mme", "vstar") else f"{O}/dreamvl_20261013g/cf_dreamvl.json")
    return {"mmstar": f"{O}/mmstar_lladav_20261022/cf_mmstar_lladav.json", "pope": f"{O}/pope_lladav_20261022/cf_pope_lladav.json",
            "mme": f"{O}/mme_20261022/cf_mme_lladav.json", "vstar": f"{O}/vstar_20261022/cf_vstar_lladav.json"}.get(
        ds, f"{O}/lladav_20261013c/cf_lladav.json")


def prior_for(m, ds):
    """{n: {label_casefold: log p_cf}} from the a_global content-free records (same as calib_decode load_priors)."""
    f = prior_path(m, ds)
    if not os.path.exists(f):
        return None, None
    if f.endswith(".jsonl"):
        recs = [json.loads(l) for l in open(f) if l.strip()]
    else:
        recs = json.load(open(f))["records"]
    out = {}
    for r in recs:
        if r.get("cond") != "a_global":
            continue
        n = int(r["n_opt"])
        v = [float(x) for x in r["letter_logit64"][:n]]
        mx = max(v)
        z = math.log(sum(math.exp(x - mx) for x in v)) + mx
        labs = [cf(x) for x in (r.get("labels") or "ABCDE"[:n])][:n]
        out[n] = dict(zip(labs, [x - z for x in v]))
    return f, out


def margin(v):
    if len(v) < 2:
        return 1.0
    mx = max(v)
    e = [math.exp(x - mx) for x in v]
    s = sum(e)
    p = sorted((x / s for x in e), reverse=True)
    return p[0] - p[1]


def majority(hist):
    if not hist:
        return None
    c = Counter(hist)
    top = max(c.values())
    for p in reversed(hist):
        if c[p] == top:
            return p


def eval_item(r, ds, prior):
    lab = labels_of(r, ds)
    gold = cf(r.get("gold"))
    stock_ans, how = parse(r, ds, lab)
    s_ok = int(stock_ans == gold)
    raw_ok = int(cf(r.get("finished_letter")) == gold)
    traj = [cf(p) for p in (r.get("pos0_argmax") or [])]
    out = {"R0_stock": s_ok}
    if not traj:
        for k in NAMES:
            out[k] = s_ok
        return out, s_ok, raw_ok, how, s_ok
    last = len(traj) - 1
    lastw = traj[last]

    def ans_of(letter):
        return s_ok if letter == lastw else int(letter == gold)

    hist = [p for p in traj if p in lab]
    maj = majority(hist)
    out["R1_vote_all"] = s_ok if maj is None else ans_of(maj)
    out["R2_back1"] = ans_of(traj[last - 1]) if last >= 1 else s_ok
    for K in (2, 4, 6):
        run = 0
        for p in reversed(traj):
            if p != lastw:
                break
            run += 1
        if run >= K or run == len(traj):
            out[f"R3_run:{K}"] = s_ok
        else:
            out[f"R3_run:{K}"] = ans_of(traj[last - run])
    n = len(lab)
    lg = r.get("letter_logits_raw_at_commit")
    v = [float(x) for x in lg[:n]] if lg and len(lg) >= n else None
    mg = margin(v) if v else None
    for tau in (0.1, 0.2, 0.3):
        key = f"R4_early_m:{tau}"
        if mg is None or mg >= tau:
            out[key] = s_ok
        else:
            half = traj[: max(1, len(traj) // 2)]
            c = Counter(half)
            best, bn = None, -1
            for p in half:
                if c[p] > bn or (c[p] == bn and p == half[-1]):
                    best, bn = p, c[p]
            out[key] = ans_of(best)
    shifted = None
    if v and prior and n in prior and all(l in prior[n] for l in lab):
        lp = [prior[n][l] for l in lab]
        a = 1 - math.exp(min(lp) - max(lp))
        shifted = [x - a * y for x, y in zip(v, lp)]
    R5_APPLIED[ds] += shifted is not None
    if shifted is None:
        out["R5_commit_spread"] = s_ok
        out["R6_agree_gate"] = s_ok
    else:
        out["R5_commit_spread"] = ans_of(lab[max(range(n), key=lambda j: shifted[j])])
        if maj is None or maj == lastw:
            out["R6_agree_gate"] = s_ok
        else:
            cand = [j for j in range(n) if lab[j] in set(hist)]
            out["R6_agree_gate"] = ans_of(lab[max(cand, key=lambda j: shifted[j])])
    oracle = int(s_ok or gold in set(hist))
    return out, s_ok, raw_ok, how, oracle


R5_APPLIED = Counter()


def split_idx(cell, n):
    idx = list(range(n))
    random.Random(f"{SEED}:{cell}").shuffle(idx)
    h = n // 2
    return np.array(sorted(idx[:h])), np.array(sorted(idx[h:]))


def boot_ci(diff_by_cell, reps=2000):
    rng = np.random.default_rng(SEED)
    tot = np.zeros(reps)
    for d in diff_by_cell:
        if len(d) == 0:
            continue
        s = rng.integers(0, len(d), size=(reps, len(d)))
        tot += d[s].sum(axis=1)
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))


def select(cells, data, idx_key):
    best = None
    for name, nc, order in CONFIGS:
        tot = int(sum(data[c]["diff"][name][data[c][idx_key]].sum() if idx_key else data[c]["diff"][name].sum() for c in cells))
        key = (tot, -nc, -order)
        if best is None or key > best[0]:
            best = (key, name, tot)
    return best[1], best[2]


def report(name, cells, data, idx_key, label):
    rows, wins, losses, ties, net = [], 0, 0, 0, 0
    for c in cells:
        d = data[c]["diff"][name]
        d = d[data[c][idx_key]] if idx_key else d
        delta = int(d.sum())
        net += delta
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
        rows.append((c, len(d), delta))
    def ci(sub):
        return boot_ci([(data[c]["diff"][name][data[c][idx_key]] if idx_key else data[c]["diff"][name]) for c in sub])
    no_yn = [c for c in cells if c[1] not in YESNO]
    no_ll = [c for c in cells if c[0] != "lladav"]
    res = {
        "label": label, "rule": name, "wins": wins, "losses": losses, "ties": ties, "net": net,
        "ci_all": ci(cells),
        "net_no_yesno": int(sum(r[2] for r in rows if r[0][1] not in YESNO)), "ci_no_yesno": ci(no_yn),
        "net_no_lladav": int(sum(r[2] for r in rows if r[0][0] != "lladav")), "ci_no_lladav": ci(no_ll),
        "per_cell": {f"{c[0]}_{c[1]}": {"n": n, "delta": dl} for c, n, dl in rows},
    }
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    parse_stats = {}
    prior_src = {}
    for (m, ds), path in CELLS.items():
        recs = load(path)
        pf, prior = prior_for(m, ds)
        prior_src[f"{m}_{ds}"] = pf
        cols = {k: [] for k in NAMES}
        stock, raw, orc, hows = [], [], [], Counter()
        for r in recs:
            out, s_ok, raw_ok, how, oracle = eval_item(r, ds, prior)
            for k in NAMES:
                cols[k].append(out[k])
            stock.append(s_ok)
            raw.append(raw_ok)
            orc.append(oracle)
            hows[how] += 1
        st = np.array(stock)
        dev, held = split_idx(f"{m}_{ds}", len(recs))
        data[(m, ds)] = {
            "n": len(recs), "stock": int(st.sum()), "raw_stock": int(sum(raw)), "oracle": int(sum(orc)),
            "diff": {k: np.array(cols[k]) - st for k in NAMES}, "dev": dev, "held": held,
            "oracle_held": int(np.array(orc)[held].sum() - st[held].sum()),
        }
        parse_stats[f"{m}_{ds}"] = dict(hows)
    cells = list(CELLS)
    print("prior sources:", {k: (os.path.basename(v) if v else None) for k, v in prior_src.items()})
    print("R5/R6 applicable items per dataset:", dict(R5_APPLIED))
    print("\n# parser effect (stock correct: leading-token only -> pre-registered parser)")
    for c in cells:
        d = data[c]
        ps = parse_stats[f"{c[0]}_{c[1]}"]
        if d["raw_stock"] != d["stock"] or ps.get("regex"):
            print(f"{c[0]}_{c[1]}\tn={d['n']}\traw={d['raw_stock']}\tparsed={d['stock']}\t{ps}")

    sel, dev_tot = select(cells, data, "dev")
    print(f"\n# selected on dev (all 28 cells): {sel}  dev_net={dev_tot:+d}")
    print("# dev net per config:", {k: int(sum(data[c]['diff'][k][data[c]['dev']].sum() for c in cells)) for k in NAMES})
    held = report(sel, cells, data, "held", "held-out, rule selected on dev")
    print(f"# held-out {sel}: wins={held['wins']} losses={held['losses']} ties={held['ties']} net={held['net']:+d} "
          f"CI95={held['ci_all']} | no-yes/no net={held['net_no_yesno']:+d} CI={held['ci_no_yesno']} | "
          f"no-LLaDA-V net={held['net_no_lladav']:+d} CI={held['ci_no_lladav']}")
    print("# held-out per cell:", {k: v["delta"] for k, v in held["per_cell"].items()})
    print("# held-out net per config (transparency only, not used for selection):",
          {k: int(sum(data[c]['diff'][k][data[c]['held']].sum() for c in cells)) for k in NAMES})

    cross = []
    for gname, A in (("models lavida+mmada -> dreamvl+lladav", [c for c in cells if c[0] in ("lavida", "mmada")]),
                     ("models dreamvl+lladav -> lavida+mmada", [c for c in cells if c[0] in ("dreamvl", "lladav")]),
                     ("datasets MCQ -> pope+mme+vstar", [c for c in cells if c[1] in ("sciqa", "m3cot", "mmbench", "mmstar")]),
                     ("datasets pope+mme+vstar -> MCQ", [c for c in cells if c[1] in ("pope", "mme", "vstar")])):
        B = [c for c in cells if c not in A]
        s, t = select(A, data, None)
        r = report(s, B, data, None, gname)
        cross.append(r)
        print(f"# cross {gname}: selected {s} (A net {t:+d}) -> B wins={r['wins']} losses={r['losses']} ties={r['ties']} "
              f"net={r['net']:+d} CI95={r['ci_all']}")

    print("\n# per cell: n, stock(parsed), held-out delta of selected rule, oracle headroom (full / held-out)")
    for c in cells:
        d = data[c]
        print(f"{c[0]}_{c[1]}\t{d['n']}\t{d['stock']}\t{held['per_cell'][c[0] + '_' + c[1]]['delta']:+d}\t+{d['oracle'] - d['stock']}\t+{d['oracle_held']}")
    json.dump({"selected": sel, "dev_net": dev_tot, "held": held, "cross": cross, "parse_stats": parse_stats,
               "cells": {f"{c[0]}_{c[1]}": {k: data[c][k] for k in ("n", "stock", "raw_stock", "oracle", "oracle_held")} for c in cells},
               "prior_src": prior_src}, open(f"{OUT}/results.json", "w"), indent=1)
    print(f"\nwrote {OUT}/results.json")


if __name__ == "__main__":
    main()
