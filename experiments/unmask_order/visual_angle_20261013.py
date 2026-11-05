#!/usr/bin/env python3
"""Visual angle on the option-letter prior: when visual evidence is weak, the letter prior takes over.

OFFLINE, CPU only, read-only on every input. ScienceQA-IMG test (2017 items), models lavida / mmada.

Shapley reconstruction (identical to hidden_shapley_offline_rules.py / letter_bias_origin.py)
------------------------------------------------------------------------------------------
outputs/hidden_shapley_split_20261010/{model}_h_subsets.npy holds, per item, the step-0 ln_f input at the
answer position (canvas index 0) for the 8 attention-key-masking subsets S of the sources {I, T, M}
(I = image keys, T = text/stem keys, M = masked answer-slot keys) in hidden_shapley_split.SUBSETS order
(none, I, T, M, IT, IM, TM, ITM).  The exact 3-player Shapley value of source x is

    phi_x = sum_{S not containing x}  |S|! (3-|S|-1)! / 3!  * ( h_{S+x} - h_S )

(hor.shapley_all), so that  h_ITM = h_none + phi_I + phi_T + phi_M  exactly (max reconstruction error
~1e-12 is checked and reported).  Any hidden vector v is projected to the 5 letter logits (ids 32..36,
letters A..E) with the float64 readout  logits(v) = (v / sqrt(mean(v^2) + 1e-5) * gamma_ln_f) @ W_letters^T
(hor.load_readout: ln_f.weight and the five ff_out rows).  The restricted step-0 prediction is the argmax of
logits(h_ITM) over the item's own n option letters; it reproduces 1448 (LaViDa) / 1064 (MMaDA) correct.
The common-component push of source x used as a sanity anchor is the first-order (RMSNorm-Jacobian at
h_ITM) letter push of the item-mean of phi_x (letter_bias_origin.first_order), B minus A averaged over items:
MMaDA T = +1.26, LaViDa T = -0.20 in outputs/letter_bias_origin_20261012/summary.json.

Per-item image-evidence measures
--------------------------------
s_I = ||phi_I||_2 ; s_T = ||phi_T||_2 ; img_spread = max - min over the item's n options of
logits(phi_I) = W . gamma . normalize(phi_I)  (the letter readout of the image share alone);
fo_spread_I = the same spread of the first-order push J(h_ITM) phi_I (secondary measure).

Content-free prior p_cf(n): a_global records of outputs/calib_vote_20261012/cf_{model}.jsonl, log-softmax over
the first n letter logits.  Stock final letters: outputs/letter_argmax_traj_20261010; gated-calibration final
letters: outputs/calib_gated_decode_20261013.  Option texts: datasets/sciqa_img_standard_v1/manifest.json.

Writes only new files into outputs/visual_angle_20261013/ (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from scipy import stats

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import hidden_shapley_offline_rules as hor  # noqa: E402

LETTERS = hor.LETTERS
IDX = hor.IDX
MODELS = ("lavida", "mmada")
CF_DIR = PROJECT / "outputs" / "calib_vote_20261012"
TRAJ_DIR = PROJECT / "outputs" / "letter_argmax_traj_20261010"
GATED_DIR = PROJECT / "outputs" / "calib_gated_decode_20261013"
LBO_SUMMARY = PROJECT / "outputs" / "letter_bias_origin_20261012" / "summary.json"
MANIFEST = PROJECT / "datasets" / "sciqa_img_standard_v1" / "manifest.json"
OUT_DIR = PROJECT / "outputs" / "visual_angle_20261013"
EXPECTED_STEP0 = {"lavida": 1448, "mmada": 1064}
EXPECTED_STOCK = {"lavida": 1460, "mmada": 1059}
EXPECTED_COMMON_T_BA = {"lavida": -0.20, "mmada": 1.26}
NQ = 5


# ---------------------------------------------------------------- small helpers

def readout(h: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    s = torch.sqrt((h * h).mean(-1, keepdim=True) + hor.EPS)
    return (h / s * gamma) @ W.T


def first_order(h: torch.Tensor, v: torch.Tensor, gamma, W) -> torch.Tensor:
    if v.dim() == 1:
        v = v[None, :].expand_as(h)
    return (gamma * hor.rms_jy_batch(h, v)) @ W.T


def log_softmax_masked(x, valid):
    return torch.log_softmax(x.masked_fill(~valid, float("-inf")), dim=1)


def let_idx(s) -> int:
    s = (s or "").strip()
    return LETTERS.index(s) if s in LETTERS else -1


def hist(pred: np.ndarray) -> dict:
    c = Counter(int(p) for p in pred)
    out = {L: c.get(k, 0) for k, L in enumerate(LETTERS)}
    out["non_letter"] = c.get(-1, 0)
    return out


def frac(mask: np.ndarray, within: np.ndarray | None = None) -> float | None:
    if within is None:
        within = np.ones_like(mask, dtype=bool)
    d = int(within.sum())
    return float(mask[within].mean()) if d else None


def quintile_rank_bins(x: np.ndarray, groups: np.ndarray | None = None, nq: int = NQ) -> np.ndarray:
    """Equal-count bins by rank (0 = lowest). If groups is given, ranks are taken within each group."""
    bins = np.empty(len(x), dtype=int)
    if groups is None:
        groups = np.zeros(len(x), dtype=int)
    for g in np.unique(groups):
        sel = np.where(groups == g)[0]
        order = np.argsort(x[sel], kind="stable")
        rank = np.empty(len(sel), dtype=int)
        rank[order] = np.arange(len(sel))
        bins[sel] = (rank * nq) // len(sel)
    return bins


def mwu(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) == 0 or len(b) == 0:
        return {"n_a": int(len(a)), "n_b": int(len(b)), "p": None}
    r = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"n_a": int(len(a)), "n_b": int(len(b)), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "U": float(r.statistic), "p": float(r.pvalue),
            "auc_a_gt_b": float(r.statistic / (len(a) * len(b)))}


def spearman(x, y) -> dict:
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


# ---------------------------------------------------------------- loading

def load_cf(model: str, ids: list, n_opt: np.ndarray):
    rows = [json.loads(l) for l in (CF_DIR / f"cf_{model}.jsonl").open(encoding="utf-8")]
    glob = {r["n_opt"]: torch.tensor(r["letter_logit64"], dtype=torch.float64) for r in rows if r["cond"] == "a_global"}
    if sorted(glob) != [2, 3, 4, 5]:
        raise RuntimeError(f"{model} a_global n_opt {sorted(glob)}")
    per = {c: {} for c in ("b_imgfree", "c_txtfree")}
    for r in rows:
        if r["cond"] in per:
            per[r["cond"]][r["id"]] = r
    for c in per:
        if set(per[c]) != set(ids):
            raise RuntimeError(f"{model} {c}: {len(per[c])} records")
        bad = [i for k, i in enumerate(ids) if per[c][i]["n_opt"] != int(n_opt[k])]
        if bad:
            raise RuntimeError(f"{model} {c} n_opt mismatch {bad[:3]}")
    L_glob = torch.stack([glob[int(n)] for n in n_opt])
    L_img = torch.stack([torch.tensor(per["b_imgfree"][i]["letter_logit64"], dtype=torch.float64) for i in ids])
    L_txt = torch.stack([torch.tensor(per["c_txtfree"][i]["letter_logit64"], dtype=torch.float64) for i in ids])
    return glob, L_glob, L_img, L_txt


def load_final_letters(model: str, ids: list, gold_letters: list):
    stock = {}
    for l in (TRAJ_DIR / f"{model}.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        stock[r["id"]] = r
    gated = {}
    for l in (GATED_DIR / f"{model}.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        gated[r["id"]] = r
    if set(stock) != set(ids) or set(gated) != set(ids):
        raise RuntimeError(f"{model} final-letter files do not cover the 2017 ids ({len(stock)}, {len(gated)})")
    for k, i in enumerate(ids):
        if stock[i]["gold"] != gold_letters[k] or gated[i]["gold"] != gold_letters[k]:
            raise RuntimeError(f"{model} gold mismatch at {i}")
    stock_fin = np.array([let_idx(stock[i]["finished_letter"]) for i in ids])
    gated_fin = np.array([let_idx(gated[i]["finished_letter"]) for i in ids])
    gated_s0_raw = np.array([let_idx(gated[i].get("step0_restricted_raw", "")) for i in ids])
    gated_s0_cal = np.array([let_idx(gated[i].get("step0_restricted_cal", "")) for i in ids])
    unmask_stock = np.array([stock[i]["letter_unmask_step"] for i in ids])
    unmask_gated = np.array([gated[i]["letter_unmask_step"] for i in ids])
    return stock_fin, gated_fin, gated_s0_raw, gated_s0_cal, unmask_stock, unmask_gated


def load_options(ids: list, n_opt: np.ndarray, gold: np.ndarray):
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in man}
    if set(by_id) != set(ids):
        raise RuntimeError("manifest ids differ from hidden-state ids")
    choices = []
    for k, i in enumerate(ids):
        m = by_id[i]
        ch = m["choices"]
        if isinstance(ch, str):
            ch = ast.literal_eval(ch)
        if len(ch) != int(n_opt[k]):
            raise RuntimeError(f"{i}: {len(ch)} choices vs n_opt {n_opt[k]}")
        if LETTERS.index(m["gold_option"]) != int(gold[k]) or int(m["answer_index"]) != int(gold[k]):
            raise RuntimeError(f"{i}: manifest gold {m['gold_option']} vs {LETTERS[gold[k]]}")
        choices.append([str(c) for c in ch])
    return choices


# ---------------------------------------------------------------- per-group statistics

def group_stats(sel: np.ndarray, V: dict) -> dict:
    n = int(sel.sum())
    g = V["gold"][sel]
    p0, st, gt, pr = V["pred0"][sel], V["stock"][sel], V["gated"][sel], V["prior"][sel]
    gold_ne_prior = g != pr
    out = {
        "n": n,
        "acc_step0": frac(p0 == g), "acc_stock": frac(st == g), "acc_gated": frac(gt == g),
        "correct_step0": int((p0 == g).sum()), "correct_stock": int((st == g).sum()), "correct_gated": int((gt == g).sum()),
        "gated_minus_stock_correct": int((gt == g).sum() - (st == g).sum()),
        "gated_minus_stock_acc_pct": float(100 * ((gt == g).mean() - (st == g).mean())),
        "n_changed_stock_to_gated": int((gt != st).sum()),
        "n_fixed": int(((gt == g) & (st != g)).sum()), "n_broke": int(((gt != g) & (st == g)).sum()),
        "p_gold_eq_prior": frac(g == pr),
        "p_imgpref_eq_gold": frac(V["img_pref"][sel] == g),
        "p_txtpref_eq_gold": frac(V["txt_pref"][sel] == g),
        "median_s_I": float(np.median(V["s_I"][sel])), "median_s_T": float(np.median(V["s_T"][sel])),
        "median_img_spread": float(np.median(V["img_spread"][sel])),
        "median_step0_margin": float(np.median(V["margin0"][sel])),
    }
    n_sel = V["n_opt"][sel]
    for name, p in (("step0", p0), ("stock", st), ("gated", gt)):
        wrong = p != g
        out[f"p_pred_eq_prior_{name}"] = frac(p == pr)
        out[f"p_prior_wins_{name}"] = frac((p == pr) & (p != g))
        out[f"p_prior_wins_given_gold_ne_prior_{name}"] = frac(p == pr, gold_ne_prior)
        out[f"acc_given_gold_ne_prior_{name}"] = frac(p == g, gold_ne_prior)
        out[f"acc_given_gold_eq_prior_{name}"] = frac(p == g, ~gold_ne_prior)
        eq, ne = out[f"acc_given_gold_eq_prior_{name}"], out[f"acc_given_gold_ne_prior_{name}"]
        out[f"acc_gap_prior_{name}"] = (eq - ne) if (eq is not None and ne is not None) else None
        # share of errors that land on the prior letter, vs chance 1(gold!=prior)/(n-1) under uniform wrong picks
        share = frac(p == pr, wrong)
        chance = float((gold_ne_prior[wrong] / (n_sel[wrong] - 1)).mean()) if wrong.sum() else None
        out[f"err_prior_share_{name}"] = share
        out[f"err_prior_share_chance_{name}"] = chance
        out[f"err_prior_share_excess_{name}"] = (share - chance) if (share is not None and chance is not None) else None
    return out


def quintile_table(x: np.ndarray, V: dict, groups: np.ndarray | None = None) -> list:
    bins = quintile_rank_bins(x, groups)
    rows = []
    for q in range(NQ):
        sel = bins == q
        r = {"quintile": q + 1, "x_min": float(x[sel].min()), "x_max": float(x[sel].max()), **group_stats(sel, V)}
        rows.append(r)
    return rows


# ---------------------------------------------------------------- main per-model routine

def run_model(model: str, lbo: dict) -> tuple[dict, dict]:
    D = hor.load_model_data(model, "cpu")
    H, W, gamma, valid, gold_t, n_opt_t, ids = D["H"], D["W"], D["gamma"], D["valid"], D["gold"], D["n_opt"], D["ids"]
    N = H.shape[0]
    gold = gold_t.numpy()
    n_opt = n_opt_t.numpy()
    if (gold < 0).any():
        raise RuntimeError(f"{model}: gold letter outside option letters for {(gold < 0).sum()} items")
    gold_letters = [LETTERS[g] for g in gold]

    # ---- Shapley reconstruction and readout
    phi = hor.shapley_all(H)
    h_none, h_full = H[:, IDX["none"]], H[:, IDX["ITM"]]
    recon_err = float(torch.linalg.vector_norm(h_none + phi["I"] + phi["T"] + phi["M"] - h_full, dim=1).max())
    L_full = readout(h_full, gamma, W)
    pred0_t = hor.pick(L_full, valid)
    pred0 = pred0_t.numpy()
    step0_correct = int((pred0 == gold).sum())
    masked_full = hor.masked(L_full, valid)
    top2 = torch.topk(masked_full, 2, dim=1).values
    margin0 = (top2[:, 0] - top2[:, 1]).numpy()

    # common-component sanity (letter_bias_origin first-order push of mean phi_T, B minus A)
    common_push = {}
    for x in ("I", "T", "M"):
        P_com = first_order(h_full, phi[x].mean(0), gamma, W)
        common_push[x] = float((P_com[:, 1] - P_com[:, 0]).mean())
    lbo_T = lbo["analysis1"][model]["norms_and_fractions"]["T"]["first_order_push_centered"]["B_minus_A_common_mean"]
    sanity = {
        "shapley_reconstruction_err_max": recon_err,
        "step0_restricted_correct": step0_correct, "step0_expected": EXPECTED_STEP0[model],
        "step0_matches_expected": step0_correct == EXPECTED_STEP0[model],
        "common_component_first_order_push_B_minus_A": common_push,
        "common_T_B_minus_A_reference_letter_bias_origin": float(lbo_T),
        "common_T_matches_reference_1e-6": abs(common_push["T"] - lbo_T) < 1e-6,
        "common_T_matches_rounded_expected": round(common_push["T"], 2) == EXPECTED_COMMON_T_BA[model],
    }
    if step0_correct != EXPECTED_STEP0[model]:
        raise RuntimeError(f"{model} step-0 restricted {step0_correct} != {EXPECTED_STEP0[model]}")

    # ---- image / text evidence measures
    s_I = torch.linalg.vector_norm(phi["I"], dim=1).numpy()
    s_T = torch.linalg.vector_norm(phi["T"], dim=1).numpy()
    s_M = torch.linalg.vector_norm(phi["M"], dim=1).numpy()
    L_img_dir = readout(phi["I"], gamma, W)  # W gamma normalize(phi_I)
    L_txt_dir = readout(phi["T"], gamma, W)
    m_img = hor.masked(L_img_dir, valid)
    img_spread = (m_img.max(1).values - torch.where(valid, L_img_dir, torch.full_like(L_img_dir, float("inf"))).min(1).values).numpy()
    m_txt = hor.masked(L_txt_dir, valid)
    txt_spread = (m_txt.max(1).values - torch.where(valid, L_txt_dir, torch.full_like(L_txt_dir, float("inf"))).min(1).values).numpy()
    img_pref = hor.pick(L_img_dir, valid).numpy()
    txt_pref = hor.pick(L_txt_dir, valid).numpy()
    fo_I = first_order(h_full, phi["I"], gamma, W)
    fo_spread_I = (hor.masked(fo_I, valid).max(1).values
                   - torch.where(valid, fo_I, torch.full_like(fo_I, float("inf"))).min(1).values).numpy()
    ratio_IT = s_I / s_T

    # ---- content-free prior and ablation conditions
    glob, L_glob, L_imgfree, L_txtfree = load_cf(model, ids, n_opt)
    prior = hor.pick(L_glob, valid).numpy()
    logp_cf = log_softmax_masked(L_glob, valid)
    p_cf_by_n = {f"n{n}": {LETTERS[j]: float(v) for j, v in enumerate(torch.softmax(glob[n][:n], 0))} for n in (2, 3, 4, 5)}
    prior_argmax_by_n = {f"n{n}": LETTERS[int(torch.argmax(glob[n][:n]))] for n in (2, 3, 4, 5)}
    pred_imgfree = hor.pick(L_imgfree, valid).numpy()
    pred_txtfree = hor.pick(L_txtfree, valid).numpy()

    # ---- final letters
    stock, gated, gated_s0_raw, gated_s0_cal, unmask_stock, unmask_gated = load_final_letters(model, ids, gold_letters)
    sanity.update({
        "stock_finished_correct": int((stock == gold).sum()), "stock_expected": EXPECTED_STOCK[model],
        "gated_finished_correct": int((gated == gold).sum()),
        "gated_step0_restricted_raw_correct": int((gated_s0_raw == gold).sum()),
        "gated_step0_restricted_cal_correct": int((gated_s0_cal == gold).sum()),
        "n_gated_step0_raw_eq_offline_pred0": int((gated_s0_raw == pred0).sum()),
        "n_stock_changed_to_gated": int((stock != gated).sum()),
        "n_gated_non_letter": int((gated == -1).sum()), "n_stock_non_letter": int((stock == -1).sum()),
    })

    V = {"gold": gold, "pred0": pred0, "stock": stock, "gated": gated, "prior": prior, "img_pref": img_pref,
         "txt_pref": txt_pref, "s_I": s_I, "s_T": s_T, "img_spread": img_spread, "margin0": margin0, "n_opt": n_opt}

    # ================================================================ A
    A = {
        "definitions": {
            "s_I": "||phi_I||_2 of the exact Shapley image share at the answer position (step 0)",
            "img_spread": "max-min over the item's n options of W.gamma.normalize(phi_I) (letter readout of the image share)",
            "fo_spread_I": "same spread of the first-order push J(h_ITM) phi_I",
            "prior_argmax": "argmax over the first n letters of the a_global content-free logits",
            "pred_eq_prior": "restricted prediction == prior argmax", "prior_wins": "pred == prior argmax and pred != gold",
            "quintiles": "equal-count rank bins (1 = weakest image evidence); *_within_n: ranks taken inside each n_opt group",
        },
        "prior_argmax_by_n": prior_argmax_by_n, "p_cf_by_n": p_cf_by_n,
        "overall": group_stats(np.ones(N, dtype=bool), V),
        "quintiles_s_I": quintile_table(s_I, V),
        "quintiles_img_spread": quintile_table(img_spread, V),
        "quintiles_s_I_within_n": quintile_table(s_I, V, n_opt),
        "quintiles_img_spread_within_n": quintile_table(img_spread, V, n_opt),
        "quintiles_fo_spread_I": quintile_table(fo_spread_I, V),
        "quintiles_fo_spread_I_within_n": quintile_table(fo_spread_I, V, n_opt),
        "quintiles_margin0": quintile_table(margin0, V),
        "quintiles_margin0_within_n": quintile_table(margin0, V, n_opt),
        "quintiles_s_T": quintile_table(s_T, V),
        "quintiles_ratio_I_over_T": quintile_table(ratio_IT, V),
        "spearman": {
            "margin0_vs_pred0_eq_prior": spearman(margin0, (pred0 == prior).astype(float)),
            "fo_spread_I_vs_margin0": spearman(fo_spread_I, margin0),
            "img_spread_vs_margin0": spearman(img_spread, margin0),
            "s_I_vs_pred0_eq_prior": spearman(s_I, (pred0 == prior).astype(float)),
            "s_I_vs_prior_wins_step0": spearman(s_I, ((pred0 == prior) & (pred0 != gold)).astype(float)),
            "s_I_vs_correct_step0": spearman(s_I, (pred0 == gold).astype(float)),
            "s_I_vs_stock_eq_prior": spearman(s_I, (stock == prior).astype(float)),
            "s_I_vs_gated_eq_prior": spearman(s_I, (gated == prior).astype(float)),
            "img_spread_vs_pred0_eq_prior": spearman(img_spread, (pred0 == prior).astype(float)),
            "img_spread_vs_correct_step0": spearman(img_spread, (pred0 == gold).astype(float)),
            "fo_spread_I_vs_pred0_eq_prior": spearman(fo_spread_I, (pred0 == prior).astype(float)),
            "s_T_vs_pred0_eq_prior": spearman(s_T, (pred0 == prior).astype(float)),
            "s_I_vs_s_T": spearman(s_I, s_T), "s_I_vs_n_opt": spearman(s_I, n_opt), "s_I_vs_margin0": spearman(s_I, margin0),
            "img_spread_vs_n_opt": spearman(img_spread, n_opt),
            "s_I_vs_pred0_eq_prior_within_n": {f"n{n}": spearman(s_I[n_opt == n], (pred0 == prior)[n_opt == n].astype(float))
                                               for n in (2, 3, 4) if (n_opt == n).sum() > 10},
        },
        "medians": {"s_I": float(np.median(s_I)), "s_T": float(np.median(s_T)), "s_M": float(np.median(s_M)),
                    "img_spread": float(np.median(img_spread)), "txt_spread": float(np.median(txt_spread)),
                    "fo_spread_I": float(np.median(fo_spread_I)), "h_ITM_norm": float(torch.linalg.vector_norm(h_full, dim=1).median())},
    }

    # ================================================================ B
    real_correct = pred0 == gold
    imgfree_correct = pred_imgfree == gold
    txtfree_correct = pred_txtfree == gold
    prior_correct = prior == gold

    def by_n(fn):
        return {f"n{n}": fn(n_opt == n) for n in (2, 3, 4, 5)}

    B = {
        "imgfree": {
            "correct": int(imgfree_correct.sum()), "acc": frac(imgfree_correct),
            "acc_by_n": by_n(lambda s: {"n": int(s.sum()), "acc": frac(imgfree_correct, s), "correct": int(imgfree_correct[s].sum())}),
            "p_imgfree_pred_eq_real_pred0": frac(pred_imgfree == pred0),
            "p_imgfree_pred_eq_real_pred0_by_n": by_n(lambda s: frac(pred_imgfree == pred0, s)),
            "p_imgfree_pred_eq_stock_final": frac(pred_imgfree == stock),
            "among_real_correct_frac_imgfree_correct": frac(imgfree_correct, real_correct),
            "among_real_correct_n": int(real_correct.sum()), "among_real_correct_n_imgfree_correct": int((imgfree_correct & real_correct).sum()),
            "among_real_wrong_frac_imgfree_correct": frac(imgfree_correct, ~real_correct),
            "among_imgfree_correct_frac_eq_prior_argmax": frac(pred_imgfree == prior, imgfree_correct),
            "among_imgfree_correct_n": int(imgfree_correct.sum()), "among_imgfree_correct_n_eq_prior": int((imgfree_correct & (pred_imgfree == prior)).sum()),
            "among_imgfree_correct_frac_eq_prior_by_n": by_n(lambda s: frac(pred_imgfree == prior, imgfree_correct & s)),
            "p_imgfree_pred_eq_prior_argmax": frac(pred_imgfree == prior),
            "p_imgfree_pred_eq_prior_argmax_by_n": by_n(lambda s: frac(pred_imgfree == prior, s)),
            "acc_given_gold_ne_prior": frac(imgfree_correct, gold != prior),
            "acc_given_gold_eq_prior": frac(imgfree_correct, gold == prior),
            "letter_hist": hist(pred_imgfree),
            "letter_hist_by_n": by_n(lambda s: hist(pred_imgfree[s])),
        },
        "txtfree": {
            "correct": int(txtfree_correct.sum()), "acc": frac(txtfree_correct),
            "acc_by_n": by_n(lambda s: {"n": int(s.sum()), "acc": frac(txtfree_correct, s)}),
            "p_txtfree_pred_eq_real_pred0": frac(pred_txtfree == pred0),
            "p_txtfree_pred_eq_prior_argmax": frac(pred_txtfree == prior),
            "p_txtfree_pred_eq_prior_argmax_by_n": by_n(lambda s: frac(pred_txtfree == prior, s)),
            "letter_hist": hist(pred_txtfree), "letter_hist_by_n": by_n(lambda s: hist(pred_txtfree[s])),
            "mean_restricted_entropy_bits": float((-(log_softmax_masked(L_txtfree, valid).exp() * log_softmax_masked(L_txtfree, valid)).nan_to_num(0).sum(1) / np.log(2)).mean()),
        },
        "reference": {
            "real_step0": {"correct": step0_correct, "acc": frac(real_correct), "letter_hist": hist(pred0),
                           "acc_by_n": by_n(lambda s: frac(real_correct, s))},
            "prior_argmax_only": {"correct": int(prior_correct.sum()), "acc": frac(prior_correct), "letter_hist": hist(prior),
                                  "acc_by_n": by_n(lambda s: frac(prior_correct, s))},
            "gold": {"letter_hist": hist(gold), "letter_hist_by_n": by_n(lambda s: hist(gold[s]))},
            "chance_acc_mean_1_over_n": float(np.mean(1.0 / n_opt)),
            "mean_restricted_entropy_bits_real": float((-(log_softmax_masked(L_full, valid).exp() * log_softmax_masked(L_full, valid)).nan_to_num(0).sum(1) / np.log(2)).mean()),
            "mean_restricted_entropy_bits_imgfree": float((-(log_softmax_masked(L_imgfree, valid).exp() * log_softmax_masked(L_imgfree, valid)).nan_to_num(0).sum(1) / np.log(2)).mean()),
        },
        "three_way": {
            "real_correct_and_imgfree_correct": int((real_correct & imgfree_correct).sum()),
            "real_correct_imgfree_wrong": int((real_correct & ~imgfree_correct).sum()),
            "real_wrong_imgfree_correct": int((~real_correct & imgfree_correct).sum()),
            "both_wrong": int((~real_correct & ~imgfree_correct).sum()),
            "real_wrong_imgfree_wrong_same_pred": int((~real_correct & ~imgfree_correct & (pred0 == pred_imgfree)).sum()),
        },
        "image_dependence_vs_s_I": {
            "note": "items answerable without the image (imgfree correct) vs not, by s_I quintile",
            "frac_imgfree_correct_by_s_I_quintile": [frac(imgfree_correct, quintile_rank_bins(s_I) == q) for q in range(NQ)],
            "frac_imgfree_eq_real_by_s_I_quintile": [frac(pred_imgfree == pred0, quintile_rank_bins(s_I) == q) for q in range(NQ)],
            "mwu_s_I_imgfree_correct_vs_wrong": mwu(s_I[imgfree_correct], s_I[~imgfree_correct]),
            "mwu_s_I_realcorrect_imgfreewrong_vs_realcorrect_imgfreecorrect": mwu(s_I[real_correct & ~imgfree_correct], s_I[real_correct & imgfree_correct]),
        },
    }

    # ================================================================ C
    changed = stock != gated
    fixed = changed & (gated == gold) & (stock != gold)
    broke = changed & (gated != gold) & (stock == gold)
    neither = changed & ~fixed & ~broke
    bins_sI = quintile_rank_bins(s_I)
    bins_sp = quintile_rank_bins(img_spread)
    C = {
        "n_changed": int(changed.sum()), "n_fixed": int(fixed.sum()), "n_broke": int(broke.sum()), "n_changed_both_wrong": int(neither.sum()),
        "changed_transitions": dict(Counter(f"{LETTERS[a] if a >= 0 else '?'}->{LETTERS[b] if b >= 0 else '?'}"
                                            for a, b in zip(stock[changed], gated[changed])).most_common()),
        "changed_vs_unchanged": {
            "s_I": mwu(s_I[changed], s_I[~changed]), "img_spread": mwu(img_spread[changed], img_spread[~changed]),
            "fo_spread_I": mwu(fo_spread_I[changed], fo_spread_I[~changed]),
            "s_T": mwu(s_T[changed], s_T[~changed]), "ratio_I_over_T": mwu(ratio_IT[changed], ratio_IT[~changed]),
            "step0_margin": mwu(margin0[changed], margin0[~changed]),
            "note": "a = changed, b = unchanged; auc = P(a > b)",
        },
        "fixed_vs_broke": {
            "s_I": mwu(s_I[fixed], s_I[broke]), "img_spread": mwu(img_spread[fixed], img_spread[broke]),
            "fo_spread_I": mwu(fo_spread_I[fixed], fo_spread_I[broke]),
            "s_T": mwu(s_T[fixed], s_T[broke]), "ratio_I_over_T": mwu(ratio_IT[fixed], ratio_IT[broke]),
            "step0_margin": mwu(margin0[fixed], margin0[broke]),
            "note": "a = fixed (wrong->right), b = broke (right->wrong)",
        },
        "fixed_vs_unchanged_s_I": mwu(s_I[fixed], s_I[~changed]),
        "broke_vs_unchanged_s_I": mwu(s_I[broke], s_I[~changed]),
        "changed_moves_toward_prior": int((changed & (gated == prior)).sum()), "changed_moves_away_from_prior": int((changed & (stock == prior)).sum()),
        "frac_changed_by_s_I_quintile": [frac(changed, bins_sI == q) for q in range(NQ)],
        "n_changed_by_s_I_quintile": [int((changed & (bins_sI == q)).sum()) for q in range(NQ)],
        "n_fixed_by_s_I_quintile": [int((fixed & (bins_sI == q)).sum()) for q in range(NQ)],
        "n_broke_by_s_I_quintile": [int((broke & (bins_sI == q)).sum()) for q in range(NQ)],
        "frac_changed_by_img_spread_quintile": [frac(changed, bins_sp == q) for q in range(NQ)],
        "n_fixed_by_img_spread_quintile": [int((fixed & (bins_sp == q)).sum()) for q in range(NQ)],
        "n_broke_by_img_spread_quintile": [int((broke & (bins_sp == q)).sum()) for q in range(NQ)],
        "frac_changed_by_s_I_quintile_within_n": [frac(changed, quintile_rank_bins(s_I, n_opt) == q) for q in range(NQ)],
        "unmask_step_hist_stock": {str(k): int(v) for k, v in sorted(Counter(unmask_stock.tolist()).items())},
        "unmask_step_hist_gated": {str(k): int(v) for k, v in sorted(Counter(unmask_gated.tolist()).items())},
        "n_unmask_step_changed": int((unmask_stock != unmask_gated).sum()),
        "image_pref_among_changed": {
            "p_gated_eq_imgpref": frac(gated == img_pref, changed), "p_stock_eq_imgpref": frac(stock == img_pref, changed),
            "p_gated_eq_imgpref_fixed": frac(gated == img_pref, fixed), "p_stock_eq_imgpref_broke": frac(stock == img_pref, broke),
            "p_gated_eq_imgpref_unchanged": frac(gated == img_pref, ~changed),
        },
    }

    # ================================================================ D
    D_out = {"by_n_opt": {}, "by_gold": {}, "by_n_opt_x_gold": {}}
    for n in (2, 3, 4, 5):
        D_out["by_n_opt"][f"n{n}"] = {"prior_argmax": prior_argmax_by_n[f"n{n}"], "chance": 1.0 / n, **group_stats(n_opt == n, V)}
        for g in range(n):
            sel = (n_opt == n) & (gold == g)
            if sel.sum():
                D_out["by_n_opt_x_gold"][f"n{n}_gold{LETTERS[g]}"] = group_stats(sel, V)
    for g in range(5):
        sel = gold == g
        if sel.sum():
            D_out["by_gold"][LETTERS[g]] = group_stats(sel, V)
    # prior agreement among gold!=prior, by s_I quintile within n (the cleanest 'takeover' measure)
    bins_w = quintile_rank_bins(s_I, n_opt)
    D_out["takeover_rate_gold_ne_prior_by_s_I_quintile_within_n"] = {
        name: [frac(p == prior, (bins_w == q) & (gold != prior)) for q in range(NQ)]
        for name, p in (("step0", pred0), ("stock", stock), ("gated", gated))}
    D_out["acc_gold_ne_prior_by_s_I_quintile_within_n"] = {
        name: [frac(p == gold, (bins_w == q) & (gold != prior)) for q in range(NQ)]
        for name, p in (("step0", pred0), ("stock", stock), ("gated", gated))}
    D_out["n_gold_ne_prior_by_s_I_quintile_within_n"] = [int(((bins_w == q) & (gold != prior)).sum()) for q in range(NQ)]
    D_out["s_I_by_n_opt_median"] = {f"n{n}": float(np.median(s_I[n_opt == n])) for n in (2, 3, 4, 5)}
    D_out["img_spread_by_n_opt_median"] = {f"n{n}": float(np.median(img_spread[n_opt == n])) for n in (2, 3, 4, 5)}

    # ================================================================ E
    choices = load_options(ids, n_opt, gold)
    lens = [np.array([len(c) for c in ch]) for ch in choices]
    wlens = [np.array([len(c.split()) for c in ch]) for ch in choices]

    def is_longest(pred, L=lens):
        return np.array([p >= 0 and L[k][p] == L[k].max() for k, p in enumerate(pred)])

    def is_shortest(pred, L=lens):
        return np.array([p >= 0 and L[k][p] == L[k].min() for k, p in enumerate(pred)])

    def rank_desc(pred, L=lens):
        return np.array([1 + int((L[k] > L[k][p]).sum()) if p >= 0 else -1 for k, p in enumerate(pred)])

    chance_longest = float(np.mean([(l == l.max()).sum() / len(l) for l in lens]))
    unique_longest = np.array([(l == l.max()).sum() == 1 for l in lens])
    gold_longest = is_longest(gold)
    E = {
        "length_unit": "characters (word-count variant in *_words)",
        "n_items_unique_longest": int(unique_longest.sum()),
        "chance_p_longest_uniform_choice": chance_longest,
        "p_gold_is_longest": frac(gold_longest), "p_gold_is_shortest": frac(is_shortest(gold)),
        "p_gold_is_longest_by_n": by_n(lambda s: frac(gold_longest, s)),
        "p_prior_argmax_option_is_longest": frac(is_longest(prior)),
        "p_prior_argmax_option_is_longest_by_n": by_n(lambda s: frac(is_longest(prior), s)),
        "p_prior_argmax_option_is_shortest": frac(is_shortest(prior)),
        "p_letter_position_is_longest": {LETTERS[g]: frac(np.array([len(l) > g and l[g] == l.max() for l in lens]), n_opt > g) for g in range(5)},
        "p_letter_position_is_longest_by_n": by_n(lambda s: {LETTERS[g]: frac(np.array([len(l) > g and l[g] == l.max() for l in lens]), s) for g in range(5) if (s & (n_opt > g)).sum()}),
        "mean_option_length_by_position": by_n(lambda s: {LETTERS[g]: float(np.mean([l[g] for l, m in zip(lens, s) if m])) for g in range(5) if (s & (n_opt > g)).sum()}),
        "chosen": {},
        "spearman_rank_corr_option_length_vs_prior_logit_within_item": None,
    }
    # within-item correlation between option length and content-free prior logit (pooled over items)
    xs, ys = [], []
    for k in range(N):
        n = int(n_opt[k])
        lp = logp_cf[k, :n].numpy()
        xs.extend((lens[k] - lens[k].mean()).tolist())
        ys.extend((lp - lp.mean()).tolist())
    E["spearman_rank_corr_option_length_vs_prior_logit_within_item"] = spearman(np.array(xs), np.array(ys))
    for name, p in (("step0", pred0), ("stock", stock), ("gated", gated), ("imgfree", pred_imgfree), ("txtfree", pred_txtfree), ("prior_argmax", prior)):
        lo, sh, rk = is_longest(p), is_shortest(p), rank_desc(p)
        wrong = p != gold
        E["chosen"][name] = {
            "p_chosen_is_longest": frac(lo), "p_chosen_is_shortest": frac(sh),
            "p_chosen_is_longest_words": frac(is_longest(p, wlens)),
            "p_chosen_is_longest_given_gold_not_longest": frac(lo, ~gold_longest),
            "p_chosen_is_longest_among_wrong": frac(lo, wrong), "p_chosen_is_shortest_among_wrong": frac(sh, wrong),
            "p_chosen_is_longest_by_n": by_n(lambda s, lo=lo: frac(lo, s)),
            "p_chosen_is_longest_n2_longer_option": frac(lo, (n_opt == 2) & unique_longest),
            "rank_hist_1_is_longest": {str(r): int((rk == r).sum()) for r in range(1, 6)},
            "rank_hist_non_letter": int((rk == -1).sum()),
            "mean_rank": float(rk[rk > 0].mean()),
        }
    E["gold_rank_hist_1_is_longest"] = {str(r): int((rank_desc(gold) == r).sum()) for r in range(1, 6)}
    E["uniform_choice_mean_rank"] = float(np.mean([np.mean([1 + (l > v).sum() for v in l]) for l in lens]))

    plots = {"s_I": s_I, "img_spread": img_spread, "changed": changed, "fixed": fixed, "broke": broke,
             "pred0": pred0, "stock": stock, "gated": gated, "prior": prior, "gold": gold, "n_opt": n_opt,
             "pred_imgfree": pred_imgfree, "pred_txtfree": pred_txtfree}
    return {"sanity": sanity, "A": A, "B": B, "C": C, "D": D_out, "E": E}, plots


# ---------------------------------------------------------------- figures

def fig_A(model: str, A: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, key, xl in ((axes[0], "quintiles_s_I", "quintile of s_I = ||phi_I|| (1 = weakest image share)"),
                        (axes[1], "quintiles_img_spread", "quintile of image-share letter spread (1 = flattest)")):
        rows = A[key]
        q = [r["quintile"] for r in rows]
        ax.plot(q, [100 * r["acc_step0"] for r in rows], "o-", color="k", label="acc step-0")
        ax.plot(q, [100 * r["acc_stock"] for r in rows], "s--", color="gray", label="acc stock final")
        ax.plot(q, [100 * r["acc_gated"] for r in rows], "^-", color="C2", label="acc gated-calib final")
        ax.plot(q, [100 * r["p_pred_eq_prior_step0"] for r in rows], "o-", color="C3", label="P(pred = prior argmax) step-0")
        ax.plot(q, [100 * r["p_prior_wins_step0"] for r in rows], "o:", color="C1", label="P(prior wins) step-0")
        ax.plot(q, [100 * r["p_prior_wins_stock"] for r in rows], "s:", color="C1", alpha=0.6, label="P(prior wins) stock final")
        ax.plot(q, [100 * r["p_gold_eq_prior"] for r in rows], "x--", color="C7", label="P(gold = prior argmax)")
        ax.bar(q, [r["gated_minus_stock_acc_pct"] for r in rows], width=0.5, color="C2", alpha=0.25, label="gated - stock (acc pts)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(q)
        ax.set_xlabel(xl)
        ax.set_ylabel("%")
        ax.set_title(f"{model}: image evidence vs prior takeover")
        ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_A_supp(model: str, A: dict, path: Path):
    """Supplementary: (1) first-order image-push spread quintiles, (2) prior-conditional accuracy gap by within-n
    s_I quintile, (3) gated-stock gain by s_I / img_spread / fo_spread / margin quintiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    rows = A["quintiles_fo_spread_I"]
    q = [r["quintile"] for r in rows]
    ax = axes[0]
    ax.plot(q, [100 * r["acc_step0"] for r in rows], "o-", color="k", label="acc step-0")
    ax.plot(q, [100 * r["acc_gated"] for r in rows], "^-", color="C2", label="acc gated final")
    ax.plot(q, [100 * r["p_pred_eq_prior_step0"] for r in rows], "o-", color="C3", label="P(pred = prior) step-0")
    ax.plot(q, [100 * r["p_prior_wins_given_gold_ne_prior_step0"] for r in rows], "s-", color="C1", label="P(pred = prior | gold != prior) step-0")
    ax.plot(q, [100 * r["err_prior_share_step0"] for r in rows], "d-", color="C4", label="prior share of errors step-0")
    ax.plot(q, [100 * r["err_prior_share_chance_step0"] for r in rows], "d:", color="C4", alpha=0.6, label="  chance 1/(n-1)")
    ax.plot(q, [100 * r["p_gold_eq_prior"] for r in rows], "x--", color="C7", label="P(gold = prior)")
    ax.bar(q, [r["gated_minus_stock_acc_pct"] for r in rows], width=0.5, color="C2", alpha=0.25, label="gated - stock (acc pts)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(q)
    ax.set_xlabel("quintile of first-order image push spread J(h)phi_I (1 = flattest)")
    ax.set_ylabel("%")
    ax.set_title(f"{model}: image push in logit space")
    ax.legend(fontsize=7)
    ax = axes[1]
    for key, lab, mk in (("quintiles_s_I_within_n", "s_I (within n)", "o-"), ("quintiles_img_spread_within_n", "img spread (within n)", "s--"),
                         ("quintiles_margin0_within_n", "step-0 margin (within n)", "d:")):
        rows = A[key]
        for name, col in (("stock", "k"), ("gated", "C2")):
            ax.plot(q, [100 * (r[f"acc_gap_prior_{name}"] or 0) for r in rows], mk, color=col,
                    alpha=0.9 if key.startswith("quintiles_s_I") else 0.5, label=f"{name} final, by {lab}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(q)
    ax.set_xlabel("quintile (1 = weakest / smallest)")
    ax.set_ylabel("acc(gold = prior) - acc(gold != prior), pts")
    ax.set_title(f"{model}: prior-conditional accuracy gap")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[2]
    w = 0.2
    for j, (key, lab) in enumerate((("quintiles_s_I_within_n", "s_I (within n)"), ("quintiles_img_spread_within_n", "img spread (within n)"),
                                    ("quintiles_fo_spread_I", "first-order image push spread"), ("quintiles_margin0", "step-0 margin"))):
        rows = A[key]
        ax.bar(np.array(q) + (j - 1.5) * w, [r["gated_minus_stock_correct"] for r in rows], w, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(q)
    ax.set_xlabel("quintile (1 = weakest / smallest)")
    ax.set_ylabel("gated - stock, correct items")
    ax.set_title(f"{model}: where the gated calibration gains")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_B(model: str, B: dict, P: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(5)
    series = [("gold", B["reference"]["gold"]["letter_hist"]), ("p_cf argmax", B["reference"]["prior_argmax_only"]["letter_hist"]),
              ("real step-0 (ITM)", B["reference"]["real_step0"]["letter_hist"]), ("gray image (b_imgfree)", B["imgfree"]["letter_hist"]),
              ("N/A text (c_txtfree)", B["txtfree"]["letter_hist"])]
    w = 0.8 / len(series)
    for j, (lb, h) in enumerate(series):
        axes[0].bar(x + (j - (len(series) - 1) / 2) * w, [h[L] for L in LETTERS], w, label=lb)
    axes[0].set_xticks(x, list(LETTERS))
    axes[0].set_ylabel("items")
    axes[0].set_title(f"{model}: step-0 restricted letter histogram")
    axes[0].legend(fontsize=8)
    ns = ["n2", "n3", "n4", "n5"]
    xn = np.arange(4)
    acc = [("real step-0", [B["reference"]["real_step0"]["acc_by_n"][k] for k in ns]),
           ("gray image", [B["imgfree"]["acc_by_n"][k]["acc"] for k in ns]),
           ("N/A text", [B["txtfree"]["acc_by_n"][k]["acc"] for k in ns]),
           ("p_cf argmax only", [B["reference"]["prior_argmax_only"]["acc_by_n"][k] for k in ns])]
    w = 0.8 / len(acc)
    for j, (lb, v) in enumerate(acc):
        axes[1].bar(xn + (j - (len(acc) - 1) / 2) * w, [100 * a for a in v], w, label=lb)
    axes[1].plot(xn, [100 / int(k[1]) for k in ns], "k--", lw=0.8, label="chance 1/n")
    axes[1].set_xticks(xn, [f"{k} (N={B['imgfree']['acc_by_n'][k]['n']})" for k in ns])
    axes[1].set_ylabel("accuracy %")
    axes[1].set_title(f"{model}: accuracy by option count")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_C(model: str, C: dict, P: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    s_I, ch, fx, br = P["s_I"], P["changed"], P["fixed"], P["broke"]
    data = [np.log10(s_I[~ch]), np.log10(s_I[ch]), np.log10(s_I[fx]), np.log10(s_I[br])]
    labels = [f"unchanged\n(n={int((~ch).sum())})", f"changed\n(n={int(ch.sum())})", f"fixed\n(n={int(fx.sum())})", f"broke\n(n={int(br.sum())})"]
    axes[0].boxplot(data, tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("log10 s_I = log10 ||phi_I||")
    p1 = C["changed_vs_unchanged"]["s_I"]["p"]
    p2 = C["fixed_vs_broke"]["s_I"]["p"]
    axes[0].set_title(f"{model}: image share, stock->gated final letter\nMWU changed vs unchanged p={p1:.2g}; fixed vs broke p={p2:.2g}")
    q = np.arange(1, NQ + 1)
    axes[1].bar(q - 0.2, C["n_fixed_by_s_I_quintile"], 0.4, color="C2", label="fixed (wrong->right)")
    axes[1].bar(q + 0.2, C["n_broke_by_s_I_quintile"], 0.4, color="C3", label="broke (right->wrong)")
    ax2 = axes[1].twinx()
    ax2.plot(q, [100 * f for f in C["frac_changed_by_s_I_quintile"]], "ko-", label="% items changed")
    ax2.set_ylabel("% of quintile changed")
    axes[1].set_xticks(q)
    axes[1].set_xlabel("quintile of s_I (1 = weakest image share)")
    axes[1].set_ylabel("items")
    axes[1].set_title(f"{model}: where the gated calibration changes the final letter")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- tables

def pct(v):
    return "-" if v is None else f"{100 * v:.1f}"


def pts(v):
    return "-" if v is None else f"{100 * v:+.1f}"


def tables_md(S: dict) -> str:
    L = ["# visual_angle_20261013 tables", "",
         "All numbers: ScienceQA-IMG test, 2017 items. Sources: outputs/hidden_shapley_split_20261010 (step-0 hidden states, Shapley shares), "
         "outputs/calib_vote_20261012 (content-free prior and gray-image / N/A-text conditions), outputs/letter_argmax_traj_20261010 (stock final letter), "
         "outputs/calib_gated_decode_20261013 (gated-calibration final letter), datasets/sciqa_img_standard_v1/manifest.json (option texts).", ""]
    L += ["## Sanity", "", "| model | recon err | step-0 restricted | expected | common-T push B-A (first order) | ref (letter_bias_origin) | stock final | gated final | gated step-0 raw/cal |", "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        s = S[m]["sanity"]
        L.append(f"| {m} | {s['shapley_reconstruction_err_max']:.1e} | {s['step0_restricted_correct']} | {s['step0_expected']} | "
                 f"{s['common_component_first_order_push_B_minus_A']['T']:+.3f} | {s['common_T_B_minus_A_reference_letter_bias_origin']:+.3f} | "
                 f"{s['stock_finished_correct']} | {s['gated_finished_correct']} | {s['gated_step0_restricted_raw_correct']}/{s['gated_step0_restricted_cal_correct']} |")
    L.append("")
    for m in MODELS:
        A = S[m]["A"]
        L += [f"## A. {m}: image evidence vs prior takeover", "",
              f"prior argmax by n: {A['prior_argmax_by_n']}; Spearman(s_I, pred0==prior) rho={A['spearman']['s_I_vs_pred0_eq_prior']['rho']:+.3f} "
              f"(p={A['spearman']['s_I_vs_pred0_eq_prior']['p']:.2g}); Spearman(s_I, correct0) rho={A['spearman']['s_I_vs_correct_step0']['rho']:+.3f}; "
              f"Spearman(img_spread, pred0==prior) rho={A['spearman']['img_spread_vs_pred0_eq_prior']['rho']:+.3f}; Spearman(s_I, n_opt) rho={A['spearman']['s_I_vs_n_opt']['rho']:+.3f}", ""]
        for key, title in (("quintiles_s_I", "quintiles of s_I = ||phi_I||"), ("quintiles_img_spread", "quintiles of image-share letter spread W.gamma.normalize(phi_I)"),
                           ("quintiles_s_I_within_n", "quintiles of s_I taken within each n_opt group"),
                           ("quintiles_img_spread_within_n", "quintiles of image-share letter spread within each n_opt group"),
                           ("quintiles_fo_spread_I", "quintiles of first-order image push spread J(h_ITM) phi_I"),
                           ("quintiles_margin0", "quintiles of step-0 restricted top-2 margin (control variable, not an image measure)")):
            L += [f"### {m}: {title}", "",
                  "| Q | range | n | acc0 | acc stock | acc gated | gated-stock | P(pred0=prior) | P(prior wins)0 | P(prior wins given gold!=prior)0 | P(stock=prior) | P(prior wins) stock | P(gated=prior) | P(prior wins) gated | P(gold=prior) | gap0 | gap stock | gap gated | err-share0 (chance) | err-share gated | P(imgpref=gold) | med margin0 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in A[key]:
                L.append(f"| {r['quintile']} | {r['x_min']:.3g}-{r['x_max']:.3g} | {r['n']} | {pct(r['acc_step0'])} | {pct(r['acc_stock'])} | {pct(r['acc_gated'])} | "
                         f"{r['gated_minus_stock_correct']:+d} ({r['gated_minus_stock_acc_pct']:+.1f}) | {pct(r['p_pred_eq_prior_step0'])} | {pct(r['p_prior_wins_step0'])} | "
                         f"{pct(r['p_prior_wins_given_gold_ne_prior_step0'])} | {pct(r['p_pred_eq_prior_stock'])} | {pct(r['p_prior_wins_stock'])} | {pct(r['p_pred_eq_prior_gated'])} | "
                         f"{pct(r['p_prior_wins_gated'])} | {pct(r['p_gold_eq_prior'])} | {pts(r['acc_gap_prior_step0'])} | {pts(r['acc_gap_prior_stock'])} | {pts(r['acc_gap_prior_gated'])} | "
                         f"{pct(r['err_prior_share_step0'])} ({pct(r['err_prior_share_chance_step0'])}) | {pct(r['err_prior_share_gated'])} | {pct(r['p_imgpref_eq_gold'])} | {r['median_step0_margin']:.2f} |")
            L.append("")
        L += [f"{m} overall: acc gap (gold=prior minus gold!=prior) step0 {pts(A['overall']['acc_gap_prior_step0'])} / stock {pts(A['overall']['acc_gap_prior_stock'])} / gated {pts(A['overall']['acc_gap_prior_gated'])} pts; "
              f"prior share of errors step0 {pct(A['overall']['err_prior_share_step0'])} (chance {pct(A['overall']['err_prior_share_chance_step0'])}) / stock {pct(A['overall']['err_prior_share_stock'])} / gated {pct(A['overall']['err_prior_share_gated'])}; "
              f"Spearman(margin0, pred0==prior) rho={A['spearman']['margin0_vs_pred0_eq_prior']['rho']:+.3f}; Spearman(fo_spread_I, margin0) rho={A['spearman']['fo_spread_I_vs_margin0']['rho']:+.3f}; "
              f"Spearman(img_spread, margin0) rho={A['spearman']['img_spread_vs_margin0']['rho']:+.3f}", ""]
    L += ["## B. Answering without the image (gray image, real text) and without the text (real image, N/A text)", "",
          "| model | real step-0 acc | imgfree acc | txtfree acc | p_cf-argmax-only acc | chance | P(imgfree=real) | among real-correct: imgfree-correct | among imgfree-correct: =prior argmax | P(imgfree=prior) | P(txtfree=prior) | imgfree acc given gold!=prior | imgfree acc given gold=prior |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        B = S[m]["B"]
        L.append(f"| {m} | {pct(B['reference']['real_step0']['acc'])} | {pct(B['imgfree']['acc'])} | {pct(B['txtfree']['acc'])} | {pct(B['reference']['prior_argmax_only']['acc'])} | "
                 f"{pct(B['reference']['chance_acc_mean_1_over_n'])} | {pct(B['imgfree']['p_imgfree_pred_eq_real_pred0'])} | {pct(B['imgfree']['among_real_correct_frac_imgfree_correct'])} "
                 f"({B['imgfree']['among_real_correct_n_imgfree_correct']}/{B['imgfree']['among_real_correct_n']}) | {pct(B['imgfree']['among_imgfree_correct_frac_eq_prior_argmax'])} "
                 f"({B['imgfree']['among_imgfree_correct_n_eq_prior']}/{B['imgfree']['among_imgfree_correct_n']}) | {pct(B['imgfree']['p_imgfree_pred_eq_prior_argmax'])} | "
                 f"{pct(B['txtfree']['p_txtfree_pred_eq_prior_argmax'])} | {pct(B['imgfree']['acc_given_gold_ne_prior'])} | {pct(B['imgfree']['acc_given_gold_eq_prior'])} |")
    L += ["", "| model | condition | A | B | C | D | E | acc n2 | acc n3 | acc n4 | acc n5 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        B = S[m]["B"]
        rows = [("gold", B["reference"]["gold"]["letter_hist"], None), ("p_cf argmax", B["reference"]["prior_argmax_only"]["letter_hist"], B["reference"]["prior_argmax_only"]["acc_by_n"]),
                ("real step-0", B["reference"]["real_step0"]["letter_hist"], B["reference"]["real_step0"]["acc_by_n"]),
                ("b_imgfree", B["imgfree"]["letter_hist"], {k: v["acc"] for k, v in B["imgfree"]["acc_by_n"].items()}),
                ("c_txtfree", B["txtfree"]["letter_hist"], {k: v["acc"] for k, v in B["txtfree"]["acc_by_n"].items()})]
        for name, h, acc in rows:
            a = " | ".join(pct(acc[k]) if acc else "-" for k in ("n2", "n3", "n4", "n5"))
            L.append(f"| {m} | {name} | {h['A']} | {h['B']} | {h['C']} | {h['D']} | {h['E']} | {a} |")
    L += ["", "## C. Items whose final letter changed between stock and gated decode", "",
          "| model | changed | fixed | broke | both wrong | med s_I changed / unchanged (MWU p) | med img_spread changed / unchanged (p) | med s_I fixed / broke (p) | med img_spread fixed / broke (p) | med margin0 changed / unchanged (p) | % changed by s_I quintile Q1..Q5 | fixed by Q | broke by Q |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        C = S[m]["C"]
        cu, fb = C["changed_vs_unchanged"], C["fixed_vs_broke"]
        L.append(f"| {m} | {C['n_changed']} | {C['n_fixed']} | {C['n_broke']} | {C['n_changed_both_wrong']} | "
                 f"{cu['s_I']['median_a']:.0f} / {cu['s_I']['median_b']:.0f} ({cu['s_I']['p']:.2g}) | {cu['img_spread']['median_a']:.2f} / {cu['img_spread']['median_b']:.2f} ({cu['img_spread']['p']:.2g}) | "
                 f"{fb['s_I']['median_a']:.0f} / {fb['s_I']['median_b']:.0f} ({fb['s_I']['p']:.2g}) | {fb['img_spread']['median_a']:.2f} / {fb['img_spread']['median_b']:.2f} ({fb['img_spread']['p']:.2g}) | "
                 f"{cu['step0_margin']['median_a']:.2f} / {cu['step0_margin']['median_b']:.2f} ({cu['step0_margin']['p']:.2g}) | "
                 f"{' / '.join(pct(f) for f in C['frac_changed_by_s_I_quintile'])} | {C['n_fixed_by_s_I_quintile']} | {C['n_broke_by_s_I_quintile']} |")
    L += ["", "## D. Gold-letter confound check", "",
          "| model | group | n | prior argmax | acc0 | acc stock | acc gated | P(pred0=prior) | P(prior wins)0 | P(prior wins given gold!=prior)0 | P(gold=prior) | P(stock=prior) | P(gated=prior) | gap0 | gap stock | gap gated | err-share0 (chance) | err-share gated | med s_I |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def drow(m, k, r, pa):
        return (f"| {m} | {k} | {r['n']} | {pa} | {pct(r['acc_step0'])} | {pct(r['acc_stock'])} | {pct(r['acc_gated'])} | {pct(r['p_pred_eq_prior_step0'])} | "
                f"{pct(r['p_prior_wins_step0'])} | {pct(r['p_prior_wins_given_gold_ne_prior_step0'])} | {pct(r['p_gold_eq_prior'])} | {pct(r['p_pred_eq_prior_stock'])} | {pct(r['p_pred_eq_prior_gated'])} | "
                f"{pts(r['acc_gap_prior_step0'])} | {pts(r['acc_gap_prior_stock'])} | {pts(r['acc_gap_prior_gated'])} | {pct(r['err_prior_share_step0'])} ({pct(r['err_prior_share_chance_step0'])}) | "
                f"{pct(r['err_prior_share_gated'])} | {r['median_s_I']:.0f} |")

    for m in MODELS:
        Dm = S[m]["D"]
        for k, r in Dm["by_n_opt"].items():
            L.append(drow(m, k, r, r["prior_argmax"]))
        for k, r in Dm["by_gold"].items():
            L.append(drow(m, f"gold {k}", r, "-"))
        for k, r in Dm["by_n_opt_x_gold"].items():
            L.append(drow(m, k, r, "-"))
        L.append("")
        L += [f"{m}: takeover rate P(pred = prior | gold != prior) by within-n s_I quintile Q1..Q5 (n = {Dm['n_gold_ne_prior_by_s_I_quintile_within_n']}): "
              + "; ".join(f"{k}: {' / '.join(pct(v) for v in vs)}" for k, vs in Dm["takeover_rate_gold_ne_prior_by_s_I_quintile_within_n"].items()), ""]
    L += ["## E. Option-length check (Atabuzzaman & Thomas 2609.01318 report an option-LENGTH prior in dLVLMs)", "",
          "| model | decode | P(chosen longest) | chance | P(gold longest) | P(chosen longest given gold not longest) | P(chosen longest among wrong) | P(chosen shortest) | rank hist 1..5 (1 = longest) | mean rank (uniform) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        E = S[m]["E"]
        for name in ("step0", "stock", "gated", "imgfree", "txtfree", "prior_argmax"):
            c = E["chosen"][name]
            L.append(f"| {m} | {name} | {pct(c['p_chosen_is_longest'])} | {pct(E['chance_p_longest_uniform_choice'])} | {pct(E['p_gold_is_longest'])} | "
                     f"{pct(c['p_chosen_is_longest_given_gold_not_longest'])} | {pct(c['p_chosen_is_longest_among_wrong'])} | {pct(c['p_chosen_is_shortest'])} | "
                     f"{list(c['rank_hist_1_is_longest'].values())} | {c['mean_rank']:.2f} ({E['uniform_choice_mean_rank']:.2f}) |")
        L.append(f"| {m} | (dataset) P(position is longest) | " + ", ".join(f"{k}:{pct(v)}" for k, v in E["p_letter_position_is_longest"].items())
                 + f" | | | | | | within-item Spearman(option length, log p_cf) rho={E['spearman_rank_corr_option_length_vs_prior_logit_within_item']['rho']:+.3f} | |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--replace-own-output", action="store_true",
                   help="allow replacing an existing summary.json/tables.md/png set ONLY if it was written by this script")
    args = p.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE must be 1 (offline analysis)")
    summary_path = args.out_dir / "summary.json"
    own = {"summary.json", "tables.md"} | {f"{k}_{m}.png" for m in MODELS for k in ("A", "A_supp", "B", "C")}
    if summary_path.exists():
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        if not (args.replace_own_output and prev.get("script") == "experiments/unmask_order/visual_angle_20261013.py"):
            raise RuntimeError(f"refusing to overwrite {summary_path}")
        for f in own:
            if (args.out_dir / f).exists():
                (args.out_dir / f).unlink()
        print(f"replaced previous output of this script in {args.out_dir}", flush=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(16)
    lbo = json.loads(LBO_SUMMARY.read_text(encoding="utf-8"))
    S, plots = {}, {}
    for m in MODELS:
        print(f"== {m}", flush=True)
        S[m], plots[m] = run_model(m, lbo)
        print(json.dumps(S[m]["sanity"], indent=1), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        fig_A(m, S[m]["A"], args.out_dir / f"A_{m}.png")
        fig_A_supp(m, S[m]["A"], args.out_dir / f"A_supp_{m}.png")
        fig_B(m, S[m]["B"], plots[m], args.out_dir / f"B_{m}.png")
        fig_C(m, S[m]["C"], plots[m], args.out_dir / f"C_{m}.png")
    summary = {
        "date": "2026-10-13", "dataset": "ScienceQA-IMG test, 2017 items", "script": "experiments/unmask_order/visual_angle_20261013.py",
        "inputs": {"hidden": str(hor.SPLIT_DIR), "content_free": str(CF_DIR), "stock_final": str(TRAJ_DIR), "gated_final": str(GATED_DIR),
                   "letter_bias_origin_reference": str(LBO_SUMMARY), "options": str(MANIFEST)},
        "decomposition": "h_ITM = h_none + phi_I + phi_T + phi_M (exact 3-player Shapley over attention-key-masking subsets); "
                         "readout = float64 RMSNorm(ln_f gamma) + ff_out letter rows (ids 32..36); argmax restricted to the item's n options",
        "models": S,
    }
    with summary_path.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    with (args.out_dir / "tables.md").open("x", encoding="utf-8") as f:
        f.write(tables_md(S))
    print(tables_md(S))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
