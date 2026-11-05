#!/usr/bin/env python3
"""LaViDa L32.mlp prior removal. Not a writer-head edit.

On real inputs the component aligned with LaViDa's letter prior is L32.mlp. The gray-image prior is subtracted
there, by adding the least-squares residual of -alpha_n * log p_cf to the layer-32 MLP output (ff_out) at the
answer slot. p6.comp_add writes attn_out, so this uses the equivalent hook on blocks[31].ff_out and never sets
head_add or comp_add.

alpha_n = 1 - p_min/p_max from calib_decode.load_priors("lavida"). A positive scalar c is fit on a few step-0
items so the measured letter-logit move along that direction equals ||alpha * log p_cf||. A non-positive slope
stops the run; the sign is not flipped.

Conditions: stock, mlp_step0 (step 0 only), mlp_every (every step while the answer slot is masked).
ScienceQA-IMG via calib_decode_multi_20261013c.rows_sciqa, gen 16, T=0, low-confidence remasking.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
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
import calib_decode as cd  # noqa: E402
import calib_decode_multi_20261013c as c  # noqa: E402
import lavida_causal_20261019 as lc  # noqa: E402
import p6_causal_20261013f as p6  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "lavida_mlp_calib_20261020").resolve()
STOCK_REF = PROJECT / "outputs" / "letter_argmax_traj_20261010" / "lavida.jsonl"
VARIANTS = ("stock", "mlp_step0", "mlp_every")
PROBE_C = 20.0
N_CALIB = 8
SEED, N_BOOT = 20261013, 1000
MLP_LAYER = 32
# Published logit-level gated decode (calib_gated_decode_20261013). Not rerun.
LOGIT_REFERENCE = {
    "source": "outputs/calib_gated_decode_20261013 + WEEK1_RESULTS_20261013 §2.1",
    "stock_strict": 1460,
    "gated_strict": 1468,
    "delta_strict": 8,
    "ci95": [-10, 25],
}


class MlpAdd:
    """Residual add on L32.mlp (ff_out) at the answer index. Independent of p6 head_add / comp_add."""

    vec = None
    index = 0
    fired = 0


def check_out(path: Path) -> None:
    r = path.resolve()
    if r != OUT_ROOT and OUT_ROOT not in r.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {path}")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def lenient_letter(text: str) -> str | None:
    m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", str(text or ""))
    return m.group(1) if m else None


def install_l32_mlp(core) -> None:
    blocks = list(core.transformer.blocks)
    if len(blocks) != MLP_LAYER:
        raise RuntimeError(f"expected {MLP_LAYER} blocks, got {len(blocks)}")
    ff = blocks[MLP_LAYER - 1].ff_out
    if ff.bias is not None:
        raise RuntimeError("L32.ff_out has a bias")

    def mlp_post(_m, _args, out):
        a = MlpAdd.vec
        if a is None:
            return None
        if isinstance(out, (tuple, list)):
            raise RuntimeError("L32.ff_out returned a tuple")
        out = out.clone()
        out[0, MlpAdd.index] = (out[0, MlpAdd.index].float() + a).to(out.dtype)
        MlpAdd.fired += 1
        p6.P.n_applied += 1
        return out

    ff.register_forward_hook(mlp_post)
    print(f"HOOK L32.mlp ff_out weight={tuple(ff.weight.shape)}", flush=True)


def load_prior() -> tuple[dict, dict]:
    log_pcf = cd.load_priors("lavida")
    alpha = {n: 1.0 - math.exp(-(max(v) - min(v))) for n, v in log_pcf.items()}
    for n in sorted(log_pcf):
        pcf = [round(math.exp(v), 4) for v in log_pcf[n]]
        print(f"PRIOR n={n} p_cf={pcf} alpha={alpha[n]:.4f}", flush=True)
    return log_pcf, alpha


def build_dirs(ctx, log_pcf, alpha) -> dict[int, dict]:
    """Unit residual added to L32.mlp. It is the least-squares prior-removal residual, so cosine with it is 1."""
    out = {}
    g = ctx["gamma64"]
    for n, lp in sorted(log_pcf.items()):
        shift = torch.tensor([alpha[n] * v for v in lp[:n]], dtype=torch.float64)
        target = -shift
        W = ctx["W64"][:n] * g
        v = torch.linalg.lstsq(W, target.unsqueeze(1)).solution.squeeze(1)
        nrm = float(v.norm())
        if nrm < 1e-8:
            raise RuntimeError(f"n={n} prior-removal residual is ~0")
        unit = v / nrm
        pred = W @ v
        cos = float((unit @ v) / (unit.norm() * v.norm() + 1e-12))
        fit = float((pred @ target) / (pred.norm() * target.norm() + 1e-12))
        out[n] = {"unit": unit, "target": target, "shift_norm": float(shift.norm()),
                  "cos_mlp_vs_prior": cos, "cos_logit_fit": fit, "alpha": alpha[n],
                  "fit_rel": float((pred - target).norm() / (target.norm() + 1e-12))}
        print(f"DIR n={n} shift_norm={float(shift.norm()):.3f} cos_mlp_vs_prior={cos:.3f} "
              f"logit_fit={fit:.3f} alpha={alpha[n]:.3f}", flush=True)
    return out


def clear_mlp() -> None:
    MlpAdd.vec = None


def calibrate(ctx, rows, dirs, out_dir: Path) -> dict[int, float]:
    """c so that c * (step-0 letter-logit move along the prior-removal direction) equals ||shift||.

    Median slope <= 0 aborts. The residual is not negated.
    """
    slopes = {n: [] for n in dirs}
    dev = ctx["device"]
    picked = {n: [] for n in dirs}
    for row in rows:
        n = int(row["n_opt"])
        if n in picked and len(picked[n]) < N_CALIB:
            picked[n].append(row)
    for row in [r for group in picked.values() for r in group]:
        n = int(row["n_opt"])
        item = ctx["make_item"](row["text"], row["image"]())
        if int(item["answer_index"]) != 0:
            raise RuntimeError(f"answer index {item['answer_index']}")
        x = p6.fresh(ctx)
        unit = dirs[n]["unit"]
        target = dirs[n]["target"]
        t_hat = target / target.norm()

        def cap():
            clear_mlp()
            p6.P.capture = {lc.N_LAYERS}

        p6.forward_once(ctx, item, x, cap)
        base = lc.exact_letters_from_capture(ctx)
        fired0 = MlpAdd.fired

        def setup(_unit=unit):
            clear_mlp()
            MlpAdd.index = 0
            MlpAdd.vec = (PROBE_C * _unit).float().to(dev)
            p6.P.capture = {lc.N_LAYERS}
            p6.P.head_add = {}
            p6.P.comp_add = {}

        try:
            _logits, napp = p6.forward_once(ctx, item, x, setup)
        finally:
            clear_mlp()
        if MlpAdd.fired != fired0 + 1 or napp != 1:
            raise RuntimeError(f"calibration hook fired {MlpAdd.fired - fired0} times, n_applied={napp}")
        moved = lc.exact_letters_from_capture(ctx)
        delta = torch.tensor(moved[:n], dtype=torch.float64) - torch.tensor(base[:n], dtype=torch.float64)
        slopes[n].append(float((delta @ t_hat) / PROBE_C))
    scale = {}
    bad = []
    for n, ss in slopes.items():
        if not ss:
            continue
        med = float(np.median(ss))
        print(f"SLOPE n={n} slope/unit={med:.6f} from {len(ss)} items values={[round(s, 5) for s in ss]}", flush=True)
        if not math.isfinite(med) or med <= 0:
            bad.append({"n": n, "slope_per_unit": med, "n_items": len(ss)})
            continue
        scale[n] = dirs[n]["shift_norm"] / med
        print(f"SCALE n={n} c={scale[n]:.2f}", flush=True)
    if bad:
        stop = {"stopped": "non-positive prior-removal slope", "sign_flipped": False, "bad": bad,
                "slopes": {str(n): ss for n, ss in slopes.items()},
                "cos_mlp_vs_prior": {str(n): dirs[n]["cos_mlp_vs_prior"] for n in dirs}}
        stop_path = out_dir / "lavida_slope_stop.json"
        check_out(stop_path)
        stop_path.write_text(json.dumps(stop, indent=1), encoding="utf-8")
        print("STOP non-positive prior-removal slope; not flipping the sign. " + json.dumps(stop), flush=True)
        raise RuntimeError("non-positive prior-removal slope; not flipping the sign. " + json.dumps(bad))
    missing = sorted(set(picked) - set(scale))
    if any(picked[n] for n in missing):
        raise RuntimeError(f"missing scale for n={missing}")
    return scale


def apply_vec(ctx, unit, cc):
    return (cc * unit).float().to(ctx["device"])


def decode_variant(ctx, item, when: str | None, vec) -> dict:
    clear_mlp()
    MlpAdd.index = 0
    mask_id = int(ctx["mask_id"])

    def setup_fn(step, x, _when=when, _vec=vec):
        clear_mlp()
        p6.P.head_add = {}
        p6.P.comp_add = {}
        if _when is None or _vec is None:
            return
        if int(x[0, 0].item()) != mask_id:
            return
        if _when == "every" or (_when == "step0" and step == 0):
            MlpAdd.index = 0
            MlpAdd.vec = _vec

    try:
        if when is None:
            return lc.full_decode(ctx, item, None, tag="stock")
        return lc.full_decode(ctx, item, setup_fn, tag=when)
    finally:
        clear_mlp()


def run(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = (out_dir / "lavida.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("lavida mlp calib already running") from exc
    files = {v: out_dir / f"lavida_{v}.jsonl" for v in VARIANTS}
    for f in files.values():
        check_out(f)
    done = set.intersection(*[{r["id"] for r in jsonl(f)} for f in files.values()])
    lc.set_dims("lavida")
    rows = c.rows_sciqa("lavida", 0)[: args.limit]
    if len(rows) != args.limit and args.limit <= 2017:
        raise RuntimeError(f"expected {args.limit} rows, got {len(rows)}")
    ctx = lc.load_backend("lavida", args.cuda)
    install_l32_mlp(ctx["core"])
    log_pcf, alpha = load_prior()
    dirs = build_dirs(ctx, log_pcf, alpha)
    scale_path = out_dir / "lavida_scale.json"
    check_out(scale_path)
    if scale_path.exists():
        payload = json.loads(scale_path.read_text(encoding="utf-8"))
        if payload.get("sign_flipped"):
            raise RuntimeError("refusing a scale file that flipped the sign")
        scale = {int(k): float(v) for k, v in payload["c"].items()}
        print(f"SCALE reused { {k: round(v, 2) for k, v in scale.items()} }", flush=True)
    else:
        scale = calibrate(ctx, rows, dirs, out_dir)
        need = {int(r["n_opt"]) for r in rows}
        if not need <= set(scale):
            raise RuntimeError(f"scale missing n_opt {sorted(need - set(scale))}")
        payload = {"c": {str(k): scale[k] for k in sorted(scale)}, "probe_c": PROBE_C, "site": "L32.mlp",
                   "sign_flipped": False,
                   "cos_mlp_vs_prior": {str(n): dirs[n]["cos_mlp_vs_prior"] for n in sorted(dirs)},
                   "cos_logit_fit": {str(n): dirs[n]["cos_logit_fit"] for n in sorted(dirs)},
                   "alpha": {str(n): dirs[n]["alpha"] for n in sorted(dirs)}}
        scale_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    cos_high = all(dirs[n]["cos_mlp_vs_prior"] >= 0.9 for n in dirs)
    print(f"COS mlp_vs_prior high={cos_high} values={ {n: round(dirs[n]['cos_mlp_vs_prior'], 4) for n in sorted(dirs)} }",
          flush=True)
    stock_ref = {r["id"]: r for r in jsonl(STOCK_REF)}
    handles = {v: f.open("a", encoding="utf-8") for v, f in files.items()}
    chk = {"stock_cmp": 0, "stock_match": 0}
    t0 = time.time()
    n_new = 0
    c_show = {k: round(v, 2) for k, v in scale.items()}
    print(f"start lavida mlp-only n={len(rows)} done={len(done)} site=L32.mlp c={c_show}", flush=True)
    try:
        for row in rows:
            if row["id"] in done:
                continue
            n = int(row["n_opt"])
            if n not in scale:
                raise RuntimeError(f"{row['id']} n_opt={n} has no scale")
            item = ctx["make_item"](row["text"], row["image"]())
            vec = apply_vec(ctx, dirs[n]["unit"], scale[n])
            fired0 = MlpAdd.fired
            res = {}
            res["stock"] = decode_variant(ctx, item, None, None)
            res["mlp_step0"] = decode_variant(ctx, item, "step0", vec)
            res["mlp_every"] = decode_variant(ctx, item, "every", vec)
            fires = {
                "stock": 0,
                "mlp_step0": 1,
                "mlp_every": int(res["mlp_every"]["letter_unmask_step"]) + 1,
            }
            got = MlpAdd.fired - fired0
            expect = fires["mlp_step0"] + fires["mlp_every"]
            if got != expect:
                raise RuntimeError(f"{row['id']} MLP hook fired {got}, expected {expect}")
            for variant, r in res.items():
                if r["n_hook_applications"] != fires[variant]:
                    raise RuntimeError(f"{row['id']} {variant} hooks {r['n_hook_applications']} != {fires[variant]}")
            sref = stock_ref.get(row["id"])
            if sref is None:
                raise RuntimeError(f"{row['id']} missing from {STOCK_REF}")
            match = (sref.get("finished_letter") == res["stock"]["finished_letter"]
                     and int(sref.get("letter_unmask_step")) == int(res["stock"]["letter_unmask_step"]))
            chk["stock_cmp"] += 1
            chk["stock_match"] += int(match)
            if args.require_stock_match and not match:
                raise RuntimeError(
                    f"stock mismatch {row['id']}: got letter={res['stock']['finished_letter']} "
                    f"step={res['stock']['letter_unmask_step']} ref letter={sref.get('finished_letter')} "
                    f"step={sref.get('letter_unmask_step')}")
            for variant, r in res.items():
                rec = {"id": row["id"], "gold": row["gold"], "n_opt": n, "backend": "lavida", "variant": variant,
                       "site": "L32.mlp", "c": round(scale[n], 4), "alpha": dirs[n]["alpha"],
                       "cos_mlp_vs_prior": dirs[n]["cos_mlp_vs_prior"],
                       "finished_letter": r["finished_letter"], "lenient_letter": lenient_letter(r["finished_text"]),
                       "finished_text": r["finished_text"][:160], "letter_unmask_step": r["letter_unmask_step"],
                       "n_hook_applications": r["n_hook_applications"],
                       "step0_letters": r["pos0_letters"][0] if r["pos0_letters"] else None}
                handles[variant].write(json.dumps(rec, ensure_ascii=False) + "\n")
                handles[variant].flush()
            done.add(row["id"])
            n_new += 1
            el = time.time() - t0
            if len(done) <= 12 or len(done) % 10 == 0 or len(done) == len(rows):
                print(f"PROGRESS {len(done)}/{len(rows)} rate={n_new / el:.3f}/s checks={chk} "
                      f"stock={res['stock']['finished_letter']} step0={res['mlp_step0']['finished_letter']} "
                      f"every={res['mlp_every']['finished_letter']} "
                      f"hooks={res['stock']['n_hook_applications']}/{res['mlp_step0']['n_hook_applications']}/"
                      f"{res['mlp_every']['n_hook_applications']}", flush=True)
    finally:
        for h in handles.values():
            h.close()
    print(f"DONE lavida n={len(done)} checks={chk} elapsed={time.time() - t0:.0f}s", flush=True)
    report(out_dir)


def boot_ci(diff: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    sums = diff[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {"delta": int(diff.sum()), "ci95": [int(round(lo)), int(round(hi))]}


def report(out_dir: Path) -> None:
    data = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for r in jsonl(out_dir / f"lavida_{v}.jsonl"):
            data[v][r["id"]] = r
    ids = [i for i in data["stock"] if all(i in data[v] for v in VARIANTS)]
    if not ids:
        return
    gold = {i: data["stock"][i]["gold"] for i in ids}
    stock = data["stock"]
    lines = [f"# LaViDa L32.mlp prior removal, ScienceQA-IMG n={len(ids)}", "",
             "Site is L32.mlp because on real inputs that component carries LaViDa's letter prior.",
             ""]
    lines += ["| variant | strict | Δ strict [CI95] | lenient | Δ lenient [CI95] | letter changed |",
              "|---|---|---|---|---|---|"]
    summary = {"n": len(ids), "site": "L32.mlp", "logit_level_gated_reference": LOGIT_REFERENCE}
    s_strict = np.array([int(stock[i]["finished_letter"] == gold[i]) for i in ids])
    s_len = np.array([int(stock[i]["lenient_letter"] == gold[i]) for i in ids])
    for v in VARIANTS:
        d = data[v]
        strict = np.array([int(d[i]["finished_letter"] == gold[i]) for i in ids])
        leng = np.array([int(d[i]["lenient_letter"] == gold[i]) for i in ids])
        ch = int(sum(d[i]["finished_letter"] != stock[i]["finished_letter"] for i in ids))
        bs, bl = boot_ci(strict - s_strict), boot_ci(leng - s_len)
        summary[v] = {"strict": int(strict.sum()), "lenient": int(leng.sum()), "d_strict": bs,
                      "d_lenient": bl, "changed": ch}
        lines.append(f"| {v} | {int(strict.sum())} | {bs['delta']:+d} {bs['ci95']} | {int(leng.sum())} | "
                     f"{bl['delta']:+d} {bl['ci95']} | {ch} |")
    same = sum(data["mlp_step0"][i]["finished_letter"] == stock[i]["finished_letter"] for i in ids)
    lines.append("")
    lines.append(f"step0 letter == stock: {same}/{len(ids)}")
    summary["step0_eq_stock"] = f"{same}/{len(ids)}"
    scale_path = out_dir / "lavida_scale.json"
    if scale_path.exists():
        sc = json.loads(scale_path.read_text(encoding="utf-8"))
        summary["c"] = sc.get("c")
        summary["cos_mlp_vs_prior"] = sc.get("cos_mlp_vs_prior")
        cos_vals = [float(v) for v in (sc.get("cos_mlp_vs_prior") or {}).values()]
        summary["cos_high"] = bool(cos_vals) and min(cos_vals) >= 0.9
        lines.append(f"c: {sc.get('c')}")
        lines.append(f"cos_mlp_vs_prior: {sc.get('cos_mlp_vs_prior')} high={summary['cos_high']}")
    lines.append(f"logit-level gated reference (not rerun): strict {LOGIT_REFERENCE['gated_strict']} vs stock "
                 f"{LOGIT_REFERENCE['stock_strict']}, Δ {LOGIT_REFERENCE['delta_strict']:+d} {LOGIT_REFERENCE['ci95']}")
    (out_dir / "lavida_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "lavida_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--limit", type=int, default=2017)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--require-stock-match", action="store_true")
    args = p.parse_args()
    args.out_dir = args.out_dir.resolve()
    check_out(args.out_dir)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
    print(f"lavida_mlp_calib_20261020 start cuda={args.cuda} limit={args.limit} out={args.out_dir}", flush=True)
    run(args)


if __name__ == "__main__":
    main()
