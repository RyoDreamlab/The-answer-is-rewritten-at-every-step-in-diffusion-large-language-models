#!/usr/bin/env python3
"""Write-site calibration: the correction the circuit analysis exports.

The letter is written by a few late attention heads, again at every step. This script subtracts the gated
content-free letter prior (alpha_n = 1 - p_min/p_max, the same prior as the logit-level calibration) from those
heads only: the prior-removal direction in residual space is projected onto the writer heads' joint output span and
added to their attn_out inputs during decoding.

Conditions, ScienceQA-IMG, first --limit items:
  stock            plain gen-16 decode
  writesite_step0  the head edit at step 0 only
  writesite_every  the head edit at every step while the answer slot is masked

Writer heads are the ones already discovered on MMStar (not re-fit here). Lavida, LLaDA-V, and Dream-VL load
through lavida_causal_20261019 (Dream-VL: 28 layers x 28 heads, answer index prefix_len-1, <|im_end|> stripped
before the leading letter). Dream-VL ScienceQA rows are calib_decode_multi_20261013c.rows_sciqa, the same prompt
the supplement stock file was decoded with. New files only, under outputs/writesite_calib_20261019/.
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

OUT_ROOT = (PROJECT / "outputs" / "writesite_calib_20261019").resolve()
HEADS = {
    "lavida": PROJECT / "outputs" / "lavida_causal_20261019" / "writer_heads.json",
    "lladav": PROJECT / "outputs" / "lladav_causal_20261019" / "writer_heads.json",
    "mmada": None,  # p6.WRITER_HEADS, discovered on the same LLaDA-8B backbone
    "dreamvl": PROJECT / "outputs" / "dreamvl_causal_20261019" / "writer_heads.json",
}
STOCK_REF = {
    "lavida": PROJECT / "outputs" / "letter_argmax_traj_20261010" / "lavida.jsonl",
    "lladav": PROJECT / "outputs" / "lladav_20261013c" / "sciqa_stock.jsonl",
    "mmada": PROJECT / "outputs" / "letter_argmax_traj_20261010" / "mmada.jsonl",
    "dreamvl": PROJECT / "outputs" / "supplement_20261016" / "dreamvl_sciqa_stock.jsonl",
}
LLADAV_PRIOR = PROJECT / "outputs" / "lladav_20261013c" / "cf_lladav.json"
# Existing Dream-VL content-free prior (sciqa template, gray 384, question N/A, n x 'N/A', letter ids 32..36).
DREAMVL_PRIOR = PROJECT / "outputs" / "dreamvl_20261013g" / "cf_dreamvl.json"
DREAMVL_HEADS = [(25, 26), (28, 6), (27, 24)]
LETTERS = "ABCDE"
VARIANTS = ("stock", "writesite_step0", "writesite_every")
PROBE_C = 20.0
N_CALIB = 8
SEED, N_BOOT = 20261013, 1000


def check_out(path: Path) -> None:
    r = path.resolve()
    if r != OUT_ROOT and OUT_ROOT not in r.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {path}")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def lenient_letter(text: str) -> str | None:
    m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", str(text or ""))
    return m.group(1) if m else None


def mmada_rows(limit: int) -> list[dict]:
    import sciqa_standard_eval as sq

    data = PROJECT / "datasets/sciqa_img_standard_v1"
    recs = sq.load_manifest(data / "manifest.json", data)
    out = []
    for r in recs[:limit]:
        ch = r["choices"]
        if isinstance(ch, str):
            import ast
            ch = ast.literal_eval(ch)
        out.append({"id": r["id"], "gold": str(r.get("gold_option") or r["gold"]).strip(),
                    "n_opt": len(list(ch)), "_raw": r})
    return out


def load_mmada(cuda: int) -> dict:
    import dla_circuit_20261013d as dla
    import sciqa_standard_eval as sq

    data = PROJECT / "datasets/sciqa_img_standard_v1"
    recs = sq.load_manifest(data / "manifest.json", data)
    ns = argparse.Namespace(backend="mmada", cuda=cuda, model_path=str(PROJECT / "checkpoints/MMaDA-8B-Base"),
                            vq_model=str(PROJECT / "checkpoints/magvitv2"), mmada_repo=str(PROJECT / "repos/MMaDA"),
                            data_dir=str(data), manifest=str(data / "manifest.json"), rows=recs)
    ctx = dla.load_backend(ns)
    p6.install_p6(ctx["core"])
    ctx["backend"] = "mmada"
    ctx["_mmada"] = True
    return ctx


def load_prior(backend: str) -> tuple[dict, dict]:
    if backend in ("lladav", "dreamvl"):
        path = LLADAV_PRIOR if backend == "lladav" else DREAMVL_PRIOR
        if backend == "dreamvl":
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("backend") != "dreamvl" or meta.get("letter_ids") != [32, 33, 34, 35, 36]:
                raise RuntimeError(f"incompatible dreamvl prior {path}: backend={meta.get('backend')} ids={meta.get('letter_ids')}")
            have = {int(r["n_opt"]) for r in meta["records"] if r.get("cond") == "a_global"}
            if have != {2, 3, 4, 5}:
                raise RuntimeError(f"dreamvl prior a_global n_opt {sorted(have)}")
        log_pcf, alpha, pcf = c.load_priors(path)
        print(f"PRIOR file={path}", flush=True)
        for n in sorted(log_pcf):
            print(f"PRIOR n={n} p_cf={[round(v, 4) for v in pcf[n]]} alpha={alpha[n]:.4f}", flush=True)
        return log_pcf, alpha
    log_pcf = cd.load_priors(backend)
    alpha = {n: 1.0 - math.exp(-(max(v) - min(v))) for n, v in log_pcf.items()}
    return log_pcf, alpha


def project_heads(ctx, heads: list[tuple[int, int]], v: torch.Tensor) -> dict[int, torch.Tensor]:
    """Minimum-norm attn_out-input add whose total residual write is the projection of v onto the heads' span.
    Returned tensors are float64 on CPU, one (d,) per layer, and the total write has unit norm."""
    d = int(v.numel())
    hs = d // lc.N_HEADS
    cols = []
    spans = []
    for l, h in heads:
        Wh = ctx["Wout64"][l - 1][:, h * hs:(h + 1) * hs]
        cols.append(Wh)
        spans.append((l, h, hs))
    A = torch.cat(cols, dim=1)
    z = torch.linalg.lstsq(A, v.unsqueeze(1)).solution.squeeze(1)
    write = A @ z
    nrm = float(write.norm())
    if nrm < 1e-8:
        raise RuntimeError("writer heads cannot represent the prior direction")
    add = {l: torch.zeros(d, dtype=torch.float64) for l, _h in heads}
    off = 0
    for l, h, hsz in spans:
        add[l][h * hsz:(h + 1) * hsz] = z[off:off + hsz] / nrm
        off += hsz
    return add


def build_dirs(ctx, heads, log_pcf, alpha) -> dict[int, dict]:
    out = {}
    g = ctx["gamma64"]
    for n, lp in sorted(log_pcf.items()):
        shift = torch.tensor([alpha[n] * v for v in lp[:n]], dtype=torch.float64)
        target = -shift
        W = ctx["W64"][:n] * g
        v = torch.linalg.lstsq(W, target.unsqueeze(1)).solution.squeeze(1)
        add = project_heads(ctx, heads, v)
        # cos between the unit residual write and the least-squares residual for `target`
        hs = v.numel() // lc.N_HEADS
        recon = torch.zeros(v.numel(), dtype=torch.float64)
        for l, h in heads:
            sl = slice(h * hs, (h + 1) * hs)
            recon = recon + ctx["Wout64"][l - 1][:, sl] @ add[l][sl]
        cos = float((recon @ v) / (recon.norm() * v.norm() + 1e-12))
        out[n] = {"add": add, "target": target, "shift_norm": float(shift.norm()), "cos_head_vs_prior": cos,
                  "alpha": alpha[n]}
        print(f"DIR n={n} shift_norm={float(shift.norm()):.3f} cos_heads={cos:.3f} alpha={alpha[n]:.3f}", flush=True)
    return out


def calibrate(ctx, rows, dirs) -> dict[int, float | None]:
    """c such that c * (step-0 letter-logit change along the prior-removal direction) equals ||shift||.

    A non-positive median slope means the writer-head edit moves letter logits toward the prior, so that n is not
    calibrated: scale[n] is None and those items are skipped at decode time.
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
        item = ctx["real_item"](row["_raw"]) if ctx.get("_mmada") else ctx["make_item"](row["text"], row["image"]())
        x = p6.fresh(ctx)

        def cap():
            p6.P.capture = {lc.N_LAYERS}

        p6.forward_once(ctx, item, x, cap)
        base = lc.exact_letters_from_capture(ctx)
        add = dirs[n]["add"]
        target = dirs[n]["target"]
        t_hat = target / target.norm()

        def setup(_add=add):
            p6.P.head_add = {l: (PROBE_C * z).float().to(dev) for l, z in _add.items()}
            p6.P.capture = {lc.N_LAYERS}

        _logits, napp = p6.forward_once(ctx, item, x, setup)
        if napp != len(add):
            raise RuntimeError(f"calibration applied {napp} hooks, expected {len(add)}")
        moved = lc.exact_letters_from_capture(ctx)
        delta = torch.tensor(moved[:n], dtype=torch.float64) - torch.tensor(base[:n], dtype=torch.float64)
        slopes[n].append(float((delta @ t_hat) / PROBE_C))
    scale = {}
    for n, ss in slopes.items():
        if not ss:
            continue
        med = float(np.median(ss))
        if med <= 0:
            scale[n] = None
            print(f"SKIP n={n} slope/unit={med:.4f} from {len(ss)} items (non-positive prior-removal slope)", flush=True)
            continue
        scale[n] = dirs[n]["shift_norm"] / med
        print(f"SCALE n={n} slope/unit={med:.4f} c={scale[n]:.2f} from {len(ss)} items", flush=True)
    return scale


def run(args):
    backend = args.backend
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = (out_dir / f"{backend}.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {backend}") from exc
    files = {v: out_dir / f"{backend}_{v}.jsonl" for v in VARIANTS}
    skip_path = out_dir / f"{backend}_skipped.jsonl"
    for f in list(files.values()) + [skip_path]:
        check_out(f)
    done = set.intersection(*[{r["id"] for r in jsonl(f)} for f in files.values()])
    done |= {r["id"] for r in jsonl(skip_path)}
    if backend == "mmada":
        lc.N_LAYERS, lc.N_HEADS = 32, 32
        p6.N_LAYERS, p6.N_HEADS = 32, 32
        heads = list(p6.WRITER_HEADS)
        rows = mmada_rows(args.limit)
        ctx = load_mmada(args.cuda)
    else:
        heads_meta = json.loads(HEADS[backend].read_text(encoding="utf-8"))
        heads = [tuple(x) for x in heads_meta["writer_heads"]]
        if backend == "dreamvl" and heads != DREAMVL_HEADS:
            raise RuntimeError(f"refusing to rediscover heads; file has {heads}, expected {DREAMVL_HEADS}")
        lc.set_dims(backend)
        rows = c.rows_sciqa(backend, 0)[: args.limit]
        ctx = lc.load_backend(backend, args.cuda)
    log_pcf, alpha = load_prior(backend)
    dirs = build_dirs(ctx, heads, log_pcf, alpha)
    scale_path = out_dir / f"{backend}_scale.json"
    if scale_path.exists():
        scale = {int(k): (None if v is None else float(v)) for k, v in json.loads(scale_path.read_text())["c"].items()}
    else:
        scale = calibrate(ctx, rows, dirs)
        payload = {"c": {str(k): v for k, v in scale.items()}, "probe_c": PROBE_C,
                   "heads": [f"L{l}.H{h}" for l, h in heads],
                   "cos": {str(n): dirs[n]["cos_head_vs_prior"] for n in dirs}}
        if backend == "dreamvl":
            payload["prior_file"] = str(DREAMVL_PRIOR)
        scale_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    dev = ctx["device"]
    stock_ref = {r["id"]: r for r in jsonl(STOCK_REF[backend])}
    handles = {v: f.open("a", encoding="utf-8") for v, f in files.items()}
    skip_handle = skip_path.open("a", encoding="utf-8")
    chk = {"stock_cmp": 0, "stock_match": 0}
    t0 = time.time()
    n_new = 0
    n_skip = 0
    c_show = {k: (None if v is None else round(v, 2)) for k, v in scale.items()}
    print(f"start writesite backend={backend} n={len(rows)} done={len(done)} heads={heads} c={c_show}",
          flush=True)
    for row in rows:
        if row["id"] in done:
            continue
        n = int(row["n_opt"])
        if n not in scale:
            raise RuntimeError(f"{row['id']} n_opt={n} has no scale")
        if scale[n] is None:
            skip_handle.write(json.dumps({"id": row["id"], "skipped": "non-positive prior-removal slope", "n_opt": n},
                                         ensure_ascii=False) + "\n")
            skip_handle.flush()
            done.add(row["id"])
            n_skip += 1
            if n_skip <= 5 or len(done) % 10 == 0 or len(done) == len(rows):
                print(f"SKIP-ITEM {row['id']} n_opt={n} skipped={n_skip} done={len(done)}/{len(rows)}", flush=True)
            continue
        item = ctx["real_item"](row["_raw"]) if ctx.get("_mmada") else ctx["make_item"](row["text"], row["image"]())
        add = dirs[n]["add"]
        cc = scale[n]
        head_add = {l: (cc * z).float().to(dev) for l, z in add.items()}

        def make_setup(when: str):
            def setup_fn(step, _x, _when=when):
                if _when == "every" or (_when == "step0" and step == 0):
                    p6.P.head_add = head_add
            return setup_fn

        res = {"stock": lc.full_decode(ctx, item, None, tag=f"stock {row['id']}")}
        res["writesite_step0"] = lc.full_decode(ctx, item, make_setup("step0"), tag=f"step0 {row['id']}")
        res["writesite_every"] = lc.full_decode(ctx, item, make_setup("every"), tag=f"every {row['id']}")
        sref = stock_ref.get(row["id"])
        if sref is not None:
            chk["stock_cmp"] += 1
            chk["stock_match"] += int(sref.get("finished_letter") == res["stock"]["finished_letter"]
                                      and sref.get("letter_unmask_step") == res["stock"]["letter_unmask_step"])
        for variant, r in res.items():
            rec = {"id": row["id"], "gold": row["gold"], "n_opt": n, "backend": backend, "variant": variant,
                   "c": round(cc, 4), "alpha": dirs[n]["alpha"], "heads": [f"L{l}.H{h}" for l, h in heads],
                   "finished_letter": r["finished_letter"], "lenient_letter": lenient_letter(r["finished_text"]),
                   "finished_text": r["finished_text"][:160], "letter_unmask_step": r["letter_unmask_step"],
                   "n_hook_applications": r["n_hook_applications"],
                   "step0_letters": r["pos0_letters"][0] if r["pos0_letters"] else None}
            handles[variant].write(json.dumps(rec, ensure_ascii=False) + "\n")
            handles[variant].flush()
        done.add(row["id"])
        n_new += 1
        el = time.time() - t0
        if len(done) % 10 == 0 or len(done) == len(rows) or args.limit <= 5:
            print(f"PROGRESS {len(done)}/{len(rows)} rate={n_new / el:.3f}/s checks={chk} "
                  f"stock={res['stock']['finished_letter']} step0={res['writesite_step0']['finished_letter']} "
                  f"every={res['writesite_every']['finished_letter']} hooks={res['writesite_every']['n_hook_applications']}",
                  flush=True)
    for h in handles.values():
        h.close()
    skip_handle.close()
    print(f"DONE {backend} n={len(done)} skipped={n_skip} checks={chk} elapsed={time.time() - t0:.0f}s", flush=True)
    report(out_dir, backend)


def boot_ci(diff: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    sums = diff[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {"delta": int(diff.sum()), "ci95": [int(round(lo)), int(round(hi))]}


def report(out_dir: Path, backend: str) -> None:
    data = {v: {r["id"]: r for r in jsonl(out_dir / f"{backend}_{v}.jsonl")} for v in VARIANTS}
    ids = [i for i in data["stock"] if all(i in data[v] for v in VARIANTS)]
    if not ids:
        return
    gold = {i: data["stock"][i]["gold"] for i in ids}
    stock = data["stock"]
    lines = [f"# write-site calibration, {backend}, ScienceQA-IMG first {len(ids)}", ""]
    lines += ["| variant | strict | Δ strict [CI95] | lenient | Δ lenient [CI95] | letter changed |", "|---|---|---|---|---|---|"]
    summary = {"n": len(ids)}
    s_strict = np.array([int(stock[i]["finished_letter"] == gold[i]) for i in ids])
    s_len = np.array([int(stock[i]["lenient_letter"] == gold[i]) for i in ids])
    for v in VARIANTS:
        d = data[v]
        strict = np.array([int(d[i]["finished_letter"] == gold[i]) for i in ids])
        leng = np.array([int(d[i]["lenient_letter"] == gold[i]) for i in ids])
        ch = int(sum(d[i]["finished_letter"] != stock[i]["finished_letter"] for i in ids))
        bs, bl = boot_ci(strict - s_strict), boot_ci(leng - s_len)
        summary[v] = {"strict": int(strict.sum()), "lenient": int(leng.sum()), "d_strict": bs, "d_lenient": bl, "changed": ch}
        lines.append(f"| {v} | {int(strict.sum())} | {bs['delta']:+d} {bs['ci95']} | {int(leng.sum())} | {bl['delta']:+d} {bl['ci95']} | {ch} |")
    same = sum(data["writesite_step0"][i]["finished_letter"] == stock[i]["finished_letter"] for i in ids)
    skipped = jsonl(out_dir / f"{backend}_skipped.jsonl")
    lines.append("")
    lines.append(f"step0 letter == stock: {same}/{len(ids)}")
    if skipped:
        lines.append(f"skipped non-positive slope: {len(skipped)} (n_opt {sorted({int(r['n_opt']) for r in skipped})})")
        summary["skipped"] = len(skipped)
        summary["skipped_n_opt"] = sorted({int(r["n_opt"]) for r in skipped})
    (out_dir / f"{backend}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{backend}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("lavida", "lladav", "mmada", "dreamvl"), required=True)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    args.out_dir = args.out_dir.resolve()
    check_out(args.out_dir)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
    run(args)


if __name__ == "__main__":
    main()
