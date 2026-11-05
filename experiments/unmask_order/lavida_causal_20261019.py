#!/usr/bin/env python3
"""Causal write-site test on a second / third model (every-step-rewrite mechanism), LaViDa and LLaDA-V.

MMaDA (P6 + supplement_20261016) showed: a write-site edit at step 0 only never changes the final letter, the same
edit at every step moves ~90% to the steered letter.  This script repeats the test on another model with that model's
OWN write site, discovered by DLA on held-out items, instead of copying the MMaDA head ids.

Backends (``--backend``)
  lavida  LaViDa (LLaDA-8B core, transformer.blocks[*].attn_out / ff_out, prefix KV cache -> canvas-only forward,
          answer index 0).  Eligible items: LaViDa stock letter commits at the last step
          (outputs/supplement_20261016/lavida_mmstar_stock.jsonl).
  lladav  LLaDA-V (LLaDA-8B core in HF-Llama layout, model.layers[*].self_attn.o_proj / mlp.down_proj, no KV cache ->
          full forward, answer index = prefix_len).  LLaDA-V commits the letter EARLY (steps 1-3) and on ~40% of MMStar
          items writes a sentence instead of a letter; eligible items are those whose stock first token is a letter
          (outputs/gap_20261019/lladav_mmstar_stock.jsonl).  The commit step is the item's own.
  dreamvl Dream-VL-7B (Qwen2.5-7B-based Dream-7B, 28 layers x 28 heads, hidden 3584, model.model.layers[*].self_attn.o_proj
          / mlp.down_proj, M-RoPE, no KV cache -> full forward).  Dream shift: canvas position 0 is predicted by hidden
          position prefix_len-1, so answer index = prefix_len-1.  Eligible items: stock letter commits at the last step
          (outputs/supplement_20261016/dreamvl_mmstar_stock.jsonl, 1268/1500).  Layer/head counts are set per backend
          (set_dims) in this module and in dla_circuit_20261013d / p6_causal_20261013f.

Modes (``--mode``)
  dla     Head discovery on the first --n-select eligible MMStar items (manifest order).  Per item: stock decode
          (checked against the prior stock file), then two capture forwards at the answer position: the step-0 canvas
          (all masked) and the commit-step canvas (stock tokens unmasked before the commit step, everything else
          masked = the canvas the model saw when it wrote the letter).  Per-head DLA letter logits (frozen-scale final
          norm readout, dla_circuit_20261013d maths) give each head's contribution to  logit[first] - logit[second]
          (first/second = top two letters of the stock step-0 restricted logits, the pair the steering uses).
          Writer heads = top-3 heads by mean commit-step margin contribution.  heads.jsonl, heads_arrays.npz,
          writer_heads.json.
  steer   The NEXT --n-steer eligible items (disjoint from the selection set).  Conditions per item: stock; writer
          step0 beta=+8; writer every beta in {+4,+8}; random every beta in {+4,+8}; random step0 beta=+8.  Random
          heads: 3 heads from the writer heads' layers excluding the top-10 writer-ranked heads (seed 20261013).
          unit_scale calibrated as in P6 (first --n-calib items, probe c=20/40, float64 exact readout); random uses the
          writer unit_scale (matched residual norm).  calib.json, steer.jsonl.
  analyze summary.md / summary.json.

New files only; outputs under outputs/{lavida,lladav}_causal_20261019/.  Resumable (jsonl append + flock).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
sys.path.insert(0, str(PROJECT / "experiments"))
import calib_decode_multi_20261013c as c  # noqa: E402
import dla_circuit_20261013d as dla  # noqa: E402
import p6_causal_20261013f as p6  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOTS = {"lavida": (PROJECT / "outputs" / "lavida_causal_20261019").resolve(),
             "lladav": (PROJECT / "outputs" / "lladav_causal_20261019").resolve(),
             "dreamvl": (PROJECT / "outputs" / "dreamvl_causal_20261019").resolve()}
PRIOR_FILES = {"lavida": PROJECT / "outputs" / "supplement_20261016" / "lavida_mmstar_stock.jsonl",
               "lladav": PROJECT / "outputs" / "gap_20261019" / "lladav_mmstar_stock.jsonl",
               "dreamvl": PROJECT / "outputs" / "supplement_20261016" / "dreamvl_mmstar_stock.jsonl"}
DIMS = {"lavida": (32, 32), "lladav": (32, 32), "dreamvl": (28, 28)}       # (n_layers, n_heads)
# second benchmark (--dataset sciqa): ScienceQA-IMG test 2017, rows / prompts from calib_decode_multi_20261013c.rows_sciqa,
# prior stock files = the existing full-set stock decodes; writer heads are NOT re-discovered: the MMStar writer_heads.json
# is copied into the sciqa out dir, so sciqa tests whether the MMStar heads transfer to a second benchmark.
SQA_OUT_ROOTS = {b: (PROJECT / "outputs" / f"{b}_causal_sqa_20261019").resolve() for b in DIMS}
SQA_PRIOR_FILES = {"lavida": PROJECT / "outputs" / "letter_argmax_traj_20261010" / "lavida.jsonl",
                   "lladav": PROJECT / "outputs" / "lladav_20261013c" / "sciqa_stock.jsonl",
                   "dreamvl": PROJECT / "outputs" / "supplement_20261016" / "dreamvl_sciqa_stock.jsonl"}
PRIOR_N = {"mmstar": 1500, "sciqa": 2017}
DATASET = "mmstar"
MANIFEST = PROJECT / "datasets" / "mmstar_20261013e" / "manifest.json"
LAVIDA_PATH = PROJECT / "checkpoints" / "lavida-llada-v1.0-instruct"
LLADAV_PATH = PROJECT / "checkpoints" / "LLaDA-V"
DREAMVL_PATH = PROJECT / "checkpoints" / "Dream-VL-7B"
LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]        # 'A'..'E' in both the LLaDA and the Qwen2 (Dream) byte-level vocabularies
N_LAYERS, N_HEADS = 32, 32


def set_dims(backend: str) -> None:
    """dla / p6 use module-level N_LAYERS / N_HEADS (comp_names, collect, head_letters, Steerer); set them per backend."""
    global N_LAYERS, N_HEADS
    N_LAYERS, N_HEADS = DIMS[backend]
    dla.N_LAYERS, dla.N_HEADS = N_LAYERS, N_HEADS
    p6.N_LAYERS, p6.N_HEADS = N_LAYERS, N_HEADS


def full_decode(ctx, item, setup_fn=None, tag="decode") -> dict:
    """p6.full_decode; for dreamvl the tokenizer keeps <|im_end|> under skip_special_tokens, so the finished text is
    cleaned as in backend_dreamvl_20261013g.Dreamvl.decode before the leading letter is read."""
    res = p6.full_decode(ctx, item, setup_fn, tag)
    if ctx["backend"] == "dreamvl":
        import backend_dreamvl_20261013g as bd

        res["finished_text_raw"] = res["finished_text"]
        res["finished_text"] = bd.strip_special_text(res["finished_text"])
        res["finished_letter"] = p6.delta.leading_letter(res["finished_text"])
    return res


def exact_letters_from_capture(ctx) -> list[float]:
    """p6.exact_letters_from_capture reads P.captured[32]; this reads the last layer of the current backend."""
    h = p6.P.captured[N_LAYERS].to(torch.float64).cpu()
    return [float(v) for v in dla.exact_letters(h, ctx["gamma64"], ctx["W64"], ctx["eps"])]
SEED = 20261013
GEN = base.GEN
CALIB_PROBES = (20.0, 40.0)
BETAS_EVERY = tuple(float(b) for b in os.environ.get("LC_BETAS_EVERY", "4,8").split(",") if b.strip())
_BETA_STEP0 = os.environ.get("LC_BETA_STEP0", "8").strip()
BETA_STEP0 = float(_BETA_STEP0) if _BETA_STEP0 else None  # "" = skip step0 conds
LC_WHEN = os.environ.get("LC_WHEN", "").strip()  # "" = step0+every; "commit" = edit only at the stock commit step
N_WRITERS = 3
EXCLUDE_TOP = 10


def check_out(path: Path) -> None:
    r = path.resolve()
    extra = (PROJECT / "outputs" / "causal_full_20261020").resolve()
    commit_root = (PROJECT / "outputs" / "causal_commit_20261021").resolve()
    roots = list(OUT_ROOTS.values()) + list(SQA_OUT_ROOTS.values()) + [extra, commit_root]
    if not any(r == root or root in r.parents for root in roots):
        raise RuntimeError(f"refusing to write outside {roots}: {path}")


def out_root(backend: str) -> Path:
    return (OUT_ROOTS if DATASET == "mmstar" else SQA_OUT_ROOTS)[backend]


def prior_file(backend: str) -> Path:
    return (PRIOR_FILES if DATASET == "mmstar" else SQA_PRIOR_FILES)[backend]


# ----------------------------------------------------------------------------- data

def load_rows(backend: str = "lavida") -> list[dict]:
    if DATASET == "sciqa":
        rows = c.rows_sciqa(backend, 0)          # id, gold, choices, n_opt, question, hint, text, image()
    else:
        rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for r in rows:
        ch = list(r["choices"])
        if r["n_opt"] != len(ch) or r["gold"] not in LETTERS[: len(ch)]:
            raise RuntimeError(f"bad {DATASET} row {r['id']}")
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError(f"duplicate {DATASET} ids")
    return rows


def load_prior(backend: str) -> dict[str, dict]:
    prior = {}
    for line in prior_file(backend).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            prior[r["id"]] = r
    if len(prior) != PRIOR_N[DATASET]:
        raise RuntimeError(f"prior stock has {len(prior)} items, expected {PRIOR_N[DATASET]}")
    return prior


def eligible(backend: str, pr: dict) -> bool:
    if backend in ("lavida", "dreamvl"):
        return int(pr["last_step"]) == 1 if "last_step" in pr else int(pr["letter_unmask_step"]) == GEN - 1
    return pr.get("finished_letter") is not None           # lladav: first token is a letter


def split_items(backend: str, rows: list[dict], n_select: int, n_steer: int):
    prior = load_prior(backend)
    keep = [r for r in rows if eligible(backend, prior[r["id"]])]
    if len(keep) < n_select + n_steer:
        raise RuntimeError(f"only {len(keep)} eligible items for {backend}")
    return keep[:n_select], keep[n_select:n_select + n_steer], prior


def text_of(backend: str, row) -> str:
    if "text" in row:                                       # sciqa rows carry the backend's own prompt
        return row["text"]
    return c.text_for(backend, row["question"], row["hint"], list(row["choices"]))


def load_image(row):
    from PIL import Image

    if callable(row["image"]):
        return row["image"]()
    return Image.open(row["image"]).convert("RGB")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------- generic hooks (HF-Llama layout, LLaDA-V)

def install_generic(layers, attn_out_of, mlp_out_of) -> None:
    """Same hook semantics as dla.install (H: capture / ablate) + p6.install_p6 (P: head_add / comp_add / patch /
    capture) for a decoder whose modules are not named transformer.blocks[*].attn_out.  Layers are 1-based in H/P."""
    H, P = dla.H, p6.P
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} layers, got {len(layers)}")
    for mod in (attn_out_of(layers[0]), mlp_out_of(layers[0])):
        if getattr(mod, "bias", None) is not None:
            raise RuntimeError(f"{mod} has a bias; DLA split assumes none")

    def embed_pre(_m, args):
        if H.armed and H.capture:
            H.comps["embed"] = args[0][0, H.index].detach()

    layers[0].register_forward_pre_hook(embed_pre)
    for li, layer in enumerate(layers):
        l = li + 1

        def attn_pre(_m, args, _l=l):
            z = args[0]
            changed = False
            if H.armed and H.capture:
                H.heads[_l] = z[0, H.index].detach()
            if P.armed:
                a = P.head_add.get(_l)
                if a is not None:
                    z = z.clone(); changed = True
                    z[0, P.index] = (z[0, P.index].float() + a).to(z.dtype)
                    P.n_applied += 1
            return (z,) + tuple(args[1:]) if changed else None

        def attn_post(_m, _args, out, _l=l):
            if H.armed and H.capture:
                H.comps[f"L{_l}.attn"] = out[0, H.index].detach()
            if P.armed:
                a = P.comp_add.get(_l)
                if a is not None:
                    out = out.clone()
                    out[0, P.index] = (out[0, P.index].float() + a).to(out.dtype)
                    P.n_applied += 1
                    return out
            return None

        def mlp_post(_m, _args, out, _l=l):
            if H.armed and H.capture:
                H.comps[f"L{_l}.mlp"] = out[0, H.index].detach()
            return None

        def layer_post(_m, _args, out, _l=l):
            is_tuple = isinstance(out, (tuple, list))
            x = out[0] if is_tuple else out
            if H.armed and H.capture and _l == N_LAYERS:
                H.final = x[0, H.index].detach()
            if P.armed:
                if _l in P.capture:
                    P.captured[_l] = x[0, P.index].detach().float().clone()
                v = P.patch.get(_l)
                if v is not None:
                    x = x.clone()
                    x[0, P.index] = v.to(x.dtype)
                    P.n_applied += 1
                    return (x,) + tuple(out[1:]) if is_tuple else x
            return None

        attn_out_of(layer).register_forward_pre_hook(attn_pre)
        attn_out_of(layer).register_forward_hook(attn_post)
        mlp_out_of(layer).register_forward_hook(mlp_post)
        layer.register_forward_hook(layer_post)


# ----------------------------------------------------------------------------- model loading

def load_backend(backend: str, cuda: int) -> dict:
    torch.cuda.set_device(cuda)
    torch.set_grad_enabled(False)
    if backend == "lavida":
        import calib_contentfree as ccf
        import hidden_shapley_split as hss
        sys.path.insert(0, str(PROJECT / "repos" / "LaViDa"))
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.model.builder import load_pretrained_model

        siglip = (PROJECT / "hf-cache/hub/models--google--siglip-so400m-patch14-384/snapshots/"
                  "9fdffc58afc957d1a03a25b10dba0329ab15c2a3")
        vision_kwargs = {"mm_vision_tower": str(siglip) if siglip.exists() else "google/siglip-so400m-patch14-384",
                         "mm_resampler_type": None, "mm_projector_type": "mlp2x_gelu", "mm_hidden_size": 1152,
                         "use_mm_proj": True}
        print("loading lavida", flush=True)
        tokenizer, model, processor, _ = load_pretrained_model(
            str(LAVIDA_PATH), None, "llava_llada", device_map=f"cuda:{cuda}",
            vision_kwargs=vision_kwargs, torch_dtype="bfloat16")
        model.eval(); model.tie_weights(); model.to(torch.bfloat16)
        lm = model.get_model()
        head, gamma64, eps = hss.prepare(lm, cuda)
        dla.install(lm)
        p6.install_p6(lm)
        lid = hss.letter_ids(tokenizer, LETTERS)
        ids = [lid[l] for l in LETTERS]
        Wout64 = [b.attn_out.weight.detach().to(torch.float64).cpu() for b in lm.transformer.blocks]
        device = model.device
        mask_id = base.MASK_ID_LAVIDA

        def make_item(text: str, image):
            dla.disarm(); p6.clear_p()
            past, plen = ccf.lavida_prefix_img(lm, model, processor, tokenizer, text, image, IMAGE_TOKEN_INDEX)

            def forward(canvas):
                logits = lm(None, input_embeddings=lm.transformer.wte(canvas), past_key_values=past,
                            use_cache=False).logits
                if logits.shape[1] != GEN:
                    raise RuntimeError(f"canvas logits {tuple(logits.shape)}")
                return logits

            return {"forward": forward, "answer_index": 0, "prefix_len": plen}
    elif backend == "dreamvl":
        from types import SimpleNamespace

        import backend_dreamvl_20261013g as bd

        dv = bd.Dreamvl(SimpleNamespace(model_path=str(DREAMVL_PATH), cuda=cuda))
        tokenizer, model, core = dv.tokenizer, dv.model, dv.core
        head, gamma64, eps = dv.prepare_readout()
        layers = list(core.layers)
        cfg = model.config
        if (int(cfg.num_hidden_layers), int(cfg.num_attention_heads)) != (N_LAYERS, N_HEADS):
            raise RuntimeError(f"dreamvl dims {cfg.num_hidden_layers}x{cfg.num_attention_heads} != {N_LAYERS}x{N_HEADS}")
        install_generic(layers, lambda L: L.self_attn.o_proj, lambda L: L.mlp.down_proj)
        ids = dv.letter_ids(LETTERS)
        Wout64 = [L.self_attn.o_proj.weight.detach().to(torch.float64).cpu() for L in layers]
        device = dv.device
        mask_id = dv.mask_id
        lm = core

        def make_item(text: str, image):
            dla.disarm(); p6.clear_p()
            pre = dv._prefix(text, image)
            fwd = dv.make_forward(pre)                    # canvas-only logits (1, GEN, vocab) with the Dream shift
            return {"forward": fwd, "answer_index": pre["plen"] - 1, "prefix_len": pre["plen"]}
    else:
        import lladav_backend_20261013c as lb

        lctx = lb.load(str(LLADAV_PATH), cuda)
        tokenizer, model, core = lctx["tokenizer"], lctx["model"], lctx["core"]
        head, gamma64, eps = lb.prepare_readout(lctx)
        layers = list(core.layers)
        install_generic(layers, lambda L: L.self_attn.o_proj, lambda L: L.mlp.down_proj)
        ids = lb.letter_ids(lctx, LETTERS)
        Wout64 = [L.self_attn.o_proj.weight.detach().to(torch.float64).cpu() for L in layers]
        device = lctx["device"]
        mask_id = lb.MASK_ID
        lm = core

        def make_item(text: str, image):
            dla.disarm(); p6.clear_p()
            pe, plen = lb.prefix(lctx, text, image)
            fwd = lb.make_forward(lctx, pe)
            return {"forward": fwd, "answer_index": plen, "prefix_len": plen}

    if ids != LETTER_IDS:
        raise RuntimeError(f"letter ids {ids}")
    W64 = torch.stack([head[i].detach().to(torch.float64).cpu() for i in ids])
    if W64.shape[1] != Wout64[0].shape[0] or Wout64[0].shape[0] != Wout64[0].shape[1]:
        raise RuntimeError(f"shapes W64 {tuple(W64.shape)} Wout {tuple(Wout64[0].shape)}")
    ctx = dict(tokenizer=tokenizer, core=lm, model=model, mask_id=mask_id, device=device, gamma64=gamma64, eps=eps,
               W64=W64, Wout64=Wout64, gen=GEN, make_item=make_item, backend=backend)
    print(f"{backend} ready mask={mask_id} d={W64.shape[1]} eps={eps}", flush=True)
    return ctx


# ----------------------------------------------------------------------------- mode: dla

def capture_margin(ctx, item, canvas, first: int, second: int) -> dict:
    """One forward with DLA capture at the answer position. Per-head and per-component margin contributions."""
    p6.clear_p()
    dla.arm(item["answer_index"], capture=True)
    logits = item["forward"](canvas)
    C, Z, final = dla.collect()
    model_letters = logits[0, 0, LETTER_IDS].detach().to(torch.float64).cpu()
    L, s = dla.dla_letters(C, ctx["gamma64"], ctx["W64"], ctx["eps"])              # (65, 5)
    HL = dla.head_letters(Z, ctx["Wout64"], ctx["gamma64"], ctx["W64"], s)        # (32, 32, 5)
    exact = dla.exact_letters(final, ctx["gamma64"], ctx["W64"], ctx["eps"])
    names = dla.comp_names()
    comp_margin = {n: float(L[i, first] - L[i, second]) for i, n in enumerate(names)}
    head_margin = (HL[:, :, first] - HL[:, :, second]).numpy()
    return {
        "model_margin": float(model_letters[first] - model_letters[second]),
        "exact_margin": float(exact[first] - exact[second]),
        "dla_sum_margin": float(sum(comp_margin.values())),
        "comp_margin": comp_margin,
        "head_margin": head_margin,
        "model_letters": [float(v) for v in model_letters],
    }


def commit_canvas(ctx, res: dict, commit: int) -> torch.Tensor:
    """Canvas the model saw at step `commit`: tokens unmasked before that step are stock, the rest are masks."""
    x = torch.tensor([res["final_ids"]], dtype=torch.long, device=ctx["device"])
    for pos, st in enumerate(res["unmask_step"]):
        if st >= commit:
            x[0, pos] = ctx["mask_id"]
    if int(x[0, 0].item()) != ctx["mask_id"]:
        raise RuntimeError("answer slot not masked at the commit canvas")
    return x


def this_run_eligible(backend: str, res: dict) -> tuple[bool, int, str | None]:
    """-> (eligible, commit_step, reason). lavida: letter at slot 0 at the last step. lladav: first token is a letter."""
    step = int(res["letter_unmask_step"])
    if backend in ("lavida", "dreamvl"):
        return step == GEN - 1, step, None if step == GEN - 1 else "not last step in this run"
    ok = res["finished_letter"] is not None
    return ok, step, None if ok else "first token is not a letter in this run"


def run_dla(args):
    backend, out_dir = args.backend, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_item = out_dir / "heads.jsonl"
    check_out(per_item)
    lock = p6.open_locked(per_item)
    rows = load_rows()
    select, _steer, prior = split_items(backend, rows, args.n_select, args.n_steer)
    if args.limit:
        select = select[: args.limit]
    done = {r["id"] for r in jsonl(per_item)}
    ctx = load_backend(backend, args.cuda)
    arrays_path = out_dir / "heads_arrays.npz"
    arr = dict(np.load(arrays_path, allow_pickle=True)) if arrays_path.exists() else {}
    t0 = time.time(); n_new = 0
    with per_item.open("a", encoding="utf-8") as handle:
        for row in select:
            if row["id"] in done:
                continue
            t1 = time.time()
            item = ctx["make_item"](text_of(backend, row), load_image(row))
            n = int(row["n_opt"])
            dla.disarm()
            res = full_decode(ctx, item, None, tag=f"stock {row['id']}")
            raw0 = res["pos0_letters"][0]
            first, second = p6.ranked(raw0, n)[:2]
            pr = prior[row["id"]]
            matches = (res["finished_letter"] == pr["finished_letter"]) and (res["letter_unmask_step"] == pr["letter_unmask_step"])
            ok, commit, reason = this_run_eligible(backend, res)
            if not ok:
                rec = {"id": row["id"], "skipped": reason, "letter_unmask_step": res["letter_unmask_step"],
                       "prior_unmask_step": pr["letter_unmask_step"], "stock_matches_prior": bool(matches),
                       "finished_text": res["finished_text"][:120]}
                p6.write_rec(handle, rec); done.add(row["id"]); continue
            canvas0 = p6.fresh(ctx)
            canvasc = commit_canvas(ctx, res, commit)
            m0 = capture_margin(ctx, item, canvas0, first, second)
            mc = capture_margin(ctx, item, canvasc, first, second)
            dla.disarm()
            arr[f"{row['id']}|step0"] = m0.pop("head_margin")
            arr[f"{row['id']}|commit"] = mc.pop("head_margin")
            rec = {"id": row["id"], "gold": row["gold"], "n_opt": n, "first": LETTERS[first], "second": LETTERS[second],
                   "final_letter": res["finished_letter"], "letter_unmask_step": res["letter_unmask_step"], "commit_step": commit,
                   "n_unmasked_at_commit": int(sum(1 for st in res["unmask_step"] if st < commit)),
                   "stock_matches_prior": bool(matches), "step0_raw": raw0, "step0": m0, "commit": mc,
                   "sec": round(time.time() - t1, 2)}
            p6.write_rec(handle, rec); done.add(row["id"]); n_new += 1
            np.savez(arrays_path, **arr)
            el = time.time() - t0
            print(f"PROGRESS dla {len(done)}/{len(select)} id={row['id']} first={LETTERS[first]} second={LETTERS[second]} "
                  f"final={res['finished_letter']} commit={commit} m0={m0['exact_margin']:.2f} mc={mc['exact_margin']:.2f} "
                  f"dla_err={abs(mc['dla_sum_margin'] - mc['exact_margin']):.3f} sec={rec['sec']} rate={n_new / el:.3f}/s",
                  flush=True)
    lock.close()
    rank_heads(out_dir)


def rank_heads(out_dir: Path) -> None:
    recs = [r for r in jsonl(out_dir / "heads.jsonl") if "skipped" not in r]
    skipped = [r for r in jsonl(out_dir / "heads.jsonl") if "skipped" in r]
    arr = np.load(out_dir / "heads_arrays.npz", allow_pickle=True)
    ids = [r["id"] for r in recs if f"{r['id']}|commit" in arr]
    M15 = np.stack([arr[f"{i}|commit"] for i in ids])
    M0 = np.stack([arr[f"{i}|step0"] for i in ids])
    mean15, mean0 = M15.mean(0), M0.mean(0)
    pos15 = (M15 > 0).mean(0)
    order15 = sorted(((float(mean15[l, h]), l + 1, h) for l in range(N_LAYERS) for h in range(N_HEADS)), reverse=True)
    order0 = sorted(((float(mean0[l, h]), l + 1, h) for l in range(N_LAYERS) for h in range(N_HEADS)), reverse=True)
    writers = [(l, h) for _v, l, h in order15[:N_WRITERS]]
    exclude = [f"L{l}.H{h}" for _v, l, h in order15[:EXCLUDE_TOP]]
    comp15 = {}
    for r in recs:
        for k, v in r["commit"]["comp_margin"].items():
            comp15.setdefault(k, []).append(v)
    comp15 = {k: float(np.mean(v)) for k, v in comp15.items()}
    layer_attn = {l: comp15[f"L{l}.attn"] for l in range(1, N_LAYERS + 1)}
    layer_mlp = {l: comp15[f"L{l}.mlp"] for l in range(1, N_LAYERS + 1)}
    total_head = float(mean15.sum())
    commit_steps = {}
    for r in recs:
        commit_steps[str(r.get("commit_step"))] = commit_steps.get(str(r.get("commit_step")), 0) + 1
    out = {
        "n_items": len(ids), "n_skipped": len(skipped), "ids_first_last": [ids[0], ids[-1]] if ids else None,
        "commit_step_hist": commit_steps,
        "criterion": "mean over held-out items of per-head DLA contribution to logit[first]-logit[second] at the commit-step canvas (tokens unmasked before the commit step = stock, rest masked); frozen-scale final-norm readout",
        "writer_heads": writers, "writer_heads_named": [f"L{l}.H{h}" for l, h in writers],
        "writer_layers": sorted({l for l, _h in writers}),
        "top10_commit": [{"head": f"L{l}.H{h}", "mean_margin": v, "frac_positive": float(pos15[l - 1, h]),
                          "mean_margin_step0": float(mean0[l - 1, h])} for v, l, h in order15[:10]],
        "bottom5_commit": [{"head": f"L{l}.H{h}", "mean_margin": v} for v, l, h in order15[-5:]],
        "top10_step0": [{"head": f"L{l}.H{h}", "mean_margin": v} for v, l, h in order0[:10]],
        "overlap_top10_step0_commit": sorted({f"L{l}.H{h}" for _v, l, h in order0[:10]} & {f"L{l}.H{h}" for _v, l, h in order15[:10]}),
        "writers_share_of_total_head_margin": float(sum(mean15[l - 1, h] for l, h in writers) / total_head) if total_head else None,
        "layer_attn_margin_commit": layer_attn, "layer_mlp_margin_commit": layer_mlp,
        "exclude_for_random": exclude,
        "mean_exact_margin_commit": float(np.mean([r["commit"]["exact_margin"] for r in recs])),
        "mean_exact_margin_step0": float(np.mean([r["step0"]["exact_margin"] for r in recs])),
        "mean_abs_dla_err_commit": float(np.mean([abs(r["commit"]["dla_sum_margin"] - r["commit"]["exact_margin"]) for r in recs])),
        "stock_matches_prior": int(sum(r["stock_matches_prior"] for r in recs)),
        "mmada_writer_heads_for_reference": p6.WRITER_HEADS,
    }
    (out_dir / "writer_heads.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("WRITERS", out["writer_heads_named"], "layers", out["writer_layers"],
          "share", out["writers_share_of_total_head_margin"], "commit_steps", commit_steps, flush=True)
    top_layers = sorted(layer_attn.items(), key=lambda kv: -kv[1])[:5]
    print("top attn layers", [(l, round(v, 3)) for l, v in top_layers], flush=True)


# ----------------------------------------------------------------------------- mode: steer

def pick_random(writers: list[tuple[int, int]], exclude: list[str]) -> list[tuple[int, int]]:
    rng = np.random.default_rng(SEED)
    layers = sorted({l for l, _h in writers})
    excl = set(exclude) | {f"L{l}.H{h}" for l, h in writers}
    cands = [(l, h) for l in layers for h in range(N_HEADS) if f"L{l}.H{h}" not in excl]
    idx = rng.choice(len(cands), size=N_WRITERS, replace=False)
    return sorted(cands[int(i)] for i in idx)


def calibrate(ctx, backend: str, items_rows: list[dict], steerers: dict[str, p6.Steerer]) -> dict:
    per = {g: {str(cc): [] for cc in CALIB_PROBES} for g in steerers}
    ids = []
    for row in items_rows:
        item = ctx["make_item"](text_of(backend, row), load_image(row))
        n = int(row["n_opt"])
        x = p6.fresh(ctx)

        def cap():
            p6.P.capture = {N_LAYERS}

        logits0, _ = p6.forward_once(ctx, item, x, cap)
        raw0 = p6.letters_of(logits0[0, 0])
        ex0 = exact_letters_from_capture(ctx)
        first, second = p6.ranked(raw0, n)[:2]
        diff0 = ex0[second] - ex0[first]
        for g, st in steerers.items():
            vec = st.vectors(first, second)
            for cc in CALIB_PROBES:
                def setup(_st=st, _vec=vec, _c=cc):
                    _st.setup(_vec, _c)(0, x)
                    p6.P.capture = {N_LAYERS}
                _lg, napp = p6.forward_once(ctx, item, x, setup)
                if napp != len(st.layers):
                    raise RuntimeError(f"calibration hook applications {napp} != {len(st.layers)}")
                r = exact_letters_from_capture(ctx)
                per[g][str(cc)].append((r[second] - r[first] - diff0) / cc)
        ids.append(row["id"])
    out = {"n_items": len(ids), "ids": ids, "probes": list(CALIB_PROBES), "groups": {}}
    for g in steerers:
        s1 = np.array(per[g][str(CALIB_PROBES[0])]); s2 = np.array(per[g][str(CALIB_PROBES[1])])
        out["groups"][g] = {"unit_scale_own": float(np.median(s1)), "slope_p1_iqr": [float(np.percentile(s1, 25)), float(np.percentile(s1, 75))],
                            "slope_p2_median": float(np.median(s2)), "slopes_p1": s1.tolist(), "slopes_p2": s2.tolist(),
                            **steerers[g].describe()}
    wh = out["groups"]["writer"]["unit_scale_own"]
    if not wh > 0:
        raise RuntimeError(f"writer unit_scale {wh} must be positive")
    for g in steerers:
        out["groups"][g]["unit_scale_used"] = wh
        out["groups"][g]["unit_scale_source"] = "writer (own)" if g == "writer" else "writer (matched residual norm)"
    return out


def run_steer(args):
    backend, out_dir = args.backend, args.out_dir
    heads_meta = json.loads((out_dir / "writer_heads.json").read_text(encoding="utf-8"))
    writers = [tuple(x) for x in heads_meta["writer_heads"]]
    randoms = pick_random(writers, heads_meta["exclude_for_random"])
    output = out_dir / "steer.jsonl"
    check_out(output)
    lock = p6.open_locked(output)
    rows = load_rows()
    _select, steer_rows, prior = split_items(backend, rows, args.n_select, args.n_steer)
    if args.limit:
        steer_rows = steer_rows[: args.limit]
    ctx = load_backend(backend, args.cuda)
    steerers = {"writer": p6.Steerer(ctx, "writer", heads=writers), "random": p6.Steerer(ctx, "random", heads=randoms)}
    calib_path = out_dir / "calib.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
    else:
        calib = calibrate(ctx, backend, steer_rows[: args.n_calib], steerers)
        calib["writer_heads"] = [f"L{l}.H{h}" for l, h in writers]
        calib["random_heads"] = [f"L{l}.H{h}" for l, h in randoms]
        calib_path.write_text(json.dumps(calib, indent=1), encoding="utf-8")
    unit = float(calib["groups"]["writer"]["unit_scale_used"])
    print(f"writers {calib['writer_heads']} randoms {calib['random_heads']} unit_scale={unit:.5f}", flush=True)
    if LC_WHEN == "commit":
        bcommit = BETA_STEP0 if BETA_STEP0 is not None else 8.0
        conds = [("writer", "commit", bcommit), ("random", "commit", bcommit)]
    else:
        conds = [("writer", "every", b) for b in BETAS_EVERY] + [("random", "every", b) for b in BETAS_EVERY]
        if BETA_STEP0 is not None:
            conds = [("writer", "step0", BETA_STEP0)] + conds + [("random", "step0", BETA_STEP0)]
    print(f"conditions {conds}", flush=True)
    done = {(r["id"], r["group"], r["when"], f"{r['beta']:g}") for r in jsonl(output)}
    by = {(r["id"], r["group"], r["when"], f"{r['beta']:g}"): r for r in jsonl(output)}
    t0 = time.time(); n_new = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in steer_rows:
            n = int(row["n_opt"])
            key_stock = (row["id"], "stock", "stock", "0")
            need = [k for k in [key_stock] + [(row["id"], g, w, f"{b:g}") for g, w, b in conds] if k not in done]
            if not need:
                continue
            t1 = time.time()
            item = ctx["make_item"](text_of(backend, row), load_image(row))
            dla.disarm()
            if key_stock in done:
                stock = by[key_stock]
            else:
                res = full_decode(ctx, item, None, tag=f"stock {row['id']}")
                raw0 = res["pos0_letters"][0]
                first, second = p6.ranked(raw0, n)[:2]
                pr = prior[row["id"]]
                ok, commit, _reason = this_run_eligible(backend, res)
                stock = {"id": row["id"], "gold": row["gold"], "n_opt": n, "group": "stock", "when": "stock", "beta": 0.0, "c": 0.0,
                         "first": LETTERS[first], "second": LETTERS[second], "step0_raw": raw0,
                         "final_letter": res["finished_letter"], "letter_unmask_step": res["letter_unmask_step"],
                         "last_step": int(res["letter_unmask_step"] == GEN - 1), "eligible": int(ok), "commit_step": commit,
                         "order": res["order"], "final_ids": res["final_ids"], "finished_text": res["finished_text"][:120],
                         "stock_matches_prior": bool(res["finished_letter"] == pr["finished_letter"] and res["letter_unmask_step"] == pr["letter_unmask_step"])}
                p6.write_rec(handle, stock); done.add(key_stock); by[key_stock] = stock; n_new += 1
            first, second = LETTERS.index(stock["first"]), LETTERS.index(stock["second"])
            for g, when, beta in conds:
                key = (row["id"], g, when, f"{beta:g}")
                if key in done:
                    continue
                st = steerers[g]
                vec = st.vectors(first, second)
                cc = beta / unit
                raw_setup = st.setup(vec, cc)

                commit_step = int(stock["commit_step"])

                def setup_fn(step, x, _raw=raw_setup, _when=when, _commit=commit_step):
                    if _when == "every" or (_when == "step0" and step == 0) or (_when == "commit" and step == _commit):
                        _raw(step, x)

                res = full_decode(ctx, item, setup_fn, tag=f"{g} {when} {beta} {row['id']}")
                final = res["finished_letter"]
                rec = {"id": row["id"], "gold": row["gold"], "n_opt": n, "group": g, "when": when, "beta": beta, "c": round(cc, 4),
                       "heads": [f"L{l}.H{h}" for l, h in st.heads], "first": stock["first"], "second": stock["second"],
                       "final_letter": final, "letter_unmask_step": res["letter_unmask_step"],
                       "edit_step": int(stock["commit_step"]) if when == "commit" else None,
                       "last_step": int(res["letter_unmask_step"] == GEN - 1),
                       "changed": int(final != stock["final_letter"]), "hit_target": int(final == stock["second"]),
                       "order_same_as_stock": int(res["order"] == stock["order"]), "order": res["order"],
                       "n_hook_applications": res["n_hook_applications"], "final_ids": res["final_ids"],
                       "finished_text": res["finished_text"][:120],
                       "step0_shift_second_minus_first": (res["pos0_letters"][0][second] - res["pos0_letters"][0][first])
                       - (stock["step0_raw"][second] - stock["step0_raw"][first])}
                p6.write_rec(handle, rec); done.add(key); by[key] = rec; n_new += 1
            el = time.time() - t0
            summ = {f"{g}/{w}/{b:g}": by[(row["id"], g, w, f"{b:g}")]["final_letter"] for g, w, b in conds}
            print(f"PROGRESS steer {row['id']} stock={stock['final_letter']} second={stock['second']} commit={stock.get('commit_step')} {summ} "
                  f"sec={time.time() - t1:.1f} rate={n_new / el:.3f}/s", flush=True)
    lock.close()
    analyze(out_dir)


# ----------------------------------------------------------------------------- analyze

def analyze(out_dir: Path) -> None:
    recs = jsonl(out_dir / "steer.jsonl")
    stock = {r["id"]: r for r in recs if r["group"] == "stock"}
    elig = {i for i, r in stock.items() if int(r.get("eligible", r.get("last_step", 0))) == 1}
    lines = [f"# causal write-site test ({out_dir.name}, MMStar held-out items)",
             f"n_stock={len(stock)} eligible_in_this_run={len(elig)} (lavida/dreamvl: letter at slot 0 written at the last step; lladav: first token is a letter)",
             f"stock_matches_prior_file={sum(r['stock_matches_prior'] for r in stock.values())}/{len(stock)}", ""]
    hm = json.loads((out_dir / "writer_heads.json").read_text()) if (out_dir / "writer_heads.json").exists() else {}
    cal = json.loads((out_dir / "calib.json").read_text()) if (out_dir / "calib.json").exists() else {}
    lines += [f"writer heads: {hm.get('writer_heads_named')}  random heads: {cal.get('random_heads')}  unit_scale={cal.get('groups', {}).get('writer', {}).get('unit_scale_used')}", ""]
    lines += ["| group | when | beta | n (eligible) | changed | hit target | order same | changed (all) |", "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for g in ("writer", "random"):
        for when in ("step0", "commit", "every"):
            for b in sorted({r["beta"] for r in recs if r["group"] == g and r["when"] == when}):
                rs = [r for r in recs if r["group"] == g and r["when"] == when and r["beta"] == b]
                ls = [r for r in rs if r["id"] in elig]
                ch = sum(r["changed"] for r in ls); hit = sum(r["hit_target"] for r in ls); same = sum(r["order_same_as_stock"] for r in ls)
                ch_all = sum(r["changed"] for r in rs)
                summary[f"{g}/{when}/{b:g}"] = {"n": len(ls), "changed": ch, "hit_target": hit, "order_same": same, "n_all": len(rs), "changed_all": ch_all}
                lines.append(f"| {g} | {when} | {b:g} | {len(ls)} | {ch} ({100 * ch / max(1, len(ls)):.1f}%) | {hit}/{ch} | {same} | {ch_all}/{len(rs)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"table": summary, "writer_heads": hm.get("writer_heads_named"),
                                                     "random_heads": cal.get("random_heads")}, indent=1), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("dla", "steer", "analyze"), required=True)
    p.add_argument("--backend", choices=("lavida", "lladav", "dreamvl"), default="lavida")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--n-select", type=int, default=200)
    p.add_argument("--n-steer", type=int, default=200)
    p.add_argument("--n-calib", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="smoke: only this many items")
    p.add_argument("--dataset", choices=("mmstar", "sciqa"), default="mmstar")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    global DATASET
    DATASET = args.dataset
    set_dims(args.backend)
    args.out_dir = (args.out_dir or out_root(args.backend)).resolve()
    check_out(args.out_dir)
    if args.mode == "dla":
        run_dla(args)
    elif args.mode == "steer":
        run_steer(args)
    else:
        analyze(args.out_dir)


if __name__ == "__main__":
    main()
