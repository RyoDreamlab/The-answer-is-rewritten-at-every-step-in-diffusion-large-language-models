#!/usr/bin/env python3
"""Upgrade 2 — direct logit attribution (DLA) of the option-letter prior at the answer position, and causal ablation.

Residual-stream decomposition at the answer position (canvas index 0, decode step 0, whole canvas masked):

    h_final = embed + sum_{l=1..32} ( attn_l + mlp_l )            (65 components; LLaDALlamaBlock adds exactly
                                                                   attn_out(.) and ff_out(.) to the residual, eval
                                                                   dropout = identity, no biases)

Each component c is read out through the FINAL ln_f and the vocab head restricted to letters A..E with the standard
"frozen-scale" approximation: the RMS scale  s = rsqrt(mean(h_final^2) + eps)  is computed once from the FULL final
residual and every component is scaled by the same  s * gamma  before the head (TransformerLens-style
`apply_ln_to_stack` / nostalgebraist logit-lens convention).  Given s, the readout is linear, so the 65 component
letter-logit vectors sum exactly to the model's letter logits (checked against the bf16 model output).  The
approximation is only that the nonlinearity of the RMS scale is attributed jointly (not per component).

Prior direction per option count n:  u_n = centered( log p_cf[:n] )  (p_cf from outputs/calib_vote_20261012/cf_{model}
cond a_global).  Component prior contribution  =  < centered(letter_logits_c[:n]) , u_n > / ||u_n||.
On the blank input the final letter logits ARE the content-free logits, so  sum_c contribution_c = ||u_n||  exactly.

Head-level: attn_l = sum_h  z_{l,h} @ W_out[:, h-slice]^T  (attn_out has no bias), so the attention component is split
into 32 per-head components at zero extra cost (forward pre-hook on attn_out).

Modes
  dla     blank input per n in 2..5 + first --n-real real items: components, heads, cumulative curves, blank ablations
  decode  stock 16-step low-confidence decode with the ablation hook applied at the answer position at EVERY step
  report  CPU: summary.json + figures dla_{model}.png, dla_ablation.png

New files only; outputs under outputs/dla_circuit_20261013d/.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import calib_contentfree as ccf  # noqa: E402
import hidden_shapley_split as hss  # noqa: E402
import image_key_block as ikb  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "dla_circuit_20261013d").resolve()
CF_DIR = PROJECT / "outputs" / "calib_vote_20261012"
STOCK_DIR = PROJECT / "outputs" / "letter_argmax_traj_20261010"
LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]
N_LAYERS = 32
N_HEADS = 32


# ----------------------------------------------------------------------------- hooks

class H:
    armed = False
    index = 0
    comps: dict[str, torch.Tensor] = {}
    heads: dict[int, torch.Tensor] = {}
    final: torch.Tensor | None = None
    ablate: dict[str, torch.Tensor | str] = {}      # 'L{l}.attn' / 'L{l}.mlp' -> 'zero' or mean vector (d,) on device
    head_ablate: set[tuple[int, int]] = set()        # (layer 1-based, head) -> zero that head's attn_out input slice
    capture = True
    n_ablations_applied = 0


def comp_names() -> list[str]:
    names = ["embed"]
    for l in range(1, N_LAYERS + 1):
        names += [f"L{l}.attn", f"L{l}.mlp"]
    return names


def install(core) -> int:
    blocks = list(core.transformer.blocks)
    if len(blocks) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} blocks, got {len(blocks)}")
    for name in ("attn_out", "ff_out"):
        if getattr(blocks[0], name).bias is not None:
            raise RuntimeError(f"{name} has a bias; DLA split assumes none")
    if type(blocks[0]).__name__ != "LLaDALlamaBlock":
        raise RuntimeError(f"expected LLaDALlamaBlock, got {type(blocks[0]).__name__}")

    def embed_pre(_m, args):
        if H.armed and H.capture:
            H.comps["embed"] = args[0][0, H.index].detach()

    blocks[0].register_forward_pre_hook(embed_pre)

    for li, block in enumerate(blocks):
        l = li + 1

        def attn_pre(_m, args, _l=l):
            if not H.armed:
                return None
            z = args[0]
            if H.capture:
                H.heads[_l] = z[0, H.index].detach()
            todo = [h for (ll, h) in H.head_ablate if ll == _l]
            if todo:
                z = z.clone()
                hs = z.shape[-1] // N_HEADS
                for h in todo:
                    z[0, H.index, h * hs:(h + 1) * hs] = 0
                H.n_ablations_applied += 1
                return (z,)
            return None

        def attn_post(_m, _args, out, _l=l):
            if not H.armed:
                return None
            if H.capture:
                H.comps[f"L{_l}.attn"] = out[0, H.index].detach()
            a = H.ablate.get(f"L{_l}.attn")
            if a is None:
                return None
            out = out.clone()
            out[0, H.index] = 0 if isinstance(a, str) else a.to(out.dtype)
            H.n_ablations_applied += 1
            return out

        def mlp_post(_m, _args, out, _l=l):
            if not H.armed:
                return None
            if H.capture:
                H.comps[f"L{_l}.mlp"] = out[0, H.index].detach()
            a = H.ablate.get(f"L{_l}.mlp")
            if a is None:
                return None
            out = out.clone()
            out[0, H.index] = 0 if isinstance(a, str) else a.to(out.dtype)
            H.n_ablations_applied += 1
            return out

        block.attn_out.register_forward_pre_hook(attn_pre)
        block.attn_out.register_forward_hook(attn_post)
        block.ff_out.register_forward_hook(mlp_post)

    def final_post(_m, _args, out):
        if H.armed and H.capture:
            x = out[0] if isinstance(out, (tuple, list)) else out
            H.final = x[0, H.index].detach()

    blocks[-1].register_forward_hook(final_post)
    return len(blocks)


def arm(index: int, capture: bool = True) -> None:
    H.armed = True
    H.index = int(index)
    H.capture = capture
    H.comps = {}
    H.heads = {}
    H.final = None
    H.n_ablations_applied = 0


def disarm() -> None:
    H.armed = False


def collect() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> C (65, d) float64 cpu, Z (32, d) attn_out inputs float64 cpu, final (d,) float64 cpu."""
    disarm()
    names = comp_names()
    missing = [n for n in names if n not in H.comps]
    if missing:
        raise RuntimeError(f"missing components {missing[:4]}")
    C = torch.stack([H.comps[n].to(torch.float64).cpu() for n in names])
    Z = torch.stack([H.heads[l].to(torch.float64).cpu() for l in range(1, N_LAYERS + 1)])
    return C, Z, H.final.to(torch.float64).cpu()


# ----------------------------------------------------------------------------- readout

def frozen_scale(h: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt((h * h).mean() + eps)


def dla_letters(C: torch.Tensor, gamma64: torch.Tensor, W64: torch.Tensor, eps: float) -> tuple[torch.Tensor, float]:
    h = C.sum(0)
    s = frozen_scale(h, eps)
    return (C * (gamma64 * s)) @ W64.T, float(s)


def exact_letters(h: torch.Tensor, gamma64: torch.Tensor, W64: torch.Tensor, eps: float) -> torch.Tensor:
    return (gamma64 * h * frozen_scale(h, eps)) @ W64.T


def head_letters(Z: torch.Tensor, Wout64: list[torch.Tensor], gamma64, W64, s: float) -> torch.Tensor:
    """Z (32 layers, d) attn_out inputs -> (32, 32 heads, 5) letter logits with the frozen scale s."""
    hs = Z.shape[1] // N_HEADS
    out = torch.zeros(N_LAYERS, N_HEADS, 5, dtype=torch.float64)
    for li in range(N_LAYERS):
        zh = Z[li].view(N_HEADS, hs)                                  # (32, hs)
        Wl = Wout64[li]                                               # (d, d)  out = z @ Wl.T
        per_head = torch.stack([zh[h] @ Wl[:, h * hs:(h + 1) * hs].T for h in range(N_HEADS)])  # (32, d)
        out[li] = (per_head * (gamma64 * s)) @ W64.T
    return out


def prior_dir(logp: list[float], n: int) -> torch.Tensor:
    u = torch.tensor(logp[:n], dtype=torch.float64)
    return u - u.mean()


def contrib(letters: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """letters (..., 5) -> projection of the centered first-n letter logits onto u/||u||."""
    n = u.numel()
    x = letters[..., :n]
    x = x - x.mean(dim=-1, keepdim=True)
    return (x @ u) / u.norm()


def kl_to_uniform(letters: torch.Tensor, n: int) -> float:
    p = torch.softmax(letters[:n], dim=0)
    return float((p * torch.log(p.clamp_min(1e-300))).sum() + math.log(n))


def load_logp(backend: str) -> dict[int, list[float]]:
    recs = [json.loads(l) for l in (CF_DIR / f"cf_{backend}.jsonl").open(encoding="utf-8") if l.strip()]
    out, raw = {}, {}
    for r in recs:
        if r["cond"] == "a_global":
            n = r["n_opt"]
            lg = torch.tensor(r["letter_logit64"][:n], dtype=torch.float64)
            out[n] = [float(v) for v in torch.log_softmax(lg, dim=0)]
            raw[n] = r["letter_logit64"]
    if sorted(out) != [2, 3, 4, 5]:
        raise RuntimeError(f"priors {sorted(out)}")
    return out, raw


def n_choices(row) -> int:
    c = row["choices"]
    if isinstance(c, str):
        c = ast.literal_eval(c)
    return len(c)


# ----------------------------------------------------------------------------- model loading (verbatim from layer_logit_lens_20261013)

def load_backend(args):
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    gen = base.GEN
    ctx = {}
    if args.backend == "lavida":
        sys.path.insert(0, str(PROJECT / "repos" / "LaViDa"))
        from PIL import Image
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.model.builder import load_pretrained_model

        siglip = (PROJECT / "hf-cache/hub/models--google--siglip-so400m-patch14-384/snapshots/"
                  "9fdffc58afc957d1a03a25b10dba0329ab15c2a3")
        vision_kwargs = {"mm_vision_tower": str(siglip) if siglip.exists() else "google/siglip-so400m-patch14-384",
                         "mm_resampler_type": None, "mm_projector_type": "mlp2x_gelu", "mm_hidden_size": 1152,
                         "use_mm_proj": True}
        print("loading lavida", flush=True)
        tokenizer, model, processor, _ = load_pretrained_model(
            args.model_path, None, "llava_llada", device_map=f"cuda:{args.cuda}",
            vision_kwargs=vision_kwargs, torch_dtype="bfloat16")
        model.eval(); model.tie_weights(); model.to(torch.bfloat16)
        lm = model.get_model()
        core = lm
        head, gamma64, eps = hss.prepare(lm, args.cuda)
        install(lm)
        mask_id = base.MASK_ID_LAVIDA
        device = model.device

        def prefix(q: str, image):
            disarm()
            return ccf.lavida_prefix_img(lm, model, processor, tokenizer, q, image, IMAGE_TOKEN_INDEX)

        def canvas_forward(past, canvas):
            embeds = lm.transformer.wte(canvas)
            logits = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
            if logits.shape[1] != gen:
                raise RuntimeError(f"canvas logits {tuple(logits.shape)}")
            return logits

        def make_item(q, image):
            past, prefix_len = prefix(q, image)
            return {"forward": (lambda canvas: canvas_forward(past, canvas)), "answer_index": 0,
                    "prefix_len": prefix_len}

        def real_item(row):
            return make_item(ikb.question_lavida(row), Image.open(row["image"]).convert("RGB"))

        def blank_item(n):
            return make_item(ccf.lavida_question_na(n), ccf.gray(ccf.GLOBAL_SIZE))
    else:
        sys.path.insert(0, str(PROJECT / "experiments"))
        sys.path.insert(0, str(PROJECT / "repos" / "MMaDA"))
        import evdlc_mmada as mm
        import sciqa_standard_eval as sq

        class Ns:
            model = args.model_path
            vq_model = args.vq_model
            mmada_repo = args.mmada_repo

        print("loading mmada", flush=True)
        device = torch.device(f"cuda:{args.cuda}")
        model, tokenizer = mm.load_mmada(Ns, device)
        vq = mm.load_vq(Ns, device)
        model.eval()
        core = model.model
        head, gamma64, eps = hss.prepare(core, args.cuda)
        install(core)
        mask_id = mm.MASK_ID
        loader = None
        for name in dir(sq):
            obj = getattr(sq, name)
            if isinstance(obj, type) and hasattr(obj, "load_rgb"):
                loader = obj(Path(args.data_dir), args.rows)
                break
        if loader is None:
            raise RuntimeError("ScienceQA image loader was not found")

        def make_item(question, image):
            disarm()
            vq_ids = mm.encode_vq(vq, image, device)
            prompt_ids, _ = mm.build_prompt_ids(tokenizer, question, vq_ids, device)
            prompt_len = int(prompt_ids.shape[1])

            def forward(canvas):
                logits = base.mmada_logits(model, prompt_ids, canvas, mask_id, None)
                if logits.shape[1] != prompt_len + gen:
                    raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
                return logits[:, prompt_len:, :]

            return {"forward": forward, "answer_index": prompt_len, "prefix_len": prompt_len}

        def real_item(row):
            return make_item(sq.build_task_prompt(row), loader.load_rgb(row))

        def blank_item(n):
            return make_item(sq.build_task_prompt({"question": "N/A", "choices": ["N/A"] * n, "hint": ""}),
                             ccf.gray(ccf.GLOBAL_SIZE))

    lid = hss.letter_ids(tokenizer, LETTERS)
    letter_ids = [lid[l] for l in LETTERS]
    if letter_ids != LETTER_IDS:
        raise RuntimeError(f"letter ids {letter_ids}")
    W64 = torch.stack([head[i].detach().to(torch.float64).cpu() for i in letter_ids])
    Wout64 = [b.attn_out.weight.detach().to(torch.float64).cpu() for b in core.transformer.blocks]
    ctx.update(dict(tokenizer=tokenizer, core=core, mask_id=mask_id, device=device, gamma64=gamma64, eps=eps,
                    W64=W64, Wout64=Wout64, real_item=real_item, blank_item=blank_item, gen=gen))
    return ctx


def step0_forward(ctx, item, capture=True):
    """Full-masked canvas, one forward; returns model letter logits (5,) float64 and the captured C/Z/final."""
    canvas = torch.full((1, ctx["gen"]), ctx["mask_id"], dtype=torch.long, device=ctx["device"])
    arm(item["answer_index"], capture=capture)
    logits = item["forward"](canvas)
    model_letters = logits[0, 0, LETTER_IDS].detach().to(torch.float64).cpu()
    if capture:
        C, Z, final = collect()
        return model_letters, C, Z, final
    disarm()
    return model_letters, None, None, None


# ----------------------------------------------------------------------------- mode: dla

def run_dla(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"dla_{args.backend}.json"
    out_npz = out_dir / f"dla_{args.backend}_arrays.npz"
    if out_json.exists() or out_npz.exists():
        raise RuntimeError(f"refusing to overwrite {out_json} / {out_npz}")
    rows = ikb.load_manifest(args)
    args.rows = rows
    rows = rows[: args.n_real]
    ctx = load_backend(args)
    gamma64, eps, W64, Wout64 = ctx["gamma64"], ctx["eps"], ctx["W64"], ctx["Wout64"]
    logp, cf_raw = load_logp(args.backend)
    names = comp_names()
    t0 = time.time()
    res = {"backend": args.backend, "components": names, "n_layers": N_LAYERS, "n_heads": N_HEADS,
           "readout": "frozen-scale ln_f: every component scaled by rsqrt(mean(h_final^2)+eps)*gamma from the FULL "
                      "final residual, then ff_out rows of A..E; linear given the scale, sums to model letter logits",
           "prior_direction": "u_n = centered log p_cf over the item's n letters (calib_vote_20261012 cf a_global); "
                              "contribution = <centered component letter logits[:n], u_n>/||u_n||",
           "log_p_cf": {str(n): v for n, v in logp.items()}, "blank": {}, "real": {}, "ablation_blank": {}}

    # ---------------- blank input, n = 2..5
    blank_C = {}
    for n in (2, 3, 4, 5):
        item = ctx["blank_item"](n)
        model_letters, C, Z, final = step0_forward(ctx, item)
        L, s = dla_letters(C, gamma64, W64, eps)                     # (65, 5)
        u = prior_dir(logp[n], n)
        c = contrib(L, u)                                            # (65,)
        total = L.sum(0)
        recon_err = float((total - model_letters).abs().max())
        sum_err = float((C.sum(0) - final).abs().max())
        cf_err = float((model_letters - torch.tensor(cf_raw[n], dtype=torch.float64)).abs().max())
        HL = head_letters(Z, Wout64, gamma64, W64, s)                # (32, 32, 5)
        hc = contrib(HL, u)                                          # (32, 32)
        # head split check: sum of heads == attn component
        attn_rows = torch.stack([L[1 + 2 * li] for li in range(N_LAYERS)])
        head_sum_err = float((HL.sum(1) - attn_rows).abs().max())
        # cumulative curves
        cum_dla, cum_kl, cum_exact_proj = [], [], []
        for l in range(0, N_LAYERS + 1):
            k = 1 + 2 * l
            cum_dla.append(float(c[:k].sum()))
            h_l = C[:k].sum(0)
            el = exact_letters(h_l, gamma64, W64, eps)
            cum_kl.append(kl_to_uniform(el, n))
            cum_exact_proj.append(float(contrib(el, u)))
        order = torch.argsort(c, descending=True)
        # "deep prior" direction: centered letter logits written by embed + layers 1..31 (everything before the last
        # block).  Projections onto it show whether layer-32 components reinforce (positive) or cancel (negative) the
        # prior that the deep layers wrote — independent of how flat the FINAL distribution is.
        d31 = L[: 1 + 2 * (N_LAYERS - 1)].sum(0)[:n]
        d31 = d31 - d31.mean()
        c31 = contrib(L, d31)
        order31 = torch.argsort(c31, descending=True)
        l32 = L[names.index("L32.attn")] + L[names.index("L32.mlp")]
        l32c = l32[:n] - l32[:n].mean()
        top_heads = torch.topk(hc.flatten(), 5)
        bot_heads = torch.topk(-hc.flatten(), 5)
        res["blank"][str(n)] = {
            "model_letter_logits": model_letters.tolist(), "cf_prior_letter_logits": cf_raw[n],
            "max_abs_diff_vs_cf_prior": cf_err, "dla_sum_vs_model_max_abs_diff": recon_err,
            "component_sum_vs_final_residual_max_abs_diff": sum_err, "frozen_scale": s,
            "prior_norm": float(u.norm()), "sum_contrib": float(c.sum()), "kl_to_uniform_final": kl_to_uniform(total, n),
            "contrib": c.tolist(), "letter_logits_per_component": L.tolist(),
            "top_writers": [(names[int(i)], float(c[int(i)])) for i in order[:5]],
            "top_cancellers": [(names[int(i)], float(c[int(i)])) for i in reversed(order[-5:])],
            "L32": {"attn": float(c[names.index("L32.attn")]), "mlp": float(c[names.index("L32.mlp")])},
            "deep31_prior_norm": float(d31.norm()), "contrib_deep31": c31.tolist(),
            "top_writers_deep31": [(names[int(i)], float(c31[int(i)])) for i in order31[:5]],
            "top_cancellers_deep31": [(names[int(i)], float(c31[int(i)])) for i in reversed(order31[-5:])],
            "L32_vs_deep31": {"attn": float(c31[names.index("L32.attn")]), "mlp": float(c31[names.index("L32.mlp")]),
                              "cos_L32sum_deep31": float((l32c @ d31) / (l32c.norm() * d31.norm() + 1e-12)),
                              "fraction_of_deep_prior_cancelled_by_L32": float(-(l32c @ d31) / (d31.norm() ** 2 + 1e-12)),
                              "kl_to_uniform_through_L31": cum_kl[N_LAYERS - 1], "kl_to_uniform_final": cum_kl[N_LAYERS]},
            "cumulative_dla_prior": cum_dla, "cumulative_exact_lens_kl_to_uniform": cum_kl,
            "cumulative_exact_lens_prior_proj": cum_exact_proj,
            "head_contrib": hc.tolist(), "head_sum_vs_attn_component_max_abs_diff": head_sum_err,
            "top_heads": [(f"L{int(i) // N_HEADS + 1}.H{int(i) % N_HEADS}", float(v)) for v, i in
                          zip(top_heads.values, top_heads.indices)],
            "bottom_heads": [(f"L{int(i) // N_HEADS + 1}.H{int(i) % N_HEADS}", float(-v)) for v, i in
                             zip(bot_heads.values, bot_heads.indices)],
        }
        blank_C[n] = C
        print(f"BLANK n={n} recon={recon_err:.4f} cf_diff={cf_err:.4f} sum_err={sum_err:.4f} |u|={float(u.norm()):.3f} "
              f"sum_c={float(c.sum()):.3f} KL={kl_to_uniform(total, n):.3f} "
              f"top={res['blank'][str(n)]['top_writers'][:3]} bottom={res['blank'][str(n)]['top_cancellers'][:3]}\n"
              f"      deep31 |d|={float(d31.norm()):.3f} KL31={cum_kl[N_LAYERS - 1]:.3f} "
              f"top31={res['blank'][str(n)]['top_writers_deep31'][:3]} bottom31={res['blank'][str(n)]['top_cancellers_deep31'][:3]} "
              f"L32={res['blank'][str(n)]['L32_vs_deep31']}", flush=True)

    # ---------------- real inputs
    n_items = len(rows)
    real_contrib = np.zeros((n_items, len(names)))
    real_letters = np.zeros((n_items, len(names), 5))
    real_head_contrib = np.zeros((n_items, N_LAYERS, N_HEADS))
    real_final_proj = np.zeros(n_items)
    real_model_letters = np.zeros((n_items, 5))
    real_n = np.zeros(n_items, dtype=int)
    real_gold = []
    real_ids = []
    mean_vec = torch.zeros(len(names), blank_C[2].shape[1], dtype=torch.float64)
    recon_errs = []
    for k, row in enumerate(rows):
        n = n_choices(row)
        item = ctx["real_item"](row)
        model_letters, C, Z, final = step0_forward(ctx, item)
        L, s = dla_letters(C, gamma64, W64, eps)
        u = prior_dir(logp[n], n)
        c = contrib(L, u)
        HL = head_letters(Z, Wout64, gamma64, W64, s)
        real_contrib[k] = c.numpy()
        real_letters[k] = L.numpy()
        real_head_contrib[k] = contrib(HL, u).numpy()
        real_final_proj[k] = float(contrib(L.sum(0), u))
        real_model_letters[k] = model_letters.numpy()
        real_n[k] = n
        real_gold.append(ikb.gold_of(row))
        real_ids.append(row["id"])
        mean_vec += C
        recon_errs.append(float((L.sum(0) - model_letters).abs().max()))
        if (k + 1) % 25 == 0 or k + 1 == n_items:
            el = time.time() - t0
            base.write_progress(args.progress, {"backend": args.backend, "phase": "real", "n_done": k + 1,
                                                "n_total": n_items, "elapsed_sec": round(el, 1)})
            print(f"REAL {k + 1}/{n_items} elapsed={el:.0f}s recon_max={max(recon_errs):.4f}", flush=True)
    mean_vec /= n_items
    mean_c = real_contrib.mean(0)
    order = np.argsort(-mean_c)
    res["real"] = {
        "n_items": n_items, "ids_first_last": [real_ids[0], real_ids[-1]], "n_opt_counts": {str(int(v)): int((real_n == v).sum()) for v in sorted(set(real_n.tolist()))},
        "dla_sum_vs_model_max_abs_diff": max(recon_errs),
        "mean_contrib": mean_c.tolist(), "std_contrib": real_contrib.std(0).tolist(),
        "top_writers_mean": [(names[int(i)], float(mean_c[int(i)])) for i in order[:5]],
        "top_cancellers_mean": [(names[int(i)], float(mean_c[int(i)])) for i in order[::-1][:5]],
        "mean_final_prior_proj": float(real_final_proj.mean()),
        "mean_head_contrib": real_head_contrib.mean(0).tolist(),
    }
    # correlation of per-item final prior projection with the sum of the blank-identified writers
    corr = {}
    for n in (2, 3):
        writers = [nm for nm, _ in res["blank"][str(n)]["top_writers"][:3]]
        idx = [names.index(w) for w in writers]
        ws = real_contrib[:, idx].sum(1)
        sel = real_n == n
        entry = {"writers": writers, "n_items_all": int(n_items),
                 "pearson_all_items": float(np.corrcoef(ws, real_final_proj)[0, 1])}
        if sel.sum() >= 3:
            entry["pearson_items_with_this_n"] = float(np.corrcoef(ws[sel], real_final_proj[sel])[0, 1])
            entry["n_items_with_this_n"] = int(sel.sum())
        # per-letter logit correlation for the top letter of the prior
        top_letter = int(np.argmax(logp[n][:n]))
        wl = real_letters[:, idx, :].sum(1)
        cl = wl[:, :n] - wl[:, :n].mean(1, keepdims=True)
        fl = real_model_letters[:, :n] - real_model_letters[:, :n].mean(1, keepdims=True)
        entry["pearson_centered_logit_of_prior_top_letter"] = float(np.corrcoef(cl[:, top_letter], fl[:, top_letter])[0, 1])
        corr[str(n)] = entry
    # same for the real-input top-3 writers
    idx = [int(i) for i in order[:3]]
    corr["real_top3"] = {"writers": [names[i] for i in idx],
                         "pearson_all_items": float(np.corrcoef(real_contrib[:, idx].sum(1), real_final_proj)[0, 1])}
    res["real"]["correlation_final_vs_writer_sum"] = corr

    # ---------------- blank ablations (zero and real-mean), answer position only
    cands = []
    for n in (2, 3):
        b = res["blank"][str(n)]
        cands += [nm for nm, _ in b["top_writers"][:3]] + [nm for nm, _ in b["top_cancellers"][:3]]
        cands += [nm for nm, _ in b["top_cancellers_deep31"][:2]]
    cands += ["L32.attn", "L32.mlp"]
    seen, cand_list = set(), []
    for c_ in cands:
        if c_ not in seen and c_ != "embed":
            seen.add(c_); cand_list.append([c_])
    for n in (2, 3):
        cand_list.append([nm for nm, _ in res["blank"][str(n)]["top_writers"][:3]])
    head_cands = []
    for n in (2, 3):
        head_cands += [h for h, _ in res["blank"][str(n)]["top_heads"][:3]] + [h for h, _ in res["blank"][str(n)]["bottom_heads"][:2]]
    head_cands = list(dict.fromkeys(head_cands))
    mean_table = {nm: mean_vec[i].to(ctx["device"]) for i, nm in enumerate(names)}
    abl = {}
    for n in (2, 3, 4, 5):
        item = ctx["blank_item"](n)
        before, _, _, _ = step0_forward(ctx, item, capture=False)
        entry = {"before": {"letters": before.tolist(), "kl_to_uniform": kl_to_uniform(before, n),
                            "p": torch.softmax(before[:n], 0).tolist()}, "runs": []}
        for comps in cand_list:
            for kind in ("zero", "mean"):
                H.ablate = {c_: ("zero" if kind == "zero" else mean_table[c_]) for c_ in comps}
                after, _, _, _ = step0_forward(ctx, item, capture=False)
                napp = H.n_ablations_applied
                H.ablate = {}
                entry["runs"].append({"components": comps, "kind": kind, "letters": after.tolist(),
                                      "p": torch.softmax(after[:n], 0).tolist(),
                                      "kl_to_uniform": kl_to_uniform(after, n),
                                      "delta_kl": kl_to_uniform(after, n) - kl_to_uniform(before, n),
                                      "argmax": LETTERS[int(torch.argmax(after[:n]))],
                                      "prior_proj": float(contrib(after, prior_dir(logp[n], n))),
                                      "n_hook_applications": napp})
        for hname in head_cands:
            l = int(hname.split(".")[0][1:]); h = int(hname.split("H")[1])
            H.head_ablate = {(l, h)}
            after, _, _, _ = step0_forward(ctx, item, capture=False)
            H.head_ablate = set()
            entry["runs"].append({"components": [hname], "kind": "zero_head", "letters": after.tolist(),
                                  "p": torch.softmax(after[:n], 0).tolist(), "kl_to_uniform": kl_to_uniform(after, n),
                                  "delta_kl": kl_to_uniform(after, n) - kl_to_uniform(before, n),
                                  "argmax": LETTERS[int(torch.argmax(after[:n]))],
                                  "prior_proj": float(contrib(after, prior_dir(logp[n], n)))})
        abl[str(n)] = entry
        best = sorted(entry["runs"], key=lambda r: r["delta_kl"])
        print(f"ABL n={n} KL_before={entry['before']['kl_to_uniform']:.3f} most_flattening={[(r['components'], r['kind'], round(r['kl_to_uniform'], 3)) for r in best[:3]]} "
              f"most_sharpening={[(r['components'], r['kind'], round(r['kl_to_uniform'], 3)) for r in best[-3:]]}", flush=True)
    res["ablation_blank"] = abl
    res["elapsed_sec"] = round(time.time() - t0, 1)
    res["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_allocated(ctx["device"]) / 2 ** 30, 2)
    print(f"PEAK_GPU_MEM {res['peak_gpu_mem_gib']} GiB", flush=True)
    np.savez_compressed(out_npz, real_contrib=real_contrib, real_letters=real_letters, real_head_contrib=real_head_contrib,
                        real_final_proj=real_final_proj, real_model_letters=real_model_letters, real_n=real_n,
                        real_ids=np.array(real_ids), real_gold=np.array([g or "" for g in real_gold]),
                        mean_component_vectors=mean_vec.numpy().astype(np.float32),
                        blank_C_n2=blank_C[2].numpy().astype(np.float32), blank_C_n3=blank_C[3].numpy().astype(np.float32),
                        component_names=np.array(names))
    out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    base.write_progress(args.progress, {"backend": args.backend, "phase": "done", "elapsed_sec": res["elapsed_sec"]})
    print(f"DONE wrote {out_json} and {out_npz} in {res['elapsed_sec']}s", flush=True)


# ----------------------------------------------------------------------------- mode: decode with ablation

def run_decode(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (("-".join(args.components) + "_" + args.ablation) if args.components else "noablation")
    output = out_dir / f"decode_{args.backend}_{tag}.jsonl"
    progress = args.progress
    lock = output.with_suffix(output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {output}") from exc
    ikb.repair_tail(output)
    done = {r["id"] for r in ikb.load_jsonl(output)}
    rows = ikb.load_manifest(args)
    args.rows = rows
    rows = rows[: args.limit] if args.limit else rows
    ctx = load_backend(args)
    gen, mask_id, tokenizer = ctx["gen"], ctx["mask_id"], ctx["tokenizer"]
    stock = {r["id"]: r for r in ikb.load_jsonl(STOCK_DIR / f"{args.backend}.jsonl")}
    # ablation table
    abl = {}
    if args.components:
        if args.ablation == "mean":
            z = np.load(out_dir / f"dla_{args.backend}_arrays.npz")
            names = [str(x) for x in z["component_names"]]
            mv = torch.from_numpy(z["mean_component_vectors"]).to(ctx["device"])
            for c_ in args.components:
                abl[c_] = mv[names.index(c_)]
        else:
            for c_ in args.components:
                abl[c_] = "zero"
    print(f"start decode backend={args.backend} ablation={args.ablation} components={args.components} n={len(rows)} "
          f"done={len(done)} output={output}", flush=True)
    t0 = time.time(); n_new = 0; n_match = 0; n_cmp = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            n = n_choices(row)
            item = ctx["real_item"](row)
            x = torch.full((1, gen), mask_id, dtype=torch.long, device=ctx["device"])
            unmask_step = [-1] * gen
            pos0_argmax, pos0_letters = [], []
            step0 = None
            napp = 0
            for step in range(gen):
                arm(item["answer_index"], capture=False)
                H.ablate = dict(abl)
                logits = item["forward"](x)
                H.ablate = {}
                napp += H.n_ablations_applied
                disarm()
                if int(x[0, 0].item()) == int(mask_id):
                    vals = [float(v) for v in logits[0, 0, LETTER_IDS]]
                    pos0_letters.append(vals)
                    pos0_argmax.append(base.piece(tokenizer, int(torch.argmax(logits[0, 0]).item()), mask_id))
                    if step == 0:
                        step0 = LETTERS[max(range(n), key=lambda j: (vals[j], -j))]
                pos = base.unmask_one(x, logits, mask_id)
                unmask_step[pos] = step
            if sorted(unmask_step) != list(range(gen)):
                raise RuntimeError(f"unmask steps {unmask_step}")
            pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
            rec = {"id": row["id"], "gold": ikb.gold_of(row), "backend": args.backend, "n_opt": n,
                   "ablation": args.ablation if args.components else None, "components": args.components,
                   "letter_unmask_step": unmask_step[0], "pos0_argmax": pos0_argmax, "pos0_letter_logits": pos0_letters,
                   "step0_restricted_raw": step0, "finished_text": pred, "finished_letter": delta.leading_letter(pred),
                   "n_hook_applications": napp, "stock_finished_letter": stock.get(row["id"], {}).get("finished_letter"),
                   "stock_letter_unmask_step": stock.get(row["id"], {}).get("letter_unmask_step")}
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n"); handle.flush(); os.fsync(handle.fileno())
            done.add(row["id"]); n_new += 1
            if rec["stock_finished_letter"] is not None:
                n_cmp += 1; n_match += int(rec["stock_finished_letter"] == rec["finished_letter"])
            el = time.time() - t0
            base.write_progress(progress, {"backend": args.backend, "mode": "decode", "tag": tag, "n_done": len(done),
                                           "n_total": len(rows), "elapsed_sec": round(el, 1),
                                           "items_per_sec": round(n_new / el, 4) if el > 0 else None,
                                           "eta_sec": round((len(rows) - len(done)) * el / n_new) if n_new else None,
                                           "match_stock_letter": f"{n_match}/{n_cmp}"})
            if len(done) % 25 == 0 or len(done) == len(rows):
                print(f"PROGRESS {len(done)}/{len(rows)} rate={n_new / el:.3f}/s match_stock={n_match}/{n_cmp}", flush=True)
    print(f"DONE decode {output} n={len(done)} match_stock_letter={n_match}/{n_cmp} "
          f"peak_gpu_mem={torch.cuda.max_memory_allocated(ctx['device']) / 2 ** 30:.2f} GiB", flush=True)


# ----------------------------------------------------------------------------- mode: report (CPU)

def boot_ci(diff: np.ndarray, seed=20261013, n_boot=1000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    sums = diff[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {"delta": int(diff.sum()), "ci95": [int(round(lo)), int(round(hi))]}


def run_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir
    summary = {"date": "2026-10-13", "dir": str(out_dir), "models": {}}
    names = comp_names()
    xs = np.arange(len(names))
    fig_models = {}
    for m in ("lavida", "mmada"):
        p = out_dir / f"dla_{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ms = {"blank": {}, "real": {}, "ablation_blank": {}, "decode": {}}
        for n in ("2", "3", "4", "5"):
            b = d["blank"][n]
            ms["blank"][n] = {k: b[k] for k in ("top_writers", "top_cancellers", "L32", "prior_norm", "sum_contrib",
                                                "kl_to_uniform_final", "dla_sum_vs_model_max_abs_diff",
                                                "max_abs_diff_vs_cf_prior", "top_heads", "bottom_heads",
                                                "head_sum_vs_attn_component_max_abs_diff")}
        ms["real"] = {k: d["real"][k] for k in ("n_items", "top_writers_mean", "top_cancellers_mean",
                                                 "mean_final_prior_proj", "correlation_final_vs_writer_sum",
                                                 "dla_sum_vs_model_max_abs_diff", "n_opt_counts")}
        for n, e in d["ablation_blank"].items():
            runs = sorted(e["runs"], key=lambda r: r["delta_kl"])
            ms["ablation_blank"][n] = {"kl_before": e["before"]["kl_to_uniform"], "p_before": e["before"]["p"],
                                       "runs": [{k: r[k] for k in ("components", "kind", "kl_to_uniform", "delta_kl", "argmax", "p")} for r in runs]}
        # decode files
        stock = {r["id"]: r for r in ikb.load_jsonl(STOCK_DIR / f"{m}.jsonl")}
        for f in sorted(out_dir.glob(f"decode_{m}_*.jsonl")):
            rows = ikb.load_jsonl(f)
            if not rows:
                continue
            ids = [r["id"] for r in rows]
            acc = np.array([int(r["finished_letter"] == r["gold"]) for r in rows])
            sacc = np.array([int(stock[i]["finished_letter"] == stock[i]["gold"]) for i in ids])
            hist = {l: sum(r["finished_letter"] == l for r in rows) for l in LETTERS}
            shist = {l: sum(stock[i]["finished_letter"] == l for i in ids) for l in LETTERS}
            ghist = {l: sum(r["gold"] == l for r in rows) for l in LETTERS}
            ms["decode"][f.stem.replace(f"decode_{m}_", "")] = {
                "n": len(rows), "acc": int(acc.sum()), "stock_acc_same_items": int(sacc.sum()),
                "paired_bootstrap_vs_stock": boot_ci(acc - sacc), "letter_hist": hist, "stock_letter_hist": shist,
                "gold_hist": ghist, "n_letter_same_as_stock": int(sum(r["finished_letter"] == stock[r["id"]]["finished_letter"] for r in rows)),
                "n_null_letter": int(sum(r["finished_letter"] is None for r in rows)),
                "letter_unmask_step15_frac": float(np.mean([r["letter_unmask_step"] == 15 for r in rows])),
                "stock_unmask_step15_frac": float(np.mean([stock[i]["letter_unmask_step"] == 15 for i in ids])),
                "step0_restricted_acc": int(sum(r["step0_restricted_raw"] == r["gold"] for r in rows)),
                "components": rows[0]["components"], "ablation": rows[0]["ablation"]}
        summary["models"][m] = ms
        fig_models[m] = d

    # ---------- figure dla_{model}.png
    for m, d in fig_models.items():
        fig, axes = plt.subplots(3, 2, figsize=(15, 11))
        for j, n in enumerate(("2", "3")):
            b = d["blank"][n]
            c = np.array(b["contrib"])
            ax = axes[0, j]
            cols = ["#888888"] + ["#1f77b4" if i % 2 == 1 else "#ff7f0e" for i in range(1, len(names))]
            ax.bar(xs, c, color=cols, width=0.9, label="projection on FINAL prior u_n (centered log p_cf)")
            c31 = np.array(b["contrib_deep31"])
            ax.plot(xs, c31, "k.", ms=3, label="projection on DEEP prior (embed+L1..L31 letter logits)")
            ax.axhline(0, color="k", lw=0.6)
            top = sorted(range(len(c)), key=lambda i: -c[i])[:3]
            bot = sorted(range(len(c)), key=lambda i: c[i])[:2]
            for i in top + bot:
                ax.annotate(names[i], (i, c[i]), fontsize=7, ha="center", va="bottom" if c[i] >= 0 else "top")
            ax.set_xticks(xs[1::4]); ax.set_xticklabels([names[i].replace(".", "\n") for i in xs[1::4]], fontsize=6)
            v = b["L32_vs_deep31"]
            ax.set_title(f"{m}: per-component prior contribution, blank input, n={n} (||u||={b['prior_norm']:.2f}, "
                         f"KL={b['kl_to_uniform_final']:.2f})  blue=attn orange=MLP\n"
                         f"KL through L31 = {v['kl_to_uniform_through_L31']:.2f}; L32 (attn+mlp) cancels "
                         f"{100 * v['fraction_of_deep_prior_cancelled_by_L32']:.0f}% of the deep prior (cos {v['cos_L32sum_deep31']:.2f})",
                         fontsize=8)
            ax.set_ylabel("contribution to prior direction"); ax.legend(fontsize=6, loc="upper left")
            ax = axes[1, j]
            L = np.arange(0, N_LAYERS + 1)
            ax.plot(L, b["cumulative_dla_prior"], "o-", ms=3, label="cumulative DLA (frozen-scale) prior projection")
            ax.plot(L, b["cumulative_exact_lens_prior_proj"], "s--", ms=3, label="exact lens prior projection of h_l")
            ax2 = ax.twinx()
            ax2.plot(L, b["cumulative_exact_lens_kl_to_uniform"], "^-", ms=3, color="tab:red", label="exact lens KL(letters||U)")
            ax2.set_ylabel("KL to uniform", color="tab:red")
            ax.set_xlabel("layer (0 = embedding)"); ax.set_ylabel("prior projection")
            ax.set_title(f"{m}: cumulative prior across layers, n={n}", fontsize=9)
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        r = d["real"]
        ax = axes[2, 0]
        mc = np.array(r["mean_contrib"]); sc = np.array(r["std_contrib"])
        cols = ["#888888"] + ["#1f77b4" if i % 2 == 1 else "#ff7f0e" for i in range(1, len(names))]
        ax.bar(xs, mc, yerr=sc / math.sqrt(r["n_items"]), color=cols, width=0.9, error_kw={"lw": 0.5})
        ax.axhline(0, color="k", lw=0.6)
        for nm, v in r["top_writers_mean"][:3] + r["top_cancellers_mean"][:2]:
            ax.annotate(nm, (names.index(nm), v), fontsize=7, ha="center", va="bottom" if v >= 0 else "top")
        ax.set_xticks(xs[1::4]); ax.set_xticklabels([names[i].replace(".", "\n") for i in xs[1::4]], fontsize=6)
        ax.set_title(f"{m}: real inputs (n={r['n_items']}), mean prior contribution per component (±s.e.m.)", fontsize=9)
        ax = axes[2, 1]
        z = np.load(out_dir / f"dla_{m}_arrays.npz")
        rc, fp, rn = z["real_contrib"], z["real_final_proj"], z["real_n"]
        corr = r["correlation_final_vs_writer_sum"]
        for n, mk in (("2", "o"), ("3", "s")):
            w = corr[n]["writers"]; idx = [names.index(x) for x in w]
            ws = rc[:, idx].sum(1)
            ax.scatter(ws, fp, s=10, marker=mk, alpha=0.6, label=f"writers(blank n={n}) {w}: r={corr[n]['pearson_all_items']:.2f}")
        lim = [min(fp.min(), 0) - 0.5, fp.max() + 0.5]
        ax.plot(lim, lim, "k:", lw=0.6)
        ax.set_xlabel("sum of writer-component prior contributions"); ax.set_ylabel("final prior projection (all 65 comps)")
        ax.set_title(f"{m}: per-item final prior projection vs writer sum (real inputs)", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out_dir / f"dla_{m}.png", dpi=95); plt.close(fig)

    # ---------- figure dla_ablation.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for j, m in enumerate(("lavida", "mmada")):
        if m not in fig_models:
            continue
        d = fig_models[m]
        ax = axes[0, j]
        key_focus = []
        for n in ("2", "3"):
            b = d["blank"][n]
            key_focus += [tuple([b["top_writers"][0][0]]), tuple([b["top_cancellers"][0][0]])]
        key_focus += [("L32.attn",), ("L32.mlp",), tuple(nm for nm, _ in d["blank"]["2"]["top_writers"][:3])]
        key_focus = list(dict.fromkeys(key_focus))
        w = 0.8 / (1 + 2 * len(key_focus))
        for ni, n in enumerate(("2", "3", "4", "5")):
            e = d["ablation_blank"][n]
            ax.bar(ni - 0.4 + w / 2, e["before"]["kl_to_uniform"], w, color="k", label="before" if ni == 0 else None)
            for ci, comps in enumerate(key_focus):
                for ki, kind in enumerate(("zero", "mean")):
                    rr = [r for r in e["runs"] if tuple(r["components"]) == comps and r["kind"] == kind]
                    if rr:
                        ax.bar(ni - 0.4 + w / 2 + w * (1 + 2 * ci + ki), rr[0]["kl_to_uniform"], w,
                               color=plt.cm.tab10(ci), alpha=1.0 if kind == "zero" else 0.45,
                               label=f"{'+'.join(comps)} ({kind})" if ni == 0 else None)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"n={n}" for n in (2, 3, 4, 5)])
        ax.set_ylabel("KL(letter dist || uniform), blank input"); ax.set_title(f"{m}: blank-input ablation at the answer position", fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        ax = axes[1, j]
        dec = {k: v for k, v in summary["models"][m]["decode"].items() if "smoke" not in k}
        if dec:
            keys = list(dec.keys())
            xl = np.arange(5)
            wd = 0.8 / (len(keys) + 2)
            k0 = keys[0]
            ax.bar(xl - 0.4 + wd / 2, [dec[k0]["gold_hist"][l] for l in LETTERS], wd, color="k", label=f"gold (n={dec[k0]['n']})")
            ax.bar(xl - 0.4 + wd * 1.5, [dec[k0]["stock_letter_hist"][l] for l in LETTERS], wd, color="gray",
                   label=f"stock acc={dec[k0]['stock_acc_same_items']}")
            for ki, k in enumerate(keys):
                ax.bar(xl - 0.4 + wd * (2.5 + ki), [dec[k]["letter_hist"][l] for l in LETTERS], wd, color=plt.cm.tab10(ki),
                       label=f"{k}: acc={dec[k]['acc']} Δ={dec[k]['paired_bootstrap_vs_stock']['delta']} CI{dec[k]['paired_bootstrap_vs_stock']['ci95']}")
            ax.set_xticks(xl); ax.set_xticklabels(list(LETTERS)); ax.set_ylabel("count")
            ax.set_title(f"{m}: full 16-step decode with ablation hook at every step (first {dec[k0]['n']} items)", fontsize=9)
            ax.legend(fontsize=6)
        else:
            ax.text(0.5, 0.5, "decode not yet available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(out_dir / "dla_ablation.png", dpi=110); plt.close(fig)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({m: {"blank": {n: {"top_writers": v["top_writers"][:3], "top_cancellers": v["top_cancellers"][:3], "L32": v["L32"]}
                                     for n, v in ms["blank"].items() if n in ("2", "3")},
                          "decode": {k: {kk: v[kk] for kk in ("n", "acc", "stock_acc_same_items", "paired_bootstrap_vs_stock", "letter_hist", "stock_letter_hist")}
                                     for k, v in ms["decode"].items()}}
                      for m, ms in summary["models"].items()}, indent=1))
    print(f"wrote {out_dir / 'summary.json'}")


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("dla", "decode", "report"), required=True)
    p.add_argument("--backend", choices=("lavida", "mmada"), default="lavida")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument("--n-real", type=int, default=200)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--components", nargs="*", default=[])
    p.add_argument("--ablation", choices=("zero", "mean"), default="mean")
    p.add_argument("--tag", default="")
    p.add_argument("--model-path", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    p.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    args = p.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir != OUT_ROOT and OUT_ROOT not in args.out_dir.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {args.out_dir}")
    if args.progress is None:
        args.progress = args.out_dir / f"progress_{args.mode}_{args.backend}.json"
    if args.mode != "report":
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
        if base.GEN != 16:
            raise RuntimeError(f"unexpected gen {base.GEN}")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"dla": run_dla, "decode": run_decode, "report": run_report}[args.mode](args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
