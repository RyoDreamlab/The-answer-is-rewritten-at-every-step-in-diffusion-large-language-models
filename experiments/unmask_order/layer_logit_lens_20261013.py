#!/usr/bin/env python3
"""B3 — layer-wise logit lens at the answer-letter position, decode step 0 (whole canvas masked).

One forward per (item, condition). Forward hooks on every transformer block capture the block output at the
letter position (canvas index 0). Each layer's residual stream h_l is read out through the model's own final norm
and LM head:  logits_l = ff_out( gamma * rmsnorm(h_l) ).  We keep (a) the A-E letter logits in float64
(same readout as hidden_shapley_split / calib_contentfree) and (b) the full-vocab argmax token and its softmax
probability (float32, GPU).  Layer index l = 1..32 is the output of block l; l = 32 equals the ln_f input, and its
letter readout is checked against the model's bf16 logits.

Conditions
  real       the item's real image + question                       (all 2017 ScienceQA-IMG test items)
  a_global   content-free: mid-gray 384x384 image, question 'N/A', n options 'N/A'  (one per n in 2..5)
  imgfree    (optional, --imgfree) item's real text, mid-gray image of the item's size

No canvas edit, no decode. Output jsonl: one record per (item, cond).
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import calib_contentfree as ccf  # noqa: E402
import hidden_shapley_split as hss  # noqa: E402
import image_key_block as ikb  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "layer_logit_lens_20261013").resolve()
LETTERS = "ABCDE"


class Lens:
    armed = False
    index = 0
    hs: dict[int, torch.Tensor] = {}
    n_layers = 0


def install_hooks(core) -> int:
    blocks = list(core.transformer.blocks)
    for li, block in enumerate(blocks):
        def hook(_m, _args, output, _li=li):
            if not Lens.armed:
                return
            x = output[0] if isinstance(output, (tuple, list)) else output
            Lens.hs[_li + 1] = x[:, Lens.index].detach()
        block.register_forward_hook(hook)
    Lens.n_layers = len(blocks)
    return len(blocks)


def arm(index: int) -> None:
    Lens.armed = True
    Lens.index = int(index)
    Lens.hs = {}


def collect() -> torch.Tensor:
    Lens.armed = False
    if len(Lens.hs) != Lens.n_layers:
        raise RuntimeError(f"captured {len(Lens.hs)} layers, expected {Lens.n_layers}")
    return torch.stack([Lens.hs[l][0] for l in range(1, Lens.n_layers + 1)])  # (L, d) bf16 on GPU


def readout_all(H: torch.Tensor, gamma_dev: torch.Tensor, eps: float, head: torch.Tensor, W64: torch.Tensor,
                gamma64: torch.Tensor, letter_ids: list[int], tokenizer, mask_id: int, eos_id: int | None) -> dict:
    """H (L, d) on GPU. Returns per-layer letter logits (float64 CPU) and vocab stats (float32 GPU)."""
    H32 = H.to(torch.float32)
    normed32 = gamma_dev * (H32 * torch.rsqrt(H32.pow(2).mean(dim=1, keepdim=True) + eps))
    V = normed32 @ head.to(torch.float32).T  # (L, vocab)
    P = torch.softmax(V, dim=1)
    top = torch.topk(P, 1, dim=1)
    arg = top.indices[:, 0].tolist()
    letter_mass = P[:, letter_ids].sum(dim=1).tolist()
    eos_p = P[:, eos_id].tolist() if eos_id is not None else None
    H64 = H.detach().to(torch.float64).cpu()
    normed64 = gamma64 * (H64 * torch.rsqrt(H64.pow(2).mean(dim=1, keepdim=True) + eps))
    L64 = (normed64 @ W64.T)  # (L, 5)
    return {
        "letter_logit64": [[float(v) for v in row] for row in L64],
        "vocab_argmax_id": arg,
        "vocab_argmax_text": [ikb.decode_token(tokenizer, a, mask_id) for a in arg],
        "vocab_top1_prob": [float(v) for v in top.values[:, 0]],
        "letter_mass": [float(v) for v in letter_mass],
        "eos_prob": [float(v) for v in eos_p] if eos_p is not None else None,
        "h_norm": [float(v) for v in H64.norm(dim=1)],
    }


def n_choices(row) -> int:
    c = row["choices"]
    if isinstance(c, str):
        c = ast.literal_eval(c)
    return len(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("lavida", "mmada"), required=True)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--imgfree", action="store_true")
    p.add_argument("--model-path", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    p.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    args = p.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
    for q in (args.output, args.progress):
        if OUT_ROOT not in q.resolve().parents:
            raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {q}")
    if base.GEN != 16:
        raise RuntimeError(f"unexpected gen {base.GEN}")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ikb.repair_tail(args.output)
    rows = ikb.load_manifest(args)
    args.rows = rows
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    lock = args.output.with_suffix(args.output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {args.output}") from exc
    done = {(r["id"], r["cond"]) for r in ikb.load_jsonl(args.output)}
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    gen = base.GEN

    # --- backend-specific forward functions (model handles are needed for the block hooks) ---
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
        n_blocks = install_hooks(lm)
        mask_id = base.MASK_ID_LAVIDA
        eos_id = tokenizer.eos_token_id

        def forward(q: str, image):
            past, prefix_len = ccf.lavida_prefix_img(lm, model, processor, tokenizer, q, image, IMAGE_TOKEN_INDEX)
            canvas = torch.full((1, gen), mask_id, dtype=torch.long, device=model.device)
            embeds = lm.transformer.wte(canvas)
            arm(0)
            logits = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
            H = collect()
            return logits[0, 0].detach().float(), H, prefix_len

        def real(row):
            return forward(ikb.question_lavida(row), Image.open(row["image"]).convert("RGB"))

        def imgfree(row):
            img = Image.open(row["image"])
            return forward(ikb.question_lavida(row), ccf.gray(img.size))

        def blank(n):
            return forward(ccf.lavida_question_na(n), ccf.gray(ccf.GLOBAL_SIZE))
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
        n_blocks = install_hooks(core)
        mask_id = mm.MASK_ID
        eos_id = tokenizer.eos_token_id
        loader = None
        for name in dir(sq):
            obj = getattr(sq, name)
            if isinstance(obj, type) and hasattr(obj, "load_rgb"):
                loader = obj(Path(args.data_dir), args.rows)
                break
        if loader is None:
            raise RuntimeError("ScienceQA image loader was not found")

        def forward(question: str, image):
            vq_ids = mm.encode_vq(vq, image, device)
            prompt_ids, _ = mm.build_prompt_ids(tokenizer, question, vq_ids, device)
            prompt_len = int(prompt_ids.shape[1])
            canvas = torch.full((1, gen), mask_id, dtype=torch.long, device=device)
            x = torch.cat([prompt_ids, canvas], dim=1)
            arm(prompt_len)
            logits = model(x).logits
            H = collect()
            return logits[0, prompt_len].detach().float(), H, prompt_len

        def real(row):
            return forward(sq.build_task_prompt(row), loader.load_rgb(row))

        def imgfree(row):
            return forward(sq.build_task_prompt(row), ccf.gray(loader.load_rgb(row).size))

        def blank(n):
            return forward(sq.build_task_prompt({"question": "N/A", "choices": ["N/A"] * n, "hint": ""}),
                           ccf.gray(ccf.GLOBAL_SIZE))

    lid = hss.letter_ids(tokenizer, LETTERS)
    letter_ids = [lid[l] for l in LETTERS]
    W64 = torch.stack([head[i].detach().to(torch.float64).cpu() for i in letter_ids])
    gamma_dev = gamma64.to(device=head.device, dtype=torch.float32)
    print(f"start backend={args.backend} cuda={args.cuda} layers={n_blocks} letter_ids={letter_ids} eos={eos_id} "
          f"n={len(rows)} done={len(done)} imgfree={args.imgfree}", flush=True)
    t0 = time.time()

    def record(item_id, cond, n, gold, res):
        logits, H, prefix_len = res
        ro = readout_all(H, gamma_dev, eps, head, W64, gamma64, letter_ids, tokenizer, mask_id, eos_id)
        model_letters = [float(logits[i]) for i in letter_ids]
        final64 = ro["letter_logit64"][-1]
        return {"id": item_id, "cond": cond, "backend": args.backend, "n_opt": int(n), "gold": gold,
                "n_layers": n_blocks, "prefix_len": prefix_len, "decode_step": 0, "canvas_index": 0,
                "letter_logit_model": model_letters,
                "final_readout_max_abs_diff_vs_model": max(abs(a - b) for a, b in zip(final64, model_letters)),
                "model_argmax_id": int(torch.argmax(logits).item()),
                "model_argmax_text": ikb.decode_token(tokenizer, int(torch.argmax(logits).item()), mask_id), **ro}

    with args.output.open("a", encoding="utf-8") as handle:
        def emit(rec):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            done.add((rec["id"], rec["cond"]))

        for n in (2, 3, 4, 5):
            key = (f"global_n{n}", "a_global")
            if key not in done:
                emit(record(key[0], key[1], n, None, blank(n)))
                print(f"BLANK n={n} final_letters64={[round(v, 3) for v in ikb.load_jsonl(args.output)[-1]['letter_logit64'][-1][:n]]}",
                      flush=True)
        for k, row in enumerate(rows):
            n = n_choices(row)
            gold = ikb.gold_of(row)
            conds = [("real", real)] + ([("imgfree", imgfree)] if args.imgfree else [])
            for cond, fn in conds:
                if (row["id"], cond) in done:
                    continue
                rec = record(row["id"], cond, n, gold, fn(row))
                emit(rec)
                if rec["final_readout_max_abs_diff_vs_model"] > 0.5:
                    print(f"WARN readout mismatch id={row['id']} cond={cond} "
                          f"diff={rec['final_readout_max_abs_diff_vs_model']:.3f}", flush=True)
            if (k + 1) % 50 == 0 or k + 1 == len(rows):
                el = time.time() - t0
                base.write_progress(args.progress, {"backend": args.backend, "n_items": k + 1, "n_total": len(rows),
                                                    "elapsed_sec": round(el, 1),
                                                    "eta_sec": round(el / (k + 1) * (len(rows) - k - 1))})
                print(f"PROGRESS {k + 1}/{len(rows)} elapsed={el:.0f}s", flush=True)
    print(f"DONE backend={args.backend} records={len(done)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
