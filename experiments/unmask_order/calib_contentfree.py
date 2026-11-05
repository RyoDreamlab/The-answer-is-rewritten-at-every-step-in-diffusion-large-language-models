#!/usr/bin/env python3
"""Content-free letter priors at the letter position, decode step 0, full canvas masked.

One stock forward per (item, condition); no canvas edit, no decode past step 0.
Conditions (option count n is always the item's own count):
  a_global_n{n}  mid-gray image, question 'N/A', n options all 'N/A' (one per n in 2..5)
  b_imgfree      the item's real question and options, mid-gray image (item image size)
  c_txtfree      the item's real image, question 'N/A', n options 'N/A', no context
  real           the item's real image and text (first --n-real-check items, readout check only)

Letter logits are recomputed in float64 from the captured ln_f input (gamma * rmsnorm(h) @ ff_out
letter rows), the same readout as hidden_shapley_offline_rules.py, and the model's own bf16 logits
at A-E are stored alongside.
"""

from __future__ import annotations

import argparse
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
import hidden_shapley_split as hss  # noqa: E402
import image_key_block as ikb  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

LETTERS = "ABCDE"
GRAY = (128, 128, 128)
GLOBAL_SIZE = (384, 384)


def gray(size):
    from PIL import Image

    return Image.new("RGB", tuple(int(s) for s in size), GRAY)


def lavida_question_na(n: int) -> str:
    opts = "\n".join(f"{LETTERS[i]}. N/A" for i in range(n))
    return f"N/A\nOptions:\n{opts}\nWhich letter is correct?"


def readout(h: torch.Tensor, gamma64: torch.Tensor, W64: torch.Tensor, eps: float) -> list[float]:
    h64 = h.to(torch.float64)
    v = gamma64 * hss.rms_norm(h64, eps)
    return [float(x) for x in (W64 @ v)]


def lavida_prefix_img(lm, model, processor, tokenizer, q: str, image, image_token_index: int):
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    conv = delta.copy_conv(conv_templates["llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + f"Question: {q}\nAnswer:")
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, image_token_index, return_tensors="pt")
    input_ids = ids.unsqueeze(0).to(model.device)
    tensors = process_images([image], processor, model.config)
    tensors = [t.to(device=model.device, dtype=torch.bfloat16) for t in tensors]
    prepared = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, tensors, ["image"], image_sizes=[image.size], return_inputs=True
    )
    embeds = prepared[4]
    base.KeyBlock.cols = None
    past = lm(None, input_embeddings=embeds, use_cache=True).attn_key_values
    return past, int(embeds.shape[1])


def make_lavida(args):
    sys.path.insert(0, str(PROJECT / "repos" / "LaViDa"))
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.model.builder import load_pretrained_model

    siglip = (
        PROJECT
        / "hf-cache/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
    )
    vision_kwargs = {
        "mm_vision_tower": str(siglip) if siglip.exists() else "google/siglip-so400m-patch14-384",
        "mm_resampler_type": None,
        "mm_projector_type": "mlp2x_gelu",
        "mm_hidden_size": 1152,
        "use_mm_proj": True,
    }
    print("loading lavida", flush=True)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, None, "llava_llada", device_map=f"cuda:{args.cuda}",
        vision_kwargs=vision_kwargs, torch_dtype="bfloat16",
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    lm = model.get_model()
    head, gamma64, eps = hss.prepare(lm, args.cuda)
    lid = hss.letter_ids(tokenizer, LETTERS)
    gen = base.GEN

    def forward(q: str, image):
        past, prefix_len = lavida_prefix_img(lm, model, processor, tokenizer, q, image, IMAGE_TOKEN_INDEX)
        canvas = torch.full((1, gen), base.MASK_ID_LAVIDA, dtype=torch.long, device=model.device)
        embeds = lm.transformer.wte(canvas)
        hss.arm(0)
        logits = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
        h = hss.collect(gen, 1)[0]
        return logits[0, 0].detach().float().cpu(), h, prefix_len

    def conditions(row, n_real):
        n = len(row["choices"]) if not isinstance(row["choices"], str) else len(eval(row["choices"]))
        img = Image.open(row["image"]).convert("RGB")
        out = [("b_imgfree", lambda: forward(ikb.question_lavida(row), gray(img.size))),
               ("c_txtfree", lambda: forward(lavida_question_na(n), img))]
        if n_real:
            out.append(("real", lambda: forward(ikb.question_lavida(row), img)))
        return n, out

    def global_forward(n):
        return forward(lavida_question_na(n), gray(GLOBAL_SIZE))

    return tokenizer, head, gamma64, eps, lid, conditions, global_forward, base.MASK_ID_LAVIDA


def make_mmada(args):
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
    if type(model).__name__ != "MMadaModelLM":
        raise RuntimeError(f"expected MMadaModelLM, got {type(model).__name__}")
    vq = mm.load_vq(Ns, device)
    model.eval()
    head, gamma64, eps = hss.prepare(model.model, args.cuda)
    lid = hss.letter_ids(tokenizer, LETTERS)
    gen = base.GEN
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
        canvas = torch.full((1, gen), mm.MASK_ID, dtype=torch.long, device=device)
        x = torch.cat([prompt_ids, canvas], dim=1)
        hss.arm(prompt_len)
        logits = model(x).logits
        h = hss.collect(prompt_len + gen, 1)[0]
        return logits[0, prompt_len].detach().float().cpu(), h, prompt_len

    def na_prompt(row, n):
        return sq.build_task_prompt({**row, "question": "N/A", "choices": ["N/A"] * n, "hint": ""})

    def conditions(row, n_real):
        n = len(row["choices"])
        img = loader.load_rgb(row)
        out = [("b_imgfree", lambda: forward(sq.build_task_prompt(row), gray(img.size))),
               ("c_txtfree", lambda: forward(na_prompt(row, n), img))]
        if n_real:
            out.append(("real", lambda: forward(sq.build_task_prompt(row), img)))
        return n, out

    def global_forward(n):
        return forward(na_prompt({"question": "", "choices": [], "hint": ""}, n), gray(GLOBAL_SIZE))

    return tokenizer, head, gamma64, eps, lid, conditions, global_forward, mm.MASK_ID


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("lavida", "mmada"), required=True)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--n-real-check", type=int, default=32)
    p.add_argument("--model-path", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    p.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    args = p.parse_args()
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
    ikb.refuse_forbidden(args.output)
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
    make = make_lavida if args.backend == "lavida" else make_mmada
    tokenizer, head, gamma64, eps, lid, conditions, global_forward, mask_id = make(args)
    W64 = torch.stack([head[lid[l]].detach().to(torch.float64).cpu() for l in LETTERS])
    ids = [lid[l] for l in LETTERS]
    print(f"start backend={args.backend} cuda={args.cuda} letter_ids={lid} n={len(rows)} done={len(done)}", flush=True)
    t0 = time.time()

    def record(item_id, cond, n, res):
        logits, h, prefix_len = res
        a = int(torch.argmax(logits).item())
        return {
            "id": item_id, "cond": cond, "backend": args.backend, "n_opt": int(n),
            "letter_logit64": readout(h, gamma64, W64, eps),
            "letter_logit_model": [float(logits[i]) for i in ids],
            "argmax_id": a, "argmax_text": ikb.decode_token(tokenizer, a, mask_id),
            "prefix_len": prefix_len, "decode_step": 0, "canvas_index": 0,
        }

    with args.output.open("a", encoding="utf-8") as handle:
        def emit(rec):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            done.add((rec["id"], rec["cond"]))

        for n in (2, 3, 4, 5):
            key = (f"global_n{n}", "a_global")
            if key not in done:
                emit(record(key[0], key[1], n, global_forward(n)))
        for k, row in enumerate(rows):
            n, conds = conditions(row, k < args.n_real_check)
            for cond, fn in conds:
                if (row["id"], cond) in done:
                    continue
                emit(record(row["id"], cond, n, fn()))
            if (k + 1) % 50 == 0 or k + 1 == len(rows):
                base.write_progress(args.progress, {"backend": args.backend, "n_items": k + 1, "n_total": len(rows),
                                                    "elapsed_sec": round(time.time() - t0, 1)})
                print(f"PROGRESS {k + 1}/{len(rows)} elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"DONE backend={args.backend} records={len(done)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
