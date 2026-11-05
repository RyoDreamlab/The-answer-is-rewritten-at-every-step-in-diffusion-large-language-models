#!/usr/bin/env python3
"""B4 — answer-slot collapse, dose control. Decode step 0, whole canvas masked, read canvas index 0.

Reuses the key-visibility machinery of hidden_shapley_split.py (queries unchanged, only key visibility changes,
every query always sees its own key). Per item, one batched forward with 16 visibility rows:

  full          nothing blocked
  M{k}          block the keys of k still-masked answer slots, canvas indices 1..k (nested), k in 1,2,4,8,15
  Trand{k}      control: block k question/prompt text keys, random (seeded per item, nested), same k
  Tlast{k}      control: block the last k text keys before the canvas (template tail, nested), same k

Image keys are never blocked. Recorded per row: letter logits A-E in float64 (ln_f-input capture + gamma*rmsnorm
+ letter rows of ff_out, as hidden_shapley_split), the bf16 model logits at A-E, the full-vocab argmax token.
Offline we compute the letter-prior shift vs 'full' as a function of dose k for M vs the two T controls.
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

import numpy as np
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import hidden_shapley_split as hss  # noqa: E402
import image_key_block as ikb  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "slot_dose_20261013").resolve()
LETTERS = "ABCDE"
DOSES = (1, 2, 4, 8, 15)
SEED = 20261013
NEG = hss.NEG


def conditions(prefix_len: int, gen: int, text_cols: torch.Tensor, item_index: int):
    """List of (name, family, k, blocked_key_cols LongTensor)."""
    rng = np.random.default_rng(SEED + item_index)
    text_np = text_cols.numpy()
    perm = rng.permutation(len(text_np))
    out = [("full", "full", 0, torch.zeros(0, dtype=torch.long))]
    for k in DOSES:
        out.append((f"M{k}", "M", k, torch.arange(prefix_len + 1, prefix_len + 1 + k)))
    for k in DOSES:
        out.append((f"Trand{k}", "Trand", k, torch.as_tensor(np.sort(text_np[perm[:k]]), dtype=torch.long)))
    for k in DOSES:
        out.append((f"Tlast{k}", "Tlast", k, torch.as_tensor(text_np[-k:], dtype=torch.long)))
    return out


def allow_matrix(n_rows: int, n_keys: int, row_key: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
    allow = torch.ones(n_rows, n_keys, dtype=torch.bool)
    allow[:, blocked] = False
    allow[torch.arange(n_rows), row_key] = True
    return allow


def pack(row_id, backend, gold, n, conds, logitsB, hB, head, gamma64, eps, W64, letter_ids, tokenizer, mask_id,
         extra) -> dict:
    recs = {}
    for i, (name, fam, k, blocked) in enumerate(conds):
        h64 = hB[i].to(torch.float64)
        l64 = (W64 @ (gamma64 * hss.rms_norm(h64, eps))).tolist()
        a = int(torch.argmax(logitsB[i]).item())
        recs[name] = {"family": fam, "k": k, "n_blocked": int(blocked.numel()),
                      "letter_logit64": [float(v) for v in l64],
                      "letter_logit_model": [float(logitsB[i][j]) for j in letter_ids],
                      "argmax_id": a, "argmax_text": ikb.decode_token(tokenizer, a, mask_id)}
    full = recs["full"]
    return {"id": row_id, "backend": backend, "gold": gold, "n_opt": int(n), "decode_step": 0, "canvas_index": 0,
            "doses": list(DOSES), "full_letter_restricted": LETTERS[int(np.argmax(full["letter_logit64"][:n]))],
            "conds": recs, **extra}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("lavida", "mmada"), required=True)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
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
    if base.GEN != 16 or base.MASK_ID_LAVIDA != 126336:
        raise RuntimeError("unexpected gen/mask")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ikb.repair_tail(args.output)
    manifest = ikb.load_manifest(args)
    rows = manifest[: args.max_samples] if args.max_samples > 0 else manifest
    index_of = {r["id"]: i for i, r in enumerate(manifest)}
    lock = args.output.with_suffix(args.output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {args.output}") from exc
    done = {r["id"] for r in ikb.load_jsonl(args.output)}
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    gen = base.GEN

    if args.backend == "lavida":
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
            args.model_path, None, "llava_llada", device_map=f"cuda:{args.cuda}",
            vision_kwargs=vision_kwargs, torch_dtype="bfloat16")
        model.eval(); model.tie_weights(); model.to(torch.bfloat16)
        lm = model.get_model()
        n_blocks = hss.install_visibility(lm)
        head, gamma64, eps = hss.prepare(lm, args.cuda)
        mask_id = base.MASK_ID_LAVIDA
        print(f"lavida blocks={n_blocks}", flush=True)

        def item(row):
            past, image_cols = ikb.lavida_prefix(lm, model, processor, tokenizer, row, IMAGE_TOKEN_INDEX)
            prefix_len = int(past[0][0].shape[-2])
            n_keys = prefix_len + gen
            image_cols = image_cols.cpu()
            is_image = torch.zeros(prefix_len, dtype=torch.bool)
            is_image[image_cols] = True
            text_cols = torch.nonzero(~is_image, as_tuple=False).flatten()
            conds = conditions(prefix_len, gen, text_cols, index_of[row["id"]])
            row_key = torch.arange(prefix_len, prefix_len + gen)
            allow = torch.stack([allow_matrix(gen, n_keys, row_key, c[3]) for c in conds])
            additive = torch.zeros(allow.shape, dtype=torch.float32)
            additive[~allow] = NEG
            additive = additive.unsqueeze(1).to(device=model.device)
            canvas = torch.full((1, gen), mask_id, dtype=torch.long, device=model.device)
            embeds = lm.transformer.wte(canvas)
            B = len(conds)
            pastB = [tuple(t.expand(B, *t.shape[1:]) for t in layer) for layer in past]
            hss.Visible.additive = additive
            try:
                hss.arm(0)
                logits = lm(None, input_embeddings=embeds.repeat(B, 1, 1), past_key_values=pastB,
                            use_cache=False).logits
                hB = hss.collect(gen, B)
            finally:
                hss.Visible.additive = None
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits")
            logitsB = logits[:, 0].detach().float().cpu()
            extra = {"prefix_len": prefix_len, "n_text_keys": int(text_cols.numel()),
                     "n_image_keys": int(image_cols.numel()),
                     "blocked_text_cols": {c[0]: c[3].tolist() for c in conds if c[1] in ("Trand", "Tlast")},
                     "blocking_scope": "canvas_queries_over_unblocked_prefix_cache"}
            return conds, logitsB, hB, extra, ikb.gold_of(row), n_choices(row)
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
        mask_id = mm.MASK_ID
        loader = None
        for name in dir(sq):
            obj = getattr(sq, name)
            if isinstance(obj, type) and hasattr(obj, "load_rgb"):
                loader = obj(Path(args.data_dir), manifest)
                break
        if loader is None:
            raise RuntimeError("ScienceQA image loader was not found")

        def item(row):
            image = loader.load_rgb(row)
            question = sq.build_task_prompt(row)
            prompt_ids, image_cols = ikb.mmada_prompt(model, tokenizer, vq, mm, image, question, device)
            prompt_len = int(prompt_ids.shape[1])
            seq = prompt_len + gen
            image_cols = image_cols.cpu()
            prompt_cpu = prompt_ids[0].cpu()
            is_image = torch.zeros(prompt_len, dtype=torch.bool)
            is_image[image_cols] = True
            is_mask = prompt_cpu.eq(mm.MASK_ID)
            text_cols = torch.nonzero(~is_image & ~is_mask, as_tuple=False).flatten()
            conds = conditions(prompt_len, gen, text_cols, index_of[row["id"]])
            row_key = torch.arange(seq)
            allow = torch.stack([allow_matrix(seq, seq, row_key, c[3]) for c in conds]).unsqueeze(1).to(device)
            canvas = torch.full((1, gen), mask_id, dtype=torch.long, device=device)
            x = torch.cat([prompt_ids, canvas], dim=1)
            B = len(conds)
            hss.arm(prompt_len)
            logits = model(x.repeat(B, 1), attention_bias=allow).logits
            hB = hss.collect(seq, B)
            answer = logits[:, prompt_len].detach().float().cpu()
            del logits
            if not torch.isfinite(answer).all():
                raise RuntimeError("non-finite logits")
            extra = {"prefix_len": prompt_len, "n_text_keys": int(text_cols.numel()),
                     "n_image_keys": int(image_cols.numel()),
                     "blocked_text_cols": {c[0]: c[3].tolist() for c in conds if c[1] in ("Trand", "Tlast")},
                     "blocking_scope": "all_query_rows_full_sequence"}
            return conds, answer, hB, extra, ikb.gold_of(row), len(row["choices"])

    lid = hss.letter_ids(tokenizer, LETTERS)
    letter_ids = [lid[l] for l in LETTERS]
    W64 = torch.stack([head[i].detach().to(torch.float64).cpu() for i in letter_ids])
    print(f"start backend={args.backend} cuda={args.cuda} doses={DOSES} n={len(rows)} done={len(done)} "
          f"letter_ids={letter_ids}", flush=True)
    t0 = time.time()
    n_new = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for k, row in enumerate(rows):
            if row["id"] in done:
                continue
            conds, logitsB, hB, extra, gold, n = item(row)
            rec = pack(row["id"], args.backend, gold, n, conds, logitsB, hB, head, gamma64, eps, W64, letter_ids,
                       tokenizer, mask_id, extra)
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(row["id"])
            n_new += 1
            if n_new % 25 == 0 or k + 1 == len(rows):
                el = time.time() - t0
                rate = n_new / el if el > 0 else 0
                base.write_progress(args.progress, {"backend": args.backend, "n_done": len(done), "n_total": len(rows),
                                                    "elapsed_sec": round(el, 1),
                                                    "eta_sec": round((len(rows) - len(done)) / rate) if rate else None})
                c = rec["conds"]
                print(f"PROGRESS {len(done)}/{len(rows)} elapsed={el:.0f}s id={row['id']} "
                      f"full={c['full']['argmax_text']!r} M15={c['M15']['argmax_text']!r} "
                      f"Trand15={c['Trand15']['argmax_text']!r} Tlast15={c['Tlast15']['argmax_text']!r}", flush=True)
    print(f"DONE backend={args.backend} n={len(done)}", flush=True)


def n_choices(row) -> int:
    c = row["choices"]
    if isinstance(c, str):
        c = ast.literal_eval(c)
    return len(c)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
