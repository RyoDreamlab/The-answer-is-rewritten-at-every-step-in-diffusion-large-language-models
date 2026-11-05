#!/usr/bin/env python3
"""Unmask-order decode plus an early bidirectional-vs-blocked contrast.

Normal low-confidence decode. At the earliest step where canvas index 0 is
still masked and at least half of the other answer positions are still masked,
a second forward blocks keys from those still-masked answer positions.
Question and image stay visible. No hidden-state edit.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
MASK_ID_LAVIDA = 126336
GEN = 16


class KeyBlock:
    cols = None  # 1D long tensor of key indices to block, or None


def install_key_block(lm) -> int:
    n = 0
    for block in lm.transformer.blocks:
        orig = block._scaled_dot_product_attention

        def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, _orig=orig):
            cols = KeyBlock.cols
            if cols is None:
                return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            num_q = q.size(1)
            num_kv = k.size(1)
            if num_q != num_kv:
                if num_q % num_kv != 0:
                    raise RuntimeError(f"cannot expand {num_kv} kv heads to {num_q}")
                rep = num_q // num_kv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            klen = k.shape[-2]
            if int(cols.max().item()) >= klen:
                raise RuntimeError(f"blocked key {int(cols.max())} outside key length {klen}")
            mask = torch.zeros(1, 1, q.shape[-2], klen, device=q.device, dtype=q.dtype)
            mask[..., cols] = torch.tensor(-1.0e4, device=q.device, dtype=q.dtype)
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )

        block._scaled_dot_product_attention = sdpa
        n += 1
    return n


def piece(tokenizer, token_id: int, mask_id: int) -> str:
    if int(token_id) == int(mask_id):
        return ""
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    text = text.replace("\n", " ").strip()
    if not text:
        text = str(tokenizer.convert_ids_to_tokens(int(token_id)))
    return text[:24]


def argmax_piece(tokenizer, logits_row: torch.Tensor, mask_id: int) -> tuple[int, str]:
    token_id = int(torch.argmax(logits_row).item())
    return token_id, piece(tokenizer, token_id, mask_id)


def unmask_one(x: torch.Tensor, logits: torch.Tensor, mask_id: int) -> int:
    mask_index = x.eq(mask_id)
    if int(mask_index.sum().item()) == 0:
        raise RuntimeError("unmask called on a full canvas")
    chosen = torch.argmax(logits, dim=-1)
    probs = F.softmax(logits.double(), dim=-1)
    conf = torch.gather(probs, -1, chosen.unsqueeze(-1)).squeeze(-1)
    conf = torch.where(mask_index, conf, torch.full_like(conf, float("-inf")))
    pos = int(torch.topk(conf[0], k=1).indices[0].item())
    if not bool(mask_index[0, pos].item()):
        raise RuntimeError(f"selected unmasked position {pos}")
    x[0, pos] = chosen[0, pos]
    return pos


def qualifies(mask_row: torch.Tensor, mask_id: int) -> bool:
    letter_masked = int(mask_row[0].item()) == int(mask_id)
    others = int((mask_row[1:] == int(mask_id)).sum().item())
    other_count = int(mask_row.numel() - 1)
    return bool(letter_masked and others * 2 >= other_count)


def write_progress(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["id"])
    return done


def maybe_pictures(rows: list[dict], tokenizer_decode, out_dir: Path, mask_id: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"picture skip: {exc}", flush=True)
        return
    changed = [r for r in rows if r.get("argmax_changed")]
    same = [r for r in rows if not r.get("argmax_changed")]
    picked = changed[:4] + same[:4]
    if len(picked) < 4 and len(rows) < 8:
        return
    if not picked:
        picked = rows[:8]
    dest = out_dir / "png"
    dest.mkdir(parents=True, exist_ok=True)
    for record in picked[:8]:
        history = record.get("canvas_ids") or []
        if len(history) != GEN:
            continue
        cell_w, cell_h = 72, 22
        width = 16 * cell_w + 80
        height = GEN * cell_h + 64
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        caption = (
            f"{record['id']}  bi={record.get('argmax_bi','')}  "
            f"blocked={record.get('argmax_blocked','')}  "
            f"step={record.get('contrast_step')}"
        )
        draw.text((4, 4), caption[:140], fill=(0, 0, 0))
        for step, ids in enumerate(history):
            y = 36 + step * cell_h
            draw.text((4, y + 4), f"{step:02d}", fill=(80, 80, 80))
            for col, tid in enumerate(ids):
                x0 = 40 + col * cell_w
                blank = int(tid) == int(mask_id)
                color = (245, 245, 245) if blank else (255, 255, 255)
                outline = (180, 40, 40) if col == 0 else (180, 180, 180)
                draw.rectangle([x0, y, x0 + cell_w - 2, y + cell_h - 2], fill=color, outline=outline)
                if not blank:
                    draw.text((x0 + 2, y + 4), piece_from_saved(record, step, col)[:10], fill=(0, 0, 0))
        image.save(dest / f"{record['id']}.png")
    print(f"pictures={len(list(dest.glob('*.png')))}", flush=True)


def piece_from_saved(record: dict, step: int, col: int) -> str:
    texts = record.get("canvas_text") or []
    if step < len(texts) and col < len(texts[step]):
        return texts[step][col] or ""
    return ""


def run_lavida(args) -> None:
    sys.path.insert(0, str(PROJECT / "repos" / "LaViDa"))
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
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
        args.model_path,
        None,
        "llava_llada",
        device_map="cuda:0",
        vision_kwargs=vision_kwargs,
        torch_dtype="bfloat16",
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    lm = model.get_model()
    n_blocks = install_key_block(lm)
    print(f"loaded lavida blocks={n_blocks} mask={MASK_ID_LAVIDA}", flush=True)
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    done = load_done(args.output)
    finished = []
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    finished.append(json.loads(line))
    t0 = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if row["id"] in done:
                continue
            record = lavida_item(lm, model, processor, tokenizer, row)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            finished.append(record)
            n_done = len(finished)
            payload = {
                "backend": "lavida",
                "n_done": n_done,
                "n_total": len(rows),
                "id": record["id"],
                "letter_unmask_step": record["letter_unmask_step"],
                "letter_unmasks_last": record["letter_unmasks_last"],
                "contrast_step": record["contrast_step"],
                "argmax_changed": record["argmax_changed"],
                "argmax_bi": record["argmax_bi"],
                "argmax_blocked": record["argmax_blocked"],
                "elapsed_sec": round(time.time() - t0, 1),
            }
            write_progress(args.progress, payload)
            print(
                f"PROGRESS n_done={n_done}/{len(rows)} id={record['id']} "
                f"letter_unmask_step={record['letter_unmask_step']} "
                f"argmax_changed={int(record['argmax_changed'])} "
                f"contrast_step={record['contrast_step']}",
                flush=True,
            )
            if n_done <= 12 or n_done % 25 == 0:
                maybe_pictures(finished, tokenizer, args.output.parent, MASK_ID_LAVIDA)


def lavida_item(lm, model, processor, tokenizer, row: dict) -> dict:
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    question = row.get("question") or row["conversations"][0]["value"]
    if "Question:" in question:
        question = question.split("Question:")[-1].split("Answer:")[0].strip()
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    conv = copy.deepcopy(conv_templates["llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + f"Question: {q}\nAnswer:")
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = ids.unsqueeze(0).to(model.device)
    image = Image.open(row["image"]).convert("RGB")
    tensors = process_images([image], processor, model.config)
    tensors = [t.to(device=model.device, dtype=torch.bfloat16) for t in tensors]
    prepared = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, tensors, ["image"], image_sizes=[image.size], return_inputs=True
    )
    inputs_embeds = prepared[4]
    past = lm(None, input_embeddings=inputs_embeds, use_cache=True).attn_key_values
    prefix_len = int(past[0][0].shape[-2])
    x = torch.full((1, GEN), MASK_ID_LAVIDA, dtype=torch.long, device=inputs_embeds.device)
    unmask_step = [-1] * GEN
    canvas_ids = []
    canvas_text = []
    contrast = None
    for step in range(GEN):
        mask_row = x[0]
        embeds = lm.transformer.wte(x)
        KeyBlock.cols = None
        logits = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
        if logits.shape[1] != GEN:
            raise RuntimeError(f"canvas logits {tuple(logits.shape)}")
        if contrast is None and qualifies(mask_row, MASK_ID_LAVIDA):
            masked_pos = torch.where(mask_row.eq(MASK_ID_LAVIDA))[0]
            cols = (masked_pos + prefix_len).to(device=logits.device)
            KeyBlock.cols = cols
            try:
                blocked = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
            finally:
                KeyBlock.cols = None
            bi_id, bi_txt = argmax_piece(tokenizer, logits[0, 0], MASK_ID_LAVIDA)
            bl_id, bl_txt = argmax_piece(tokenizer, blocked[0, 0], MASK_ID_LAVIDA)
            contrast = {
                "contrast_step": step,
                "others_masked": int((mask_row[1:] == MASK_ID_LAVIDA).sum().item()),
                "n_blocked_keys": int(masked_pos.numel()),
                "argmax_bi_id": bi_id,
                "argmax_blocked_id": bl_id,
                "argmax_bi": bi_txt,
                "argmax_blocked": bl_txt,
                "argmax_changed": bi_id != bl_id,
            }
        pos = unmask_one(x, logits, MASK_ID_LAVIDA)
        unmask_step[pos] = step
        ids = [int(v) for v in x[0].tolist()]
        canvas_ids.append(ids)
        canvas_text.append([piece(tokenizer, tid, MASK_ID_LAVIDA) for tid in ids])
    if contrast is None:
        raise RuntimeError(f"{row['id']} never reached the early contrast")
    if sorted(unmask_step) != list(range(GEN)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return {
        "id": row["id"],
        "gold": row.get("gold"),
        "backend": "lavida",
        "pred": pred,
        "unmask_step": unmask_step,
        "letter_unmask_step": unmask_step[0],
        "letter_unmasks_last": unmask_step[0] == GEN - 1,
        "canvas_ids": canvas_ids,
        "canvas_text": canvas_text,
        **contrast,
    }


def run_mmada(args) -> None:
    sys.path.insert(0, str(PROJECT / "experiments"))
    sys.path.insert(0, str(PROJECT / "repos" / "MMaDA"))
    import evdlc_mmada as mm
    import sciqa_standard_eval as sq

    class Ns:
        model = args.model_path
        vq_model = args.vq_model
        mmada_repo = args.mmada_repo

    print("loading mmada", flush=True)
    device = torch.device("cuda:0")
    model, tokenizer = mm.load_mmada(Ns, device)
    vq = mm.load_vq(Ns, device)
    model.eval()
    print(f"loaded mmada mask={mm.MASK_ID}", flush=True)
    records = sq.load_manifest(Path(args.manifest), Path(args.data_dir))
    loader = None
    for name in dir(sq):
        obj = getattr(sq, name)
        if isinstance(obj, type) and hasattr(obj, "load_rgb"):
            loader = obj(Path(args.data_dir), records)
            break
    if loader is None:
        raise RuntimeError("ScienceQA image loader was not found")
    if args.max_samples > 0:
        records = records[: args.max_samples]
    done = load_done(args.output)
    finished = []
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    finished.append(json.loads(line))
    t0 = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        for row in records:
            if row["id"] in done:
                continue
            image = loader.load_rgb(row)
            question = sq.build_task_prompt(row)
            record = mmada_item(model, tokenizer, vq, mm, image, question, row, device)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            finished.append(record)
            n_done = len(finished)
            payload = {
                "backend": "mmada",
                "n_done": n_done,
                "n_total": len(records),
                "id": record["id"],
                "letter_unmask_step": record["letter_unmask_step"],
                "letter_unmasks_last": record["letter_unmasks_last"],
                "contrast_step": record["contrast_step"],
                "argmax_changed": record["argmax_changed"],
                "argmax_bi": record["argmax_bi"],
                "argmax_blocked": record["argmax_blocked"],
                "elapsed_sec": round(time.time() - t0, 1),
            }
            write_progress(args.progress, payload)
            print(
                f"PROGRESS n_done={n_done}/{len(records)} id={record['id']} "
                f"letter_unmask_step={record['letter_unmask_step']} "
                f"argmax_changed={int(record['argmax_changed'])} "
                f"contrast_step={record['contrast_step']}",
                flush=True,
            )
            if n_done <= 12 or n_done % 25 == 0:
                maybe_pictures(finished, tokenizer, args.output.parent, mm.MASK_ID)


def mmada_logits(model, prompt_ids, canvas, mask_id: int, block_canvas: torch.Tensor | None):
    x = torch.full(
        (1, prompt_ids.shape[1] + canvas.shape[1]),
        mask_id,
        dtype=torch.long,
        device=canvas.device,
    )
    x[:, : prompt_ids.shape[1]] = prompt_ids
    x[:, prompt_ids.shape[1] :] = canvas
    if block_canvas is None:
        return model(x).logits
    seq = x.shape[1]
    allow = torch.ones(1, 1, seq, seq, dtype=torch.bool, device=x.device)
    abs_cols = block_canvas.to(device=x.device) + prompt_ids.shape[1]
    allow[0, 0, :, abs_cols] = False
    return model(x, attention_bias=allow).logits


def mmada_item(model, tokenizer, vq, mm, image, question: str, row: dict, device) -> dict:
    vq_ids = mm.encode_vq(vq, image, device)
    prompt_ids, _text = mm.build_prompt_ids(tokenizer, question, vq_ids, device)
    prompt_len = int(prompt_ids.shape[1])
    x = torch.full((1, GEN), mm.MASK_ID, dtype=torch.long, device=device)
    unmask_step = [-1] * GEN
    canvas_ids = []
    canvas_text = []
    contrast = None
    for step in range(GEN):
        logits = mmada_logits(model, prompt_ids, x, mm.MASK_ID, None)
        if logits.shape[1] != prompt_len + GEN:
            raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
        answer_logits = logits[:, prompt_len:, :]
        if contrast is None and qualifies(x[0], mm.MASK_ID):
            masked_pos = torch.where(x[0].eq(mm.MASK_ID))[0]
            blocked = mmada_logits(model, prompt_ids, x, mm.MASK_ID, masked_pos)
            bi_id, bi_txt = argmax_piece(tokenizer, answer_logits[0, 0], mm.MASK_ID)
            bl_id, bl_txt = argmax_piece(tokenizer, blocked[0, prompt_len], mm.MASK_ID)
            contrast = {
                "contrast_step": step,
                "others_masked": int((x[0, 1:] == mm.MASK_ID).sum().item()),
                "n_blocked_keys": int(masked_pos.numel()),
                "argmax_bi_id": bi_id,
                "argmax_blocked_id": bl_id,
                "argmax_bi": bi_txt,
                "argmax_blocked": bl_txt,
                "argmax_changed": bi_id != bl_id,
            }
        pos = unmask_one(x, answer_logits, mm.MASK_ID)
        unmask_step[pos] = step
        ids = [int(v) for v in x[0].tolist()]
        canvas_ids.append(ids)
        canvas_text.append([piece(tokenizer, tid, mm.MASK_ID) for tid in ids])
    if contrast is None:
        raise RuntimeError(f"{row['id']} never reached the early contrast")
    if sorted(unmask_step) != list(range(GEN)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    gold = row.get("gold") or row.get("gold_option") or row.get("answer")
    return {
        "id": row["id"],
        "gold": gold,
        "backend": "mmada",
        "pred": pred,
        "unmask_step": unmask_step,
        "letter_unmask_step": unmask_step[0],
        "letter_unmasks_last": unmask_step[0] == GEN - 1,
        "canvas_ids": canvas_ids,
        "canvas_text": canvas_text,
        **contrast,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("lavida", "mmada"), required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--model-path", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    p.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"start backend={args.backend} output={args.output} cuda={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    try:
        if args.backend == "lavida":
            run_lavida(args)
        else:
            run_mmada(args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
