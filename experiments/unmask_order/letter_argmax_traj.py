#!/usr/bin/env python3
"""Record position-0 argmax while that canvas position is still masked.

Normal low-confidence unmask, gen length 16, temperature 0.
No token override. No hidden-state edit. No weight update.
At each step, before unmasking, if position 0 is still masked, store the
argmax token text at position 0. Also store letter_unmask_step.
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
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

FORBIDDEN = (
    (PROJECT / "outputs/letter_hidden_delta_20261009").resolve(),
    (PROJECT / "outputs/letter_hidden_delta2_20261009").resolve(),
    (PROJECT / "outputs/unmask_order_bidir_20261009").resolve(),
)


def repair_tail(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, sep, _tail = data.rpartition(b"\n")
    path.write_bytes(head + sep)


def refuse_forbidden(path: Path) -> None:
    resolved = path.resolve()
    for banned in FORBIDDEN:
        if resolved == banned or banned in resolved.parents:
            raise RuntimeError(f"refusing to write into {banned}")


def gold_of(row: dict) -> str | None:
    gold = row.get("gold") or row.get("gold_option") or row.get("answer")
    if gold is None:
        return None
    return str(gold).strip()


def lavida_decode(lm, model, processor, tokenizer, row: dict) -> dict:
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
    conv = delta.copy_conv(conv_templates["llada"])
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
    x = torch.full((1, base.GEN), base.MASK_ID_LAVIDA, dtype=torch.long, device=inputs_embeds.device)
    return decode_canvas(
        x,
        base.MASK_ID_LAVIDA,
        tokenizer,
        forward=lambda canvas: lavida_forward(lm, past, canvas),
    )


def lavida_forward(lm, past, canvas):
    embeds = lm.transformer.wte(canvas)
    logits = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
    if logits.shape[1] != base.GEN:
        raise RuntimeError(f"canvas logits {tuple(logits.shape)}")
    return logits


def mmada_decode(model, tokenizer, vq, mm, image, question, device) -> dict:
    vq_ids = mm.encode_vq(vq, image, device)
    prompt_ids, _text = mm.build_prompt_ids(tokenizer, question, vq_ids, device)
    prompt_len = int(prompt_ids.shape[1])
    x = torch.full((1, base.GEN), mm.MASK_ID, dtype=torch.long, device=device)

    def forward(canvas):
        logits = base.mmada_logits(model, prompt_ids, canvas, mm.MASK_ID, None)
        if logits.shape[1] != prompt_len + base.GEN:
            raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
        return logits[:, prompt_len:, :]

    return decode_canvas(x, mm.MASK_ID, tokenizer, forward=forward)


def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
    unmask_step = [-1] * base.GEN
    pos0_argmax = []
    for step in range(base.GEN):
        logits = forward(x)
        if int(x[0, 0].item()) == int(mask_id):
            token_id = int(torch.argmax(logits[0, 0]).item())
            pos0_argmax.append(base.piece(tokenizer, token_id, mask_id))
        pos = base.unmask_one(x, logits, mask_id)
        unmask_step[pos] = step
    letter_step = unmask_step[0]
    if letter_step < 0:
        raise RuntimeError("letter position was never selected")
    if len(pos0_argmax) != letter_step + 1:
        raise RuntimeError(
            f"recorded {len(pos0_argmax)} argmax steps, letter unmasked at {letter_step}"
        )
    if sorted(unmask_step) != list(range(base.GEN)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return {
        "letter_unmask_step": int(letter_step),
        "pos0_argmax": pos0_argmax,
        "finished_letter": delta.leading_letter(pred),
    }


def pack(row: dict, backend: str, decoded: dict) -> dict:
    gold = gold_of(row)
    return {
        "id": row["id"],
        "gold": gold,
        "backend": backend,
        "letter_unmask_step": decoded["letter_unmask_step"],
        "pos0_argmax": decoded["pos0_argmax"],
        "finished_letter": decoded["finished_letter"],
    }


def run_lavida(args) -> None:
    sys.path.insert(0, str(PROJECT / "repos" / "LaViDa"))
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
    if not hasattr(lm, "transformer"):
        raise RuntimeError(f"LaViDa get_model() returned {type(lm).__name__} without transformer")
    print(f"lavida core={type(lm).__name__} cuda={args.cuda}", flush=True)
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if len(rows) != 2017:
        raise RuntimeError(f"expected 2017 LaViDa items, got {len(rows)}")
    loop(args, rows, lambda row: lavida_decode(lm, model, processor, tokenizer, row))


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
    device = torch.device(f"cuda:{args.cuda}")
    model, tokenizer = mm.load_mmada(Ns, device)
    if type(model).__name__ != "MMadaModelLM":
        raise RuntimeError(f"expected MMadaModelLM, got {type(model).__name__}")
    vq = mm.load_vq(Ns, device)
    model.eval()
    core = model.model
    print(
        f"mmada outer={type(model).__name__} core={type(core).__name__} "
        f"mask={mm.MASK_ID} cuda={args.cuda}",
        flush=True,
    )
    records = sq.load_manifest(Path(args.manifest), Path(args.data_dir))
    if len(records) != 2017:
        raise RuntimeError(f"expected 2017 MMaDA items, got {len(records)}")
    loader = None
    for name in dir(sq):
        obj = getattr(sq, name)
        if isinstance(obj, type) and hasattr(obj, "load_rgb"):
            loader = obj(Path(args.data_dir), records)
            break
    if loader is None:
        raise RuntimeError("ScienceQA image loader was not found")

    def one(row):
        image = loader.load_rgb(row)
        question = sq.build_task_prompt(row)
        return mmada_decode(model, tokenizer, vq, mm, image, question, device)

    loop(args, records, one)


def loop(args, rows, fn) -> None:
    done = base.load_done(args.output)
    finished = len(done)
    t0 = time.time()
    print(f"resume n_done={finished}/{len(rows)}", flush=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            with torch.no_grad():
                decoded = fn(row)
            record = pack(row, args.backend, decoded)
            if len(record["pos0_argmax"]) != record["letter_unmask_step"] + 1:
                raise RuntimeError(f"{record['id']} trajectory length mismatch")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(record["id"])
            finished += 1
            payload = {
                "backend": args.backend,
                "n_done": finished,
                "n_total": len(rows),
                "id": record["id"],
                "letter_unmask_step": record["letter_unmask_step"],
                "elapsed_sec": round(time.time() - t0, 1),
            }
            base.write_progress(args.progress, payload)
            print(
                f"PROGRESS n_done={finished}/{len(rows)} id={record['id']} "
                f"letter_unmask_step={record['letter_unmask_step']}",
                flush=True,
            )
    if finished != len(rows):
        raise RuntimeError(f"finished {finished} of {len(rows)}")
    print(f"SUMMARY n={finished}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("lavida", "mmada"), required=True)
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    parser.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    args = parser.parse_args()
    if base.GEN != 16:
        raise RuntimeError(f"expected gen length 16, got {base.GEN}")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    refuse_forbidden(args.output)
    refuse_forbidden(args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    repair_tail(args.output)
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_handle = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"trajectory already running for {args.output}") from exc
    torch.cuda.set_device(args.cuda)
    print(
        f"start backend={args.backend} cuda={args.cuda} gen={base.GEN} temperature=0 "
        f"override=0 output={args.output}",
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
