#!/usr/bin/env python3
"""Commit canvas position 0 at decode step 12, then finish low-confidence decode.

Same 16-step temperature-0 low-confidence loop as letter_argmax_traj.py.
Step k is the same index as pos0_argmax[k]: the forward while that canvas is
still the one produced by steps 0..k-1, before this step writes a token.

At k=12, if position 0 is still the mask token, write the position-0 argmax
from that forward and count it as this step's only unmask. Steps 13..15 then
run the normal low-confidence rule, so the still-masked positions are chosen
from forwards that see the written letter. If position 0 was already unmasked,
leave it unchanged and let step 12 unmask some other position.

Gold is not an input. No curvature threshold. No image-key block.
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
sys.path.insert(0, str(PROJECT / "experiments" / "caft"))
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402
from graders import scienceqa_mc  # noqa: E402

COMMIT_STEP = 12
N_TOTAL = 2017
STOCK_LETTER = {"lavida": 1460, "mmada": 1059}
TRAJ_DIR = PROJECT / "outputs" / "letter_argmax_traj_20261010"
FORBIDDEN = (
    (PROJECT / "outputs/letter_hidden_delta_20261009").resolve(),
    (PROJECT / "outputs/letter_hidden_delta2_20261009").resolve(),
    (PROJECT / "outputs/letter_second_override_20261010").resolve(),
    (PROJECT / "outputs/letter_argmax_traj_20261010").resolve(),
    (PROJECT / "outputs/unmask_order_bidir_20261009").resolve(),
    (PROJECT / "outputs/image_key_block_20261010").resolve(),
    (PROJECT / "outputs/image_key_blocked_delta2_20261010").resolve(),
)


def repair_tail(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, sep, _tail = data.rpartition(b"\n")
    path.write_bytes(head + sep)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


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


def official_row(row: dict) -> dict | None:
    """Fields scienceqa_mc reads, taken from the item. No second model pass."""
    gold = gold_of(row)
    if not gold:
        return None
    if row.get("gold") and (row.get("gold_text") or row.get("golds")):
        return row
    choices = row.get("choices")
    idx = row.get("answer_idx", row.get("answer_index"))
    if isinstance(choices, list) and idx is not None:
        return {"gold": gold, "gold_text": str(choices[int(idx)])}
    return None


def load_traj(backend: str) -> list[dict]:
    path = TRAJ_DIR / f"{backend}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != N_TOTAL:
        raise RuntimeError(f"{path} has {len(rows)} rows, expected {N_TOTAL}")
    return rows


def check_step(record: dict, traj_row: dict, announce: bool) -> None:
    step = int(traj_row["letter_unmask_step"])
    still_masked = step >= COMMIT_STEP
    forced = bool(record["forced_at_step12"])
    expected = None
    token_ok = True
    if still_masked:
        pieces = traj_row["pos0_argmax"]
        if len(pieces) <= COMMIT_STEP:
            raise RuntimeError(
                f"{record['id']} traj letter_unmask_step={step} but pos0_argmax has {len(pieces)} entries"
            )
        expected = pieces[COMMIT_STEP]
        token_ok = record["written_token"] == expected
    ok = forced == still_masked and token_ok
    if announce or not ok:
        print(
            f"STEP_CHECK id={record['id']} forced={int(forced)} "
            f"written_id={record['written_id']} written_token={record['written_token']!r} "
            f"traj_pos0_argmax_12={expected!r} traj_letter_unmask_step={step} match={int(ok)}",
            flush=True,
        )
    if not ok:
        raise RuntimeError(
            f"step-index check failed id={record['id']} forced={forced} "
            f"traj_letter_unmask_step={step} written_id={record['written_id']} "
            f"written_token={record['written_token']!r} traj_pos0_argmax_12={expected!r}"
        )


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
    forced = False
    written_id = None
    for step in range(base.GEN):
        logits = forward(x)
        if step == COMMIT_STEP and int(x[0, 0].item()) == int(mask_id):
            token_id = int(torch.argmax(logits[0, 0]).item())
            if token_id == int(mask_id):
                raise RuntimeError("position-0 argmax at step 12 is the mask token")
            x[0, 0] = token_id
            unmask_step[0] = step
            forced = True
            written_id = token_id
            continue
        pos = base.unmask_one(x, logits, mask_id)
        if unmask_step[pos] >= 0:
            raise RuntimeError(f"position {pos} unmasked twice")
        unmask_step[pos] = step
    if sorted(unmask_step) != list(range(base.GEN)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    if forced:
        if unmask_step[0] != COMMIT_STEP or int(x[0, 0].item()) != int(written_id):
            raise RuntimeError("forced letter was not kept at position 0")
    elif unmask_step[0] >= COMMIT_STEP:
        raise RuntimeError("position 0 was still masked at step 12 but was not committed")
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return {
        "forced_at_step12": forced,
        "written_id": None if written_id is None else int(written_id),
        "written_token": None if written_id is None else base.piece(tokenizer, written_id, mask_id),
        "finished_text": pred,
        "finished_letter": delta.leading_letter(pred),
        "letter_unmask_step": int(unmask_step[0]),
    }


def pack(row: dict, backend: str, decoded: dict) -> dict:
    gold = gold_of(row)
    finished = decoded["finished_letter"]
    graded = official_row(row)
    official = None if graded is None else bool(scienceqa_mc(decoded["finished_text"], graded))
    return {
        "id": row["id"],
        "gold": gold,
        "backend": backend,
        "commit_step": COMMIT_STEP,
        "forced_at_step12": decoded["forced_at_step12"],
        "written_id": decoded["written_id"],
        "written_token": decoded["written_token"],
        "finished_text": decoded["finished_text"],
        "finished_letter": finished,
        "finished_letter_correct": finished == gold,
        "scienceqa_mc": official,
        "letter_unmask_step": decoded["letter_unmask_step"],
    }


def tally(path: Path) -> dict:
    counts = {"n": 0, "forced": 0, "letter": 0, "official": 0, "official_scored": 0}
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            counts["n"] += 1
            counts["forced"] += int(bool(record["forced_at_step12"]))
            counts["letter"] += int(bool(record["finished_letter_correct"]))
            if record["scienceqa_mc"] is not None:
                counts["official_scored"] += 1
                counts["official"] += int(bool(record["scienceqa_mc"]))
    return counts


def run_lavida(args, rows) -> None:
    sys.path.insert(0, str(PROJECT / "repos" / "LaViDa"))
    from llava.model.builder import load_pretrained_model

    if base.MASK_ID_LAVIDA != 126336:
        raise RuntimeError(f"expected LaViDa mask id 126336, got {base.MASK_ID_LAVIDA}")
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
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if int(mask_token_id) != 126336:
        raise RuntimeError(f"LaViDa config mask_token_id={mask_token_id}")
    print(
        f"lavida core={type(lm).__name__} mask={base.MASK_ID_LAVIDA} "
        f"prefix_lm={getattr(model.config, 'prefix_lm', 'unset')} cuda={args.cuda}",
        flush=True,
    )
    loop(args, rows, lambda row: lavida_decode(lm, model, processor, tokenizer, row))


def run_mmada(args, rows) -> None:
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
    if hasattr(model, "transformer"):
        raise RuntimeError("MMadaModelLM unexpectedly has .transformer; language core is model.model.transformer")
    core = model.model
    if not hasattr(core, "transformer"):
        raise RuntimeError(f"MMaDA language core {type(core).__name__} has no transformer")
    vq = mm.load_vq(Ns, device)
    model.eval()
    print(
        f"mmada outer={type(model).__name__} core={type(core).__name__} "
        f"language={type(core.transformer).__name__} mask={mm.MASK_ID} cuda={args.cuda}",
        flush=True,
    )
    loader = None
    for name in dir(sq):
        obj = getattr(sq, name)
        if isinstance(obj, type) and hasattr(obj, "load_rgb"):
            loader = obj(Path(args.data_dir), rows)
            break
    if loader is None:
        raise RuntimeError("ScienceQA image loader was not found")

    def one(row):
        image = loader.load_rgb(row)
        question = sq.build_task_prompt(row)
        return mmada_decode(model, tokenizer, vq, mm, image, question, device)

    loop(args, rows, one)


def loop(args, rows, fn) -> None:
    done = base.load_done(args.output)
    counts = tally(args.output)
    finished = counts["n"]
    if finished != len(done):
        raise RuntimeError(f"jsonl has {finished} records but {len(done)} ids")
    t0 = time.time()
    seen = 0
    print(f"resume n_done={finished}/{len(rows)}", flush=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            with torch.no_grad():
                decoded = fn(row)
            record = pack(row, args.backend, decoded)
            seen += 1
            check_step(record, args.traj_by_id[row["id"]], announce=seen <= 8)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(record["id"])
            finished += 1
            counts["forced"] += int(bool(record["forced_at_step12"]))
            counts["letter"] += int(bool(record["finished_letter_correct"]))
            if record["scienceqa_mc"] is not None:
                counts["official_scored"] += 1
                counts["official"] += int(bool(record["scienceqa_mc"]))
            payload = {
                "backend": args.backend,
                "n_done": finished,
                "n_total": len(rows),
                "id": record["id"],
                "commit_step": COMMIT_STEP,
                "forced_at_step12": record["forced_at_step12"],
                "forced_count": counts["forced"],
                "written_id": record["written_id"],
                "written_token": record["written_token"],
                "finished_letter": record["finished_letter"],
                "letter_correct_count": counts["letter"],
                "scienceqa_mc_count": counts["official"],
                "scienceqa_mc_scored": counts["official_scored"],
                "elapsed_sec": round(time.time() - t0, 1),
            }
            base.write_progress(args.progress, payload)
            print(
                f"PROGRESS n_done={finished}/{len(rows)} id={record['id']} "
                f"forced={int(record['forced_at_step12'])} written={record['written_token']!r} "
                f"letter={record['finished_letter']} letter_correct={counts['letter']} "
                f"scienceqa_mc={counts['official']}/{counts['official_scored']}",
                flush=True,
            )
    if finished != len(rows):
        raise RuntimeError(f"finished {finished} of {len(rows)}")
    stock = STOCK_LETTER[args.backend]
    official_text = (
        f"{counts['official']}/{counts['official_scored']}"
        if counts["official_scored"] == finished
        else "unavailable"
    )
    print(
        f"SUMMARY backend={args.backend} n={finished} forced_at_step12={counts['forced']} "
        f"letter_correct={counts['letter']}/{finished} stock_letter={stock}/{finished} "
        f"letter_minus_stock={counts['letter'] - stock} scienceqa_mc={official_text}",
        flush=True,
    )


def load_rows(args) -> list[dict]:
    if args.backend == "lavida":
        rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    else:
        sys.path.insert(0, str(PROJECT / "experiments"))
        import sciqa_standard_eval as sq

        rows = sq.load_manifest(Path(args.manifest), Path(args.data_dir))
    if len(rows) != N_TOTAL:
        raise RuntimeError(f"expected {N_TOTAL} items, got {len(rows)}")
    return rows


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
    if not 0 <= COMMIT_STEP < base.GEN:
        raise RuntimeError(f"commit step {COMMIT_STEP} outside 0..{base.GEN - 1}")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    refuse_forbidden(args.output)
    refuse_forbidden(args.progress)
    if line_count(args.output) >= N_TOTAL:
        print(f"ALREADY_DONE n={line_count(args.output)} {args.output}", flush=True)
        return
    rows = load_rows(args)
    traj_rows = load_traj(args.backend)
    if [row["id"] for row in rows] != [row["id"] for row in traj_rows]:
        raise RuntimeError(f"{args.backend} item order does not match {TRAJ_DIR}")
    args.traj_by_id = {row["id"]: row for row in traj_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_handle = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"letter commit already running for {args.output}") from exc
    repair_tail(args.output)
    torch.cuda.set_device(args.cuda)
    print(
        f"start backend={args.backend} cuda={args.cuda} gen={base.GEN} steps={base.GEN} "
        f"block_length={base.GEN} temperature=0 remasking=low_confidence commit_step={COMMIT_STEP} "
        f"gold_choice=0 curvature=0 image_key_block=0 output={args.output}",
        flush=True,
    )
    try:
        if args.backend == "lavida":
            run_lavida(args, rows)
        else:
            run_mmada(args, rows)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
