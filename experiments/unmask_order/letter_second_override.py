#!/usr/bin/env python3
"""Write the step-0 argmax at canvas position 0 when |second| is large.

Normal low-confidence unmask, gen length 16, temperature 0.
At the forward that selects position 0, if the absolute second-order term of
RMSNorm+ff_out for the token about to be written is at least the frozen
threshold, write the step-0 argmax id instead. No other edit. No weight update.
Gold is recorded after the decision and is not an input to it.

Thresholds are the frozen 75th percentiles of |second_final| from
outputs/letter_hidden_delta2_20261009. This script does not recompute them.
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
import torch.nn.functional as F

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

# Frozen. Do not recompute from this run.
THRESHOLDS = {
    "lavida": 2.3044655,
    "mmada": 0.12879625,
}
FORBIDDEN = (
    (PROJECT / "outputs/letter_hidden_delta_20261009").resolve(),
    (PROJECT / "outputs/letter_hidden_delta2_20261009").resolve(),
)


def second_order(h0: torch.Tensor, h_pre: torch.Tensor, head, gamma, eps: float, token_new: int) -> float:
    direction = h_pre - h0
    head_row = head[token_new].detach().to(dtype=torch.float32).cpu()
    return 0.5 * float(delta.quadratic_form(h0, direction, gamma, eps, head_row))


def commit_position(
    x, logits, mask_id: int, h0, h_pre, step0_id: int, head, gamma, eps: float, threshold: float, allow_override: bool
):
    """Same low-confidence choice as unmask_one, with one optional letter rewrite."""
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
    token_new = int(chosen[0, pos].item())
    fired = False
    second = None
    token_write = token_new
    if pos == 0 and allow_override:
        second = second_order(h0, h_pre, head, gamma, eps, token_new)
        if abs(second) >= threshold:
            token_write = int(step0_id)
            fired = True
    x[0, pos] = token_write
    return pos, token_new, token_write, fired, second


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


def lavida_decode(lm, model, processor, tokenizer, row: dict, head, gamma, eps: float, threshold: float) -> dict:
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
        head,
        gamma,
        eps,
        threshold,
        forward=lambda canvas: lavida_forward(lm, past, canvas),
    )


def lavida_forward(lm, past, canvas):
    embeds = lm.transformer.wte(canvas)
    delta.take_hidden(0)
    logits = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
    hidden = delta.read_hidden(base.GEN)
    if logits.shape[1] != base.GEN:
        raise RuntimeError(f"canvas logits {tuple(logits.shape)}")
    return logits, hidden


def mmada_decode(model, tokenizer, vq, mm, image, question, row, device, head, gamma, eps, threshold) -> dict:
    vq_ids = mm.encode_vq(vq, image, device)
    prompt_ids, _text = mm.build_prompt_ids(tokenizer, question, vq_ids, device)
    prompt_len = int(prompt_ids.shape[1])
    x = torch.full((1, base.GEN), mm.MASK_ID, dtype=torch.long, device=device)

    def forward(canvas):
        delta.take_hidden(prompt_len)
        logits = base.mmada_logits(model, prompt_ids, canvas, mm.MASK_ID, None)
        hidden = delta.read_hidden(prompt_len + base.GEN)
        if logits.shape[1] != prompt_len + base.GEN:
            raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
        return logits[:, prompt_len:, :], hidden

    return decode_canvas(
        x,
        mm.MASK_ID,
        tokenizer,
        head,
        gamma,
        eps,
        threshold,
        forward=forward,
    )


def decode_canvas(x, mask_id, tokenizer, head, gamma, eps, threshold, forward) -> dict:
    unmask_step = [-1] * base.GEN
    h0 = None
    step0_id = None
    wrote = None
    fired = False
    second = None
    pre_id = None
    for step in range(base.GEN):
        logits, hidden = forward(x)
        if step == 0:
            h0 = hidden
            logits0 = logits[0, 0].detach().float().cpu()
            step0_id = int(torch.argmax(logits0).item())
            delta.check_reconstruction(h0, gamma, eps, head, step0_id, float(logits0[step0_id]))
        pos, token_new, token_write, this_fired, this_second = commit_position(
            x, logits, mask_id, h0, hidden, step0_id, head, gamma, eps, threshold, wrote is None
        )
        unmask_step[pos] = step
        if pos == 0 and wrote is None:
            wrote = token_write
            fired = this_fired
            second = this_second
            pre_id = token_new
    if h0 is None or step0_id is None or wrote is None or pre_id is None or second is None:
        raise RuntimeError("letter position was never selected")
    if sorted(unmask_step) != list(range(base.GEN)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    if int(x[0, 0].item()) != int(wrote):
        raise RuntimeError(f"position 0 holds {int(x[0, 0].item())} after committing {wrote}")
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return {
        "step0_id": int(step0_id),
        "step0_token": base.piece(tokenizer, step0_id, mask_id),
        "pre_id": int(pre_id),
        "pre_token": base.piece(tokenizer, pre_id, mask_id),
        "written_id": int(wrote),
        "written_token": base.piece(tokenizer, wrote, mask_id),
        "override_fired": bool(fired),
        "second": float(second),
        "finished_letter": delta.leading_letter(pred),
        "letter_unmask_step": unmask_step[0],
        "pred": pred,
    }


def pack(row: dict, backend: str, decoded: dict) -> dict:
    gold = gold_of(row)
    finished = decoded["finished_letter"]
    step0_letter = delta.leading_letter(decoded["step0_token"])
    return {
        "id": row["id"],
        "gold": gold,
        "backend": backend,
        "step0_token": decoded["step0_token"],
        "pre_token": decoded["pre_token"],
        "written_token": decoded["written_token"],
        "override_fired": decoded["override_fired"],
        "second": decoded["second"],
        "finished_letter": finished,
        "finished_correct": finished == gold,
        "step0_correct": step0_letter == gold,
        "letter_unmask_step": decoded["letter_unmask_step"],
    }


def prepare_head(module, cuda: int):
    norm, head, gamma, eps = delta.head_of(module)
    if abs(float(eps) - 1e-5) > 1e-12:
        raise RuntimeError(f"expected RMSNorm eps 1e-5, got {eps}")
    if int(norm.weight.device.index) != int(cuda):
        raise RuntimeError(f"RMSNorm weight on {norm.weight.device}, expected cuda:{cuda}")
    if head.dim() != 2 or head.shape[0] < head.shape[1]:
        raise RuntimeError(f"unexpected ff_out weight shape {tuple(head.shape)}")
    return head, gamma, float(eps)


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
    head, gamma, eps = prepare_head(lm, args.cuda)
    print(f"lavida core={type(lm).__name__} eps={eps} cuda={args.cuda}", flush=True)
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if len(rows) != 2017:
        raise RuntimeError(f"expected 2017 LaViDa items, got {len(rows)}")
    loop(args, rows, lambda row: lavida_decode(
        lm, model, processor, tokenizer, row, head, gamma, eps, THRESHOLDS["lavida"]
    ))


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
    head, gamma, eps = prepare_head(core, args.cuda)
    print(
        f"mmada outer={type(model).__name__} core={type(core).__name__} eps={eps} "
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
        return mmada_decode(
            model, tokenizer, vq, mm, image, question, row, device, head, gamma, eps, THRESHOLDS["mmada"]
        )

    loop(args, records, one)


def loop(args, rows, fn) -> None:
    done = base.load_done(args.output)
    finished = len(done)
    correct = 0
    overrides = 0
    step0_correct = 0
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                correct += int(bool(record["finished_correct"]))
                overrides += int(bool(record["override_fired"]))
                step0_correct += int(bool(record["step0_correct"]))
    t0 = time.time()
    print(f"resume n_done={finished}/{len(rows)}", flush=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            with torch.no_grad():
                decoded = fn(row)
            record = pack(row, args.backend, decoded)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(record["id"])
            finished += 1
            correct += int(bool(record["finished_correct"]))
            overrides += int(bool(record["override_fired"]))
            step0_correct += int(bool(record["step0_correct"]))
            payload = {
                "backend": args.backend,
                "n_done": finished,
                "n_total": len(rows),
                "id": record["id"],
                "override_fired": record["override_fired"],
                "override_count": overrides,
                "finished_correct_count": correct,
                "step0_correct_count": step0_correct,
                "second": record["second"],
                "letter_unmask_step": record["letter_unmask_step"],
                "elapsed_sec": round(time.time() - t0, 1),
            }
            base.write_progress(args.progress, payload)
            print(
                f"PROGRESS n_done={finished}/{len(rows)} id={record['id']} "
                f"override={int(record['override_fired'])} second={record['second']:.6g} "
                f"finished_correct={correct} step0_correct={step0_correct}",
                flush=True,
            )
    if finished != len(rows):
        raise RuntimeError(f"finished {finished} of {len(rows)}")
    print(
        f"SUMMARY n={finished} override={overrides} finished_correct={correct} "
        f"step0_correct={step0_correct}",
        flush=True,
    )


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
        raise RuntimeError(f"override already running for {args.output}") from exc
    torch.cuda.set_device(args.cuda)
    print(
        f"start backend={args.backend} cuda={args.cuda} threshold={THRESHOLDS[args.backend]} "
        f"gen={base.GEN} temperature=0 output={args.output}",
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
