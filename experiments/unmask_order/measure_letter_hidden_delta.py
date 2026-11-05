#!/usr/bin/env python3
"""Record the letter-position hidden state at two decode times.

Time 0: the whole answer canvas is still masked.
Time pre: the forward that writes position 0, while that position is still masked.

No hidden-state edit, no weight update, no blocked keys, no Newton update.
The first-order score change is the RMSNorm Jacobian at time 0 applied to
h_pre - h_0, then the vocabulary row. The second-order term is one half of
the same vocabulary row applied to gamma times D2y(dh, dh).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import run_unmask_order as base  # noqa: E402


class Capture:
    armed = False
    index = 0
    h = None
    seq = None


def _pre_hook(_module, args):
    if not Capture.armed:
        return
    hidden = args[0]
    Capture.seq = int(hidden.shape[1])
    if Capture.seq <= Capture.index:
        Capture.h = None
        return
    Capture.h = hidden[0, Capture.index].detach().to(dtype=torch.float32).cpu()


def leading_letter(text: str | None) -> str | None:
    match = re.match(r"^([A-E])(?:$|[\s\.\:\)])", (text or "").strip())
    return match.group(1) if match else None


class Gate:
    fd_only = False
    analytic = None
    fd = None
    rel = None


def rms_jvp(hidden: torch.Tensor, direction: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    width = hidden.shape[0]
    mean_square = torch.dot(hidden, hidden) / width
    scale = mean_square + eps
    rms = torch.sqrt(scale)
    alignment = torch.dot(hidden, direction)
    directional = (direction - hidden * (alignment / (width * scale))) / rms
    return gamma * directional


def d2y(hidden: torch.Tensor, direction: torch.Tensor, eps: float) -> torch.Tensor:
    width = hidden.shape[0]
    scale = torch.dot(hidden, hidden) / width + eps
    alignment = torch.dot(hidden, direction) / width
    scale_inv_3_2 = scale ** -1.5
    scale_inv_5_2 = scale ** -2.5
    direction_sq = torch.dot(direction, direction)
    return (
        -2 * alignment * scale_inv_3_2 * direction
        - (direction_sq / width) * scale_inv_3_2 * hidden
        + 3 * (alignment ** 2) * scale_inv_5_2 * hidden
    )


def scalar_logit(hidden: torch.Tensor, gamma: torch.Tensor, eps: float, head_row: torch.Tensor) -> torch.Tensor:
    width = hidden.shape[0]
    scale = torch.dot(hidden, hidden) / width + eps
    normalized = hidden * torch.rsqrt(scale)
    return torch.dot(head_row, gamma * normalized)


def quadratic_form(hidden: torch.Tensor, direction: torch.Tensor, gamma: torch.Tensor, eps: float, head_row: torch.Tensor) -> torch.Tensor:
    return torch.dot(head_row, gamma * d2y(hidden, direction, eps))


def finite_difference_check(hidden: torch.Tensor, gamma: torch.Tensor, eps: float, head_row: torch.Tensor) -> None:
    hidden = hidden.detach().to(dtype=torch.float64).cpu()
    gamma = gamma.detach().to(dtype=torch.float64).cpu()
    head_row = head_row.detach().to(dtype=torch.float64).cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261009)
    direction = torch.randn(hidden.shape, generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)
    delta = 1e-2
    analytic = quadratic_form(hidden, direction, gamma, eps, head_row)
    plus = scalar_logit(hidden + delta * direction, gamma, eps, head_row)
    minus = scalar_logit(hidden - delta * direction, gamma, eps, head_row)
    base = scalar_logit(hidden, gamma, eps, head_row)
    fd = (plus + minus - 2 * base) / (delta * delta)
    denom = torch.maximum(analytic.abs(), fd.abs()).clamp_min(1e-30)
    rel = (analytic - fd).abs() / denom
    Gate.analytic = float(analytic)
    Gate.fd = float(fd)
    Gate.rel = float(rel)
    print(
        f"FD_CHECK analytic={Gate.analytic:.10g} fd={Gate.fd:.10g} rel={Gate.rel:.10g}",
        flush=True,
    )
    if Gate.rel >= 1e-3:
        print(f"FD_CHECK_FAIL analytic={Gate.analytic:.10g} fd={Gate.fd:.10g}", flush=True)
        raise RuntimeError(
            f"finite-difference check failed analytic={Gate.analytic} fd={Gate.fd} rel={Gate.rel}"
        )


def install_hook(norm) -> None:
    if getattr(norm, "_letter_delta_hook", False):
        return
    norm.register_forward_pre_hook(_pre_hook)
    norm._letter_delta_hook = True


def take_hidden(index: int) -> torch.Tensor:
    Capture.armed = True
    Capture.index = int(index)
    Capture.h = None
    Capture.seq = None


def read_hidden(expected_seq: int) -> torch.Tensor:
    Capture.armed = False
    if Capture.h is None:
        raise RuntimeError(f"hidden state was not captured, seq={Capture.seq} index={Capture.index}")
    if Capture.seq != expected_seq:
        raise RuntimeError(f"norm sequence {Capture.seq} != expected {expected_seq}")
    return Capture.h


def logit_at(row: torch.Tensor, token_id: int) -> float:
    return float(row[token_id].detach().float().cpu())


def predicted_delta(head_row: torch.Tensor, directional: torch.Tensor) -> float:
    return float(torch.dot(head_row.detach().float().cpu(), directional))


def check_reconstruction(hidden, gamma, eps, head, token_id: int, actual: float) -> None:
    mean_square = torch.dot(hidden, hidden) / hidden.shape[0]
    normalized = hidden * torch.rsqrt(mean_square + eps)
    reconstructed = float(torch.dot(head[token_id].detach().float().cpu(), gamma * normalized))
    if abs(reconstructed - actual) > 1.0:
        raise RuntimeError(
            f"position-0 logit reconstruction {reconstructed:.4f} vs actual {actual:.4f} "
            f"at token {token_id}"
        )


def measure_pair(h0, h_pre, logits0, logits_pre, head, gamma, eps, step0_id, final_id, gold_id):
    dh = h_pre - h0
    gamma_cpu = gamma.detach().float().cpu()
    directional = rms_jvp(h0, dh, gamma_cpu, eps)
    tokens = {
        "step0": step0_id,
        "final": final_id,
        "gold": gold_id,
    }
    out = {
        "dh_l2": float(torch.linalg.vector_norm(dh)),
    }
    for name, token_id in tokens.items():
        if token_id is None:
            out[f"actual_{name}"] = None
            out[f"predicted_{name}"] = None
            out[f"abs_err_{name}"] = None
            out[f"first_{name}"] = None
            out[f"second_{name}"] = None
            out[f"abs_err_first_{name}"] = None
            out[f"abs_err_second_{name}"] = None
            continue
        head_row = head[token_id].detach().float().cpu()
        actual = logit_at(logits_pre, token_id) - logit_at(logits0, token_id)
        first = float(torch.dot(head_row, directional))
        second = 0.5 * float(quadratic_form(h0, dh, gamma_cpu, eps, head_row))
        out[f"actual_{name}"] = actual
        out[f"predicted_{name}"] = first
        out[f"abs_err_{name}"] = abs(actual - first)
        out[f"first_{name}"] = first
        out[f"second_{name}"] = second
        out[f"abs_err_first_{name}"] = abs(actual - first)
        out[f"abs_err_second_{name}"] = abs(actual - first - second)
    return out


def gold_token_id(tokenizer, gold: str | None) -> int | None:
    if not gold or len(gold) != 1:
        return None
    ids = tokenizer.encode(gold, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    return None


def lavida_measure(lm, model, processor, tokenizer, row: dict, norm, head, gamma, eps) -> dict:
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
    conv = copy_conv(conv_templates["llada"])
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
    unmask_step = [-1] * base.GEN
    h0 = h_pre = None
    logits0 = logits_pre = None
    for step in range(base.GEN):
        embeds = lm.transformer.wte(x)
        take_hidden(0)
        logits = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
        hidden = read_hidden(base.GEN)
        if logits.shape[1] != base.GEN:
            raise RuntimeError(f"canvas logits {tuple(logits.shape)}")
        if step == 0:
            h0 = hidden
            logits0 = logits[0, 0].detach().float().cpu()
            step0_id = int(torch.argmax(logits0).item())
            check_reconstruction(h0, gamma, eps, head, step0_id, float(logits0[step0_id]))
            if Gate.fd_only:
                finite_difference_check(h0, gamma, eps, head[step0_id])
                return None
        pos = base.unmask_one(x, logits, base.MASK_ID_LAVIDA)
        unmask_step[pos] = step
        if pos == 0:
            h_pre = hidden
            logits_pre = logits[0, 0].detach().float().cpu()
    if h0 is None or h_pre is None:
        raise RuntimeError(f"{row['id']} missing letter hidden state")
    final_id = int(x[0, 0].item())
    step0_id = int(torch.argmax(logits0).item())
    pre_id = int(torch.argmax(logits_pre).item())
    gold = row.get("gold")
    measured = measure_pair(
        h0, h_pre, logits0, logits_pre, head, gamma, eps, step0_id, final_id, gold_token_id(tokenizer, gold)
    )
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return pack(
        row["id"], gold, "lavida", tokenizer, base.MASK_ID_LAVIDA,
        step0_id, pre_id, final_id, pred, unmask_step, measured,
    )


def copy_conv(template):
    import copy
    return copy.deepcopy(template)


def mmada_measure(model, tokenizer, vq, mm, image, question, row, device, norm, head, gamma, eps) -> dict:
    vq_ids = mm.encode_vq(vq, image, device)
    prompt_ids, _text = mm.build_prompt_ids(tokenizer, question, vq_ids, device)
    prompt_len = int(prompt_ids.shape[1])
    x = torch.full((1, base.GEN), mm.MASK_ID, dtype=torch.long, device=device)
    unmask_step = [-1] * base.GEN
    h0 = h_pre = None
    logits0 = logits_pre = None
    for step in range(base.GEN):
        take_hidden(prompt_len)
        logits = base.mmada_logits(model, prompt_ids, x, mm.MASK_ID, None)
        hidden = read_hidden(prompt_len + base.GEN)
        answer_logits = logits[:, prompt_len:, :]
        if answer_logits.shape[1] != base.GEN:
            raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
        if step == 0:
            h0 = hidden
            logits0 = answer_logits[0, 0].detach().float().cpu()
            step0_id = int(torch.argmax(logits0).item())
            check_reconstruction(h0, gamma, eps, head, step0_id, float(logits0[step0_id]))
            if Gate.fd_only:
                finite_difference_check(h0, gamma, eps, head[step0_id])
                return None
        pos = base.unmask_one(x, answer_logits, mm.MASK_ID)
        unmask_step[pos] = step
        if pos == 0:
            h_pre = hidden
            logits_pre = answer_logits[0, 0].detach().float().cpu()
    if h0 is None or h_pre is None:
        raise RuntimeError(f"{row['id']} missing letter hidden state")
    final_id = int(x[0, 0].item())
    step0_id = int(torch.argmax(logits0).item())
    pre_id = int(torch.argmax(logits_pre).item())
    gold = row.get("gold") or row.get("gold_option") or row.get("answer")
    measured = measure_pair(
        h0, h_pre, logits0, logits_pre, head, gamma, eps, step0_id, final_id, gold_token_id(tokenizer, gold)
    )
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return pack(
        row["id"], gold, "mmada", tokenizer, mm.MASK_ID,
        step0_id, pre_id, final_id, pred, unmask_step, measured,
    )


def pack(item_id, gold, backend, tokenizer, mask_id, step0_id, pre_id, final_id, pred, unmask_step, measured):
    step0_txt = base.piece(tokenizer, step0_id, mask_id)
    pre_txt = base.piece(tokenizer, pre_id, mask_id)
    final_txt = base.piece(tokenizer, final_id, mask_id)
    finished = leading_letter(pred)
    record = {
        "id": item_id,
        "gold": gold,
        "backend": backend,
        "step0_token": step0_txt,
        "pre_token": pre_txt,
        "final_token": final_txt,
        "finished_letter": finished,
        "letter_unmask_step": unmask_step[0],
        "step0_correct": leading_letter(step0_txt) == gold,
        "pre_correct": leading_letter(pre_txt) == gold,
        "finished_correct": finished == gold,
        "argmax_changed": step0_id != pre_id,
        "hidden_state_edited": False,
    }
    record.update(measured)
    return record


def language_core(module):
    if hasattr(module, "transformer") and hasattr(module.transformer, "ln_f"):
        return module
    inner = getattr(module, "model", None)
    if inner is not None and hasattr(inner, "transformer") and hasattr(inner.transformer, "ln_f"):
        return inner
    raise RuntimeError(f"{type(module).__name__} has no transformer.ln_f")


def head_of(lm):
    lm = language_core(lm)
    norm = lm.transformer.ln_f
    if type(norm).__name__ != "RMSLayerNorm":
        raise RuntimeError(f"expected RMSLayerNorm, got {type(norm).__name__}")
    head = lm.transformer.ff_out
    install_hook(norm)
    gamma = norm.weight.detach().float().cpu()
    return norm, head.weight, gamma, float(norm.eps)


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
        args.model_path, None, "llava_llada", device_map="cuda:0",
        vision_kwargs=vision_kwargs, torch_dtype="bfloat16",
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    lm = model.get_model()
    norm, head, gamma, eps = head_of(lm)
    print(f"lavida norm={type(norm).__name__} eps={eps}", flush=True)
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    loop(args, rows, lambda row: lavida_measure(lm, model, processor, tokenizer, row, norm, head, gamma, eps))


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
    norm, head, gamma, eps = head_of(model)
    print(f"mmada norm={type(norm).__name__} eps={eps} mask={mm.MASK_ID}", flush=True)
    records = sq.load_manifest(Path(args.manifest), Path(args.data_dir))
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
        return mmada_measure(model, tokenizer, vq, mm, image, question, row, device, norm, head, gamma, eps)

    loop(args, records, one)


def loop(args, rows, fn) -> None:
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    done = base.load_done(args.output)
    if not done:
        if not rows:
            raise RuntimeError("no items to measure")
        print(f"FD preflight id={rows[0]['id']}", flush=True)
        Gate.fd_only = True
        try:
            fn(rows[0])
        finally:
            Gate.fd_only = False
        if Gate.rel is None:
            raise RuntimeError("finite-difference check did not run")
        print(
            f"FD_CHECK_PASS analytic={Gate.analytic:.10g} fd={Gate.fd:.10g} rel={Gate.rel:.10g}",
            flush=True,
        )
    finished = len(done)
    t0 = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            record = fn(row)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            finished += 1
            payload = {
                "backend": record["backend"],
                "n_done": finished,
                "n_total": len(rows),
                "id": record["id"],
                "letter_unmask_step": record["letter_unmask_step"],
                "argmax_changed": record["argmax_changed"],
                "dh_l2": record["dh_l2"],
                "abs_err_final": record["abs_err_final"],
                "abs_err_second_final": record["abs_err_second_final"],
                "second_final": record["second_final"],
                "fd_analytic": Gate.analytic,
                "fd_fd": Gate.fd,
                "fd_rel": Gate.rel,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            base.write_progress(args.progress, payload)
            print(
                f"PROGRESS n_done={finished}/{len(rows)} id={record['id']} "
                f"dh_l2={record['dh_l2']:.4f} abs_err_final={record['abs_err_final']}",
                flush=True,
            )


def main() -> None:
    args = base.parse_args()
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    old_output = (PROJECT / "outputs/letter_hidden_delta_20261009").resolve()
    output_dir = args.output.resolve().parent
    if output_dir == old_output or old_output in args.output.resolve().parents:
        raise RuntimeError(f"refusing to write into {old_output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    print(f"start backend={args.backend} output={args.output}", flush=True)
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
