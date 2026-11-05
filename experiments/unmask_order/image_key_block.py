#!/usr/bin/env python3
"""Image-key block on the ScienceQA image split.

This is not the image-minus-text Jacobian. Gold is never used to choose a token.

contrast: at step 0, while the answer canvas is still fully masked, two frozen
forwards. Image keys stay visible, then image keys are blocked. Question tokens
and still-masked answer keys stay visible. No hidden edit and no decode rewrite.

delta2: the normal 16-step low-confidence unmask, with image keys blocked on
every canvas forward. At canvas index 0, record the float64 RMSNorm second-order
term of the token about to be written. Do not apply a curvature rewrite.

Image-key columns are read from that item's input ids. LaViDa uses the expanded
IMAGE_TOKEN_INDEX span. MMaDA uses the VQ codes between its image markers.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

N_TOTAL = 2017
STOCK_CORRECT = {"lavida": 1460, "mmada": 1059}
OLD_ABS_SECOND_MEDIAN = {"lavida": 1.0902, "mmada": 0.0857}
CONTRAST_DIR = PROJECT / "outputs" / "image_key_block_20261010"
FORBIDDEN = (
    (PROJECT / "outputs/letter_hidden_delta_20261009").resolve(),
    (PROJECT / "outputs/letter_hidden_delta2_20261009").resolve(),
    (PROJECT / "outputs/letter_second_override_20261010").resolve(),
    (PROJECT / "outputs/letter_argmax_traj_20261010").resolve(),
    (PROJECT / "outputs/unmask_order_bidir_20261009").resolve(),
    (PROJECT / "outputs/lavida_jacobian_image_minus_text_20261009").resolve(),
)


def refuse_forbidden(path: Path) -> None:
    resolved = path.resolve()
    if "jacobian" in str(resolved):
        raise RuntimeError(f"refusing jacobian path {resolved}")
    for banned in FORBIDDEN:
        if resolved == banned or banned in resolved.parents:
            raise RuntimeError(f"refusing to write into {banned}")


def repair_tail(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, sep, _tail = data.rpartition(b"\n")
    path.write_bytes(head + sep)


def gold_of(row: dict) -> str | None:
    gold = row.get("gold") or row.get("gold_option") or row.get("answer")
    if gold is None:
        return None
    return str(gold).strip()


def decode_token(tokenizer, token_id: int, mask_id: int) -> str:
    if int(token_id) == int(mask_id):
        return ""
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    text = text.replace("\n", " ").strip()
    if not text:
        text = str(tokenizer.convert_ids_to_tokens(int(token_id)))
    return text


def contiguous_positions(mask_1d: torch.Tensor) -> torch.Tensor:
    pos = torch.nonzero(mask_1d, as_tuple=False).flatten()
    if pos.numel() == 0:
        raise RuntimeError("no image tokens in input ids")
    start = int(pos[0].item())
    end = int(pos[-1].item())
    if end - start + 1 != int(pos.numel()):
        raise RuntimeError(
            f"image tokens are not one contiguous span start={start} end={end} n={int(pos.numel())}"
        )
    return pos


def lavida_image_positions(expanded: torch.Tensor, embeds: torch.Tensor, image_token_index: int) -> torch.Tensor:
    if expanded.ndim != 2 or embeds.ndim != 3:
        raise RuntimeError(
            f"expanded ids {tuple(expanded.shape)} embeds {tuple(embeds.shape)} are not a prefix pair"
        )
    if expanded.shape[0] != embeds.shape[0] or expanded.shape[1] != embeds.shape[1]:
        raise RuntimeError(
            f"expanded ids {tuple(expanded.shape)} do not align with embeds {tuple(embeds.shape[:2])}"
        )
    known = expanded.ge(0) | expanded.eq(image_token_index)
    if not bool(known.all()):
        n_bad = int((~known).sum().item())
        raise RuntimeError(
            f"{n_bad} prefix positions are neither text ids nor IMAGE_TOKEN_INDEX={image_token_index}"
        )
    return contiguous_positions(expanded[0].eq(image_token_index))


def mmada_image_positions(prompt_ids: torch.Tensor, eoi_open: int, eoi_close: int, vq_offset: int) -> torch.Tensor:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise RuntimeError(f"prompt ids {tuple(prompt_ids.shape)}")
    row = prompt_ids[0]
    open_at = torch.nonzero(row.eq(eoi_open), as_tuple=False).flatten()
    close_at = torch.nonzero(row.eq(eoi_close), as_tuple=False).flatten()
    if open_at.numel() != 1 or close_at.numel() != 1:
        raise RuntimeError(
            f"expected one image open/close marker, got open={int(open_at.numel())} close={int(close_at.numel())}"
        )
    start = int(open_at[0].item()) + 1
    end = int(close_at[0].item())
    if end <= start:
        raise RuntimeError(f"empty image span between markers {start}..{end}")
    span = row[start:end]
    if not bool(span.ge(vq_offset).all()):
        raise RuntimeError("tokens between image markers are not VQ image codes")
    outside = row.ge(vq_offset).clone()
    outside[start:end] = False
    if bool(outside.any()):
        raise RuntimeError("VQ-range ids sit outside the image markers")
    pos = torch.arange(start, end, device=prompt_ids.device, dtype=torch.long)
    if int(pos.numel()) != end - start:
        raise RuntimeError("image span length mismatch")
    return pos


def span_stats(pos: torch.Tensor) -> tuple[int, int, int]:
    start = int(pos[0].item())
    end = int(pos[-1].item())
    return start, end, int(pos.numel())


def assert_image_cols(pos: torch.Tensor, prefix_len: int) -> None:
    if pos.numel() == 0:
        raise RuntimeError("no image keys to block")
    if int(pos.min().item()) < 0 or int(pos.max().item()) >= int(prefix_len):
        raise RuntimeError(
            f"image keys {int(pos.min())}..{int(pos.max())} outside prefix length {prefix_len}"
        )
    n_question = int(prefix_len) - int(pos.numel())
    if n_question <= 0:
        raise RuntimeError("blocking image keys would hide every question token")


def second_of(h0: torch.Tensor, h_pre: torch.Tensor, head: torch.Tensor, gamma: torch.Tensor, eps: float, token_id: int) -> float:
    h0_64 = h0.detach().to(dtype=torch.float64).cpu()
    direction = h_pre.detach().to(dtype=torch.float64).cpu() - h0_64
    gamma64 = gamma.detach().to(dtype=torch.float64).cpu()
    head_row = head[int(token_id)].detach().to(dtype=torch.float64).cpu()
    quad = delta.quadratic_form(h0_64, direction, gamma64, float(eps), head_row)
    return 0.5 * float(quad)


def bf16_ulp(value: float) -> float:
    if value == 0.0 or value != value:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def assert_logit_reconstruction(hidden: torch.Tensor, gamma: torch.Tensor, eps: float, head: torch.Tensor, logits_row: torch.Tensor) -> None:
    token_id = int(torch.argmax(logits_row).item())
    actual = float(logits_row[token_id].detach().float().cpu())
    delta.check_reconstruction(hidden, gamma, eps, head, token_id, actual)
    h = hidden.detach().to(dtype=torch.float32).cpu()
    g = gamma.detach().to(dtype=torch.float32).cpu()
    scale = torch.dot(h, h) / h.shape[0] + float(eps)
    vec = (g * (h * torch.rsqrt(scale))).to(device=head.device, dtype=torch.float32)
    vocab = int(head.shape[0])
    best_val = None
    best_id = None
    winner = None
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            for start in range(0, vocab, 8192):
                rows = head[start : start + 8192].detach().to(dtype=torch.float32)
                scores = rows @ vec
                local = int(torch.argmax(scores).item())
                val = float(scores[local].item())
                if best_val is None or val > best_val:
                    best_val = val
                    best_id = start + local
                if start <= token_id < start + int(scores.shape[0]):
                    winner = float(scores[token_id - start].item())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    if winner is None or best_id is None or best_val is None:
        raise RuntimeError("reconstruction scan missed the vocabulary")
    gap = abs(winner - actual)
    if gap > 1.0:
        raise RuntimeError(f"position-0 logit reconstruction {winner:.4f} vs actual {actual:.4f}")
    if best_id == token_id:
        return
    other_actual = float(logits_row[best_id].detach().float().cpu())
    other_gap = abs(best_val - other_actual)
    ulp = max(bf16_ulp(actual), bf16_ulp(other_actual))
    if other_gap > 1.0 or max(gap, other_gap) > ulp:
        raise RuntimeError(
            f"reconstruction argmax {best_id} != logit argmax {token_id} "
            f"(recon {winner:.6f} vs best {best_val:.6f}, actual {actual:.6f}, "
            f"other_actual {other_actual:.6f}, gap {gap:.6g}, other_gap {other_gap:.6g}, ulp {ulp:.6g})"
        )
    print(
        f"RECON_ULP token={token_id} other={best_id} gap={gap:.6g} other_gap={other_gap:.6g} ulp={ulp:.6g}",
        flush=True,
    )


def logit_gap(left: torch.Tensor, right: torch.Tensor) -> float:
    gap = float((left.detach().float() - right.detach().float()).abs().max().item())
    if gap == 0.0 or gap != gap:
        raise RuntimeError(f"image-key block did not change logits (max abs diff {gap})")
    return gap


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no} is not json") from exc
            rows.append(row)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate ids in {path}")
    return rows


def require_contrast_complete() -> None:
    for name in ("lavida", "mmada"):
        path = CONTRAST_DIR / f"{name}.jsonl"
        rows = load_jsonl(path)
        if len(rows) < N_TOTAL:
            raise RuntimeError(f"part A {name} has {len(rows)} lines, refusing part B")
        if any("argmax_differs" not in row for row in rows):
            raise RuntimeError(f"part A {name} is not a contrast file")


def question_lavida(row: dict) -> str:
    question = row.get("question") or row["conversations"][0]["value"]
    if "Question:" in question:
        question = question.split("Question:")[-1].split("Answer:")[0].strip()
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    return q


def lavida_prefix(lm, model, processor, tokenizer, row: dict, image_token_index: int):
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    q = question_lavida(row)
    conv = delta.copy_conv(conv_templates["llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + f"Question: {q}\nAnswer:")
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, image_token_index, return_tensors="pt")
    input_ids = ids.unsqueeze(0).to(model.device)
    image = Image.open(row["image"]).convert("RGB")
    tensors = process_images([image], processor, model.config)
    tensors = [t.to(device=model.device, dtype=torch.bfloat16) for t in tensors]
    prepared = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, tensors, ["image"], image_sizes=[image.size], return_inputs=True
    )
    if not isinstance(prepared, tuple) or len(prepared) < 7 or prepared[4] is None or prepared[6] is None:
        raise RuntimeError("prepare_inputs did not return embeddings and expanded ids")
    embeds = prepared[4]
    expanded = prepared[6]
    image_cols = lavida_image_positions(expanded, embeds, image_token_index)
    base.KeyBlock.cols = None
    past = lm(None, input_embeddings=embeds, use_cache=True).attn_key_values
    prefix_len = int(past[0][0].shape[-2])
    if prefix_len != int(embeds.shape[1]):
        raise RuntimeError(f"prefix cache {prefix_len} != embeddings {int(embeds.shape[1])}")
    assert_image_cols(image_cols, prefix_len)
    return past, image_cols


def lavida_canvas_forward(lm, past, canvas: torch.Tensor, image_cols: torch.Tensor, block: bool, capture: bool):
    base.KeyBlock.cols = image_cols if block else None
    try:
        embeds = lm.transformer.wte(canvas)
        if capture:
            delta.take_hidden(0)
        logits = lm(None, input_embeddings=embeds, past_key_values=past, use_cache=False).logits
        hidden = delta.read_hidden(base.GEN) if capture else None
    finally:
        base.KeyBlock.cols = None
    if logits.shape[1] != base.GEN:
        raise RuntimeError(f"canvas logits {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite LaViDa logits")
    return logits, hidden


def mmada_prompt(model, tokenizer, vq, mm, image, question: str, device):
    vq_ids = mm.encode_vq(vq, image, device)
    prompt_ids, _text = mm.build_prompt_ids(tokenizer, question, vq_ids, device)
    image_cols = mmada_image_positions(prompt_ids, mm.EOI_OPEN_ID, mm.EOI_CLOSE_ID, mm.VQ_OFFSET)
    assert_image_cols(image_cols, int(prompt_ids.shape[1]))
    return prompt_ids, image_cols


def mmada_forward(model, prompt_ids, canvas, mask_id: int, image_cols: torch.Tensor, block: bool, capture: bool):
    prompt_len = int(prompt_ids.shape[1])
    x = torch.full(
        (1, prompt_len + canvas.shape[1]),
        mask_id,
        dtype=torch.long,
        device=canvas.device,
    )
    x[:, :prompt_len] = prompt_ids
    x[:, prompt_len:] = canvas
    if capture:
        delta.take_hidden(prompt_len)
    if not block:
        logits = model(x).logits
    else:
        seq = int(x.shape[1])
        allow = torch.ones(1, 1, seq, seq, dtype=torch.bool, device=x.device)
        cols = image_cols.to(device=x.device)
        if int(cols.max().item()) >= prompt_len:
            raise RuntimeError("image columns overlap the canvas")
        allow[0, 0, :, cols] = False
        blocked = ~allow[0, 0, 0]
        if int(blocked.sum().item()) != int(cols.numel()):
            raise RuntimeError("blocked column count does not match image keys")
        if bool(blocked[prompt_len:].any()):
            raise RuntimeError("image-key block hid a canvas answer key")
        logits = model(x, attention_bias=allow).logits
    hidden = delta.read_hidden(prompt_len + base.GEN) if capture else None
    if logits.shape[1] != prompt_len + base.GEN:
        raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
    answer = logits[:, prompt_len:, :]
    if not torch.isfinite(answer).all():
        raise RuntimeError("non-finite MMaDA logits")
    return answer, hidden


def contrast_record(row, backend, tokenizer, mask_id, visible_logits, blocked_logits, image_cols) -> dict:
    visible_id = int(torch.argmax(visible_logits[0, 0]).item())
    blocked_id = int(torch.argmax(blocked_logits[0, 0]).item())
    visible_text = decode_token(tokenizer, visible_id, mask_id)
    blocked_text = decode_token(tokenizer, blocked_id, mask_id)
    gold = gold_of(row)
    visible_letter = delta.leading_letter(visible_text)
    blocked_letter = delta.leading_letter(blocked_text)
    start, end, n_image = span_stats(image_cols)
    return {
        "id": row["id"],
        "gold": gold,
        "backend": backend,
        "visible_id": visible_id,
        "blocked_id": blocked_id,
        "visible_text": visible_text,
        "blocked_text": blocked_text,
        "argmax_differs": visible_id != blocked_id,
        "visible_letter": visible_letter,
        "blocked_letter": blocked_letter,
        "visible_correct": visible_letter is not None and visible_letter == gold,
        "blocked_correct": blocked_letter is not None and blocked_letter == gold,
        "n_image_keys": n_image,
        "image_key_start": start,
        "image_key_end": end,
        "logit_max_abs_diff": logit_gap(visible_logits, blocked_logits),
        "contrast_step": 0,
        "hidden_edited": False,
        "decode_rewritten": False,
    }


def prepare_head(module, cuda: int):
    norm, head, gamma, eps = delta.head_of(module)
    if type(norm).__name__ != "RMSLayerNorm":
        raise RuntimeError(f"expected RMSLayerNorm, got {type(norm).__name__}")
    if abs(float(eps) - 1e-5) > 1e-12:
        raise RuntimeError(f"expected RMSNorm eps 1e-5, got {eps}")
    if int(norm.weight.device.index) != int(cuda):
        raise RuntimeError(f"RMSNorm weight on {norm.weight.device}, expected cuda:{cuda}")
    if head.dim() != 2 or head.shape[0] < head.shape[1]:
        raise RuntimeError(f"unexpected ff_out weight shape {tuple(head.shape)}")
    return head, gamma, float(eps)


def run_fd(h0, gamma, eps, head, token_id: int, fd_state: dict) -> None:
    if fd_state["done"]:
        return
    delta.finite_difference_check(h0, gamma, eps, head[int(token_id)])
    if delta.Gate.rel is None or delta.Gate.rel >= 1e-3:
        raise RuntimeError(f"finite-difference check failed rel={delta.Gate.rel}")
    print(
        f"FD_CHECK_PASS analytic={delta.Gate.analytic:.10g} fd={delta.Gate.fd:.10g} rel={delta.Gate.rel:.10g}",
        flush=True,
    )
    fd_state["done"] = True


def decode_blocked(x, mask_id, tokenizer, head, gamma, eps, forward, fd_state: dict) -> dict:
    unmask_step = [-1] * base.GEN
    h0 = None
    second = None
    wrote = None
    for _step in range(base.GEN):
        logits, hidden = forward(x)
        if h0 is None:
            h0 = hidden
            logits0 = logits[0, 0].detach().float().cpu()
            step0_id = int(torch.argmax(logits0).item())
            assert_logit_reconstruction(h0, gamma, eps, head, logits0)
            run_fd(h0, gamma, eps, head, step0_id, fd_state)
        pos = base.unmask_one(x, logits, mask_id)
        unmask_step[pos] = _step
        if pos == 0 and second is None:
            wrote = int(x[0, 0].item())
            second = second_of(h0, hidden, head, gamma, eps, wrote)
    if h0 is None or second is None or wrote is None:
        raise RuntimeError("letter position was never selected")
    if sorted(unmask_step) != list(range(base.GEN)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    if int(x[0, 0].item()) != int(wrote):
        raise RuntimeError(f"position 0 holds {int(x[0, 0].item())} after committing {wrote}")
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return {
        "written_id": int(wrote),
        "written_token": decode_token(tokenizer, wrote, mask_id),
        "second": float(second),
        "abs_second": abs(float(second)),
        "finished_letter": delta.leading_letter(pred),
        "letter_unmask_step": int(unmask_step[0]),
        "pred": pred,
    }


def delta_record(row, backend, decoded, image_cols) -> dict:
    gold = gold_of(row)
    finished = decoded["finished_letter"]
    start, end, n_image = span_stats(image_cols)
    return {
        "id": row["id"],
        "gold": gold,
        "backend": backend,
        "written_id": decoded["written_id"],
        "written_token": decoded["written_token"],
        "second": decoded["second"],
        "abs_second": decoded["abs_second"],
        "finished_letter": finished,
        "finished_correct": finished is not None and finished == gold,
        "letter_unmask_step": decoded["letter_unmask_step"],
        "n_image_keys": n_image,
        "image_key_start": start,
        "image_key_end": end,
        "curvature_rewrite": False,
    }


def load_manifest(args):
    if args.backend == "lavida":
        rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    else:
        sys.path.insert(0, str(PROJECT / "experiments"))
        import sciqa_standard_eval as sq

        rows = sq.load_manifest(Path(args.manifest), Path(args.data_dir))
    if len(rows) != N_TOTAL:
        raise RuntimeError(f"expected {N_TOTAL} items, got {len(rows)}")
    return rows


def ids_complete(rows: list[dict], manifest: list[dict]) -> bool:
    return {row["id"] for row in rows} == {row["id"] for row in manifest}


def validate_existing(rows: list[dict], part: str) -> None:
    key = "argmax_differs" if part == "contrast" else "abs_second"
    for row in rows:
        if key not in row or row.get("backend") is None:
            raise RuntimeError(f"existing row {row.get('id')} is not a {part} record")


def loop(args, manifest, fn, tally) -> None:
    done = {row["id"] for row in load_jsonl(args.output)}
    finished = len(done)
    t0 = time.time()
    print(f"resume n_done={finished}/{len(manifest)}", flush=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in manifest:
            if row["id"] in done:
                continue
            with torch.no_grad():
                record = fn(row)
            if record["id"] != row["id"]:
                raise RuntimeError(f"record id {record['id']} != {row['id']}")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(record["id"])
            finished += 1
            payload = tally(record, finished, len(manifest), t0)
            base.write_progress(args.progress, payload)
            print(
                f"PROGRESS n_done={finished}/{len(manifest)} id={record['id']} "
                + " ".join(f"{key}={payload[key]}" for key in payload if key in (
                    "argmax_differs", "differ_count", "visible_correct_count",
                    "blocked_correct_count", "abs_second", "finished_correct_count",
                    "running_abs_second_median",
                )),
                flush=True,
            )
    if finished != len(manifest):
        raise RuntimeError(f"finished {finished} of {len(manifest)}")


def contrast_tally_factory(state):
    def tally(record, finished, total, t0):
        state["differ"] += int(bool(record["argmax_differs"]))
        state["visible"] += int(bool(record["visible_correct"]))
        state["blocked"] += int(bool(record["blocked_correct"]))
        return {
            "backend": record["backend"],
            "part": "contrast",
            "n_done": finished,
            "n_total": total,
            "id": record["id"],
            "argmax_differs": record["argmax_differs"],
            "differ_count": state["differ"],
            "visible_correct_count": state["visible"],
            "blocked_correct_count": state["blocked"],
            "n_image_keys": record["n_image_keys"],
            "image_key_start": record["image_key_start"],
            "image_key_end": record["image_key_end"],
            "logit_max_abs_diff": record["logit_max_abs_diff"],
            "elapsed_sec": round(time.time() - t0, 1),
        }
    return tally


def delta_tally_factory(state):
    def tally(record, finished, total, t0):
        state["correct"] += int(bool(record["finished_correct"]))
        state["abs"].append(float(record["abs_second"]))
        return {
            "backend": record["backend"],
            "part": "delta2",
            "n_done": finished,
            "n_total": total,
            "id": record["id"],
            "abs_second": record["abs_second"],
            "finished_correct_count": state["correct"],
            "running_abs_second_median": statistics.median(state["abs"]),
            "finished_letter": record["finished_letter"],
            "letter_unmask_step": record["letter_unmask_step"],
            "elapsed_sec": round(time.time() - t0, 1),
        }
    return tally


def seed_contrast_state(rows: list[dict]) -> dict:
    return {
        "differ": sum(int(bool(row["argmax_differs"])) for row in rows),
        "visible": sum(int(bool(row["visible_correct"])) for row in rows),
        "blocked": sum(int(bool(row["blocked_correct"])) for row in rows),
    }


def seed_delta_state(rows: list[dict]) -> dict:
    return {
        "correct": sum(int(bool(row["finished_correct"])) for row in rows),
        "abs": [float(row["abs_second"]) for row in rows],
    }


def run_lavida(args, manifest) -> None:
    sys.path.insert(0, str(PROJECT / "repos" / "LaViDa"))
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.model.builder import load_pretrained_model

    if IMAGE_TOKEN_INDEX >= 0:
        raise RuntimeError(f"IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX} is not an image sentinel")
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
    n_blocks = base.install_key_block(lm)
    if n_blocks <= 0:
        raise RuntimeError("LaViDa key block hooked no attention blocks")
    head = gamma = eps = None
    if args.part == "delta2":
        head, gamma, eps = prepare_head(lm, args.cuda)
    print(
        f"lavida core={type(lm).__name__} blocks={n_blocks} mask={base.MASK_ID_LAVIDA} "
        f"cuda={args.cuda} part={args.part}",
        flush=True,
    )
    existing = load_jsonl(args.output)
    fd_state = {"done": False}

    def one(row):
        past, image_cols = lavida_prefix(lm, model, processor, tokenizer, row, IMAGE_TOKEN_INDEX)
        x = torch.full((1, base.GEN), base.MASK_ID_LAVIDA, dtype=torch.long, device=model.device)
        if args.part == "contrast":
            visible, _hidden = lavida_canvas_forward(lm, past, x, image_cols, block=False, capture=False)
            blocked, _hidden = lavida_canvas_forward(lm, past, x, image_cols, block=True, capture=False)
            if not bool(x.eq(base.MASK_ID_LAVIDA).all()):
                raise RuntimeError("contrast forward rewrote the canvas")
            return contrast_record(row, "lavida", tokenizer, base.MASK_ID_LAVIDA, visible, blocked, image_cols)

        def forward(canvas):
            return lavida_canvas_forward(lm, past, canvas, image_cols, block=True, capture=True)

        decoded = decode_blocked(x, base.MASK_ID_LAVIDA, tokenizer, head, gamma, eps, forward, fd_state)
        return delta_record(row, "lavida", decoded, image_cols)

    if args.part == "contrast":
        loop(args, manifest, one, contrast_tally_factory(seed_contrast_state(existing)))
    else:
        loop(args, manifest, one, delta_tally_factory(seed_delta_state(existing)))


def run_mmada(args, manifest) -> None:
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
        raise RuntimeError("MMadaModelLM unexpectedly has .transformer")
    core = model.model
    if not hasattr(core, "transformer"):
        raise RuntimeError(f"MMaDA language core {type(core).__name__} has no transformer")
    vq = mm.load_vq(Ns, device)
    model.eval()
    head = gamma = eps = None
    if args.part == "delta2":
        head, gamma, eps = prepare_head(core, args.cuda)
    print(
        f"mmada outer={type(model).__name__} core={type(core).__name__} "
        f"mask={mm.MASK_ID} cuda={args.cuda} part={args.part}",
        flush=True,
    )
    loader = None
    for name in dir(sq):
        obj = getattr(sq, name)
        if isinstance(obj, type) and hasattr(obj, "load_rgb"):
            loader = obj(Path(args.data_dir), manifest)
            break
    if loader is None:
        raise RuntimeError("ScienceQA image loader was not found")
    existing = load_jsonl(args.output)
    fd_state = {"done": False}

    def one(row):
        image = loader.load_rgb(row)
        question = sq.build_task_prompt(row)
        prompt_ids, image_cols = mmada_prompt(model, tokenizer, vq, mm, image, question, device)
        x = torch.full((1, base.GEN), mm.MASK_ID, dtype=torch.long, device=device)
        if args.part == "contrast":
            visible, _hidden = mmada_forward(model, prompt_ids, x, mm.MASK_ID, image_cols, block=False, capture=False)
            blocked, _hidden = mmada_forward(model, prompt_ids, x, mm.MASK_ID, image_cols, block=True, capture=False)
            if not bool(x.eq(mm.MASK_ID).all()):
                raise RuntimeError("contrast forward rewrote the canvas")
            return contrast_record(row, "mmada", tokenizer, mm.MASK_ID, visible, blocked, image_cols)

        def forward(canvas):
            return mmada_forward(model, prompt_ids, canvas, mm.MASK_ID, image_cols, block=True, capture=True)

        decoded = decode_blocked(x, mm.MASK_ID, tokenizer, head, gamma, eps, forward, fd_state)
        return delta_record(row, "mmada", decoded, image_cols)

    if args.part == "contrast":
        loop(args, manifest, one, contrast_tally_factory(seed_contrast_state(existing)))
    else:
        loop(args, manifest, one, delta_tally_factory(seed_delta_state(existing)))


def summarize_contrast(path: Path) -> dict:
    rows = load_jsonl(path)
    differ = visible = blocked = 0
    for row in rows:
        letter_visible = row["visible_letter"]
        letter_blocked = row["blocked_letter"]
        gold = row["gold"]
        visible_ok = letter_visible is not None and letter_visible == gold
        blocked_ok = letter_blocked is not None and letter_blocked == gold
        if visible_ok != bool(row["visible_correct"]):
            raise RuntimeError(f"{row['id']} visible_correct does not match the letter")
        if blocked_ok != bool(row["blocked_correct"]):
            raise RuntimeError(f"{row['id']} blocked_correct does not match the letter")
        if (row["visible_id"] != row["blocked_id"]) != bool(row["argmax_differs"]):
            raise RuntimeError(f"{row['id']} argmax_differs flag mismatch")
        differ += int(bool(row["argmax_differs"]))
        visible += int(bool(row["visible_correct"]))
        blocked += int(bool(row["blocked_correct"]))
    starts = sorted({(row["image_key_start"], row["image_key_end"], row["n_image_keys"]) for row in rows})
    return {
        "n": len(rows),
        "differ": differ,
        "visible_correct": visible,
        "blocked_correct": blocked,
        "spans": starts[:8],
        "n_spans": len(starts),
    }


def summarize_delta(path: Path) -> dict:
    rows = load_jsonl(path)
    correct = 0
    values = []
    for row in rows:
        finished_ok = row["finished_letter"] is not None and row["finished_letter"] == row["gold"]
        if finished_ok != bool(row["finished_correct"]):
            raise RuntimeError(f"{row['id']} finished_correct does not match the letter")
        if abs(float(row["second"])) != float(row["abs_second"]):
            raise RuntimeError(f"{row['id']} abs_second mismatch")
        if row.get("curvature_rewrite"):
            raise RuntimeError(f"{row['id']} applied a curvature rewrite")
        correct += int(bool(row["finished_correct"]))
        values.append(float(row["abs_second"]))
    median = None if not values else statistics.median(values)
    return {"n": len(rows), "finished_correct": correct, "abs_second_median": median}


def summarize_old_delta() -> dict:
    out = {}
    root = PROJECT / "outputs" / "letter_hidden_delta2_20261009"
    for name in ("lavida", "mmada"):
        path = root / f"{name}.jsonl"
        if not path.exists():
            out[name] = None
            continue
        values = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    values.append(abs(float(json.loads(line)["second_final"])))
        out[name] = None if not values else statistics.median(values)
    return out


def print_summary() -> None:
    old = summarize_old_delta()
    print("SUMMARY", flush=True)
    for name in ("lavida", "mmada"):
        contrast_path = CONTRAST_DIR / f"{name}.jsonl"
        delta_path = PROJECT / "outputs" / "image_key_blocked_delta2_20261010" / f"{name}.jsonl"
        if contrast_path.exists():
            contrast = summarize_contrast(contrast_path)
            print(
                f"PART_A {name} n={contrast['n']} differ={contrast['differ']} "
                f"visible_correct={contrast['visible_correct']} blocked_correct={contrast['blocked_correct']} "
                f"n_spans={contrast['n_spans']} spans={contrast['spans']}",
                flush=True,
            )
        if delta_path.exists():
            measured = summarize_delta(delta_path)
            print(
                f"PART_B {name} n={measured['n']} abs_second_median={measured['abs_second_median']} "
                f"finished_correct={measured['finished_correct']} stock={STOCK_CORRECT[name]} "
                f"old_abs_second_median_file={old[name]} old_abs_second_median_stated={OLD_ABS_SECOND_MEDIAN[name]}",
                flush=True,
            )


def self_check() -> None:
    image_index = -200
    expanded = torch.arange(30).view(1, 30)
    expanded[0, 5:12] = image_index
    embeds = torch.zeros(1, 30, 4)
    pos = lavida_image_positions(expanded, embeds, image_index)
    if [int(v) for v in pos.tolist()] != list(range(5, 12)):
        raise RuntimeError(f"shifted image span detected as {pos.tolist()}")
    example = torch.arange(400).view(1, 400)
    example[0, 47:393] = image_index
    example_pos = lavida_image_positions(example, torch.zeros(1, 400, 2), image_index)
    if int(example_pos[0]) != 47 or int(example_pos[-1]) != 392:
        raise RuntimeError("detector failed on a 47..392 example")
    if [int(v) for v in pos.tolist()] == [int(v) for v in example_pos.tolist()]:
        raise RuntimeError("detector returned the example span for a shifted item")
    holed = expanded.clone()
    holed[0, 8] = 3
    try:
        lavida_image_positions(holed, embeds, image_index)
    except RuntimeError as exc:
        if "not one contiguous span" not in str(exc):
            raise
    else:
        raise RuntimeError("a hole in the image span was accepted")

    eoi_open, eoi_close, vq_offset, soi = 126084, 126085, 126349, 126089
    vq = torch.arange(4) + vq_offset
    text = torch.tensor([11, 12, 13, mm_mask()])
    prompt = torch.cat([torch.tensor([soi, eoi_open]), vq, torch.tensor([eoi_close]), text]).view(1, -1)
    found = mmada_image_positions(prompt, eoi_open, eoi_close, vq_offset)
    if [int(v) for v in found.tolist()] != [2, 3, 4, 5]:
        raise RuntimeError(f"MMaDA image positions {found.tolist()}")
    leaked = prompt.clone()
    leaked[0, -1] = vq_offset
    try:
        mmada_image_positions(leaked, eoi_open, eoi_close, vq_offset)
    except RuntimeError as exc:
        if "outside the image markers" not in str(exc):
            raise
    else:
        raise RuntimeError("a VQ id in the question was accepted as inside the image")

    generator = torch.Generator()
    generator.manual_seed(20261010)
    hidden = torch.randn(64, generator=generator, dtype=torch.float64)
    gamma = torch.rand(64, generator=generator, dtype=torch.float64) + 0.1
    head = torch.randn(6, 64, generator=generator, dtype=torch.float64)
    delta.finite_difference_check(hidden, gamma, 1e-5, head[2])
    if delta.Gate.rel is None or delta.Gate.rel >= 1e-3:
        raise RuntimeError(f"self-check finite difference failed rel={delta.Gate.rel}")
    direction = torch.randn(64, generator=generator, dtype=torch.float64)
    got = second_of(hidden, hidden + direction, head, gamma, 1e-5, 2)
    quad = 0.5 * float(delta.quadratic_form(hidden, direction, gamma, 1e-5, head[2]))
    if abs(got - quad) > 1e-12:
        raise RuntimeError(f"second_of {got} != half quadratic {quad}")
    print(
        f"SELF_CHECK_PASS fd_rel={delta.Gate.rel:.10g} second={got:.10g}",
        flush=True,
    )


def mm_mask() -> int:
    return 126336


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("lavida", "mmada"))
    parser.add_argument("--part", choices=("contrast", "delta2"))
    parser.add_argument("--cuda", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    parser.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.summarize_only:
        print_summary()
        return
    if args.backend is None or args.part is None or args.cuda is None or args.output is None or args.progress is None:
        raise RuntimeError("backend, part, cuda, output, and progress are required")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
    if base.GEN != 16 or base.MASK_ID_LAVIDA != 126336:
        raise RuntimeError(f"unexpected gen/mask {base.GEN} {base.MASK_ID_LAVIDA}")
    if args.backend == "lavida":
        args.model_path = args.model_path or str(PROJECT / "checkpoints/lavida-llada-v1.0-instruct")
        args.manifest = args.manifest or str(PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json")
    else:
        args.model_path = args.model_path or str(PROJECT / "checkpoints/MMaDA-8B-Base")
        args.manifest = args.manifest or str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json")
        args.data_dir = args.data_dir or str(PROJECT / "datasets/sciqa_img_standard_v1")
    refuse_forbidden(args.output)
    refuse_forbidden(args.progress)
    if args.part == "delta2":
        require_contrast_complete()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    repair_tail(args.output)
    manifest = load_manifest(args)
    existing = load_jsonl(args.output)
    validate_existing(existing, args.part)
    manifest_ids = {row["id"] for row in manifest}
    extra = {row["id"] for row in existing} - manifest_ids
    if extra:
        raise RuntimeError(f"output has ids outside the manifest, for example {sorted(extra)[:3]}")
    if ids_complete(existing, manifest):
        print(f"ALREADY_DONE n={len(existing)} part={args.part} backend={args.backend}", flush=True)
        return
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_handle = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {args.output}") from exc
    torch.cuda.set_device(args.cuda)
    print(
        f"start backend={args.backend} part={args.part} cuda={args.cuda} gen={base.GEN} "
        f"temperature=0 remasking=low_confidence prefix_lm=1 output={args.output}",
        flush=True,
    )
    try:
        if args.backend == "lavida":
            run_lavida(args, manifest)
        else:
            run_mmada(args, manifest)
        if args.part == "contrast":
            stats = summarize_contrast(args.output)
            print(
                f"DONE_PART_A backend={args.backend} n={stats['n']} differ={stats['differ']} "
                f"visible_correct={stats['visible_correct']} blocked_correct={stats['blocked_correct']}",
                flush=True,
            )
        else:
            stats = summarize_delta(args.output)
            print(
                f"DONE_PART_B backend={args.backend} n={stats['n']} "
                f"abs_second_median={stats['abs_second_median']} "
                f"finished_correct={stats['finished_correct']} stock={STOCK_CORRECT[args.backend]}",
                flush=True,
            )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
