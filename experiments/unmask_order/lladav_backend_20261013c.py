#!/usr/bin/env python3
"""LLaDA-V backend for the gen-16 low-confidence letter decodes (new file; existing backends untouched).

LLaDA-V (checkpoints/LLaDA-V, class LlavaLLaDAModelLM from repos/LLaDA-V/train, HF-Llama-style LLaDA core:
model.model.embed_tokens / model.model.norm (LLaDARMSNorm) / model.lm_head). Unlike LaViDa there is no prefix
KV cache ("The kvcache is not supported for MDM"), so every decode step is one full forward over
[prefix embeddings (text + SigLIP2 anyres_max_4 image tokens) ; canvas token embeddings].

Interface used by calib_decode_multi_20261013c.py:
    load(model_path, cuda)                -> ctx (dict)
    prefix(ctx, text, image)              -> (prefix_embeds, prefix_len)   (prompt via conv 'llava_llada')
    make_forward(ctx, prefix_embeds)      -> forward(canvas) -> logits over the canvas only (1, GEN, V)
    letter_ids(ctx, letters)              -> list[int]
    prepare_readout(ctx)                  -> (head weight, gamma64, eps)   float64 letter readout at ln_f input
    arm(index) / collect(seq, batch)      -> captured pre-norm hidden state at one position
    MASK_ID = 126336

The repo's llava package for LLaDA-V (repos/LLaDA-V/train) shadows LaViDa's llava package, so a process that
loads this backend must not also load LaViDa.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
LLADAV_REPO = PROJECT / "repos" / "LLaDA-V" / "train"
MASK_ID = 126336
EOS_ID = 126081
SIGLIP2_SNAPSHOTS = PROJECT / "hf-cache/hub/models--google--siglip2-so400m-patch14-384/snapshots"


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
    Capture.h = hidden[:, Capture.index].detach().to(dtype=torch.float32).cpu()


def arm(index: int) -> None:
    Capture.armed = True
    Capture.index = int(index)
    Capture.h = None
    Capture.seq = None


def collect(expected_seq: int, expected_batch: int) -> torch.Tensor:
    Capture.armed = False
    if Capture.h is None:
        raise RuntimeError("norm input was not captured")
    if Capture.seq != expected_seq:
        raise RuntimeError(f"norm sequence {Capture.seq} != expected {expected_seq}")
    if int(Capture.h.shape[0]) != expected_batch:
        raise RuntimeError(f"captured batch {tuple(Capture.h.shape)} != {expected_batch}")
    return Capture.h


def _siglip2_local() -> str:
    if SIGLIP2_SNAPSHOTS.exists():
        hits = sorted(p for p in SIGLIP2_SNAPSHOTS.iterdir() if (p / "config.json").exists())
        if hits:
            return str(hits[-1])
    return ""


def load(model_path: str, cuda: int) -> dict:
    if "llava" in sys.modules:
        mod = sys.modules["llava"]
        if "LLaDA-V" not in str(getattr(mod, "__file__", "")):
            raise RuntimeError("another llava package is already imported; LLaDA-V needs its own process")
    sys.path.insert(0, str(LLADAV_REPO))
    from llava.model.builder import load_pretrained_model

    vision = _siglip2_local()
    kwargs = dict(device_map=f"cuda:{cuda}", torch_dtype="bfloat16", attn_implementation="sdpa")
    if vision:
        kwargs["overwrite_config"] = {"mm_vision_tower": vision}
    print(f"loading lladav model={model_path} vision={vision or '(checkpoint default)'}", flush=True)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, "llava_llada", **kwargs)
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.to(torch.bfloat16)
    if type(model).__name__ != "LlavaLLaDAModelLM":
        raise RuntimeError(f"expected LlavaLLaDAModelLM, got {type(model).__name__}")
    core = model.get_model()
    if not hasattr(core, "embed_tokens") or not hasattr(core, "norm"):
        raise RuntimeError(f"LLaDA-V core {type(core).__name__} lacks embed_tokens/norm")
    mask_tok = tokenizer.convert_ids_to_tokens(MASK_ID)
    print(f"lladav loaded class={type(model).__name__} core={type(core).__name__} mask_id={MASK_ID} "
          f"mask_token={mask_tok!r} cuda={cuda}", flush=True)
    return {"tokenizer": tokenizer, "model": model, "processor": processor, "core": core, "cuda": int(cuda),
            "device": torch.device(f"cuda:{cuda}"), "vision": vision, "conv": "llava_llada"}


def letter_ids(ctx: dict, letters: str) -> list[int]:
    tok = ctx["tokenizer"]
    out = []
    for letter in letters:
        ids = tok.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"letter {letter!r} tokenizes to {ids}")
        out.append(int(ids[0]))
    return out


def user_turn(text: str) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN

    return DEFAULT_IMAGE_TOKEN + "\n" + text


def prompt_string(ctx: dict, text: str) -> str:
    from llava.conversation import conv_templates

    conv = copy.deepcopy(conv_templates[ctx["conv"]])
    conv.tokenizer = ctx["tokenizer"]  # avoid the offline AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-V') path
    conv.append_message(conv.roles[0], user_turn(text))
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prefix(ctx: dict, text: str, image) -> tuple[torch.Tensor, int]:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token

    model, processor, tokenizer = ctx["model"], ctx["processor"], ctx["tokenizer"]
    ids = tokenizer_image_token(prompt_string(ctx, text), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = ids.unsqueeze(0).to(model.device)
    tensors = process_images([image], processor, model.config)
    tensors = [t.to(device=model.device, dtype=torch.bfloat16) for t in tensors]
    prepared = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, tensors, ["image"], image_sizes=[image.size]
    )
    embeds = prepared[4]
    if embeds is None or embeds.dim() != 3 or embeds.shape[0] != 1:
        raise RuntimeError(f"bad prefix embeddings {None if embeds is None else tuple(embeds.shape)}")
    return embeds, int(embeds.shape[1])


def logits_from_embeds(ctx: dict, embeds: torch.Tensor) -> torch.Tensor:
    model = ctx["model"]
    out = model.model(inputs_embeds=embeds, use_cache=False)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return model.lm_head(hidden)


def make_forward(ctx: dict, prefix_embeds: torch.Tensor):
    core = ctx["core"]
    plen = int(prefix_embeds.shape[1])

    def forward(canvas: torch.Tensor) -> torch.Tensor:
        emb = core.embed_tokens(canvas).to(prefix_embeds.dtype)
        x = torch.cat([prefix_embeds, emb], dim=1)
        logits = logits_from_embeds(ctx, x)
        if logits.shape[1] != plen + canvas.shape[1]:
            raise RuntimeError(f"lladav logits {tuple(logits.shape)} vs prefix {plen} + canvas {canvas.shape[1]}")
        return logits[:, plen:, :]

    return forward


def prepare_readout(ctx: dict):
    core, model = ctx["core"], ctx["model"]
    norm = core.norm
    if "RMSNorm" not in type(norm).__name__:
        raise RuntimeError(f"expected an RMSNorm final norm, got {type(norm).__name__}")
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm, "eps", None)
    if eps is None:
        raise RuntimeError("final norm has no eps attribute")
    eps = float(eps)
    if int(norm.weight.device.index) != ctx["cuda"]:
        raise RuntimeError(f"norm on {norm.weight.device}, expected cuda:{ctx['cuda']}")
    head = model.lm_head.weight
    if head.dim() != 2 or head.shape[0] < head.shape[1]:
        raise RuntimeError(f"unexpected lm_head weight {tuple(head.shape)}")
    norm.register_forward_pre_hook(_pre_hook)
    gamma64 = norm.weight.detach().to(dtype=torch.float64).cpu()
    return head, gamma64, eps


def rms_norm(h: torch.Tensor, eps: float) -> torch.Tensor:
    return h * torch.rsqrt(torch.dot(h, h) / h.shape[0] + eps)
