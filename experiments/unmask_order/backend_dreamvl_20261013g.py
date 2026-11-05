#!/usr/bin/env python3
"""Dream-VL-7B backend for the gen-16 low-confidence letter decodes (new file 2026-10-13g; nothing existing is edited).

Model   checkpoints/Dream-VL-7B  == Dream-org/Dream-VL-7B revision 95b9018899ebcfd3824afec86bac295f6c5c1c08
        (arXiv 2512.22615; Dream-7B = Qwen2.5-7B-initialised masked-diffusion LM, Qwen2-VL ViT (depth 32, 1280-d,
        patch 14, 2x2 spatial merge) + 2-layer projector; class DreamVLModel via trust_remote_code).
Prompt  official chat template (processor.apply_chat_template, system "You are a helpful assistant."), user turn =
        [image, text]; the text is the same lmms-eval sqa template used for the mmada / lladav backends.
Image   official DreamVLProcessor (Qwen2-VL smart_resize) with the pixel budget of the official lmms-eval evaluator
        (vl/eval/lmms_eval/models/dream_vl.py: min_pixels 200704, max_pixels 12845056), i.e. ~ (h/28)*(w/28) tokens.
Decode  every step is one full bidirectional forward over [prompt embeddings (text + projected image features) ;
        canvas token embeddings] with the official 3-D M-RoPE position ids (model.get_rope_index on prompt + canvas)
        and no KV cache.  Dream is AR-initialised: the official sampler (generation_utils._sample) reads the
        prediction for position k from output k-1 (`logits = cat([logits[:, :1], logits[:, :-1]], 1)`).  We apply the
        same shift, so canvas position j is read from hidden state prompt_len-1+j, and the content-free readout
        (prior_forward) captures the final-norm input at prompt_len-1.
Rule    OUR remasking (run_unmask_order.unmask_one: one token per step, highest argmax probability among masked
        positions, temperature 0), gen 16 / 16 steps -- NOT Dream's default maskgit_plus schedule (which transfers
        int(n_mask*(1-s/t)) tokens per step with a linear t-schedule, 0 tokens at step 0 for 16 steps).  The official
        sampler is exposed as official_generate() for the agreement check only.

Interface (as calib_decode_multi_20261013c.Lladav / _20261013e.LavidaE):
    Dreamvl(args)            args.model_path, args.cuda
    .name .tokenizer .model .mask_id .cuda
    .letter_ids()            ids of 'A'..'E' (no leading space: the answer directly follows "<|im_start|>assistant\n")
    .decode(text, image)     -> dict from calib_decode_multi_20261013e.decode_canvas (+ prefix_len, n_image_tokens)
    .step0(text, image)      -> (position-0 logits float cpu, prefix_len)
    .prepare_readout()       -> (lm_head weight, gamma64, eps)
    .prior_forward(text, image) -> (pos-0 logits, captured final-norm input, prefix_len, prompt string)
    .official_generate(text, image, **gen) -> dict (official diffusion_generate, for the agreement check)
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import run_unmask_order as base  # noqa: E402

GEN = base.GEN
MODEL_DIR = PROJECT / "checkpoints" / "Dream-VL-7B"
HF_ID = "Dream-org/Dream-VL-7B"
HF_REVISION = "95b9018899ebcfd3824afec86bac295f6c5c1c08"
MASK_ID = 151666          # <|mask|>
IMAGE_TOKEN_ID = 151655   # <|image_pad|>
IM_END_ID = 151645        # <|im_end|>  (chat turn end; config.eos_token_id is <|endoftext|> 151643)
EOS_ID = 151643
# official lmms-eval evaluator pixel budget (Dream-VLX/vl/eval/lmms_eval/models/dream_vl.py)
MIN_PIXELS = 200704       # 256 * 28 * 28
MAX_PIXELS = 12845056     # 16384 * 28 * 28
OFFICIAL_README = dict(max_new_tokens=128, steps=128, temperature=0.1, top_p=1.0, alg="maskgit_plus", alg_temp=0)
OFFICIAL_LMMS_MCQ = dict(max_new_tokens=16, steps=16, temperature=0.0, top_p=1.0, alg="maskgit_plus", alg_temp=0)


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
        raise RuntimeError("final-norm input was not captured")
    if Capture.seq != expected_seq:
        raise RuntimeError(f"captured sequence {Capture.seq} != expected {expected_seq}")
    if int(Capture.h.shape[0]) != expected_batch:
        raise RuntimeError(f"captured batch {tuple(Capture.h.shape)} != {expected_batch}")
    return Capture.h


def cut_official(text: str) -> str:
    for stop in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(stop)[0]
    return text.strip()


# DreamTokenizer.decode(skip_special_tokens=True) keeps <|im_end|> (it is an added token but not in the tokenizer's
# special set), so the canvas text comes out as 'B<|im_end|>' and leading_letter() would return None.  Same fix as the
# earlier Dream-VL probes (experiments/evdlc_dream_vl.strip_special_text).
DREAM_SPECIAL_STRINGS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|mask|>", "<|image_pad|>")


def strip_special_text(text: str) -> str:
    cleaned = str(text or "")
    for piece in DREAM_SPECIAL_STRINGS:
        cleaned = cleaned.replace(piece, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


class Dreamvl:
    name = "dreamvl"

    def __init__(self, args):
        from transformers import AutoModel, AutoProcessor

        path = str(getattr(args, "model_path", "") or MODEL_DIR)
        self.cuda = int(args.cuda)
        # DREAMVL_DEVICE=cpu is a debug knob for CPU-only code-path tests (never set by the queue / smoke scripts)
        self.device = torch.device(os.environ.get("DREAMVL_DEVICE") or f"cuda:{self.cuda}")
        t0 = time.time()
        print(f"loading dreamvl model={path} attn=sdpa dtype=bf16 cuda={self.cuda}", flush=True)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                               attn_implementation="sdpa").to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True,
                                                       min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        self.tokenizer = self.processor.tokenizer
        ip = self.processor.image_processor
        if int(getattr(ip, "min_pixels", -1)) != MIN_PIXELS or int(getattr(ip, "max_pixels", -1)) != MAX_PIXELS:
            raise RuntimeError(f"image processor pixel budget {getattr(ip, 'min_pixels', None)}/"
                               f"{getattr(ip, 'max_pixels', None)} != {MIN_PIXELS}/{MAX_PIXELS}")
        cfg = self.model.config
        if type(self.model).__name__ != "DreamVLModel":
            raise RuntimeError(f"expected DreamVLModel, got {type(self.model).__name__}")
        if int(cfg.mask_token_id) != MASK_ID or int(cfg.image_token_id) != IMAGE_TOKEN_ID:
            raise RuntimeError(f"config ids mask={cfg.mask_token_id} image={cfg.image_token_id}")
        if int(self.tokenizer.convert_tokens_to_ids("<|mask|>")) != MASK_ID:
            raise RuntimeError("tokenizer <|mask|> id mismatch")
        if int(self.tokenizer.convert_tokens_to_ids("<|im_end|>")) != IM_END_ID:
            raise RuntimeError("tokenizer <|im_end|> id mismatch")
        self.mask_id = MASK_ID
        self.core = self.model.model          # DreamVLBaseModel (embed_tokens, layers, norm)
        self.embed = self.model.get_input_embeddings()
        print(f"dreamvl loaded in {time.time() - t0:.1f}s class={type(self.model).__name__} layers={cfg.num_hidden_layers} "
              f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} mask_id={self.mask_id} image_token_id={IMAGE_TOKEN_ID} "
              f"im_end={IM_END_ID} eos={EOS_ID} pixels=[{MIN_PIXELS},{MAX_PIXELS}] "
              f"full_attn_mask={getattr(cfg, 'full_attn_mask', None)} use_cache={cfg.use_cache}", flush=True)

    # ------------------------------------------------------------------ tokens

    def letter_ids(self, letters: str = "ABCDE") -> list[int]:
        out = []
        for letter in letters:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"letter {letter!r} tokenizes to {ids}")
            out.append(int(ids[0]))
        return out

    def letter_id_report(self, letters: str = "ABCDE") -> dict:
        rep = {}
        for letter in letters:
            rep[letter] = {"no_space": self.tokenizer.encode(letter, add_special_tokens=False),
                           "space": self.tokenizer.encode(" " + letter, add_special_tokens=False)}
        return rep

    # ------------------------------------------------------------------ prompt / prefix

    def prompt_string(self, text: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _inputs(self, text: str, image):
        prompt = self.prompt_string(text)
        inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        return prompt, inputs

    def _prefix(self, text: str, image) -> dict:
        """Prompt embeddings (text + projected image features), M-RoPE position ids for prompt+canvas, lengths."""
        prompt, inputs = self._inputs(text, image)
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)
        grid = inputs["image_grid_thw"].to(self.device)
        if input_ids.shape[0] != 1:
            raise RuntimeError(f"batch {tuple(input_ids.shape)}")
        if int(input_ids[0, -1].item()) == self.mask_id:
            raise RuntimeError("prompt ends with a mask token")
        emb = self.embed(input_ids)
        img_mask = input_ids == IMAGE_TOKEN_ID
        n_img = int(img_mask.sum().item())
        feats = torch.cat(self.model.get_image_features(pixel_values, grid), dim=0)
        if feats.shape[0] != n_img:
            raise RuntimeError(f"image features {feats.shape[0]} != image tokens {n_img}")
        proj = self.model.projector(feats.to(emb.device, emb.dtype))
        emb = emb.masked_scatter(img_mask.unsqueeze(-1).expand_as(emb), proj)
        plen = int(input_ids.shape[1])
        canvas_ids = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.device)
        full_ids = torch.cat([input_ids, canvas_ids], dim=1)
        position_ids, rope_deltas = self.model.get_rope_index(full_ids, grid, None, None)
        if tuple(position_ids.shape) != (3, 1, plen + GEN):
            raise RuntimeError(f"position ids {tuple(position_ids.shape)}")
        return {"embeds": emb, "plen": plen, "position_ids": position_ids, "prompt": prompt, "n_image_tokens": n_img,
                "grid_thw": [int(v) for v in grid[0].tolist()], "input_ids": input_ids,
                "rope_delta": int(rope_deltas.reshape(-1)[0].item())}

    def make_forward(self, pre: dict):
        plen, pe, pos = pre["plen"], pre["embeds"], pre["position_ids"]

        def forward(canvas: torch.Tensor) -> torch.Tensor:
            if tuple(canvas.shape) != (1, GEN):
                raise RuntimeError(f"canvas {tuple(canvas.shape)}")
            x = torch.cat([pe, self.embed(canvas).to(pe.dtype)], dim=1)
            out = self.core(inputs_embeds=x, position_ids=pos, use_cache=False, return_dict=True)
            hidden = out.last_hidden_state
            if hidden.shape[1] != plen + GEN:
                raise RuntimeError(f"hidden {tuple(hidden.shape)} vs {plen}+{GEN}")
            # Dream shift: canvas position j is predicted by output position plen-1+j
            return self.model.lm_head(hidden[:, plen - 1: plen - 1 + GEN, :])

        return forward

    # ------------------------------------------------------------------ interface

    def decode(self, text: str, image) -> dict:
        import calib_decode_multi_20261013e as e

        pre = self._prefix(text, image)
        x = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.device)
        d = e.decode_canvas(x, self.mask_id, self.tokenizer, self.make_forward(pre))
        d["finished_text_raw"] = d["finished_text"]
        d["finished_text"] = strip_special_text(d["finished_text"])
        d["finished_letter"] = e.finished_label(d["finished_text"])
        d["canvas_tokens"] = [self.tokenizer.convert_ids_to_tokens(int(t)) for t in x[0].tolist()]
        d["prefix_len"] = pre["plen"]
        d["n_image_tokens"] = pre["n_image_tokens"]
        d["grid_thw"] = pre["grid_thw"]
        return d

    def step0(self, text: str, image):
        pre = self._prefix(text, image)
        x = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.device)
        logits = self.make_forward(pre)(x)
        return logits[0, 0].detach().float().cpu(), pre["plen"]

    def prepare_readout(self):
        norm = self.core.norm
        if "RMSNorm" not in type(norm).__name__:
            raise RuntimeError(f"expected RMSNorm final norm, got {type(norm).__name__}")
        eps = float(norm.variance_epsilon)
        head = self.model.lm_head.weight
        if head.dim() != 2 or head.shape[0] < head.shape[1]:
            raise RuntimeError(f"unexpected lm_head weight {tuple(head.shape)}")
        if not getattr(self, "_hooked", False):
            norm.register_forward_pre_hook(_pre_hook)
            self._hooked = True
        gamma64 = norm.weight.detach().to(dtype=torch.float64).cpu()
        return head, gamma64, eps

    def prior_forward(self, text: str, image):
        pre = self._prefix(text, image)
        canvas = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.device)
        arm(pre["plen"] - 1)                      # the hidden state that predicts canvas position 0 (Dream shift)
        logits = self.make_forward(pre)(canvas)
        h = collect(pre["plen"] + GEN, 1)[0]
        return logits[0, 0].detach().float().cpu(), h, pre["plen"], pre["prompt"]

    # ------------------------------------------------------------------ official sampler (agreement check only)

    def official_generate(self, text: str, image, seed: int = 0, **gen) -> dict:
        """Official diffusion_generate exactly as the HF README / lmms-eval evaluator call it (use_cache False)."""
        cfg = dict(OFFICIAL_LMMS_MCQ)
        cfg.update(gen)
        prompt, inputs = self._inputs(text, image)
        inputs = inputs.to(self.device)
        input_ids = inputs.pop("input_ids")
        torch.manual_seed(seed)
        t0 = time.time()
        out = self.model.diffusion_generate(input_ids, output_history=False, return_dict_in_generate=True,
                                            use_cache=False, **cfg, **inputs)
        sec = time.time() - t0
        gen_ids = out.sequences[0, input_ids.shape[1]:].cpu().tolist()
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return {"official_cfg": cfg, "seed": seed, "gen_ids": gen_ids, "raw_text": raw, "text": cut_official(raw),
                "sec": round(sec, 3), "prefix_len": int(input_ids.shape[1])}
