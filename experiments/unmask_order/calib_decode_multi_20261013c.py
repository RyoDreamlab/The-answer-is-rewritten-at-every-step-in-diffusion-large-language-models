#!/usr/bin/env python3
"""Stock / gated content-free-calibrated gen-16 low-confidence letter decode for several backends and datasets.

New file (2026-10-13c); existing scripts are imported, not modified.

backends  lavida  (calib_decode_m3cot.load_lavida; prefix KV cache + canvas forward, as calib_decode.py)
          mmada   (calib_decode_m3cot.load_mmada; full forward prompt+canvas, as letter_argmax_traj.mmada_decode)
          lladav  (lladav_backend_20261013c; full forward prefix-embeds+canvas, no KV cache)
datasets  sciqa   ScienceQA-IMG test 2017  (lavida: datasets/urd_eval_full manifest and its composed question,
                                            byte-identical to the earlier LaViDa runs; mmada/lladav:
                                            datasets/sciqa_img_standard_v1 + sciqa_standard_eval.build_task_prompt)
          m3cot   M3CoT validation 1108    (calib_decode_m3cot.load_rows)
          mmbench MMBench-dev EN           (datasets/mmbench_dev_v1/manifest.json, build_mmbench_dev_20261013c.py)
templates lavida  "{question}[\nContext: {hint}]\nOptions:\nA. ..\nWhich letter is correct?" wrapped as
                  "<image>\nQuestion: {..}\nAnswer:" (conv 'llada')  -- the LaViDa ScienceQA template
          mmada / lladav  lmms-eval sqa_doc_to_text: "[Context: {hint}\n]{question}\nA. ..\n..
                  \nAnswer with the option's letter from the given choices directly."  (lladav: conv 'llava_llada')
modes     stock   no logit change
          gated   while canvas position 0 is masked, position-0 logits of the item's n letters are shifted by
                  -alpha_n * alpha_scale * log p_cf(y); alpha_n = 1 - p_min/p_max (--alpha-rule spread) or 1 (full).
                  --alpha-scale 0 must reproduce stock exactly (sanity).
          prior   no decode: one step-0 forward per n in 2..5 with a mid-gray 384x384 image, question 'N/A',
                  n options 'N/A', hint '' (cond a_global, the convention of outputs/calib_vote_20261012) and
                  hint 'N/A' (cond a_global_hintna, reference only). float64 letter readout at the final-norm input.
Decode: gen 16, 16 steps, temperature 0, low-confidence remasking (run_unmask_order.unmask_one), one token per step.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
sys.path.insert(0, str(PROJECT / "experiments"))
import calib_decode_m3cot as m  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402
import sciqa_standard_eval as sq  # noqa: E402

LETTERS = "ABCDE"
GEN = base.GEN
GRAY = (128, 128, 128)
GLOBAL_SIZE = (384, 384)
SUFFIX = "_20261013c"
STATE = {"mode": "stock", "log_pcf": None, "alpha": None, "alpha_scale": 1.0, "n_opt": None, "letter_ids": None}


# ------------------------------------------------------------------ text templates

def text_for(backend: str, question: str, hint: str, choices: list[str]) -> str:
    if backend == "lavida":
        return m.lavida_wrapped(m.lavida_text(question, hint, list(choices)))
    return sq.build_task_prompt({"question": question, "choices": list(choices), "hint": hint})


def gray_image(size=GLOBAL_SIZE):
    from PIL import Image

    return Image.new("RGB", tuple(int(s) for s in size), GRAY)


# ------------------------------------------------------------------ datasets

def _choices(c):
    if isinstance(c, str):
        import ast

        c = ast.literal_eval(c)
    return [str(x) for x in c]


def rows_sciqa(backend: str, limit: int) -> list[dict]:
    from PIL import Image

    out = []
    if backend == "lavida":
        rows = json.loads((PROJECT / "datasets/urd_eval_full/scienceqa_img_test.json").read_text(encoding="utf-8"))
        if len(rows) != 2017:
            raise RuntimeError(f"expected 2017 rows, got {len(rows)}")
        for r in rows:
            ch = _choices(r["choices"])
            path = r["image"]
            out.append({"id": r["id"], "gold": str(r["gold"]).strip(), "gold_text": str(r["gold_text"]), "choices": ch,
                        "n_opt": len(ch), "question": r["question"], "hint": r.get("hint") or "",
                        "meta": {"subject": r.get("subject")},
                        # composed question already contains Context/Options; identical to the earlier LaViDa runs
                        "text": m.lavida_wrapped(r["question"]),
                        "image": (lambda p=path: Image.open(p).convert("RGB"))})
    else:
        data_dir = PROJECT / "datasets/sciqa_img_standard_v1"
        recs = sq.load_manifest(data_dir / "manifest.json", data_dir)
        if len(recs) != 2017:
            raise RuntimeError(f"expected 2017 rows, got {len(recs)}")
        store = sq.ParquetImageStore(data_dir, recs)
        for r in recs:
            ch = _choices(r["choices"])
            gold = str(r["gold_option"]).strip()
            out.append({"id": r["id"], "gold": gold, "gold_text": ch[LETTERS.index(gold)], "choices": ch,
                        "n_opt": len(ch), "question": r["question"], "hint": r.get("hint") or "",
                        "meta": {"subject": r.get("subject"), "category": r.get("category"), "topic": r.get("topic")},
                        "text": text_for(backend, r["question"], r.get("hint") or "", ch),
                        "image": (lambda rec=r: store.load_rgb(rec))})
    return out[:limit] if limit else out


def rows_m3cot(backend: str, limit: int) -> list[dict]:
    out = []
    for r in m.load_rows(limit):
        out.append({"id": r["id"], "gold": r["gold"], "gold_text": r["gold_text"], "choices": r["choices"],
                    "n_opt": r["n_opt"], "question": r["question"], "hint": r["context"],
                    "meta": {"domain": r["domain"], "topic": r["topic"], "category": r["category"]},
                    "text": text_for(backend, r["question"], r["context"], r["choices"]),
                    "image": (lambda row=r: m.rgb(row))})
    return out


def rows_mmbench(backend: str, limit: int) -> list[dict]:
    from PIL import Image

    path = PROJECT / "datasets/mmbench_dev_v1/manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        ch = list(r["choices"])
        if r["n_opt"] != len(ch) or r["gold"] not in LETTERS[:len(ch)]:
            raise RuntimeError(f"bad mmbench row {r['id']}")
        out.append({"id": r["id"], "gold": r["gold"], "gold_text": r["gold_text"], "choices": ch, "n_opt": len(ch),
                    "question": r["question"], "hint": r["hint"],
                    "meta": {"index": r["index"], "category": r["category"], "l2_category": r["l2_category"],
                             "source": r["source"]},
                    "text": text_for(backend, r["question"], r["hint"], ch),
                    "image": (lambda p=r["image"]: Image.open(p).convert("RGB"))})
    if len({r["id"] for r in out}) != len(out):
        raise RuntimeError("duplicate mmbench ids")
    return out[:limit] if limit else out


DATASETS = {"sciqa": rows_sciqa, "m3cot": rows_m3cot, "mmbench": rows_mmbench}


# ------------------------------------------------------------------ priors

def load_priors(path: Path) -> tuple[dict, dict, dict]:
    """Returns log_pcf{n}, alpha_spread{n}, p_cf{n}. Accepts calib_vote .jsonl (cond a_global) or this script's .json."""
    if path.suffix == ".jsonl":
        recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    else:
        recs = json.loads(path.read_text(encoding="utf-8"))["records"]
    glob = {int(r["n_opt"]): r for r in recs if r["cond"] == "a_global"}
    if not set(glob) >= {2, 3, 4}:
        raise RuntimeError(f"global priors only for n in {sorted(glob)}")
    log_pcf, alpha, pcf = {}, {}, {}
    for n, r in glob.items():
        lp = torch.log_softmax(torch.tensor(r["letter_logit64"][:n], dtype=torch.float64), dim=0)
        log_pcf[n] = [float(v) for v in lp]
        pcf[n] = [float(v) for v in lp.exp()]
        alpha[n] = 1.0 - math.exp(-float(lp.max() - lp.min()))
    return log_pcf, alpha, pcf


# ------------------------------------------------------------------ decode

def letter_vals(logits_pos0, ids) -> list[float]:
    return [float(logits_pos0[i]) for i in ids]


def restricted_letter(vals: list[float], n: int) -> str:
    return LETTERS[max(range(n), key=lambda j: (vals[j], -j))]


def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
    if x.shape[1] != GEN:
        raise RuntimeError(f"canvas {tuple(x.shape)} != gen {GEN}")
    n = STATE["n_opt"]
    ids = STATE["letter_ids"][:n]
    calib = STATE["mode"] == "gated"
    shift = None
    if calib:
        a = STATE["alpha"][n] * STATE["alpha_scale"]
        shift = (a * torch.tensor(STATE["log_pcf"][n], dtype=torch.float64)).to(torch.float32)
    unmask_step = [-1] * GEN
    pos0_argmax = []
    raw_at_commit = cal_at_commit = None
    step0 = None
    for step in range(GEN):
        logits = forward(x)
        if int(x[0, 0].item()) == int(mask_id):
            raw = letter_vals(logits[0, 0], STATE["letter_ids"])
            if calib:
                logits = logits.float()
                logits[0, 0, ids] = logits[0, 0, ids] - shift.to(logits.device)
            cal = letter_vals(logits[0, 0], STATE["letter_ids"])
            pos0_argmax.append(base.piece(tokenizer, int(torch.argmax(logits[0, 0]).item()), mask_id))
            if step == 0:
                step0 = {"raw": restricted_letter(raw, n), "cal": restricted_letter(cal, n)}
            raw_at_commit, cal_at_commit = raw, cal
        pos = base.unmask_one(x, logits, mask_id)
        unmask_step[pos] = step
    letter_step = unmask_step[0]
    if letter_step < 0:
        raise RuntimeError("letter position was never selected")
    if len(pos0_argmax) != letter_step + 1:
        raise RuntimeError(f"recorded {len(pos0_argmax)} argmax steps, letter unmasked at {letter_step}")
    if sorted(unmask_step) != list(range(GEN)):
        raise RuntimeError(f"unmask steps {unmask_step}")
    pred = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    return {
        "letter_unmask_step": int(letter_step),
        "pos0_argmax": pos0_argmax,
        "letter_logits_raw_at_commit": raw_at_commit,
        "letter_logits_cal_at_commit": cal_at_commit,
        "step0_restricted_raw": step0["raw"],
        "step0_restricted_cal": step0["cal"],
        "finished_text": pred,
        "finished_letter": delta.leading_letter(pred),
    }


# ------------------------------------------------------------------ backends

class Lavida:
    name = "lavida"

    def __init__(self, args):
        import letter_argmax_traj as traj

        self.traj = traj
        self.tokenizer, self.model, self.processor, self.lm = m.load_lavida(args)
        self.mask_id = base.MASK_ID_LAVIDA
        self.cuda = args.cuda

    def letter_ids(self):
        import hidden_shapley_split as hss

        lid = hss.letter_ids(self.tokenizer, LETTERS)
        return [lid[l] for l in LETTERS]

    def _prefix(self, text: str, image):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        conv = delta.copy_conv(conv_templates["llada"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + f"Question: {text}\nAnswer:")
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = ids.unsqueeze(0).to(self.model.device)
        tensors = process_images([image], self.processor, self.model.config)
        tensors = [t.to(device=self.model.device, dtype=torch.bfloat16) for t in tensors]
        prepared = self.model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, tensors, ["image"], image_sizes=[image.size], return_inputs=True
        )
        embeds = prepared[4]
        base.KeyBlock.cols = None
        past = self.lm(None, input_embeddings=embeds, use_cache=True).attn_key_values
        return past, int(embeds.shape[1]), f"<image>\nQuestion: {text}\nAnswer:"

    def decode(self, text: str, image) -> dict:
        past, plen, _ = self._prefix(text, image)
        x = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.model.device)
        d = decode_canvas(x, self.mask_id, self.tokenizer, lambda c: self.traj.lavida_forward(self.lm, past, c))
        d["prefix_len"] = plen
        return d

    def prepare_readout(self):
        import hidden_shapley_split as hss

        self.hss = hss
        return hss.prepare(self.lm, self.cuda)

    def prior_forward(self, text: str, image):
        past, plen, prompt = self._prefix(text, image)
        canvas = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.model.device)
        self.hss.arm(0)
        logits = self.lm(None, input_embeddings=self.lm.transformer.wte(canvas), past_key_values=past,
                         use_cache=False).logits
        h = self.hss.collect(GEN, 1)[0]
        return logits[0, 0].detach().float().cpu(), h, plen, prompt


class Mmada:
    name = "mmada"

    def __init__(self, args):
        self.mm, self.model, self.tokenizer, self.vq, self.device = m.load_mmada(args)
        self.mask_id = self.mm.MASK_ID
        self.cuda = args.cuda

    def letter_ids(self):
        import hidden_shapley_split as hss

        lid = hss.letter_ids(self.tokenizer, LETTERS)
        return [lid[l] for l in LETTERS]

    def _prompt(self, text: str, image):
        vq_ids = self.mm.encode_vq(self.vq, image, self.device)
        prompt_ids, ptext = self.mm.build_prompt_ids(self.tokenizer, text, vq_ids, self.device)
        return prompt_ids, int(prompt_ids.shape[1]), ptext

    def decode(self, text: str, image) -> dict:
        prompt_ids, plen, _ = self._prompt(text, image)
        x = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.device)

        def forward(canvas):
            logits = base.mmada_logits(self.model, prompt_ids, canvas, self.mask_id, None)
            if logits.shape[1] != plen + GEN:
                raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
            return logits[:, plen:, :]

        d = decode_canvas(x, self.mask_id, self.tokenizer, forward)
        d["prefix_len"] = plen
        return d

    def prepare_readout(self):
        import hidden_shapley_split as hss

        self.hss = hss
        return hss.prepare(self.model.model, self.cuda)

    def prior_forward(self, text: str, image):
        prompt_ids, plen, ptext = self._prompt(text, image)
        x = torch.cat([prompt_ids, torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.device)], dim=1)
        self.hss.arm(plen)
        logits = self.model(x).logits
        h = self.hss.collect(plen + GEN, 1)[0]
        return logits[0, plen].detach().float().cpu(), h, plen, str(ptext)


class Lladav:
    name = "lladav"

    def __init__(self, args):
        import lladav_backend_20261013c as lb

        self.lb = lb
        self.ctx = lb.load(args.model_path, args.cuda)
        self.tokenizer = self.ctx["tokenizer"]
        self.model = self.ctx["model"]
        self.mask_id = lb.MASK_ID
        self.cuda = args.cuda

    def letter_ids(self):
        return self.lb.letter_ids(self.ctx, LETTERS)

    def decode(self, text: str, image) -> dict:
        pe, plen = self.lb.prefix(self.ctx, text, image)
        x = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=pe.device)
        d = decode_canvas(x, self.mask_id, self.tokenizer, self.lb.make_forward(self.ctx, pe))
        d["prefix_len"] = plen
        return d

    def prepare_readout(self):
        return self.lb.prepare_readout(self.ctx)

    def prior_forward(self, text: str, image):
        pe, plen = self.lb.prefix(self.ctx, text, image)
        canvas = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=pe.device)
        fwd = self.lb.make_forward(self.ctx, pe)
        self.lb.arm(plen)
        logits = fwd(canvas)
        h = self.lb.collect(plen + GEN, 1)[0]
        return logits[0, 0].detach().float().cpu(), h, plen, self.lb.prompt_string(self.ctx, text)


BACKENDS = {"lavida": Lavida, "mmada": Mmada, "lladav": Lladav}


# ------------------------------------------------------------------ runs

def run_prior(args, be) -> None:
    head, gamma64, eps = be.prepare_readout()
    ids = STATE["letter_ids"]
    W64 = torch.stack([head[i].detach().to(torch.float64).cpu() for i in ids])
    records = []
    for cond, hint in (("a_global", ""), ("a_global_hintna", "N/A")):
        for n in (2, 3, 4, 5):
            text = text_for(be.name, "N/A", hint, ["N/A"] * n)
            with torch.no_grad():
                logits, h, plen, prompt = be.prior_forward(text, gray_image())
            h64 = h.to(torch.float64)
            l64 = [float(v) for v in (W64 @ (gamma64 * (h64 * torch.rsqrt(torch.dot(h64, h64) / h64.shape[0] + eps))))]
            lm = [float(logits[i]) for i in ids]
            lp = torch.log_softmax(torch.tensor(l64[:n], dtype=torch.float64), dim=0)
            a = int(torch.argmax(logits).item())
            rec = {"id": f"global_n{n}" if cond == "a_global" else f"global_hintna_n{n}", "cond": cond,
                   "backend": be.name, "n_opt": n, "template": "lavida_sqa" if be.name == "lavida" else "lmms_sqa",
                   "prompt_text": prompt, "letter_ids": ids, "letter_logit64": l64, "letter_logit_model": lm,
                   "max_abs_diff_readout_vs_model": max(abs(x - y) for x, y in zip(l64, lm)),
                   "argmax_id": a, "argmax_text": base.piece(be.tokenizer, a, be.mask_id), "prefix_len": plen,
                   "decode_step": 0, "canvas_index": 0, "p_cf": [float(v) for v in lp.exp()],
                   "log_p_cf": [float(v) for v in lp],
                   "alpha_1_minus_pmin_over_pmax": 1.0 - math.exp(-float(lp.max() - lp.min()))}
            records.append(rec)
            print(f"PRIOR backend={be.name} cond={cond} n={n} p_cf={[round(v, 4) for v in rec['p_cf']]} "
                  f"alpha={rec['alpha_1_minus_pmin_over_pmax']:.6f} argmax={rec['argmax_text']!r} "
                  f"readout_vs_model_maxdiff={rec['max_abs_diff_readout_vs_model']:.4f} prefix_len={plen}", flush=True)
    by_n = {str(r["n_opt"]): {"p_cf": r["p_cf"], "log_p_cf": r["log_p_cf"], "alpha_spread": r["alpha_1_minus_pmin_over_pmax"]}
            for r in records if r["cond"] == "a_global"}
    out = {"backend": be.name, "dataset_template_of": args.dataset, "letter_ids": ids, "gray": GRAY, "image_size": GLOBAL_SIZE,
           "gen": GEN, "rule": "alpha_n = 1 - p_min/p_max over the n letters of cond a_global", "by_n_opt": by_n,
           "records": records}
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


def run_decode(args, be, rows) -> None:
    done = base.load_done(args.output)
    finished = len(done)
    t0 = time.time()
    n_new = 0
    print(f"resume n_done={finished}/{len(rows)} mode={STATE['mode']}", flush=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            STATE["n_opt"] = row["n_opt"]
            t1 = time.time()
            with torch.no_grad():
                d = be.decode(row["text"], row["image"]())
            alpha = STATE["alpha"][row["n_opt"]] * STATE["alpha_scale"] if STATE["mode"] == "gated" else 0.0
            record = {"id": row["id"], "gold": row["gold"], "gold_text": row["gold_text"], "backend": be.name,
                      "dataset": args.dataset, "mode": STATE["mode"], "calib": STATE["mode"] == "gated",
                      "alpha_rule": args.alpha_rule if STATE["mode"] == "gated" else None, "alpha": alpha,
                      "n_opt": row["n_opt"], **row["meta"], "gen_length": GEN, **d, "sec": round(time.time() - t1, 3)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(record["id"])
            finished += 1
            n_new += 1
            el = time.time() - t0
            rate = n_new / el if el > 0 else 0.0
            base.write_progress(args.progress, {
                "backend": be.name, "dataset": args.dataset, "mode": STATE["mode"], "n_done": finished,
                "n_total": len(rows), "id": record["id"], "letter_unmask_step": record["letter_unmask_step"],
                "elapsed_sec": round(el, 1), "items_per_sec": round(rate, 4),
                "eta_sec": round((len(rows) - finished) / rate, 0) if rate > 0 else None, "output": str(args.output),
            })
            if finished % 25 == 0 or finished == len(rows) or args.limit:
                print(f"PROGRESS n_done={finished}/{len(rows)} rate={rate:.3f}/s id={record['id']} gold={record['gold']} "
                      f"letter={record['finished_letter']} step={record['letter_unmask_step']} "
                      f"text={record['finished_text']!r}", flush=True)
    if finished != len(rows):
        raise RuntimeError(f"finished {finished} of {len(rows)}")
    print(f"SUMMARY n={finished} mode={STATE['mode']} backend={be.name} dataset={args.dataset}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--mode", choices=("stock", "gated", "prior"), required=True)
    p.add_argument("--alpha-rule", choices=("spread", "full"), default="spread")
    p.add_argument("--alpha-scale", type=float, default=1.0, help="multiplies alpha_n; 0 reproduces stock")
    p.add_argument("--prior-file", type=Path, default=None)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument("--model-path", default="")
    p.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    p.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    p.add_argument("--print-prompts", type=int, default=0)
    args = p.parse_args()
    if args.print_prompts:
        for row in DATASETS[args.dataset](args.backend, args.print_prompts):
            print(json.dumps({"id": row["id"], "gold": row["gold"], "n_opt": row["n_opt"], "text": row["text"]},
                             ensure_ascii=False, indent=1))
        return
    if args.cuda == 0:
        raise RuntimeError("GPU 0 belongs to another agent; use --cuda 1")
    for q in (args.output, args.progress):
        if q is None:
            continue
        r = q.resolve()
        if (PROJECT / "outputs").resolve() not in r.parents or not r.parent.name.endswith(SUFFIX):
            raise RuntimeError(f"refusing to write outside outputs/*{SUFFIX}/: {q}")
    if GEN != 16:
        raise RuntimeError(f"expected gen 16, got {GEN}")
    defaults = {"lavida": "checkpoints/lavida-llada-v1.0-instruct", "mmada": "checkpoints/MMaDA-8B-Base",
                "lladav": "checkpoints/LLaDA-V"}
    args.model_path = args.model_path or str(PROJECT / defaults[args.backend])
    STATE["mode"] = args.mode
    STATE["alpha_scale"] = float(args.alpha_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "prior" and args.output.exists():
        raise RuntimeError(f"prior file exists, refusing to overwrite: {args.output}")
    lock_handle = args.output.with_suffix(args.output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {args.output}") from exc
    if args.mode == "gated":
        if args.prior_file is None:
            raise RuntimeError("--prior-file required for gated")
        log_pcf, alpha, pcf = load_priors(args.prior_file)
        if args.alpha_rule == "full":
            alpha = {n: 1.0 for n in alpha}
        STATE["log_pcf"], STATE["alpha"] = log_pcf, alpha
        for n in sorted(log_pcf):
            print(f"PRIOR n={n} p_cf={dict(zip(LETTERS, [round(v, 6) for v in pcf[n]]))} alpha_{args.alpha_rule}={alpha[n]:.6f} "
                  f"alpha_scale={args.alpha_scale} shift(-a*logp)={[round(-alpha[n] * args.alpha_scale * v, 4) for v in log_pcf[n]]}",
                  flush=True)
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    print(f"start backend={args.backend} dataset={args.dataset} mode={args.mode} cuda={args.cuda} gen={GEN} steps={GEN} "
          f"temperature=0 remasking=low_confidence limit={args.limit} output={args.output}", flush=True)
    try:
        rows = None
        if args.mode != "prior":
            rows = DATASETS[args.dataset](args.backend, args.limit)
            need = {r["n_opt"] for r in rows}
            if args.mode == "gated" and not need <= set(STATE["log_pcf"]):
                raise RuntimeError(f"prior missing for n_opt {sorted(need - set(STATE['log_pcf']))}")
            print(f"rows={len(rows)} n_opt_hist={ {n: sum(r['n_opt'] == n for r in rows) for n in sorted(need)} }", flush=True)
            print(f"example prompt id={rows[0]['id']}:\n{rows[0]['text']}", flush=True)
        be = BACKENDS[args.backend](args)
        STATE["letter_ids"] = be.letter_ids()
        print(f"letter_ids={dict(zip(LETTERS, STATE['letter_ids']))} mask_id={be.mask_id}", flush=True)
        if args.backend in ("lavida", "mmada") and STATE["letter_ids"] != [32, 33, 34, 35, 36]:
            raise RuntimeError(f"unexpected letter ids {STATE['letter_ids']}")
        if args.mode == "prior":
            run_prior(args, be)
            return
        if args.progress is None:
            raise RuntimeError("--progress required for decode")
        import letter_argmax_traj as traj

        traj.repair_tail(args.output)
        run_decode(args, be, rows)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
