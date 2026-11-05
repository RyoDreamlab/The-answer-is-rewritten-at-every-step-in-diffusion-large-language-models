#!/usr/bin/env python3
"""Perception-heavy benchmark package (2026-10-13e): MMStar (MCQ) and POPE (yes/no) decodes for LaViDa / MMaDA.

New file; imports (does not modify) calib_decode_multi_20261013c.py for the backends, prompt templates and the
gated rule, and the dataset builders build_mmstar_20261013e.py / build_pope_20261013e.py for the manifests.

datasets  mmstar  datasets/mmstar_20261013e/manifest.json   (1500 MCQ, 4 options; same manifest schema as MMBench)
          pope    datasets/pope_20261013e/manifest.json     (1000 yes/no, balanced, POPE adversarial)
labels    mmstar  option letters A..E (ids 32..36 on both tokenizers), n = the item's option count
          pope    the two answer tokens 'Yes' / 'No' (single tokens, checked at start), n = 2
templates mmstar  exactly calib_decode_multi_20261013c.text_for (LaViDa ScienceQA template / lmms-eval sqa template)
          pope    lmms-eval pope doc_to_text: "{question}\\nAnswer the question using a single word or phrase."
                  (LaViDa: wrapped by the backend as "<image>\\nQuestion: {..}\\nAnswer:" exactly like the MCQ sets)
modes     stock   no logit change, real image
          gated   while canvas position 0 is masked, the n label logits at position 0 are shifted by
                  -alpha_n * alpha_scale * log p_cf(y), alpha_n = 1 - p_min/p_max (rule 'spread') from --prior-file;
                  --alpha-scale 0 must reproduce stock exactly (sanity check ALPHA0_EQUALS_STOCK in the smoke log)
          gray    stock decode with the image replaced by a mid-gray (128,128,128) image of the item's own size
                  (= cond b_imgfree of calib_contentfree.py): image-free control with the real question
          prior   no decode: content-free prior = step-0 forward with a 384x384 mid-gray image and question 'N/A'
                  (mmstar: n 'N/A' options for n in 2..5, cond a_global, as outputs/mmbench_20261013c/cf_*.json;
                   pope: 'N/A' question + the pope instruction, labels Yes/No). Writes cf json (refuses overwrite).
Decode: gen 16, 16 steps, temperature 0, low-confidence remasking (run_unmask_order.unmask_one), one token per step.
Outputs only under outputs/*_20261013e/ ; resumable (jsonl append + id skip); progress json; flock per output.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
sys.path.insert(0, str(PROJECT / "experiments"))
import calib_decode_multi_20261013c as c  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

LETTERS = "ABCDE"
YESNO = ["Yes", "No"]
GEN = base.GEN
GRAY = c.GRAY
GLOBAL_SIZE = c.GLOBAL_SIZE
SUFFIX = "_20261013e"
POPE_POST = "\nAnswer the question using a single word or phrase."
STATE = {"mode": "stock", "log_pcf": None, "alpha": None, "alpha_scale": 1.0, "n_opt": None, "label_ids": None,
         "labels": None, "task": "mcq"}


# ------------------------------------------------------------------ datasets

def gray_like(image):
    from PIL import Image

    return Image.new("RGB", image.size, GRAY)


def pope_text(question: str) -> str:
    return f"{question}{POPE_POST}"


def rows_mmstar(backend: str, limit: int, image_cond: str) -> list[dict]:
    from PIL import Image

    path = PROJECT / "datasets/mmstar_20261013e/manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        ch = list(r["choices"])
        if r["n_opt"] != len(ch) or r["gold"] not in LETTERS[:len(ch)]:
            raise RuntimeError(f"bad mmstar row {r['id']}")

        def load(p=r["image"]):
            im = Image.open(p).convert("RGB")
            return gray_like(im) if image_cond == "gray" else im

        out.append({"id": r["id"], "gold": r["gold"], "gold_text": r["gold_text"], "choices": ch, "n_opt": len(ch),
                    "question": r["question"], "hint": r["hint"],
                    "meta": {"index": r["index"], "category": r["category"], "l2_category": r["l2_category"],
                             "source": r["source"], "image_cond": image_cond},
                    "text": c.text_for(backend, r["question"], r["hint"], ch), "image": load})
    if len({r["id"] for r in out}) != len(out):
        raise RuntimeError("duplicate mmstar ids")
    return out[:limit] if limit else out


def rows_pope(backend: str, limit: int, image_cond: str) -> list[dict]:
    from PIL import Image

    path = PROJECT / "datasets/pope_20261013e/manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if r["gold"] not in YESNO:
            raise RuntimeError(f"bad pope row {r['id']}")

        def load(p=r["image"]):
            im = Image.open(p).convert("RGB")
            return gray_like(im) if image_cond == "gray" else im

        out.append({"id": r["id"], "gold": r["gold"], "gold_text": r["gold"], "choices": list(YESNO), "n_opt": 2,
                    "question": r["question"], "hint": "",
                    "meta": {"question_id": r["question_id"], "category": r["category"],
                             "image_source": r["image_source"], "image_cond": image_cond},
                    "text": pope_text(r["question"]), "image": load})
    if len({r["id"] for r in out}) != len(out):
        raise RuntimeError("duplicate pope ids")
    return out[:limit] if limit else out


DATASETS = {"mmstar": rows_mmstar, "pope": rows_pope}
TASK = {"mmstar": "mcq", "pope": "yesno"}


# ------------------------------------------------------------------ priors

def load_priors(path: Path) -> tuple[dict, dict, dict, list]:
    """cf json written by --mode prior (records with cond a_global). Returns log_pcf{n}, alpha{n}, p_cf{n}, labels."""
    d = json.loads(path.read_text(encoding="utf-8"))
    recs = d["records"]
    glob = {int(r["n_opt"]): r for r in recs if r["cond"] == "a_global"}
    if not glob:
        raise RuntimeError(f"no a_global records in {path}")
    log_pcf, alpha, pcf = {}, {}, {}
    for n, r in glob.items():
        lp = torch.log_softmax(torch.tensor(r["letter_logit64"][:n], dtype=torch.float64), dim=0)
        log_pcf[n] = [float(v) for v in lp]
        pcf[n] = [float(v) for v in lp.exp()]
        alpha[n] = 1.0 - math.exp(-float(lp.max() - lp.min()))
    return log_pcf, alpha, pcf, list(d["labels"])


# ------------------------------------------------------------------ decode

def label_vals(logits_pos0, ids) -> list[float]:
    return [float(logits_pos0[i]) for i in ids]


def restricted_label(vals: list[float], n: int) -> str:
    return STATE["labels"][max(range(n), key=lambda j: (vals[j], -j))]


def finished_label(text: str) -> str | None:
    if STATE["task"] == "mcq":
        return delta.leading_letter(text)
    m = re.match(r"^(yes|no)(?:$|[\s\.\,\!\:\)])", (text or "").strip(), re.I)
    return m.group(1).capitalize() if m else None


def decode_canvas(x, mask_id, tokenizer, forward) -> dict:
    if x.shape[1] != GEN:
        raise RuntimeError(f"canvas {tuple(x.shape)} != gen {GEN}")
    n = STATE["n_opt"]
    all_ids = STATE["label_ids"]
    ids = all_ids[:n]
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
            raw = label_vals(logits[0, 0], all_ids)
            if calib:
                logits = logits.float()
                logits[0, 0, ids] = logits[0, 0, ids] - shift.to(logits.device)
            cal = label_vals(logits[0, 0], all_ids)
            pos0_argmax.append(base.piece(tokenizer, int(torch.argmax(logits[0, 0]).item()), mask_id))
            if step == 0:
                step0 = {"raw": restricted_label(raw, n), "cal": restricted_label(cal, n)}
            raw_at_commit, cal_at_commit = raw, cal
        pos = base.unmask_one(x, logits, mask_id)
        unmask_step[pos] = step
    letter_step = unmask_step[0]
    if letter_step < 0:
        raise RuntimeError("answer position was never selected")
    if len(pos0_argmax) != letter_step + 1:
        raise RuntimeError(f"recorded {len(pos0_argmax)} argmax steps, answer unmasked at {letter_step}")
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
        "finished_letter": finished_label(pred),
    }


# ------------------------------------------------------------------ backends (reuse the 20261013c prefix / prompt code)

def label_ids_of(tokenizer, labels) -> list[int]:
    out = []
    for lab in labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"label {lab!r} tokenizes to {ids} (need a single token)")
        out.append(int(ids[0]))
    return out


class LavidaE(c.Lavida):
    def decode(self, text: str, image) -> dict:
        past, plen, _ = self._prefix(text, image)
        x = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.model.device)
        d = decode_canvas(x, self.mask_id, self.tokenizer, lambda cv: self.traj.lavida_forward(self.lm, past, cv))
        d["prefix_len"] = plen
        return d

    def step0(self, text: str, image):
        """One forward with the full canvas masked; returns (position-0 logits float cpu, prefix_len)."""
        past, plen, _ = self._prefix(text, image)
        x = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.model.device)
        logits = self.traj.lavida_forward(self.lm, past, x)
        return logits[0, 0].detach().float().cpu(), plen


class MmadaE(c.Mmada):
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

    def step0(self, text: str, image):
        prompt_ids, plen, _ = self._prompt(text, image)
        x = torch.full((1, GEN), self.mask_id, dtype=torch.long, device=self.device)
        logits = base.mmada_logits(self.model, prompt_ids, x, self.mask_id, None)
        if logits.shape[1] != plen + GEN:
            raise RuntimeError(f"mmada logits {tuple(logits.shape)}")
        return logits[0, plen].detach().float().cpu(), plen


BACKENDS = {"lavida": LavidaE, "mmada": MmadaE}
DEFAULT_MODEL = {"lavida": "checkpoints/lavida-llada-v1.0-instruct", "mmada": "checkpoints/MMaDA-8B-Base"}


def load_backend(backend: str, cuda: int, model_path: str = ""):
    """Shared by image_dose_step0_20261013e.py. Returns the backend object with label ids set in STATE."""
    ns = argparse.Namespace(cuda=cuda, model_path=model_path or str(PROJECT / DEFAULT_MODEL[backend]),
                            vq_model=str(PROJECT / "checkpoints/magvitv2"), mmada_repo=str(PROJECT / "repos/MMaDA"))
    return BACKENDS[backend](ns)


def set_labels(be, task: str) -> None:
    STATE["task"] = task
    STATE["labels"] = list(LETTERS) if task == "mcq" else list(YESNO)
    STATE["label_ids"] = label_ids_of(be.tokenizer, STATE["labels"])
    if task == "mcq" and STATE["label_ids"] != [32, 33, 34, 35, 36]:
        raise RuntimeError(f"unexpected letter ids {STATE['label_ids']}")
    print(f"labels={dict(zip(STATE['labels'], STATE['label_ids']))} mask_id={be.mask_id}", flush=True)


# ------------------------------------------------------------------ runs

def prior_text(backend: str, dataset: str, n: int) -> str:
    if dataset == "pope":
        return pope_text("N/A")
    return c.text_for(backend, "N/A", "", ["N/A"] * n)


def run_prior(args, be) -> None:
    head, gamma64, eps = be.prepare_readout()
    ids = STATE["label_ids"]
    labels = STATE["labels"]
    W64 = torch.stack([head[i].detach().to(torch.float64).cpu() for i in ids])
    ns = (2,) if args.dataset == "pope" else (2, 3, 4, 5)
    records = []
    for n in ns:
        text = prior_text(be.name, args.dataset, n)
        with torch.no_grad():
            logits, h, plen, prompt = be.prior_forward(text, c.gray_image())
        h64 = h.to(torch.float64)
        l64 = [float(v) for v in (W64 @ (gamma64 * (h64 * torch.rsqrt(torch.dot(h64, h64) / h64.shape[0] + eps))))]
        lm = [float(logits[i]) for i in ids]
        lp = torch.log_softmax(torch.tensor(l64[:n], dtype=torch.float64), dim=0)
        a = int(torch.argmax(logits).item())
        rec = {"id": f"global_n{n}", "cond": "a_global", "backend": be.name, "dataset": args.dataset, "n_opt": n,
               "labels": labels[:n], "prompt_text": prompt, "letter_ids": ids, "letter_logit64": l64,
               "letter_logit_model": lm, "max_abs_diff_readout_vs_model": max(abs(x - y) for x, y in zip(l64, lm)),
               "argmax_id": a, "argmax_text": base.piece(be.tokenizer, a, be.mask_id), "prefix_len": plen,
               "decode_step": 0, "canvas_index": 0, "p_cf": [float(v) for v in lp.exp()],
               "log_p_cf": [float(v) for v in lp],
               "alpha_1_minus_pmin_over_pmax": 1.0 - math.exp(-float(lp.max() - lp.min()))}
        records.append(rec)
        print(f"PRIOR backend={be.name} dataset={args.dataset} n={n} p_cf={dict(zip(labels[:n], [round(v, 4) for v in rec['p_cf']]))} "
              f"alpha={rec['alpha_1_minus_pmin_over_pmax']:.6f} argmax={rec['argmax_text']!r} "
              f"readout_vs_model_maxdiff={rec['max_abs_diff_readout_vs_model']:.4f} prefix_len={plen}", flush=True)
    by_n = {str(r["n_opt"]): {"p_cf": r["p_cf"], "log_p_cf": r["log_p_cf"], "alpha_spread": r["alpha_1_minus_pmin_over_pmax"],
                              "prior_label": labels[int(torch.tensor(r["p_cf"]).argmax())]} for r in records}
    out = {"backend": be.name, "dataset_template_of": args.dataset, "labels": labels, "letter_ids": ids, "gray": GRAY,
           "image_size": GLOBAL_SIZE, "gen": GEN, "rule": "alpha_n = 1 - p_min/p_max over the n labels of cond a_global",
           "by_n_opt": by_n, "records": records}
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
                      "n_opt": row["n_opt"], "labels": STATE["labels"][:row["n_opt"]], **row["meta"],
                      "gen_length": GEN, **d, "sec": round(time.time() - t1, 3)}
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
                      f"letter={record['finished_letter']} step0={record['step0_restricted_raw']}->{record['step0_restricted_cal']} "
                      f"step={record['letter_unmask_step']} text={record['finished_text']!r}", flush=True)
    if finished != len(rows):
        raise RuntimeError(f"finished {finished} of {len(rows)}")
    print(f"SUMMARY n={finished} mode={STATE['mode']} backend={be.name} dataset={args.dataset}", flush=True)


def check_output_path(q: Path) -> None:
    r = q.resolve()
    if (PROJECT / "outputs").resolve() not in r.parents or not r.parent.name.endswith(SUFFIX):
        raise RuntimeError(f"refusing to write outside outputs/*{SUFFIX}/: {q}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--mode", choices=("stock", "gated", "gray", "prior"), required=True)
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
    image_cond = "gray" if args.mode == "gray" else "real"
    if args.print_prompts:
        for row in DATASETS[args.dataset](args.backend, args.print_prompts, image_cond):
            print(json.dumps({"id": row["id"], "gold": row["gold"], "n_opt": row["n_opt"], "text": row["text"]},
                             ensure_ascii=False, indent=1))
        return
    if args.cuda not in (0, 1):
        raise RuntimeError("--cuda must be 0 or 1")
    for q in (args.output, args.progress):
        if q is not None:
            check_output_path(q)
    if GEN != 16:
        raise RuntimeError(f"expected gen 16, got {GEN}")
    args.model_path = args.model_path or str(PROJECT / DEFAULT_MODEL[args.backend])
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
    task = TASK[args.dataset]
    labels = list(LETTERS) if task == "mcq" else list(YESNO)
    if args.mode == "gated":
        if args.prior_file is None:
            raise RuntimeError("--prior-file required for gated")
        log_pcf, alpha, pcf, plabels = load_priors(args.prior_file)
        if plabels != labels:
            raise RuntimeError(f"prior labels {plabels} != dataset labels {labels}")
        if args.alpha_rule == "full":
            alpha = {n: 1.0 for n in alpha}
        STATE["log_pcf"], STATE["alpha"] = log_pcf, alpha
        for n in sorted(log_pcf):
            print(f"PRIOR n={n} p_cf={dict(zip(labels, [round(v, 6) for v in pcf[n]]))} alpha_{args.alpha_rule}={alpha[n]:.6f} "
                  f"alpha_scale={args.alpha_scale} shift(-a*logp)={[round(-alpha[n] * args.alpha_scale * v, 4) for v in log_pcf[n]]}",
                  flush=True)
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    print(f"start backend={args.backend} dataset={args.dataset} mode={args.mode} image_cond={image_cond} cuda={args.cuda} "
          f"gen={GEN} steps={GEN} temperature=0 remasking=low_confidence limit={args.limit} output={args.output}", flush=True)
    try:
        rows = None
        if args.mode != "prior":
            rows = DATASETS[args.dataset](args.backend, args.limit, image_cond)
            need = {r["n_opt"] for r in rows}
            if args.mode == "gated" and not need <= set(STATE["log_pcf"]):
                raise RuntimeError(f"prior missing for n_opt {sorted(need - set(STATE['log_pcf']))}")
            print(f"rows={len(rows)} n_opt_hist={ {n: sum(r['n_opt'] == n for r in rows) for n in sorted(need)} }", flush=True)
            print(f"example prompt id={rows[0]['id']}:\n{rows[0]['text']}", flush=True)
        be = BACKENDS[args.backend](args)
        set_labels(be, task)
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
