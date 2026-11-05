#!/usr/bin/env python3
"""Stock / gated / gray / prior gen-16 low-confidence decodes for the Dream-VL-7B backend on all five benchmark sets
(new file 2026-10-13g; imports, never edits, the 20261013c / 20261013e packages).

backend   dreamvl  (backend_dreamvl_20261013g.Dreamvl; prompt = official chat template around the lmms-eval sqa
                    template of the mmada / lladav backends; official DreamVLProcessor image tokens; Dream logit shift;
                    OUR low-confidence remasking rule, gen 16 / 16 steps / temperature 0)
datasets  sciqa    calib_decode_multi_20261013c.rows_sciqa   (ScienceQA-IMG test 2017, datasets/sciqa_img_standard_v1)
          m3cot    calib_decode_multi_20261013c.rows_m3cot   (M3CoT validation 1108)
          mmbench  calib_decode_multi_20261013c.rows_mmbench (MMBench-dev EN, datasets/mmbench_dev_v1)
          mmstar   calib_decode_multi_20261013e.rows_mmstar  (1500)
          pope     calib_decode_multi_20261013e.rows_pope    (1000 balanced, labels Yes / No)
modes     stock | gated (needs --prior-file; --alpha-scale 0 must reproduce stock: ALPHA0_EQUALS_STOCK) | gray (image
          replaced by a mid-gray image of the item's own size) | prior (content-free step-0 forward, 384x384 gray image,
          question 'N/A', n 'N/A' options, n in 2..5 for MCQ sets / Yes-No for pope; writes the cf json, refuses overwrite)
scoring   calib_decode_multi_20261013e.decode_canvas / run_decode / run_prior (identical record schema to the other
          backends; extra fields n_image_tokens, grid_thw)
official  --official-check N : after the stock decode, run the official diffusion_generate sampler on the same rows
          (lmms-eval MCQ setting 16/16 temp 0 and the HF README setting 128/128 temp 0.1 seed 0) -> *_official.jsonl
Outputs only under outputs/dreamvl_20261013g/ ; resumable (jsonl append + id skip); progress json; flock per output.
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
sys.path.insert(0, str(PROJECT / "experiments"))
import calib_decode_multi_20261013c as c  # noqa: E402
import calib_decode_multi_20261013e as e  # noqa: E402
import backend_dreamvl_20261013g as bd  # noqa: E402
import run_unmask_order as base  # noqa: E402

GEN = base.GEN
OUT_ROOT = PROJECT / "outputs" / "dreamvl_20261013g"
LETTERS = e.LETTERS
YESNO = e.YESNO


# ------------------------------------------------------------------ datasets (c-sets get the gray option of the e-sets)

def _with_gray(rows: list[dict], image_cond: str) -> list[dict]:
    for r in rows:
        r["meta"] = {**r.get("meta", {}), "image_cond": image_cond}
        if image_cond == "gray":
            load = r["image"]
            r["image"] = (lambda f=load: e.gray_like(f()))
    return rows


def rows_sciqa(backend, limit, image_cond):
    return _with_gray(c.rows_sciqa(backend, limit), image_cond)


def rows_m3cot(backend, limit, image_cond):
    return _with_gray(c.rows_m3cot(backend, limit), image_cond)


def rows_mmbench(backend, limit, image_cond):
    return _with_gray(c.rows_mmbench(backend, limit), image_cond)


DATASETS = {"sciqa": rows_sciqa, "m3cot": rows_m3cot, "mmbench": rows_mmbench, "mmstar": e.rows_mmstar, "pope": e.rows_pope}
TASK = {"sciqa": "mcq", "m3cot": "mcq", "mmbench": "mcq", "mmstar": "mcq", "pope": "yesno"}


def check_output_path(q: Path) -> None:
    r = q.resolve()
    root = OUT_ROOT.resolve()
    if r != root and root not in r.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {q}")


def set_labels(be, task: str) -> None:
    """Like calib_decode_multi_20261013e.set_labels but reports the leading-space variants instead of hard-coding ids."""
    e.STATE["task"] = task
    e.STATE["labels"] = list(LETTERS) if task == "mcq" else list(YESNO)
    e.STATE["label_ids"] = e.label_ids_of(be.tokenizer, e.STATE["labels"])
    rep = {lab: {"no_space": be.tokenizer.encode(lab, add_special_tokens=False),
                 "space": be.tokenizer.encode(" " + lab, add_special_tokens=False)} for lab in e.STATE["labels"]}
    print(f"labels={dict(zip(e.STATE['labels'], e.STATE['label_ids']))} mask_id={be.mask_id} variants={rep} "
          f"(using the no-space ids: the answer follows '<|im_start|>assistant\\n' directly)", flush=True)
    if task == "mcq" and e.STATE["label_ids"] != [32, 33, 34, 35, 36]:
        print(f"NOTE letter ids differ from the LLaDA-family ids [32..36]: {e.STATE['label_ids']}", flush=True)


def finished_label(text: str):
    return e.finished_label(text)


def run_official(args, be, rows, path: Path) -> None:
    done = base.load_done(path)
    configs = {"lmms_mcq_16_16_t0": bd.OFFICIAL_LMMS_MCQ, "readme_128_128_t0.1_seed0": bd.OFFICIAL_README}
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            rec = {"id": row["id"], "gold": row["gold"], "n_opt": row["n_opt"], "backend": be.name, "dataset": args.dataset,
                   "mode": "official", "sampler": "diffusion_generate (official generation_utils._sample)"}
            image = row["image"]()
            for name, cfg in configs.items():
                with torch.no_grad():
                    o = be.official_generate(row["text"], image, seed=0, **cfg)
                o["letter"] = finished_label(o["text"])
                rec[name] = o
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"OFFICIAL id={row['id']} gold={row['gold']} " + " ".join(
                f"{k}: letter={rec[k]['letter']} text={rec[k]['text']!r} sec={rec[k]['sec']}" for k in configs), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("dreamvl",), default="dreamvl")
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--mode", choices=("stock", "gated", "gray", "prior"), required=True)
    p.add_argument("--alpha-rule", choices=("spread", "full"), default="spread")
    p.add_argument("--alpha-scale", type=float, default=1.0, help="multiplies alpha_n; 0 reproduces stock")
    p.add_argument("--prior-file", type=Path, default=None)
    p.add_argument("--cuda", type=int, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument("--model-path", default=str(bd.MODEL_DIR))
    p.add_argument("--official-check", type=int, default=0,
                   help="stock mode only: also run the official sampler on the first N rows -> <output stem>_official.jsonl")
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
    e.STATE["mode"] = args.mode
    e.STATE["alpha_scale"] = float(args.alpha_scale)
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
        log_pcf, alpha, pcf, plabels = e.load_priors(args.prior_file)
        if plabels != labels:
            raise RuntimeError(f"prior labels {plabels} != dataset labels {labels}")
        if args.alpha_rule == "full":
            alpha = {n: 1.0 for n in alpha}
        e.STATE["log_pcf"], e.STATE["alpha"] = log_pcf, alpha
        for n in sorted(log_pcf):
            print(f"PRIOR n={n} p_cf={dict(zip(labels, [round(v, 6) for v in pcf[n]]))} alpha_{args.alpha_rule}={alpha[n]:.6f} "
                  f"alpha_scale={args.alpha_scale} shift(-a*logp)={[round(-alpha[n] * args.alpha_scale * v, 4) for v in log_pcf[n]]}",
                  flush=True)
    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    print(f"start backend={args.backend} dataset={args.dataset} mode={args.mode} image_cond={image_cond} cuda={args.cuda} "
          f"gen={GEN} steps={GEN} temperature=0 remasking=low_confidence(run_unmask_order.unmask_one) limit={args.limit} "
          f"model={args.model_path} output={args.output}", flush=True)
    t_start = time.time()
    try:
        rows = None
        if args.mode != "prior":
            rows = DATASETS[args.dataset](args.backend, args.limit, image_cond)
            need = {r["n_opt"] for r in rows}
            if args.mode == "gated" and not need <= set(e.STATE["log_pcf"]):
                raise RuntimeError(f"prior missing for n_opt {sorted(need - set(e.STATE['log_pcf']))}")
            print(f"rows={len(rows)} n_opt_hist={ {n: sum(r['n_opt'] == n for r in rows) for n in sorted(need)} }", flush=True)
            print(f"example prompt id={rows[0]['id']}:\n{rows[0]['text']}", flush=True)
        be = bd.Dreamvl(args)
        set_labels(be, task)
        if rows:
            print(f"example chat prompt:\n{be.prompt_string(rows[0]['text'])}", flush=True)
        torch.cuda.reset_peak_memory_stats(args.cuda)
        if args.mode == "prior":
            e.run_prior(args, be)
        else:
            if args.progress is None:
                raise RuntimeError("--progress required for decode")
            import letter_argmax_traj as traj

            traj.repair_tail(args.output)
            e.run_decode(args, be, rows)
            if args.official_check and args.mode == "stock":
                off = args.output.with_name(args.output.stem + "_official.jsonl")
                run_official(args, be, rows[:args.official_check], off)
        peak = torch.cuda.max_memory_allocated(args.cuda) / 2**30
        resv = torch.cuda.max_memory_reserved(args.cuda) / 2**30
        print(f"PEAK_GPU_MEM allocated_gib={peak:.2f} reserved_gib={resv:.2f} wall_sec={time.time() - t_start:.1f}", flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
