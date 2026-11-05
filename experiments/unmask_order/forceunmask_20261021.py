#!/usr/bin/env python3
"""Intervention 3: force-unmask the answer-slot TOKEN at step 0 vs. the hidden-state write at step 0.

Phenomenon: the writer heads rewrite the (masked) answer slot at every step, so a hidden-state edit at step 0 only
does not change the final letter (LaViDa MMStar: writer step0 beta=8 -> 15/1240 changed; every step -> 1166/1240).
Question: if the answer-slot TOKEN itself is unmasked at step 0 to the item's second-choice letter (the same target
the steer runs use), does the model keep it, and does the rest of the generation follow it?

Items: the SAME items as outputs/causal_full_20261020/<backend> (stock rows of its steer.jsonl: id, first, second,
final_letter, commit_step).  Writer heads / unit_scale are read from that dir (writer_heads.json, calib.json); nothing
there is modified.  Prompts / images come from the MMStar manifest via lavida_causal_20261019 (same text_of /
load_image as the steer runs).  All decodes: greedy, one position per step, GEN = 16, same as the steer runs.

Conditions per item (answer slot = canvas position 0 for all three backends)
  stock                   re-decode (16 forwards; deterministic, checked against the causal_full stock row)
  force_step0             BEFORE the step-0 forward: canvas[0] = token id of the target letter (slot unmasked at
                          step 0).  The 15 other positions follow the normal confidence schedule (15 forwards).  No
                          remasking exists in this decoder, so the slot token is still the target at the end; the
                          record keeps the whole 16-token text and whether the other 15 positions contradict it.
  hidden_step0            writer-head residual write (beta = 8, c = beta / unit_scale) at step 0 only, toward the same
                          target (= causal_full writer/step0/8; the 15/1240 control, reproduced inside this run).
  force_step0_then_remask like force_step0 for step 0 (the model sees the forced slot once and decodes one other
                          position under its influence), then BEFORE the step-1 forward the slot is set back to the
                          mask id; the schedule decides the slot again later (16 forwards in total, like stock).

Outputs (new dir only): outputs/forceunmask_20261021/<backend>/steer.jsonl (one line per item per cond; append +
resume by (id, cond)), summary.md / summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
sys.path.insert(0, str(PROJECT / "experiments"))
import lavida_causal_20261019 as S  # noqa: E402
import p6_causal_20261013f as p6  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "forceunmask_20261021").resolve()
SRC_ROOT = (PROJECT / "outputs" / "causal_full_20261020").resolve()
LETTERS = S.LETTERS
LETTER_IDS = S.LETTER_IDS
GEN = S.GEN
SLOT = 0
BETA = 8.0
CONDS = ("stock", "force_step0", "hidden_step0", "force_step0_then_remask")


def allow_out(path: Path) -> None:
    """Monkeypatch for lavida_causal_20261019.check_out: this script writes only under OUT_ROOT (incl. smoke dirs)."""
    r = Path(path).resolve()
    if r == OUT_ROOT or OUT_ROOT in r.parents:
        return
    raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {path}")


S.check_out = allow_out


# ----------------------------------------------------------------------------- items

def load_source(backend: str) -> tuple[list[dict], dict]:
    """Stock rows of causal_full_20261020/<backend>/steer.jsonl (eligible == 1), writer heads, unit_scale."""
    src = SRC_ROOT / backend
    stock = [r for r in S.jsonl(src / "steer.jsonl") if r["group"] == "stock" and int(r.get("eligible", 0)) == 1]
    ids = [r["id"] for r in stock]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate stock ids in source")
    heads_meta = json.loads((src / "writer_heads.json").read_text(encoding="utf-8"))
    calib = json.loads((src / "calib.json").read_text(encoding="utf-8"))
    meta = {"writer_heads": [tuple(x) for x in heads_meta["writer_heads"]],
            "writer_heads_named": heads_meta["writer_heads_named"],
            "unit_scale": float(calib["groups"]["writer"]["unit_scale_used"]),
            "source_dir": str(src), "n_source_items": len(stock)}
    return stock, meta


# ----------------------------------------------------------------------------- text helpers

def clean_text(ctx, ids) -> str:
    text = ctx["tokenizer"].decode(ids, skip_special_tokens=True).strip()
    if ctx["backend"] == "dreamvl":
        import backend_dreamvl_20261013g as bd

        text = bd.strip_special_text(text)
    return text


def lenient_letter(text: str | None) -> str | None:
    """Leading letter if present, else the first stand-alone A-E anywhere in the text (LLaDA-V writes sentences)."""
    lead = p6.delta.leading_letter(text)
    if lead is not None:
        return lead
    m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", text or "")
    return m.group(1) if m else None


def final_letter_of(backend: str, text: str) -> str | None:
    return lenient_letter(text) if backend == "lladav" else p6.delta.leading_letter(text)


def letters_in(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", text or "")


def token_letter(tok: int) -> str | None:
    return LETTERS[LETTER_IDS.index(int(tok))] if int(tok) in LETTER_IDS else None


# ----------------------------------------------------------------------------- decodes

def forced_decode(ctx, item, target_id: int, remask_at: int | None, tag: str) -> dict:
    """Slot forced to target_id before the step-0 forward; others follow the normal schedule.  If remask_at is not
    None, the slot is set back to the mask id before the forward of that step and decided again by the schedule.
    Loops until no mask is left (15 forwards without remask, 16 with)."""
    mask_id = int(ctx["mask_id"])
    x = p6.fresh(ctx)
    x[0, SLOT] = int(target_id)
    unmask = [-1] * GEN
    unmask[SLOT] = 0
    trace = [{"step": 0, "pos": SLOT, "tok": int(target_id), "forced": True}]
    slot_raws = []
    remasked = False
    step = 0
    while True:
        if remask_at is not None and step == remask_at and not remasked:
            x[0, SLOT] = mask_id
            unmask[SLOT] = -1
            remasked = True
            trace.append({"step": step, "pos": SLOT, "tok": mask_id, "remask": True})
        if not bool(x.eq(mask_id).any().item()):
            break
        if step > GEN + 2:
            raise RuntimeError(f"{tag}: decode did not finish")
        logits, _napp = p6.forward_once(ctx, item, x, None)
        if int(x[0, SLOT].item()) == mask_id:
            slot_raws.append(p6.letters_of(logits[0, SLOT]))
        pos = base.unmask_one(x, logits, mask_id)
        unmask[pos] = step
        trace.append({"step": step, "pos": pos, "tok": int(x[0, pos].item())})
        step += 1
    if min(unmask) < 0:
        raise RuntimeError(f"{tag}: unmask steps {unmask}")
    ids = [int(v) for v in x[0].tolist()]
    text = clean_text(ctx, x[0])
    other_text = clean_text(ctx, torch.tensor(ids[:SLOT] + ids[SLOT + 1:], dtype=torch.long))
    return {"unmask_step": unmask, "order": p6.order_of(unmask), "n_forwards": step, "final_ids": ids,
            "finished_text": text, "other_text": other_text, "slot_token_at_end": ids[SLOT],
            "slot_letter_at_end": token_letter(ids[SLOT]), "slot_unmask_step": unmask[SLOT], "remasked": remasked,
            "slot_raws_when_masked": slot_raws, "trace": trace}


def stock_decode(ctx, item, tag: str) -> dict:
    res = S.full_decode(ctx, item, None, tag=tag)
    ids = res["final_ids"]
    res["other_text"] = clean_text(ctx, torch.tensor(ids[:SLOT] + ids[SLOT + 1:], dtype=torch.long))
    res["slot_token_at_end"] = ids[SLOT]
    res["slot_letter_at_end"] = token_letter(ids[SLOT])
    return res


def text_flags(backend: str, res: dict, target: str, stock_letter: str) -> dict:
    other = res["other_text"]
    letters = letters_in(other)
    slot_letter = res["slot_letter_at_end"]
    return {"other_text_letters": letters,
            "text_has_target": target in letters_in(res["finished_text"]),
            "text_has_stock": stock_letter in letters_in(res["finished_text"]),
            "other_has_stock": stock_letter in letters,
            # contradiction: the non-slot tokens name a letter different from the letter sitting in the slot
            "contradiction": bool(slot_letter is not None and any(l != slot_letter for l in letters))}


# ----------------------------------------------------------------------------- run

def run(args) -> None:
    backend, out_dir = args.backend, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "steer.jsonl"
    S.check_out(output)
    lock = p6.open_locked(output)
    src_rows, meta = load_source(backend)
    if args.limit:
        src_rows = src_rows[: args.limit]
    manifest = {r["id"]: r for r in S.load_rows(backend)}
    missing = [r["id"] for r in src_rows if r["id"] not in manifest]
    if missing:
        raise RuntimeError(f"{len(missing)} source ids not in manifest: {missing[:5]}")
    (out_dir / "meta.json").write_text(json.dumps({**meta, "beta": BETA, "conds": CONDS, "gen": GEN, "slot": SLOT,
                                                   "n_items_this_run": len(src_rows), "backend": backend}, indent=1),
                                       encoding="utf-8")
    ctx = S.load_backend(backend, args.cuda)
    steerer = p6.Steerer(ctx, "writer", heads=meta["writer_heads"])
    unit = meta["unit_scale"]
    cc = BETA / unit
    print(f"items={len(src_rows)} writers={meta['writer_heads_named']} unit_scale={unit:.5f} c={cc:.3f} conds={CONDS}",
          flush=True)
    recs = S.jsonl(output)
    done = {(r["id"], r["cond"]) for r in recs}
    by = {(r["id"], r["cond"]): r for r in recs}
    t0 = time.time(); n_new = 0
    with output.open("a", encoding="utf-8") as handle:
        for src in src_rows:
            need = [cnd for cnd in CONDS if (src["id"], cnd) not in done]
            if not need:
                continue
            t1 = time.time()
            row = manifest[src["id"]]
            n = int(row["n_opt"])
            item = ctx["make_item"](S.text_of(backend, row), S.load_image(row))
            S.dla.disarm()
            target = src["second"]
            first = src["first"]
            target_id = LETTER_IDS[LETTERS.index(target)]
            common = {"id": src["id"], "gold": src["gold"], "n_opt": n, "first": first, "target": target,
                      "stock_letter_causal_full": src["final_letter"], "commit_step_stock": int(src["commit_step"])}
            key_stock = (src["id"], "stock")
            if key_stock in done:
                stock = by[key_stock]
            else:
                res = stock_decode(ctx, item, f"stock {src['id']}")
                letter = final_letter_of(backend, res["finished_text"])
                stock = {**common, "cond": "stock", "stock_letter": letter, "final_letter": letter,
                         "slot_token_at_end": res["slot_token_at_end"], "slot_letter_at_end": res["slot_letter_at_end"],
                         "finished_text": res["finished_text"][:200], "other_text": res["other_text"][:200],
                         "letter_unmask_step": res["letter_unmask_step"], "order": res["order"], "final_ids": res["final_ids"],
                         "step0_raw": res["pos0_letters"][0] if res["pos0_letters"] else None,
                         "matches_causal_full": bool(letter == src["final_letter"] and res["letter_unmask_step"] == int(src["commit_step"])),
                         "changed": 0, "hit": int(letter == target), "target_eq_stock": int(target == letter),
                         **text_flags(backend, res, target, letter)}
                p6.write_rec(handle, stock); done.add(key_stock); by[key_stock] = stock; n_new += 1
            stock_letter = stock["stock_letter"]
            for cnd in CONDS[1:]:
                key = (src["id"], cnd)
                if key in done:
                    continue
                if cnd == "hidden_step0":
                    vec = steerer.vectors(LETTERS.index(first), LETTERS.index(target))
                    raw_setup = steerer.setup(vec, cc)

                    def setup_fn(step, x, _raw=raw_setup):
                        if step == 0:
                            _raw(step, x)

                    res = S.full_decode(ctx, item, setup_fn, tag=f"hidden {src['id']}")
                    ids = res["final_ids"]
                    res["other_text"] = clean_text(ctx, torch.tensor(ids[:SLOT] + ids[SLOT + 1:], dtype=torch.long))
                    res["slot_token_at_end"] = ids[SLOT]
                    res["slot_letter_at_end"] = token_letter(ids[SLOT])
                    extra = {"beta": BETA, "c": round(cc, 4), "heads": meta["writer_heads_named"],
                             "n_hook_applications": res["n_hook_applications"], "letter_unmask_step": res["letter_unmask_step"],
                             "n_forwards": GEN}
                else:
                    res = forced_decode(ctx, item, target_id, 1 if cnd == "force_step0_then_remask" else None, f"{cnd} {src['id']}")
                    extra = {"n_forwards": res["n_forwards"], "slot_unmask_step": res["slot_unmask_step"],
                             "remasked": res["remasked"], "trace": res["trace"], "letter_unmask_step": res["slot_unmask_step"],
                             "slot_raws_when_masked": res["slot_raws_when_masked"]}
                letter = final_letter_of(backend, res["finished_text"])
                rec = {**common, "cond": cnd, "stock_letter": stock_letter, "final_letter": letter,
                       "slot_token_at_end": res["slot_token_at_end"], "slot_letter_at_end": res["slot_letter_at_end"],
                       "slot_kept_target": int(res["slot_letter_at_end"] == target),
                       "finished_text": res["finished_text"][:200], "other_text": res["other_text"][:200],
                       "order": res["order"], "order_same_as_stock": int(res["order"] == stock["order"]),
                       "final_ids": res["final_ids"],
                       "changed": int(letter != stock_letter), "hit": int(letter == target),
                       "target_eq_stock": int(target == stock_letter), **extra,
                       **text_flags(backend, res, target, stock_letter)}
                p6.write_rec(handle, rec); done.add(key); by[key] = rec; n_new += 1
            el = time.time() - t0
            f = by[(src["id"], "force_step0")]; h = by[(src["id"], "hidden_step0")]; m = by[(src["id"], "force_step0_then_remask")]
            print(f"PROGRESS forceunmask {backend} {len([1 for s in src_rows if (s['id'], CONDS[-1]) in done])}/{len(src_rows)} "
                  f"id={src['id']} stock={stock_letter} target={target} commit={stock.get('letter_unmask_step')} "
                  f"force_step0={f['final_letter']} slot={f['slot_letter_at_end']} contra={int(f['contradiction'])} "
                  f"hidden_step0={h['final_letter']} remask={m['final_letter']} remask_slot_step={m['slot_unmask_step']} "
                  f"text={f['finished_text'][:40]!r} sec={time.time() - t1:.1f} rate={n_new / max(el, 1e-6):.3f}/s", flush=True)
    lock.close()
    analyze(out_dir)


# ----------------------------------------------------------------------------- analyze

def analyze(out_dir: Path) -> None:
    recs = S.jsonl(out_dir / "steer.jsonl")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8")) if (out_dir / "meta.json").exists() else {}
    stock = {r["id"]: r for r in recs if r["cond"] == "stock"}
    lines = [f"# force-unmask vs hidden-state write at step 0 ({out_dir.name}, items = causal_full_20261020/{meta.get('backend')})",
             f"n_stock={len(stock)} matches_causal_full={sum(int(r.get('matches_causal_full', 0)) for r in stock.values())}/{len(stock)} "
             f"target==stock_letter: {sum(int(r.get('target_eq_stock', 0)) for r in stock.values())}",
             f"writer heads: {meta.get('writer_heads_named')} unit_scale={meta.get('unit_scale')} beta={meta.get('beta')}", ""]
    lines += ["| cond | n | changed | hit target | slot kept target | contradiction (other tokens name another letter) | order same as stock | n (target!=stock) | changed (target!=stock) | hit (target!=stock) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for cnd in CONDS[1:]:
        rs = [r for r in recs if r["cond"] == cnd]
        ne = [r for r in rs if not int(r.get("target_eq_stock", 0))]
        ch = sum(r["changed"] for r in rs); hit = sum(r["hit"] for r in rs)
        kept = sum(int(r.get("slot_kept_target", 0)) for r in rs)
        contra = sum(int(r.get("contradiction", False)) for r in rs)
        same = sum(int(r.get("order_same_as_stock", 0)) for r in rs)
        summary[cnd] = {"n": len(rs), "changed": ch, "hit": hit, "slot_kept_target": kept, "contradiction": contra,
                        "order_same_as_stock": same, "n_target_ne_stock": len(ne),
                        "changed_target_ne_stock": sum(r["changed"] for r in ne), "hit_target_ne_stock": sum(r["hit"] for r in ne)}
        s = summary[cnd]
        lines.append(f"| {cnd} | {s['n']} | {ch} ({100 * ch / max(1, len(rs)):.1f}%) | {hit} ({100 * hit / max(1, len(rs)):.1f}%) | {kept} | {contra} | {same} | "
                     f"{s['n_target_ne_stock']} | {s['changed_target_ne_stock']} | {s['hit_target_ne_stock']} |")
    rm = [r for r in recs if r["cond"] == "force_step0_then_remask"]
    if rm:
        from collections import Counter

        hist = Counter(r.get("slot_unmask_step") for r in rm)
        back = sum(1 for r in rm if r["final_letter"] == r["stock_letter"])
        lines += ["", f"force_step0_then_remask: slot re-decided at step hist={dict(sorted(hist.items(), key=lambda kv: str(kv[0])))}; "
                      f"final == stock letter: {back}/{len(rm)}; final == target: {sum(r['hit'] for r in rm)}/{len(rm)}"]
        summary["force_step0_then_remask"]["slot_redecide_step_hist"] = {str(k): v for k, v in hist.items()}
    fs = [r for r in recs if r["cond"] == "force_step0"]
    if fs:
        ex = [r for r in fs if r.get("contradiction")][:5]
        lines += ["", "force_step0 contradiction examples (slot letter | finished text):"]
        lines += [f"- {r['id']}: slot={r['slot_letter_at_end']} stock={r['stock_letter']} text={r['finished_text'][:100]!r}" for r in ex]
        lines += ["", f"force_step0 finished_text == slot letter only: {sum(1 for r in fs if r['finished_text'].strip() == (r['slot_letter_at_end'] or '#'))}/{len(fs)}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"table": summary, **meta}, indent=1), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("run", "analyze"), default="run")
    p.add_argument("--backend", choices=("lavida", "lladav", "dreamvl"), default="lavida")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="smoke: only this many items")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    S.DATASET = "mmstar"
    S.set_dims(args.backend)
    args.out_dir = (args.out_dir or OUT_ROOT / args.backend).resolve()
    S.check_out(args.out_dir)
    if args.mode == "run":
        run(args)
    else:
        analyze(args.out_dir)


if __name__ == "__main__":
    main()
