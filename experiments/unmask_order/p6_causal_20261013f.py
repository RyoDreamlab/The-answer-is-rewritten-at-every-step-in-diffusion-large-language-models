#!/usr/bin/env python3
"""P6 — causal perturbation of the step-0 decision (MMaDA only, ScienceQA-IMG standard manifest).

Reviewer objection: "decide-then-write" (the step-0 restricted argmax predicts the final letter on ~94% of items) is
correlational.  P6 asks whether CHANGING the internal step-0 decision changes the final letter while the unmask order
stays the same.  Structural fact (METHOD_DLA_ATTRACTOR_20261013d §B2): the 16-step decode is memoryless except via the
canvas, so a step-0-only logit edit can only act through the step-0 commit.  Arm 1 therefore perturbs something that
persists across steps (the writer heads' output at the answer position at EVERY step); Arm 2 runs the literal
step-0 activation patch (expected null under §B2) AND a persistent variant; Arm 3 asks whether the written template
(the 15 non-answer canvas tokens) carries the decision.

Arms (``--arm``)
  steer   ``--group {writer_heads,random_heads,writer_comps}``.  At every step t = 0..15, at the answer position only,
          add c·u_G to the output of the heads/components in group G, where
              v   = normalize( gamma_lnf ⊙ (W_U[second] − W_U[first]) )          (d,)  — gradient of
                    (logit_second − logit_first) w.r.t. the final residual under the frozen-scale ln_f readout
                    logits = (gamma ⊙ h · s) W_U^T  (s = rsqrt(mean h² + eps) is a positive scalar, so it does not
                    change the direction); first/second = the two top letters of the STOCK step-0 restricted logits.
              heads:  each head (l, h) can only write into the column space of its slice W_h = W_out^{(l)}[:, h-slice]
                      (d × 128).  a_h = pinv(W_h) v  (so W_h a_h = P_h v is the orthogonal projection of v onto that
                      subspace); the z-space (attn_out input) add for the head is a_h / ||Σ_h P_h v||, so that the total
                      residual-space write Σ_h W_h a_h has UNIT norm.  Writes happen at the heads' own layers
                      (L30 writes are processed by blocks 31–32).
              comps:  add v / |G| to the attn_out output of every component in G (total write = v, unit norm).
          β-units: β = 1 shifts (logit_second − logit_first) at step 0 by ≈ 1.  c = β / unit_scale, where unit_scale
          (logit shift per unit residual-norm write) is the median over the first 20 items of the measured step-0 shift
          per unit c (probe c = 20, linearity checked with c = 40; logits recomputed in float64 from the block-32 residual).  writer_heads and writer_comps use their own
          unit_scale; random_heads uses the WRITER-HEADS unit_scale (same injected residual norm as the writer heads —
          the control is matched on perturbation size, not on logit effect).  Random heads: 3 heads drawn (seed
          20261013) from layers 30–32 excluding every head named in the DLA top/bottom-head lists (same depth as the
          writers, i.e. the stricter control).
  patch   Step-0 activation patching.  Recipients = first --limit manifest items; donors drawn (seed 20261013) from the
          first --pool items with the same option count and a DIFFERENT stock step-0 restricted letter ("differ") or
          the SAME letter ("same" control).  For L in {24, 28, 32}: replace the residual stream at the answer position
          after block L with the donor's step-0 (all-masked canvas) residual.  Variant ``step0``: at step 0 only, then
          decode normally (literal spec; §B2 predicts a null unless the step-0 commit changes — the stock trajectory is
          reused when the canvas after step 0 equals stock's).  Variant ``every``: patch at every step while the answer
          position is masked (persistent; L = 32 is a positive control because it fixes the answer-position logits).
  canvas  Canvas counterfactual for items whose stock commits the letter at step 15: answer position masked, the 15
          other positions carry a template, one forward, read the letter.  Templates: full stock canvas (= stock step
          15), blank (= step 0), EOS run 14 (last / first slot masked), shuffled stock template (identical when all
          tokens are EOS — recorded, no forward), EOT×15, '.'+EOS×14, '\\n'+EOS×14, random vocabulary tokens.

New files only; outputs under outputs/p6_causal_20261013f/.  Resumable (jsonl append, keyed records).
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

import numpy as np
import torch

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import dla_circuit_20261013d as dla  # noqa: E402  (model loading, hooks stay disarmed; frozen-scale readout maths)
import image_key_block as ikb  # noqa: E402
import measure_letter_hidden_delta as delta  # noqa: E402
import run_unmask_order as base  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "p6_causal_20261013f").resolve()
STOCK_FILE = PROJECT / "outputs" / "letter_argmax_traj_20261010" / "mmada.jsonl"
DLA_SUMMARY = PROJECT / "outputs" / "dla_circuit_20261013d" / "summary.json"
LETTERS = "ABCDE"
LETTER_IDS = [32, 33, 34, 35, 36]
N_LAYERS, N_HEADS = 32, 32
WRITER_HEADS = [(32, 27), (30, 28), (32, 4)]
WRITER_COMPS = [30, 31, 32]
RANDOM_LAYERS = [30, 31, 32]
PATCH_LAYERS = [24, 28, 32]
DEFAULT_BETAS = [-4.0, -2.0, 0.0, 2.0, 4.0, 8.0]
SEED = 20261013
N_CALIB = 20
CALIB_PROBES = (20.0, 40.0)
GROUPS = ("writer_heads", "random_heads", "writer_comps")


# ----------------------------------------------------------------------------- hooks (own state; DLA hooks stay disarmed)

class P:
    armed = False
    index = 0
    head_add: dict[int, torch.Tensor] = {}    # layer -> (d,) float32 add in attn_out INPUT space (zeros outside slices)
    comp_add: dict[int, torch.Tensor] = {}    # layer -> (d,) float32 add to attn_out OUTPUT
    patch: dict[int, torch.Tensor] = {}       # layer -> (d,) replacement of the residual after block `layer`
    capture: set[int] = set()
    captured: dict[int, torch.Tensor] = {}
    n_applied = 0


def install_p6(core) -> None:
    blocks = list(core.transformer.blocks)
    if len(blocks) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} blocks, got {len(blocks)}")
    for li, block in enumerate(blocks):
        l = li + 1

        def attn_pre(_m, args, _l=l):
            if not P.armed:
                return None
            a = P.head_add.get(_l)
            if a is None:
                return None
            z = args[0].clone()
            z[0, P.index] = (z[0, P.index].float() + a).to(z.dtype)
            P.n_applied += 1
            return (z,) + tuple(args[1:])

        def attn_post(_m, _args, out, _l=l):
            if not P.armed:
                return None
            a = P.comp_add.get(_l)
            if a is None:
                return None
            out = out.clone()
            out[0, P.index] = (out[0, P.index].float() + a).to(out.dtype)
            P.n_applied += 1
            return out

        def block_post(_m, _args, out, _l=l):
            if not P.armed:
                return None
            is_tuple = isinstance(out, (tuple, list))
            x = out[0] if is_tuple else out
            if _l in P.capture:
                P.captured[_l] = x[0, P.index].detach().float().clone()
            v = P.patch.get(_l)
            if v is None:
                return None
            x = x.clone()
            x[0, P.index] = v.to(x.dtype)
            P.n_applied += 1
            return (x,) + tuple(out[1:]) if is_tuple else x

        block.attn_out.register_forward_pre_hook(attn_pre)
        block.attn_out.register_forward_hook(attn_post)
        block.register_forward_hook(block_post)


def clear_p() -> None:
    P.head_add = {}
    P.comp_add = {}
    P.patch = {}
    P.capture = set()
    P.armed = False


# ----------------------------------------------------------------------------- small helpers

def letters_of(row) -> list[float]:
    return [float(v) for v in row[LETTER_IDS]]


def ranked(vals, n) -> list[int]:
    return sorted(range(n), key=lambda j: (-float(vals[j]), j))


def restricted(vals, n) -> str:
    return LETTERS[ranked(vals, n)[0]]


def order_of(unmask: list[int]) -> list[int]:
    return sorted(range(len(unmask)), key=lambda p: unmask[p])


def levenshtein(a: list[int], b: list[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def load_jsonl(path: Path) -> list[dict]:
    """jsonl loader that allows several records per item id (one per β / condition)."""
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_no} is not json") from exc
    return rows


def fresh(ctx) -> torch.Tensor:
    return torch.full((1, ctx["gen"]), ctx["mask_id"], dtype=torch.long, device=ctx["device"])


def forward_once(ctx, item, x, setup=None):
    """One forward through the P hooks. setup() sets P.head_add / comp_add / patch / capture. Returns logits, n_applied."""
    clear_p()
    P.armed = True
    P.index = int(item["answer_index"])
    P.n_applied = 0
    if setup is not None:
        setup()
    try:
        logits = item["forward"](x)
    finally:
        napp = P.n_applied
        clear_p()
    return logits, napp


def run_steps(ctx, item, x, start: int, setup_fn=None, logits0=None) -> list[dict]:
    """Decode x (in place) from step `start` to the end. setup_fn(step, x) configures the hooks for that forward."""
    recs = []
    for step in range(start, ctx["gen"]):
        masked0 = int(x[0, 0].item()) == int(ctx["mask_id"])
        if step == start and logits0 is not None:
            logits, napp = logits0, 0
        else:
            logits, napp = forward_once(ctx, item, x, (lambda: setup_fn(step, x)) if setup_fn else None)
        raw = letters_of(logits[0, 0]) if masked0 else None
        pos = base.unmask_one(x, logits, ctx["mask_id"])
        rec = {"step": step, "masked0": masked0, "raw": raw, "pos": pos, "n_applied": napp}
        if step == 0:
            rec["canvas_after"] = [int(v) for v in x[0].tolist()]
        recs.append(rec)
    return recs


def finish(ctx, x, recs: list[dict], tag: str) -> dict:
    gen = ctx["gen"]
    unmask = [-1] * gen
    for r in recs:
        unmask[r["pos"]] = r["step"]
    if sorted(unmask) != list(range(gen)):
        raise RuntimeError(f"{tag}: unmask steps {unmask}")
    text = ctx["tokenizer"].decode(x[0], skip_special_tokens=True).strip()
    return {"unmask_step": unmask, "letter_unmask_step": unmask[0], "order": order_of(unmask), "finished_text": text,
            "finished_letter": delta.leading_letter(text), "final_ids": [int(v) for v in x[0].tolist()],
            "pos0_letters": [r["raw"] for r in recs if r["raw"] is not None],
            "canvas_after_step0": recs[0].get("canvas_after") if recs and recs[0]["step"] == 0 else None,
            "n_hook_applications": int(sum(r["n_applied"] for r in recs))}


def full_decode(ctx, item, setup_fn=None, tag="decode") -> dict:
    x = fresh(ctx)
    recs = run_steps(ctx, item, x, 0, setup_fn=setup_fn)
    return finish(ctx, x, recs, tag)


def open_locked(path: Path):
    lock = path.with_suffix(path.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"already running for {path}") from exc
    return lock


def write_rec(handle, rec: dict) -> None:
    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def load_stock_ref() -> dict[str, dict]:
    return {r["id"]: r for r in ikb.load_jsonl(STOCK_FILE)}


def load_prior_top() -> dict[int, str]:
    logp, _ = dla.load_logp("mmada")
    return {n: LETTERS[int(np.argmax(v[:n]))] for n, v in logp.items()}


def load_backend(args):
    rows = ikb.load_manifest(args)
    args.rows = rows
    ctx = dla.load_backend(args)
    install_p6(ctx["core"])
    ctx["backend"] = "mmada"
    return rows, ctx


def stock_from_steer_file(out_dir: Path) -> dict[str, dict]:
    """β = 0 records of the writer_heads run (the shared stock reference), if present."""
    f = out_dir / "steer_writer_heads.jsonl"
    return {r["id"]: r for r in load_jsonl(f) if r.get("beta") == 0}


# ----------------------------------------------------------------------------- steering vectors

def letter_dir(ctx, first: int, second: int) -> torch.Tensor:
    g = ctx["gamma64"] * (ctx["W64"][second] - ctx["W64"][first])
    return g / g.norm()


class Steerer:
    def __init__(self, ctx, name: str, heads: list[tuple[int, int]] | None = None, comps: list[int] | None = None):
        self.ctx, self.name, self.heads, self.comps = ctx, name, heads, comps
        self.cache: dict[tuple[int, int], dict] = {}
        self.W: dict[tuple[int, int], torch.Tensor] = {}
        self.pinv: dict[tuple[int, int], torch.Tensor] = {}
        if heads:
            d = ctx["Wout64"][0].shape[0]
            hs = d // N_HEADS
            for (l, h) in heads:
                Wh = ctx["Wout64"][l - 1][:, h * hs:(h + 1) * hs]           # (d, hs)  out = z @ W^T  -> head out = W_h z_h
                self.W[(l, h)] = Wh
                self.pinv[(l, h)] = torch.linalg.pinv(Wh)                      # (hs, d)
        self.layers = sorted({l for (l, _h) in heads}) if heads else sorted(comps)

    def vectors(self, first: int, second: int) -> dict:
        key = (first, second)
        if key in self.cache:
            return self.cache[key]
        v = letter_dir(self.ctx, first, second)
        d = v.numel()
        dev = self.ctx["device"]
        if self.heads:
            hs = d // N_HEADS
            a = {k: self.pinv[k] @ v for k in self.heads}
            U = sum(self.W[k] @ a[k] for k in self.heads)                    # Σ_h P_h v
            nrm = float(U.norm())
            z_add = {l: torch.zeros(d, dtype=torch.float64) for l in self.layers}
            for (l, h) in self.heads:
                z_add[l][h * hs:(h + 1) * hs] += a[(l, h)] / nrm
            # check: total residual write has unit norm
            tot = sum(self.W[k] @ (a[k] / nrm) for k in self.heads)
            ent = {"kind": "heads", "head_add": {l: z.float().to(dev) for l, z in z_add.items()},
                   "proj_norm": nrm, "cos_total_write_vs_v": float((tot @ v) / (tot.norm() * v.norm())),
                   "unit_norm_check": float(tot.norm()),
                   "per_head_proj_norm": {f"L{l}.H{h}": float((self.W[(l, h)] @ a[(l, h)]).norm()) for (l, h) in self.heads}}
        else:
            share = (v / len(self.comps)).float().to(dev)
            ent = {"kind": "comps", "comp_add": {l: share for l in self.comps}, "proj_norm": 1.0,
                   "cos_total_write_vs_v": 1.0, "unit_norm_check": 1.0}
        self.cache[key] = ent
        return ent

    def setup(self, vec: dict, c: float):
        """Returns a setup_fn(step, x) applying c · unit write at every step."""
        def fn(_step, _x):
            if vec["kind"] == "heads":
                P.head_add = {l: c * z for l, z in vec["head_add"].items()}
            else:
                P.comp_add = {l: c * z for l, z in vec["comp_add"].items()}
        return fn

    def describe(self) -> dict:
        return {"name": self.name, "heads": [f"L{l}.H{h}" for (l, h) in self.heads] if self.heads else None,
                "comps": [f"L{l}.attn" for l in self.comps] if self.comps else None}


def pick_random_heads(rng: np.random.Generator) -> tuple[list[tuple[int, int]], list[str]]:
    excl = {f"L{l}.H{h}" for (l, h) in WRITER_HEADS}
    if DLA_SUMMARY.exists():
        blank = json.loads(DLA_SUMMARY.read_text())["models"]["mmada"]["blank"]
        for n, b in blank.items():
            excl |= {h for h, _ in b.get("top_heads", [])} | {h for h, _ in b.get("bottom_heads", [])}
    cands = [(l, h) for l in RANDOM_LAYERS for h in range(N_HEADS) if f"L{l}.H{h}" not in excl]
    idx = rng.choice(len(cands), size=3, replace=False)
    chosen = sorted(cands[int(i)] for i in idx)
    return chosen, sorted(excl)


def build_steerers(ctx) -> tuple[dict[str, Steerer], dict]:
    rng = np.random.default_rng(SEED)
    rand_heads, excluded = pick_random_heads(rng)
    st = {"writer_heads": Steerer(ctx, "writer_heads", heads=WRITER_HEADS),
          "random_heads": Steerer(ctx, "random_heads", heads=rand_heads),
          "writer_comps": Steerer(ctx, "writer_comps", comps=WRITER_COMPS)}
    meta = {"random_heads": [f"L{l}.H{h}" for (l, h) in rand_heads], "random_excluded": excluded,
            "random_layers": RANDOM_LAYERS, "seed": SEED}
    return st, meta


def exact_letters_from_capture(ctx) -> list[float]:
    """Letter logits recomputed in float64 from the captured final residual (block 32 output) through ln_f + head.
    The model's own logits are bf16 (resolution 0.125 at |logit| ~ 20), too coarse for a slope calibration."""
    h = P.captured[32].to(torch.float64).cpu()
    return [float(v) for v in dla.exact_letters(h, ctx["gamma64"], ctx["W64"], ctx["eps"])]


def calibrate(ctx, rows, steerers: dict[str, Steerer]) -> dict:
    """unit_scale[group] = median over items of  Δ(logit_second − logit_first at step 0) / c  for probe c = CALIB_PROBES[0],
    with the step-0 letter logits recomputed in float64 from the captured final residual."""
    per = {g: {str(c): [] for c in CALIB_PROBES} for g in steerers}
    geo = {g: {"proj_norm": [], "cos": []} for g in steerers}
    ids = []
    bf16_vs_exact = []

    def cap():
        P.capture = {32}

    for row in rows:
        n = dla.n_choices(row)
        item = ctx["real_item"](row)
        x = fresh(ctx)
        logits0, _ = forward_once(ctx, item, x, cap)
        raw0 = letters_of(logits0[0, 0])
        ex0 = exact_letters_from_capture(ctx)
        bf16_vs_exact.append(max(abs(a - b) for a, b in zip(raw0, ex0)))
        first, second = ranked(raw0, n)[:2]            # ranking by the model's own (bf16) logits, as in the decode
        diff0 = ex0[second] - ex0[first]
        for g, st in steerers.items():
            vec = st.vectors(first, second)
            geo[g]["proj_norm"].append(vec["proj_norm"]); geo[g]["cos"].append(vec["cos_total_write_vs_v"])
            for c in CALIB_PROBES:
                def setup(_st=st, _vec=vec, _c=c):
                    _st.setup(_vec, _c)(0, x)
                    P.capture = {32}
                _lg, napp = forward_once(ctx, item, x, setup)
                if napp != len(st.layers):
                    raise RuntimeError(f"calibration hook applications {napp} != {len(st.layers)}")
                r = exact_letters_from_capture(ctx)
                per[g][str(c)].append((r[second] - r[first] - diff0) / c)
        ids.append(row["id"])
    out = {"n_items": len(ids), "ids": ids, "probes": list(CALIB_PROBES), "groups": {},
           "bf16_logits_vs_float64_exact_max_abs_diff": float(max(bf16_vs_exact)),
           "readout": "step-0 letter logits for the slopes recomputed in float64 from the captured block-32 residual through ln_f (exact, not frozen-scale)"}
    for g in steerers:
        s10 = np.array(per[g][str(CALIB_PROBES[0])]); s20 = np.array(per[g][str(CALIB_PROBES[1])])
        out["groups"][g] = {"unit_scale_own": float(np.median(s10)), "slope_p1_median": float(np.median(s10)),
                            "slope_p1_iqr": [float(np.percentile(s10, 25)), float(np.percentile(s10, 75))],
                            "slope_p2_median": float(np.median(s20)),
                            "linearity_median_ratio_p2_over_p1": float(np.median(s20 / np.where(np.abs(s10) > 1e-9, s10, np.nan))) if np.any(np.abs(s10) > 1e-9) else None,
                            "slopes_p1": s10.tolist(), "slopes_p2": s20.tolist(),
                            "proj_norm_mean": float(np.mean(geo[g]["proj_norm"])), "cos_total_write_vs_v_mean": float(np.mean(geo[g]["cos"])),
                            **steerers[g].describe()}
    wh = out["groups"]["writer_heads"]["unit_scale_own"]
    for g in steerers:
        out["groups"][g]["unit_scale_used"] = wh if g == "random_heads" else out["groups"][g]["unit_scale_own"]
        out["groups"][g]["unit_scale_source"] = "writer_heads (matched residual norm)" if g == "random_heads" else "own"
    out["formula"] = ("v = normalize(gamma_lnf * (W_U[second] - W_U[first])); heads: z-add_h = pinv(W_h) v / ||sum_h P_h v||; "
                      "comps: add v/|G| at each attn_out output; c = beta / unit_scale; unit_scale = median_items[(Δ(l2-l1) at step 0)/c], c=20, float64 exact readout")
    return out


# ----------------------------------------------------------------------------- arm 1: steer

def run_steer(args):
    out_dir = args.out_dir
    group = args.group
    output = out_dir / f"steer_{group}.jsonl"
    lock = open_locked(output)
    ikb.repair_tail(output)
    done = {(r["id"], f"{r['beta']:g}") for r in load_jsonl(output)}
    rows, ctx = load_backend(args)
    rows = rows[: args.limit]
    betas = args.betas
    steerers, meta = build_steerers(ctx)
    st = steerers[group]
    calib_path = out_dir / "calib_steer.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        print(f"CALIB loaded {calib_path}", flush=True)
    else:
        t0 = time.time()
        calib = calibrate(ctx, rows[: min(N_CALIB, len(rows))], steerers)
        calib.update(meta)
        calib["elapsed_sec"] = round(time.time() - t0, 1)
        calib_path.write_text(json.dumps(calib, indent=1), encoding="utf-8")
        print(f"CALIB wrote {calib_path}", flush=True)
    for g, e in calib["groups"].items():
        print(f"CALIB {g}: unit_scale_own={e['unit_scale_own']:.5f} used={e['unit_scale_used']:.5f} "
              f"iqr={[round(v, 5) for v in e['slope_p1_iqr']]} lin_p2/p1={e['linearity_median_ratio_p2_over_p1']} "
              f"proj_norm={e['proj_norm_mean']:.3f} cos={e['cos_total_write_vs_v_mean']:.3f} "
              f"{e['heads'] or e['comps']}", flush=True)
    unit_scale = float(calib["groups"][group]["unit_scale_used"])
    if not (unit_scale > 0):
        raise RuntimeError(f"unit_scale {unit_scale} must be positive")
    stock_ref = load_stock_ref()
    prior_top = load_prior_top()
    shared_stock = stock_from_steer_file(out_dir) if group != "writer_heads" else {}
    beta0_ok = beta0_cmp = 0
    t0 = time.time(); n_dec = 0
    print(f"start steer group={group} {st.describe()} betas={betas} unit_scale={unit_scale:.5f} n={len(rows)} "
          f"done_recs={len(done)} shared_stock={len(shared_stock)}", flush=True)
    with output.open("a", encoding="utf-8") as handle:
        for k, row in enumerate(rows):
            todo = [b for b in betas if (row["id"], f"{b:g}") not in done]
            if not todo:
                continue
            n = dla.n_choices(row)
            item = ctx["real_item"](row)
            zero_vec = {"kind": "heads" if st.heads else "comps",
                        ("head_add" if st.heads else "comp_add"): {l: torch.zeros(ctx["W64"].shape[1], device=ctx["device"]) for l in st.layers}}
            # ---- stock (β = 0 through the hook path with zero adds; the shared reference)
            sh = shared_stock.get(row["id"])
            if sh is not None:
                stock = {kk: sh[kk] for kk in ("unmask_step", "letter_unmask_step", "order", "finished_text", "finished_letter",
                                               "final_ids", "pos0_letters", "canvas_after_step0", "n_hook_applications")}
                stock_source = "steer_writer_heads"
            else:
                stock = full_decode(ctx, item, st.setup(zero_vec, 0.0), tag="stock"); n_dec += 1
                stock_source = "computed"
            raw0 = stock["pos0_letters"][0]
            first, second = ranked(raw0, n)[:2]
            diff0 = raw0[second] - raw0[first]
            ref = stock_ref.get(row["id"])
            if ref is not None:
                beta0_cmp += 1
                beta0_ok += int(ref["finished_letter"] == stock["finished_letter"] and ref["letter_unmask_step"] == stock["letter_unmask_step"])
            vec = st.vectors(first, second)
            common = {"id": row["id"], "gold": ikb.gold_of(row), "n_opt": n, "group": group, "heads": [f"L{l}.H{h}" for (l, h) in st.heads] if st.heads else None,
                      "comps": [f"L{l}.attn" for l in st.comps] if st.comps else None, "unit_scale": unit_scale,
                      "first": LETTERS[first], "second": LETTERS[second], "prior_top": prior_top[n],
                      "stock_finished_letter": stock["finished_letter"], "stock_letter_unmask_step": stock["letter_unmask_step"],
                      "stock_order": stock["order"], "stock_step0_letters": raw0, "stock_step0_diff_second_minus_first": diff0,
                      "stock_source": stock_source, "proj_norm": vec["proj_norm"], "cos_total_write_vs_v": vec["cos_total_write_vs_v"],
                      "ref_stock_finished_letter": ref.get("finished_letter") if ref else None,
                      "ref_stock_letter_unmask_step": ref.get("letter_unmask_step") if ref else None}
            for b in todo:
                if b == 0:
                    res = stock
                    c = 0.0
                else:
                    c = b / unit_scale
                    res = full_decode(ctx, item, st.setup(vec, c), tag=f"{group} beta={b}"); n_dec += 1
                s0 = res["pos0_letters"][0]
                final = res["finished_letter"]
                rec = {**common, "beta": float(b), "c_raw": c, "finished_letter": final, "finished_text": res["finished_text"],
                       "flipped": final != stock["finished_letter"], "final_is_second": final == LETTERS[second],
                       "final_is_first": final == LETTERS[first], "final_is_prior_top": final == prior_top[n],
                       "final_is_other": final not in (LETTERS[first], LETTERS[second]),
                       "letter_unmask_step": res["letter_unmask_step"], "unmask_step": res["unmask_step"], "order": res["order"],
                       "order_edit_distance": levenshtein(res["order"], stock["order"]),
                       "order_identical": res["order"] == stock["order"],
                       "step0_letters": s0, "step0_restricted": restricted(s0, n),
                       "step0_shift_second_minus_first": (s0[second] - s0[first]) - diff0,
                       "step0_flipped": restricted(s0, n) != LETTERS[first],
                       "n_hook_applications": res["n_hook_applications"], "final_ids": res["final_ids"],
                       "canvas_after_step0": res["canvas_after_step0"], "pos0_letters": res["pos0_letters"],
                       "correct": final == ikb.gold_of(row)}
                write_rec(handle, rec)
                done.add((row["id"], f"{b:g}"))
            el = time.time() - t0
            base.write_progress(args.progress, {"arm": "steer", "group": group, "n_done_items": k + 1, "n_total": len(rows),
                                                "elapsed_sec": round(el, 1), "n_decodes": n_dec,
                                                "sec_per_decode": round(el / n_dec, 3) if n_dec else None,
                                                "eta_sec": round((len(rows) - k - 1) * el / (k + 1)),
                                                "beta0_equals_stock": f"{beta0_ok}/{beta0_cmp}"})
            if (k + 1) % 10 == 0 or k + 1 == len(rows):
                print(f"PROGRESS {k + 1}/{len(rows)} elapsed={el:.0f}s decodes={n_dec} sec/decode={el / max(n_dec, 1):.2f} "
                      f"beta0_equals_stock={beta0_ok}/{beta0_cmp}", flush=True)
    print(f"BETA0_EQUALS_STOCK={beta0_ok == beta0_cmp and beta0_cmp > 0} ({beta0_ok}/{beta0_cmp})", flush=True)
    print(f"DONE steer {group} n={len(rows)} decodes={n_dec} elapsed={time.time() - t0:.0f}s "
          f"peak_gpu_mem={torch.cuda.max_memory_allocated(ctx['device']) / 2 ** 30:.2f} GiB", flush=True)
    lock.close()


# ----------------------------------------------------------------------------- arm 2: patch

def run_patch(args):
    out_dir = args.out_dir
    output = out_dir / "patch.jsonl"
    lock = open_locked(output)
    ikb.repair_tail(output)
    done = {(r["id"], r["cond"], r["layer"], r["variant"]) for r in load_jsonl(output)}
    rows, ctx = load_backend(args)
    pool = rows[: args.pool]
    recips = rows[: args.limit]
    # ---- phase A: step-0 residuals + step-0 letters for the pool (one forward each)
    t0 = time.time()
    cache: dict[str, dict] = {}
    for row in pool:
        n = dla.n_choices(row)
        item = ctx["real_item"](row)
        x = fresh(ctx)
        def cap():
            P.capture = set(PATCH_LAYERS)
        logits, _ = forward_once(ctx, item, x, cap)
        raw0 = letters_of(logits[0, 0])
        cache[row["id"]] = {"n": n, "raw0": raw0, "letter": restricted(raw0, n),
                            "resid": {L: P.captured[L].cpu() for L in PATCH_LAYERS}}
        P.captured = {}
    print(f"POOL {len(pool)} step-0 forwards in {time.time() - t0:.0f}s; letters="
          f"{ {l: sum(c['letter'] == l for c in cache.values()) for l in LETTERS} }", flush=True)
    # ---- pairing (deterministic, persisted)
    pairs_path = out_dir / "patch_pairs.json"
    if pairs_path.exists():
        pairs = json.loads(pairs_path.read_text())
    else:
        rng = np.random.default_rng(SEED)
        pairs = {}
        pool_ids = [r["id"] for r in pool]
        for row in recips:
            i = row["id"]; n = cache[i]["n"]; li = cache[i]["letter"]
            diff = [j for j in pool_ids if j != i and cache[j]["n"] == n and cache[j]["letter"] != li]
            same = [j for j in pool_ids if j != i and cache[j]["n"] == n and cache[j]["letter"] == li]
            pairs[i] = {"differ": str(rng.choice(diff)) if diff else None, "same": str(rng.choice(same)) if same else None}
        pairs_path.write_text(json.dumps({"seed": SEED, "pool": len(pool), "pairs": pairs}, indent=1), encoding="utf-8")
        pairs = {"seed": SEED, "pool": len(pool), "pairs": pairs}
    pairs = pairs["pairs"]
    shared_stock = stock_from_steer_file(out_dir)
    stock_ref = load_stock_ref()
    conds = ("differ", "same")
    variants = ("step0", "every")
    n_dec = 0; n_fwd_short = 0; t0 = time.time(); beta0_ok = beta0_cmp = 0
    print(f"start patch recipients={len(recips)} pool={len(pool)} layers={PATCH_LAYERS} done_recs={len(done)} "
          f"shared_stock={len(shared_stock)}", flush=True)
    with output.open("a", encoding="utf-8") as handle:
        for k, row in enumerate(recips):
            i = row["id"]
            todo = [(c, L, v) for c in conds for L in PATCH_LAYERS for v in variants
                    if pairs[i][c] is not None and (i, c, L, v) not in done]
            if not todo:
                continue
            n = cache[i]["n"]
            item = ctx["real_item"](row)
            sh = shared_stock.get(i)
            if sh is not None:
                stock = {kk: sh[kk] for kk in ("unmask_step", "letter_unmask_step", "order", "finished_text", "finished_letter",
                                               "final_ids", "pos0_letters", "canvas_after_step0")}
                stock_source = "steer_writer_heads"
            else:
                stock = full_decode(ctx, item, None, tag="stock"); n_dec += 1; stock_source = "computed"
            ref = stock_ref.get(i)
            if ref is not None:
                beta0_cmp += 1
                beta0_ok += int(ref["finished_letter"] == stock["finished_letter"] and ref["letter_unmask_step"] == stock["letter_unmask_step"])
            for (cond, L, variant) in todo:
                j = pairs[i][cond]
                donor = cache[j]["resid"][L].to(ctx["device"])
                if variant == "step0":
                    x = fresh(ctx)
                    def s0():
                        P.patch = {L: donor}
                    lg, napp = forward_once(ctx, item, x, s0)
                    if napp != 1:
                        raise RuntimeError(f"patch hook applied {napp} times")
                    # step 0 with the patched logits, done by hand so the canvas can be compared with stock
                    s0_letters = letters_of(lg[0, 0])
                    pos = base.unmask_one(x, lg, ctx["mask_id"])
                    rec0 = {"step": 0, "masked0": True, "raw": s0_letters, "pos": pos, "n_applied": napp,
                            "canvas_after": [int(v) for v in x[0].tolist()]}
                    same = rec0["canvas_after"] == stock["canvas_after_step0"]
                    if same:
                        res = {**stock, "n_hook_applications": napp}
                        branch = False; n_fwd_short += 1
                    else:
                        recs = [rec0] + run_steps(ctx, item, x, 1)
                        res = finish(ctx, x, recs, "patch step0"); res["n_hook_applications"] = napp
                        branch = True; n_dec += 1
                else:
                    def every(step, xx, _L=L, _d=donor):
                        if int(xx[0, 0].item()) == int(ctx["mask_id"]):
                            P.patch = {_L: _d}
                    res = full_decode(ctx, item, every, tag="patch every"); n_dec += 1
                    s0_letters = res["pos0_letters"][0]
                    branch = res["final_ids"] != stock["final_ids"]
                final = res["finished_letter"]
                rec = {"id": i, "gold": ikb.gold_of(row), "n_opt": n, "donor": j, "cond": cond, "layer": L, "variant": variant,
                       "recipient_step0_letter": cache[i]["letter"], "donor_step0_letter": cache[j]["letter"],
                       "donor_resid_norm": float(cache[j]["resid"][L].norm()), "recipient_resid_norm": float(cache[i]["resid"][L].norm()),
                       "stock_finished_letter": stock["finished_letter"], "stock_letter_unmask_step": stock["letter_unmask_step"],
                       "stock_source": stock_source, "finished_letter": final, "finished_text": res["finished_text"],
                       "final_eq_donor_step0": final == cache[j]["letter"], "final_eq_stock": final == stock["finished_letter"],
                       "changed": final != stock["finished_letter"], "branch": branch,
                       "step0_patched_letters": s0_letters, "step0_patched_restricted": restricted(s0_letters, n),
                       "step0_patched_eq_donor": restricted(s0_letters, n) == cache[j]["letter"],
                       "committed_at_step0": res["unmask_step"][0] == 0,
                       "letter_unmask_step": res["letter_unmask_step"], "order": res["order"],
                       "order_edit_distance": levenshtein(res["order"], stock["order"]),
                       "n_hook_applications": res.get("n_hook_applications"), "final_ids": res["final_ids"], "correct": final == ikb.gold_of(row)}
                write_rec(handle, rec)
                done.add((i, cond, L, variant))
            el = time.time() - t0
            base.write_progress(args.progress, {"arm": "patch", "n_done_items": k + 1, "n_total": len(recips), "elapsed_sec": round(el, 1),
                                                "n_decodes": n_dec, "n_shortcut": n_fwd_short, "sec_per_decode": round(el / n_dec, 3) if n_dec else None,
                                                "eta_sec": round((len(recips) - k - 1) * el / (k + 1)), "beta0_equals_stock": f"{beta0_ok}/{beta0_cmp}"})
            if (k + 1) % 10 == 0 or k + 1 == len(recips):
                print(f"PROGRESS {k + 1}/{len(recips)} elapsed={el:.0f}s decodes={n_dec} shortcut={n_fwd_short} "
                      f"stock_equals_ref={beta0_ok}/{beta0_cmp}", flush=True)
    print(f"BETA0_EQUALS_STOCK={beta0_ok == beta0_cmp and beta0_cmp > 0} ({beta0_ok}/{beta0_cmp})", flush=True)
    print(f"DONE patch n={len(recips)} decodes={n_dec} elapsed={time.time() - t0:.0f}s "
          f"peak_gpu_mem={torch.cuda.max_memory_allocated(ctx['device']) / 2 ** 30:.2f} GiB", flush=True)
    lock.close()


# ----------------------------------------------------------------------------- arm 3: canvas

def run_canvas(args):
    out_dir = args.out_dir
    output = out_dir / "canvas.jsonl"
    lock = open_locked(output)
    ikb.repair_tail(output)
    done = {r["id"] for r in load_jsonl(output)}
    rows, ctx = load_backend(args)
    rows = rows[: args.limit]
    tok, mask_id, gen = ctx["tokenizer"], ctx["mask_id"], ctx["gen"]
    sys.path.insert(0, str(PROJECT / "experiments"))
    import evdlc_mmada as mm
    eos, eot = int(mm.EOS_ID), int(getattr(mm, "EOT_ID", 126348))

    def single(s: str) -> int:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{s!r} tokenizes to {ids}")
        return int(ids[0])

    period, newline = single("."), single("\n")
    shared_stock = stock_from_steer_file(out_dir)
    stock_ref = load_stock_ref()
    rng = np.random.default_rng(SEED)
    vocab = int(ctx["core"].transformer.wte.weight.shape[0])
    t0 = time.time(); n_dec = 0; n_fwd = 0; beta0_ok = beta0_cmp = 0
    print(f"start canvas n={len(rows)} done={len(done)} eos={eos} eot={eot} period={period} newline={newline}", flush=True)
    with output.open("a", encoding="utf-8") as handle:
        for k, row in enumerate(rows):
            if row["id"] in done:
                continue
            n = dla.n_choices(row)
            item = ctx["real_item"](row)
            sh = shared_stock.get(row["id"])
            if sh is not None:
                stock = {kk: sh[kk] for kk in ("letter_unmask_step", "finished_letter", "final_ids", "pos0_letters", "order")}
                stock_source = "steer_writer_heads"
            else:
                stock = full_decode(ctx, item, None, tag="stock"); n_dec += 1; stock_source = "computed"
            ref = stock_ref.get(row["id"])
            if ref is not None:
                beta0_cmp += 1
                beta0_ok += int(ref["finished_letter"] == stock["finished_letter"] and ref["letter_unmask_step"] == stock["letter_unmask_step"])
            step0_letter = restricted(stock["pos0_letters"][0], n)
            rec = {"id": row["id"], "gold": ikb.gold_of(row), "n_opt": n, "stock_finished_letter": stock["finished_letter"],
                   "stock_letter_unmask_step": stock["letter_unmask_step"], "step0_letter": step0_letter, "stock_source": stock_source,
                   "eligible": stock["letter_unmask_step"] == gen - 1, "templates": {}}
            if rec["eligible"]:
                fin = stock["final_ids"]
                tmpl = fin[1:]
                templates = {
                    "full_stock": list(tmpl),
                    "blank": [mask_id] * (gen - 1),
                    "eos_run14_last_masked": list(tmpl[:-1]) + [mask_id],
                    "eos_run14_first_masked": [mask_id] + list(tmpl[1:]),
                    "eos_run8_first_half_masked": [mask_id] * 7 + list(tmpl[7:]),
                    "shuffled_stock": [tmpl[int(i)] for i in rng.permutation(gen - 1)],
                    "eot_x15": [eot] * (gen - 1),
                    "period_then_eos": [period] + [eos] * (gen - 2),
                    "newline_then_eos": [newline] + [eos] * (gen - 2),
                    "random_tokens": [int(v) for v in rng.integers(1000, min(vocab, 100000), size=gen - 1)],
                }
                templates["random_tokens"] = [v if v not in LETTER_IDS else v + 10 for v in templates["random_tokens"]]
                for name, t in templates.items():
                    ent = {"template": t, "identical_to_full_stock": t == list(tmpl)}
                    if name != "full_stock" and ent["identical_to_full_stock"]:
                        ent["skipped"] = True
                        rec["templates"][name] = ent
                        continue
                    x = fresh(ctx)
                    x[0, 1:] = torch.tensor(t, device=x.device)
                    logits, _ = forward_once(ctx, item, x); n_fwd += 1
                    row0 = logits[0, 0].detach().float()
                    ls = letters_of(row0)
                    rk = ranked(ls, n)
                    full_arg = int(torch.argmax(row0).item())
                    ent.update({"letters": ls, "restricted": LETTERS[rk[0]], "margin": ls[rk[0]] - ls[rk[1]] if n > 1 else None,
                                "full_argmax_id": full_arg, "full_argmax_piece": base.piece(tok, full_arg, mask_id),
                                "full_argmax_is_letter": full_arg in LETTER_IDS[:n],
                                "eq_stock_final": LETTERS[rk[0]] == stock["finished_letter"], "eq_step0": LETTERS[rk[0]] == step0_letter,
                                "correct": LETTERS[rk[0]] == ikb.gold_of(row)})
                    rec["templates"][name] = ent
                if rec["templates"]["full_stock"]["restricted"] != stock["finished_letter"]:
                    rec["full_stock_mismatch"] = True
            write_rec(handle, rec)
            done.add(row["id"])
            el = time.time() - t0
            base.write_progress(args.progress, {"arm": "canvas", "n_done_items": len(done), "n_total": len(rows), "elapsed_sec": round(el, 1),
                                                "n_decodes": n_dec, "n_forwards": n_fwd, "eta_sec": round((len(rows) - len(done)) * el / max(k + 1, 1)),
                                                "beta0_equals_stock": f"{beta0_ok}/{beta0_cmp}"})
            if len(done) % 25 == 0 or len(done) == len(rows):
                print(f"PROGRESS {len(done)}/{len(rows)} elapsed={el:.0f}s decodes={n_dec} forwards={n_fwd} stock_equals_ref={beta0_ok}/{beta0_cmp}", flush=True)
    print(f"BETA0_EQUALS_STOCK={beta0_ok == beta0_cmp and beta0_cmp > 0} ({beta0_ok}/{beta0_cmp})", flush=True)
    print(f"DONE canvas n={len(done)} elapsed={time.time() - t0:.0f}s peak_gpu_mem={torch.cuda.max_memory_allocated(ctx['device']) / 2 ** 30:.2f} GiB", flush=True)
    lock.close()


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("steer", "patch", "canvas"), required=True)
    p.add_argument("--group", choices=GROUPS, default="writer_heads")
    p.add_argument("--betas", default=",".join(f"{b:g}" for b in DEFAULT_BETAS))
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--pool", type=int, default=400)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument("--model-path", default=str(PROJECT / "checkpoints/MMaDA-8B-Base"))
    p.add_argument("--manifest", default=str(PROJECT / "datasets/sciqa_img_standard_v1/manifest.json"))
    p.add_argument("--data-dir", default=str(PROJECT / "datasets/sciqa_img_standard_v1"))
    p.add_argument("--vq-model", default=str(PROJECT / "checkpoints/magvitv2"))
    p.add_argument("--mmada-repo", default=str(PROJECT / "repos/MMaDA"))
    args = p.parse_args()
    args.backend = "mmada"
    args.betas = [float(b) for b in args.betas.split(",") if b.strip()]
    if 0.0 not in args.betas:
        args.betas = [0.0] + args.betas
    args.betas = sorted(set(args.betas))
    args.out_dir = args.out_dir.resolve()
    if args.out_dir != OUT_ROOT and OUT_ROOT not in args.out_dir.parents:
        raise RuntimeError(f"refusing to write outside {OUT_ROOT}: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.progress is None:
        suffix = f"_{args.group}" if args.arm == "steer" else ""
        args.progress = args.out_dir / f"progress_{args.arm}{suffix}.json"
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")
    if base.GEN != 16:
        raise RuntimeError(f"unexpected gen {base.GEN}")
    if args.pool < args.limit:
        args.pool = args.limit
    {"steer": run_steer, "patch": run_patch, "canvas": run_canvas}[args.arm](args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
