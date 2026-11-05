#!/usr/bin/env python3
"""Gated (bias-strength-adaptive) content-free calibration, FULL ScienceQA-IMG test decode (gen 16).

Thin wrapper over calib_decode.py (unchanged). The only difference: at every step while canvas
position 0 is still masked, the n option-letter logits are shifted by

    - alpha_n * log p_cf(y),   alpha_n = 1 - p_min / p_max  over the item's n letters of the global
                                content-free prior (outputs/calib_vote_20261012/cf_{model}.jsonl, cond a_global)

i.e. rule 'apriori_spread' of calib_gated_offline.py (Stage-1 cross-fit: LaViDa 1456 = +8, MMaDA 1071 = +7 vs
step-0 raw). No fitted constant, no gold. calib_decode.py uses alpha = 1 (full strength); here alpha_n is folded
into the stored log-prior so the decode path is byte-identical to calib_decode.py.
Outputs go to outputs/calib_gated_decode_20261013/. --no-calib reproduces the stock path (sanity).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import calib_decode as cd  # noqa: E402

OUT_ROOT = (PROJECT / "outputs" / "calib_gated_decode_20261013").resolve()
_orig_load_priors = cd.load_priors


def load_priors_gated(backend: str) -> dict[int, list[float]]:
    lp = _orig_load_priors(backend)
    out, table = {}, {}
    for n, l in sorted(lp.items()):
        alpha = 1.0 - math.exp(-(max(l) - min(l)))  # = 1 - p_min/p_max
        out[n] = [alpha * v for v in l]
        table[n] = {"alpha_1_minus_pmin_over_pmax": alpha, "log_p_cf": l,
                    "p_cf": [math.exp(v) for v in l], "shift": out[n]}
        print(f"GATE backend={backend} n={n} alpha={alpha:.6f} p_cf={[round(math.exp(v), 4) for v in l]} "
              f"shift(-alpha*log_p)={[round(-v, 4) for v in out[n]]}", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    side = OUT_ROOT / f"alpha_table_{backend}.json"
    if not side.exists():
        side.write_text(json.dumps({"rule": "apriori_spread: alpha_n = 1 - p_min/p_max", "backend": backend,
                                    "by_n_opt": table}, indent=1), encoding="utf-8")
    return out


cd.load_priors = load_priors_gated
cd.OUT_ROOT = OUT_ROOT

if __name__ == "__main__":
    cd.main()
