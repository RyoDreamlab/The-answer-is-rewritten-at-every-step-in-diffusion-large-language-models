#!/usr/bin/env python3
"""M3CoT validation (1108) calibrated decode, gen 16 — thin wrapper over calib_decode_m3cot.py (unchanged).

Adds --alpha-rule:
  spread  alpha_n = 1 - p_min/p_max  (gated rule; identical to the original --mode calib)
  full    alpha_n = 1                (full-strength content-free calibration, Zhao et al. 2021, as in the
                                      ScienceQA calib_decode_20261012 run)
Outputs are written under outputs/m3cot_20261013/ instead of outputs/m3cot_20261012/.
All other flags are passed through (--backend --cuda --mode calib --prior-file --output --progress ...).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0, str(PROJECT / "experiments" / "unmask_order"))
import calib_decode_m3cot as m  # noqa: E402

RULE = "spread"
if "--alpha-rule" in sys.argv:
    i = sys.argv.index("--alpha-rule")
    RULE = sys.argv[i + 1]
    del sys.argv[i:i + 2]
if RULE not in ("spread", "full"):
    raise SystemExit(f"--alpha-rule must be spread|full, got {RULE}")

_orig_load_priors = m.load_priors


def load_priors_rule(path: Path):
    log_pcf, alpha = _orig_load_priors(path)
    if RULE == "full":
        alpha = {n: 1.0 for n in alpha}
    print(f"ALPHA_RULE {RULE} alpha_by_n={ {n: round(a, 6) for n, a in sorted(alpha.items())} }", flush=True)
    return log_pcf, alpha


m.load_priors = load_priors_rule
m.OUT_ROOT = (PROJECT / "outputs" / "m3cot_20261013").resolve()

if __name__ == "__main__":
    m.main()
