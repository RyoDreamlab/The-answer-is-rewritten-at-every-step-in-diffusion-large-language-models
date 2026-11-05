#!/usr/bin/env python3
"""Run a stock decoder after installing the hysteresis answer-slot rule.

Usage: run_hyst_20261029.py {e|dream|dreamvm|lladav} <decoder args>
HYST_RULE must be set.
"""
from __future__ import annotations

import sys

from release_hyst_20261029 import install

install()
which = sys.argv[1]
sys.argv = [sys.argv[0], *sys.argv[2:]]
if which == "e":
    import calib_decode_multi_20261013e as target
elif which == "dream":
    import calib_decode_dreamvl_20261013g as target
elif which == "dreamvm":
    import calib_decode_vstar_mme_dreamvl_20261022 as dreamvm
    target = dreamvm.g
elif which == "lladav":
    import calib_decode_lladav_e_20261022 as target
else:
    raise SystemExit(f"unknown target {which}")
target.main()
