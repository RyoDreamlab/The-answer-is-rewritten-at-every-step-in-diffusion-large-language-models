#!/usr/bin/env python3
"""Run an existing gated decoder after installing the prior-tie shift.

Usage: run_priortie_20261026.py {e|vstar|dream|dreamvm|lladav|c} <args forwarded to that script>

LLaDA-V POPE uses the lladav entry (calib_decode_lladav_e_20261022). The other cells use the
run_release_20261026.py targets e / vstar / dream / dreamvm / c.
"""
from __future__ import annotations

import sys

from release_priortie_20261026 import install

install()
which = sys.argv[1]
sys.argv = [sys.argv[0], *sys.argv[2:]]
if which == "e":
    import calib_decode_multi_20261013e as target
elif which == "vstar":
    import calib_decode_vstar_mme_20261022 as target
elif which == "dream":
    import calib_decode_dreamvl_20261013g as target
elif which == "dreamvm":
    # Importing the wrapper registers V*/MME loaders and the 20261022 path check.
    import calib_decode_vstar_mme_dreamvl_20261022 as dreamvm
    target = dreamvm.g
elif which == "lladav":
    import calib_decode_lladav_e_20261022 as target
elif which == "c":
    import calib_decode_multi_20261013c as target
else:
    raise SystemExit(f"unknown target {which}")
target.main()
