#!/usr/bin/env python3
"""Run a stock decoder after installing the commit-step margin shift.

Usage: run_commitmargin_20261027.py {dream|e} <args forwarded to that script>
"""
from __future__ import annotations

import sys

from release_commitmargin_20261027 import install

install()
which = sys.argv[1]
sys.argv = [sys.argv[0], *sys.argv[2:]]
if which == "dream":
    import calib_decode_dreamvl_20261013g as target
elif which == "e":
    import calib_decode_multi_20261013e as target
else:
    raise SystemExit(f"unknown target {which}")
target.main()
