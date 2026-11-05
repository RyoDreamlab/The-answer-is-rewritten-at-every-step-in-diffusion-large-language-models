#!/usr/bin/env python3
"""Run a stock decoder after installing the bump answer-slot rule.

Usage: run_bump_20261030.py {e|c|dream|vstar} <decoder args>
HYST_RULE must be set.
"""
from __future__ import annotations

import sys

from release_bump_20261030 import install

install()
which = sys.argv[1]
sys.argv = [sys.argv[0], *sys.argv[2:]]
if which == "e":
    import calib_decode_multi_20261013e as target
elif which == "c":
    import inspect

    import calib_decode_multi_20261013c as target

    # The shared decoder refuses GPU 0. Both cards are free for this queue,
    # so drop that guard in this process only and leave the file unchanged.
    guard = (
        '    if args.cuda == 0:\n'
        '        raise RuntimeError("GPU 0 belongs to another agent; use --cuda 1")\n'
    )
    source = inspect.getsource(target.main)
    if guard not in source:
        raise SystemExit("GPU 0 guard not found; refusing to run c decoder")
    exec(source.replace(guard, "", 1), target.__dict__)
elif which == "dream":
    import calib_decode_dreamvl_20261013g as target
elif which == "vstar":
    import calib_decode_vstar_mme_20261022 as target
else:
    raise SystemExit(f"unknown target {which}")
target.main()
