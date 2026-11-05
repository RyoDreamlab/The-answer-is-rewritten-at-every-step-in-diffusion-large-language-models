#!/usr/bin/env python3
"""Run a stock decoder after installing the retune2 unmask-step rewrite.

Usage: run_retune2_20261030.py {lavida_mme|dream_vstar} <decoder args>
RETUNE_RULE must match the target. RETUNE_IDS=id1,id2 restricts the row set (smoke).
"""
from __future__ import annotations

import os
import sys

from release_retune2_20261030 import install

install()
which = sys.argv[1]
sys.argv = [sys.argv[0], *sys.argv[2:]]


def _filter(rows):
    raw = os.environ.get("RETUNE_IDS", "").strip()
    if not raw:
        return rows
    want = [item for item in raw.split(",") if item]
    by = {row["id"]: row for row in rows}
    missing = [item for item in want if item not in by]
    if missing:
        raise RuntimeError(f"RETUNE_IDS missing {missing}")
    return [by[item] for item in want]


if which == "lavida_mme":
    import calib_decode_vstar_mme_20261022 as target
    orig = target.DATASETS["mme"]

    def rows_mme(backend, limit, image_cond, orig=orig):
        return _filter(orig(backend, limit, image_cond))

    target.DATASETS["mme"] = rows_mme
    target.main()
elif which == "dream_vstar":
    import calib_decode_vstar_mme_dreamvl_20261022 as dreamvm
    target = dreamvm.g
    orig = target.DATASETS["vstar"]

    def rows_vstar(backend, limit, image_cond, orig=orig):
        return _filter(orig(backend, limit, image_cond))

    target.DATASETS["vstar"] = rows_vstar
    target.main()
else:
    raise SystemExit(f"unknown target {which}")
