#!/usr/bin/env python3
"""Build the MMBench-dev (English, lmms-lab format) manifest for the option-letter calibration study.

Source: data/benchmarks/mmbench/lmms-lab_MMBench/en/dev-00000-of-00001.parquet
Output (new directory only, refuses to overwrite): datasets/mmbench_dev_v1/
    manifest.json   one record per kept item
    images/<index>.<ext>   the exact source image bytes (extension from the decoded format)

Kept: items with image bytes, a gold letter in the item's option range, and 2..4 non-empty options.
Options = the non-empty A..D columns ('nan' strings count as empty); they must form a prefix A..A+n-1.
Prompt templates are NOT stored here: the decode script composes them per backend, with the same structure
as ScienceQA (hint/context + question + lettered options + the backend's answer instruction).
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import Counter
from pathlib import Path

PROJECT = Path(os.environ.get("URD_PROJECT", "/mnt/pfs/peng_gao/visual-hysteresis"))
PARQUET = PROJECT / "data/benchmarks/mmbench/lmms-lab_MMBench/en/dev-00000-of-00001.parquet"
OUT_DIR = PROJECT / "datasets/mmbench_dev_v1"
LETTERS = "ABCD"


def clean(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image

    if (args.out_dir / "manifest.json").exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_dir / 'manifest.json'}")
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.parquet)
    rows = table.to_pylist()
    kept, drop = [], Counter()
    for r in rows:
        img = r.get("image")
        if not isinstance(img, dict) or not img.get("bytes"):
            drop["no_image"] += 1
            continue
        opts = [clean(r.get(L)) for L in LETTERS]
        present = [i for i, o in enumerate(opts) if o]
        n = len(present)
        if present != list(range(n)) or not 2 <= n <= 4:
            drop["options_not_prefix_or_count"] += 1
            continue
        gold = clean(r.get("answer")).upper()
        if gold not in LETTERS[:n]:
            drop["gold_not_in_range"] += 1
            continue
        idx = int(r["index"])
        with Image.open(io.BytesIO(img["bytes"])) as im:
            fmt = (im.format or "PNG").lower()
            w, h = im.size
        ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "gif": ".gif", "bmp": ".bmp"}.get(fmt, f".{fmt}")
        path = img_dir / f"{idx}{ext}"
        if not path.exists():
            path.write_bytes(img["bytes"])
        choices = opts[:n]
        kept.append({
            "id": f"mmbench_dev_{idx}", "index": idx, "question": clean(r.get("question")),
            "hint": clean(r.get("hint")), "choices": choices, "n_opt": n, "gold": gold,
            "gold_text": choices[LETTERS.index(gold)], "category": clean(r.get("category")),
            "l2_category": clean(r.get("L2-category")), "source": clean(r.get("source")),
            "split": clean(r.get("split")), "image": str(path), "image_size": [w, h], "image_format": fmt,
        })
    ids = [k["id"] for k in kept]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate ids")
    (args.out_dir / "manifest.json").write_text(json.dumps(kept, ensure_ascii=False, indent=0), encoding="utf-8")
    meta = {
        "source_parquet": str(args.parquet), "n_source_rows": len(rows), "n_kept": len(kept), "dropped": dict(drop),
        "n_opt_hist": dict(sorted(Counter(k["n_opt"] for k in kept).items())),
        "gold_hist": dict(sorted(Counter(k["gold"] for k in kept).items())),
        "gold_hist_by_n_opt": {str(n): dict(sorted(Counter(k["gold"] for k in kept if k["n_opt"] == n).items()))
                               for n in (2, 3, 4)},
        "n_with_hint": sum(1 for k in kept if k["hint"]),
        "category_hist": dict(Counter(k["category"] for k in kept).most_common()),
        "l2_category_hist": dict(Counter(k["l2_category"] for k in kept).most_common()),
        "image_format_hist": dict(Counter(k["image_format"] for k in kept)),
    }
    (args.out_dir / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"wrote {args.out_dir / 'manifest.json'} n={len(kept)}")


if __name__ == "__main__":
    main()
