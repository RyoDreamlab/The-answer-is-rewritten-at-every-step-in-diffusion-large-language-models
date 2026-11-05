#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

root = Path("/mnt/pfs/peng_gao/visual-hysteresis/outputs/unmask_order_bidir_20261009")
for name in ("lavida", "mmada"):
    rows = [json.loads(line) for line in (root / f"{name}.jsonl").open()]
    counts = Counter((row.get("argmax_blocked") or "")[:24] for row in rows)
    print(name, "blocked top", counts.most_common(6))
    changed = [row for row in rows if row["argmax_changed"]]
    same = [row for row in rows if not row["argmax_changed"]]
    picked = []
    seen = set()
    for row in changed[:4] + same[:4] + changed + same:
        if row["id"] in seen:
            continue
        picked.append(row)
        seen.add(row["id"])
        if len(picked) == 8:
            break
    dest = root / "strips" / name
    dest.mkdir(parents=True, exist_ok=True)
    for record in picked:
        history = record["canvas_text"]
        cell_w, cell_h = 78, 22
        image = Image.new("RGB", (16 * cell_w + 70, 16 * cell_h + 56), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        caption = f"{record['id']}  bi={record['argmax_bi']}  blocked={record['argmax_blocked']}"
        draw.text((4, 4), caption[:150], fill=(0, 0, 0))
        for step, texts in enumerate(history):
            y = 28 + step * cell_h
            draw.text((2, y + 4), f"{step:02d}", fill=(90, 90, 90))
            for col, text in enumerate(texts):
                x0 = 36 + col * cell_w
                blank = not text
                outline = (180, 40, 40) if col == 0 else (190, 190, 190)
                fill = (245, 245, 245) if blank else (255, 255, 255)
                draw.rectangle([x0, y, x0 + cell_w - 2, y + cell_h - 2], fill=fill, outline=outline)
                if text:
                    draw.text((x0 + 2, y + 4), text[:9], fill=(0, 0, 0))
        image.save(dest / f"{record['id']}.png")
    print(name, "png", len(list(dest.glob("*.png"))))
