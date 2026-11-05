#!/usr/bin/env python3
"""Keep the 11 not-yet-up cells moving.

The raise12 queues already own the first attempt. When a GPU goes idle, this
process starts the next every-step prior scale on cells whose best score is
still at or below stock. A cell with any positive delta is left alone.
Dream-VL MMStar is not in this list.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/pfs/peng_gao/visual-hysteresis")
LOGDIR = ROOT / "logs" / "raise12_20261027"
STATE = LOGDIR / "follow_state.json"
PY = ROOT / ".venv" / "bin" / "python3"
E = "experiments/unmask_order/calib_decode_multi_20261013e.py"
V = "experiments/unmask_order/calib_decode_vstar_mme_20261022.py"
D = "experiments/unmask_order/calib_decode_dreamvl_20261013g.py"
DV = "experiments/unmask_order/calib_decode_vstar_mme_dreamvl_20261022.py"
L = "experiments/unmask_order/calib_decode_lladav_e_20261022.py"

# Later every-step scales. 1.0 already failed on these cells before this round.
LADDER = [0.10, 0.50, -0.25, -0.10, -0.50, 0.05, 2.0]
TAG = {
    0.10: "weak010",
    0.50: "weak050",
    -0.25: "sign025",
    -0.10: "sign010",
    -0.50: "sign050",
    0.05: "weak005",
    2.0: "weak200",
}

# gpu: which raise12 queue currently owns the cell
CELLS = [
    {"name": "lavida_pope", "n": 1000, "gpu": 0, "stock": "outputs/pope_20261013e/lavida_stock.jsonl",
     "script": E, "pre": ["--backend", "lavida", "--dataset", "pope"],
     "prior": "outputs/pope_20261013e/cf_pope_lavida.json", "bucket": "e"},
    {"name": "lavida_mmstar", "n": 1500, "gpu": 0, "stock": "outputs/mmstar_20261013e/lavida_stock.jsonl",
     "script": E, "pre": ["--backend", "lavida", "--dataset", "mmstar"],
     "prior": "outputs/mmstar_20261013e/cf_lavida.json", "bucket": "e"},
    {"name": "lavida_mme", "n": 2374, "gpu": 0, "stock": "outputs/mme_20261022/lavida_stock.jsonl",
     "script": V, "pre": ["--backend", "lavida", "--dataset", "mme"],
     "prior": "outputs/mme_20261022/cf_mme_lavida.json", "bucket": "mme"},
    {"name": "mmada_mme", "n": 2374, "gpu": 0, "stock": "outputs/mme_20261022/mmada_stock.jsonl",
     "script": V, "pre": ["--backend", "mmada", "--dataset", "mme"],
     "prior": "outputs/mme_20261022/cf_mme_mmada.json", "bucket": "mme"},
    {"name": "dream_sciqa", "n": 2017, "gpu": 0, "stock": "outputs/dreamvl_20261013g/sciqa_20261022/dreamvl_stock.jsonl",
     "script": D, "pre": ["--dataset", "sciqa"],
     "prior": "outputs/dreamvl_20261013g/cf_dreamvl.json", "bucket": "dream"},
    {"name": "dream_mmbench", "n": 4329, "gpu": 0, "stock": "outputs/dreamvl_20261013g/mmbench_20261022/dreamvl_stock.jsonl",
     "script": D, "pre": ["--dataset", "mmbench"],
     "prior": "outputs/dreamvl_20261013g/cf_dreamvl.json", "bucket": "dream"},
    {"name": "dream_m3cot", "n": 1108, "gpu": 1, "stock": "outputs/dreamvl_20261013g/m3cot_20261022/dreamvl_stock.jsonl",
     "script": D, "pre": ["--dataset", "m3cot"],
     "prior": "outputs/dreamvl_20261013g/cf_dreamvl.json", "bucket": "dream"},
    {"name": "mmada_mmstar", "n": 1500, "gpu": 1, "stock": "outputs/mmstar_20261013e/mmada_stock.jsonl",
     "script": E, "pre": ["--backend", "mmada", "--dataset", "mmstar"],
     "prior": "outputs/mmstar_20261013e/cf_mmada.json", "bucket": "e"},
    {"name": "dream_mme", "n": 2374, "gpu": 1, "stock": "outputs/mme_20261022/dreamvl_stock.jsonl",
     "script": DV, "pre": ["--dataset", "mme"],
     "prior": "outputs/mme_20261022/cf_mme_dreamvl.json", "bucket": "dreamvm"},
    {"name": "dream_vstar", "n": 191, "gpu": 1, "stock": "outputs/vstar_20261022/dreamvl_stock.jsonl",
     "script": DV, "pre": ["--dataset", "vstar"],
     "prior": "outputs/vstar_20261022/cf_vstar_dreamvl.json", "bucket": "dreamvm"},
    {"name": "lladav_pope", "n": 1000, "gpu": 1, "stock": "outputs/pope_lladav_20261022/lladav_stock.jsonl",
     "script": L, "pre": ["--backend", "lladav", "--dataset", "pope"],
     "prior": "outputs/pope_lladav_20261022/cf_pope_lladav.json", "bucket": "lladav"},
]

RESULT_RE = re.compile(
    r"RESULT (\S+) n=\d+ stock=\d+ .*? delta=(-?\d+) "
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {msg}"
    print(line, flush=True)
    with (LOGDIR / "follow_supervisor.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def raise12_alive(gpu: int) -> bool:
    needle = f"queue_raise12_20261027.sh {gpu}"
    out = subprocess.run(["pgrep", "-f", needle], capture_output=True, text=True)
    return bool(out.stdout.strip())


def parse_results() -> dict[str, list[tuple[str, int]]]:
    found: dict[str, list[tuple[str, int]]] = {c["name"]: [] for c in CELLS}
    if not LOGDIR.exists():
        return found
    for path in LOGDIR.glob("*.log"):
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            m = RESULT_RE.search(line)
            if not m or m.group(1) not in found:
                continue
            tag = "unknown"
            tm = re.search(r" (commit|weak\d+|sign\d+)=", line)
            if tm:
                tag = tm.group(1)
            found[m.group(1)].append((tag, int(m.group(2))))
    return found


def best_delta(hits: list[tuple[str, int]]) -> int | None:
    if not hits:
        return None
    return max(d for _, d in hits)


def out_paths(cell: dict, tag: str) -> tuple[Path, Path]:
    name = cell["name"]
    bucket = cell["bucket"]
    if bucket == "e":
        parent = ROOT / "outputs" / tag / "e_20261013e"
    elif bucket == "mme":
        parent = ROOT / "outputs" / tag / "mme_20261022"
    elif bucket == "lladav":
        parent = ROOT / "outputs" / tag / "lladav_20261022"
    elif bucket == "dreamvm":
        parent = ROOT / "outputs" / tag / "dreamvm_20261022"
    elif bucket == "dream":
        parent = ROOT / "outputs" / "dreamvl_20261013g" / tag
    else:
        raise RuntimeError(bucket)
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{name}_gated.jsonl", parent / f"{name}.progress.json"


def launch(cell: dict, scale: float, gpu: int) -> int:
    tag = TAG[scale]
    out, prog = out_paths(cell, tag)
    script = LOGDIR / f"follow_{cell['name']}_{tag}_gpu{gpu}.sh"
    cmd = [
        str(PY), str(ROOT / cell["script"]),
        *cell["pre"],
        "--mode", "gated", "--alpha-rule", "spread",
        f"--alpha-scale={scale}",
        "--prior-file", cell["prior"],
        "--cuda", str(gpu),
        "--output", str(out),
        "--progress", str(prog),
    ]
    body = f"""#!/bin/bash
set -u
cd {ROOT}
export PYTHONUNBUFFERED=1 HF_HOME={ROOT}/hf-cache TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export URD_PROJECT={ROOT}
LOG={LOGDIR}/follow_gpu{gpu}.log
exec >>"$LOG" 2>&1
echo "[$(date -Is)] LAUNCH {cell['name']} scale={scale} tag={tag}"
if ! {' '.join(cmd)}; then
  echo "[$(date -Is)] FAIL {cell['name']} scale={scale} rc=$?"
  exit 1
fi
echo "[$(date -Is)] EXIT {cell['name']} scale={scale} rc=0"
{PY} - {cell['name']} {ROOT / cell['stock']} {out} {tag} << 'PY'
import json, sys
name, stock, new, tag = sys.argv[1:]
def load(p):
    d = {{}}
    with open(p) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[r["id"]] = r.get("finished_letter") == r.get("gold")
    return d
a, b = load(stock), load(new)
ids = sorted(set(a) & set(b))
sa, sb = sum(a[i] for i in ids), sum(b[i] for i in ids)
n = len(ids)
if n == 0:
    raise SystemExit("RESULT %s n=0 empty" % name)
print("RESULT %s n=%d stock=%d (%.2f%%) %s=%d (%.2f%%) delta=%d (%+.2fpp)" % (
    name, n, sa, sa / n * 100, tag, sb, sb / n * 100, sb - sa, (sb - sa) / n * 100), flush=True)
PY
"""
    script.write_text(body)
    script.chmod(0o755)
    proc = subprocess.Popen(
        ["bash", str(script)],
        cwd=str(ROOT),
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )
    log(f"START gpu={gpu} {cell['name']} scale={scale} pid={proc.pid}")
    return proc.pid


def main() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    state = {"gpu": {"0": 0, "1": 0}}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    log("supervisor up")
    while True:
        results = parse_results()
        summary = []
        eligible = []
        for cell in CELLS:
            hits = results[cell["name"]]
            delta = best_delta(hits)
            tags = {t for t, _ in hits}
            summary.append(f"{cell['name']}:{delta if delta is not None else 'pending'}")
            if delta is not None and delta > 0:
                continue
            if delta is None and raise12_alive(cell["gpu"]):
                continue
            nxt = next((s for s in LADDER if TAG[s] not in tags), None)
            if nxt is None:
                continue
            eligible.append((cell, nxt))
        eligible.sort(key=lambda item: item[0]["n"])
        busy = {int(g): int(pid) for g, pid in state.get("gpu", {}).items()}
        free = []
        for gpu in (0, 1):
            pid = busy.get(gpu, 0)
            if pid and not alive(pid):
                log(f"DONE gpu={gpu} pid={pid}")
                pid = 0
            if pid and alive(pid):
                continue
            if raise12_alive(gpu):
                continue
            free.append(gpu)
        assigned = set()
        for gpu in free:
            pick = next(((c, s) for c, s in eligible if c["name"] not in assigned), None)
            if pick is None:
                break
            cell, scale = pick
            assigned.add(cell["name"])
            busy[gpu] = launch(cell, scale, gpu)
        state["gpu"] = {str(g): busy.get(g, 0) for g in (0, 1)}
        STATE.write_text(json.dumps(state))
        running = [f"gpu{g}={busy.get(g, 0)}" for g in (0, 1) if busy.get(g, 0) and alive(busy.get(g, 0))]
        log("status " + " ".join(summary) + " | " + (" ".join(running) or "idle"))
        if not eligible and not any(raise12_alive(g) for g in (0, 1)) and not running:
            log("STOPPED nothing left to run")
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
