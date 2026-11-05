#!/usr/bin/env bash
# Master queue for the 20261013d mechanism upgrades: Upgrade 2 (DLA circuit + ablation decodes) then Upgrade 1
# (attractor decodes).  Strictly sequential => at most one GPU job at a time.
# Usage: nohup bash queue_20261013d.sh > /dev/null 2>&1 &
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
mkdir -p "$ROOT/logs/dla_20261013d"
M=$ROOT/logs/dla_20261013d/master_queue.log
echo "MASTER START $(date -Iseconds) pid=$$" >> "$M"
bash "$S/run_dla_20261013d.sh" "${1:-all}"; echo "MASTER dla rc=$? $(date -Iseconds)" >> "$M"
bash "$S/run_attractor_20261013d.sh" "${2:-all}"; echo "MASTER attractor rc=$? $(date -Iseconds)" >> "$M"
echo "MASTER END $(date -Iseconds)" >> "$M"
