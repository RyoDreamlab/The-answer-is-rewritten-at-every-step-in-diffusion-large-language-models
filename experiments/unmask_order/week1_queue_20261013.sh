#!/usr/bin/env bash
# Week-1 GPU queue for one model on one GPU.
#   lane A (background): B2 ScienceQA gated full decode
#   lane B (background): B1 M3CoT gated (spread) -> B1 M3CoT full-strength
# Both lanes share the GPU (batch-1 decode leaves headroom on a 96 GB Blackwell).
# Usage: week1_queue_20261013.sh <lavida|mmada> <cuda>
set -u
model=$1; cuda=$2
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
S=$ROOT/experiments/unmask_order
mkdir -p "$ROOT/logs/week1_20261013"
qlog=$ROOT/logs/week1_20261013/queue_${model}.log
echo "QUEUE START $model cuda=$cuda $(date -Iseconds)" >> "$qlog"
bash "$S/run_b2_sqa_gated_20261013.sh" "$model" "$cuda" &
pa=$!
( bash "$S/run_b1_m3cot_20261013.sh" "$model" "$cuda" spread; echo "LANE_B spread done $(date -Iseconds)" >> "$qlog";
  bash "$S/run_b1_m3cot_20261013.sh" "$model" "$cuda" full;   echo "LANE_B full done $(date -Iseconds)" >> "$qlog" ) &
pb=$!
wait $pa; echo "LANE_A (B2 sqa gated) done $(date -Iseconds)" >> "$qlog"
wait $pb
echo "QUEUE END $model $(date -Iseconds)" >> "$qlog"
