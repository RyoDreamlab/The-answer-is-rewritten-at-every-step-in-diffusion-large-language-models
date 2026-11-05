#!/usr/bin/env bash
# Score each long_canvas_20261012 output once it has 2017 rows and its decode exited. Never overwrites.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
OUT=$ROOT/outputs/long_canvas_20261012
LOG=$ROOT/logs/long_canvas_autoscore_20261012.log
N=2017
cd "$ROOT" || exit 1
export URD_PROJECT=$ROOT
exec 9>"$ROOT/logs/long_canvas_autoscore_20261012.lock"
flock -n 9 || { echo "[$(date '+%F %T')] lock held; exit" >> "$LOG"; exit 0; }
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "watcher start pid=$$"
while :; do
  pending=0
  for m in lavida mmada; do
    for tag in c32 c64; do
      js=$OUT/score_${m}_${tag}.json
      [ -e "$js" ] && continue
      pending=1
      out=$OUT/${m}_${tag}.jsonl
      n=$(grep -c . "$out" 2>/dev/null); n=${n:-0}
      if pgrep -f "^[^ ]*python3 [^ ]*long_canvas_decode.py --backend $m .*--output $out" >/dev/null; then alive=1; else alive=0; fi
      if [ "$n" -ge "$N" ] && [ "$alive" = 0 ]; then
        tmp=$(mktemp "$OUT/.score_${m}_${tag}.XXXX")
        if .venv/bin/python3 experiments/unmask_order/score_long_canvas_20261012.py "$m" "$tag" > "$tmp" 2>> "$LOG"; then
          tail -n 1 "$tmp" > "$tmp.json"
          [ -e "$OUT/score_${m}_${tag}.txt" ] || mv "$tmp" "$OUT/score_${m}_${tag}.txt"
          [ -e "$js" ] || mv "$tmp.json" "$js"
          rm -f "$tmp" "$tmp.json"
          log "$m $tag scored: $(head -n 1 "$OUT/score_${m}_${tag}.txt")"
        else
          log "$m $tag scorer FAILED; will retry"; rm -f "$tmp"
        fi
      elif [ -e "$out" ] && [ "$alive" = 0 ] && [ "$n" -lt "$N" ]; then
        log "$m $tag WARNING: decode not alive with n=$n < $N"
      fi
    done
  done
  [ "$pending" = 0 ] && { log "all scored; exit"; exit 0; }
  sleep 120
done
