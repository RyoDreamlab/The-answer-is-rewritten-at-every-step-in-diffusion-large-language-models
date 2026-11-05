#!/usr/bin/env bash
# Score each calib_decode_20261012 output once it has 2017 rows and its decode exited. Never overwrites.
set -u
ROOT=/mnt/pfs/peng_gao/visual-hysteresis
OUT=$ROOT/outputs/calib_decode_20261012
LOG=$ROOT/logs/calib_decode_autoscore_20261012.log
N=2017
cd "$ROOT" || exit 1
export URD_PROJECT=$ROOT
exec 9>"$ROOT/logs/calib_decode_autoscore_20261012.lock"
flock -n 9 || { echo "[$(date '+%F %T')] lock held; exit" >> "$LOG"; exit 0; }
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "watcher start pid=$$"
while :; do
  pending=0
  for m in lavida mmada; do
    js=$OUT/score_${m}.json
    [ -e "$js" ] && continue
    pending=1
    out=$OUT/${m}.jsonl
    n=$(grep -c . "$out" 2>/dev/null); n=${n:-0}
    if pgrep -f "^[^ ]*python3 [^ ]*calib_decode.py --backend $m .*--output $out" >/dev/null; then alive=1; else alive=0; fi
    if [ "$n" -ge "$N" ] && [ "$alive" = 0 ]; then
      tmp=$(mktemp "$OUT/.score_${m}.XXXX")
      if .venv/bin/python3 experiments/unmask_order/score_calib_decode_20261012.py "$m" > "$tmp" 2>> "$LOG"; then
        tail -n 1 "$tmp" > "$tmp.json"
        [ -e "$OUT/score_${m}.txt" ] || mv "$tmp" "$OUT/score_${m}.txt"
        [ -e "$js" ] || mv "$tmp.json" "$js"
        rm -f "$tmp" "$tmp.json"
        log "$m scored: $(head -n 1 "$OUT/score_${m}.txt")"
      else
        log "$m scorer FAILED; will retry"; rm -f "$tmp"
      fi
    elif [ -e "$out" ] && [ "$alive" = 0 ] && [ "$n" -lt "$N" ]; then
      log "$m WARNING: decode not alive with n=$n < $N"
    fi
  done
  [ "$pending" = 0 ] && { log "all scored; exit"; exit 0; }
  sleep 120
done
