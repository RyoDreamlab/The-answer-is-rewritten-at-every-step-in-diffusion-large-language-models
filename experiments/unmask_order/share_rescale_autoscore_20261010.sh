#!/usr/bin/env bash
# Auto-score share_rescale_decode_20261010 per model once decode finishes. Never overwrites.
set -u
PROJ=/mnt/pfs/peng_gao/visual-hysteresis
OUT=$PROJ/outputs/share_rescale_decode_20261010
LOG=$PROJ/logs/share_rescale_autoscore_20261010.log
N=2017
declare -A PID=([lavida]=775457 [mmada]=775962)
cd "$PROJ" || exit 1
export URD_PROJECT=$PROJ
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "watcher start pid=$$"
while :; do
  pending=0
  for m in lavida mmada; do
    js=$OUT/score_$m.json
    [ -e "$js" ] && continue
    pending=1
    n=$(grep -c . "$OUT/$m.jsonl" 2>/dev/null || echo 0)
    if kill -0 "${PID[$m]}" 2>/dev/null; then alive=1; else alive=0; fi
    log "$m n=$n/$N pid_alive=$alive"
    if [ "$n" -ge "$N" ] && [ "$alive" = 0 ]; then
      tmp=$(mktemp "$OUT/.score_$m.XXXX")
      if .venv/bin/python3 experiments/unmask_order/score_share_rescale_20261010.py "$m" > "$tmp" 2>> "$LOG"; then
        tail -n 1 "$tmp" > "$tmp.json"
        [ -e "$OUT/score_$m.txt" ] || mv "$tmp" "$OUT/score_$m.txt"
        [ -e "$js" ] || mv "$tmp.json" "$js"
        rm -f "$tmp" "$tmp.json"
        log "$m scored: $(head -n 1 "$OUT/score_$m.txt")"
      else
        log "$m scorer FAILED; will retry"; rm -f "$tmp"
      fi
    elif [ "$alive" = 0 ]; then
      log "$m WARNING: pid exited with n=$n < $N"
    fi
  done
  [ "$pending" = 0 ] && { log "both scored; exit"; exit 0; }
  sleep 120
done
