#!/bin/bash
# Watchdog for the NQ Stella encode on the RTX box. Exits (waking the session) on completion,
# failure, process exit, or no new shard for 60 min. Heartbeat: $HB.
HB=${HB:-/tmp/claude-501/hb_nq.txt}
mkdir -p "$(dirname "$HB")"
last=""; same=0
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 dylan@100.114.121.66 '
    L=~/constella-talk/encode_nq.log
    if grep -q "NQ COMPLETE" $L; then echo DONE
    elif grep -qiE "Traceback|Error|Killed|OOM" $L; then echo FAILED; tail -5 $L
    elif ! pgrep -f talk_encode_nq.py >/dev/null; then echo EXITED; tail -5 $L
    else ls ~/constella-talk/work/nq_stella | grep -c shard_; fi' 2>&1)
  echo "$(date +%T) $out" > "$HB"
  case "$out" in DONE*|FAILED*|EXITED*) echo "$out"; exit 0;; esac
  if [ "$out" = "$last" ]; then same=$((same+1)); else same=0; last="$out"; fi
  if [ $same -ge 6 ]; then echo "STALLED: shard count unchanged for 60 min ($out)"; exit 0; fi
  sleep 600
done
