#!/bin/bash
# Wait for the VM end-to-end sweep: exits on completion, a traceback, or the process ending.
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 dylan@34.26.83.255 '
    if [ -s /home/dylan/e2e_nq.json ]; then echo DONE
    elif grep -qiE "Traceback|Error|Killed" /home/dylan/e2e_run.log; then echo FAILED; tail -8 /home/dylan/e2e_run.log
    elif ! pgrep -f talk_throughput.py >/dev/null; then echo EXITED; tail -8 /home/dylan/e2e_run.log
    else grep -c "^e2e" /home/dylan/e2e_run.log; fi' 2>&1)
  case "$out" in DONE*|FAILED*|EXITED*) echo "$out"; exit 0;; esac
  sleep 60
done
