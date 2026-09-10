#!/bin/bash
# Tracked here, not in gitignored work/: a fresh clone needs it (CLAUDE.md, durable knowledge).
# Survives a VM kill: append + flush every sample. Columns: ts, guest_used_MB, guest_avail_MB,
# arm_rss_MB, gpu_MB, host_free_MB (every 6th sample only -- powershell is slow), arm_hwm_MB
# (VmHWM, MONOTONIC -- trust this for peaks; a sampled RSS max is only a lower bound), n_match
# (0 matches => rss/hwm are -1, not 0, so a bad pattern cannot look like an idle arm).
# Usage: memtrace.sh [pgrep-pattern] [out.csv]
PAT="${1:-run_arm.py F-bge-small}"
OUT="${2:-work/memtrace.csv}"
echo "# pattern: $PAT" >> $OUT
echo "ts,guest_used_mb,guest_avail_mb,arm_rss_mb,gpu_mb,host_free_mb,arm_hwm_mb,n_match" >> $OUT
i=0
while true; do
  read used avail < <(free -m | awk '/^Mem:/{print $3, $7}')
  # `pgrep -f` also matches the tracer AND the shell that launched the arm, because the pattern
  # sits in their argv too (same trap as `pkill -f` matching your own shell). Excluding our own
  # pids is not enough -- the launcher is a different process. So take the LARGEST RSS among the
  # matches: the arm outweighs any bookkeeping shell by three orders of magnitude.
  RSS=0; HWM=0; NMATCH=0
  for pid in $(pgrep -f "$PAT" | grep -vx -e "$$" -e "$PPID"); do
    NMATCH=$((NMATCH+1))
    r=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null || echo 0)
    # VmHWM is MONOTONIC, so it cannot be missed between 5-second samples the way VmRSS can.
    # Every "peak" in the ledger before 2026-09-07 was a sampled VmRSS max -- a LOWER BOUND.
    h=$(awk '/VmHWM/{print int($2/1024)}' /proc/$pid/status 2>/dev/null || echo 0)
    [ -n "$r" ] && [ "$r" -gt "$RSS" ] && RSS=$r
    [ -n "$h" ] && [ "$h" -gt "$HWM" ] && HWM=$h
  done
  # -1, not 0: a pattern that matches nothing looks exactly like an idle arm otherwise, and this
  # CSV is the only forensic instrument that survives a WSL kill (dmesg does not).
  [ "$NMATCH" -eq 0 ] && { RSS=-1; HWM=-1; }
  GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  HOST=""
  if [ $((i % 6)) -eq 0 ]; then
    HOST=$(timeout 25 powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" 2>/dev/null | tr -d '\r')
  fi
  echo "$(date +%H:%M:%S),$used,$avail,$RSS,$GPU,$HOST,$HWM,$NMATCH" >> $OUT
  sync -d "$(dirname "$OUT")" 2>/dev/null || sync
  i=$((i+1))
  sleep 5
done
