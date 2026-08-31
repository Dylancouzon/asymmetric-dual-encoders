#!/usr/bin/env bash
# Guardian for the M9.3 watchdog (Codex unattended review, BLOCKER 2).
#
# Nothing supervised the watchdog itself: if it was OOM-killed or died outside its protected
# loop, a later trainer death would go unnoticed for the rest of a 3-day unattended window,
# with "status commits stopped" as the only symptom. This relaunches it.
#
# Safe by construction: the watchdog takes an flock on work/m9long/watchdog.lock that the
# kernel releases only on process death, so a relaunch while one is alive exits immediately
# via SystemExit. Two watchdogs can never supervise one trainer.
#
# It deliberately does NOT relaunch once the run has legitimately finished.
set -u
cd /home/dylan/asymetric-dual-encoders || exit 1
RUN=work/m9long
LOG=logs/m9_guardian.log
WD_ARGS="--hours 168 --eval-stale 18000 --ckpt-stale 7200"

say() { echo "[$(date -Is)] $*" >> "$LOG"; }
say "guardian start (pid $$), supervising: watchdog.py $WD_ARGS"

while true; do
  sleep 60
  # The run ended on its own terms -- a completed cooldown, a registered stop, or a watchdog
  # give-up. Nothing left to supervise; exit rather than resurrect a finished run.
  if [ -f "$RUN/terminal.json" ]; then
    say "terminal.json present; run is over, guardian exiting"
    exit 0
  fi
  if pgrep -f "^\.venv/bin/python m9src/watchdog\.py" > /dev/null 2>&1; then
    continue
  fi
  say "WATCHDOG ABSENT -- relaunching"
  setsid nohup .venv/bin/python m9src/watchdog.py $WD_ARGS >> logs/m9_watchdog.log 2>&1 &
  sleep 20
  if pgrep -f "^\.venv/bin/python m9src/watchdog\.py" > /dev/null 2>&1; then
    say "relaunch OK"
  else
    say "RELAUNCH FAILED -- see logs/m9_watchdog.log"
  fi
done
