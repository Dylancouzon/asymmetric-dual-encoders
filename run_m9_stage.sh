#!/usr/bin/env bash
# M9.1 arm chain. Runs the registered arms in the registered order, stopping at the first failure.
# `decide` is re-derivable from the artifacts, so its one-use token is cleared before each call;
# every ARM token is left alone, so an arm still cannot be re-run over its own result.
set -u
cd /home/dylan/asymetric-dual-encoders
PY=.venv/bin/python
log() { echo "[$(date -Is)] $*" | tee -a logs/m9_chain.log; }

decide() {
  rm -f work/m9tokens/m9-decisions.json
  log "decide"
  $PY m9src/screen.py decide >> logs/m9_chain.log 2>&1 || { log "DECIDE FAILED"; exit 1; }
}

for arm in "$@"; do
  if [ "$arm" = "decide" ]; then decide; continue; fi
  log "START $arm"
  $PY m9src/screen.py arm "$arm" > "logs/m9_arm_${arm}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "FAILED $arm rc=$rc"; tail -25 "logs/m9_arm_${arm}.log" | tee -a logs/m9_chain.log; exit $rc
  fi
  log "DONE $arm $(grep -o 'DONE.*' "logs/m9_arm_${arm}.log" | tail -1 | cut -c1-300)"
done
log "CHAIN COMPLETE"
