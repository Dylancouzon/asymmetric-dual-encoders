#!/usr/bin/env bash
# M9.1 chain: gates, then the registered arms in the registered order, stopping at the first
# failure. `decide` is re-derivable from the artifacts, so its one-use token is cleared before each
# call; ARM tokens are never cleared, so an arm still cannot be re-run over its own result.
set -u
cd /home/dylan/asymetric-dual-encoders
PY=.venv/bin/python
log() { echo "[$(date -Is)] $*" | tee -a logs/m9_chain.log; }

for step in "$@"; do
  case "$step" in
    gate:*)
      spec="${step#gate:}"; mod="${spec%%:*}"; arg="${spec#*:}"
      [ "$arg" = "$mod" ] && arg=""
      log "GATE $mod $arg"
      $PY "m9src/${mod}.py" $arg > "logs/m9_gate_${mod}.log" 2>&1 || {
        log "GATE FAILED $mod"; tail -20 "logs/m9_gate_${mod}.log" | tee -a logs/m9_chain.log; exit 1; }
      log "GATE OK $mod"
      ;;
    decide)
      rm -f work/m9tokens/m9-decisions.json
      log "decide"
      $PY m9src/screen.py decide >> logs/m9_chain.log 2>&1 || { log "DECIDE FAILED"; exit 1; }
      ;;
    adequacy)
      rm -f work/m9tokens/m9-adequacy.json
      log "adequacy"
      $PY m9src/screen.py adequacy >> logs/m9_chain.log 2>&1 || { log "ADEQUACY FAILED"; exit 1; }
      ;;
    *)
      log "START $step"
      $PY m9src/screen.py arm "$step" > "logs/m9_arm_${step}.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        log "FAILED $step rc=$rc"; tail -25 "logs/m9_arm_${step}.log" | tee -a logs/m9_chain.log; exit $rc
      fi
      log "DONE $step $(grep -o 'DONE.*' "logs/m9_arm_${step}.log" | tail -1 | cut -c1-320)"
      ;;
  esac
done
log "CHAIN COMPLETE"
