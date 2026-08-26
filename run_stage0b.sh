#!/usr/bin/env bash
# Stage 0 (representation compatibility) + the go/no-go gate. Sequential, one GPU job at a time.
# Usage: ./run_stage0b.sh [first_step]
set -u
FIRST=${1:-1}
cd /home/dylan/asymetric-dual-encoders/m7src
PY=../.venv/bin/python
export HF_HUB_ENABLE_HF_TRANSFER=1
L=../logs
N=4
step () {
  local i=$1; shift
  local title=$1; shift
  if [ "$i" -lt "$FIRST" ]; then echo "--- skip $i/$N $title" | tee -a $L/stage0b.log; return 0; fi
  echo "=========== $i/$N $title  $(date -Is) ===========" | tee -a $L/stage0b.log
  "$@" >> $L/stage0b.log 2>&1
  local rc=$?
  echo "--- exit $rc  $(date -Is)  mem: $(free -m | awk '/Mem:/{print $3"/"$2" MB used"}')" | tee -a $L/stage0b.log
  return $rc
}
# 0.1: the closed-form MSE-optimal flat-weight table -- an exact upper bound on what flat
# distillation can reach, from one linear solve.
step 1 "stage-0.1 ridge probe"                          $PY stage0_ridge.py teacher noprefix || exit 1
step 2 "stage-0.2 capacity probe (gate-ineligible)"     $PY capacity_probe.py noprefix 6000  || exit 1
step 3 "objective grid A/B/C"                           $PY -c "import program; program.save('p1', program.phase1_objective())" || exit 1
step 4 "go/no-go gate"                                  $PY gate.py p1-objC p1-objB          || exit 1
echo "STAGE0B COMPLETE $(date -Is)" | tee -a $L/stage0b.log
