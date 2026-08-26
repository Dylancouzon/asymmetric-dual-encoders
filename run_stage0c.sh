#!/usr/bin/env bash
# Held-out dev slices (they need the frozen pool), their decontamination pass, and their
# reference rows. Sequential. Usage: ./run_stage0c.sh [first_step]
set -u
FIRST=${1:-1}
cd /home/dylan/asymetric-dual-encoders/m7src
PY=../.venv/bin/python
export HF_HUB_ENABLE_HF_TRANSFER=1
L=../logs
N=3
step () {
  local i=$1; shift; local title=$1; shift
  if [ "$i" -lt "$FIRST" ]; then echo "--- skip $i/$N $title" | tee -a $L/stage0c.log; return 0; fi
  echo "=========== $i/$N $title  $(date -Is) ===========" | tee -a $L/stage0c.log
  "$@" >> $L/stage0c.log 2>&1
  local rc=$?
  echo "--- exit $rc  $(date -Is)  mem: $(free -m | awk '/Mem:/{print $3"/"$2" MB used"}')" | tee -a $L/stage0c.log
  return $rc
}
step 1 "build held-out dev slices"                $PY heldout.py            || exit 1
step 2 "TRAIN <-> held-out decontamination"       $PY decontam_heldout.py   || exit 1
step 3 "dev reference rows (all components)"      $PY dev_eval.py           || exit 1
echo "STAGE0C COMPLETE $(date -Is)" | tee -a $L/stage0c.log
