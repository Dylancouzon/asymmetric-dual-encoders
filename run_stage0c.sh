#!/usr/bin/env bash
# Held-out dev slices (they need the frozen pool), the decontamination passes that depend on
# them, and the reference rows for the full pinned dev suite. Sequential.
# Usage: ./run_stage0c.sh [first_step]
set -u
FIRST=${1:-1}
cd /home/dylan/asymetric-dual-encoders/m7src
PY=../.venv/bin/python
export HF_HUB_ENABLE_HF_TRANSFER=1
L=../logs
N=5
step () {
  local i=$1; shift; local title=$1; shift
  if [ "$i" -lt "$FIRST" ]; then echo "--- skip $i/$N $title" | tee -a $L/stage0c.log; return 0; fi
  echo "=========== $i/$N $title  $(date -Is) ===========" | tee -a $L/stage0c.log
  "$@" >> $L/stage0c.log 2>&1
  local rc=$?
  echo "--- exit $rc  $(date -Is)  mem: $(free -m | awk '/Mem:/{print $3"/"$2" MB used"}')" | tee -a $L/stage0c.log
  return $rc
}
step 1 "build held-out dev slices"                 $PY heldout.py            || exit 1
# must precede any training or probe: it removes from TRAIN, not from dev
step 2 "TRAIN <-> held-out decontamination"        $PY decontam_heldout.py   || exit 1
step 3 "refresh the field table counts"            $PY field_table.py        || exit 1
step 4 "R3 sweeps missed by the first run"         $PY decontam_r3_extra.py  || exit 1
step 5 "dev reference rows (full pinned suite)"    $PY dev_eval.py           || exit 1
echo "STAGE0C COMPLETE $(date -Is)" | tee -a $L/stage0c.log
