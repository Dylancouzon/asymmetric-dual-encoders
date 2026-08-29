#!/usr/bin/env bash
# Strictly sequential. The 2026-08-25 WSL OOM came from running three memory-heavy jobs at once;
# CLAUDE.md already recorded that lesson from the M4 incident and it is now enforced here.
# Usage: ./run_stage0.sh [first_step]   (steps are idempotent; caches make re-runs cheap)
set -u
FIRST=${1:-1}
cd /home/dylan/asymetric-dual-encoders/m7src
PY=../.venv/bin/python
export HF_HUB_ENABLE_HF_TRANSFER=1
L=../logs
N=7
step () {
  local i=$1; shift
  local title=$1; shift
  if [ "$i" -lt "$FIRST" ]; then echo "--- skip $i/$N $title" | tee -a $L/stage0.log; return 0; fi
  echo "=========== $i/$N $title  $(date -Is) ===========" | tee -a $L/stage0.log
  "$@" >> $L/stage0.log 2>&1
  local rc=$?
  echo "--- exit $rc  $(date -Is)  mem: $(free -m | awk '/Mem:/{print $3"/"$2" MB used"}')" | tee -a $L/stage0.log
  return $rc
}
step 1 "encode dev corpora (resumable)"   $PY encode_dev.py            || exit 1
step 2 "freeze m7 assets"                 $PY freeze_m7_assets.py      || exit 1
step 3 "decontaminate pairs"              $PY decontam.py              || exit 1
step 4 "decontaminate query-text sources" $PY decontam_querytext.py    || exit 1
step 5 "objective-by-dataset field table" $PY field_table.py           || exit 1
step 6 "build frozen doc-vector pool"     $PY pool.py                  || exit 1
step 7 "dev reference rows (all four)"    $PY dev_eval.py              || exit 1
echo "STAGE0 DRIVER COMPLETE $(date -Is)" | tee -a $L/stage0.log
