#!/usr/bin/env bash
# Strictly sequential. The 2026-08-25 WSL OOM came from running three memory-heavy jobs at once;
# CLAUDE.md already recorded that lesson from the M4 incident and it is now enforced here.
set -u
cd /home/dylan/asymetric-dual-encoders/m7src
PY=../.venv/bin/python
export HF_HUB_ENABLE_HF_TRANSFER=1
L=../logs
step () {
  echo "=========== $1  $(date -Is) ===========" | tee -a $L/stage0.log
  shift
  "$@" >> $L/stage0.log 2>&1
  local rc=$?
  echo "--- exit $rc  $(date -Is)  mem: $(free -m | awk '/Mem:/{print $3"/"$2" MB used"}')" | tee -a $L/stage0.log
  return $rc
}
step "1/7 encode dev corpora (resumable)"   $PY encode_dev.py            || exit 1
step "2/7 freeze m7 assets"                 $PY freeze_m7_assets.py      || exit 1
step "3/7 decontaminate pairs"              $PY decontam.py              || exit 1
step "4/7 decontaminate query-text sources" $PY decontam_querytext.py    || exit 1
step "5/7 build frozen doc-vector pool"     $PY pool.py                  || exit 1
step "6/7 dev reference rows (all four)"    $PY dev_eval.py              || exit 1
step "7/7 stage-0 ridge probe"              $PY stage0_ridge.py teacher noprefix || exit 1
echo "STAGE0 DRIVER COMPLETE $(date -Is)" | tee -a $L/stage0.log
