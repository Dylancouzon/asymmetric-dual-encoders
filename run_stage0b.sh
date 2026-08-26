#!/usr/bin/env bash
# Stage 0.2 + the go/no-go gate. Sequential, one GPU job at a time.
set -u
cd /home/dylan/asymetric-dual-encoders/m7src
PY=../.venv/bin/python
export HF_HUB_ENABLE_HF_TRANSFER=1
L=../logs
step () {
  echo "=========== $1  $(date -Is) ===========" | tee -a $L/stage0b.log
  shift
  "$@" >> $L/stage0b.log 2>&1
  local rc=$?
  echo "--- exit $rc  $(date -Is)  mem: $(free -m | awk '/Mem:/{print $3"/"$2" MB used"}')" | tee -a $L/stage0b.log
  return $rc
}
step "1/3 capacity probe (diagnostic, gate-ineligible)" $PY capacity_probe.py noprefix 6000 || exit 1
step "2/3 objective grid A/B/C"  $PY -c "import program, json; program.save('p1', program.phase1_objective())" || exit 1
step "3/3 go/no-go gate"         $PY gate.py p1-objC p1-objB || exit 1
echo "STAGE0B COMPLETE $(date -Is)" | tee -a $L/stage0b.log
