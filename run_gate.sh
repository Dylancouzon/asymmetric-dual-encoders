#!/usr/bin/env bash
# The go/no-go gate on both named checkpoints: p1-objC (named in the driver before any result
# existed) and p1-objB (the better checkpoint from the same TRAIN data). Both reported; the
# substitution is logged in m7/LEDGER.md.
set -u
cd /home/dylan/asymetric-dual-encoders/m7src
PY=../.venv/bin/python
L=../logs
for pair in "p1-objC p1-objB" "p1-objB p1-objB"; do
  set -- $pair
  echo "=========== gate candidate=$1 stage0=$2  $(date -Is) ===========" | tee -a $L/gate.log
  $PY gate.py "$1" "$2" >> $L/gate.log 2>&1
  echo "--- exit $?  $(date -Is)  mem: $(free -m | awk '/Mem:/{print $3"/"$2" MB"}')" | tee -a $L/gate.log
done
echo "GATE RUNS COMPLETE $(date -Is)" | tee -a $L/gate.log
