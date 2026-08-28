#!/usr/bin/env bash
# The simplification arm that faces the bar, after the negatives closure changed the baseline.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5 PYTHONPATH=m7src
PY=.venv/bin/python
LOG=logs/simplify.log
echo "=========== simplification (nohn) $(date -Is) ===========" >> "$LOG"
if [ ! -f work/runs/p5s-simple-nohn-a.npz ]; then
  $PY -u m7src/run_arm.py p5s simple-nohn a >> "$LOG" 2>&1 || { echo "FAILED arm" >> "$LOG"; exit 1; }
fi
# Baseline served under its own frozen rule (sqrt, adopted on this artifact and only this one);
# the simplified arm under its own (mean). Each as it would ship.
$PY -u m7src/compare_full.py simplify p35w-2m-s2500 p5s-simple-nohn-a p5s-simple-a >> "$LOG" 2>&1
echo "=========== simplification done $(date -Is) ===========" >> "$LOG"
tail -8 "$LOG"
