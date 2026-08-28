#!/usr/bin/env bash
# Re-decide the negatives ablation on the STEP-RULE-CORRECTED arms, in one corpus pass.
#
# The four p4n arms were promoted and full-suite-compared at the inherited steps_a=2500, but the
# pre-registered rule is that an arm's step count is its best proxy eval. LEDGER.md, "The
# step-selection rule was NOT applied to the negatives arms", fixes what happens next, including
# the part that binds: the proxy picked the step, so a corrected arm ships even if it scores lower
# on the full dev suite than the 2500-step version did.
#
# Baseline is `p35w-2m-s2500` -- the candidate the negatives bar was written against -- served
# under its own frozen rule (sqrt, adopted for that artifact only). The corrected arms are served
# under theirs (mean; lever #4 does not survive on them, see m7_lever4_pooling_full.json). Each
# artifact under its own rule is the like-for-like comparison, because that is what would ship.
#
# `p4n-teacher16-a` is included so the step-rule correction's own effect is visible rather than
# inferred from two separately-run passes.
set -eu
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTHONPATH=m7src
PY=.venv/bin/python
LOG=logs/negatives_redecide.log
echo "=========== negatives re-decision $(date -Is) ===========" >> "$LOG"

$PY -u m7src/compare_full.py steprule p35w-2m-s2500 \
    p4n-teacher16-a \
    p4n-teacher16-s1500-a p4n-bm2516-s1500-a p4n-mixed32-s1000-a >> "$LOG" 2>&1

echo "=========== negatives re-decision done $(date -Is) ===========" >> "$LOG"
tail -12 "$LOG"
