#!/usr/bin/env bash
# DIAGNOSTIC (m7/LEDGER.md, under "Recipe simplification"): score the seven MANDATORY ablations on
# the full pinned dev suite. They have only ever been reported on the three-component proxy, and
# that instrument failed twice today -- its peak did not reproduce on re-run, and it inverted the
# full-suite ordering of three arms. Every artifact already exists, so this trains nothing.
#
# Cannot change the released recipe: the simplification test already decided it. This makes the
# mandate's ablation table read on the suite that decisions actually use, and decomposes the
# simplification's -0.0048 into single-knob effects.
#
# Baseline at :mean to match the arms, which all trained mean-pooled. p4-base-a is the replay of
# the candidate's own recipe and doubles as a determinism check.
set -eu
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5 PYTHONPATH=m7src
LOG=logs/ablations_full.log
echo "=========== mandatory ablations, full suite $(date -Is) ===========" >> "$LOG"
.venv/bin/python -u m7src/compare_full.py ablations p35w-2m-s2500:mean \
    p4-base-a p4-input-emb-a p4-random-a p4-prefix-a p4-flat-a p4-uniform-w-a p4-reg0-a \
    p4e-prefix-init-a >> "$LOG" 2>&1
echo "=========== mandatory ablations done $(date -Is) ===========" >> "$LOG"
tail -14 "$LOG"
