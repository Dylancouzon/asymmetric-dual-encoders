#!/usr/bin/env bash
# DIAGNOSTIC (m7/LEDGER.md, "RECIPE-PERTURBATION SPREAD"): how much does a nuisance step count
# move the dev macro? Three negatives arms at both step counts, all served under MATCHED `mean`
# pooling -- the 2500-step versions have only been scored sqrt-served, so the pair that exists is
# confounded by the pooling rule and cannot answer this.
#
# Cannot change any adoption. The negatives avenue is closed; this only prices the noise floor
# that every bar in the ledger is read against.
set -eu
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5 PYTHONPATH=m7src
LOG=logs/stepspread.log
echo "=========== step-count spread $(date -Is) ===========" >> "$LOG"
.venv/bin/python -u m7src/compare_full.py stepspread p35w-2m-s2500:mean \
    p4n-teacher16-a p4n-teacher16-s1500-a \
    p4n-bm2516-a:mean p4n-bm2516-s1500-a \
    p4n-mixed32-a:mean p4n-mixed32-s1000-a >> "$LOG" 2>&1
echo "=========== step-count spread done $(date -Is) ===========" >> "$LOG"
tail -12 "$LOG"
