#!/usr/bin/env bash
# Everything between "the recipe is final" and "freeze". Takes the final run id.
#
#   ./run_freeze_prep.sh p35w-2m-s2500
#
# NOT chained to the lever queue on purpose. Fusion must be re-selected on whatever artifact the
# levers leave standing, and the gate is a one-way door -- review #3 is explicit that the recipe
# may not change after the gate has been seen. So a human decides the run id before this starts.
#
# This script deliberately stops BEFORE freeze.write() and before the final run: the freeze commit
# and the single confirmatory access to the six are Dylan's calls.
set -eu
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTHONPATH=m7src
PY=.venv/bin/python
RUN_ID="$1"
LOG=logs/freeze_prep.log
echo "=========== freeze prep for ${RUN_ID} $(date -Is) ===========" >> "$LOG"

step () { echo -e "\n----------- $1 $(date -Is) -----------" >> "$LOG"; }

# 1. Fusion re-selection. Fitted against the RELEASE int8 artifact, at DEPTH=1000, on the four
#    text-backed components (BM25 has no run on the held-out slices). The convex grid now carries
#    w=1.0, the dense-only endpoint, so "do not fuse" can win the same selection -- and
#    `released_system` is DERIVED from which point wins, not chosen later at freeze time.
#    The spec records the artifact hashes, preproc fingerprint, encoder identity and the BM25
#    cache keys it was fitted against; freeze.write re-derives all of them and refuses a mismatch.
#    The BM25 caches are content-keyed as of 2026-08-28, so any cache written before that is
#    rebuilt here (~30 min, dominated by HotpotQA's 5.23M documents).
step "fusion re-selection"
$PY -u m7src/select_fusion.py "$RUN_ID" >> "$LOG" 2>&1

# 2. ANN behaviour on real HNSW, and the three cost numbers. The ANN sweep reads the query rule
#    from the artifact's own metadata, so it exercises the adopted pooling rule.
step "ann sweep"
$PY -u m7src/ann_sweep.py "${RUN_ID}.release" >> "$LOG" 2>&1 || \
  echo "ann sweep failed (non-fatal for the gate)" >> "$LOG"
step "costs"
$PY -u m7src/costs.py "${RUN_ID}.release" >> "$LOG" 2>&1 || echo "costs failed" >> "$LOG"

# 3. The gate, as a mechanical eligibility audit: pinned six verified including the pool's bytes,
#    released QueryTable path, strict alignment, dependence-aware int8 bound, unrounded per-query
#    dumps, plus the exploratory audit against the pre-lever winner.
step "gate (eligibility audit)"
# s1-objB is the STELLA-era Stage-0 distilled table G1 is defined on, and is what GO #2 used.
# This line said `p1-objB` -- the BGE-era one -- which is a 768-d table from a document space
# this project left on 2026-08-26. It got here as the fix for an earlier argv bug that fed
# `s2w-1e3-s1000` in as stage0_id: the omission was corrected with the wrong id. The gate now
# also refuses any checkpoint whose teacher is not the active encoder, and exits nonzero on
# NO-GO so `set -e` stops here instead of continuing toward the freeze.
$PY -u m7src/gate.py "$RUN_ID" s1-objB --audit-vs s2w-1e3-s1000 >> "$LOG" 2>&1

echo "=========== freeze prep done $(date -Is) ===========" >> "$LOG"
echo "NEXT, BY HAND: review results/m7_gate_${RUN_ID}.json, then" >> "$LOG"
echo "  freeze.write('${RUN_ID}')  -- it loads the fusion selection and the gate result itself;" >> "$LOG"
echo "  the fusion spec and released_system are NOT arguments -- then commit + push the tag," >> "$LOG"
echo "then the single final run. Both are Dylan's calls." >> "$LOG"
