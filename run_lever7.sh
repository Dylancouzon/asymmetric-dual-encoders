#!/usr/bin/env bash
# Capacity lever #7, long-span distillation. Pre-registered in m7/LEDGER.md before any number,
# including the part that makes it falsifiable: the PRIMARY bar is the length probe, and the dev
# suite is only a veto, because four of six dev components are short-query and would dilute
# exactly the effect being bought.
#
# Takes the current candidate's run id as $1 -- the arm is compared against it on both bars.
#
#   ./run_lever7.sh p4n-teacher16-s1500-a
#
# Cost, in order: a mixed pseudo-query pool (CPU, reads the doc stores one at a time), R1
# decontamination of every pool (CPU), a teacher encode of 500k spans half of which are 64-320
# words (GPU, the expensive step), a 16,000-step B leg and an A leg, then two adjudications.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTHONPATH=m7src
PY=.venv/bin/python
CAND="$1"
# The arm is "the candidate plus long spans": every other knob comes from the candidate's own
# committed config, so whichever way the simplification test and the negatives tie-break went,
# this comparison isolates the span distribution and nothing else.
export M7_RECIPE_FROM="$CAND"
LOG=logs/lever7.log
echo "=========== lever 7 (long-span) vs ${CAND} $(date -Is) ===========" >> "$LOG"
step () { echo -e "\n----------- $1 $(date -Is) -----------" >> "$LOG"; }

# 1. The pool. Half first-sentence <=32-word spans, half sentence-aligned 64-320-word ones.
#    Deterministic in SEED, so a rerun reuses it and the teacher encode cache stays valid.
step "build mixed pseudo-query pool"
$PY -u - >> "$LOG" 2>&1 <<'EOF'
import numpy as np, program, pseudoq
# the SIZE comes from the candidate's own recipe too: 500k if the simplification was accepted,
# 2m if it was not. Hardcoding it would change two things at once.
_, b, _ = program.ablation_recipe()
n = b["b_pseudo_queries"]
assert n, "the candidate's B phase uses no pseudo-queries; lever #7 has nothing to lengthen"
qs = pseudoq.build(n, kind="mixed")
w = np.array([len(q.split()) for q in qs])
print(f"pool {len(qs):,} (n={n:,})  words p5/p50/p95 {np.percentile(w,[5,50,95]).round(1)}  "
      f"share >= {pseudoq.LONG_MIN_WORDS} words: {(w >= pseudoq.LONG_MIN_WORDS).mean():.3f}")
EOF

# 2. R1. A long span carries more word-8-grams than a short one and so matches the protected-query
#    index more often; the counts land in results/m7_decontam_querytext.json like every source's.
step "decontaminate (R1) -- rewrites kept-*.json for every pool"
$PY -u m7src/decontam_querytext.py >> "$LOG" 2>&1

# 3. Smoke. The mixed pool has never been tokenized, encoded or trained on; 512-token truncation
#    on a 320-word span is a path with no execution history.
step "smoke"
$PY -u - >> "$LOG" 2>&1 <<'EOF'
import program, sweep
surv, b, a = program.ablation_recipe()     # M7_RECIPE_FROM -> the candidate's own recipe
spec = program.P7_ARMS["longspan"]
sweep.smoke_chain(program.BASE, {**b, **spec["b"]}, {**a, **spec.get("a", {})})
EOF
if ! tail -60 "$LOG" | grep -q "SMOKE CHAIN ok"; then
  echo "SMOKE FAILED -- not launching lever 7. See $LOG" | tee -a "$LOG"; exit 1
fi

# 4. Train. One process per leg (CODEMAP 14).
for leg in b a; do
  rid="p7-longspan-${leg}"
  if [ -f "work/runs/${rid}.npz" ]; then echo "--- skip ${rid} ---" >> "$LOG"; continue; fi
  step "$rid"
  $PY -u m7src/run_arm.py p7 longspan "$leg" >> "$LOG" 2>&1 || { echo "FAILED $rid" >> "$LOG"; exit 1; }
done

# 5. Primary bar: the length probe, paired per span, pooled 128- and 256-word buckets.
step "lever 7 primary bar (length probe)"
$PY -u m7src/longspan_probe.py "$CAND" p7-longspan-a >> "$LOG" 2>&1

# 6. Guardrail: the full dev suite must be non-inferior at the same -0.0040 margin. It can veto
#    on its own; passing it is not adoption.
step "lever 7 guardrail (full dev suite)"
$PY -u m7src/compare_full.py lever7 "$CAND" p7-longspan-a >> "$LOG" 2>&1

echo "=========== lever 7 done $(date -Is) ===========" >> "$LOG"
tail -25 "$LOG"
