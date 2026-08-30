#!/usr/bin/env bash
# B3's scoring and verdict, as one command. Run AFTER `b3_pool.py train` has finished all 9 arms
# (the three f=1.00 arms are the adopted floor arms and are already on disk).
#
# Only int8 / sqrt is scored, because that is what B3's bar reads -- the release format and R0's
# adopted pooling rule. Scoring both precisions and both pooling modes would quadruple the variant
# count for three quantities the bar never looks at, and the pool component already peaks at
# 15.7 GB RSS with 12 variants on a 25 GB box.
#
# Order matters: `b3_decide.py` refuses to score arms that `collect()` says are not a dose curve,
# so a missing sidecar or a broken nesting stops this before a number is ever produced.
set -euo pipefail
cd "$(dirname "$0")/.."
export M7_ENCODER=stella-400M-v5

ARMS=(m8b3-p025-s0 m8b3-p025-s1 m8b3-p025-s2
      m8b3-p050-s0 m8b3-p050-s1 m8b3-p050-s2
      m8b3-p075-s0 m8b3-p075-s1 m8b3-p075-s2
      m8nf-seed0   m8nf-seed1   m8nf-seed2)

echo "=== 0. are these a dose curve at all? ==="
.venv/bin/python m8src/b3_pool.py collect

echo "=== 1. dense: compare_full over the 12 sqrt variants (~20 min) ==="
SQRT=("${ARMS[@]/%/:sqrt}")
PYTHONPATH=m7src:bench .venv/bin/python -u m7src/compare_full.py m8b3 "${SQRT[@]}"

echo "=== 2. fused: frozen operator, int8/sqrt only (~8 min) ==="
.venv/bin/python -u m8src/fused_floor.py \
  --arms "${ARMS[@]}" --seed-arms m8nf-seed0 m8nf-seed1 m8nf-seed2 \
  --modes sqrt --precisions int8 --out m8_b3_fused.json

echo "=== 3. the verdict, as code ==="
.venv/bin/python m8src/b3_decide.py \
  --dump results/m7_devperquery_m8b3.json.gz \
  --fused results/m8_b3_fused.json
