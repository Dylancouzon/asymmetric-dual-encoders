#!/usr/bin/env bash
# Everything queued behind the ablation night, in priority order, each pre-registered in
# m7/LEDGER.md BEFORE its numbers exist. Strictly sequential and GPU-exclusive.
#
#  1. attribution on the FULL suite  -- the proxy decomposition of lever #2 is not the statistic
#     any decision uses; this is.
#  2. capacity lever #6 arm (a)      -- train the A phase THROUGH the adopted sqrt rule. Smoked
#     first: the training forward has never run with pooling weights.
#  3. long-span teacher agreement    -- diagnostic, decides whether a long-span chain is worth
#     buying before it costs one.
#  4. capacity lever #5              -- update-count row shrinkage, eval-only.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTHONPATH=m7src
PY=.venv/bin/python
LOG=logs/post_ablation.log
echo "=========== post-ablation $(date -Is) ===========" >> "$LOG"

step () { echo -e "\n----------- $1 $(date -Is) -----------" >> "$LOG"; }

# 1. Attribution, full suite. All three artifacts served under the ADOPTED sqrt rule so the
#    comparison is like-for-like and matches what ships; all three were trained mean-pooled.
step "attribution (full suite)"
$PY -u m7src/compare_full.py attrib p35w-2m-s2500 \
    p4x-nopseudo-a:sqrt p4x-pseudo500k-a:sqrt >> "$LOG" 2>&1

# 2. Lever #6 arm (a): A-phase only, through the sqrt rule, from the candidate's own B checkpoint.
step "lever 6 smoke"
$PY -u - >> "$LOG" 2>&1 <<'EOF'
import program, sweep
surv, b, a = program.ablation_recipe()
sweep.smoke_chain(program.BASE, {**b, "pool_mode": "sqrt"}, {**a, "pool_mode": "sqrt"})
EOF
if tail -40 "$LOG" | grep -q "SMOKE CHAIN ok"; then
  step "lever 6 arm (a)"
  $PY -u - >> "$LOG" 2>&1 <<'EOF'
import json, program, sweep
from _paths import WORK
surv, b, a = program.ablation_recipe()
bid = json.loads((WORK / "runs" / f"{surv}.json").read_text())["cfg"]["init"]
print(f"lever 6(a): A through pool_mode=sqrt from {bid}", flush=True)
sweep.one("p4p-sqrt-a", program.BASE, init=bid, pool_mode="sqrt", **a)
EOF
  $PY -u m7src/compare_full.py lever6 p35w-2m-s2500 p4p-sqrt-a >> "$LOG" 2>&1
else
  echo "lever 6 SMOKE FAILED -- skipping the arm" >> "$LOG"
fi

# 3. Long-span teacher agreement (diagnostic, no qrels).
step "long-span probe"
$PY -u m7src/longspan_probe.py --smoke >> "$LOG" 2>&1
$PY -u m7src/longspan_probe.py >> "$LOG" 2>&1

# 4. Capacity lever #5.
step "lever 5 smoke"
$PY -u m7src/lever5_shrinkage.py --smoke >> "$LOG" 2>&1
step "lever 5"
$PY -u m7src/lever5_shrinkage.py >> "$LOG" 2>&1

echo "=========== post-ablation done $(date -Is) ===========" >> "$LOG"
