#!/usr/bin/env bash
# The mandatory ablations, rebuilt after Codex review #3: seven chains of TWO runs each (a B run,
# then a fresh A run from that exact checkpoint), fixed at the surviving candidate's step counts,
# no per-arm step selection. Plus the two attribution controls that decide whether the lever-#2
# gain can be credited to pseudo-query COVERAGE at all, and one labelled exploratory chain.
#
# Strictly sequential and GPU-exclusive (flock), per the OOM incident in m7/LEDGER.md.
# Every arm lands in m7/RESULTS.md whatever it says -- that is what "mandatory" means here.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTHONPATH=m7src
PY=.venv/bin/python
LOG=logs/ablations.log

echo "=========== ablations $(date -Is) ===========" >> "$LOG"

# Smoke the two-run chain path first, and smoke it on the arm with NO execution history: the
# runtime-prefix arm, where `init_preproc` makes the runtime tokenization differ from the rule the
# teacher rows were built under. A grid's arms share code, so one crash here is nine crashes later.
$PY -u - >> "$LOG" 2>&1 <<'EOF'
import program, sweep
surv, b, a = program.ablation_recipe()
sweep.smoke_chain(program.BASE,
                  {**b, "preproc": "prefix", "init_preproc": "noprefix"},
                  {**a, "preproc": "prefix", "init_preproc": "noprefix"})
EOF
if ! tail -40 "$LOG" | grep -q "SMOKE CHAIN ok"; then
  echo "SMOKE FAILED -- not launching the ablations. See $LOG" | tee -a "$LOG"
  exit 1
fi

# The attribution controls run FIRST: they are what the report's causal claim depends on, and if
# the night is cut short the mandatory ablations are the ones that can be described as pending.
$PY -u - >> "$LOG" 2>&1 <<'EOF'
import json
import program
from _paths import REPO

surv, b, a = program.ablation_recipe()
print(f"surviving candidate {surv}: B {b['steps_b']} -> A {a['steps_a']}", flush=True)

res = {}
res.update(program.phase4_attribution(program.BASE))
program.save("phase4_attribution", res)

res2 = program.phase4_mandatory(program.BASE)
program.save("phase4_mandatory", res2)

# The mandated negatives ablation, which phase2_negatives was written for and never ran.
res4 = program.phase4_negatives(program.BASE)
program.save("phase4_negatives", res4)

res3 = program.phase4_exploratory(program.BASE)
program.save("phase4_exploratory", res3)

print(json.dumps({"attribution": res, "mandatory": res2, "negatives": res4,
                  "exploratory": res3}, indent=1), flush=True)
EOF

echo "=========== ablations done $(date -Is) ===========" >> "$LOG"
