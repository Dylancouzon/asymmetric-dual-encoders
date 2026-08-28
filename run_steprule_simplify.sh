#!/usr/bin/env bash
# Two pre-registered pieces of work, in one sequential driver (LEDGER.md, "The step-selection rule
# was NOT applied to the negatives arms" and "Recipe simplification"):
#
#   1. The three promoted negatives arms re-run at their own best proxy step (1500/1500/1000).
#      `warmup_linear` decays over steps_a, so these are real re-runs, not checkpoints of the
#      2500-step versions. The `bank` control peaks at 2500 and is unchanged.
#   2. The one simplification arm: init=input_emb, 500k pseudo-queries, no IDF seeding, no
#      reg_init -- every ablation-inert component removed at once, as a B chain plus a fresh A.
#
# ONE PROCESS PER LEG (run_arm.py), strictly sequential: the ablation night reached 24.7 GB RSS on
# a 25 GB box running arms in one process. Idempotent -- an arm whose artifact exists is skipped.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTHONPATH=m7src
PY=.venv/bin/python
LOG=logs/steprule_simplify.log

echo "=========== step-rule corrections + simplification $(date -Is) ===========" >> "$LOG"

leg () {   # phase suffix leg
  local rid="$1-$2-$3"
  if [ -f "work/runs/${rid}.npz" ]; then
    echo "--- skip ${rid} (already trained) ---" >> "$LOG"
    return 0
  fi
  echo -e "\n----------- ${rid} $(date -Is) -----------" >> "$LOG"
  $PY -u m7src/run_arm.py "$1" "$2" "$3" >> "$LOG" 2>&1
  local rc=$?
  echo "--- ${rid} exit ${rc} ---" >> "$LOG"
  free -g | sed -n 2p >> "$LOG"
  return $rc
}

# Smoke the path with NO execution history. That is the p5s B leg: init=input_emb has run, and
# idf_init_weights=False has run, and reg_init=0.0 has run, but never together -- and with
# reg_init=0 the W0 anchor is built and then never used, a branch combination no arm has taken.
# The A leg is smoked in the same chain because it is where the B artifact is loaded back.
if ! grep -q "SMOKE CHAIN ok" "$LOG"; then
  echo -e "\n----------- smoke p5s-simple $(date -Is) -----------" >> "$LOG"
  $PY -u - >> "$LOG" 2>&1 <<'EOF'
import program, sweep
surv, b, a = program.ablation_recipe()
spec = program.P5S_ARMS["simple"]
sweep.smoke_chain(program.BASE, {**b, **spec["b"]}, {**a, **spec["a"]})
EOF
  if ! grep -q "SMOKE CHAIN ok" "$LOG"; then
    echo "SMOKE FAILED -- not launching. See $LOG" | tee -a "$LOG"
    exit 1
  fi
fi

# 1. Step-rule corrections. A-only arms from the candidate's own B checkpoint, so they cost ~5 min
#    each and the control needs no re-run.
for a in teacher16-s1500 bm2516-s1500 mixed32-s1000; do leg p4n "$a" a; done

# 2. The simplification chain. Its A-phase step count is left at the base recipe's 2500 here; the
#    step-selection rule is applied to its own proxy curve afterwards, like any other arm, and the
#    re-run at the selected step is a separate invocation so that the selection is visible.
leg p5s simple b && leg p5s simple a

echo "=========== step-rule + simplification done $(date -Is) ===========" >> "$LOG"
$PY -u - >> "$LOG" 2>&1 <<'EOF'
import json
from _paths import WORK
for rid in ("p4n-teacher16-s1500-a", "p4n-bm2516-s1500-a", "p4n-mixed32-s1000-a",
            "p5s-simple-b", "p5s-simple-a"):
    p = WORK / "runs" / f"{rid}.json"
    if not p.exists():
        print(f"{rid}: MISSING")
        continue
    d = json.loads(p.read_text())
    pts = [(e["step"], round(e["macro"], 5)) for e in d["history"] if e["phase"] in ("A", "B")]
    print(f"{rid}: final {d['final_macro']:.5f}  curve {pts}")
EOF
tail -8 "$LOG"
