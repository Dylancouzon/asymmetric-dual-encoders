#!/usr/bin/env bash
# The mandatory ablations: seven chains of TWO runs each (a B run, then a fresh A run from that
# exact checkpoint), fixed at the surviving candidate's step counts, no per-arm step selection.
# Plus the attribution controls that decide whether lever #2's gain can be credited to
# pseudo-query COVERAGE, the mandate's never-run negatives ablation, and one exploratory chain.
#
# ONE PROCESS PER LEG. Running the whole night in a single python process accumulated this repo's
# deliberate module-level caches on top of each arm's ~8 GB of working set; the third chain hit
# 24.7 GB RSS on a 25 GB box and thrashed (see m7src/run_arm.py). A fresh process per leg also
# means every arm starts from the same memory state, which is what makes them comparable.
#
# Idempotent: an arm whose artifact already exists is skipped, so this can be re-run after an
# interruption. Every arm lands in m7/RESULTS.md whatever it says -- that is what "mandatory" means.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTHONPATH=m7src
PY=.venv/bin/python
LOG=logs/ablations.log

echo "=========== ablations $(date -Is) ===========" >> "$LOG"

leg () {   # phase suffix leg
  local rid="$1-$2-$3"
  if [ -f "work/runs/${rid}.npz" ]; then
    echo "--- skip ${rid} (already trained) ---" >> "$LOG"
    return 0
  fi
  echo -e "\n----------- ${rid} $(date -Is) -----------" >> "$LOG"
  $PY -u m7src/run_arm.py "$1" "$2" "$3" >> "$LOG" 2>&1
  local rc=$?
  echo "--- ${rid} exit ${rc}, peak RSS check ---" >> "$LOG"
  free -g | sed -n 2p >> "$LOG"
  return $rc
}

chain () {  # phase suffix -- B then A; A is skipped if B failed
  leg "$1" "$2" b && leg "$1" "$2" a
}

# Smoke the two-run path first, on the arm with NO execution history: the runtime-prefix arm,
# where `init_preproc` makes runtime tokenization differ from the rule the teacher rows were
# built under. A grid's arms share code, so one crash here is nine crashes later.
if ! grep -q "SMOKE CHAIN ok" "$LOG"; then
  echo -e "\n----------- smoke $(date -Is) -----------" >> "$LOG"
  $PY -u - >> "$LOG" 2>&1 <<'EOF'
import program, sweep
surv, b, a = program.ablation_recipe()
sweep.smoke_chain(program.BASE,
                  {**b, "preproc": "prefix", "init_preproc": "noprefix"},
                  {**a, "preproc": "prefix", "init_preproc": "noprefix"})
EOF
  if ! grep -q "SMOKE CHAIN ok" "$LOG"; then
    echo "SMOKE FAILED -- not launching the ablations. See $LOG" | tee -a "$LOG"
    exit 1
  fi
fi

# Attribution first: the report's causal claim depends on these, so if the night is cut short the
# MANDATORY ablations are the ones that can honestly be described as pending.
for a in nopseudo pseudo500k; do chain p4x "$a"; done

# The seven mandatory chains. `base` is both the nondeterminism replay and the regularization-ON
# control at 1e-3, which is why there is no separate reg-on arm.
for a in base input-emb random prefix flat uniform-w reg0; do chain p4 "$a"; done

# The negatives ablation the mandate ordered and that never ran. A-only from the candidate's own
# B checkpoint, so `bank` IS the candidate and is the control.
for a in bank teacher16 bm2516 mixed32; do leg p4n "$a" a; done

# Labelled exploratory: prefix-CONDITIONED teacher rows.
chain p4e prefix-init

$PY -u - >> "$LOG" 2>&1 <<'EOF'
import json
import program
from _paths import REPO, WORK
out = {}
for phase, arms in program.ARMS.items():
    for suffix in arms:
        for leg in ("b", "a"):
            rid = f"{phase}-{suffix}-{leg}"
            p = WORK / "runs" / f"{rid}.json"
            if p.exists():
                out[rid] = json.loads(p.read_text())["final_macro"]
program.save("phase4_all", out)
print(json.dumps(out, indent=1), flush=True)
EOF

echo "=========== ablations done $(date -Is) ===========" >> "$LOG"
