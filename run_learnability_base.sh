#!/usr/bin/env bash
# Teacher learnability for the two BASE-SIZED shortlist survivors the 2026-08-26 sweep never ran
# through the adopted criterion: arctic-embed-m-v1.5 and gte-base-en-v1.5.
#
# Both were dismissed on MTEB v1 ordering -- the criterion this project subsequently REFUTED
# (Spearman(ceiling, table) = 0.000 over eight candidates) -- and the refutation was applied only
# to the rows a reviewer named. EXPLORED.md records that as an open item; LEDGER.md's
# "Teacher re-examination" fixes the swap bar, the tie-break and the cost before any number.
#
# Measurement only. A swap is Dylan's call, costs an ~8-12 h re-encode, re-adjudicates levers
# #4/#5/#6, and -- per the ONE-ACCESS RULE -- may only happen BEFORE the freeze and the final run.
#
# Order per candidate is the one CODEMAP.md mandates and that exists because a Spec once silently
# omitted stella's published Dense head: validate_encoder.py FIRST, then encodes, then the probe.
# A candidate that fails validation is skipped, loudly -- a probe on an unvalidated loader ranks
# the wrong model.
set -u
cd /home/dylan/asymetric-dual-encoders
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/learnability_base.log
exec >> "$LOG" 2>&1
echo "=========== learnability (base-sized candidates) $(date -Is) ==========="

# No wait loop: launch this under `flock -n /tmp/m7.gpu.lock`, which is the one mechanism that
# actually serialises GPU work here. The older drivers polled `pgrep` instead, and CODEMAP 4
# records two ways that goes wrong -- a pattern matching the shell that wrote it, and a driver
# that execs python leaving no script name in any cmdline.

for enc in arctic-embed-m-v1.5 gte-base-en-v1.5; do
  echo "=========== $enc validate  $(date -Is) ==========="
  PYTHONPATH=m7src .venv/bin/python -u m7src/validate_encoder.py "$enc" \
      || { echo "FAILED validation $enc -- SKIPPING; an unvalidated loader ranks the wrong model"; continue; }
  echo "=========== $enc trainq  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/encode_trainq.py encode \
      || { echo "FAILED trainq $enc"; continue; }
  echo "=========== $enc dev encodes  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u m7src/encode_dev.py cqadup-programmers cqadup-physics \
      || { echo "FAILED dev $enc"; continue; }
  echo "=========== $enc learnability  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/teacher_learnability.py 1e-4 1e-3 1e-2 1e-1 \
      || echo "FAILED learnability $enc"
done

echo "=========== report (INCUMBENT must be stella)  $(date -Is) ==========="
.venv/bin/python -u scripts/learnability_report.py || echo "FAILED report"
echo "LEARNABILITY BASE COMPLETE $(date -Is)"
