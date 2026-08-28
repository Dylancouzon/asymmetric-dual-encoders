#!/usr/bin/env bash
# Teacher learnability probe, on a SECOND MACHINE (Apple silicon / MPS).
#
# Runs the two registered base-sized candidates plus a stella replication row. Rules are
# pre-registered in m7/LEDGER.md, "RUNNING THE PROBES ON A SECOND MACHINE" -- the stella row is
# mandatory (it makes the ranking internally self-consistent and doubles as a cross-platform
# check), and any winner here is re-probed on the RTX box before it can move anything.
#
# Not runnable for the ModernBERT candidates: `stage0_ridge.solve_ridge` builds the Gram in
# float64, so V=50,368 needs 20.3 GB and does not fit in 24 GB. See the ledger.
#
# Prerequisites, checked below rather than assumed:
#   * .venv with the repo's requirements (torch with MPS, transformers, scipy, datasets)
#   * work/trainq_texts.json  -- 21 MB, COPIED from the RTX box; it is gitignored and is the one
#     artifact this machine cannot re-derive without most of run_stage0.sh
# Everything else (dev corpora, encodes) is derived here from HF.
set -u
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Both watermarks, or MPS crashes rather than spilling -- the M2-era incident in CLAUDE.md.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
PY=.venv/bin/python
LOG=logs/learnability_mac.log
mkdir -p logs
echo "=========== learnability (mac) $(date -Is) ===========" >> "$LOG"

if [ ! -f work/trainq_texts.json ]; then
  echo "MISSING work/trainq_texts.json -- copy it from the RTX box first. It is gitignored, and" | tee -a "$LOG"
  echo "re-deriving it here means running most of run_stage0.sh." | tee -a "$LOG"
  exit 1
fi

# stella LAST: the two candidates are the point, and if the machine dies overnight the
# replication row is the one thing we can still get from the other box.
for enc in arctic-embed-m-v1.5 gte-base-en-v1.5 stella-400M-v5; do
  echo "=========== $enc validate  $(date -Is) ===========" >> "$LOG"
  PYTHONPATH=m7src $PY -u m7src/validate_encoder.py "$enc" >> "$LOG" 2>&1 \
      || { echo "FAILED validation $enc -- SKIPPING; an unvalidated loader ranks the wrong model" | tee -a "$LOG"; continue; }
  echo "=========== $enc trainq  $(date -Is) ===========" >> "$LOG"
  M7_ENCODER=$enc $PY -u scripts/encode_trainq.py encode >> "$LOG" 2>&1 \
      || { echo "FAILED trainq $enc" | tee -a "$LOG"; continue; }
  echo "=========== $enc dev encodes  $(date -Is) ===========" >> "$LOG"
  M7_ENCODER=$enc PYTHONPATH=m7src $PY -u m7src/encode_dev.py cqadup-programmers cqadup-physics >> "$LOG" 2>&1 \
      || { echo "FAILED dev $enc" | tee -a "$LOG"; continue; }
  echo "=========== $enc learnability  $(date -Is) ===========" >> "$LOG"
  M7_ENCODER=$enc $PY -u scripts/teacher_learnability.py 1e-4 1e-3 1e-2 1e-1 >> "$LOG" 2>&1 \
      || echo "FAILED learnability $enc" | tee -a "$LOG"
  echo "--- $enc done $(date -Is) ---" | tee -a "$LOG"
done

echo "=========== report  $(date -Is) ===========" >> "$LOG"
$PY -u scripts/learnability_report.py >> "$LOG" 2>&1 || echo "FAILED report" | tee -a "$LOG"
echo "LEARNABILITY MAC COMPLETE $(date -Is)" | tee -a "$LOG"
grep -E "dev_macro_2|ranking by what ships|vs incumbent" "$LOG" | tail -30
