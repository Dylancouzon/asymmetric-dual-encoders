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
#   * a venv with the repo's requirements (torch with MPS, transformers, scipy, datasets). PY
#     overrides which one; the default matches the lock except for torch's build.
#   * the TRAIN query list, either as work/trainq_texts.json or as the gzipped transfer copy this
#     branch ships. `encode_trainq.load_texts` restores it and verifies its sha256 either way.
# Everything else (dev corpora, encodes) is derived here from HF.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Both watermarks, or MPS crashes rather than spilling -- the M2-era incident in CLAUDE.md.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
PY=${PY:-.venv-mac/bin/python}
LOG=logs/learnability_mac.log
mkdir -p logs
echo "=========== learnability (mac) $(date -Is) ===========" >> "$LOG"

if [ ! -f work/trainq_texts.json ] && [ ! -f transfer/trainq_texts.json.gz ]; then
  echo "MISSING the TRAIN query list: neither work/trainq_texts.json nor the transfer copy" | tee -a "$LOG"
  echo "transfer/trainq_texts.json.gz is present. Re-deriving it here means running most of" | tee -a "$LOG"
  echo "run_stage0.sh, which needs the pool index this machine does not have." | tee -a "$LOG"
  exit 1
fi

# Nothing else records the environment for these artifacts, and the stella row is a cross-platform
# replication check whose only meaning is against a stated toolchain.
echo "=========== environment  $(date -Is) ===========" >> "$LOG"
$PY -c "import platform, torch, transformers, scipy, numpy, sentence_transformers as st; \
print(f'python {platform.python_version()} {platform.machine()} | torch {torch.__version__} ' \
      f'mps={torch.backends.mps.is_available()} | transformers {transformers.__version__} | ' \
      f'scipy {scipy.__version__} | numpy {numpy.__version__} | sentence-transformers {st.__version__}')" \
    2>&1 | tee -a "$LOG"

# The two dev components carry the probe's queries and qrels. A different HF snapshot here would
# make every Mac row incomparable to the committed stella row, silently.
echo "=========== dev component hashes  $(date -Is) ===========" >> "$LOG"
PYTHONPATH=m7src $PY -u scripts/verify_dev_hashes.py cqadup-programmers cqadup-physics 2>&1 | tee -a "$LOG" \
    || { echo "ABORT: dev components do not match the pin" | tee -a "$LOG"; exit 1; }

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
