#!/usr/bin/env bash
# Learnability for the remaining registry candidates. The symmetric-ceiling probe ranked arctic
# first and its closed-form TABLE lands below the incumbent's, so ceiling does not predict the
# quantity that ships and every candidate needs measuring on this axis before the teacher is chosen.
# Per candidate: TRAIN query vectors, the two CQADupStack dev components, then the ridge grid.
# Strictly sequential; the wait loop is anchored to a python cmdline, not a script name (CODEMAP 4).
set -u
cd /home/dylan/asymetric-dual-encoders
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "^[^ ]*python[0-9.]* -u scripts/teacher_learnability" >/dev/null; do sleep 15; done

for enc in gte-large-en-v1.5 bge-large-en-v1.5; do
  echo "=========== $enc trainq  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/encode_trainq.py encode || { echo "FAILED trainq $enc"; continue; }
  echo "=========== $enc dev encodes  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u m7src/encode_dev.py cqadup-programmers cqadup-physics \
      || { echo "FAILED dev $enc"; continue; }
  echo "=========== $enc learnability  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/teacher_learnability.py 1e-4 1e-3 1e-2 \
      || echo "FAILED learnability $enc"
done
echo "LEARNABILITY ALL COMPLETE $(date -Is)"
