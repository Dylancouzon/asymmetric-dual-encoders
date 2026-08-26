#!/usr/bin/env bash
# Test the pooling hypothesis: does a MEAN-pooled teacher get approximated better by a lookup table?
# Per candidate: loader validation, symmetric ceiling on the two dev components, TRAIN query vectors,
# then the ridge grid. Strictly sequential; wait loop anchored to a python cmdline (CODEMAP 4).
#
# arctic-embed-l-mean is expected to FAIL validate_encoder.py -- it is a deliberate off-spec read-out
# of a CLS-trained tower, and sentence-transformers implements the published CLS pipeline. The loop
# therefore reports validation instead of gating on it, and the Spec records why.
set -u
cd /home/dylan/asymetric-dual-encoders
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "^[^ ]*python[0-9.]* -u (scripts/(teacher_learnability|encode_trainq)|m7src/encode_dev)" \
      >/dev/null; do sleep 15; done

for enc in e5-large-v2 e5-base-v2 arctic-embed-l-mean; do
  echo "=========== $enc validate  $(date -Is) ==========="
  (cd m7src && M7_ENCODER=$enc ../.venv/bin/python -u validate_encoder.py "$enc") \
      || echo "VALIDATION FAILED $enc (expected for the off-spec mean read-out)"
  echo "=========== $enc ceiling  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u m7src/encode_dev.py cqadup-programmers cqadup-physics \
      || { echo "FAILED dev $enc"; continue; }
  (cd m7src && M7_ENCODER=$enc ../.venv/bin/python -u teacher_probe.py "$enc") \
      || echo "FAILED probe $enc"
  echo "=========== $enc trainq + learnability  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/encode_trainq.py encode \
      || { echo "FAILED trainq $enc"; continue; }
  M7_ENCODER=$enc .venv/bin/python -u scripts/teacher_learnability.py 1e-4 1e-3 1e-2 \
      || echo "FAILED learnability $enc"
done
echo "LEARNABILITY MEAN COMPLETE $(date -Is)"
