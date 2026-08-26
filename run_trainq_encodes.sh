#!/usr/bin/env bash
# TRAIN query vectors for the two live teacher candidates, for the learnability probe in
# m7/STATUS.md. Queries only -- no pool, no documents. Strictly sequential.
set -u
cd /home/dylan/asymetric-dual-encoders
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for enc in arctic-embed-l stella-400M-v5; do
  echo "=========== $enc  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/encode_trainq.py encode || echo "FAILED $enc"
done
echo "TRAINQ ENCODES COMPLETE $(date -Is)"
