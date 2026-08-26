#!/usr/bin/env bash
# Extension: lambda=1e-4 for both live candidates (their best was 1e-3, the first grid's lower
# EDGE), plus bge-base across the grid as the reference point -- "is a new teacher more or less
# table-approximable than the one every committed number was produced with" is the question the
# two-candidate comparison cannot answer. Waits for the first probe rather than sharing the GPU.
# Per-lambda results merge into the existing JSONs, so nothing already paid for is recomputed.
set -u
cd /home/dylan/asymetric-dual-encoders
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "[t]eacher_learnability.py" >/dev/null; do sleep 15; done

for enc in arctic-embed-l stella-400M-v5; do
  echo "=========== ext $enc lam 1e-4  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/teacher_learnability.py 1e-4 || echo "FAILED $enc"
done
echo "=========== reference bge-base-en-v1.5  $(date -Is) ==========="
.venv/bin/python -u scripts/teacher_learnability.py 1e-4 1e-3 1e-2 || echo "FAILED bge-base"
echo "LEARNABILITY EXT COMPLETE $(date -Is)"
