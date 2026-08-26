#!/usr/bin/env bash
# Extension: lambda=1e-4 for both live candidates (their best was 1e-3, the first grid's lower
# EDGE), plus bge-base across the grid as the reference point -- "is a new teacher more or less
# table-approximable than the one every committed number was produced with" is the question the
# two-candidate comparison cannot answer. Waits for the first probe rather than sharing the GPU.
# Per-lambda results merge into the existing JSONs, so nothing already paid for is recomputed.
set -u
cd /home/dylan/asymetric-dual-encoders
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Anchored to a PYTHON cmdline. `pgrep -f "[t]eacher_learnability.py"` matched the shell that
# WROTE this script -- the heredoc text is in that shell's own cmdline -- so the loop never ended
# and the job sat idle. The bracket trick only hides the pattern from itself, not from every other
# process carrying the same string. ^[^ ]*python excludes /bin/bash cmdlines.
while pgrep -f "^[^ ]*python[0-9.]* -u scripts/teacher_learnability" >/dev/null; do sleep 15; done

for enc in arctic-embed-l stella-400M-v5; do
  echo "=========== ext $enc lam 1e-4  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/teacher_learnability.py 1e-4 || echo "FAILED $enc"
done
echo "=========== reference bge-base-en-v1.5  $(date -Is) ==========="
.venv/bin/python -u scripts/teacher_learnability.py 1e-4 1e-3 1e-2 || echo "FAILED bge-base"
echo "LEARNABILITY EXT COMPLETE $(date -Is)"
