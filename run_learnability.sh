#!/usr/bin/env bash
# Teacher learnability probe, end to end and STRICTLY SEQUENTIAL: stella's dev encodes (arctic's
# already exist), then the closed-form ridge fit per candidate. One GPU job at a time -- launching
# a second one alongside the TRAIN-query encodes halved its throughput, which is the cheap version
# of the lesson in m7/LEDGER.md.
set -u
cd /home/dylan/asymetric-dual-encoders
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait out anything still holding the GPU from an earlier launch.
while pgrep -f "[e]ncode_trainq.py" >/dev/null; do sleep 10; done

echo "=========== stella dev encodes  $(date -Is) ==========="
M7_ENCODER=stella-400M-v5 .venv/bin/python -u m7src/encode_dev.py \
    cqadup-programmers cqadup-physics || exit 1

for enc in arctic-embed-l stella-400M-v5; do
  echo "=========== learnability $enc  $(date -Is) ==========="
  M7_ENCODER=$enc .venv/bin/python -u scripts/teacher_learnability.py || echo "FAILED $enc"
done
echo "LEARNABILITY COMPLETE $(date -Is)"
