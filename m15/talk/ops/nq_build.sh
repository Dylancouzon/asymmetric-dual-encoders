#!/bin/bash
# NQ on the Mac: box-vs-Mac parity, then the Edge build. Log: work/m15/logs/nq_build.log
cd "$(dirname "$0")/../../.." || exit 1
M7_ENCODER=stella-400M-v5 .venv-mac/bin/python -u m15src/talk_encode_nq.py --parity work/m15/vecs/nq_stella || { echo "PARITY STEP FAILED"; exit 1; }
.venv/bin/python -u m15src/talk_edge.py build nq && echo "BUILD COMPLETE"
