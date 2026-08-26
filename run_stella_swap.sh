#!/usr/bin/env bash
# The teacher swap to stella_en_400M_v5, chosen on the distilled table (m7_learnability_report.json)
# after the symmetric-ceiling criterion was refuted. ~3.5 h: 6.17M documents at 1024-d.
#
# Gates first, in the order m7/CODEMAP.md mandates, because a bad loader here costs the whole encode:
# cache-key stability, then the init-row read-out check (stella is the mean-pooled + post-Dense Spec
# that motivated that check existing).
#
# BM25 and potion reference rows were copied from the bge refs file rather than recomputed -- they do
# not depend on the teacher, and BM25 over the 5.23M-doc HotpotQA component would burn an hour to
# reproduce identical numbers. Only the teacher rows are computed here.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
L=logs/stella_swap.log

step () { echo "=========== $* $(date -Is) ==========="; }

step "gate: encode cache keys";      .venv/bin/python m7src/test_encoders.py    || exit 1
step "gate: init-row read-out";      (cd m7src && M7_ENCODER=stella-400M-v5 ../.venv/bin/python test_init_rows.py) || exit 1
step "dev encodes nq-250k";          .venv/bin/python -u m7src/encode_dev.py nq-250k   || exit 1
step "dev encodes hotpotqa (5.23M)"; .venv/bin/python -u m7src/encode_dev.py hotpotqa  || exit 1
step "doc-vector pool (6.17M)";      .venv/bin/python -u m7src/pool.py                 || exit 1
step "dev reference rows (teacher)"; .venv/bin/python -u m7src/dev_eval.py             || exit 1
step "closed-form ridge, full suite"; .venv/bin/python -u m7src/stage0_ridge.py        || echo "ridge FAILED"
echo "STELLA SWAP COMPLETE $(date -Is)"
