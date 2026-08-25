#!/bin/sh
# Detached, idempotent benchmark chain. Every stage resumes from cache; rerunning is safe.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
P=.venv/bin/python
LOG=results/overnight.log
caffeinate -dims -w $$ &
{
echo "=== STAGE lr-tables-bos $(date +%H:%M)" && $P bench/run_lightretriever.py tables && \
echo "=== STAGE lr-eval $(date +%H:%M)" && $P bench/run_lightretriever.py eval && \
echo "=== STAGE sparse-mask-diag $(date +%H:%M)" && $P bench/diag_sparse_mask.py && \
echo "=== STAGE opensearch $(date +%H:%M)" && .venv-os/bin/python bench/run_opensearch.py && \
echo "=== STAGE costs $(date +%H:%M)" && $P bench/measure_cost.py && \
echo "=== STAGE tc-baselines $(date +%H:%M)" && BENCH_DATASETS=trec-covid $P bench/run_st.py bge-small-en-v1.5 e5-small-v2 all-MiniLM-L6-v2 gte-small arctic-embed-xs arctic-embed-s granite-small-r2 static-retrieval-mrl-en-v1 arctic-embed-m-v1.5 mdbr-leaf-ir && \
echo "=== STAGE tc-m2v-bm25-asym $(date +%H:%M)" && BENCH_DATASETS=trec-covid $P bench/run_model2vec.py && BENCH_DATASETS=trec-covid $P bench/run_bm25.py && BENCH_DATASETS=trec-covid $P bench/run_asym.py && \
echo "=== STAGE tc-lr-docs $(date +%H:%M)" && BENCH_DATASETS=trec-covid $P bench/run_lightretriever.py docs && \
echo "=== STAGE tc-lr-eval $(date +%H:%M)" && BENCH_DATASETS=trec-covid $P bench/run_lightretriever.py eval && \
echo "=== STAGE tc-opensearch $(date +%H:%M)" && BENCH_DATASETS=trec-covid .venv-os/bin/python bench/run_opensearch.py && \
echo "=== STAGE projection-encode $(date +%H:%M)" && $P bench/run_projection.py encode && \
echo "=== STAGE projection-fit $(date +%H:%M)" && $P bench/run_projection.py fit && \
echo "=== STAGE edge-build $(date +%H:%M)" && $P bench/edge_prototype.py build && \
echo "=== STAGE edge-measure $(date +%H:%M)" && $P bench/edge_prototype.py measure && \
echo "=== STAGE significance $(date +%H:%M)" && BENCH_DATASETS=scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid $P bench/significance.py && \
echo "=== ALL DONE $(date +%H:%M)"
} >> "$LOG" 2>&1
