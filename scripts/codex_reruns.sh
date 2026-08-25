#!/bin/sh
# Codex-gate reruns: projection fix, extended significance, ANN sweeps, sparse index cost.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
P=.venv/bin/python
LOG=results/codex_reruns.log
caffeinate -dims -w $$ &
{
echo "=== STAGE proj-encode $(date +%H:%M)" && $P bench/run_projection.py encode && \
echo "=== STAGE proj-fit $(date +%H:%M)" && $P bench/run_projection.py fit && \
echo "=== STAGE ann-sweep $(date +%H:%M)" && $P bench/edge_ann_sweep.py && \
echo "=== STAGE os-index-cost $(date +%H:%M)" && .venv-os/bin/python bench/opensearch_index_cost.py && \
echo "=== STAGE significance6 $(date +%H:%M)" && BENCH_DATASETS=scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid $P bench/significance.py && \
echo "=== ALL RERUNS DONE $(date +%H:%M)"
} >> "$LOG" 2>&1
