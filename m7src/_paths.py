"""Shared path/env setup for M7 modules: puts bench/ on sys.path and pins the six datasets."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("BENCH_DATASETS", "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid")
if str(REPO / "bench") not in sys.path:
    sys.path.insert(0, str(REPO / "bench"))
M7 = REPO / "m7"
WORK = REPO / "work"          # gitignored: encode caches, tables, checkpoints
WORK.mkdir(exist_ok=True)
