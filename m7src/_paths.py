"""Shared path/env setup for M7 modules: puts bench/ on sys.path and pins the six datasets."""
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("BENCH_DATASETS", "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid")
if str(REPO / "bench") not in sys.path:
    sys.path.insert(0, str(REPO / "bench"))
M7 = REPO / "m7"
WORK = REPO / "work"          # gitignored: encode caches, tables, checkpoints
WORK.mkdir(exist_ok=True)

# The accelerator this process runs on. One resolver, because porting `device="cuda"` defaults one
# call site at a time gives a mixed-device bug rather than a crash. M7_DEVICE overrides.
DEVICE = os.environ.get("M7_DEVICE") or (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")


def empty_cache():
    """Release cached accelerator memory, whichever accelerator this is."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
