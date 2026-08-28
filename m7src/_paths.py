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

# The accelerator this process runs on. Every module used to hardcode "cuda" as a default argument,
# so the teacher-learnability probe could not run at all on the second machine the ledger sends it
# to. One resolver, so a mixed-device bug (rows on mps, query block on cpu) cannot be introduced by
# porting one call site and missing its neighbour. M7_DEVICE overrides for a CPU fallback run.
DEVICE = os.environ.get("M7_DEVICE") or (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")


def empty_cache():
    """Release cached accelerator memory, whichever accelerator this is. `torch.cuda.empty_cache()`
    is a silent no-op on a CUDA-less build in some versions and an AssertionError in others, and the
    peak-RAM incidents in CLAUDE.md are the reason these calls exist at all."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
