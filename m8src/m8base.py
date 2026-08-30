"""Shared path/env setup for M8 modules.

M8 reuses the M7 harness rather than forking it: `m7src/` is frozen (LEDGER G3), `bench/` and
`scripts/` are the shared M1-M6 harness. Importing this module puts `m8src/`, `m7src/` and
`bench/` on `sys.path` (flat modules, the convention m7src already uses -- there is no package
`__init__.py` anywhere in this repo) and installs the protected-path guard (LEDGER G2)
process-wide, so a module that merely forgets to think about protected data still cannot open it.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "bench", REPO / "m7src", REPO / "m8src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("BENCH_DATASETS", "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid")
# THE INCUMBENT TEACHER, pinned for every M8 process. `m7src/encoders.active()` reads M7_ENCODER
# and defaults to bge-base -- M7's PRE-SWAP teacher. Without this pin, every arm initialized from
# a stella checkpoint dies with "init was trained against stella but the active encoder is
# bge-base", which is exactly what the first noise-floor smoke did, five times in a row.
# m7/LEDGER.md: "All work keys on M7_ENCODER=stella-400M-v5 ... so no comparison can mix
# teachers." A default beats remembering.
os.environ.setdefault("M7_ENCODER", "stella-400M-v5")

M8 = REPO / "m8"
M7 = REPO / "m7"
WORK = REPO / "work"
RESULTS = REPO / "results"
LOGS = REPO / "logs"
WORK.mkdir(exist_ok=True)

LEDGER = M8 / "LEDGER.md"
STATUS = M8 / "STATUS.md"

# The reserved confirmatory four (LEDGER 2.1). Order is the reporting order; the estimand is the
# EQUAL-WEIGHT macro over them, so this list is also the weighting.
RESERVED_FOUR = ("fever", "dbpedia-entity", "cqadup-android", "cqadup-english")

# M7's six, development-informed for M8 (LEDGER 2.2). Scored descriptively only, plus the
# one-directional six-set no-regression ship guard.
SIX = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")

# Registered dev groups (LEDGER 2.2 / 4.5). Selection reads median / worst-group over these,
# never the arithmetic-mean macro.
DEV_GROUPS = {
    "out-of-domain": ("cqadup-programmers", "cqadup-physics"),
    "wikipedia": ("nq-250k", "hotpotqa"),
    "heldout": ("heldout-train", "heldout-longq"),
}

import paths_guard  # noqa: E402  (needs m8src on sys.path, done above)

paths_guard.install()

_DEVICE = None


def device():
    """Resolved lazily so importing this module costs no torch import."""
    global _DEVICE
    if _DEVICE is None:
        override = os.environ.get("M8_DEVICE") or os.environ.get("M7_DEVICE")
        if override:
            _DEVICE = override
        else:
            import torch
            _DEVICE = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    return _DEVICE


def empty_cache():
    import torch
    d = device()
    if d == "cuda":
        torch.cuda.empty_cache()
    elif d == "mps":
        torch.mps.empty_cache()
