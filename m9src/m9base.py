"""M9 foundation. Mirrors m8src/m8base.py: puts the shared harness on sys.path, pins the
incumbent encoder, and installs the protected-path guard at import time so no M9 module can
opt out by forgetting to import it (m8/CODEMAP.md pitfall 9 + LEDGER G2).

M9 owns `m9src/`. It IMPORTS m7src and m8src and never edits them.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "work"
RESULTS = REPO / "results"
M9 = REPO / "m9"

for p in (REPO / "m7src", REPO / "m8src", REPO / "bench", REPO):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# The teacher M9 inherits. ASSIGNED, never defaulted: `setdefault` would let an operator's stale
# export silently replace the incumbent (Codex BLOCKER-4; CODEMAP pitfall 9 is the same failure
# from the other direction). A conflicting value is refused, not overridden in silence.
INCUMBENT = "stella-400M-v5"
_env = os.environ.get("M7_ENCODER")
if _env not in (None, "", INCUMBENT):
    raise SystemExit(f"M7_ENCODER={_env!r} in the environment conflicts with M9's incumbent "
                     f"{INCUMBENT!r}. Unset it; M9 does not inherit an encoder from a shell.")
os.environ["M7_ENCODER"] = INCUMBENT

# The six (confirmatory, one access) and the reserved four (descriptive, one access).
SIX = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
RESERVED = ("fever", "dbpedia-entity", "cqadup-android", "cqadup-english")

# M9.0-locked decision surfaces (m9/LEDGER.md §4.1). DEV_FULL decides student/prompt/mix/batch
# and the seed floor; SCREEN_DEV (family-weighted) decides the teacher only, because a challenger
# teacher cannot re-encode the two large dev components inside this milestone.
SCREEN_DEV = ("nq-250k", "cqadup-programmers", "cqadup-physics")
DEV_FULL = ("nq-250k", "hotpotqa", "cqadup-programmers", "cqadup-physics",
            "heldout-train", "heldout-longq")

import paths_guard  # noqa: E402
paths_guard.install()
