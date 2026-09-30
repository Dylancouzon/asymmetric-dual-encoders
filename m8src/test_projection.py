"""The projection gate's arming rule: it must refuse a doomed run and not refuse a viable one.

Stage C's first launch (2026-09-19) refused a run the token-weighted estimate clears by ~35 h,
because `min_rows=100_000` armed it inside BEIR-15's six longest corpora -- 1.1% of the volume at
~300 tokens against the batch's 92.2-token mean -- and projected 59 docs/s across 23.7M documents
that are mostly 68-117 tokens. These tests pin BOTH directions of the fix: the larger sample lets
a viable run through, and the gate still stops a genuinely too-slow one.

Rates here are the measured ones: 59 docs/s over the 272,117 long documents (stage C attempt 1),
and 13,400 tokens/s / 77.1 tokens = ~174 docs/s for msmarco (`m20/FINDINGS.md`).
"""
import sys
import time

import pre_encode as P

FAILED = []
EXPECTED_DOCS = 23_744_806
LONG_DOCS, LONG_RATE = 272_117, 59.0        # BEIR-15's six long corpora, as measured
MSMARCO_RATE = 174.0                         # conservative, from the token throughput
COST = {"nano-dense": 1.0, "bge-small-en-v1.5": 0.1349, "leaf-ir-asym": 0.3308}
ORDER = ["nano-dense", "bge-small-en-v1.5", "leaf-ir-asym"]


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r}, want {want!r}")
        print(f"FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"ok   {name}")


def drive(min_rows, budget_hours, msmarco_rate=MSMARCO_RATE, stop_at=None):
    """Feed the gate the real batch order in shard-sized observations.

    Returns the row count at which it refused, or None if it never did.
    """
    gate = P.Projection(
        tower="nano-dense", tower_order=ORDER, expected_docs=EXPECTED_DOCS,
        deadline_epoch=time.time() + budget_hours * 3600.0, cost_ratio=COST,
        label="test", min_rows=min_rows)
    done = 0
    limit = stop_at or (LONG_DOCS + 3_000_000)
    while done < limit:
        rate = LONG_RATE if done < LONG_DOCS else msmarco_rate
        rows = min(50_000, limit - done)
        try:
            gate.observe("nano-dense", rows, rows / rate)
        except RuntimeError:
            return done + rows
        done += rows
    return None


# The regression itself: at the old arming point the gate refuses a run that fits.
check("old min_rows refuses a viable run",
      drive(100_000, budget_hours=102.27) is not None, True)

# And at the new one it does not, because the sample has reached the volume-carrying corpora.
check("new min_rows admits the same viable run",
      drive(1_000_000, budget_hours=102.27), None)

# The gate must still be a gate. A box half this speed does not fit in the cap, and is refused
# even with the larger sample.
check("still refuses a genuinely too-slow run",
      drive(1_000_000, budget_hours=102.27, msmarco_rate=MSMARCO_RATE / 2) is not None, True)

# It refuses at a shard boundary once armed, not before: nothing can refuse under min_rows.
check("never refuses below min_rows",
      drive(1_000_000, budget_hours=0.001, stop_at=900_000), None)
check("refuses promptly once armed",
      drive(1_000_000, budget_hours=0.001, stop_at=1_050_000), 1_000_000)

# The batch wiring: reserved keeps the value stage A actually ran under, BEIR-15 gets the larger.
check("reserved batch min_rows", P.batch_min_projection_rows("reserved"), 100_000)
check("beir15 batch min_rows", P.batch_min_projection_rows("beir15"), 1_000_000)

# The arming point has to be reachable inside the batch, or the gate is disarmed by arithmetic.
check("min_rows is a real prefix of the batch",
      P.batch_min_projection_rows("beir15") < EXPECTED_DOCS, True)
check("reserved min_rows is a real prefix too",
      P.batch_min_projection_rows("reserved") < P.expected_batch_docs("reserved"), True)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED")
    sys.exit(1)
print("all projection-gate checks pass")
