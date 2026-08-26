"""Witness for the mining loop-order fix: the fast path must return what the slow path returned.

train.mine_hard_negatives was re-reading the 9.5 GB doc pool once per query chunk (1.6 TB of
traffic, 3.6 hours). Inverting the loops touches the pool once. The arithmetic is meant to be
identical, but fp16 accumulation order changes with tiling, so near-ties can swap: this measures
agreement rather than asserting bit-equality, and reports top-1 agreement separately because that
is the negative that matters most.

    ../.venv/bin/python scripts/check_mining.py            # from m7src/, or with m7src on sys.path
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))

import pool as poolmod
import train
from _paths import REPO

N_Q, K, POOL_ROWS = 8192, 16, 1_500_000  # N_Q spans 4 query chunks: the query-outer path re-reads the pool once per chunk, which is the cost being fixed

def main():
    _, vecs, meta = poolmod.build()
    sub = vecs[:POOL_ROWS]
    rng = np.random.default_rng(0)
    # Queries drawn from the pool itself, then perturbed: random unit vectors would make every
    # candidate equally uninteresting and hide ordering differences in the tail.
    idx = rng.choice(POOL_ROWS, N_Q, replace=False)
    q = np.asarray(sub[idx], dtype=np.float32) + 0.35 * rng.standard_normal((N_Q, sub.shape[1])).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = q.astype(np.float16)
    exclude = [[] for _ in range(N_Q)]

    t0 = time.time()
    empty_mask = (np.zeros(0, np.int64), set(), "none")  # synthetic pool: global B2 rows do not apply
    fast = train.mine_hard_negatives("checkfast", q, sub, K, exclude, banned=empty_mask)
    t_fast = time.time() - t0
    t0 = time.time()
    slow = train._mine_hard_negatives_qouter("checkslow", q, sub, K, exclude)
    t_slow = time.time() - t0

    top1 = float((fast[:, 0] == slow[:, 0]).mean())
    jac = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(fast, slow)]))
    exact = float(np.mean([np.array_equal(a, b) for a, b in zip(fast, slow)]))
    out = {"_note": "Equivalence witness for the mining loop-order fix. Same k, same pool slice, "
                    "same queries; only the loop nesting differs. Agreement is measured, not "
                    "asserted bit-equal, because fp16 accumulation order changes with tiling.",
           "n_queries": N_Q, "k": K, "pool_rows": POOL_ROWS,
           "seconds_pool_outer_fast": round(t_fast, 1),
           "seconds_query_outer_original": round(t_slow, 1),
           "speedup_on_this_slice": round(t_slow / max(t_fast, 1e-9), 1),
           "_speedup_caveat": "This slice CANNOT demonstrate the speedup and does not claim to: "
                              "1.5M rows stay in the OS page cache, so the original's repeated "
                              "reads are nearly free here (1.1x). The real cost was ~170 COLD "
                              "re-reads of the full 9.5 GB pool, measured at 76 s per query chunk "
                              "in logs/phase2_screen.log. This file's job is EQUIVALENCE; the "
                              "speed claim is settled by the mining line of the real run.",
           "top1_agreement": top1, "mean_set_overlap_at_k": jac, "exact_row_match": exact}
    (REPO / "results" / "m7_mining_equivalence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    assert top1 >= 0.98 and jac >= 0.99, "fast mining disagrees with the reference beyond fp16 noise"
    print("PASS")

if __name__ == "__main__":
    main()
