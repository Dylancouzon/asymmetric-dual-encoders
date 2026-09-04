"""Qdrant's two shipping fusion operators, on the run dicts m7src/fusion.py already speaks.

M12 exists because M7's published fused row uses `convex0 w=0.8`, which Qdrant does not ship. These
are the operators it does ship, implemented to parity rather than to a paraphrase -- the whole point
of the audit is lost if we compare against our idea of RRF instead of Qdrant's.

Parity source, vendored in `reference.py` and pinned in m12/LEDGER.md:
  qdrant-client qdrant_client/hybrid/fusion.py @ e50eb17f49851eb710c1f6f502e16cd338898703
  sha256 92e438121d817c28e7b54103fb421518beb4d74af53a31ef38ce3169d42082ff
The server (lib/segment/src/common/reciprocal_rank_fusion.rs) computes the same formula; the client
is vendored because it is the one we can execute in a test.

TWO THINGS THAT ARE NOT OUR CONVENTIONS, kept because they are Qdrant's:
  * RRF rank is 0-BASED and the constant sits outside it: `1/((pos+1)/w + k - 1)`. So a `k` here is
    NOT a `k` in `fusion.rrf`, which computes `w/(k + rank)` on 1-based ranks. The two agree only
    under `k_qdrant = k_ours + 1` at w=1. Report every recommended k in Qdrant's units.
  * A document absent from a prefetch contributes 0. Under DBSF's `(s-(mu-3sigma))/(6sigma)` that is
    NOT the bottom of the normalised range, so an absent document can outrank a present one whose
    score sits below mu-3sigma. That is Qdrant's behaviour and a registered choice (m12/LEDGER.md),
    not an oversight -- it is the same trap M7 hit with convex's `floor_zero`.
"""

DEFAULT_K = 2      # Qdrant's default. The original RRF paper's 60 is k=61 here.


def _ordered(docs):
    """A prefetch's documents best-first. Stable on ties, matching `fusion.rrf` -- a run dict has
    no order of its own, so this is where the rank a fused score depends on is actually decided."""
    return sorted(docs.items(), key=lambda kv: -kv[1])


def rrf(runs, k=DEFAULT_K, weights=None):
    """Qdrant reciprocal rank fusion. runs: list of {qid: {docid: score}}.

    `w <= 0` contributes 0.0 rather than dividing by zero or flipping sign. `k == 0` is refused:
    it divides by zero at pos 0 and the SERVER does not validate it either (validate.rs returns
    Ok(())), so a k=0 recommendation would be a crash we handed to a user.
    """
    if k == 0:
        raise ValueError("qfusion.rrf: k=0 divides by zero at rank 0; Qdrant does not validate it")
    ws = weights or [1.0] * len(runs)
    if len(ws) != len(runs):
        raise ValueError(f"weights {len(ws)} != runs {len(runs)}")
    out = {}
    for run, w in zip(runs, ws):
        for qid, docs in run.items():
            o = out.setdefault(qid, {})
            for pos, (d, _) in enumerate(_ordered(docs)):
                o[d] = o.get(d, 0.0) + (0.0 if w <= 0 else 1.0 / ((pos + 1.0) / w + k - 1.0))
    return out


def dbsf(runs):
    """Qdrant distribution-based score fusion. Parameter-free.

    Per prefetch: `(s - (mu - 3*sd)) / (6*sd)` with the SAMPLE sd (ddof=1), no clamping, then sum
    across prefetches. A singleton or zero-variance list normalises to 0.5 throughout (Qdrant's
    guard against dividing by zero); an empty list contributes nothing at all.
    """
    out = {}
    for run in runs:
        for qid, docs in run.items():
            o = out.setdefault(qid, {})
            if not docs:                      # `if not response: continue` upstream
                continue
            n = len(docs)
            if n == 1:
                o[next(iter(docs))] = o.get(next(iter(docs)), 0.0) + 0.5
                continue
            mean = sum(docs.values()) / n
            var = sum((s - mean) ** 2 for s in docs.values()) / (n - 1)
            if var == 0:
                for d in docs:
                    o[d] = o.get(d, 0.0) + 0.5
                continue
            sd = var ** 0.5
            low, span = mean - 3 * sd, 6 * sd
            for d, s in docs.items():
                o[d] = o.get(d, 0.0) + (s - low) / span
    return out


def truncate(run, depth):
    """A run cut to its top `depth`, on the same stable order the operators rank by.

    Depth is not cosmetic: DBSF's mu and sigma are computed over whatever the prefetch returned, so
    the operator at `limit: 20` is a different function from the one at 1000. Truncating an exact
    top-1000 is exactly what a Qdrant prefetch at that limit would have returned."""
    return {q: dict(_ordered(d)[:depth]) for q, d in run.items()}
