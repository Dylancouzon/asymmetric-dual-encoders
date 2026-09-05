"""Score a trained student on the COV surface: student query vectors against stella's documents.

The asymmetry is the whole architecture, so the evaluation has to respect it — the student encodes
QUERIES into stella's space and retrieves against the SAME frozen stella document vectors the
server would hold. Those document vectors already exist from the COV admission encode
(`m10src/cov_encode.py`), so a student evaluation costs one query encode and a matmul, not a
re-encode of 452,757 documents.

Returns per-query nDCG@10 per unit, which is exactly what `m10src/cov_macro` consumes, so a screen
contrast and this calibration read the same estimator.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np

KEY = "stella-400M-v5"


def _doc_vecs(name, docs):
    """The cached stella document vectors for one unit. Content-keyed, so a text list that has
    moved builds a new cache rather than silently serving the wrong vectors."""
    import teacher9
    return np.asarray(teacher9.encode_cached(KEY, f"m10cov-{name}-d", docs, "doc",
                                             verbose=False), dtype=np.float32)


def _unit_cache_name(uid):
    """BRIGHT is cached as ONE component of 404,416 documents across its six slices, so a slice's
    vectors are a span of that array, not their own cache."""
    return "BRIGHT" if uid.startswith("BRIGHT/") else uid


def score_student(encode_queries, units=None, verbose=True):
    """`encode_queries(list[str]) -> (n, 1024) fp32 unit-norm` -> {unit: {qid: ndcg@10}}.

    The callable is passed in rather than a model, so the same function scores a trained student,
    stella itself (the teacher ceiling) or any probe, with no branch inside the scorer.
    """
    import evalkit
    import cov_probe
    us = units if units is not None else cov_probe.units()
    bright_docs, bright_off = None, {}
    if any(u[0].startswith("BRIGHT/") for u in us):
        from cov_admit import COMPONENTS, BRIGHT_SLICES
        rev = dict((n, r) for _f, cs in COMPONENTS.items() for n, _rp, r in cs)["BRIGHT"]
        from datasets import load_dataset
        alld, off = [], 0
        for sl in BRIGHT_SLICES:
            d = load_dataset("xlangai/BRIGHT", "documents", revision=rev, split=sl)
            bright_off[f"BRIGHT/{sl}"] = (off, off + len(d))
            off += len(d)
            alld += list(d["content"])
        bright_docs = _doc_vecs("BRIGHT", alld)

    out = {}
    for uid, family, qs, qids, ds, dids, qrels in us:
        if uid.startswith("BRIGHT/"):
            lo, hi = bright_off[uid]
            dv = bright_docs[lo:hi]
        else:
            dv = _doc_vecs(_unit_cache_name(uid), ds)
        if len(dv) != len(ds):
            raise SystemExit(f"{uid}: {len(dv)} cached document vectors for {len(ds)} documents")
        qv = encode_queries(qs)
        out[uid] = evalkit.score(qv, qids, dv, dids, qrels, k=200)
        if verbose:
            print(f"  {uid:28s} nDCG@10 {np.mean(list(out[uid].values())):.4f}", flush=True)
    return out


def macro(per_unit, units=None):
    import cov_macro
    import cov_probe
    us = units if units is not None else cov_probe.units()
    uf = {u[0]: u[1] for u in us}
    return cov_macro.macro(per_unit, uf)


def teacher_ceiling(verbose=True):
    """stella scoring its own documents — the ceiling every retention figure is read against."""
    import teacher9
    return score_student(
        lambda texts: np.asarray(teacher9.encode(KEY, texts, "query"), dtype=np.float32),
        verbose=verbose)
