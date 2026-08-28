"""Fusion of the zero-neural-query-compute dense table with BM25.

One family only, picked on dev; every parameter (including BM25's, at bm25s-lucene defaults)
frozen on dev before any test access. No per-dataset weights, normalization, or routing --
a single (family, parameter) pair applies to every dataset.
"""
import json

import numpy as np

# RRF and min-max convex fusion are both depth-sensitive: ranks beyond the retrieval cut simply
# do not exist to be fused. So dev selection and final application must retrieve to the SAME
# depth, or the parameter frozen on dev is applied to a different function at test time.
DEPTH = 1000

# The complete family list. `apply_frozen` used to fall through to convex for anything that was
# not "rrf", so a typo or a future family name in the frozen spec would have been applied as
# convex-with-that-param in the one-shot run (Codex one-shot-path review 2026-08-28, MAJOR 1).
FAMILIES = ("rrf", "convex", "convex0")

# Everything about the BM25 function that a cached run depends on. Part of the cache key, so a
# parameter change invalidates every cache instead of being silently inherited.
BM25_CONFIG = {"impl": "bm25s", "method": "lucene", "k1": 1.2, "b": 0.75,
               "stopwords": "en", "stemmer": "english-snowball-PyStemmer",
               "drop_zero_scores": True, "drop_self_hits": True}
CACHE_FORMAT = 2         # 1 == the keyless caches written before 2026-08-28


def rrf(runs, k=60, weights=None):
    """Reciprocal rank fusion. runs: list of {qid: {docid: score}}."""
    weights = weights or [1.0] * len(runs)
    out = {}
    for run, w in zip(runs, weights):
        for qid, docs in run.items():
            o = out.setdefault(qid, {})
            for rank, (d, _) in enumerate(sorted(docs.items(), key=lambda kv: -kv[1]), start=1):
                o[d] = o.get(d, 0.0) + w / (k + rank)
    return out


def convex(runs, w, eps=1e-9, floor_zero=False):
    """Per-query min-max normalize each run, then a convex combination. w applies to runs[0].

    floor_zero anchors each query's min at 0 (the absent-document baseline) instead of the
    minimum returned score: with padding gone, a query with ONE BM25 hit otherwise normalizes
    that hit to 0 -- indistinguishable from no lexical evidence (review #2 MAJOR 19). Both
    variants are in the dev selection grid; the frozen spec records which won."""
    ws = [w, 1.0 - w] if len(runs) == 2 else [1.0 / len(runs)] * len(runs)
    out = {}
    for run, wi in zip(runs, ws):
        for qid, docs in run.items():
            o = out.setdefault(qid, {})
            if not docs:      # a query with zero positive-score matches contributes nothing
                continue      # (padding used to hide this case; test_fusion_paths.py covers it)
            v = np.fromiter(docs.values(), dtype=np.float64, count=len(docs))
            lo, hi = (0.0, float(v.max())) if floor_zero else (float(v.min()), float(v.max()))
            for d, s in docs.items():
                o[d] = o.get(d, 0.0) + wi * (s - lo) / (hi - lo + eps)
    return out


def _to_run(ids, sc, doc_ids, q_ids):
    """Raw bm25s (ids, scores) arrays -> run dict. THE one conversion for anything that gets
    fused, shared by the cached, fresh, selection, and final paths so they cannot diverge.

    The `s > 0` filter and the self-hit drop are PART OF the frozen fusion function: bm25s pads
    to k with zero-score rows whenever a query matches fewer than DEPTH docs (guaranteed on the
    small six corpora), padding drags convex's per-query min-max `lo` to 0 and hands RRF rank
    mass to docs BM25 never retrieved. Codex B5: selection dropped the padding and the final run
    kept it, so the Tier-1 system would not have been the function selected on dev."""
    return {q_ids[i]: {doc_ids[int(d)]: float(s) for d, s in zip(ids[i], sc[i])
                       if s > 0 and doc_ids[int(d)] != q_ids[i]}
            for i in range(len(q_ids))}


def _pkg_versions():
    """Versions of the two packages that define the BM25 function, without importing either --
    `importlib.metadata` reads the installed distribution metadata, so the cache-hit path stays
    cheap."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in ("bm25s", "PyStemmer"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:      # pragma: no cover -- both are in requirements.lock.txt
            out[p] = None
    return out


def cache_key(doc_ids, doc_texts, q_ids, q_texts, depth=DEPTH):
    """The identity of a cached BM25 run: its exact inputs, depth, parameters and library versions.

    The caches used to be keyed by PATHNAME alone and held nothing but integer doc positions and
    scores. `_to_run` then re-attaches whatever `doc_ids`/`q_ids` the caller passes, so a corpus of
    the same shape -- a re-pinned dev component, a different subforum, a regenerated pool slice --
    would have been silently accepted, and a fusion parameter selected on one lexical run applied
    to another. Positions are meaningless without the list they index into, which is exactly what
    was not being checked (Codex one-shot-path review 2026-08-28, MAJOR 2).
    """
    from hashing import sha_stream_list
    return {"format": CACHE_FORMAT,
            "n_docs": len(doc_ids), "n_queries": len(q_ids), "depth": int(depth),
            "doc_ids_sha256": sha_stream_list(doc_ids),
            "doc_texts_sha256": sha_stream_list(doc_texts),
            "q_ids_sha256": sha_stream_list(q_ids),
            "q_texts_sha256": sha_stream_list(q_texts),
            "config": BM25_CONFIG, "versions": _pkg_versions()}


def _read_cache(cache_path, key):
    """-> (ids, scores) if the cache is provably the run `key` describes, else (None, reason)."""
    z = np.load(cache_path, allow_pickle=False)
    if "key" not in z.files:
        return None, "written before content keying (no `key` array); it cannot be validated"
    got = json.loads(bytes(z["key"]).decode())
    if got != key:
        differ = sorted(k for k in set(got) | set(key) if got.get(k) != key.get(k))
        return None, f"key mismatch on {differ}"
    if z["ids"].shape[0] != key["n_queries"]:      # the key is a claim; the arrays are the fact
        return None, f"row count {z['ids'].shape[0]} != {key['n_queries']} queries"
    return (z["ids"], z["scores"]), None


def bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=None, key=None):
    """BM25 at DEPTH (bm25s-lucene defaults, frozen). Optional raw-array cache: indexing
    HotpotQA's 5.23M documents is the single most expensive repeated step on this box.

    The cache is CONTENT-keyed (see `cache_key`). An unvalidatable cache is rebuilt, loudly and
    never silently reused: correctness of the frozen fusion parameter is worth half an hour of
    CPU, and a stale cache is exactly the failure that cannot be seen in the output. `key` lets a
    caller that already computed it (to record as provenance) avoid re-hashing 5.23M documents.
    """
    if cache_path is not None and key is None:
        key = cache_key(doc_ids, doc_texts, q_ids, q_texts)
    if cache_path is not None and cache_path.exists():
        arrays, why = _read_cache(cache_path, key)
        if arrays is not None:
            return _to_run(arrays[0], arrays[1], doc_ids, q_ids)
        print(f"[fusion] REBUILDING BM25 cache {cache_path.name}: {why}", flush=True)
    import Stemmer
    import bm25s
    st = Stemmer.Stemmer("english")
    r = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    r.index(bm25s.tokenize(doc_texts, stopwords="en", stemmer=st, show_progress=False),
            show_progress=False)
    ids, sc = r.retrieve(bm25s.tokenize(q_texts, stopwords="en", stemmer=st, show_progress=False),
                         k=min(DEPTH, len(doc_ids)), show_progress=False)
    ids, sc = ids.astype(np.int32), sc.astype(np.float32)
    if cache_path is not None:
        np.savez_compressed(cache_path, ids=ids, scores=sc,
                            key=np.frombuffer(json.dumps(key, sort_keys=True).encode(),
                                              dtype=np.uint8))
    return _to_run(ids, sc, doc_ids, q_ids)


RRF_K = [10, 20, 30, 60, 100]
# 1.0 is the DENSE-ONLY endpoint, i.e. "do not fuse". It belongs in the grid so that whether the
# released system fuses at all is decided by the same mechanical selection as the parameter,
# rather than by a later judgement call comparing two separately reported macros. Added
# 2026-08-28, before the fusion re-selection on the post-lever candidate (m7/LEDGER.md).
CONVEX_W = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def select_on_dev(dense_runs, bm25_runs, qrels_by_comp, report=print):
    """Grid-search both families on the dev macro; returns the single winning (family, param).

    dense_runs/bm25_runs: {component: run dict}, both retrieved to DEPTH. Selection happens here
    and nowhere else, and the winner is written into m7/FREEZE.json before any test access.
    """
    from evalkit import per_query_ndcg
    comps = sorted(dense_runs)

    def macro(fused):
        per = {c: per_query_ndcg(fused[c], qrels_by_comp[c]) for c in comps}
        return float(np.mean([np.mean(list(per[c].values())) for c in comps])), per

    best = None
    grid = []
    for k in RRF_K:
        m, per = macro({c: rrf([dense_runs[c], bm25_runs[c]], k=k) for c in comps})
        grid.append({"family": "rrf", "param": k, "macro": m})
        report(f"  fusion rrf k={k:<4} dev macro {m:.4f}")
        if best is None or m > best[0]:
            best = (m, "rrf", k, per)
    for w in CONVEX_W:
        m, per = macro({c: convex([dense_runs[c], bm25_runs[c]], w=w) for c in comps})
        grid.append({"family": "convex", "param": w, "macro": m})
        report(f"  fusion convex w={w:<4} dev macro {m:.4f}")
        if m > best[0]:
            best = (m, "convex", w, per)
    for w in CONVEX_W:
        m, per = macro({c: convex([dense_runs[c], bm25_runs[c]], w=w, floor_zero=True)
                        for c in comps})
        grid.append({"family": "convex0", "param": w, "macro": m})
        report(f"  fusion convex0 w={w:<4} dev macro {m:.4f}")
        if m > best[0]:
            best = (m, "convex0", w, per)
    report(f"  -> frozen fusion: {best[1]} param={best[2]} dev macro {best[0]:.4f}")
    return {"family": best[1], "param": best[2], "dev_macro": best[0], "grid": grid}, best[3]


def is_dense_only(spec):
    """True iff the selected point in the grid IS the dense-only endpoint.

    `CONVEX_W` carries w=1.0 precisely so that "do not fuse" can win the same mechanical
    selection as the parameter (m7/LEDGER.md, Fusion). This function is how that decision is
    read back out, so `released_system` is derived from the selection rather than asserted on a
    freeze command line. RRF always mixes both runs, so it is never dense-only.
    """
    return spec["family"] in ("convex", "convex0") and float(spec["param"]) == 1.0


def apply_frozen(spec, dense_run, bm25_run):
    fam = spec["family"]
    if fam not in FAMILIES:
        # Was a silent fall-through to convex. In the one-shot run that would have applied a
        # different function than the one the spec names, with nothing in the output to show it.
        raise SystemExit(f"FUSION REFUSED: unknown fusion family {fam!r}; known: {FAMILIES}")
    if fam == "rrf":
        return rrf([dense_run, bm25_run], k=spec["param"])
    return convex([dense_run, bm25_run], w=spec["param"], floor_zero=(fam == "convex0"))
