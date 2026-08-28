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
    cheap. A MISSING version is fatal, not `None`: two source or editable installs with no
    distribution metadata would otherwise produce equal keys and share each other's caches."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in ("bm25s", "PyStemmer"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            raise SystemExit(
                f"BM25 REFUSED: no installed distribution metadata for {p!r}, so the lexical "
                "function cannot be pinned into the cache key. Install it as a distribution "
                "(see m7/requirements.lock.txt) rather than from a bare source tree.")
    return out


def cache_key(doc_ids, doc_texts, q_ids, q_texts):
    """The identity of a cached BM25 run: its exact inputs, depth, parameters and library versions.

    The caches used to be keyed by PATHNAME alone and held nothing but integer doc positions and
    scores. `_to_run` then re-attaches whatever `doc_ids`/`q_ids` the caller passes, so a corpus of
    the same shape -- a re-pinned dev component, a different subforum, a regenerated pool slice --
    would have been silently accepted, and a fusion parameter selected on one lexical run applied
    to another. Positions are meaningless without the list they index into, which is exactly what
    was not being checked (Codex one-shot-path review 2026-08-28, MAJOR 2).
    """
    from hashing import sha_stream_list
    # Sequences, not iterators. A generator would be CONSUMED here and then handed on empty to
    # the tokenizer; and `depth` is not a parameter because retrieval always uses the module-level
    # DEPTH, so accepting one only lets the key mislabel the cache (Codex review #4, fusion-cache).
    for name, seq in (("doc_ids", doc_ids), ("doc_texts", doc_texts),
                      ("q_ids", q_ids), ("q_texts", q_texts)):
        if not hasattr(seq, "__len__") or not hasattr(seq, "__getitem__"):
            raise TypeError(f"fusion.cache_key needs a re-iterable sequence for {name}, got "
                            f"{type(seq).__name__}")
    return {"format": CACHE_FORMAT,
            "n_docs": len(doc_ids), "n_queries": len(q_ids), "depth": int(DEPTH),
            "doc_ids_sha256": sha_stream_list(doc_ids),
            "doc_texts_sha256": sha_stream_list(doc_texts),
            "q_ids_sha256": sha_stream_list(q_ids),
            "q_texts_sha256": sha_stream_list(q_texts),
            "config": BM25_CONFIG, "versions": _pkg_versions()}


def _read_cache(cache_path, key):
    """-> (ids, scores) if the cache is provably the run `key` describes, else (None, reason).

    An unreadable or truncated file is a REASON, not an exception: a cache killed mid-write must
    be rebuilt like any other unvalidatable one, not crash the run (Codex review #4)."""
    try:
        z = np.load(cache_path, allow_pickle=False)
        if "key" not in z.files:
            return None, "written before content keying (no `key` array); it cannot be validated"
        got = json.loads(bytes(z["key"]).decode())
        ids, sc = z["ids"], z["scores"]
    except Exception as e:
        return None, f"unreadable ({type(e).__name__}: {e}); treating as unvalidatable"
    if got != key:
        differ = sorted(k for k in set(got) | set(key) if got.get(k) != key.get(k))
        return None, f"key mismatch on {differ}"
    # The key is a CLAIM; the arrays are the fact. Check both axes, that they agree, and that
    # every stored position is in range for the corpus the key names.
    want_k = min(key["depth"], key["n_docs"])
    if ids.shape != sc.shape:
        return None, f"ids {ids.shape} and scores {sc.shape} disagree"
    if ids.shape != (key["n_queries"], want_k):
        return None, f"shape {ids.shape} != ({key['n_queries']}, {want_k})"
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= key["n_docs"]):
        return None, f"doc positions out of range for {key['n_docs']} documents"
    return (ids, sc), None


def bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=None):
    """BM25 at DEPTH (bm25s-lucene defaults, frozen). Optional raw-array cache: indexing
    HotpotQA's 5.23M documents is the single most expensive repeated step on this box.

    The cache is CONTENT-keyed (see `cache_key`). An unvalidatable cache is rebuilt, loudly and
    never silently reused: correctness of the frozen fusion parameter is worth half an hour of
    CPU, and a stale cache is exactly the failure that cannot be seen in the output.

    The key is always computed HERE, from the arguments actually used. An earlier version let the
    caller pass one in to avoid re-hashing 5.23M documents; that put the cache's identity in the
    caller's hands, which is the hole this keying exists to close (Codex review #4).
    """
    key = cache_key(doc_ids, doc_texts, q_ids, q_texts) if cache_path is not None else None
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
        # Atomic: an interrupted write must leave the old cache or nothing, never a half file.
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
        np.savez_compressed(tmp, ids=ids, scores=sc,
                            key=np.frombuffer(json.dumps(key, sort_keys=True).encode(),
                                              dtype=np.uint8))
        tmp.replace(cache_path)
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

    grid, pers = [], {}
    for fam, params, kw in (("rrf", RRF_K, {}), ("convex", CONVEX_W, {}),
                            ("convex0", CONVEX_W, {"floor_zero": True})):
        for p in params:
            fused = ({c: rrf([dense_runs[c], bm25_runs[c]], k=p) for c in comps} if fam == "rrf"
                     else {c: convex([dense_runs[c], bm25_runs[c]], w=p, **kw) for c in comps})
            m, per = macro(fused)
            grid.append({"family": fam, "param": p, "macro": m})
            pers[(fam, p)] = per
            report(f"  fusion {fam} {'k' if fam == 'rrf' else 'w'}={p:<5} dev macro {m:.4f}")

    # TIE POLICY, fixed 2026-08-28 BEFORE this selection was run on the shipping candidate, and
    # therefore before its numbers exist. A running `best` with strict `>` scanned RRF first, then
    # convex from w=0.3 upward, so the dense-only endpoint w=1.0 -- which is the LAST convex point
    # -- could never displace an equal earlier one. Ties therefore silently favoured the more
    # complex system, and the release could have been called "fused" on a parameter with no dev
    # benefit at all. On an exact tie the SIMPLER system now wins: dense-only first, then the
    # first point in grid order (deterministic). This implements the intent already recorded for
    # w=1.0 -- that whether we fuse at all is decided by the same mechanical selection as the
    # parameter -- rather than changing it (Codex review #4, "Grid ties").
    best = max(grid, key=lambda r: (r["macro"], is_dense_only(r)))
    tied = [r for r in grid if r["macro"] == best["macro"]]
    if len(tied) > 1:
        report(f"  {len(tied)} grid points tie at {best['macro']:.6f}; tie policy takes "
               f"{'the dense-only endpoint' if is_dense_only(best) else 'the first in grid order'}")
    report(f"  -> frozen fusion: {best['family']} param={best['param']} "
           f"dev macro {best['macro']:.4f}"
           + ("  [DENSE-ONLY: the released system does not fuse]" if is_dense_only(best) else ""))
    return ({"family": best["family"], "param": best["param"], "dev_macro": best["macro"],
             "grid": grid, "n_tied_at_best": len(tied)},
            pers[(best["family"], best["param"])])


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
