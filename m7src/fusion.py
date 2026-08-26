"""Fusion of the zero-neural-query-compute dense table with BM25.

One family only, picked on dev; every parameter (including BM25's, at bm25s-lucene defaults)
frozen on dev before any test access. No per-dataset weights, normalization, or routing --
a single (family, parameter) pair applies to every dataset.
"""
import numpy as np

# RRF and min-max convex fusion are both depth-sensitive: ranks beyond the retrieval cut simply
# do not exist to be fused. So dev selection and final application must retrieve to the SAME
# depth, or the parameter frozen on dev is applied to a different function at test time.
DEPTH = 1000


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


def bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=None):
    """BM25 at DEPTH (bm25s-lucene defaults, frozen). Optional raw-array cache: indexing
    HotpotQA's 5.23M documents is the single most expensive repeated step on this box."""
    if cache_path is not None and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        return _to_run(z["ids"], z["scores"], doc_ids, q_ids)
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
        np.savez_compressed(cache_path, ids=ids, scores=sc)
    return _to_run(ids, sc, doc_ids, q_ids)


RRF_K = [10, 20, 30, 60, 100]
CONVEX_W = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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


def apply_frozen(spec, dense_run, bm25_run):
    if spec["family"] == "rrf":
        return rrf([dense_run, bm25_run], k=spec["param"])
    return convex([dense_run, bm25_run], w=spec["param"],
                  floor_zero=(spec["family"] == "convex0"))
