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


def convex(runs, w, eps=1e-9):
    """Per-query min-max normalize each run, then a convex combination. w applies to runs[0]."""
    ws = [w, 1.0 - w] if len(runs) == 2 else [1.0 / len(runs)] * len(runs)
    out = {}
    for run, wi in zip(runs, ws):
        for qid, docs in run.items():
            o = out.setdefault(qid, {})
            v = np.fromiter(docs.values(), dtype=np.float64, count=len(docs))
            lo, hi = v.min(), v.max()
            for d, s in docs.items():
                o[d] = o.get(d, 0.0) + wi * (s - lo) / (hi - lo + eps)
    return out


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
    report(f"  -> frozen fusion: {best[1]} param={best[2]} dev macro {best[0]:.4f}")
    return {"family": best[1], "param": best[2], "dev_macro": best[0], "grid": grid}, best[3]


def apply_frozen(spec, dense_run, bm25_run):
    return (rrf([dense_run, bm25_run], k=spec["param"]) if spec["family"] == "rrf"
            else convex([dense_run, bm25_run], w=spec["param"]))
