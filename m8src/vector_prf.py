"""VECTOR-PRF -- train-free dense pseudo-relevance feedback on R0's frozen table.

WHY THIS PROBE EXISTS, and why it is not another vocabulary idea. The table is a BAG of
context-free rows; the document tower is contextual. `D2-PRE` (§24) just falsified the hypothesis
that finer vocabulary granularity repairs that mismatch -- all four new-row classes came back
negative. This attacks the same gap from the other side and without training anything: take the
documents the first retrieval already returned and use them to pull the query vector toward the
document manifold.

    q' = normalize( alpha * q + beta * mean(d_1..d_k) )

The config is the PUBLISHED one, alpha=0.4, beta=0.6, k=3 (Vector-PRF, arXiv 2205.00235), used
verbatim. There is no grid and no per-component choice, deliberately: a probe that may tune its
feedback weights can always find a positive somewhere, and the registry froze this before any
number existed.

WHAT IT IS NOT. It trains nothing and changes no row -- R0's shipped table is used byte for byte,
and the comparison is strictly paired against that same table under single-pass retrieval. So this
is a change to the retrieval PROCEDURE: a `qualifying_system` key under Dylan's condition-4 ruling,
NEVER a better table. It also costs a second ANN query and a k-vector fetch per search, which the
result reports beside any gain because the edge story goes from one round trip to two.

Query drift is the known failure mode and a negative is entirely plausible.
"""
import argparse
import gc
import json
import sys
import time

import numpy as np

import m8base
import probe_guard
from d2_pre import CLS_ID, counts_of, encode, ids_of, load_incumbent, tok_pre

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_vector_prf.json"
OOD = ("cqadup-programmers", "cqadup-physics")
FUSED_COMPONENTS = ("nq-250k", "hotpotqa", "cqadup-programmers", "cqadup-physics")

# THE PUBLISHED CONFIG, frozen before any number (arXiv 2205.00235). Not a search space.
ALPHA, BETA, DEPTH = 0.4, 0.6, 3
BAR = 0.0040


def prf(qv, dv, k=DEPTH, alpha=ALPHA, beta=BETA, chunk=250_000):
    """One feedback round. Returns q' and the mean rank-1..k document vector, both normalized.

    The feedback vectors come from the SAME exact search the baseline uses, so the two arms differ
    only in the second pass -- which is what makes the comparison paired.
    """
    from evalkit import topk_arrays
    bi, _ = topk_arrays(qv, dv, k=k, chunk=chunk)
    fb = np.asarray(dv[np.asarray(bi).ravel()], dtype=np.float32).reshape(len(qv), k, -1).mean(1)
    fb /= np.maximum(np.linalg.norm(fb, axis=1, keepdims=True), 1e-12)
    q2 = alpha * qv + beta * fb
    q2 /= np.maximum(np.linalg.norm(q2, axis=1, keepdims=True), 1e-12)
    return q2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    import dev_eval
    import fusion
    import select_fusion
    from evalkit import per_query_ndcg, topk_ids_scores

    tok, pre = tok_pre()
    W_inc, _ = load_incumbent()
    comps = list(OOD) if a.smoke else dev_eval.dev_components()
    spec = json.loads((RESULTS / "m7_fusion_p35w-2m-s2500.json").read_text())
    spec = {"family": spec["family"], "param": spec["param"]}

    per = {"base": {}, "prf": {}}
    fused = {"base": {}, "prf": {}}
    drift = {}
    for c in comps:
        tc = time.time()
        doc_ids, doc_texts, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(c)
        qv = encode([counts_of(x) for x in ids_of(tok, list(q_texts), pre)], W_inc)
        q2 = prf(qv, dv, chunk=dev_eval.CHUNK.get(c, 200_000))
        drift[c] = float(np.mean((qv * q2).sum(1)))
        per["base"][c] = dev_eval.eval_query_vecs(c, qv)
        per["prf"][c] = dev_eval.eval_query_vecs(c, q2)
        if c in FUSED_COMPONENTS and doc_texts is not None:
            b_run, _ = select_fusion.bm25_run_and_key(c)
            for who, v in (("base", qv), ("prf", q2)):
                d_run = topk_ids_scores(v, dv, doc_ids, k=fusion.DEPTH, qids=list(q_ids),
                                        chunk=dev_eval.CHUNK.get(c, 200_000))
                fused[who][c] = {q: float(x) for q, x in per_query_ndcg(
                    fusion.apply_frozen(spec, d_run, b_run), qrels).items()}
            del b_run
        print(f"  {c}: base {np.mean(list(per['base'][c].values())):.4f} -> "
              f"prf {np.mean(list(per['prf'][c].values())):.4f}  "
              f"(cos(q,q')={drift[c]:.3f}, {time.time()-tc:.0f}s)", flush=True)
        del dv, doc_ids, doc_texts, q_texts
        gc.collect()
        m8base.empty_cache()

    import noise_floor
    gv = {w: (noise_floor._group_vector(per[w]) if not a.smoke else
              {"group_vector_median": float(np.mean(
                  [np.mean(list(per[w][c].values())) for c in OOD]))}) for w in per}
    fm = {w: (float(np.mean([np.mean(list(v.values())) for v in fused[w].values()]))
              if fused[w] else None) for w in fused}
    d_dense = gv["prf"]["group_vector_median"] - gv["base"]["group_vector_median"]
    d_fused = None if fm["base"] is None else fm["prf"] - fm["base"]

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "config": {"alpha": ALPHA, "beta": BETA, "depth": DEPTH,
                   "source": "arXiv 2205.00235, published config, used verbatim -- no grid"},
        "group_vector": gv, "fused_macro": fm,
        "per_component_means": {w: {c: float(np.mean(list(v.values())))
                                    for c, v in per[w].items()} for w in per},
        "delta_group_vector": d_dense, "delta_fused_macro": d_fused,
        "mean_cosine_q_to_qprime": drift,
        "bar": BAR, "cleared_bar": bool(d_dense >= BAR),
        "verdict": ("PRF CLEARS -- route to a registered system candidate" if d_dense >= BAR
                    else "NO SURVIVOR -- train-free query-side post-hoc refinement does not "
                         "recover the bag-query gap"),
        "cost_disclosure": ("TWO ANN queries per search plus a k-vector fetch, against one for R0. "
                            "The edge story changes from a single round trip to two, and the "
                            "feedback vectors must be retrievable at query time."),
        "is_a_system_change": ("qualifying_system, never qualifying_table: no row changed. Any "
                               "write-up must decompose it as a retrieval-procedure change."),
        "components": comps, "smoke": bool(a.smoke),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps({"delta_group_vector": d_dense, "delta_fused_macro": d_fused,
                      "verdict": out["verdict"]}, indent=2))
    dest = (RESULTS / "m8_vector_prf.SMOKE.json") if a.smoke else OUT
    if a.smoke:
        dest.write_text(json.dumps(out, indent=2, default=str))
    else:
        probe_guard.write_result(dest, out, "VECTOR-PRF")
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
