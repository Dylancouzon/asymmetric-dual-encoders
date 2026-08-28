"""Evaluate many query-side variants on the full pinned dev suite with ONE corpus pass each.

Why this exists: `bigram_residual.eval_variants` already put components on the outer loop so each
corpus JSON is parsed once, but it still reads the corpus VECTORS once per variant. The two
held-out components' corpus is the 6.17M-row pool (12.6 GB fp16) and HotpotQA's is 5.23M rows, so
a 22-variant audit that way is ~36 GB of memmap reads per component-variant pair. Here the
variants' query vectors are stacked into one matrix and the corpus is read exactly once per
CORPUS, not per component: `heldout-train` and `heldout-longq` share the pool (longq is literally
a 55-query subset), so they are scored in the same pass, which is also what makes their per-query
values exactly comparable for the dependence-preserving statistics.

A `maker` is `fn(comp, q_texts) -> (n, dim) float32`. That is the whole interface: the released
`QueryTable` path, the matrix shortcut, an int8 dequantize, and a count-saturation pooling rule
are all just makers, so nothing here knows what is being compared.
"""
import numpy as np

import dev_eval
from evalkit import per_query_ndcg, run_from_arrays, topk_arrays


def rss_gb():
    return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1e6


def corpus_key(comp):
    """Components that share a corpus share a pass. The held-out slices' corpus IS the pool."""
    return "pool" if comp.startswith("heldout-") else comp


def same_corpus(ids_a, vecs_a, ids_b, vecs_b):
    """Object identity is the cheap proof and is what the pool path now gives (heldout memoizes
    both), but equality is the actual requirement, so fall back to it rather than refusing a
    legitimately shared corpus that arrived as two equal objects."""
    if ids_a is ids_b and vecs_a is vecs_b:
        return True
    if getattr(vecs_a, "shape", None) != getattr(vecs_b, "shape", None) or len(ids_a) != len(ids_b):
        return False
    fa, fb = getattr(vecs_a, "filename", None), getattr(vecs_b, "filename", None)
    if fa is None or fa != fb:
        return False
    return ids_a == ids_b


def groups(components):
    out = {}
    for c in components:
        out.setdefault(corpus_key(c), []).append(c)
    return out


def eval_makers(makers, components=None, k=100, verbose=True, max_docs=None):
    """-> {tag: {comp: {qid: ndcg}}} with unrounded per-query values.

    `max_docs` truncates every corpus, which makes the nDCG values meaningless but exercises the
    real code path cheaply -- the only honest way to smoke a component whose corpus is 6.17M rows.
    Callers must label any result produced with it."""
    comps = list(components or dev_eval.dev_components())
    per = {tag: {} for tag in makers}
    for key, gcomps in groups(comps).items():
        blocks = []          # [tag, comp, q_ids, qrels, start, n]
        total = 0
        meta = {}
        doc_ids = dv = None
        for comp in gcomps:
            d_ids, _, q_ids, q_texts, qrels, d_vecs = dev_eval.doc_vecs(comp)
            if doc_ids is None:
                doc_ids, dv = d_ids, d_vecs
            elif not same_corpus(d_ids, d_vecs, doc_ids, dv):
                raise AssertionError(f"{comp} does not share {gcomps[0]}'s corpus; the shared-pass "
                                     "assumption is false and their scores are not comparable")
            meta[comp] = (q_ids, q_texts, qrels)
            for tag in makers:
                blocks.append([tag, comp, q_ids, qrels, total, len(q_texts)])
                total += len(q_texts)
        dim = None
        Q = None
        for blk in blocks:
            tag, comp, _, _, start, n = blk
            qv = np.ascontiguousarray(makers[tag](comp, meta[comp][1]), dtype=np.float32)
            if qv.shape[0] != n:
                raise AssertionError(f"{tag}/{comp}: maker returned {qv.shape[0]} rows, want {n}")
            if Q is None:
                dim = qv.shape[1]
                Q = np.empty((total, dim), dtype=np.float32)
            Q[start:start + n] = qv
            del qv
        if max_docs is not None and len(doc_ids) > max_docs:
            doc_ids, dv = doc_ids[:max_docs], dv[:max_docs]
        if verbose:
            print(f"  [{key}] {len(gcomps)} component(s) x {len(makers)} variants = {total:,} "
                  f"query rows over {len(doc_ids):,} docs, rss {rss_gb():.1f} GB", flush=True)
        bi, bs = topk_arrays(Q, dv, k=k, chunk=dev_eval.CHUNK.get(gcomps[0], 200_000))
        del Q
        for tag, comp, q_ids, qrels, start, n in blocks:
            run = run_from_arrays(bi[start:start + n], bs[start:start + n], doc_ids, q_ids)
            per[tag][comp] = per_query_ndcg(run, qrels)
            del run
        del bi, bs
        if verbose:
            print(f"  [{key}] scored, rss {rss_gb():.1f} GB", flush=True)
    return per


def macro(per_comp):
    return float(np.mean([float(np.mean(list(v.values()))) for v in per_comp.values()]))


def means(per_comp):
    return {c: float(np.mean(list(v.values()))) for c, v in per_comp.items()}
