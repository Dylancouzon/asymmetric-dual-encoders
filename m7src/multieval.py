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


def rank_compare(bi_a, bs_a, bi_b, bs_b, cut=10):
    """How two retrieval paths differ in what they RETURN, not only in what they score.

    Identical per-query nDCG proves the historical statistics are unchanged; it does not prove the
    rankings match, because top-10 membership can change entirely among non-relevant documents and
    leave nDCG untouched (Codex review #3b MAJOR 4). This reports ordered top-`cut` changes, set
    changes at `cut` and at full depth, and the largest score deviation over documents both paths
    retrieved."""
    n = bi_a.shape[0]
    ord_changed = int((bi_a[:, :cut] != bi_b[:, :cut]).any(1).sum())
    set_cut = sum(1 for i in range(n) if set(bi_a[i, :cut]) != set(bi_b[i, :cut]))
    set_full = sum(1 for i in range(n) if set(bi_a[i]) != set(bi_b[i]))
    worst = 0.0
    for i in range(n):
        m_a = dict(zip(bi_a[i].tolist(), bs_a[i].tolist()))
        for j, s in zip(bi_b[i].tolist(), bs_b[i].tolist()):
            if j in m_a:
                worst = max(worst, abs(m_a[j] - s))
    return {"n": n, f"changed_ordered_top{cut}": ord_changed,
            f"changed_top{cut}_set": set_cut, "changed_topk_set": set_full,
            "max_score_dev_matched_docs": float(worst)}


def eval_makers(makers, components=None, k=100, verbose=True, max_docs=None, pair_checks=()):
    """-> {tag: {comp: {qid: ndcg}}} with unrounded per-query values.

    `max_docs` truncates every corpus, which makes the nDCG values meaningless but exercises the
    real code path cheaply -- the only honest way to smoke a component whose corpus is 6.17M rows.
    Callers must label any result produced with it.

    `pair_checks` is a list of (tag_a, tag_b): both are compared with `rank_compare` inside the
    same pass, since the top-k arrays are gone once the pass ends."""
    if k < 11:
        raise ValueError(f"k={k}: nDCG@10 needs at least 11 retrieved documents, because one "
                         "self-hit may be dropped from every run")
    comps = list(components or dev_eval.dev_components())
    per = {tag: {} for tag in makers}
    ranks = {}
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
        span = {(tag, comp): (start, n) for tag, comp, _, _, start, n in blocks}
        for tag, comp, q_ids, qrels, start, n in blocks:
            if len(set(q_ids)) != len(q_ids):
                raise AssertionError(f"{comp}: duplicate qids")
            run = run_from_arrays(bi[start:start + n], bs[start:start + n], doc_ids, q_ids)
            nd = per_query_ndcg(run, qrels)
            # pytrec_eval silently drops a query with no qrels entry, so equality here is what
            # proves the component was scored whole (Codex review #3b MINOR).
            if set(nd) != set(q_ids):
                raise AssertionError(f"{tag}/{comp}: scored {len(nd)} of {len(q_ids)} queries "
                                     f"(missing {sorted(set(q_ids) - set(nd))[:5]})")
            per[tag][comp] = nd
            del run, nd
        for ta, tb in pair_checks:
            for c in gcomps:
                (sa, na), (sb, nb) = span[(ta, c)], span[(tb, c)]
                ranks.setdefault(f"{ta}|vs|{tb}", {})[c] = rank_compare(
                    bi[sa:sa + na], bs[sa:sa + na], bi[sb:sb + nb], bs[sb:sb + nb])
        del bi, bs
        if verbose:
            print(f"  [{key}] scored, rss {rss_gb():.1f} GB", flush=True)
    return (per, ranks) if pair_checks else per


def macro(per_comp):
    return float(np.mean([float(np.mean(list(v.values()))) for v in per_comp.values()]))


def means(per_comp):
    return {c: float(np.mean(list(v.values()))) for c, v in per_comp.items()}
