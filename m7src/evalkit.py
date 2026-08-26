"""Chunked GPU brute-force retrieval + per-query nDCG@10. Never assumes a corpus fits VRAM.

The score matrix is tiled over BOTH axes under an explicit byte budget: the 5.23M-doc HotpotQA
dev component has 7,405 queries, so a naive (n_queries x 400K) fp32 block would be 11.8 GB on a
10 GB card. Documents are the outer loop so each chunk is read from disk exactly once.
"""
import numpy as np
import pytrec_eval
import torch

NEG = -3.4e38


def topk_ids_scores(q_vecs, doc_vecs, doc_ids, k=100, chunk=250_000, device="cuda", qids=None,
                    budget_bytes=1 << 30):
    nq, nd = len(q_vecs), len(doc_vecs)
    k_eff = min(k, nd)
    qtile = max(1, min(nq, budget_bytes // (4 * min(chunk, nd))))
    q = torch.from_numpy(np.ascontiguousarray(q_vecs, dtype=np.float32)).to(device)
    best_s = torch.full((nq, k_eff), NEG, device=device)
    best_i = torch.zeros((nq, k_eff), dtype=torch.long, device=device)
    for lo in range(0, nd, chunk):
        hi = min(lo + chunk, nd)
        d = torch.from_numpy(np.ascontiguousarray(doc_vecs[lo:hi])).to(device).float()
        for qlo in range(0, nq, qtile):
            qhi = min(qlo + qtile, nq)
            s = q[qlo:qhi] @ d.T
            kk = min(k_eff, s.shape[1])
            cs, ci = torch.topk(s, kk, dim=1)
            del s
            cat_s = torch.cat([best_s[qlo:qhi], cs], 1)
            cat_i = torch.cat([best_i[qlo:qhi], ci + lo], 1)
            ns, order = torch.topk(cat_s, k_eff, dim=1)
            best_s[qlo:qhi] = ns
            best_i[qlo:qhi] = torch.gather(cat_i, 1, order)
        del d
    bi, bs = best_i.cpu().numpy(), best_s.cpu().numpy()
    del q, best_s, best_i
    torch.cuda.empty_cache()
    run = {}
    for qi, qid in enumerate(qids):
        run[qid] = {doc_ids[int(j)]: float(bs[qi, r]) for r, j in enumerate(bi[qi])
                    if bs[qi, r] > NEG and doc_ids[int(j)] != qid}
    return run


def per_query_ndcg(run, qrels, cut=10):
    ev = pytrec_eval.RelevanceEvaluator({k: v for k, v in qrels.items() if k in run}, {f"ndcg_cut.{cut}"})
    return {q: s[f"ndcg_cut_{cut}"] for q, s in ev.evaluate(run).items()}


def score(q_vecs, qids, doc_vecs, doc_ids, qrels, k=100, chunk=250_000):
    return per_query_ndcg(topk_ids_scores(q_vecs, doc_vecs, doc_ids, k=k, chunk=chunk, qids=qids), qrels)


def macro(per_component):
    """per_component: {name: {qid: ndcg}} -> equal weight per component."""
    means = {n: float(np.mean(list(v.values()))) for n, v in per_component.items()}
    return float(np.mean(list(means.values()))), means
