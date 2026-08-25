"""Chunked GPU brute-force retrieval + per-query nDCG@10. Never assumes a corpus fits VRAM."""
import numpy as np
import pytrec_eval
import torch


def topk_ids_scores(q_vecs, doc_vecs, doc_ids, k=1000, chunk=200_000, device="cuda", qids=None):
    """q_vecs (Nq,d) fp32 normalized; doc_vecs (Nd,d) fp16/fp32 array or memmap.

    Returns a trec run dict. Drops doc_id == query_id (BEIR ignore_identical_ids), matching
    bench/core.topk_run.
    """
    nq, nd = len(q_vecs), len(doc_vecs)
    k_eff = min(k, nd)
    q = torch.from_numpy(np.ascontiguousarray(q_vecs, dtype=np.float32)).to(device)
    best_s = torch.zeros((nq, 0), device=device)
    best_i = torch.zeros((nq, 0), dtype=torch.long, device=device)
    for lo in range(0, nd, chunk):
        hi = min(lo + chunk, nd)
        d = torch.from_numpy(np.ascontiguousarray(doc_vecs[lo:hi])).to(device).float()
        s = q @ d.T
        del d
        kk = min(k_eff, s.shape[1])
        cs, ci = torch.topk(s, kk, dim=1)
        del s
        cat_s = torch.cat([best_s, cs], 1)
        cat_i = torch.cat([best_i, ci + lo], 1)
        kk2 = min(k_eff, cat_s.shape[1])
        best_s, order = torch.topk(cat_s, kk2, dim=1)
        best_i = torch.gather(cat_i, 1, order)
    bi, bs = best_i.cpu().numpy(), best_s.cpu().numpy()
    del q, best_s, best_i
    torch.cuda.empty_cache()
    run = {}
    for qi, qid in enumerate(qids):
        run[qid] = {doc_ids[int(j)]: float(bs[qi, r]) for r, j in enumerate(bi[qi]) if doc_ids[int(j)] != qid}
    return run


def per_query_ndcg(run, qrels, cut=10):
    ev = pytrec_eval.RelevanceEvaluator({k: v for k, v in qrels.items() if k in run}, {f"ndcg_cut.{cut}"})
    return {q: s[f"ndcg_cut_{cut}"] for q, s in ev.evaluate(run).items()}


def score(q_vecs, qids, doc_vecs, doc_ids, qrels, k=1000, chunk=200_000):
    return per_query_ndcg(topk_ids_scores(q_vecs, doc_vecs, doc_ids, k=k, chunk=chunk, qids=qids), qrels)


def macro(per_component):
    """per_component: {name: {qid: ndcg}} -> equal weight per component."""
    means = {n: float(np.mean(list(v.values()))) for n, v in per_component.items()}
    return float(np.mean(list(means.values()))), means
