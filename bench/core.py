"""BEIR loading, brute-force retrieval, and nDCG scoring. Shared by every model runner."""
import json
import os
from pathlib import Path

import numpy as np
import pytrec_eval
from datasets import load_dataset

DATASETS = os.environ.get("BENCH_DATASETS", "scifact,nfcorpus,fiqa,arguana,scidocs").split(",")
REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts"
RESULTS = REPO / "results" / "quality.json"


def doc_text(row):
    title = (row.get("title") or "").strip()
    return f"{title} {row['text']}".strip() if title else row["text"].strip()


def load_beir(name):
    """Returns (doc_ids, doc_texts, query_ids, query_texts, qrels_dict) for test split."""
    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_rows = load_dataset(f"BeIR/{name}-qrels", split="test")
    qrels = {}
    for r in qrels_rows:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    # only queries that have test qrels (BEIR convention: queries file spans all splits)
    doc_ids = [str(x) for x in corpus["_id"]]
    doc_texts = [doc_text(r) for r in corpus]
    q_ids, q_texts = [], []
    for qid, text in zip(queries["_id"], queries["text"]):
        if str(qid) in qrels:
            q_ids.append(str(qid))
            q_texts.append(text)
    return doc_ids, doc_texts, q_ids, q_texts, qrels


def score_run(run, qrels):
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
    scores = ev.evaluate(run)
    return {
        "ndcg@10": float(np.mean([s["ndcg_cut_10"] for s in scores.values()])),
        "recall@100": float(np.mean([s["recall_100"] for s in scores.values()])),
        "n_queries": len(scores),
    }


def topk_run(doc_ids, sims, q_ids, k=1000):
    """sims: (N_q, N_docs) score matrix -> trec run dict of top-k per query.

    Drops doc_id == query_id rows (BEIR's ignore_identical_ids): in ArguAna the query
    itself is a corpus doc with the same id and must not count as a retrieved result.
    """
    k = min(k, sims.shape[1])
    # stable sort so score ties (e.g. all-zero sparse scores) break by doc index deterministically
    topk = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return {
        qid: {doc_ids[i]: float(sims[qi, i]) for i in topk[qi] if doc_ids[i] != qid}
        for qi, qid in enumerate(q_ids)
    }


def evaluate(doc_ids, doc_vecs, q_ids, q_vecs, qrels, k=1000):
    """Brute-force dot-product top-k + pytrec_eval. Vectors must be L2-normalized."""
    sims = q_vecs.astype(np.float32) @ doc_vecs.astype(np.float32).T
    return score_run(topk_run(doc_ids, sims, q_ids, k), qrels)


def fuse_linear(run_a, run_b, w_a=0.7, w_b=0.3, eps=1e-8):
    """LightRetriever's hybrid fusion: per-query min-max normalize each run's scores, weighted sum."""
    fused = {}
    for run, w in ((run_a, w_a), (run_b, w_b)):
        for qid, passages in run.items():
            out = fused.setdefault(qid, {})
            vals = np.array(list(passages.values()))
            lo, hi = vals.min(), vals.max()
            for pid, s in passages.items():
                out[pid] = out.get(pid, 0.0) + w * (s - lo) / (hi - lo + eps)
    return fused


def save_vecs(model_slug, dataset, kind, ids, vecs, meta=None):
    d = ARTIFACTS / model_slug / dataset
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{kind}_vecs.npy", vecs.astype(np.float16))
    (d / f"{kind}_ids.json").write_text(json.dumps(ids))
    if meta:
        (d / f"{kind}_meta.json").write_text(json.dumps(meta, sort_keys=True))


def load_vecs(model_slug, dataset, kind, expect_meta=None):
    d = ARTIFACTS / model_slug / dataset
    p = d / f"{kind}_vecs.npy"
    if not p.exists():
        return None, None
    mp = d / f"{kind}_meta.json"
    if expect_meta is not None and mp.exists():
        cached = json.loads(mp.read_text())
        if cached != expect_meta:
            raise RuntimeError(f"stale cache {p}: {cached} != {expect_meta}; delete the directory to re-encode")
    return json.loads((d / f"{kind}_ids.json").read_text()), np.load(p)


def record(model_slug, dataset, metrics, extra=None):
    # read-modify-write of one JSON file: NOT safe under concurrent runners (jobs run sequentially)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    all_r = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    all_r.setdefault(model_slug, {})[dataset] = {**metrics, **(extra or {})}
    RESULTS.write_text(json.dumps(all_r, indent=1, sort_keys=True))
    print(f"[{model_slug} / {dataset}] {metrics}")
