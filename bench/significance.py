"""Paired bootstrap CIs for the report's key comparisons.

Regenerates each system's run from cached artifacts, collects per-query nDCG@10 via
pytrec_eval, then bootstraps (B=10k, resample queries within each dataset) the
macro-average difference. Prints delta, 95% CI, p (two-sided sign of the effect).
"""
import json
import sys
from collections import Counter

import numpy as np
import pytrec_eval

from core import DATASETS, load_beir, load_vecs, topk_run
from run_lightretriever import LR_DIR, SLUG as LR_SLUG

B = 10_000
rng = np.random.default_rng(0)


def per_query_ndcg(run, qrels):
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    return {q: s["ndcg_cut_10"] for q, s in ev.evaluate(run).items()}


def run_dense(slug, ds, qrels):
    doc_ids, doc_vecs = load_vecs(slug, ds, "doc")
    q_ids, q_vecs = load_vecs(slug, ds, "query")
    sims = q_vecs.astype(np.float32) @ doc_vecs.astype(np.float32).T
    return topk_run(doc_ids, sims, q_ids)


def run_asym_pair(doc_slug, q_slug, ds, qrels):
    doc_ids, doc_vecs = load_vecs(doc_slug, ds, "doc")
    q_ids, q_vecs = load_vecs(q_slug, ds, "query")
    sims = q_vecs.astype(np.float32) @ doc_vecs.astype(np.float32).T
    return topk_run(doc_ids, sims, q_ids)


def run_lr_dense(table_name, ds, qrels):
    from transformers import AutoTokenizer

    from run_lightretriever import queries_from_table, query_token_ids

    tok = AutoTokenizer.from_pretrained("lightretriever/lightretriever-qwen2.5-1.5b")
    doc_ids, doc_vecs = load_vecs(LR_SLUG, ds, "doc")
    _, _, q_ids, q_texts, _ = load_beir(ds)
    table = np.load(LR_DIR / f"table_{table_name}.npy").astype(np.float32)
    qv = queries_from_table(table, query_token_ids(tok, q_texts))
    sims = qv @ doc_vecs.astype(np.float32).T
    return topk_run(doc_ids, sims, q_ids)


def run_lr_sparse(ds, qrels):
    from transformers import AutoTokenizer

    from run_lightretriever import query_token_ids

    tok = AutoTokenizer.from_pretrained("lightretriever/lightretriever-qwen2.5-1.5b")
    doc_ids, _ = load_vecs(LR_SLUG, ds, "doc")
    _, _, q_ids, q_texts, _ = load_beir(ds)
    cols = json.loads((LR_DIR / ds / "sparse_cols.json").read_text())
    col_pos = {c: i for i, c in enumerate(cols)}
    sparse_docs = np.load(LR_DIR / ds / "doc_sparse.npy").astype(np.float32)
    qs = np.zeros((len(q_ids), len(cols)), dtype=np.float32)
    for i, ids in enumerate(query_token_ids(tok, q_texts)):
        for t, c in Counter(ids).items():
            qs[i, col_pos[t]] = c
    return topk_run(doc_ids, qs @ sparse_docs.T, q_ids)


def run_lr_hybrid(table_name, ds, qrels):
    from core import fuse_linear

    return fuse_linear(run_lr_dense(table_name, ds, qrels), run_lr_sparse(ds, qrels))


def run_opensearch(ds, qrels):
    from transformers import AutoTokenizer

    from core import ARTIFACTS
    from run_opensearch import MODEL_ID, load_idf

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    idf = load_idf(tok)
    d = ARTIFACTS / "opensearch-doc-v3-gte" / ds
    cols = json.loads((d / "sparse_cols.json").read_text())
    col_pos = {c: i for i, c in enumerate(cols)}
    docs = np.load(d / "doc_sparse.npy").astype(np.float32)
    doc_ids, _, q_ids, q_texts, _ = load_beir(ds)
    qv = np.zeros((len(q_ids), len(cols)), dtype=np.float32)
    for i, text in enumerate(q_texts):
        for t in set(tok(text, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]):
            if t in col_pos:
                qv[i, col_pos[t]] = idf.get(t, 0.0)
    return topk_run(doc_ids, qv @ docs.T, q_ids)


def run_bm25(ds, qrels):
    import Stemmer
    import bm25s

    stemmer = Stemmer.Stemmer("english")
    doc_ids, doc_texts, q_ids, q_texts, _ = load_beir(ds)
    retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    retriever.index(bm25s.tokenize(doc_texts, stopwords="en", stemmer=stemmer, show_progress=False), show_progress=False)
    ids, sc = retriever.retrieve(bm25s.tokenize(q_texts, stopwords="en", stemmer=stemmer, show_progress=False), k=min(1000, len(doc_ids)), show_progress=False)
    return {qid: {doc_ids[d_]: float(s) for d_, s in zip(ids[qi], sc[qi]) if doc_ids[d_] != qid} for qi, qid in enumerate(q_ids)}


SYSTEMS = {
    "lr-dense-websearch": lambda ds, qr: run_lr_dense("websearch", ds, qr),
    "lr-dense-pertask": lambda ds, qr: run_lr_dense(ds, ds, qr),
    "lr-hybrid-websearch": lambda ds, qr: run_lr_hybrid("websearch", ds, qr),
    "lr-hybrid-pertask": lambda ds, qr: run_lr_hybrid(ds, ds, qr),
    "opensearch-doc-v3-gte": run_opensearch,
    "bm25": run_bm25,
    "bge-small-en-v1.5": lambda ds, qr: run_dense("bge-small-en-v1.5", ds, qr),
    "granite-small-r2": lambda ds, qr: run_dense("granite-small-r2", ds, qr),
    "gte-small": lambda ds, qr: run_dense("gte-small", ds, qr),
    "e5-small-v2": lambda ds, qr: run_dense("e5-small-v2", ds, qr),
    "arctic-embed-s": lambda ds, qr: run_dense("arctic-embed-s", ds, qr),
    "potion-retrieval-32M": lambda ds, qr: run_dense("potion-retrieval-32M", ds, qr),
    "static-retrieval-mrl-en-v1": lambda ds, qr: run_dense("static-retrieval-mrl-en-v1", ds, qr),
    "leaf-ir-asym": lambda ds, qr: run_asym_pair("arctic-embed-m-v1.5", "mdbr-leaf-ir", ds, qr),
    "mdbr-leaf-ir": lambda ds, qr: run_dense("mdbr-leaf-ir", ds, qr),
    "arctic-embed-m-v1.5": lambda ds, qr: run_dense("arctic-embed-m-v1.5", ds, qr),
    "all-MiniLM-L6-v2": lambda ds, qr: run_dense("all-MiniLM-L6-v2", ds, qr),
}

PAIRS = [
    ("lr-dense-websearch", "bge-small-en-v1.5"),
    ("lr-dense-websearch", "granite-small-r2"),
    ("lr-dense-websearch", "potion-retrieval-32M"),
    ("lr-dense-websearch", "static-retrieval-mrl-en-v1"),
    ("lr-dense-websearch", "all-MiniLM-L6-v2"),
    ("lr-dense-websearch", "e5-small-v2"),
    ("lr-dense-websearch", "bm25"),
    ("lr-dense-websearch", "leaf-ir-asym"),
    ("lr-dense-pertask", "lr-dense-websearch"),
    ("lr-hybrid-websearch", "lr-dense-websearch"),
    ("lr-hybrid-pertask", "lr-dense-pertask"),
    ("opensearch-doc-v3-gte", "lr-hybrid-pertask"),
    ("opensearch-doc-v3-gte", "lr-hybrid-websearch"),
    ("opensearch-doc-v3-gte", "bm25"),
    ("opensearch-doc-v3-gte", "bge-small-en-v1.5"),
    ("opensearch-doc-v3-gte", "gte-small"),
    ("leaf-ir-asym", "mdbr-leaf-ir"),
    ("leaf-ir-asym", "arctic-embed-m-v1.5"),
    ("leaf-ir-asym", "bge-small-en-v1.5"),
    ("leaf-ir-asym", "arctic-embed-s"),
    ("granite-small-r2", "bge-small-en-v1.5"),
]

if __name__ == "__main__":
    scores = {}  # (system, ds) -> ordered per-query ndcg array (aligned by qid list per ds)
    qid_lists = {}
    for ds in DATASETS:
        *_, qrels = load_beir(ds)
        qids = None
        for name, fn in SYSTEMS.items():
            pq = per_query_ndcg(fn(ds, qrels), qrels)
            if qids is None:
                qids = sorted(pq)
                qid_lists[ds] = qids
            scores[(name, ds)] = np.array([pq[q] for q in qids])
        print(f"{ds}: per-query scores for {len(SYSTEMS)} systems", flush=True)

    out = {}
    for a, b in PAIRS:
        # macro-average over datasets; bootstrap resamples queries within each dataset
        deltas = np.zeros(B)
        base = 0.0
        for ds in DATASETS:
            da, db = scores[(a, ds)], scores[(b, ds)]
            n = len(da)
            idx = rng.integers(0, n, size=(B, n))
            deltas += (da[idx].mean(1) - db[idx].mean(1)) / len(DATASETS)
            base += (da.mean() - db.mean()) / len(DATASETS)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = 2 * min((deltas < 0).mean(), (deltas > 0).mean())
        p_str = f"<{2/B}" if p == 0 else f"{p:.4f}"  # 0 observed sign-flips is a bound, not zero
        out[f"{a} vs {b}"] = {"delta": round(base, 4), "ci95": [round(lo, 4), round(hi, 4)], "p": p_str}
        print(f"{a:28s} - {b:28s} d={base:+.4f} CI=[{lo:+.4f},{hi:+.4f}] p={p_str}", flush=True)
    json.dump(out, open("results/significance.json", "w"), indent=1)
