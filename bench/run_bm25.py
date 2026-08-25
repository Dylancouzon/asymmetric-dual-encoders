"""BM25 baseline (bm25s, Lucene scoring, Snowball English stemmer, k1=1.2 b=0.75).

The canonical zero-compute-everywhere baseline. Convention differs from BEIR's official
Elasticsearch multi-field setup, so validate against published numbers before trusting deltas.
"""
import Stemmer
import bm25s

from core import DATASETS, load_beir, record, score_run

stemmer = Stemmer.Stemmer("english")

for ds in DATASETS:
    doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
    corpus_tokens = bm25s.tokenize(doc_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    q_tokens = bm25s.tokenize(q_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    ids, scores = retriever.retrieve(q_tokens, k=min(1000, len(doc_ids)), show_progress=False)
    run = {}
    for qi, qid in enumerate(q_ids):
        run[qid] = {doc_ids[d]: float(s) for d, s in zip(ids[qi], scores[qi]) if doc_ids[d] != qid}
    record("bm25", ds, score_run(run, qrels), extra={"impl": "bm25s-lucene k1=1.2 b=0.75 snowball"})
