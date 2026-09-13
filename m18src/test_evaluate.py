from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m18src"))
import evaluate as E


def test_exact_search_ties_use_document_id():
    q = np.asarray([[1.0, 0.0]], np.float32)
    d = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], np.float32)
    run = E.search(q, d, k=2, doc_ids=["b", "a", "c"], query_ids=["q"])
    assert list(run["q"]) == ["a", "b"]


def test_metrics_include_dense_fused_recall_and_mrr():
    qrels = {"q": {"d": 2}}
    got = E.metric_per_query({"q": {"x": 2.0, "d": 1.0}}, qrels)
    assert got["recall@10"]["q"] == 1.0
    assert got["mrr@10"]["q"] == 0.5
    assert 0 < got["ndcg@10"]["q"] < 1


def test_bm25_roundtrip(tmp_path):
    idx = E.BM25Index(["a", "b"], ["qdrant hnsw tuning", "cooking recipe"])
    assert list(idx.query("hnsw")) == ["a"]
    idx.save(tmp_path / "bm25")
    assert E.BM25Index.load(tmp_path / "bm25").query("hnsw") == idx.query("hnsw")
