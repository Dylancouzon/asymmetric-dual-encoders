"""Asymmetric mixes from cached vectors (no new encoding).

leaf-asym: docs encoded by teacher arctic-embed-m-v1.5 (109M, cloud), queries by mdbr-leaf-ir (23M, edge).
"""
import numpy as np

from core import DATASETS, evaluate, load_beir, load_vecs, record

PAIRS = {"leaf-ir-asym": ("arctic-embed-m-v1.5", "mdbr-leaf-ir")}


def main():
    for slug, (doc_model, query_model) in PAIRS.items():
        for ds in DATASETS:
            doc_ids, doc_vecs = load_vecs(doc_model, ds, "doc")
            q_ids, q_vecs = load_vecs(query_model, ds, "query")
            if doc_vecs is None or q_vecs is None:
                print(f"skip {slug}/{ds}: missing artifacts")
                continue
            assert doc_vecs.shape[1] == q_vecs.shape[1], f"dim mismatch {doc_vecs.shape} vs {q_vecs.shape}"
            *_, qrels = load_beir(ds)
            record(slug, ds, evaluate(doc_ids, doc_vecs, q_ids, q_vecs, qrels))


if __name__ == "__main__":
    main()
