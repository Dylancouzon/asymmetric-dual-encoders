"""Re-encode Model2Vec statics with their native loader (matches official MTEB runs;
the sentence-transformers wrapper deviates by up to +0.003, see research/verification-m2.md B2)."""
import numpy as np
from model2vec import StaticModel

from core import DATASETS, evaluate, load_beir, record, save_vecs

MODELS = {"potion-base-8M": "minishlab/potion-base-8M", "potion-retrieval-32M": "minishlab/potion-retrieval-32M"}


def norm(v):
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


for slug, hf_id in MODELS.items():
    m = StaticModel.from_pretrained(hf_id)
    for ds in DATASETS:
        doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
        dv, qv = norm(m.encode(doc_texts)), norm(m.encode(q_texts))
        meta = {"loader": "model2vec", "dtype": "fp32"}
        save_vecs(slug, ds, "doc", doc_ids, dv, meta=meta)
        save_vecs(slug, ds, "query", q_ids, qv, meta=meta)
        record(slug, ds, evaluate(doc_ids, dv, q_ids, qv, qrels), extra=meta)
