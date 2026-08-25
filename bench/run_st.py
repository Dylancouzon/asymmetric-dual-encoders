"""Run sentence-transformers-compatible models over the BEIR subset.

Usage: python bench/run_st.py [model_slug ...]   (default: all in REGISTRY)
"""
import gc
import sys

import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer

from core import DATASETS, evaluate, load_beir, load_vecs, record, save_vecs

BGE_Q = "Represent this sentence for searching relevant passages: "

# slug -> (hf_id, query_prefix, passage_prefix)
REGISTRY = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", BGE_Q, ""),
    "e5-small-v2": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "gte-small": ("thenlper/gte-small", "", ""),
    "arctic-embed-xs": ("Snowflake/snowflake-arctic-embed-xs", BGE_Q, ""),
    "arctic-embed-s": ("Snowflake/snowflake-arctic-embed-s", BGE_Q, ""),
    "granite-small-r2": ("ibm-granite/granite-embedding-small-english-r2", "", ""),
    "potion-base-8M": ("minishlab/potion-base-8M", "", ""),
    "potion-retrieval-32M": ("minishlab/potion-retrieval-32M", "", ""),
    "static-retrieval-mrl-en-v1": ("sentence-transformers/static-retrieval-mrl-en-v1", "", ""),
    "arctic-embed-m-v1.5": ("Snowflake/snowflake-arctic-embed-m-v1.5", BGE_Q, ""),
    "mdbr-leaf-ir": ("MongoDB/mdbr-leaf-ir", BGE_Q, ""),
}


def encode(model, texts, prefix, bs=256):
    return model.encode(
        [prefix + t for t in texts],
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def run(slug):
    hf_id, q_pre_default, p_pre = REGISTRY[slug]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # dtype must be pinned: transformers 5.x defaults to the checkpoint's own dtype
    # (granite=bf16, gte=fp16), which made those two configs non-comparable (verification B1)
    model = SentenceTransformer(hf_id, device=device, model_kwargs={"dtype": torch.float32})
    bs = 256 if (model.max_seq_length or 512) <= 512 else 4  # long-context models: quadratic attention OOMs on long-doc corpora
    for ds in DATASETS:
        q_pre = q_pre_default
        doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
        doc_meta = {"dtype": "fp32", "p_prefix": p_pre, "max_seq": model.max_seq_length}
        cached_ids, cached = load_vecs(slug, ds, "doc", expect_meta=doc_meta)
        if cached is None:
            vecs = encode(model, doc_texts, p_pre, bs=bs)
            save_vecs(slug, ds, "doc", doc_ids, vecs, meta=doc_meta)
        else:
            doc_ids, vecs = cached_ids, cached
        q_vecs = encode(model, q_texts, q_pre, bs=bs)
        save_vecs(slug, ds, "query", q_ids, q_vecs)
        prov = {"dtype": "fp32", "q_prefix": q_pre, "p_prefix": p_pre,
                "max_seq": model.max_seq_length, "transformers": transformers.__version__}
        record(slug, ds, evaluate(doc_ids, vecs, q_ids, q_vecs, qrels), extra=prov)
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    slugs = sys.argv[1:] or list(REGISTRY)
    for s in slugs:
        run(s)
