#!/usr/bin/env python3
"""Receipt for the numbers in qdrant/fastembed#752: output dtypes and, with --compare, value drift.

Run once per FastEmbed install (e.g. PyPI 0.6.0, 0.6.1, 0.8.1 and the patched checkout):

    python m22/dtype_issue_measure.py ROWS.npz [--compare OTHER_ROWS.npz]

Prints one JSON line. ROWS.npz holds this install's embeddings for the four mean-pooling models.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

DTYPE_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
DRIFT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "nomic-ai/nomic-embed-text-v1",
    "thenlper/gte-base",
]
DOCS = ["hello world", "flag embedding", "retrieval " * 300, "a"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows")
    parser.add_argument("--compare")
    args = parser.parse_args()

    import fastembed
    from fastembed import TextEmbedding

    receipt = {"fastembed_version": fastembed.__version__, "numpy": np.__version__, "dtype": {}}
    for name in DTYPE_MODELS:
        receipt["dtype"][name] = str(next(iter(TextEmbedding(name).embed(["hello world"]))).dtype)

    rows = {name: np.stack(list(TextEmbedding(name).embed(DOCS))) for name in DRIFT_MODELS}
    np.savez(args.rows, **{k.replace("/", "__"): v for k, v in rows.items()})
    if args.compare:
        other = np.load(args.compare)
        receipt["drift_vs_other"] = {}
        for name, mine in rows.items():
            theirs = other[name.replace("/", "__")]
            receipt["drift_vs_other"][name] = {
                "dtype": str(mine.dtype),
                "other_dtype": str(theirs.dtype),
                "maximum_absolute_difference": float(
                    np.abs(mine.astype(np.float64) - theirs.astype(np.float64)).max()
                ),
                "norms": np.linalg.norm(mine.astype(np.float64), axis=1).round(6).tolist(),
                "other_norms": np.linalg.norm(theirs.astype(np.float64), axis=1).round(6).tolist(),
            }
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
