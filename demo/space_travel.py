#!/usr/bin/env python3
"""Search a playful travel guide with both Constella query encoders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

DOCUMENT_MODEL = "DylanCouzon/stella-en-400M-v5-doc-onnx"
QUERY_MODELS = {
    "nano": "DylanCouzon/constella-nano",
    "zero": "DylanCouzon/constella-zero",
}
MODEL_PATH_VARIABLES = {
    DOCUMENT_MODEL: "CONSTELLA_DOC_PATH",
    QUERY_MODELS["nano"]: "CONSTELLA_NANO_PATH",
    QUERY_MODELS["zero"]: "CONSTELLA_ZERO_PATH",
}


def load_model(model_name: str) -> TextEmbedding:
    """Load from Hugging Face, or from an optional pre-downloaded model directory."""
    path = os.environ.get(MODEL_PATH_VARIABLES[model_name])
    if path:
        return TextEmbedding(model_name, specific_model_path=path)
    return TextEmbedding(model_name)


def load_guide() -> dict:
    path = Path(__file__).with_name("space_travel.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(documents: list[dict]) -> QdrantClient:
    print("Encoding the travel guide once with Stella...")
    encoder = load_model(DOCUMENT_MODEL)
    embeddings = encoder.embed(document["text"] for document in documents)

    client = QdrantClient(":memory:")
    client.create_collection(
        "travel-guide",
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE,
        ),
    )
    client.upsert(
        "travel-guide",
        points=[
            models.PointStruct(
                id=index,
                vector=embedding.tolist(),
                payload=document,
            )
            for index, (document, embedding) in enumerate(zip(documents, embeddings))
        ],
    )
    return client


def search(
    client: QdrantClient,
    model_label: str,
    queries: list[dict],
) -> int:
    encoder = load_model(QUERY_MODELS[model_label])
    query_embeddings = encoder.embed(query["text"] for query in queries)
    failures = 0

    print(f"\n{model_label.upper()} — same document index")
    for query, embedding in zip(queries, query_embeddings):
        result = client.query_points(
            "travel-guide",
            query=embedding.tolist(),
            limit=1,
        ).points[0]
        matched = result.payload["id"] == query["expected_id"]
        failures += not matched
        marker = "✓" if matched else "✗"
        print(f'{marker} {query["text"]}')
        print(f'  → {result.payload["title"]} ({result.score:.4f})')

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("both", "nano", "zero"),
        default="both",
        help="query encoder to demonstrate (default: both)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    guide = load_guide()
    client = build_index(guide["documents"])
    selected = QUERY_MODELS if args.model == "both" else (args.model,)
    failures = sum(search(client, label, guide["queries"]) for label in selected)

    if failures:
        print(f"\nSmoke test failed: {failures} unexpected top result(s).")
        return 1

    print("\nSmoke test passed: every expected destination ranked first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
