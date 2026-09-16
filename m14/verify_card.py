#!/usr/bin/env python3
"""Execute the M14 card usage block against local preview bytes with Hub access refused."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "m14/MODEL_CARD.md"
NANO = REPO / "work/m14-preview/staging"
DOC = REPO / "work/release/stella-doc-onnx"
FASTEMBED_CHECKOUT = REPO / "work/m14-preview/fastembed"
FASTEMBED_BRANCH = "constella-research-preview"
FASTEMBED_COMMIT = "47a50907415a90f26d4b49d7a01efe210848813e"
NANO_NAME = "DylanCouzon/constella-nano"
DOC_NAME = "DylanCouzon/stella-en-400M-v5-doc-onnx"
EXPECTED_HASHES = {
    NANO / "model.onnx": "9ba0acf57b71dc31bc5512c5445078a797fa51cf3e85587d6b8a506bfc55dbc2",
    DOC / "model.onnx": "fe31555e2b40767e17487885fb67dcdf0dcee11bef31f42478e55c1ec69a4ea9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    card_bytes = subprocess.run(
        ["git", "show", ":m14/MODEL_CARD.md"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if CARD.read_bytes() != card_bytes:
        raise RuntimeError("working-tree model card differs from the staged bytes")
    card = card_bytes.decode()
    card_blob = subprocess.run(
        ["git", "rev-parse", ":m14/MODEL_CARD.md"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert 'NAME = "DylanCouzon/constella-nano"' in card
    assert 'DOC_NAME = "DylanCouzon/stella-en-400M-v5-doc-onnx"' in card
    assert "REPO_ID" not in card
    assert "add_custom_model(" not in card
    assert "Dylancouzon/fastembed.git@constella-research-preview" in card

    for directory in (NANO, DOC, FASTEMBED_CHECKOUT):
        if not directory.is_dir():
            raise FileNotFoundError(f"required offline directory is missing: {directory}")
    actual_hashes = {path: sha256_file(path) for path in EXPECTED_HASHES}
    assert actual_hashes == EXPECTED_HASHES

    commit = subprocess.run(
        ["git", "-C", str(FASTEMBED_CHECKOUT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == FASTEMBED_COMMIT
    branch = subprocess.run(
        ["git", "-C", str(FASTEMBED_CHECKOUT), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == FASTEMBED_BRANCH

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CONSTELLA_NANO_PATH": str(NANO),
            "CONSTELLA_DOC_PATH": str(DOC),
        }
    )
    os.environ.pop("QDRANT_URL", None)

    import fastembed
    import fastembed.common.model_management as model_management
    from fastembed import TextEmbedding

    module_path = Path(fastembed.__file__).resolve()
    if not module_path.is_relative_to(FASTEMBED_CHECKOUT.resolve()):
        raise RuntimeError(f"stale FastEmbed import path: {module_path}")

    hub_calls = 0

    def refuse_hub(*args, **kwargs):
        nonlocal hub_calls
        hub_calls += 1
        raise RuntimeError("Hub access refused during offline card verification")

    model_management.snapshot_download = refuse_hub

    def refuse_bridge(*args, **kwargs):
        raise RuntimeError("add_custom_model bridge is not permitted on the native card path")

    TextEmbedding.add_custom_model = refuse_bridge

    start = card.index("<!-- card-usage-start -->")
    end = card.index("<!-- card-usage-end -->")
    usage = card[start:end]
    snippets = re.findall(r"```python\n(.*?)\n```", usage, flags=re.DOTALL)
    assert len(snippets) == 1
    script = "\n\n".join(snippets)
    script = script.replace(
        "TextEmbedding(DOC_NAME)",
        "TextEmbedding(DOC_NAME, specific_model_path=str(DOC))",
    )
    script = script.replace(
        "TextEmbedding(NAME)",
        "TextEmbedding(NAME, specific_model_path=str(NANO))",
    )
    namespace: dict[str, object] = {}
    namespace.update({"DOC": DOC, "NANO": NANO})
    exec(compile(script, f"{CARD} (staged usage snippets)", "exec"), namespace)

    query_model = namespace["query_model"]
    doc_model = namespace["document_model"]
    q = namespace["query_embedding"]
    docs = namespace["documents"]
    hits = namespace["results"]
    assert namespace["NAME"] == NANO_NAME
    assert namespace["DOC_NAME"] == DOC_NAME
    assert type(query_model.model).__name__ == "PooledNormalizedEmbedding"
    assert type(doc_model.model).__name__ == "OnnxTextEmbedding"
    assert q.shape == (1024,)
    assert q.dtype == np.dtype(np.float32)
    assert q.nbytes == 4096
    assert np.isfinite(q).all()
    assert namespace["client"]._client.__class__.__name__ == "QdrantLocal"
    assert hits[0].payload["text"] == docs[0]
    assert hub_calls == 0

    receipt = {
        "stage": "S5",
        "status": "PASSED",
        "card": str(CARD.relative_to(REPO)),
        "card_source": "git index (:m14/MODEL_CARD.md)",
        "card_git_blob": card_blob,
        "card_sha256": hashlib.sha256(card_bytes).hexdigest(),
        "working_tree_matches_staged": True,
        "usage_block_executed": True,
        "usage_python_blocks_executed": len(snippets),
        "repo_ids": {"query": NANO_NAME, "document": DOC_NAME},
        "offline": {
            "hub_access_refused": True,
            "hub_calls": hub_calls,
            "nano_directory": str(NANO.relative_to(REPO)),
            "document_directory": str(DOC.relative_to(REPO)),
            "model_sha256": {
                str(path.relative_to(REPO)): digest for path, digest in actual_hashes.items()
            },
        },
        "fastembed": {
            "checkout_path": str(FASTEMBED_CHECKOUT.resolve()),
            "branch": branch,
            "commit": commit,
            "module_path": str(module_path),
            "stale_import_path_refused": True,
            "query_family": type(query_model.model).__name__,
            "document_family": type(doc_model.model).__name__,
            "add_custom_model_refused": True,
        },
        "outputs": {
            "native_query": {
                "dtype": q.dtype.name,
                "per_vector_bytes": int(q.nbytes),
            },
            "query_shape": list(q.shape),
            "query_norm": float(np.linalg.norm(q)),
            "qdrant": {
                "mode": "local in-memory",
                "collection": "documents",
                "size": 1024,
                "distance": "Cosine",
                "nano_ranking": [
                    {"document": hit.payload["text"], "score": float(hit.score)} for hit in hits
                ],
            },
            "all_finite": True,
        },
        "passed": True,
    }
    print("M14_CARD_RECEIPT=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
