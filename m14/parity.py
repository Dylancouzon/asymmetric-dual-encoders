#!/usr/bin/env python3
"""M14 S3: frozen real-fixture parity for the staged constella-nano bundle."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STAGING = REPO / "work/m14-preview/staging"
ARTIFACT_REPO = REPO / "work/m13cloud"
FREEZE_PATH = REPO / "m10/FREEZE.json"
FIXTURE_PATH = REPO / "m11/release/doc_fixtures.json"

# Fixed before the first M14 parity inference. The cosine bar is M13's registered release bar.
THRESHOLDS = {
    "minimum_true_cosine": 0.9999,
    "maximum_absolute_error": 1e-5,
    "maximum_row_norm_deviation": 1e-5,
    "negative_control_maximum_cosine": 0.99,
    "negative_control_minimum_max_absolute_difference": 1e-3,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def true_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(an * bn, 1e-12)


def comparison(a: np.ndarray, b: np.ndarray) -> dict:
    cosines = true_cosines(a, b)
    return {
        "n_rows": int(a.shape[0]),
        "minimum_true_cosine": float(cosines.min()),
        "maximum_absolute_error": float(np.max(np.abs(a - b))),
        "passed": bool(
            cosines.min() >= THRESHOLDS["minimum_true_cosine"]
            and np.max(np.abs(a - b)) <= THRESHOLDS["maximum_absolute_error"]
        ),
    }


def encode_torch(model, tokenizer, texts: list[str], batch_size: int = 2) -> np.ndarray:
    import torch

    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            rows.append(model(batch["input_ids"], batch["attention_mask"]).cpu().numpy())
    return np.concatenate(rows).astype(np.float32, copy=False)


def encode_ort(session, tokenizer, texts: list[str], batch_size: int = 2) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        mask = batch["attention_mask"].astype(np.float32)[..., None]
        token_embeddings = session.run(
            ["token_embeddings"],
            {
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
            },
        )[0]
        pooled = (token_embeddings * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        rows.append(pooled)
    return np.concatenate(rows).astype(np.float32, copy=False)


def main() -> None:
    # Keep model construction local-only; parity must not silently fetch a different backbone.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("M14 S3 thresholds fixed before inference: " + json.dumps(THRESHOLDS, sort_keys=True),
          file=sys.stderr, flush=True)

    import fastembed
    import onnx
    import onnxruntime as ort
    import torch
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    model_path = STAGING / "model.onnx"
    graph = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(graph)

    opsets = [{"domain": x.domain or "ai.onnx", "version": int(x.version)}
              for x in graph.opset_import]
    node_domains = Counter(n.domain or "ai.onnx" for n in graph.graph.node)
    initializer_dtypes = Counter(
        onnx.TensorProto.DataType.Name(x.data_type) for x in graph.graph.initializer
    )
    input_dtypes = {
        x.name: onnx.TensorProto.DataType.Name(x.type.tensor_type.elem_type)
        for x in graph.graph.input
    }
    output_dtypes = {
        x.name: onnx.TensorProto.DataType.Name(x.type.tensor_type.elem_type)
        for x in graph.graph.output
    }
    custom_domains = sorted(set(node_domains) - {"ai.onnx", "ai.onnx.ml"})
    assert opsets == [{"domain": "ai.onnx", "version": 17}]
    assert custom_domains == []
    assert initializer_dtypes == {"FLOAT": 199}
    assert input_dtypes == {"input_ids": "INT64", "attention_mask": "INT64"}
    assert output_dtypes == {"token_embeddings": "FLOAT"}

    tokenizer = AutoTokenizer.from_pretrained(STAGING, local_files_only=True)
    fixture_source = json.loads(FIXTURE_PATH.read_text())
    selections = [
        ("tiny_0", "tiny", 0, None),
        ("tiny_1", "tiny", 1, None),
        ("short_0", "short", 0, None),
        ("mid_0", "mid", 0, None),
        ("near_0", "near", 0, None),
        ("boundary_511", "boundary", 0, 511),
        ("boundary_512", "boundary", 1, 512),
        ("boundary_513", "boundary", 8, 513),
        ("over_0", "over", 0, None),
    ]
    over_counts = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in fixture_source["over"]
    ]
    selections.append(("over_longest", "over", int(np.argmax(over_counts)), None))

    fixtures = []
    texts = []
    for fixture_id, group, index, required_tokens in selections:
        text = fixture_source[group][index]
        raw_tokens = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        served_tokens = len(tokenizer(
            text, add_special_tokens=True, truncation=True, max_length=512
        )["input_ids"])
        if required_tokens is not None:
            assert raw_tokens == required_tokens
        fixtures.append({
            "id": fixture_id,
            "source_group": group,
            "source_index": index,
            "characters": len(text),
            "text_sha256": sha256_text(text),
            "tokens_untruncated": raw_tokens,
            "tokens_served": served_tokens,
            "truncated": raw_tokens > 512,
        })
        texts.append(text)

    by_id = {row["id"]: row for row in fixtures}
    assert [by_id[f"boundary_{n}"]["tokens_untruncated"] for n in (511, 512, 513)] \
        == [511, 512, 513]
    assert by_id["boundary_511"]["tokens_served"] == 511
    assert by_id["boundary_512"]["tokens_served"] == 512
    assert by_id["boundary_513"]["tokens_served"] == 512
    assert by_id["over_0"]["tokens_untruncated"] > 512
    assert by_id["over_longest"]["tokens_untruncated"] > 512
    assert by_id["over_longest"]["tokens_served"] == 512

    freeze = json.loads(FREEZE_PATH.read_text())
    checkpoint_path = ARTIFACT_REPO / freeze["checkpoint"]
    checkpoint_sha = sha256_file(checkpoint_path)
    assert checkpoint_sha == freeze["checkpoint_sha256"]

    sys.path.insert(0, str(REPO / "m13src"))
    sys.path.insert(0, str(REPO / "m10src"))
    from score13 import Nano10Student

    student = Nano10Student(freeze, device="cpu", repo=ARTIFACT_REPO)
    student.model.tok = tokenizer
    torch_rows = encode_torch(student.model, tokenizer, texts)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_rows = encode_ort(session, tokenizer, texts)

    local_name = "m14-local/constella-nano-s3"
    TextEmbedding.add_custom_model(
        model=local_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=local_name),
        dim=1024,
        model_file="model.onnx",
        description="local M14 S3 constella-nano parity",
        license="mit",
        size_in_gb=model_path.stat().st_size / 1e9,
    )
    fastembed_model = TextEmbedding(
        model_name=local_name,
        specific_model_path=str(STAGING),
        threads=4,
    )
    fastembed_rows = np.stack(list(fastembed_model.embed(texts, batch_size=2))).astype(
        np.float32, copy=False
    )

    assert torch_rows.shape == ort_rows.shape == fastembed_rows.shape == (len(texts), 1024)
    assert np.isfinite(torch_rows).all()
    assert np.isfinite(ort_rows).all()
    assert np.isfinite(fastembed_rows).all()

    comparisons = {
        "torch_vs_direct_ort": comparison(torch_rows, ort_rows),
        "torch_vs_stock_fastembed": comparison(torch_rows, fastembed_rows),
        "direct_ort_vs_stock_fastembed": comparison(ort_rows, fastembed_rows),
    }
    norm_ranges = {}
    for name, rows in (
        ("torch_reference", torch_rows),
        ("direct_ort", ort_rows),
        ("stock_fastembed", fastembed_rows),
    ):
        norms = np.linalg.norm(rows, axis=1)
        deviation = float(np.max(np.abs(norms - 1.0)))
        norm_ranges[name] = {
            "minimum": float(norms.min()),
            "maximum": float(norms.max()),
            "maximum_deviation_from_one": deviation,
            "passed": bool(deviation <= THRESHOLDS["maximum_row_norm_deviation"]),
        }

    negative_a, negative_b = 0, 1
    neg_cos = float(true_cosines(torch_rows[[negative_a]], torch_rows[[negative_b]])[0])
    neg_max_abs = float(np.max(np.abs(torch_rows[negative_a] - torch_rows[negative_b])))
    negative_control = {
        "kind": "distinct real passages must not behave like a matched parity row",
        "first_fixture": fixtures[negative_a]["id"],
        "second_fixture": fixtures[negative_b]["id"],
        "true_cosine": neg_cos,
        "maximum_absolute_difference": neg_max_abs,
        "passed": bool(
            neg_cos <= THRESHOLDS["negative_control_maximum_cosine"]
            and neg_max_abs
            >= THRESHOLDS["negative_control_minimum_max_absolute_difference"]
        ),
    }

    passed = (
        all(row["passed"] for row in comparisons.values())
        and all(row["passed"] for row in norm_ranges.values())
        and negative_control["passed"]
    )
    receipt = {
        "stage": "S3",
        "status": "PASSED" if passed else "FAILED",
        "thresholds_fixed_before_inference": THRESHOLDS,
        "inputs": {
            "staging_directory": str(STAGING.relative_to(REPO)),
            "model_sha256": sha256_file(model_path),
            "tokenizer_sha256": sha256_file(STAGING / "tokenizer.json"),
            "freeze_path": str(FREEZE_PATH.relative_to(REPO)),
            "freeze_sha256": sha256_file(FREEZE_PATH),
            "checkpoint_path": str(checkpoint_path.relative_to(REPO)),
            "checkpoint_sha256": checkpoint_sha,
            "fixture_source": str(FIXTURE_PATH.relative_to(REPO)),
            "fixture_source_sha256": sha256_file(FIXTURE_PATH),
        },
        "fixtures": fixtures,
        "overlength_fixture_ids": [
            row["id"] for row in fixtures if row["tokens_untruncated"] > 512
        ],
        "onnx": {
            "checker_passed": True,
            "node_count": len(graph.graph.node),
            "opsets": opsets,
            "node_domains": dict(sorted(node_domains.items())),
            "custom_domains": custom_domains,
            "initializer_dtypes": dict(sorted(initializer_dtypes.items())),
            "input_dtypes": input_dtypes,
            "output_dtypes": output_dtypes,
        },
        "reference": {
            "class": "m13src/score13.py:Nano10Student",
            "tokenization": "padding=True, truncation=True, max_length=512",
            "pooling": "masked mean of per-token head output, then L2 normalization",
        },
        "direct_ort": {
            "provider": session.get_providers()[0],
            "pooling": "masked mean of token_embeddings, then L2 normalization",
        },
        "stock_fastembed": {
            "version": fastembed.__version__,
            "module_path": str(Path(fastembed.__file__).resolve()),
            "bridge": "TextEmbedding.add_custom_model",
            "pooling": "MEAN",
            "normalization": True,
        },
        "comparisons": comparisons,
        "row_norms": norm_ranges,
        "negative_control": negative_control,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "fastembed": fastembed.__version__,
            "numpy": np.__version__,
            "execution": "CPU, four threads, Hugging Face and Transformers offline",
        },
        "passed": passed,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("M14 S3 parity or negative control failed; receipt printed above")


if __name__ == "__main__":
    main()
