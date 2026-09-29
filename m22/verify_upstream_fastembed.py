#!/usr/bin/env python3
"""Check the Constella registrations in an installed FastEmbed against direct ONNX Runtime.

Written for upstream qdrant/fastembed#751. Uses the published Hub bytes, the M14 length-stratified
fixtures (including raw token lengths 511/512/513) and the M14 parity thresholds. The Torch leg is
not rerun: M14 measured direct ORT within 1.2e-07 of the frozen Torch reference.

    python m22/verify_upstream_fastembed.py --cache DIR --rows OUT.npz [--compare OTHER.npz]

Prints a JSON receipt. Exits non-zero if a registered parity threshold fails; a dtype that
differs from the graph's fp32 output is reported, not fatal, because it is the finding under test.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO / "m11/release/doc_fixtures.json"
MODELS = ["Qdrant/constella-nano", "Qdrant/constella-zero", "Qdrant/stella-en-400M-v5-doc-onnx"]
# Same thresholds and fixture selection as m14/parity.py.
MIN_COSINE, MAX_ABS, MAX_NORM_DEV = 0.9999, 1e-5, 1e-5
SELECTIONS = [("tiny", 0), ("tiny", 1), ("short", 0), ("mid", 0), ("near", 0),
              ("boundary", 0), ("boundary", 1), ("boundary", 8), ("over", 0)]


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.astype(np.float64), b.astype(np.float64)
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return {"minimum_true_cosine": float(cos.min()), "maximum_absolute_error": float(np.abs(a - b).max())}


def direct_ort(model_dir: Path, texts: list[str]) -> np.ndarray:
    import onnxruntime as ort
    from transformers import PreTrainedTokenizerFast

    # tokenizer.json directly: Stella's config.json names remote code, which the reference skips.
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"),
                                        pad_token="[PAD]", model_max_length=512)
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    output = session.get_outputs()[0].name
    rows = []
    for start in range(0, len(texts), 2):
        batch = tokenizer(texts[start:start + 2], padding=True, truncation=True, max_length=512,
                          return_tensors="np")
        feeds = {k: batch[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
        out = session.run([output], feeds)[0]
        if out.ndim == 3:  # Nano: token embeddings, pooled outside the graph
            mask = batch["attention_mask"].astype(np.float32)[..., None]
            out = (out * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        rows.append(out)
    return np.concatenate(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rows", required=True, help="write this install's FastEmbed rows here")
    parser.add_argument("--compare", help="rows written by another FastEmbed install")
    args = parser.parse_args()

    import fastembed
    import onnxruntime
    from fastembed import TextEmbedding

    checkout = Path(fastembed.__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, str(checkout))
    from tests.test_text_onnx_embeddings import CANONICAL_VECTOR_VALUES

    source = json.loads(FIXTURE_PATH.read_text())
    texts = [source[group][index] for group, index in SELECTIONS]
    texts.append(max(source["over"], key=len))

    receipt = {"fastembed_commit": commit, "fastembed_version": fastembed.__version__,
               "numpy": np.__version__, "onnxruntime": onnxruntime.__version__,
               "thresholds": {"minimum_true_cosine": MIN_COSINE, "maximum_absolute_error": MAX_ABS,
                              "maximum_row_norm_deviation": MAX_NORM_DEV},
               "models": {}}
    saved, passed = {}, True
    for name in MODELS:
        model = TextEmbedding(name, cache_dir=args.cache, threads=4)
        inner = model.model
        model_dir = Path(inner._model_dir)
        tokens = [len(inner.tokenizer.encode(t).ids) for t in texts]
        batched = np.stack(list(model.embed(texts, batch_size=2)))
        single = np.stack([next(iter(model.embed([t], batch_size=1))) for t in texts])
        reference = direct_ort(model_dir, texts)
        canonical = CANONICAL_VECTOR_VALUES[name]
        hello = next(iter(model.embed(["hello world"])))[: canonical.shape[0]]
        vs_ort = compare(batched, reference)
        norm_dev = float(np.abs(np.linalg.norm(batched.astype(np.float64), axis=1) - 1).max())
        row = {
            "class": type(inner).__name__,
            "hub_revision": model_dir.name,
            "graph_output_dtype": str(reference.dtype),
            "fastembed_dtype": str(batched.dtype),
            "dtype_matches_graph": batched.dtype == reference.dtype,
            "served_token_counts": tokens,
            "padding": {k: inner.tokenizer.padding[k] for k in ("length", "direction", "pad_token")},
            "fastembed_vs_direct_ort": vs_ort,
            "batch2_vs_batch1": compare(batched, single),
            "maximum_row_norm_deviation": norm_dev,
            "canonical_vector_max_abs_error": float(np.abs(hello - canonical).max()),
        }
        row["parity_passed"] = bool(
            vs_ort["minimum_true_cosine"] >= MIN_COSINE and vs_ort["maximum_absolute_error"] <= MAX_ABS
            and norm_dev <= MAX_NORM_DEV and row["canonical_vector_max_abs_error"] <= 1e-3
        )
        passed &= row["parity_passed"]
        receipt["models"][name] = row
        saved[name] = batched

    assert receipt["models"]["Qdrant/constella-nano"]["served_token_counts"][5:8] == [511, 512, 512]
    np.savez(args.rows, **{k.replace("/", "__"): v for k, v in saved.items()})
    if args.compare:
        other = np.load(args.compare)
        receipt["vs_other_install"] = {
            name: {**compare(saved[name], other[name.replace("/", "__")]),
                   "other_dtype": str(other[name.replace("/", "__")].dtype)}
            for name in MODELS
        }
    receipt["parity_passed"] = passed
    print(json.dumps(receipt, indent=2, default=bool))
    if not passed:
        raise SystemExit("parity threshold failed; receipt printed above")


if __name__ == "__main__":
    main()
