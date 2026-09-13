import json

import numpy as np
import pytest

from m19src import inherit
from m19src.common import sha_array


def test_effective_row_hash_is_chunk_invariant_and_canonical():
    codes = np.array([[1, -2], [3, 4], [-5, 6]], dtype=np.int8)
    scales = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    effective = codes.astype(np.float32) * scales[:, None]
    assert inherit._effective_rows_hash(codes, scales, chunk_rows=1) == sha_array(effective)
    assert inherit._effective_rows_hash(codes, scales, chunk_rows=2) == sha_array(effective)


@pytest.mark.parametrize("control_key", ["instructions", "m19_registry"])
def test_production_instruction_and_registry_pins_reject_mutation(control_key, monkeypatch):
    assert {"instructions", "m19_registry"}.issubset(inherit.EXPECTED_CONTROL_HASHES)
    expected_by_path = {path: expected for path, expected in inherit.EXPECTED_CONTROL_HASHES.values()}
    target = inherit.EXPECTED_CONTROL_HASHES[control_key][0]
    monkeypatch.setattr(
        inherit, "sha_file",
        lambda path: "mutated" if path == target else expected_by_path[path],
    )
    with pytest.raises(SystemExit, match=f"{control_key} hash"):
        inherit.build_lock()


def test_control_hash_mismatch_stops_before_data(monkeypatch):
    monkeypatch.setattr(inherit, "EXPECTED_CONTROL_HASHES", {"registry": (inherit.REGISTRY_PATH, "fixed")})
    monkeypatch.setattr(inherit, "sha_file", lambda path: "changed")
    with pytest.raises(SystemExit, match="registry hash"):
        inherit.build_lock()


def test_inherited_data_hash_mismatch_stops(monkeypatch, tmp_path):
    fixture = tmp_path / "index.bin"
    fixture.write_bytes(b"changed")
    monkeypatch.setattr(inherit, "EXPECTED_CONTROL_HASHES", {})
    monkeypatch.setattr(inherit, "INHERITED_DATA", {"bm25_data": (fixture, "fixed")})
    monkeypatch.setattr(inherit, "sha_file", lambda path: "changed")
    with pytest.raises(SystemExit, match="bm25_data hash"):
        inherit.build_lock()


def test_publish_never_replaces_different_existing_lock(monkeypatch, tmp_path):
    lock_path = tmp_path / "inheritance-lock.json"
    lock_path.write_text(json.dumps({"identity_sha256": "old"}))
    monkeypatch.setattr(inherit, "build_lock", lambda: {"identity_sha256": "new"})
    monkeypatch.setattr(inherit, "load_json", lambda path: json.loads(path.read_text()))
    with pytest.raises(SystemExit, match="replacement forbidden"):
        inherit.publish(lock_path)
    assert json.loads(lock_path.read_text()) == {"identity_sha256": "old"}


def test_publish_existing_identical_lock_is_idempotent(monkeypatch, tmp_path):
    expected = {"identity_sha256": "same"}
    lock_path = tmp_path / "inheritance-lock.json"
    lock_path.write_text(json.dumps(expected))
    monkeypatch.setattr(inherit, "build_lock", lambda: expected)
    monkeypatch.setattr(inherit, "load_json", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(inherit, "create_json", lambda *args: pytest.fail("must not rewrite"))
    assert inherit.publish(lock_path) == expected


def test_publish_race_never_replaces_concurrent_lock(monkeypatch, tmp_path):
    current = {"identity_sha256": "current"}
    concurrent = {"identity_sha256": "concurrent"}
    lock_path = tmp_path / "inheritance-lock.json"
    monkeypatch.setattr(inherit, "build_lock", lambda: current)
    monkeypatch.setattr(inherit, "load_json", lambda path: json.loads(path.read_text()))

    def concurrent_create(path, obj):
        path.write_text(json.dumps(concurrent))
        raise FileExistsError(path)

    monkeypatch.setattr(inherit, "create_json", concurrent_create)
    with pytest.raises(SystemExit, match="concurrent lock differs"):
        inherit.publish(lock_path)
    assert json.loads(lock_path.read_text()) == concurrent


def _synthetic_identity_inputs():
    doc_ids = ["a", "b"]
    corpus_hash = "corpus"
    table_identity = {
        "table": {"effective_rows_sha256": "effective"},
        "pooling": {
            "preproc": {"prefix": "", "add_special_tokens": True, "max_length": 512},
            "fallback_token_id": 101,
            "weights_folded": True,
        },
    }
    source = {"git": {"commit": "commit"}, "github": {"github_cutoff_utc": "cutoff"}, "sha256": "source"}
    corpus = {"corpus_sha256": corpus_hash, "indexable_documents": 2, "artifacts": 2}
    index = {
        "indexed_text_sha256": "texts", "doc_ids_sha256": inherit.sha_texts(doc_ids),
        "stella": {"combined": {"sha256": "vectors"}}, "vector_shape": [2, 2],
        "vector_dtype": "float16", "sha256": "index", "bm25": {}, "fusion": {},
    }
    system = {"selected_query_encoder": {
        "model_sha256": "model", "tokenizer_sha256": "tokenizer", "config_sha256": "config",
    }}
    data = {
        "index_corpus_jsonl": {"sha256": corpus_hash}, "document_vectors": {"sha256": "vectors"},
        "zero_v1_model": {"sha256": "model"}, "zero_v1_tokenizer": {"sha256": "tokenizer"},
        "zero_v1_config": {"sha256": "config"},
    }
    bm25 = {"doc_ids": list(doc_ids)}
    recipe = {"inheritance": {
        "zero_v1_effective_int8_rows_sha256": "effective",
        "pooling": {"prefix": "", "add_special_tokens": True, "max_length": 512,
                    "fallback_token_id": 101, "weights_folded": True},
    }}
    expected = {
        "qdrant_commit": "commit", "github_cutoff_utc": "cutoff", "corpus_documents": 2,
        "corpus_artifacts": 2, "document_vector_shape": [2, 2],
        "document_vector_dtype": "float16", "effective_rows_sha256": "effective",
    }
    return source, corpus, index, system, data, table_identity, doc_ids, bm25, recipe, expected


def test_semantic_document_order_mismatch_is_refused(monkeypatch):
    args = list(_synthetic_identity_inputs())
    expected = args.pop()
    monkeypatch.setattr(inherit, "EXPECTED_IDENTITIES", expected)
    args[7]["doc_ids"] = ["b", "a"]
    with pytest.raises(SystemExit, match="BM25 document order"):
        inherit._identities(*args)


def test_semantic_indexed_corpus_mismatch_is_refused(monkeypatch):
    args = list(_synthetic_identity_inputs())
    expected = args.pop()
    monkeypatch.setattr(inherit, "EXPECTED_IDENTITIES", expected)
    args[4]["index_corpus_jsonl"]["sha256"] = "different"
    with pytest.raises(SystemExit, match="indexed corpus copy"):
        inherit._identities(*args)
