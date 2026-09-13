"""Verify and publish M19's immutable inheritance lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m19src.common import (M18_WORK, M19, RELEASE_BUNDLE, REPO, RESULTS, load_json, sha_file,
                           sha_json, write_json)

EXPECTED_MANIFEST_HASHES = {
    "m18_registry": (REPO / "m18/registry.json", "e3d16ce06efbd40458006d67724af89a35a6658def0767d17a58c65db61b0a31"),
    "m18_execution_lock": (REPO / "m18/execution-lock.json", "3251a72c573d171cbb6544eb83ea10052beb29a744e883498e48122de52754da"),
    "source_manifest": (RESULTS / "m18_source_manifest.json", "435b8c3a4ce301f6c662a2a7c7147ffed1e073f1dda9e49ede1b3beeccf20f25"),
    "corpus_manifest": (RESULTS / "m18_corpus_manifest.json", "fc21a2bb54db3a34ad4a116d639a1bdbc04de5ffd74510c38f2f7831db22c804"),
    "index_manifest": (RESULTS / "m18_index_manifest.json", "037a9e3f5ccf7be1264ba50f427d6f062bc6e2cdfbaeeaa2c538ef7c23a11025"),
    "system_manifest": (RESULTS / "m18_system_manifest.json", "642ded576530b9ad32caaa4c1710803a55155cb2e2c3103ecb9e385a2820443f"),
}

INHERITED_DATA = {
    "corpus_jsonl": M18_WORK / "derived/corpus.jsonl",
    "index_corpus_jsonl": M18_WORK / "derived/index/corpus.jsonl",
    "doc_ids": M18_WORK / "derived/index/doc_ids.json",
    "document_vectors": M18_WORK / "derived/index/documents/vectors.f16.npy",
    "bm25_data": M18_WORK / "derived/index/bm25/data.csc.index.npy",
    "bm25_indices": M18_WORK / "derived/index/bm25/indices.csc.index.npy",
    "bm25_indptr": M18_WORK / "derived/index/bm25/indptr.csc.index.npy",
    "bm25_metadata": M18_WORK / "derived/index/bm25/m18.json",
    "bm25_params": M18_WORK / "derived/index/bm25/params.index.json",
    "bm25_vocab": M18_WORK / "derived/index/bm25/vocab.index.json",
    "zero_v1_model": RELEASE_BUNDLE / "model.npz",
    "zero_v1_tokenizer": RELEASE_BUNDLE / "tokenizer.json",
    "zero_v1_config": RELEASE_BUNDLE / "config.json",
}


def build_lock():
    files = {}
    for key, (path, expected) in EXPECTED_MANIFEST_HASHES.items():
        actual = sha_file(path)
        if actual != expected:
            raise SystemExit(f"M19 INHERITANCE REFUSED: {key} hash {actual} != {expected}")
        files[key] = {"path": str(path.relative_to(REPO)), "sha256": actual}

    data = {}
    for key, path in INHERITED_DATA.items():
        if not path.is_file():
            raise SystemExit(f"M19 INHERITANCE REFUSED: missing {key}: {path}")
        data[key] = {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}

    source = load_json(RESULTS / "m18_source_manifest.json")
    corpus = load_json(RESULTS / "m18_corpus_manifest.json")
    index = load_json(RESULTS / "m18_index_manifest.json")
    system = load_json(RESULTS / "m18_system_manifest.json")
    identities = {
        "qdrant_commit": source["git"]["commit"],
        "github_cutoff_utc": source["github"]["github_cutoff_utc"],
        "source_identity_sha256": source["sha256"],
        "corpus_identity_sha256": corpus["corpus_sha256"],
        "corpus_documents": corpus["indexable_documents"],
        "corpus_artifacts": corpus["artifacts"],
        "ordered_doc_ids_sha256": index["doc_ids_sha256"],
        "indexed_text_sha256": index["indexed_text_sha256"],
        "document_vectors_sha256": index["stella"]["combined"]["sha256"],
        "document_vector_shape": index["vector_shape"],
        "document_vector_dtype": index["vector_dtype"],
        "index_identity_sha256": index["sha256"],
        "bm25_recipe": index["bm25"],
        "fusion_recipe": index["fusion"],
        "zero_v1": system["selected_query_encoder"],
    }
    expected_identity = {
        "qdrant_commit": "5e32ea89cb5ea9a6827a68b27291d2e2251e1b1c",
        "github_cutoff_utc": "2026-09-13T00:36:26Z",
        "corpus_documents": 79269,
        "corpus_artifacts": 11574,
        "document_vector_shape": [79269, 1024],
        "document_vector_dtype": "float16",
    }
    for key, expected in expected_identity.items():
        if identities[key] != expected:
            raise SystemExit(f"M19 INHERITANCE REFUSED: {key} {identities[key]!r} != {expected!r}")
    for key in ("model_sha256", "tokenizer_sha256", "config_sha256"):
        data_key = "zero_v1_" + key.removesuffix("_sha256")
        if data[data_key]["sha256"] != identities["zero_v1"][key]:
            raise SystemExit(f"M19 INHERITANCE REFUSED: released {key} does not match M18")
    if data["corpus_jsonl"]["sha256"] != identities["corpus_identity_sha256"]:
        raise SystemExit("M19 INHERITANCE REFUSED: inherited corpus bytes do not match M18")
    if data["document_vectors"]["sha256"] != identities["document_vectors_sha256"]:
        raise SystemExit("M19 INHERITANCE REFUSED: inherited document vectors do not match M18")

    body = {
        "_schema": "m19-inheritance-lock-v1",
        "state": "locked",
        "source_commit": "bfa7257fe369a41ce048e150f861703a662a08f0",
        "manifest_files": files,
        "inherited_data": data,
        "identities": identities,
        "protected_inputs": {
            "m18_confirmation_queries_or_qrels": "forbidden",
            "historical_spent_or_reserved_evaluation": "forbidden",
            "m18_closed_state_mutation": "forbidden",
        },
    }
    return {**body, "identity_sha256": sha_json(body)}


def verify(lock_path=M19 / "inheritance-lock.json"):
    recorded = load_json(lock_path)
    current = build_lock()
    if recorded != current:
        raise SystemExit("M19 INHERITANCE REFUSED: current inputs differ from inheritance lock")
    return recorded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out", default=str(M19 / "inheritance-lock.json"))
    args = parser.parse_args(argv)
    if args.verify:
        lock = verify(Path(args.out))
    else:
        lock = build_lock()
        write_json(args.out, lock)
    print(json.dumps({"state": lock["state"], "identity_sha256": lock["identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
