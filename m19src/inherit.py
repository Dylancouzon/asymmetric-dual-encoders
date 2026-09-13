"""Verify and exclusively publish M19's immutable inheritance/recipe lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m19src.common import (M18_WORK, M19, REGISTRY_PATH, RELEASE_BUNDLE, REPO, RESULTS,
                           admit_read, load_json, sha_array, sha_file, sha_json, sha_texts,
                           write_json)

EXPECTED_CONTROL_HASHES = {
    "claude": (REPO / "CLAUDE.md", "ebab0f2c864d514fe2b41f7554c4d5fada3adf2a89d86428ff137eb8be1539ce"),
    "instructions": (REPO / "instructions-m19.md", "e7a1da986c042075e10b31f9199efb8318a3bbe8f0219b47d1cb6b0fde377fd5"),
    "m19_registry": (REGISTRY_PATH, "3796b18c16215405137e9b6e5e2fa07df72e95b9d0f90362a4e3dcbe346df0fe"),
    "m18_registry": (REPO / "m18/registry.json", "e3d16ce06efbd40458006d67724af89a35a6658def0767d17a58c65db61b0a31"),
    "m18_execution_lock": (REPO / "m18/execution-lock.json", "3251a72c573d171cbb6544eb83ea10052beb29a744e883498e48122de52754da"),
    "source_manifest": (RESULTS / "m18_source_manifest.json", "435b8c3a4ce301f6c662a2a7c7147ffed1e073f1dda9e49ede1b3beeccf20f25"),
    "corpus_manifest": (RESULTS / "m18_corpus_manifest.json", "fc21a2bb54db3a34ad4a116d639a1bdbc04de5ffd74510c38f2f7831db22c804"),
    "index_manifest": (RESULTS / "m18_index_manifest.json", "037a9e3f5ccf7be1264ba50f427d6f062bc6e2cdfbaeeaa2c538ef7c23a11025"),
    "system_manifest": (RESULTS / "m18_system_manifest.json", "642ded576530b9ad32caaa4c1710803a55155cb2e2c3103ecb9e385a2820443f"),
}

SELF_BOUND_FILES = {
    "m19_common_implementation": REPO / "m19src/common.py",
    "m19_inheritance_implementation": REPO / "m19src/inherit.py",
}

INHERITED_DATA = {
    "corpus_jsonl": (M18_WORK / "derived/corpus.jsonl", "0cede39eb47b5680ef0f560917bd93ec78d02ac0e607ae0c4209d98f4430f394"),
    "index_corpus_jsonl": (M18_WORK / "derived/index/corpus.jsonl", "0cede39eb47b5680ef0f560917bd93ec78d02ac0e607ae0c4209d98f4430f394"),
    "doc_ids": (M18_WORK / "derived/index/doc_ids.json", "85fac5a24e5714281c0cc4046c4b2971b8a28d6c553ea8896b21c3cf80cbb939"),
    "document_vectors": (M18_WORK / "derived/index/documents/vectors.f16.npy", "3422ddccae17df01bf85a375c3cb4815a0e86e8d3447f8be0cfe46583ef7a10e"),
    "bm25_data": (M18_WORK / "derived/index/bm25/data.csc.index.npy", "2e1e5550edd1fbb7177a6ce94c4a1d4444f80168a9e515f9ce0b321778fb2398"),
    "bm25_indices": (M18_WORK / "derived/index/bm25/indices.csc.index.npy", "2d99baa326b82c0702b9c72be8f5b64a576b727fc1852da65cb2d3540e465b2c"),
    "bm25_indptr": (M18_WORK / "derived/index/bm25/indptr.csc.index.npy", "e0c8e75804198f64f1c41db7ea31a9693ccfa73c3b1c341e1bac18e3041244e5"),
    "bm25_metadata": (M18_WORK / "derived/index/bm25/m18.json", "f7ef649d0402e00eee9ed5f2742755215b8f605d822b8ed214293f0c8388ccb0"),
    "bm25_params": (M18_WORK / "derived/index/bm25/params.index.json", "3075fb8ccf7bad6f4f34094e2703d97df5221612193be6d46f72f378f9d2ad9f"),
    "bm25_vocab": (M18_WORK / "derived/index/bm25/vocab.index.json", "6b0c6092bef6ed07067d138d007632a2795a0b91d163899c92bcd99044ba8f21"),
    "zero_v1_model": (RELEASE_BUNDLE / "model.npz", "a7007b1a6af120b976f093fd69ddcb5001996ec0b84b5864b4fd25d7af878abf"),
    "zero_v1_tokenizer": (RELEASE_BUNDLE / "tokenizer.json", "997eaeff157a7d4db899b8f2278ad3db6e323e609ae956b9c5035550f581eeee"),
    "zero_v1_config": (RELEASE_BUNDLE / "config.json", "fd784397925e9ec32b6f1412af85b46faea79bb618b17197dc6d3a0068669636"),
}

EXPECTED_IDENTITIES = {
    "qdrant_commit": "5e32ea89cb5ea9a6827a68b27291d2e2251e1b1c",
    "github_cutoff_utc": "2026-09-13T00:36:26Z",
    "corpus_documents": 79269,
    "corpus_artifacts": 11574,
    "document_vector_shape": [79269, 1024],
    "document_vector_dtype": "float16",
    "effective_rows_sha256": "f0a62a358104b04d4937829898a2beb00cc57737eb4a7ee5bff22c47edf6bb1a",
}


def _effective_rows_hash(codes, scales, chunk_rows=2048):
    """Hash dequantized float32 rows without allocating the full 125 MB table."""
    if codes.ndim != 2 or scales.shape != (codes.shape[0],):
        raise SystemExit("M19 INHERITANCE REFUSED: malformed int8 codes/scales")
    digest = hashlib.sha256()
    for lo in range(0, codes.shape[0], chunk_rows):
        hi = min(lo + chunk_rows, codes.shape[0])
        block = codes[lo:hi].astype(np.float32) * scales[lo:hi, None]
        digest.update(np.ascontiguousarray(block).tobytes())
    digest.update(b"float32")
    digest.update(str(codes.shape).encode())
    return digest.hexdigest()


def released_table_identity(model_path, config_path):
    with np.load(admit_read(model_path)) as archive:
        codes = np.asarray(archive["rows_int8"], dtype=np.int8)
        scales = np.asarray(archive["int8_scale"], dtype=np.float32)
        table = {
            "codes_sha256": sha_array(codes),
            "scales_sha256": sha_array(scales),
            "effective_rows_sha256": _effective_rows_hash(codes, scales),
            "shape": list(codes.shape),
            "codes_dtype": str(codes.dtype),
            "scales_dtype": str(scales.dtype),
            "resident_bytes": int(codes.nbytes + scales.nbytes),
            "hash_convention": "C float32 dequantized row bytes, then dtype and shape strings",
        }
    config = load_json(config_path)
    pooling = {
        "preproc": config["preproc"],
        "fallback_token_id": config["fallback_token_id"],
        "weights_folded": config["weights_folded"],
        "learned_weights": config["learned_weights"],
    }
    return {"table": table, "pooling": pooling, "pooling_sha256": sha_json(pooling)}


def _identities(source, corpus, index, system, data, table_identity, doc_ids, bm25, recipe):
    identities = {
        "qdrant_commit": source["git"]["commit"],
        "github_cutoff_utc": source["github"]["github_cutoff_utc"],
        "source_identity_sha256": source["sha256"],
        "corpus_identity_sha256": corpus["corpus_sha256"],
        "corpus_documents": corpus["indexable_documents"],
        "corpus_artifacts": corpus["artifacts"],
        "ordered_doc_ids_sha256": sha_texts(doc_ids),
        "indexed_text_sha256": index["indexed_text_sha256"],
        "document_vectors_sha256": index["stella"]["combined"]["sha256"],
        "document_vector_shape": index["vector_shape"],
        "document_vector_dtype": index["vector_dtype"],
        "index_identity_sha256": index["sha256"],
        "bm25_recipe": index["bm25"],
        "fusion_recipe": index["fusion"],
        "zero_v1": system["selected_query_encoder"],
        "released_effective_table": table_identity,
        "effective_rows_sha256": table_identity["table"]["effective_rows_sha256"],
    }
    for key, expected in EXPECTED_IDENTITIES.items():
        if identities[key] != expected:
            raise SystemExit(f"M19 INHERITANCE REFUSED: {key} {identities[key]!r} != {expected!r}")
    registered = recipe["inheritance"]
    if registered["zero_v1_effective_int8_rows_sha256"] != identities["effective_rows_sha256"]:
        raise SystemExit("M19 INHERITANCE REFUSED: effective rows differ from registered identity")
    pooling = registered["pooling"]
    actual_pooling = table_identity["pooling"]
    for key in ("prefix", "add_special_tokens", "max_length"):
        if pooling[key] != actual_pooling["preproc"][key]:
            raise SystemExit(f"M19 INHERITANCE REFUSED: pooling {key} differs from registry")
    for key in ("fallback_token_id", "weights_folded"):
        if pooling[key] != actual_pooling[key]:
            raise SystemExit(f"M19 INHERITANCE REFUSED: pooling {key} differs from registry")
    if identities["ordered_doc_ids_sha256"] != index["doc_ids_sha256"]:
        raise SystemExit("M19 INHERITANCE REFUSED: ordered document-ID identity differs from M18")
    if bm25.get("doc_ids") != doc_ids:
        raise SystemExit("M19 INHERITANCE REFUSED: BM25 document order differs from index")
    if data["index_corpus_jsonl"]["sha256"] != corpus["corpus_sha256"]:
        raise SystemExit("M19 INHERITANCE REFUSED: indexed corpus copy differs from M18 corpus")
    if data["document_vectors"]["sha256"] != identities["document_vectors_sha256"]:
        raise SystemExit("M19 INHERITANCE REFUSED: document vectors differ from M18 identity")
    for key in ("model_sha256", "tokenizer_sha256", "config_sha256"):
        data_key = "zero_v1_" + key.removesuffix("_sha256")
        if data[data_key]["sha256"] != identities["zero_v1"][key]:
            raise SystemExit(f"M19 INHERITANCE REFUSED: released {key} differs from M18")
    return identities


def build_lock():
    controls = {}
    for key, (path, expected) in EXPECTED_CONTROL_HASHES.items():
        actual = sha_file(path)
        if actual != expected:
            raise SystemExit(f"M19 INHERITANCE REFUSED: {key} hash {actual} != {expected}")
        controls[key] = {"path": str(path), "sha256": actual}
    for key, path in SELF_BOUND_FILES.items():
        controls[key] = {"path": str(path), "sha256": sha_file(path)}

    data = {}
    for key, (path, expected) in INHERITED_DATA.items():
        if not path.is_file():
            raise SystemExit(f"M19 INHERITANCE REFUSED: missing {key}: {path}")
        actual = sha_file(path)
        if actual != expected:
            raise SystemExit(f"M19 INHERITANCE REFUSED: {key} hash {actual} != {expected}")
        data[key] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}

    source = load_json(RESULTS / "m18_source_manifest.json")
    corpus = load_json(RESULTS / "m18_corpus_manifest.json")
    index = load_json(RESULTS / "m18_index_manifest.json")
    system = load_json(RESULTS / "m18_system_manifest.json")
    doc_ids = load_json(M18_WORK / "derived/index/doc_ids.json")
    bm25 = load_json(M18_WORK / "derived/index/bm25/m18.json")
    recipe = load_json(REGISTRY_PATH)
    table_identity = released_table_identity(RELEASE_BUNDLE / "model.npz",
                                             RELEASE_BUNDLE / "config.json")
    identities = _identities(source, corpus, index, system, data, table_identity, doc_ids, bm25,
                             recipe)

    body = {
        "_schema": "m19-inheritance-lock-v2",
        "state": "locked",
        "source_commit": "bfa7257fe369a41ce048e150f861703a662a08f0",
        "control_files": controls,
        "inherited_data": data,
        "identities": identities,
        "protected_inputs": {
            "m18_confirmation_queries_or_qrels": "forbidden",
            "historical_spent_or_reserved_evaluation": "forbidden",
            "m18_closed_state_mutation": "forbidden",
            "fresh_m19_confirmation_before_claim": "sealed",
        },
    }
    return {**body, "identity_sha256": sha_json(body)}


def verify(lock_path=M19 / "inheritance-lock.json"):
    recorded = load_json(lock_path)
    current = build_lock()
    if recorded != current:
        raise SystemExit("M19 INHERITANCE REFUSED: current inputs differ from inheritance lock")
    return recorded


def publish(lock_path=M19 / "inheritance-lock.json"):
    current = build_lock()
    path = Path(lock_path)
    if path.exists():
        recorded = load_json(path)
        if recorded != current:
            raise SystemExit("M19 INHERITANCE REFUSED: existing lock differs; replacement forbidden")
        return recorded
    write_json(path, current)
    return current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out", default=str(M19 / "inheritance-lock.json"))
    args = parser.parse_args(argv)
    lock = verify(Path(args.out)) if args.verify else publish(Path(args.out))
    print(json.dumps({"state": lock["state"], "identity_sha256": lock["identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
