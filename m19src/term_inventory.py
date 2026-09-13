"""Prospectively inventory and freeze M19's fragmented Qdrant term roster.

This stage reads corpus text and tokenizer behavior only. It never loads document/query vectors,
runs retrieval, reads M18 protocol data, or authors M19 evaluation queries.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tokenizers import AddedToken, Tokenizer

from m19src import inherit
from m19src.common import (M18_WORK, M19, REGISTRY_PATH, RELEASE_BUNDLE, RESULTS, admit_read,
                           create_json, load_json, sha_file, sha_json)

# Fixed before the support scan. Order is demand/stable-meaning priority, never retrieval quality.
# M18's spent bare-term observations may motivate a term but supply no M19 quality evidence.
TERM_CATALOG = [
    {"term": "k8s", "meaning": "Kubernetes deployment shorthand",
     "demand": "owner-requested Qdrant project-memory term in M18",
     "intent_axes": ["deployment", "persistence", "networking"]},
    {"term": "s3", "meaning": "Amazon-S3-compatible object storage",
     "demand": "recorded Qdrant snapshot/project-memory use case",
     "intent_axes": ["snapshot storage", "credentials", "compatibility"]},
    {"term": "hnsw", "meaning": "hierarchical navigable small-world vector index",
     "demand": "core Qdrant index configuration and operations",
     "intent_axes": ["construction", "search tuning", "persistence"]},
    {"term": "grpc", "meaning": "gRPC transport/API",
     "demand": "Qdrant client and server interface",
     "intent_axes": ["connectivity", "client behavior", "server configuration"]},
    {"term": "rocksdb", "meaning": "RocksDB storage engine",
     "demand": "Qdrant storage implementation and migration history",
     "intent_axes": ["migration", "failure recovery", "performance"]},
    {"term": "mmap", "meaning": "memory-mapped storage access",
     "demand": "Qdrant storage configuration and diagnostics",
     "intent_axes": ["configuration", "resource use", "failure diagnosis"]},
    {"term": "arm64", "meaning": "64-bit Arm target architecture",
     "demand": "Qdrant build and deployment platform",
     "intent_axes": ["build", "deployment", "compatibility"]},
    {"term": "tls", "meaning": "transport layer security",
     "demand": "Qdrant network security configuration",
     "intent_axes": ["certificates", "configuration", "connection failure"]},
    {"term": "cuda", "meaning": "NVIDIA CUDA compute platform",
     "demand": "Qdrant GPU build and acceleration work",
     "intent_axes": ["build", "device support", "runtime failure"]},
    {"term": "simd", "meaning": "single-instruction multiple-data CPU execution",
     "demand": "Qdrant architecture and performance work",
     "intent_axes": ["build features", "architecture support", "performance"]},
    {"term": "turboquant", "meaning": "Qdrant TurboQuant quantization implementation",
     "demand": "named Qdrant quantization component",
     "intent_axes": ["configuration", "accuracy", "platform support"]},
    {"term": "gridstore", "meaning": "Qdrant Gridstore storage implementation",
     "demand": "named Qdrant storage component",
     "intent_axes": ["configuration", "migration", "performance"]},
    {"term": "qdrant", "meaning": "the Qdrant vector-search project",
     "demand": "the fixed project-memory domain",
     "intent_axes": ["installation", "configuration", "client compatibility"]},
    {"term": "quantization", "meaning": "reduced-precision vector representation",
     "demand": "Qdrant storage and search feature",
     "intent_axes": ["configuration", "accuracy", "memory use"]},
    {"term": "vectorstore", "meaning": "vector-store integration abstraction",
     "demand": "Qdrant ecosystem integration term",
     "intent_axes": ["integration", "configuration", "compatibility"]},
]


def _term_pattern(term):
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)


def original_pieces(tokenizer, term):
    encoded = tokenizer.encode(term, add_special_tokens=False)
    return {"tokens": list(encoded.tokens), "ids": list(encoded.ids)}


def _added_token_audit(tokenizer_path, terms):
    tokenizer = Tokenizer.from_file(str(admit_read(tokenizer_path)))
    base_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    added = tokenizer.add_tokens([
        AddedToken(term, single_word=True, normalized=True) for term in terms
    ])
    if added != len(terms):
        raise SystemExit(f"M19 ROSTER REFUSED: requested {len(terms)} exact rows but added {added}")
    rows = {}
    for offset, term in enumerate(terms):
        encoded = tokenizer.encode(term, add_special_tokens=False)
        expected_id = base_vocab + offset
        if encoded.ids != [expected_id] or encoded.tokens != [term]:
            raise SystemExit(f"M19 ROSTER REFUSED: AddedToken behavior failed for {term!r}")
        rows[term] = expected_id
    return {
        "base_vocab": base_vocab,
        "added": added,
        "final_vocab": tokenizer.get_vocab_size(with_added_tokens=True),
        "term_ids": rows,
        "policy": {"single_word": True, "normalized": True, "special": False,
                   "lstrip": False, "rstrip": False},
    }


def inventory(corpus_path, tokenizer_path, catalog=TERM_CATALOG):
    tokenizer = Tokenizer.from_file(str(admit_read(tokenizer_path)))
    patterns = {entry["term"]: _term_pattern(entry["term"]) for entry in catalog}
    support = {entry["term"]: set() for entry in catalog}
    title_support = {entry["term"]: set() for entry in catalog}
    occurrences = Counter()
    by_kind = {entry["term"]: Counter() for entry in catalog}
    examples = {entry["term"]: set() for entry in catalog}
    documents = 0
    artifacts = set()
    with open(admit_read(corpus_path), encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "issue_event" or not row.get("text"):
                continue
            documents += 1
            artifact = str(row["artifact_id"])
            artifacts.add(artifact)
            title = str(row.get("title") or "")
            searchable = title + "\n" + str(row["text"])
            for entry in catalog:
                term = entry["term"]
                matches = patterns[term].findall(searchable)
                if not matches:
                    continue
                support[term].add(artifact)
                occurrences[term] += len(matches)
                by_kind[term][str(row["kind"])] += 1
                if patterns[term].search(title):
                    title_support[term].add(artifact)
                if len(examples[term]) < 8:
                    examples[term].add(artifact)

    rows = []
    for priority, entry in enumerate(catalog):
        pieces = original_pieces(tokenizer, entry["term"])
        rows.append({
            **entry,
            "priority": priority,
            "fragment_count": len(pieces["ids"]),
            "original_tokens": pieces["tokens"],
            "original_ids": pieces["ids"],
            "artifact_support": len(support[entry["term"]]),
            "title_artifact_support": len(title_support[entry["term"]]),
            "occurrences": occurrences[entry["term"]],
            "support_by_kind": dict(sorted(by_kind[entry["term"]].items())),
            "example_artifact_ids": sorted(examples[entry["term"]]),
        })
    return {"indexable_documents_scanned": documents, "artifacts_scanned": len(artifacts),
            "catalog": rows}


def select_roster(report, minimum=8, maximum=12, minimum_support=5):
    selected = []
    rejected = []
    for row in report["catalog"]:
        reasons = []
        if row["fragment_count"] < 2:
            reasons.append("released_tokenizer_not_fragmented")
        if row["artifact_support"] < minimum_support:
            reasons.append("artifact_support_below_minimum")
        if len(row["intent_axes"]) < 3:
            reasons.append("fewer_than_three_credible_intent_axes")
        if reasons:
            rejected.append({"term": row["term"], "reasons": reasons})
        elif len(selected) < maximum:
            selected.append(row)
        else:
            rejected.append({"term": row["term"], "reasons": ["roster_maximum_reached"]})
    if len(selected) < minimum:
        raise SystemExit(f"M19 STOP: only {len(selected)} demanded fragmented terms survived")
    return selected, rejected


def build_outputs():
    inheritance = inherit.verify()
    registry = load_json(REGISTRY_PATH)
    corpus_path = M18_WORK / "derived/corpus.jsonl"
    tokenizer_path = RELEASE_BUNDLE / "tokenizer.json"
    report = inventory(corpus_path, tokenizer_path)
    policy = registry["term_roster"]
    selected, rejected = select_roster(
        report, minimum=policy["minimum"], maximum=policy["maximum"],
        minimum_support=policy["minimum_artifact_support"],
    )
    terms = [row["term"] for row in selected]
    added = _added_token_audit(tokenizer_path, terms)
    inventory_result = {
        "_schema": "m19-term-inventory-v1",
        "quality_access": False,
        "query_authoring": False,
        "corpus_sha256": sha_file(corpus_path),
        "tokenizer_sha256": sha_file(tokenizer_path),
        "inheritance_identity_sha256": inheritance["identity_sha256"],
        **report,
        "selected_terms": terms,
        "rejected": rejected,
    }
    body = {
        "_schema": "m19-term-roster-lock-v1",
        "state": "locked",
        "selection_policy": {
            "catalog_frozen_before_support_scan": True,
            "order": "fixed demand/stable-meaning priority",
            "minimum_terms": policy["minimum"],
            "maximum_terms": policy["maximum"],
            "minimum_distinct_artifacts": policy["minimum_artifact_support"],
            "minimum_fragment_pieces": policy["pieces_minimum"],
            "minimum_intent_axes": policy["minimum_distinct_short_intents"],
        },
        "terms": selected,
        "added_token_audit": added,
        "bindings": {
            "registry_sha256": sha_file(REGISTRY_PATH),
            "inheritance_lock_sha256": sha_file(M19 / "inheritance-lock.json"),
            "inventory_implementation_sha256": sha_file(Path(__file__)),
            "corpus_sha256": inventory_result["corpus_sha256"],
            "tokenizer_sha256": inventory_result["tokenizer_sha256"],
            "inventory_result_identity_sha256": sha_json(inventory_result),
        },
    }
    roster = {**body, "identity_sha256": sha_json(body)}
    return inventory_result, roster


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    inventory_result, roster = build_outputs()
    result_path = RESULTS / "m19_term_inventory.json"
    roster_path = M19 / "term-roster-lock.json"
    if args.verify:
        if load_json(result_path) != inventory_result or load_json(roster_path) != roster:
            raise SystemExit("M19 ROSTER REFUSED: current inventory differs from frozen outputs")
    else:
        if result_path.exists() or roster_path.exists():
            raise SystemExit("M19 ROSTER REFUSED: immutable inventory outputs already exist; use --verify")
        create_json(result_path, inventory_result)
        create_json(roster_path, roster)
    print(json.dumps({"terms": [row["term"] for row in roster["terms"]],
                      "identity_sha256": roster["identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
