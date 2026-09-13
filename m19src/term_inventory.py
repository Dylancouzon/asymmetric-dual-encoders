"""Consume, qualify and freeze M19's fragmented Qdrant term roster.

The scanner uses the real joint AddedToken tokenizer, hashes the exact corpus bytes while parsing
them, and binds those consumed hashes to the verified inheritance lock. It never loads vectors,
runs retrieval, reads M18 protocol data, or authors evaluation queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from tokenizers import AddedToken, Tokenizer

from m19src import inherit
from m19src.common import (M18_WORK, M19, REGISTRY_PATH, RELEASE_BUNDLE, RESULTS, WORK, admit_read,
                           create_json, load_json, sha_bytes, sha_file, sha_json, write_json)

# Catalog and first observed support were committed together in 1736d93. The runtime order was
# fixed before that scan, but there is no independent pre-scan git receipt; the v2 lock discloses
# this limitation instead of asserting stronger prospectivity. No retrieval quality informed it.
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

# Added after inspecting only the ignored source-qualification packet. Every witness must be a
# serving-consistent, non-equivalent natural occurrence in the catalogued sense. This is not a
# relevance label or an evaluation query. Filled before v2 publication.
QUALIFICATIONS = {
    "k8s": {
        "witness_artifact_ids": ["gh:thread:1640", "gh:thread:2401", "gh:thread:6431",
                                 "gh:thread:1060", "gh:thread:1704"],
        "intent_evidence": {
            "deployment": ["gh:thread:6431", "gh:thread:1704"],
            "persistence": ["gh:thread:1640", "gh:thread:1060"],
            "networking": ["gh:thread:2401", "gh:thread:6431"],
        },
        "notes": "Kubernetes shorthand only; pod lifecycle, persistent storage and peer/health networking.",
    },
    "s3": {
        "witness_artifact_ids": ["gh:thread:10085", "gh:thread:1703", "gh:thread:3430",
                                 "gh:thread:4701", "gh:thread:4705"],
        "intent_evidence": {
            "snapshot storage": ["gh:thread:1703", "gh:thread:3430"],
            "credentials": ["gh:thread:4701"],
            "compatibility": ["gh:thread:4705"],
        },
        "notes": "Object-storage sense only; excludes local identifiers that happen to spell s3.",
    },
    "hnsw": {
        "witness_artifact_ids": ["gh:thread:10", "gh:thread:10010", "gh:thread:10044",
                                 "gh:thread:10046", "gh:thread:1044"],
        "intent_evidence": {
            "construction": ["gh:thread:10", "gh:thread:10046"],
            "search tuning": ["gh:thread:10010", "gh:thread:1044"],
            "persistence": ["gh:thread:10044"],
        },
    },
    "grpc": {
        "witness_artifact_ids": ["gh:thread:10048", "gh:thread:10106", "gh:thread:10234",
                                 "gh:thread:10306", "gh:thread:10522"],
        "intent_evidence": {
            "connectivity": ["gh:thread:10306"],
            "client behavior": ["gh:thread:10048", "gh:thread:10522"],
            "server configuration": ["gh:thread:10234", "gh:thread:10306"],
        },
    },
    "rocksdb": {
        "witness_artifact_ids": ["gh:thread:10561", "gh:thread:1725", "gh:thread:1736",
                                 "gh:thread:2709", "gh:thread:3036"],
        "intent_evidence": {
            "migration": ["gh:thread:10561"],
            "failure recovery": ["gh:thread:1725", "gh:thread:1736"],
            "performance": ["gh:thread:2709", "gh:thread:3036"],
        },
    },
    "mmap": {
        "witness_artifact_ids": ["gh:thread:10180", "gh:thread:1308", "gh:thread:1791",
                                 "gh:thread:1873", "gh:thread:2408"],
        "intent_evidence": {
            "configuration": ["gh:thread:1308", "gh:thread:2408"],
            "resource use": ["gh:thread:1791", "gh:thread:1308"],
            "failure diagnosis": ["gh:thread:10180", "gh:thread:1873"],
        },
    },
    "arm64": {
        "witness_artifact_ids": ["gh:thread:10303", "gh:thread:1529", "gh:thread:2474",
                                 "gh:thread:3056", "gh:thread:3714"],
        "intent_evidence": {
            "build": ["gh:thread:1529", "gh:thread:3714"],
            "deployment": ["gh:thread:2474", "gh:thread:3056"],
            "compatibility": ["gh:thread:10303", "gh:thread:2474"],
        },
    },
    "tls": {
        "witness_artifact_ids": ["gh:thread:1497", "gh:thread:1641", "gh:thread:1808",
                                 "gh:thread:1948", "gh:thread:6273"],
        "intent_evidence": {
            "certificates": ["gh:thread:1808", "gh:thread:1948"],
            "configuration": ["gh:thread:1497", "gh:thread:1641"],
            "connection failure": ["gh:thread:6273"],
        },
        "notes": "Transport-security sense only; excludes thread-local-storage uses.",
    },
    "cuda": {
        "witness_artifact_ids": ["gh:thread:1669", "gh:thread:10210", "gh:thread:10211",
                                 "gh:thread:5172", "gh:thread:5872"],
        "intent_evidence": {
            "build": ["gh:thread:5172"],
            "device support": ["gh:thread:1669", "gh:thread:5872"],
            "runtime failure": ["gh:thread:10210", "gh:thread:10211"],
        },
        "notes": "CUDA itself is stable; Qdrant's GPU path is Vulkan, so coexistence distinctions remain explicit.",
    },
    "simd": {
        "witness_artifact_ids": ["gh:thread:10209", "gh:thread:10291", "gh:thread:10461",
                                 "gh:thread:508", "gh:thread:7238"],
        "intent_evidence": {
            "build features": ["gh:thread:10209", "gh:thread:7238"],
            "architecture support": ["gh:thread:10461", "gh:thread:7238"],
            "performance": ["gh:thread:10291", "gh:thread:508"],
        },
    },
    "turboquant": {
        "witness_artifact_ids": ["gh:thread:10165", "gh:thread:10344", "gh:thread:10437",
                                 "gh:thread:8524", "gh:thread:8544"],
        "intent_evidence": {
            "configuration": ["gh:thread:10344", "gh:thread:8544"],
            "accuracy": ["gh:thread:8524"],
            "platform support": ["gh:thread:10437"],
        },
    },
    "gridstore": {
        "witness_artifact_ids": ["gh:thread:10399", "gh:thread:5918", "gh:thread:6445",
                                 "gh:thread:6609", "gh:thread:6722"],
        "intent_evidence": {
            "configuration": ["gh:thread:6609", "gh:thread:6722"],
            "migration": ["gh:thread:5918", "gh:thread:6609"],
            "performance": ["gh:thread:6445", "gh:thread:6722"],
        },
    },
}


def _load_tokenizer_bytes(path):
    payload = admit_read(path).read_bytes()
    tokenizer = Tokenizer.from_str(payload.decode("utf-8"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return payload, tokenizer


def original_pieces(tokenizer, term):
    encoded = tokenizer.encode(term, add_special_tokens=False)
    return {"tokens": list(encoded.tokens), "ids": list(encoded.ids)}


def _extend_tokenizer(base_payload, terms):
    tokenizer = Tokenizer.from_str(base_payload.decode("utf-8"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    before = tokenizer.get_vocab(with_added_tokens=True)
    base_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    added = tokenizer.add_tokens([
        AddedToken(term, single_word=True, normalized=True) for term in terms
    ])
    if added != len(terms):
        raise SystemExit(f"M19 ROSTER REFUSED: requested {len(terms)} exact rows but added {added}")
    after = tokenizer.get_vocab(with_added_tokens=True)
    if any(after.get(token) != token_id for token, token_id in before.items()):
        raise SystemExit("M19 ROSTER REFUSED: inherited tokenizer IDs changed")
    term_ids = {}
    for offset, term in enumerate(terms):
        encoded = tokenizer.encode(term, add_special_tokens=False)
        expected_id = base_vocab + offset
        if encoded.ids != [expected_id] or encoded.tokens != [term]:
            raise SystemExit(f"M19 ROSTER REFUSED: AddedToken behavior failed for {term!r}")
        term_ids[term] = expected_id
    return tokenizer, {
        "base_vocab": base_vocab,
        "added": added,
        "final_vocab": tokenizer.get_vocab_size(with_added_tokens=True),
        "term_ids": term_ids,
        "inherited_vocab_sha256": sha_json(before),
        "inherited_ids_unchanged": True,
        "policy": {"single_word": True, "normalized": True, "special": False,
                   "lstrip": False, "rstrip": False},
    }


def _witness_digest(row):
    return sha_json({
        "title": " ".join(str(row.get("title") or "").casefold().split()),
        "normalized_text_sha256": row.get("normalized_text_sha256")
            or sha_bytes(str(row.get("text") or "").casefold().encode()),
    })


def _snippet(text, term, width=600):
    match = re.search(re.escape(term), text, flags=re.IGNORECASE)
    if not match:
        return text[:width]
    lo = max(0, match.start() - width // 2)
    return text[lo:lo + width]


def inventory(corpus_path, tokenizer_path, catalog=TERM_CATALOG, packet_limit=40):
    tokenizer_payload, base = _load_tokenizer_bytes(tokenizer_path)
    terms = [entry["term"] for entry in catalog]
    matcher, added_audit = _extend_tokenizer(tokenizer_payload, terms)
    term_by_id = {token_id: term for term, token_id in added_audit["term_ids"].items()}
    raw_occurrences = Counter()
    # One representative per (term, normalized title+body digest), chosen deterministically.
    witnesses = {term: {} for term in terms}
    documents = 0
    artifacts = set()
    corpus_digest = hashlib.sha256()
    with open(admit_read(corpus_path), "rb") as handle:
        for raw_line in handle:
            corpus_digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if (row.get("kind") == "issue_event" or row.get("indexable") is False
                    or not row.get("text")):
                continue
            documents += 1
            artifact = str(row["artifact_id"])
            artifacts.add(artifact)
            title = str(row.get("title") or "")
            text = title + "\n" + str(row["text"])
            counts = Counter(matcher.encode(text, add_special_tokens=False).ids)
            title_ids = set(matcher.encode(title, add_special_tokens=False).ids)
            digest = _witness_digest(row)
            for token_id, count in counts.items():
                term = term_by_id.get(token_id)
                if term is None:
                    continue
                raw_occurrences[term] += count
                record = {
                    "artifact_id": artifact,
                    "doc_id": str(row.get("doc_id") or ""),
                    "kind": str(row["kind"]),
                    "title": title,
                    "source_url": str(row.get("source_url") or ""),
                    "witness_digest": digest,
                    "title_match": token_id in title_ids,
                    "snippet": _snippet(str(row["text"]), term),
                }
                previous = witnesses[term].get(digest)
                key = (record["artifact_id"], record["doc_id"])
                if previous is None or key < (previous["artifact_id"], previous["doc_id"]):
                    witnesses[term][digest] = record

    rows = []
    packets = {}
    for priority, entry in enumerate(catalog):
        term = entry["term"]
        unique = sorted(witnesses[term].values(), key=lambda row: (
            not row["title_match"], row["artifact_id"], row["doc_id"]
        ))
        support_artifacts = sorted({row["artifact_id"] for row in unique})
        title_artifacts = sorted({row["artifact_id"] for row in unique if row["title_match"]})
        by_kind = Counter(row["kind"] for row in unique)
        pieces = original_pieces(base, term)
        rows.append({
            **entry,
            "priority": priority,
            "fragment_count": len(pieces["ids"]),
            "original_tokens": pieces["tokens"],
            "original_ids": pieces["ids"],
            "artifact_support": len(support_artifacts),
            "non_equivalent_witnesses": len(unique),
            "title_artifact_support": len(title_artifacts),
            "raw_serving_match_occurrences": raw_occurrences[term],
            "support_by_kind": dict(sorted(by_kind.items())),
            "example_artifact_ids": support_artifacts[:8],
        })
        packets[term] = unique[:packet_limit]
    return {
        "consumed_corpus_sha256": corpus_digest.hexdigest(),
        "consumed_tokenizer_sha256": sha_bytes(tokenizer_payload),
        "indexable_documents_scanned": documents,
        "artifacts_scanned": len(artifacts),
        "catalog": rows,
        "added_token_audit": added_audit,
    }, packets


def select_roster(report, minimum=8, maximum=12, minimum_support=5):
    selected = []
    rejected = []
    for row in report["catalog"]:
        reasons = []
        if row["fragment_count"] < 2:
            reasons.append("released_tokenizer_not_fragmented")
        if row["artifact_support"] < minimum_support:
            reasons.append("artifact_support_below_minimum")
        axes = row.get("intent_axes") or []
        if len(axes) < 3 or len(set(axes)) != len(axes):
            reasons.append("fewer_than_three_distinct_credible_intent_axes")
        if not str(row.get("meaning") or "").strip():
            reasons.append("missing_stable_meaning")
        if not str(row.get("demand") or "").strip():
            reasons.append("missing_demand_basis")
        if reasons:
            rejected.append({"term": row["term"], "reasons": reasons})
        elif len(selected) < maximum:
            selected.append(row)
        else:
            rejected.append({"term": row["term"], "reasons": ["roster_maximum_reached"]})
    if len(selected) < minimum:
        raise SystemExit(f"M19 STOP: only {len(selected)} demanded fragmented terms survived")
    return selected, rejected


def _qualify(selected, packets, qualifications=QUALIFICATIONS):
    receipts = {}
    for row in selected:
        term = row["term"]
        receipt = qualifications.get(term)
        if not isinstance(receipt, dict):
            raise SystemExit(f"M19 ROSTER REFUSED: {term} lacks a manual source qualification")
        qualified = receipt.get("witness_artifact_ids") or []
        if len(qualified) < 5 or len(set(qualified)) != len(qualified):
            raise SystemExit(f"M19 ROSTER REFUSED: {term} lacks five distinct qualified artifacts")
        available = {item["artifact_id"]: item for item in packets[term]}
        if any(artifact not in available for artifact in qualified):
            raise SystemExit(f"M19 ROSTER REFUSED: {term} qualification is outside source packet")
        digests = {available[artifact]["witness_digest"] for artifact in qualified}
        if len(digests) < 5:
            raise SystemExit(f"M19 ROSTER REFUSED: {term} witnesses are text-equivalent")
        evidence = receipt.get("intent_evidence") or {}
        if set(evidence) != set(row["intent_axes"]):
            raise SystemExit(f"M19 ROSTER REFUSED: {term} intent evidence is incomplete")
        if any(not refs or any(ref not in available for ref in refs) for refs in evidence.values()):
            raise SystemExit(f"M19 ROSTER REFUSED: {term} intent evidence is outside source packet")
        receipts[term] = {
            "judgment": "source-only intended-sense and multi-intent plausibility; not relevance",
            "witness_artifact_ids": qualified,
            "witness_digests": sorted(digests),
            "intent_evidence": evidence,
            "notes": str(receipt.get("notes") or ""),
        }
    return receipts


def _bind_consumed(report, inheritance):
    expected = inheritance["inherited_data"]
    checks = {
        "consumed_corpus_sha256": expected["corpus_jsonl"]["sha256"],
        "consumed_tokenizer_sha256": expected["zero_v1_tokenizer"]["sha256"],
    }
    for key, value in checks.items():
        if report[key] != value:
            raise SystemExit(f"M19 ROSTER REFUSED: {key} differs from inheritance")


def build_outputs(*, require_qualification=True):
    implementation_before = sha_file(Path(__file__))
    inheritance = inherit.verify()
    registry = load_json(REGISTRY_PATH)
    corpus_path = M18_WORK / "derived/corpus.jsonl"
    tokenizer_path = RELEASE_BUNDLE / "tokenizer.json"
    report, packets = inventory(corpus_path, tokenizer_path)
    _bind_consumed(report, inheritance)
    policy = registry["term_roster"]
    selected, rejected = select_roster(
        report, minimum=policy["minimum"], maximum=policy["maximum"],
        minimum_support=policy["minimum_artifact_support"],
    )
    qualifications = _qualify(selected, packets) if require_qualification else None
    # Re-verify after the scan and compare the implementation at both ends. The exact consumed
    # byte hashes above catch a mutation during parsing, not merely before/after snapshots.
    inheritance_after = inherit.verify()
    if inheritance_after != inheritance or sha_file(Path(__file__)) != implementation_before:
        raise SystemExit("M19 ROSTER REFUSED: inputs or implementation changed during build")
    inventory_result = {
        "_schema": "m19-term-inventory-v2",
        "quality_access": False,
        "query_authoring": False,
        "consumed_inputs": {
            "corpus_sha256": report["consumed_corpus_sha256"],
            "tokenizer_sha256": report["consumed_tokenizer_sha256"],
        },
        "inheritance_identity_sha256": inheritance["identity_sha256"],
        "indexable_documents_scanned": report["indexable_documents_scanned"],
        "artifacts_scanned": report["artifacts_scanned"],
        "catalog": report["catalog"],
        "selected_terms": [row["term"] for row in selected],
        "rejected": rejected,
        "added_token_audit": report["added_token_audit"],
    }
    body = {
        "_schema": "m19-term-roster-lock-v2",
        "state": "locked",
        "supersedes": {
            "path": "m19/term-roster-lock.json",
            "reason": "serving-boundary and consumed-input corrections after Astra review",
        },
        "selection_policy": {
            "provenance_limit": "catalog and first support result first committed together at 1736d93; runtime order fixed but no independent pre-scan receipt",
            "quality_informed": False,
            "order": "fixed demand/stable-meaning priority",
            "minimum_terms": policy["minimum"],
            "maximum_terms": policy["maximum"],
            "minimum_distinct_non_equivalent_artifacts": policy["minimum_artifact_support"],
            "minimum_fragment_pieces": policy["pieces_minimum"],
            "minimum_distinct_intent_axes": policy["minimum_distinct_short_intents"],
        },
        "terms": selected,
        "source_qualifications": qualifications,
        "added_token_audit": report["added_token_audit"],
        "bindings": {
            "registry_sha256": sha_file(REGISTRY_PATH),
            "inheritance_lock_sha256": sha_file(M19 / "inheritance-lock.json"),
            "inventory_implementation_sha256": implementation_before,
            "consumed_corpus_sha256": report["consumed_corpus_sha256"],
            "consumed_tokenizer_sha256": report["consumed_tokenizer_sha256"],
            "inventory_result_identity_sha256": sha_json(inventory_result),
        },
    }
    roster = {**body, "identity_sha256": sha_json(body)}
    return inventory_result, roster, packets


def _publish_or_verify(path, obj):
    path = Path(path)
    if path.exists():
        if load_json(path) != obj:
            raise SystemExit(f"M19 ROSTER REFUSED: existing immutable output differs: {path}")
        return "verified"
    try:
        create_json(path, obj)
        return "created"
    except FileExistsError:
        if load_json(path) != obj:
            raise SystemExit(f"M19 ROSTER REFUSED: competing immutable output differs: {path}")
        return "verified-after-race"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--qualification-packet", action="store_true")
    args = parser.parse_args(argv)
    if args.qualification_packet:
        inventory_result, roster, packets = build_outputs(require_qualification=False)
        packet = {
            "_schema": "m19-source-qualification-packet-v1",
            "not_queries_or_relevance_labels": True,
            "selected_terms": inventory_result["selected_terms"],
            "candidates": packets,
        }
        write_json(WORK / "term-source-qualification.json", packet)
        print(json.dumps({"packet": str(WORK / "term-source-qualification.json"),
                          "terms": inventory_result["selected_terms"]}, sort_keys=True))
        return
    inventory_result, roster, _ = build_outputs()
    result_path = RESULTS / "m19_term_inventory_v2.json"
    roster_path = M19 / "term-roster-lock-v2.json"
    if args.verify:
        if not result_path.exists() or not roster_path.exists():
            raise SystemExit("M19 ROSTER REFUSED: v2 immutable outputs are incomplete")
        if load_json(result_path) != inventory_result or load_json(roster_path) != roster:
            raise SystemExit("M19 ROSTER REFUSED: current inventory differs from frozen v2 outputs")
    else:
        _publish_or_verify(result_path, inventory_result)
        _publish_or_verify(roster_path, roster)
    print(json.dumps({"terms": inventory_result["selected_terms"],
                      "identity_sha256": roster["identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
