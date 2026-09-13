"""Seven-route artifact pools, blinded evidence packets and independent audit sampling."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from m19src.common import sha_bytes, sha_json

ROUTES = (
    "bm25", "v1_dense", "v1_dbsf", "v0_compose_dense", "t0_teacher_dense",
    "t0_teacher_dbsf", "stella_dense",
)
CONCEALED = ("system_identity", "route", "score", "rank", "first_seen_phase")
FORBIDDEN_PACKET_KEYS = set(CONCEALED)
METRIC_ROUTE_ROLES = {"dense_candidate": "t0_teacher_dense", "dense_v1": "v1_dense",
                      "hybrid_candidate": "t0_teacher_dbsf", "hybrid_v1": "v1_dbsf"}
PACKET_ITEM_KEYS = {"item_id", "query_id", "query_text", "term",
                    "source_exclusion_identity", "artifact_id", "title", "url_or_path",
                    "kind", "passages"}


def _row_passages(row):
    if row.get("route_support"):
        return [(support["passage_id"], support["passage"])
                for support in row["route_support"].values()]
    return [(row["passage_id"], row["passage"])]


def build_pool(routes, query_specs, artifact_metadata, *, seed=19019, cap=3000):
    if set(routes) != set(ROUTES):
        raise ValueError(f"pool routes differ from registered seven: {sorted(routes)}")
    specs = {row["query_id"]: row for row in query_specs}
    if any(set(run) != set(specs) for run in routes.values()):
        raise ValueError("every route must contain every query")
    pool_manifest, packet_items = {}, []
    for query_id in sorted(specs):
        artifacts = {}
        provenance = {}
        passage_ranks = defaultdict(dict)
        passage_rows = {}
        for route_name in ROUTES:
            rows = list(routes[route_name][query_id])[:10]
            provenance[route_name] = [row["artifact_id"] for row in rows]
            for rank, row in enumerate(rows, start=1):
                artifact_id = str(row["artifact_id"])
                artifacts.setdefault(artifact_id, row)
                for passage_id, passage in _row_passages(row):
                    passage_id = str(passage_id)
                    previous = passage_ranks[artifact_id].get(passage_id)
                    passage_ranks[artifact_id][passage_id] = min(previous or rank, rank)
                    passage_rows[(artifact_id, passage_id)] = passage
        pool_manifest[query_id] = {"route_top10": provenance,
                                   "artifact_ids": sorted(artifacts)}
        spec = specs[query_id]
        for artifact_id in sorted(artifacts):
            metadata = artifact_metadata.get(artifact_id)
            if (not isinstance(metadata, dict) or not str(metadata.get("title") or "").strip() or
                    not str(metadata.get("url_or_path") or "").strip() or
                    not str(metadata.get("kind") or "").strip() or
                    not isinstance(metadata.get("parent_metadata"), dict) or
                    not metadata["parent_metadata"]):
                raise ValueError(f"artifact metadata is incomplete for {artifact_id}")
            ranked_passages = sorted(
                passage_ranks[artifact_id].items(), key=lambda pair: (-1.0 / pair[1], pair[0])
            )[:3]
            evidence = []
            for passage_id, _ in ranked_passages:
                passage = passage_rows[(artifact_id, passage_id)]
                full_text = str(passage.get("text") or "")
                evidence.append({
                    "passage_id": passage_id,
                    "text": full_text[:1200],
                    "full_text_sha256": sha_bytes(full_text.encode("utf-8", "surrogatepass")),
                    "parent_metadata": dict(metadata["parent_metadata"]),
                })
            packet_items.append({
                "item_id": sha_json({"query_id": query_id, "artifact_id": artifact_id})[:24],
                "query_id": query_id,
                "query_text": spec["text"],
                "term": spec["term"],
                "source_exclusion_identity": spec.get("source_exclusion_identity"),
                "artifact_id": artifact_id,
                "title": str(metadata["title"]),
                "url_or_path": str(metadata["url_or_path"]),
                "kind": str(metadata["kind"]),
                "passages": evidence,
            })
    if len(packet_items) > int(cap):
        raise SystemExit(f"M19 JUDGMENT STOP: pool has {len(packet_items)} items above cap {cap}")
    random.Random(seed).shuffle(packet_items)
    return {
        "_schema": "m19-artifact-pool-v1",
        "routes": list(ROUTES),
        "queries": pool_manifest,
        "unique_query_artifact_items": len(packet_items),
        "cap": int(cap),
        "query_specs_sha256": sha_json({row["query_id"]: {
            "text": row["text"], "term": row["term"],
            "source_exclusion_identity": row.get("source_exclusion_identity")}
            for row in query_specs}),
    }, {
        "_schema": "m19-blinded-evidence-packet-v1",
        "seed": int(seed),
        "items": packet_items,
        "concealed": list(CONCEALED),
    }


def assert_blinded(packet):
    def visit(value):
        if isinstance(value, dict):
            bad = set(value) & FORBIDDEN_PACKET_KEYS
            if bad:
                raise ValueError(f"blinded packet exposes {sorted(bad)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    if set(packet) != {"_schema", "seed", "items", "concealed"}:
        raise ValueError("blinded packet has unexpected top-level fields")
    visit(packet)
    for item in packet["items"]:
        if set(item) != PACKET_ITEM_KEYS:
            raise ValueError("packet item has unexpected fields")
        if (not str(item["title"]).strip() or not str(item["url_or_path"]).strip() or
                not str(item["kind"]).strip() or not item["passages"] or
                len(item["passages"]) > 3 or any(
                    len(row["text"]) > 1200 or
                    not isinstance(row.get("parent_metadata"), dict) or
                    not row["parent_metadata"] for row in item["passages"])):
            raise ValueError("evidence packet metadata or passage limits differ")
    return True


def validate_frozen_pool(manifest, packet, metric_runs, query_specs, *, seed=19019, cap=3000):
    """Join the blinded packet and every scored top-ten to the registered seven-route union."""
    if (manifest.get("_schema") != "m19-artifact-pool-v1" or
            packet.get("_schema") != "m19-blinded-evidence-packet-v1" or
            manifest.get("routes") != list(ROUTES) or manifest.get("cap") != int(cap) or
            packet.get("seed") != int(seed) or
            packet.get("concealed") != list(CONCEALED)):
        raise ValueError("pool/packet schema or route registry differs")
    assert_blinded(packet)
    specs = {row["query_id"]: row for row in query_specs}
    expected_spec_hash = sha_json({query_id: {
        "text": row["text"], "term": row["term"],
        "source_exclusion_identity": row.get("source_exclusion_identity")}
        for query_id, row in specs.items()})
    if (set(specs) != set(manifest.get("queries", {})) or
            manifest.get("query_specs_sha256") != expected_spec_hash):
        raise ValueError("pool query specifications differ from authenticated queries")
    by_query = defaultdict(set)
    seen_items = set()
    for item in packet["items"]:
        spec = specs.get(item["query_id"])
        if (spec is None or item["query_text"] != spec["text"] or
                item["term"] != spec["term"] or
                item["source_exclusion_identity"] != spec.get("source_exclusion_identity")):
            raise ValueError("packet query semantics differ from authenticated queries")
        expected_id = sha_json({"query_id": item["query_id"], "artifact_id": item["artifact_id"]})[:24]
        if item.get("item_id") != expected_id:
            raise ValueError("packet item ID differs from query/artifact identity")
        key = (item["query_id"], item["artifact_id"])
        if key in seen_items:
            raise ValueError("packet repeats a query-artifact item")
        seen_items.add(key)
        by_query[item["query_id"]].add(item["artifact_id"])
    if set(by_query) != set(manifest["queries"]):
        raise ValueError("pool/packet query coverage differs")
    for query_id, row in manifest["queries"].items():
        union = set(row["artifact_ids"])
        if by_query[query_id] != union:
            raise ValueError("pool/packet artifact union differs")
        if set(row["route_top10"]) != set(ROUTES):
            raise ValueError("pool route provenance is incomplete")
        for route in ROUTES:
            ranked = row["route_top10"][route]
            if (len(ranked) > 10 or len(ranked) != len(set(ranked)) or
                    not set(ranked).issubset(union)):
                raise ValueError("route top-ten is duplicate or outside pool union")
    if (manifest.get("unique_query_artifact_items") != len(packet["items"]) or
            len(packet["items"]) > int(cap)):
        raise ValueError("pool/packet item count differs")
    required_runs = set(METRIC_ROUTE_ROLES)
    if set(metric_runs) != required_runs:
        raise ValueError("metric run roles differ from frozen evaluator")
    for run in metric_runs.values():
        if set(run) != set(by_query):
            raise ValueError("metric run query coverage differs from packet")
        for query_id, ranked in run.items():
            top = list(map(str, ranked[:10]))
            if len(top) != len(set(top)) or not set(top).issubset(by_query[query_id]):
                raise ValueError("metric top-ten is duplicate or outside judged union")
    for metric_role, route in METRIC_ROUTE_ROLES.items():
        for query_id in by_query:
            if list(map(str, metric_runs[metric_role][query_id][:10])) != list(map(
                    str, manifest["queries"][query_id]["route_top10"][route])):
                raise ValueError("metric run differs from its registered frozen route")
    return True


def select_audit(packet, primary_labels, *, seed=19019, fraction=0.20):
    items = {row["item_id"]: row for row in packet["items"]}
    if set(items) != set(primary_labels):
        raise ValueError("primary labels do not cover complete pool")
    valid = {0, 1, "unjudgeable"}
    if any(label not in valid for label in primary_labels.values()):
        raise ValueError("labels must be binary or unjudgeable")
    chosen = {item_id for item_id, label in primary_labels.items() if label in (1, "unjudgeable")}
    negatives = sorted(item_id for item_id, label in primary_labels.items() if label == 0)
    random.Random(seed).shuffle(negatives)
    # Seed negatives to cover every query and term when a negative exists for that group.
    for field in ("query_id", "term"):
        groups = defaultdict(list)
        for item_id in negatives:
            groups[items[item_id][field]].append(item_id)
        for group in sorted(groups):
            chosen.add(groups[group][0])
    target = math.ceil(len(items) * float(fraction))
    for item_id in negatives:
        if len(chosen) >= target:
            break
        chosen.add(item_id)
    ordered = [item_id for item_id in (row["item_id"] for row in packet["items"])
               if item_id in chosen]
    return {
        "seed": int(seed),
        "item_ids": ordered,
        "items": [{**items[item_id], "repeat_id": sha_json({"seed": seed, "id": item_id})[:24]}
                  for item_id in ordered],
        "pool_items": len(items),
        "audit_items": len(ordered),
        "fraction": len(ordered) / max(1, len(items)),
        "fraction_minimum": float(fraction),
        "all_primary_positive_and_unjudgeable": all(
            item_id in chosen for item_id, label in primary_labels.items()
            if label in (1, "unjudgeable")
        ),
    }


def audit_agreement(audit, primary_labels, auditor_labels, minimum=0.90):
    expected = set(audit["item_ids"])
    if set(auditor_labels) != expected:
        raise ValueError("auditor labels do not cover exact audit sample")
    if any(type(label) is not int or label not in (0, 1) for label in auditor_labels.values()):
        raise ValueError("auditor labels must be exact binary integers")
    agreements = sum(primary_labels[item_id] == auditor_labels[item_id] for item_id in expected)
    disagreements = sorted(item_id for item_id in expected
                             if primary_labels[item_id] != auditor_labels[item_id])
    rate = agreements / max(1, len(expected))
    return {"agreement": rate, "minimum": float(minimum), "pass": rate >= float(minimum),
            "agreements": agreements, "audited": len(expected), "disagreements": disagreements}


def freeze_binary_labels(packet, primary_labels, audit, auditor_labels, adjudicated_labels, *,
                         primary_reviewer_id, auditor_id, query_authors,
                         seed=19019, fraction=0.20, minimum_agreement=0.90,
                         clarification=None):
    items = {row["item_id"]: row for row in packet["items"]}
    if set(query_authors) != {row["query_id"] for row in packet["items"]}:
        raise SystemExit("M19 JUDGMENT STOP: query-author mapping is incomplete")
    if any(not isinstance(value, str) or not value.strip() or value != value.strip()
           for value in query_authors.values()):
        raise SystemExit("M19 JUDGMENT STOP: query author IDs are invalid")
    authors = set(query_authors.values())
    if (not isinstance(primary_reviewer_id, str) or not primary_reviewer_id.strip() or
            primary_reviewer_id != primary_reviewer_id.strip() or
            not isinstance(auditor_id, str) or not auditor_id.strip() or
            auditor_id != auditor_id.strip() or primary_reviewer_id == auditor_id):
        raise SystemExit("M19 JUDGMENT STOP: primary reviewer and auditor must be independent")
    if primary_reviewer_id in authors:
        raise SystemExit("M19 JUDGMENT STOP: primary reviewer must be independent of query authors")
    if set(items) != set(primary_labels) or any(
            type(label) is not int and label != "unjudgeable" or
            type(label) is int and label not in (0, 1)
            for label in primary_labels.values()):
        raise SystemExit("M19 JUDGMENT STOP: primary labels are incomplete or invalid")
    generation = 0
    expected_seed = int(seed)
    if clarification is not None:
        generation = int(clarification.get("generation", -1))
        if (generation != 1 or clarification.get("complete_pool_relabel") is not True or
                not isinstance(clarification.get("prior_primary_sha256"), str) or
                len(clarification["prior_primary_sha256"]) != 64 or
                not isinstance(clarification.get("rubric_sha256"), str) or
                len(clarification["rubric_sha256"]) != 64):
            raise SystemExit("M19 JUDGMENT STOP: clarification lacks complete-pool relabel proof")
        expected_seed += 1
    expected_audit = select_audit(packet, primary_labels, seed=expected_seed, fraction=fraction)
    if audit != expected_audit:
        raise SystemExit("M19 JUDGMENT STOP: audit sample differs from deterministic protocol")
    audit_report = audit_agreement(
        audit, primary_labels, auditor_labels, minimum=float(minimum_agreement)
    )
    if not audit_report["pass"]:
        raise SystemExit("M19 JUDGMENT STOP: independent agreement is below the gate")
    labels = dict(primary_labels)
    unresolved = set(audit_report["disagreements"]) | {
        item_id for item_id, label in primary_labels.items() if label == "unjudgeable"
    }
    for item_id in unresolved:
        if item_id not in adjudicated_labels:
            raise SystemExit("M19 JUDGMENT STOP: unresolved blind disagreement")
        labels[item_id] = adjudicated_labels[item_id]
    if set(adjudicated_labels) != unresolved or any(
            type(label) is not int or label not in (0, 1)
            for label in adjudicated_labels.values()):
        raise SystemExit("M19 JUDGMENT STOP: adjudications differ from exact unresolved set")
    if set(labels) != set(items) or any(type(label) is not int or label not in (0, 1)
                                       for label in labels.values()):
        raise SystemExit("M19 JUDGMENT STOP: qrels contain missing/non-binary labels")
    qrels = defaultdict(dict)
    for item_id, label in labels.items():
        item = items[item_id]
        qrels[item["query_id"]][item["artifact_id"]] = int(label)
    return {
        "_schema": "m19-frozen-judgments-v1",
        "qrels": {query_id: dict(sorted(rows.items())) for query_id, rows in sorted(qrels.items())},
        "audit_report": audit_report,
        "clarification_generation": generation,
        "reviewers": {"primary": primary_reviewer_id, "auditor": auditor_id},
    }
