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
FORBIDDEN_PACKET_KEYS = {"route", "routes", "score", "rank", "first_seen_phase", "system"}


def _row_passages(row):
    if row.get("route_support"):
        return [(support["passage_id"], support["passage"])
                for support in row["route_support"].values()]
    return [(row["passage_id"], row["passage"])]


def build_pool(routes, query_specs, *, seed=19019, cap=3000):
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
                    "parent_metadata": dict(passage.get("parent_metadata") or {}),
                })
            representative = artifacts[artifact_id]
            passage = _row_passages(representative)[0][1]
            packet_items.append({
                "item_id": sha_json({"query_id": query_id, "artifact_id": artifact_id})[:24],
                "query_id": query_id,
                "query_text": spec["text"],
                "term": spec["term"],
                "source_exclusion_identity": spec.get("source_exclusion_identity"),
                "artifact_id": artifact_id,
                "title": str(passage.get("title") or ""),
                "url_or_path": str(passage.get("source_url") or passage.get("path") or ""),
                "kind": str(passage.get("kind") or ""),
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
    }, {
        "_schema": "m19-blinded-evidence-packet-v1",
        "seed": int(seed),
        "items": packet_items,
        "concealed": sorted(FORBIDDEN_PACKET_KEYS),
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
    visit(packet["items"])
    for item in packet["items"]:
        if len(item["passages"]) > 3 or any(len(row["text"]) > 1200 for row in item["passages"]):
            raise ValueError("evidence packet exceeds passage limits")
    return True


def select_audit(packet, primary_labels, *, seed=19019, fraction=0.20):
    items = {row["item_id"]: row for row in packet["items"]}
    if set(items) != set(primary_labels):
        raise ValueError("primary labels do not cover complete pool")
    valid = {0, 1, "unjudgeable"}
    if any(label not in valid for label in primary_labels.values()):
        raise ValueError("labels must be binary or unjudgeable")
    chosen = {item_id for item_id, label in primary_labels.items() if label in (1, "unjudgeable")}
    negatives = [item_id for item_id, label in primary_labels.items() if label == 0]
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
        "item_ids": ordered,
        "items": [{**items[item_id], "repeat_id": sha_json({"seed": seed, "id": item_id})[:24]}
                  for item_id in ordered],
        "pool_items": len(items),
        "audit_items": len(ordered),
        "fraction": len(ordered) / max(1, len(items)),
        "all_primary_positive_and_unjudgeable": all(
            item_id in chosen for item_id, label in primary_labels.items()
            if label in (1, "unjudgeable")
        ),
    }


def audit_agreement(audit, primary_labels, auditor_labels, minimum=0.90):
    expected = set(audit["item_ids"])
    if set(auditor_labels) != expected:
        raise ValueError("auditor labels do not cover exact audit sample")
    agreements = sum(primary_labels[item_id] == auditor_labels[item_id] for item_id in expected)
    disagreements = sorted(item_id for item_id in expected
                             if primary_labels[item_id] != auditor_labels[item_id])
    rate = agreements / max(1, len(expected))
    return {"agreement": rate, "minimum": float(minimum), "pass": rate >= float(minimum),
            "agreements": agreements, "audited": len(expected), "disagreements": disagreements}


def freeze_binary_labels(packet, primary_labels, adjudicated_labels, audit_report, *,
                         primary_reviewer_id, auditor_id, query_author_ids,
                         rubric_clarifications=0):
    items = {row["item_id"]: row for row in packet["items"]}
    authors = {str(value) for value in query_author_ids}
    if not primary_reviewer_id or not auditor_id or primary_reviewer_id == auditor_id:
        raise SystemExit("M19 JUDGMENT STOP: primary reviewer and auditor must be independent")
    if primary_reviewer_id in authors and auditor_id in authors:
        raise SystemExit("M19 JUDGMENT STOP: query authors cannot be the sole relevance judges")
    if not audit_report["pass"]:
        raise SystemExit("M19 JUDGMENT STOP: independent agreement is below the gate")
    if int(rubric_clarifications) > 1:
        raise SystemExit("M19 JUDGMENT STOP: more than one rubric clarification")
    labels = dict(primary_labels)
    for item_id in audit_report["disagreements"]:
        if item_id not in adjudicated_labels:
            raise SystemExit("M19 JUDGMENT STOP: unresolved blind disagreement")
        labels[item_id] = adjudicated_labels[item_id]
    if set(labels) != set(items) or any(label not in (0, 1) for label in labels.values()):
        raise SystemExit("M19 JUDGMENT STOP: qrels contain missing/non-binary labels")
    qrels = defaultdict(dict)
    for item_id, label in labels.items():
        item = items[item_id]
        qrels[item["query_id"]][item["artifact_id"]] = int(label)
    return {query_id: dict(sorted(rows.items())) for query_id, rows in sorted(qrels.items())}
