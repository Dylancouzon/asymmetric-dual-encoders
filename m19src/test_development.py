import json

import pytest

from m19src import common, development, judgments, rehearse


def _write(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    common.atomic_create_bytes(path, payload)
    return path


def test_development_transaction_orders_reviews_judgments_qrels_and_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "WORK", tmp_path)
    tx = development.DevelopmentTransaction(tmp_path / "development")
    tx.initialize()
    with pytest.raises(SystemExit, match="frozen qrels"):
        tx.score()

    specs, routes, metadata = rehearse._pool_fixture()
    pool, packet = judgments.build_pool(routes, specs, metadata, cap=3000)
    metric_runs = {
        role: {query_id: pool["queries"][query_id]["route_top10"][route]
               for query_id in pool["queries"]}
        for role, route in judgments.METRIC_ROUTE_ROLES.items()
    }
    paths = {
        "queries": _write(tmp_path / "queries.json", specs),
        "pool_manifest": _write(tmp_path / "pool.json", pool),
        "evidence_packet": _write(tmp_path / "packet.json", packet),
        "metric_runs": _write(tmp_path / "runs.json", metric_runs),
    }
    tx._transition("locked", "pool-frozen",
                   added={role: development._binding(path) for role, path in paths.items()})

    scope = _write(tmp_path / "scope.json", {"_schema": "m19-review-scope-v1"})
    scope_sha = common.sha_file(scope)
    reviews = []
    for role in ("implementation", "astra"):
        reviews.append(_write(tmp_path / f"{role}.json", {
            "_schema": "m19-review-go-v1", "role": role, "decision": "GO",
            "reviewer_id": role, "reviewed_commit": "1" * 40,
            "scope_sha256": scope_sha,
        }))
    tx.begin_judgments(reviews[0], reviews[1], scope)

    primary = {row["item_id"]: int(row["artifact_id"] == "a1") for row in packet["items"]}
    audit = judgments.select_audit(packet, primary)
    auditor = {item_id: primary[item_id] for item_id in audit["item_ids"]}
    primary_path = _write(tmp_path / "primary.json", primary)
    audit_path = _write(tmp_path / "audit.json", {"sample": audit, "labels": auditor})
    adjudication_path = _write(tmp_path / "adjudication.json", {})
    support_path = _write(tmp_path / "support.json", [
        {"query_id": "synthetic-q1", "artifact_id": "a1", "pass": True}])
    tx.freeze_qrels(
        primary_path, audit_path, adjudication_path, support_path,
        primary_reviewer_id="primary", auditor_id="auditor")
    result = tx.score()
    assert result["state"] == "scored"
    assert (tmp_path / "development" / "qrels.json").exists()
    assert (tmp_path / "development" / "evaluation.json").exists()
