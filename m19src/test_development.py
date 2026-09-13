import json

import pytest

from m19src import common, development, judgments, pool_pilot, rehearse


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

    reviewed_commit = "1" * 40
    scope = _write(tmp_path / "scope.json", {
        "_schema": "m19-review-scope-v1", "roles": ["implementation", "astra"],
        "reviewed_commit": reviewed_commit,
        "files": {name: common.sha_file(common.REPO / name)
                  for name in development.REQUIRED_REVIEW_FILES},
    })
    scope_sha = common.sha_file(scope)
    reviews = []
    for role in ("implementation", "astra"):
        reviews.append(_write(tmp_path / f"{role}.json", {
            "_schema": "m19-review-go-v1", "role": role, "decision": "GO",
            "reviewer_id": role, "reviewed_commit": reviewed_commit,
            "scope_sha256": scope_sha,
        }))
    tx.begin_judgments(reviews[0], reviews[1], scope)

    primary = {row["item_id"]: int(row["artifact_id"] == "a1") for row in packet["items"]}
    audit = judgments.select_audit(packet, primary)
    failed_auditor = {item_id: 1 - primary[item_id] for item_id in audit["item_ids"]}
    primary_path = _write(tmp_path / "primary.json", primary)
    failed_audit_path = _write(
        tmp_path / "audit-failed.json", {"sample": audit, "labels": failed_auditor})
    adjudication_path = _write(tmp_path / "adjudication.json", {})
    support_path = _write(tmp_path / "support.json", [
        {"query_id": "synthetic-q1", "artifact_id": "a1", "pass": True}])
    with pytest.raises(SystemExit, match="agreement is below"):
        tx.freeze_qrels(
            primary_path, failed_audit_path, adjudication_path, support_path,
            primary_reviewer_id="primary", auditor_id="auditor")
    assert (tmp_path / "development" / "judgment-attempt-0.json").exists()

    clarification_path = _write(tmp_path / "clarification.json", {
        "generation": 1, "complete_pool_relabel": True,
        "prior_primary_sha256": common.sha_file(primary_path), "rubric_sha256": "4" * 64,
    })
    fresh_audit = judgments.select_audit(packet, primary, seed=19020)
    auditor = {item_id: primary[item_id] for item_id in fresh_audit["item_ids"]}
    audit_path = _write(tmp_path / "audit-fresh.json", {
        "sample": fresh_audit, "labels": auditor})
    tx.freeze_qrels(
        primary_path, audit_path, adjudication_path, support_path,
        primary_reviewer_id="primary", auditor_id="auditor",
        clarification_path=clarification_path)
    result = tx.score()
    assert result["state"] == "scored"
    assert (tmp_path / "development" / "qrels.json").exists()
    assert (tmp_path / "development" / "evaluation.json").exists()
    evaluation = json.loads((tmp_path / "development" / "evaluation.json").read_text())
    assert evaluation["headroom"]["pass"] is False
    assert evaluation["evaluation"] is None
    assert evaluation["eligible"] is False


def test_development_state_chain_rejects_changed_earlier_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "WORK", tmp_path)
    tx = development.DevelopmentTransaction(tmp_path / "development")
    tx.initialize()
    tx._transition("locked", "pool-frozen")
    locked = tx.states["locked"]
    value = json.loads(locked.read_text())
    value["tampered"] = True
    locked.write_text(json.dumps(value))
    with pytest.raises(SystemExit, match="state receipt chain differs"):
        tx.current()


def test_development_review_scope_rejects_empty_file_set(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "WORK", tmp_path)
    scope = _write(tmp_path / "scope.json", {
        "_schema": "m19-review-scope-v1", "roles": ["implementation", "astra"],
        "reviewed_commit": "1" * 40, "files": {},
    })
    with pytest.raises(SystemExit, match="review scope is incomplete"):
        development.validate_review_scope(scope)


def test_pilot_runner_still_publishes_receipt_after_pool_builder_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "WORK", tmp_path)
    pilot_ids = [f"q{index}" for index in range(10)]
    development_rows = [
        {"query_id": f"q{index}", "term": f"term-{index}",
         "primary_class": "short_context"}
        for index in range(60)
    ]
    lock = _write(tmp_path / "pilot-lock.json", {
        "query_count": 10, "pilot_query_ids": pilot_ids,
        "development_queries_sha256": "1" * 64,
        "query_seal_sha256": "2" * 64, "roster_identity_sha256": "3" * 64,
    })
    monkeypatch.setattr(pool_pilot, "PILOT_LOCK", lock)
    monkeypatch.setattr(pool_pilot, "POOL_OUT", tmp_path / "pool.json")
    monkeypatch.setattr(pool_pilot, "PACKET_OUT", tmp_path / "packet.json")
    monkeypatch.setattr(pool_pilot, "RESULT_OUT", tmp_path / "result.json")
    monkeypatch.setattr(pool_pilot, "build_frozen_pool", lambda *args, **kwargs: {
        "queries": development_rows[:10], "development": development_rows,
        "pool": {"queries": {}},
        "packet": {"items": [{"artifact_id": "a1", "title": "title",
                                "url_or_path": "path", "passages": [{"text": "evidence"}]}]},
        "inputs": {},
    })
    result = pool_pilot.run()
    assert result["state"] == "complete"
    assert result["development_projection"]["cap"] == 3000
