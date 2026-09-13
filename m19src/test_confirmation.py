import json

import pytest

from m19src import common, rehearse
from m19src.confirmation import ConfirmationTransaction


def _write(path, value):
    payload = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode()
    common.atomic_create_bytes(path, payload)
    return common.sha_bytes(payload)


def _fixture(tmp_path, monkeypatch, *, wrong_query_hash=False):
    registry = common.load_json(common.REGISTRY_PATH)
    work = tmp_path / "work" / "m19"
    confirmation = work / "confirmation"
    monkeypatch.setattr(common, "WORK", work)
    monkeypatch.setattr(common, "CONFIRMATION_WORK", confirmation)
    query_path = confirmation / "queries.jsonl"
    query_hash = _write(
        query_path,
        (b'{"author_id":"author","primary_class":"short_context",'
         b'"query_id":"synthetic-q1",'
         b'"tags":[],"term":"k8s","text":"k8s probes"}\n'
         b'{"author_id":"author","primary_class":"longer_control",'
         b'"query_id":"synthetic-q2",'
         b'"tags":["version"],"term":"k8s",'
         b'"text":"k8s probes changed after version 2 upgrade"}\n'),
    )
    decision = rehearse._prepare_decision(work, query_path, query_hash, registry)
    decision["transaction_id"] = "synthetic-1"
    if wrong_query_hash:
        decision["confirmation_query_sha256"] = "2" * 64
    decision_path = work / "decision.json"
    _write(decision_path, decision)
    return ConfirmationTransaction(decision_path, work / "receipts"), confirmation


def test_interrupted_transaction_resumes_across_judgment_boundary(tmp_path, monkeypatch):
    tx, confirmation = _fixture(tmp_path, monkeypatch)
    tx.initialize()
    with pytest.raises(common.ProtectedRead):
        common.admit_read(confirmation / "queries.jsonl")
    tx.claim()
    assert b"k8s probes" in tx.read_bound_bytes(confirmation / "queries.jsonl")

    tx = ConfirmationTransaction(tx.decision_path, tx.receipt_dir)
    assert tx.current()["state"] == "claimed"
    specs, routes = rehearse._pool_fixture()
    pool_value, packet_value = rehearse.judgments.build_pool(routes, specs)
    pool = confirmation / "pool.json"
    packet = confirmation / "packet.json"
    runs = confirmation / "runs.json"
    _write(pool, pool_value)
    _write(packet, packet_value)
    candidate = {spec["query_id"]: ["a1"] for spec in specs}
    baseline = {spec["query_id"]: ["a4"] for spec in specs}
    _write(runs, {"dense_candidate": candidate, "dense_v1": baseline,
                  "hybrid_candidate": candidate, "hybrid_v1": baseline})
    tx.freeze_pools({"pool_manifest": pool, "evidence_packet": packet, "metric_runs": runs})
    tx.begin_judgments()
    primary = confirmation / "primary-01.json"
    audit = confirmation / "audit-01.json"
    primary_labels = {row["item_id"]: int(row["artifact_id"] == "a1")
                      for row in packet_value["items"]}
    audit_sample = rehearse.judgments.select_audit(packet_value, primary_labels)
    auditor_labels = {item_id: primary_labels[item_id] for item_id in audit_sample["item_ids"]}
    support = confirmation / "support.json"
    _write(primary, primary_labels)
    _write(audit, {"sample": audit_sample, "labels": auditor_labels})
    _write(support, [{"query_id": "synthetic-q1", "artifact_id": "a1", "pass": True}])
    tx.checkpoint_judgment_batch("primary-01", primary)
    tx.checkpoint_judgment_batch("audit-01", audit)
    tx.checkpoint_judgment_batch("support-01", support)

    tx = ConfirmationTransaction(tx.decision_path, tx.receipt_dir)
    early_metrics = confirmation / "early-metrics.json"
    with pytest.raises(SystemExit, match="before qrels freeze"):
        tx.score(early_metrics)
    assert not early_metrics.exists()
    with pytest.raises(common.ProtectedRead):
        tx.read_bound_bytes(early_metrics)
    qrels = confirmation / "qrels.json"
    tx.freeze_qrels(
        qrels, packet_path=packet, primary_batch_id="primary-01", audit_batch_id="audit-01",
        supporting_batch_id="support-01",
    )
    metrics = confirmation / "metrics.json"
    tx.score(metrics)
    tx.complete()
    receipt = tx.reconcile()
    assert receipt["state"] == "complete"
    assert receipt["outcome"] == "pass"
    assert receipt["bound_file_count"] == 9


def test_wrong_query_hash_refuses_claim_and_never_exposes_content(tmp_path, monkeypatch):
    tx, confirmation = _fixture(tmp_path, monkeypatch, wrong_query_hash=True)
    tx.initialize()
    with pytest.raises(SystemExit, match="query hash differs"):
        tx.claim()
    assert tx.current()["state"] == "locked"
    with pytest.raises(common.ProtectedRead):
        tx.read_bound_bytes(confirmation / "queries.jsonl")


def test_consumed_failure_is_terminal_and_resumable(tmp_path, monkeypatch):
    tx, _ = _fixture(tmp_path, monkeypatch)
    tx.initialize()
    tx.claim()
    failed = tx.mark_incomplete(reason="unrecoverable synthetic failure")
    assert failed["outcome"] == "consumed-incomplete"
    assert tx.mark_incomplete(reason="ignored on resume") == failed
    assert tx.reconcile()["outcome"] == "consumed-incomplete"


def test_decision_requires_two_distinct_go_reviews(tmp_path, monkeypatch):
    tx, _ = _fixture(tmp_path, monkeypatch)
    decision = common.load_json(tx.decision_path)
    decision["review_gos"][1]["reviewer_id"] = decision["review_gos"][0]["reviewer_id"]
    bad = tmp_path / "work" / "m19" / "bad-decision.json"
    _write(bad, decision)
    with pytest.raises(ValueError, match="not independent"):
        ConfirmationTransaction(bad, tx.receipt_dir)


def test_current_refuses_mutated_decision_binding(tmp_path, monkeypatch):
    tx, _ = _fixture(tmp_path, monkeypatch)
    tx.initialize()
    row_receipt = tx._bound_path("row_receipt")
    row_receipt.write_text("tampered\n")
    with pytest.raises(SystemExit, match="binding changed: row_receipt"):
        tx.current()
