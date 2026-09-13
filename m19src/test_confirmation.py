import json

import pytest

from m19src import common
from m19src.confirmation import ConfirmationTransaction


def _write(path, value):
    payload = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode()
    common.atomic_create_bytes(path, payload)
    return common.sha_bytes(payload)


def _fixture(tmp_path, monkeypatch, *, wrong_query_hash=False):
    work = tmp_path / "work" / "m19"
    confirmation = work / "confirmation"
    monkeypatch.setattr(common, "WORK", work)
    monkeypatch.setattr(common, "CONFIRMATION_WORK", confirmation)
    query_path = confirmation / "queries.jsonl"
    query_hash = _write(query_path, b'{"query_id":"c1","text":"k8s probes"}\n')
    decision = {
        "_schema": "m19-confirmation-decision-v1", "transaction_id": "synthetic-1",
        "candidate_id": "T0-teacher", "eligible": True,
        "bundle_hashes": {"model.npz": "a" * 64, "tokenizer.json": "b" * 64},
        "development_qrels_sha256": "c" * 64,
        "development_results_sha256": "d" * 64,
        "development_eligibility_sha256": "e" * 64,
        "term_roster_sha256": "f" * 64, "inheritance_identity": "1" * 64,
        "row_formula": {"kind": "teacher-minus-fixed", "scale": "original-bare-norm"},
        "pool_recipe": {"routes": 7, "depth": 10},
        "evidence_recipe": {"passages": 3, "characters": 1200},
        "judgment_recipe": {"labels": [0, 1], "audit_minimum": 0.2},
        "metric_recipe": {"primary": "P@10", "denominator": 10},
        "numerical_gates": {"headroom": 0.05},
        "primary_judge_id": "primary", "auditor_id": "auditor",
        "confirmation_query_path": str(query_path.resolve()),
        "confirmation_query_sha256": "2" * 64 if wrong_query_hash else query_hash,
        "review_gos": [
            {"role": "implementation", "reviewer_id": "reviewer-1", "decision": "GO",
             "findings_sha256": "3" * 64},
            {"role": "astra", "reviewer_id": "reviewer-2", "decision": "GO",
             "findings_sha256": "4" * 64},
        ],
    }
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
    pool = confirmation / "pool.json"
    packet = confirmation / "packet.json"
    _write(pool, {"items": 1})
    _write(packet, {"blinded": True})
    tx.freeze_pools({"pool": pool, "packet": packet})
    tx.begin_judgments()
    primary = confirmation / "primary-01.json"
    audit = confirmation / "audit-01.json"
    _write(primary, {"labels": {"i1": 1}})
    _write(audit, {"labels": {"i1": 1}})
    tx.checkpoint_judgment_batch("primary-01", primary)
    tx.checkpoint_judgment_batch("audit-01", audit)

    tx = ConfirmationTransaction(tx.decision_path, tx.receipt_dir)
    metrics = confirmation / "metrics.json"
    _write(metrics, {"p10": 1.0})
    with pytest.raises(SystemExit, match="before qrels freeze"):
        tx.score(metrics)
    with pytest.raises(common.ProtectedRead):
        tx.read_bound_bytes(metrics)
    qrels = confirmation / "qrels.json"
    _write(qrels, {"c1": {"a1": 1}})
    tx.freeze_qrels({"qrels": qrels}, batch_ids=["primary-01", "audit-01"])
    tx.score(metrics)
    tx.complete(outcome="pass")
    receipt = tx.reconcile()
    assert receipt["state"] == "complete"
    assert receipt["outcome"] == "pass"
    assert receipt["bound_file_count"] == 7


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
    decision["review_gos"][1]["reviewer_id"] = "reviewer-1"
    bad = tmp_path / "work" / "m19" / "bad-decision.json"
    _write(bad, decision)
    with pytest.raises(ValueError, match="not independent"):
        ConfirmationTransaction(bad, tx.receipt_dir)
