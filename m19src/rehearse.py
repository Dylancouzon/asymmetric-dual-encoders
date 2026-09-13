"""One-command synthetic rehearsal of pooling, judgments, metrics and confirmation resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m19src import common, judgments, metrics
from m19src.confirmation import ConfirmationTransaction


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_or_verify(path, value):
    payload = value if isinstance(value, bytes) else _json_bytes(value)
    try:
        common.atomic_create_bytes(path, payload)
    except FileExistsError:
        # This helper is confined to the declared synthetic root and compares only a digest.
        if common.sha_file_unchecked(path) != common.sha_bytes(payload):
            raise SystemExit(f"M19 REHEARSAL STOP: synthetic fixture differs: {path}")
    return common.sha_bytes(payload)


def _route_row(route, artifact, rank):
    passage_id = f"{route}-{artifact}"
    return {
        "artifact_id": artifact, "score": float(20 - rank), "passage_id": passage_id,
        "passage": {
            "passage_id": passage_id, "artifact_id": artifact,
            "text": f"Synthetic evidence for k8s probes in artifact {artifact} via {route}.",
            "title": f"Synthetic {artifact}", "kind": "issue",
            "source_url": f"https://synthetic.invalid/{artifact}",
            "parent_metadata": {"synthetic": True},
        },
    }


def _pool_fixture():
    spec = {"query_id": "synthetic-q1", "text": "k8s probes", "term": "k8s",
            "primary_class": "short_context", "source_exclusion_identity": "synthetic-source"}
    routes = {}
    for route_index, route in enumerate(judgments.ROUTES):
        artifacts = ["a1", "a2", "a3"] if route_index % 2 == 0 else ["a2", "a4", "a5"]
        routes[route] = {spec["query_id"]: [
            _route_row(route, artifact, rank)
            for rank, artifact in enumerate(artifacts, start=1)
        ]}
    return spec, routes


def _decision(query_path, query_sha256):
    return {
        "_schema": "m19-confirmation-decision-v1", "transaction_id": "synthetic-rehearsal-v1",
        "candidate_id": "T0-teacher", "eligible": True,
        "bundle_hashes": {"model.npz": "a" * 64, "tokenizer.json": "b" * 64},
        "development_qrels_sha256": "c" * 64,
        "development_results_sha256": "d" * 64,
        "development_eligibility_sha256": "e" * 64,
        "term_roster_sha256": "f" * 64, "inheritance_identity": "1" * 64,
        "row_formula": {"kind": "teacher-minus-fixed", "scale": "original-bare-norm"},
        "pool_recipe": {"routes": list(judgments.ROUTES), "depth": 10},
        "evidence_recipe": {"passages": 3, "characters": 1200},
        "judgment_recipe": {"labels": [0, 1], "audit_minimum": 0.20,
                            "agreement_minimum": 0.90},
        "metric_recipe": {"primary": "precision_at_10", "denominator": 10},
        "numerical_gates": {"headroom": 0.05},
        "primary_judge_id": "synthetic-primary", "auditor_id": "synthetic-auditor",
        "confirmation_query_path": str(query_path.resolve()),
        "confirmation_query_sha256": query_sha256,
        "review_gos": [
            {"role": "implementation", "reviewer_id": "synthetic-reviewer-1",
             "decision": "GO", "findings_sha256": "2" * 64},
            {"role": "astra", "reviewer_id": "synthetic-reviewer-2",
             "decision": "GO", "findings_sha256": "3" * 64},
        ],
    }


def run_rehearsal(root=None):
    """Run or resume a fixed synthetic transaction, including an object reconstruction."""
    original_work, original_confirmation = common.WORK, common.CONFIRMATION_WORK
    synthetic_root = Path(root or (original_work / "rehearsal"))
    common.WORK = synthetic_root
    common.CONFIRMATION_WORK = synthetic_root / "confirmation"
    try:
        query_path = common.CONFIRMATION_WORK / "queries.jsonl"
        query_sha = _write_or_verify(
            query_path,
            b'{"query_id":"synthetic-q1","text":"k8s probes","term":"k8s"}\n',
        )
        decision_path = synthetic_root / "decision-lock.json"
        _write_or_verify(decision_path, _decision(query_path, query_sha))
        receipts = synthetic_root / "receipts"
        tx = ConfirmationTransaction(decision_path, receipts)
        if not (receipts / "00-locked.json").exists():
            tx.initialize()
        state = tx.current()["state"]
        if state == "locked":
            tx.claim()
            json.loads(tx.read_bound_bytes(query_path))
            state = "claimed"

        spec, routes = _pool_fixture()
        pool, packet = judgments.build_pool(routes, [spec], seed=19019, cap=3000)
        judgments.assert_blinded(packet)
        pool_path, packet_path = common.CONFIRMATION_WORK / "pool.json", (
            common.CONFIRMATION_WORK / "packet.json"
        )
        _write_or_verify(pool_path, pool)
        _write_or_verify(packet_path, packet)
        if state == "claimed":
            tx.freeze_pools({"pool": pool_path, "packet": packet_path})
            state = "pools-frozen"
        if state == "pools-frozen":
            tx.begin_judgments()
            # Re-instantiation is the rehearsed interruption across the judgment boundary.
            tx = ConfirmationTransaction(decision_path, receipts)
            state = tx.current()["state"]

        primary = {row["item_id"]: 1 for row in packet["items"]}
        audit = judgments.select_audit(packet, primary, seed=19019, fraction=0.20)
        auditor = {item_id: 1 for item_id in audit["item_ids"]}
        agreement = judgments.audit_agreement(audit, primary, auditor, minimum=0.90)
        primary_path = common.CONFIRMATION_WORK / "primary-01.json"
        audit_path = common.CONFIRMATION_WORK / "audit-01.json"
        _write_or_verify(primary_path, primary)
        _write_or_verify(audit_path, {"sample": audit, "labels": auditor, "agreement": agreement})
        if state == "judgments-in-progress":
            tx.checkpoint_judgment_batch("primary-01", primary_path)
            tx.checkpoint_judgment_batch("audit-01", audit_path)
            qrels = judgments.freeze_binary_labels(
                packet, primary, {}, agreement, primary_reviewer_id="synthetic-primary",
                auditor_id="synthetic-auditor", query_author_ids={"synthetic-author"},
            )
            qrels_path = common.CONFIRMATION_WORK / "qrels.json"
            _write_or_verify(qrels_path, qrels)
            tx.freeze_qrels({"qrels": qrels_path}, batch_ids=["primary-01", "audit-01"])
            state = "qrels-frozen"
        else:
            qrels = judgments.freeze_binary_labels(
                packet, primary, {}, agreement, primary_reviewer_id="synthetic-primary",
                auditor_id="synthetic-auditor", query_author_ids={"synthetic-author"},
            )

        if state == "qrels-frozen":
            selected = {spec["query_id"]: [
                row["artifact_id"] for row in routes["t0_teacher_dense"][spec["query_id"]]
            ]}
            result = metrics.score_run(selected, qrels)
            metrics_path = common.CONFIRMATION_WORK / "metrics.json"
            _write_or_verify(metrics_path, result)
            tx.score(metrics_path)
            state = "scored"
        if state == "scored":
            tx.complete(outcome="pass")
        result = tx.reconcile()
        result.update({
            "_schema": "m19-synthetic-rehearsal-v1", "synthetic": True,
            "interruption_boundary": "judgments-in-progress", "pool_items": len(packet["items"]),
            "audit_fraction": audit["fraction"], "audit_agreement": agreement["agreement"],
            "metric_exposure_after_qrels_freeze": True,
        })
        return result
    finally:
        common.WORK, common.CONFIRMATION_WORK = original_work, original_confirmation


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(common.RESULTS / "m19_rehearsal.json"))
    args = parser.parse_args(argv)
    result = run_rehearsal()
    common.write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
