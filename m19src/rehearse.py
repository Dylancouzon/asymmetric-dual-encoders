"""One-command synthetic rehearsal of pooling, judgments, metrics and confirmation resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from m19src import common, judgments, metrics, zero
from m19src.confirmation import ConfirmationTransaction
from m19src.term_inventory import _extend_tokenizer


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
    specs = [
        {"query_id": "synthetic-q1", "text": "k8s probes", "term": "k8s",
         "primary_class": "short_context", "tags": [],
         "source_exclusion_identity": "synthetic-source"},
        {"query_id": "synthetic-q2", "text": "k8s probes changed after version 2 upgrade",
         "term": "k8s", "primary_class": "longer_control", "tags": ["version"],
         "source_exclusion_identity": "synthetic-source-2"},
    ]
    routes = {}
    for route in judgments.ROUTES:
        if route in ("t0_teacher_dense", "t0_teacher_dbsf"):
            artifacts = ["a1"]
        elif route in ("v1_dense", "v1_dbsf"):
            artifacts = ["a4"]
        else:
            artifacts = ["a2", "a3"]
        routes[route] = {
            spec["query_id"]: [_route_row(route, artifact, rank)
                               for rank, artifact in enumerate(artifacts, start=1)]
            for spec in specs
        }
    return specs, routes


def _prepare_decision(root, query_path, query_sha256, registry):
    """Create the small real artifacts needed to exercise decision authentication."""
    registry = json.loads(json.dumps(registry))
    registry["development_eligibility"]["net_term_wins_minimum"] = 1
    registry["confirmation_eligibility"]["net_term_wins_minimum"] = 1
    registry["numerical_gates"]["added_row_bytes"] = 8
    registry_path = root / "registry.json"
    _write_or_verify(registry_path, registry)

    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "k": 3, "##8": 4, "##s": 5,
             "other": 6}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    base_tokenizer = tokenizer.to_str().encode()
    base_tokenizer_path = root / "base-tokenizer.json"
    _write_or_verify(base_tokenizer_path, base_tokenizer)
    float_rows = np.array([
        [0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.0, -0.1], [-0.1, 0.1, 0.2, 0.1],
        [0.5, -0.2, 0.1, 0.0], [0.0, 0.3, -0.1, 0.2], [0.1, 0.0, 0.4, -0.2],
        [0.3, 0.2, 0.1, 0.0],
    ], dtype=np.float32)
    base_codes, base_scales = zero.quantize_rows(float_rows)
    base_model_path = root / "base-model.npz"
    _write_or_verify(base_model_path, zero._deterministic_npz(
        {"rows_int8": base_codes, "int8_scale": base_scales}
    ))
    extended, audit = _extend_tokenizer(base_tokenizer, ["k8s"])
    roster_body = {"_schema": "m19-synthetic-roster-v1", "terms": [{"term": "k8s"}],
                   "selected_added_token_audit": audit}
    roster = {**roster_body, "identity_sha256": common.sha_json(roster_body)}
    roster_path = root / "roster.json"
    _write_or_verify(roster_path, roster)
    pooling = {"fallback_token_id": 1, "learned_weights": False,
               "preproc": {"add_special_tokens": True, "max_length": 512,
                           "pool_mode": "sqrt", "prefix": ""}, "weights_folded": True}
    base_config = {**pooling, "document_encoder": {"dim": 4}}
    base_config_path = root / "base-config.json"
    _write_or_verify(base_config_path, base_config)
    inheritance_body = {"_schema": "m19-synthetic-inheritance-v1",
                        "identities": {"released_effective_table": {
                            "pooling_sha256": common.sha_json(pooling)}},
                        "inherited_data": {
                            "zero_v1_model": {"path": str(base_model_path.resolve()),
                                              "sha256": common.sha_file_unchecked(base_model_path)},
                            "zero_v1_tokenizer": {"path": str(base_tokenizer_path.resolve()),
                                                  "sha256": common.sha_file_unchecked(base_tokenizer_path)},
                            "zero_v1_config": {"path": str(base_config_path.resolve()),
                                               "sha256": common.sha_file_unchecked(base_config_path)},
                            **{role: {"sha256": str(index + 1) * 64}
                               for index, role in enumerate((
                                   "index_corpus_jsonl", "doc_ids", "document_vectors",
                                   "bm25_data", "bm25_indices", "bm25_indptr", "bm25_vocab",
                                   "bm25_params"))}}}
    inheritance = {**inheritance_body, "identity_sha256": common.sha_json(inheritance_body)}
    inheritance_path = root / "inheritance.json"
    _write_or_verify(inheritance_path, inheritance)
    teacher = {"k8s": zero._normalize(np.array([0.2, 0.7, -0.1, 0.4], dtype=np.float32))}
    teacher_array = np.stack([teacher["k8s"]])
    teacher_path = root / "teacher-vectors.npy"
    _write_or_verify(teacher_path, zero._npy_bytes(teacher_array))
    teacher_receipt = {"_schema": "m19-teacher-vectors-v1",
                       "model": registry["inheritance"]["document_encoder"],
                       "query_prefix": registry["candidate"]["query_prefix"], "terms": ["k8s"],
                       "dtype": str(teacher_array.dtype), "shape": list(teacher_array.shape),
                       "vectors_sha256": common.sha_array(teacher_array)}
    teacher_receipt_path = root / "teacher-receipt.json"
    _write_or_verify(teacher_receipt_path, teacher_receipt)
    built = zero.construct_added_rows(base_codes, base_scales, base_tokenizer, roster, teacher)
    codes, scales = zero.compact_table(base_codes, base_scales, built["T0-teacher"])
    verification = {"base_codes": base_codes, "base_scales": base_scales,
                    "base_tokenizer_payload": base_tokenizer, "roster": roster,
                    "inheritance_identity": inheritance["identity_sha256"],
                    "pooling_identity_sha256": common.sha_json(pooling)}
    provenance = {"inheritance_identity": inheritance["identity_sha256"],
                  "roster_identity": roster["identity_sha256"],
                  "base_codes_sha256": common.sha_array(base_codes),
                  "base_scales_sha256": common.sha_array(base_scales),
                  "base_tokenizer_sha256": common.sha_bytes(base_tokenizer),
                  "pooling_identity_sha256": common.sha_json(pooling),
                  "selected_added_token_audit": audit}
    payload = zero.bundle_payload("T0-teacher", codes, scales, extended, base_config, provenance)
    bundle_dir = root / "bundle"
    bundle_report = zero.publish_bundle(bundle_dir, payload, verification=verification)
    algebra = zero.algebra_gates(base_codes, base_scales, built, teacher,
                                 registry=registry, base_config=base_config)
    row_receipt = {"_schema": "m19-row-receipt-v1", "variant": "T0-teacher",
                   "codes_sha256": common.sha_array(codes),
                   "scales_sha256": common.sha_array(scales),
                   "tokenizer_sha256": common.sha_bytes(extended.to_str().encode()),
                   "algebra_receipts_sha256": common.sha_json(built["receipts"]),
                   "algebra_gates_sha256": common.sha_json(algebra)}
    row_receipt_path = root / "row-receipt.json"
    _write_or_verify(row_receipt_path, row_receipt)
    dev_queries = [
        {"query_id": "d-short", "text": "k8s probes", "term": "k8s",
         "primary_class": "short_context", "tags": []},
        {"query_id": "d-long", "text": "k8s probes changed after version 2 upgrade",
         "term": "k8s", "primary_class": "longer_control", "tags": ["version"]},
    ]
    development_split_path = root / "development.jsonl"
    development_split_payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in dev_queries)
    _write_or_verify(development_split_path, development_split_payload)
    query_seal = {"_schema": "m19-query-split-seal-v1", "splits": {"development": {
        "sha256": common.sha_bytes(development_split_payload),
        "query_ids": [row["query_id"] for row in dev_queries]}}}
    query_seal_path = root / "query-seal.json"
    _write_or_verify(query_seal_path, query_seal)
    candidate_build = {
        "_schema": "m19-candidate-build-v1", "state": "complete",
        "inheritance_identity": inheritance["identity_sha256"],
        "roster_identity": roster["identity_sha256"],
        "bundles": {"T0-teacher": {"identity_sha256": bundle_report["identity_sha256"]}},
    }
    candidate_build_path = root / "candidate-build.json"
    _write_or_verify(candidate_build_path, candidate_build)
    sequence = [dev_queries[index % len(dev_queries)]["text"]
                for index in range(registry["numerical_gates"]["latency_queries"])]
    serving_inputs = {
        "registry_sha256": common.sha_file_unchecked(registry_path),
        "inheritance_identity_sha256": inheritance["identity_sha256"],
        "roster_identity_sha256": roster["identity_sha256"],
        "query_seal_sha256": common.sha_file_unchecked(query_seal_path),
        "development_queries_sha256": common.sha_file_unchecked(development_split_path),
        "candidate_build_sha256": common.sha_file_unchecked(candidate_build_path),
        "bundle_identity_sha256": bundle_report["identity_sha256"],
        "index_inputs_sha256": {
            role: inheritance["inherited_data"][role]["sha256"]
            for role in ("index_corpus_jsonl", "doc_ids", "document_vectors", "bm25_data",
                         "bm25_indices", "bm25_indptr", "bm25_vocab", "bm25_params")
        },
    }
    route_semantics = {
        "dense_dtype": "float32", "exclusion_before_truncation": True,
        "passage_depth": 500, "artifact_depth": 100,
        "tie_break": "descending score then ascending passage_id",
    }
    serving_receipt = {"_schema": "m19-serving-gates-v2", "variant": "T0-teacher",
                       "bundle_identity_sha256": bundle_report["identity_sha256"],
                       "checks": {key: True for key in (
                           "loader_parity", "tokenizer_boundaries", "no_match_ranking_parity",
                           "pooling_identity", "resident_memory", "encoder_latency",
                           "end_to_end_latency")},
                       "measurements": {"loader_parity_max_abs": 0.0,
                                        "released_loader_parity_max_abs": 0.0,
                                        "added_row_bytes": 8,
                                        "encoder_latency_median_ratio": 1.0,
                                        "encoder_latency_p95_ratio": 1.0,
                                        "encoder_latency_median_additive_ms": 0.0,
                                        "encoder_latency_p95_additive_ms": 0.0,
                                        "end_to_end_latency_median_ratio": 1.0,
                                        "end_to_end_latency_p95_ratio": 1.0,
                                        "temporary_peak_bytes": 0, "rss_high_water_kib": 0},
                       "benchmark": {"query_count": registry["numerical_gates"]["latency_queries"],
                                     "query_sequence_sha256": common.sha_texts(sequence),
                                     "host": {"node": "synthetic", "platform": "synthetic",
                                              "gpu": "synthetic"},
                                     "threads": {"torch": 1, "bm25": 1},
                                     "warmup": 20, "repetitions": 1},
                       "inputs": serving_inputs, "route_semantics": route_semantics}
    serving_receipt_path = root / "serving-receipt.json"
    _write_or_verify(serving_receipt_path, serving_receipt)

    dev_qrels = {row["query_id"]: {"good": 1, "bad": 0} for row in dev_queries}
    candidate = {row["query_id"]: ["good"] for row in dev_queries}
    baseline = {row["query_id"]: ["bad"] for row in dev_queries}
    dev_runs = {"dense_candidate": candidate, "dense_v1": baseline,
                "hybrid_candidate": candidate, "hybrid_v1": baseline}
    dev_pool = {"_schema": "m19-artifact-pool-v1", "routes": list(judgments.ROUTES),
                "queries": {row["query_id"]: {"artifact_ids": ["bad", "good"],
                    "route_top10": {route: (candidate[row["query_id"]]
                        if route in ("t0_teacher_dense", "t0_teacher_dbsf")
                        else baseline[row["query_id"]]) for route in judgments.ROUTES}}
                    for row in dev_queries}}
    dev_support = [{"query_id": "d-short", "artifact_id": "good", "pass": True}]
    computed = metrics.evaluate_frozen(
        dev_runs, dev_qrels, dev_queries, registry["development_eligibility"],
        {("d-short", "good"): True},
    )
    artifacts = {"development_qrels": dev_qrels, "development_runs": dev_runs,
                 "development_pool_manifest": dev_pool,
                 "development_queries": dev_queries, "development_support": dev_support}
    paths = {}
    for role, value in artifacts.items():
        paths[role] = root / f"{role}.json"
        _write_or_verify(paths[role], value)
    evaluation = {"_schema": "m19-development-evaluation-v1",
                  "input_sha256": {role: common.sha_file_unchecked(path)
                                   for role, path in paths.items()}, "result": computed}
    paths["development_evaluation"] = root / "development_evaluation.json"
    _write_or_verify(paths["development_evaluation"], evaluation)
    review_scope_path = root / "review-scope.json"
    _write_or_verify(review_scope_path, {"_schema": "m19-review-scope-v1",
                                        "files": {"synthetic": "7" * 64}})
    reviewed_commit = "0" * 40
    review_paths = {"implementation_review": root / "implementation-review.json",
                    "astra_review": root / "astra-review.json"}
    for role, path in review_paths.items():
        short_role = role.removesuffix("_review")
        _write_or_verify(path, {"_schema": "m19-review-go-v1", "role": short_role,
                                "reviewer_id": f"synthetic-{short_role}-reviewer",
                                "decision": "GO", "reviewed_commit": reviewed_commit,
                                "scope_sha256": common.sha_file_unchecked(review_scope_path)})

    role_paths = {"registry": registry_path, "inheritance_lock": inheritance_path,
                  "roster": roster_path, "base_model": base_model_path,
                  "base_tokenizer": base_tokenizer_path, "base_config": base_config_path,
                  "candidate_build": candidate_build_path, "query_seal": query_seal_path,
                  "development_query_split": development_split_path,
                  "teacher_vectors": teacher_path, "teacher_receipt": teacher_receipt_path,
                  "row_receipt": row_receipt_path, "serving_receipt": serving_receipt_path,
                  "review_scope": review_scope_path,
                  **paths, **review_paths}
    for name in ("model.npz", "config.json", "tokenizer.json", "provenance.json", "complete.json"):
        role_paths["bundle_" + name.split(".")[0]] = bundle_dir / name
    bindings = {role: {"path": str(path.resolve()), "sha256": common.sha_file_unchecked(path)}
                for role, path in role_paths.items()}
    bundle_hashes = {Path(row["path"]).name: row["sha256"] for role, row in bindings.items()
                     if role.startswith("bundle_")}
    return {
        "_schema": "m19-confirmation-decision-synthetic-v1", "mode": "synthetic",
        "reviewed_commit": reviewed_commit, "transaction_id": "synthetic-rehearsal-v1",
        "candidate_id": "T0-teacher", "eligible": True,
        "bundle_hashes": bundle_hashes,
        "development_qrels_sha256": bindings["development_qrels"]["sha256"],
        "development_results_sha256": bindings["development_evaluation"]["sha256"],
        "development_eligibility_sha256": common.sha_json(computed),
        "term_roster_sha256": roster["identity_sha256"],
        "inheritance_identity": inheritance["identity_sha256"],
        "row_formula": {"formula": registry["candidate"]["formula"],
                        "scale_convention": registry["candidate"]["scale_convention"],
                        "version": registry["versions"]["row_formula"]},
        "pool_recipe": registry["retrieval"], "evidence_recipe": registry["judgments"],
        "judgment_recipe": registry["judgments"], "metric_recipe": registry["metrics"],
        "numerical_gates": registry["numerical_gates"],
        "primary_judge_id": "synthetic-primary", "auditor_id": "synthetic-auditor",
        "confirmation_query_path": str(query_path.resolve()),
        "confirmation_query_sha256": query_sha256,
        "bindings": bindings,
        "registry_sections": {key: registry[key] for key in (
            "versions", "candidate", "retrieval", "judgments", "metrics", "numerical_gates",
            "development_eligibility", "confirmation_eligibility", "confirmation_states",
        )},
        "review_gos": [
            {"role": "implementation", "reviewer_id": "synthetic-implementation-reviewer",
             "decision": "GO", "findings_sha256": bindings["implementation_review"]["sha256"]},
            {"role": "astra", "reviewer_id": "synthetic-astra-reviewer",
             "decision": "GO", "findings_sha256": bindings["astra_review"]["sha256"]},
        ],
    }


def run_rehearsal(root=None):
    """Run or resume a fixed synthetic transaction, including an object reconstruction."""
    registry = common.load_json(common.REGISTRY_PATH)
    original_work, original_confirmation = common.WORK, common.CONFIRMATION_WORK
    synthetic_root = Path(root or (original_work / "rehearsal-v4"))
    common.WORK = synthetic_root
    common.CONFIRMATION_WORK = synthetic_root / "confirmation"
    try:
        query_path = common.CONFIRMATION_WORK / "queries.jsonl"
        query_sha = _write_or_verify(
            query_path,
            (b'{"author_id":"synthetic-author","primary_class":"short_context",'
             b'"query_id":"synthetic-q1",'
             b'"tags":[],"term":"k8s","text":"k8s probes"}\n'
             b'{"author_id":"synthetic-author","primary_class":"longer_control",'
             b'"query_id":"synthetic-q2",'
             b'"tags":["version"],"term":"k8s",'
             b'"text":"k8s probes changed after version 2 upgrade"}\n'),
        )
        decision_path = synthetic_root / "decision-lock.json"
        _write_or_verify(decision_path, _prepare_decision(
            synthetic_root, query_path, query_sha, registry
        ))
        receipts = synthetic_root / "receipts"
        tx = ConfirmationTransaction(decision_path, receipts)
        if not (receipts / "00-locked.json").exists():
            tx.initialize()
        state = tx.current()["state"]
        if state == "locked":
            tx.claim()
            [json.loads(line) for line in tx.read_bound_bytes(query_path).splitlines() if line]
            state = "claimed"

        specs, routes = _pool_fixture()
        pool, packet = judgments.build_pool(
            routes, specs, seed=19019, cap=int(registry["judgments"]["confirmation_cap"])
        )
        judgments.assert_blinded(packet)
        pool_path, packet_path = common.CONFIRMATION_WORK / "pool.json", (
            common.CONFIRMATION_WORK / "packet.json"
        )
        _write_or_verify(pool_path, pool)
        _write_or_verify(packet_path, packet)
        candidate = {spec["query_id"]: ["a1"] for spec in specs}
        baseline = {spec["query_id"]: ["a4"] for spec in specs}
        metric_runs = {"dense_candidate": candidate, "dense_v1": baseline,
                       "hybrid_candidate": candidate, "hybrid_v1": baseline}
        metric_runs_path = common.CONFIRMATION_WORK / "metric-runs.json"
        _write_or_verify(metric_runs_path, metric_runs)
        if state == "claimed":
            tx.freeze_pools({"pool_manifest": pool_path, "evidence_packet": packet_path,
                             "metric_runs": metric_runs_path})
            state = "pools-frozen"
        if state == "pools-frozen":
            tx.begin_judgments()
            # Re-instantiation is the rehearsed interruption across the judgment boundary.
            tx = ConfirmationTransaction(decision_path, receipts)
            state = tx.current()["state"]
            packet = json.loads(tx.read_bound_bytes(packet_path))

        primary = {row["item_id"]: int(row["artifact_id"] == "a1")
                   for row in packet["items"]}
        audit = judgments.select_audit(packet, primary, seed=19019, fraction=0.20)
        auditor = {item_id: primary[item_id] for item_id in audit["item_ids"]}
        agreement = judgments.audit_agreement(audit, primary, auditor, minimum=0.90)
        primary_path = common.CONFIRMATION_WORK / "primary-01.json"
        audit_path = common.CONFIRMATION_WORK / "audit-01.json"
        support_path = common.CONFIRMATION_WORK / "supporting-passages.json"
        support = [{"query_id": spec["query_id"], "artifact_id": "a1", "pass": True}
                   for spec in specs if spec["primary_class"] == "short_context"]
        _write_or_verify(primary_path, primary)
        _write_or_verify(audit_path, {"sample": audit, "labels": auditor, "agreement": agreement})
        _write_or_verify(support_path, support)
        if state == "judgments-in-progress":
            tx.checkpoint_judgment_batch("primary-01", primary_path)
            tx.checkpoint_judgment_batch("audit-01", audit_path)
            tx.checkpoint_judgment_batch("supporting-01", support_path)
            qrels_path = common.CONFIRMATION_WORK / "qrels.json"
            tx.freeze_qrels(
                qrels_path, packet_path=packet_path, primary_batch_id="primary-01",
                audit_batch_id="audit-01", supporting_batch_id="supporting-01",
            )
            state = "qrels-frozen"

        if state == "qrels-frozen":
            metrics_path = common.CONFIRMATION_WORK / "metrics.json"
            tx.score(metrics_path)
            state = "scored"
        if state == "scored":
            tx.complete()
        result = tx.reconcile()
        result.update({
            "_schema": "m19-synthetic-rehearsal-v2", "synthetic": True,
            "interruption_boundary": "judgments-in-progress", "pool_items": len(packet["items"]),
            "audit_fraction": audit["fraction"], "audit_agreement": agreement["agreement"],
            "metric_exposure_after_qrels_freeze": True,
        })
        return result
    finally:
        common.WORK, common.CONFIRMATION_WORK = original_work, original_confirmation


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(common.RESULTS / "m19_rehearsal_v4.json"))
    args = parser.parse_args(argv)
    result = run_rehearsal()
    payload = _json_bytes(result)
    try:
        common.atomic_create_bytes(args.output, payload)
    except FileExistsError:
        if common.sha_file_unchecked(args.output) != common.sha_bytes(payload):
            raise SystemExit("M19 REHEARSAL STOP: immutable result differs")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
