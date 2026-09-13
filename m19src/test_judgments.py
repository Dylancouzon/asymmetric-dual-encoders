import pytest

from m19src import judgments


def _row(artifact, route, rank):
    return {"artifact_id": artifact, "score": 10 - rank, "passage_id": f"{route}-{artifact}",
            "passage": {"text": (f"evidence {route} {artifact} " * 200), "title": artifact,
                        "kind": "issue", "source_url": f"https://x/{artifact}",
                        "parent_metadata": {"thread": artifact}}}


def _fixture():
    specs = [
        {"query_id": "q1", "text": "k8s probes", "term": "k8s", "source_exclusion_identity": "x"},
        {"query_id": "q2", "text": "s3 credentials", "term": "s3", "source_exclusion_identity": None},
    ]
    routes = {}
    for r_index, route in enumerate(judgments.ROUTES):
        routes[route] = {}
        for query in specs:
            artifacts = ["a", "b", "c"] if r_index % 2 == 0 else ["b", "d", "e"]
            routes[route][query["query_id"]] = [_row(a, route, rank) for rank, a in enumerate(artifacts, 1)]
    return routes, specs


def test_seven_route_pool_is_blinded_capped_and_deterministic():
    routes, specs = _fixture()
    manifest, packet = judgments.build_pool(routes, specs)
    manifest2, packet2 = judgments.build_pool(routes, specs)
    assert (manifest, packet) == (manifest2, packet2)
    assert manifest["unique_query_artifact_items"] == 10
    assert judgments.assert_blinded(packet)
    assert all(len(p["text"]) == 1200 for item in packet["items"] for p in item["passages"])
    with pytest.raises(SystemExit, match="above cap"):
        judgments.build_pool(routes, specs, cap=9)


def test_pool_requires_exact_routes_and_queries():
    routes, specs = _fixture()
    routes.pop("bm25")
    with pytest.raises(ValueError, match="registered seven"):
        judgments.build_pool(routes, specs)


def test_frozen_pool_refuses_metric_artifacts_outside_union():
    manifest, packet = judgments.build_pool(*_fixture())
    runs = {role: {query_id: manifest["queries"][query_id]["route_top10"][route]
                   for query_id in manifest["queries"]}
            for role, route in judgments.METRIC_ROUTE_ROLES.items()}
    assert judgments.validate_frozen_pool(manifest, packet, runs)
    runs["dense_candidate"]["q1"] = ["outside"]
    with pytest.raises(ValueError, match="outside judged union"):
        judgments.validate_frozen_pool(manifest, packet, runs)


def test_audit_includes_positives_unjudgeable_and_negative_coverage():
    _, packet = judgments.build_pool(*_fixture())
    labels = {row["item_id"]: 0 for row in packet["items"]}
    labels[packet["items"][0]["item_id"]] = 1
    labels[packet["items"][1]["item_id"]] = "unjudgeable"
    audit = judgments.select_audit(packet, labels)
    assert audit["all_primary_positive_and_unjudgeable"]
    assert audit["fraction"] >= 0.20
    assert {row["query_id"] for row in audit["items"]} == {"q1", "q2"}
    assert {row["term"] for row in audit["items"]} == {"k8s", "s3"}


def test_agreement_and_freeze_require_binary_resolved_labels():
    _, packet = judgments.build_pool(*_fixture())
    primary = {row["item_id"]: 0 for row in packet["items"]}
    primary[packet["items"][0]["item_id"]] = 1
    audit = judgments.select_audit(packet, primary)
    auditor = {item_id: primary[item_id] for item_id in audit["item_ids"]}
    judge_args = {"primary_reviewer_id": "primary", "auditor_id": "auditor",
                  "query_authors": {"q1": "author", "q2": "author"}}
    frozen = judgments.freeze_binary_labels(packet, primary, audit, auditor, {}, **judge_args)
    assert set(frozen["qrels"]) == {"q1", "q2"}
    primary[packet["items"][1]["item_id"]] = "unjudgeable"
    audit = judgments.select_audit(packet, primary)
    auditor = {item_id: 0 for item_id in audit["item_ids"]}
    with pytest.raises(SystemExit, match="unresolved"):
        judgments.freeze_binary_labels(
            packet, primary, audit, auditor, {}, minimum_agreement=0.0, **judge_args
        )


def test_freeze_requires_independent_judges_and_not_only_query_authors():
    _, packet = judgments.build_pool(*_fixture())
    primary = {row["item_id"]: 0 for row in packet["items"]}
    audit = judgments.select_audit(packet, primary)
    auditor = {item_id: 0 for item_id in audit["item_ids"]}
    with pytest.raises(SystemExit, match="must be independent"):
        judgments.freeze_binary_labels(
            packet, primary, audit, auditor, {}, primary_reviewer_id="same", auditor_id="same",
            query_authors={"q1": "author", "q2": "author"},
        )
    with pytest.raises(SystemExit, match="independent of query authors"):
        judgments.freeze_binary_labels(
            packet, primary, audit, auditor, {}, primary_reviewer_id="author-a",
            auditor_id="author-b", query_authors={"q1": "author-a", "q2": "author-b"},
        )


def test_freeze_recomputes_audit_and_requires_fresh_complete_relabel_after_clarification():
    _, packet = judgments.build_pool(*_fixture())
    primary = {row["item_id"]: 0 for row in packet["items"]}
    audit = judgments.select_audit(packet, primary)
    auditor = {item_id: 0 for item_id in audit["item_ids"]}
    authors = {"q1": "author", "q2": "author"}
    altered = {**audit, "fraction": 1.0}
    with pytest.raises(SystemExit, match="differs"):
        judgments.freeze_binary_labels(
            packet, primary, altered, auditor, {}, primary_reviewer_id="primary",
            auditor_id="auditor", query_authors=authors,
        )
    fresh = judgments.select_audit(packet, primary, seed=19020)
    fresh_labels = {item_id: 0 for item_id in fresh["item_ids"]}
    clarification = {"generation": 1, "complete_pool_relabel": True,
                     "prior_primary_sha256": "a" * 64, "rubric_sha256": "b" * 64}
    frozen = judgments.freeze_binary_labels(
        packet, primary, fresh, fresh_labels, {}, primary_reviewer_id="primary",
        auditor_id="auditor", query_authors=authors, clarification=clarification,
    )
    assert frozen["clarification_generation"] == 1
