from fractions import Fraction

import pytest

from m19src.sensitivity import (contrast_bounds, query_states, resolve_labels,
                                threshold_status)


def test_exact_contrast_bounds_are_not_all_zero_all_one_imputations():
    query_id = "q"
    artifacts = ["candidate-only", "v1-only"]
    unknown = {
        "000000000000000000000001",
        "000000000000000000000002",
    }
    # Patch the identity boundary for this pure fixture by supplying the real derived IDs.
    from m19src.sensitivity import original_item_id
    unknown = {original_item_id(query_id, artifact) for artifact in artifacts}
    states, count = query_states(
        query_id, artifacts,
        {"candidate": ["candidate-only"], "baseline": ["v1-only"]},
        known={}, unknown=unknown,
    )
    assert count == 2
    values = {row["candidate"]["precision_at_10"] - row["baseline"]["precision_at_10"]
              for row in states}
    assert values == {-0.1, 0.0, 0.1}
    bounds = contrast_bounds(
        {query_id: states},
        {query_id: {"query_id": query_id, "term": "term", "primary_class": "short_context"}},
        "candidate", "baseline", "precision_at_10",
        lambda row: row["primary_class"] == "short_context",
    )
    assert bounds["fixed_roster_mean"] == {
        "minimum": Fraction(-1, 10), "maximum": Fraction(1, 10)}


def test_label_resolution_keeps_only_binary_agreement_known():
    from m19src.sensitivity import original_item_id
    query_id = "q"
    artifacts = ["agree", "disagree", "abstain", "unaudited"]
    ids = {artifact: original_item_id(query_id, artifact) for artifact in artifacts}
    pool = {"queries": {query_id: {"artifact_ids": artifacts}}}
    audit_packet = {"items": [
        {"item_id": "r1", "query_id": query_id, "artifact_id": "agree"},
        {"item_id": "r2", "query_id": query_id, "artifact_id": "disagree"},
        {"item_id": "r3", "query_id": query_id, "artifact_id": "abstain"},
    ]}
    primary = {ids["agree"]: 1, ids["disagree"]: 1, ids["abstain"]: 0,
               ids["unaudited"]: 0}
    auditor = {"r1": 1, "r2": 0, "r3": "unjudgeable"}
    known, unknown, summary = resolve_labels(pool, primary, audit_packet, auditor)
    assert known == {ids["agree"]: 1, ids["unaudited"]: 0}
    assert unknown == {ids["disagree"], ids["abstain"]}
    assert summary["audited_binary_agreement"] == 1
    assert summary["audited_binary_disagreement"] == 1
    assert summary["audited_either_unjudgeable"] == 1


def test_precision_threshold_comparison_remains_exact():
    exact = {"minimum": Fraction(1, 20), "maximum": Fraction(1, 20)}
    assert threshold_status(exact, 0.05) == "robust_pass"


def test_query_states_refuses_ranked_artifact_outside_pool():
    with pytest.raises(ValueError, match="outside the frozen pool"):
        query_states("q", ["inside"], {"candidate": ["outside"]},
                     known={}, unknown=set())
