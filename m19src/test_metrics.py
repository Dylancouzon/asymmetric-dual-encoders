import pytest

from m19src import metrics


def _specs():
    return [
        {"query_id": "q1", "term": "a", "primary_class": "short_context"},
        {"query_id": "q2", "term": "a", "primary_class": "short_context"},
        {"query_id": "q3", "term": "b", "primary_class": "short_context"},
        {"query_id": "q4", "term": "b", "primary_class": "longer_control"},
    ]


def test_binary_metrics_use_fixed_precision_denominator_and_pooled_idcg():
    row = metrics.score_query(["a", "x"], {"a": 1, "b": 1, "x": 0})
    assert row["precision_at_10"] == 0.1
    assert row["pooled_recall_at_10"] == 0.5
    assert row["mrr_at_10"] == 1.0
    assert row["success_at_10"] == 1.0
    assert 0 < row["ndcg_at_10"] < 1


def test_term_macro_weights_terms_equally_and_reports_complete_differences():
    baseline = {
        "q1": {"precision_at_10": 0.0}, "q2": {"precision_at_10": 0.0},
        "q3": {"precision_at_10": 0.2}, "q4": {"precision_at_10": 0.0},
    }
    candidate = {
        "q1": {"precision_at_10": 0.1}, "q2": {"precision_at_10": 0.1},
        "q3": {"precision_at_10": 0.2}, "q4": {"precision_at_10": 0.5},
    }
    paired = metrics.paired_differences(candidate, baseline, _specs())
    assert paired["per_term"] == {"a": 0.1, "b": 0.0}
    assert paired["fixed_roster_mean"] == 0.05
    assert paired["query_outcomes"] == {"win": 2, "tie": 1, "loss": 0}
    assert paired["term_outcomes"] == {"win": 1, "tie": 1, "loss": 0}
    assert paired["population_inference"] is None


def test_eligibility_requires_strict_majority_and_all_safety_checks():
    specs = _specs()[:3]
    base = {qid: {"precision_at_10": 0.0} for qid in ("q1", "q2", "q3")}
    cand = {qid: {"precision_at_10": 0.1} for qid in ("q1", "q2", "q3")}
    rules = {
        "short_precision_delta_minimum": 0.03, "net_term_wins_minimum": 1,
        "hybrid_precision_delta_minimum": -0.01,
        "numeric_version_control_delta_minimum": -0.02,
        "longer_control_delta_minimum": -0.02,
    }
    out = metrics.eligibility(
        dense_candidate=cand, dense_v1=base, hybrid_delta=0, ndcg_delta=0.01,
        numeric_version_delta=0, longer_delta=0, supporting_passage_audit=True,
        query_specs=specs, rules=rules,
    )
    assert out["eligible"]
    out = metrics.eligibility(
        dense_candidate=cand, dense_v1=base, hybrid_delta=0, ndcg_delta=0.01,
        numeric_version_delta=0, longer_delta=0, supporting_passage_audit=False,
        query_specs=specs, rules=rules,
    )
    assert not out["eligible"] and not out["checks"]["supporting_passages"]


def test_score_run_refuses_missing_queries():
    with pytest.raises(ValueError, match="differ"):
        metrics.score_run({"q": []}, {"other": {}})


def test_empty_safety_slice_is_refused():
    with pytest.raises(ValueError, match="empty"):
        metrics.slice_term_macro_delta({}, {}, _specs(), lambda row: "numeric" in row.get("tags", []))


def test_metrics_refuse_duplicate_runs_and_nonbinary_qrels():
    with pytest.raises(ValueError, match="unique"):
        metrics.score_query(["a", "a"], {"a": 1})
    with pytest.raises(ValueError, match="binary"):
        metrics.score_query(["a"], {"a": 2})


def test_frozen_evaluator_derives_slices_and_support_gate():
    specs = [
        {"query_id": "a-s", "term": "a", "primary_class": "short_context", "tags": []},
        {"query_id": "a-l", "term": "a", "primary_class": "longer_control",
         "tags": ["version"]},
        {"query_id": "b-s", "term": "b", "primary_class": "short_context", "tags": []},
        {"query_id": "b-l", "term": "b", "primary_class": "longer_control",
         "tags": ["numeric"]},
    ]
    qrels = {row["query_id"]: {"good": 1, "bad": 0} for row in specs}
    candidate = {row["query_id"]: ["good"] for row in specs}
    baseline = {row["query_id"]: ["bad"] for row in specs}
    runs = {"dense_candidate": candidate, "dense_v1": baseline,
            "hybrid_candidate": candidate, "hybrid_v1": baseline}
    rules = {"short_precision_delta_minimum": 0.03, "net_term_wins_minimum": 1,
             "hybrid_precision_delta_minimum": -0.01,
             "numeric_version_control_delta_minimum": -0.02,
             "longer_control_delta_minimum": -0.02}
    support = {("a-s", "good"): True, ("b-s", "good"): True}
    result = metrics.evaluate_frozen(runs, qrels, specs, rules, support)
    assert result["eligibility"]["eligible"]
    result = metrics.evaluate_frozen(runs, qrels, specs, rules, {})
    assert not result["eligibility"]["eligible"]
