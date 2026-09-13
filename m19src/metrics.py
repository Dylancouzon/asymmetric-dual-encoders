"""Registered binary artifact metrics and fixed-roster aggregation/decision helpers."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


METRICS = ("precision_at_10", "ndcg_at_10", "pooled_recall_at_10", "mrr_at_10", "success_at_10")


def score_query(ranked_artifact_ids, qrels, k=10):
    ranked = list(ranked_artifact_ids)[:k]
    positive = {str(artifact) for artifact, label in qrels.items() if int(label) == 1}
    gains = [1 if artifact in positive else 0 for artifact in ranked]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_count = min(k, len(positive))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_count))
    relevant_ranks = [rank + 1 for rank, artifact in enumerate(ranked) if artifact in positive]
    return {
        "precision_at_10": sum(gains) / 10.0,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "pooled_recall_at_10": len(set(ranked) & positive) / len(positive) if positive else 0.0,
        "mrr_at_10": 1.0 / min(relevant_ranks) if relevant_ranks else 0.0,
        "success_at_10": 1.0 if relevant_ranks else 0.0,
        "returned": len(ranked),
        "pool_positives": len(positive),
    }


def score_run(run, qrels):
    if set(run) != set(qrels):
        raise ValueError("run/qrels query IDs differ")
    return {query_id: score_query(run[query_id], qrels[query_id]) for query_id in sorted(qrels)}


def aggregate(per_query, query_specs, primary_class=None):
    specs = {row["query_id"]: row for row in query_specs}
    if set(per_query) != set(specs):
        raise ValueError("metrics/query specs differ")
    selected = [query_id for query_id in sorted(per_query)
                if primary_class is None or specs[query_id]["primary_class"] == primary_class]
    by_term = defaultdict(list)
    for query_id in selected:
        by_term[specs[query_id]["term"]].append(query_id)
    output = {"queries": len(selected), "terms": len(by_term), "per_term": {}}
    for term, query_ids in sorted(by_term.items()):
        output["per_term"][term] = {
            metric: float(np.mean([per_query[qid][metric] for qid in query_ids]))
            for metric in METRICS
        } | {"queries": len(query_ids)}
    output["term_macro"] = {
        metric: float(np.mean([row[metric] for row in output["per_term"].values()]))
            if output["per_term"] else 0.0
        for metric in METRICS
    }
    return output


def paired_differences(candidate, baseline, query_specs, metric="precision_at_10",
                       primary_class="short_context"):
    specs = {row["query_id"]: row for row in query_specs}
    if set(candidate) != set(baseline) or set(candidate) != set(specs):
        raise ValueError("paired query IDs differ")
    per_query, by_term = {}, defaultdict(list)
    for query_id in sorted(candidate):
        if specs[query_id]["primary_class"] != primary_class:
            continue
        delta = float(candidate[query_id][metric] - baseline[query_id][metric])
        per_query[query_id] = delta
        by_term[specs[query_id]["term"]].append(delta)
    per_term = {term: float(np.mean(values)) for term, values in sorted(by_term.items())}

    def label(value):
        return "win" if value > 0 else "loss" if value < 0 else "tie"

    query_counts = Counter(label(value) for value in per_query.values())
    term_counts = Counter(label(value) for value in per_term.values())
    values = list(per_query.values())
    term_values = list(per_term.values())
    return {
        "metric": metric,
        "primary_class": primary_class,
        "per_query": per_query,
        "per_term": per_term,
        "query_outcomes": {key: query_counts.get(key, 0) for key in ("win", "tie", "loss")},
        "term_outcomes": {key: term_counts.get(key, 0) for key in ("win", "tie", "loss")},
        "fixed_roster_mean": float(np.mean(list(per_term.values()))) if per_term else 0.0,
        "query_median": float(np.median(values)) if values else 0.0,
        "query_range": [float(min(values)), float(max(values))] if values else [0.0, 0.0],
        "term_median": float(np.median(term_values)) if term_values else 0.0,
        "term_range": [float(min(term_values)), float(max(term_values))]
            if term_values else [0.0, 0.0],
        "term_dependence_disclosed": True,
        "population_inference": None,
    }


def headroom_gate(stella, v1, query_specs, registry):
    paired = paired_differences(stella, v1, query_specs)
    gate = registry["headroom_gate"]
    outcomes = paired["term_outcomes"]
    short_counts = Counter(row["term"] for row in query_specs
                           if row["primary_class"] == "short_context")
    checks = {
        "judgeable_terms": len(paired["per_term"]) >= int(gate["minimum_judgeable_terms"]),
        "short_intents_per_term": bool(short_counts) and min(short_counts.values()) >= int(
            gate["minimum_development_short_intents_per_term"]
        ),
        "precision_delta": paired["fixed_roster_mean"] >= float(gate["minimum_term_macro_precision_delta"]),
        "net_term_wins": outcomes["win"] - outcomes["loss"] >= int(gate["minimum_net_term_wins"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "paired": paired}


def slice_term_macro_delta(candidate, baseline, query_specs, predicate, metric="precision_at_10"):
    """Point delta on a registered safety slice; an empty slice is a hard protocol error."""
    chosen = [row for row in query_specs if predicate(row)]
    if not chosen:
        raise ValueError("registered safety slice is empty")
    query_ids = {row["query_id"] for row in chosen}
    if not query_ids.issubset(candidate) or not query_ids.issubset(baseline):
        raise ValueError("safety slice metrics are incomplete")
    by_term = defaultdict(list)
    for row in chosen:
        query_id = row["query_id"]
        by_term[row["term"]].append(candidate[query_id][metric] - baseline[query_id][metric])
    return float(np.mean([np.mean(values) for values in by_term.values()]))


def eligibility(*, dense_candidate, dense_v1, hybrid_delta, ndcg_delta, numeric_version_delta,
                longer_delta, supporting_passage_audit, query_specs, rules):
    paired = paired_differences(dense_candidate, dense_v1, query_specs)
    outcomes = paired["term_outcomes"]
    terms = len(paired["per_term"])
    checks = {
        "short_precision": paired["fixed_roster_mean"] >= float(rules["short_precision_delta_minimum"]),
        "majority_terms_win": outcomes["win"] > terms / 2,
        "net_term_wins": outcomes["win"] - outcomes["loss"] >= int(rules["net_term_wins_minimum"]),
        "short_ndcg_positive": float(ndcg_delta) > 0,
        "hybrid_preserved": float(hybrid_delta) >= float(rules["hybrid_precision_delta_minimum"]),
        "numeric_version_preserved": float(numeric_version_delta) >= float(
            rules.get("numeric_version_control_delta_minimum", rules.get("safety_slice_delta_minimum"))
        ),
        "longer_preserved": float(longer_delta) >= float(
            rules.get("longer_control_delta_minimum", rules.get("safety_slice_delta_minimum"))
        ),
        "supporting_passages": bool(supporting_passage_audit),
    }
    return {"eligible": all(checks.values()), "checks": checks, "paired_precision": paired}
