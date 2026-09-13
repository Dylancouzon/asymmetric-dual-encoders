"""Exact aggregate bounds for the post-hoc M19 development uncertainty diagnostic."""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from m19src import metrics
from m19src.common import RESULTS, WORK, create_json, load_json, registry, sha_file, sha_json


INPUT_NAMES = (
    "pool-manifest.json",
    "metric-runs.json",
    "queries.json",
    "primary-labels.json",
    "audit-packet.json",
    "audit-review-labels.json",
)
ROUTE_MAP = {
    "dense_candidate": "t0_teacher_dense",
    "dense_v1": "v1_dense",
    "hybrid_candidate": "t0_teacher_dbsf",
    "hybrid_v1": "v1_dbsf",
    "stella_dense": "stella_dense",
}


def original_item_id(query_id, artifact_id):
    return sha_json({"query_id": query_id, "artifact_id": artifact_id})[:24]


def resolve_labels(pool, primary, audit_packet, auditor):
    audit_rows = {row["item_id"]: row for row in audit_packet["items"]}
    if set(audit_rows) != set(auditor):
        raise ValueError("audit packet and labels differ")
    audited = {}
    for repeat_id, row in audit_rows.items():
        item_id = original_item_id(row["query_id"], row["artifact_id"])
        if item_id not in primary or item_id in audited:
            raise ValueError("concealed audit identity does not map uniquely to primary labels")
        audited[item_id] = auditor[repeat_id]

    expected = {
        original_item_id(query_id, artifact_id)
        for query_id, row in pool["queries"].items()
        for artifact_id in row["artifact_ids"]
    }
    if expected != set(primary):
        raise ValueError("primary labels differ from frozen pool")

    known, unknown = {}, set()
    counts = Counter()
    for item_id, primary_label in primary.items():
        audit_label = audited.get(item_id)
        if item_id not in audited:
            if type(primary_label) is int and primary_label in (0, 1):
                known[item_id] = primary_label
                counts["unaudited_primary_binary"] += 1
            else:
                unknown.add(item_id)
                counts["unaudited_primary_unjudgeable"] += 1
        elif (type(primary_label) is int and primary_label in (0, 1)
              and type(audit_label) is int and audit_label in (0, 1)):
            if primary_label == audit_label:
                known[item_id] = primary_label
                counts["audited_binary_agreement"] += 1
            else:
                unknown.add(item_id)
                counts["audited_binary_disagreement"] += 1
        else:
            unknown.add(item_id)
            counts["audited_either_unjudgeable"] += 1
    binary_total = counts["audited_binary_agreement"] + counts["audited_binary_disagreement"]
    summary = {
        **dict(sorted(counts.items())),
        "audit_items": len(audited),
        "known_items": len(known),
        "unknown_items": len(unknown),
        "resolved_binary_agreement": (
            counts["audited_binary_agreement"] / binary_total if binary_total else None
        ),
        "resolved_binary_agreement_denominator": binary_total,
    }
    return known, unknown, summary


def query_states(query_id, artifacts, route_rows, known, unknown, max_unknown=20):
    artifacts = list(map(str, artifacts))
    artifact_set = set(artifacts)
    if len(artifact_set) != len(artifacts):
        raise ValueError("frozen pool repeats an artifact")
    for route, ranked in route_rows.items():
        top_ten = list(map(str, ranked[:10]))
        if len(top_ten) != len(set(top_ten)) or not set(top_ten).issubset(artifact_set):
            raise ValueError(f"{route} top ten is duplicate or outside the frozen pool")
    item_ids = {a: original_item_id(query_id, a) for a in artifacts}
    unknown_artifacts = sorted(a for a, item_id in item_ids.items() if item_id in unknown)
    if len(unknown_artifacts) > max_unknown:
        raise ValueError(f"query has {len(unknown_artifacts)} unknowns above exact-enumeration cap")
    fixed = {a: known[item_id] for a, item_id in item_ids.items() if item_id in known}
    if len(fixed) + len(unknown_artifacts) != len(item_ids):
        raise ValueError("query labels are neither known nor unknown")
    states = []
    for assignment in itertools.product((0, 1), repeat=len(unknown_artifacts)):
        qrels = {**fixed, **dict(zip(unknown_artifacts, assignment))}
        states.append({
            route: metrics.score_query(ranked, qrels)
            for route, ranked in route_rows.items()
        })
    return states, len(unknown_artifacts)


def contrast_bounds(states, specs, candidate, baseline, metric, predicate):
    by_term = defaultdict(list)
    for query_id, rows in states.items():
        spec = specs[query_id]
        if predicate(spec):
            if metric == "precision_at_10":
                values = sorted({
                    Fraction(round(row[candidate][metric] * 10)
                             - round(row[baseline][metric] * 10), 10)
                    for row in rows
                })
            else:
                values = sorted({row[candidate][metric] - row[baseline][metric]
                                 for row in rows})
            by_term[spec["term"]].append(values)
    if not by_term:
        raise ValueError("contrast slice is empty")

    term_bounds = {}
    sign_sets = {}
    for term, query_values in sorted(by_term.items()):
        zero = Fraction(0) if metric == "precision_at_10" else 0.0
        attainable = {zero}
        for values in query_values:
            attainable = {total + value for total in attainable for value in values}
        attainable = {value / len(query_values) for value in attainable}
        term_bounds[term] = [min(attainable), max(attainable)]
        sign_sets[term] = {1 if value > 0 else -1 if value < 0 else 0
                           for value in attainable}
    zero = Fraction(0) if metric == "precision_at_10" else 0.0
    lower = sum((value[0] for value in term_bounds.values()), zero) / len(term_bounds)
    upper = sum((value[1] for value in term_bounds.values()), zero) / len(term_bounds)
    return {
        "metric": metric,
        "terms": len(term_bounds),
        "queries": sum(len(values) for values in by_term.values()),
        "fixed_roster_mean": {"minimum": lower, "maximum": upper},
        "term_outcomes": {
            "minimum_wins": sum(signs == {1} for signs in sign_sets.values()),
            "maximum_wins": sum(1 in signs for signs in sign_sets.values()),
            "minimum_net_wins": sum(min(signs) for signs in sign_sets.values()),
            "maximum_net_wins": sum(max(signs) for signs in sign_sets.values()),
            "ambiguous_terms": sum(len(signs) > 1 for signs in sign_sets.values()),
        },
        "per_term_bounds": term_bounds,
    }


def threshold_status(bounds, threshold, *, strict=False):
    lower = bounds["minimum"]
    upper = bounds["maximum"]
    if isinstance(lower, Fraction):
        threshold = Fraction(str(threshold))
    lower_passes = lower > threshold if strict else lower >= threshold
    upper_passes = upper > threshold if strict else upper >= threshold
    return "robust_pass" if lower_passes else "robust_fail" if not upper_passes else "label_sensitive"


def term_status(outcomes, threshold, field):
    minimum = outcomes[f"minimum_{field}"]
    maximum = outcomes[f"maximum_{field}"]
    return "robust_pass" if minimum >= threshold else "robust_fail" if maximum < threshold else "label_sensitive"


def jsonable(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def analyze(root: Path):
    paths = {name: root / name for name in INPUT_NAMES}
    values = {name: load_json(path) for name, path in paths.items()}
    pool = values["pool-manifest.json"]
    runs = values["metric-runs.json"]
    query_rows = values["queries.json"]
    specs = {row["query_id"]: row for row in query_rows}
    if set(specs) != set(pool["queries"]) or any(set(run) != set(specs) for run in runs.values()):
        raise ValueError("query coverage differs across frozen development inputs")
    known, unknown, judgment_summary = resolve_labels(
        pool, values["primary-labels.json"], values["audit-packet.json"],
        values["audit-review-labels.json"],
    )

    states, unknown_by_query = {}, {}
    for query_id, pool_row in pool["queries"].items():
        route_rows = {name: runs[name][query_id] for name in runs}
        route_rows["stella_dense"] = pool_row["route_top10"][ROUTE_MAP["stella_dense"]]
        states[query_id], unknown_by_query[query_id] = query_states(
            query_id, pool_row["artifact_ids"], route_rows, known, unknown,
        )

    short = lambda row: row["primary_class"] == "short_context"
    longer = lambda row: row["primary_class"] == "longer_control"
    numeric = lambda row: longer(row) and bool(set(row.get("tags") or []) & {"numeric", "version"})
    bounds = {
        "headroom_precision": contrast_bounds(
            states, specs, "stella_dense", "dense_v1", "precision_at_10", short),
        "dense_precision": contrast_bounds(
            states, specs, "dense_candidate", "dense_v1", "precision_at_10", short),
        "dense_ndcg": contrast_bounds(
            states, specs, "dense_candidate", "dense_v1", "ndcg_at_10", short),
        "hybrid_precision": contrast_bounds(
            states, specs, "hybrid_candidate", "hybrid_v1", "precision_at_10", short),
        "numeric_version_precision": contrast_bounds(
            states, specs, "dense_candidate", "dense_v1", "precision_at_10", numeric),
        "longer_precision": contrast_bounds(
            states, specs, "dense_candidate", "dense_v1", "precision_at_10", longer),
    }
    rules = registry()
    headroom = rules["headroom_gate"]
    eligibility = rules["development_eligibility"]
    gates = {
        "headroom_precision": threshold_status(
            bounds["headroom_precision"]["fixed_roster_mean"],
            headroom["minimum_term_macro_precision_delta"]),
        "headroom_net_term_wins": term_status(
            bounds["headroom_precision"]["term_outcomes"],
            headroom["minimum_net_term_wins"], "net_wins"),
        "short_precision": threshold_status(
            bounds["dense_precision"]["fixed_roster_mean"],
            eligibility["short_precision_delta_minimum"]),
        "majority_terms_win": (
            "robust_pass" if bounds["dense_precision"]["term_outcomes"]["minimum_wins"]
            > bounds["dense_precision"]["terms"] / 2 else
            "robust_fail" if bounds["dense_precision"]["term_outcomes"]["maximum_wins"]
            <= bounds["dense_precision"]["terms"] / 2 else "label_sensitive"),
        "net_term_wins": term_status(
            bounds["dense_precision"]["term_outcomes"],
            eligibility["net_term_wins_minimum"], "net_wins"),
        "short_ndcg_positive": threshold_status(
            bounds["dense_ndcg"]["fixed_roster_mean"], 0.0, strict=True),
        "hybrid_preserved": threshold_status(
            bounds["hybrid_precision"]["fixed_roster_mean"],
            eligibility["hybrid_precision_delta_minimum"]),
        "numeric_version_preserved": threshold_status(
            bounds["numeric_version_precision"]["fixed_roster_mean"],
            eligibility["numeric_version_control_delta_minimum"]),
        "longer_preserved": threshold_status(
            bounds["longer_precision"]["fixed_roster_mean"],
            eligibility["longer_control_delta_minimum"]),
        "supporting_passages": "not_evaluated",
    }
    measured = [value for value in gates.values() if value != "not_evaluated"]
    if "robust_fail" in measured:
        interpretation = "ROBUST_NUMERIC_FAILURE"
    elif "label_sensitive" in measured:
        interpretation = "LABEL_SENSITIVE"
    else:
        interpretation = "NUMERIC_GATES_ROBUST_PASS_SUPPORT_UNASSESSED"
    return jsonable({
        "_schema": "m19-development-sensitivity-v1",
        "status": "posthoc-development-only",
        "original_outcome_unchanged": "ENCODER_INCONCLUSIVE",
        "confirmation_accessed": False,
        "qrels_created": False,
        "input_sha256": {name: sha_file(path) for name, path in sorted(paths.items())},
        "judgments": judgment_summary,
        "enumeration": {
            "queries": len(states),
            "queries_with_unknowns": sum(value > 0 for value in unknown_by_query.values()),
            "maximum_unknowns_per_query": max(unknown_by_query.values()),
            "binary_states_evaluated": sum(len(value) for value in states.values()),
        },
        "bounds": bounds,
        "gate_status": gates,
        "interpretation": interpretation,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=WORK / "development")
    parser.add_argument("--out", type=Path, default=RESULTS / "m19_development_sensitivity.json")
    args = parser.parse_args(argv)
    result = analyze(args.root)
    create_json(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()
