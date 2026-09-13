"""Create-only M19 development pool, judgment, qrels and scoring transaction."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from m19src import common, judgments, metrics
from m19src.build_candidate import _json_bytes, _publish
from m19src.common import M19, WORK, admit_read, load_json, sha_file, sha_json
from m19src.pool_pilot import build_frozen_pool

ROOT = WORK / "development"
STATE_NAMES = ("locked", "pool-frozen", "judgments-in-progress", "qrels-frozen", "scored")
STATE_FILES = {name: ROOT / f"{index:02d}-{name}.json"
               for index, name in enumerate(STATE_NAMES)}
REQUIRED_REVIEW_FILES = {
    "m19src/development.py", "m19src/pool_pilot.py", "m19src/judgments.py",
    "m19src/test_development.py",
}


def _load(path):
    return json.loads(admit_read(path).read_text())


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha_file(path)}


def validate_review_scope(path, *, reviewed_commit=None):
    scope = _load(path)
    files = scope.get("files")
    if (scope.get("_schema") != "m19-review-scope-v1" or
            scope.get("roles") != ["implementation", "astra"] or
            not re.fullmatch(r"[0-9a-f]{40}", str(scope.get("reviewed_commit", ""))) or
            (reviewed_commit is not None and scope["reviewed_commit"] != reviewed_commit) or
            not isinstance(files, dict) or not files or
            not REQUIRED_REVIEW_FILES.issubset(files)):
        raise SystemExit("M19 DEVELOPMENT STOP: review scope is incomplete")
    for relative, expected in files.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SystemExit("M19 DEVELOPMENT STOP: review scope path is unsafe")
        resolved = (common.REPO / candidate).resolve()
        try:
            resolved.relative_to(common.REPO.resolve())
        except ValueError:
            raise SystemExit("M19 DEVELOPMENT STOP: review scope path escapes repository")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or sha_file(resolved) != expected:
            raise SystemExit(f"M19 DEVELOPMENT STOP: reviewed file changed: {relative}")
    return scope


class DevelopmentTransaction:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.states = {name: self.root / STATE_FILES[name].name for name in STATE_NAMES}

    def current(self):
        existing = [name for name in STATE_NAMES if self.states[name].exists()]
        if not existing:
            return None
        expected = list(STATE_NAMES[:STATE_NAMES.index(existing[-1]) + 1])
        if existing != expected:
            raise SystemExit("M19 DEVELOPMENT STOP: state chain is not contiguous")
        state = None
        previous_name = None
        previous_bound = {}
        for name in existing:
            state = _load(self.states[name])
            expected_previous = sha_file(self.states[previous_name]) if previous_name else None
            bound = state.get("bound_files", {})
            if (state.get("_schema") != "m19-development-state-v1" or
                    state.get("state") != name or
                    state.get("previous_sha256") != expected_previous or
                    any(bound.get(role) != binding for role, binding in previous_bound.items())):
                raise SystemExit("M19 DEVELOPMENT STOP: state receipt chain differs")
            previous_name, previous_bound = name, bound
        for binding in state.get("bound_files", {}).values():
            if sha_file(binding["path"]) != binding["sha256"]:
                raise SystemExit("M19 DEVELOPMENT STOP: bound file changed")
        return state

    def _transition(self, expected, target, *, added=None, extra=None):
        current = self.current()
        if (current is None and expected is not None) or (
                current is not None and current["state"] != expected):
            raise SystemExit(f"M19 DEVELOPMENT STOP: {target} requires state {expected}")
        bound = dict(current.get("bound_files", {}) if current else {})
        bound.update(added or {})
        receipt = {
            "_schema": "m19-development-state-v1", "state": target,
            "previous_sha256": sha_file(self.states[expected]) if expected else None,
            "bound_files": dict(sorted(bound.items())), **(extra or {}),
        }
        _publish(self.states[target], _json_bytes(receipt))
        return receipt

    def initialize(self):
        if self.current() is not None:
            return self.current()
        registry = M19 / "registry.json"
        seal = M19 / "query-split-seal-v1.json"
        roster = M19 / "term-roster-lock-v3.json"
        snapshot = M19 / "teacher-snapshot-lock-v1.json"
        return self._transition(None, "locked", added={
            "registry": _binding(registry), "query_seal": _binding(seal),
            "roster": _binding(roster), "teacher_snapshot_lock": _binding(snapshot),
        })

    def freeze_pool(self):
        current = self.current()
        if current and current["state"] == "pool-frozen":
            return current
        if not current or current["state"] != "locked":
            raise SystemExit("M19 DEVELOPMENT STOP: pool freeze requires locked state")
        seal = _load(current["bound_files"]["query_seal"]["path"])
        query_ids = seal["splits"]["development"]["query_ids"]
        registry = _load(current["bound_files"]["registry"]["path"])
        built = build_frozen_pool(
            query_ids, cap=registry["judgments"]["development_cap"],
            expected_inputs={"query_seal_sha256": current["bound_files"]["query_seal"]["sha256"]})
        paths = {
            "queries": self.root / "queries.json",
            "pool_manifest": self.root / "pool-manifest.json",
            "evidence_packet": self.root / "evidence-packet.json",
            "metric_runs": self.root / "metric-runs.json",
        }
        values = {"queries": built["query_specs"], "pool_manifest": built["pool"],
                  "evidence_packet": built["packet"], "metric_runs": built["metric_runs"]}
        for role, path in paths.items():
            _publish(path, _json_bytes(values[role]))
        return self._transition("locked", "pool-frozen",
                                added={role: _binding(path) for role, path in paths.items()},
                                extra={"query_count": len(query_ids),
                                       "pool_items": built["pool"]["unique_query_artifact_items"]})

    def begin_judgments(self, implementation_review, astra_review, review_scope):
        current = self.current()
        if not current or current["state"] != "pool-frozen":
            raise SystemExit("M19 DEVELOPMENT STOP: judgments require frozen pool")
        scope_binding = _binding(review_scope)
        receipts = {"implementation_review": _binding(implementation_review),
                    "astra_review": _binding(astra_review)}
        parsed = [_load(receipts[role]["path"]) for role in receipts]
        if ({row.get("role") for row in parsed} != {"implementation", "astra"} or
                any(row.get("_schema") != "m19-review-go-v1" or row.get("decision") != "GO" or
                    row.get("scope_sha256") != scope_binding["sha256"] or
                    not row.get("reviewer_id") for row in parsed) or
                len({row.get("reviewer_id") for row in parsed}) != 2 or
                len({row.get("reviewed_commit") for row in parsed}) != 1):
            raise SystemExit("M19 DEVELOPMENT STOP: two independent scoped review GOs required")
        validate_review_scope(review_scope, reviewed_commit=parsed[0]["reviewed_commit"])
        return self._transition("pool-frozen", "judgments-in-progress",
                                added={**receipts, "review_scope": scope_binding},
                                extra={"reviewed_commit": parsed[0]["reviewed_commit"]})

    def freeze_qrels(self, primary_path, audit_path, adjudication_path, support_path,
                     *, primary_reviewer_id, auditor_id, clarification_path=None):
        current = self.current()
        if not current or current["state"] != "judgments-in-progress":
            raise SystemExit("M19 DEVELOPMENT STOP: qrels require judgments-in-progress")
        new = {"primary_labels": _binding(primary_path), "audit_labels": _binding(audit_path),
               "adjudications": _binding(adjudication_path),
               "supporting_passages": _binding(support_path)}
        generation = 1 if clarification_path else 0
        clarification = None
        if generation:
            prior_path = self.root / "judgment-attempt-0.json"
            if not prior_path.exists():
                raise SystemExit("M19 DEVELOPMENT STOP: clarification requires a failed first attempt")
            prior = _load(prior_path)
            clarification = _load(clarification_path)
            prior_inputs = prior.get("input_files", {})
            if (prior.get("_schema") != "m19-development-judgment-attempt-v1" or
                    prior.get("generation") != 0 or
                    set(prior_inputs) != {
                        "primary_labels", "audit_labels", "adjudications",
                        "supporting_passages"} or
                    any(sha_file(binding["path"]) != binding["sha256"]
                        for binding in prior_inputs.values())):
                raise SystemExit("M19 DEVELOPMENT STOP: failed judgment input changed")
            if clarification.get("prior_primary_sha256") != prior_inputs[
                    "primary_labels"]["sha256"]:
                raise SystemExit("M19 DEVELOPMENT STOP: clarification does not bind first labels")
            new["clarification"] = _binding(clarification_path)
        attempt_path = self.root / f"judgment-attempt-{generation}.json"
        attempt = {
            "_schema": "m19-development-judgment-attempt-v1", "generation": generation,
            "input_files": dict(sorted(new.items())),
            "reviewers": {"primary": primary_reviewer_id, "auditor": auditor_id},
        }
        _publish(attempt_path, _json_bytes(attempt))
        packet = _load(current["bound_files"]["evidence_packet"]["path"])
        query_specs = _load(current["bound_files"]["queries"]["path"])
        primary = _load(primary_path)
        audit_payload = _load(audit_path)
        adjudications = _load(adjudication_path)
        if any(not isinstance(row.get("author_id"), str) or not row["author_id"].strip() or
               row["author_id"] != row["author_id"].strip() for row in query_specs):
            raise SystemExit("M19 DEVELOPMENT STOP: sealed query author IDs are invalid")
        query_authors = {row["query_id"]: row["author_id"] for row in query_specs}
        frozen = judgments.freeze_binary_labels(
            packet, primary, audit_payload["sample"], audit_payload["labels"], adjudications,
            primary_reviewer_id=primary_reviewer_id, auditor_id=auditor_id,
            query_authors=query_authors, clarification=clarification)
        qrels_path = self.root / "qrels.json"
        judgment_path = self.root / "judgment-freeze.json"
        _publish(qrels_path, _json_bytes(frozen["qrels"]))
        freeze_receipt = {**frozen, "qrels": None,
                          "input_sha256": {role: value["sha256"] for role, value in new.items()}}
        _publish(judgment_path, _json_bytes(freeze_receipt))
        attempt_bindings = {"judgment_attempt": _binding(attempt_path)}
        if generation:
            attempt_bindings["failed_judgment_attempt"] = _binding(
                self.root / "judgment-attempt-0.json")
            attempt_bindings.update({f"failed_{role}": binding
                                     for role, binding in prior_inputs.items()})
        return self._transition("judgments-in-progress", "qrels-frozen", added={
            **new, **attempt_bindings, "qrels": _binding(qrels_path),
            "judgment_freeze": _binding(judgment_path)})

    def score(self):
        current = self.current()
        if not current or current["state"] != "qrels-frozen":
            raise SystemExit("M19 DEVELOPMENT STOP: scoring requires frozen qrels")
        query_specs = _load(current["bound_files"]["queries"]["path"])
        pool = _load(current["bound_files"]["pool_manifest"]["path"])
        packet = _load(current["bound_files"]["evidence_packet"]["path"])
        runs = _load(current["bound_files"]["metric_runs"]["path"])
        qrels = _load(current["bound_files"]["qrels"]["path"])
        registry = _load(current["bound_files"]["registry"]["path"])
        judgments.validate_frozen_pool(
            pool, packet, runs, query_specs, seed=registry["judgments"]["randomization_seed"],
            cap=registry["judgments"]["development_cap"])
        support_rows = _load(current["bound_files"]["supporting_passages"]["path"])
        support = {(row["query_id"], row["artifact_id"]): row["pass"] for row in support_rows}
        stella_run = {query_id: pool["queries"][query_id]["route_top10"]["stella_dense"]
                      for query_id in pool["queries"]}
        stella = metrics.score_run(stella_run, qrels)
        v1 = metrics.score_run(runs["dense_v1"], qrels)
        headroom = metrics.headroom_gate(stella, v1, query_specs, registry)
        evaluation = None
        if headroom["pass"]:
            evaluation = metrics.evaluate_frozen(
                runs, qrels, query_specs, registry["development_eligibility"], support)
        result = {"_schema": "m19-development-evaluation-v2", "evaluation": evaluation,
                  "headroom": headroom, "eligible": bool(
                      evaluation and evaluation["eligibility"]["eligible"]),
                  "input_sha256": {role: binding["sha256"]
                                   for role, binding in current["bound_files"].items()}}
        result_path = self.root / "evaluation.json"
        _publish(result_path, _json_bytes(result))
        return self._transition("qrels-frozen", "scored",
                                added={"evaluation": _binding(result_path)},
                                extra={"eligible": result["eligible"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "freeze-pool", "begin-judgments",
                                           "freeze-qrels", "score"))
    parser.add_argument("--implementation-review")
    parser.add_argument("--astra-review")
    parser.add_argument("--review-scope")
    parser.add_argument("--primary-labels")
    parser.add_argument("--audit-labels")
    parser.add_argument("--adjudications")
    parser.add_argument("--supporting-passages")
    parser.add_argument("--primary-reviewer-id")
    parser.add_argument("--auditor-id")
    parser.add_argument("--clarification")
    args = parser.parse_args(argv)
    tx = DevelopmentTransaction()
    if args.action == "initialize":
        result = tx.initialize()
    elif args.action == "freeze-pool":
        result = tx.freeze_pool()
    elif args.action == "begin-judgments":
        result = tx.begin_judgments(
            args.implementation_review, args.astra_review, args.review_scope)
    elif args.action == "freeze-qrels":
        result = tx.freeze_qrels(
            args.primary_labels, args.audit_labels, args.adjudications,
            args.supporting_passages, primary_reviewer_id=args.primary_reviewer_id,
            auditor_id=args.auditor_id, clarification_path=args.clarification)
    else:
        result = tx.score()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
