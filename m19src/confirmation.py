"""Immutable, resumable M19 one-shot confirmation transaction."""
from __future__ import annotations

import re
from pathlib import Path

from m19src import common

STATES = (
    "locked", "claimed", "pools-frozen", "judgments-in-progress", "qrels-frozen",
    "scored", "complete",
)
STATE_FILES = {state: f"{index:02d}-{state}.json" for index, state in enumerate(STATES)}
REQUIRED_DECISION_KEYS = {
    "_schema", "transaction_id", "candidate_id", "eligible", "bundle_hashes",
    "development_qrels_sha256", "development_results_sha256",
    "development_eligibility_sha256", "term_roster_sha256", "row_formula",
    "pool_recipe", "evidence_recipe", "judgment_recipe", "metric_recipe",
    "numerical_gates", "inheritance_identity", "primary_judge_id", "auditor_id",
    "confirmation_query_path", "confirmation_query_sha256", "review_gos",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


def _is_hash(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def validate_decision(decision):
    missing = REQUIRED_DECISION_KEYS - set(decision)
    if missing:
        raise ValueError(f"decision lock missing {sorted(missing)}")
    if decision["_schema"] != "m19-confirmation-decision-v1":
        raise ValueError("wrong confirmation decision schema")
    if not SAFE_ID.fullmatch(str(decision["transaction_id"])):
        raise ValueError("unsafe transaction ID")
    if decision["candidate_id"] != "T0-teacher" or decision["eligible"] is not True:
        raise ValueError("confirmation is available only to an eligible T0-teacher decision")
    hash_fields = (
        "development_qrels_sha256", "development_results_sha256",
        "development_eligibility_sha256", "term_roster_sha256", "inheritance_identity",
        "confirmation_query_sha256",
    )
    if any(not _is_hash(decision[field]) for field in hash_fields):
        raise ValueError("decision lock contains a malformed required hash")
    if not decision["bundle_hashes"] or any(
            not _is_hash(value) for value in decision["bundle_hashes"].values()):
        raise ValueError("decision lock must bind every selected-bundle file hash")
    if not decision["primary_judge_id"] or not decision["auditor_id"] or (
            decision["primary_judge_id"] == decision["auditor_id"]):
        raise ValueError("decision lock must name independent confirmation reviewers")
    gos = decision["review_gos"]
    if not isinstance(gos, list) or len(gos) != 2:
        raise ValueError("decision lock requires exactly two independent review GOs")
    if {row.get("role") for row in gos} != {"implementation", "astra"}:
        raise ValueError("review GOs must cover implementation and fresh Astra roles")
    if any(row.get("decision") != "GO" or not row.get("reviewer_id") or
           not _is_hash(row.get("findings_sha256")) for row in gos):
        raise ValueError("review GO record is incomplete")
    if len({row["reviewer_id"] for row in gos}) != 2:
        raise ValueError("review GO identities are not independent")
    return decision


class ConfirmationTransaction:
    """Manage a hash-chained transaction without opening unclaimed content."""

    def __init__(self, decision_path, receipt_dir):
        self.decision_path = Path(decision_path)
        self.receipt_dir = Path(receipt_dir)
        self.decision = validate_decision(common.load_json(self.decision_path))
        self.decision_sha256 = common.sha_file(self.decision_path)

    def _receipt_path(self, state):
        return self.receipt_dir / STATE_FILES[state]

    @staticmethod
    def _receipt_bytes(obj):
        import json
        return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()

    def _create_or_resume(self, path, obj):
        payload = self._receipt_bytes(obj)
        try:
            common.atomic_create_bytes(path, payload)
        except FileExistsError:
            if common.load_json(path) != obj:
                raise SystemExit(f"M19 CONFIRMATION STOP: immutable receipt differs: {path}")
        return obj

    def initialize(self):
        obj = {
            "_schema": "m19-confirmation-state-v1", "state": "locked",
            "transaction_id": self.decision["transaction_id"],
            "decision_sha256": self.decision_sha256, "previous_receipt_sha256": None,
            "bound_files": {}, "outcome": None,
        }
        return self._create_or_resume(self._receipt_path("locked"), obj)

    def current(self):
        prior = None
        current = None
        seen_gap = False
        for state in STATES:
            path = self._receipt_path(state)
            if not path.exists():
                seen_gap = True
                continue
            terminal_failure = (
                seen_gap and state == "complete" and prior is not None
            )
            if seen_gap and not terminal_failure:
                raise SystemExit("M19 CONFIRMATION STOP: state receipt sequence has a gap")
            row = common.load_json(path)
            if terminal_failure and (
                    row.get("outcome") != "consumed-incomplete" or
                    row.get("failed_from_state") != prior):
                raise SystemExit("M19 CONFIRMATION STOP: invalid incomplete terminal receipt")
            expected_previous = None if prior is None else common.sha_file(self._receipt_path(prior))
            if (row.get("state") != state or
                    row.get("transaction_id") != self.decision["transaction_id"] or
                    row.get("decision_sha256") != self.decision_sha256 or
                    row.get("previous_receipt_sha256") != expected_previous):
                raise SystemExit(f"M19 CONFIRMATION STOP: invalid state receipt {path}")
            if current is not None and row.get("bound_files", {}) != {
                    **current.get("bound_files", {}), **row.get("added_files", {})}:
                raise SystemExit("M19 CONFIRMATION STOP: bound-file ledger is not cumulative")
            prior, current = state, row
        if current is None:
            raise SystemExit("M19 CONFIRMATION STOP: transaction is not initialized")
        return current

    @staticmethod
    def _confirmation_path(path):
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(common.CONFIRMATION_WORK.resolve())
        except ValueError as exc:
            raise SystemExit(
                f"M19 CONFIRMATION STOP: {resolved} is outside the sealed namespace"
            ) from exc
        return resolved

    def _hash_confirmation_file(self, path):
        path = self._confirmation_path(path)
        if not path.is_file():
            raise SystemExit(f"M19 CONFIRMATION STOP: missing checkpoint file {path}")
        return str(path), common.sha_file_unchecked(path)

    def _transition(self, expected, target, *, added_files=None, extra=None):
        current = self.current()
        if current["state"] == target:
            return current
        if current["state"] != expected:
            raise SystemExit(
                f"M19 CONFIRMATION STOP: cannot transition {current['state']} to {target}"
            )
        added_files = dict(sorted((added_files or {}).items()))
        overlap = set(current["bound_files"]) & set(added_files)
        if overlap:
            raise SystemExit(f"M19 CONFIRMATION STOP: files rebound at a later state: {sorted(overlap)}")
        obj = {
            "_schema": "m19-confirmation-state-v1", "state": target,
            "transaction_id": self.decision["transaction_id"],
            "decision_sha256": self.decision_sha256,
            "previous_receipt_sha256": common.sha_file(self._receipt_path(expected)),
            "added_files": added_files,
            "bound_files": {**current["bound_files"], **added_files}, "outcome": None,
        }
        obj.update(extra or {})
        return self._create_or_resume(self._receipt_path(target), obj)

    def claim(self):
        current = self.current()
        if current["state"] == "claimed":
            return current
        if current["state"] != "locked":
            raise SystemExit("M19 CONFIRMATION STOP: claim is not the next transition")
        path, actual = self._hash_confirmation_file(self.decision["confirmation_query_path"])
        if actual != self.decision["confirmation_query_sha256"]:
            raise SystemExit("M19 CONFIRMATION STOP: sealed query hash differs from decision lock")
        # Only a digest is computed above. This receipt is durable before content can be returned.
        return self._transition("locked", "claimed", added_files={path: actual})

    def read_bound_bytes(self, path):
        current = self.current()
        admitted = common.admit_confirmation_read(
            path, state=current["state"], claimed_files=current["bound_files"]
        )
        return admitted.read_bytes()

    def freeze_pools(self, files):
        current = self.current()
        if current["state"] == "pools-frozen":
            return current
        if current["state"] != "claimed":
            raise SystemExit("M19 CONFIRMATION STOP: pools require the claimed state")
        bound = dict(self._hash_confirmation_file(path) for path in files.values())
        return self._transition("claimed", "pools-frozen", added_files=bound,
                                extra={"pool_roles": {k: str(v) for k, v in sorted(files.items())}})

    def begin_judgments(self):
        return self._transition("pools-frozen", "judgments-in-progress")

    def checkpoint_judgment_batch(self, batch_id, path):
        current = self.current()
        if current["state"] != "judgments-in-progress":
            raise SystemExit("M19 CONFIRMATION STOP: judgment batch outside judgment state")
        if not SAFE_ID.fullmatch(str(batch_id)):
            raise ValueError("unsafe judgment batch ID")
        resolved, digest = self._hash_confirmation_file(path)
        obj = {
            "_schema": "m19-confirmation-judgment-batch-v1",
            "transaction_id": self.decision["transaction_id"],
            "decision_sha256": self.decision_sha256,
            "judgments_state_sha256": common.sha_file(self._receipt_path("judgments-in-progress")),
            "batch_id": str(batch_id), "path": resolved, "sha256": digest,
        }
        return self._create_or_resume(self.receipt_dir / f"batch-{batch_id}.json", obj)

    def _batch_receipt(self, batch_id):
        if not SAFE_ID.fullmatch(str(batch_id)):
            raise ValueError("unsafe judgment batch ID")
        row = common.load_json(self.receipt_dir / f"batch-{batch_id}.json")
        if (row.get("transaction_id") != self.decision["transaction_id"] or
                row.get("decision_sha256") != self.decision_sha256 or
                row.get("judgments_state_sha256") != common.sha_file(
                    self._receipt_path("judgments-in-progress"))):
            raise SystemExit("M19 CONFIRMATION STOP: invalid judgment batch receipt")
        path, digest = self._hash_confirmation_file(row["path"])
        if path != row["path"] or digest != row["sha256"]:
            raise SystemExit("M19 CONFIRMATION STOP: judgment batch bytes changed")
        return row

    def freeze_qrels(self, files, *, batch_ids):
        current = self.current()
        if current["state"] == "qrels-frozen":
            return current
        if current["state"] != "judgments-in-progress":
            raise SystemExit("M19 CONFIRMATION STOP: qrels require judgments in progress")
        batches = [self._batch_receipt(batch_id) for batch_id in batch_ids]
        if len(batches) != len(set(row["batch_id"] for row in batches)) or not batches:
            raise SystemExit("M19 CONFIRMATION STOP: qrels need unique checkpointed batches")
        bound = dict(self._hash_confirmation_file(path) for path in files.values())
        bound.update({row["path"]: row["sha256"] for row in batches})
        return self._transition(
            "judgments-in-progress", "qrels-frozen", added_files=bound,
            extra={
                "qrel_roles": {k: str(v) for k, v in sorted(files.items())},
                "judgment_batch_receipts": {
                    row["batch_id"]: common.sha_file(
                        self.receipt_dir / f"batch-{row['batch_id']}.json"
                    ) for row in batches
                },
            },
        )

    def score(self, metrics_path):
        current = self.current()
        if current["state"] == "scored":
            return current
        if current["state"] != "qrels-frozen":
            raise SystemExit("M19 CONFIRMATION STOP: metrics cannot be exposed before qrels freeze")
        path, digest = self._hash_confirmation_file(metrics_path)
        return self._transition("qrels-frozen", "scored", added_files={path: digest})

    def complete(self, *, outcome):
        if outcome not in {"pass", "fail", "inconclusive"}:
            raise ValueError("invalid scored confirmation outcome")
        return self._transition("scored", "complete", extra={"outcome": outcome})

    def mark_incomplete(self, *, reason):
        current = self.current()
        if STATES.index(current["state"]) < STATES.index("claimed"):
            raise SystemExit("M19 CONFIRMATION STOP: an unclaimed transaction is not consumed")
        if current["state"] == "complete":
            return current
        obj = {
            "_schema": "m19-confirmation-state-v1", "state": "complete",
            "transaction_id": self.decision["transaction_id"],
            "decision_sha256": self.decision_sha256,
            "previous_receipt_sha256": common.sha_file(self._receipt_path(current["state"])),
            "added_files": {}, "bound_files": current["bound_files"],
            "outcome": "consumed-incomplete", "failure_reason": str(reason),
            "failed_from_state": current["state"],
        }
        return self._create_or_resume(self._receipt_path("complete"), obj)

    def reconcile(self):
        current = self.current()
        for path, expected in current["bound_files"].items():
            resolved, actual = self._hash_confirmation_file(path)
            if resolved != path or actual != expected:
                raise SystemExit(f"M19 CONFIRMATION STOP: bound file changed: {path}")
        return {
            "transaction_id": self.decision["transaction_id"],
            "decision_sha256": self.decision_sha256, "state": current["state"],
            "outcome": current["outcome"], "bound_file_count": len(current["bound_files"]),
            "final_receipt_sha256": common.sha_file(self._receipt_path(current["state"])),
        }
