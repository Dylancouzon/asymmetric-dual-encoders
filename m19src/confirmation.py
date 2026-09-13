"""Immutable, resumable M19 one-shot confirmation transaction."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from m19src import common, judgments, metrics, zero

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
    "bindings", "registry_sections", "mode", "reviewed_commit",
}
BOUND_ROLES = {
    "registry", "inheritance_lock", "roster", "base_model", "base_tokenizer", "base_config",
    "bundle_model", "bundle_config", "bundle_tokenizer", "bundle_provenance",
    "bundle_complete", "development_qrels", "development_runs", "development_queries",
    "development_support", "development_evaluation", "implementation_review", "astra_review",
    "teacher_vectors", "teacher_receipt", "row_receipt", "serving_receipt", "review_scope",
    "development_pool_manifest",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


def _is_hash(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def validate_decision(decision):
    missing = REQUIRED_DECISION_KEYS - set(decision)
    if missing:
        raise ValueError(f"decision lock missing {sorted(missing)}")
    schemas = {"production": "m19-confirmation-decision-v1",
               "synthetic": "m19-confirmation-decision-synthetic-v1"}
    if decision.get("mode") not in schemas or decision["_schema"] != schemas[decision["mode"]]:
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
    if not re.fullmatch(r"[0-9a-f]{40}", str(decision["reviewed_commit"])):
        raise ValueError("reviewed commit must be a full git object ID")
    if set(decision["bindings"]) != BOUND_ROLES:
        raise ValueError(f"decision bindings must have exact roles {sorted(BOUND_ROLES)}")
    for role, binding in decision["bindings"].items():
        if (set(binding) != {"path", "sha256"} or not Path(binding["path"]).is_absolute() or
                not _is_hash(binding["sha256"])):
            raise ValueError(f"malformed decision binding for {role}")
    return decision


class ConfirmationTransaction:
    """Manage a hash-chained transaction without opening unclaimed content."""

    def __init__(self, decision_path, receipt_dir):
        self.decision_path = Path(decision_path)
        self.receipt_dir = Path(receipt_dir)
        self.decision = validate_decision(common.load_json(self.decision_path))
        self.decision_sha256 = common.sha_file(self.decision_path)
        self._authenticate_decision()

    def _bound_path(self, role):
        return Path(self.decision["bindings"][role]["path"])

    def _load_bound_json(self, role):
        return json.loads(common.admit_read(self._bound_path(role)).read_text())

    def _authenticate_decision(self):
        if common.sha_file(self.decision_path) != self.decision_sha256:
            raise SystemExit("M19 CONFIRMATION STOP: decision lock bytes changed")
        for role, binding in self.decision["bindings"].items():
            if common.sha_file(binding["path"]) != binding["sha256"]:
                raise SystemExit(f"M19 CONFIRMATION STOP: decision binding changed: {role}")
        registry = self._load_bound_json("registry")
        expected_sections = {
            key: registry[key] for key in (
                "versions", "candidate", "retrieval", "judgments", "metrics", "numerical_gates",
                "development_eligibility", "confirmation_eligibility", "confirmation_states",
            )
        }
        if self.decision["registry_sections"] != expected_sections:
            raise SystemExit("M19 CONFIRMATION STOP: decision recipes/gates differ from registry")
        inheritance = self._load_bound_json("inheritance_lock")
        roster = self._load_bound_json("roster")
        if (common.sha_json({k: v for k, v in inheritance.items() if k != "identity_sha256"}) !=
                inheritance.get("identity_sha256") or
                common.sha_json({k: v for k, v in roster.items() if k != "identity_sha256"}) !=
                roster.get("identity_sha256")):
            raise SystemExit("M19 CONFIRMATION STOP: lock body identity is invalid")
        if (inheritance.get("identity_sha256") != self.decision["inheritance_identity"] or
                roster.get("identity_sha256") != self.decision["term_roster_sha256"]):
            raise SystemExit("M19 CONFIRMATION STOP: roster/inheritance identity differs")
        if self.decision["mode"] == "production":
            canonical = {
                "registry": common.REGISTRY_PATH.resolve(),
                "inheritance_lock": (common.M19 / "inheritance-lock.json").resolve(),
                "roster": (common.M19 / "term-roster-lock-v3.json").resolve(),
                "base_model": (common.RELEASE_BUNDLE / "model.npz").resolve(),
                "base_tokenizer": (common.RELEASE_BUNDLE / "tokenizer.json").resolve(),
                "base_config": (common.RELEASE_BUNDLE / "config.json").resolve(),
            }
            if any(self._bound_path(role).resolve() != path for role, path in canonical.items()):
                raise SystemExit("M19 CONFIRMATION STOP: production trust root is noncanonical")
        for role, inherited_role in (("base_model", "zero_v1_model"),
                                     ("base_tokenizer", "zero_v1_tokenizer"),
                                     ("base_config", "zero_v1_config")):
            inherited = inheritance["inherited_data"][inherited_role]
            binding = self.decision["bindings"][role]
            if (Path(inherited["path"]).resolve() != self._bound_path(role).resolve() or
                    inherited["sha256"] != binding["sha256"]):
                raise SystemExit("M19 CONFIRMATION STOP: released base binding differs from lock")
        base_model = common.admit_read(self._bound_path("base_model"))
        with np.load(base_model) as archive:
            base_codes = np.asarray(archive["rows_int8"], dtype=np.int8).copy()
            base_scales = np.asarray(archive["int8_scale"], dtype=np.float32).copy()
        base_tokenizer_payload = common.admit_read(self._bound_path("base_tokenizer")).read_bytes()
        bundle = self._bound_path("bundle_complete").parent
        bundle_report = zero.verify_bundle(bundle, verification={
            "base_codes": base_codes, "base_scales": base_scales,
            "base_tokenizer_payload": base_tokenizer_payload, "roster": roster,
            "inheritance_identity": inheritance["identity_sha256"],
            "pooling_identity_sha256": inheritance["identities"]["released_effective_table"][
                "pooling_sha256"
            ],
        })
        if bundle_report["variant"] != "T0-teacher":
            raise SystemExit("M19 CONFIRMATION STOP: selected bundle is not T0-teacher")
        teacher_array = np.load(common.admit_read(self._bound_path("teacher_vectors")))
        terms = [row["term"] for row in roster["terms"]]
        if (teacher_array.shape[0] != len(terms) or teacher_array.dtype != np.float32 or
                not np.isfinite(teacher_array).all() or
                float(np.max(np.abs(np.linalg.norm(teacher_array, axis=1) - 1))) > 1e-6):
            raise SystemExit("M19 CONFIRMATION STOP: teacher vector term count differs")
        teachers = {term: np.asarray(teacher_array[index], dtype=np.float32)
                    for index, term in enumerate(terms)}
        teacher_receipt = self._load_bound_json("teacher_receipt")
        expected_teacher = {"model": registry["inheritance"]["document_encoder"],
                            "query_prefix": registry["candidate"]["query_prefix"],
                            "terms": terms, "dtype": str(teacher_array.dtype),
                            "shape": list(teacher_array.shape),
                            "vectors_sha256": common.sha_array(teacher_array)}
        if (teacher_receipt.get("_schema") != "m19-teacher-vectors-v1" or
                {key: teacher_receipt.get(key) for key in expected_teacher} != expected_teacher):
            raise SystemExit("M19 CONFIRMATION STOP: teacher provenance differs from registry")
        built = zero.construct_added_rows(base_codes, base_scales, base_tokenizer_payload,
                                          roster, teachers)
        expected_codes, expected_scales = zero.compact_table(
            base_codes, base_scales, built["T0-teacher"]
        )
        with np.load(common.admit_read(self._bound_path("bundle_model"))) as archive:
            if (not np.array_equal(archive["rows_int8"], expected_codes) or
                    not np.array_equal(archive["int8_scale"], expected_scales)):
                raise SystemExit("M19 CONFIRMATION STOP: added rows differ from deterministic T0")
        if common.admit_read(self._bound_path("bundle_tokenizer")).read_bytes() != (
                built["tokenizer"].to_str().encode()):
            raise SystemExit("M19 CONFIRMATION STOP: bundle tokenizer is not deterministic extension")
        base_config = self._load_bound_json("base_config")
        algebra = zero.algebra_gates(base_codes, base_scales, built, teachers,
                                     registry=registry, base_config=base_config)
        row_receipt = self._load_bound_json("row_receipt")
        expected_row_receipt = {
            "_schema": "m19-row-receipt-v1", "variant": "T0-teacher",
            "codes_sha256": common.sha_array(expected_codes),
            "scales_sha256": common.sha_array(expected_scales),
            "tokenizer_sha256": common.sha_bytes(built["tokenizer"].to_str().encode()),
            "algebra_receipts_sha256": common.sha_json(built["receipts"]),
            "algebra_gates_sha256": common.sha_json(algebra),
        }
        if row_receipt != expected_row_receipt:
            raise SystemExit("M19 CONFIRMATION STOP: deterministic row receipt differs")
        serving = self._load_bound_json("serving_receipt")
        required_checks = {"loader_parity", "tokenizer_boundaries", "no_match_ranking_parity",
                           "pooling_identity", "resident_memory", "encoder_latency",
                           "end_to_end_latency"}
        if (serving.get("_schema") != "m19-serving-gates-v1" or
                serving.get("variant") != "T0-teacher" or
                serving.get("bundle_identity_sha256") != bundle_report["identity_sha256"] or
                set(serving.get("checks", {})) != required_checks or
                not all(serving["checks"].values())):
            raise SystemExit("M19 CONFIRMATION STOP: serving gate receipt is incomplete or failed")
        measured = serving.get("measurements", {})
        gates = registry["numerical_gates"]
        measure_keys = {"loader_parity_max_abs", "added_row_bytes",
                        "encoder_latency_median_ratio", "encoder_latency_p95_ratio",
                        "encoder_latency_median_additive_ms", "encoder_latency_p95_additive_ms",
                        "end_to_end_latency_median_ratio", "end_to_end_latency_p95_ratio",
                        "temporary_peak_bytes", "rss_high_water_kib"}
        numeric = [value for value in measured.values()
                   if type(value) in (int, float) and not isinstance(value, bool)]
        provenance_keys = {"query_count", "query_sequence_sha256", "host", "threads",
                           "warmup", "repetitions"}
        bench = serving.get("benchmark", {})
        if (set(measured) != measure_keys or len(numeric) != len(measured) or
                not all(np.isfinite(value) and value >= 0 for value in numeric) or
                set(bench) != provenance_keys or bench["query_count"] != gates["latency_queries"] or
                not _is_hash(bench["query_sequence_sha256"]) or
                any(type(serving["checks"][key]) is not bool for key in required_checks) or
                measured["loader_parity_max_abs"] > gates["loader_parity_max_abs"] or
                measured["added_row_bytes"] != gates["added_row_bytes"] or
                max(measured["encoder_latency_median_ratio"], measured["encoder_latency_p95_ratio"]) >
                gates["encoder_latency_ratio_maximum"] or
                max(measured["encoder_latency_median_additive_ms"],
                    measured["encoder_latency_p95_additive_ms"]) >
                gates["encoder_latency_additive_ms_maximum"] or
                max(measured["end_to_end_latency_median_ratio"],
                    measured["end_to_end_latency_p95_ratio"]) >
                gates["end_to_end_latency_ratio_maximum"]):
            raise SystemExit("M19 CONFIRMATION STOP: serving measurements fail registered gates")
        bundle_hashes = {role.removeprefix("bundle_").replace("complete", "complete.json"):
                         binding["sha256"]
                         for role, binding in self.decision["bindings"].items()
                         if role.startswith("bundle_")}
        bundle_hashes = {
            (name if name == "complete.json" else name + ({"model": ".npz"}.get(name, ".json"))): value
            for name, value in bundle_hashes.items()
        }
        if self.decision["bundle_hashes"] != bundle_hashes:
            raise SystemExit("M19 CONFIRMATION STOP: selected bundle hash set differs")
        qrels = self._load_bound_json("development_qrels")
        runs = self._load_bound_json("development_runs")
        dev_pool = self._load_bound_json("development_pool_manifest")
        if (dev_pool.get("_schema") != "m19-artifact-pool-v1" or
                dev_pool.get("routes") != list(judgments.ROUTES) or
                set(dev_pool.get("queries", {})) != set(qrels)):
            raise SystemExit("M19 CONFIRMATION STOP: development pool manifest differs")
        for metric_role, route in judgments.METRIC_ROUTE_ROLES.items():
            for query_id in qrels:
                if list(map(str, runs[metric_role][query_id][:10])) != list(map(
                        str, dev_pool["queries"][query_id]["route_top10"][route])):
                    raise SystemExit("M19 CONFIRMATION STOP: development run differs from route")
        query_rows = self._load_bound_json("development_queries")
        support_rows = self._load_bound_json("development_support")
        support = {(row["query_id"], row["artifact_id"]): row["pass"] for row in support_rows}
        computed = metrics.evaluate_frozen(
            runs, qrels, query_rows, registry["development_eligibility"], support
        )
        evaluation = self._load_bound_json("development_evaluation")
        expected_inputs = {
            role: self.decision["bindings"][role]["sha256"] for role in (
                "development_qrels", "development_runs", "development_queries",
                "development_support", "development_pool_manifest",
            )
        }
        if (evaluation != {"_schema": "m19-development-evaluation-v1",
                           "input_sha256": expected_inputs, "result": computed} or
                not computed["eligibility"]["eligible"]):
            raise SystemExit("M19 CONFIRMATION STOP: development eligibility does not recompute")
        if (self.decision["development_qrels_sha256"] != expected_inputs["development_qrels"] or
                self.decision["development_results_sha256"] !=
                self.decision["bindings"]["development_evaluation"]["sha256"] or
                self.decision["development_eligibility_sha256"] != common.sha_json(computed)):
            raise SystemExit("M19 CONFIRMATION STOP: development identities differ")
        review_roles = {row["role"]: row for row in self.decision["review_gos"]}
        if (review_roles["implementation"]["findings_sha256"] !=
                self.decision["bindings"]["implementation_review"]["sha256"] or
                review_roles["astra"]["findings_sha256"] !=
                self.decision["bindings"]["astra_review"]["sha256"]):
            raise SystemExit("M19 CONFIRMATION STOP: review GO bindings differ")
        for role, binding_role in (("implementation", "implementation_review"),
                                   ("astra", "astra_review")):
            receipt = self._load_bound_json(binding_role)
            declared = review_roles[role]
            if (receipt.get("_schema") != "m19-review-go-v1" or
                    receipt.get("role") != role or receipt.get("decision") != "GO" or
                    receipt.get("reviewer_id") != declared["reviewer_id"] or
                    receipt.get("reviewed_commit") != self.decision["reviewed_commit"] or
                    receipt.get("scope_sha256") !=
                    self.decision["bindings"]["review_scope"]["sha256"]):
                raise SystemExit("M19 CONFIRMATION STOP: bound review is not an authenticated GO")
        candidate = registry["candidate"]
        expected_formula = {"formula": candidate["formula"],
                            "scale_convention": candidate["scale_convention"],
                            "version": registry["versions"]["row_formula"]}
        if (self.decision["row_formula"] != expected_formula or
                self.decision["pool_recipe"] != registry["retrieval"] or
                self.decision["evidence_recipe"] != registry["judgments"] or
                self.decision["judgment_recipe"] != registry["judgments"] or
                self.decision["metric_recipe"] != registry["metrics"] or
                self.decision["numerical_gates"] != registry["numerical_gates"]):
            raise SystemExit("M19 CONFIRMATION STOP: top-level decision recipe differs")

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
        self._authenticate_decision()
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
        if set(files) != {"pool_manifest", "evidence_packet", "metric_runs"}:
            raise SystemExit("M19 CONFIRMATION STOP: pool freeze requires exact registered roles")
        preview = dict(self._hash_confirmation_file(path) for path in files.values())
        def preview_json(path):
            admitted = common.admit_confirmation_read(
                path, state=current["state"],
                claimed_files={**current["bound_files"], **preview},
            )
            return json.loads(admitted.read_text())
        manifest = preview_json(files["pool_manifest"])
        packet = preview_json(files["evidence_packet"])
        metric_runs = preview_json(files["metric_runs"])
        rules = self.decision["registry_sections"]["judgments"]
        judgments.validate_frozen_pool(
            manifest, packet, metric_runs, seed=int(rules["randomization_seed"]),
            cap=int(rules["confirmation_cap"]),
        )
        bound = preview
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

    def _batch_json(self, batch_id):
        row = self._batch_receipt(batch_id)
        current = self.current()
        path = common.admit_confirmation_read(
            row["path"], state=current["state"],
            claimed_files={**current["bound_files"], row["path"]: row["sha256"]},
        )
        return row, json.loads(path.read_text())

    def _create_confirmation_json(self, path, value):
        path = self._confirmation_path(path)
        payload = self._receipt_bytes(value)
        try:
            common.atomic_create_bytes(path, payload)
        except FileExistsError:
            if common.sha_file_unchecked(path) != common.sha_bytes(payload):
                raise SystemExit(f"M19 CONFIRMATION STOP: owned output differs: {path}")
        return path, common.sha_bytes(payload)

    def freeze_qrels(self, output_path, *, packet_path, primary_batch_id, audit_batch_id,
                     supporting_batch_id, adjudication_batch_id=None,
                     clarification_batch_ids=None):
        current = self.current()
        if current["state"] == "qrels-frozen":
            return current
        if current["state"] != "judgments-in-progress":
            raise SystemExit("M19 CONFIRMATION STOP: qrels require judgments in progress")
        if str(self._confirmation_path(packet_path)) not in current["bound_files"]:
            raise SystemExit("M19 CONFIRMATION STOP: qrels packet is not pool-bound")
        packet = json.loads(self.read_bound_bytes(packet_path))
        query_rows = [json.loads(line) for line in self.read_bound_bytes(
            self.decision["confirmation_query_path"]
        ).splitlines() if line]
        query_ids = [row.get("query_id") for row in query_rows]
        if (len(query_ids) != len(set(query_ids)) or any(
                not isinstance(row.get("author_id"), str) or not row["author_id"]
                for row in query_rows)):
            raise SystemExit("M19 CONFIRMATION STOP: sealed query authors/IDs are invalid")
        query_authors = {row["query_id"]: row["author_id"] for row in query_rows}
        if set(query_authors) != set(packet["queries"] if "queries" in packet else
                                     {row["query_id"] for row in packet["items"]}):
            raise SystemExit("M19 CONFIRMATION STOP: sealed queries and packet differ")
        primary_row, primary = self._batch_json(primary_batch_id)
        audit_row, audit_payload = self._batch_json(audit_batch_id)
        support_row, _ = self._batch_json(supporting_batch_id)
        batches = [primary_row, audit_row, support_row]
        adjudications = {}
        if adjudication_batch_id is not None:
            adjudication_row, adjudications = self._batch_json(adjudication_batch_id)
            batches.append(adjudication_row)
        clarification = None
        if clarification_batch_ids is not None:
            if set(clarification_batch_ids) != {"prior_primary", "rubric"}:
                raise SystemExit("M19 CONFIRMATION STOP: clarification batch roles differ")
            prior_row = self._batch_receipt(clarification_batch_ids["prior_primary"])
            rubric_row = self._batch_receipt(clarification_batch_ids["rubric"])
            batches.extend([prior_row, rubric_row])
            clarification = {"generation": 1, "complete_pool_relabel": True,
                             "prior_primary_sha256": prior_row["sha256"],
                             "rubric_sha256": rubric_row["sha256"]}
        if len(batches) != len(set(row["batch_id"] for row in batches)):
            raise SystemExit("M19 CONFIRMATION STOP: qrels need unique checkpointed batches")
        rules = self.decision["registry_sections"]["judgments"]
        frozen = judgments.freeze_binary_labels(
            packet, primary, audit_payload["sample"], audit_payload["labels"], adjudications,
            primary_reviewer_id=self.decision["primary_judge_id"],
            auditor_id=self.decision["auditor_id"], query_authors=query_authors,
            seed=int(rules["randomization_seed"]),
            fraction=float(rules["audit_fraction_minimum"]),
            minimum_agreement=float(rules["audit_exact_agreement_minimum"]),
            clarification=clarification,
        )
        output_path, output_hash = self._create_confirmation_json(output_path, frozen)
        bound = {str(output_path): output_hash}
        bound.update({row["path"]: row["sha256"] for row in batches})
        return self._transition(
            "judgments-in-progress", "qrels-frozen", added_files=bound,
            extra={
                "qrel_roles": {"frozen_judgments": str(output_path),
                               "supporting_passages": support_row["path"]},
                "judgment_batch_receipts": {
                    row["batch_id"]: common.sha_file(
                        self.receipt_dir / f"batch-{row['batch_id']}.json"
                    ) for row in batches
                },
            },
        )

    def score(self, output_path):
        current = self.current()
        if current["state"] == "scored":
            return current
        if current["state"] != "qrels-frozen":
            raise SystemExit("M19 CONFIRMATION STOP: metrics cannot be exposed before qrels freeze")
        frozen = json.loads(self.read_bound_bytes(current["qrel_roles"]["frozen_judgments"]))
        pool_receipt = common.load_json(self._receipt_path("pools-frozen"))
        runs_path = pool_receipt["pool_roles"]["metric_runs"]
        runs = json.loads(self.read_bound_bytes(runs_path))
        queries = [json.loads(line) for line in self.read_bound_bytes(
            self.decision["confirmation_query_path"]
        ).splitlines() if line]
        support_rows = json.loads(self.read_bound_bytes(
            current["qrel_roles"]["supporting_passages"]
        ))
        support = {(row["query_id"], row["artifact_id"]): row["pass"] for row in support_rows}
        result = metrics.evaluate_frozen(
            runs, frozen["qrels"], queries,
            self.decision["registry_sections"]["confirmation_eligibility"], support,
        )
        output_path, digest = self._create_confirmation_json(output_path, result)
        return self._transition("qrels-frozen", "scored",
                                added_files={str(output_path): digest},
                                extra={"eligible": result["eligibility"]["eligible"]})

    def complete(self):
        current = self.current()
        if current["state"] == "complete":
            return current
        if current["state"] != "scored":
            raise SystemExit("M19 CONFIRMATION STOP: completion requires transaction scoring")
        outcome = "pass" if current["eligible"] else "inconclusive"
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
        self._authenticate_decision()
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
