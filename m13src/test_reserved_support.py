import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import reserved_support as R


def test_nano_dependency_identity_is_derived_and_registered(tmp_path):
    class Backend:
        @staticmethod
        def to_str():
            return '{"tokenizer":"pinned"}'

    class Tokenizer:
        backend_tokenizer = Backend()

    class Config:
        @staticmethod
        def to_json_string():
            return '{"hidden_size":384}'

    class Backbone:
        config = Config()

    class Model:
        key = "bge-small"
        tok = Tokenizer()
        backbone = Backbone()

    got = R.nano_dependency_identity(Model())
    h = hashlib.sha256()
    for value in ("BAAI/bge-small-en-v1.5", Backend.to_str(), Config.to_json_string()):
        h.update(value.encode())
    assert got == {"repo": "BAAI/bge-small-en-v1.5", "sha256": h.hexdigest()}

    (tmp_path / "m13").mkdir()
    expected = {"repo": got["repo"], "sha256": got["sha256"]}
    (tmp_path / "m13" / "LOTTE_GATE_MANIFEST.json").write_text(json.dumps(
        {"candidate": {"dependencies": expected}}))
    assert R.registered_nano_dependency(tmp_path) == expected


def test_sharded_vectors_crosses_boundaries(tmp_path):
    values = np.arange(35, dtype=np.float16).reshape(7, 5)
    shards = {}
    for i, part in enumerate((values[:3], values[3:6], values[6:])):
        path = tmp_path / f"shard_{i:05d}.npy"
        np.save(path, part)
        shards[f"{i:05d}"] = {"bytes": path.stat().st_size, "sha256": R.sha_file(path)}
    manifest = {"status": "COMPLETE", "n_docs": 7, "dim": 5, "shard_rows": 3,
                "shards": shards}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    got = R.ShardedVectors(path)
    assert got.shape == values.shape
    np.testing.assert_array_equal(got[2:7], values[2:7])


def _outputs(shift_b=.1, shift_leaf=.2):
    base = {dataset: {f"q{i}": .4 + i / 10 for i in range(1, 5)} for dataset in R.DATASETS}

    def shifted(amount):
        return {dataset: {qid: value - amount for qid, value in row.items()}
                for dataset, row in base.items()}

    # Every registered system must be present for `summarize` to run; only the two contrast pairs
    # carry meaning, and the other five deliberately sit at arbitrary offsets to prove they do not
    # move R1 or R2.
    systems = {"nano-dense": base, "bge-small-en-v1.5": shifted(shift_b),
               "leaf-ir-asym": shifted(shift_leaf)}
    for i, other in enumerate(s for s in R.SYSTEMS if s not in systems):
        systems[other] = shifted(0.03 * (i + 1))
    return {system: {"status": "COMPLETE", "datasets": {
        dataset: {"scores": row, "mean_ndcg10": float(np.mean(list(row.values())))}
        for dataset, row in datasets.items()}} for system, datasets in systems.items()}


def _conf():
    return {"reserved": {"B": 100, "seed": 902,
                         "ndo3_weights": {"dbpedia": .5, "cqadup-android": .25,
                                          "cqadup-english": .25}}}


def test_summary_known_deltas_and_zero_alpha_scope():
    got = R.summarize(_outputs(), _conf())
    assert got["status"] == "complete"
    assert got["roster"] == list(R.SYSTEMS) and len(R.SYSTEMS) == 8
    assert set(got["systems"]) == set(R.SYSTEMS)
    assert set(got["contrasts"]) == {"R1", "R2"}
    assert got["contrasts"]["R1"]["ndo3"]["delta_raw"] == pytest.approx(.1)
    assert got["contrasts"]["R2"]["ndo3"]["delta_raw"] == pytest.approx(.2)
    assert got["contrasts"]["R1"]["query_pooled"]["delta_raw"] == pytest.approx(.1)
    assert len(got["contrasts"]["R1"]["query_pooled"]["ci95_raw"]) == 2
    assert "zero alpha" in got["scope"]
    assert got["contrasts"]["R1"]["per_dataset"]["fever"]["classification"].startswith(
        "double-contaminated")


def test_summary_refuses_mismatched_queries():
    outputs = _outputs()
    del outputs["leaf-ir-asym"]["datasets"]["fever"]["scores"]["q1"]
    with pytest.raises(ValueError, match="different query ids"):
        R.summarize(outputs, _conf())


def test_saved_output_is_bound_to_transaction_payload_and_cache(tmp_path):
    import access13 as A

    identity = {"begin_commit": "abc", "code_sha256": "def", "spent_tag": "spent"}
    manifest = {"m7_untouched_final": {}}
    datasets = {}
    system = "bge-small-en-v1.5"
    for dataset in R.DATASETS:
        scores = {"q1": .2, "q2": .6}
        payload = {"qids_sha256": A.sha_json(sorted(scores)),
                   "qtexts_sha256": f"texts-{dataset}",
                   "qrels_sha256": f"qrels-{dataset}"}
        manifest["m7_untouched_final"][dataset] = payload
        cache = (tmp_path / "work" / "m13-reserved-enc" / R.cache_dir_for(system)
                 / dataset / "manifest.json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"dataset": dataset}))
        datasets[dataset] = {
            "scores": scores, "mean_ndcg10": .4, "n_queries": 2,
            "payload_hashes": payload,
            "document_cache_manifest": str(cache.relative_to(tmp_path)),
            "document_cache_manifest_sha256": R.sha_file(cache),
        }
    manifest_path = tmp_path / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    cfg = SimpleNamespace(repo=tmp_path, manifest_path=manifest_path,
                          extra={"reserved_identity": identity})
    record = {"status": "COMPLETE", "system": system, "datasets": datasets,
              "document_encoder": R.DOCUMENT_ENCODERS[system],
              "document_compute_dtype": "fp32 on CUDA; normalized vectors stored fp16",
              "query_compute_dtype": "fp32", "transaction": identity}
    assert R.validate_output(record, cfg, system) is record
    record["datasets"]["fever"]["scores"]["q2"] = .7
    with pytest.raises(ValueError, match="count or mean"):
        R.validate_output(record, cfg, system)


def test_score13_production_reserved_resumes_validated_system_files(tmp_path, monkeypatch):
    import score13 as S

    systems = list(R.SYSTEMS)
    conf = {"reserved": {"systems_included": systems}}
    cfg = SimpleNamespace(scores_dir=tmp_path, extra={"reserved_production": True})
    existing = {"status": "COMPLETE", "system": systems[0]}
    (tmp_path / "reserved").mkdir()
    (tmp_path / "reserved" / f"{R.R20.slug(systems[0])}.json").write_text(json.dumps(existing))
    scored = []
    validated = []

    def score_system(_cfg, _conf, system, rows, outputs=None):
        assert rows is None
        # A derived row may only be produced after both of its inputs have been accepted.
        for needed in R.DERIVED_SYSTEMS.get(system, ()):
            assert needed in outputs, f"{system} scored before {needed}"
        scored.append(system)
        return {"status": "COMPLETE", "system": system}

    def validate(record, _cfg, system):
        assert record["system"] == system
        validated.append(system)
        return record

    monkeypatch.setattr(R, "score_system", score_system)
    monkeypatch.setattr(R, "validate_output", validate)
    monkeypatch.setattr(R, "summarize", lambda outputs, _conf: {
        "status": "complete", "systems": sorted(outputs)})

    got = S.reserved_batch(cfg, conf, None)
    assert got == {"status": "complete", "systems": sorted(systems)}
    assert scored == systems[1:]
    assert validated == systems
    for system in systems:
        path = tmp_path / "reserved" / f"{R.R20.slug(system)}.json"
        assert json.loads(path.read_text())["system"] == system


def test_reserved_only_claims_before_transaction(monkeypatch):
    import paths_guard
    import reserved_transaction
    import score13 as S

    events = []
    cfg = object()
    monkeypatch.setattr(paths_guard, "claim", lambda entry, note="": events.append(("claim", entry)))
    monkeypatch.setattr(paths_guard, "install", lambda: events.append(("install", None)))
    monkeypatch.setattr(reserved_transaction, "run",
                        lambda received: events.append(("run", received)) or 17)
    assert S.reserved_only_run(cfg) == 17
    assert events == [("claim", "m13src.score13"), ("install", None), ("run", cfg)]


# ---------------------------------------------------------------- M20 roster extension

def test_three_systems_share_one_stella_document_index():
    """The premise of the project: Nano, Zero and the Stella query tower read the SAME shards, so
    adding them authorizes no second corpus-scale encode."""
    stella = {R.cache_dir_for(s) for s in ("nano-dense", "zero-dense", "stella-query")}
    assert stella == {"nano-dense"}
    assert R.cache_dir_for("bge-small-en-v1.5") == "bge-small-en-v1.5"
    assert R.cache_dir_for("leaf-ir-asym") == "leaf-ir-asym"
    assert len({R.cache_dir_for(s) for s in R.DENSE_SYSTEMS}) == 3


def test_executor_and_pushed_registration_agree():
    assert R.R20.assert_registered_identities() is True


def test_registration_drift_is_caught(tmp_path, monkeypatch):
    registry = json.loads((Path(R.REPO) / "m20" / "beir15_registry.json").read_text())
    for row in registry["systems"]:
        if row["key"] == "stella-query":
            row["query_prompt"] = "Query: "
    (tmp_path / "m20").mkdir()
    (tmp_path / "m20" / "beir15_registry.json").write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="Stella query prompt"):
        R.R20.assert_registered_identities(tmp_path)


def test_dbsf_truncates_each_prefetch_before_fusing():
    """Depth-100 truncation happens BEFORE fusion: DBSF normalizes by the prefetch's own mean and
    sample sd, so fusing first and cutting after is a different function."""
    import qfusion

    dense = {"q": {f"d{i}": 1.0 - i / 1000 for i in range(300)}}
    lexical = {"q": {f"d{i}": float(300 - i) for i in range(300)}}
    got = R.R20.dbsf_at_depth(dense, lexical, depth=100)
    want = qfusion.dbsf([qfusion.truncate(dense, 100), qfusion.truncate(lexical, 100)])
    assert got == want
    wrong = qfusion.truncate(qfusion.dbsf([dense, lexical]), 100)
    assert got != wrong


def test_persisted_run_round_trips_and_is_hash_bound(tmp_path):
    # Scores are persisted fp32, which is what every producer already emits: bm25s returns
    # float32 and the dense top-k comes from fp16 blocks promoted to fp32. The derived rows fuse
    # exactly these bytes, so the persisted run IS the registered fusion input.
    run = {"q1": {"d1": 0.75, "d2": 0.5}, "q2": {}, "q3": {"dz": 0.25}}
    path = tmp_path / "r.npz"
    digest = R.R20.save_run(path, run, ["q1", "q2", "q3"])
    back, qids = R.R20.load_run(path, digest)
    assert qids == ["q1", "q2", "q3"] and back == run
    import numpy as _np
    wide = R.R20.save_run(tmp_path / "w.npz", {"q": {"d": 0.9}}, ["q"])
    got, _ = R.R20.load_run(tmp_path / "w.npz", wide)
    assert got["q"]["d"] == float(_np.float32(0.9))
    with pytest.raises(ValueError, match="hash changed"):
        R.R20.load_run(path, "0" * 64)


def test_persisted_run_refuses_more_than_the_registered_depth(tmp_path):
    deep = {"q": {f"d{i}": float(i) for i in range(R.R20.FUSION_DEPTH + 1)}}
    with pytest.raises(ValueError, match="deeper than"):
        R.R20.save_run(tmp_path / "r.npz", deep, ["q"])


def test_registry_amendment_reverses_to_the_six_set_bytes():
    """The dated M20 amendment must be provably confined to the roster: reversing it has to
    reproduce the exact registry the frozen six-set result pinned."""
    import reserved_transaction as T

    undone = T.registry_without_m20_amendment(Path(R.REPO) / "m10" / "final_run_registry.json")
    pinned = json.loads((Path(R.REPO) / "results" / "m10_final_run.json").read_text())
    assert T._sha_bytes(undone) == pinned["registry_sha256"]


def test_registry_amendment_reversal_catches_an_extra_edit(tmp_path):
    import reserved_transaction as T

    live = json.loads((Path(R.REPO) / "m10" / "final_run_registry.json").read_text())
    live["reserved"]["seed"] = 903
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(live, indent=1) + "\n")
    pinned = json.loads((Path(R.REPO) / "results" / "m10_final_run.json").read_text())
    assert T._sha_bytes(T.registry_without_m20_amendment(path)) != pinned["registry_sha256"]


def test_validate_output_requires_the_persisted_run_for_a_fusion_input(tmp_path):
    import access13 as A

    identity = {"begin_commit": "abc", "code_sha256": "def", "spent_tag": "spent"}
    manifest = {"m7_untouched_final": {}}
    datasets = {}
    system = "zero-dense"
    for dataset in R.DATASETS:
        scores = {"q1": .2, "q2": .6}
        payload = {"qids_sha256": A.sha_json(sorted(scores)),
                   "qtexts_sha256": f"texts-{dataset}", "qrels_sha256": f"qrels-{dataset}"}
        manifest["m7_untouched_final"][dataset] = payload
        cache = (tmp_path / "work" / "m13-reserved-enc" / R.cache_dir_for(system)
                 / dataset / "manifest.json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"dataset": dataset}))
        run_path = tmp_path / "work" / "m20-runs" / "reserved" / system / f"{dataset}.npz"
        run_sha = R.R20.save_run(run_path, {"q1": {"d": .5}, "q2": {}}, ["q1", "q2"])
        datasets[dataset] = {
            "scores": scores, "mean_ndcg10": .4, "n_queries": 2, "payload_hashes": payload,
            "document_cache_manifest": str(cache.relative_to(tmp_path)),
            "document_cache_manifest_sha256": R.sha_file(cache),
            "run_path": str(run_path.relative_to(tmp_path)), "run_sha256": run_sha,
        }
    manifest_path = tmp_path / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    cfg = SimpleNamespace(repo=tmp_path, manifest_path=manifest_path,
                          extra={"reserved_identity": identity})
    record = {"status": "COMPLETE", "system": system, "datasets": datasets,
              "document_encoder": R.DOCUMENT_ENCODERS[system],
              "document_compute_dtype": R.R20.DOC_COMPUTE_NOTE,
              "query_compute_dtype": "fp32", "transaction": identity}
    assert R.validate_output(record, cfg, system) is record
    datasets["fever"]["run_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="persisted top-100 run"):
        R.validate_output(record, cfg, system)


def test_reserved_support_never_imports_the_corpus_only_entry_point():
    """`paths_guard.claim` refuses a second, different claim in one process. The tagged
    transaction holds `m13src.score13`, so importing `m8src/pre_encode.py` -- which claims the
    corpus-only entry at import time -- would raise inside the transaction, AFTER the reserved
    access had been spent. BM25 needs corpus text, so this is a live path, not a hypothetical."""
    source = (Path(R.REPO) / "m13src" / "reserved_support.py").read_text()
    offenders = [line.strip() for line in source.splitlines()
                 if "pre_encode" in line and not line.lstrip().startswith("#")
                 and "deliberately does NOT import" not in line]
    assert not offenders, offenders


def test_only_one_allowlist_claim_is_reachable_from_the_transaction(monkeypatch):
    import paths_guard

    monkeypatch.setattr(paths_guard, "_claim", "m13src.score13", raising=False)
    with pytest.raises(paths_guard.ProtectedPathRefusal, match="already claimed"):
        import pre_encode  # noqa: F401  -- claims m8src.pre_encode at import time
