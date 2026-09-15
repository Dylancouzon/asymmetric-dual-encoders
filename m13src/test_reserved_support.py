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
    systems = {"nano-dense": base, "bge-small-en-v1.5": shifted(shift_b),
               "leaf-ir-asym": shifted(shift_leaf)}
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
        cache = tmp_path / "work" / "m13-reserved-enc" / system / dataset / "manifest.json"
        cache.parent.mkdir(parents=True)
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
    (tmp_path / "reserved" / f"{systems[0]}.json").write_text(json.dumps(existing))
    scored = []
    validated = []

    def score_system(_cfg, _conf, system, rows):
        assert rows is None
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
        assert json.loads((tmp_path / "reserved" / f"{system}.json").read_text())["system"] == system


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
