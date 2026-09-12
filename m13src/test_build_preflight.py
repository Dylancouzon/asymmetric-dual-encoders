"""Synthetic DEV-6 cache checks; no experiment data, models, network or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_preflight", Path(__file__).parents[1] / "scripts/m13_build_preflight.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def fixture_cache(tmp_path):
    teacher = SimpleNamespace(ENC=tmp_path, SHARD=1, TEACHER="synthetic", TEACHER_REV="rev",
                              PROVENANCE={}, reads=0)
    teacher.sha_texts = lambda texts: hashlib.sha256(json.dumps(texts).encode()).hexdigest()
    teacher.cache_key = lambda name, *args: (name, "synthetic metadata")

    def unexpected(*args, **kwargs):
        pytest.fail("preflight must never encode or load a teacher")

    teacher.encode = teacher.load_teacher = unexpected
    for name in P.TEXT_CACHES:
        directory = tmp_path / name
        directory.mkdir()
        parts = [np.full((1, 1024), i, dtype=np.float16) for i in range(2)]
        manifest = {"shards": {}}
        for i, values in enumerate(parts):
            part = directory / f"shard_{i:05d}.npy"
            np.save(part, values)
            manifest["shards"][f"{i:05d}"] = {"sha256": P.sha(part), "trusted_on_first_use": True}
        combined = directory / "combined.f16"
        combined.write_bytes(np.concatenate(parts).tobytes())
        manifest["combined"] = {"sha256": P.sha(combined), "n_rows": 2,
                                "from_shard_sha256": [row["sha256"] for row in manifest["shards"].values()],
                                "trusted_on_first_use": True}
        (directory / "shards.json").write_text(json.dumps(manifest))

    def reader(name, texts, **kwargs):
        assert kwargs["verify"] is False
        teacher.reads += 1
        teacher.PROVENANCE[name] = {"shards_written_now": 0}
        return np.memmap(tmp_path / name / "combined.f16", mode="r", dtype=np.float16, shape=(2, 1024))

    teacher.encode_cached = reader

    def doc_vecs(component, incumbent):
        vectors = (teacher.encode_cached("dev-" + component + "-docs", ["a", "b"], dtype="fp16")
                   if component in P.DEV6[:4] else np.zeros((2, 1024), dtype=np.float16))
        return ["a", "b"], ["q"], ["synthetic query"], {}, vectors

    evaluator = SimpleNamespace(components=lambda surface: P.DEV6,
                                doc_vecs=doc_vecs, INCUMBENT="synthetic")
    return teacher, evaluator


def test_actual_cache_reader_reused_without_upgrading_legacy_provenance(tmp_path):
    teacher, evaluator = fixture_cache(tmp_path)
    prior = {name: (tmp_path / name / "shards.json").read_bytes() for name in P.TEXT_CACHES}
    original = teacher.encode_cached
    result = P.check_dev6(teacher, evaluator)
    assert teacher.reads == 4 and teacher.encode_cached is original
    assert set(result["components"]) == set(P.DEV6)
    assert all(row["legacy_shards"] == 2 and row["legacy_combined"] for row in result["cache_checks"])
    assert all((tmp_path / name / "shards.json").read_bytes() == data for name, data in prior.items())


@pytest.mark.parametrize("failure", ["missing", "corrupt_stitch", "missing_inventory"])
def test_bad_cache_refuses_before_reader_can_repair_it(tmp_path, failure):
    teacher, evaluator = fixture_cache(tmp_path)
    directory = tmp_path / "dev-nq-250k-docs"
    if failure == "missing":
        (directory / "shard_00000.npy").unlink()
    elif failure == "corrupt_stitch":
        combined = directory / "combined.f16"
        original = combined.read_bytes()
        combined.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    else:
        path = directory / "shards.json"
        manifest = json.loads(path.read_text())
        del manifest["shards"]["00001"]
        path.write_text(json.dumps(manifest))
    original_reader, original_encoder = teacher.encode_cached, teacher.encode
    with pytest.raises((RuntimeError, FileNotFoundError)):
        P.check_dev6(teacher, evaluator)
    assert teacher.reads == 0
    assert teacher.encode_cached is original_reader and teacher.encode is original_encoder


def test_encoder_and_model_loading_are_blocked_even_for_heldout_reader(tmp_path):
    teacher, evaluator = fixture_cache(tmp_path)
    evaluator.doc_vecs = lambda *args: teacher.load_teacher()
    with pytest.raises(RuntimeError, match="refuses teacher encoding or model loading"):
        P.check_dev6(teacher, evaluator)
    assert teacher.reads == 0
