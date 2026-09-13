import copy
import json

import numpy as np
import pytest

from m19src import inherit
from m19src.common import sha_array, sha_json


def test_effective_row_hash_is_chunk_invariant_and_canonical():
    codes = np.array([[1, -2], [3, 4], [-5, 6]], dtype=np.int8)
    scales = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    effective = codes.astype(np.float32) * scales[:, None]
    assert inherit._effective_rows_hash(codes, scales, chunk_rows=1) == sha_array(effective)
    assert inherit._effective_rows_hash(codes, scales, chunk_rows=2) == sha_array(effective)


@pytest.mark.parametrize("section", [
    "term_roster", "candidate", "retrieval", "query_protocol", "judgments", "metrics",
    "headroom_gate", "numerical_gates", "development_eligibility", "confirmation_eligibility",
])
def test_each_consequential_registry_section_changes_recipe_identity(section):
    recipe = {
        "term_roster": {"minimum": 8},
        "candidate": {"formula": "fixed"},
        "retrieval": {"passage_depth": 500},
        "query_protocol": {"short_max": 5},
        "judgments": {"audit": 0.2},
        "metrics": {"primary": "P@10"},
        "headroom_gate": {"delta": 0.05},
        "numerical_gates": {"cosine": 0.999},
        "development_eligibility": {"delta": 0.03},
        "confirmation_eligibility": {"delta": 0.03},
    }
    changed = copy.deepcopy(recipe)
    changed[section]["mutation"] = True
    assert sha_json(changed) != sha_json(recipe)


def test_control_hash_mismatch_stops_before_data(monkeypatch):
    monkeypatch.setattr(inherit, "EXPECTED_CONTROL_HASHES", {"registry": (inherit.REGISTRY_PATH, "fixed")})
    monkeypatch.setattr(inherit, "sha_file", lambda path: "changed")
    with pytest.raises(SystemExit, match="registry hash"):
        inherit.build_lock()


def test_inherited_data_hash_mismatch_stops(monkeypatch, tmp_path):
    fixture = tmp_path / "index.bin"
    fixture.write_bytes(b"changed")
    monkeypatch.setattr(inherit, "EXPECTED_CONTROL_HASHES", {})
    monkeypatch.setattr(inherit, "INHERITED_DATA", {"bm25_data": (fixture, "fixed")})
    monkeypatch.setattr(inherit, "sha_file", lambda path: "changed")
    with pytest.raises(SystemExit, match="bm25_data hash"):
        inherit.build_lock()


def test_publish_never_replaces_different_existing_lock(monkeypatch, tmp_path):
    lock_path = tmp_path / "inheritance-lock.json"
    lock_path.write_text(json.dumps({"identity_sha256": "old"}))
    monkeypatch.setattr(inherit, "build_lock", lambda: {"identity_sha256": "new"})
    monkeypatch.setattr(inherit, "load_json", lambda path: json.loads(path.read_text()))
    with pytest.raises(SystemExit, match="replacement forbidden"):
        inherit.publish(lock_path)
    assert json.loads(lock_path.read_text()) == {"identity_sha256": "old"}


def test_publish_existing_identical_lock_is_idempotent(monkeypatch, tmp_path):
    expected = {"identity_sha256": "same"}
    lock_path = tmp_path / "inheritance-lock.json"
    lock_path.write_text(json.dumps(expected))
    monkeypatch.setattr(inherit, "build_lock", lambda: expected)
    monkeypatch.setattr(inherit, "load_json", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(inherit, "write_json", lambda *args: pytest.fail("must not rewrite"))
    assert inherit.publish(lock_path) == expected
