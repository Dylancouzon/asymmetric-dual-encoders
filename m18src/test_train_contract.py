"""Identity, immutable snapshot and pair-aware sampling tests."""
from __future__ import annotations

import json

import numpy as np
import pytest

import train
from common import sha_array


def _data(order=(0, 1)):
    inherited = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    new = np.ones((1, 4), np.float32)
    rows = [[0, 3], [1, 3]]
    value = {"model": train.build_model(inherited, new),
             "query_ids": [f"q{i}" for i in order], "ids": [rows[i] for i in order],
             "teacher_q": np.eye(2, 4, dtype=np.float32)[list(order)],
             "bank": np.eye(4, dtype=np.float32),
             "candidate_ids": np.asarray([[0, 1], [1, 0]], np.int32)[list(order)],
             "teacher_scores": np.asarray([[1, 0], [1, 0]], np.float32)[list(order)],
             "alias_pair_ids": np.asarray(["p", "p"])[list(order)]}
    return value


def test_prepared_identity_rejects_same_length_permutation():
    original = _data()
    identity = train.training_data_identity(original)
    permuted = _data((1, 0))
    assert train.training_data_identity(permuted)["sha256"] != identity["sha256"]


def test_pair_complete_batch_adds_alias_mate():
    pair_ids = np.asarray(["pair", "pair", "", ""])
    got = train._pair_complete_batch(np.asarray([0, 2]), pair_ids)
    assert set(got) == {0, 1}
    assert train._alias_slots(got, pair_ids) == [(0, 1)]


def test_alias_slots_use_two_distinct_views_despite_repeated_queries():
    pair_ids = np.asarray(["pair", "pair", ""])
    assert train._alias_slots(np.asarray([0, 0, 1, 1, 2]), pair_ids) == [(0, 2)]
    assert train._alias_slots(np.asarray([0, 0, 2]), pair_ids) == []


def test_cross_epoch_batch_stream_is_deterministic_and_may_repeat():
    first = {"epoch": 0, "position": 0}
    got_a = train._batch(first, n=3, batch=5, seed=18001)
    got_b = train._batch(first, n=3, batch=5, seed=18001)
    second = {"epoch": 0, "position": 0}
    assert np.array_equal(got_a, train._batch(second, n=3, batch=5, seed=18001))
    assert np.array_equal(got_b, train._batch(second, n=3, batch=5, seed=18001))
    assert first == second and len(set(got_a)) < len(got_a)


def test_trainer_version_is_bound_into_resume_identity():
    a = train.RunCfg(rehearsal=True, trainer_version="sampler-a")
    b = train.RunCfg(rehearsal=True, trainer_version="sampler-b")
    assert a.resume_identity()["sha256"] != b.resume_identity()["sha256"]


def test_load_snapshot_rejects_rows_array_not_equal_verified_blocks(tmp_path):
    inherited = np.zeros((2, 3), np.float32)
    new = np.ones((1, 3), np.float32)
    cfg = train.RunCfg(rehearsal=True, variant="T0", schedule_steps=1, stop_after=0,
                       checkpoint_steps=(0,), tokenizer_sha256="t", preprocessing_sha256="p",
                       vocabulary_sha256="v", cache_artifact_sha256="c", prepared_data_sha256="d")
    model = train.build_model(inherited, new)
    lineage = {"inherited_scalars_sha256": sha_array(np.ones(2, np.float32))}
    root = tmp_path / "snapshot"
    train.save_snapshot(root, model, cfg, 0, lineage, sha_array(new))
    z = np.load(root / "table.npz")
    np.savez(root / "table.npz", rows=np.full((3, 3), 7, np.float32),
             inherited=z["inherited"], new_rows=z["new_rows"])
    with pytest.raises(SystemExit, match="concatenated table mismatch"):
        train.load_snapshot(root)
