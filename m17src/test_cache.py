"""The candidate cache must obey the registered construction, or refuse.

Each check below breaks one registered rule and asserts the builder does not silently accept
it — quota counting, the positive's origin, the tie-break, the per-query RNG, and the fact that
a teacher hit is never a label.
"""
from __future__ import annotations

import numpy as np
import pytest

import cache


def _build(w, reg, **kw):
    return cache.build(w["queries"], w["bank"], w["teacher_q"], w["v1_q"], reg, **kw)


def test_mix_and_slots(tiny_world, reg):
    arrays, side = _build(tiny_world, reg)
    K = reg["training"]["candidate_k"]
    assert arrays["candidate_ids"].shape == (len(tiny_world["queries"]), K)
    for qi in range(arrays["candidate_ids"].shape[0]):
        ids = arrays["candidate_ids"][qi]
        assert len(set(ids.tolist())) == K, "candidates must be unique per query"
        src = arrays["candidate_source"][qi]
        mix = (reg["training"]["candidate_mix_labeled"] if arrays["has_label"][qi]
               else reg["training"]["candidate_mix_query_only"])
        assert (src == cache.SRC_POSITIVE).sum() == mix["known_positive"]
        assert (src == cache.SRC_TEACHER).sum() == mix["teacher_top"]
        assert (src == cache.SRC_V1).sum() == mix["zero_v1_top"]
        assert (src == cache.SRC_UNIFORM).sum() == mix["uniform"]
    assert side["counts"]["alias_pairs"] * 2 == side["counts"]["alias_views"]


def test_positive_is_the_first_in_qrels_order_and_nullable(tiny_world, reg):
    arrays, _ = _build(tiny_world, reg)
    for qi, q in enumerate(tiny_world["queries"]):
        if not q.positive_ids:
            assert not arrays["has_label"][qi]
            assert arrays["positive_id"][qi] == -1
            continue
        assert arrays["has_label"][qi]
        want = tiny_world["bank"].index[q.positive_ids[0]]
        assert int(arrays["positive_id"][qi]) == want
        assert int(arrays["candidate_ids"][qi][0]) == want


def test_teacher_hits_are_not_labels(tiny_world, reg):
    """A teacher-sourced candidate must never be recorded as the positive."""
    arrays, _ = _build(tiny_world, reg)
    src, pos = arrays["candidate_source"], arrays["positive_id"]
    for qi in range(src.shape[0]):
        for slot, code in enumerate(src[qi]):
            if code != cache.SRC_POSITIVE:
                continue
            assert int(arrays["candidate_ids"][qi][slot]) == int(pos[qi])


def test_ranking_tie_break_is_ascending_document_id(tiny_world, reg):
    """With identical scores the walk must admit documents in ascending id order."""
    bank = tiny_world["bank"]
    flat = np.zeros(len(bank.doc_ids), dtype=np.float32)
    got = [i for _, i in zip(range(5), cache._ranked(flat, bank.id_rank))]
    assert [bank.doc_ids[i] for i in got] == sorted(bank.doc_ids)[:5]


def test_uniform_rng_is_per_query_and_reproducible(tiny_world, reg):
    a, _ = _build(tiny_world, reg, cache_seed=3)
    b, _ = _build(tiny_world, reg, cache_seed=3)
    c, _ = _build(tiny_world, reg, cache_seed=4)
    assert np.array_equal(a["candidate_ids"], b["candidate_ids"])
    assert not np.array_equal(a["candidate_ids"], c["candidate_ids"])


def test_identity_covers_the_registered_fields(tiny_world, reg):
    _, side = _build(tiny_world, reg, cache_seed=1)
    parts = side["identity"]["parts"]
    for key in ("raw_query_text_sha256", "source_split_manifest", "alias_manifest", "teacher",
                "bank", "v1_artifact", "candidates"):
        assert key in parts
    # changing a single query's text must change the identity
    w2 = dict(tiny_world)
    qs = list(w2["queries"])
    qs[0] = cache.QuerySpec(**{**qs[0].__dict__, "text": qs[0].text + " changed"})
    w2["queries"] = qs
    _, side2 = _build(w2, reg, cache_seed=1)
    assert side2["identity"]["sha256"] != side["identity"]["sha256"]


def test_entropy_block_has_the_m8_shape(tiny_world, reg):
    _, side = _build(tiny_world, reg)
    ent = side["entropy_diagnostic"]
    assert set(ent) >= {"setting", "entropy_nats", "entropy_as_fraction_of_ceiling",
                        "p_max_of_teacher_distribution", "share_of_queries_below_1e-4_nats",
                        "share_of_queries_below_1e-2_nats"}
    assert set(ent["entropy_nats"]) >= {"mean", "p50", "p95"}
    assert 0.0 <= ent["entropy_as_fraction_of_ceiling"] <= 1.0


def test_bad_mix_is_refused(tiny_world, reg):
    bad = {**reg, "training": {**reg["training"],
                               "candidate_mix_labeled": {"known_positive": 1, "teacher_top": 1,
                                                         "zero_v1_top": 1, "uniform": 1}}}
    with pytest.raises(ValueError, match="does not sum to candidate_k"):
        _build(tiny_world, bad)


def test_roundtrip(tiny_world, reg, tmp_path):
    arrays, side = _build(tiny_world, reg)
    cache.save(tmp_path / "c", arrays, side)
    back, side2 = cache.load(tmp_path / "c")
    assert np.array_equal(back["candidate_ids"], arrays["candidate_ids"])
    assert side2["identity"]["sha256"] == side["identity"]["sha256"]
