"""The candidate cache must obey the registered construction, or refuse.

Each check below breaks one registered rule and asserts the builder does not silently accept
it — quota counting, the positive's origin, the tie-break, the per-query RNG, and the fact that
a teacher hit is never a label.
"""
from __future__ import annotations

import numpy as np
import pytest

import cache


MANIFESTS = {"v1_artifact": {"fixture": "test v1 rows"},
             "teacher_query_preprocessing": "fixture: raw text, no prefix"}


def _build(w, reg, **kw):
    kw.setdefault("manifests", MANIFESTS)
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


def test_ties_spanning_block_boundaries_on_shuffled_ids(tiny_world):
    """A tie wider than any partition block, with ids permuted independently of row order.

    A partial `argpartition` selects the cutoff on score alone, so it can drop the smaller
    bank id a tie requires; the full ordering cannot.
    """
    import cache as C
    rng = np.random.default_rng(11)
    n = 400
    doc_ids = [f"d{i:04d}" for i in range(n)]
    rng.shuffle(doc_ids)                                   # ids unrelated to row order
    vecs = np.zeros((n, 4), dtype=np.float32)
    vecs[:, 0] = 1.0
    bank = C.Bank(doc_ids, vecs, ["synthetic"] * n, seed=0)
    scores = np.zeros(n, dtype=np.float32)                 # one giant tie
    order = list(C._ranked(scores, bank.id_rank))
    assert [bank.doc_ids[i] for i in order] == sorted(bank.doc_ids)
    # and the same holds for a tie band straddling several block sizes
    scores = np.zeros(n, dtype=np.float32)
    scores[:70] = 1.0
    top = [bank.doc_ids[i] for _, i in zip(range(70), C._ranked(scores, bank.id_rank))]
    assert top == sorted(bank.doc_ids[i] for i in range(70))


def test_positives_outside_the_bank_are_refused_by_qid(tiny_world, reg):
    """registry positive_bank_policy: never silently converted to a query-only example."""
    w = dict(tiny_world)
    qs = list(w["queries"])
    victim = next(i for i, q in enumerate(qs) if q.positive_ids)
    qs[victim] = cache.QuerySpec(**{**qs[victim].__dict__,
                                    "positive_ids": qs[victim].positive_ids + ("not-in-bank",)})
    w["queries"] = qs
    with pytest.raises(ValueError, match=f"{qs[victim].qid}.*outside the bank"):
        _build(w, reg)


def test_uniform_rng_matches_an_independently_computed_golden_seed():
    """`training.candidate_construction.rng`, recomputed here from hashlib, not from cache.py."""
    import hashlib
    seed, text = 7, "s3 bucket policy"
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    digest = hashlib.sha256(b"7" + b"\x00" + text_sha.encode("utf-8")).digest()
    want = np.random.default_rng(int.from_bytes(digest[:8], "big")).integers(0, 1000, size=5)
    got = cache._query_rng(seed, text).integers(0, 1000, size=5)
    assert np.array_equal(got, want)
    # the recipe and its version are recorded, so a later change is visible in the identity
    assert "SHA-256(raw query text as UTF-8) hex digest" in cache.RNG_RECIPE


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


def test_identity_refuses_empty_v1_and_preprocessing(tiny_world, reg):
    with pytest.raises(ValueError, match="v1_artifact"):
        cache.identity(tiny_world["queries"], tiny_world["bank"], reg, 0, manifests={})
    with pytest.raises(ValueError, match="teacher_query_preprocessing"):
        cache.identity(tiny_world["queries"], tiny_world["bank"], reg, 0,
                       manifests={"v1_artifact": {"x": 1}})


def test_identity_notices_swapped_per_query_metadata(tiny_world, reg):
    """Aggregate sets and counts survive a swap of two queries' source/bucket; the hash must not."""
    _, side = _build(tiny_world, reg)
    qs = list(tiny_world["queries"])
    i, j = 0, next(k for k, q in enumerate(qs) if q.source != qs[0].source)
    qs[i], qs[j] = (cache.QuerySpec(**{**qs[i].__dict__, "source": qs[j].source}),
                    cache.QuerySpec(**{**qs[j].__dict__, "source": qs[i].source}))
    _, side2 = _build({**tiny_world, "queries": qs}, reg)
    assert side2["identity"]["parts"]["source_split_manifest"]["sources"] == \
        side["identity"]["parts"]["source_split_manifest"]["sources"]
    assert side2["identity"]["sha256"] != side["identity"]["sha256"]


def test_load_verifies_the_stored_arrays(tiny_world, reg, tmp_path):
    arrays, side = _build(tiny_world, reg)
    cache.save(tmp_path / "c", arrays, side)
    z = dict(np.load(tmp_path / "c" / "candidates.npz", allow_pickle=False))
    z["candidate_ids"] = np.zeros_like(z["candidate_ids"])
    np.savez(tmp_path / "c" / "candidates.npz", **z)
    with pytest.raises(SystemExit, match="has been altered"):
        cache.load(tmp_path / "c")


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
