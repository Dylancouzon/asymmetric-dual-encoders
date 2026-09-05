"""The teacher-target cache: content keys, fp16 round-trip, and crash-safe appends.

The property that matters is COLD == WARM (`m10/CODEMAP.md` pitfall 8): the run that encodes a
text and the run that reads it back must give the trainer the same vector, or an arm and its
re-run are different experiments.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

import targets10 as TG

IDENT = {"model": "test/stub", "revision": "0", "pooling": "mean-l2", "dim": 8,
         "post_dense": None, "role": "query", "prefix": "Q: ", "max_length": 512,
         "store_dtype": "fp16", "hash": "blake2b-128"}


def _cache(d, dim=8):
    return TG.TargetCache(d=Path(d) / "c", ident=IDENT, dim=dim)


def _put(c, texts, vecs):
    """Every append goes through the writer lock -- that is the invariant, so the tests obey it."""
    with c.writer_lock():
        c.append(TG.keys_of(texts), vecs)


def _vecs(n, dim=8, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_the_key_is_the_text_and_nothing_else():
    assert TG.key_of("abc") == TG.key_of("abc") != TG.key_of("abd")
    assert len(TG.key_of("abc")) == TG.KEY_BYTES
    assert TG.keys_of(["a", "b"]).shape == (2, TG.KEY_BYTES)


def test_a_text_encoded_once_is_never_encoded_twice():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        texts = ["a", "b", "c"]
        _put(c, texts, _vecs(3))
        rows = c.rows_for(["b", "zzz", "a"])
        assert rows[0] == 1 and rows[2] == 0 and rows[1] == -1


def test_fp16_round_trip_is_identical_cold_and_warm():
    """The encoding run must not hand the trainer fp32 vectors the re-run will never see."""
    with tempfile.TemporaryDirectory() as d:
        texts = [f"t{i}" for i in range(64)]
        v = _vecs(64)
        c = _cache(d)
        _put(c, texts, v.astype(np.float16))
        cold = TG.targets(texts, cache=c)
        warm = TG.targets(texts, cache=_cache(d))              # re-opened from disk
        assert np.array_equal(cold, warm)
        assert cold.dtype == np.float32
        want = TG.normalize(np.asarray(v.astype(np.float16), dtype=np.float32))
        assert np.array_equal(cold, want)
        assert not np.array_equal(cold, v), "fp16 storage must be visible on the cold path too"


def test_targets_are_returned_in_the_ORDER_ASKED_not_in_cache_order():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        texts = [f"t{i}" for i in range(16)]
        _put(c, texts, _vecs(16).astype(np.float16))
        a = TG.targets(["t9", "t0", "t9"], cache=c)
        b = TG.targets(texts, cache=c)
        assert np.array_equal(a[0], b[9]) and np.array_equal(a[1], b[0])
        assert np.array_equal(a[0], a[2])


def test_a_missing_target_raises_with_the_command_that_fixes_it():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        _put(c, ["a"], _vecs(1).astype(np.float16))
        with pytest.raises(SystemExit, match="targets10.py"):
            TG.targets(["a", "b"], cache=c)


def test_an_append_survives_reopening_and_grows_rather_than_replaces():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        _put(c, ["a", "b"], _vecs(2).astype(np.float16))
        c2 = _cache(d)
        assert c2.n == 2
        _put(c2, ["c"], _vecs(1, seed=3).astype(np.float16))
        assert _cache(d).n == 3
        assert (_cache(d).rows_for(["a", "b", "c"]) == [0, 1, 2]).all()


def test_a_crash_between_the_vector_write_and_the_meta_write_loses_the_CHUNK_not_the_CACHE():
    """`n` is the authority; trailing bytes are unreferenced. A READER ignores them -- it must not
    truncate a store another process may be appending to -- and a WRITER repairs them on refresh."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        _put(c, ["a", "b"], _vecs(2).astype(np.float16))
        with open(c.vecs_p, "ab") as fh:                       # a half-written chunk
            fh.write(np.zeros((1, 8), dtype=np.float16).tobytes())
        c2 = _cache(d)
        assert c2.n == 2 and c2.vecs_p.stat().st_size == 3 * 8 * 2, "a reader leaves the tail"
        assert (c2.rows_for(["a", "b"]) == [0, 1]).all()
        with c2.writer_lock():
            c2.refresh()
        assert c2.vecs_p.stat().st_size == 2 * 8 * 2, "a writer truncates it"


def test_a_cache_written_under_another_encoder_identity_is_refused():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        m = json.loads(c.meta_p.read_text())
        m["identity"] = {**IDENT, "prefix": "something else"}
        c.meta_p.write_text(json.dumps(m))
        with pytest.raises(SystemExit, match="different encoder identity"):
            _cache(d)


def test_the_cache_identity_is_the_stella_query_prompt_the_M9_targets_used():
    """Retyping the prompt would give M9's targets and M10's a silent one-token disagreement."""
    import teacher
    ident = TG.identity()
    assert ident["prefix"] == teacher.QUERY_PREFIX
    assert ident["prefix"].startswith("Instruct: Given a web search query")
    assert ident["model"] == "NovaSearch/stella_en_400M_v5" and ident["max_length"] == 512
    assert str(TG.cache_dir()).endswith(TG.cache_dir().name) and "stella" in TG.cache_dir().name


def test_encode_missing_collapses_duplicates_and_skips_what_is_cached(monkeypatch):
    """PAQ's build sample is nested inside its A2 sample: the same text arrives twice."""
    calls = []

    def fake_encode(texts, prefix="", max_length=512, batch_tokens=0, verbose=False):
        calls.append(list(texts))
        return _vecs(len(texts), seed=len(calls))

    monkeypatch.setattr(TG.teacher, "encode", fake_encode)
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        rep = TG.encode_missing(["a", "b", "a", "c"], cache=c, chunk=10, verbose=False)
        assert rep["to_encode"] == 3 and rep["duplicates_collapsed"] == 1
        assert calls == [["a", "b", "c"]]
        rep2 = TG.encode_missing(["a", "b", "c"], cache=c, chunk=10, verbose=False)
        assert rep2["to_encode"] == 0 and len(calls) == 1


def test_a_non_finite_teacher_vector_stops_the_encode_rather_than_being_stored(monkeypatch):
    def bad(texts, **kw):
        v = _vecs(len(texts))
        v[0, 0] = np.nan
        return v

    monkeypatch.setattr(TG.teacher, "encode", bad)
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        with pytest.raises(SystemExit, match="non-finite"):
            TG.encode_missing(["a", "b"], cache=c, chunk=10, verbose=False)
        assert _cache(d).n == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_a_short_store_is_refused_rather_than_zero_padded():
    """Codex 2026-09-05 finding 8: `truncate(want)` EXTENDS a short file, so half of a lost vector
    was normalized and trained as a target. A partial vector is not a target."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        _put(c, ["a", "b"], _vecs(2).astype(np.float16))
        with open(c.vecs_p, "r+b") as fh:                      # half of the second vector lost
            fh.truncate(3 * 8)
        with pytest.raises(SystemExit, match="SHORT by"):
            _cache(d)


def test_only_one_writer_may_append_at_a_time():
    """Two writers can append vectors A,B and keys B,A under the same published `n`, and the
    reopened cache then maps keyB to vectorA (Codex 2026-09-05 finding 5)."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        other = _cache(d)
        with c.writer_lock():
            assert (c.dir / "lock").exists()
            with pytest.raises(SystemExit, match="another process is appending"):
                with other.writer_lock():
                    pass
        assert not (c.dir / "lock").exists(), "the lock is released even so"
        with other.writer_lock():                              # and is takeable afterwards
            pass


def test_encode_missing_holds_the_lock_and_a_read_path_does_not(monkeypatch):
    seen = {}

    def fake_encode(texts, prefix="", max_length=512, batch_tokens=0, verbose=False):
        seen["locked_during_encode"] = (Path(seen["dir"]) / "lock").exists()
        return _vecs(len(texts))

    import teacher
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        seen["dir"] = c.dir
        monkeypatch.setattr(teacher, "encode", fake_encode)
        TG.encode_missing(["a", "b"], cache=c, verbose=False)
        assert seen["locked_during_encode"] is True
        assert not (c.dir / "lock").exists()
        TG.targets(["a", "b"], cache=_cache(d))                # a read takes nothing
        assert not (c.dir / "lock").exists()


def test_append_outside_the_writer_lock_is_refused():
    """A lock nobody checks is a comment: `append` and `refresh` were public and unguarded, so a
    second process could append with a stale `n` while another held the lock."""
    with tempfile.TemporaryDirectory() as d:
        c, other = _cache(d), _cache(d)
        with pytest.raises(SystemExit, match="without the writer lock"):
            c.append(TG.keys_of(["a"]), _vecs(1).astype(np.float16))
        with pytest.raises(SystemExit, match="without the writer lock"):
            c.refresh()
        with c.writer_lock():
            c.append(TG.keys_of(["a"]), _vecs(1).astype(np.float16))
            # the OTHER handle still holds no lock, even while one is out
            with pytest.raises(SystemExit, match="without the writer lock"):
                other.append(TG.keys_of(["b"]), _vecs(1).astype(np.float16))
        assert c.n == 1
