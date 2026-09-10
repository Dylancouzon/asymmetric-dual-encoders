"""The properties an arm's corpus has to have before it is worth spending a GPU-hour on.

The expensive failures this guards are the silent ones: a source that loads but whose targets are
another source's rows, a "balanced" sampler that is balanced only on the sources that happen to be
present, a resumed arm that draws different data, and the FORMS-12 hold-out being trainable.
"""
import json
import os
import sys
import types
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
import torch

import corpus_loader as CL
import data10 as D
import trainer10 as T


def _jsonl(d, name, rows):
    p = Path(d) / name
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def _ids(n, seed=0, lo=3, hi=40):
    rng = np.random.default_rng(seed)
    lens = rng.integers(lo, hi, size=n)
    offs = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    flat = rng.integers(1, 900, size=int(offs[-1])).astype(np.int32)
    return CL.PackedIds(flat, offs)


def _targets(n, dim=8, seed=1):
    rng = np.random.default_rng(seed)
    T_ = rng.normal(size=(n, dim)).astype(np.float32)
    return T_ / np.linalg.norm(T_, axis=1, keepdims=True)


# ---------------------------------------------------------------------------- reading rows ----

def test_a_source_manifest_hashes_the_file_and_counts_every_form():
    with tempfile.TemporaryDirectory() as d:
        rows = [{"text": f"t{i}", "form": "title" if i % 2 else "claim"} for i in range(10)]
        p = _jsonl(d, "s.jsonl", rows)
        texts, forms = CL._rows_from_jsonl(p)
        assert len(texts) == 10 and forms.count("title") == 5
        import hashlib
        assert CL.sha_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
        before = CL.sha_file(p)
        _jsonl(d, "s.jsonl", rows + [{"text": "extra", "form": "claim"}])
        assert CL.sha_file(p) != before, "the manifest hash must move when the corpus does"


def test_the_generated_half_needs_no_code_change():
    """`build_form` emits `query`/`seed_id`; PAQ emits `question`; the harvest emits `text`."""
    with tempfile.TemporaryDirectory() as d:
        p = _jsonl(d, "g.jsonl", [{"query": "how do I reset it", "form": "howto", "seed_id": "7"}])
        assert CL._rows_from_jsonl(p) == (["how do I reset it"], ["howto"])
        p = _jsonl(d, "q.jsonl", [{"question": "who wrote hamlet"}])
        assert CL._rows_from_jsonl(p, default_form="factoid") == (["who wrote hamlet"], ["factoid"])


def test_an_unregistered_form_is_refused_rather_than_coded_as_something_else():
    with tempfile.TemporaryDirectory() as d:
        p = _jsonl(d, "s.jsonl", [{"text": "x", "form": "not-a-form"}])
        with pytest.raises(SystemExit, match="not one of the 12"):
            CL._rows_from_jsonl(p)


def test_the_forms12_holdout_cannot_be_read_as_training_data():
    """"500 seed documents per form are set aside first; queries generated or harvested from them
    are never trained on" -- so the file is refused by PATH, before it is opened."""
    p = CL.WORK / "m10harvest" / "harvest_forms12.jsonl"
    assert str(p.resolve()) in CL.HOLDOUT_FILES
    with pytest.raises(SystemExit, match="FORMS-12 hold-out"):
        CL._rows_from_jsonl(p)


def test_every_registered_form_name_comes_from_the_frozen_rubric():
    import forms
    assert CL.FORMS == tuple(forms.RUBRIC) and len(CL.FORMS) == 12
    assert set(CL.M9_SOURCE_FORM.values()) | {CL.PAQ_FORM} <= set(CL.FORMS)


# -------------------------------------------------------------------------- packed tokens ----

def test_packed_ids_behave_like_the_list_they_replace():
    p = _ids(64)
    assert len(p) == 64
    assert list(p.lengths) == [len(p[i]) for i in range(64)]
    x, m = D.collate(p, np.arange(8), pad_id=0)
    assert x.shape == m.shape and int(m.sum()) == int(p.lengths[:8].sum())


def test_length_buckets_uses_the_lengths_a_packed_corpus_already_knows():
    p = _ids(256)
    a = D.length_buckets(p, 16, seed=0)
    b = D.length_buckets([p[i] for i in range(len(p))], 16, seed=0)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]


# ------------------------------------------------------------------------------- sampling ----

def _stream(n_per_form, balanced=True, batch_size=8, seed=0, forms=("title", "claim", "keyword")):
    counts = n_per_form if isinstance(n_per_form, dict) else {f: n_per_form for f in forms}
    fid = []
    for f, c in counts.items():
        fid += [CL.FORM_ID[f]] * c
    n = len(fid)
    return CL.FormBalancedStream(_ids(n), _targets(n), np.array(fid), pad_id=0,
                                 batch_size=batch_size, seed=seed, balanced=balanced), np.array(fid)


def test_balanced_shares_are_equal_across_the_forms_present():
    st, _ = _stream({"title": 400, "claim": 80, "keyword": 40})
    sh = st.realized_shares(300)
    assert set(sh) == {"title", "claim", "keyword"}
    assert all(abs(v - 1 / 3) < 0.01 for v in sh.values()), sh


def test_the_unbalanced_variant_is_proportional_to_the_corpus_and_stays_available():
    """Family A2's volume control and the reported diagnostic. It must NOT be balanced."""
    st, _ = _stream({"title": 400, "claim": 80, "keyword": 40}, balanced=False)
    sh = st.realized_shares(60)
    assert 0.70 < sh["title"] < 0.85, sh                # ~400/520, not 1/3
    assert abs(sum(sh.values()) - 1.0) < 1e-6


def test_a_small_form_is_sampled_with_replacement_rather_than_dropped():
    """"texts drawn with replacement within a form" -- a form smaller than a batch still runs, and
    the batch REPEATS rows rather than being short. Merely appearing is not the property: a fixed
    cycled batch would satisfy that too (Codex re-review 2026-09-05)."""
    st, forms = _stream({"title": 100, "claim": 3}, batch_size=8)
    drawn = [st._pick(k)[1] for k in range(60) if st._pick(k)[0] == CL.FORM_ID["claim"]]
    assert drawn, "the small form must still be presented"
    for idx in drawn:
        assert len(idx) == 8 and (forms[idx] == CL.FORM_ID["claim"]).all()
        assert len(set(idx.tolist())) < 8, "a 3-row form at batch 8 must repeat rows"


def test_every_batch_is_one_form_so_length_bucketing_survives_balancing():
    st, forms = _stream(120)
    for k in range(30):
        f, idx = st._pick(k)
        assert set(forms[idx].tolist()) == {f}


def test_batch_k_is_a_pure_function_of_k_and_targets_travel_with_their_rows():
    st, _ = _stream(120)
    Tm = st.T
    for k in (0, 1, 5, 31, len(st) + 3):
        a, b = st.batch(k), st.batch(k)
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
        idx = st._pick(k)[1]
        assert np.allclose(a[2].numpy(), Tm[idx], atol=1e-6)


def test_shares_and_order_move_with_the_seed_but_not_with_the_wall_clock():
    a, _ = _stream(120, seed=0)
    b, _ = _stream(120, seed=1)
    ka = [a._pick(k)[0] for k in range(30)]
    kb = [b._pick(k)[0] for k in range(30)]
    assert ka != kb, "two seeds must not present the forms in the same order"
    assert ka == [_stream(120, seed=0)[0]._pick(k)[0] for k in range(30)]


# ------------------------------------------------------------------------------- data cut ----

def _segs(n_list, dim=8):
    out, base = [], 0
    for i, n in enumerate(n_list):
        arr = _targets(n, dim=dim, seed=10 + i).astype(np.float16)
        out.append(CL.Segment(f"s{i}", [f"s{i}-{j}" for j in range(n)],
                              np.full(n, CL.FORM_ID["claim"], dtype=np.int16), arr,
                              np.arange(n)))
        base += n
    return out


def test_the_data_cut_downsamples_the_whole_corpus_deterministically():
    segs = _segs([300, 200])
    a, rep = CL.apply_data_cut(segs, 250, seed=0)
    b, _ = CL.apply_data_cut(segs, 250, seed=0)
    assert rep["applied"] and rep["n_after"] == 250
    assert [s.texts for s in a] == [s.texts for s in b]
    assert sum(len(s) for s in a) == 250


def test_no_registered_cut_means_no_cut_and_says_so():
    """The HELPER still reports honestly with no count -- but a cut ARM must not reach it; that is
    `test_a_cut_arm_refuses_to_train_uncut` below. This test used to be the whole story and blessed
    A2/A3/A4 training at three different volumes (Codex 2026-09-05 finding 1)."""
    segs = _segs([50])
    out, rep = CL.apply_data_cut(segs, None)
    assert out is segs and not rep["applied"] and "§0b" in rep["_why"]


def test_the_cut_arms_are_read_from_the_registry_and_name_the_anchor():
    assert CL.cut_arms() == {"A2", "A3", "A4", "ANCHOR"}, CL.cut_arms()


class _Tok:
    pad_token_id = 0

    def __call__(self, texts, **kw):
        return {"input_ids": [[7] * (len(t) % 5 + 2) for t in texts]}


def _fake_corpus(monkeypatch, tmp, n=40):
    monkeypatch.setattr(CL, "TOKCACHE", Path(tmp))
    segs = _segs([n])
    monkeypatch.setattr(CL, "load_segments",
                        lambda names, head_per_source=None, verbose=True, consumed=None: (
                            segs, {"sources": [], "sha256": "abc", "n_rows": n}))
    monkeypatch.setattr(CL, "ARM_SOURCES", {**CL.ARM_SOURCES, "A3": ("harvest",)})
    return segs


def test_a_cut_arm_refuses_to_train_uncut(monkeypatch):
    """A2/A3/A4 are cut to the identical post-screen unique-text count; without the count they
    would train the full corpus and family A's forms contrast becomes a volume contrast."""
    with tempfile.TemporaryDirectory() as d:
        _fake_corpus(monkeypatch, d)
        monkeypatch.setattr(CL, "data_cut_count", lambda registry=None: None)
        with pytest.raises(SystemExit, match="registered cut arm"):
            CL.build_query_stream("A3", _Tok(), "bge-small", batch_size=4, verbose=False)


def test_a_smoke_may_train_uncut_only_by_saying_so_in_the_artifact(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _fake_corpus(monkeypatch, d)
        monkeypatch.setattr(CL, "data_cut_count", lambda registry=None: None)
        _st, man = CL.build_query_stream("A3", _Tok(), "bge-small", batch_size=4,
                                         allow_uncut=True, verbose=False)
        assert man["uncut"] is True and man["is_cut_arm"] is True


def test_a_registered_cut_is_applied_to_a_cut_arm_without_being_asked(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _fake_corpus(monkeypatch, d)
        monkeypatch.setattr(CL, "data_cut_count", lambda registry=None: 25)
        _st, man = CL.build_query_stream("A3", _Tok(), "bge-small", batch_size=4, verbose=False)
        assert man["data_cut"]["applied"] and man["data_cut"]["n_after"] == 25
        assert "uncut" not in man


# --------------------------------------------------------------------------- target view ----

def test_the_target_view_gathers_the_right_segment_and_normalizes_in_fp32():
    segs = _segs([40, 25])
    v = CL.TargetView(segs)
    assert len(v) == 65
    got = v[np.array([0, 39, 40, 64])]
    assert got.dtype == np.float32 and got.shape == (4, 8)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
    for row, (s, i) in zip(got, [(0, 0), (0, 39), (1, 0), (1, 24)]):
        want = np.asarray(segs[s].array[i], dtype=np.float32)
        assert np.allclose(row, want / np.linalg.norm(want), atol=1e-6)


# ---------------------------------------------------------------------- resume determinism ----

class Toy(torch.nn.Module):
    def __init__(self, d_out=8):
        super().__init__()
        self.head = torch.nn.Linear(4, d_out)

    def forward(self, ids, mask):
        x = ids.float() @ torch.ones(ids.shape[-1], 4) / ids.shape[-1]
        return torch.nn.functional.normalize(self.head(x), dim=-1)


def test_resume_reproduces_an_uninterrupted_run_on_the_new_loader():
    """`test_trainer10` proves this for a synthetic batch_fn; the corpus path must keep it."""
    def run(**kw):
        torch.manual_seed(0)
        m = Toy()
        st, _ = _stream(200, batch_size=8)
        d, _ = _stream(200, batch_size=8, seed=3)
        return m, T.train_arm(m, D.batch_fn(st, d), total_steps=40, seed=0, **kw)

    _m1, ref = run()
    with tempfile.TemporaryDirectory() as tdir:
        ck = Path(tdir) / "ck.pt"
        _m2, part = run(ckpt_path=ck, ckpt_every=17)
        m3, rest = run(resume_from=ck)
    assert part["losses"] == ref["losses"]
    assert rest["start_step"] == 34 and rest["steps_run"] == 6
    for a, b in zip(rest["losses"], ref["losses"]):
        assert abs(a - b) < 1e-6, (a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_the_document_role_marker_is_the_registered_one_and_is_actually_applied():
    """"document-role examples carry M9's fixed document-role marker" -- `data10.pretokenize` had
    no prefix argument at all, so documents reached the student as raw bytes."""
    import json as _json
    reg = _json.loads((CL.REPO / "m9" / "registry.json").read_text())
    assert CL.doc_marker() == reg["templates"]["doc_student"] == "passage: "

    class Tok:
        pad_token_id = 0

        def __call__(self, texts, **kw):
            return {"input_ids": [[len(t)] for t in texts]}

    assert D.pretokenize(Tok(), ["x"], prefix="passage: ")[0][0] == len("passage: x")
    assert D.pretokenize(Tok(), ["x"])[0][0] == 1, "the query role stays raw bytes"


def test_the_token_cache_identity_binds_the_cut_not_just_the_source_list(monkeypatch):
    """Two corpora that differ only in the data cut must not share pretokenized ids."""
    class Tok:
        pad_token_id = 0

        def __call__(self, texts, **kw):
            return {"input_ids": [[7] * (len(t) % 5 + 2) for t in texts]}

    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(CL, "TOKCACHE", Path(d))
        segs = _segs([40])
        man = {"sha256": "abc"}
        a = CL.tokenize_corpus(Tok(), segs, man, "bge-small", verbose=False,
                               extra_ident={"data_cut": {"applied": False}})
        b = CL.tokenize_corpus(Tok(), segs, man, "bge-small", verbose=False,
                               extra_ident={"data_cut": {"applied": True, "n_after": 20}})
        assert len(a) == len(b) == 40
        assert len(list(Path(d).iterdir())) == 2, "the cut must be part of the cache identity"


def test_the_manifest_identity_is_the_corpus_and_not_the_run(monkeypatch):
    """`seconds` is a wall-clock measurement: leaving it inside the hashed view would put a fresh
    identity -- and so a fresh pretokenization of 5.3M texts -- on every load."""
    seen = {}

    def fake(name, limit=None):
        seen[name] = seen.get(name, 0) + 1
        return ["a", "b"], ["claim", "title"], None, {"source": name, "sha256": "x", "n_rows": 2,
                                                       "by_form": {"claim": 1, "title": 1}}

    class Cache:
        def rows_for(self, texts):
            return np.arange(len(texts))

        def vecs(self):
            return np.ones((2, 8), dtype=np.float16)

    import targets10
    monkeypatch.setattr(CL, "source_texts", fake)
    monkeypatch.setattr(targets10, "TargetCache", lambda *a, **k: Cache())
    monkeypatch.setitem(CL.SOURCES, "fake", {"kind": "jsonl", "path": "/dev/null", "what": "t"})
    a = CL.load_segments(["fake"], verbose=False)[1]
    b = CL.load_segments(["fake"], verbose=False)[1]
    assert a["sha256"] == b["sha256"] and a["by_form"] == {"claim": 1, "title": 1}


# ------------------------------------------------------------- the FORMS-12 hold-out, by hash ----

def test_a_copy_of_the_holdout_under_another_name_is_still_refused(monkeypatch):
    """Codex 2026-09-05 finding 3: the guard protected a PATHNAME. `cp harvest_forms12.jsonl
    generated_queries.jsonl` walked straight past it."""
    with tempfile.TemporaryDirectory() as d:
        hold = _jsonl(d, "holdout.jsonl", [{"text": "a held-out query", "form": "claim"}])
        monkeypatch.setattr(CL, "_HOLDOUT_HASHES", {})
        hs = CL.holdout_hashes(hold)
        assert len(hs) == 1
        segs = [CL.Segment("generated", ["fine", "a held-out query"],
                           [CL.FORM_ID["claim"]] * 2, _targets(2, dim=8).astype(np.float16),
                           np.arange(2))]
        monkeypatch.setattr(CL, "holdout_hashes", lambda path=None: hs)
        with pytest.raises(SystemExit, match="FORMS-12 hold-out"):
            CL.refuse_holdout_texts(segs)
        clean = [CL.Segment("generated", ["fine", "also fine"], [CL.FORM_ID["claim"]] * 2,
                            _targets(2, dim=8).astype(np.float16), np.arange(2))]
        assert CL.refuse_holdout_texts(clean) == 0


# -------------------------------------------------------------------------- unique-text cut ----

def test_the_cut_counts_UNIQUE_texts_across_sources():
    """"post-screen unique-text count": `["x", "x", "y"]` is two texts, and keeping both copies of
    `x` would also double its presentation weight inside its form."""
    a = CL.Segment("a", ["x", "x", "y"], [CL.FORM_ID["claim"]] * 3,
                   _targets(3, dim=8).astype(np.float16), np.arange(3))
    b = CL.Segment("b", ["y", "z"], [CL.FORM_ID["claim"]] * 2,
                   _targets(2, dim=8).astype(np.float16), np.arange(2))
    segs, removed = CL.dedup_segments([a, b])
    assert [s.texts for s in segs] == [["x", "y"], ["z"]]
    assert removed == {"a": 1, "b": 1}
    # the rows travel with their texts: `y` keeps segment a's row 2, `z` keeps b's row 1
    assert segs[0].rowmap.tolist() == [0, 2] and segs[1].rowmap.tolist() == [1]


# ----------------------------------------------------------- with-replacement within a form ----

def test_a_small_form_does_not_repeat_the_identical_batch():
    """A three-row form at batch 8 used to yield `0,1,2,0,1,2,0,1` every single time it came up."""
    st, forms = _stream({"title": 100, "claim": 3}, batch_size=8)
    seen = [tuple(st._pick(k)[1]) for k in range(60) if st._pick(k)[0] == CL.FORM_ID["claim"]]
    assert len(seen) >= 3 and len(set(seen)) > 1, seen


def test_a_drawn_batch_is_sorted_by_length_so_the_padded_chunk_is_tight():
    st, _forms = _stream(120, batch_size=8)
    for k in range(10):
        idx = st._pick(k)[1]
        L = st.lengths[idx]
        assert list(L) == sorted(L), L


def test_the_draw_is_a_pure_function_of_seed_and_step():
    a, _ = _stream(120, batch_size=8, seed=0)
    b, _ = _stream(120, batch_size=8, seed=0)
    c, _ = _stream(120, batch_size=8, seed=1)
    assert [tuple(a._pick(k)[1]) for k in range(20)] == [tuple(b._pick(k)[1]) for k in range(20)]
    assert [tuple(a._pick(k)[1]) for k in range(20)] != [tuple(c._pick(k)[1]) for k in range(20)]


# ------------------------------------------------------------------- cross-role collisions ----

def test_the_same_student_input_cannot_carry_two_teacher_targets():
    """The query "passage: X" and the document "X" tokenize identically once the document marker
    is applied, and their teacher targets differ."""
    tok = _Tok()
    q = D.pretokenize(tok, ["passage: hello"], prefix="")
    d = D.pretokenize(tok, ["hello"], prefix=CL.doc_marker())
    assert [x.tolist() for x in q] == [x.tolist() for x in d], "the stub must actually collide"
    assert CL.cross_role_collisions(q, d) == 1
    with pytest.raises(SystemExit, match="BOTH the query and the document role"):
        CL.guard_cross_role(q, d)
    assert CL.guard_cross_role(q, d, skip=True)["checked"] is False
    clean = D.pretokenize(tok, ["a much longer document body here"], prefix=CL.doc_marker())
    assert CL.guard_cross_role(q, clean)["collisions"] == 0


# ------------------------------------------------------------------- the tokenizer identity ----

def test_the_token_cache_identity_binds_the_TOKENIZER_not_the_students_nickname(monkeypatch):
    """Codex 2026-09-05 finding 11: `student="bge-small"` is a label, and two revisions of the same
    repo give different ids for the same text."""
    class Tok(_Tok):
        """ONE class, so the class name cannot be what separates the two caches -- the earlier
        version of this test used two classes and would have passed with vocabulary hashing
        removed entirely (Codex re-review 2026-09-05)."""
        name_or_path = "BAAI/bge-small-en-v1.5"

        def __init__(self, vocab):
            self._v = vocab

        def get_vocab(self):
            return self._v

    a, b = Tok({"a": 0, "b": 1}), Tok({"a": 0, "b": 1, "c": 2})
    assert CL.tokenizer_ident(a)["class"] == CL.tokenizer_ident(b)["class"]
    assert CL.tokenizer_ident(a)["vocab_sha256"] != CL.tokenizer_ident(b)["vocab_sha256"]
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(CL, "TOKCACHE", Path(d))
        segs, man = _segs([20]), {"sha256": "abc"}
        CL.tokenize_corpus(a, segs, man, "bge-small", verbose=False)
        CL.tokenize_corpus(b, segs, man, "bge-small", verbose=False)
        assert len(list(Path(d).iterdir())) == 2, "the vocabulary must be part of the identity"
    assert CL.tokenizer_ident(a)["name_or_path"] == "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------------- the M10 re-screen ----

def test_the_m9_pools_cannot_load_without_the_M10_rescreen_mask(monkeypatch):
    """instructions-m10.md:462 -- the M9 pools are re-screened against the COV additions. The mask
    is computed by `rescreen10`'s CLI; a training path reads it and REFUSES if it is absent."""
    import rescreen10
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(rescreen10, "CACHE", Path(d))
        monkeypatch.setattr(rescreen10, "protected_ident", lambda: {"version": "test"})
        with pytest.raises(SystemExit, match="rescreen10.py --queries"):
            rescreen10.query_keep_mask(["a", "b"], "m9-test", compute=False)
        monkeypatch.setattr(rescreen10, "_screen",
                            lambda texts, verbose=True, label="", log_every=0: (
                                np.array([True, False]), {"near": 1}))
        keep, rep = rescreen10.query_keep_mask(["a", "b"], "m9-test", verbose=False)
        assert rep["removed"] == 1 and keep.tolist() == [True, False]
        again, _ = rescreen10.query_keep_mask(["a", "b"], "m9-test", compute=False)
        assert again.tolist() == [True, False], "the mask is cached on the pool identity"
        # a DIFFERENT protected index invalidates it rather than serving the stale mask
        monkeypatch.setattr(rescreen10, "protected_ident", lambda: {"version": "test+cov"})
        with pytest.raises(SystemExit, match="rescreen10.py --queries"):
            rescreen10.query_keep_mask(["a", "b"], "m9-test", compute=False)


def test_the_document_pool_drops_the_rescreened_rows_and_still_returns_n(monkeypatch):
    """"matching pool documents are removed too" -- and the arm still gets the `n` documents it
    asked for, because the draw takes a margin and trims after the removal."""
    import data10 as _D

    def fake_pool_rows(k, seed):
        return np.arange(k, dtype=np.int64), {"n_drawn": k, "seed": seed}

    class FakeM9:
        doc_pool_rows = staticmethod(fake_pool_rows)

        @staticmethod
        def row_texts(rows):
            return [f"doc{int(r)}" for r in rows]

    class FakePool:
        @staticmethod
        def build():
            v = np.eye(16, 8, dtype=np.float32) + 0.5
            return None, np.tile(v, (2000, 1)), {}

    monkeypatch.setitem(sys.modules, "data", FakeM9)
    monkeypatch.setitem(sys.modules, "pool", FakePool)
    banned = {0, 1, 2, 5}
    rows, V, meta = CL._screened_doc_pool(8, 0, banned, margin=1.0, floor=8)
    # ROW INDICES, not texts: holding 5,000,000 document strings was a measured 19 GB, and they
    # exist only to be tokenized (`_stream_doc_ids` fetches and drops them per chunk).
    assert list(rows) == [3, 4, 6, 7, 8, 9, 10, 11]
    assert FakeM9.row_texts(rows) == ["doc3", "doc4", "doc6", "doc7", "doc8", "doc9", "doc10",
                                      "doc11"]
    assert meta["n_removed_by_rescreen"] == 4 and len(V) == 8
    # V is now a LAZY DocTargetView, not a materialized array: it must be indexed to get vectors,
    # and every gathered batch is unit-norm.
    assert isinstance(V, CL.DocTargetView)
    gathered = V[np.arange(len(V))]
    assert gathered.shape == (8, 8) and gathered.dtype == np.float32
    assert np.allclose(np.linalg.norm(gathered, axis=1), 1.0, atol=1e-6)
    # and a single batch is unit-norm too, which is how `data10.Stream` consumes it
    assert np.allclose(np.linalg.norm(V[np.array([0, 3, 7])], axis=1), 1.0, atol=1e-6)


def test_the_token_cache_identity_binds_the_M10_RESCREEN(monkeypatch):
    """A corpus screened against a different protected index is a different corpus, and must not
    be served the ids the previous one cached."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(CL, "TOKCACHE", Path(d))
        segs, man = _segs([12]), {"sha256": "abc"}
        for r in (None, {"removed": 709, "protected10": "aa"},
                  {"removed": 709, "protected10": "bb"}):
            CL.tokenize_corpus(_Tok(), segs, man, "bge-small", verbose=False,
                               extra_ident={"rescreen10": r})
        assert len(list(Path(d).iterdir())) == 3


def test_a_missing_holdout_refuses_the_corpus_rather_than_waving_it_through(monkeypatch):
    """A guard that turns itself off when its input disappears protects nothing, and looks exactly
    like a clean pass in the artifact."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(CL, "_HOLDOUT_HASHES", {})
        missing = Path(d) / "nope.jsonl"
        assert CL.holdout_hashes(missing) == set()
        with pytest.raises(SystemExit, match="missing or empty"):
            CL.refuse_holdout_texts(_segs([3]), path=missing)


def test_the_holdout_hash_cache_is_keyed_on_the_files_contents_not_its_name(monkeypatch):
    import os
    import time as _time
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(CL, "_HOLDOUT_HASHES", {})
        p = _jsonl(d, "h.jsonl", [{"text": "one", "form": "claim"}])
        assert len(CL.holdout_hashes(p)) == 1
        _jsonl(d, "h.jsonl", [{"text": "one", "form": "claim"},
                              {"text": "two", "form": "claim"}])
        os.utime(p, (_time.time() + 10, _time.time() + 10))
        assert len(CL.holdout_hashes(p)) == 2, "a hold-out extended after a first read"


def test_a_training_document_stream_refuses_without_the_rescreen_mask(monkeypatch):
    """The enforcement, not just the helper: `build_doc_stream` asks `rescreen10` for the mask
    before it draws a single document."""
    import rescreen10
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(rescreen10, "CACHE", Path(d))
        monkeypatch.setattr(rescreen10, "protected_ident", lambda: {"version": "test"})
        monkeypatch.setattr(rescreen10, "doc_pool_ident", lambda: {"n": 3})
        called = {"drew": False}
        monkeypatch.setattr(CL, "_screened_doc_pool",
                            lambda *a, **k: called.update(drew=True) or ([], None, {}))
        with pytest.raises(SystemExit, match="rescreen10.py --documents"):
            CL.build_doc_stream(4, _Tok(), verbose=False)
        assert called["drew"] is False, "it refused BEFORE drawing"


# ============================================================== assemble_arm (the mandatory path)

def test_assemble_arm_refuses_a_source_list():
    """The ONE thing a launcher may never pass: a source list bypasses every guard registered
    against an ARM (the cut, the 12-form check, the masks)."""
    with pytest.raises(SystemExit, match="never a source list"):
        CL.assemble_arm(["harvest"], _Tok(), "bge-small", registry={"anchor_aliases": {}})


def test_resolve_arm_name_resolves_an_alias_and_refuses_a_prose_one():
    reg = {"anchor_aliases": {"anchor": "ANCHOR",
                              "F-winner": "the winner of contrast F1 at its 20M checkpoint"}}
    assert CL.resolve_arm_name("A1", reg) == "A1"
    assert CL.resolve_arm_name("anchor", reg) == "ANCHOR"
    with pytest.raises(SystemExit, match="does not resolve"):
        CL.resolve_arm_name("F-winner", reg)
    with pytest.raises(SystemExit, match="not a registered arm"):
        CL.resolve_arm_name("not-a-real-arm", reg)


def test_assemble_arm_refuses_an_uncut_cut_arm(monkeypatch):
    """A3 is a registered cut arm: `assemble_arm` never accepts `allow_uncut` at all."""
    with tempfile.TemporaryDirectory() as d:
        _fake_corpus(monkeypatch, d)
        monkeypatch.setattr(CL, "data_cut_count", lambda registry=None: None)
        with pytest.raises(SystemExit, match="registered cut arm"):
            CL.assemble_arm("A3", _Tok(), "bge-small", registry={"anchor_aliases": {}},
                            verbose=False)


def test_screened_doc_pool_refuses_an_empty_or_missing_ban_set_as_mask_missing():
    """An empty ban set is indistinguishable from a wiring bug that never actually screened
    anything, so it is treated exactly like a missing mask -- refused, not "nothing to remove"."""
    with pytest.raises(SystemExit, match="mask missing"):
        CL._screened_doc_pool(4, 0, set())
    with pytest.raises(SystemExit, match="mask missing"):
        CL._screened_doc_pool(4, 0, None)


def test_form_balanced_stream_requires_all_forms_when_asked_and_names_the_missing_ones():
    """The old check tested only forms already in `self.present`, built FROM the forms that
    survived `np.unique` -- so a form missing entirely could never trigger it."""
    n = 40
    fid = np.array([CL.FORM_ID["title"]] * (n // 2) + [CL.FORM_ID["claim"]] * (n // 2))
    with pytest.raises(ValueError, match="keyword"):
        CL.FormBalancedStream(_ids(n), _targets(n), fid, pad_id=0, batch_size=8, seed=0,
                              balanced=True, require_forms=CL.FORMS)
    # unchanged when nothing is required
    st = CL.FormBalancedStream(_ids(n), _targets(n), fid, pad_id=0, batch_size=8, seed=0,
                               balanced=True, require_forms=None)
    assert len(st) > 0


def _assemble_arm_mocks(monkeypatch, calls, guard=None):
    """Stub out everything `assemble_arm` calls except its own orchestration logic."""
    class FakeStream:
        def __init__(self, tag):
            self.ids = tag

    def fake_bqs(name, tok, student, *, batch_size, seed, balanced, max_len, prefix, allow_uncut,
                require_forms, verbose, registry=None, consumed=None):
        calls["build_query_stream"] = dict(name=name, allow_uncut=allow_uncut,
                                           require_forms=require_forms, registry=registry)
        if consumed is not None:
            import rescreen10 as _R
            for seg in getattr(_R, "QUERY_SEGMENTS", ()):
                consumed[seg] = np.zeros(0, dtype=bool)
        return FakeStream("q"), {"n_rows": 10}

    def fake_bds(n, tok, *, batch_size, seed, max_len, allow_unscreened, verbose, consumed=None):
        calls["build_doc_stream"] = dict(n=n, allow_unscreened=allow_unscreened)
        if consumed is not None:
            consumed["documents"] = np.zeros(0, dtype=np.int64)
        return FakeStream("d"), {"n": n}

    def fake_guard(q_ids, d_ids, skip=False):
        calls["guard"] = (q_ids, d_ids)
        if guard:
            return guard(q_ids, d_ids)
        return {"checked": True, "collisions": 0}

    class FakeRescreen:
        @staticmethod
        def load_report():
            return {"ok": True}

        @staticmethod
        def protected_ident():
            return {"v": 1}

        @staticmethod
        def query_keep_mask(texts, name, compute=False):
            return np.zeros(0, dtype=bool), {}

        @staticmethod
        def doc_banned_rows(compute=False, verbose=False):
            return np.zeros(0, dtype=np.int64), {}

        @staticmethod
        def validate(report, masks):
            calls["validate"] = (report, masks)

    monkeypatch.setattr(CL, "build_query_stream", fake_bqs)
    monkeypatch.setattr(CL, "build_doc_stream", fake_bds)
    monkeypatch.setattr(CL, "guard_cross_role", fake_guard)
    monkeypatch.setattr(CL, "_m9_segments", lambda screen=True: [])
    monkeypatch.setitem(sys.modules, "rescreen10", FakeRescreen)


def test_assemble_arm_never_passes_allow_uncut_or_allow_unscreened(monkeypatch):
    calls = {}
    _assemble_arm_mocks(monkeypatch, calls)
    bf, man = CL.assemble_arm("A1", _Tok(), "bge-small", registry={"anchor_aliases": {}},
                              verbose=False)
    bqs = calls["build_query_stream"]
    assert (bqs["name"], bqs["allow_uncut"], bqs["require_forms"]) == ("A1", False, None)
    assert bqs["registry"] is not None, "the resolved registry must reach the corpus/cut lookup"
    assert calls["build_doc_stream"]["allow_unscreened"] is False
    assert calls["validate"][0] == {"ok": True}
    assert man["arm"] == "A1" and man["rescreen10_report_validated"] is True
    assert callable(bf)


def test_assemble_arm_requires_all_12_forms_for_the_anchor(monkeypatch):
    calls = {}
    _assemble_arm_mocks(monkeypatch, calls)
    CL.assemble_arm("ANCHOR", _Tok(), "bge-small", registry={"anchor_aliases": {}}, verbose=False)
    assert calls["build_query_stream"]["require_forms"] == CL.FORMS
    calls.clear()
    # A4 resolves to ANCHOR via `anchor_aliases` FIRST (item A), so the alias must be present here
    # -- `resolve_arm_name` never checks "A4" itself, which the registry marks untrained.
    CL.assemble_arm("A4", _Tok(), "bge-small",
                    registry={"anchor_aliases": {"A4": "ANCHOR"}}, verbose=False)
    assert calls["build_query_stream"]["require_forms"] == CL.FORMS
    assert calls["build_query_stream"]["name"] == "ANCHOR"


def test_assemble_arm_runs_the_cross_role_guard_and_propagates_a_collision(monkeypatch):
    calls = {}

    def collide(q_ids, d_ids):
        raise SystemExit("REFUSED: 1 student inputs appear in BOTH the query and the "
                         "document role")

    _assemble_arm_mocks(monkeypatch, calls, guard=collide)
    with pytest.raises(SystemExit, match="BOTH the query and the document role"):
        CL.assemble_arm("A1", _Tok(), "bge-small", registry={"anchor_aliases": {}}, verbose=False)
    assert calls["guard"] == ("q", "d")


# ---------------------------------------------------------------- item A: the registry-driven arm

def test_resolve_arm_name_a4_resolves_to_anchor_via_the_real_registry():
    assert CL.resolve_arm_name("A4") == "ANCHOR"


def test_resolve_arm_name_refuses_arms_that_are_cut_or_never_trained():
    with pytest.raises(SystemExit, match="CUT"):
        CL.resolve_arm_name("C-M9init")
    with pytest.raises(SystemExit, match="CUT"):
        CL.resolve_arm_name("F-MiniLM-L12")


def test_assemble_arm_reads_family_bs_own_mix_pattern_from_the_registry(monkeypatch):
    """B-100/0 and B-50/50 carry their OWN `pattern` in the real registry, encoded as 'kQ[+mD]'
    ('4Q', '2Q+2D') rather than the '100/0'/'50/50' string a stream wants -- translated by
    counting, not hand-copied."""
    calls = {}
    _assemble_arm_mocks(monkeypatch, calls)
    _bf, man = CL.assemble_arm("B-100/0", _Tok(), "bge-small", verbose=False)
    assert man["pattern"] == "100/0" and man["pattern_source"]["raw"] == "4Q"
    calls.clear()
    _bf, man = CL.assemble_arm("B-50/50", _Tok(), "bge-small", verbose=False)
    assert man["pattern"] == "50/50" and man["pattern_source"]["raw"] == "2Q+2D"


def test_assemble_arm_refuses_caller_overrides_of_registry_knobs():
    """`n_docs`, `pattern` and `balanced` are removed from the signature entirely: every
    data-affecting knob comes from the registry, not the caller (item A)."""
    for kw in ({"n_docs": 10}, {"pattern": "50/50"}, {"balanced": False}):
        with pytest.raises(TypeError):
            CL.assemble_arm("A1", _Tok(), "bge-small", registry={"anchor_aliases": {}}, **kw)


# ==================================================================== held-out document ids ----

def test_a_training_row_with_a_held_out_document_id_is_refused(monkeypatch):
    """`_rows_from_jsonl` used to drop the `doc`/`seed_id` field entirely, so a query harvested or
    generated from a held-out document was checked only by TEXT -- a held title and its sibling
    held heading are different strings from the same document."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(CL, "TOKCACHE", Path(d))
        rows = [{"text": "a safe row", "form": "claim", "doc": "doc-safe"},
               {"text": "an unsafe row", "form": "claim", "doc": "doc-held"}]
        p = _jsonl(d, "gen.jsonl", rows)
        monkeypatch.setitem(CL.SOURCES, "fakegen", {"kind": "jsonl", "path": p, "what": "t"})
        monkeypatch.setattr(CL, "held_out_doc_ids", lambda path=None: {"doc-held"})

        class Cache:
            def rows_for(self, texts):
                return np.arange(len(texts))

            def vecs(self):
                return np.ones((2, 8), dtype=np.float16)

        import targets10
        monkeypatch.setattr(targets10, "TargetCache", lambda *a, **k: Cache())
        with pytest.raises(SystemExit, match="FORMS-12 hold-out"):
            CL.load_segments(["fakegen"], verbose=False)


def test_rows_from_jsonl_with_ids_is_opt_in_and_reads_doc_or_seed_id():
    with tempfile.TemporaryDirectory() as d:
        rows = [{"text": "a", "form": "claim", "doc": "d1"},
               {"text": "b", "form": "claim", "seed_id": "s2"},
               {"text": "c", "form": "claim"}]
        p = _jsonl(d, "g.jsonl", rows)
        # default: unchanged 2-tuple
        assert CL._rows_from_jsonl(p) == (["a", "b", "c"], ["claim"] * 3)
        texts, forms, ids = CL._rows_from_jsonl(p, with_ids=True)
        assert ids == ["d1", "s2", None]


def test_a_harvest_row_without_doc_is_refused_by_row_index():
    """Provenance is required for harvest (`doc`) and generated (`seed_id`) rows -- without it the
    held-out-document check silently treats the row as clean (item C). `m9-pool` and PAQ carry no
    such field and are exempt by source (no `require_id` on their `SOURCES` entries)."""
    with tempfile.TemporaryDirectory() as d:
        rows = [{"text": "a", "form": "claim", "doc": "d1"},
               {"text": "b", "form": "claim"}]
        p = _jsonl(d, "harvest.jsonl", rows)
        with pytest.raises(SystemExit, match=r"row 1 carries no usable 'doc'"):
            CL._rows_from_jsonl(p, with_ids=True, require_id="doc")
        # unaffected when the source carries no requirement (PAQ / m9-pool)
        CL._rows_from_jsonl(p, with_ids=True)

        grows = [{"text": "a", "form": "claim", "seed_id": "s1"},
                {"text": "b", "form": "claim", "doc": "not-what-generated-uses"}]
        gp = _jsonl(d, "generated.jsonl", grows)
        with pytest.raises(SystemExit, match=r"row 1 carries no usable 'seed_id'"):
            CL._rows_from_jsonl(gp, with_ids=True, require_id="seed_id")


def test_held_out_doc_ids_reads_the_forms12_files_own_doc_ids(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        p = _jsonl(d, "forms12.jsonl", [{"text": "x", "form": "claim", "doc": "h1"},
                                        {"text": "y", "form": "title", "doc": "h2"},
                                        {"text": "z", "form": "title", "doc": "h1"}])
        assert CL.held_out_doc_ids(p) == {"h1", "h2"}
        missing = Path(d) / "nope.jsonl"
        assert CL.held_out_doc_ids(missing) == set()


# ============================================================= holdout cache keyed by digest ----

def test_holdout_hashes_is_keyed_by_content_not_size_and_mtime(monkeypatch):
    """A rewrite that lands at the same byte length, within the same wall-clock second, is
    invisible to a size+mtime key but is a different corpus."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(CL, "_HOLDOUT_HASHES", {})
        p = _jsonl(d, "h.jsonl", [{"text": "aaaa", "form": "claim"}])
        hs1 = CL.holdout_hashes(p)
        assert CL.text_hash("aaaa") in hs1
        p.write_text(json.dumps({"text": "bbbb", "form": "claim"}) + "\n")
        hs2 = CL.holdout_hashes(p)
        assert CL.text_hash("bbbb") in hs2 and CL.text_hash("aaaa") not in hs2


# ================================================================= the tokenizer identity ----

def test_tokenizer_identity_catches_a_hidden_max_length_the_attribute_does_not(monkeypatch):
    """A `max_length` key beside `model_max_length` in `tokenizer_config.json` is not loaded onto
    the tokenizer object by `transformers` at all -- the attribute alone is blind to it, and it is
    the exact bug that disqualified both MiniLM ONNX exports (`m10/CODEMAP.md` pitfall 5)."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        for dd, ml in ((d1, 128), (d2, 512)):
            (Path(dd) / "tokenizer_config.json").write_text(json.dumps({"model_max_length": ml}))

        class Tok:
            vocab_size = 100

            def __init__(self, path):
                self.name_or_path = path
                # both instances report the SAME (default) attribute value
                self.model_max_length = 1_000_000_000_000

        a, b = CL.tokenizer_ident(Tok(d1)), CL.tokenizer_ident(Tok(d2))
        assert a["model_max_length"] == b["model_max_length"]
        assert a["tokenizer_config_sha256"] != b["tokenizer_config_sha256"]
        assert a["tokenizer_config_sha256"] is not None


def test_arm_doc_count_is_the_document_example_count_not_a_fixed_draw():
    """A 5M arm at 75/25 presents 1.25M document examples; drawing fewer documents than that
    repeats them. M9 drew from every eligible pool row."""
    import corpus_loader as C
    assert C.arm_doc_count({"dose_examples": 5_000_000}, "75/25") == 1_250_000
    assert C.arm_doc_count({"dose_examples": 20_000_000}, "50/50") == 10_000_000
    assert C.arm_doc_count({"dose_examples": 5_000_000}, "100/0") == 32


def test_a_blank_provenance_id_is_refused_not_just_a_missing_one(tmp_path):
    """Codex pass 5: `{"doc": ""}` and `{"seed_id": " "}` passed the None check and could never
    match the held-document set, so a query from a held document could train."""
    import json
    import corpus_loader as C
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps({"text": "a title", "form": "title", "doc": "  "}) + "\n")
    try:
        C._rows_from_jsonl(p, with_ids=True, require_id="doc")
    except SystemExit as e:
        assert "usable 'doc'" in str(e)
    else:
        raise AssertionError("a blank doc id was accepted")


def test_assemble_arm_validates_the_masks_the_streams_consumed_not_a_reload(monkeypatch):
    """Codex pass 5: validation reloaded the masks from the cache; a swap between the load that
    built the stream and the reload would pass. Now the builders record the exact arrays they
    used and `validate` receives those; a builder that records nothing is refused."""
    import corpus_loader as C
    import rescreen10 as R
    calls = {}

    class FakeStream:
        ids = [[1, 2]]

    def fake_q(name, tok, student, **kw):
        assert "consumed" in kw and kw["consumed"] is not None
        for seg in R.QUERY_SEGMENTS:
            kw["consumed"][seg] = f"mask-{seg}"
        return FakeStream(), {}

    def fake_d(n, tok, **kw):
        kw["consumed"]["documents"] = "mask-docs"
        return FakeStream(), {}

    monkeypatch.setattr(C, "build_query_stream", fake_q)
    monkeypatch.setattr(C, "build_doc_stream", fake_d)
    monkeypatch.setattr(C, "guard_cross_role", lambda a, b: {"checked": True, "collisions": 0})
    monkeypatch.setattr(R, "load_report", lambda: {"report": True})
    monkeypatch.setattr(R, "protected_ident", lambda: "pid")
    monkeypatch.setattr(R, "validate", lambda rep, masks: calls.setdefault("masks", masks))
    monkeypatch.setattr(C.D, "batch_fn", lambda q, d, pattern: "bf")

    class Tok:
        pad_token_id = 0

    C.assemble_arm("A1", Tok(), "bge-small", verbose=False)
    assert calls["masks"]["documents"] == "mask-docs"
    assert all(calls["masks"][s] == f"mask-{s}" for s in R.QUERY_SEGMENTS)

    # a builder that records nothing must be refused, never validated from a reload
    monkeypatch.setattr(C, "build_doc_stream", lambda n, tok, **kw: (FakeStream(), {}))
    try:
        C.assemble_arm("A1", Tok(), "bge-small", verbose=False)
    except SystemExit as e:
        assert "did not record the masks" in str(e)
    else:
        raise AssertionError("a stream with no recorded mask was validated")


def test_arms_inheriting_the_anchor_corpus_are_cut_like_the_anchor():
    """Codex pass 6: F/G/B/E/D arms train on 'A4, the CUT corpus' but were never in
    data_cut.applies_to by name, so they would have loaded the uncut A4 sources."""
    import corpus_loader as C
    for arm in ("B-50/50", "F-bge-small", "G-MLP", "D-NORM", "E-bs128"):
        assert C.is_cut_corpus(arm), arm
    assert C.is_cut_corpus("A3") and C.is_cut_corpus("ANCHOR")
    assert not C.is_cut_corpus("A1"), "A1 is the M9 pool and is registered as NOT cut"


def test_the_registry_owns_the_batch_size(monkeypatch):
    """Codex pass 6: E-bs128 registers batch 128; a caller default of 32 silently won."""
    import json
    import corpus_loader as C
    reg = json.loads((C.REPO / "m10" / "screen_registry.json").read_text())
    assert C.arm_batch(reg["arms"]["E-bs128"], reg) == 128
    assert C.arm_batch(reg["arms"]["D-NORM"], reg) == 32
    calls = {}
    _assemble_arm_mocks(monkeypatch, calls)
    try:
        C.assemble_arm("E-bs128", _Tok(), "bge-small", batch_size=32, verbose=False)
    except SystemExit as e:
        assert "registered batch 128" in str(e)
    else:
        raise AssertionError("a caller batch of 32 was accepted for E-bs128")


def test_a_generated_row_with_a_blank_doc_uses_its_seed_id_as_provenance(tmp_path):
    """Codex pass 6: the generic doc-first fallback returned " " for a row whose required id was
    seed_id, so a held seed slipped past the membership check."""
    import json
    import corpus_loader as C
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"text": "novel", "form": "title", "seed_id": "held", "doc": " "}) + "\n")
    texts, forms, ids = C._rows_from_jsonl(p, with_ids=True, require_id="seed_id")
    assert ids == ["held"]


# ---- token cache: atomic write, validated read (2026-09-07 WSL crash) -----------------------

def test_token_cache_guard_reads_the_LAST_file_written_not_the_second(tmp_path, monkeypatch):
    """The guard used to read `offs.npy`, the SECOND of three writes. `np.save` is not atomic, so
    a crash partway through writing it left a truncated file that still `exists()` and was served
    as a complete cache -- silently training an arm on a prefix of its corpus. The 2026-09-07 WSL
    crash landed 2 minutes into family F and proved the window is real."""
    import inspect
    src = inspect.getsource(CL.tokenize_segments) if hasattr(CL, "tokenize_segments") else ""
    if not src:
        import re
        whole = (Path(CL.__file__)).read_text()
        m = re.search(r'd = TOKCACHE / hashlib.*?\n    return p', whole, re.S)
        src = m.group(0) if m else whole
    assert 'if cache and (d / "meta.json").exists()' in src, \
        "the cache-hit guard must read meta.json, the last file written"
    assert '_publish_dir(tmp, d)' in src, \
        "the cache directory must be renamed into place, not built in place"
    assert 'CORRUPT' in src, "a cache that fails validation must refuse, not be trained on"


def test_a_truncated_offs_is_refused_by_the_same_predicates_the_loader_uses(tmp_path):
    """The three predicates, exercised on a deliberately truncated cache."""
    n = 1000
    offs = np.arange(n + 1, dtype=np.int64) * 5
    flat = np.zeros(int(offs[-1]), dtype=np.int32)

    def check(offs, flat, n):
        if len(offs) != n + 1:
            return "length"
        if int(offs[-1]) != len(flat):
            return "offs[-1]"
        if offs[0] != 0 or not bool(np.all(np.diff(offs) >= 0)):
            return "monotone"
        return None

    assert check(offs, flat, n) is None, "the intact cache must pass"
    assert check(offs[:500], flat, n) == "length", "a crash mid-offs write"
    assert check(offs, flat[:10], n) == "offs[-1]", "a crash mid-flat write"
    scrambled = offs.copy(); scrambled[10] = 0
    assert check(scrambled, flat, n) == "monotone", "a non-monotone packing"


def test_the_shipped_F_token_cache_is_intact():
    """The cache family F wrote 2 minutes before the crash. It happened to complete -- verified
    here rather than assumed, because the arm is being restarted onto it."""
    d = Path("/home/dylan/asymetric-dual-encoders/work/m10tok/dfc8d12618876554")
    if not (d / "meta.json").exists():
        pytest.skip("that cache directory is gone")
    meta = json.loads((d / "meta.json").read_text())
    flat = np.load(d / "flat.npy", mmap_mode="r")
    offs = np.load(d / "offs.npy")
    assert len(offs) == meta["n"] + 1 == 2_651_573
    assert int(offs[-1]) == len(flat)
    assert offs[0] == 0 and bool(np.all(np.diff(offs) >= 0))
    assert int(np.diff(offs).max()) <= meta["max_len"]


# ---- document targets are gathered per batch, never materialized (2026-09-07 WSL kills) -----

def test_DocTargetView_does_not_materialize_the_pool():
    """`_screened_doc_pool` used to build `n x 1024` fp32 up front -- 19.1 GiB at the registered
    5,000,000 documents, which with the texts and id arrays reached a measured 23.8 GiB RSS and
    took WSL (26 GB of a 31.8 GB host) down twice. The view must hold no per-row float storage."""
    store = np.tile(np.eye(4, 8, dtype=np.float32) + 0.5, (250, 1))     # 1000 x 8
    rows = np.arange(0, 1000, 2)
    v = CL.DocTargetView(store, rows)
    assert len(v) == 500 and v.shape == (500, 8)
    # the view's own arrays are the row index and nothing else
    assert v.rows.nbytes == rows.astype(np.int64).nbytes
    assert v.store is store, "the view must reference the pool store, not a copy of it"
    # gathering is what normalizes, and only the gathered rows are touched
    got = v[np.array([0, 1, 499])]
    assert got.shape == (3, 8) and np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-6)


def test_DocTargetView_matches_the_old_eager_normalisation():
    """Same numbers as `V = asarray(vecs[rows], f32); V / norm(V)`, gathered instead of stored."""
    rng = np.random.default_rng(3)
    store = rng.normal(size=(400, 16)).astype(np.float32)
    rows = np.sort(rng.choice(400, 120, replace=False))
    eager = np.asarray(store[rows], dtype=np.float32)
    eager = eager / np.maximum(np.linalg.norm(eager, axis=1, keepdims=True), 1e-12)
    lazy = CL.DocTargetView(store, rows)[np.arange(120)]
    assert np.allclose(lazy, eager, atol=1e-6)


def test_DocTargetView_REFUSES_a_zero_target_on_the_BATCH_that_would_carry_it():
    """The old path floored a ~zero norm with `maximum(norm, 1e-12)`, serving it as a huge unit
    vector. Validation is PER BATCH, like `TargetView`: an earlier version checked all rows at
    construction, which read the whole store through the page cache and cost +12 GB on a
    1,000,000-row probe -- a worse bug than the one it was guarding."""
    store = np.ones((50, 8), dtype=np.float32)
    store[37] = 0.0
    v = CL.DocTargetView(store, np.arange(50))          # construction is free, touches nothing
    v[np.array([0, 1, 36])]                              # a clean batch is served
    with pytest.raises(SystemExit, match="never reach a trainer"):
        v[np.array([36, 37, 38])]                        # the batch carrying it is refused
    with pytest.raises(SystemExit, match="never reach a trainer"):
        v[np.arange(50)]


def test_DocTargetView_construction_reads_no_target_rows():
    """Construction must not touch the store: that is the whole point."""
    class CountingStore:
        shape = (100, 8)
        def __init__(self): self.reads = 0
        def __getitem__(self, i):
            self.reads += 1
            return np.ones((len(np.atleast_1d(i)), 8), dtype=np.float32)
    st = CountingStore()
    v = CL.DocTargetView(st, np.arange(100))
    assert st.reads == 0, "construction read the store"
    v[np.array([0, 1])]
    assert st.reads == 1


def test_stream_doc_ids_never_holds_the_whole_corpus(monkeypatch):
    """The document texts are ~19 GB as Python strings and exist only to be tokenized. Streaming
    must fetch, tokenize and DROP them per chunk -- so `row_texts` is called once per chunk with
    only that chunk's rows, never once with all of them."""
    calls = []

    class FakeM9:
        @staticmethod
        def row_texts(rows):
            calls.append(len(rows))
            return [f"d{int(r)}" for r in rows]

    monkeypatch.setitem(sys.modules, "data", FakeM9)
    class FakeTok:
        def __call__(self, texts, truncation=None, max_length=None, add_special_tokens=None):
            return {"input_ids": [[7, 8, 9] for _ in texts]}

    rows = np.arange(250)
    # cache=False: this test is about STREAMING, and it must not read or write the real token
    # cache -- it did on its first run and then hit its own cache on the second, so `row_texts`
    # was never called and the assertions below read `[]`.
    ids, n = CL._stream_doc_ids(rows, tok=FakeTok(), prefix="doc: ", max_len=512, chunk=100,
                                verbose=False, cache=False)
    assert n == 250 and len(ids) == 250
    assert calls == [100, 100, 50], f"row_texts must be called per chunk, got {calls}"
    assert max(calls) == 100, "no call may ever request the whole corpus"
    # PACKED, not a list of 250 small arrays ("M9's pitfall 14"). The list cost +4,546 MB at one
    # million documents; do NOT read that as a per-row rate -- the successor memmap path was
    # measured at BOTH 200k and 1M and the residue is a per-chunk constant, not per-row, so the
    # "25.7 GiB at 5M" this comment used to claim was a bad one-point extrapolation (LEDGER
    # 2026-09-07). The structural assertions below are the guard; no projected number is.
    assert isinstance(ids, CL.PackedIds)
    assert ids.flat.dtype == np.int32 and len(ids.offs) == 251
    assert int(ids.offs[-1]) == len(ids.flat)
    assert np.all(np.diff(ids.offs) >= 0) and hasattr(ids, "lengths")
    assert list(ids[0]) == [7, 8, 9]


def test_doc_id_cache_is_atomic_validated_and_keyed_on_the_ROW_DRAW(monkeypatch, tmp_path):
    """Same contract as the query-side token cache: renamed into place, guarded on the last file
    written, validated on read, and a DIFFERENT row draw must not be served another draw's ids."""
    class FakeM9:
        @staticmethod
        def row_texts(rows):
            return [f"d{int(r)}" for r in rows]

    class FakeTok:
        def __call__(self, texts, truncation=None, max_length=None, add_special_tokens=None):
            return {"input_ids": [[7, 8, 9] for _ in texts]}
        name_or_path = "fake"
        vocab_size = 9
        model_max_length = 512
        truncation_side = "right"
        def __len__(self): return 9

    monkeypatch.setitem(sys.modules, "data", FakeM9)
    monkeypatch.setattr(CL, "TOKCACHE", tmp_path)
    monkeypatch.setattr(CL, "tokenizer_ident", lambda t: {"class": "Fake"})

    rows_a = np.arange(60)
    ids, n = CL._stream_doc_ids(rows_a, FakeTok(), "doc: ", 512, chunk=25, verbose=False)
    assert n == 60 and len(ids) == 60 and isinstance(ids, CL.PackedIds)
    dirs = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    assert len(dirs) == 1 and dirs[0].startswith("doc-")
    d = tmp_path / dirs[0]
    assert (d / "meta.json").exists() and (d / "flat.npy").exists() and (d / "offs.npy").exists()
    assert not (d / "flat.bin").exists(), "the raw append file must not be left behind"
    assert not any(x.name.startswith(dirs[0] + ".partial") for x in tmp_path.iterdir())

    # THE PROPERTY THAT MAKES THE COST CONSTANT: `flat` is memmap-backed, never resident. This is
    # why the doc-stream residue is a per-chunk constant (+3,425 MB at 200k vs +3,454 MB at 1M,
    # LEDGER 2026-09-07) instead of growing with the corpus. An "optimisation" that np.load()s
    # flat.npy without mmap_mode would restore linear growth and reintroduce the crash silently,
    # so assert the mmap rather than trusting a projected megabyte count.
    assert isinstance(ids.flat, np.memmap), f"flat must be a memmap, got {type(ids.flat)}"

    # second call hits the cache -- proven by making row_texts explode if touched
    monkeypatch.setitem(sys.modules, "data", type("Boom", (), {
        "row_texts": staticmethod(lambda r: (_ for _ in ()).throw(AssertionError("re-read")))}))
    again, _ = CL._stream_doc_ids(rows_a, FakeTok(), "doc: ", 512, chunk=25, verbose=False)
    assert list(again[0]) == list(ids[0]) and len(again) == 60
    # the CACHE-HIT load is a SEPARATE np.load from the cache-miss return, so asserting the
    # memmap on one does not cover the other -- and the hit path is the one every arm after the
    # first takes (Codex 2026-09-07).
    assert isinstance(again.flat, np.memmap), f"cache-hit flat must be a memmap, got {type(again.flat)}"

    # a DIFFERENT row draw must miss: same n, same tokenizer, different rows
    monkeypatch.setitem(sys.modules, "data", FakeM9)
    other, _ = CL._stream_doc_ids(np.arange(100, 160), FakeTok(), "doc: ", 512, chunk=25,
                                  verbose=False)
    assert len(other) == 60
    assert len([x for x in tmp_path.iterdir() if x.is_dir()]) == 2, "row draw must key the cache"

    # a corrupt cache is refused, not served
    (d / "offs.npy").unlink()
    np.save(d / "offs.npy", np.arange(10, dtype=np.int64))
    with pytest.raises(SystemExit, match="CORRUPT"):
        CL._stream_doc_ids(rows_a, FakeTok(), "doc: ", 512, chunk=25, verbose=False)


def test_publish_dir_replaces_a_NON_EMPTY_destination(tmp_path):
    """`os.replace` is atomic for files but raises ENOTEMPTY on a non-empty directory, so the
    build-in-a-sibling-and-rename pattern needs this. Both caches hit it the moment a stale or
    corrupt cache directory already existed."""
    d = tmp_path / "cache"
    d.mkdir()
    (d / "stale.npy").write_bytes(b"old")
    tmp = tmp_path / "cache.partial-1"
    tmp.mkdir()
    (tmp / "meta.json").write_text("{}")
    # the bare call this replaced would raise
    with pytest.raises(OSError):
        os.replace(tmp, d)
    CL._publish_dir(tmp, d)
    assert (d / "meta.json").exists() and not (d / "stale.npy").exists()
    assert not tmp.exists(), "the partial directory must be gone once published"


def test_release_arena_is_safe_and_the_doc_build_actually_calls_it(monkeypatch, tmp_path):
    """`release_arena` is what turns the doc build's ~3 GB of anonymous allocator residue back
    into host memory (anon 3,839 -> 788 MB measured, LEDGER 2026-09-07). The residue is NOT live
    data and NOT reclaimable page cache, so nothing else gives it back -- if a refactor drops this
    call the arm silently carries 3 GB it does not need, which is 3 GB Windows does not get, and
    that is the shape of the two 2026-09-07 crashes. Guard the CALL, not the allocator."""
    # never raises, always an int, and reports WHICH outcome it was
    monkeypatch.setattr(CL, "_TRIM", None)          # force re-resolution
    assert isinstance(CL.release_arena(), int) and CL.release_arena() >= 0
    assert CL.release_arena.status in ("freed", "nothing"), CL.release_arena.status

    # no malloc_trim: must degrade to 0 AND say so. `_TRIM` is cached after the calls above, so
    # without resetting it the monkeypatch below is a no-op and the 0 comes from "nothing to
    # trim" -- the assertion would pass for the wrong reason (it did, until this was fixed).
    monkeypatch.setattr(CL, "_TRIM", None)
    monkeypatch.setattr(CL, "_TRIM_REPORTED", False)
    monkeypatch.setattr(CL.ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("musl")))
    assert CL.release_arena() == 0, "must degrade to 0, not raise, without malloc_trim"
    assert CL.release_arena.status == "unsupported", (
        "a silent 0 lets a run whose safety argument rests on trimming look identical to one "
        f"where the mechanism never ran; got {CL.release_arena.status!r}")
    monkeypatch.setattr(CL, "_TRIM", None)

    calls = []
    monkeypatch.setattr(CL, "release_arena", lambda: (calls.append(1), 7)[1])

    class FakeM9:
        @staticmethod
        def row_texts(rows): return [f"d{int(r)}" for r in rows]

    class FakeTok:
        def __call__(self, texts, truncation=None, max_length=None, add_special_tokens=None):
            return {"input_ids": [[7, 8, 9] for _ in texts]}
        name_or_path = "fake"; vocab_size = 9; model_max_length = 512; truncation_side = "right"
        def __len__(self): return 9

    monkeypatch.setitem(sys.modules, "data", FakeM9)
    monkeypatch.setattr(CL, "TOKCACHE", tmp_path)
    monkeypatch.setattr(CL, "tokenizer_ident", lambda t: {"class": "Fake"})

    CL._stream_doc_ids(np.arange(60), FakeTok(), "doc: ", 512, chunk=25, verbose=False)
    assert calls, "the cached doc build must release the allocator residue"
    calls.clear()
    CL._stream_doc_ids(np.arange(60), FakeTok(), "doc: ", 512, chunk=25, verbose=False,
                       cache=False)
    assert calls, "the in-memory path concatenates (holding flat twice) and must release too"


def test_tokenize_corpus_serves_the_cache_HIT_as_a_memmap_too(monkeypatch, tmp_path):
    """The query side has the same hazard as the document side and had NO guard for it: mutating
    its cache-hit `np.load` to an eager read left the whole suite green. The corpus is 5.24M
    texts, so an eager flat array is resident RAM for every arm after the first -- the exact class
    of cost that took the box down twice (Codex 2026-09-07 finding 6, generalized)."""
    class FakeTok:
        def __call__(self, texts, truncation=None, max_length=None, add_special_tokens=None):
            return {"input_ids": [[3, 4] for _ in texts]}
        name_or_path = "fake"; vocab_size = 9; model_max_length = 512; truncation_side = "right"
        def __len__(self): return 9

    monkeypatch.setattr(CL, "TOKCACHE", tmp_path)
    monkeypatch.setattr(CL, "tokenizer_ident", lambda t: {"class": "Fake"})
    seg = types.SimpleNamespace(texts=[f"q{i}" for i in range(40)])
    man = {"sha256": "deadbeef"}

    # The MISS path builds in RAM on purpose here and that is affordable: query texts are ~20
    # tokens, so 5.24M of them is ~419 MB (double during the concatenate). It is the HIT path --
    # taken by every arm after the first -- that must not pull the flat array into memory.
    miss = CL.tokenize_corpus(FakeTok(), [seg], man, "fake", verbose=False)
    assert len(miss) == 40
    hit = CL.tokenize_corpus(FakeTok(), [seg], man, "fake", verbose=False)
    assert len(hit) == 40 and list(hit[0]) == [3, 4]
    assert isinstance(hit.flat, np.memmap), f"cache-HIT flat must be a memmap, got {type(hit.flat)}"


def test_DOC_TEXT_CHUNK_is_bounded_and_a_bigger_chunk_is_refused():
    """`chunk` is the only bound on `row_texts(rows[i:i+chunk])`, which materializes that many
    408-char document strings -- 5M of them is the 19 GB that helped kill the box twice. Nothing
    asserted it: DOC_TEXT_CHUNK could be set to 5,000,000 and the whole suite stayed green
    (Opus 2026-09-07, open item 4)."""
    assert 0 < CL.DOC_TEXT_CHUNK <= 200_000, (
        f"DOC_TEXT_CHUNK={CL.DOC_TEXT_CHUNK:,} -- at ~408 chars/doc this materializes "
        f"~{CL.DOC_TEXT_CHUNK * 408 / 2**30:.1f} GiB of Python strings per chunk")
    with pytest.raises(SystemExit, match="must be in"):
        CL._stream_doc_ids(np.arange(10), tok=None, prefix="", max_len=512,
                           chunk=CL.DOC_TEXT_CHUNK + 1, verbose=False, cache=False)
    with pytest.raises(SystemExit, match="must be in"):
        CL._stream_doc_ids(np.arange(10), tok=None, prefix="", max_len=512, chunk=0,
                           verbose=False, cache=False)


def test_the_lazy_n_produces_the_SAME_cache_key_as_flattening_first():
    """`tokenize_corpus` stopped flattening 5.3M texts before the cache check and now computes
    `n` as `sum(len(s.texts))`. That is a change to a CACHE KEY: if it disagreed with
    `len([t for s in segs for t in s.texts])` by even one, every existing token cache would miss
    and 16 arms would silently re-tokenize. Prove the two agree rather than asserting it -- the
    ragged and empty-segment cases are where a sum and a flatten part company."""
    for segs in ([], [[]], [["a"]], [["a", "b"], []], [[], ["c"]], [["a"], ["b", "c"], ["d"] * 7]):
        objs = [types.SimpleNamespace(texts=list(t)) for t in segs]
        lazy = sum(len(s.texts) for s in objs)
        eager = len([t for s in objs for t in s.texts])
        assert lazy == eager, f"{segs}: lazy {lazy} != eager {eager} -- cache key would change"


def test_assemble_arm_trims_BETWEEN_the_two_stream_builds(monkeypatch):
    """The doc build peaks at ~8.2 GB (measured at the registered 5,000,000) and runs SECOND, so
    the code returns the query build's allocator residue BEFORE that peak rather than after it.
    That ordering is written into a comment and was covered by no test (Codex + Opus 2026-09-07):
    the mutation test on `release_arena` only reaches the two `_stream_doc_ids` branches, so this
    call could be deleted, or moved after the doc build where it cannot help, with every test
    still green."""
    calls = {}
    _assemble_arm_mocks(monkeypatch, calls)
    order = []
    monkeypatch.setattr(CL, "release_arena", lambda: (order.append("trim"), 0)[1])
    real_bqs = CL.build_query_stream
    real_bds = CL.build_doc_stream

    def spy_bqs(*a, **k):
        order.append("query"); return real_bqs(*a, **k)

    def spy_bds(*a, **k):
        order.append("doc"); return real_bds(*a, **k)

    monkeypatch.setattr(CL, "build_query_stream", spy_bqs)
    monkeypatch.setattr(CL, "build_doc_stream", spy_bds)
    CL.assemble_arm("A1", _Tok(), "bge-small", registry={"anchor_aliases": {}}, verbose=False)
    assert order == ["query", "trim", "doc"], (
        f"the trim must sit BETWEEN the two builds, got {order}")


def test_the_doc_id_WRITER_never_accumulates_the_whole_flat_array(monkeypatch, tmp_path):
    """The memmap guards assert the READ artifact. They do NOT constrain the writer: a
    "simplification" that appended each chunk's `parts` to a list, `np.concatenate`d them and
    `np.save`d the result would still return a memmap on read, pass every other guard, and
    reinstate the original crash -- `np.concatenate` holds the flat array TWICE (+3,813 MB at 1M,
    this module's own docstring table). Codex + Opus 2026-09-07 both flagged it.

    The property: `flat.bin` is written ONE CHUNK AT A TIME. Observed by wrapping the file handle
    and recording each `write` size -- NOT by sampling `st_size`, which stays 0 until close
    because the handle is buffered (the first version of this test asserted five zeros)."""
    writes = []

    class FakeM9:
        @staticmethod
        def row_texts(rows): return [f"d{int(r)}" for r in rows]

    class FakeTok:
        def __call__(self, texts, truncation=None, max_length=None, add_special_tokens=None):
            return {"input_ids": [[1, 2, 3] for _ in texts]}
        name_or_path = "fake"; vocab_size = 9; model_max_length = 512; truncation_side = "right"
        def __len__(self): return 9

    class Spy:
        def __init__(self, fh): self._fh = fh
        def write(self, b): writes.append(len(b)); return self._fh.write(b)
        def __enter__(self): return self
        def __exit__(self, *a): return self._fh.__exit__(*a)
        def __getattr__(self, k): return getattr(self._fh, k)

    real_open = Path.open

    def spy_open(self, *a, **k):
        fh = real_open(self, *a, **k)
        return Spy(fh) if self.name == "flat.bin" else fh

    monkeypatch.setitem(sys.modules, "data", FakeM9)
    monkeypatch.setattr(CL, "TOKCACHE", tmp_path)
    monkeypatch.setattr(CL, "tokenizer_ident", lambda t: {"class": "Fake"})
    monkeypatch.setattr(Path, "open", spy_open)

    ids, n = CL._stream_doc_ids(np.arange(100), FakeTok(), "doc: ", 512, chunk=20, verbose=False)
    assert n == 100 and len(ids) == 100
    # 5 chunks of 20 docs x 3 ids x 4 bytes: five equal writes, never one big one
    assert writes == [240] * 5, (
        f"flat.bin must be written one chunk at a time, got {writes} -- a single large write means "
        f"the whole flat array was accumulated in RAM first, which is the crash this path removed")


def test_DocTargetView_accepts_a_SLICE_not_only_an_index_array():
    """`nano10.cov_matrix` reads its width with `doc_vecs[0:1]` before the chunked pass, so a view
    that only accepts index arrays makes D-COV -- the very arm this view was rewritten to make
    affordable -- fail to construct. The 90-step shape smoke caught it at 11/12; without that it
    would have surfaced as the LAST arm of the registered order, after a day of other arms."""
    store = np.arange(40 * 4, dtype=np.float32).reshape(40, 4) + 1.0
    v = CL.DocTargetView(store, np.arange(10, 20))
    assert v.shape == (10, 4) and len(v) == 10
    by_slice = v[0:1]
    by_array = v[np.arange(1)]
    assert by_slice.shape == (1, 4)
    assert np.array_equal(by_slice, by_array), "slice and index-array must agree"
    assert np.array_equal(v[2:5], v[np.array([2, 3, 4])])
    assert np.array_equal(v[:], v[np.arange(10)]), "a bare [:] must gather every row"
    # rows are unit-norm fp32
    assert np.allclose(np.linalg.norm(v[0:3], axis=1), 1.0)


def test_cov_matrix_runs_against_a_DocTargetView():
    """The D-COV path end to end: the chunked accumulation must work on the lazy view, not just on
    a dense array. This is the pairing that was half-fixed -- cov_matrix was made chunked while the
    view still refused the slice it chunks with."""
    import nano10 as N
    store = np.random.default_rng(0).normal(size=(200, 8)).astype(np.float32)
    v = CL.DocTargetView(store, np.arange(200))
    sig = N.cov_matrix(v, chunk=37)
    dense = N.cov_matrix(np.asarray([v[np.array([i])][0] for i in range(200)]), chunk=37)
    assert sig.shape == (8, 8)
    assert np.allclose(sig, dense, atol=1e-6), "view and dense must agree"
