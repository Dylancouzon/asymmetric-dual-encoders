"""The properties an arm's corpus has to have before it is worth spending a GPU-hour on.

The expensive failures this guards are the silent ones: a source that loads but whose targets are
another source's rows, a "balanced" sampler that is balanced only on the sources that happen to be
present, a resumed arm that draws different data, and the FORMS-12 hold-out being trainable.
"""
import json
import sys
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
                        lambda names, head_per_source=None, verbose=True: (
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
    texts, V, meta = CL._screened_doc_pool(8, 0, banned, margin=1.0, floor=8)
    assert texts == ["doc3", "doc4", "doc6", "doc7", "doc8", "doc9", "doc10", "doc11"]
    assert meta["n_removed_by_rescreen"] == 4 and len(V) == 8
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-6)


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
                require_forms, verbose):
        calls["build_query_stream"] = dict(name=name, allow_uncut=allow_uncut,
                                           require_forms=require_forms)
        return FakeStream("q"), {"n_rows": 10}

    def fake_bds(n, tok, *, batch_size, seed, max_len, allow_unscreened, verbose):
        calls["build_doc_stream"] = dict(n=n, allow_unscreened=allow_unscreened)
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
    assert calls["build_query_stream"] == {"name": "A1", "allow_uncut": False,
                                           "require_forms": None}
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
    CL.assemble_arm("A4", _Tok(), "bge-small", registry={"anchor_aliases": {}}, verbose=False)
    assert calls["build_query_stream"]["require_forms"] == CL.FORMS


def test_assemble_arm_runs_the_cross_role_guard_and_propagates_a_collision(monkeypatch):
    calls = {}

    def collide(q_ids, d_ids):
        raise SystemExit("REFUSED: 1 student inputs appear in BOTH the query and the "
                         "document role")

    _assemble_arm_mocks(monkeypatch, calls, guard=collide)
    with pytest.raises(SystemExit, match="BOTH the query and the document role"):
        CL.assemble_arm("A1", _Tok(), "bge-small", registry={"anchor_aliases": {}}, verbose=False)
    assert calls["guard"] == ("q", "d")


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
