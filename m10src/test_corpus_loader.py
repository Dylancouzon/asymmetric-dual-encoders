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
    """"texts drawn with replacement within a form" -- a form smaller than a batch still runs."""
    st, forms = _stream({"title": 100, "claim": 3}, batch_size=8)
    idx = np.concatenate([st._pick(k)[1] for k in range(40)])
    assert (forms[idx] == CL.FORM_ID["claim"]).sum() > 0


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
        return ["a", "b"], ["claim", "title"], {"source": name, "sha256": "x", "n_rows": 2,
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
    class TokA(_Tok):
        name_or_path = "BAAI/bge-small-en-v1.5"

        def get_vocab(self):
            return {"a": 0, "b": 1}

    class TokB(TokA):
        def get_vocab(self):
            return {"a": 0, "b": 1, "c": 2}

    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(CL, "TOKCACHE", Path(d))
        segs, man = _segs([20]), {"sha256": "abc"}
        CL.tokenize_corpus(TokA(), segs, man, "bge-small", verbose=False)
        CL.tokenize_corpus(TokB(), segs, man, "bge-small", verbose=False)
        assert len(list(Path(d).iterdir())) == 2, "the vocabulary must be part of the identity"
    assert CL.tokenizer_ident(TokA())["name_or_path"] == "BAAI/bge-small-en-v1.5"


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
