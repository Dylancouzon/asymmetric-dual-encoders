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
    sh = st.realized_shares(100)
    assert sh == {"ALL": 1.0}
    forms = np.array([CL.FORM_ID["title"]] * 400 + [CL.FORM_ID["claim"]] * 80
                     + [CL.FORM_ID["keyword"]] * 40)
    seen = np.concatenate([st._pick(k)[1] for k in range(60)])
    share_title = float((forms[seen] == CL.FORM_ID["title"]).mean())
    assert 0.70 < share_title < 0.85, share_title       # ~400/520, not 1/3


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
    segs = _segs([50])
    out, rep = CL.apply_data_cut(segs, None)
    assert out is segs and not rep["applied"] and "§0b" in rep["_why"]


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
    assert rest["start_step"] == 34
    for a, b in zip(rest["losses"], ref["losses"][34:]):
        assert abs(a - b) < 1e-6, (a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
