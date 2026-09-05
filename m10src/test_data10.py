"""The batching properties `trainer10`'s resume guarantee depends on."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import data10 as D


def _corpus(n=500, lo=3, hi=60, seed=0):
    rng = np.random.default_rng(seed)
    ids = [rng.integers(1, 900, size=int(rng.integers(lo, hi))).astype(np.int32) for _ in range(n)]
    T = rng.normal(size=(n, 8)).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return ids, T


def test_batch_k_is_a_pure_function_of_k():
    """Resume is exact only if the same step draws the same batch. This is that property."""
    ids, T = _corpus()
    s = D.Stream(ids, T, pad_id=0, batch_size=16, seed=0)
    for k in (0, 1, 7, 33, len(s) + 5):
        a, b = s.batch(k), s.batch(k)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def test_streams_wrap_rather_than_run_out():
    ids, T = _corpus(n=64)
    s = D.Stream(ids, T, pad_id=0, batch_size=16, seed=0)
    assert torch.equal(s.batch(0)[0], s.batch(len(s))[0])


def test_padding_is_to_the_BATCH_maximum_not_the_corpus_maximum():
    """The whole point of bucketing: 400 examples/s vs 890 on this box."""
    ids, T = _corpus(n=512, lo=3, hi=200)
    s = D.Stream(ids, T, pad_id=0, batch_size=32, seed=0)
    widths = [s.batch(k)[0].shape[1] for k in range(len(s))]
    corpus_max = max(len(x) for x in ids)
    assert max(widths) <= corpus_max
    assert np.mean(widths) < 0.6 * corpus_max, (np.mean(widths), corpus_max)


def test_every_row_is_padded_and_masked_consistently():
    ids, T = _corpus()
    s = D.Stream(ids, T, pad_id=0, batch_size=16, seed=0)
    x, m, t = s.batch(3)
    assert x.shape == m.shape and t.shape[0] == x.shape[0]
    for r in range(len(x)):
        k = int(m[r].sum())
        assert (x[r, k:] == 0).all(), "padding must be the pad id"
        assert (m[r, :k] == 1).all() and (m[r, k:] == 0).all()


def test_targets_travel_with_their_rows():
    """A batching bug that shuffles targets against inputs trains on noise and looks fine."""
    ids, T = _corpus(n=256)
    s = D.Stream(ids, T, pad_id=0, batch_size=8, seed=0)
    for k in range(len(s)):
        idx = s.batches[k % len(s.batches)]
        assert np.allclose(s.batch(k)[2].numpy(), T[idx])


def test_batches_are_shuffled_so_length_does_not_track_step():
    """Sorting alone makes every batch a length band in a fixed order, correlating batch content
    with training step. The batch ORDER is shuffled to break that."""
    ids, T = _corpus(n=2048, lo=3, hi=200)
    s = D.Stream(ids, T, pad_id=0, batch_size=32, seed=0)
    w = np.array([s.batch(k)[0].shape[1] for k in range(len(s))], dtype=float)
    r = abs(np.corrcoef(w, np.arange(len(w)))[0, 1])
    assert r < 0.25, f"batch width still tracks step order, r = {r:.3f}"


def test_the_two_streams_advance_independently():
    """Family B re-weights the streams; it must not re-order either of them."""
    qi, qt = _corpus(n=256, seed=1)
    di, dt = _corpus(n=256, seed=2)
    q = D.Stream(qi, qt, 0, batch_size=8, seed=0)
    d = D.Stream(di, dt, 0, batch_size=8, seed=0)
    f = D.batch_fn(q, d)
    seen_q = [f(0, "Q")[0], f(1, "Q")[0], f(2, "D")[0], f(3, "Q")[0]]
    assert torch.equal(seen_q[0], q.batch(0)[0])
    assert torch.equal(seen_q[1], q.batch(1)[0])
    assert torch.equal(seen_q[2], d.batch(0)[0])
    assert torch.equal(seen_q[3], q.batch(2)[0]), "a document step must not consume a query batch"


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
