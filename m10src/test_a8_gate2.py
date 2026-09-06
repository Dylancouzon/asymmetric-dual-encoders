"""CPU-only tests for a8_gate2's math and path guard. No stella, no network, no cache."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))

import numpy as np
import pytest

import a8_gate2 as G


def _rng():
    return np.random.default_rng(0)


def _unit(rng, n, dim=8):
    v = rng.normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_energy_distance_zero_for_identical_samples():
    x = _unit(_rng(), 40)
    assert G.energy_distance(x, x) == pytest.approx(0.0, abs=1e-9)


def test_energy_distance_zero_for_same_distribution_disjoint_draws():
    rng = _rng()
    x = _unit(rng, 200)
    y = _unit(rng, 200)
    # not exactly zero on a finite sample from the SAME distribution, but small and nonnegative
    d = G.energy_distance(x, y)
    assert d >= -1e-9
    assert d < 0.15


def test_energy_distance_positive_for_shifted_samples():
    rng = _rng()
    x = _unit(rng, 200)
    # a second cluster: unit vectors concentrated near a different mean direction
    shift = rng.normal(size=8)
    shift[0] += 5.0
    y = _unit(rng, 200) + shift
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    d = G.energy_distance(x, y)
    assert d > 0.05


def test_nn_cosine_tiny_case():
    # 3 orthonormal-ish query rows against a pool of 3 known vectors
    pool = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
    q = np.array([[1.0, 0.0]])
    out = G.nn_cosine(q, pool, k_list=(1, 2))
    # nearest neighbour of [1,0] in the pool is itself (cos=1)
    assert out["mean_cos_k1"] == pytest.approx(1.0, abs=1e-6)
    # k=2 mean of the top two cosines: 1.0 and 0.7071
    assert out["mean_cos_k2"] == pytest.approx((1.0 + 0.7071) / 2, abs=1e-4)


def test_nn_cosine_exclude_self():
    # with exclude_self, a vector's own row must not be its own nearest neighbour
    v = np.array([[1.0, 0.0], [0.9995, 0.0316], [0.0, 1.0]])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    out = G.nn_cosine(v, v, k_list=(1,), exclude_self=True)
    # row 0's nearest OTHER row is row 1 (cos close to 1, not the diagonal's exact 1.0 self-match
    # -- both are close here, so just check it's not artificially forced to 1.0 for every row via
    # the diagonal)
    assert out["mean_cos_k1"] < 1.0


def test_msmarco_path_refused_under_train_sources():
    bad = REPO / "work" / "train" / "sources" / "msmarco_queries.dev.tsv"
    with pytest.raises(SystemExit):
        G.assert_not_training_path(bad)


def test_msmarco_ok_path_not_refused():
    ok = REPO / "work" / "m10msmarco" / "msmarco_queries.dev.tsv"
    assert G.assert_not_training_path(ok) == ok
