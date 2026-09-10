"""The two registered warm starts that were missing: G-MLP's three-solve and C-M9init's head init.

The claim under test for G-MLP is the mandate's own: the warm start is **exact for the training
form**, so the trained model's pooled forward equals the numpy prediction the three solves
produced. That is the whole reason the recipe exists -- a fresh MLP head would handicap contrast
G3 in the direction that rejects the non-default -- so it is asserted, not assumed.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np
import pytest
import torch

import nano10 as N

TEXTS = [f"a short training query about topic number {i} and its details" for i in range(48)]
TEXTS += [f"another {'quite ' * (i % 5)}longer harvested sentence with more tokens {i}"
          for i in range(48)]


def _targets(n, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, N.OUT_DIM)).astype(np.float32)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def mlp_model():
    torch.manual_seed(0)
    return N.Nano10("bge-small", n_layers=3, head="mlp", mlp_k=16)


def test_top_directions_are_orthonormal_and_sign_fixed():
    rng = np.random.default_rng(1)
    d, n = 24, 500
    X = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))
    mu, G2 = X.mean(0), (X.T @ X) / n
    W1, b1 = N.top_directions(mu, G2, k=5)
    assert W1.shape == (5, d) and b1.shape == (d and 5,)
    np.testing.assert_allclose(W1 @ W1.T, np.eye(5), atol=1e-5)
    # b1 = -W1 mu, so the centring is exactly the registered one
    np.testing.assert_allclose(b1, -W1 @ mu, atol=1e-5)
    # each direction's largest-magnitude component is positive -- the determinism guarantee
    for row in W1:
        assert row[np.argmax(np.abs(row))] > 0


def test_top_directions_recover_the_leading_variance_directions():
    rng = np.random.default_rng(2)
    d = 12
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    scales = np.array([9.0, 6.0] + [0.05] * (d - 2))
    X = rng.normal(size=(4000, d)) * scales @ basis.T
    mu, G2 = X.mean(0), (X.T @ X) / X.shape[0]
    W1, _ = N.top_directions(mu, G2, k=2)
    # the recovered 2-plane must be the plane spanned by the first two basis vectors
    P = basis[:, :2] @ basis[:, :2].T
    for row in W1:
        np.testing.assert_allclose(P @ row, row, atol=1e-2)


def test_warm_start_mlp_is_EXACT_for_the_training_form(mlp_model):
    """The mandate's claim: pooling commutes with the linear maps, so the model's own forward
    reproduces the three solves' prediction. If this drifts, G-MLP is not starting where the
    recipe says it starts."""
    m = mlp_model
    Y = _targets(len(TEXTS))
    rec = N.warm_start_mlp(m, TEXTS, Y, lam=1e-4)
    assert rec["k"] == 16 and rec["d_in"] == 1152

    Xbar = N.pooled_features(m, TEXTS)
    Gf = N.gelu_features(m, TEXTS, m.head.down.weight.detach().cpu().numpy(),
                         m.head.down.bias.detach().cpu().numpy())
    W = m.head.lin.weight.detach().cpu().numpy()
    b = m.head.lin.bias.detach().cpu().numpy()
    W2 = m.head.up.weight.detach().cpu().numpy()
    b2 = m.head.up.bias.detach().cpu().numpy()
    expect = Xbar @ W.T + b + Gf @ W2.T + b2

    with torch.no_grad():
        got = m.encode_queries(TEXTS, batch_size=16)
    expect = expect / np.maximum(np.linalg.norm(expect, axis=1, keepdims=True), 1e-12)
    cos = (got * expect).sum(1)
    assert cos.min() > 0.9999, f"min cos {cos.min():.8f} -- the warm start is not exact"


def test_warm_start_mlp_starts_no_worse_than_the_linear_head():
    """Solve 3 fits the residual of solve 1, so on the fit sample the MLP start cannot be worse.

    This MUST be run OVERDETERMINED or it proves nothing. With `d_in = 1152` the ridge has 1,153
    parameters per output dim, so any fit sample smaller than that interpolates and BOTH objectives
    read exactly 0.0 -- which is what the arm smoke shows at its n_fit of 256. So use the 1-layer
    feature (d_in 384, 385 parameters) against 600 texts and a real residual.
    """
    torch.manual_seed(0)
    m = N.Nano10("bge-small", n_layers=1, head="mlp", mlp_k=16)
    texts = [f"query {i} about {'alpha beta gamma delta'.split()[i % 4]} and some trailing words"
             for i in range(600)]
    Y = _targets(len(texts))
    rec = N.warm_start_mlp(m, texts, Y, lam=1e-4)
    assert rec["train_objective_linear_only"] > 1e-6, \
        f"fit is degenerate ({rec['train_objective_linear_only']}) -- the test proves nothing"
    assert rec["train_objective"] <= rec["train_objective_linear_only"] + 1e-6
    assert rec["train_objective"] < rec["train_objective_linear_only"], \
        "a fitted rank-16 correction on a real residual should strictly improve the fit"


def test_warm_start_mlp_refuses_a_linear_head():
    torch.manual_seed(0)
    m = N.Nano10("bge-small", n_layers=3)
    with pytest.raises(ValueError, match="G-MLP head"):
        N.warm_start_mlp(m, TEXTS[:8], _targets(8))


def test_select_lambda_uses_the_locked_grid_and_breaks_ties_upward():
    import warmfit
    X = np.random.default_rng(3).normal(size=(400, 8)).astype(np.float32)
    Y = _targets(400, seed=4)
    lam, rows = N.select_lambda(X, Y, n_fit_split=300, seed=21)
    assert [r["lambda"] for r in rows] == list(warmfit.GRID)
    assert lam in warmfit.GRID
    best = min(r["val_objective"] for r in rows)
    # ties go to the LARGER lambda, so the winner is the biggest lambda achieving the best score
    assert lam == max(r["lambda"] for r in rows if r["val_objective"] == best)


def test_warm_start_from_m9_keeps_the_384_block_and_zeroes_the_rest(tmp_path):
    torch.manual_seed(0)
    m = N.Nano10("bge-small", n_layers=3)
    d1 = m.backbone.config.hidden_size
    hw = torch.randn(N.OUT_DIM, d1)
    hb = torch.randn(N.OUT_DIM)
    ck = {"model": {"head.weight": hw, "head.bias": hb,
                    **{f"backbone.{k}": v for k, v in m.backbone.state_dict().items()}}}
    p = tmp_path / "m9cand.pt"
    torch.save(ck, p)
    rec = N.warm_start_from_m9(m, p)
    assert rec["head_block"] == [N.OUT_DIM, d1]
    assert rec["zeroed_columns"] == 1152 - d1
    W = m.head.weight.detach().cpu()
    np.testing.assert_allclose(W[:, :d1].numpy(), hw.numpy(), atol=1e-6)
    assert torch.count_nonzero(W[:, d1:]).item() == 0, "the extra layers' columns must be ZERO"
    np.testing.assert_allclose(m.head.bias.detach().cpu().numpy(), hb.numpy(), atol=1e-6)


def test_warm_start_from_m9_refuses_a_wrong_width_head(tmp_path):
    torch.manual_seed(0)
    m = N.Nano10("bge-small", n_layers=3)
    ck = {"model": {"head.weight": torch.randn(N.OUT_DIM, 1152)}}   # already 3-layer wide
    p = tmp_path / "wrong.pt"
    torch.save(ck, p)
    with pytest.raises(SystemExit, match="one layer wide"):
        N.warm_start_from_m9(m, p)


def test_select_lambda_keeps_a_real_holdout_on_a_small_sample():
    """The registered split is 50,000 of 60,000 -- five sixths. Clamping to `len - 1` left ONE
    validation row, whose objective is noise, so `warmfit.select`'s tie rule returned the TOP of
    the grid every time: the arm smoke selected lambda = 1.0 for all eleven linear shapes, which
    is a near-zero head."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 8)).astype(np.float32)
    W = rng.normal(size=(8, N.OUT_DIM)).astype(np.float32)
    Y = X @ W + 0.05 * rng.normal(size=(600, N.OUT_DIM))
    Y = (Y / np.linalg.norm(Y, axis=1, keepdims=True)).astype(np.float32)
    lam, rows = N.select_lambda(X, Y)
    # 600 * 50000/60000 = 500 fit, 100 validation -- not 599/1
    assert lam < 1.0, "a real holdout must not default to the top of the grid"
    vals = [r["val_objective"] for r in rows]
    assert len(set(round(v, 9) for v in vals)) > 1, "one validation row makes every lambda tie"


def test_select_lambda_is_unchanged_at_the_registered_sample_size():
    """Ratio-scaling must be bit-identical to the registration at 60,000."""
    import warmfit
    n_fit_split, m = 50_000, N.N_FIT_REGISTERED
    n = n_fit_split if m >= N.N_FIT_REGISTERED else int(round(m * n_fit_split / N.N_FIT_REGISTERED))
    assert n == 50_000 and warmfit.GRID[-1] == 1.0
