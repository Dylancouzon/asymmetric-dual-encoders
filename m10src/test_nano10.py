"""Tests for the M10 student. The algebra the whole architecture rests on, proved not asserted."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import nano10 as N


def _tiny(head="linear", n_layers=3):
    """A 2-layer toy BERT, so the tests run on CPU in seconds and exercise the real code path."""
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-BertModel")
    m = object.__new__(N.Nano10)
    torch.nn.Module.__init__(m)
    m.backbone = AutoModel.from_config(cfg)
    m.key, m.n_layers, m.head_kind = "toy", n_layers, head
    m.layers = tuple(range(cfg.num_hidden_layers, cfg.num_hidden_layers - n_layers, -1))
    m.d_in = cfg.hidden_size * n_layers
    m.out_dim, m.max_seq, m.tok = 16, 64, None
    m.head = torch.nn.Linear(m.d_in, 16) if head == "linear" else N.MLPHead(m.d_in, 4, 16)
    return m.eval()


def _batch(b=3, s=7, vocab=90):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (b, s), generator=g)
    mask = torch.ones(b, s, dtype=torch.long)
    mask[0, 5:] = 0                       # a real padded row, or the mask is untested
    mask[1, 3:] = 0
    return ids, mask


def test_pooling_after_a_linear_head_equals_pooling_before_it():
    """THE algebraic claim the warm start and the export both rest on."""
    m = _tiny("linear")
    ids, mask = _batch()
    with torch.inference_mode():
        after = m(ids, mask, normalize=False)
        f = m.features(ids, mask)
        mm = mask.unsqueeze(-1).to(f.dtype)
        before = m.head((f * mm).sum(1) / mm.sum(1).clamp(min=1e-9))
    assert torch.allclose(after, before, atol=1e-5), (after - before).abs().max()


def test_pooling_after_the_MLP_head_does_NOT_equal_pooling_before_it():
    """The other half of the same claim: a check that cannot fail is not a check. G-MLP is why
    the warm start is a three-solve recipe rather than one pooled ridge."""
    m = _tiny("mlp")
    ids, mask = _batch()
    with torch.inference_mode():
        after = m(ids, mask, normalize=False)
        f = m.features(ids, mask)
        mm = mask.unsqueeze(-1).to(f.dtype)
        before = m.head((f * mm).sum(1) / mm.sum(1).clamp(min=1e-9))
    assert not torch.allclose(after, before, atol=1e-4)


def test_padding_is_excluded_from_the_pool():
    """A padded position must not move the output — the failure mode that silently degrades
    every short query in a mixed-length batch."""
    m = _tiny("linear")
    ids, mask = _batch()
    ids2 = ids.clone()
    ids2[0, 5:] = 41                      # change ONLY masked-out positions
    ids2[1, 3:] = 17
    with torch.inference_mode():
        assert torch.allclose(m(ids, mask), m(ids2, mask), atol=1e-6)


def test_output_is_unit_norm():
    m = _tiny("linear")
    ids, mask = _batch()
    with torch.inference_mode():
        v = m(ids, mask)
    assert torch.allclose(v.norm(dim=-1), torch.ones(len(v)), atol=1e-5)


def test_ridge_on_pooled_features_reproduces_the_per_token_head():
    """The warm start solves on pooled features; the head is applied per token. Same map."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 24)).astype(np.float32)
    Y = rng.normal(size=(200, 16)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    m = _tiny("linear")
    m.d_in = 24
    m.head = torch.nn.Linear(24, 16)
    N.warm_start_linear(m, X, Y, lam=1e-3)
    A, _ = N.ridge_head(X, Y, 1e-3)
    with torch.inference_mode():
        got = m.head(torch.from_numpy(X)).numpy()
    want = np.hstack([X, np.ones((len(X), 1), np.float32)]) @ A
    assert np.abs(got - want).max() < 1e-4, np.abs(got - want).max()


def test_the_mix_window_is_exact_not_expected():
    for pat, want in (("100/0", 1.0), ("75/25", 0.75), ("50/50", 0.5)):
        for steps in (4, 40, 400, 4000):
            assert N.window_shares(pat, steps)["q_share"] == want, (pat, steps)


def test_the_schedule_restarts_at_every_cycle_and_ends_at_final():
    total, cycles, peak, final = 300, 3, 1e-4, 1e-5
    lrs = [N.lr_at(s, total, cycles, peak, final) for s in range(total)]
    assert abs(lrs[0] - peak) < 1e-12
    ends = N.cycle_ends(total, cycles)
    assert ends == [99, 199, 299], ends
    for e in ends:
        assert abs(lrs[e] - final) < 1e-9, (e, lrs[e])
        if e + 1 < total:
            assert abs(lrs[e + 1] - peak) < 1e-12, "a cycle must RESTART at peak"


def test_the_objectives_are_distinct_and_leaf_is_the_norm_not_the_square():
    g = torch.Generator().manual_seed(1)
    p = torch.nn.functional.normalize(torch.randn(64, 32, generator=g), dim=-1)
    t = torch.nn.functional.normalize(torch.randn(64, 32, generator=g), dim=-1)
    sq, nm = N.loss_sq_l2(p, t), N.loss_norm_e2(p, t)
    assert abs(float(nm) - float((p - t).norm(dim=-1).mean())) < 1e-6
    assert abs(float(sq) - float(nm)) > 1e-3, "D-NORM must not collapse onto the anchor"
    # D-COV with Σ = 2I must be exactly twice the anchor: the quadratic form's sanity check
    assert abs(float(N.loss_cov_weighted(p, t, torch.eye(32) * 2.0)) - 2 * float(sq)) < 1e-5


def test_layer_indices_match_the_registered_spec():
    assert N.LAYERS["bge-small"][3] == (12, 8, 4)
    assert N.LAYERS["MiniLM-L6"][3] == (6, 4, 2)
    assert N.LAYERS["MiniLM-L6"][4] == (6, 4, 2, 1)
    assert N.LAYERS["MiniLM-L12"][4] == (12, 8, 4, 2)




# ---- D-COV, and the kill and plateau rules ----------------------------------------------------

def test_dcov_is_a_quadratic_form_not_a_diagonal_weight():
    """The registered loss is `(s-t)^T Σ (s-t)`. A diagonal would keep the coordinate basis, and
    the entire claim is that error should be charged along the directions documents differ in."""
    g = torch.Generator().manual_seed(2)
    d = 8
    p = torch.randn(16, d, generator=g)
    t = torch.randn(16, d, generator=g)
    sig = torch.eye(d)
    assert abs(float(N.loss_cov_weighted(p, t, sig)) - float(N.loss_sq_l2(p, t))) < 1e-5
    # a rotation with the SAME diagonal must change the loss, or it is diagonal in disguise
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    lam = torch.linspace(0.2, 2.0, d)
    rot = q @ torch.diag(lam) @ q.T
    diag_only = torch.diag(torch.diagonal(rot))
    assert abs(float(N.loss_cov_weighted(p, t, rot))
               - float(N.loss_cov_weighted(p, t, diag_only))) > 1e-3


def test_cov_matrix_is_unit_trace_symmetric_psd_at_every_alpha():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 12))
    for a in (0.0, 0.1, 0.5, 1.0):
        S = N.cov_matrix(X, alpha=a)
        assert abs(np.trace(S) - 1.0) < 1e-9, (a, np.trace(S))
        assert np.abs(S - S.T).max() < 1e-12
        assert np.linalg.eigvalsh(S).min() > -1e-12
    assert np.abs(N.cov_matrix(X, alpha=1.0) - np.eye(12) / 12).max() < 1e-12


def test_kill_needs_two_CONSECUTIVE_drops_of_its_own_kind():
    kind = lambda i: "mid" if i % 2 == 0 else "end"
    # T2-9: "two consecutive" is read as consecutive IN THE SCHEDULE, so a midpoint and the cycle
    # end after it must BOTH be below their own kind's best.
    evals = [0.50, 0.60, 0.4900, 0.5900, 0.48, 0.58]
    fired, why = N.kill_fires(evals, kind)
    assert fired, why
    # the reading that was NOT taken: two successive MIDPOINTS drop while every cycle end keeps
    # improving. Under the schedule reading this does not fire, and that is the point of T2-9.
    assert not N.kill_fires([0.50, 0.60, 0.4900, 0.61, 0.4800, 0.62], kind)[0]
    # one drop, then a recovery of the same kind, must NOT fire
    assert not N.kill_fires([0.50, 0.60, 0.49, 0.61, 0.505, 0.62], kind)[0]
    # a drop within the 0.0056 tolerance is not a drop
    assert not N.kill_fires([0.50, 0.60, 0.4950, 0.61, 0.4951, 0.62], kind)[0]
    # a non-finite evaluation fires on its own
    assert N.kill_fires([0.5, float("nan")], kind)[0]
    assert N.kill_fires([0.5, None], kind)[0]


def test_kill_compares_within_kind_only():
    """Midpoints sit below cycle ends by construction; comparing across kinds would fire at once."""
    kind = lambda i: "mid" if i % 2 == 0 else "end"
    assert not N.kill_fires([0.40, 0.60, 0.41, 0.61, 0.42, 0.62], kind)[0]


def test_plateau_is_best_to_best_from_cycle_three():
    assert N.plateau_fires([0.50, 0.60, 0.70, 0.80]) == (False, None)
    assert N.plateau_fires([0.50, 0.60, 0.6020]) == (True, 3)      # gain 0.002 < 0.003
    assert N.plateau_fires([0.50, 0.60, 0.6031]) == (False, None)  # gain 0.0031 >= 0.003
    # best-to-best, not last-to-last: a dip then a small recovery still plateaus
    assert N.plateau_fires([0.50, 0.70, 0.60, 0.7010])[0] is True
    # cycles 1 and 2 can never fire it
    assert N.plateau_fires([0.50, 0.50]) == (False, None)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
