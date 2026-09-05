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
    w = torch.ones(32) * 2.0
    assert abs(float(N.loss_cov_weighted(p, t, w)) - 2 * float(sq)) < 1e-5


def test_layer_indices_match_the_registered_spec():
    assert N.LAYERS["bge-small"][3] == (12, 8, 4)
    assert N.LAYERS["MiniLM-L6"][3] == (6, 4, 2)
    assert N.LAYERS["MiniLM-L6"][4] == (6, 4, 2, 1)
    assert N.LAYERS["MiniLM-L12"][4] == (12, 8, 4, 2)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
