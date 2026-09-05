"""Resume determinism, the mix window under way, and the stop rules firing in a real loop.

A seven-day build that cannot resume EXACTLY is a build whose result depends on when it crashed.
`test_resume_reproduces_an_uninterrupted_run` is the one that matters: it is not "the losses look
similar", it is the same trajectory and the same weights to float tolerance.
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import nano10 as N
import trainer10 as T


class Toy(torch.nn.Module):
    """Stands in for Nano10 with the same call signature, so the loop is tested and not the BERT."""

    def __init__(self, d_in=12, d_out=6):
        super().__init__()
        self.head = torch.nn.Linear(d_in, d_out)

    def forward(self, ids, mask):
        x = ids.float() @ torch.ones(ids.shape[-1], self.head.in_features) / ids.shape[-1]
        return torch.nn.functional.normalize(self.head(x), dim=-1)


def make_batch_fn(n=32, s=5, seed=0):
    """Deterministic in `step`, so two runs see identical data — otherwise resume is untestable."""
    def f(step, kind):
        g = torch.Generator().manual_seed(1000 + step)
        ids = torch.randint(0, 50, (n, s), generator=g)
        mask = torch.ones(n, s, dtype=torch.long)
        tgt = torch.nn.functional.normalize(torch.randn(n, 6, generator=g), dim=-1)
        return ids, mask, tgt
    return f


def test_resume_reproduces_an_uninterrupted_run():
    torch.manual_seed(0)
    m1 = Toy()
    ref = T.train_arm(m1, make_batch_fn(), total_steps=40, seed=0)

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.pt"
        torch.manual_seed(0)
        m2 = Toy()
        part = T.train_arm(m2, make_batch_fn(), total_steps=40, seed=0,
                           ckpt_path=ck, ckpt_every=17)
        # a fresh model, restored from the step-34 checkpoint, must finish identically
        m3 = Toy()
        rest = T.train_arm(m3, make_batch_fn(), total_steps=40, seed=0, resume_from=ck)

    assert part["losses"] == ref["losses"], "the checkpointing run itself must not drift"
    assert rest["start_step"] == 34, rest["start_step"]
    tail = ref["losses"][34:]
    assert len(rest["losses"]) == len(tail)
    for a, b in zip(rest["losses"], tail):
        assert abs(a - b) < 1e-6, (a, b)
    for p, q in zip(m1.parameters(), m3.parameters()):
        assert torch.allclose(p, q, atol=1e-6), (p - q).abs().max()


def test_a_non_finite_loss_stops_the_arm_rather_than_training_on():
    def bad(step, kind):
        ids, mask, tgt = make_batch_fn()(step, kind)
        if step == 5:
            tgt = tgt * float("inf")
        return ids, mask, tgt
    r = T.train_arm(Toy(), bad, total_steps=20, seed=0)
    assert r["stopped"] and "non-finite" in r["stopped"], r["stopped"]
    assert len(r["losses"]) == 6, len(r["losses"])


def test_the_mix_window_holds_across_a_real_run():
    seen = []
    def f(step, kind):
        seen.append(kind)
        return make_batch_fn()(step, kind)
    T.train_arm(Toy(), f, total_steps=40, pattern="50/50", seed=0)
    assert seen.count("Q") == 20 and seen.count("D") == 20, seen.count("Q")


def test_the_plateau_rule_stops_a_flat_curve_at_cycle_three():
    vals = {0: 0.50, 1: 0.60, 2: 0.6005}          # cycle-end readings, gain 0.0005 < 0.003
    state = {"n": 0}
    def ev(model, step, kind):
        if kind != "end":
            return 0.40 + 0.001 * state["n"]
        v = vals[state["n"]]; state["n"] += 1
        return v
    r = T.train_arm(Toy(), make_batch_fn(), total_steps=30, seed=0, eval_fn=ev)
    assert r["stopped"] == "plateau at cycle 3", r["stopped"]


def test_the_examples_per_second_counter_counts_examples():
    r = T.train_arm(Toy(), make_batch_fn(n=32), total_steps=10, seed=0)
    assert r["examples"] == 320, r["examples"]
    assert r["examples_per_s"] > 0


def test_checkpoints_save_the_eager_module_even_behind_a_compiled_wrapper():
    """§T: a compiled wrapper's state_dict carries `_orig_mod.` prefixes and is the wrong shape."""
    class FakeCompiled(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

        def forward(self, *a, **k):
            return self._orig_mod(*a, **k)

    m = Toy()
    wrapped = FakeCompiled(m)
    opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as d:
        p = T.save(Path(d) / "c.pt", wrapped, opt, 7)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert all(not k.startswith("_orig_mod") for k in ck["model"]), list(ck["model"])[:3]
        # and it loads straight into an UNwrapped model, which is what export and parity use
        T.load(p, Toy(), torch.optim.AdamW(Toy().parameters(), lr=1e-4))


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
