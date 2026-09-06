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
    # the resumed record carries the WHOLE arm's history, the restored prefix included
    assert rest["steps_run"] == 6 and len(rest["losses"]) == 40
    for a, b in zip(rest["losses"], ref["losses"]):
        assert abs(a - b) < 1e-6, (a, b)
    for p, q in zip(m1.parameters(), m3.parameters()):
        assert torch.allclose(p, q, atol=1e-6), (p - q).abs().max()


def test_a_resumed_arm_can_still_fire_the_plateau_rule():
    """Codex 2026-09-05 finding 4: `evals`/`cycle_end_evals` were re-initialised on resume, so the
    third cycle-end reading saw one value where the rule needs three and the registered plateau
    could not fire on any resumed run -- i.e. on any real seven-day build."""
    vals = [0.50, 0.60, 0.6005]                   # gain 0.0005 < 0.003 at cycle 3
    st = {"n": 0}

    def ev(model, step, kind):
        if kind != "end":
            return 0.40 + 0.001 * st["n"]
        v = vals[st["n"]]; st["n"] += 1
        return v

    ends = N.cycle_ends(30, 3)
    crash_at = ends[1] + 2                        # two cycle ends already read, then the crash

    def crashing(step, kind):
        if step >= crash_at:
            raise RuntimeError("the box went down")
        return make_batch_fn()(step, kind)

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.pt"
        try:
            T.train_arm(Toy(), crashing, total_steps=30, seed=0, eval_fn=ev,
                        ckpt_path=ck, ckpt_every=1)
            raise AssertionError("the crash did not happen")
        except RuntimeError:
            pass
        ck_extra = torch.load(ck, map_location="cpu", weights_only=False)["extra"]
        assert ck_extra["cycle_end_evals"] == vals[:2], ck_extra["cycle_end_evals"]
        rest = T.train_arm(Toy(), make_batch_fn(), total_steps=30, seed=0, eval_fn=ev,
                           resume_from=ck)
    # the third reading is compared against BOTH earlier cycles, so the plateau fires
    assert rest["cycle_end_evals"] == vals, rest["cycle_end_evals"]
    assert rest["stopped"] == "plateau at cycle 3", rest["stopped"]


def test_a_crash_during_save_leaves_the_previous_checkpoint_intact(monkeypatch):
    """The checkpoint is a build's only recovery point, so `torch.save` must never write onto it
    in place."""
    m = Toy()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as d:
        p = Path(T.save(Path(d) / "c.pt", m, opt, 3))
        good = p.read_bytes()
        real = torch.save

        def dies(obj, fh, *a, **k):
            real(obj, fh, *a, **k)                  # the temp file is half-written, then:
            raise OSError("the box went down mid-save")

        monkeypatch.setattr(torch, "save", dies)
        try:
            T.save(p, m, opt, 4)
        except OSError:
            pass
        assert p.read_bytes() == good, "the previous checkpoint survived"
        assert torch.load(p, map_location="cpu", weights_only=False)["step"] == 3
    monkeypatch.undo()
    with tempfile.TemporaryDirectory() as d:
        p = Path(T.save(Path(d) / "c.pt", m, opt, 3))
        T.save(p, m, opt, 4)
        assert [x.name for x in Path(d).iterdir()] == ["c.pt"], "no temp file left behind"
        assert torch.load(p, map_location="cpu", weights_only=False)["step"] == 4


def test_a_stopped_arm_checkpoints_the_stop_and_the_evaluation_that_caused_it():
    vals = {0: 0.50, 1: 0.60, 2: 0.6005}
    state = {"n": 0}

    def ev(model, step, kind):
        if kind != "end":
            return 0.40 + 0.001 * state["n"]
        v = vals[state["n"]]; state["n"] += 1
        return v

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.pt"
        r = T.train_arm(Toy(), make_batch_fn(), total_steps=30, seed=0, eval_fn=ev, ckpt_path=ck,
                        ckpt_every=1000)
        assert r["stopped"] == "plateau at cycle 3"
        ex = torch.load(ck, map_location="cpu", weights_only=False)["extra"]
        assert ex["stopped"] == "plateau at cycle 3" and ex["cycle_end_evals"] == [0.50, 0.60,
                                                                                  0.6005]


def test_a_resume_that_fails_on_its_very_first_step_still_checkpoints_the_stop():
    """`if stopped and ckpt_path and run_steps:` used to skip the save whenever `run_steps == 0` --
    exactly the resumed-and-immediately-non-finite case, since the break happens before the step
    counter increments. A build that crashes there would lose WHY it stopped."""
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.pt"
        T.train_arm(Toy(), make_batch_fn(), total_steps=10, seed=0, ckpt_path=ck, ckpt_every=10)

        def bad(step, kind):
            ids, mask, tgt = make_batch_fn()(step, kind)
            return ids, mask, tgt * float("inf")

        r = T.train_arm(Toy(), bad, total_steps=20, seed=0, resume_from=ck, ckpt_path=ck,
                        ckpt_every=1000)
        assert r["steps_run"] == 0
        assert r["stopped"] and "non-finite" in r["stopped"]
        ex = torch.load(ck, map_location="cpu", weights_only=False)["extra"]
        assert ex["stopped"] == r["stopped"], "the checkpoint must record the stop even at 0 steps"


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
