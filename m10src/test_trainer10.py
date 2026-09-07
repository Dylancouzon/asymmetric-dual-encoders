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


# ---- §Recipe: train mode, the parameter groups, bf16 autocast (Codex 2026-09-07, 1/2) --------

class ModeSpy(Toy):
    """Records `self.training` at every forward — the warm start leaves the model in eval()."""

    def __init__(self):
        super().__init__()
        self.modes = []
        self.norm = torch.nn.LayerNorm(6)

    def forward(self, ids, mask):
        self.modes.append(self.training)
        return self.norm(super().forward(ids, mask))


def test_the_backbone_trains_in_TRAIN_mode_even_when_the_warm_start_left_it_in_eval():
    """`nano10.pooled_features` calls `model.eval()`, so every warm-started arm reached the loop
    with dropout off and (had it been a real BERT) frozen batch statistics."""
    m = ModeSpy()
    m.eval()                                        # exactly what the warm start leaves behind
    r = T.train_arm(m, make_batch_fn(), total_steps=6, seed=0)
    assert m.modes and all(m.modes), m.modes
    assert m.training is True and r["steps_run"] == 6


def test_an_evaluation_restores_train_mode():
    m = ModeSpy()

    def ev(model, step, kind):
        model.eval()                                # what every real evaluator's encode does
        return 0.4 + 0.01 * step
    T.train_arm(m, make_batch_fn(), total_steps=30, seed=0, eval_fn=ev)
    assert all(m.modes), "a step after an evaluation trained in eval mode"


def test_weight_decay_is_registered_on_dim_gt_1_only():
    m = ModeSpy()
    gs = T.param_groups(m, 0.01)
    assert [g["weight_decay"] for g in gs] == [0.01, 0.0]
    assert all(p.dim() > 1 for p in gs[0]["params"]) and gs[0]["params"]
    assert all(p.dim() <= 1 for p in gs[1]["params"]) and gs[1]["params"]
    # every parameter is in exactly one group
    ids = [id(p) for g in gs for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in m.parameters()) and len(set(ids)) == len(ids)
    # and the loop actually uses them, at the registered betas/eps
    r = T.train_arm(m, make_batch_fn(), total_steps=2, seed=0)
    assert r["weight_decay_on_dim_gt_1"] == 0.01


def test_bf16_autocast_is_requested_on_cuda_and_skipped_on_cpu():
    cuda_ctx, on = T.autocast_for("cuda")
    assert on is True and cuda_ctx.device == "cuda" and cuda_ctx.fast_dtype is torch.bfloat16
    cpu_ctx, off = T.autocast_for("cpu")
    assert off is False
    with cpu_ctx:
        assert not torch.is_autocast_enabled("cuda")
    r = T.train_arm(Toy(), make_batch_fn(), total_steps=2, seed=0, device="cpu")
    assert r["bf16_autocast"] is False


def test_the_loss_is_fp32_even_when_the_forward_is_reduced_precision():
    """The registered form: bf16 forward, fp32 loss. On CPU the cast is the observable half."""
    seen = []

    class Bf16(Toy):
        def forward(self, ids, mask):
            return super().forward(ids, mask).to(torch.bfloat16)

    def loss(pred, tgt):
        seen.append(pred.dtype)
        return ((pred - tgt) ** 2).sum(-1).mean()

    N.LOSSES["_test_dtype"] = loss
    try:
        T.train_arm(Bf16(), make_batch_fn(), total_steps=2, seed=0, loss_name="_test_dtype")
    finally:
        N.LOSSES.pop("_test_dtype")
    assert seen == [torch.float32, torch.float32], seen


# ---- the checkpoint carries the evaluator and its recipe (Codex 2026-09-07, 7/8) -------------

class FamilyEval:
    """A stand-in for `run_arm.CovEval`: per-family macros the contrast step needs."""

    def __init__(self):
        self.records = []

    def __call__(self, model, step, kind):
        n = len(self.records) + 1
        self.n_end = getattr(self, "n_end", 0) + (kind == "end")
        label = f"cycle{self.n_end}" if kind == "end" else f"{kind}{step}"
        self.records.append({"label": label, "step": int(step), "kind": kind,
                             "macro": 0.40 + 0.01 * n, "by_family": {"legal": 0.3 + 0.01 * n}})
        return 0.40 + 0.01 * n

    def state(self):
        return {"records": self.records}

    def restore(self, d):
        self.records = list(d["records"])


def test_a_resume_restores_the_evaluators_own_records():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fmt = str(d / "cycle{cycle}.pt")
        first = FamilyEval()
        T.train_arm(Toy(), make_batch_fn(), total_steps=30, seed=0, eval_fn=first,
                    eval_state=first, cycle_ckpt_fmt=fmt, evals_per_cycle=1)
        assert len(first.records) >= 2
        # a FRESH evaluator resuming from the cycle-2 checkpoint must recover cycles 1 and 2
        second = FamilyEval()
        r = T.train_arm(Toy(), make_batch_fn(), total_steps=30, seed=0, eval_fn=second,
                        eval_state=second, resume_from=d / "cycle2.pt")
        ends = [x for x in second.records if x["kind"] == "end"]
        assert [x["label"] for x in ends[:2]] == ["cycle1", "cycle2"], second.records
        for rec in ends[:2]:
            assert rec["by_family"]["legal"] > 0, "the per-family macro must survive a resume"
        assert r["cycle_end_evals"][:2] == [x["macro"] for x in ends[:2]]


def test_a_resume_refuses_a_checkpoint_written_by_another_recipe():
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ckpt.pt"
        T.train_arm(Toy(), make_batch_fn(), total_steps=10, seed=0, ckpt_path=ck, ckpt_every=10,
                    fingerprint="A1:abc")
        # same fingerprint: allowed
        T.train_arm(Toy(), make_batch_fn(), total_steps=20, seed=0, resume_from=ck,
                    fingerprint="A1:abc")
        for other in ("A2:abc", "A1:def"):
            try:
                T.train_arm(Toy(), make_batch_fn(), total_steps=20, seed=0, resume_from=ck,
                            fingerprint=other)
            except SystemExit as e:
                assert "different recipe" in str(e)
            else:
                raise AssertionError(f"a resume under {other} must be refused")


def test_a_fingerprinted_resume_refuses_a_checkpoint_that_carries_none():
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ckpt.pt"
        T.train_arm(Toy(), make_batch_fn(), total_steps=10, seed=0, ckpt_path=ck, ckpt_every=10)
        try:
            T.train_arm(Toy(), make_batch_fn(), total_steps=20, seed=0, resume_from=ck,
                        fingerprint="A1:abc")
        except SystemExit as e:
            assert "different recipe" in str(e)
        else:
            raise AssertionError("an unfingerprinted checkpoint must not satisfy a fingerprint")
