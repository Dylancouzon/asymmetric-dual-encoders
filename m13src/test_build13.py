"""What the 200M build controller must get right, and what it must refuse.

CPU only, synthetic streams, a `Toy` model and scripted COV macros — the same shape as
`m10src/test_run_arm.py`. The point is the controller's contract: the 200M pin's arithmetic, the
extension/plateau/kill decisions and their cap and budget gates, resume equivalence across a cycle
boundary and inside an extension, the document policy, and every refusal that stands between a
mistake and a rented GPU. Nothing here trains a BERT, reads a corpus, or touches an evaluation
surface.
"""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m10src"))

import numpy as np
import pytest
import torch

import build13 as BD
import build_lock as BL
import corpus_loader as CL
import data10 as D

REPO = Path(__file__).resolve().parents[1]
TINY_DOSE = 3840            # 120 steps at bs32: cycle ends 39 / 79 / 119
TINY_EXT = 1280             # 40 steps at bs32
BS = 32


# --------------------------------------------------------------------------- the 200M pin -----

def test_the_cycle_table_at_bs32_sums_to_exactly_200M():
    p = BL.cycle_plan(200_000_000, 32)
    assert p["total_steps"] == 6_250_000 and p["divides_exactly"]
    assert [r["examples"] for r in p["cycles"]] == [66_666_656, 66_666_656, 66_666_688]
    assert p["sum_examples"] == 200_000_000


def test_the_cycle_table_at_bs128_sums_to_exactly_200M():
    p = BL.cycle_plan(200_000_000, 128)
    assert p["total_steps"] == 1_562_500 and p["divides_exactly"]
    assert [r["examples"] for r in p["cycles"]] == [66_666_624, 66_666_624, 66_666_752]
    assert p["sum_examples"] == 200_000_000


def test_the_extension_floor_at_bs128_is_recorded():
    """66,700,000 is not a whole number of batches of 128: 521,093 steps = 66,699,904 examples,
    96 fewer. Recorded, never silent (the same rule as `run_arm.schedule`)."""
    e = BL.extension_plan(66_700_000, 128)
    assert (e["steps"], e["examples"], e["examples_dropped"]) == (521_093, 66_699_904, 96)
    assert BL.extension_plan(66_700_000, 32)["examples_dropped"] == 0


def test_a_config_totalling_200_100_000_is_refused(tmp_path):
    """"Do not silently train 200.1M" (m10/M102_LOCK.md): the old 'three x 66.7M' is rounded."""
    cfg = _config(tmp_path, dose=200_100_000)[0]
    with pytest.raises(SystemExit, match="registered build dose is 200,000,000"):
        BL.validate(cfg)


def test_a_dose_that_does_not_divide_by_the_batch_is_refused(tmp_path, monkeypatch):
    cfg = _config(tmp_path, dose=3841)[0]
    monkeypatch.setattr(BL, "DOSE", 3841)
    BL.validate(cfg)                                    # the dose itself is accepted...
    with pytest.raises(SystemExit, match="not a whole number of batches"):
        BL.check_dose(cfg, 32)                          # ...its arithmetic is not


def test_the_cap_formula_prices_the_mandatory_lines_first(tmp_path):
    cfg = _config(tmp_path, reserved_hours=2.0)[0]
    cap = BL.cap_arithmetic(cfg, 1000.0, 2.0)
    hours = cap["mandatory_hours"]
    assert hours["build"] == pytest.approx(200_000_000 / 1000 / 3600, abs=1e-3)
    assert hours["cloud_E_arms"] == pytest.approx(10_000_000 / 1000 / 3600, abs=1e-3)
    assert cap["committed_usd"] == pytest.approx(2.0 * cap["mandatory_hours_total"] + 25.0,
                                                 abs=0.05)
    cycle = 2.0 * 66_700_000 / 1000 / 3600
    assert cap["extension_cycle_usd"] == pytest.approx(cycle, rel=1e-3)
    assert cap["max_extension_cycles"] == int((1000 - cap["committed_usd"]) // cycle)


def test_the_cap_refuses_an_unpriced_reserved_batch(tmp_path):
    cfg = _config(tmp_path)[0]                          # reserved_batch_allowance is null
    with pytest.raises(SystemExit, match="reserved_batch_allowance"):
        BL.cap_arithmetic(cfg, 1000.0, 2.0)


# ------------------------------------------------------------------ the validator's refusals ----

def test_a_data_knob_that_differs_from_the_anchor_is_refused(tmp_path):
    cfg = _config(tmp_path)[0]
    cfg["data_knobs_equal_to_anchor"]["objective"] = "leaf_norm_e2"
    with pytest.raises(SystemExit, match="differ from ANCHOR"):
        BL.validate(cfg)


def test_an_unexpected_knob_in_the_build_arm_entry_is_refused(tmp_path):
    cfg = _config(tmp_path)[0]
    cfg["arm_entry"]["objective"] = "leaf_norm_e2"
    with pytest.raises(SystemExit, match="not among"):
        BL.validate(cfg)


def test_the_document_policy_and_the_uncut_corpus_are_both_required(tmp_path):
    cfg = _config(tmp_path)[0]
    cfg["arm_entry"].pop("documents")
    with pytest.raises(SystemExit, match="50,000,000 unique documents"):
        BL.validate(cfg)
    cfg = _config(tmp_path)[0]
    cfg["arm_entry"]["data_cut"] = "registered"
    with pytest.raises(SystemExit, match="FULL uncut A4"):
        BL.validate(cfg)


def test_a_smoke_only_document_count_is_refused_outside_a_smoke(tmp_path):
    cfg = _config(tmp_path)[0]
    cfg["arm_entry"]["documents"]["n"] = 1000
    BL.validate(cfg, smoke=True)
    with pytest.raises(SystemExit, match="SMOKE-ONLY override"):
        BL.validate(cfg, smoke=False)


def test_a_pending_E1_batch_is_refused_outside_a_smoke(tmp_path):
    """`selected.batch` is 'PENDING' until both E arms have run on the cloud GPU."""
    cfg = _config(tmp_path)[0]
    with pytest.raises(SystemExit, match="E1 verdict is PENDING"):
        BL.resolve_batch(cfg)
    assert BL.resolve_batch(cfg, smoke=True, override=32)[0] == 32
    with pytest.raises(SystemExit, match="--batch may only be passed"):
        BL.resolve_batch(cfg, smoke=False, override=32)


# ------------------------------------------------------------------------ the document policy ----

def test_arm_doc_count_is_unchanged_without_a_policy():
    assert CL.arm_doc_count({"dose_examples": 5_000_000}, "75/25", 32) == 1_250_000
    assert CL.arm_doc_count({"dose_examples": 5_000_000}, "100/0", 32) == 32


def test_the_build_policy_asks_for_every_eligible_document_not_50_million():
    entry = {"dose_examples": 200_000_000,
             "documents": {"policy": "all_eligible_rescreened", "n": None}}
    assert CL.arm_doc_count(entry, "75/25", 32) == CL.ALL_ELIGIBLE
    entry["documents"]["n"] = 512                       # the smoke-only override
    assert CL.arm_doc_count(entry, "75/25", 32) == 512
    with pytest.raises(SystemExit, match="unknown document policy"):
        CL.arm_document_policy({"documents": {"policy": "something-else"}})


def test_the_all_eligible_draw_takes_every_eligible_row_minus_the_rescreen(monkeypatch):
    """The margin-and-trim draw cannot serve `ALL_ELIGIBLE`; the branch reads `n_eligible` off
    `doc_pool_rows`'s own meta rather than reimplementing the eligibility mask."""
    class FakeM9:
        @staticmethod
        def doc_pool_rows(k, seed):
            return np.arange(k, dtype=np.int64), {"n_eligible": 20, "n_drawn": k, "seed": seed}

    class FakePool:
        @staticmethod
        def build():
            return None, np.ones((20, 8), dtype=np.float32), {}

    monkeypatch.setitem(sys.modules, "data", FakeM9)
    monkeypatch.setitem(sys.modules, "pool", FakePool)
    rows, _V, meta = CL._screened_doc_pool(CL.ALL_ELIGIBLE, 0, {3, 4})
    assert list(rows) == [r for r in range(20) if r not in (3, 4)]
    assert meta["policy"] == CL.DOC_POLICY_ALL and meta["n_eligible_before_rescreen"] == 20


def _epoch_stream(n_docs=64, batch=8, seed=7):
    ids = [np.arange(3 + (i % 4), dtype=np.int32) for i in range(n_docs)]
    T = np.arange(n_docs * 4, dtype=np.float32).reshape(n_docs, 4)
    return CL.EpochShuffledStream(ids, T, pad_id=0, batch_size=batch, seed=seed, epoch_seed=seed)


def test_every_document_is_presented_once_per_epoch_in_a_different_order():
    s = _epoch_stream()
    nb = len(s.batches)
    e0 = [int(i) for i in s.epoch_order(0)]
    e1 = [int(i) for i in s.epoch_order(1)]
    assert sorted(e0) == sorted(e1) == list(range(nb)), "every batch exactly once per epoch"
    assert e0 != e1, "and in a different order"
    seen = sorted(int(i) for k in range(nb) for i in s.batches[int(s.epoch_order(0)[k])])
    assert seen == list(range(nb * 8)), "so every document appears exactly once in the epoch"


def test_the_document_position_is_a_pure_function_of_the_global_step():
    """CODEMAP pitfall 11: a stream position that depends on how many times the process has been
    asked cannot resume. Two fresh streams, one asked in order and one asked backwards, must
    return the identical batch for the identical global position."""
    a, b = _epoch_stream(), _epoch_stream()
    ks = list(range(len(a.batches) * 3))
    fwd = {k: a.batch(k)[2] for k in ks}
    for k in reversed(ks):
        assert torch.equal(b.batch(k)[2], fwd[k])
    assert not torch.equal(fwd[0], fwd[len(a.batches)]), "epoch 2 is reshuffled, not replayed"


def test_the_default_document_stream_is_untouched():
    ids = [np.arange(3, dtype=np.int32) for _ in range(64)]
    T = np.arange(64 * 4, dtype=np.float32).reshape(64, 4)
    plain = D.Stream(ids, T, pad_id=0, batch_size=8, seed=7)
    assert torch.equal(plain.batch(0)[2], plain.batch(len(plain.batches))[2]), \
        "the default stream still replays its batch order every epoch"


# ------------------------------------------------------------------------- the tiny build run ----

class FakeTok:
    pad_token_id = 0


class Toy(torch.nn.Module):
    """The stand-in `test_run_arm` and `test_trainer10` use: exercises the loop, not the model."""
    d_in = 1152
    tok = FakeTok()

    def __init__(self, d_in=12, d_out=6):
        super().__init__()
        self.head = torch.nn.Linear(d_in, d_out)

    def forward(self, ids, mask):
        x = ids.float() @ torch.ones(ids.shape[-1], self.head.in_features) / ids.shape[-1]
        return torch.nn.functional.normalize(self.head(x), dim=-1)

    def to(self, *a, **k):
        return self

    def n_params(self):
        return 34_540_672

    def under_cap(self):
        return True


class FakeStream:
    """A step-addressable stream whose batch is a pure function of its position, so a resumed run
    draws exactly what an uninterrupted one drew."""

    def __init__(self, tag, n=64, rows=BS):
        self.tag, self.n, self.rows = tag, n, rows
        self.calls = []

    def batch(self, k):
        self.calls.append(int(k))
        _ABRUPT.tick()
        g = torch.Generator().manual_seed(1000 * self.tag + int(k) % self.n)
        return (torch.randint(0, 50, (self.rows, 5), generator=g),
                torch.ones(self.rows, 5, dtype=torch.long),
                torch.nn.functional.normalize(torch.randn(self.rows, 6, generator=g), dim=-1))


class _Abrupt:
    after = None
    n = 0

    def tick(self):
        self.n += 1
        if self.after is not None and self.n >= self.after:
            os.kill(os.getpid(), signal.SIGKILL)


_ABRUPT = _Abrupt()


class ScriptedEval:
    """The scripted COV surface: a macro per LABEL, so the sequence is reproducible across a
    process restart and a restored evaluator lands on the same values."""
    MACROS = {}

    def __init__(self, *a, **k):
        self.records = []

    def __call__(self, step, label):
        m = float(self.MACROS.get(label, 0.40))
        self.records.append({"label": label, "step": int(step), "macro": m, "scripted": True})
        return m

    def state(self):
        return {"records": self.records}

    def restore(self, d):
        self.records = list(d.get("records") or [])


SCENARIOS = {
    # gain(cycle3) = 0.004 >= 0.003 -> one extension; then 0.0005 -> plateau
    "extend": {"cycle1": 0.40, "cycle2": 0.41, "cycle3": 0.414, "ext1": 0.4145},
    # gain = 0.002 -> plateau, no extension
    "reject": {"cycle1": 0.40, "cycle2": 0.41, "cycle3": 0.412},
    # gain = 0.004, but the extension's own evaluations collapse -> kill at the extension end
    "kill_ext": {"cycle1": 0.40, "cycle2": 0.41, "cycle3": 0.414,
                 "ext1mid20": 0.30, "ext1": 0.30},
}


def _config(root, *, dose=200_000_000, ext=66_700_000, reserved_hours=None, gate=True,
            lotte_gate=None):
    """A COPY of the real m13/build_config.json, with the test's overrides. The real file is the
    thing under test — a hand-written fixture would test the fixture."""
    Path(root).mkdir(parents=True, exist_ok=True)
    cfg = json.loads((REPO / "m13" / "build_config.json").read_text())
    cfg["arm_entry"]["dose_examples"] = dose
    cfg["extension_examples"] = ext
    cfg["budget"]["mandatory_hours_fixed"]["reserved_batch_allowance"] = reserved_hours
    cfg["budget"]["mandatory_examples"]["build"] = dose
    if lotte_gate is None and gate:
        g = Path(root) / "lotte_gate.json"
        g.write_text(json.dumps({"decision": "skipped", "_what": "test gate record"}))
        lotte_gate = str(g)
    cfg["lotte_gate"] = lotte_gate or str(Path(root) / "missing_gate.json")
    p = Path(root) / "build_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1))
    return cfg, p


def _state(root, *, cap=1, spent=100.0, cycle_usd=10.0, rate=2.7, smoke=False):
    d = Path(root) / "m13build" / ("smoke" if smoke else "") / "BUILD-200M"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({
        "max_extension_cycles": cap, "spent_usd": spent,
        "benchmark": {"rate_ex_per_s": rate, "price_usd_per_h": 2.0,
                      "extension_cycle_usd": cycle_usd, "max_extension_cycles": cap}}))
    return d


def _tiny_build_run(root, scenario="extend", *, abrupt_after=None, resume=False, cap=1,
                    spent=100.0, cycle_usd=10.0, smoke_steps=None):
    """A whole build on a Toy model, callable from a fresh interpreter so a SIGKILL cannot take
    pytest with it."""
    with pytest.MonkeyPatch.context() as mp:
        return _tiny_build_run_patched(mp, Path(root), scenario, abrupt_after=abrupt_after,
                                       resume=resume, cap=cap, spent=spent, cycle_usd=cycle_usd,
                                       smoke_steps=smoke_steps)


def _tiny_build_run_patched(mp, root, scenario, *, abrupt_after, resume, cap, spent, cycle_usd,
                            smoke_steps=None):
    import run_arm as R
    import trainer10 as Tr

    _cfg, cfg_path = _config(root, dose=TINY_DOSE, ext=TINY_EXT, reserved_hours=1.0)
    if not resume:
        _state(root, cap=cap, spent=spent, cycle_usd=cycle_usd, smoke=smoke_steps is not None)
    mp.setattr(BD, "WORK", root / "m13build")
    mp.setattr(BD, "SMOKE_WORK", root / "m13build" / "smoke")
    mp.setattr(BD, "RESULTS", root / "results")
    mp.setattr(BD.BL, "DOSE", TINY_DOSE)
    mp.setattr(BD.BL, "verdicts",
               lambda path=None: {"selected": {"batch": BS, "student": "bge-small"}})
    mp.setattr(BD.N, "Nano10", lambda *a, **k: Toy())
    mp.setattr(BD.R, "warm_start", lambda *a, **k: {"registered": "linear", "stub": True})
    mp.setattr(BD, "freeze_checkpoint",
               lambda m, d, ck, smoke=False, verbose=True: {"stub": True, "from": str(ck)})
    mp.setattr(torch.cuda, "is_available", lambda: False)
    ScriptedEval.MACROS = SCENARIOS[scenario]
    mp.setattr(BD.R, "CovEval", ScriptedEval)
    mp.setattr(BD.R, "StubEval", ScriptedEval)
    q, d = FakeStream(1), FakeStream(2)
    mp.setattr(BD.CL, "assemble_arm",
               lambda *a, **k: (D.batch_fn(q, d, "75/25"), {"arm": a[0], "mocked": True}))
    mp.setattr(BD.R, "streams_of", lambda bf: (q, d))
    real_train = Tr.train_arm

    def cpu_train(*a, **k):
        k["device"] = "cpu"
        return real_train(*a, **k)
    mp.setattr(BD.Tr, "train_arm", cpu_train)
    _ABRUPT.after, _ABRUPT.n = abrupt_after, 0
    try:
        return BD.run(str(cfg_path), device="cuda", resume=resume, ckpt_minutes=1.0,
                      verbose=False, smoke_steps=smoke_steps,
                      batch=(BS if smoke_steps else None),
                      lotte_gate=(str(Path(root) / "lotte_gate.json") if smoke_steps else None))
    finally:
        _ABRUPT.after = None


def _child(root, scenario, abrupt_after):
    """Run the abrupt half in a fresh interpreter: the SIGKILL is real."""
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from test_build13 import _tiny_build_run; "
         "_tiny_build_run(sys.argv[1], sys.argv[2], abrupt_after=int(sys.argv[3]))",
         str(root), scenario, str(abrupt_after)],
        cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True, timeout=180)


# ------------------------------------------------------------------- the extension decisions ----

def test_an_extension_runs_when_the_gain_clears_the_bar(tmp_path):
    rec = _tiny_build_run(tmp_path / "a", "extend", cap=1)
    assert rec["complete"] and rec["outcome"]["outcome"] == "PLATEAU"
    d0, d1 = rec["decisions"]
    assert d0["decision"] == "EXTEND" and d0["gain"] == pytest.approx(0.004, abs=1e-9)
    assert d1["decision"] == "PLATEAU"
    ext = rec["extensions"][0]
    assert ext["steps"] == TINY_EXT // BS and ext["examples"] == TINY_EXT
    assert rec["dose_run_examples"] == TINY_DOSE + TINY_EXT
    assert rec["final_checkpoint"].endswith("ext1.pt")


def test_a_gain_below_the_bar_is_a_plateau_freeze(tmp_path):
    rec = _tiny_build_run(tmp_path / "b", "reject")
    assert rec["outcome"]["outcome"] == "PLATEAU" and rec["extensions"] == []
    assert rec["decisions"][0]["gain"] == pytest.approx(0.002, abs=1e-9)
    assert rec["final_checkpoint"].endswith("cycle3.pt")
    assert rec["dose_run_examples"] == TINY_DOSE


def test_the_cap_stops_an_extension_the_gain_would_have_allowed(tmp_path):
    rec = _tiny_build_run(tmp_path / "c", "extend", cap=0)
    assert rec["outcome"]["outcome"] == "CAPPED" and rec["extensions"] == []
    assert rec["decisions"][0]["decision"] == "CAPPED"
    assert rec["complete"] and rec["final_checkpoint"].endswith("cycle3.pt")


def test_the_budget_stops_an_extension_before_the_cap_does(tmp_path):
    """"A cycle whose projected cost plus billed spend to date would exceed the ceiling does not
    start" — with cap 5 left, the money is what refuses."""
    rec = _tiny_build_run(tmp_path / "d", "extend", cap=5, spent=995.0, cycle_usd=10.0)
    assert rec["outcome"]["outcome"] == "CAPPED" and rec["extensions"] == []
    dec = rec["decisions"][0]
    assert dec["decision"] == "CAPPED_BUDGET" and "ceiling" in rec["outcome"]["why"]


def test_a_kill_at_an_extension_end_is_terminal_and_has_no_final_checkpoint(tmp_path):
    rec = _tiny_build_run(tmp_path / "e", "kill_ext", cap=2)
    assert rec["status"] == "failed" and rec["complete"] is False and rec["terminal"] is True
    assert rec["outcome"]["outcome"] == "FAILED" and "kill" in rec["outcome"]["why"]
    assert rec["final_checkpoint"] is None and rec["final_checkpoint_sha256"] is None
    assert rec["freeze"] == {}
    assert (tmp_path / "e" / "m13build" / "BUILD-200M" / "ext1.pt").exists(), \
        "the retained extension checkpoint is still on disk; it is simply never called final"


def test_the_three_cycles_are_one_train_arm_schedule(tmp_path, monkeypatch):
    seen = []
    import trainer10 as Tr
    real = Tr.train_arm

    def spy(*a, **k):
        seen.append(k.get("total_steps"))
        k["device"] = "cpu"
        return real(*a, **k)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD.Tr, "train_arm", spy)
        rec = _tiny_build_run_inner(mp, tmp_path / "f", "extend")
    assert seen[0] == TINY_DOSE // BS, "cycles 1-3 are ONE schedule, not three calls"
    assert seen[1] == TINY_EXT // BS and len(seen) == 2
    assert rec["schedule"]["extension_warmup_steps"] == 0


def _tiny_build_run_inner(mp, root, scenario, **kw):
    """`_tiny_build_run_patched` under a monkeypatch context the caller already owns."""
    return _tiny_build_run_patched(mp, Path(root), scenario, abrupt_after=None, resume=False,
                                   cap=kw.get("cap", 1), spent=kw.get("spent", 100.0),
                                   cycle_usd=kw.get("cycle_usd", 10.0))


# ---------------------------------------------------------------------- resume equivalence ----

@pytest.mark.parametrize("abrupt_after,where", [(45, "across the cycle-1 boundary"),
                                                (132, "inside the extension")])
def test_sigkill_and_resume_reproduce_the_uninterrupted_build(tmp_path, abrupt_after, where):
    """The build's own SIGKILL test. `abrupt_after` counts stream draws: 45 lands in cycle 2
    (the cycle-1 end and its COV read are already behind it), 132 lands 12 steps into the
    extension, past its first rolling checkpoint."""
    base = _tiny_build_run(tmp_path / "base", "extend")
    base_model = torch.load(tmp_path / "base/m13build/BUILD-200M/ext1.pt",
                            weights_only=False)["model"]
    root = tmp_path / "killed"
    child = _child(root, "extend", abrupt_after)
    assert child.returncode == -signal.SIGKILL, child.stdout + child.stderr
    d = root / "m13build" / "BUILD-200M"
    receipt = json.loads((d / "receipt.json").read_text())
    assert receipt["status"] == "running" and receipt["terminal"] is False
    assert not (root / "results" / "m13_build_record.json").exists()
    with pytest.raises(SystemExit, match="already exists"):
        _tiny_build_run(root, "extend")

    resumed = _tiny_build_run(root, "extend", resume=True)
    assert resumed["complete"] is True, where
    assert resumed["cov"]["cycle_macros"] == base["cov"]["cycle_macros"]
    assert resumed["dose_run_examples"] == base["dose_run_examples"] == TINY_DOSE + TINY_EXT
    assert [e["j"] for e in resumed["extensions"]] == [1]
    if abrupt_after > TINY_DOSE // BS:
        assert resumed["extensions"][0]["start_step"] > 0, "resumed INSIDE the extension"
    else:
        assert resumed["training"]["start_step"] > 0, "resumed inside cycles 1-3"
        assert resumed["extensions"][0]["start_step"] == 0
    got = torch.load(d / "ext1.pt", weights_only=False)["model"]
    assert all(torch.equal(v, got[k]) for k, v in base_model.items()), where
    assert json.loads((d / "receipt.json").read_text())["terminal"] is True


def test_a_smokes_extension_is_the_smokes_own_dose_not_a_30_hour_cloud_cycle(tmp_path):
    """Found by the first box smoke of this controller: a 20-step smoke whose gain cleared the bar
    started a 2,084,375-step extension, because the extension size came from the registered
    66,700,000 while everything else was smoke-scaled."""
    rec = _tiny_build_run(tmp_path / "s", "extend", cap=1, smoke_steps=TINY_DOSE // BS)
    assert rec["smoke"] is True
    assert rec["schedule"]["extension"]["registered_examples"] == TINY_DOSE
    assert rec["extensions"][0]["examples"] == TINY_DOSE
    assert rec["batch_source"] == "--batch (SMOKE ONLY)"
    assert not (tmp_path / "s" / "results").exists(), "a smoke never publishes a record"


def test_the_freeze_exports_on_the_cpu_after_dev6(tmp_path, monkeypatch):
    """§T: export, parity and encoding run EAGER and on CPU. The first box smoke of this
    controller died at the export with "Expected all tensors to be on the same device", because
    `nano10.export_onnx` builds its trace inputs on the CPU."""
    model = Toy()
    seen = {}
    ck = tmp_path / "cycle3.pt"
    torch.save({"model": model.state_dict()}, ck)
    monkeypatch.setattr(BD.R, "dev6", lambda m, verbose=True: {"macro": 0.5, "device_at_call":
                                                               next(m.parameters()).device.type})
    def fake_export(m, d, max_len=512):
        seen["export_device"] = next(m.parameters()).device.type
        return {"path": str(d)}
    monkeypatch.setattr(BD.N, "export_onnx", fake_export)
    monkeypatch.setattr(BD.N, "export_parity", lambda m, d, t: {"min_cos": 1.0})
    monkeypatch.setattr(BD, "fastembed_parity", lambda m, d, t: {"served": False})
    monkeypatch.setattr(BD, "sha256_file", lambda p: "0" * 64)
    out = BD.freeze_checkpoint(model, tmp_path, ck, smoke=False, verbose=False)
    assert "export_error" not in out
    assert seen["export_device"] == "cpu"
    assert out["dev6"]["macro"] == 0.5 and out["ort_parity"]["min_cos"] == 1.0


def test_the_loss_history_lives_in_the_sidecar_not_the_checkpoint(tmp_path):
    _tiny_build_run(tmp_path / "g", "reject")
    d = tmp_path / "g" / "m13build" / "BUILD-200M"
    lines = [json.loads(x) for x in (d / "losses.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == list(range(TINY_DOSE // BS))
    ck = torch.load(d / "cycle3.pt", weights_only=False)["extra"]
    assert ck["n_losses"] == TINY_DOSE // BS
    assert len(ck["losses"]) <= 200


def test_each_phase_keeps_its_own_loss_sidecar(tmp_path):
    """A fresh `train_arm` truncates the sidecar it is given, so one shared file would lose
    cycles 1-3's losses the moment extension 1 started (the box smoke left 20 of 40 lines)."""
    rec = _tiny_build_run(tmp_path / "g2", "extend", cap=1)
    d = tmp_path / "g2" / "m13build" / "BUILD-200M"
    assert sorted(p.name for p in d.glob("losses*.jsonl")) == ["losses.jsonl",
                                                              "losses_ext1.jsonl"]
    assert len((d / "losses.jsonl").read_text().splitlines()) == TINY_DOSE // BS
    assert len((d / "losses_ext1.jsonl").read_text().splitlines()) == TINY_EXT // BS
    assert len(rec["loss_logs"]) == 2


# ------------------------------------------------------------------------------- refusals ----

def _run_refusing(root, **kw):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD, "WORK", Path(root) / "m13build")
        mp.setattr(BD, "RESULTS", Path(root) / "results")
        mp.setattr(BD.BL, "DOSE", TINY_DOSE)
        mp.setattr(BD.BL, "verdicts",
                   lambda path=None: {"selected": {"batch": BS, "student": "bge-small"}})
        cfg_kw = {k: kw.pop(k) for k in ("gate", "dose") if k in kw}
        _cfg, p = _config(root, dose=cfg_kw.get("dose", TINY_DOSE), ext=TINY_EXT,
                          reserved_hours=1.0, gate=cfg_kw.get("gate", True))
        return BD.run(str(p), **kw)


def test_compile_and_every_recipe_knob_are_smoke_only(tmp_path):
    with pytest.raises(SystemExit, match="only be passed with --smoke-steps"):
        _run_refusing(tmp_path / "h", compile_step=True, device="cuda")
    with pytest.raises(SystemExit, match="only be passed with --smoke-steps"):
        _run_refusing(tmp_path / "h2", batch=64, device="cuda")
    with pytest.raises(SystemExit, match="only be passed with --smoke-steps"):
        _run_refusing(tmp_path / "h3", max_len=64, device="cuda")


def test_a_registered_build_refuses_a_non_cuda_device(tmp_path):
    with pytest.raises(SystemExit, match="requires --device cuda"):
        _run_refusing(tmp_path / "i", device="cpu")


def test_a_missing_lotte_gate_record_refuses_the_build(tmp_path):
    with pytest.raises(SystemExit, match="no LoTTE gate record"):
        _run_refusing(tmp_path / "j", gate=False, device="cuda")


def test_a_missing_cap_refuses_the_build(tmp_path):
    """`max_extension_cycles` is fixed at the day-one benchmark, not now."""
    with pytest.raises(SystemExit, match="max_extension_cycles"):
        _run_refusing(tmp_path / "k", device="cuda")


def test_an_existing_terminal_record_refuses_a_bare_re_run_and_a_resume(tmp_path):
    root = tmp_path / "l"
    _state(root)
    d = root / "m13build" / "BUILD-200M"
    (d / "record.json").write_text(json.dumps({"arm": "BUILD-200M", "complete": True}))
    with pytest.raises(SystemExit, match="already exists"):
        _run_refusing(root, device="cuda")
    with pytest.raises(SystemExit, match="TERMINAL"):
        _run_refusing(root, device="cuda", resume=True)


def test_the_pending_batch_refusal_reaches_the_controller(tmp_path):
    root = tmp_path / "m"
    _state(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD, "WORK", root / "m13build")
        mp.setattr(BD.BL, "DOSE", TINY_DOSE)
        _cfg, p = _config(root, dose=TINY_DOSE, ext=TINY_EXT, reserved_hours=1.0)
        with pytest.raises(SystemExit, match="E1 verdict is PENDING"):
            BD.run(str(p), device="cuda")


def test_the_benchmark_writes_the_cap_into_state_and_needs_both_inputs(tmp_path):
    root = tmp_path / "n"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD, "WORK", root / "m13build")
        mp.setattr(BD, "RESULTS", root / "results")
        mp.setattr(BD.BL, "DOSE", TINY_DOSE)
        mp.setattr(BD.BL, "verdicts",
                   lambda path=None: {"selected": {"batch": BS, "student": "bge-small"}})
        _cfg, p = _config(root, dose=TINY_DOSE, ext=TINY_EXT, reserved_hours=1.0)
        with pytest.raises(SystemExit, match="--benchmark needs"):
            BD.run(str(p), device="cuda", benchmark_only=True, benchmark_rate=900)
        st = BD.run(str(p), device="cuda", benchmark_only=True, benchmark_rate=900,
                    benchmark_price=2.0)
    assert st["max_extension_cycles"] == st["benchmark"]["max_extension_cycles"]
    on_disk = json.loads((root / "m13build" / "BUILD-200M" / "state.json").read_text())
    assert on_disk["max_extension_cycles"] == st["max_extension_cycles"]
    assert on_disk["spent_usd"] == st["benchmark"]["committed_usd"]


def test_the_real_config_validates_against_the_live_registry():
    """The shipped `m13/build_config.json`, against the registry that is on disk now."""
    cfg, _ = BL.load()
    rep = BL.validate(cfg, smoke=True)
    assert rep["registry_sha256"] == BL.R.sha256_file(BL.R.REGISTRY)
    assert rep["dose_examples"] == 200_000_000 and rep["data_cut"] == "none"
    assert cfg["arm_entry"]["batch"] == "PENDING", "the batch comes from the E1 verdict"
