"""What the 200M build controller must get right, and what it must refuse.

CPU only, synthetic streams, a `Toy` model and scripted COV macros — the same shape as
`m10src/test_run_arm.py`. The point is the controller's contract: the 200M pin's arithmetic, the
kill and plateau readings (ruling R13: no extension cycles), the LoTTE gate contract and its veto,
the FROZEN_UNVERIFIED freeze and its resumable retry, resume equivalence across a cycle boundary,
the document policy including its ragged tail, the byte-identity of the default artifacts, and
every refusal that stands between a mistake and a rented GPU. Nothing here trains a BERT, reads a
corpus, or touches an evaluation surface.
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
BIG_DOSE = 7040             # 220 steps at bs32: more than trainer10.LOSS_TAIL_KEEP
BS = 32
VERDICTS_SHA = BL.R.sha256_file(BL.VERDICTS)
REGISTRY_SHA = BL.R.sha256_file(BL.R.REGISTRY)
SELECTED = {"registry_sha256": REGISTRY_SHA, "selected": {"batch": BS, "student": "bge-small"}}


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


def test_the_allocation_table_prices_every_mandatory_line(tmp_path):
    """Ruling R13 leaves no cap formula: the budget is a fixed table and the only judgement in it
    is whether the total fits under the recorded ceiling."""
    cfg = _config(tmp_path, reserved_hours=2.0)[0]
    al = BL.allocation(cfg, 1000.0, 2.0)
    hours = al["mandatory_hours"]
    assert hours["build"] == pytest.approx(200_000_000 / 1000 / 3600, abs=1e-3)
    assert hours["cloud_E_arms"] == pytest.approx(10_000_000 / 1000 / 3600, abs=1e-3)
    assert al["committed_usd"] == pytest.approx(2.0 * al["mandatory_hours_total"] + 25.0, abs=0.05)
    assert al["headroom_usd"] == pytest.approx(1000 - al["committed_usd"], abs=0.05)
    assert "max_extension_cycles" not in al and "extension_cycle_usd" not in al


def test_a_mandatory_plan_above_the_ceiling_is_refused_not_capped_at_zero(tmp_path):
    """The old cap arithmetic clamped a negative remainder to zero, so an $11,724 plan reported
    itself as 'max_extension_cycles 0' (Codex 2026-09-10, B7)."""
    cfg = _config(tmp_path, reserved_hours=2.0)[0]
    with pytest.raises(SystemExit, match="above the recorded"):
        BL.allocation(cfg, 10.0, 20.0)               # 10 ex/s: the build alone is 5,555 hours


def test_the_allocation_refuses_an_unpriced_reserved_batch(tmp_path):
    cfg = _config(tmp_path)[0]                          # reserved_batch_allowance is null
    with pytest.raises(SystemExit, match="reserved_batch_allowance"):
        BL.allocation(cfg, 1000.0, 2.0)
    assert BL.allocation(cfg, 1000.0, 2.0, smoke=True)["unpriced_lines_zeroed_for_smoke"] == \
        ["reserved_batch_allowance"]


def test_a_configuration_that_reintroduces_extension_cycles_is_refused(tmp_path):
    """Ruling R13, Dylan 2026-09-10: no extension cycles."""
    cfg = _config(tmp_path)[0]
    cfg["extension_examples"] = 66_700_000
    with pytest.raises(SystemExit, match="R13"):
        BL.validate(cfg)


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


def test_the_validator_pins_require_all_forms_and_the_document_subfields(tmp_path):
    """B10: the validator accepted `require_all_forms: false` and never looked at the document
    policy's own fields, so an arm could carry the policy name and none of its meaning."""
    cfg = _config(tmp_path)[0]
    cfg["arm_entry"]["require_all_forms"] = False
    with pytest.raises(SystemExit, match="require_all_forms"):
        BL.validate(cfg)
    for field, bad in (("repeat", False), ("reshuffle_per_epoch", False), ("policy", "other")):
        cfg = _config(tmp_path)[0]
        cfg["arm_entry"]["documents"][field] = bad
        with pytest.raises(SystemExit):
            BL.validate(cfg)
    cfg = _config(tmp_path)[0]
    cfg["plateau_min_gain"] = 0.004
    with pytest.raises(SystemExit, match="plateau_min_gain"):
        BL.validate(cfg)


def test_screen_verdicts_bound_to_another_registry_are_refused(tmp_path, monkeypatch):
    """B6: `selected.batch` was trusted without checking the verdicts' own registry binding, and
    ruling R9 changes that sha before the build starts."""
    cfg = _config(tmp_path)[0]
    monkeypatch.setattr(BL, "verdicts", lambda path=None: {"registry_sha256": "0" * 64,
                                                           "selected": {"batch": 32}})
    with pytest.raises(SystemExit, match="bound to registry sha"):
        BL.resolve_batch(cfg)
    with pytest.raises(SystemExit, match="bound to registry sha"):
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


def test_every_eligible_document_is_presented_on_a_non_divisible_pool():
    """B9: `length_buckets` sorts by length and then DROPS the ragged tail, so the longest
    documents in the pool were excluded from every epoch for ever — which is not "every eligible
    re-screened document once per epoch" (R5). 65 documents at batch 8 is 8 full batches and a
    short one; the epoch must present 65 rows, not 64."""
    s = _epoch_stream(n_docs=65, batch=8)
    assert len(s.batches) == 9
    sizes = sorted(len(b) for b in s.batches)
    assert sizes == [1] + [8] * 8
    seen = sorted(int(i) for k in range(len(s.batches))
                  for i in s.batches[int(s.epoch_order(0)[k])])
    assert seen == list(range(65)), "every eligible document exactly once per epoch"
    rows = sum(int(s.batch(k)[0].shape[0]) for k in range(len(s.batches)))
    assert rows == 65, "and the presentations the trainer actually draws add up to the pool"


def test_the_default_document_stream_is_untouched():
    """The default (screen) stream keeps BOTH of its old behaviours: it replays its batch order
    every epoch, and it still drops the ragged tail. `keep_tail` is additive."""
    ids = [np.arange(3, dtype=np.int32) for _ in range(64)]
    T = np.arange(64 * 4, dtype=np.float32).reshape(64, 4)
    plain = D.Stream(ids, T, pad_id=0, batch_size=8, seed=7)
    assert torch.equal(plain.batch(0)[2], plain.batch(len(plain.batches))[2]), \
        "the default stream still replays its batch order every epoch"
    ragged = D.Stream([np.arange(3, dtype=np.int32) for _ in range(65)],
                      np.zeros((65, 4), dtype=np.float32), pad_id=0, batch_size=8, seed=7)
    assert len(ragged.batches) == 8, "the default still drops the ragged tail"
    assert all(len(b) == 8 for b in ragged.batches)


# ------------------------------------------------------------------------- the tiny build run ----

class FakeTok:
    pad_token_id = 0


@pytest.fixture(autouse=True)
def _fake_e_records(tmp_path_factory, monkeypatch):
    """Published E arm records do not exist yet (family E is CLOUD_ONLY and unrun), and
    `check_gate` refuses without them outside a smoke. Every controller test therefore sees a fake
    results dir whose cycle-3 shas match `_gate`'s defaults: E-bs32 -> "a"*64, E-bs128 -> "c"*64."""
    d = tmp_path_factory.mktemp("e_records")
    (d / "m10_arm_E-bs32.json").write_text(json.dumps({"checkpoints": {"cycle3": {"sha256": "a" * 64}}}))
    (d / "m10_arm_E-bs128.json").write_text(json.dumps({"checkpoints": {"cycle3": {"sha256": "c" * 64}}}))
    monkeypatch.setattr(BD, "ARM_RECORDS_DIR", d)
    return d


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
    # evaluation order at TINY_DOSE: mid20, cycle1, mid60, cycle2, mid100, cycle3
    # gain(cycle3) = 0.004 >= 0.003: the schedule simply ends. Nothing follows it (R13).
    "gain": {"mid20": 0.40, "cycle1": 0.40, "mid60": 0.41, "cycle2": 0.41,
             "mid100": 0.415, "cycle3": 0.414},
    # gain(cycle3) = 0.002 < 0.003: `plateau_fires` at the last cycle end, read as the freeze
    "plateau": {"mid20": 0.40, "cycle1": 0.40, "mid60": 0.405, "cycle2": 0.41,
                "mid100": 0.412, "cycle3": 0.412},
    # two consecutive evaluations far below the best of their own kind -> the kill rule
    "kill": {"mid20": 0.40, "cycle1": 0.40, "mid60": 0.41, "cycle2": 0.41,
             "mid100": 0.30, "cycle3": 0.30},
}


def _gate(root, **over):
    """A gate record carrying everything `build13.check_gate` requires, plus the committed
    checkpoint manifest it is bound to (`check_gate_manifest`), written under `root/e_records/` so
    `_tiny_build_run_patched` can point `MANIFEST_PATH` at it. The default is the bs32 branch,
    where the veto is SKIPPED and forfeited (m13/LOTTE_GATE_REGISTRATION.json)."""
    g = {"executed": True, "decision": "skipped", "branch": "bs32",
         "candidate_sha256": "a" * 64, "comparator_sha256": None,
         "e1_verdict_sha256": VERDICTS_SHA, "read_at": "2026-09-10T00:00:00+0000",
         "_what": "test gate record"}
    g.update(over)
    man_p = _manifest_path(root)
    man_p.parent.mkdir(parents=True, exist_ok=True)
    man_p.write_text(json.dumps({
        "_what": "test manifest", "branch": g["branch"],
        "candidate": {"arm": "E-bs128" if g["branch"] == "bs128" else "E-bs32",
                      "sha256": g["candidate_sha256"]},
        "comparator": (None if g["comparator_sha256"] is None
                       else {"arm": "E-bs32", "sha256": g["comparator_sha256"]}),
        "e1_verdict_sha256": g["e1_verdict_sha256"], "git_head": "f" * 40}))
    g.setdefault("manifest_sha256", BD.sha256_file(man_p))
    p = Path(root) / "lotte_gate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g))
    return p


def _manifest_path(root):
    return Path(root) / "e_records" / "LOTTE_GATE_MANIFEST.json"


def _config(root, *, dose=200_000_000, reserved_hours=None, gate=True, lotte_gate=None):
    """A COPY of the real m13/build_config.json, with the test's overrides. The real file is the
    thing under test — a hand-written fixture would test the fixture."""
    Path(root).mkdir(parents=True, exist_ok=True)
    cfg = json.loads((REPO / "m13" / "build_config.json").read_text())
    cfg["arm_entry"]["dose_examples"] = dose
    cfg["budget"]["mandatory_hours_fixed"]["reserved_batch_allowance"] = reserved_hours
    cfg["budget"]["mandatory_examples"]["build"] = dose
    if lotte_gate is None and gate:
        lotte_gate = str(_gate(root) if BS == 32 else
                         _gate(root, decision="no_veto", branch="bs128",
                               candidate_sha256="c" * 64, comparator_sha256="a" * 64))
    cfg["lotte_gate"] = lotte_gate or str(Path(root) / "missing_gate.json")
    p = Path(root) / "build_config.json"
    p.write_text(json.dumps(cfg, indent=1))
    return cfg, p


def _out_dir(root, *, smoke=False):
    d = Path(root) / "m13build" / ("smoke" if smoke else "") / "BUILD-200M"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tiny_build_run(root, scenario="plateau", *, abrupt_after=None, resume=False, dose=TINY_DOSE,
                    smoke_steps=None, freeze=None):
    """A whole build on a Toy model, callable from a fresh interpreter so a SIGKILL cannot take
    pytest with it."""
    with pytest.MonkeyPatch.context() as mp:
        return _tiny_build_run_patched(mp, Path(root), scenario, abrupt_after=abrupt_after,
                                       resume=resume, dose=dose, smoke_steps=smoke_steps,
                                       freeze=freeze)


def _tiny_build_run_patched(mp, root, scenario, *, abrupt_after=None, resume=False,
                            dose=TINY_DOSE, smoke_steps=None, freeze=None):
    import trainer10 as Tr

    _cfg, cfg_path = _config(root, dose=dose, reserved_hours=1.0)
    mp.setattr(BD, "WORK", root / "m13build")
    mp.setattr(BD, "SMOKE_WORK", root / "m13build" / "smoke")
    mp.setattr(BD, "RESULTS", root / "results")
    # the published E arm records the gate check authenticates against (fake, matching `_gate`),
    # kept OUT of root/results so "a smoke never publishes a record" stays checkable
    (root / "e_records").mkdir(parents=True, exist_ok=True)
    for arm, sha in (("E-bs32", "a" * 64), ("E-bs128", "c" * 64)):
        (root / "e_records" / f"m10_arm_{arm}.json").write_text(
            json.dumps({"checkpoints": {"cycle3": {"sha256": sha}}}))
    mp.setattr(BD, "ARM_RECORDS_DIR", root / "e_records")
    mp.setattr(BD, "MANIFEST_PATH", _manifest_path(root))
    mp.setattr(BD.BL, "DOSE", dose)
    mp.setattr(BD.BL, "verdicts", lambda path=None: SELECTED)
    mp.setattr(BD.N, "Nano10", lambda *a, **k: Toy())
    mp.setattr(BD.R, "warm_start", lambda *a, **k: {"registered": "linear", "stub": True})
    mp.setattr(BD, "freeze_checkpoint",
               freeze or (lambda m, d, ck, smoke=False, verbose=True:
                          {"stub": True, "from": str(ck), "verified": True}))
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


# ------------------------------------------------------------- the stop rules and the freeze ----

def test_the_schedule_ends_at_cycle_3_with_no_extension(tmp_path):
    """Ruling R13: a gain that clears the old extension bar buys nothing. The build is 200M."""
    rec = _tiny_build_run(tmp_path / "a", "gain")
    assert rec["complete"] and rec["status"] == "complete"
    assert rec["outcome"]["outcome"] == "COMPLETE"
    assert rec["outcome"]["final_gain"] == pytest.approx(0.004, abs=1e-9)
    assert rec["outcome"]["plateau_at"] is None
    assert rec["dose_run_examples"] == TINY_DOSE
    assert rec["final_checkpoint"].endswith("cycle3.pt")
    assert "extensions" not in rec and "decisions" not in rec
    assert rec["schedule"]["extension_cycles"].startswith("none")


def test_a_plateau_at_the_last_cycle_end_is_the_ordinary_freeze(tmp_path):
    rec = _tiny_build_run(tmp_path / "b", "plateau")
    assert rec["complete"] and rec["outcome"]["outcome"] == "COMPLETE"
    assert rec["outcome"]["plateau_at"] == 3 and "plateau" in rec["outcome"]["why"]
    assert rec["final_checkpoint"].endswith("cycle3.pt")
    assert rec["dose_run_examples"] == TINY_DOSE


def test_a_kill_is_terminal_and_has_no_final_checkpoint(tmp_path):
    rec = _tiny_build_run(tmp_path / "e", "kill")
    assert rec["status"] == "failed" and rec["complete"] is False and rec["terminal"] is True
    assert rec["outcome"]["outcome"] == "FAILED" and "kill" in rec["outcome"]["why"]
    assert rec["final_checkpoint"] is None and rec["final_checkpoint_sha256"] is None
    assert rec["freeze"] == {}
    assert (tmp_path / "e" / "m13build" / "BUILD-200M" / "cycle3.pt").exists(), \
        "the retained cycle checkpoint is still on disk; it is simply never called final"


def test_the_three_cycles_are_one_train_arm_schedule(tmp_path):
    seen = []
    import trainer10 as Tr
    real = Tr.train_arm

    def spy(*a, **k):
        seen.append(k.get("total_steps"))
        k["device"] = "cpu"
        return real(*a, **k)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD.Tr, "train_arm", spy)
        rec = _tiny_build_run_patched(mp, tmp_path / "f", "plateau")
    assert seen == [TINY_DOSE // BS], "cycles 1-3 are ONE schedule and nothing follows them"
    assert rec["schedule"]["warmup_steps_cycle1"] == BD.N.warmup_steps_for(BS)


# ---------------------------------------------------------------------- resume equivalence ----

def test_sigkill_and_resume_reproduce_the_uninterrupted_build(tmp_path):
    """The build's own SIGKILL test. `abrupt_after` counts stream draws: 45 lands in cycle 2, so
    the cycle-1 end and its COV read are already behind it."""
    base = _tiny_build_run(tmp_path / "base", "plateau")
    base_model = torch.load(tmp_path / "base/m13build/BUILD-200M/cycle3.pt",
                            weights_only=False)["model"]
    root = tmp_path / "killed"
    child = _child(root, "plateau", 45)
    assert child.returncode == -signal.SIGKILL, child.stdout + child.stderr
    d = root / "m13build" / "BUILD-200M"
    receipt = json.loads((d / "receipt.json").read_text())
    assert receipt["status"] == "running" and receipt["terminal"] is False
    assert not (root / "results" / "m13_build_record.json").exists()
    with pytest.raises(SystemExit, match="already exists"):
        _tiny_build_run(root, "plateau")

    resumed = _tiny_build_run(root, "plateau", resume=True)
    assert resumed["complete"] is True
    assert resumed["cov"]["cycle_macros"] == base["cov"]["cycle_macros"]
    assert resumed["dose_run_examples"] == base["dose_run_examples"] == TINY_DOSE
    assert resumed["training"]["start_step"] > 0, "resumed inside cycles 1-3"
    got = torch.load(d / "cycle3.pt", weights_only=False)["model"]
    assert all(torch.equal(v, got[k]) for k, v in base_model.items())
    assert json.loads((d / "receipt.json").read_text())["terminal"] is True


def test_a_smoke_writes_only_under_its_own_tree(tmp_path):
    rec = _tiny_build_run(tmp_path / "s", "plateau", smoke_steps=TINY_DOSE // BS)
    assert rec["smoke"] is True and rec["batch_source"] == "--batch (SMOKE ONLY)"
    assert not (tmp_path / "s" / "results").exists(), "a smoke never publishes a record"
    assert (tmp_path / "s" / "m13build" / "smoke" / "BUILD-200M" / "cycle3.pt").exists()


# --------------------------------------------------------------------------------- the freeze ----

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
    monkeypatch.setattr(BD, "fastembed_parity",
                        lambda m, d, t: {"served": True, "min_cos": 1.0, "pass_min_cos_1e-4": True})
    monkeypatch.setattr(BD, "sha256_file", lambda p: "0" * 64)
    out = BD.freeze_checkpoint(model, tmp_path, ck, smoke=False, verbose=False)
    assert "export_error" not in out and out["verified"] is True
    assert seen["export_device"] == "cpu"
    assert out["dev6"]["macro"] == 0.5 and out["ort_parity"]["min_cos"] == 1.0


@pytest.mark.parametrize("how", ["raises", "parity"])
def test_a_failed_export_or_parity_is_not_a_verified_freeze(tmp_path, monkeypatch, how):
    """B8: the export exception was swallowed into a field nothing read, and neither parity result
    changed `ok`, so a build whose ONNX never existed still published `complete: true`."""
    model = Toy()
    ck = tmp_path / "cycle3.pt"
    torch.save({"model": model.state_dict()}, ck)
    monkeypatch.setattr(BD.R, "dev6", lambda m, verbose=True: {"macro": 0.5})
    monkeypatch.setattr(BD, "sha256_file", lambda p: "0" * 64)
    if how == "raises":
        monkeypatch.setattr(BD.N, "export_onnx",
                            lambda m, d, max_len=512: (_ for _ in ()).throw(RuntimeError("boom")))
    else:
        monkeypatch.setattr(BD.N, "export_onnx", lambda m, d, max_len=512: {"path": str(d)})
        monkeypatch.setattr(BD.N, "export_parity", lambda m, d, t: {"min_cos": 0.93})
        monkeypatch.setattr(BD, "fastembed_parity", lambda m, d, t: {"served": False})
    out = BD.freeze_checkpoint(model, tmp_path, ck, smoke=False, verbose=False)
    assert out["verified"] is False and out["unverified_why"]


def test_an_unverified_freeze_ends_frozen_unverified_and_the_retry_does_not_retrain(tmp_path):
    """The checkpoint is retained, `complete` is false, the record is NOT terminal, and a
    --resume run retries the freeze alone."""
    calls = {"freeze": 0, "train": 0}

    def flaky_freeze(m, d, ck, smoke=False, verbose=True):
        calls["freeze"] += 1
        if calls["freeze"] == 1:
            return {"from_checkpoint": str(ck), "verified": False,
                    "unverified_why": "onnxruntime is not installed on this box"}
        return {"from_checkpoint": str(ck), "verified": True}

    root = tmp_path / "u"
    first = _tiny_build_run(root, "plateau", freeze=flaky_freeze)
    assert first["status"] == "frozen_unverified"
    assert first["complete"] is False and first["terminal"] is False
    assert first["final_checkpoint"].endswith("cycle3.pt")
    assert first["final_checkpoint_sha256"] and "UNVERIFIED" in first["_final_checkpoint"]
    d = root / "m13build" / "BUILD-200M"
    assert json.loads((d / "receipt.json").read_text())["status"] == "running"

    import trainer10 as Tr
    real = Tr.train_arm

    def spy(*a, **k):
        calls["train"] += 1
        k["device"] = "cpu"
        return real(*a, **k)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD.Tr, "train_arm", spy)
        second = _tiny_build_run_patched(mp, root, "plateau", resume=True, freeze=flaky_freeze)
    assert calls["train"] == 0, "finalization is resumable: the retry never retrains"
    assert second["complete"] is True and second["status"] == "complete"
    assert second["terminal"] is True
    assert second["final_checkpoint"] == first["final_checkpoint"]


# ----------------------------------------------------------------------- the loss sidecar ----

def test_the_loss_history_lives_in_the_sidecar_not_the_checkpoint(tmp_path):
    """220 steps against `trainer10.LOSS_TAIL_KEEP` of 200: the in-memory list is trimmed, the
    sidecar is complete, and `n_losses` carries the true count (the 120-step version of this test
    never reached the cap it was written for — Codex 2026-09-10, B11)."""
    _tiny_build_run(tmp_path / "g", "plateau", dose=BIG_DOSE)
    d = tmp_path / "g" / "m13build" / "BUILD-200M"
    steps = BIG_DOSE // BS
    assert steps > BD.Tr.LOSS_TAIL_KEEP
    lines = [json.loads(x) for x in (d / "losses.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == list(range(steps))
    ck = torch.load(d / "cycle3.pt", weights_only=False)["extra"]
    assert ck["n_losses"] == steps
    assert len(ck["losses"]) == BD.Tr.LOSS_TAIL_KEEP
    assert sorted(p.name for p in d.glob("losses*.jsonl")) == ["losses.jsonl"]


# ------------------------------------------------------------------------------- refusals ----

def _run_refusing(root, **kw):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD, "WORK", Path(root) / "m13build")
        mp.setattr(BD, "RESULTS", Path(root) / "results")
        mp.setattr(BD.BL, "DOSE", TINY_DOSE)
        mp.setattr(BD.BL, "verdicts", lambda path=None: SELECTED)
        mp.setattr(BD, "MANIFEST_PATH", _manifest_path(root))
        cfg_kw = {k: kw.pop(k) for k in ("gate", "dose") if k in kw}
        _cfg, p = _config(root, dose=cfg_kw.get("dose", TINY_DOSE), reserved_hours=1.0,
                          gate=cfg_kw.get("gate", True))
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


# ------------------------------------------------------------------------ the LoTTE gate ----

def test_an_empty_or_unexecuted_gate_record_is_refused(tmp_path):
    """B6: `{}` used to pass, because the controller only read `gate.get("decision")`."""
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(SystemExit, match="executed"):
        BD.check_gate(empty)
    with pytest.raises(SystemExit, match="executed"):
        BD.check_gate(_gate(tmp_path / "x", executed=False))
    # the REGISTRATION is not the gate record
    with pytest.raises(SystemExit, match="executed"):
        BD.check_gate(REPO / "m13" / "LOTTE_GATE_REGISTRATION.json")


@pytest.mark.parametrize("over,match", [
    ({"decision": None}, "decision"),
    ({"decision": "maybe"}, "decision"),
    ({"branch": "bs64"}, "branch"),
    ({"candidate_sha256": None}, "candidate_sha256"),
    ({"decision": "no_veto", "branch": "bs128", "comparator_sha256": None}, "comparator_sha256"),
    ({"e1_verdict_sha256": "0" * 64}, "e1_verdict_sha256"),
])
def test_every_required_gate_field_is_enforced(tmp_path, over, match):
    with pytest.raises(SystemExit, match=match):
        BD.check_gate(_gate(tmp_path / "g", **over))


def test_a_complete_gate_record_is_accepted_and_summarised(tmp_path):
    g = BD.check_gate(_gate(tmp_path / "ok", decision="no_veto", branch="bs128",
                            candidate_sha256="c" * 64, comparator_sha256="a" * 64))
    assert g["executed"] is True and g["decision"] == "no_veto" and g["branch"] == "bs128"
    assert g["candidate_sha256"] == "c" * 64 and g["comparator_sha256"] == "a" * 64
    assert g["e1_verdict_sha256"] == VERDICTS_SHA and BD._is_sha(g["sha256"])


def test_a_recorded_veto_selects_bs32_whatever_e1_says(tmp_path):
    """m10/LOTTE_LOCK.md: "the comparator's recipe (bs32) is what the 200M build trains". The
    version this replaces recorded the veto and trained bs128 anyway (B6)."""
    root = tmp_path / "v"
    _gate(root, decision="veto", branch="bs128", candidate_sha256="c" * 64,
          comparator_sha256="a" * 64)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD.BL, "verdicts", lambda path=None: {
            "registry_sha256": REGISTRY_SHA,
            "selected": {"batch": 128, "student": "bge-small"}})
        _cfg, p = _config(root, dose=TINY_DOSE, reserved_hours=1.0,
                          lotte_gate=str(root / "lotte_gate.json"))
        mp.setattr(BD.BL, "DOSE", TINY_DOSE)
        mp.setattr(BD, "MANIFEST_PATH", _manifest_path(root))
        cfg, _q = BL.load(p)
        ctx = {}
        _reg, batch, source, gate, _rep = BD._preflight(
            ctx, cfg, p, smoke=False, device="cuda", batch=None, lotte_gate=None,
            compile_step=False, real_eval=False, max_len=None, ckpt_every=None, n_fit=None)
    assert batch == 32, "a veto selects the comparator's bs32 recipe"
    assert "VETO" in source and gate["decision"] == "veto"
    assert gate["manifest"]["branch"] == "bs128" and gate["manifest"]["sha256"] == gate["manifest_sha256"]


def test_the_gate_must_be_bound_to_the_committed_manifest(tmp_path):
    """Astra 2026-09-10, finding 2: the arm records are mutable; the committed manifest is the
    anchor. Outside a smoke a missing manifest refuses; a stale or disagreeing one refuses."""
    root = tmp_path / "m"
    g = BD.check_gate(_gate(root, decision="no_veto", branch="bs128", candidate_sha256="c" * 64,
                            comparator_sha256="a" * 64), smoke=True)
    assert BD.check_gate_manifest(g, manifest_path=_manifest_path(root))["branch"] == "bs128"
    assert BD.check_gate_manifest(g, manifest_path=root / "missing.json", smoke=True) is None
    with pytest.raises(SystemExit, match="no checkpoint manifest"):
        BD.check_gate_manifest(g, manifest_path=root / "missing.json")
    # the manifest changed after the gate read it
    man = json.loads(_manifest_path(root).read_text())
    man["written_at"] = "later"
    _manifest_path(root).write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="would run under another"):
        BD.check_gate_manifest(g, manifest_path=_manifest_path(root))
    # a manifest for another checkpoint, with the record's sha pointing at it
    man["candidate"]["sha256"] = "d" * 64
    _manifest_path(root).write_text(json.dumps(man))
    g2 = dict(g, manifest_sha256=BD.sha256_file(_manifest_path(root)))
    with pytest.raises(SystemExit, match="the gate record says"):
        BD.check_gate_manifest(g2, manifest_path=_manifest_path(root))


def test_an_unaffordable_allocation_refuses_before_any_training(tmp_path):
    with pytest.raises(SystemExit, match="above the recorded"):
        _run_refusing(tmp_path / "aff", device="cuda", rate=10.0, price=20.0)
    assert not (tmp_path / "aff" / "m13build" / "BUILD-200M" / "receipt.json").exists()


def test_an_existing_terminal_record_refuses_a_bare_re_run_and_a_resume(tmp_path):
    root = tmp_path / "l"
    d = _out_dir(root)
    (d / "record.json").write_text(json.dumps({"arm": "BUILD-200M", "complete": True}))
    with pytest.raises(SystemExit, match="already exists"):
        _run_refusing(root, device="cuda")
    with pytest.raises(SystemExit, match="TERMINAL"):
        _run_refusing(root, device="cuda", resume=True)


def test_the_pending_batch_refusal_reaches_the_controller(tmp_path):
    root = tmp_path / "m"
    _out_dir(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BD, "WORK", root / "m13build")
        mp.setattr(BD.BL, "DOSE", TINY_DOSE)
        _cfg, p = _config(root, dose=TINY_DOSE, reserved_hours=1.0)
        with pytest.raises(SystemExit, match="E1 verdict is PENDING"):
            BD.run(str(p), device="cuda")


# ------------------------------------------------- the default artifacts stay byte-identical ----

# The pre-M13 checkpoint `extra()` schema, copied from `trainer10.train_arm` before the loss
# sidecar was added. A checkpoint written on the DEFAULT path must still carry exactly these keys.
EXTRA_KEYS_BEFORE = ["cycle_end_evals", "eval_kinds", "evals", "examples", "fingerprint",
                     "losses", "read_evals", "stopped"]


def test_a_default_checkpoint_gains_no_keys(tmp_path):
    """B11: `n_losses` was added unconditionally, so every default checkpoint's schema changed
    even though `n_losses == len(losses)` without a sidecar."""
    import trainer10 as Tr
    q, d = FakeStream(1), FakeStream(2)
    ck = tmp_path / "ck.pt"
    r = Tr.train_arm(Toy(), D.batch_fn(q, d, "75/25"), total_steps=4, cycles=1,
                     ckpt_path=ck, ckpt_every=2, device="cpu", batch_size=BS)
    assert sorted(torch.load(ck, weights_only=False)["extra"]) == EXTRA_KEYS_BEFORE
    assert "n_losses" not in r and "loss_log" not in r
    assert len(r["losses"]) == 4                    # the whole history, untrimmed

    ck2, log = tmp_path / "ck2.pt", tmp_path / "losses.jsonl"
    r2 = Tr.train_arm(Toy(), D.batch_fn(q, d, "75/25"), total_steps=4, cycles=1,
                      ckpt_path=ck2, ckpt_every=2, device="cpu", batch_size=BS,
                      loss_log=str(log))
    assert sorted(torch.load(ck2, weights_only=False)["extra"]) == \
        sorted(EXTRA_KEYS_BEFORE + ["n_losses"])
    assert r2["n_losses"] == 4 and r2["loss_log"] == str(log)


def _stub_assemble(mp, entry):
    """The smallest set of stubs that lets `assemble_arm` build its manifest without a corpus."""
    class FakeRescreen:
        QUERY_SEGMENTS = ()
        protected_ident = staticmethod(lambda: {"stub": True})
        load_report = staticmethod(lambda: {"stub": True})
        validate = staticmethod(lambda rep, consumed: True)

    ids = [np.arange(3, dtype=np.int32) for _ in range(16)]
    T = np.zeros((16, 4), dtype=np.float32)
    mp.setitem(sys.modules, "rescreen10", FakeRescreen)
    mp.setattr(CL, "resolve_arm_name", lambda name, reg: name)
    mp.setattr(CL, "_registry_arms", lambda reg: {"X": entry})
    mp.setattr(CL, "arm_batch", lambda e, reg: 8)
    mp.setattr(CL, "resolve_arm_pattern", lambda n, e, r: ("75/25", {"from": "stub"}))
    mp.setattr(CL, "arm_doc_count", lambda e, p, b: 16)
    mp.setattr(CL, "release_arena", lambda *a, **k: None)
    mp.setattr(CL, "guard_cross_role", lambda a, b: {"ok": True})
    mp.setattr(CL, "build_query_stream",
               lambda *a, **k: (D.Stream(ids, T, pad_id=0, batch_size=8), {"stub": "q"}))

    def doc(n, tok, **k):
        k.setdefault("consumed", {})["documents"] = np.array([], dtype=np.int64)
        meta = {"stub": "d"}
        if k.get("epoch_shuffle"):
            meta["epoch_shuffle"] = True
        return D.Stream(ids, T, pad_id=0, batch_size=8), meta
    mp.setattr(CL, "build_doc_stream", doc)


def test_the_default_assemble_manifest_gains_no_keys(tmp_path):
    """B11: `document_policy: null` was emitted for every arm. The manifest is hashed into every
    checkpoint's recipe fingerprint, so that changed each screen arm's identity for nothing."""
    policy = {"policy": "all_eligible_rescreened", "repeat": True, "reshuffle_per_epoch": True,
              "n": None}
    with pytest.MonkeyPatch.context() as mp:
        _stub_assemble(mp, {"dose_examples": 1024})
        _fn, plain = CL.assemble_arm("X", FakeTok(), "bge-small", registry={"arms": {}})
    with pytest.MonkeyPatch.context() as mp:
        _stub_assemble(mp, {"dose_examples": 1024, "documents": policy})
        _fn, built = CL.assemble_arm("X", FakeTok(), "bge-small", registry={"arms": {}})
    assert "document_policy" not in plain, "a default arm's manifest is the pre-M13 one"
    assert "epoch_shuffle" not in plain["document"]
    assert built["document_policy"] == policy and built["document"]["epoch_shuffle"] is True
    assert sorted(built) == sorted(list(plain) + ["document_policy"])


def test_the_fingerprint_covers_the_sampler_and_the_gate(tmp_path, monkeypatch):
    """B10: `run_arm.code_identity` covers run_arm/trainer10/nano10 but not the corpus path, so a
    sampler change left an unchanged manifest and a resumable checkpoint."""
    assert set(BD.LOADER_FILES) == {"corpus_loader.py", "data10.py"}
    plan = {"arm": "X", "dose_examples": 1, "batch": 32, "pattern": "75/25", "student": "s",
            "n_layers": 3, "head": "linear", "objective": "squared_l2", "warm_start": "linear",
            "total_steps": 1, "cycle_end_steps": [0]}
    g1 = {"sha256": "a" * 64}
    base = BD.fingerprint(plan, {}, 0, None, "cpu", "cfg", gate=g1)
    assert base == BD.fingerprint(plan, {}, 0, None, "cpu", "cfg", gate=g1)
    assert base != BD.fingerprint(plan, {}, 0, None, "cpu", "cfg", gate={"sha256": "b" * 64})
    monkeypatch.setattr(BD, "loader_identity", lambda: "changed")
    assert base != BD.fingerprint(plan, {}, 0, None, "cpu", "cfg", gate=g1)


def test_the_real_config_validates_against_the_live_registry():
    """The shipped `m13/build_config.json`, against the registry that is on disk now."""
    cfg, _ = BL.load()
    rep = BL.validate(cfg, smoke=True)
    assert rep["registry_sha256"] == BL.R.sha256_file(BL.R.REGISTRY)
    assert rep["dose_examples"] == 200_000_000 and rep["data_cut"] == "none"
    assert cfg["arm_entry"]["batch"] == "PENDING", "the batch comes from the E1 verdict"


def test_the_gate_must_match_the_e1_selection(tmp_path):
    """Codex re-check 2026-09-10 (B6): a `skipped` gate under a bs128 selection is the bypass the
    veto exists to prevent; a gate for the other branch is a gate for a different build."""
    with pytest.raises(SystemExit):
        BD.check_gate(_gate(tmp_path / "a"), e1_batch=128, smoke=True,
                      arm_records_dir=tmp_path / "none")                          # skipped under bs128
    with pytest.raises(SystemExit):
        BD.check_gate(_gate(tmp_path / "b", decision="no_veto", branch="bs128",
                            comparator_sha256="b" * 64), e1_batch=32, smoke=True,
                      arm_records_dir=tmp_path / "none")                          # ran for bs128, E1 said 32
    with pytest.raises(SystemExit):
        BD.check_gate(_gate(tmp_path / "c", decision="veto", branch="bs32",
                            comparator_sha256="b" * 64), e1_batch=32, smoke=True,
                      arm_records_dir=tmp_path / "none")                          # a veto in the skipped branch
    assert BD.check_gate(_gate(tmp_path / "d"), e1_batch=32, smoke=True,
                         arm_records_dir=tmp_path / "none")["e1_batch"] == 32
    g = BD.check_gate(_gate(tmp_path / "e", decision="no_veto", branch="bs128",
                            comparator_sha256="b" * 64), e1_batch=128, smoke=True,
                      arm_records_dir=tmp_path / "none")
    assert g["branch"] == "bs128" and g["arm_records_checked"] is False


def test_the_gate_checkpoints_must_be_the_registered_e_checkpoints(tmp_path):
    """Codex re-check 2026-09-10 (B6): sha-shaped is not enough; the gate must name the two A100 E
    checkpoints the published arm records carry, and outside a smoke both records must exist."""
    recs = tmp_path / "results"
    recs.mkdir()
    c32, c128 = "1" * 64, "2" * 64
    (recs / "m10_arm_E-bs32.json").write_text(json.dumps({"checkpoints": {"cycle3": {"sha256": c32}}}))
    (recs / "m10_arm_E-bs128.json").write_text(json.dumps({"checkpoints": {"cycle3": {"sha256": c128}}}))
    with pytest.raises(SystemExit):                       # default candidate "a"*64 is nobody's
        BD.check_gate(_gate(tmp_path / "a"), e1_batch=32, arm_records_dir=recs)
    g = BD.check_gate(_gate(tmp_path / "b", candidate_sha256=c32), e1_batch=32, arm_records_dir=recs)
    assert g["arm_records_checked"] is True
    with pytest.raises(SystemExit):                       # comparator must be E-bs32's
        BD.check_gate(_gate(tmp_path / "c", decision="veto", branch="bs128", candidate_sha256=c128,
                            comparator_sha256="9" * 64), e1_batch=128, arm_records_dir=recs)
    g = BD.check_gate(_gate(tmp_path / "d", decision="veto", branch="bs128", candidate_sha256=c128,
                            comparator_sha256=c32), e1_batch=128, arm_records_dir=recs)
    assert g["decision"] == "veto"
    with pytest.raises(SystemExit):                       # no records outside a smoke
        BD.check_gate(_gate(tmp_path / "e", candidate_sha256=c32), e1_batch=32,
                      arm_records_dir=tmp_path / "nowhere")
    assert BD.check_gate(_gate(tmp_path / "f", candidate_sha256=c32), e1_batch=32,
                         arm_records_dir=tmp_path / "nowhere", smoke=True)["arm_records_checked"] is False


def test_a_failing_fastembed_parity_leaves_the_freeze_unverified(tmp_path, monkeypatch):
    """Codex re-check 2026-09-10 (B8): fastembed is the shipped serving path, so a served result
    below the bar, or no served result, must not verify the freeze even when ORT passes."""
    model = Toy()
    ck = tmp_path / "cycle3.pt"
    torch.save({"model": model.state_dict()}, ck)
    monkeypatch.setattr(BD.R, "dev6", lambda m, verbose=True: {"macro": 0.5})
    monkeypatch.setattr(BD, "sha256_file", lambda p: "0" * 64)
    monkeypatch.setattr(BD.N, "export_onnx", lambda m, d, max_len=512: {"path": str(d)})
    monkeypatch.setattr(BD.N, "export_parity", lambda m, d, t: {"min_cos": 1.0})
    for fe in ({"served": False, "error": "no fastembed"},
               {"served": True, "min_cos": 0.9, "pass_min_cos_1e-4": False}):
        monkeypatch.setattr(BD, "fastembed_parity", lambda m, d, t, _fe=fe: _fe)
        out = BD.freeze_checkpoint(model, tmp_path, ck, smoke=False, verbose=False)
        assert out["verified"] is False and "fastembed" in out["unverified_why"]
    monkeypatch.setattr(BD, "fastembed_parity",
                        lambda m, d, t: {"served": True, "min_cos": 1.0, "pass_min_cos_1e-4": True})
    assert BD.freeze_checkpoint(model, tmp_path, ck, smoke=False, verbose=False)["verified"] is True
