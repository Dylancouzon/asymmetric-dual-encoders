"""What the screen-arm runner must refuse, and the arithmetic it must get right.

CPU only, streams and evaluations mocked: the point is the runner's contract with the registry
(`m10/screen_registry.json`) and with `trainer10`, not the BERT. The four refusals are each a way
a screen arm could be spent on nothing, and the two schedule tests are the family-F reading points
— 5M inside a 20M schedule — which is the one piece of arithmetic no other test covers.
"""
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
import torch

import run_arm as R
import trainer10 as T


# --------------------------------------------------------------------------------- fixtures ----

class FakeTok:
    pad_token_id = 0


class FakeModel(torch.nn.Module):
    d_in = 1152

    def __init__(self):
        super().__init__()
        self.tok = FakeTok()
        self.head = torch.nn.Linear(4, 4)

    def to(self, *a, **k):
        return self

    def n_params(self):
        return 34_540_672

    def under_cap(self):
        return True


class Toy(torch.nn.Module):
    """The same stand-in `test_trainer10` uses: exercises the loop, not the model."""

    def __init__(self, d_in=12, d_out=6):
        super().__init__()
        self.head = torch.nn.Linear(d_in, d_out)

    def forward(self, ids, mask):
        x = ids.float() @ torch.ones(ids.shape[-1], self.head.in_features) / ids.shape[-1]
        return torch.nn.functional.normalize(self.head(x), dim=-1)


def make_batch_fn(n=8, s=5):
    def f(step, kind):
        g = torch.Generator().manual_seed(1000 + step)
        return (torch.randint(0, 50, (n, s), generator=g),
                torch.ones(n, s, dtype=torch.long),
                torch.nn.functional.normalize(torch.randn(n, 6, generator=g), dim=-1))
    return f


@pytest.fixture
def sandbox(monkeypatch):
    """Redirect every output path, so no test can touch a real arm record (§Hazards)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        monkeypatch.setattr(R, "WORK", d / "m10arms")
        monkeypatch.setattr(R, "SMOKE_WORK", d / "m10arms" / "smoke")
        monkeypatch.setattr(R, "RESULTS", d / "results")
        (d / "results").mkdir()
        yield d


def mock_pipeline(monkeypatch, *, cycle_ends=(0.40, 0.41, 0.42), stopped=None):
    """Everything below the runner: the model, the corpus, the warm start, the loop, DEV-6."""
    monkeypatch.setattr(R.N, "Nano10", lambda *a, **k: FakeModel())
    monkeypatch.setattr(R.CL, "assemble_arm",
                        lambda *a, **k: (make_batch_fn(), {"arm": a[0], "mocked": True}))
    monkeypatch.setattr(R, "streams_of", lambda bf: (object(), object()))
    monkeypatch.setattr(R, "warm_start", lambda *a, **k: {"registered": "linear", "n_fit": 60000})
    monkeypatch.setattr(R, "dev6", lambda m, verbose=True: {"macro": 0.55, "per_component": {}})
    rec = {"steps_run": 10, "start_step": 0, "total_steps": 10, "stopped": stopped,
           "examples": 320, "examples_this_run": 320, "seconds": 1.0, "examples_per_s": 320.0,
           "mix": {"q_share": 0.75}, "evals": list(cycle_ends), "eval_kinds": ["end"] * 3,
           "cycle_end_evals": list(cycle_ends), "read_evals": [], "losses": [1.0, 0.9]}
    monkeypatch.setattr(R.Tr, "train_arm", lambda *a, **k: dict(rec))
    return rec


# --------------------------------------------------------------------------------- refusals ----

def test_it_refuses_when_the_screen_lock_does_not_validate(monkeypatch, sandbox):
    monkeypatch.setattr(R.SL, "validate", lambda r=None: ["a contrast names a missing arm"])
    with pytest.raises(SystemExit, match="screen_lock.validate"):
        R.run("A1")


def test_it_refuses_an_unregistered_arm(sandbox):
    with pytest.raises(SystemExit, match="not a registered arm"):
        R.run("A99")


def test_it_refuses_an_arm_that_is_not_trained(sandbox):
    # A4 is not trained separately -- it IS the ANCHOR arm
    with pytest.raises(SystemExit, match="not `trained: true`"):
        R.run("A4")


def test_it_refuses_a_cut_arm(monkeypatch, sandbox):
    reg = copy.deepcopy(R.SL.cfg())
    reg["arms"]["A1"]["cut"] = "CUT for this test"
    monkeypatch.setattr(R.SL, "cfg", lambda: reg)
    # `screen_lock` itself refuses a cut-but-trained arm, so stub it out to reach the runner's
    # own check -- which must stand on its own for a cut arm the lock accepts (C-M9init: cut and
    # `trained: false`, where the trained check fires first).
    monkeypatch.setattr(R.SL, "validate", lambda r=None: [])
    with pytest.raises(SystemExit, match="is CUT"):
        R.run("A1")


def test_it_refuses_a_cut_corpus_arm_while_the_data_cut_is_unregistered(sandbox):
    """A2/A3/ANCHOR and every arm inheriting the anchor's corpus. `assemble_arm` is the enforcer;
    this only fails before a 5M-row corpus is read."""
    assert R.CL.data_cut_count(R.SL.cfg()) is None, "§0b has been filled: update this test"
    with pytest.raises(SystemExit, match="unique_text_count"):
        R.run("A2")


def test_it_refuses_to_re_run_a_complete_record(monkeypatch, sandbox):
    d = R.WORK / "A1"
    d.mkdir(parents=True)
    (d / "record.json").write_text(json.dumps({"arm": "A1", "complete": True}))
    with pytest.raises(SystemExit, match="already exists and is complete"):
        R.run("A1")


def test_a_failed_warm_start_refuses_rather_than_training_a_fresh_head(monkeypatch):
    class Boom:
        head_kind = "linear"
    monkeypatch.setattr(R, "fit_sample", lambda *a, **k: (["x"], np.zeros((1, 4), np.float32)))
    monkeypatch.setattr(R.N, "pooled_features", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("shape")))
    with pytest.raises(SystemExit, match="warm start .* FAILED"):
        R.warm_start(Boom(), {"warm_start": "linear"}, object(), {"n_fit": 8, "seed": 21})


# ------------------------------------------------------------------------ schedule arithmetic --

def test_a_20M_family_F_arm_reads_at_5M_10M_and_20M_in_steps():
    reg = R.SL.cfg()
    p = R.arm_plan("F-bge-small", reg)
    assert p["dose_examples"] == 20_000_000 and p["batch"] == 32
    assert p["total_steps"] == 625_000
    # 3 cycles over the arm's OWN 20M dose, the last absorbing the remainder (`nano10.lr_at`)
    assert p["cycle_end_steps"] == [208_332, 416_665, 624_999]
    assert p["read_points"] == [{"examples": 5_000_000, "step": 156_249},
                                {"examples": 10_000_000, "step": 312_499},
                                {"examples": 20_000_000, "step": 624_999}]
    # the 20M read IS the final cycle end and the 10M read IS a schedule midpoint, so only the
    # 5M point needs an extra evaluation
    per = p["total_steps"] // 3
    mids = {c * per + per // 2 for c in range(3)}
    extra = [s for _e, s in [(r["examples"], r["step"]) for r in p["read_points"]]
             if s not in set(p["cycle_end_steps"]) and s not in mids]
    assert extra == [156_249]
    assert p["projected_hours"] == round(20_000_000 / 890.0 / 3600, 2)
    assert p["dose_rounding"]["examples_dropped"] == 0


def test_a_5M_arm_has_three_annealed_cycle_ends_and_no_read_points():
    p = R.arm_plan("A1", R.SL.cfg())
    assert p["total_steps"] == 156_250
    assert p["cycle_end_steps"] == [52_082, 104_165, 156_249]
    assert p["read_points"] == []
    # the last step of training IS the final cycle end: no step runs after the last COV read
    assert p["cycle_end_steps"][-1] == p["total_steps"] - 1


def test_the_bs128_arm_carries_its_own_batch_and_rate():
    p = R.arm_plan("E-bs128", R.SL.cfg())
    assert p["batch"] == 128 and p["total_steps"] == 5_000_000 // 128
    assert p["projected_at_ex_per_s"] == 1517.0
    assert p["cloud_only"], "E-bs128 does not run on the box at realistic sequence lengths"


def test_the_plan_covers_the_w8_band_1_order():
    rows = R.plan(R.SL.cfg())
    assert [r["arm"] for r in rows] == R.BAND1_ORDER
    assert {r["arm"] for r in rows if r["cloud_only"]} == {"E-bs128"}
    # every band-1 arm is registered, trained and not cut
    arms = R.SL.cfg()["arms"]
    for r in rows:
        assert arms[r["arm"]]["trained"] is True and "cut" not in arms[r["arm"]], r["arm"]


def test_a_dose_that_does_not_divide_by_the_batch_is_FLOORED_and_recorded():
    """`E-bs128`: 5,000,000 examples at batch 128 is 39,062.5 steps. The dose is a cap, so it
    floors -- 64 examples short of bs32's 5,000,000 -- and the record says so."""
    total, ends, reads, rounding = R.schedule(5_000_000, 128, {})
    assert total == 39_062 and rounding["examples_run"] == 4_999_936
    assert rounding["examples_dropped"] == 64 and "floor" in rounding["rule"]
    assert ends[-1] == total - 1 and reads == []


def test_the_shape_check_catches_a_registry_that_disagrees_with_arm_smoke():
    with pytest.raises(SystemExit, match="student"):
        R.check_shape("F-MiniLM-L6", {"student": "bge-small"}, R.shape_for("F-MiniLM-L6"))
    with pytest.raises(SystemExit, match="objective"):
        R.check_shape("D-NORM", {"objective": "squared_l2"}, R.shape_for("D-NORM"))
    # the registry's own strings must pass, aliases included
    reg = R.SL.cfg()
    for name in R.BAND1_ORDER:
        R.check_shape(name, reg["arms"][name], R.arm_plan(name, reg) and R.shape_for(
            R.CL.resolve_arm_name(name, reg)))


# ------------------------------------------------------------------------------- the record ----

def test_the_record_schema(monkeypatch, sandbox):
    mock_pipeline(monkeypatch)
    rec = R.run("A1", device="cpu", verbose=False)
    for k in ("arm", "family", "status", "complete", "smoke", "device", "seed", "recipe",
              "schedule", "warm_start", "cov", "dev6", "training", "throughput_ex_per_s",
              "checkpoints", "final_checkpoint", "final_checkpoint_sha256", "assemble_manifest",
              "registry_sha256", "git_head", "evaluation_mode"):
        assert k in rec, k
    assert rec["arm"] == "A1" and rec["status"] == "complete" and rec["seed"] == 0
    assert rec["cov"]["final_macro"] == 0.42
    assert rec["cov"]["cycle_end_macros"] == [0.40, 0.41, 0.42]
    assert rec["dev6"]["macro"] == 0.55
    assert rec["throughput_ex_per_s"] == 320.0
    assert rec["recipe"]["objective"] == "squared_l2" and rec["recipe"]["pattern"] == "75/25"
    # both artifacts, identical
    on_disk = json.loads((R.WORK / "A1" / "record.json").read_text())
    assert on_disk == json.loads((R.RESULTS / "m10_arm_A1.json").read_text())
    assert on_disk["registry_sha256"] == R.sha256_file(R.REGISTRY)


def test_a_stopped_arm_is_reported_failed_and_gets_no_dev6(monkeypatch, sandbox):
    mock_pipeline(monkeypatch, cycle_ends=(0.40,), stopped="kill: two consecutive evaluations")
    rec = R.run("A1", device="cpu", verbose=False)
    assert rec["status"] == "failed" and rec["dev6"] is None
    assert rec["complete"] is True, "the record exists so the arm is not silently re-run"


def test_plateau_at_the_final_cycle_is_completion_not_failure(monkeypatch, sandbox):
    mock_pipeline(monkeypatch, stopped="plateau at cycle 3")
    rec = R.run("A1", device="cpu", verbose=False)
    assert rec["status"] == "complete" and rec["stopped_is_completion"] is True
    assert rec["dev6"] is not None


def test_a_smoke_never_writes_the_real_record(monkeypatch, sandbox):
    mock_pipeline(monkeypatch)
    rec = R.run("A1", device="cpu", smoke_steps=6, verbose=False)
    assert rec["smoke"] is True and rec["evaluation_mode"] == "stub" and rec["dev6"] is None
    assert (R.SMOKE_WORK / "A1" / "record.json").exists()
    assert not (R.WORK / "A1" / "record.json").exists()
    assert not (R.RESULTS / "m10_arm_A1.json").exists()
    # the smoke's dose is its own, so `assemble_arm` derives a tiny document count
    assert rec["registry_dose_overridden_for_smoke"] == 6 * 32
    assert R.SL.cfg()["arms"]["A1"]["dose_examples"] == 5_000_000, "the real registry is untouched"


# ------------------------------------------------- the two trainer hooks the runner depends on --

def test_read_points_are_recorded_and_kept_out_of_the_kill_and_plateau_state():
    seen = []

    def ev(m, step, kind):
        seen.append((step, kind))
        return {"end": 0.5, "mid": 0.5}.get(kind, 0.1)      # reads collapse; ends do not

    # total_steps 30, 3 cycles: ends [9, 19, 29], midpoints {5, 15, 25} -- so 3 and 11 are
    # genuinely extra reading points, and a read step that IS a midpoint stays a midpoint.
    r = T.train_arm(Toy(), make_batch_fn(), total_steps=30, eval_fn=ev, read_steps=[3, 11, 5])
    assert [k for _s, k in seen].count("read") == 2
    assert [x["step"] for x in r["read_evals"]] == [3, 11]
    assert 0.1 not in r["evals"], "a read must not enter the kill rule's sequence"
    assert r["stopped"] is None or r["stopped"].startswith("plateau"), r["stopped"]


def test_cycle_end_checkpoints_are_retained_and_resume_carries_the_evals():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fmt = str(d / "cycle{cycle}.pt")
        n = [0]

        def ev(m, step, kind):
            n[0] += 1
            return 0.4 + 0.01 * n[0]

        torch.manual_seed(0)
        part = T.train_arm(Toy(), make_batch_fn(), total_steps=30, eval_fn=ev,
                           ckpt_path=d / "ckpt.pt", ckpt_every=11, cycle_ckpt_fmt=fmt,
                           read_steps=[3])
        assert (d / "cycle1.pt").exists() and (d / "cycle3.pt").exists()
        assert len(part["cycle_end_evals"]) == 3

        # a resume from the ROLLING checkpoint carries the evaluation history, reads included
        ck = d / "ckpt.pt"
        step, extra = T.load(ck, Toy(), torch.optim.AdamW(Toy().parameters()))
        assert extra["cycle_end_evals"], "a cycle-end eval must be in the rolling checkpoint"
        assert "read_evals" in extra
        rest = T.train_arm(Toy(), make_batch_fn(), total_steps=30, eval_fn=ev,
                           resume_from=ck, cycle_ckpt_fmt=fmt)
        assert rest["cycle_end_evals"][:len(extra["cycle_end_evals"])] == \
            extra["cycle_end_evals"]
        assert rest["read_evals"] == extra["read_evals"]


def test_streams_of_recovers_the_two_streams_assemble_arm_built():
    import corpus_loader as CL
    import data10 as D

    class FakeQ(CL.FormBalancedStream):
        def __init__(self):
            pass

    class FakeD(D.Stream):
        def __init__(self):
            pass

    q, dd = FakeQ(), FakeD()
    bf = D.batch_fn(q, dd, pattern="75/25")
    assert R.streams_of(bf) == (q, dd)
    with pytest.raises(SystemExit, match="could not recover"):
        R.streams_of(lambda step, kind: None)
