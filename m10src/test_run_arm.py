"""What the screen-arm runner must refuse, and the arithmetic it must get right.

CPU only, streams and evaluations mocked: the point is the runner's contract with the registry
(`m10/screen_registry.json`) and with `trainer10`, not the BERT. The four refusals are each a way
a screen arm could be spent on nothing, and the two schedule tests are the family-F reading points
— 5M inside a 20M schedule — which is the one piece of arithmetic no other test covers.
"""
import copy
import hashlib
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


def write_f_verdict(winner="bge-small", *, registry_sha=None, record_shas=None,
                    f_record_status="complete", f_record_final_sha="0" * 64, contrast=None):
    """A stub of the file the CONTRAST step writes (`f_verdict`'s schema). Written into the
    sandbox's RESULTS, never the repo's."""
    R.RESULTS.mkdir(parents=True, exist_ok=True)
    reg = R.SL.cfg()
    shas = []
    for n in R.f_arms(reg):
        rp = R.RESULTS / f"m10_arm_{R.slug(n)}.json"
        rec = {"arm": n, "status": f_record_status, "complete": f_record_status == "complete"}
        if f_record_final_sha:
            rec["final_checkpoint_sha256"] = f_record_final_sha
        rp.write_text(json.dumps(rec))
        shas.append(R.sha256_file(rp))
    v = {"winner": winner,
         "contrast": contrast or {"rule": "C1", "point": 0.01, "lower": 0.004, "resolved": True},
         "registry_sha256": registry_sha or R.sha256_file(R.REGISTRY),
         "sha256_of_F_records": record_shas if record_shas is not None else shas}
    p = R.RESULTS / R.F_VERDICT_NAME
    p.write_text(json.dumps(v))
    return p


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


def test_it_refuses_a_cut_corpus_arm_while_the_data_cut_is_unregistered(sandbox, monkeypatch):
    """A2/A3/ANCHOR and every arm inheriting the anchor's corpus. `assemble_arm` is the enforcer;
    this only fails before a 5M-row corpus is read. §0b was REGISTERED 2026-09-07
    (2,651,572), so the unregistered state is simulated by removing the count from the registry."""
    reg = json.loads(json.dumps(R.SL.cfg()))
    assert reg["data_cut"].pop("unique_text_count") == 2651572, "the registered §0b count moved"
    monkeypatch.setattr(R.SL, "cfg", lambda: reg)
    monkeypatch.setattr(R.CL, "data_cut_count", lambda registry=None: None)
    with pytest.raises(SystemExit, match="unique_text_count"):
        R.run("A2")


def test_the_registered_data_cut_is_the_A3_minimum():
    """§0b, registered 2026-09-07 before any arm: A2 4,439,407 · A3 2,651,572 · A4 3,486,034."""
    dc = R.SL.cfg()["data_cut"]
    assert dc["unique_text_count"] == 2651572
    c = dc["registered"]["counts"]
    assert min(c.values()) == c["A3"] == dc["unique_text_count"]
    assert dc["registered"]["before_any_arm"] is True


def test_it_refuses_to_re_run_a_complete_record(monkeypatch, sandbox):
    d = R.WORK / "A1"
    d.mkdir(parents=True)
    (d / "record.json").write_text(json.dumps({"arm": "A1", "complete": True}))
    with pytest.raises(SystemExit, match="already exists"):
        R.run("A1")


def test_it_refuses_when_EITHER_published_path_already_carries_a_complete_record(sandbox):
    """finding 13: only `work/m10arms/<arm>/record.json` was checked, so an arm whose PUBLISHED
    `results/m10_arm_<arm>.json` existed was re-run and overwrote it."""
    (R.RESULTS / "m10_arm_A1.json").write_text(json.dumps({"arm": "A1", "complete": True}))
    with pytest.raises(SystemExit, match="already exists"):
        R.run("A1")


def test_an_incomplete_record_does_not_block_a_run(monkeypatch, sandbox):
    """item E, new contract: ANY record file at a published path refuses a BARE re-run --
    parseable or not, complete or not. `--resume` is what continues a NON-terminal one (a
    TERMINAL one, complete or failed, still refuses `--resume` -- see the tests below)."""
    mock_pipeline(monkeypatch)
    write_f_verdict()
    (R.RESULTS / "m10_arm_A1.json").write_text(json.dumps({"arm": "A1", "complete": False}))
    with pytest.raises(SystemExit, match="already exists"):
        R.run("A1", device="cuda", verbose=False)
    # a valid --resume needs the non-terminal WORK record and the rolling checkpoint (pass 3)
    d = R.WORK / "A1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps({"arm": "A1", "complete": False}))
    (d / "ckpt.pt").write_bytes(b"ckpt")
    assert R.run("A1", device="cuda", resume=True, verbose=False)["status"] == "complete"


def test_a_malformed_record_refuses_a_bare_re_run_but_permits_resume(monkeypatch, sandbox):
    """item E: "parseable or not, complete or not" -- a malformed record still blocks a bare
    re-run, and is treated as non-terminal (not proof the arm finished), so `--resume` proceeds."""
    mock_pipeline(monkeypatch)
    write_f_verdict()
    (R.RESULTS / "m10_arm_A1.json").write_text("{not valid json")
    with pytest.raises(SystemExit, match="already exists"):
        R.run("A1", device="cuda", verbose=False)
    # Codex pass 3: an unparseable record makes terminality UNKNOWABLE, so --resume refuses too
    with pytest.raises(SystemExit, match="unparseable"):
        R.run("A1", device="cuda", resume=True, verbose=False)


def _non_terminal_record(arm="A1"):
    d = R.WORK / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps({"arm": arm, "complete": False, "status": "running"}))
    return d


def test_resume_refuses_when_there_is_no_record_to_continue(monkeypatch, sandbox):
    """Codex pass 3: `--resume` with no record silently became a fresh 20M run."""
    mock_pipeline(monkeypatch)
    write_f_verdict()
    with pytest.raises(SystemExit, match="no record"):
        R.run("A1", device="cuda", resume=True, verbose=False)


def test_resume_refuses_when_the_rolling_checkpoint_is_missing(monkeypatch, sandbox):
    mock_pipeline(monkeypatch)
    write_f_verdict()
    _non_terminal_record()
    with pytest.raises(SystemExit, match="no rolling checkpoint"):
        R.run("A1", device="cuda", resume=True, verbose=False)


def test_a_valid_resume_hands_the_checkpoint_to_the_trainer_never_None(monkeypatch, sandbox):
    mock_pipeline(monkeypatch)
    write_f_verdict()
    d = _non_terminal_record()
    (d / "ckpt.pt").write_bytes(b"ckpt")
    seen = {}
    real = R.Tr.train_arm

    def spy(*a, **k):
        seen["resume_from"] = k.get("resume_from")
        return real(*a, **k)
    monkeypatch.setattr(R.Tr, "train_arm", spy)
    R.run("A1", device="cuda", resume=True, verbose=False)
    assert seen["resume_from"] == str(d / "ckpt.pt")


def test_resume_refuses_when_the_existing_record_is_already_terminal(monkeypatch, sandbox):
    """item E: `--resume` only continues a NON-terminal run; a record that already carries an
    outcome (complete success, or a FAILED record with `terminal: true`) has nothing to resume."""
    mock_pipeline(monkeypatch)
    write_f_verdict()
    (R.RESULTS / "m10_arm_A1.json").write_text(json.dumps({"arm": "A1", "complete": True}))
    with pytest.raises(SystemExit, match="TERMINAL"):
        R.run("A1", device="cuda", resume=True, verbose=False)
    (R.RESULTS / "m10_arm_A1.json").write_text(
        json.dumps({"arm": "A1", "status": "failed", "complete": False, "terminal": True}))
    with pytest.raises(SystemExit, match="TERMINAL"):
        R.run("A1", device="cuda", resume=True, verbose=False)


def test_the_record_is_written_atomically_to_both_published_paths(monkeypatch, sandbox):
    """A truncated JSON on the real path is indistinguishable from an arm that reported nothing,
    so the record lands via a temp file and `os.replace` (finding 13)."""
    seen = []
    real = R.os.replace
    monkeypatch.setattr(R.os, "replace", lambda a, b: (seen.append((str(a), str(b))), real(a, b)))
    mock_pipeline(monkeypatch)
    write_f_verdict()
    rec = R.run("A1", device="cuda", verbose=False)
    dests = [b for _a, b in seen]
    assert str(R.WORK / "A1" / "record.json") in dests
    assert str(R.RESULTS / "m10_arm_A1.json") in dests
    assert all(".tmp" in a for a, _b in seen), seen
    on_disk = json.loads((R.WORK / "A1" / "record.json").read_text())
    assert on_disk == json.loads((R.RESULTS / "m10_arm_A1.json").read_text()) == rec


# ------------------------------------------------------------ the smoke-only knobs (finding 5) --

@pytest.mark.parametrize("kw", [{"max_len": 128}, {"ckpt_every": 5}, {"n_fit": 256},
                                {"compile_step": True}, {"real_eval": True}])
def test_a_real_arm_refuses_every_recipe_knob(sandbox, kw):
    with pytest.raises(SystemExit, match="only be passed with --smoke-steps"):
        R.run("A1", **kw)


def test_the_smoke_may_use_them(monkeypatch, sandbox):
    mock_pipeline(monkeypatch)
    rec = R.run("A1", device="cpu", smoke_steps=6, max_len=128, n_fit=64, verbose=False)
    assert rec["max_len"] == 128 and rec["smoke"] is True


# --------------------------------------------- registered runs require CUDA (item A) -----------

def test_a_registered_run_refuses_a_non_cuda_device(monkeypatch, sandbox):
    """item A: §Recipe is bf16 autocast, and CPU silently trains fp32 -- a confound, not a smoke.
    A real (non-smoke) run must REFUSE anything but `cuda`."""
    mock_pipeline(monkeypatch)
    write_f_verdict()
    with pytest.raises(SystemExit, match="requires --device cuda"):
        R.run("A1", device="cpu", verbose=False)


def test_a_smoke_may_still_pass_cpu(monkeypatch, sandbox):
    """item A's counterpoint: a smoke is exempt -- it is a 60-step CPU path check, not a
    measurement, and this milestone's CPU box has no CUDA device to smoke on at all."""
    mock_pipeline(monkeypatch)
    rec = R.run("A1", device="cpu", smoke_steps=6, verbose=False)
    assert rec["device"] == "cpu" and rec["smoke"] is True


def test_the_cli_default_device_is_cuda():
    """item A: the CLI default for `--device` becomes `cuda`; a smoke may still override it."""
    ap = R.build_argparser()
    assert ap.parse_args(["A1"]).device == "cuda"
    assert ap.parse_args(["A1", "--smoke-steps", "6", "--device", "cpu"]).device == "cpu"


def test_a_real_evaluation_is_prohibited_under_a_smoke(sandbox):
    """finding 10: `--real-eval` reached the 13,416-query COV surface and DEV-6's ~13 GB from a
    60-step CPU smoke. With finding 5 it is not selectable at all, and the CLI no longer has it."""
    with pytest.raises(SystemExit, match="prohibited under --smoke-steps"):
        R.run("A1", smoke_steps=6, real_eval=True)
    # and it is not on the command line at all
    with pytest.raises(SystemExit):
        R.main(["A1", "--smoke-steps", "6", "--real-eval"])       # argparse: unknown argument


# ---------------------------------------------------------------- F's winner (finding 11) ------

def test_a_post_F_arm_refuses_without_F_s_verdict(sandbox):
    with pytest.raises(SystemExit, match="F's WINNER backbone|does not exist"):
        R.run("A1")


def test_a_post_F_arm_refuses_a_verdict_from_another_registry(sandbox):
    write_f_verdict(registry_sha="0" * 64)
    with pytest.raises(SystemExit, match="decided under registry"):
        R.run("A1")


def test_a_post_F_arm_refuses_a_verdict_whose_F_records_have_moved(sandbox):
    write_f_verdict(record_shas=["1" * 64, "2" * 64])
    with pytest.raises(SystemExit, match="the verdict is stale"):
        R.run("A1")


def test_a_post_F_arm_refuses_a_verdict_naming_an_unknown_student(sandbox):
    write_f_verdict(winner="gte-tiny")
    with pytest.raises(SystemExit, match="not a known nano10 student"):
        R.run("A1")


def test_a_post_F_arm_refuses_a_verdict_naming_a_cut_untrained_student(sandbox):
    """item F: `MiniLM-L12-v2` IS a known nano10 student (`STUDENT_ALIAS` covers it), but its only
    registry arm, F-MiniLM-L12, is CUT and never trained -- the verdict cannot be read off it."""
    write_f_verdict(winner="MiniLM-L12-v2")
    with pytest.raises(SystemExit, match="TRAINED, uncut F arm"):
        R.run("A1")


def test_a_post_F_arm_refuses_a_verdict_whose_F_record_is_not_complete(sandbox):
    """item F: a verdict can only be read off a COMPLETED F arm, not a failed one -- the hash
    check alone does not catch this, since a failed record hashes just as reproducibly."""
    write_f_verdict(f_record_status="failed")
    with pytest.raises(SystemExit, match="not 'complete'"):
        R.run("A1")


def test_a_post_F_arm_refuses_a_verdict_whose_F_record_has_no_final_checkpoint_sha(sandbox):
    write_f_verdict(f_record_final_sha=None)
    with pytest.raises(SystemExit, match="final_checkpoint_sha256"):
        R.run("A1")


@pytest.mark.parametrize("bad_contrast", [
    {"point": 0.01, "lower": 0.004, "resolved": True},          # missing `rule`
    {"rule": "C1", "point": 0.01, "lower": 0.004},               # missing `resolved`
    "not even a dict",
])
def test_a_post_F_arm_refuses_a_malformed_contrast_schema(sandbox, bad_contrast):
    write_f_verdict(contrast=bad_contrast)
    with pytest.raises(SystemExit, match="contrast block must carry"):
        R.run("A1")


def test_the_student_of_a_post_F_arm_comes_from_the_verdict_not_from_SHAPES(monkeypatch, sandbox):
    """The anchor's shape says bge-small; if F selects MiniLM-L6, every later arm must train the
    MiniLM backbone (`anchor.init: "F's winner backbone"`)."""
    assert R.shape_for("A1")["student"] == "bge-small"
    write_f_verdict(winner="MiniLM-L6-v2")
    seen = {}
    mock_pipeline(monkeypatch)
    monkeypatch.setattr(R.N, "Nano10",
                        lambda student, **k: (seen.setdefault("student", student), FakeModel())[1])
    rec = R.run("A1", device="cuda", verbose=False)
    assert seen["student"] == "MiniLM-L6"
    assert rec["recipe"]["student"] == "MiniLM-L6"
    assert rec["student_source"].endswith(R.F_VERDICT_NAME)
    assert rec["f_verdict"]["winner"] == "MiniLM-L6-v2"


def test_a_family_F_arm_needs_no_verdict(monkeypatch, sandbox):
    """F runs FIRST, so it cannot wait on its own verdict; its student is in the registry."""
    mock_pipeline(monkeypatch)
    monkeypatch.setattr(R.CL, "data_cut_count", lambda reg=None: 4_000_000)
    rec = R.run("F-bge-small", device="cuda", verbose=False)
    assert rec["recipe"]["student"] == "bge-small"
    assert rec["student_source"].startswith("registry")


def test_a_smoke_needs_no_verdict(monkeypatch, sandbox):
    """`arm_smoke` keeps hard-coded students (shapes only), and a 60-step CPU smoke is the same
    kind of path check — the record says which student it actually ran."""
    mock_pipeline(monkeypatch)
    rec = R.run("A1", device="cpu", smoke_steps=6, verbose=False)
    assert rec["recipe"]["student"] == "bge-small" and rec["f_verdict"] is None
    assert "PENDING" in rec["student_source"]


def test_a_failed_warm_start_refuses_rather_than_training_a_fresh_head(monkeypatch):
    class Boom:
        head_kind = "linear"
    monkeypatch.setattr(R, "fit_sample", lambda *a, **k: (["x"], np.zeros((1, 4), np.float32)))
    monkeypatch.setattr(R.N, "pooled_features", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("shape")))
    with pytest.raises(SystemExit, match="warm start .* FAILED"):
        R.warm_start(Boom(), {"warm_start": "linear"}, object(), {"n_fit": 8, "seed": 21})


def test_a_warm_start_failure_through_run_writes_a_terminal_FAILED_record(monkeypatch, sandbox):
    """item D: an exception INSIDE the warm start happens after the model and corpus are already
    built -- a FAILURE, not a pre-flight refusal (`ctx["started"]` distinguishes the two) -- so it
    must write a terminal FAILED record and re-raise, unlike a refusal before any work starts."""
    monkeypatch.setattr(R.N, "Nano10", lambda *a, **k: FakeModel())
    monkeypatch.setattr(R.CL, "assemble_arm",
                        lambda *a, **k: (make_batch_fn(), {"arm": a[0], "mocked": True}))
    monkeypatch.setattr(R, "streams_of", lambda bf: (object(), object()))
    monkeypatch.setattr(R, "fit_sample", lambda *a, **k: (["x"], np.zeros((1, 4), np.float32)))
    monkeypatch.setattr(R.N, "pooled_features", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("pooled_features shape mismatch")))
    write_f_verdict()
    with pytest.raises(SystemExit, match="warm start .* FAILED"):
        R.run("A1", device="cuda", verbose=False)
    for path in (R.WORK / "A1" / "record.json", R.RESULTS / "m10_arm_A1.json"):
        rec = json.loads(path.read_text())
        assert rec["status"] == "failed" and rec["complete"] is False and rec["terminal"] is True
        assert rec["failure"]["type"] == "SystemExit"
        assert "pooled_features shape mismatch" in rec["failure"]["message"]
        assert rec["final_checkpoint"] is None and rec["dev6"] is None


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
    # against the CONSTANT, not a copy of it: this line hardcoded 890.0 and broke the moment
    # the rate was re-measured at 512 in the runner itself (2026-09-07).
    assert p["projected_hours"] == round(20_000_000 / R.PLAN_RATES[32] / 3600, 2)
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
    reg = R.SL.cfg()

    def recipe(name, **over):
        e = dict(reg["arms"][name])
        e.update(over)
        pat = R.CL.resolve_arm_pattern(name, e, reg)[0]
        return R.resolved_recipe(name, e, reg, pat, R.CL.arm_batch(e, reg))

    with pytest.raises(SystemExit, match="student"):
        R.check_shape("F-MiniLM-L6", recipe("F-MiniLM-L6", student="bge-small"),
                      R.shape_for("F-MiniLM-L6"))
    with pytest.raises(SystemExit, match="objective"):
        R.check_shape("D-NORM", recipe("D-NORM", objective="squared_l2"), R.shape_for("D-NORM"))
    # the registry's own strings must pass, every band-1 arm, aliases included
    for name in R.BAND1_ORDER:
        R.arm_plan(name, reg)


def test_the_shape_check_compares_the_UNTOUCHED_shape_against_the_resolved_recipe(monkeypatch):
    """finding 6: `arm_plan` wrote the registry's batch INTO the shape and then compared the two,
    so `SHAPES` could disagree with the registry about the batch and nothing failed."""
    reg = R.SL.cfg()
    shapes = {k: dict(v) for k, v in R.AS.SHAPES.items()}
    shapes["E-bs128"]["batch"] = 32                     # the registry says 128
    monkeypatch.setattr(R.AS, "SHAPES", shapes)
    with pytest.raises(SystemExit, match="batch: registry 128 vs shape 32"):
        R.arm_plan("E-bs128", reg)


@pytest.mark.parametrize("arm,field,value,msg", [
    ("B-50/50", "pattern", "4Q", "pattern"),
    ("G-MLP", "head", None, "head"),
    ("C-M9init", "head_init", None, "warm_start"),
    ("G-1536", "feature_layers", 3, "feature_layers"),
])
def test_every_shared_field_is_compared_not_just_the_student(arm, field, value, msg):
    """head, mix pattern and warm start were never cross-checked at all (finding 6)."""
    reg = copy.deepcopy(R.SL.cfg())
    e = reg["arms"][arm]
    if value is None:
        e.pop(field, None)
    else:
        e[field] = value
    e["trained"] = True
    e.pop("cut", None)
    with pytest.raises(SystemExit, match=msg):
        R.arm_plan(arm, reg)


# ------------------------------------------------------------------------------- the record ----

def test_the_record_schema(monkeypatch, sandbox):
    mock_pipeline(monkeypatch)
    write_f_verdict()
    rec = R.run("A1", device="cuda", verbose=False)
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
    write_f_verdict()
    rec = R.run("A1", device="cuda", verbose=False)
    assert rec["status"] == "failed" and rec["dev6"] is None
    assert rec["complete"] is False, "item E: a failed record is not `complete`"
    assert rec["terminal"] is True, "the record exists so the arm is not silently re-run"


def test_plateau_at_the_final_cycle_is_completion_not_failure(monkeypatch, sandbox):
    mock_pipeline(monkeypatch, stopped="plateau at cycle 3")
    write_f_verdict()
    rec = R.run("A1", device="cuda", verbose=False)
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
        m = Toy()
        step, extra = T.load(ck, m, torch.optim.AdamW(T.param_groups(m)))
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


# --------------------------------------------- a kill is never a completion (finding 4) --------

def test_a_kill_at_the_FINAL_cycle_end_is_a_FAILURE_not_a_completion(monkeypatch, sandbox):
    """The macro sequence mid/end = .50/.50, .51/.51, .49/.49 fires the kill rule at the LAST
    cycle end, so the arm has all three cycle ends AND a kill. `n_ends >= CYCLES` alone called
    that complete, read DEV-6 on it and labelled `cycle3.pt` final."""
    seq = [0.50, 0.50, 0.51, 0.51, 0.49, 0.49]
    calls = []

    def ev(_m, step, kind):
        calls.append(kind)
        return seq[len(calls) - 1]

    # the real loop, so the real kill rule decides: total 30 -> ends [9, 19, 29], mids {5, 15, 25}
    r = T.train_arm(Toy(), make_batch_fn(), total_steps=30, seed=0, eval_fn=ev)
    assert calls == ["mid", "end"] * 3
    assert r["stopped"].startswith("kill:"), r["stopped"]
    assert len(r["cycle_end_evals"]) == 3, "the kill fired AT the final cycle end"

    monkeypatch.setattr(R.Tr, "train_arm", lambda *a, **k: dict(r))
    mock_pipeline(monkeypatch)
    monkeypatch.setattr(R.Tr, "train_arm", lambda *a, **k: dict(r))
    write_f_verdict()
    rec = R.run("A1", device="cuda", verbose=False)
    assert rec["status"] == "failed" and rec["stopped_is_completion"] is False
    assert rec["dev6"] is None, "a killed arm gets no DEV-6 read"
    assert rec["final_checkpoint"] is None and rec["final_checkpoint_sha256"] is None


@pytest.mark.parametrize("stopped,n_ends,status", [
    (None, 3, "complete"),
    ("plateau at cycle 3", 3, "complete"),
    ("plateau at cycle 2", 3, "failed"),
    ("kill: two consecutive evaluations more than 0.0056 below", 3, "failed"),
    ("non-finite loss at step 12345", 3, "failed"),
    ("non-finite grad norm at step 9", 3, "failed"),
    ("non-finite evaluation at index 5", 3, "failed"),
    (None, 2, "failed"),
])
def test_success_is_stopped_None_or_exactly_the_final_cycle_plateau(stopped, n_ends, status):
    got, ok, _pl = R.classify(stopped, n_ends)
    assert got == status and ok is (status == "complete")


def test_a_failed_arm_never_labels_a_checkpoint_final(monkeypatch, sandbox):
    """`cycle3.pt` is on disk after a kill at the final cycle end; it is listed under its cycle
    key and is NOT the arm's final checkpoint."""
    mock_pipeline(monkeypatch, stopped="kill: at the last cycle end")
    write_f_verdict()
    (R.WORK / "A1").mkdir(parents=True, exist_ok=True)
    (R.WORK / "A1" / "cycle3.pt").write_bytes(b"partial")
    rec = R.run("A1", device="cuda", verbose=False)
    assert rec["status"] == "failed"
    assert rec["final_checkpoint"] is None and rec["final_checkpoint_sha256"] is None
    assert rec["checkpoints"]["cycle3"]["sha256"], "it is still reported, under its cycle key"


# ------------------------------------- a crash is an outcome, recorded (finding 9) -------------

def test_a_crash_writes_a_terminal_FAILED_record_and_re_raises(monkeypatch, sandbox):
    mock_pipeline(monkeypatch)
    write_f_verdict()

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    monkeypatch.setattr(R.Tr, "train_arm", boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        R.run("A1", device="cuda", verbose=False)
    for path in (R.WORK / "A1" / "record.json", R.RESULTS / "m10_arm_A1.json"):
        rec = json.loads(path.read_text())
        assert rec["status"] == "failed" and rec["complete"] is False and rec["terminal"] is True
        assert rec["failure"]["type"] == "RuntimeError"
        assert "out of memory" in rec["failure"]["message"]
        assert rec["final_checkpoint"] is None and rec["dev6"] is None
        assert rec["recipe"]["student"] and rec["registry_sha256"]
    # and it is not silently re-run
    with pytest.raises(SystemExit, match="already exists"):
        R.run("A1", device="cuda", verbose=False)


def test_a_REFUSAL_writes_no_record(sandbox):
    """A refusal means the arm never started, and a record would block the run after the fix."""
    with pytest.raises(SystemExit):
        R.run("A1")                                     # no F verdict
    assert not (R.WORK / "A1" / "record.json").exists()
    assert not (R.RESULTS / "m10_arm_A1.json").exists()


# --------------------------- the evaluator's records survive a resume (finding 7) --------------

def test_the_cov_evaluator_checkpoints_and_restores_its_own_records(sandbox):
    """finding 7: `CovEval.records` — per-family macros and the per-query score paths the contrast
    step bootstraps over — lived only in the runner's memory, so a resumed arm reported the
    cycles of THIS process only."""
    out = sandbox / "arm"
    out.mkdir()
    (out / "cov_cycle1.json").write_text('{"stub": 1}')
    (out / "cov_cycle2.json").write_text('{"stub": 2}')
    ev = R.CovEval(model=None, out_dir=out, verbose=False)
    ev.records = [{"label": "cycle1", "step": 9, "macro": 0.41,
                   "by_family": {"legal": 0.31, "finance": 0.5},
                   "per_query_scores": str(out / "cov_cycle1.json"),
                   "per_query_scores_sha256": R.sha256_file(out / "cov_cycle1.json")},
                  {"label": "cycle2", "step": 19, "macro": 0.42,
                   "by_family": {"legal": 0.33, "finance": 0.5},
                   "per_query_scores": str(out / "cov_cycle2.json"),
                   "per_query_scores_sha256": R.sha256_file(out / "cov_cycle2.json")}]

    fresh = R.CovEval(model=None, out_dir=out, verbose=False)
    assert fresh.records == []
    fresh.restore(ev.state())
    assert [r["label"] for r in fresh.records] == ["cycle1", "cycle2"]
    assert fresh.records[0]["by_family"]["legal"] == 0.31
    assert fresh.records[1]["per_query_scores"].endswith("cov_cycle2.json")
    assert R.StubEval().state() == {"records": []}


def test_a_call_records_its_own_evidence_files_hash(monkeypatch, sandbox):
    """item C: `CovEval.__call__` writes the per-query score file BEFORE it appends the record, so
    the hash it records is the hash of the file actually on disk."""
    out = sandbox / "arm"
    out.mkdir()

    class M:
        training = True

        def encode_queries(self, texts, batch_size=256):
            return texts

        def train(self):
            pass

    import cov_eval10
    monkeypatch.setattr(cov_eval10, "score_student", lambda *a, **k: {"u": {"q1": 1.0}})
    monkeypatch.setattr(cov_eval10, "macro", lambda per, units=None: (0.5, {"legal": 0.5}, {}))
    ev = R.CovEval(M(), out, verbose=False)
    ev.units = lambda: []
    ev(9, "cycle1")
    p = out / "cov_cycle1.json"
    assert p.exists()
    assert ev.records[0]["per_query_scores_sha256"] == R.sha256_file(p)


def test_resume_refuses_when_an_evidence_file_is_missing_or_altered(sandbox):
    """item C: delete or alter a `cov_cycle1.json` after its checkpoint and resume must refuse --
    the contrast step's per-query scores would no longer be what the checkpoint says they are."""
    out = sandbox / "arm"
    out.mkdir()
    p = out / "cov_cycle1.json"
    p.write_text('{"stub": 1}')
    state = {"records": [{"label": "cycle1", "step": 9, "macro": 0.41,
                          "per_query_scores": str(p), "per_query_scores_sha256": R.sha256_file(p)}]}

    altered = R.CovEval(model=None, out_dir=out, verbose=False)
    p.write_text('{"stub": "TAMPERED"}')
    with pytest.raises(SystemExit, match="does not match|hashes to"):
        altered.restore(state)

    p.unlink()
    deleted = R.CovEval(model=None, out_dir=out, verbose=False)
    with pytest.raises(SystemExit, match="missing"):
        deleted.restore(state)


def test_the_runner_hands_the_evaluator_and_a_fingerprint_to_the_trainer(monkeypatch, sandbox):
    seen = {}
    mock_pipeline(monkeypatch)
    write_f_verdict()
    real = R.Tr.train_arm

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(R.Tr, "train_arm", spy)
    rec = R.run("A1", device="cpu", smoke_steps=4, verbose=False)
    assert isinstance(seen["eval_state"], R.StubEval)
    assert seen["fingerprint"] == rec["recipe_fingerprint"] and len(seen["fingerprint"]) == 64


def test_the_fingerprint_moves_with_the_recipe(sandbox):
    reg = R.SL.cfg()
    p = R.arm_plan("A1", reg)
    man = {"arm": "A1", "sources": ["harvest"]}
    base = R.fingerprint("A1", p, man, 0, None, "cuda")
    assert base == R.fingerprint("A1", p, man, 0, None, "cuda")
    assert base != R.fingerprint("A2", p, man, 0, None, "cuda")
    assert base != R.fingerprint("A1", p, man, 1, None, "cuda")
    assert base != R.fingerprint("A1", p, {"arm": "A1", "sources": ["harvest", "gen"]}, 0, None,
                                 "cuda")
    other = dict(p, student="MiniLM-L6")
    assert base != R.fingerprint("A1", other, man, 0, None, "cuda")


def test_the_fingerprint_carries_device_autocast_warmup_and_optimizer_settings(sandbox):
    """item B: device, autocast dtype, `nano10.WARMUP_STEPS` and the optimizer settings are all
    part of the fingerprint, so a checkpoint from one is refused a resume under another."""
    reg = R.SL.cfg()
    p = R.arm_plan("A1", reg)
    man = {"arm": "A1", "sources": ["harvest"]}
    cuda_fp = R.fingerprint("A1", p, man, 0, None, "cuda")
    cpu_fp = R.fingerprint("A1", p, man, 0, None, "cpu")
    assert cuda_fp != cpu_fp, "a cuda/bf16 checkpoint must not resume under cpu/fp32"
    # deterministic and stable across two identical calls
    assert cuda_fp == R.fingerprint("A1", p, man, 0, None, "cuda")


def test_the_fingerprint_moves_with_the_code(monkeypatch, sandbox, tmp_path):
    """item B: a sha256 of `run_arm.py` + `trainer10.py` + `nano10.py` is part of the fingerprint
    (code identity), so an edit to any of the three refuses a resume."""
    reg = R.SL.cfg()
    p = R.arm_plan("A1", reg)
    man = {"arm": "A1", "sources": ["harvest"]}
    base = R.fingerprint("A1", p, man, 0, None, "cuda")
    real = R.code_identity()

    def changed():
        return hashlib.sha256(b"not the same code at all").hexdigest()
    monkeypatch.setattr(R, "code_identity", changed)
    assert R.fingerprint("A1", p, man, 0, None, "cuda") != base
    assert changed() != real


def test_the_seed_is_a_registered_property_of_the_arm_not_a_launch_argument():
    """`seed_rule` fixes seed 0 for every screen arm; `ANCHOR-seed1` carries its own seed 1 (the
    descriptive seed-sensitivity run, astra's decision 2). Reading it per-arm means a seed can
    never be supplied at a launch -- it is registered, like the dose."""
    reg = R.cfg()
    assert reg["arms"]["ANCHOR-seed1"]["seed"] == 1
    assert reg["arms"]["ANCHOR"].get("seed") is None, \
        "the anchor takes the global seed_rule; only the descriptive arm overrides"
    assert "--seed" not in R.build_argparser().format_usage()


def test_A3_20M_and_F_bge_small_differ_ONLY_in_the_corpus():
    """The parity astra's decision 1 was conditional on. `n_docs` is dose-derived
    (`corpus_loader.arm_doc_count`), so matching the dose matches the document side too; after
    that the only difference left is `sources`, which is the contrast."""
    reg = R.cfg()
    v = R.f_verdict(reg)
    a = R.arm_plan("A3-20M", reg, verdict=v)
    b = R.arm_plan("F-bge-small", reg, verdict=v)
    for k in ("dose_examples", "total_steps", "cycle_end_steps", "n_docs", "batch", "pattern",
              "student", "n_layers", "head", "objective", "warm_start", "cut_corpus"):
        assert a[k] == b[k], f"{k}: {a[k]!r} != {b[k]!r} -- parity is the ruling's precondition"
    assert a["sources"] != b["sources"]
    assert set(b["sources"]) - set(a["sources"]) == {"generated"}
