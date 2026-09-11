"""What the deferred-DEV-6 filler must refuse, and that it changes exactly one field. CPU only,
the student and `run_arm.dev6` mocked: the contract under test is the record's, not the BERT's.
Every path is redirected into a tmp tree, so no real arm record can be touched (§Hazards)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "m7src", REPO / "m10src", REPO / "m13src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dev6_from_checkpoint as D                                          # noqa: E402
import run_arm as R                                                       # noqa: E402

ARM = "E-bs32"


class Toy(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.head = torch.nn.Linear(4, 4)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def under_cap(self):
        return True


DEV6_ROW = {"teacher": "stella-400M-v5", "components": ["x"], "per_component": {"x": 0.5},
            "macro": 0.55, "ceiling_DEV6": 0.6, "retention": 0.9167,
            "_scope": "DEV-6 only; no six-set, reserved or LoTTE surface was read"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """-> (record, checkpoint path). Both published copies written by `run_arm.write_record`."""
    monkeypatch.setattr(R, "WORK", tmp_path / "m10arms")
    monkeypatch.setattr(R, "SMOKE_WORK", tmp_path / "m10arms" / "smoke")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(D.N, "Nano10", Toy)
    monkeypatch.setattr(D.R, "dev6", lambda m, verbose=True: dict(DEV6_ROW))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out_dir, rec_path, results_path = R.record_paths(ARM, False)
    out_dir.mkdir(parents=True)
    ck = out_dir / "cycle3.pt"
    torch.save({"model": Toy().state_dict(), "step": 5}, ck)
    sha = R.sha256_file(ck)
    rec = {"_what": "fixture arm record", "arm": ARM, "family": "E", "status": "complete",
           "complete": True, "terminal": True, "smoke": False, "device": "cuda", "seed": 0,
           "recipe": {"student": "bge-small", "n_layers": 3, "head": "linear",
                      "dose_examples": 5_000_000, "batch": 32},
           "params": Toy().n_params(),
           "cov": {"final_macro": 0.42}, "dev6": R.dev6_deferred({"sha256": sha}, ARM),
           "checkpoints": {"cycle3": {"path": R.rel(ck), "sha256": sha}},
           "final_checkpoint": R.rel(ck), "final_checkpoint_sha256": sha,
           "registry_sha256": "0" * 64, "git_head": "f" * 40}
    R.write_record(rec, rec_path, results_path)
    return rec, ck


def _rewrite(**over):
    _o, rec_path, results_path = R.record_paths(ARM, False)
    rec = json.loads(rec_path.read_text())
    rec.update(over)
    R.write_record(rec, rec_path, results_path)
    return rec


def test_it_fills_dev6_and_changes_nothing_else(world):
    before, ck = world
    assert before["dev6"]["deferred"] is True and before["dev6"]["macro"] is None
    rec = D.run(ARM, device="cpu", verbose=False)
    assert rec["dev6"]["macro"] == 0.55 and rec["dev6"]["per_component"] == {"x": 0.5}
    prov = rec["dev6"]["filled_from_checkpoint"]
    assert prov["checkpoint_sha256"] == before["final_checkpoint_sha256"]
    assert prov["deferred_by_runner"] == before["dev6"]
    assert prov["code_sha256"] == D.code_identity() and prov["device"] == "cpu"
    assert {k: v for k, v in rec.items() if k != "dev6"} == \
        {k: v for k, v in before.items() if k != "dev6"}
    _o, rec_path, results_path = R.record_paths(ARM, False)
    on_disk = json.loads(rec_path.read_text())
    assert on_disk == json.loads(results_path.read_text())
    assert on_disk["dev6"]["macro"] == 0.55 and on_disk["final_checkpoint_sha256"] == before["final_checkpoint_sha256"]


def test_a_record_that_already_carries_dev6_refuses_a_second_read(world):
    D.run(ARM, device="cpu", verbose=False)
    with pytest.raises(SystemExit, match="read ONCE"):
        D.run(ARM, device="cpu", verbose=False)
    _rewrite(dev6=dict(DEV6_ROW))
    with pytest.raises(SystemExit, match="read ONCE"):
        D.run(ARM, device="cpu", verbose=False)


@pytest.mark.parametrize("over, match", [
    ({"status": "failed", "complete": False}, "failed arm has no final checkpoint"),
    ({"smoke": True}, "SMOKE record"),
    ({"arm": "E-bs128"}, "not 'E-bs32'"),
    ({"final_checkpoint_sha256": "1" * 64}, "do not name one final checkpoint"),
    ({"dev6": {"deferred": True, "checkpoint_sha256": "2" * 64}}, "deferral names checkpoint"),
    ({"recipe": {"student": "bge-small"}}, "recipe lacks"),
    ({"params": 12345}, "does not describe it"),
])
def test_a_record_that_is_not_a_complete_deferred_arm_refuses(world, over, match):
    _rewrite(**over)
    with pytest.raises(SystemExit, match=match):
        D.run(ARM, device="cpu", verbose=False)


def test_a_checkpoint_whose_bytes_moved_refuses(world):
    _rec, ck = world
    ck.write_bytes(b"not the published checkpoint")
    with pytest.raises(SystemExit, match="not the published final checkpoint"):
        D.run(ARM, device="cpu", verbose=False)


def test_a_missing_checkpoint_refuses(world):
    _rec, ck = world
    ck.unlink()
    with pytest.raises(SystemExit, match="not on this machine"):
        D.run(ARM, device="cpu", verbose=False)


def test_the_two_published_copies_must_agree(world):
    _o, rec_path, _results_path = R.record_paths(ARM, False)
    rec = json.loads(rec_path.read_text())
    rec["cov"] = {"final_macro": 0.99}
    rec_path.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="differ"):
        D.run(ARM, device="cpu", verbose=False)


def test_a_missing_copy_refuses(world):
    _o, _rec_path, results_path = R.record_paths(ARM, False)
    results_path.unlink()
    with pytest.raises(SystemExit, match="both published paths"):
        D.run(ARM, device="cpu", verbose=False)


def test_cuda_is_the_default_and_is_verified(world):
    assert D.build_argparser().parse_args([ARM]).device == "cuda"
    with pytest.raises(SystemExit, match="not a GPU"):
        D.run(ARM, device="cuda", verbose=False)
