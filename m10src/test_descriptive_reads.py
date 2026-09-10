"""Examples for the two descriptive reads. They select nothing, which is exactly why the thing
printed beside their numbers has to be pinned: a number with no label becomes a verdict.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import descriptive_reads as DRD


def test_it_reads_only_registered_descriptive_arms():
    """A screen arm must not be readable through this module -- that would produce a
    verdict-shaped artifact outside the registered contrast machinery."""
    reg = DRD.cfg()
    for name, (arm, comparator, _at) in DRD.READS.items():
        assert reg["arms"][arm].get("descriptive") is True, f"{name}: {arm} must be descriptive"
        assert arm in reg["descriptive_runs"]["members"]
    bad = {**reg, "arms": {**reg["arms"],
                           "ANCHOR-seed1": {**reg["arms"]["ANCHOR-seed1"], "descriptive": False}}}
    with pytest.raises(ValueError, match="not registered"):
        DRD.read("seed_sensitivity", bad, verbose=False)


def test_the_comparators_are_the_ones_the_rulings_named():
    """Decision 1 compares A3's corpus against A4's AT 20M, using the arm that already exists;
    decision 2 compares the two anchor seeds at the screen dose."""
    assert DRD.READS["corpus_at_20M"] == ("A3-20M", "F-bge-small", 20_000_000)
    assert DRD.READS["seed_sensitivity"] == ("ANCHOR-seed1", "ANCHOR", None)


def test_no_interval_is_computed_anywhere_in_this_module():
    """Deliberate: astra's ruling says compare the checkpoints DESCRIPTIVELY, and an interval
    beside a point estimate reads as a test however it is captioned."""
    src = (Path(__file__).resolve().parent / "descriptive_reads.py").read_text()
    for banned in ("cov_macro.contrast(", "lower_bound", "quantile", "ONE_SIDED"):
        assert banned not in src, f"{banned} must not appear in a descriptive read"


def test_a_completed_read_carries_its_prohibitions(tmp_path, monkeypatch):
    """The seed read must ship the registered sentence AND the do-not-claim list with the number,
    from the registry rather than from this module."""
    reg = DRD.cfg()
    e = reg["arms"]["ANCHOR-seed1"]
    assert "SENSITIVITY" in e["_what_it_licenses"]
    for phrase in ("reproducibility", "bounded seed noise", "EVEN IF"):
        assert phrase in e["_what_it_does_NOT_license"]
    assert "NOTHING" in e["selects"] and "retained regardless of outcome" in e["selects"]


def test_main_skips_an_arm_that_has_not_finished(capsys, tmp_path, monkeypatch):
    """An unfinished arm produces no read, regardless of machine-local experiment state."""
    monkeypatch.setattr(DRD, "RESULTS", tmp_path)
    def unexpected_read(*args, **kwargs):
        pytest.fail("an unfinished arm must not be read")
    monkeypatch.setattr(DRD, "read", unexpected_read)
    (tmp_path / "m10_arm_ANCHOR-seed1.json").write_text('{"status": "running"}')
    rc = DRD.main(["seed_sensitivity"])
    out = capsys.readouterr().out
    assert rc == 1 and "skipped" in out
    assert not (tmp_path / "m10_descriptive_seed_sensitivity.json").exists()


def test_a_read_point_resolves_under_EITHER_label_convention(monkeypatch):
    """The bug this caught in flight: an arm with registered `read_at` points carries them in the
    label (`cycle3_read20000000`), while an arm whose DOSE is that number just has `cycle3`. Both
    are the same read, and requiring the tag made the 20M comparison fail on the arm trained
    specifically for it."""
    reg = DRD.cfg()
    labels = {"A3-20M": "cycle3", "F-bge-small": "cycle3_read20000000"}
    monkeypatch.setattr(DRD, "cov_files", lambda arm: {labels[arm]: Path("unused")})
    calls = []
    def read_point(arm, at=None):
        calls.append((arm, at))
        return labels[arm], {"unit": {"q0": 0.5}}
    monkeypatch.setattr(DRD, "read_point", read_point)
    la, sa = DRD.read_at("A3-20M", 20_000_000, reg)
    lb, sb = DRD.read_at("F-bge-small", 20_000_000, reg)
    assert la == "cycle3" and lb == "cycle3_read20000000"
    assert set(sa) == set(sb)
    assert calls == [("A3-20M", None), ("F-bge-small", 20_000_000)]


def test_it_refuses_a_read_at_a_count_the_arm_never_reached(monkeypatch):
    """The fallback accepts the final cycle end ONLY when the arm's registered dose IS the
    requested count -- never on the assumption that the last checkpoint is close enough."""
    reg = DRD.cfg()
    monkeypatch.setattr(DRD, "cov_files", lambda arm: {"cycle3": Path("unused")})
    with pytest.raises(ValueError, match="no read at|no cycle end tagged"):
        DRD.read_at("ANCHOR-seed1", 20_000_000, reg)      # a 5M arm
