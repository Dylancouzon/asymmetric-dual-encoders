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


def test_main_skips_an_arm_that_has_not_finished(capsys):
    """The chain is hours long; running the read early must say so, not half-report."""
    rc = DRD.main(["seed_sensitivity"])
    out = capsys.readouterr().out
    done = (Path(DRD.RESULTS) / "m10_arm_ANCHOR-seed1.json").exists() and json.loads(
        (Path(DRD.RESULTS) / "m10_arm_ANCHOR-seed1.json").read_text()).get("status") == "complete"
    if done:
        assert rc == 0 and "delta" in out
    else:
        assert rc == 1 and "skipped" in out
