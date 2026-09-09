"""Examples for the contrast step, because a rule nobody ran examples against is not a rule.

The three things that can go wrong here and cannot be seen in the output: reading the wrong
checkpoint, passing the wrong quantile, and collapsing "the arm never ran" into "the contrast did
not resolve". Each has its own test. `test_F1_reproduces_the_registered_artifact` is the end-to-end
guard: the committed F1 record was computed before this module existed, and the driver reproduces
its draws digest exactly.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contrasts as CT
import cov_macro


def _scores(offset=0.0):
    """One score per unit per query on the admitted surface, deterministic and paired."""
    return {u: {f"q{i}": 0.5 + offset + 0.001 * ((i * 7) % 11)
                for i in range(12)} for u in cov_macro.SURFACE}


def _arm(tmp, name, labels):
    d = tmp / name
    d.mkdir(parents=True)
    for label, offset in labels.items():
        s = _scores(offset)
        (d / f"cov_{label}.json").write_text(json.dumps(
            {"label": label, "step": 1, "macro": 0.0, "by_family": {}, "by_unit": {},
             "per_unit_query": s}))
    return d


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(CT, "WORK", tmp_path)
    return tmp_path


# ------------------------------------------------------------------ 1. reading the right file ---

def test_cycle_ends_are_ordered_numerically_and_mid_reads_excluded(work):
    """`cov_mid*` files exist for the plateau rule and are NOT cycle ends; and lexicographic order
    would put cycle10 before cycle2, which is how a sign-stability check reads the wrong pair."""
    _arm(work, "X", {"cycle1": 0.0, "cycle2": 0.01, "cycle10": 0.02, "mid999": 0.9})
    assert list(CT.cov_files("X")) == ["cycle1", "cycle2", "cycle10"]
    assert CT.read_point("X")[0] == "cycle10"
    assert CT.previous_cycle_end("X", "cycle10")[0] == "cycle2"


def test_a_read_point_is_selected_by_its_tag_not_by_position(work):
    _arm(work, "F", {"cycle1": 0.0, "cycle2": 0.01, "cycle3_read20000000": 0.02})
    assert CT.read_point("F", 20_000_000)[0] == "cycle3_read20000000"
    with pytest.raises(KeyError):
        CT.read_point("F", 10_000_000)          # a read that is not a cycle end


def test_first_cycle_has_no_previous_cycle_end(work):
    _arm(work, "X", {"cycle1": 0.0})
    assert CT.previous_cycle_end("X", "cycle1") == (None, None)


# ------------------------------------------------------------------------- 2. the quantile ------

def test_quantile_is_checked_against_cov_macro_per_contrast():
    """The `_runner_contract` hazard: 11 contrasts carry no quantile of their own, so a stale
    registry default decides them all -- and it used to be the superseded alpha/13."""
    boot = {"quantile": cov_macro.ONE_SIDED}
    assert CT.quantile_for("G2", {}, boot) == (cov_macro.ONE_SIDED, 1)
    assert CT.quantile_for("F1", {"tails": 2, "quantile": cov_macro.F_PER_TAIL}, boot) == \
        (cov_macro.F_PER_TAIL, 2)
    with pytest.raises(ValueError):
        CT.quantile_for("G2", {}, {"quantile": cov_macro.HISTORICAL_13})
    with pytest.raises(ValueError):
        CT.quantile_for("F1", {"tails": 2, "quantile": cov_macro.ONE_SIDED}, boot)


def test_the_live_registry_passes_the_quantile_check_for_every_contrast():
    reg = CT.cfg()
    boot = reg["statistics"]["bootstrap"]
    total = sum(t * q for q, t in
                (CT.quantile_for(cid, c, boot) for cid, c in reg["contrasts"].items()))
    assert total == pytest.approx(boot["alpha_family"], abs=1e-15)


# --------------------------------------------------------------- 3. the labels and the rules ----

def test_family_A_uses_the_corrected_bar_and_reports_three_labels(work, monkeypatch):
    """`three_outcome`/`generated_half_in_build` resolve on `lower > MDE`, not `lower > 0`, and are
    exempt from sign stability. A3-A2 landed at lower 0.005363 against MDE 0.0056 -- 0.00024 short
    -- so the boundary this test pins is the one the screen actually sat on."""
    _arm(work, "A3", {"cycle1": 0.0, "cycle2": 0.0, "cycle3": 0.02})
    _arm(work, "A2", {"cycle1": 0.0, "cycle2": 0.0, "cycle3": 0.0})
    reg = {"contrasts": {"A3-A2": {"a": "A3", "b": "A2", "rule": "three_outcome"}},
           "anchor_aliases": {}, "arms": {"A3": {}, "A2": {}},
           "statistics": {"MDE": 0.0056, "measured_resolution_distance": 0.008619,
                          "bootstrap": {"quantile": cov_macro.ONE_SIDED, "B": 2000, "seed": 0,
                                        "quantile_method": "inverted_cdf", "chunk": 500,
                                        "n_contrasts": 12}}}
    monkeypatch.setattr(CT, "RESULTS", work)
    out = CT.compute("A3-A2", reg, verbose=False)
    assert out["resolve"]["sign_stability_exempt"] is True
    assert out["resolve"]["label"] == "RESOLVED" and out["resolve"]["lower_bound_gt_MDE"]
    # the same arms with no effect at all fall to the third label, not to "unresolved"
    _arm(work, "A2b", {"cycle1": 0.0, "cycle2": 0.0, "cycle3": 0.02})
    reg["contrasts"]["A3-A2"]["b"] = "A2b"
    reg["arms"]["A2b"] = {}
    assert CT.compute("A3-A2", reg, verbose=False)["resolve"]["label"] == "NOT POSITIVE"


def test_a_cut_or_unrun_arm_yields_not_computed_never_unresolved(work):
    """`rules.C_skipped` and `outcome_to_action.E` are different dispositions from
    `rules.arm_failure`, and a screen that reported them as unresolved would silently take a
    default decision on a measurement nobody made."""
    reg = {"contrasts": {"C1": {"a": "C-M9init", "b": "C-fresh"},
                         "E1": {"a": "E-bs32", "b": "E-bs128"}},
           "anchor_aliases": {"C-fresh": "ANCHOR", "E-bs32": "ANCHOR"},
           "arms": {"C-M9init": {"cut": "CUT under W8 band 1"}, "ANCHOR": {}, "E-bs128": {}},
           "statistics": {"MDE": 0.0056, "bootstrap": {"quantile": cov_macro.ONE_SIDED,
                                                       "n_contrasts": 12}}}
    _arm(work, "ANCHOR", {"cycle1": 0.0})
    assert "CUT" in CT.compute("C1", reg, verbose=False)["not_computed"]
    assert "has not run" in CT.compute("E1", reg, verbose=False)["not_computed"]


def test_selection_carries_E_as_PENDING_when_its_contrast_was_never_read():
    """The live selection: an unread E1 must not be read through `rules.E_cost`'s "in every other
    case, bs128" branch, which would take E's decision from a measurement that was never made."""
    out = json.loads((CT.RESULTS / "m10_screen_verdicts.json").read_text())
    assert out["selected"]["batch"] == "PENDING"
    assert "E1" in out["not_computed"] and "C1" in out["not_computed"]


# --------------------------------------------------------------------- 4. the end-to-end guard --

def test_F1_reproduces_the_registered_artifact():
    """The F1 record was computed before this module existed. Same arms, same plan, same digest."""
    f1 = json.loads((CT.RESULTS / "m10_contrast_F1.json").read_text())
    assert f1["draws_sha256"] == \
        "cb392db4f02aa8774afe092c20da7f11616e6fd28da189424ff60797dd5c00bd"
    assert f1["delta_raw"] == pytest.approx(0.011595234778582406, abs=1e-15)
    assert f1["quantile"] == cov_macro.F_PER_TAIL and f1["resolve"]["resolved"] is True
    v = json.loads((CT.RESULTS / "m10_F_verdict.json").read_text())
    assert v["winner"] == "bge-small"
    assert v["contrast"]["draws_sha256"] == f1["draws_sha256"]
    assert v["registry_sha256"] == CT.sha256_file(CT.REGISTRY), \
        "the verdict must name the CURRENT registry, or run_arm refuses every post-F arm"
