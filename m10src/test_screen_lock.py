"""The lock's validator must FAIL on the mutations it exists to catch."""
import copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_lock as L


def _mut(fn):
    r = copy.deepcopy(L.cfg()); fn(r); return L.validate(r)


def test_clean_registry_validates():
    assert L.validate() == [], L.validate()


def test_a_fourteenth_contrast_is_refused():
    """The Bonferroni denominator is 13. A contrast added without moving it changes every bound."""
    bad = _mut(lambda r: r["contrasts"].__setitem__("X1", {"a": "D-NORM", "b": "anchor"}))
    assert any("Bonferroni" in b for b in bad), bad


def test_a_contrast_naming_a_nonexistent_arm_is_refused():
    bad = _mut(lambda r: r["contrasts"]["D1"].__setitem__("a", "D-SOMETHING"))
    assert any("neither an arm nor an anchor alias" in b for b in bad), bad


def test_an_arm_no_contrast_reads_is_refused():
    def f(r):
        r["arms"]["Z-orphan"] = {"family": "D", "dose_examples": 5000000, "trained": True}
        r["trained_arms_expected"] += 1
    bad = _mut(f)
    assert any("no contrast reads it" in b for b in bad), bad


def test_a_family_without_an_outcome_map_is_refused():
    bad = _mut(lambda r: r["outcome_to_action"].pop("E"))
    assert any("no outcome->action entry" in b for b in bad), bad


def test_a_quantile_that_is_not_alpha_over_the_contrast_count_is_refused():
    bad = _mut(lambda r: r["statistics"]["bootstrap"].__setitem__("quantile", 0.025 / 12))
    assert any("alpha_family / n_contrasts" in b for b in bad), bad


def test_numpy_default_quantile_method_is_refused():
    bad = _mut(lambda r: r["statistics"]["bootstrap"].__setitem__("quantile_method", "linear"))
    assert any("inverted_cdf" in b for b in bad), bad


def test_a_registry_resolution_number_that_drifts_from_its_artifact_is_refused():
    bad = _mut(lambda r: r["statistics"].__setitem__("measured_resolution_distance", 0.004))
    assert any("resolution" in b for b in bad), bad


def test_families_must_match_the_macro_implementation():
    bad = _mut(lambda r: r["statistics"].__setitem__("families", ["BRIGHT", "legal", "finance"]))
    assert any("cov_macro.FAMILIES" in b for b in bad), bad




# ---- S10 (Fable pass, 2026-09-05): each of these mutations PASSED the first validator ----

def test_an_alias_pointing_at_a_nonexistent_arm_is_refused():
    bad = _mut(lambda r: r["anchor_aliases"].__setitem__("E-bs32", "ANCHOR-TYPO"))
    assert any("is not an arm" in b for b in bad), bad


def test_a_loosened_alpha_with_a_coherent_quantile_is_refused():
    """0.05/13 is internally consistent; only checking the RATIO would let it through."""
    def f(r):
        r["statistics"]["bootstrap"]["alpha_family"] = 0.05
        r["statistics"]["bootstrap"]["quantile"] = 0.05 / 13
    bad = _mut(f)
    assert any("alpha_family is 0.05" in b for b in bad), bad


def test_bootstrap_constants_drifting_from_the_artifact_are_refused():
    for k, v in (("B", 20000), ("seed", 1), ("chunk", 1000)):
        bad = _mut(lambda r, k=k, v=v: r["statistics"]["bootstrap"].__setitem__(k, v))
        assert any(f"bootstrap.{k}" in b for b in bad), (k, bad)


def test_a_contrast_read_beyond_its_arms_dose_is_refused():
    bad = _mut(lambda r: r["contrasts"]["D1"].__setitem__("at_examples", 20000000))
    assert any("beyond its dose" in b for b in bad), bad


def test_an_anchor_field_contradicting_the_axis_it_stands_in_for_is_refused():
    bad = _mut(lambda r: r["anchor"].__setitem__("batch", 128))
    assert any("alias E-bs32 claims" in b for b in bad), bad


def test_an_arm_moved_into_the_untrained_bucket_is_refused():
    """`trained` is a boolean, so prose cannot demote an arm out of the count."""
    bad = _mut(lambda r: r["arms"]["D-COV"].pop("trained"))
    assert any("no boolean `trained`" in b for b in bad), bad
    bad = _mut(lambda r: r["arms"]["D-COV"].__setitem__("trained", False))
    assert any("trained arms" in b for b in bad), bad


def test_an_emptied_outcome_entry_is_refused():
    bad = _mut(lambda r: r["outcome_to_action"].__setitem__("D", "  "))
    assert any("no outcome->action entry" in b for b in bad), bad


def test_a_contrast_naming_an_unregistered_rule_is_refused():
    bad = _mut(lambda r: r["contrasts"]["A3-A2"].__setitem__("rule", "whatever_wins"))
    assert any("not in `rules`" in b for b in bad), bad


def test_the_trained_count_and_the_order_are_actually_checked():
    def f(r):
        r["arms"]["Z"] = {"family": "D", "dose_examples": 5000000, "trained": True}
    bad = _mut(f)
    assert any("trained arms" in b for b in bad) and any("no contrast reads it" in b for b in bad), bad
    bad = _mut(lambda r: r["order"].remove("E"))
    assert any("does not cover the decision families" in b for b in bad), bad


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
