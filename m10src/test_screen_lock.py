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
        r["arms"]["Z-orphan"] = {"family": "D", "dose_examples": 5000000}
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


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
