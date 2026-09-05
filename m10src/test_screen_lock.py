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
    bad = _mut(lambda r: r["statistics"]["bootstrap"].__setitem__("quantile", 0.025 / 13))
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


# ---- W9: the eleven registry defects, ruled "fix and have it reviewed" by Dylan 2026-09-05 ----

def test_E1_is_oriented_so_a_bs32_win_can_actually_resolve():
    """Codex finding 1. Read `bs128 - bs32`, a bs32 quality win is a NEGATIVE point estimate and
    `point >= threshold` refuses it -- making the mandate's own action, 'a RESOLVED bs32 win is
    adopted', unreachable. The contrast must be oriented so the adoptable outcome is positive."""
    r = L.cfg()
    e1 = r["contrasts"]["E1"]
    assert (e1["a"], e1["b"]) == ("E-bs32", "E-bs128"), \
        "a bs32 win must be a POSITIVE point estimate or it can never resolve"
    assert r["anchor_aliases"]["E-bs32"] == "ANCHOR"
    assert e1["rule"] == "E_cost" and "E_cost" in r["rules"]


def test_the_W9_fixes_are_STRUCTURALLY_present_not_just_described():
    """Four W9 fixes, checked by STRUCTURE rather than by matching a sentence. The earlier version
    of these asserted that particular words appeared in a docstring field -- which breaks on a
    wording edit and passes on a wrong rule, i.e. exactly backwards (whole-plan review 2026-09-05).
    G1's orientation, the alpha and the C-skip counts are checked numerically elsewhere."""
    r = L.cfg()
    # G1 keeps the mandate's orientation; only its ACTION text changed
    assert (r["contrasts"]["G1"]["a"], r["contrasts"]["G1"]["b"]) == ("G-1152", "G-384")
    # F2 and L12 are CUT (Dylan 2026-09-05); the retired rule stays as a record only
    assert "F2" not in r["contrasts"] and "F2" in r["_cut_contrasts"]
    a = r["arms"]["F-MiniLM-L12"]
    assert a["trained"] is False and a.get("cut") and "conditional_extension" not in a
    # the multi-arm families all name the tie rule, and it exists
    assert "multi_arm_winner" in r["rules"]
    for cid in ("G1", "G2", "G3", "B1", "B2", "D1", "D2"):
        assert r["contrasts"][cid]["family_rule"] == "multi_arm_winner", cid
    # the confirmation cap's consequence is a registry field, not prose elsewhere
    assert r["confirmation"].get("unconfirmed_non_defaults")


def test_the_familywise_alpha_is_exactly_the_registered_0_025():
    """Codex finding 3. F1 and F2 are two-sided (F is oriented after the readings), so at alpha/13
    per tail the union bound was 15 tails = 0.02885, not 0.025."""
    r = L.cfg()
    b = r["statistics"]["bootstrap"]
    q, alpha = b["quantile"], b["alpha_family"]
    total = 0.0
    for c in r["contrasts"].values():
        tails = c.get("tails", 1)
        total += tails * c.get("quantile", q)
    assert abs(total - alpha) < 1e-12, f"familywise {total} != {alpha}"
    assert b["n_contrasts"] == 12 and abs(q - 0.025 / 12) < 1e-15
    assert abs(15 * (0.025 / 13) - 0.02884615384615) < 1e-9, "the pre-W9 value, kept as the regression"
    for cid in ("F1",):
        assert r["contrasts"][cid]["tails"] == 2
        assert abs(r["contrasts"][cid]["quantile"] - q / 2) < 1e-15


    """Codex finding 4, now retired with F2: no live contrast may name an adaptively-selected comparator."""
    r = L.cfg()
    for cid, c in r["contrasts"].items():
        assert c["b"] != "F-winner", cid


def test_C_is_CUT_and_the_denominator_is_12_after_L12():
    """W8 band 1 cuts C-M9init: it warm-starts from 3.69B tokens and wins its own contrast by
    construction. The registry and STATUS disagreed on this for several hours."""
    r = L.cfg()
    c = r["arms"]["C-M9init"]
    assert c["trained"] is False and c.get("cut")
    assert "skipped_iff" not in c, "C is CUT, not conditionally skipped"
    assert r.get("trained_arms_if_C_skipped") is None
    assert r["trained_arms_expected"] == 14
    assert r["statistics"]["bootstrap"]["n_contrasts"] == 12, "13 minus F2, cut with L12 (Dylan 2026-09-05)"


def test_an_arm_marked_cut_but_still_trained_is_refused():
    bad = _mut(lambda r: r["arms"]["C-M9init"].__setitem__("trained", True))
    assert any("cut but still trained" in b or "trained arms" in b for b in bad), bad
