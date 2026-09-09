"""The final-run preregistration. It governs one irreversible access, so every constant in it is
checked against the artifact it came from rather than trusted as typed.

M7's near-miss was a rounded CI endpoint in exactly this kind of file — caught by review, not by a
test. These are the tests.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
REG = json.loads((REPO / "m10" / "final_run_registry.json").read_text())
BARS = json.loads((REPO / "results" / "m10_bars.json").read_text())


def test_every_bar_matches_the_frozen_comparator_table():
    """A bar typed into the registry that disagrees with `m10_bars.json` would decide the release
    on a number nobody measured."""
    want = {"C1a": BARS["registered"]["C1a_release_avg6"],
            "C1b": BARS["registered"]["C1b_release_clean4"],
            "C2a": BARS["registered"]["C2a_aim_avg6"],
            "C2b": BARS["registered"]["C2b_aim_clean4"]}
    for cid, bar in want.items():
        assert REG["conjuncts"][cid]["bar"] == bar, cid
    # and each bar is the comparator's OWN score on that conjunct's OWN partition
    rows = BARS["rows"]
    for cid, c in REG["conjuncts"].items():
        assert rows[c["b"]][c["partition"].replace("clean4", "clean4").replace("all6", "all6")] \
            == c["bar"], f"{cid}: bar is not {c['b']}'s score on {c['partition']}"


def test_the_partitions_are_the_registered_ones_and_clean4_is_a_subset_of_all6():
    assert REG["partitions"]["clean4"] == BARS["clean4"]
    assert sorted(REG["partitions"]["all6"]) == sorted(BARS["all6"])
    assert set(REG["partitions"]["clean4"]) < set(REG["partitions"]["all6"])
    assert set(REG["partitions"]["all6"]) - set(REG["partitions"]["clean4"]) == {"fiqa", "arguana"}, \
        "the two datasets stella discloses as training data are exactly the ones clean-4 drops"


def test_the_sequence_is_fixed_gatekeeping_at_the_full_alpha_not_a_split():
    """Fixed-sequence gatekeeping spends the family alpha once in a pre-fixed order. Splitting it
    (Holm, Bonferroni) would be a DIFFERENT and more conservative design, and M9's was the split
    one — reusing its 0.0125 here would silently change the test."""
    s = REG["sequence"]
    assert s["order"] == ["C1b", "C1a", "C2a", "C2b"]
    assert s["alpha_per_conjunct"] == 0.025
    assert [REG["conjuncts"][c]["order"] for c in s["order"]] == [1, 2, 3, 4]
    assert "holm" not in {k.lower() for k in REG}, "gatekeeping replaces Holm; naming both invites " \
                                                   "a reader to apply the wrong one"
    # the release conjunct on the headline partition goes first
    first = REG["conjuncts"][s["order"][0]]
    assert first["gate"] == "release" and first["partition"] == "clean4"
    # an aim conjunct never precedes its own release conjunct
    for pair in (("C1a", "C2a"), ("C1b", "C2b")):
        assert REG["conjuncts"][pair[0]]["order"] < REG["conjuncts"][pair[1]]["order"]


def test_the_decision_field_is_the_empirical_quantile_at_0_025():
    b = REG["bootstrap"]
    assert b["quantile"] == 0.025 and b["quantile_method"] == "inverted_cdf"
    assert b["decision_field"] == "lower_q025_raw"
    assert "250th order statistic" in b["decision_field_def"]
    assert b["B"] == 10000 and int(b["quantile"] * b["B"]) == 250
    assert "linear" in b["_method_note"], "the rejected method must be named, or it comes back"
    assert REG["signflip"]["alpha"] == 0.025, "both conjuncts of the pass rule sit at the same alpha"
    assert "BOTH" in REG["pass_rule"]


def test_the_frozen_comparator_file_is_pinned_by_hash_and_unchanged():
    """`results/perquery.json` is irreplaceable — regenerated from caches that no longer exist."""
    src = REG["comparator_source"]
    p = REPO / src["path"]
    assert p.exists()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == src["sha256"], \
        "perquery.json has CHANGED since the registry pinned it"
    assert "NEVER overwrite" in src["_irreplaceable"]


def test_the_M9_decide_is_explicitly_not_reused():
    """`m9src/final9.py:decide()` is hard-coded to two conjuncts under Holm-2 at 0.0125. Reusing it
    would apply the wrong quantile AND the wrong multiplicity correction, silently."""
    impl = REG["implementation"]
    assert "final9.py:decide()" in impl["does_NOT_reuse"]
    assert "NOT YET WRITTEN" in impl["module"]
    for required in ("first non-rejection", "NOT TESTED", "250th order statistic"):
        assert required in impl["requirement"], required


def test_the_reserved_trigger_is_ANY_conjunct_not_the_first():
    """So an aim claim never stands without its descriptive reserved rows."""
    r = REG["reserved"]
    assert r["trigger"].startswith("if ANY C-conjunct rejects")
    assert r["alpha"] == 0.0 and r["gate"] is None
    assert "DOUBLE-CONTAMINATED" in r["fever"]
    assert set(r["datasets"]) == {"FEVER", "dbpedia-entity", "cqadup-android", "cqadup-english"}


def test_it_is_a_preregistration_not_an_amendment():
    """M9's equivalent was written after its build and had to be classified an amendment. This one
    is fixed before the build has run, and the distinction must not blur."""
    assert REG["classification"] == "preregistration"
    assert "before" in REG["_classification_note"].lower()
    assert REG["ratified_by_owner"] is False
