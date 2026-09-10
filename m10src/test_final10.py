"""Executable examples for the final-run decision procedure.

It governs one irreversible access, and its failure modes are all silent: the wrong quantile, the
wrong multiplicity correction, an untested conjunct reported as a failure, a partition quietly
shrinking. Each has a test.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import final10 as F


CONF = F.cfg()


def _stat(lower, delta=None, cid="C1b", **override):
    """A REGISTRY-CONFORMANT stat dict. It has to be, now: `assert_evidence_matches_registry`
    refuses evidence whose metadata does not match the registered procedure, and the helper that
    builds test evidence must therefore build the real shape or every test is testing the guard."""
    part = CONF["partitions"][CONF["conjuncts"][cid]["partition"]]
    b = CONF["bootstrap"]
    st = {"lower_q025_raw": lower,
          "delta_raw": delta if delta is not None else lower + 0.01,
          "quantile": b["quantile"], "quantile_method": b["quantile_method"], "B": b["B"],
          "k_datasets": len(part), "n_by_dataset": {ds: 100 for ds in part}}
    st.update(override)
    return st


def _ev(**kw):
    """{cid: (lower, p)} -> the evidence dict."""
    return {cid: {"stat": _stat(l, cid=cid), "signflip_p": p} for cid, (l, p) in kw.items()}


# ---------------------------------------------------------------- the sequence -------------------

def test_the_registered_order_is_C1b_C1a_C2a_C2b():
    """Fixed 2026-09-04, before any M10 six-set output existed. Release on the headline partition
    first, and no aim conjunct before its own release conjunct."""
    assert F.sequence() == ["C1b", "C1a", "C2a", "C2b"]


def test_a_registry_whose_two_statements_of_the_order_disagree_is_refused():
    conf = F.cfg()
    conf["conjuncts"]["C2b"]["order"] = 1
    with pytest.raises(ValueError, match="disagrees"):
        F.sequence(conf)


def test_it_stops_at_the_first_non_rejection_and_calls_the_rest_NOT_TESTED():
    """The one that matters most: an untested conjunct has NO verdict. Reporting it as a failure
    would invent evidence against us as readily as reporting it as a pass would invent evidence
    for us."""
    d = F.decide(_ev(C1b=(0.01, 0.001), C1a=(-0.002, 0.4)))
    assert d["conjuncts"]["C1b"]["status"] == F.REJECTED
    assert d["conjuncts"]["C1a"]["status"] == F.NOT_REJECTED
    assert d["conjuncts"]["C2a"]["status"] == F.NOT_TESTED
    assert d["conjuncts"]["C2b"]["status"] == F.NOT_TESTED
    assert d["not_tested"] == ["C2a", "C2b"] and d["rejected"] == ["C1b"]
    for c in ("C2a", "C2b"):
        assert "never reported as failed" in d["conjuncts"][c]["_why"]


def test_a_failure_at_the_very_first_conjunct_tests_nothing_else():
    d = F.decide(_ev(C1b=(-0.001, 0.9)))
    assert d["rejected"] == [] and d["not_tested"] == ["C1a", "C2a", "C2b"]
    assert d["reserved_batch_runs"] is False


def test_all_four_can_reject():
    d = F.decide(_ev(C1b=(0.02, 0.0001), C1a=(0.02, 0.0001),
                     C2a=(0.01, 0.001), C2b=(0.005, 0.01)))
    assert d["rejected"] == ["C1b", "C1a", "C2a", "C2b"] and d["not_tested"] == []


def test_missing_evidence_for_a_conjunct_the_sequence_REACHED_is_an_error():
    """Absent-because-unreached is NOT_TESTED; absent-because-nobody-scored-it is a bug, and the
    difference must not be silently collapsed."""
    with pytest.raises(ValueError, match="still untested"):
        F.decide(_ev(C1b=(0.01, 0.001)))


# ---------------------------------------------------------------- the pass rule ------------------

def test_BOTH_conjuncts_of_the_pass_rule_bind():
    alpha = 0.025
    assert F.conjunct_rejects(_stat(0.01), 0.001, alpha) is True
    assert F.conjunct_rejects(_stat(0.01), 0.30, alpha) is False, "sign-flip alone can refuse"
    assert F.conjunct_rejects(_stat(-0.01), 0.001, alpha) is False, "the bound alone can refuse"
    assert F.conjunct_rejects(_stat(0.0), 0.001, alpha) is False, "the bound must EXCEED zero"


def test_the_alpha_is_the_full_0_025_not_a_split():
    """Gatekeeping spends the family alpha once in a fixed order. M9 split it (Holm-2 at 0.0125);
    applying that here would silently make every conjunct harder."""
    conf = F.cfg()
    assert conf["sequence"]["alpha_per_conjunct"] == 0.025
    d = F.decide(_ev(C1b=(0.01, 0.02), C1a=(-0.001, 0.9)), conf)
    assert d["conjuncts"]["C1b"]["status"] == F.REJECTED, \
        "p = 0.02 rejects at 0.025 and would NOT at M9's Holm-2 0.0125"


# ---------------------------------------------------------------- the statistics -----------------

def _aligned(datasets, n=40, shift=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return {ds: (list(range(n)), rng.normal(0.5 + shift, 0.1, n), rng.normal(0.5, 0.1, n))
            for ds in datasets}


def test_the_bootstrap_gate_is_the_250th_order_statistic_not_an_interpolated_one():
    """`linear` interpolates toward the next order statistic and returns a weakly MORE PERMISSIVE
    bound — the defect caught in M9 by review before any six-set number existed."""
    al = _aligned(F.cfg()["partitions"]["clean4"])
    plan, _ = F.draw_plan(al, B=2000, seed=900)
    inv = F.bootstrap(al, plan, 0.025, method="inverted_cdf")
    lin = F.bootstrap(al, plan, 0.025, method="linear")
    assert lin["lower_q025_raw"] >= inv["lower_q025_raw"], \
        "linear must be the weakly more permissive one, which is why it is banned"
    assert inv["quantile_method"] == "inverted_cdf"
    conf = F.cfg()["bootstrap"]
    assert int(conf["quantile"] * conf["B"]) == 250


def test_the_bootstrap_works_on_BOTH_partition_sizes():
    """M9's `final_stats.bootstrap` raises unless k == 6, which is why this module has its own."""
    for part in ("clean4", "all6"):
        ds = F.cfg()["partitions"][part]
        al = _aligned(ds)
        plan, _ = F.draw_plan(al, B=500, seed=900)
        r = F.bootstrap(al, plan, 0.025)
        assert r["k_datasets"] == len(ds) and set(r["per_dataset_delta_raw"]) == set(ds)


def test_a_partition_missing_a_dataset_RAISES_and_does_not_renormalize():
    """A 3-dataset clean-4 macro must be impossible. M9's `_assert_six` lesson, generalized."""
    ds = F.cfg()["partitions"]["clean4"]
    full = _aligned(ds)
    a = {d: dict(zip(v[0], v[1])) for d, v in full.items()}
    b = {d: dict(zip(v[0], v[2])) for d, v in full.items()}
    a.pop(ds[0]); b.pop(ds[0])                     # missing from BOTH: strict alignment allows it
    with pytest.raises(ValueError, match="must be impossible"):
        F.align_partition(a, b, ds)


def test_the_draw_plan_is_shared_byte_identically_within_a_partition():
    ds = F.cfg()["partitions"]["all6"]
    al = _aligned(ds)
    p1, h1 = F.draw_plan(al, B=200, seed=900)
    p2, h2 = F.draw_plan(al, B=200, seed=900)
    assert h1 == h2 and all(np.array_equal(p1[d], p2[d]) for d in ds)


def test_the_vocabulary_of_the_screen_is_kept_out_of_the_VERDICTS():
    """A final-run conjunct REJECTS or does not. It never "resolves" and is never "confirmed" --
    those belong to the screen and are on `forbidden_words`. Keeping the vocabularies apart is what
    stops a screen verdict being read as a release claim. The check is on the STATUS values, which
    are what a report quotes -- not on the prose, which names the banned words deliberately."""
    d = F.decide(_ev(C1b=(0.01, 0.001), C1a=(-0.002, 0.4)))
    statuses = {v["status"] for v in d["conjuncts"].values()}
    assert statuses <= {F.REJECTED, F.NOT_REJECTED, F.NOT_TESTED}
    for w in F.cfg()["forbidden_words"]:
        assert not any(w in st.lower() for st in statuses), w


def test_the_reserved_batch_runs_iff_ANY_conjunct_rejected():
    """So an aim claim never stands without its descriptive reserved rows."""
    assert F.decide(_ev(C1b=(0.01, 0.001), C1a=(-0.1, 0.9)))["reserved_batch_runs"] is True
    assert F.decide(_ev(C1b=(-0.1, 0.9)))["reserved_batch_runs"] is False


# ------------------------------------- the guard both reviewers said was missing -----------------

def test_decide_REFUSES_evidence_that_does_not_match_the_registered_procedure():
    """M9 had `final_stats._assert_matches_registry`; M10 removed that protection and replaced it
    with nothing. Both reviewers demonstrated the consequence on 2026-09-09: evidence produced at
    quantile 0.0125 with `linear` at B=9, and even `signflip_p=-1`, was accepted as REJECTED —
    because the field is NAMED `lower_q025_raw` however it was computed."""
    for bad, match in (
            ({"quantile": 0.0125}, "quantile"),
            ({"quantile_method": "linear"}, "quantile_method"),
            ({"B": 9}, "B="),
            ({"k_datasets": 3}, "k_datasets"),
            ({"n_by_dataset": {"scifact": 100}}, "scored datasets")):
        ev = {"C1b": {"stat": _stat(0.01, **bad), "signflip_p": 0.001}}
        with pytest.raises(ValueError, match="does not match the registered procedure"):
            F.decide(ev)
    for p in (-1, 1.5, None, "0.01"):
        with pytest.raises(ValueError, match="not a probability"):
            F.decide({"C1b": {"stat": _stat(0.01), "signflip_p": p}})


def test_bootstrap_REFUSES_a_non_registered_quantile_or_method_when_given_the_registry():
    """The direction test proves `linear` is more permissive; this proves the code will not compute
    it. A value named `lower_q025_raw` that was not computed at 0.025 by `inverted_cdf` is a lie in
    the field name."""
    al = _aligned(F.cfg()["partitions"]["clean4"])
    plan, _ = F.draw_plan(al, B=200, seed=900)
    conf = F.cfg()
    for q, m in ((0.0125, "inverted_cdf"), (0.025, "linear")):
        with pytest.raises(ValueError, match="refusing to compute"):
            F.bootstrap(al, plan, q, method=m, conf=conf)
    F.bootstrap(al, plan, 0.025, method="inverted_cdf", conf=conf)      # the registered pair passes


def test_signflip_p_refuses_a_short_partition_exactly_as_align_partition_does():
    """Both halves of a conjunctive pass rule must refuse. The reviewers found `signflip_p`
    filtering to a subset and calling the primitive directly, so a dataset missing from BOTH inputs
    silently produced a p-value over three datasets."""
    ds = F.cfg()["partitions"]["clean4"]
    full = _aligned(ds, n=30)
    a = {d: dict(zip(v[0], v[1])) for d, v in full.items()}
    b = {d: dict(zip(v[0], v[2])) for d, v in full.items()}
    p_ok = F.signflip_p(a, b, ds)
    assert 0.0 <= p_ok <= 1.0
    a.pop(ds[0]); b.pop(ds[0])
    with pytest.raises(ValueError, match="must be impossible"):
        F.signflip_p(a, b, ds)


def test_signflip_p_actually_runs_the_primitive_on_both_partition_sizes():
    """It had no executing test at all, which is how a wrong return shape would have survived."""
    for part in ("clean4", "all6"):
        ds = F.cfg()["partitions"][part]
        full = _aligned(ds, n=25, shift=0.2)
        a = {d: dict(zip(v[0], v[1])) for d, v in full.items()}
        b = {d: dict(zip(v[0], v[2])) for d, v in full.items()}
        p = F.signflip_p(a, b, ds)
        assert isinstance(p, float) and 0.0 <= p <= 1.0
        assert p < 0.5, "a large positive shift should not look null"


def test_NOT_TESTED_entries_still_identify_themselves():
    """A tabulating report writer gets blanks otherwise, and a blank row invites a reader to fill
    it in with the failure that was never measured."""
    d = F.decide(_ev(C1b=(0.01, 0.001), C1a=(-0.002, 0.4)))
    for c in d["not_tested"]:
        e = d["conjuncts"][c]
        assert e["partition"] and e["gate"] and e["bar_comparator"]
