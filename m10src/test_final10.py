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


def _scores(cid, shift, n=40, seed=0):
    """Synthetic per-query scores for a conjunct's two systems, on its REGISTERED partition,
    clipped into the nDCG@10 range."""
    c = CONF["conjuncts"][cid]
    ds = CONF["partitions"][c["partition"]]
    rng = np.random.default_rng(seed)
    return {c["a"]: {d: {i: float(v) for i, v in
                         enumerate(np.clip(rng.normal(0.5 + shift, 0.1, n), 0.0, 1.0))} for d in ds},
            c["b"]: {d: {i: float(v) for i, v in
                         enumerate(np.clip(rng.normal(0.5, 0.1, n), 0.0, 1.0))} for d in ds}}


def _aligned_from_scores(scores, cid, conf):
    """The same alignment `evidence_for` builds, so a test can redraw the plan independently."""
    c = conf["conjuncts"][cid]
    return F.align_partition(scores[c["a"]], scores[c["b"]], conf["partitions"][c["partition"]])


def _stat(lower, delta=None, cid="C1b", **override):
    """A REGISTRY-CONFORMANT stat dict. It has to be, now: `assert_evidence_matches_registry`
    refuses evidence whose metadata does not match the registered procedure, and the helper that
    builds test evidence must therefore build the real shape or every test is testing the guard."""
    part = CONF["partitions"][CONF["conjuncts"][cid]["partition"]]
    b = CONF["bootstrap"]
    st = {"lower_q025_raw": lower,
          "delta_raw": delta if delta is not None else lower + 0.01,
          "quantile": b["quantile"], "quantile_method": b["quantile_method"], "B": b["B"],
          "bootstrap_seed": b["seed"],
          "k_datasets": len(part), "n_by_dataset": {ds: 100 for ds in part}}
    st.update(override)
    return st


def _ev(**kw):
    """{cid: (lower, p)} -> registry-conformant evidence, including every provenance field the
    guard now requires. It has to be complete: the guard refuses absence as well as mismatch."""
    sf, out = CONF["signflip"], {}
    for cid, (l, p) in kw.items():
        c = CONF["conjuncts"][cid]
        out[cid] = {"stat": _stat(l, cid=cid), "signflip_p": p, "conjunct": cid,
                    "signflip_B": sf["B"], "signflip_seed": sf["seed"],
                    "signflip_alternative": sf["alternative"],
                    "signflip_unit_of": "boot.unit_key",
                    "a": c["a"], "b": c["b"], "partition": c["partition"],
                    "comparator_source_sha256": CONF["comparator_source"]["sha256"],
                    "draw_plan_sha256": "0" * 64, "qid_sha256": "1" * 64,
                    "registry_sha256": F.registry_sha256(),
                    "signflip_per_dataset_n": dict(_stat(0, cid=cid)["n_by_dataset"])}
    return out


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
    """nDCG@10 values, so CLIPPED into [0, 1] — the aligner now refuses anything outside it, and a
    fixture that generates 1.007 is not a fixture of nDCG."""
    rng = np.random.default_rng(seed)
    return {ds: (list(range(n)),
                 np.clip(rng.normal(0.5 + shift, 0.1, n), 0.0, 1.0),
                 np.clip(rng.normal(0.5, 0.1, n), 0.0, 1.0))
            for ds in datasets}


def test_the_bootstrap_gate_is_the_250th_order_statistic_not_an_interpolated_one():
    """`linear` interpolates toward the next order statistic and returns a weakly MORE PERMISSIVE
    bound — the defect caught in M9 by review before any six-set number existed."""
    # The DIRECTION is demonstrated with numpy directly. `bootstrap()` no longer takes a quantile
    # or a method at all, so there is no unsafe path here to demonstrate it through.
    rng = np.random.default_rng(0)
    for _ in range(200):
        draws = rng.normal(0.01, 0.005, 10_000)
        inv = float(np.quantile(draws, 0.025, method="inverted_cdf"))
        lin = float(np.quantile(draws, 0.025, method="linear"))
        assert lin >= inv, "linear must be the weakly MORE PERMISSIVE one, which is why it is banned"
        assert inv == float(np.sort(draws)[249]), "inverted_cdf IS the 250th order statistic"
    conf = F.cfg()["bootstrap"]
    assert int(conf["quantile"] * conf["B"]) == 250


def test_the_bootstrap_works_on_BOTH_partition_sizes():
    """M9's `final_stats.bootstrap` raises unless k == 6, which is why this module has its own."""
    for part in ("clean4", "all6"):
        ds = F.cfg()["partitions"][part]
        al = _aligned(ds)
        conf = F.cfg()
        plan, _ = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
        r = F.bootstrap(al, plan, conf)
        assert r["k_datasets"] == len(ds) and set(r["per_dataset_delta_raw"]) == set(ds)
        assert r["quantile"] == 0.025 and r["quantile_method"] == "inverted_cdf"


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
    for bad, needle in (
            ({"quantile": 0.0125}, "quantile=0.0125"),
            ({"quantile_method": "linear"}, "quantile_method='linear'"),
            ({"B": 9}, "B=9"),
            ({"k_datasets": 3}, "k_datasets=3"),
            ({"bootstrap_seed": 12345}, "bootstrap_seed=12345")):
        # ONE field mutated on OTHERWISE COMPLETE evidence, and the message must name THAT field.
        # The first version of this test built evidence with no provenance at all, so every case
        # raised with nine problems and matched only the generic prefix -- deleting any individual
        # check would have left it green (Fable, 2026-09-10).
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"]["stat"].update(bad)
        with pytest.raises(ValueError) as exc:
            F.decide(ev)
        assert needle in str(exc.value), (needle, str(exc.value))
        assert str(exc.value).count(";") == 0, \
            f"exactly ONE problem should be reported for a single mutation: {exc.value}"
    # dropping datasets legitimately trips TWO clauses -- the partition check and the
    # bootstrap-vs-signflip cross-check -- so it is asserted by needle, not by count.
    ev = _ev(C1b=(0.01, 0.001))
    ev["C1b"]["stat"]["n_by_dataset"] = {"scifact": 100}
    with pytest.raises(ValueError, match="scored datasets"):
        F.decide(ev)
    for p in (-1, 1.5, None, "0.01"):
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"]["signflip_p"] = p
        with pytest.raises(ValueError, match="not a probability"):
            F.decide(ev)


def test_bootstrap_takes_no_quantile_argument_at_all_and_refuses_a_wrong_sized_plan():
    """The unsafe API is gone rather than guarded: there is no parameter through which an arbitrary
    quantile or method could be labelled `lower_q025_raw`. And a plan whose replicate count is not
    the registered B is refused, which the old test violated without noticing (it used B=200
    against a registry requiring 10,000)."""
    import inspect
    params = list(inspect.signature(F.bootstrap).parameters)
    assert params == ["aligned", "plan", "conf"], params
    al = _aligned(F.cfg()["partitions"]["clean4"])
    short, _ = F.draw_plan(al, B=200, seed=900)
    with pytest.raises(ValueError, match="replicate counts"):
        F.bootstrap(al, short, F.cfg())


def test_signflip_consumes_the_SAME_aligned_object_and_returns_full_provenance():
    """The partition is no longer a caller argument — `aligned` arrives already validated against
    the conjunct's REGISTERED partition, so a caller cannot pass a valid three-dataset list. And it
    returns `R`, the seed, the alternative and the strata instead of discarding them."""
    for part in ("clean4", "all6"):
        ds = F.cfg()["partitions"][part]
        al = _aligned(ds, n=25, shift=0.2)
        r = F.signflip(al, F.cfg())
        assert 0.0 <= r["signflip_p"] <= 1.0 and r["signflip_p"] < 0.5
        assert r["signflip_B"] == F.cfg()["signflip"]["B"]
        assert r["signflip_seed"] == F.cfg()["signflip"]["seed"]
        assert r["signflip_alternative"] == "greater"
        assert r["signflip_unit_of"] == "boot.unit_key"
        assert len(r["signflip_strata"]) == len(ds)


def test_NOT_TESTED_entries_still_identify_themselves():
    """A tabulating report writer gets blanks otherwise, and a blank row invites a reader to fill
    it in with the failure that was never measured."""
    d = F.decide(_ev(C1b=(0.01, 0.001), C1a=(-0.002, 0.4)))
    for c in d["not_tested"]:
        e = d["conjuncts"][c]
        assert e["partition"] and e["gate"] and e["bar_comparator"]


# ------------------------- the attacks Codex REPRODUCED on 2026-09-10 ---------------------------

def test_a_doctored_registry_cannot_be_handed_to_decide():
    """Codex set `alpha_per_conjunct` to 1.0 and a p-value of 0.9 was accepted as REJECTED. The
    guard was checking the evidence against the CALLER'S ruler."""
    bad = F.cfg()
    bad["sequence"]["alpha_per_conjunct"] = 1.0
    with pytest.raises(ValueError, match="not .m10/final_run_registry.json. on disk"):
        F.decide(_ev(C1b=(0.01, 0.9)), bad)
    # and the same evidence is refused under the real registry
    d = F.decide(_ev(C1b=(0.01, 0.9), C1a=(-0.1, 0.9)))
    assert d["conjuncts"]["C1b"]["status"] == F.NOT_REJECTED, "p = 0.9 must not reject at 0.025"


def test_False_is_not_a_probability():
    """`bool` is a subclass of `int`, so `isinstance(True, int)` is True. `False` slipped through
    and behaved as p = 0, i.e. it REJECTED. Codex reproduced it."""
    for p in (False, True):
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"]["signflip_p"] = p
        with pytest.raises(ValueError, match="not a probability"):
            F.decide(ev)


def test_ABSENCE_of_provenance_is_not_equality():
    """`ev.get(k, registered_value)` treated a MISSING key as a match, so evidence carrying no
    sign-flip provenance at all passed the guard."""
    for k in ("signflip_B", "signflip_seed", "signflip_alternative", "signflip_unit_of",
              "signflip_per_dataset_n"):
        ev = _ev(C1b=(0.01, 0.001))
        del ev["C1b"][k]
        with pytest.raises(ValueError, match=f"no '{k}'"):
            F.decide(ev)
    # the digests report their absence as a malformed value, which is the same refusal
    for k in ("draw_plan_sha256", "qid_sha256", "registry_sha256"):
        ev = _ev(C1b=(0.01, 0.001))
        del ev["C1b"][k]
        with pytest.raises(ValueError, match=f"{k}=None"):
            F.decide(ev)


def test_a_REVERSED_contrast_with_conforming_metadata_is_refused():
    """The orientation was unchecked, so b-minus-a could be relabelled as a-minus-b."""
    ev = _ev(C1b=(0.01, 0.001))
    ev["C1b"]["a"], ev["C1b"]["b"] = ev["C1b"]["b"], ev["C1b"]["a"]
    with pytest.raises(ValueError, match="oriented"):
        F.decide(ev)


def test_the_wrong_partition_is_refused():
    """The comparator-hash half of this test is gone with the check: `evidence_for` STAMPS that
    field from the registry, so verifying it against the registry was a tautology, not a
    measurement (both reviewers, round 8). Verifying the real comparator FILE is executor work,
    registered as such."""
    ev = _ev(C1b=(0.01, 0.001))
    ev["C1b"]["partition"] = "all6"
    with pytest.raises(ValueError, match="partition="):
        F.decide(ev)


def test_invented_query_counts_are_refused():
    ev = _ev(C1b=(0.01, 0.001))
    ev["C1b"]["stat"]["n_by_dataset"] = {d: 0 for d in ev["C1b"]["stat"]["n_by_dataset"]}
    with pytest.raises(ValueError, match="non-positive"):
        F.decide(ev)


# ------------------------------- the one production path, end to end ----------------------------

def test_evidence_for_derives_everything_from_the_registry():
    """The structural fix: the caller supplies raw per-query scores and NOTHING else. There is no
    parameter through which a partition, an orientation, a quantile, a method or a seed could be
    chosen, so there is nothing to forge."""
    import inspect
    assert list(inspect.signature(F.evidence_for).parameters) == ["cid", "scores", "conf"]
    ev = F.evidence_for("C1b", _scores("C1b", shift=0.05))
    c = F.cfg()["conjuncts"]["C1b"]
    assert ev["a"] == c["a"] and ev["b"] == c["b"] and ev["partition"] == "clean4"
    assert ev["stat"]["quantile"] == 0.025 and ev["stat"]["quantile_method"] == "inverted_cdf"
    assert ev["stat"]["B"] == 10_000 and ev["stat"]["k_datasets"] == 4
    assert len(ev["draw_plan_sha256"]) == 64 and len(ev["qid_sha256"]) == 64
    # and it passes its own guard, which is the point
    F.assert_evidence_matches_registry("C1b", ev, F.cfg())


def test_evidence_for_is_reproducible_and_refuses_a_missing_system():
    a = F.evidence_for("C1b", _scores("C1b", shift=0.05))
    b = F.evidence_for("C1b", _scores("C1b", shift=0.05))
    assert a["draw_plan_sha256"] == b["draw_plan_sha256"]
    assert a["stat"]["draws_sha256"] == b["stat"]["draws_sha256"]
    with pytest.raises(ValueError, match="no scores supplied"):
        F.evidence_for("C1b", {})


def test_a_real_positive_effect_rejects_end_to_end():
    """No hand-built stat dicts: scores in, verdict out, through the production path only."""
    ev = {cid: F.evidence_for(cid, _scores(cid, shift=0.08)) for cid in F.sequence()}
    d = F.decide(ev)
    assert d["rejected"] == ["C1b", "C1a", "C2a", "C2b"], d["conjuncts"]
    h = F.headline(d)
    assert [x["conjunct"] for x in h["emit"]] == ["C1b", "C1a", "C2a", "C2b"]


# ------------------------------------------ the headline ----------------------------------------

def test_the_headline_emits_NOTHING_for_a_conjunct_that_was_not_tested():
    """The failure mode a report writer would otherwise commit: claiming an aim result that the
    sequence never tested."""
    d = F.decide(_ev(C1b=(0.01, 0.001), C1a=(-0.002, 0.4)))
    h = F.headline(d)
    assert [x["conjunct"] for x in h["emit"]] == ["C1b"]
    blob = " ".join(x["sentence"] for x in h["emit"]).lower()
    assert "arctic" not in blob, "no aim sentence may be emitted when C2a/C2b were NOT_TESTED"
    assert "clean-4" in blob and "nfcorpus" in blob.replace("NFCorpus".lower(), "nfcorpus")


def test_the_headline_emits_nothing_at_all_when_nothing_rejected():
    d = F.decide(_ev(C1b=(-0.01, 0.9)))
    h = F.headline(d)
    assert h["emit"] == [] and h["emit_nothing_because"] == "no conjunct rejected"


def test_every_emitted_sentence_avoids_the_forbidden_vocabulary():
    ev = {cid: F.evidence_for(cid, _scores(cid, shift=0.08)) for cid in F.sequence()}
    h = F.headline(F.decide(ev))
    for item in h["emit"]:
        low = item["sentence"].lower()
        for w in F.cfg()["forbidden_words"]:
            assert w not in low, (item["conjunct"], w)


def test_the_gate_is_PRODUCTION_bootstrap_s_250th_order_statistic():
    """The previous version of this demonstrated `inverted_cdf` on `rng.normal` draws and never
    touched `bootstrap()` — it proved a numpy property, not ours (Fable, 2026-09-10). This
    reconstructs the draw vector from the production plan and checks the emitted field IS the
    250th order statistic of it, and that the recorded digest matches."""
    import hashlib
    conf = F.cfg()
    ds = conf["partitions"]["clean4"]
    al = _aligned(ds, n=30)
    plan, _ = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
    stat = F.bootstrap(al, plan, conf)
    diffs = {d: (x - y) for d, (_q, x, y) in al.items()}
    draws = np.zeros(conf["bootstrap"]["B"], dtype=np.float64)
    for d, v in diffs.items():
        draws += v[plan[d]].mean(axis=1)
    draws /= len(diffs)
    assert stat["lower_q025_raw"] == float(np.sort(draws)[249]), \
        "the emitted gate must be the 250th order statistic of the production draw vector"
    assert stat["draws_sha256"] == hashlib.sha256(
        np.ascontiguousarray(draws).tobytes()).hexdigest()


def test_canonical_returns_the_FILE_not_an_object_that_merely_compares_equal():
    """`dict.__eq__` compares stored items, so a subclass overriding `__getitem__` compares equal
    and reads differently — and the first version returned the caller's object, reopening the
    alpha=1.0 attack it was written to close (Fable, 2026-09-10)."""
    class Sneaky(dict):
        def __getitem__(self, k):
            if k == "sequence":
                return {"order": ["C1b", "C1a", "C2a", "C2b"], "alpha_per_conjunct": 1.0}
            return super().__getitem__(k)

    sneaky = Sneaky(F.cfg())
    assert sneaky["sequence"]["alpha_per_conjunct"] == 1.0
    assert F.canonical(sneaky)["sequence"]["alpha_per_conjunct"] == 0.025
    d = F.decide(_ev(C1b=(0.01, 0.9), C1a=(-0.1, 0.9)), sneaky)
    assert d["conjuncts"]["C1b"]["status"] == F.NOT_REJECTED, \
        "p = 0.9 must not reject, whatever alpha the caller's object reports"


def test_a_single_non_finite_score_is_refused_rather_than_flipping_the_verdict():
    """Not an attack — a plausible accident (an unscored query, a divide-by-zero nDCG, a missing
    qrel). One NaN made the bound NaN, `nan > 0` False, and the sign-flip p its MINIMUM: a true
    positive silently became NOT_REJECTED with a record that looked valid."""
    sc = _scores("C1b", shift=0.08)
    c = F.cfg()["conjuncts"]["C1b"]
    ds = F.cfg()["partitions"]["clean4"][0]
    first = next(iter(sc[c["a"]][ds]))
    for bad in (float("nan"), float("inf")):
        sc[c["a"]][ds][first] = bad
        with pytest.raises(ValueError, match="non-finite per-query score"):
            F.evidence_for("C1b", sc)


def test_the_two_halves_of_the_pass_rule_must_have_scored_the_same_queries():
    ev = _ev(C1b=(0.01, 0.001))
    ev["C1b"]["signflip_per_dataset_n"] = {d: 7 for d in ev["C1b"]["stat"]["n_by_dataset"]}
    with pytest.raises(ValueError, match="the sign-flip scored"):
        F.decide(ev)


def test_an_empty_or_mixed_draw_plan_is_a_ValueError_not_a_StopIteration():
    conf = F.cfg()
    al = _aligned(conf["partitions"]["clean4"])
    with pytest.raises(ValueError, match="empty draw plan"):
        F.bootstrap(al, {}, conf)
    good, _ = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
    short, _ = F.draw_plan(al, B=50, seed=900)
    mixed = dict(good); mixed[conf["partitions"]["clean4"][0]] = short[conf["partitions"]["clean4"][0]]
    with pytest.raises(ValueError, match="replicate counts"):
        F.bootstrap(al, mixed, conf)


def test_the_decision_field_must_be_a_plausible_nDCG_difference():
    """Both decision numbers must stay within the range of nDCG differences."""
    for field in ("lower_q025_raw", "delta_raw"):
        for value in (-1.5, 1.5, 1e300):
            ev = _ev(C1b=(0.01, 0.001))
            ev["C1b"]["stat"][field] = value
            with pytest.raises(ValueError, match=r"outside \[-1, 1\]"):
                F.decide(ev)
    # `lower > delta` is REPORTED, never refused: it is not a bootstrap invariant, and a
    # categorical refusal on the decision path would consume the irreversible access on a
    # legitimate run (Codex, 2026-09-10, constructed a fixture producing it).
    ev = _ev(C1b=(0.01, 0.001), C1a=(-0.1, 0.9))
    ev["C1b"]["stat"]["lower_q025_raw"], ev["C1b"]["stat"]["delta_raw"] = 0.5, 0.3
    F.decide(ev)      # must NOT raise


def test_every_type_that_must_not_be_the_decision_field():
    from decimal import Decimal
    from fractions import Fraction
    for bad in (True, False, None, "0.01", Decimal("0.01"), Fraction(1, 100),
                float("nan"), float("inf"), float("-inf"), [0.01]):
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"]["stat"]["lower_q025_raw"] = bad
        with pytest.raises(ValueError):
            F.decide(ev)
    # numpy's float64 IS accepted, deliberately: the exact-type/`__float__` arms race was dropped
    # as over-engineering (2026-09-10). `evidence_for` being the only production path is the
    # control, not type forensics at the comparison site.
    ev = _ev(C1b=(0.01, 0.001), C1a=(-0.1, 0.9))
    ev["C1b"]["stat"]["lower_q025_raw"] = np.float64(0.01)
    assert F.decide(ev)["conjuncts"]["C1b"]["status"] == F.REJECTED


def test_the_evidence_must_be_labelled_with_its_own_conjunct():
    for bad in (None, "C1a", "c1b", ""):
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"]["conjunct"] = bad
        with pytest.raises(ValueError, match="is labelled conjunct"):
            F.decide(ev)


def test_provenance_MISMATCH_is_refused_not_only_absence():
    """Only absence was tested; a WRONG value survived mutation."""
    for k, bad in (("signflip_B", 999), ("signflip_seed", 12345),
                   ("signflip_alternative", "less"), ("signflip_unit_of", "something-else")):
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"][k] = bad
        with pytest.raises(ValueError, match=f"{k}="):
            F.decide(ev)


def test_a_PARTIAL_orientation_change_is_refused_not_only_a_full_swap():
    for side in ("a", "b"):
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"][side] = "some-other-system"
        with pytest.raises(ValueError, match="oriented"):
            F.decide(ev)


def test_a_digest_must_be_a_sha256_not_merely_truthy():
    for k in ("draw_plan_sha256", "qid_sha256", "registry_sha256"):
        for bad in ("", "abc", None, 0, "0" * 63):
            ev = _ev(C1b=(0.01, 0.001))
            ev["C1b"][k] = bad
            with pytest.raises(ValueError, match=f"{k}="):
                F.decide(ev)


def test_query_counts_must_be_positive_ints_and_present():
    for bad in ({}, {"nfcorpus": True}, {"nfcorpus": 1.5}, {"nfcorpus": -1}):
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"]["stat"]["n_by_dataset"] = bad
        with pytest.raises(ValueError):
            F.decide(ev)


def test_the_two_halves_cross_check_is_REQUIRED_not_skipped_when_absent():
    """Fable mutation S1: dropping `signflip_per_dataset_n` made the check vanish silently — the
    exact `is not None` pattern named twenty lines above it in the same file, in the same batch."""
    ev = _ev(C1b=(0.01, 0.001))
    del ev["C1b"]["signflip_per_dataset_n"]
    with pytest.raises(ValueError, match="no 'signflip_per_dataset_n'"):
        F.decide(ev)


def test_a_malformed_descriptive_record_on_a_NOT_TESTED_conjunct_is_still_refused():
    """The guard runs on NOT_TESTED conjuncts too, so their descriptive numbers cannot be junk."""
    ev = _ev(C1b=(0.01, 0.001), C1a=(-0.002, 0.4), C2a=(0.01, 0.001))
    ev["C2a"]["stat"]["quantile"] = 0.0125
    with pytest.raises(ValueError, match="quantile=0.0125"):
        F.decide(ev)


def test_the_primitives_themselves_refuse_non_finite_and_out_of_range_scores():
    """The guard lives in `align_partition` — the one object BOTH halves consume — so no other
    route reaches `bootstrap` or `signflip` with a NaN (Fable, 2026-09-10)."""
    ds = F.cfg()["partitions"]["clean4"]
    for bad in (float("nan"), float("inf"), 5.0, -3.0):
        full = _aligned(ds, n=20)
        a = {d: dict(zip(v[0], v[1])) for d, v in full.items()}
        b = {d: dict(zip(v[0], v[2])) for d, v in full.items()}
        a[ds[0]][0] = bad
        with pytest.raises(ValueError, match="non-finite|outside"):
            F.align_partition(a, b, ds)


def test_an_inverted_bound_is_REPORTED_and_never_refused():
    """A guard that can refuse a legitimate run is worse than the hole it closes, because the
    access is already spent by the time `decide()` runs. 4,150 fixtures at the registered seed and
    B produced zero inversions — but 'astronomically unlikely' is not 'impossible'."""
    ev = _ev(C1b=(0.6, 0.001), C1a=(-0.1, 0.9))
    ev["C1b"]["stat"]["delta_raw"] = 0.4          # lower > delta
    d = F.decide(ev)                              # does not raise
    assert d["conjuncts"]["C1b"]["status"] == F.REJECTED


def test_the_primitives_refuse_bad_scores_DIRECTLY_not_only_through_the_aligner():
    """The check lived only in `align_partition`, so `bootstrap` and `signflip` stayed reachable
    with a NaN by any other route — and I wrote a test named 'the primitives themselves refuse'
    that only exercised the aligner. A claim in a test name is not a check."""
    conf = F.cfg()
    ds = conf["partitions"]["clean4"]
    al = _aligned(ds, n=20)
    plan, _ = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
    for bad in (float("nan"), 5.0):
        broken = {d: (q, x.copy(), y.copy()) for d, (q, x, y) in al.items()}
        broken[ds[0]][1][0] = bad
        with pytest.raises(ValueError, match="non-finite|outside"):
            F.bootstrap(broken, plan, conf)
        with pytest.raises(ValueError, match="non-finite|outside"):
            F.signflip(broken, conf)


def test_the_comparator_side_is_validated_too_not_only_the_candidate():
    ds = F.cfg()["partitions"]["clean4"]
    full = _aligned(ds, n=20)
    a = {d: dict(zip(v[0], v[1])) for d, v in full.items()}
    b = {d: dict(zip(v[0], v[2])) for d, v in full.items()}
    b[ds[0]][0] = float("nan")          # the COMPARATOR, not the candidate
    with pytest.raises(ValueError, match="non-finite"):
        F.align_partition(a, b, ds)


def test_a_p_value_exactly_at_alpha_rejects():
    """`p <= alpha`, not `p < alpha`. The mutation survived because nothing tested the boundary."""
    alpha = F.cfg()["sequence"]["alpha_per_conjunct"]
    d = F.decide(_ev(C1b=(0.01, alpha), C1a=(-0.1, 0.9)))
    assert d["conjuncts"]["C1b"]["status"] == F.REJECTED
    d = F.decide(_ev(C1b=(0.01, alpha + 1e-12), C1a=(-0.1, 0.9)))
    assert d["conjuncts"]["C1b"]["status"] == F.NOT_REJECTED


def test_counts_must_be_exactly_int_even_when_every_dataset_is_present():
    """The mutation survived because the dataset-set check masked it: bools and floats slipped
    through whenever the KEYS were still complete."""
    part = F.cfg()["partitions"]["clean4"]
    for bad in (True, 100.0):
        ev = _ev(C1b=(0.01, 0.001))
        ev["C1b"]["stat"]["n_by_dataset"] = {d: bad for d in part}
        ev["C1b"]["signflip_per_dataset_n"] = {d: bad for d in part}
        with pytest.raises(ValueError, match="non-positive or non-integer"):
            F.decide(ev)


def test_delta_raw_range_is_checked_not_only_the_bound():
    ev = _ev(C1b=(0.01, 0.001))
    ev["C1b"]["stat"]["delta_raw"] = 1e300
    with pytest.raises(ValueError, match=r"delta_raw.*outside \[-1, 1\]"):
        F.decide(ev)


# ---------- the two halves of the pass rule, PINNED (Fable round 7: both were unpinned) --------

def test_the_signflip_p_is_pinned_at_both_ends_of_its_range():
    """A mutant that halved `signflip_p` survived all 54 tests: the bootstrap half had the
    order-statistic test and the sign-flip half had nothing checking its VALUE. These two
    identities are exact, so an accidental edit to `signflip()` cannot pass."""
    conf = F.cfg()
    ds = conf["partitions"]["clean4"]
    n = 25
    # identical systems -> every paired difference is 0 -> every sign-flip statistic equals the
    # observed one -> p is exactly 1.0
    same = {d: (list(range(n)), np.full(n, 0.5), np.full(n, 0.5)) for d in ds}
    assert F.signflip(same, conf)["signflip_p"] == 1.0
    # a large uniform positive shift -> no permutation reaches the observed statistic -> p is the
    # Monte Carlo floor, 1/(B+1)
    big = {d: (list(range(n)), np.full(n, 0.9), np.full(n, 0.1)) for d in ds}
    assert F.signflip(big, conf)["signflip_p"] == 1.0 / (conf["signflip"]["B"] + 1)


def test_delta_raw_is_the_mean_of_the_per_dataset_deltas():
    """A mutant that dropped the division by k survived: `delta_raw` decides nothing, but it is the
    point estimate the headline quotes."""
    conf = F.cfg()
    ds = conf["partitions"]["all6"]
    al = _aligned(ds, n=30)
    plan, _ = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
    stat = F.bootstrap(al, plan, conf)
    per = list(stat["per_dataset_delta_raw"].values())
    assert stat["delta_raw"] == pytest.approx(sum(per) / len(per), abs=1e-15)


def test_an_inverted_bound_actually_REACHES_the_verdict():
    """I wrote that `lower > delta` is "REPORTED, never raised" — and then dropped the flag on the
    floor: the guard returned it and `decide()` ignored it. The code and the claim must agree."""
    ev = _ev(C1b=(0.6, 0.001), C1a=(-0.1, 0.9))
    ev["C1b"]["stat"]["delta_raw"] = 0.4
    d = F.decide(ev)
    assert d["conjuncts"]["C1b"]["lower_exceeds_point_estimate"] is True
    d2 = F.decide(_ev(C1b=(0.01, 0.001), C1a=(-0.1, 0.9)))
    assert "lower_exceeds_point_estimate" not in d2["conjuncts"]["C1b"]


def test_a_partitions_two_conjuncts_must_share_one_plan_and_one_qid_set():
    """Otherwise they were computed from different alignments and are not comparable. Free to
    check from the evidence alone, and it cannot fire on a run that came through `evidence_for`."""
    ev = {cid: F.evidence_for(cid, _scores(cid, shift=0.08)) for cid in F.sequence()}
    assert ev["C1b"]["draw_plan_sha256"] == ev["C2b"]["draw_plan_sha256"], "clean4 shares a plan"
    assert ev["C1a"]["draw_plan_sha256"] == ev["C2a"]["draw_plan_sha256"], "all6 shares a plan"
    F.decide(ev)
    ev["C2b"]["qid_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="different alignments or plans"):
        F.decide(ev)


def test_a_comparator_scored_on_only_its_own_partition_does_not_raise():
    """`align_partition` used to demand dataset-set equality BEFORE restricting to the partition,
    so a comparator built per-partition against a candidate carrying all six raised — inside
    `evidence_for`, i.e. after the access is spent."""
    all6 = F.cfg()["partitions"]["all6"]
    clean4 = F.cfg()["partitions"]["clean4"]
    c = F.cfg()["conjuncts"]["C1b"]
    full = _scores("C1a", shift=0.05)              # C1a's partition is all6
    scores = {c["a"]: full[c["a"]], c["b"]: {d: v for d, v in full[c["b"]].items() if d in clean4}}
    assert set(scores[c["a"]]) == set(all6) and set(scores[c["b"]]) == set(clean4)
    ev = F.evidence_for("C1b", scores)             # must not raise
    assert ev["stat"]["k_datasets"] == 4


# ---- the meaningful mutation survivors, round 8 (both reviewers found the same ones) ----------

def test_a_qid_missing_on_ONE_side_raises():
    """`strict=True` in the aligner had no test: only the missing-DATASET case was covered, so a
    mutant flipping it to strict=False survived. A silently-dropped query changes the macro."""
    ds = F.cfg()["partitions"]["clean4"]
    full = _aligned(ds, n=20)
    a = {d: dict(zip(v[0], v[1])) for d, v in full.items()}
    b = {d: dict(zip(v[0], v[2])) for d, v in full.items()}
    b[ds[0]].pop(0)                       # one query missing from the comparator only
    with pytest.raises(Exception):
        F.align_partition(a, b, ds)


def test_evidence_for_uses_the_REGISTERED_plan_seed():
    """`bootstrap_seed` is COPIED from the registry, not measured from the plan that was drawn, so
    a mutant drawing at seed 901 while reporting 900 survived — and it flipped a verdict. Pinning
    the plan digest against an independently drawn plan is what makes the seed observable."""
    conf = F.cfg()
    ev = F.evidence_for("C1b", _scores("C1b", shift=0.05))
    ds = conf["partitions"]["clean4"]
    al = _aligned_from_scores(_scores("C1b", shift=0.05), "C1b", conf)
    want, want_digest = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
    assert ev["draw_plan_sha256"] == want_digest, \
        "the plan `evidence_for` drew is not the plan the registered seed produces"
    wrong, wrong_digest = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=901)
    assert wrong_digest != want_digest, "the digest must actually distinguish seeds"


def test_the_plan_must_cover_every_dataset_at_the_right_width():
    conf = F.cfg()
    ds = conf["partitions"]["clean4"]
    al = _aligned(ds, n=20)
    plan, _ = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
    short = {d: v for d, v in plan.items() if d != ds[0]}
    with pytest.raises(ValueError, match="does not cover"):
        F.bootstrap(al, short, conf)
    wrong_width = dict(plan)
    wrong_width[ds[0]] = plan[ds[0]][:, :-1]
    with pytest.raises(ValueError, match="plan width"):
        F.bootstrap(al, wrong_width, conf)


def test_an_unscorable_conjunct_does_not_discard_the_verdicts_already_established():
    """One NaN in a comparator row made C2a unscorable and `decide()` raised — after the access was
    spent, with C1b and C1a already REJECTED. Real rejections must not be thrown away by a later
    conjunct's data problem (Fable, round 9)."""
    ev = _ev(C1b=(0.02, 0.001), C1a=(0.02, 0.001), C2b=(0.01, 0.001))
    ev["C2a"] = {"unscorable": "leaf-ir-asym's fiqa row has a non-finite score"}
    d = F.decide(ev)
    assert d["rejected"] == ["C1b", "C1a"], "the established rejections stand"
    assert d["conjuncts"]["C2a"]["status"] == F.UNSCORABLE
    assert "fiqa" in d["conjuncts"]["C2a"]["reason"]
    assert d["conjuncts"]["C2b"]["status"] == F.NOT_TESTED, "the sequence stops at the unscorable"
    assert d["unscorable"] == ["C2a"] and d["reserved_batch_runs"] is True


def test_the_registry_hash_in_evidence_is_the_live_one():
    """`registry_sha256` is recover's only drift anchor and was a stamped constant no test pinned."""
    ev = F.evidence_for("C1b", _scores("C1b", shift=0.05))
    assert ev["registry_sha256"] == F.registry_sha256()


def test_the_draw_plan_covers_every_query_index():
    """A mutant drawing `0..n-2` — never selecting the last query of each dataset — survived. The
    m9 tolerance is too loose to catch it by inspection."""
    conf = F.cfg()
    ds = conf["partitions"]["clean4"]
    al = _aligned(ds, n=12)
    plan, _ = F.draw_plan(al, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
    for d in ds:
        n = len(al[d][0])
        assert set(np.unique(plan[d])) == set(range(n)), \
            f"{d}: the plan never draws index {sorted(set(range(n)) - set(np.unique(plan[d])))}"


def test_the_qid_digest_actually_depends_on_the_qids():
    a = F.evidence_for("C1b", _scores("C1b", shift=0.05))
    sc = _scores("C1b", shift=0.05)
    c = F.cfg()["conjuncts"]["C1b"]
    for side in (c["a"], c["b"]):                       # relabel every qid
        for dset in sc[side]:
            sc[side][dset] = {f"q{k}": v for k, v in sc[side][dset].items()}
    b = F.evidence_for("C1b", sc)
    assert a["qid_sha256"] != b["qid_sha256"], "the digest must depend on the qids themselves"
