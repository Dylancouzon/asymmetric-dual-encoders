"""The confirmatory rule's own refusals and reductions. LEDGER §4.4 gap-list item 1.

A ship rule that has only ever been run on a case that ships is not tested -- it is demonstrated.
Every condition here is exercised in BOTH directions, because the whole value of the rule is what
it refuses, and the refusals are the branches nobody runs by accident.

Two things this file establishes that the ledger asserts in prose:

  * that `paired_dep` REDUCES to the ordinary stratified path on the four reserved sets. §4.1 says
    the confirmatory route may use `boot.paired` because the four sets are disjoint -- no shared
    query, no nesting -- so the dependence-preserving estimator degenerates to it. That is an
    argument, and this makes it a measurement.
  * that the alpha/3 bound computed at the EXACT level agrees with `boot.paired`'s rounded
    "0.8333" percentile key, which is the assertion §4.1 registers and which decide.py enforces.

Zero-padded synthetic qids throughout: `boot._align_ids` sorts qids as STRINGS, so `q0..q10`
permutes the arrays relative to their source order and a correct implementation looks wrong
(m7/CODEMAP.md pitfall 19).
"""
import sys

import numpy as np

import m8base
import boot
import decide

FAILED = []
RESERVED = list(m8base.RESERVED_FOUR)
N = {"fever": 6666, "dbpedia-entity": 400, "cqadup-android": 699, "cqadup-english": 1570}


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILED.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:                                    # noqa: BLE001
        FAILED.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


def make(shift=0.0, seed=0, datasets=None, n=None):
    rng = np.random.default_rng(seed)
    ds = datasets or RESERVED
    n = n or N
    return {d: {f"q{i:07d}": float(v) for i, v in
                enumerate(np.clip(rng.normal(0.45 + shift, 0.30, n[d]), 0, 1))} for d in ds}


def shifted(base, delta):
    return {d: {q: min(1.0, max(0.0, v + delta)) for q, v in base[d].items()} for d in base}


# ---------------------------------------------------------------- reductions ----------------

def t_paired_dep_reduces_on_disjoint_sets():
    """§4.1's load-bearing claim, measured rather than argued."""
    a = make(seed=1)
    b = shifted(a, -0.01)
    ord_ = boot.paired(a, b, alternative="greater", strict=True)
    dep = boot.paired_dep(a, b, alternative="two-sided", strict=True)
    assert abs(ord_["delta_raw"] - dep["delta_raw"]) < 1e-12, \
        f"point estimates differ: {ord_['delta_raw']} vs {dep['delta_raw']}"
    # The interval is a resample, so it agrees to sampling noise, not to machine precision. A 5%
    # tolerance on the half-width is generous and still catches a genuinely different estimator
    # (the dependence-blind interval is 1.43x too narrow under full duplication).
    hw_o = (ord_["ci95_raw"][1] - ord_["ci95_raw"][0]) / 2
    hw_d = (dep["ci95_raw"][1] - dep["ci95_raw"][0]) / 2
    assert abs(hw_o - hw_d) / max(hw_o, 1e-30) < 0.05, \
        f"half-widths differ by more than 5%: {hw_o:.6f} vs {hw_d:.6f} -- the four reserved sets " \
        f"are supposed to be disjoint, so the stratified estimator should degenerate"


def t_exact_bonferroni_matches_the_rounded_key():
    """The assertion §4.1 registers: the exact 100*alpha/3 quantile and boot's '0.8333' string."""
    a = make(seed=2)
    b = shifted(a, -0.02)
    pr = boot.paired(a, b, alternative="greater", strict=True)
    rounded = decide._bonf_lookup(pr["one_sided_lower_raw"])
    exact = decide.exact_bonferroni_lower(a, b)
    assert rounded is not None, "boot.paired did not return the simultaneous bound"
    assert abs(exact - rounded) < 1e-6, \
        f"exact {exact!r} vs rounded {rounded!r} differ by {abs(exact - rounded):.2e}"


def t_decide_family_raises_when_the_two_bounds_disagree():
    """The guard must fire, not be decorative. Feed it a mismatched pair on purpose."""
    a = make(seed=3)
    b = shifted(a, -0.02)
    legs = {"C1": decide.leg("C1", a, b, hypothesis="x"),
            "C2": decide.leg("C2", a, b, hypothesis="x"),
            "C3": decide.leg("C3", a, b, hypothesis="x")}
    legs["C1"]["one_sided_lower_raw"] = {"2.5": 0.0, "0.8333": legs["C1"]["ci95_raw"][0] + 1.0}
    try:
        decide.decide_family(legs, per_query={"C1": (a, b)})
    except AssertionError as e:
        assert "1e-06" in str(e) or "differ" in str(e), str(e)
        return
    raise AssertionError("decide_family accepted an alpha/3 bound that disagrees with the exact one")


# ---------------------------------------------------------------- refusals ------------------

def t_worst_group_aborts_on_a_missing_dataset():
    a = make(seed=4)
    b = shifted(a, -0.005)
    short = {k: v for k, v in a.items() if k != "dbpedia-entity"}
    try:
        decide.worst_group(short, b)
    except AssertionError as e:
        assert "dbpedia-entity" in str(e)
        return
    raise AssertionError("the worst-group guard was evaluated over three of four datasets")


def t_worst_group_blocks_a_single_regressing_dataset():
    a = make(seed=5)
    b = dict(a)
    # everything up 0.02 except DBpedia, which drops 0.02 -- one group regressing must block
    m8 = {d: ({q: min(1.0, v + 0.02) for q, v in a[d].items()} if d != "dbpedia-entity"
              else {q: max(0.0, v - 0.02) for q, v in a[d].items()}) for d in a}
    r = decide.worst_group(m8, b)
    assert not r["pass"], f"a -0.02 regression on one reserved set passed: {r['per_dataset']}"
    assert r["worst"] < r["threshold"]


def t_six_set_guard_aborts_on_misalignment():
    six = make(seed=6, datasets=list(m8base.SIX), n={d: 200 for d in m8base.SIX})
    short = {k: v for k, v in six.items() if k != "trec-covid"}
    try:
        decide.six_set_no_regression(short, six)
    except ValueError as e:
        assert "dataset" in str(e).lower()
        return
    raise AssertionError("the six-set guard compared two different dataset sets")


def t_six_set_guard_margin_comes_from_the_registry():
    six = make(seed=7, datasets=list(m8base.SIX), n={d: 200 for d in m8base.SIX})
    r = decide.six_set_no_regression(six, six)
    assert abs(r["margin"] + 0.0075) < 1e-12, f"margin {r['margin']} is not the registered -0.0075"
    worse = shifted(six, -0.01)
    assert not decide.six_set_no_regression(worse, six)["pass"], \
        "a -0.01 six-set regression passed a -0.0075 margin"
    ok = shifted(six, -0.005)
    assert decide.six_set_no_regression(ok, six)["pass"], "-0.005 should be inside the margin"


def t_point_guard_is_strict():
    """§5.5 registers `> 0.005`. Exactly 0.005 must FAIL."""
    assert decide.POINT_GUARD_C1 == 0.005
    fam = {"C1": {"delta_raw": 0.005, "resolved": True},
           "C2": {"resolved": True}, "C3": {"resolved": True}}
    v = decide.ship(fam, qualifying=decide.qualifying_table({"pool_composition"}),
                    worst={"pass": True}, six_guard={"pass": True})
    assert not v["point_guard"]["pass"], "delta exactly at the guard passed a STRICT > rule"


def t_qualifying_branches():
    q = decide.qualifying_table
    assert q({"pool_composition", "steps_a"})["pass"], "a real data-construction lever must pass"
    assert not q({"doc_side_head"})["pass"], "a doc-side head ALONE must not qualify (E11/G4-4)"
    assert not q({"seed", "steps_a", "lr"})["pass"], "ordinary tuning must not qualify"
    assert not q({"some_new_knob"})["pass"], "an unknown key must FAIL, not be ignored"
    assert q({"some_new_knob"})["unknown"] == ["some_new_knob"]
    swap = q({"teacher_id", "tokenizer_id", "vocab"})
    assert swap["teacher_swap_in_diff"] and not swap["pass"], \
        "a teacher swap alone must not satisfy the qualifying-table condition"
    both = q({"teacher_id", "tokenizer_id", "vocab", "pool_composition"})
    assert both["pass"] and both["qualifying_table_net_of_swap"] == ["pool_composition"]


def t_every_ship_condition_can_block_alone():
    """Seven conditions; each must be able to sink an otherwise-perfect result."""
    ok_fam = {k: {"resolved": True, "delta_raw": 0.02} for k in ("C1", "C2", "C3")}
    good = dict(qualifying=decide.qualifying_table({"pool_composition"}),
                worst={"pass": True}, six_guard={"pass": True})
    assert decide.ship(ok_fam, **good)["ship"], "the all-good case must ship"
    for leg in ("C1", "C2", "C3"):
        fam = {k: dict(v) for k, v in ok_fam.items()}
        fam[leg]["resolved"] = False
        assert not decide.ship(fam, **good)["ship"], f"{leg} unresolved must block"
    assert not decide.ship(ok_fam, **{**good,
                                      "qualifying": decide.qualifying_table({"seed"})})["ship"], \
        "no qualifying table change must block"
    assert not decide.ship(ok_fam, **{**good, "worst": {"pass": False}})["ship"], \
        "the worst-group guard must block"
    assert not decide.ship(ok_fam, **{**good, "six_guard": {"pass": False}})["ship"], \
        "the six-set guard must block"
    small = {k: dict(v) for k, v in ok_fam.items()}
    small["C1"]["delta_raw"] = 0.004
    assert not decide.ship(small, **good)["ship"], "the +0.005 point guard must block"


def t_holm_ordering_is_real():
    """A leg with a large p must not be rejected because its siblings were small."""
    h = boot.holm({"C1": 0.001, "C2": 0.004, "C3": 0.90}, alpha=decide.ALPHA)
    assert h["C1"]["reject"] and not h["C3"]["reject"]
    assert h["C1"]["threshold"] == decide.ALPHA / 3
    # step-down: once one fails, everything above it fails too
    h2 = boot.holm({"C1": 0.02, "C2": 0.021, "C3": 0.022}, alpha=decide.ALPHA)
    assert not any(v["reject"] for v in h2.values()), "step-down did not stop at the first failure"


def main():
    print("m8 confirmatory-rule suite (LEDGER 4 + 5)")
    print(" reductions:")
    for n in ("t_paired_dep_reduces_on_disjoint_sets", "t_exact_bonferroni_matches_the_rounded_key",
              "t_decide_family_raises_when_the_two_bounds_disagree"):
        check(n[2:], globals()[n])
    print(" refusals:")
    for n in sorted(k for k in globals()
                    if k.startswith("t_") and "reduce" not in k and "bonferroni" not in k
                    and "disagree" not in k):
        check(n[2:], globals()[n])
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES")
        return 1
    print("all confirmatory-rule checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
