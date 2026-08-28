"""Type-behaviour checks for the dependence-preserving statistics (Codex review #3 BLOCKER 1).

Three properties, because the failure mode being repaired is silent: a dependence-blind interval
is not obviously wrong when you look at it, it is just too narrow.

1. REDUCTION -- with no query shared across components, the stratified machinery must reproduce
   the ordinary one (up to Monte-Carlo error). If it did not, adopting it would move every
   number for reasons unrelated to dependence.
2. FULL DUPLICATION -- if component B is a verbatim copy of component A, the macro IS A's mean,
   so the dependence-preserving interval must match a single-component bootstrap, and the
   ordinary one must be ~sqrt(2) too narrow. This is the bug, in its purest form.
3. DETERMINISM -- fixed seed, fixed answer.

Run: .venv/bin/python m7src/test_dep_stats.py
"""
import numpy as np

import boot


def _mk(n, delta, sd, seed, prefix="q"):
    rng = np.random.default_rng(seed)
    b = rng.random(n)
    a = b + delta + rng.normal(0, sd, n)
    ids = [f"{prefix}{i}" for i in range(n)]
    return dict(zip(ids, a)), dict(zip(ids, b))


def test_reduction_without_sharing():
    a1, b1 = _mk(400, 0.01, 0.05, 1, "x")
    a2, b2 = _mk(300, 0.02, 0.05, 2, "y")
    A = {"ds1": a1, "ds2": a2}
    B = {"ds1": b1, "ds2": b2}
    o, d = boot.paired(A, B), boot.paired_dep(A, B)
    assert o["delta"] == d["delta"], (o["delta"], d["delta"])
    wo = o["ci95"][1] - o["ci95"][0]
    wd = d["ci95"][1] - d["ci95"][0]
    assert abs(wo - wd) < 0.05 * wo, (wo, wd)
    so, sd_ = boot.signflip(A, B), boot.signflip_dep(A, B)
    assert abs(so["p"] - sd_["p"]) < 0.01, (so["p"], sd_["p"])
    assert sd_["shared_units"] == 0
    print(f"1. reduction: ci width ordinary {wo:.5f} vs dep {wd:.5f}; "
          f"p {so['p']:.4f} vs {sd_['p']:.4f}  OK")


def test_full_duplication():
    """dsA and dsB hold the SAME queries -- exactly heldout-longq's relation to heldout-train,
    taken to its limit. unit_key would not tie them (neither name starts with 'heldout-'), so the
    unit function is passed explicitly: the tie is a property of the data, not of the name."""
    a1, b1 = _mk(200, 0.01, 0.05, 3, "z")
    A = {"dsA": a1, "dsB": dict(a1)}
    B = {"dsA": b1, "dsB": dict(b1)}
    one = boot.paired({"dsA": a1}, {"dsA": b1})
    o = boot.paired(A, B)
    d = boot.paired_dep(A, B, unit_of=lambda ds, q: q)
    w1 = one["ci95"][1] - one["ci95"][0]
    wo = o["ci95"][1] - o["ci95"][0]
    wd = d["ci95"][1] - d["ci95"][0]
    assert abs(wd - w1) < 0.05 * w1, (wd, w1)
    assert wo < 0.85 * wd, (wo, wd)           # the dependence-blind interval is ~1/sqrt(2) too narrow
    sd_ = boot.signflip_dep(A, B, unit_of=lambda ds, q: q)
    so = boot.signflip(A, B)
    assert sd_["shared_units"] == 200 and len(sd_["strata"]) == 1
    assert sd_["p"] > so["p"], (sd_["p"], so["p"])
    print(f"1-component ci {w1:.5f} | dep {wd:.5f} (matches) | ordinary {wo:.5f} "
          f"(too narrow by {wd/wo:.2f}x); p ordinary {so['p']:.5f} < dep {sd_['p']:.5f}  OK")


def test_determinism():
    a1, b1 = _mk(150, 0.01, 0.05, 4)
    A, B = {"d": a1}, {"d": b1}
    assert boot.paired_dep(A, B) == boot.paired_dep(A, B)
    assert boot.signflip_dep(A, B, R=5000) == boot.signflip_dep(A, B, R=5000)
    print("3. determinism  OK")


def test_realistic_nesting():
    """The actual shape: 6 components, one of them 55 queries drawn from another's 7,325."""
    comps = {"nq-250k": 3452, "hotpotqa": 7405, "cqadup-programmers": 876,
             "cqadup-physics": 1039, "heldout-train": 7325}
    A, B = {}, {}
    for i, (c, n) in enumerate(comps.items()):
        A[c], B[c] = _mk(n, 0.002, 0.2, 10 + i, f"{c}:")
    long_ids = sorted(A["heldout-train"])[:55]
    A["heldout-longq"] = {q: A["heldout-train"][q] for q in long_ids}
    B["heldout-longq"] = {q: B["heldout-train"][q] for q in long_ids}
    o, d = boot.paired(A, B), boot.paired_dep(A, B)
    so, sd_ = boot.signflip(A, B), boot.signflip_dep(A, B)
    assert sd_["shared_units"] == 55, sd_["shared_units"]
    assert {tuple(s["components"]) for s in sd_["strata"]} >= {("heldout-longq", "heldout-train")}
    print(f"4. realistic nesting: delta {o['delta']} | ci ordinary {o['ci95']} dep {d['ci95']} | "
          f"p ordinary {so['p']:.5f} dep {sd_['p']:.5f}  OK")


if __name__ == "__main__":
    test_reduction_without_sharing()
    test_full_duplication()
    test_determinism()
    test_realistic_nesting()
    print("dep-stats checks passed")
