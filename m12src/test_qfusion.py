"""Parity of qfusion's RRF/DBSF against the vendored qdrant-client reference, plus the three
degenerate cases that can actually change a ranking.

The audit's entire value is that we compare convex0 against QDRANT's operators, not against our
idea of them. A paraphrase that scores 0.57 tells us nothing, so parity is the deliverable and this
file is what makes it a fact. Run after touching m12src/qfusion.py.
"""
import random
import sys

import qfusion
import reference as ref

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def as_lists(runs, qid):
    """The prefetch lists Qdrant would have received, in qfusion's own stable order -- so a parity
    failure is a formula difference and never a tie-break difference."""
    return [[ref.P(d, s) for d, s in qfusion._ordered(r[qid])] for r in runs if qid in r]


def close(ours, theirs, tol=1e-12):
    got = {p.id: p.score for p in theirs}
    if set(got) != set(ours):
        return False, f"ids differ: {sorted(set(ours) ^ set(got))}"
    bad = {d: (ours[d], got[d]) for d in ours if abs(ours[d] - got[d]) > tol}
    return not bad, str(bad)


def main():
    rng = random.Random(7)

    # ---- parity on random overlapping runs, across the whole registered k grid ----------------
    dense = {"q": {f"d{i}": rng.random() for i in range(40)}}
    lex = {"q": {f"d{i}": rng.random() * 30 for i in rng.sample(range(60), 25)}}
    lists = as_lists([dense, lex], "q")

    for k in (1, 2, 3, 4, 6, 11, 21, 31, 61, 101):
        ok, why = close(qfusion.rrf([dense, lex], k=k)["q"],
                        ref.reciprocal_rank_fusion(as_lists([dense, lex], "q"), limit=10**6,
                                                   ranking_constant_k=k))
        check(f"rrf parity at k={k}", ok, why)

    for w in ((1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (1, 3)):
        ok, why = close(qfusion.rrf([dense, lex], k=6, weights=list(w))["q"],
                        ref.reciprocal_rank_fusion(as_lists([dense, lex], "q"), limit=10**6,
                                                   ranking_constant_k=6, weights=list(w)))
        check(f"weighted rrf parity at w={w}", ok, why)

    ok, why = close(qfusion.dbsf([dense, lex])["q"],
                    ref.distribution_based_score_fusion(as_lists([dense, lex], "q"), limit=10**6))
    check("dbsf parity", ok, why)

    # k=0 is a crash upstream, not a score. We refuse it rather than recommend it to a user.
    try:
        qfusion.rrf([dense], k=0)
        check("rrf refuses k=0", False, "no error raised")
    except ValueError:
        check("rrf refuses k=0", True)
    try:
        ref.reciprocal_rank_fusion([[ref.P("a", 1.0)]], limit=1, ranking_constant_k=0)
        check("reference crashes at k=0", False, "upstream returned a score")
    except ZeroDivisionError:
        check("reference crashes at k=0", True)   # why we refuse it rather than pass it through

    # ---- degenerate case 1: singleton and constant lists normalise to 0.5, not to 1 -----------
    single = {"q": {"only": 12.3}}
    flat = {"q": {"a": 2.0, "b": 2.0, "c": 2.0}}
    check("dbsf singleton -> 0.5", qfusion.dbsf([single])["q"] == {"only": 0.5})
    check("dbsf constant list -> 0.5", set(qfusion.dbsf([flat])["q"].values()) == {0.5})
    ok, why = close(qfusion.dbsf([single, flat])["q"],
                    ref.distribution_based_score_fusion(as_lists([single, flat], "q"), limit=10**6))
    check("dbsf degenerate parity", ok, why)

    # ---- degenerate case 2: a doc absent from a prefetch contributes 0 ------------------------
    # Registered, and NOT the bottom of DBSF's range. `lo` sits below every present score, so a
    # present-but-weak doc can be normalised BELOW an absent one. If this ever stops being true,
    # the operator has drifted from Qdrant and every M12 number is about something else.
    a = {"q": {"x": 10.0, "y": 9.9, "z": 0.0}}
    b = {"q": {"x": 5.0}}
    fused = qfusion.dbsf([a, b])["q"]
    check("absent doc contributes exactly 0", abs(fused["y"] - qfusion.dbsf([a])["q"]["y"]) < 1e-12)
    check("dbsf missing-doc parity", *close(fused,
          ref.distribution_based_score_fusion(as_lists([a, b], "q"), limit=10**6)))

    # ---- degenerate case 3: a score below mu-3sigma normalises NEGATIVE ------------------------
    # so it ranks below a document that was never retrieved at all. Qdrant's behaviour; the reason
    # it is a test is that it is the one result a reader will assume is a bug.
    outlier = {"q": {f"d{i}": 1.0 for i in range(30)} | {"low": -50.0}}
    n = qfusion.dbsf([outlier])["q"]
    check("dbsf can emit a negative score (no clamping)", n["low"] < 0, f"{n['low']:.4f}")
    check("a below-3-sigma doc ranks under an absent one", n["low"] < 0.0)
    check("dbsf negative-score parity", *close(
        n, ref.distribution_based_score_fusion(as_lists([outlier], "q"), limit=10**6)))

    # ---- our k is not Qdrant's k -------------------------------------------------------------
    # The whole reason M7's sweep never reached the optimum. k_qdrant = k_ours + 1 at w=1.
    import fusion as m7fusion
    ours = m7fusion.rrf([dense, lex], k=59)["q"]
    theirs = qfusion.rrf([dense, lex], k=60)["q"]
    check("k_qdrant == k_ours + 1", set(ours) == set(theirs)
          and max(abs(ours[d] - theirs[d]) for d in ours) < 1e-12)
    check("and they differ at equal k", any(
        abs(ours[d] - qfusion.rrf([dense, lex], k=59)["q"][d]) > 1e-9 for d in ours))

    # ---- truncation is what a shorter prefetch would have returned ---------------------------
    t = qfusion.truncate(dense, 10)
    check("truncate keeps the top-d of the same stable order",
          list(t["q"]) == [d for d, _ in qfusion._ordered(dense["q"])[:10]])
    check("truncating below list length is a no-op", qfusion.truncate(b, 1000) == b)

    print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
