"""Tests for the COV family-macro contrast rule. A check that cannot fail is not a check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
import cov_macro as cm

UF = dict(cm.SURFACE)          # the tests run on the real admitted surface, never a synthetic one
BR0 = "BRIGHT/biology"


def test_weights():
    w = cm.weights(UF)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert abs(w["MedicalQARetrieval"] - 0.25) < 1e-12
    assert abs(w["LegalBenchCorporateLobbying"] - 0.125) < 1e-12
    assert abs(w[BR0] - 0.25 / 6) < 1e-12
    # A family or unit absent from the scored set must not silently reweight the macro: a
    # 3-family macro, or a 5-slice BRIGHT, would otherwise report as a clean number
    # (Codex pass 2026-09-05). Every one of these must be refused.
    for bad in ({"x": "not-a-family"},
                {u: f for u, f in UF.items() if f != "finance"},          # a family dropped
                {u: f for u, f in UF.items() if u != BR0},                # one slice dropped
                {**UF, "BRIGHT/stackoverflow": "BRIGHT"},                 # an unadmitted slice
                {**UF, BR0: "legal"}):                                    # a unit relabelled
        try:
            cm.weights(bad)
            raise AssertionError(f"accepted a surface that is not the lock: {sorted(bad)[:3]}")
        except ValueError:
            pass


def test_macro_is_family_weighted_not_query_weighted():
    """The point of the estimator: 10,000 LEDGER queries do not outvote 340 legal ones."""
    per = {u: {f"q{i}": (1.0 if f == "finance" else 0.0) for i in range(50)}
           for u, f in UF.items()}
    m, fm, _ = cm.macro(per, UF)
    assert abs(m - 0.25) < 1e-12 and fm["finance"] == 1.0


def test_align_refuses_ragged():
    a = {u: {"q1": 1.0} for u in UF}
    for bad in ({**a, "MedicalQARetrieval": {"q2": 1.0}},                 # a qid moved
                {u: v for u, v in a.items() if u != "LEDGER"},            # a unit dropped
                {**a, "MedicalQARetrieval": {}}):                         # a unit emptied
        try:
            cm.align(a, bad, UF)
            raise AssertionError("accepted a mismatched pairing")
        except ValueError:
            pass


def test_bootstrap_matches_analytic_se_and_is_reproducible():
    rng = np.random.default_rng(7)
    ns = {"MedicalQARetrieval": 2048, "LEDGER": 10000,
          "LegalBenchCorporateLobbying": 340, "LegalBenchConsumerContractsQA": 396,
          **{u: 100 for u, f in UF.items() if f == "BRIGHT"}}
    a, b = {}, {}
    for u, n in ns.items():
        x = rng.normal(0.5, 0.2, n)
        a[u] = {f"q{i}": float(v) for i, v in enumerate(x)}
        b[u] = {f"q{i}": float(v) for i, v in enumerate(x + rng.normal(0.0, 0.15, n))}
    al = cm.align(a, b, UF)
    r = cm.contrast(al, UF, B=20_000, seed=0, chunk=5_000, quantile=cm.ONE_SIDED)
    w = cm.weights(UF)
    se = np.sqrt(sum(w[u] ** 2 * (al[u][1] - al[u][2]).var(ddof=1) / ns[u] for u in ns))
    assert abs(r["draws_sd"] - se) / se < 0.05, (r["draws_sd"], se)
    # the lower bound sits ~z*SE below the point estimate at the registered quantile
    z = 2.8907                                     # one-sided 0.025/13
    assert abs(r["distance_raw"] - z * se) / (z * se) < 0.10, (r["distance_raw"], z * se)
    r2 = cm.contrast(al, UF, B=20_000, seed=0, chunk=5_000, quantile=cm.ONE_SIDED)
    assert r2["draws_sha256"] == r["draws_sha256"]
    r3 = cm.contrast(al, UF, B=20_000, seed=1, chunk=5_000, quantile=cm.ONE_SIDED)
    assert r3["draws_sha256"] != r["draws_sha256"]


def test_quantile_is_the_registered_order_statistic():
    x = np.arange(200_000, dtype=np.float64)
    q = float(np.quantile(x, 0.025 / 13, method="inverted_cdf"))
    assert q == 384.0, q                           # 0-based index 384 = the 385th order statistic
    assert float(np.quantile(x, 0.025 / 13)) > q   # numpy's default is strictly more permissive


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)


def test_score_student_asserts_the_surface_before_it_scores_anything():
    """finding 12: `macro()` checked the surface, but only after the whole encode was spent."""
    import cov_eval10

    scored = []

    def encode(texts):
        scored.append(texts)
        raise AssertionError("no unit may be scored on a surface that does not match the lock")

    # one admitted unit, so the surface is INCOMPLETE: units are (uid, family, qs, qids, ds,
    # dids, qrels)
    units = [("LEDGER", "finance", ["q"], ["q1"], ["d"], ["d1"], {"q1": {"d1": 1}})]
    with pytest.raises(ValueError, match="COV surface does not match"):
        cov_eval10.score_student(encode, units=units, verbose=False)
    assert scored == [], "the refusal must precede the encode"


def test_contrast_REFUSES_a_silent_quantile_and_the_constants_match_amendment_C2():
    """The quantile decides 11 registered contrasts, one of which (`A4-A3`) controls whether the
    ~1.0M generated queries enter the build. It used to default silently to 0.025/13 -- superseded
    by amendment C2 -- and 0.025/13 sits FURTHER into the tail than ALPHA/12, so the stale value
    was strictly HARDER to resolve and biased toward discarding data. No silent default now."""
    rng = np.random.default_rng(0)
    a = {u: {f"q{i}": float(v) for i, v in enumerate(rng.normal(0.5, 0.2, 40))} for u in UF}
    b = {u: {f"q{i}": float(v) for i, v in enumerate(rng.normal(0.5, 0.2, 40))} for u in UF}
    al = cm.align(a, b, UF)
    with pytest.raises(TypeError, match="quantile` is required"):
        cm.contrast(al, UF, B=1_000, seed=0)
    # C2: 11 one-sided at ALPHA/12 + F1 two-sided at ALPHA/24 per tail = ALPHA exactly
    assert 11 * cm.ONE_SIDED + 2 * cm.F_PER_TAIL == pytest.approx(cm.ALPHA)
    # and the stale value does NOT close, which is why it is superseded
    assert 11 * (cm.ALPHA / 13) + 2 * (cm.ALPHA / 26) != pytest.approx(cm.ALPHA)
    assert cm.HISTORICAL_13 < cm.ONE_SIDED, "the stale value is stricter, hence false-negative-prone"
