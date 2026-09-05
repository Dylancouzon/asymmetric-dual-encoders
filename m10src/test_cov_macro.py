"""Tests for the COV family-macro contrast rule. A check that cannot fail is not a check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
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
    r = cm.contrast(al, UF, B=20_000, seed=0, chunk=5_000)
    w = cm.weights(UF)
    se = np.sqrt(sum(w[u] ** 2 * (al[u][1] - al[u][2]).var(ddof=1) / ns[u] for u in ns))
    assert abs(r["draws_sd"] - se) / se < 0.05, (r["draws_sd"], se)
    # the lower bound sits ~z*SE below the point estimate at the registered quantile
    z = 2.8907                                     # one-sided 0.025/13
    assert abs(r["distance_raw"] - z * se) / (z * se) < 0.10, (r["distance_raw"], z * se)
    r2 = cm.contrast(al, UF, B=20_000, seed=0, chunk=5_000)
    assert r2["draws_sha256"] == r["draws_sha256"]
    r3 = cm.contrast(al, UF, B=20_000, seed=1, chunk=5_000)
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
