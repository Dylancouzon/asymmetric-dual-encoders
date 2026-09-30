import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import common
import e5_oracle
import e6_router
import roster_ids


def test_ids_match_m20_roster():
    text = (common.REPO / "m20src" / "roster.py").read_text()
    for name in ("STELLA_REVISION", "BGE_REVISION", "LEAF_QUERY_REVISION", "ZERO_HUB_REVISION"):
        value = getattr(roster_ids, name.replace("LEAF_QUERY", "LEAF"))
        assert f'{name} = "{value}"' in text


def test_reserved_refused():
    for name in ("fever", "dbpedia-entity", "cqadup-android", "cqadup-english"):
        with pytest.raises(SystemExit):
            common.refuse_reserved(name)
    assert not set(common.EVAL12) & common.RESERVED
    assert not set(common.FIT_FORUMS) & set(common.EVAL12)


def test_frontier():
    e5_oracle.selfcheck()


def test_router_bounds():
    rng = np.random.default_rng(0)
    zero, nano = rng.random(200), rng.random(200)
    oracle_route = e6_router.evaluate(nano - zero, zero, nano, 0.0, rng)
    assert abs(oracle_route["efficiency"] - 1.0) < 1e-9
    everyone = e6_router.evaluate(np.ones(200), zero, nano, 0.0, rng)
    assert everyone["nano_fraction"] == 1.0 and abs(everyone["router_minus_random"]) < 1e-12


def test_router_interval_zero_width_for_constant_gain():
    # Nano better by a constant: routing advantage at equal share is identically zero.
    rng = np.random.default_rng(1)
    zero = rng.random(300)
    out = e6_router.evaluate(rng.random(300), zero, zero + 0.1, 0.5, rng, draws=500)
    lo, hi = out["router_minus_random_ci95"]
    assert abs(lo) < 1e-12 and abs(hi) < 1e-12


def test_e8_convergence_gate():
    sys.path.insert(0, str(common.REPO / "m8src"))
    import e8_towers
    ok = {"converged": True, "worst_rel_residual": 5e-7, "tol": 1e-6, "iterations": 3,
          "seconds": 1.0, "preconditioner": "jacobi"}
    assert e8_towers._gate(ok, "t")["iterations"] == 3
    for bad in ({**ok, "converged": False}, {**ok, "worst_rel_residual": 2e-6}):
        with pytest.raises(e8_towers.GateFailure):
            e8_towers._gate(bad, "t")
