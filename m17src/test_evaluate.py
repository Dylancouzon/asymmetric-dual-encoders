"""Retrieval metrics, the family bootstrap, DBSF@100 and the surface gate.

The surface gate is the one that matters operationally: M17 has no protected access, and the
development suite and the M17 panel are registered reads for the execution session, not a
development convenience.
"""
from __future__ import annotations

import numpy as np
import pytest

import evaluate as E


def test_exact_search_ranks_by_inner_product():
    docs = np.eye(4, dtype=np.float32)
    q = np.asarray([[0.9, 0.4, 0.0, 0.0]], dtype=np.float32)
    run = E.search(q / np.linalg.norm(q), docs, k=2, doc_ids=list("abcd"))
    assert list(run[0]) == ["a", "b"]


def test_ndcg_and_recall():
    run = {0: {"a": 3.0, "b": 2.0, "c": 1.0}}
    assert E.ndcg_at_k(run, {0: {"a": 1}}, k=10)[0] == pytest.approx(1.0)
    assert E.ndcg_at_k(run, {0: {"c": 1}}, k=10)[0] == pytest.approx(0.5)
    assert E.recall_at_k(run, {0: {"a": 1, "z": 1}}, k=10) == pytest.approx({0: 0.5})
    assert E.ndcg_at_k(run, {0: {}}, k=10) == {0: 0.0}


def test_per_domain_macro_is_over_domains_not_queries():
    vals = {0: 1.0, 1: 1.0, 2: 0.0}
    doms = {0: "a", 1: "a", 2: "b"}
    out = E.per_domain(vals, doms)
    assert out["per_domain"] == {"a": 1.0, "b": 0.0} and out["macro"] == pytest.approx(0.5)
    assert out["n_per_domain"] == {"a": 2, "b": 1}


def test_family_bootstrap_resamples_families_not_queries():
    """Twelve queries in two families must give a far wider interval than twelve families."""
    a = {i: (1.0 if i < 6 else 0.0) for i in range(12)}
    b = {i: 0.0 for i in range(12)}
    tied = E.paired_family_bootstrap(a, b, {i: ("f0" if i < 6 else "f1") for i in range(12)},
                                     replicates=2000, seed=0)
    loose = E.paired_family_bootstrap(a, b, {i: f"f{i}" for i in range(12)},
                                      replicates=2000, seed=0)
    assert tied["n_families"] == 2 and loose["n_families"] == 12
    assert (tied["ci"][1] - tied["ci"][0]) > (loose["ci"][1] - loose["ci"][0])
    assert tied["delta"] == pytest.approx(0.5)


def test_dbsf_uses_the_m12_operator():
    import sys
    from common import REPO
    sys.path.insert(0, str(REPO / "m12src"))
    import qfusion
    dense = {0: {f"d{i}": float(100 - i) for i in range(120)}}
    bm25 = {0: {f"d{i}": float(i) for i in range(120)}}
    got = E.dbsf_at(dense, bm25, prefetch=100)
    want = qfusion.dbsf([qfusion.truncate(dense, 100), qfusion.truncate(bm25, 100)])
    assert got == want


def test_alias_test_reports_overlap_and_rank_correlation():
    a = {0: {f"d{i}": float(10 - i) for i in range(10)}}
    b = {0: {f"d{i}": float(10 - i) for i in range(10)}}
    same = E.alias_test(a, b)
    assert same["top10_overlap_mean"] == 1.0
    assert same["rank_correlation_mean"] == pytest.approx(1.0)
    c = {0: {f"x{i}": float(10 - i) for i in range(10)}}
    assert E.alias_test(a, c)["top10_overlap_mean"] == 0.0
    assert "descriptive" in same["_role"]


def test_protected_paths_are_refused_by_name():
    for bad in ("results/frozen_eval/untouched-final", "work/m9reserve/x", "data/lotte/pooled"):
        with pytest.raises(SystemExit, match="protected surface"):
            E._check_path(bad)
    E._check_path("work/m17/rehearsal/fixtures")


def test_dev_suite_and_panel_need_their_flag_and_the_lock():
    with pytest.raises(SystemExit, match="needs its explicit"):
        E.main(["--fixtures", "x", "--surface", "dev-suite"])
    with pytest.raises(SystemExit, match="registry status"):
        E.main(["--fixtures", "x", "--surface", "panel", "--allow-panel"])


def test_rehearsal_evaluation_stayed_synthetic(rehearsal):
    ev = rehearsal["record"]["stages"]["evaluate"]
    assert ev["surface"] == "synthetic fixtures only"
    assert set(ev["ndcg@10"]["per_domain"]) <= {"general", "cloud-software",
                                                "science-engineering"}
    assert ev["alias_test"]["n_pairs"] > 0
