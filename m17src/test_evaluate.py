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


def test_document_blocking_is_exact_and_tie_broken_by_document_id():
    """A 4096 x 5.2M score matrix is 85 GB, so the corpus is blocked; blocking must not change
    the answer, including where scores tie."""
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(37, 6)).astype(np.float32)
    docs[5] = docs[11] = docs[23] = docs[2]          # deliberate exact ties across blocks
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = docs[[2, 30]] * 1.0
    ids = [f"doc{i:03d}" for i in range(37)]
    ref = E.search(q, docs, k=5, doc_ids=ids, doc_block=10_000)
    for db in (1, 3, 8, 37):
        got = E.search(q, docs, k=5, doc_ids=ids, doc_block=db)
        assert list(got[0]) == list(ref[0]) and list(got[1]) == list(ref[1])
    brute = []
    for r in range(2):
        scored = sorted(((float(q[r] @ docs[j]), ids[j]) for j in range(37)),
                        key=lambda t: (-t[0], t[1]))[:5]
        brute.append([d for _, d in scored])
    assert [list(ref[0]), list(ref[1])] == brute


def test_a_tie_wider_than_k_is_still_exact_and_block_independent():
    """`argpartition` used to discard all but an arbitrary k inside each block, so a tie of
    more than k documents lost smaller document ids before the ascending-id sort."""
    docs = np.zeros((40, 4), dtype=np.float32)
    docs[:, 0] = 1.0                                  # all 40 documents score identically
    q = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ids = [f"doc{i:03d}" for i in range(40)]
    want = ids[:5]                                    # the five smallest ids win the tie
    for db in (1, 3, 7, 40, 10_000):
        got = E.search(q, docs, k=5, doc_ids=ids, doc_block=db)
        assert list(got[0]) == want, f"doc_block={db} changed the answer"
    # a zero query: every score is 0.0, the tie spans the whole corpus
    z = np.zeros((1, 4), dtype=np.float32)
    assert list(E.search(z, docs, k=5, doc_ids=ids, doc_block=6)[0]) == want


def test_search_refuses_duplicate_document_ids():
    docs = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="doc_ids contains duplicates"):
        E.search(docs[:1], docs, k=2, doc_ids=["a", "a", "b"])


def test_comparison_surfaces_must_have_identical_keys():
    a, b = {0: 1.0, 1: 0.5}, {0: 0.9}
    with pytest.raises(ValueError, match="baseline keys do not match"):
        E.paired_family_bootstrap(a, b, {}, replicates=10)
    with pytest.raises(ValueError, match="alias views do not cover the same pairs"):
        E.alias_test({0: {"d": 1.0}, 1: {"d": 1.0}}, {0: {"d": 1.0}})


def test_bm25_run_keys_must_match_the_dense_run(reg):
    docs = np.eye(4, dtype=np.float32)
    q = docs[:2]
    ids = list("abcd")
    with pytest.raises(ValueError, match="bm25_run keys do not match"):
        E.evaluate(q, docs, {"q0": {"a": 1}, "q1": {"b": 1}},
                   {"q0": "general", "q1": "general"}, doc_ids=ids,
                   bm25_run={0: {"a": 1.0}, 1: {"b": 1.0}}, reg=reg,
                   query_ids=["q0", "q1"])


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


def test_the_bootstrap_estimates_the_domain_macro_not_the_query_mean():
    """100 queries improving by 1.0 in one domain against one worsening by 1.0 in another: the
    macro difference is 0, the query mean is 0.98. The interval must be about the macro."""
    a = {i: 1.0 for i in range(100)}
    a[100] = 0.0
    b = {i: 0.0 for i in range(100)}
    b[100] = 1.0
    fams = {i: f"f{i}" for i in range(101)}
    doms = {i: ("A" if i < 100 else "B") for i in range(101)}
    out = E.paired_family_bootstrap(a, b, fams, replicates=500, seed=0, domains=doms)
    assert out["delta"] == pytest.approx(0.0)
    assert out["delta_query_mean"] == pytest.approx(99 / 101, abs=1e-6)
    assert out["statistic"] == "equal-weight domain macro"
    assert out["ci"][0] < 0.5 < out["ci"][1] or out["ci"][1] <= 0.5
    plain = E.paired_family_bootstrap(a, b, fams, replicates=500, seed=0)
    assert plain["delta"] == pytest.approx(99 / 101, abs=1e-6)
    assert plain["statistic"] == "query mean"


def test_evaluate_accepts_panel_query_ids_and_refuses_a_key_mismatch():
    docs = np.eye(4, dtype=np.float32)
    qids = [f"s:{i}" for i in range(4)]
    doc_ids = [f"d{i}" for i in range(4)]
    qrels = {q: {f"d{i}": 1} for i, q in enumerate(qids)}
    domains = {q: ("general" if i % 2 else "medicine") for i, q in enumerate(qids)}
    rep = E.evaluate(docs, docs, qrels, domains, doc_ids=doc_ids, query_ids=qids)
    assert rep["ndcg@10"]["macro"] == pytest.approx(1.0)      # perfect retrieval
    assert rep["ndcg@10"]["n_per_domain"] == {"general": 2, "medicine": 2}
    with pytest.raises(ValueError, match="qrels keys"):
        E.evaluate(docs, docs, {0: {"d0": 1}}, domains, doc_ids=doc_ids, query_ids=qids)


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
