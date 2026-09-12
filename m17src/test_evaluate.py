"""Retrieval metrics, the family bootstrap, DBSF@100 and the surface gate.

The surface gate is the one that matters operationally: M17 has no protected access, and the
development suite and the M17 panel are registered reads for the execution session, not a
development convenience.
"""
from __future__ import annotations

import json
from pathlib import Path

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


def test_dev_suite_and_panel_need_their_flag_and_the_lock(monkeypatch):
    with pytest.raises(SystemExit, match="needs its explicit"):
        E.main(["--fixtures", "x", "--surface", "dev-suite"])
    # the status gate, on a draft copy: the real registry reached EXECUTABLE at step 6a
    draft = dict(E.registry(), status="DRAFT_NOT_EXECUTABLE")
    monkeypatch.setattr(E, "registry", lambda *a, **k: draft)
    with pytest.raises(SystemExit, match="registry status"):
        E.main(["--fixtures", "x", "--surface", "panel", "--allow-panel"])


@pytest.fixture
def synthetic_dev_suite(tmp_path, monkeypatch):
    """A two-component pinned suite in tmp_path: one text-backed, one 'full-pool' slice.

    The real components are never opened: `devsuite.CACHE` and `heldout.HELD` are redirected, and
    the manifest the reader is pointed at is this fixture's own.
    """
    import devsuite
    import heldout
    from hashing import sha, sha_stream_list
    monkeypatch.setattr(devsuite, "CACHE", tmp_path)
    monkeypatch.setattr(heldout, "HELD", tmp_path)
    text = {"doc_ids": [f"d{i}" for i in range(5)],
            "doc_texts": [f"document {i}" for i in range(5)],
            "q_ids": ["q0", "q1"], "q_texts": ["first query", "second query"],
            "qrels": {"q0": {"d0": 1}, "q1": {"d3": 1}}}
    held = {"corpus": "full-pool", "n_docs": 7, "q_ids": ["s:1", "s:2"],
            "q_texts": ["held out one", "held out two"],
            "qrels": {"s:1": {"4": 1}, "s:2": {"0": 1}}, "n_tokens": [3, 3], "by_source": {"s": 2}}
    (tmp_path / "syn-text.json").write_text(json.dumps(text))
    held_p = tmp_path / "syn-held.json"
    held_p.write_text(json.dumps(held))
    man = {
        "syn-text": {"n_docs": 5, "n_queries": 2,
                     "corpus_ids_sha256": sha(text["doc_ids"]),
                     "corpus_text_sha256": sha(text["doc_texts"]),
                     "qids_sha256": sha(sorted(text["q_ids"])),
                     "qrels_sha256": sha(text["qrels"])},
        "syn-held": {"corpus": "full-pool", "n_docs": 7, "n_queries": 2,
                     "qids_ordered_sha256": sha_stream_list(held["q_ids"]),
                     "qids_sha256": sha(sorted(held["q_ids"])),
                     "qtexts_ordered_sha256": sha_stream_list(held["q_texts"]),
                     "qrels_sha256": sha(held["qrels"]),
                     "json_sha256": E.sha_file(held_p)},
        "_pinned": {"components": ["syn-text", "syn-held"],
                    "macro": "equal weight per component"},
    }
    p = tmp_path / "m7_dev_manifest.json"
    p.write_text(json.dumps(man))
    return {"manifest_path": p, "dir": tmp_path, "man": man, "text": text, "held": held}


def test_dev_suite_reader_needs_its_opt_in_flag(synthetic_dev_suite):
    """The gate is on the reader itself, not only on the CLI."""
    with pytest.raises(SystemExit, match="needs its explicit --allow-dev-suite"):
        E.load_dev_component("syn-text", manifest_path=synthetic_dev_suite["manifest_path"])
    with pytest.raises(SystemExit, match="needs its explicit --allow-dev-suite"):
        E.dev_suite_read("some/bundle")
    with pytest.raises(SystemExit, match="needs its explicit --allow-dev-suite"):
        E._dev_suite_read_fixture("some/bundle",
                                  manifest_path=synthetic_dev_suite["manifest_path"])


def test_dev_suite_reader_serves_the_pinned_components(synthetic_dev_suite):
    mp = synthetic_dev_suite["manifest_path"]
    assert E.dev_components(path=mp) == ["syn-text", "syn-held"]
    txt = E.load_dev_component("syn-text", allow_dev_suite=True, manifest_path=mp)
    assert (txt["n_docs"], txt["n_queries"]) == (5, 2)
    assert txt["doc_texts"] == synthetic_dev_suite["text"]["doc_texts"]
    assert set(txt["qrels"]) <= set(txt["q_ids"])
    held = E.load_dev_component("syn-held", allow_dev_suite=True, manifest_path=mp)
    assert (held["n_docs"], held["n_queries"]) == (7, 2)
    assert held["corpus"] == "full-pool" and held["doc_texts"] is None
    assert held["doc_ids"] == [str(i) for i in range(7)]     # pool row indices, as the qrels are
    assert {d for v in held["qrels"].values() for d in v} <= set(held["doc_ids"])
    assert txt["doc_vecs"] is None and held["doc_vecs"] is None    # not requested, not loaded


def test_dev_suite_reader_refuses_a_component_that_changed_under_it(synthetic_dev_suite):
    mp = synthetic_dev_suite["manifest_path"]
    for name, field in (("syn-text", "corpus_text_sha256"), ("syn-held", "json_sha256")):
        man = json.loads(mp.read_text())
        man[name][field] = "0" * 64
        with pytest.raises(SystemExit, match=f"pinned dev component {name} does not match"):
            E.load_dev_component(name, allow_dev_suite=True, manifest=man)
    # ... and the hashes are what makes it pass, so an unverified load still works
    assert E.load_dev_component("syn-text", allow_dev_suite=True, manifest_path=mp,
                                verify=False)["n_queries"] == 2


def test_dev_suite_may_not_shrink_silently(synthetic_dev_suite):
    mp = synthetic_dev_suite["manifest_path"]
    (synthetic_dev_suite["dir"] / "syn-held.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        E.load_dev_component("syn-held", allow_dev_suite=True, manifest_path=mp)
    with pytest.raises(SystemExit, match="not a pinned dev component"):
        E.load_dev_component("cqadup-android", allow_dev_suite=True, manifest_path=mp)
    unpinned = {k: v for k, v in json.loads(mp.read_text()).items() if k != "_pinned"}
    with pytest.raises(SystemExit, match="no _pinned.components"):
        E.dev_components(manifest=unpinned)


@pytest.fixture
def v0_world(tmp_path, monkeypatch, rehearsal):
    """One tiny suite read end to end: the rehearsal's REAL exported bundle through `loader_np`,
    the M7 scorer, and a two-query text component whose documents that same bundle encodes.

    Nothing real is opened: `devsuite.CACHE` and `teacher.ENC` are redirected and the manifest is
    this fixture's own, so the documents come from a TINY REAL encode cache built here — the
    production `_teacher_doc_vecs` path, verification included — and the frozen pool is never
    touched. The corpus carries an exact duplicate pair (a tie) and one query whose id IS a
    document id (the self-hit the M7 scorer drops).
    """
    import devsuite
    import export
    import teacher
    import torch
    from hashing import sha
    from loader_np import M17QueryEncoder
    bundle = Path(rehearsal["root"]) / "bundle-endpoint"
    digests = export.gate_artifact(bundle)
    reg = json.loads(json.dumps(E.registry()))
    reg["status"] = "LOCKED_EXECUTABLE"
    reg.setdefault("lock", {})["executed"] = {
        "v0_export": {k: digests[k] for k in E.V0_EXPORT_DIGESTS}}
    monkeypatch.setattr(devsuite, "CACHE", tmp_path)
    comp = {"doc_ids": ["d0", "d1", "d2", "d3", "d4"],
            "doc_texts": ["storage bucket", "cluster ingress", "cluster ingress",
                          "vector index", "network service"],       # d1 == d2: an exact tie
            "q_ids": ["d0", "q1"],                                  # "d0" is a self-hit
            "q_texts": ["storage bucket", "vector index"],
            "qrels": {"d0": {"d0": 1}, "q1": {"d3": 1}}}
    (tmp_path / "syn-a.json").write_text(json.dumps(comp))
    man = {"syn-a": {"n_docs": 5, "n_queries": 2,
                     "corpus_ids_sha256": sha(comp["doc_ids"]),
                     "corpus_text_sha256": sha(comp["doc_texts"]),
                     "qids_sha256": sha(sorted(comp["q_ids"])),
                     "qrels_sha256": sha(comp["qrels"])},
           "_pinned": {"components": ["syn-a"], "macro": "equal weight per component"}}
    mp = tmp_path / "man.json"
    mp.write_text(json.dumps(man))

    # the document vectors live in a real (tiny) teacher encode cache, hashes recorded
    monkeypatch.setattr(teacher, "ENC", tmp_path / "enc")
    doc_vecs = np.asarray(M17QueryEncoder(bundle, variant="int8", mode="resident_int8")
                          .encode(comp["doc_texts"]), dtype=np.float16)
    cache = write_teacher_cache(tmp_path / "enc", "dev-syn-a-docs", comp["doc_texts"], doc_vecs)
    return {"bundle": bundle, "reg": reg, "manifest_path": mp, "man": man, "dir": tmp_path,
            "digests": digests, "cache": cache, "doc_vecs": doc_vecs, "comp": comp}


def write_teacher_cache(enc_dir, name, texts, vecs, tofu=False, corrupt=False):
    """A one-shard M7 teacher encode cache with its hashes recorded, as `encode_cached` writes
    it. `tofu` marks the shard trust-on-first-use; `corrupt` changes the bytes after hashing."""
    import teacher
    import torch
    key, blob = teacher.cache_key(name, "", 512, teacher.TEACHER, teacher.TEACHER_REV,
                                  teacher.sha_texts(texts), torch.float16)
    d = Path(enc_dir) / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(blob)
    shard = d / "shard_00000.npy"
    np.save(shard, np.asarray(vecs, dtype=np.float16))
    rec = {"bytes": shard.stat().st_size, "sha256": teacher.sha_file(shard), "rows": len(vecs),
           "shard_size": teacher.SHARD, "trusted_on_first_use": bool(tofu)}
    if corrupt:
        flipped = np.load(shard)
        flipped[0] = -flipped[0]
        np.save(shard, flipped)
    (d / "shards.json").write_text(json.dumps({"shards": {"00000": rec}}))
    return d


def test_dev_suite_read_scores_through_the_loader_and_the_m7_scorer(v0_world, tmp_path):
    out = tmp_path / "read.json"
    rep = E._dev_suite_read_fixture(v0_world["bundle"], out, allow_dev_suite=True,
                                    reg=v0_world["reg"],
                                    manifest_path=v0_world["manifest_path"])
    # "q1" retrieves its own text exactly; "d0"'s only judged document is its own self-hit,
    # which `evalkit.run_from_arrays` drops — so it is unreachable, not rank 1.
    assert rep["per_query_ndcg@10"]["syn-a"] == pytest.approx({"d0": 0.0, "q1": 1.0})
    assert rep["ndcg@10"]["macro"] == pytest.approx(0.5)
    assert rep["recall@10"]["per_component"]["syn-a"] == pytest.approx(0.5)
    assert rep["recall@10"]["macro"] == pytest.approx(0.5)
    assert rep["reads"] == 1 and rep["bundle_digests"] == {k: v0_world["digests"][k]
                                                           for k in E.V0_EXPORT_DIGESTS}
    q = rep["component_identities"]["syn-a"]["queries"]
    assert len(q["query_pairs_sha256"]) == 64 and q["pinned_fields_verified"] == []
    assert json.loads(out.read_text())["ndcg@10"]["macro"] == pytest.approx(0.5)
    rc = json.loads(E._receipt_path(out).read_text())
    assert rc["state"] == "complete" and rc["reads"] == 1
    assert rc["completed_components"] == ["syn-a"] and rc["git_sha"]
    assert rc["bundle_digests"] == rep["bundle_digests"] and rc["started_utc"].endswith("Z")
    assert rc["ndcg_at_10"]["macro"] == pytest.approx(0.5)
    assert rc["metrics"]["syn-a"]["recall@10"] == pytest.approx({"d0": 0.0, "q1": 1.0})
    # the single-read boundary: no overwrite, no --force
    with pytest.raises(SystemExit, match="already exists"):
        E._dev_suite_read_fixture(v0_world["bundle"], out, allow_dev_suite=True,
                                  reg=v0_world["reg"],
                                  manifest_path=v0_world["manifest_path"])


def test_dev_suite_read_needs_an_output_path(v0_world):
    with pytest.raises(SystemExit, match="needs its output path"):
        E._dev_suite_read_fixture(v0_world["bundle"], allow_dev_suite=True, reg=v0_world["reg"],
                                  manifest_path=v0_world["manifest_path"])


def test_dev_suite_read_needs_the_executed_lock_half(v0_world, tmp_path):
    """`{"status": "EXECUTABLE"}` used to pass the gate."""
    pre = json.loads(json.dumps(v0_world["reg"]))
    pre["status"] = "EXECUTABLE"
    with pytest.raises(SystemExit, match="needs 'LOCKED_EXECUTABLE'"):
        E._dev_suite_read_fixture(v0_world["bundle"], tmp_path / "a.json", allow_dev_suite=True,
                                  reg=pre, manifest_path=v0_world["manifest_path"])
    no_exec = json.loads(json.dumps(v0_world["reg"]))
    no_exec["lock"].pop("executed")
    with pytest.raises(SystemExit, match="lock.executed.v0_export is missing"):
        E._dev_suite_read_fixture(v0_world["bundle"], tmp_path / "b.json", allow_dev_suite=True,
                                  reg=no_exec, manifest_path=v0_world["manifest_path"])
    assert not (tmp_path / "a.json").exists() and not (tmp_path / "b.json").exists()


def test_dev_suite_read_refuses_a_bundle_that_is_not_the_locked_v0(v0_world, tmp_path):
    swapped = json.loads(json.dumps(v0_world["reg"]))
    swapped["lock"]["executed"]["v0_export"]["tokenizer_sha256"] = "0" * 64
    out = tmp_path / "c.json"
    with pytest.raises(SystemExit, match="is not the locked V0 export"):
        E._dev_suite_read_fixture(v0_world["bundle"], out, allow_dev_suite=True,
                                  reg=swapped, manifest_path=v0_world["manifest_path"])
    # the empty claim is released: a preflight refusal is not a spent read
    assert not out.exists() and not E._receipt_path(out).exists()


def test_the_production_surface_cannot_be_redefined_by_arguments(v0_world, tmp_path):
    import inspect
    reg, man = v0_world["reg"], v0_world["man"]
    # the production entry point has no manifest/subset/loader/depth/fixture parameter at all,
    # so a Python caller cannot redefine the surface (Sol dev-reader-fix review P1)
    assert set(inspect.signature(E.dev_suite_read).parameters) == {"bundle_dir", "out",
                                                                   "allow_dev_suite"}
    with pytest.raises(SystemExit, match="fixture-only argument"):
        E._enforce_production_surface(reg, ["syn-a"], man, man, None, "int8", "resident_int8",
                                      None)
    full = ["syn-a", "syn-b"]
    man2 = dict(man, **{"syn-b": {}}, _pinned={"components": full})
    with pytest.raises(SystemExit, match="the complete pinned list"):
        E._enforce_production_surface(reg, ["syn-a"], man2, None, None, "int8",
                                      "resident_int8", None)
    for variant, mode in (("fp16", "eager_fp32"), ("int8", "eager_fp32")):
        with pytest.raises(SystemExit, match="the registered read is"):
            E._enforce_production_surface(reg, full, man2, None, None, variant, mode, None)
    with pytest.raises(SystemExit, match="registered retrieval depth"):
        E._enforce_production_surface(reg, full, man2, None, None, "int8", "resident_int8", 10)
    assert E._enforce_production_surface(reg, full, man2, None, None, "int8", "resident_int8",
                                         None) == int(reg["serving"]["prefetch"])


def test_the_production_read_refuses_another_destination_a_spent_read_and_a_dirty_tree(
        monkeypatch, tmp_path):
    """The production entry point loads the registry itself, writes only to the canonical path,
    needs `v0_export.read: false`, and refuses a dirty tree (Sol dev-reader-fix review P1)."""
    reg = json.loads(json.dumps(E.registry()))
    reg["status"] = "LOCKED_EXECUTABLE"
    reg.setdefault("lock", {}).setdefault("executed", {})["v0_export"] = {
        "model_npz_sha256": "a" * 64, "tokenizer_sha256": "b" * 64, "config_sha256": "c" * 64,
        "read": False}
    monkeypatch.setattr(E, "registry", lambda *a, **k: reg)
    with pytest.raises(SystemExit, match="needs its explicit"):
        E.dev_suite_read("bundle", tmp_path / "x.json")
    with pytest.raises(SystemExit, match="needs its output path"):
        E.dev_suite_read("bundle", allow_dev_suite=True)
    with pytest.raises(SystemExit, match="registered destination"):
        E.dev_suite_read("bundle", tmp_path / "x.json", allow_dev_suite=True)
    monkeypatch.setattr(E, "_git_porcelain", lambda: " M m17src/evaluate.py")
    with pytest.raises(SystemExit, match="uncommitted tracked changes"):
        E.dev_suite_read("bundle", E.V0_READ_PATH, allow_dev_suite=True)
    reg["lock"]["executed"]["v0_export"]["read"] = True
    with pytest.raises(SystemExit, match="v0_export.read is True"):
        E.dev_suite_read("bundle", E.V0_READ_PATH, allow_dev_suite=True)
    monkeypatch.setattr(E, "_git_porcelain", lambda: "")
    assert E._require_clean_tree() == ""


def test_an_interrupted_receipt_continues_only_when_every_identity_matches(v0_world, tmp_path):
    out = tmp_path / "r.json"
    kw = dict(allow_dev_suite=True, reg=v0_world["reg"],
              manifest_path=v0_world["manifest_path"])
    first = E._dev_suite_read_fixture(v0_world["bundle"], out, **kw)
    rp = E._receipt_path(out)
    done = json.loads(rp.read_text())

    def interrupted(**over):
        out.unlink(missing_ok=True)
        base = dict(done, state="started", metrics={}, completed_components=[])
        base.update(over)
        rp.write_text(json.dumps(base))

    interrupted()                                   # killed after `started`: rescore everything
    again = E._dev_suite_read_fixture(v0_world["bundle"], out, **kw)
    assert again["ndcg@10"] == first["ndcg@10"]
    assert json.loads(rp.read_text())["state"] == "complete"
    # a component already persisted is NOT rescored
    interrupted(state="failed", completed_components=["syn-a"],
                metrics={"syn-a": {"ndcg@10": {"d0": 1.0, "q1": 1.0},
                                   "recall@10": {"d0": 1.0, "q1": 1.0}}})
    assert E._dev_suite_read_fixture(v0_world["bundle"], out,
                                     **kw)["ndcg@10"]["macro"] == pytest.approx(1.0)
    # an identity that moved is refused by name, not continued
    interrupted(git_sha="0" * 40)
    with pytest.raises(SystemExit, match="git_sha"):
        E._dev_suite_read_fixture(v0_world["bundle"], out, **kw)
    # and a complete receipt is never a continuation
    out.unlink(missing_ok=True)
    rp.write_text(json.dumps(done))
    with pytest.raises(SystemExit, match="already exists"):
        E._dev_suite_read_fixture(v0_world["bundle"], out, **kw)


def test_teacher_doc_vecs_serves_the_verified_cache_it_scores(v0_world):
    comp = dict(v0_world["comp"], name="syn-a", corpus="text")
    vecs, ident = E._teacher_doc_vecs(comp, v0_world["man"])
    assert np.asarray(vecs).shape == v0_world["doc_vecs"].shape
    assert Path(ident["path"]).parent == v0_world["cache"]
    assert ident["verified_against"] == "teacher shards.json" and ident["n_rows"] == 5
    assert ident["vectors_sha256"] == E.sha_file(ident["path"])


def _text_comp(name, texts, doc_ids=None):
    return {"name": name, "corpus": "text", "doc_texts": texts,
            "doc_ids": doc_ids if doc_ids is not None else [f"d{i}" for i in range(len(texts))]}


def test_teacher_doc_vecs_refuses_tofu_corrupted_wrong_shape_and_another_teacher(tmp_path,
                                                                                 monkeypatch):
    import teacher
    enc = tmp_path / "enc"
    monkeypatch.setattr(teacher, "ENC", enc)
    texts = ["alpha document", "beta document"]
    vecs = np.arange(2 * 4, dtype=np.float16).reshape(2, 4)
    for name, kw in (("tofu", {"tofu": True}), ("bad", {"corrupt": True}), ("ok", {})):
        write_teacher_cache(enc, f"dev-{name}-docs", texts, vecs, **kw)
    with pytest.raises(SystemExit, match="predate hash recording"):
        E._teacher_doc_vecs(_text_comp("tofu", texts), {})
    with pytest.raises(SystemExit, match="does not match the hash recorded"):
        E._teacher_doc_vecs(_text_comp("bad", texts), {})
    with pytest.raises(SystemExit, match="but its encode cache holds 2 rows"):
        E._teacher_doc_vecs(_text_comp("ok", texts, ["d0", "d1", "d2"]), {})
    pinned = {"_pinned": {"active_encoder": {"repo": teacher.TEACHER,
                                             "revision": teacher.TEACHER_REV, "dim": 1024}}}
    with pytest.raises(SystemExit, match="dimensional, the manifest pins 1024"):
        E._teacher_doc_vecs(_text_comp("ok", texts), pinned)
    other = {"_pinned": {"active_encoder": {"repo": "someone/else", "revision": "x", "dim": 4}}}
    with pytest.raises(SystemExit, match="the manifest pins someone/else"):
        E._teacher_doc_vecs(_text_comp("ok", texts), other)
    # a missing shard is refused rather than encoded
    (enc / [d.name for d in enc.iterdir() if d.name.startswith("dev-ok-docs")][0]
     / "shard_00000.npy").unlink()
    with pytest.raises(SystemExit, match="is missing 1 of 1 shards"):
        E._teacher_doc_vecs(_text_comp("ok", texts), {})


@pytest.fixture
def tiny_pool(tmp_path, monkeypatch):
    """A 4-row stand-in for the frozen pool memmap, pinned exactly as `_pinned.pool` pins it."""
    import prepare_data
    monkeypatch.setattr(prepare_data, "POOL_DIR", tmp_path / "pool")
    E._POOL_VERIFIED.clear()
    n, dim = 4, 8
    d = tmp_path / "pool" / "stella-400M-v5"
    d.mkdir(parents=True)
    (d / "vecs.f16").write_bytes(np.arange(n * dim, dtype=np.float16).tobytes())
    meta = {"n": n, "dim": dim, "encoder": "stella-400M-v5",
            "encoder_revision": E.registry()["teacher_revision"],
            "spans": {"s": [0, n]}, "id_sha256": {"s": "0" * 64}}
    (d / "meta.json").write_text(json.dumps(meta))
    man = {"_pinned": {"pool": {"n": n, "dim": dim, "encoder": meta["encoder"],
                                "encoder_revision": meta["encoder_revision"],
                                "spans": meta["spans"],
                                "store_id_sha256": meta["id_sha256"],
                                "vectors_bytes": (d / "vecs.f16").stat().st_size,
                                "vectors_sha256": E.sha_file(d / "vecs.f16")}}}
    return {"dir": d, "man": man, "n": n}


def test_pool_identity_binds_the_memmap_it_scores(tiny_pool):
    vecs, ident = E._pool_identity(tiny_pool["man"], tiny_pool["n"])
    assert vecs.shape == (4, 8) and Path(vecs.filename) == tiny_pool["dir"] / "vecs.f16"
    assert ident["path"] == str(tiny_pool["dir"] / "vecs.f16")
    assert ident["vectors_sha256"] == tiny_pool["man"]["_pinned"]["pool"]["vectors_sha256"]


def test_pool_identity_refuses_a_changed_pool_and_another_active_encoder(tiny_pool):
    man, n = tiny_pool["man"], tiny_pool["n"]
    with pytest.raises(SystemExit, match="component pins 9"):
        E._pool_identity(man, 9)
    swapped = json.loads(json.dumps(man))
    swapped["_pinned"]["pool"]["encoder"] = "bge-base-en-v1.5"
    with pytest.raises(SystemExit, match="the active encoder is"):
        E._pool_identity(swapped, n)
    p = tiny_pool["dir"] / "vecs.f16"
    p.write_bytes(np.full(4 * 8, 7, dtype=np.float16).tobytes())      # same size, new content
    E._POOL_VERIFIED.clear()
    with pytest.raises(SystemExit, match="same size, different content"):
        E._pool_identity(man, n)
    p.write_bytes(b"\x00" * 8)                                        # and a truncated file
    with pytest.raises(SystemExit, match="is not 4 x 8 fp16"):
        E._pool_identity(man, n)


def test_a_missing_pinned_hash_field_refuses_instead_of_skipping_its_check(v0_world):
    entry = dict(v0_world["man"]["syn-a"])
    entry.pop("corpus_text_sha256")
    with pytest.raises(SystemExit, match="has no \\['corpus_text_sha256'\\]"):
        E._require_pinned_fields("syn-a", entry)
    with pytest.raises(SystemExit, match="qtexts_ordered_sha256"):
        E._require_pinned_fields("heldout-x", {"corpus": "full-pool"})
    E._require_pinned_fields("syn-a", v0_world["man"]["syn-a"])


def test_rehearsal_evaluation_stayed_synthetic(rehearsal):
    ev = rehearsal["record"]["stages"]["evaluate"]
    assert ev["surface"] == "synthetic fixtures only"
    assert set(ev["ndcg@10"]["per_domain"]) <= {"general", "cloud-software",
                                                "science-engineering"}
    assert ev["alias_test"]["n_pairs"] > 0
