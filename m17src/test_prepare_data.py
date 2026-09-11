"""Checks for the step-5 prepared-data builder.

Synthetic fixtures and scratch outputs only: no test reads a real store, the frozen pool, a
development component, the M17 panel or any protected surface, and none writes a registered
artifact or depends on a local checkpoint.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import cache as m17cache
import prepare_data as pd
import train as m17train


# --------------------------------------------------------------------------- pool planning

def test_plan_respects_the_query_cap_and_four_pass_ceiling(reg):
    avail = {"general": 10 ** 7, "coverage_views": 10 ** 7, "alias_pairs": 10 ** 6}
    plan, total, factor = pd._plan(reg, avail, None)
    dose = reg["data"]["measured_dose_after_pre_lock_rule"]
    assert factor == 1.0
    assert total <= int(reg["data"]["training_query_cap"])
    # no bucket holds more distinct rows than four passes at the registered dose need
    assert plan["coverage"] <= dose["unpaired_coverage_views_per_batch"] * dose["steps"] / 4 + 1
    assert plan["alias_pairs"] <= dose["alias_pairs_per_batch"] * dose["steps"] / 4 + 1


def test_plan_subsamples_every_bucket_proportionally(reg):
    avail = {"general": 500000, "coverage_views": 10 ** 6, "alias_pairs": 20000}
    full, total, _ = pd._plan(reg, avail, None)
    small, _t, factor = pd._plan(reg, avail, total // 10)
    assert 0.09 < factor < 0.11
    for k in ("general", "coverage", "alias_pairs"):
        assert small[k] == pytest.approx(full[k] * factor, rel=0.05)


def test_stratified_is_deterministic_and_keeps_source_proportions():
    rows = [{"s": "a", "i": i} for i in range(800)] + [{"s": "b", "i": i} for i in range(200)]
    one = pd._stratified(rows, lambda r: r["s"], 100, (0, "x"))
    two = pd._stratified(rows, lambda r: r["s"], 100, (0, "x"))
    other = pd._stratified(rows, lambda r: r["s"], 100, (1, "x"))
    assert one == two and len(one) == 100
    assert one != other
    counts = pd._tally(r["s"] for r in one)
    assert counts["a"] == 80 and counts["b"] == 20


def test_stratified_returns_everything_when_the_target_exceeds_the_population():
    rows = [{"s": "a", "i": i} for i in range(5)]
    assert pd._stratified(rows, lambda r: r["s"], 50, (0, "x")) == rows


# --------------------------------------------------------------------------- held-out slice

def test_heldout_draw_keeps_whole_families_and_skips_giant_ones():
    by_family = {f"f{i}": [f"q{i}-{j}" for j in range(3)] for i in range(20)}
    by_family["giant"] = [f"g{j}" for j in range(500)]
    picked, n = draw = pd.draw_heldout_families(by_family, 12, seed=0)
    assert "giant" not in picked
    assert n >= 12 and n % 3 == 0                       # whole families only
    assert draw == pd.draw_heldout_families(by_family, 12, seed=0)


def test_heldout_families_are_excluded_from_the_training_buckets():
    """The bucket split a prepared pool must satisfy: `train._check_heldout`'s own rule."""
    by_family = {f"f{i}": [f"q{i}-{j}" for j in range(2)] for i in range(10)}
    picked, _ = pd.draw_heldout_families(by_family, 4, seed=0)
    specs = []
    for fam, qs in sorted(by_family.items()):
        for q in qs:
            specs.append(m17cache.QuerySpec(qid=q, text=q, family=fam,
                                            bucket="heldout" if fam in picked else "general"))
    heldout = np.asarray([i for i, s in enumerate(specs) if s.bucket == "heldout"])
    streams, pair_index, _ = m17train.build_streams(
        [s.bucket for s in specs], [s.alias_pair_id for s in specs],
        [s.alias_view for s in specs], [s.family for s in specs], seed=0)
    cfg = m17train.RunCfg(rehearsal=True, heldout_queries=len(heldout))
    m17train._check_heldout(heldout, streams, pair_index, cfg)   # must not raise


# --------------------------------------------------------------------------- denied sources

@pytest.mark.parametrize("name", ["msmarco-train", "work/train/stores/msmarco-pos.json",
                                  "MSMARCO", "fineweb-edu"])
def test_denied_sources_are_refused_by_name(name):
    with pytest.raises(SystemExit) as e:
        pd._denied(name)
    assert "not a training input" in str(e.value)


def test_admitted_sources_pass_the_name_check():
    for name in pd.sm.PAIR_SOURCES + pd.sm.QUERYTEXT_SOURCES + ["k8s-docs-en", "fever-pos"]:
        assert pd._denied(name) == name


# --------------------------------------------------------------------------- domain join

def test_domain_join_refuses_two_labels_for_one_document():
    dd, gd, dg = {}, {}, {}
    pd.join_domain(dd, gd, dg, "squad-train:d1", "g1", "medicine")
    pd.join_domain(dd, gd, dg, "squad-train:d1", "g1", "medicine")       # idempotent
    with pytest.raises(SystemExit) as e:
        pd.join_domain(dd, gd, dg, "squad-train:d1", "g1", "finance")
    assert "one domain" in str(e.value) or "labelled" in str(e.value)


def test_domain_join_refuses_two_labels_for_one_text_group():
    dd, gd, dg = {}, {}, {}
    pd.join_domain(dd, gd, dg, "squad-train:d1", "g1", "legal")
    with pytest.raises(SystemExit) as e:
        pd.join_domain(dd, gd, dg, "squad-train:d2", "g1", "general")
    assert "own text alone" in str(e.value)


# --------------------------------------------------------------------------- alias views

def test_both_alias_views_share_one_family_and_one_pair_id():
    pair = {"pair_id": "p1", "family_id": "f1", "source": "k8s-docs-en",
            "view_a": "the api server rejects it", "view_b": "the kube-apiserver rejects it"}
    a, b = pd.alias_specs(pair, "cloud-software")
    assert a.family == b.family == "alias:f1"
    assert a.alias_pair_id == b.alias_pair_id == "p1"
    assert {a.alias_view, b.alias_view} == {"a", "b"}
    assert a.bucket == b.bucket == "coverage"
    # and `train.build_streams` accepts them as a well-formed pair
    specs = [m17cache.QuerySpec(qid=f"g{i}", text=f"g{i}", bucket="general", family=f"g{i}")
             for i in range(4)] + [a, b]
    _s, pair_index, rep = m17train.build_streams(
        [s.bucket for s in specs], [s.alias_pair_id for s in specs],
        [s.alias_view for s in specs], [s.family for s in specs], seed=0)
    assert rep["n_pairs"] == 1 and pair_index.shape == (1, 2)


def test_a_split_that_separates_the_two_views_is_refused_downstream():
    pair = {"pair_id": "p1", "family_id": "f1", "source": "k8s-docs-en",
            "view_a": "a", "view_b": "b"}
    a, b = pd.alias_specs(pair, "general")
    specs = [a, b]
    with pytest.raises(SystemExit) as e:
        m17train.build_streams([a.bucket, "heldout"], [a.alias_pair_id, b.alias_pair_id],
                               [a.alias_view, b.alias_view], [s.family for s in specs], seed=0)
    assert "malformed alias pairs" in str(e.value)


# --------------------------------------------------------------------------- bank / cache

def test_a_positive_outside_the_bank_is_refused_by_qid(tiny_world, reg):
    q = m17cache.QuerySpec(qid="q-outside", text="a query", positive_ids=("not-in-the-bank",))
    with pytest.raises(ValueError) as e:
        m17cache.build([q], tiny_world["bank"], tiny_world["teacher_q"][:1],
                       tiny_world["v1_q"][:1], reg, cache_seed=0,
                       manifests={"v1_artifact": {"x": 1}, "teacher_query_preprocessing": "t"})
    assert "q-outside" in str(e.value) and "outside the bank" in str(e.value)


# --------------------------------------------------------------------------- teacher cache

def test_teacher_cache_reuses_vectors_by_text_sha(tmp_path):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return np.stack([np.full(8, float(len(t))) for t in texts])

    c = pd.TextVectorCache(tmp_path / "tc", 8)
    v1, r1 = c.get(["alpha", "beta", "alpha"], encode)
    assert r1["encoded"] == 2 and r1["requested"] == 3          # the duplicate is not re-encoded
    assert np.allclose(v1[0], v1[2])
    c2 = pd.TextVectorCache(tmp_path / "tc", 8)                  # a fresh process
    v2, r2 = c2.get(["beta", "alpha"], encode)
    assert r2["encoded"] == 0 and r2["hit_rate"] == 1.0
    assert len(calls) == 1
    assert np.allclose(v2[1], v1[0])
    v3, r3 = c2.get(["alpha", "gamma"], encode)
    assert r3["encoded"] == 1 and r3["hit_rate"] == 0.5


def test_teacher_cache_refuses_an_inconsistent_store(tmp_path):
    c = pd.TextVectorCache(tmp_path / "tc", 4)
    c.get(["x"], lambda ts: np.ones((len(ts), 4)))
    (tmp_path / "tc" / "index.json").write_text(json.dumps({"a": 0, "b": 1}))
    with pytest.raises(SystemExit) as e:
        pd.TextVectorCache(tmp_path / "tc", 4)
    assert "inconsistent" in str(e.value)


# --------------------------------------------------------------------------- warm start

def _fake_checkpoint(path, rows, weights, meta):
    import sys
    sys.path.insert(0, str(pd.REPO / "m7src"))
    np.savez(path, rows_fp16=rows.astype(np.float16),
             rows_int8=np.zeros(rows.shape, dtype=np.int8),
             int8_scale=np.ones(rows.shape[0], dtype=np.float32),
             token_weights=weights.astype(np.float32),
             updates=np.zeros(rows.shape[0], dtype=np.int64))
    (path.parent / (path.stem + ".meta.json")).write_text(json.dumps(meta))


def test_warm_start_materialization_refuses_a_folded_artifact(tmp_path, monkeypatch, reg):
    """A release has no per-token scalars; folding it again would double-scale every row."""
    src = tmp_path / "folded.npz"
    _fake_checkpoint(src, np.ones((4, 3), np.float32), np.zeros(0, np.float32),
                     {"teacher": reg["teacher"], "teacher_revision": reg["teacher_revision"]})
    monkeypatch.setattr(pd, "WARM_START", src)
    ctx = type("C", (), {})()
    ctx.shared = tmp_path / "shared"
    ctx.shared.mkdir()
    ctx.freeze = {"training_checkpoint_sha256": pd.sha_file(src)}
    with pytest.raises(SystemExit) as e:
        pd._materialize_warm_start(ctx)
    assert "unfolded" in str(e.value)


def test_warm_start_materialization_refuses_a_hash_mismatch(tmp_path, monkeypatch, reg):
    src = tmp_path / "ck.npz"
    _fake_checkpoint(src, np.ones((4, 3), np.float32), np.ones(4, np.float32), {})
    monkeypatch.setattr(pd, "WARM_START", src)
    ctx = type("C", (), {})()
    ctx.shared = tmp_path / "shared"
    ctx.shared.mkdir()
    ctx.freeze = {"training_checkpoint_sha256": "0" * 64}
    with pytest.raises(SystemExit) as e:
        pd._materialize_warm_start(ctx)
    assert "FREEZE records" in str(e.value)


def test_old_vocab_parity_refuses_doubly_folded_rows():
    rows = np.random.default_rng(0).normal(size=(32, 8)).astype(np.float32)
    w = 1.0 + np.random.default_rng(1).random(32).astype(np.float32)
    eff = pd.m17vocab.effective_rows(rows, w)
    assert pd.m17vocab.verify_old_vocab_parity(eff, eff) == 0.0
    with pytest.raises(AssertionError) as e:
        pd.m17vocab.verify_old_vocab_parity(pd.m17vocab.effective_rows(eff, w), eff)
    assert "never fold twice" in str(e.value).lower() or "parity FAILED" in str(e.value)


# --------------------------------------------------------------------------- manifests

def test_base_and_ext_forms_match_the_arms_they_claim():
    for form, arms in pd.ARMS_FOR_FORM.items():
        for arm in arms:
            assert m17train.ARMS[arm]["vocab_extension"] is (form == "ext")
    assert sorted(a for arms in pd.ARMS_FOR_FORM.values() for a in arms) == sorted(m17train.ARMS)


def test_source_doc_is_the_supporting_document_not_the_query():
    specs = [
        m17cache.QuerySpec(qid="squad-train:1", text="q", source="squad-train",
                           positive_ids=("squad-train:d7",)),
        m17cache.QuerySpec(qid="nqopen:5", text="q", source="nqopen"),
        m17cache.QuerySpec(qid="cov:k8s-docs-en:content/en/x.md:title", text="v",
                           source="k8s-docs-en", bucket="coverage"),
        *pd.alias_specs({"pair_id": "p", "family_id": "f", "source": "hotpotqa-train",
                         "view_a": "a", "view_b": "b"}, "general"),
    ]
    overrides = {s.qid: "hotpotqa-train:doc-group:gg" for s in specs if s.alias_pair_id}
    m = pd._source_doc_map(specs, overrides)
    assert m["squad-train:1"] == "squad-train:d7"
    # one sentinel unit per query-text-only source, not one per query
    assert m["nqopen:5"] == "nqopen:querytext-no-document"
    assert m["cov:k8s-docs-en:content/en/x.md:title"] == "k8s-docs-en:content/en/x.md"
    # both alias views point at the same EVIDENCE DOCUMENT GROUP, never at the family
    alias = {m[s.qid] for s in specs if s.alias_pair_id}
    assert alias == {"hotpotqa-train:doc-group:gg"}


def test_document_views_are_deterministic_and_deduplicated():
    text = " ".join(f"w{i}" for i in range(200))
    a = pd.document_views("k8s-docs-en:x", "Cluster Architecture", text)
    b = pd.document_views("k8s-docs-en:x", "Cluster Architecture", text)
    assert a == b and 2 <= len(a) <= 3
    assert len({v["text"] for v in a}) == len(a)
    short = pd.document_views("s:y", "", "one two three four")
    assert len(short) == 1 and short[0]["kind"] == "span_head"


def test_abbreviation_inventory_prefers_the_single_word_form():
    pairs = [{"form_a": "Pennsylvania Interscholastic Athletic Association", "form_b": "PIAA"},
             {"form_a": "kube-apiserver", "form_b": "API server"}]
    abbrs, exp = pd.abbreviation_inventory(pairs)
    assert "piaa" in abbrs and exp["piaa"].startswith("pennsylvania")
    assert "kube-apiserver" in abbrs and exp["kube-apiserver"] == "api server"
