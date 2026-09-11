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

def test_plan_respects_the_query_cap_and_the_four_pass_minimum(reg):
    """The four-pass population is a MINIMUM to check against the dose, never a cap."""
    avail = {"general": 10 ** 7, "coverage_views": 10 ** 7, "alias_pairs": 10 ** 6}
    plan, total, factor = pd._plan(reg, avail, None)
    dose = reg["data"]["measured_dose_after_pre_lock_rule"]
    assert factor == 1.0
    assert total <= int(reg["data"]["training_query_cap"])
    # each bucket holds AT LEAST the distinct rows four passes at the registered dose need
    assert plan["coverage"] >= dose["unpaired_coverage_views_per_batch"] * dose["steps"] / 4
    assert plan["alias_pairs"] >= dose["alias_pairs_per_batch"] * dose["steps"] / 4
    # and the general bucket takes everything the cap still allows, not a four-pass slice
    assert plan["general"] == int(reg["data"]["training_query_cap"]) - plan["coverage"] \
        - 2 * plan["alias_pairs"]


def test_the_uniform_sampler_does_not_favour_the_head_of_the_file():
    """Lowest-seeded-hash selection over the whole population (P1-7).

    The old streaming sampler accepted with probability ~4*quota/population and stopped at the
    quota, so it usually ended around the first quarter of the file.
    """
    keys = [f"doc-{i:06d}" for i in range(10000)]
    picked = pd._lowest_hash_pick(keys, 1000, 0, "bank", "src")
    assert len(picked) == len(set(picked)) == 1000
    assert picked == pd._lowest_hash_pick(reversed(keys), 1000, 0, "bank", "src")
    deciles = pd._tally(f"d{keys.index(k) // 1000}" for k in picked)
    first, last = deciles["d0"], deciles["d9"]
    # both deciles hold ~100 of the 1000 draws; +-4 sd of a binomial(1000, 0.1) is +-38
    assert abs(first - 100) < 38 and abs(last - 100) < 38
    assert set(deciles) == {f"d{i}" for i in range(10)}


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


def test_source_doc_is_one_canonical_key_per_supporting_document():
    """P1-10: the same document keeps ONE identity in every role, and documentless is None."""
    specs = [
        m17cache.QuerySpec(qid="squad-train:1", text="q", source="squad-train",
                           positive_ids=("squad-train:d7",)),
        m17cache.QuerySpec(qid="squad-train:2", text="q2", source="squad-train"),   # budget-bound
        m17cache.QuerySpec(qid="nqopen:5", text="q", source="nqopen"),
        m17cache.QuerySpec(qid="cov:squad-train:d7:title", text="v", source="squad-train",
                           bucket="coverage"),
        *pd.alias_specs({"pair_id": "p", "family_id": "f", "source": "squad-train",
                         "view_a": "a", "view_b": "b"}, "general"),
    ]
    groups = {"squad-train:1": ("squad-train", "g7"), "squad-train:2": ("squad-train", "g7"),
              "cov:squad-train:d7:title": ("squad-train", "g7")}
    groups.update({s.qid: ("squad-train", "g7") for s in specs if s.alias_pair_id})
    m = pd._source_doc_map(specs, groups)
    # one document, one key: general (labeled and budget-bound), coverage and alias agree
    assert len({m[q] for q in m if q != "nqopen:5"}) == 1
    assert m["squad-train:1"] == "squad-train:doc-group:g7"
    # a documentless source casts NO vote (data.documentless_sources_vote == "none", A4)
    assert m["nqopen:5"] is None


def test_documentless_sources_cannot_reach_the_document_minimum(tok_and_vocab):
    """18 real documents plus nqopen and triviaqa must NOT count as 20 (P1-10)."""
    tok, _n = tok_and_vocab
    term = "kubectl"
    records = [{"text": f"{term} context {i}", "qid": f"q{i}", "domain": "cloud-software",
                "source_doc": f"squad-train:doc-group:g{i}"} for i in range(18)]
    records += [{"text": f"{term} open question", "qid": "nqopen:1", "domain": "general",
                 "source_doc": None},
                {"text": f"{term} trivia question", "qid": "triviaqa:1", "domain": "general",
                 "source_doc": None}]
    stats, _single = pd.m17vocab.discover(records, np.full(len(records), 0.5), tok)
    st = stats[term]
    assert st.n_docs == 18                      # not 20: the two sentinels cast no vote
    assert st.n_contexts == 20                  # contexts and residuals still count
    assert st.residual_n == 20


def test_discover_still_refuses_two_domains_for_one_real_document(tok_and_vocab):
    tok, _n = tok_and_vocab
    records = [{"text": "kubectl a", "qid": "a", "domain": "cloud-software",
                "source_doc": "squad-train:doc-group:g1"},
               {"text": "kubectl b", "qid": "b", "domain": "medicine",
                "source_doc": "squad-train:doc-group:g1"}]
    with pytest.raises(ValueError) as e:
        pd.m17vocab.discover(records, np.full(2, 0.5), tok)
    assert "labelled" in str(e.value)
    # and a MISSING key is still an error, unlike an explicit None
    with pytest.raises(ValueError):
        pd.m17vocab.discover([{"text": "kubectl a", "qid": "a"}], np.full(1, 0.5), tok)


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


# --------------------------------------------------------------------------- exclusions

def _exclusion_world():
    """One excluded item per category, and one admitted control for each."""
    ex = {"families": {"famX"}, "text_shas": {pd.sm.group_id(pd.sm.normalize("held out text"))},
          "terms": {pd.sm.normalize("PDB")},
          "doc_groups": {"gX"}}
    return pd.Exclusions(ex, doc_ids={"squad-train": {"dX"}})


def test_one_exclusion_predicate_applies_by_role_and_counts_realized_drops():
    e = _exclusion_world()
    assert e.general("famX", "an ordinary query", "squad-train", ["d1"])          # family
    assert e.general("fam1", "held out text", "squad-train", ["d1"])              # text sha
    assert e.general("fam1", "pdb", "squad-train", ["d1"])                        # alias term
    assert e.general("fam1", "an ordinary query", "squad-train", ["dX"])          # positive group
    assert not e.general("fam1", "an ordinary query", "squad-train", ["d1"])
    assert e.coverage_document("squad-train", "d2", "gX")                         # source group
    assert e.coverage_document("squad-train", "dX", "g2")
    assert not e.coverage_document("squad-train", "d2", "g2")
    assert e.coverage_view("held out text") and e.coverage_view("PDB")
    assert not e.coverage_view("an ordinary coverage view")
    pair = {"family_id": "f1", "doc_group": "g1", "view_a": "a view", "view_b": "b view",
            "form_a": "x", "form_b": "y"}
    assert e.alias_pair({**pair, "family_id": "famX"})
    assert e.alias_pair({**pair, "doc_group": "gX"})
    assert e.alias_pair({**pair, "view_b": "held out text"})
    assert e.alias_pair({**pair, "form_b": "PDB"})
    assert not e.alias_pair(pair)
    # every category realized exactly the drops above -- these are counts, not inventory sizes
    assert e.realized == {"general_family": 1, "general_text_sha": 1, "general_alias_term": 1,
                          "general_positive_doc_group": 1, "coverage_doc_group": 2,
                          "coverage_text_sha": 1, "coverage_alias_term": 1, "alias_family": 1,
                          "alias_doc_group": 1, "alias_view_sha": 1, "alias_term": 1}


# --------------------------------------------------------------------------- cross-bucket split

def test_a_document_supporting_a_heldout_family_is_excluded_from_the_other_buckets():
    general = [{"src": "squad-train", "qid": "1", "pos": ["d1"], "family": "f1"},
               {"src": "squad-train", "qid": "2", "pos": ["d2"], "family": "f2"}]
    doc_group = {"squad-train:d1": "g1", "squad-train:d2": "g2"}
    held = pd.heldout_document_groups(general, [True, False], doc_group)
    assert held == {"g1"}                       # only the held-out query's document
    assert "g2" not in held


# --------------------------------------------------------------------------- labeled subset

def _general(n, per_query_positives=1):
    return [{"src": "squad-train", "qid": str(i), "family": f"f{i}",
             "pos": [f"d{i}-{j}" for j in range(per_query_positives)]} for i in range(n)]


def test_labeled_subset_excludes_the_heldout_slice_and_respects_the_budget():
    general = _general(10)
    doc_domain = {f"squad-train:{d}": "general" for r in general for d in r["pos"]}
    is_heldout = [i < 3 for i in range(10)]
    eligible, positives, q_only = pd.select_labeled_subset(
        general, doc_domain, is_heldout, bank_budget=4, seed=0)
    assert not (eligible & {0, 1, 2})           # the divergence slice never spends the budget
    assert len(positives) == 4 and len(eligible) == 4
    assert q_only == 3                          # the remaining seven: four admitted, three not


def test_a_labeled_query_whose_positives_never_joined_is_refused_by_qid():
    general = _general(3)
    doc_domain = {"squad-train:d0-0": "general", "squad-train:d2-0": "general"}
    with pytest.raises(SystemExit) as e:
        pd.select_labeled_subset(general, doc_domain, [False] * 3, bank_budget=99, seed=0)
    assert "squad-train:1" in str(e.value)


# --------------------------------------------------------------------------- protected screen

class _Ctx:
    """The handful of `Ctx` attributes the screen stages touch."""

    def __init__(self, tmp_path, reg, protected=False):
        self.out = tmp_path
        self.reg = reg
        self.protected_screen = protected
        self.stages, self.timings, self.notes, self.cache_state = {}, {}, {}, {}


def test_the_protected_screen_gates_on_the_registry_before_importing_protected10(tmp_path,
                                                                                 monkeypatch):
    """P1-1: the flag must not reach protected10.build() while the registry is a draft."""
    import common
    real = json.loads((pd.REPO / "m17" / "registry.json").read_text())
    assert real["status"] not in common.EXECUTABLE_STATUSES
    ctx = _Ctx(tmp_path, real, protected=True)
    ctx.cache_state["specs"] = []
    monkeypatch.delitem(__import__("sys").modules, "protected10", raising=False)
    with pytest.raises(common.NotExecutable):
        pd.stage_protected(ctx)
    assert "protected10" not in __import__("sys").modules


def test_the_deferred_screen_counts_the_unscreened_new_source_rows(tmp_path, reg):
    ctx = _Ctx(tmp_path, reg)
    ctx.cache_state["specs"] = [
        m17cache.QuerySpec(qid="k8s-docs-en:1", text="a", source="k8s-docs-en"),
        m17cache.QuerySpec(qid="squad-train:1", text="b", source="squad-train")]
    rec = pd.stage_protected(ctx)
    assert rec["state"] == "deferred_to_clock" and rec["unscreened_new_source_rows"] == 1


def _screen_specs():
    a, b = pd.alias_specs({"pair_id": "p1", "family_id": "f1", "source": "k8s-docs-en",
                           "view_a": "view a", "view_b": "view b"}, "cloud-software")
    return [m17cache.QuerySpec(qid="q0", text="protected text", source="squad-train",
                               bucket="heldout", family="q:f0"),
            m17cache.QuerySpec(qid="q1", text="ordinary", source="squad-train",
                               bucket="heldout", family="q:f9"),
            m17cache.QuerySpec(qid="q2", text="ordinary two", source="squad-train",
                               positive_ids=("squad-train:d1",), family="q:f2"),
            a, b]


def test_a_screen_drop_rebuilds_the_pool_and_survives_a_restart(tmp_path, reg):
    """P1-4: persist the screened pool, rebuild the indices, drop BOTH views of a pair."""
    ctx = _Ctx(tmp_path, reg)
    specs = _screen_specs()
    ctx.cache_state["specs"] = specs
    ctx.cache_state["heldout_idx"] = [0, 1]
    ctx.cache_state["source_doc"] = {s.qid: "k8s-docs-en:doc-group:g1" for s in specs}
    ctx.cache_state["positives"] = ["squad-train:d1"]
    rec = pd._apply_screen(ctx, {"q0", "alias:p1:a"})
    assert rec["rows_dropped"] == 3 and rec["alias_pairs_dropped"] == 1
    kept = ctx.cache_state["specs"]
    assert [s.qid for s in kept] == ["q1", "q2"]
    assert ctx.cache_state["heldout_idx"] == [0]          # rebuilt, not stale positions
    assert ctx.cache_state["positives"] == ["squad-train:d1"]
    # and a restart rehydrates the SCREENED pool from disk, never the original one
    fresh = _Ctx(tmp_path, reg)
    pd._rehydrate(fresh, "protected")
    assert [s.qid for s in fresh.cache_state["specs"]] == ["q1", "q2"]
    assert fresh.cache_state["heldout_idx"] == [0]


class _FakeProtected:
    """A stand-in for `m10src/protected10.py`: nothing protected is opened in a test."""
    VERSION = "fixture"

    @staticmethod
    def build(verbose=False):
        return ("idx",)

    @staticmethod
    def hits(text, idx):
        return "exact" if "PROTECTED" in text else None


def test_the_screen_reads_full_document_text_and_the_bank_follows_the_receipt(tmp_path,
                                                                             monkeypatch, reg):
    """P1-2: a document can match while every view sampled from it does not."""
    import sys as _sys
    docs = [("content/en/a.md", "A", "ordinary admitted document text"),
            ("content/en/b.md", "B", "this body contains PROTECTED payload text")]
    monkeypatch.setattr(pd.sm, "k8s_excluded_paths", lambda: set())
    monkeypatch.setattr(pd.sm, "iter_k8s", lambda excluded: iter(docs))
    monkeypatch.setitem(_sys.modules, "protected10", _FakeProtected)
    groups = {p: pd.sm.group_id(pd.sm.normalize(t)) for p, _t2, t in docs}
    specs = [m17cache.QuerySpec(qid=f"cov:k8s-docs-en:{p}:title", text=f"view of {p}",
                                source="k8s-docs-en", bucket="coverage") for p, _t, _x in docs]
    ctx = _Ctx(tmp_path, {**reg, "status": "EXECUTABLE"}, protected=True)
    ctx.cache_state["specs"] = specs
    ctx.cache_state["heldout_idx"] = []
    ctx.cache_state["positives"] = []
    ctx.cache_state["source_doc"] = {s.qid: f"k8s-docs-en:doc-group:{groups[p]}"
                                     for s, (p, _t, _x) in zip(specs, docs)}
    rec = pd.stage_protected(ctx)
    assert rec["state"] == "complete"
    assert rec["documents_screened"] == 2 and rec["documents_dropped"] == 1
    # the view text itself is innocent; the row goes because its DOCUMENT was not admitted
    assert rec["dropped_by_text"] == 0 and rec["dropped_by_document_receipt"] == 1
    assert [s.qid for s in ctx.cache_state["specs"]] == ["cov:k8s-docs-en:content/en/a.md:title"]
    ctx.stages["protected"] = rec
    assert pd._screen_receipt(ctx) == {"content/en/a.md"}
    assert rec["unscreened_new_source_rows"] == 0


def test_a_deferred_receipt_is_no_receipt_at_all(tmp_path, reg):
    ctx = _Ctx(tmp_path, reg)
    ctx.stages["protected"] = {"state": "deferred_to_clock"}
    assert pd._screen_receipt(ctx) is None       # the bank filter is a no-op pre-clock


# --------------------------------------------------------------------------- pool layout

def test_pool_ids_are_authenticated_against_the_vector_layout(tmp_path, monkeypatch):
    """P2-13: reordering `ids-<store>.json` beside the memmap must not pass."""
    from hashing import sha_stream_list
    ids = ["d0", "d1", "d2"]
    d = tmp_path / "stella-400M-v5"
    d.mkdir(parents=True)
    meta = {"encoder": "stella-400M-v5",
            "encoder_revision": pd.registry()["teacher_revision"], "dim": 2, "n": 3,
            "spans": {"squad-ctx": [0, 3]}, "id_sha256": {"squad-ctx": sha_stream_list(ids)}}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "vecs.f16").write_bytes(np.zeros((3, 2), dtype=np.float16).tobytes())
    (tmp_path / "ids-squad-ctx.json").write_text(json.dumps(ids))
    monkeypatch.setattr(pd, "POOL_DIR", tmp_path)
    r = pd.PoolReader()
    assert r.rows_for("squad-ctx", ["d1"]) == {"d1": 1}
    (tmp_path / "ids-squad-ctx.json").write_text(json.dumps(["d2", "d1", "d0"]))
    with pytest.raises(SystemExit, match="do not describe the stored vectors"):
        r.rows_for("squad-ctx", ["d1"])
    (tmp_path / "ids-squad-ctx.json").write_text(json.dumps(["d0", "d1"]))
    with pytest.raises(SystemExit, match="span"):
        r.rows_for("squad-ctx", ["d1"])


# --------------------------------------------------------------------------- stage identity

def test_a_changed_seed_refuses_a_cached_stage_unless_forced():
    prev = {"_identity": {"seed": 0, "size": 2000}}
    ident = {"seed": 1, "size": 2000}
    assert pd._check_stage_identity({"_identity": ident}, ident, "pool", False)
    with pytest.raises(SystemExit) as e:
        pd._check_stage_identity(prev, ident, "pool", False)
    assert "different inputs" in str(e.value) and "seed" in str(e.value)
    assert pd._check_stage_identity(prev, ident, "pool", True) is False   # --force rebuilds


def test_forcing_a_stage_invalidates_its_dependents():
    """The fixed dependency order: everything after a rebuilt stage is rebuilt."""
    i = pd.STAGES.index("teacher")
    assert pd.STAGES[i + 1:] == ("bank", "v1", "parity", "vocab", "cache", "manifests")


# --------------------------------------------------------------------------- v1 variant

def test_v1_mining_refuses_anything_but_the_released_int8_table(tmp_path):
    assert pd.V1_VARIANT == "int8"
    with pytest.raises(SystemExit) as e:
        pd._load_v1_table(tmp_path / "x.npz", "cpu", variant="fp16")
    assert "int8" in str(e.value)


def test_int8_and_fp16_rows_are_different_encoders(tmp_path):
    """The npz hash cannot distinguish the variants; the rows can (P1-9)."""
    from table import dequantize_int8, load_table
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(8, 4)).astype(np.float32)
    scale = np.abs(rows).max(1) / 127.0
    codes = np.round(rows / scale[:, None]).astype(np.int8)
    p = tmp_path / "t.npz"
    np.savez(p, rows_fp16=rows.astype(np.float16), rows_int8=codes,
             int8_scale=scale.astype(np.float32), token_weights=np.zeros(0, np.float32))
    int8 = pd._load_v1_table(p, "cpu").rows.detach().numpy()
    fp16 = load_table(p, variant="fp16", device="cpu").rows.detach().numpy()
    assert np.allclose(int8, dequantize_int8(codes, scale.astype(np.float32)))
    assert not np.array_equal(int8, fp16)


# --------------------------------------------------------------------------- vector caches

def test_a_text_vector_cache_refuses_a_changed_preprocessing_manifest(tmp_path):
    m = {"instruction": "one", "encode_dtype": "float32", "storage_dtype": "float16"}
    c = pd.TextVectorCache(tmp_path / "tc", 4, manifest=m)
    assert c.manifest_adopted
    c.get(["x"], lambda ts: np.ones((len(ts), 4)))
    pd.TextVectorCache(tmp_path / "tc", 4, manifest=m)               # same manifest: reused
    with pytest.raises(SystemExit) as e:
        pd.TextVectorCache(tmp_path / "tc", 4, manifest={**m, "instruction": "two"})
    assert "preprocessing manifest" in str(e.value)


def test_the_teacher_manifest_states_encode_and_storage_precision_separately(reg):
    ctx = type("C", (), {})()
    ctx.reg = reg
    import teacher as m7teacher
    ctx.freeze = {"encoder_spec": {"query_prefix": m7teacher.QUERY_PREFIX, "max_length": 512,
                                   "pooling": "cls", "tokenizer_id": "fixture", "dim": 8}}
    pre = pd.teacher_preprocessing(ctx)
    assert pre["encode_dtype"] == "float32" and pre["storage_dtype"] == "float16"
    assert "dtype" not in pre


# --------------------------------------------------------------------------- v1 rounding

def test_fresh_and_resumed_v1_vectors_are_identical(tmp_path):
    """P2-16: the stage rounds to fp16 before any use and hashes that representation."""
    rng = np.random.default_rng(3)
    v1 = rng.normal(size=(16, 8)).astype(np.float32)
    v1 /= np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-9)
    fresh = v1.astype(np.float16)
    np.save(tmp_path / "v1_q.npy", fresh)
    resumed = np.load(tmp_path / "v1_q.npy").astype(np.float32)
    assert np.array_equal(fresh.astype(np.float32), resumed)
    assert pd.sha_array(fresh) == pd.sha_array(np.load(tmp_path / "v1_q.npy"))
    assert pd.sha_array(v1) != pd.sha_array(fresh)      # the old fp32 digest was not the stored one
