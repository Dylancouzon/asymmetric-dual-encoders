"""Support-manifest counting, grouping, family linking and the registry's dose rule.

Every check builds a synthetic work tree in `tmp_path`. Nothing reads the real
`work/train/`, `results/`, a development component, the panel or any protected surface.
"""
from __future__ import annotations

import gzip
import json

import pytest

import support_manifest as S


# --------------------------------------------------------------------------- synthetic world

def _world(tmp_path, monkeypatch, docs=None, pairs=None, querytext=None, flagged=()):
    train = tmp_path / "train"
    (train / "stores").mkdir(parents=True)
    (train / "sources").mkdir()
    (train / "querytext").mkdir()
    (tmp_path / "decontam").mkdir()
    docs = docs or {}
    pairs = pairs or {}
    kept = {}
    for src in S.PAIR_SOURCES:
        store = S.STORE_OF[src]
        ids, texts = zip(*docs.get(src, [("d0", f"{src} filler document")]))
        (train / "stores" / f"{store}.json").write_text(
            json.dumps({"ids": list(ids), "texts": list(texts)}))
        ps = pairs.get(src, [])
        (train / "sources" / f"{src}.json").write_text(
            json.dumps({"pairs": ps, "docstore": store}))
        kept[src] = [str(p["qid"]) for p in ps]
    qt = querytext or {}
    kept_qt = {}
    for src in S.QUERYTEXT_SOURCES:
        rows = qt.get(src, [])
        (train / "querytext" / f"{src}.json").write_text(json.dumps(rows))
        kept_qt[src] = list(range(len(rows)))
    (tmp_path / "decontam" / "kept.json").write_text(json.dumps(kept))
    (tmp_path / "decontam" / "kept_querytext.json").write_text(json.dumps(kept_qt))

    k8s = tmp_path / "k8s.jsonl"
    with open(k8s, "w", encoding="utf-8") as f:
        for p, t in [("content/en/docs/a.md", "Kubernetes pods run containers."),
                     ("content/en/docs/b.md", "Kubernetes pods run containers."),
                     ("content/en/docs/test.md", "excluded near duplicate")]:
            f.write(json.dumps({"path": p, "title": p, "text": t, "sha256": "x"}) + "\n")
    step2a = tmp_path / "step2a.json"
    step2a.write_text(json.dumps({"decontamination": {"result": {"flagged_paths": list(flagged)}}}))

    monkeypatch.setattr(S, "WORK", tmp_path)
    monkeypatch.setattr(S, "TRAIN", train)
    monkeypatch.setattr(S, "MANIFEST_DIR", tmp_path / "manifest")
    monkeypatch.setattr(S, "K8S_JSONL", k8s)
    monkeypatch.setattr(S, "K8S_STEP2A", step2a)
    monkeypatch.setattr(S, "RESULT", tmp_path / "m17_support_manifest.json")
    return tmp_path / "manifest"


# --------------------------------------------------------------------------- primitives

def test_normalization_is_nfkc_casefold_and_whitespace_collapsed():
    assert S.normalize("  K8s  INGRESS\n") == S.normalize("k8s ingress")
    assert S.group_id(S.normalize("A  b")) == S.group_id(S.normalize("a b"))


def test_distinct_text_gets_a_distinct_group():
    assert S.group_id(S.normalize("a b")) != S.group_id(S.normalize("a c"))


def test_union_find_is_transitive_and_order_independent():
    u = S.Union()
    for k in "abcd":
        u.add(k)
    u.union("a", "b")
    u.union("c", "d")
    u.union("d", "b")
    assert len({u.find(k) for k in "abcd"}) == 1


def test_denied_source_names_cover_msmarco():
    assert {"msmarco-train", "msmarco-pos"} <= S.DENIED_SOURCES


def test_every_mapped_domain_is_a_panel_domain():
    assert set(S.SOURCE_DOMAIN.values()) <= set(S.PANEL_DOMAINS)
    assert S.DEFAULT_DOMAIN in S.PANEL_DOMAINS


# ------------------------------------------------------------- ruling A3 document domains

MEDICAL = "The patient's disease required clinical therapy from a physician."
BLAND = "A short note about a thing that happened somewhere on a day."


def test_mapped_source_keeps_its_domain_whatever_the_text_says():
    assert S.document_domain("k8s-docs-en", MEDICAL) == "cloud-software"
    assert S.document_domain("k8s-docs-en", BLAND) == "cloud-software"


def test_general_source_with_medical_text_is_classified_as_medicine():
    assert S.SOURCE_DOMAIN["squad-train"] == "general"
    assert S.document_domain("squad-train", MEDICAL) == "medicine"


def test_general_source_with_bland_text_stays_general():
    assert S.document_domain("squad-train", BLAND) == "general"
    assert S.document_domain("squad-train", "") == "general"


def _pin(monkeypatch, tmp_path, min_score):
    p = tmp_path / "panel_manifest.json"
    p.write_text(json.dumps({"classifier_min_score": min_score}))
    monkeypatch.setattr(S, "PANEL_MANIFEST", p)
    monkeypatch.setattr(S, "_PINNED_MIN_SCORE", None)
    return p


def test_document_domain_matches_the_panel_classifier_at_the_same_threshold(tmp_path,
                                                                            monkeypatch):
    import panel_build as P
    _pin(monkeypatch, tmp_path, P.MIN_SCORE)
    assert S.classifier_min_score() == P.MIN_SCORE
    assert S.domain_method() == P.DOMAIN_METHOD
    for text in (MEDICAL, BLAND, "inflation and monetary policy raised the interest rate"):
        assert S.document_domain("squad-train", text) == P.classify(text, "general")[0]
    # and the threshold is honoured, not hard-coded: an unreachable one falls back to the map
    assert S.document_domain("squad-train", MEDICAL, min_score=1000) == "general"
    assert S.document_domain("squad-train", MEDICAL, min_score=2) == "medicine"


def test_the_pinned_panel_threshold_wins_over_the_module_default(tmp_path, monkeypatch):
    """Ruling A3's "same threshold" is the one the sealed panel pinned, not panel_build's own
    default: at 3 a document the default (4) leaves 'general' is labelled."""
    import panel_build as P
    text = "A treaty on trade law was signed."          # legal scores exactly 3
    _pin(monkeypatch, tmp_path, 4)
    assert S.classifier_min_score() == 4
    default_label = S.document_domain("squad-train", text)
    _pin(monkeypatch, tmp_path, 3)
    ms, src = S.classifier_min_score_with_source()
    assert ms == 3 and src.endswith("panel_manifest.json")
    assert S.document_domain("squad-train", text) == P.classify(text, "general", 3)[0]
    assert S.document_domain("squad-train", text) != default_label


def test_absent_panel_manifest_falls_back_to_the_module_default(tmp_path, monkeypatch):
    import panel_build as P
    monkeypatch.setattr(S, "PANEL_MANIFEST", tmp_path / "absent.json")
    monkeypatch.setattr(S, "_PINNED_MIN_SCORE", None)
    ms, src = S.classifier_min_score_with_source()
    assert ms == P.MIN_SCORE and "module default" in src


def test_unmapped_source_falls_back_to_the_general_branch():
    assert "brand-new-source" not in S.SOURCE_DOMAIN
    assert S.document_domain("brand-new-source", MEDICAL) == "medicine"


def test_document_pass_assigns_deduplicated_documents_one_at_a_time(tmp_path, monkeypatch):
    out = _world(tmp_path, monkeypatch,
                 docs={"squad-train": [("a", MEDICAL), ("b", MEDICAL.upper()), ("c", BLAND)]})
    per_source, _ = S.document_pass(out, hub_fanout=50)
    rec = per_source["squad-train"]
    assert rec["documents_by_domain"] == {"general": 1, "medicine": 1}, (
        "the duplicate must be classified once, so the counts partition the deduplicated total")
    assert sum(rec["documents_by_domain"].values()) == rec["documents_deduplicated"] == 2
    k8s = per_source["k8s-docs-en"]
    assert k8s["documents_by_domain"] == {"cloud-software": k8s["documents_deduplicated"]}


def test_per_document_rollup_populates_a_domain_the_source_map_cannot(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch,
           docs={"squad-train": [("a", MEDICAL), ("c", BLAND)]})
    res = S.build(hub_fanout=50, reuse_counts=False)
    assert res["per_domain"]["medicine"]["documents_deduplicated"] == 1
    assert "medicine" not in res["unpopulated_domains_gap"]
    # the step-2b view is still readable beside it
    assert res["per_domain_source_level"]["medicine"]["documents_deduplicated"] == 0
    assert "medicine" in res["unpopulated_domains_gap_source_level"]
    assert res["domain_assignment"]["applied_per_document"] is True
    assert res["domain_assignment"]["classifier_min_score"] == S.classifier_min_score()
    assert "HEURISTIC" in res["domain_assignment"]["method"]
    assert "medicine" in res["breadth_note"]


def test_reuse_counts_without_the_new_key_falls_back_instead_of_crashing(
        tmp_path, monkeypatch, capsys):
    out = _world(tmp_path, monkeypatch)
    S.build(hub_fanout=50, reuse_counts=False)
    # a counts_cache.json written before ruling A3: no documents_by_domain anywhere
    (out / "counts_cache.json").write_text(json.dumps({
        "per_source": dict(
            {s: {"documents_deduplicated": 3, "queries_deduplicated": 1} for s in S.PAIR_SOURCES},
            **{"k8s-docs-en": {"documents_deduplicated": 5, "queries_deduplicated": 0}}),
        "family_stats": {"queries_deduplicated_total": 100, "families": 100,
                         "queries_total_train": 100, "largest_family": 1,
                         "hub_documents_excluded_from_linking": 0, "hub_fanout": 50}}))
    monkeypatch.setattr(S, "TRAIN", tmp_path / "does-not-exist")
    res = S.build(hub_fanout=50, reuse_counts=True)
    assert "document pass" in capsys.readouterr().out
    assert res["domain_assignment"]["applied_per_document"] is False
    assert res["domain_assignment"]["fallback"]
    assert res["per_domain"] == res["per_domain_source_level"], "fallback is the source-level view"
    assert res["per_domain"]["general"]["documents_deduplicated"] == 3 * len(S.PAIR_SOURCES)


# --------------------------------------------------------------------------- passes

def test_document_pass_deduplicates_and_drops_flagged_k8s_paths(tmp_path, monkeypatch):
    out = _world(tmp_path, monkeypatch,
                 docs={"squad-train": [("a", "same text"), ("b", "Same  TEXT"), ("c", "other")]},
                 flagged=["content/en/docs/test.md"])
    per_source, doc_group = S.document_pass(out, hub_fanout=50)
    assert per_source["squad-train"]["documents"] == 3
    assert per_source["squad-train"]["documents_deduplicated"] == 2
    assert doc_group["squad-train"]["a"] == doc_group["squad-train"]["b"]
    # a.md and b.md carry identical text; test.md is excluded by path.
    assert per_source["k8s-docs-en"]["documents"] == 2
    assert per_source["k8s-docs-en"]["documents_deduplicated"] == 1
    rows = gzip.open(out / "doc_groups" / "squad-train.tsv.gz", "rt").read().strip().split("\n")
    assert len(rows) == 3


def test_k8s_is_candidate_only_until_the_executor_screen(tmp_path, monkeypatch):
    out = _world(tmp_path, monkeypatch)
    per_source, _ = S.document_pass(out, hub_fanout=50)
    assert "protected screen" in per_source["k8s-docs-en"]["admission"]


def test_query_pass_counts_the_decontaminated_train_split_only(tmp_path, monkeypatch):
    kept_qid, dropped_qid = "1", "2"
    out = _world(tmp_path, monkeypatch,
                 docs={"squad-train": [("d1", "doc one")]},
                 pairs={"squad-train": [{"qid": kept_qid, "query": "alpha", "pos": ["d1"]}]})
    # the dropped qid never enters kept.json, so it must not be counted
    src = json.loads((tmp_path / "train" / "sources" / "squad-train.json").read_text())
    src["pairs"].append({"qid": dropped_qid, "query": "beta", "pos": ["d1"]})
    (tmp_path / "train" / "sources" / "squad-train.json").write_text(json.dumps(src))
    per_source, doc_group = S.document_pass(out, hub_fanout=50)
    per_source, stats = S.query_pass(out, per_source, doc_group, hub_fanout=50)
    assert per_source["squad-train"]["queries_train"] == 1
    assert stats["queries_total_train"] == 1


def test_identical_query_text_shares_a_family(tmp_path, monkeypatch):
    out = _world(tmp_path, monkeypatch,
                 docs={"squad-train": [("d1", "one"), ("d2", "two")]},
                 pairs={"squad-train": [{"qid": "1", "query": "Alpha Beta", "pos": ["d1"]},
                                        {"qid": "2", "query": "alpha  beta", "pos": ["d2"]}]})
    per_source, doc_group = S.document_pass(out, hub_fanout=50)
    _, stats = S.query_pass(out, per_source, doc_group, hub_fanout=50)
    assert stats["families"] == 1


def test_shared_positive_document_joins_a_family(tmp_path, monkeypatch):
    out = _world(tmp_path, monkeypatch,
                 docs={"squad-train": [("d1", "one")]},
                 pairs={"squad-train": [{"qid": "1", "query": "alpha", "pos": ["d1"]},
                                        {"qid": "2", "query": "beta", "pos": ["d1"]}]})
    per_source, doc_group = S.document_pass(out, hub_fanout=50)
    _, stats = S.query_pass(out, per_source, doc_group, hub_fanout=50)
    assert stats["families"] == 1


def test_hub_document_does_not_collapse_everything_into_one_family(tmp_path, monkeypatch):
    ps = [{"qid": str(i), "query": f"q{i}", "pos": ["hub"]} for i in range(5)]
    out = _world(tmp_path, monkeypatch,
                 docs={"squad-train": [("hub", "a much reused document")]},
                 pairs={"squad-train": ps})
    per_source, doc_group = S.document_pass(out, hub_fanout=2)
    _, stats = S.query_pass(out, per_source, doc_group, hub_fanout=2)
    assert stats["hub_documents_excluded_from_linking"] == 1
    assert stats["families"] == 5, "a hub must link nothing"


def test_query_only_sources_are_counted_and_deduplicated(tmp_path, monkeypatch):
    out = _world(tmp_path, monkeypatch, querytext={"nqopen": ["who is x", "WHO IS X", "other"]})
    per_source, doc_group = S.document_pass(out, hub_fanout=50)
    per_source, stats = S.query_pass(out, per_source, doc_group, hub_fanout=50)
    kept = [i for i in range(3) if not S.heldout("nqopen", str(i))]
    assert per_source["nqopen"]["queries_train"] == len(kept)
    assert per_source["nqopen"]["documents"] == 0


# --------------------------------------------------------------------------- dose rule

def test_registered_dose_passes_are_the_registry_arithmetic():
    p = S.passes_at_registered_dose(576000, 192000, 48000)
    assert p["general"] == 2.0 and p["unpaired_coverage"] == 1.0 and p["alias_pairs"] == 2.0
    assert p["exceeds_four_passes"] == []


def test_ample_populations_leave_the_registered_shares_untouched():
    d = S.dose_rule(576000, 6000000, 48000)
    assert d["resulting_shares"]["steps"] == 6000
    assert d["resulting_shares"]["general_views_per_batch"] == 192
    assert d["resulting_shares"]["alias_pairs_per_batch"] == 16
    assert not any(d["shrunk"].values())


def test_a_thin_alias_pool_shrinks_the_alias_share_to_its_floor():
    d = S.dose_rule(576000, 6000000, 500)
    assert d["resulting_shares"]["alias_pairs_per_batch"] == 4, "registered floor is 4 pairs"
    assert d["shrunk"]["alias"]
    assert d["resulting_shares"]["general_views_per_batch"] == 192 + (32 - 8)


def test_shrinks_never_raise_steps_above_six_thousand():
    d = S.dose_rule(10 ** 9, 10 ** 9, 10 ** 9)
    assert d["resulting_shares"]["steps"] == 6000


def test_a_thin_general_pool_reduces_steps_last():
    d = S.dose_rule(10000, 6000000, 48000)
    r = d["resulting_shares"]
    assert d["shrunk"]["steps"] and r["steps"] < 6000
    assert r["general_views"] / 10000 <= 4 + 1e-9


def test_batch_shares_always_sum_to_the_batch():
    for alias_pop in (500, 5000, 48000):
        r = S.dose_rule(576000, 6000000, alias_pop)["resulting_shares"]
        assert (r["general_views_per_batch"] + r["unpaired_coverage_views_per_batch"]
                + 2 * r["alias_pairs_per_batch"]) == 256


def test_zero_alias_pairs_is_the_floor_not_a_crash():
    r = S.dose_rule(576000, 6000000, 0)["resulting_shares"]
    assert r["alias_pairs_per_batch"] == 4


def test_new_source_share_is_measured_on_processed_views():
    n = S.new_source_share(1536000, 192000, 1648)
    assert n["new_source_view_cap"] == 153600
    assert n["cap_binds_before_the_coverage_bucket"] is True
    assert n["passes_over_the_new_source_if_it_filled_the_cap"] > 4, (
        "a 1648-document slice cannot fill a 153600-view share without heavy repetition")
    assert n["new_source_views_at_four_passes"] == 6592


def test_coverage_population_separates_the_new_source_half():
    per = {s: {"documents_deduplicated": 10} for s in S.PAIR_SOURCES}
    per["k8s-docs-en"] = {"documents_deduplicated": 7}
    cov = S.coverage_population(per)
    assert cov["new_source_views_min"] == 7
    assert cov["existing_source_views_min"] == 50
    assert cov["total_min"] == 57


def test_unpopulated_panel_domains_are_reported_as_gaps(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    res = S.build(hub_fanout=50, reuse_counts=False)
    assert set(res["unpopulated_domains_gap"]) == {
        "science-engineering", "medicine", "finance", "legal"}
    assert res["per_domain"]["cloud-software"]["sources"] == ["k8s-docs-en"]
    assert res["protected_payloads_opened"] is False and res["gpu_used"] is False


def test_result_carries_no_document_or_query_text(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch,
           docs={"squad-train": [("d1", "SENTINELDOCTEXT here")]},
           pairs={"squad-train": [{"qid": "1", "query": "SENTINELQUERYTEXT", "pos": ["d1"]}]})
    res = S.build(hub_fanout=50, reuse_counts=False)
    blob = json.dumps(res)
    assert "SENTINELDOCTEXT" not in blob and "SENTINELQUERYTEXT" not in blob


def test_reuse_counts_reapplies_the_rule_without_the_heavy_pass(tmp_path, monkeypatch):
    out = _world(tmp_path, monkeypatch)
    S.build(hub_fanout=50, reuse_counts=False)
    (out / "counts_cache.json").write_text(json.dumps({
        "per_source": dict(
            {s: {"documents_deduplicated": 1, "queries_deduplicated": 1} for s in S.PAIR_SOURCES},
            **{"k8s-docs-en": {"documents_deduplicated": 5, "queries_deduplicated": 0}}),
        "family_stats": {"queries_deduplicated_total": 100, "families": 100,
                         "queries_total_train": 100, "largest_family": 1,
                         "hub_documents_excluded_from_linking": 0, "hub_fanout": 50}}))
    monkeypatch.setattr(S, "TRAIN", tmp_path / "does-not-exist")
    res = S.build(hub_fanout=50, reuse_counts=True)
    assert res["bucket_populations"]["general"]["population"] == 100


def test_missing_decontamination_survivors_fail_loudly(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    (tmp_path / "decontam" / "kept.json").unlink()
    with pytest.raises(FileNotFoundError):
        S.load_kept()


def test_load_exclusions_accepts_the_panel_steps_schema(tmp_path):
    f = tmp_path / "alias_test_families.json"
    f.write_text(json.dumps({"excluded_family_ids": ["fam:a"], "excluded_text_shas": ["abc"],
                             "excluded_alias_terms": ["Pod Disruption Budget", "PDB"],
                             "excluded_evidence_doc_groups": ["deadbeefdeadbeef"],
                             "key_conventions": "sha256(normalized text)[:16]"}))
    ex = S.load_exclusions(f)
    assert ex["families"] == {"fam:a"} and ex["text_shas"] == {"abc"}
    assert ex["terms"] == {"pod disruption budget", "pdb"}       # normalized on load
    assert ex["doc_groups"] == {"deadbeefdeadbeef"}
    absent = S.load_exclusions(tmp_path / "absent.json")
    assert all(v == set() for v in absent.values())


def test_an_exclusion_file_without_the_term_keys_is_refused(tmp_path):
    """Family ids and view shas alone removed nothing from the real pool; a file that carries
    only them is a silent no-op, so it is refused instead of applied."""
    f = tmp_path / "old.json"
    f.write_text(json.dumps({"excluded_family_ids": ["fam:a"], "excluded_text_shas": ["abc"]}))
    with pytest.raises(SystemExit, match="excluded_alias_terms"):
        S.load_exclusions(f)


def test_iter_store_refuses_ms_marco_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "TRAIN", tmp_path)
    (tmp_path / "stores").mkdir()
    (tmp_path / "stores" / "MSMARCO-pos.json").write_text(
        json.dumps({"ids": ["d0"], "texts": ["never a training input"]}))
    with pytest.raises(SystemExit, match="MS MARCO"):
        list(S.iter_store("MSMARCO-pos"))


def test_heldout_query_text_is_removed_from_the_training_population(tmp_path, monkeypatch):
    out = _world(tmp_path, monkeypatch,
                 docs={"squad-train": [("d1", "one")]},
                 pairs={"squad-train": [{"qid": "1", "query": "keep me", "pos": ["d1"]},
                                        {"qid": "3", "query": "hold me out", "pos": ["d1"]}]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "alias_test_families.json").write_text(json.dumps(
        {"excluded_text_shas": [S.group_id(S.normalize("hold me out"))],
         "excluded_alias_terms": [], "excluded_evidence_doc_groups": []}))
    per_source, doc_group = S.document_pass(out, hub_fanout=50)
    per_source, stats = S.query_pass(out, per_source, doc_group, hub_fanout=50)
    assert stats["queries_removed_as_heldout"] == 1
    assert per_source["squad-train"]["queries_train"] == 1
    assert stats["heldout_keys_available"] is True


def test_a_bare_held_out_alias_term_is_dropped_from_the_training_queries(tmp_path, monkeypatch):
    """A training query that IS the held-out short form or expansion is held-out material even
    though its text sha is not in the pair list."""
    out = _world(tmp_path, monkeypatch,
                 docs={"squad-train": [("d1", "one")]},
                 pairs={"squad-train": [{"qid": "1", "query": "keep me", "pos": ["d1"]},
                                        {"qid": "2", "query": "Pod Disruption Budget",
                                         "pos": ["d1"]}]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "alias_test_families.json").write_text(json.dumps(
        {"excluded_text_shas": [], "excluded_evidence_doc_groups": [],
         "excluded_alias_terms": ["Pod Disruption Budget", "PDB"]}))
    per_source, doc_group = S.document_pass(out, hub_fanout=50)
    per_source, stats = S.query_pass(out, per_source, doc_group, hub_fanout=50)
    assert stats["queries_removed_as_heldout_alias_term"] == 1
    assert per_source["squad-train"]["queries_train"] == 1
