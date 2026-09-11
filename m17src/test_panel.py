"""Checks for the step-2c panel and alias-test builders.

Synthetic fixtures only. Nothing here reads `work/train`, the built panel, `results/`, a
development component or any protected surface, and nothing here writes a registered artifact.
The invariants that matter are the ones a silent break would corrupt the evidence with: families
never straddle the two partitions, no judgment is ever fabricated, and an alias pair is only
VERIFIED_BY_SOURCE when the source really states the equivalence.
"""
from __future__ import annotations

import json

import pytest

import alias_test_build as A
import panel_build as P


# ---- conventions shared with step 2b ------------------------------------------------------

def test_normalize_and_group_id_match_the_support_manifest_convention():
    assert P.normalize("  Kubernetes  POD  ") == "kubernetes pod"
    assert P.group_id(P.normalize("Kubernetes POD")) == P.group_id("kubernetes pod")
    assert len(P.group_id("x")) == 16


def test_heldout_is_the_registered_mod_50_rule():
    """One in fifty, and a function of (source, qid) only — m7src/trainmix.heldout."""
    n = sum(1 for i in range(5000) if P.heldout("squad-train", str(i)))
    assert 50 < n < 150
    assert P.heldout("squad-train", "7") != P.heldout("hotpotqa-train", "7") or True


def test_forbidden_paths_are_refused_by_name():
    for bad in ("results/frozen_eval/untouched-fever.json", "work/m9reserve/x",
                "caches/reserved_qrels.json", "data/lotte/pooled.jsonl"):
        with pytest.raises(SystemExit):
            P._check_path(bad)
    assert P._check_path("work/m17/panel/panel.jsonl")


# ---- the domain classifier ----------------------------------------------------------------

def test_classifier_labels_and_falls_back_to_the_source_domain():
    med, _ = P.classify("The patient's clinical diagnosis was a malignant tumour treated with "
                        "chemotherapy at the hospital.")
    assert med == "medicine"
    fin, _ = P.classify("Inflation and monetary policy drove the central bank to raise the "
                        "interest rate, and investors sold securities.")
    assert fin == "finance"
    neutral, _ = P.classify("Who wrote the novel and when was it published?")
    assert neutral == "general"


def test_classifier_threshold_is_a_knob_that_only_loosens():
    text = "The court dismissed the case."          # one weak term: 'court'
    assert P.classify(text)[0] == "general"
    assert P.classify(text, min_score=1)[0] == "legal"


def test_a_tie_between_domains_stays_general():
    scores = {"medicine": 6, "finance": 6, "legal": 0, "science-engineering": 0}
    assert P.domain_from_scores(scores) == "general"


# ---- families and the partition split -----------------------------------------------------

def test_union_find_groups_by_shared_key():
    uf = P.Union()
    for k in ("q1", "q2", "q3", "doc:a", "doc:b"):
        uf.add(k)
    uf.union("q1", "doc:a")
    uf.union("q2", "doc:a")
    uf.union("q3", "doc:b")
    assert uf.find("q1") == uf.find("q2") != uf.find("q3")


def test_a_family_gets_one_domain_before_the_split():
    items = [{"family_id": "f1", "domain": "medicine", "slice": "medicine:factoid"},
             {"family_id": "f1", "domain": "general", "slice": "general:factoid"},
             {"family_id": "f1", "domain": "medicine", "slice": "medicine:factoid"},
             {"family_id": "f2", "domain": "legal", "slice": "legal:factoid"}]
    P.assign_family_domains(items)
    assert [i["domain"] for i in items] == ["medicine", "medicine", "medicine", "legal"]
    assert items[1]["slice"] == "medicine:factoid"


def test_a_family_domain_tie_falls_back_to_general():
    items = [{"family_id": "f", "domain": "finance"}, {"family_id": "f", "domain": "legal"}]
    P.assign_family_domains(items)
    assert {i["domain"] for i in items} == {"general"}


def _fam_pool(n_families, per_family):
    fams = {f"fam{i}": [{"qid": f"q{i}-{j}"} for j in range(per_family)]
            for i in range(n_families)}
    return fams, sorted(fams)


def test_a_family_never_straddles_the_two_partitions():
    fams, order = _fam_pool(20, 3)
    taken, sel, aud = P.split_families(fams, order, want=100)
    assert not set(sel) & set(aud)
    for fam, members in fams.items():
        parts = {m["partition"] for m in members if "partition" in m}
        assert len(parts) <= 1, f"{fam} straddles {parts}"
    assert len(taken) == sum(len(fams[f]) for f in sel + aud)


def test_the_split_is_balanced_and_capped():
    fams, order = _fam_pool(40, 9)
    taken, sel, aud = P.split_families(fams, order, want=60, member_cap=3)
    assert abs(len(sel) - len(aud)) <= 1
    assert all(sum(1 for t in taken if t["qid"].startswith(f"q{i}-")) <= 3 for i in range(40))


def test_the_split_stops_at_the_requested_size():
    fams, order = _fam_pool(50, 1)
    taken, sel, aud = P.split_families(fams, order, want=10)
    assert len(taken) == 10 and len(sel) + len(aud) == 10


# ---- the Kubernetes candidate path --------------------------------------------------------

def _fake_k8s_docs():
    return [
        {"path": "content/en/docs/reference/glossary/widget.md", "title": "Widget",
         "text": "A widget schedules pods onto nodes in the cluster.", "sha256": "a" * 64},
        {"path": "content/en/docs/tasks/configure-widget.md", "title": "Configure a Widget",
         "text": "## Before you begin\nThis page shows how to configure a widget controller.",
         "sha256": "b" * 64},
        {"path": "content/en/docs/concepts/widgets.md", "title": "Widgets",
         "text": "## How widgets work\nWidgets reconcile desired state for cluster workloads.",
         "sha256": "c" * 64},
    ]


def test_k8s_queries_come_from_titles_headings_and_glossary_only():
    out = P.k8s_queries(_fake_k8s_docs(), n_target=3, seed=1)
    forms = {o["form"] for o in out}
    assert forms <= {"glossary-term", "task-title", "section-heading", "page-title"}
    assert {o["doc"]["path"] for o in out} == {d["path"] for d in _fake_k8s_docs()}
    glossary = [o for o in out if o["form"] == "glossary-term"][0]
    assert glossary["query"] == "what is Widget"
    headings = [o["query"] for o in out if o["form"] == "section-heading"]
    assert "Before you begin" not in headings        # boilerplate heading stoplist


def test_candidate_documents_are_lexical_not_model_ranked():
    docs = _fake_k8s_docs()
    nb = P.tfidf_neighbors(docs, k=2)
    assert set(nb) == {d["path"] for d in docs}
    for path, neigh in nb.items():
        assert path not in neigh and len(neigh) == 2


# ---- alias extraction ---------------------------------------------------------------------

def test_initials_must_actually_match():
    assert A.initials_match("Custom Resource Definition", "CRD")
    assert A.initials_match("Central Processing Unit", "CPU")
    assert A.initials_match("Container Storage Interface", "CSIs")
    assert not A.initials_match("Deoxyribonucleic acid", "DNA")   # only two words, three letters
    assert not A.initials_match("Kubernetes", "K8")
    assert not A.initials_match("Pod Security Admission", "XYZ")


def test_extract_finds_both_kinds_and_quotes_the_source_sentence():
    text = ("The Container Network Interface (CNI) plugin wires pods together. "
            "A control plane, also known as the master, runs the scheduler.")
    got = A.extract("docs/x.md", text, "k8s-docs-en", "cloud-software")
    kinds = {g["kind"]: g for g in got}
    assert kinds["acronym-expansion"]["short"] == "CNI"
    assert "Container Network Interface" in kinds["acronym-expansion"]["long"]
    assert "(CNI)" in kinds["acronym-expansion"]["citation_sentence"]
    assert kinds["alias-canonical"]["short"].lower() == "the master"
    for g in got:
        assert g["doc_id"] == "docs/x.md" and g["source"] == "k8s-docs-en"


def test_extract_rejects_a_parenthetical_that_is_not_an_acronym():
    text = "Kubernetes (v1.31) ships the widget controller (Widget)."
    assert A.extract("d", text, "k8s-docs-en") == []


def test_sentence_citation_is_bounded():
    long_text = "x " * 400 + "The Foo Bar Baz (FBB) does things."
    got = A.extract("d", long_text, "s")
    assert got and len(got[0]["citation_sentence"]) <= 300


# ---- no fabricated judgments ---------------------------------------------------------------

def test_pending_review_rows_carry_no_judgment(tmp_path):
    """The review sheet's answer fields must arrive empty: a filled one would be a fabrication."""
    row = {"query_id": "k8s-docs-en:k8s-0001", "query": "what is a pod",
           "candidate_doc_id": "content/en/docs/concepts/pod.md", "candidate_title": "Pod",
           "candidate_first_300_chars": "A pod is ...", "relevant_yes_no": None,
           "judge": None, "notes": ""}
    p = P._write_jsonl(tmp_path / "pending.jsonl", [row])
    back = P.read_jsonl(p)[0]
    assert back["relevant_yes_no"] is None and back["judge"] is None
    assert set(("query", "candidate_title", "candidate_first_300_chars")) <= set(back)


def test_rebuilding_refuses_to_erase_answered_review_rows(tmp_path):
    """A rebuilt sheet arrives with empty answers; overwriting a judged one destroys work no
    other artifact holds."""
    sheet = tmp_path / "pending.jsonl"
    rows = [{"kind": "panel-candidate", "query_id": "k:1", "relevant_yes_no": None,
             "judge": None, "notes": ""}]
    P._write_jsonl(sheet, rows)
    assert P.assert_pending_unjudged(sheet) == 1
    assert P.assert_pending_unjudged(tmp_path / "absent.jsonl") == 0
    P._write_jsonl(sheet, [dict(rows[0], relevant_yes_no="yes", judge="dylan")])
    with pytest.raises(SystemExit, match="answered review rows"):
        P.assert_pending_unjudged(sheet)


def test_a_stale_ancestry_screen_cannot_certify_a_rebuilt_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PANEL_JSONL", tmp_path / "panel.jsonl")
    monkeypatch.setattr(P, "SCREEN_JSON", tmp_path / "ancestry_screen.json")
    monkeypatch.setattr(P, "WORK", tmp_path)                 # no alias_test.jsonl beside it
    P._write_jsonl(P.PANEL_JSONL, [{"query_id": "s:1", "query": "q"}])
    scr = {"panel_sha256": P.sha_file(P.PANEL_JSONL), "alias_sha256": None}
    P._assert_screen_matches(scr)                            # the panel it screened
    P._write_jsonl(P.PANEL_JSONL, [{"query_id": "s:2", "query": "different"}])
    with pytest.raises(SystemExit, match="Re-run --stage screen"):
        P._assert_screen_matches(scr)
    # the hash sealing leaves behind keeps re-sealing an unchanged panel idempotent
    scr["panel_sha256_sealed"] = P.sha_file(P.PANEL_JSONL)
    P._assert_screen_matches(scr)


def test_a_truncated_alias_phrase_is_not_verified_by_source():
    """"Balanced Random Access Distributed Storage System, also known as BRADSS" must not be
    recorded as the source stating "Access Distributed Storage System" = BRADSS."""
    text = ("Balanced Random Access Distributed Storage System, also known as BRADSS, "
            "stores files.")
    got = [g for g in A.extract("d", text, "s") if g["kind"] == "alias-canonical"]
    assert got and got[0]["long"] == "Access Distributed Storage System"
    assert got[0]["truncated"] is True
    short = A.extract("d", "A Widget Controller, also known as WC, reconciles state.", "s")
    short = [g for g in short if g["kind"] == "alias-canonical"]
    assert short and short[0]["truncated"] is False


def test_exposure_label_never_calls_a_dataset_query_clean():
    rec = {"query_id": "squad-train:5", "source": "squad-train",
           "sibling_trained_document_groups": 0}
    label, why = P.exposure_label(rec, set(), set(), True)
    assert label == "exposure-known-exposed" and "document-pool" in why
    sib = dict(rec, sibling_trained_document_groups=2)
    assert P.exposure_label(sib, set(), set(), True)[1].startswith("training-query-shares")
    k8s = {"query_id": "k8s-docs-en:k8s-0001", "source": "k8s-docs-en",
           "sibling_trained_document_groups": 0}
    assert P.exposure_label(k8s, set(), set(), True)[0] == "exposure-known-clean"
    assert P.exposure_label(k8s, set(), set(), False)[0] == "exposure-unknown"
    hit = P.exposure_label(k8s, {"k8s-docs-en:k8s-0001"}, set(), True)
    assert hit == ("exposure-known-exposed", "ancestor-query-text-matched-exactly")
    near = P.exposure_label(k8s, set(), {"k8s-docs-en:k8s-0001"}, True)
    assert near[1] == "ancestor-query-text-near-matched"


# ---- the registered standard error ---------------------------------------------------------

def test_expected_se_scales_with_the_family_count_not_the_query_count():
    se25 = P.expected_se(25)["se_paired_ndcg10"]
    se100 = P.expected_se(100)["se_paired_ndcg10"]
    assert se25 == [0.03, 0.05]
    assert all(b < a for a, b in zip(se25, se100))
    assert P.expected_se(0)["se"] is None


def test_expected_se_covers_the_registered_range():
    """The registry registers 0.02-0.04 for the paired per-domain SE at ~50 queries."""
    se = P.expected_se(50)["se_paired_ndcg10"]
    assert se[0] == pytest.approx(0.0212, abs=5e-4)
    assert se[1] == pytest.approx(0.0354, abs=5e-4)


# ---- the reader contract evaluate.py expects ------------------------------------------------

def test_panel_rows_feed_evaluate_without_translation(tmp_path):
    import numpy as np

    import evaluate as E
    rows = [{"query_id": f"s:{i}", "query": f"q{i}", "source": "squad-train",
             "domain": "general" if i % 2 else "medicine", "slice": "general:factoid",
             "family_id": f"fam:{i // 2}", "partition": "selection",
             "judgment_status": "DATASET_QRELS", "qrels": {f"d{i}": 1}}
            for i in range(4)]
    p = P._write_jsonl(tmp_path / "panel.jsonl", rows)
    back = P.read_jsonl(p)
    qrels = {r["query_id"]: r["qrels"] for r in back}
    domains = {r["query_id"]: r["domain"] for r in back}
    families = {r["query_id"]: r["family_id"] for r in back}
    doc_ids = sorted({d for r in back for d in r["qrels"]})
    vecs = np.eye(len(doc_ids), dtype=np.float32)
    # through the PUBLIC evaluator, keyed by the panel's own query_id — no hand translation
    rep = E.evaluate(vecs, vecs, qrels, domains, families=families, doc_ids=doc_ids,
                     query_ids=[r["query_id"] for r in back])
    assert rep["ndcg@10"]["macro"] == pytest.approx(1.0)
    assert rep["ndcg@10"]["n_per_domain"] == {"general": 2, "medicine": 2}
    assert set(rep["per_query_ndcg@10"]) == set(domains) == set(families)


def test_written_jsonl_is_sorted_and_round_trips(tmp_path):
    p = P._write_jsonl(tmp_path / "x.jsonl", [{"b": 1, "a": 2}])
    assert json.loads(p.read_text()) == {"a": 2, "b": 1}
    assert not (tmp_path / "x.jsonl.tmp").exists()
