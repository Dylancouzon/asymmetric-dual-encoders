"""Leakage and sealed-read contract tests for the M18 protocol."""
from __future__ import annotations

import hashlib
import json

import pytest

import protocol
import vocab


def _digest(text):
    return hashlib.sha256(protocol.re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()


def _candidate(i, stratum="concept_howto"):
    return {"query_id": f"q{i}", "text": f"How does feature {i} work", "family": f"gh:thread:{i}",
            "family_group": f"f{i}", "family_members": [f"gh:thread:{i}"],
            "near_duplicate_family": f"shape {i}", "source_doc": f"o{i}",
            "target_doc": f"a{i}", "timestamp": f"2025-01-{i:02d}T00:00:00Z",
            "stratum": stratum}


def test_exactly_35_families_realizes_required_25_10_split():
    rows = [_candidate(i) for i in range(1, 36)]
    reg = {"strata": ["concept_howto"], "split": {
        "development_per_stratum": 35, "confirmation_per_stratum": 15,
        "minimum_development_per_available_stratum": 25,
        "minimum_confirmation_per_available_stratum": 10}}
    train, dev, conf, realized = protocol._split(rows, reg)
    assert not train and len(dev) == 25 and len(conf) == 10
    assert not ({q["family_group"] for q in dev} & {q["family_group"] for q in conf})
    assert realized["concept_howto"]["development"] == 25


def test_duplicate_answer_and_linked_threads_form_one_union_family():
    answer = "Use indexed payload filtering because it resolves this distinct query behavior."
    corpus = [
        {"doc_id": "a1", "artifact_id": "gh:thread:1", "text": answer,
         "normalized_text_sha256": _digest(answer), "outbound_links": []},
        {"doc_id": "a2", "artifact_id": "gh:thread:2", "text": answer,
         "normalized_text_sha256": _digest(answer),
         "outbound_links": ["https://github.com/qdrant/qdrant/issues/3"]},
        {"doc_id": "a3", "artifact_id": "gh:thread:3", "text": "another answer",
         "normalized_text_sha256": _digest("another answer"), "outbound_links": []},
    ]
    candidates = []
    for i in (1, 2, 3):
        q = _candidate(i); q["target_doc"] = f"a{i}"; candidates.append(q)
    got = protocol._assign_union_families(candidates, corpus, {r["doc_id"]: r for r in corpus})
    assert len({q["family_group"] for q in got}) == 1
    assert got[0]["family_members"] == ["gh:thread:1", "gh:thread:2", "gh:thread:3"]


def test_generic_log_request_is_not_an_answer():
    opening = {"doc_id": "o", "kind": "issue_opening", "artifact_id": "gh:thread:1",
               "github_number": 1, "title": "Why does my collection fail", "text": "question",
               "timestamp": "2025-01-01T00:00:00Z", "chunk_index": 1,
               "normalized_text_sha256": _digest("question"), "outbound_links": []}
    request = {"doc_id": "a", "kind": "issue_comment", "artifact_id": "gh:thread:1",
               "github_id": 2, "text": "You should provide the logs because we need more information to investigate this problem fully.",
               "timestamp": "2025-01-01T01:00:00Z", "author_association": "MEMBER",
               "normalized_text_sha256": _digest("request"), "outbound_links": []}
    assert protocol.structural_candidates([opening, request]) == []


def test_thread_closure_does_not_turn_clarification_into_answer():
    opening = {"doc_id": "o", "kind": "issue_opening", "artifact_id": "gh:thread:1",
               "github_number": 1, "title": "Snapshot restore panic", "text": "question",
               "timestamp": "2025-01-01T00:00:00Z", "closed_at": "2025-01-02T00:00:00Z",
               "state": "closed", "chunk_index": 1,
               "normalized_text_sha256": _digest("question"), "outbound_links": []}
    clarification = {"doc_id": "a", "kind": "issue_comment", "artifact_id": "gh:thread:1",
                     "github_id": 2,
                     "text": "What versions did you use, and can you share enough data for us to reproduce this problem?",
                     "timestamp": "2025-01-01T01:00:00Z", "author_association": "MEMBER",
                     "normalized_text_sha256": _digest("clarification"), "outbound_links": []}
    assert protocol.structural_candidates([opening, clarification]) == []


def test_status_only_fix_is_not_answer_to_unrelated_pr_title():
    ev = {"explicit_resolution": True, "thread_closed_after_answer": True,
          "project_link": False, "closing_or_link_event": True}
    opening = {"title": "Clear joint consensus fields on first peer reinit"}
    assert not protocol._credible_issue_answer(
        opening, {"text": "Fixed the Codespell typo and pushed it; CI needs maintainer approval."}, ev)


@pytest.mark.parametrize("text", [
    "Need logs because we cannot reproduce this failure without more details from the user.",
    "Is this actually fixed? Because I still see the same failure in my local environment.",
    "I do not think this is fixed because the same failure still occurs in production.",
])
def test_review_requests_and_non_resolutions_are_not_answers(text):
    assert not protocol._credible_review_answer({"text": text})


@pytest.mark.parametrize("text", [
    "Is this actually fixed? Because I still see the same failure in my local environment.",
    "I do not think this is fixed because the same failure still occurs in production.",
])
def test_non_resolution_does_not_pass_closed_issue_gate(text):
    ev = {"explicit_resolution": True, "thread_closed_after_answer": True,
          "project_link": False, "closing_or_link_event": True}
    assert not protocol._credible_issue_answer({"title": "Production failure"}, {"text": text}, ev)


def test_imperative_pr_title_is_not_concept_howto_audit_query():
    opening = {"kind": "pull_request_opening"}
    assert not protocol._credible_audit_query(opening, "fix: clear joint consensus fields", "concept_howto")
    assert protocol._credible_audit_query(opening, "Why are joint consensus fields retained?", "concept_howto")
    assert protocol._credible_audit_query(opening, "fix: timeout 500 in API", "error_troubleshooting")


def test_context_dependent_review_question_is_not_concept_candidate():
    parent = {"doc_id": "p", "kind": "review_comment", "artifact_id": "gh:thread:1",
              "github_id": 1, "text": "Can't we return a reference here?",
              "timestamp": "2025-01-01T00:00:00Z", "author_association": "MEMBER",
              "normalized_text_sha256": _digest("parent"), "outbound_links": []}
    reply = {"doc_id": "r", "kind": "review_comment", "artifact_id": "gh:thread:1",
             "github_id": 2, "in_reply_to_id": 1,
             "text": "We should return an owned value because the segment lifetime is shorter than the collection lifetime.",
             "timestamp": "2025-01-01T01:00:00Z", "author_association": "MEMBER",
             "normalized_text_sha256": _digest("reply"), "outbound_links": []}
    assert protocol.structural_candidates([parent, reply]) == []


def test_code_question_operator_is_not_a_prose_question():
    assert not protocol._has_prose_question("Consider this change:\n```rust\nlet x = value?;\n```")
    assert protocol._has_prose_question("Should this propagate the error?\n```rust\nvalue?\n```")


def test_duplicate_source_digest_unions_candidate_families():
    rows = [_candidate(1), _candidate(2)]
    corpus = []
    for i in (1, 2):
        corpus.extend([
            {"doc_id": f"o{i}", "artifact_id": f"gh:thread:{i}", "text": "same question",
             "normalized_text_sha256": _digest("same question"), "outbound_links": []},
            {"doc_id": f"a{i}", "artifact_id": f"gh:thread:{i}", "text": f"answer {i}",
             "normalized_text_sha256": _digest(f"answer {i}"), "outbound_links": []},
        ])
    got = protocol._assign_union_families(rows, corpus, {r["doc_id"]: r for r in corpus})
    assert got[0]["family_group"] == got[1]["family_group"]


def test_adjudication_is_exact_and_content_bound(tmp_path):
    q = _candidate(1)
    q["label_provenance"] = {"rule": "fixture"}
    corpus = {
        "o1": {"doc_id": "o1", "kind": "issue_opening", "text": "question body", "path": None},
        "a1": {"doc_id": "a1", "kind": "issue_comment", "text": "direct answer", "path": None},
    }
    decision_path = tmp_path / "decisions.jsonl"
    reg = {"strata": ["concept_howto"], "evaluation": {"qrel_adjudication": {
        "pool_per_stratum": 2, "decisions_path": str(decision_path), "judge": "fixture"}}}
    sha = protocol.adjudication_record(q, corpus)["candidate_sha256"]
    decision_path.write_text(json.dumps({"query_id": "q1", "candidate_sha256": sha,
        "label": "clear", "stratum": "concept_howto", "reason": "direct answer"}) + "\n")
    accepted, meta = protocol.apply_adjudications([q], reg, corpus)
    assert [x["query_id"] for x in accepted] == ["q1"] and meta["accepted_clear"] == 1
    decision_path.write_text(decision_path.read_text().replace(sha, "0" * 64))
    with pytest.raises(SystemExit, match="stale adjudication"):
        protocol.apply_adjudications([q], reg, corpus)


def test_unjudged_titles_are_query_only_and_heldout_families_are_excluded():
    issue, heldout, review = _candidate(1), _candidate(2), _candidate(3)
    corpus = {
        "o1": {"kind": "issue_opening"}, "o2": {"kind": "pull_request_opening"},
        "o3": {"kind": "review_comment"},
    }
    got = protocol._query_only_training([issue, heldout, review], [], {heldout["family_group"]}, corpus)
    assert [q["query_id"] for q in got] == [issue["query_id"]]
    assert got[0]["target_doc"] is None and not got[0]["label_provenance"]["positive_label"]


def test_confirmation_cannot_use_general_surface_loader(tmp_path):
    with pytest.raises(SystemExit, match="run_confirmation"):
        protocol.load_surface("confirmation", tmp_path)


def test_project_compounds_are_not_mistaken_for_generated_suffixes():
    assert not vocab.is_near_unique("scalar-quantization")
    assert not vocab.is_near_unique("collection-configuration")
    assert vocab.is_near_unique("qdrant-7f8c9d2a")
    assert vocab.is_near_unique("qdrant-7d9f8c7b6c-abc12")
