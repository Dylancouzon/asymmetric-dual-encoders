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


def test_imperative_pr_title_is_not_concept_howto_audit_query():
    opening = {"kind": "pull_request_opening"}
    assert not protocol._credible_audit_query(opening, "fix: clear joint consensus fields", "concept_howto")
    assert protocol._credible_audit_query(opening, "Why are joint consensus fields retained?", "concept_howto")
    assert protocol._credible_audit_query(opening, "fix: timeout 500 in API", "error_troubleshooting")


def test_confirmation_cannot_use_general_surface_loader(tmp_path):
    with pytest.raises(SystemExit, match="run_confirmation"):
        protocol.load_surface("confirmation", tmp_path)


def test_project_compounds_are_not_mistaken_for_generated_suffixes():
    assert not vocab.is_near_unique("scalar-quantization")
    assert not vocab.is_near_unique("collection-configuration")
    assert vocab.is_near_unique("qdrant-7f8c9d2a")
    assert vocab.is_near_unique("qdrant-7d9f8c7b6c-abc12")
