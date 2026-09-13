"""Evaluation-policy tests for the M18 experiment driver."""
from __future__ import annotations

import pytest

import experiment


def test_query_source_is_removed_without_marking_it_relevant():
    rows = [{"query_id": "q1", "source_doc": "opening",
             "source_exclusion_docs": ["opening", "opening-chunk-2"]},
            {"query_id": "q2", "source_doc": "question"}]
    run = {"q1": {"opening": 4.0, "opening-chunk-2": 3.5, "answer": 3.0},
           "q2": {"answer": 2.0, "question": 1.0}}
    got = experiment._exclude_query_sources(run, rows)
    assert got == {"q1": {"answer": 3.0}, "q2": {"answer": 2.0}}
    assert run["q1"]["opening"] == 4.0


def test_confirmation_decision_rejects_unlocked_bundle_name():
    decision = {"_schema": "m18-encoder-decision-v1",
                "selected_bundle": {"name": "zero_v1"}}
    with pytest.raises(SystemExit, match="bundle name"):
        experiment._verify_confirmation_decision(decision, "candidate", "/not/read")
