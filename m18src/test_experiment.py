"""Evaluation-policy tests for the M18 experiment driver."""
from __future__ import annotations

import experiment


def test_query_source_is_removed_without_marking_it_relevant():
    rows = [{"query_id": "q1", "source_doc": "opening"},
            {"query_id": "q2", "source_doc": "question"}]
    run = {"q1": {"opening": 4.0, "answer": 3.0},
           "q2": {"answer": 2.0, "question": 1.0}}
    got = experiment._exclude_query_sources(run, rows)
    assert got == {"q1": {"answer": 3.0}, "q2": {"answer": 2.0}}
    assert run["q1"]["opening"] == 4.0
