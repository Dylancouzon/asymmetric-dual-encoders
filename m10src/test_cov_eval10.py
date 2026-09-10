"""`cov_eval10.score_student`'s pre-flight surface check (item G).

CPU only, no network: a duplicate unit id must refuse BEFORE anything is scored, so the encoder
callable and the document-vector cache are never even touched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import cov_eval10 as CE


def _unit(uid, family="consumer-health"):
    return (uid, family, ["q1"], ["1"], ["d1"], ["1"], {"1": {"1": 1}})


def test_duplicate_unit_ids_refuse_before_any_encoding():
    calls = []

    def encode_queries(texts):
        calls.append(texts)
        raise AssertionError("scoring must not be reached when units() carries a duplicate id")

    units = [_unit("nq"), _unit("nq")]
    with pytest.raises(SystemExit, match="duplicate unit ids"):
        CE.score_student(encode_queries, units=units)
    assert calls == [], "the encoder must never be called once a duplicate id is found"


def test_three_way_and_cross_family_duplicates_are_both_named():
    units = [_unit("nq"), _unit("nq"), _unit("nq"), _unit("fiqa", family="finance"),
            _unit("fiqa", family="finance")]
    with pytest.raises(SystemExit, match=r"\['fiqa', 'nq'\]"):
        CE.score_student(lambda t: t, units=units)


def test_a_surface_with_no_duplicates_reaches_assert_surface(monkeypatch):
    """The duplicate check must not fire on ordinary units -- it should fall through to the
    existing surface check (which then rejects this made-up family/unit set on its own terms,
    proving the duplicate guard did not swallow the call)."""
    units = [_unit("nq"), _unit("fiqa", family="finance")]
    with pytest.raises(ValueError, match="COV surface does not match"):
        CE.score_student(lambda t: t, units=units)
