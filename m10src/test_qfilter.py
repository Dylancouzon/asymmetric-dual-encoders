"""The filter must enforce the rubric's own numbers and nothing else."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import forms
import qfilter as Q


def test_ranges_come_from_the_frozen_rubric_for_every_form():
    assert set(Q.RANGES) == set(forms.RUBRIC)
    assert Q.RANGES["health"] == (8, 30) and Q.RANGES["yesno"] == (6, 20)
    assert Q.RANGES["argument"] == (120, 220) and Q.RANGES["keyword"] == (2, 4)


def test_boundaries_are_inclusive():
    lo, hi = Q.RANGES["health"]
    assert Q.in_range("health", " ".join(["w"] * lo))
    assert Q.in_range("health", " ".join(["w"] * hi))
    assert not Q.in_range("health", " ".join(["w"] * (lo - 1)))
    assert not Q.in_range("health", " ".join(["w"] * (hi + 1)))


def test_short_and_long_are_counted_separately():
    """A form failing SHORT is a prompt problem; failing LONG is a different one. One number
    hides which."""
    rows = ([{"form": "health", "query": " ".join(["w"] * 3)}] * 2
            + [{"form": "health", "query": " ".join(["w"] * 40)}]
            + [{"form": "health", "query": " ".join(["w"] * 12)}] * 7)
    kept, rep = Q.filter_queries(rows)
    assert len(kept) == 7
    assert rep["health"] == {"seen": 10, "kept": 7, "short": 2, "long": 1,
                             "range": [8, 30], "drop_rate": 0.3}


def test_a_rubric_edit_moves_the_filter_with_it():
    """The two must not be able to disagree — that is the whole reason the range is parsed."""
    import importlib
    old = forms.RUBRIC["yesno"]
    try:
        forms.RUBRIC["yesno"] = "N yes/no questions. 9 to 11 words."
        importlib.reload(Q)
        assert Q.RANGES["yesno"] == (9, 11)
    finally:
        forms.RUBRIC["yesno"] = old
        importlib.reload(Q)
    assert Q.RANGES["yesno"] == (6, 20)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
