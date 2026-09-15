import pytest

import m13_paired_comparison as P


SIX = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
CLEAN4 = ("nfcorpus", "scidocs", "scifact", "trec-covid")


def rows(offset):
    return {dataset: {"q1": 0.2 + offset, "q2": 0.6 + offset} for dataset in SIX}


def test_summarize_uses_exact_six_and_registered_partitions():
    conf = {"bootstrap": {"B": 100, "seed": 900},
            "partitions": {"all6": list(SIX), "clean4": list(CLEAN4)}}
    got = P.summarize(rows(.1), rows(0), conf)
    assert got["partitions"]["all6"]["delta"] == pytest.approx(.1)
    assert got["partitions"]["clean4"]["delta"] == pytest.approx(.1)
    assert set(got["per_dataset"]) == set(SIX)
    assert all(value["n"] == 2 for value in got["per_dataset"].values())


def test_summarize_refuses_query_mismatch():
    m9 = rows(0)
    del m9["fiqa"]["q2"]
    conf = {"bootstrap": {"B": 10, "seed": 900},
            "partitions": {"all6": list(SIX), "clean4": list(CLEAN4)}}
    with pytest.raises(ValueError):
        P.summarize(rows(.1), m9, conf)
