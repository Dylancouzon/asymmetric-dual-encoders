"""The rescreen10 report merge and its validator.

`main()` used to start every invocation from `out = {}`, so running `--queries` and then
`--documents` (or the reverse) as two separate CLI calls left only the LAST pass's section in
`results/m10_rescreen10.json` -- the 709-query section was gone from the shipped report. These
tests are on `merge_report` and `validate` directly, which is what a launcher actually depends on;
`main()` itself needs a live protected10 index and is not unit-testable cheaply.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

import rescreen10 as R


def test_merge_report_preserves_the_other_sections_on_a_partial_rerun():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "report.json"
        R.merge_report(p, {"m9-queries_pair": {"n": 10}, "protected10": {"v": 1}})
        R.merge_report(p, {"documents": {"n_dropped": 3}})
        out = json.loads(p.read_text())
        assert out["m9-queries_pair"]["n"] == 10, "the query section must survive"
        assert out["documents"]["n_dropped"] == 3
        # re-running the SAME section overwrites only itself
        R.merge_report(p, {"m9-queries_pair": {"n": 11}})
        out = json.loads(p.read_text())
        assert out["m9-queries_pair"]["n"] == 11 and out["documents"]["n_dropped"] == 3


def test_validate_requires_both_sections():
    report = {"m9-queries_pair": {"n": 2}, "m9-nqopen": {"n": 2}, "m9-triviaqa": {"n": 2},
             "protected10": {"v": 1}}
    with pytest.raises(SystemExit, match="missing sections"):
        R.validate(report, {})
    with pytest.raises(SystemExit, match="missing sections"):
        R.validate({"documents": {"n_dropped": 0}}, {})


def test_validate_checks_mask_lengths_against_the_reports_pool_sizes():
    report = {"m9-queries_pair": {"n": 2}, "m9-nqopen": {"n": 1}, "m9-triviaqa": {"n": 1},
             "documents": {"n_dropped": 2}, "protected10": {"v": 1}}
    good = {"m9-queries_pair": np.zeros(2, dtype=bool), "m9-nqopen": np.zeros(1, dtype=bool),
           "m9-triviaqa": np.zeros(1, dtype=bool), "documents": np.zeros(2, dtype=np.int64),
           "protected10": {"v": 1}}
    R.validate(report, good)                            # does not raise

    bad = dict(good)
    bad["m9-nqopen"] = np.zeros(5, dtype=bool)
    with pytest.raises(SystemExit, match="mask length"):
        R.validate(report, bad)

    bad2 = dict(good)
    bad2["documents"] = np.zeros(9, dtype=np.int64)
    with pytest.raises(SystemExit, match="mask length"):
        R.validate(report, bad2)


def test_validate_catches_a_stale_protected10_identity():
    report = {"m9-queries_pair": {"n": 1}, "m9-nqopen": {"n": 1}, "m9-triviaqa": {"n": 1},
             "documents": {"n_dropped": 0}, "protected10": {"v": 1}}
    masks = {"m9-queries_pair": np.zeros(1, dtype=bool), "m9-nqopen": np.zeros(1, dtype=bool),
            "m9-triviaqa": np.zeros(1, dtype=bool), "documents": np.zeros(0, dtype=np.int64),
            "protected10": {"v": 2}}
    with pytest.raises(SystemExit, match="identity"):
        R.validate(report, masks)
