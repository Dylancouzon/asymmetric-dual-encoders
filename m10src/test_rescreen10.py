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


def _good_masks():
    return {"m9-queries_pair": np.zeros(2, dtype=bool), "m9-nqopen": np.zeros(1, dtype=bool),
           "m9-triviaqa": np.zeros(1, dtype=bool), "documents": np.array([3, 5], dtype=np.int64),
           "protected10": {"v": 1}}


def _good_report(masks):
    def qsect(name, n):
        m = masks[name]
        return {"n": n, "removed": int((~m).sum()), "mask_sha256": R.mask_sha256(m)}
    return {"m9-queries_pair": qsect("m9-queries_pair", 2),
           "m9-nqopen": qsect("m9-nqopen", 1),
           "m9-triviaqa": qsect("m9-triviaqa", 1),
           "documents": {"n_dropped": 2, "complete": True,
                        "mask_sha256": R.mask_sha256(masks["documents"]),
                        "ident": {"pool": {"n": 10}}},
           "protected10": {"v": 1}}


def test_validate_checks_mask_lengths_against_the_reports_pool_sizes():
    good = _good_masks()
    report = _good_report(good)
    R.validate(report, good)                            # does not raise

    bad = dict(good)
    bad["m9-nqopen"] = np.zeros(5, dtype=bool)
    with pytest.raises(SystemExit, match="mask length"):
        R.validate(report, bad)

    bad2 = dict(good)
    bad2["documents"] = np.zeros(9, dtype=np.int64)
    with pytest.raises(SystemExit, match="mask length"):
        R.validate(report, bad2)


def test_validate_rejects_a_mask_that_does_not_match_its_digest():
    """The old check was length-only: an all-True query mask of the RIGHT LENGTH, or a document
    mask of unrelated rows, passed it silently."""
    good = _good_masks()
    report = _good_report(good)

    all_true = dict(good)
    all_true["m9-queries_pair"] = np.ones(2, dtype=bool)
    with pytest.raises(SystemExit, match="removes|digest"):
        R.validate(report, all_true)

    wrong_docs = dict(good)
    wrong_docs["documents"] = np.array([1, 2], dtype=np.int64)      # same length, different rows
    with pytest.raises(SystemExit, match="digest"):
        R.validate(report, wrong_docs)


def test_validate_rejects_document_rows_out_of_range_or_duplicated():
    good = _good_masks()
    report = _good_report(good)

    dup = np.array([3, 3], dtype=np.int64)
    report_dup = dict(report)
    report_dup["documents"] = {**report["documents"], "mask_sha256": R.mask_sha256(dup)}
    masks_dup = dict(good)
    masks_dup["documents"] = dup
    with pytest.raises(SystemExit, match="duplicate"):
        R.validate(report_dup, masks_dup)

    oob = np.array([3, 99], dtype=np.int64)                          # pool n is 10
    report_oob = dict(report)
    report_oob["documents"] = {**report["documents"], "mask_sha256": R.mask_sha256(oob)}
    masks_oob = dict(good)
    masks_oob["documents"] = oob
    with pytest.raises(SystemExit, match="out of range"):
        R.validate(report_oob, masks_oob)


def test_validate_requires_the_document_screen_to_be_complete():
    good = _good_masks()
    report = _good_report(good)
    report["documents"] = {**report["documents"], "complete": False}
    with pytest.raises(SystemExit, match="COMPLETE"):
        R.validate(report, good)


def test_validate_catches_a_stale_protected10_identity():
    report = {"m9-queries_pair": {"n": 1}, "m9-nqopen": {"n": 1}, "m9-triviaqa": {"n": 1},
             "documents": {"n_dropped": 0}, "protected10": {"v": 1}}
    masks = {"m9-queries_pair": np.zeros(1, dtype=bool), "m9-nqopen": np.zeros(1, dtype=bool),
            "m9-triviaqa": np.zeros(1, dtype=bool), "documents": np.zeros(0, dtype=np.int64),
            "protected10": {"v": 2}}
    with pytest.raises(SystemExit, match="identity"):
        R.validate(report, masks)


def test_the_real_shipped_report_validates_against_its_own_cached_masks():
    """No screens recomputed: reads the mask CACHE files directly (the same bytes a real
    `assemble_arm` call would consume) and checks them against `results/m10_rescreen10.json`'s
    own recorded digests -- the retrofit item B adds them by."""
    if not R.CACHE.exists() or not R.REPORT_PATH.exists():
        pytest.skip("no cached M10 re-screen masks on this box")
    report = json.loads(R.REPORT_PATH.read_text())
    masks = {}
    for jf in R.CACHE.glob("q-*.json"):
        side = json.loads(jf.read_text())
        name = side["ident"]["name"]
        z = np.load(jf.with_suffix(".npz"), allow_pickle=False)
        masks[name] = z["keep"].astype(bool)
    dfiles = list(R.CACHE.glob("d-*.json"))
    if dfiles:
        z = np.load(dfiles[0].with_suffix(".npz"), allow_pickle=False)
        masks["documents"] = z["rows"].astype(np.int64)
    if not all(s in masks for s in R.QUERY_SEGMENTS) or "documents" not in masks:
        pytest.skip("cached masks incomplete on this box")
    R.validate(report, masks)                            # does not raise
