# Codex gpt-6-astra, focused re-review of the three P1 fixes (4ac2b62)

Date 2026-09-30. Brief: `briefs-2026-09-30/astra_measurements_rereview.md`. Access audited from the exec log: the diff and the named files only. Returned text below, verbatim.

---

- **P1 #1 — Resolved**, `m15src/e6_router.py:65`: each bootstrap draw resamples aligned rows and recomputes the realized Nano fraction and matched-fraction contrast with the threshold fixed.
- **P1 #2 — Resolved at the specification level**, `m15/MEASUREMENTS.md:238`: the frozen gate matches the solver’s tolerances, covers every candidate and final re-solve, prohibits scoring failed solves, and requires diagnostics. Runtime enforcement is not demonstrated by this diff.
- **P1 #3 — Resolved**, `m15src/e6_router.py:89` and `m15src/common.py:118`: thresholds and fit-input receipts are persisted before evaluation reads, their file hash is bound into the final receipt, and query loads invoke the six-set access logger.

**New correctness defects:** None identified in the scoped fixes.

**Files opened** (including diff-only and partial reads): `m15/MEASUREMENTS.md`; `m15/REVIEWS/2026-09-30-astra-measurements-review.md`; `m15src/common.py`; `m15src/e6_router.py`; `m15src/encoders15.py`; `m15src/test_m15.py`; `m8src/blockcg.py`; `bench/core.py`.

**Verdict:** The E6 fixes and E8 frozen acceptance rule are correct; E8 runtime enforcement remains unverified.
