# Project status — 2026-09-10

**M10 is closed as preparation. M13 owns cloud execution.** The repository has useful measured
results and reusable components, but no final nano quality result or complete cost frontier yet.
Current plan: `ROADMAP.md`; next action: `m13/STATUS.md`.

## Research and product

| Area | Status |
|---|---|
| Zero | Frozen/published in M11's record. Dense avg-6 0.4339 misses LightRetriever's 0.4583 bar. Valid negative result, deployable lookup table |
| Fusion | M12 recommends Qdrant DBSF@100: 0.4887 avg-6 / 0.4912 clean-4. Small observed differences from convex0; no equivalence CI. Edge fusion is not measured |
| First nano (M9) | Frozen after plateau; strong dataset dependence (93.8% retention on NQ, 50–71% on two forum components). Final six-set close-out pending |
| Nano preparation (M10) | Data pipeline, 13 box screen arms and two descriptive arms complete. Selected bge-small/A4/1152-linear/75:25/squared-L2; both cloud E arms and batch decision remain pending |
| Cloud/final execution (M13) | Build controller, final scorer, LoTTE definition, access recovery and comparable costs remain explicit work; see `m13/EXECUTION.md` |
| Paper | Substantial evidence already exists. Teacher distillability, task dependence, fusion deployment and negative results support a paper even if nano misses |

The generated-form intervention helps the screen, chiefly consumer health; it does not establish
broad recovery. Nano has roughly bge-small's query compute; zero is the near-zero-compute path.
Evidence/claim limits: `m10/FINDINGS.md`, `instructions-m15.md`.

## Cleanup and verification

- Preserved the original mandate, guidance, ledger and future plans byte-for-byte with hashes
  under `research/archive/m10-cleanup-2026-09-10/`; git commits remain the provenance record.
- Reduced the five core entry documents from **3,678 to 210 lines**; full history stays archived. Shifted
  release/paper/image/better-zero to M14/M15/M16/M17; kept experiment paths and screen constants.
- Added one active check command and component map (`run_checks.sh`, `HARNESS.md`). Isolated a
  test that could rewrite a real descriptive result; covered both decision fields outside ±1.
- Fixed abrupt-process resume: a durable run receipt now connects the CLI to its checkpoint.
  A SIGKILL/restart test matches uninterrupted weights, losses, evaluations and example count.
  Caught failures remain terminal; resume still requires a saved checkpoint.
- Corrected the bridge's false “minimum nDCG quantum” rationale, leaving thresholds unchanged.
  The concrete counterexample and independent audit are in `research/m10-cleanup-review-2026-09-10.md`.

**Final verification: 469 M10 tests, 16 M9 statistics tests and 31 fusion checks passed.**
Archive copies match their source commit; screen registry, comparator and screen verdict bytes are
unchanged. The final-run registry changed only its bridge rationale, not decision constants.
Legacy M7: eight suites passed, encoder inventory failed on seven historical cache/spec mismatches;
one lengthy calibration was stopped and the last not started. All legacy outputs were restored
byte-for-byte. The full legacy suite is not green; no cache was deleted to hide the mismatch.

No cloud rental, experimental training, protected evaluation or model publication was performed by this
cleanup. Closing preparation does not claim M13's execution blockers have been solved.
