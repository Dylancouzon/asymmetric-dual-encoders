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
- Replaced long active narratives with short scope, status, result and finding indexes. Shifted
  release/paper/image/better-zero to M14/M15/M16/M17; kept experiment paths and screen constants.
- Added one active check command and component map (`run_checks.sh`, `HARNESS.md`). Isolated a
  test that could rewrite a real descriptive result; covered both decision fields outside ±1.
- Corrected the bridge's false “minimum nDCG quantum” rationale, leaving thresholds unchanged.
  The concrete counterexample and independent audit are in `research/m10-cleanup-review-2026-09-10.md`.

Verification before the final runner repair: **459 M10 tests, 16 M9 statistics tests and 31 fusion
checks passed**. Final verification and merge receipt will be recorded after the repair is tested.
Legacy M7 has a known seven-cache encoder-spec mismatch, retained and disclosed rather than erased.

No cloud rental, training run, protected evaluation or model publication was performed by this
cleanup. Closing preparation does not claim M13's execution blockers have been solved.
