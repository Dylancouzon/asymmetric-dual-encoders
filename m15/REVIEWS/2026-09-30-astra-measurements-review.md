# Codex gpt-6-astra, read-only, adversarial review of m15/MEASUREMENTS.md at 39c9b54

Date 2026-09-30. Brief: `briefs-2026-09-30/astra_measurements_brief.md`. Access audited from the exec log: named files only, no `work/`, no protected paths, no recursive search under `results/` or `work/`. Returned text below, verbatim. Dispositions in `m15/LOG.md`.

---

1. **P1 — E6 bootstraps a different contrast from its reported matched-fraction comparison.** In [e6_router.py:54](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e6_router.py:54), `f` is estimated from the evaluation queries. Lines 64–66 then resample differences calculated using that original `f`, although each resample changes the router’s realized Nano fraction. This adds traffic-share variation to an interval intended to measure routing advantage at equal traffic share. For example, if Nano improves every query by the same constant, matched-fraction routing advantage is identically zero in every resample; the implemented interval can nevertheless have positive width.  
   **Minimal fix:** keep the fitted threshold fixed, resample aligned `(route, zero, nano)` rows, and recompute `f`, router mean and random mean within each draw. Take percentiles of those recomputed contrasts.

2. **P1 — E8 lacks an enforced convergence condition for the claimed closed-form solution.** [MEASUREMENTS.md:230](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/MEASUREMENTS.md:230) treats block CG as the ridge solution and extends the grid to a potentially harder `1e-5`. But [teacher_screen.py:94](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m8src/teacher_screen.py:94) immediately scores the returned matrix; lines 104–105 merely record `converged` and the residual, and line 113 selects among all returned scores. The cited B7 evidence establishes agreement for Stella at the original four lambdas, not convergence for every E8 configuration and extension (`m8/RESULTS.md:20`). An unconverged candidate would measure solver truncation as well as the registered recipe.  
   **Minimal fix:** freeze a numerical acceptance rule before execution and require it before scoring any candidate, including the final re-solve. Use the existing solver’s convergence/residual criteria; stop on failure rather than selecting or reporting that matrix as the closed-form result. Preserve these diagnostics in the E8 receipt.

3. **P1 — E6 does not implement its registered pre-evaluation freeze or access receipt.** [MEASUREMENTS.md:81](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/MEASUREMENTS.md:81) requires thresholds to be written before evaluation data are read. [e6_router.py:85](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/e6_router.py:85) only prints them; the result is written at line 104, after every evaluation dataset. Separately, its direct query loader at line 44 never appends the six-set access log required by `MEASUREMENTS.md:18`. The calculation order prevents evaluation-driven fitting, but the promised durable evidence of that order is absent.  
   **Minimal fix:** persist a small threshold record, including fit-input hashes, before entering the evaluation loop; bind its hash into the final result. Log the six-set query loads as required. No transaction framework is needed.

The quantile router, fixed direction, disjoint fit forums and matched-fraction point baselines otherwise answer E6’s narrowed question. I found no essential E5 arithmetic defect. E2’s loss rule and subset boundary, E1’s runtime/parity protocol, and E4’s reproduction gate and controls address the earlier design objections. E8’s prescribed freeze order, e5 document-prefix correction, bounded grid extension and checkpoint roster are sound at the method level. Its fit-list hash gate remains mandatory; I did not inspect the protected storage to verify availability. No specified measurement assigns a role to the reserved four.

Files opened, including partial reads and content searches:

- `CLAUDE.md`; `instructions-m15.md`
- `m15/MEASUREMENTS.md`; `m15/PLAN.md`; `m15/HANDOFF.md`; `m15/LOG.md`; `m15/REVIEWS/2026-09-30-astra-plan-review.md`; `m15/REVIEWS/briefs-2026-09-30/astra_measurements_brief.md`
- `m15src/common.py`; `m15src/e5_oracle.py`; `m15src/e6_router.py`; `m15src/roster_ids.py`; `m15src/test_m15.py`
- `m8src/teacher_screen.py`; `m8src/m8base.py`; `m8src/paths_guard.py`; `m8src/d2_pre.py` **only lines 600–660**
- `m7src/dev_eval.py`; `m7src/teacher_probe.py`; `m7src/encoders.py`; `m7src/stage0_ridge.py`; `m7src/teacher.py`; `m7src/evalkit.py`
- `m20src/roster.py`; `m20src/beir15.py`
- `scripts/m13_serving_costs.py`; `scripts/teacher_learnability.py`; `m11/release/zero_encoder.py`
- `m7/CODEMAP.md`; `m8/RESULTS.md`; `m8/CODEMAP.md`; `m11/STATUS.md`; `m13/SHIP_LIST.md`; `m20/STATUS.md`; `m20/FINDINGS.md`
- `results/m8_trainq_manifest.json`; `results/m7_learnability_report.json`; `results/m8_t1_stella-400M-v5.json`

No scripts, tests or measurements were run; no protected paths were opened or listed.

**Verdict: Fix the E6 interval and audit trail, and enforce E8 solver convergence before proceeding with the affected runs.**
