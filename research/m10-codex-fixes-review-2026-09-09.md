# Codex (gpt-5.6-sol, high) adversarial review of the eleven fixes — 2026-09-09, VERBATIM

Brief: `research/m10-review-briefs-2026-09-09.md` §2. Invocation:
`codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort="high"`. Read-exclusion
audited clean (the only hits on reserved names were the brief's own exclusion text and two
unrelated code comments containing the word UNTOUCHED).

The transcript's first ~3,790 lines are the files it read; the review itself follows.

```
/bin/bash: line 1: jq: command not found

codex
Overall verdict: **NO-GO on the eleven-fix bundle as claimed.** The live warmup path works, but fixes 1, 2, 4, 8, and 11 are incomplete or misleading. Fix 4 remains the most serious integrity defect.

## Per-fix verdict

1. **PARTIAL** — Training really is batch-aware: `run_arm` passes `p["batch"]` to both the fingerprint and trainer, and the trainer passes `warmup_steps_for(batch_size)` into `lr_at` ([run_arm.py:701](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:701), [run_arm.py:932](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:932), [trainer10.py:166](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:166)). Thus bs32 gets 2,000 steps and bs128 gets 500 ([nano10.py:327](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:327)).  
   However, “already-run fingerprints unchanged” is false: fingerprints also contain the changed registry hash and a hash of `run_arm.py`, `trainer10.py`, and `nano10.py`—all changed in this commit ([run_arm.py:674](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:674), [run_arm.py:704](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:704), [run_arm.py:707](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:707)). Only the `warmup_steps` subfield remains 2,000. Also, registry `warmup_examples` is not read; code independently hardcodes 64,000, and `lr_at` still defaults to the old fixed `WARMUP_STEPS` ([screen_registry.json:138](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:138), [nano10.py:336](/home/dylan/asymetric-dual-encoders/m10src/nano10.py:336)). The stale interpretation still says bs128 gets 256,000 warmup examples ([screen_registry.json:454](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:454)).

2. **PARTIAL** — `pending: CLOUD_ONLY` is registered and `compute()` checks it before looking for COV files ([screen_registry.json:136](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:136), [contrasts.py:158](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:158)). But it has no lifecycle: after a legitimate E run, the same static field still blocks computation until someone manually edits the registry. Worse, `selection()` trusts whatever contrast artifact exists without verifying its registry hash or registered spec ([contrasts.py:300](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:300)). The committed verdict still carries the old missing-file inference, not the new registered-pending reason ([m10_screen_verdicts.json:69](/home/dylan/asymetric-dual-encoders/results/m10_screen_verdicts.json:69)).

3. **WORKS** — Generic resolution now requires `sign_stable is True`, so a missing previous cycle cannot resolve ([contrasts.py:201](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:201), [contrasts.py:215](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:215)). The test exercises the production `compute()` path ([test_contrasts.py:157](/home/dylan/asymetric-dual-encoders/m10src/test_contrasts.py:157)).

4. **DOES NOT WORK** — Only the final files used for the bootstrap are checked ([contrasts.py:171](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:171)). The previous cycle ends used for sign stability are subsequently read without validation ([contrasts.py:190](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:190)). In addition, a record entry with a missing, null, or empty `per_query_scores_sha256` passes because comparison is conditional on the expected hash being truthy ([contrasts.py:104](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:104), [contrasts.py:109](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:109)). Finally, the arm record is not independently pinned; changing both the current arm record and COV file passes. Only family F has a separate verdict-to-arm-record hash check ([run_arm.py:246](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:246)). The test covers only a populated wrong hash and misses all three holes ([test_contrasts.py:192](/home/dylan/asymetric-dual-encoders/m10src/test_contrasts.py:192)).

5. **WORKS, but the test is worthless as a regression guard** — Production `best()` correctly returns `None` for multiple equal maxima, causing fallback to the default ([contrasts.py:327](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:327)). The test reimplements that algorithm locally and never calls production code; its `best = CT.selection.__globals__` assignment is unused ([test_contrasts.py:213](/home/dylan/asymetric-dual-encoders/m10src/test_contrasts.py:213)). It documents intent but will remain green if production `best()` is reverted or deleted.

6. **PARTIAL** — The current two-arm unresolved-F case returns the correct cheapest student, MiniLM-L6 ([contrasts.py:340](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:340)). But it is hardcoded rather than implementing the registered parameter/layer/backbone ordering ([screen_registry.json:413](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:413)), and there is no test of this dead branch. Candidate or cost changes can silently make it stale.

7. **WORKS** — Both the exemption and family-A resolution branch use the same registered rule-name set; no contrast IDs remain hardcoded ([contrasts.py:196](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:196), [contrasts.py:205](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:205)).

8. **PARTIAL** — The registry now correctly retracts clipping as established and calls both mechanisms candidates ([screen_registry.json:458](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:458)). But the executable selection explanation and committed verdict still assert that D2 “is confounded with an optimizer-scale difference” ([contrasts.py:379](/home/dylan/asymetric-dual-encoders/m10src/contrasts.py:379), [m10_screen_verdicts.json:17](/home/dylan/asymetric-dual-encoders/results/m10_screen_verdicts.json:17)). The retraction has not reached the downstream output.

9. **WORKS** — `_exposure_timing` now explicitly says macros were visible but no contrast had been computed, consistent with the amendment history ([screen_registry.json:452](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:452), [screen_registry.json:461](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:461)).

10. **WORKS as accounting prose** — The arithmetic is correct: realized old “plus” 15M minus SYNTH-20M 20M gives −5M; adding the conditional 66.7M top-up gives +61.7M ([screen_registry.json:327](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:327)). None of the reviewed executable files reads `budget_released`; the obsolete actionable scalar was removed, so this currently changes no decision.

11. **DOES NOT WORK** — `_VACUOUS_AS_REGISTERED` is only an annotation. The active `synth_20M` block still specifies 20M and retains the obsolete combined-components rationale ([screen_registry.json:319](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:319)). Moreover, batch remains pending, so SYNTH is not necessarily anchor-vs-itself: if E selects bs128, batch differs ([m10_screen_verdicts.json:10](/home/dylan/asymetric-dual-encoders/results/m10_screen_verdicts.json:10)). `seed_variance_gap` is likewise unconsumed prose; the current verdict does not carry its required interval disclaimer. And one observed seed difference with `n=1` is a measurement or sensitivity datum, not a statistical “bound” ([screen_registry.json:341](/home/dylan/asymetric-dual-encoders/m10/screen_registry.json:341)). No reviewed code reads `synth_20M` or `seed_variance_gap`.

## Ranked remaining breakage

1. **COV integrity remains bypassable:** missing hashes pass; previous-cycle files are unchecked; general arm records are not pinned.
2. **The fingerprint-compatibility claim is false:** every recomputed fingerprint changes through both registry and code hashes.
3. **E has no safe state transition:** `pending` prevents legitimate post-run computation, while selection accepts stale artifacts without hash validation.
4. **Fix 11 is exactly the registry-prose failure mode:** neither the vacuous SYNTH disposition nor reporting obligation is enforced.
5. **The D2 retraction does not propagate to the generated verdict.**
6. **Dead branches are weakly guarded:** the tie test never touches production, and serve-cost selection is hardcoded and untested.
7. **Warmup has three competing sources:** registry field, `WARMUP_EXAMPLES`, and the old `lr_at` default; the registry interpretation is already stale.

## Is equal-example warmup defensible?

Yes—I would defend the 64,000-example rule in a paper, provided the estimand is stated as “batch size under an example-indexed LR schedule.” It preserves the same 1.28% warmup fraction and the same LR trajectory versus examples. Keeping 2,000 optimizer steps at bs128 would instead make warmup 5.12% of its dose.

The cost is real: 500 AdamW updates may be insufficient even if 64,000 examples are sufficient. You cannot simultaneously equalize examples and update count when batch differs. If the scientific target is the best deployable recipe rather than a controlled batch contrast, each batch should receive independently preregistered LR/warmup tuning. For this E contrast, equal-example warmup is the cleaner choice, but it does not remove the remaining optimizer-update or hardware confounds.

## Is PENDING defensible?

Conceptually, yes: “not run” is not “unresolved,” and selecting bs128 from an absent measurement would collapse missingness into evidence. The bs32-based budget is conservative because bs128 is cheaper.

Operationally, the implementation is not yet sound. `pending` needs an explicit registered transition such as “pending until a complete arm record matching its fingerprint exists”; otherwise the legitimate result remains permanently blocked or requires an outcome-visible registry edit.

## A4−A3 arithmetic and interpretation

The arithmetic checks:

| Family | Family delta | Weighted contribution |
|---|---:|---:|
| consumer-health / MedicalQA | +0.0395390 | +0.00988475 |
| finance / LEDGER | +0.00766168 | +0.00191542 |
| legal | +0.00646009 | +0.00161502 |
| BRIGHT | −0.00533907 | −0.00133477 |
| Total |  | **+0.01208043** |

MedicalQA therefore contributes **81.82%**. Removing health and renormalizing the other three family weights gives **+0.00292757**, matching the claimed +0.0029. A3 has five forms; A4/ANCHOR has twelve, including the added `health` and `finance` forms ([m10_exposure_table.json:43](/home/dylan/asymetric-dual-encoders/results/m10_exposure_table.json:43), [m10_exposure_table.json:68](/home/dylan/asymetric-dual-encoders/results/m10_exposure_table.json:68)).

“This is a form-match effect, not a coverage effect” is over-claimed. This is one bundled intervention adding seven forms while diluting every existing form’s exposure; there is no per-form ablation. The evidence supports: **the gain is concentrated in health and is consistent with domain/form matching, while broad cross-domain benefit is not established.** It cannot distinguish form matching from targeted semantic coverage, generated-data quality, or another correlated property of those added datasets.
```
