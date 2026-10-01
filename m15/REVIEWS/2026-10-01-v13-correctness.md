# V13 focused claim-scope closure, 2026-10-01

**Outcome:** E16–E18 are otherwise represented accurately. One narrow causal-exclusion sentence should be corrected; no new experiment or repeated audit is needed.

## Required narrow correction

In Section 4.2, “Mean cosine … is almost identical … so greater angular distortion does not explain that absolute difference” excludes more than the aggregate measurement establishes. Similar means do not exclude different distributions or query-level relationships between angle and candidate loss.

Use: **“Zero does not show a larger mean angular distortion; the source of the coverage difference remains unresolved.”** The displayed range should be approximately 0.794–0.798 (FiQA Nano is 0.79443), or simply “about 0.80.” This preserves the useful negative observation without claiming a mechanism has been ruled out.

## Scope checks passed

- E17's primary interaction is described as weaker, with Zero-versus-Nano uncertainty including zero. C14 also correctly renders the Zero-versus-Stella endpoint as touching zero. Secondary settings are not substituted for the primary result.
- E18 is exhaustive scoring on fixed document sign codes, with each tier's own original exact-top10 target and expected boundary-tie inclusion. C40 is identified as primary, and all budgets remain in C15.
- The larger Zero gain is explicitly an absolute percentage-point result. The paper and C15 disclose differing miss headroom and do not claim greater proportional or intrinsic architectural sensitivity.
- The paper distinguishes graded original-query scoring from native scalar8, global candidate budgets from segment oversampling, and candidate coverage from nDCG, HNSW performance, and latency. No unsupported 2.5-fold serving or candidate-budget claim is introduced.
- Abstract and conclusion frame query precision as a measured scoring choice that motivates a deployment test, not a demonstrated native remedy. Existing shared-index timing is kept separate.

This closure reuses the prior E16/E17/E18 source, arithmetic, tie-function, and provenance checks. I read the changed v13 main text and evidence cards C14/C15, plus targeted occurrences of precision-related claims in `m15/PAPER.md`. No new result or raw data was opened, no experiment was launched, and only this review file was written.

## Correction disposition

Verified only the amended Section 4.2 sentence in `m15/PAPER.md`: it now reports mean cosine as about 0.80, states that Zero does not show a larger mean angular distortion, and leaves the source of the coverage difference unresolved. **The narrow finding is resolved; no essential claim-scope issue remains from this focused v13 review.** No repeated audit or experiment was performed.
