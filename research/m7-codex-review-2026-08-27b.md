Do not use the full-run numbers yet. The dependence statistic itself is basically right, but the audit still has three decision-changing blockers: pinning is not enforced end-to-end, lever 4 may be evaluated on the wrong surviving artifact, and its int8 rule does not implement the pre-registration.

I reviewed the latest working tree, including the newly added `freeze_heldout.py` and memoization changes that appeared during the review.

## BLOCKER

1. **The audit can still run without the six-component pin, and the pinned pool is recorded but not verified.**

The committed [m7_dev_manifest.json](/home/dylan/asymetric-dual-encoders/results/m7_dev_manifest.json:1) currently contains only four components. When `_pinned.components` is absent, [dev_eval.py](/home/dylan/asymetric-dual-encoders/m7src/dev_eval.py:43) falls back to the old existence-dependent behavior, and [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:69) does not require a completed pin. Thus the “full pinned” audit can still start before `freeze_heldout.py` runs.

Additionally, `freeze_heldout.py` records the pool vector SHA and metadata at [freeze_heldout.py](/home/dylan/asymetric-dual-encoders/m7src/freeze_heldout.py:86), but runtime verification checks only the two JSON byte hashes at [heldout.py](/home/dylan/asymetric-dual-encoders/m7src/heldout.py:97). A changed same-size vector file or inconsistent pool metadata is therefore not rejected.

Before the audit:

- Make `dev_audit` require exactly the pinned six names; no fallback.
- Recompute and verify the pool identity, vector SHA, active encoder, and the six asset hashes against the manifest.
- Do not let `--no-pool-hash` write an authoritative pin; it currently can at [freeze_heldout.py](/home/dylan/asymetric-dual-encoders/m7src/freeze_heldout.py:95).
- Assert uniqueness, equal vector lengths, exact long-query subset membership, and identical long/train text and qrels. The actual files satisfy these properties—7,325/55 unique qids and zero mismatches—but [freeze_heldout.py](/home/dylan/asymetric-dual-encoders/m7src/freeze_heldout.py:111) currently records the intersection without asserting them.

2. **Lever 4 is always evaluated on `CHAIN[-1]`, even if the dependence audit reverts it.**

`CANDIDATE` is fixed to the last artifact at [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:40). Pooling makers are constructed only for that artifact at [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:119), while the actual survivor is determined later at [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:143).

If s2500 or an earlier decision reverts, lever 4 is probing a rejected table, contrary to “the frozen candidate.” Either:

- Finish the dependence audit first and run lever 4 on `surviving`; or
- Include pooling variants for every possible survivor in the one corpus pass, then adjudicate only the surviving artifact.

3. **Lever 4 does not implement “int8 independently.”**

Holm is computed only from fp16 p-values at [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:193). Eligibility then checks fp16 and int8 CIs, but never the int8 sign-flip p-value or an int8 Holm result at [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:195).

The cleanest interpretation is two three-hypothesis families: Holm across fp16 arms and separately across int8 arms, requiring the selected arm to pass both Holm decisions and both CIs. If int8 is intended only as a robustness gate on the fp16-selected winner, say that explicitly instead—the current prose and implementation disagree.

## MAJOR

1. **The bootstrap repair is correct only as a conditional, fixed-stratum-size bootstrap. Report that choice.**

For each long query, the macro coefficient is:

`1/(6×7325) + 1/(6×55)`.

Using one sign for both appearances is exactly correct; [signflip_dep](/home/dylan/asymetric-dual-encoders/m7src/boot.py:205) applies that combined coefficient under one exchangeability flip.

[paired_dep](/home/dylan/asymetric-dual-encoders/m7src/boot.py:242) is also internally correct: it independently resamples 55 long and 7,270 non-long units, holds both stratum sizes fixed, and reuses the long draw in both component means.

But that changes more than covariance. Ordinary heldout-train bootstrap allows the long-query fraction to fluctuate; the new bootstrap conditions on observing exactly 55 long queries. It does not bias the observed point estimate, but it estimates a different, conditional variance. That is defensible because the suite and component sizes are frozen, and it is the repair I previously prescribed. Still, report it separately from the covariance effect. A useful three-way diagnostic is:

- Ordinary componentwise bootstrap.
- Fixed-stratum bootstrap with independent long draws.
- Fixed-stratum bootstrap with the shared long draw.

The second-to-third difference isolates dependence; first-to-second isolates conditioning/stratification.

2. **Decisions are made using rounded CI endpoints.**

Both ordinary and dependent bootstrap routines round CIs to four decimals at [boot.py](/home/dylan/asymetric-dual-encoders/m7src/boot.py:280). `dev_audit` then compares that rounded lower bound to zero at [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:139) and [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:197).

A true lower endpoint of `+0.00004` becomes `0.0000` and incorrectly reverts. Return raw endpoints for decisions and separate formatted fields for display.

3. **“Original bar” is historically inaccurate, although the new conservative audit is honest.**

The original lever-1 language required fp16 sign-flip plus CI, but only int8 CI at [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:447). The 2m cross-arm pick and s2500 extension did not prospectively specify this adoption threshold at [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:520).

Because the recomputation rule was written before the recomputed numbers, applying the stricter rule is honest as a new conservative survival audit. Change “still meet its original bar” at [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:546) to “meet the newly fixed common survival bar.”

The walk-forward logic itself is right: stopping at the first REVERTS is necessary because later comparisons do not establish superiority over an earlier surviving baseline.

4. **The matrix evidence proves nDCG-decision equivalence, not ranking equivalence.**

Exact per-query nDCG equality is enough to show that the historical nDCG statistics would be unchanged. It does not discharge the full MAJOR 3 request: top-10 membership can change entirely among non-relevant documents while nDCG remains identical. Current reporting only counts changed nDCG at [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:158).

Record, per artifact/quantization/component:

- Changed ordered top-10 count.
- Changed top-10 set count.
- Maximum score deviation for matched retrieved documents.
- Preferably top-100 set changes too, since retrieval is computed at 100.

The `4.47e-08` vector deviation and zero nDCG deviation are excellent evidence; membership counts complete it.

5. **MAJOR 6 provenance is improved but incomplete.**

The audit now stores table hashes, active encoder, unrounded macros, and per-query values. Missing pieces are:

- Unrounded CI endpoints and deltas.
- Evaluator/code identity—particularly important while these files are uncommitted.
- Dev-manifest hash and verified live asset hashes.
- A clearly labelled compressed-file SHA versus decompressed JSON SHA; [dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:219) currently stores the hash of uncompressed JSON under a path naming the gzip file.

A committed code revision or explicit hashes of the evaluator/statistics source files would close this.

## MINOR

- The reduction test’s sign-flip assertion is nearly vacuous: its generated effect puts both p-values at the Monte Carlo floor, so `abs(p1-p2)<0.01` at [test_dep_stats.py](/home/dylan/asymetric-dual-encoders/m7src/test_dep_stats.py:39) would tolerate many broken implementations. Use a near-null fixture with p around 0.2–0.8.
- The full-duplication CI invariant is good, but the test bypasses production `unit_key` with a lambda at [test_dep_stats.py](/home/dylan/asymetric-dual-encoders/m7src/test_dep_stats.py:55). Add the same test using `heldout-train`/`heldout-longq`, and compare dependent sign-flip directly with a single-component reference rather than merely asserting `p_dep > p_ordinary`.
- `eval_makers` should assert unique qids and `set(per_query_ndcg)==set(q_ids)==set(qrels)` after scoring. At present it binds arrays correctly but does not prove no query was omitted at [multieval.py](/home/dylan/asymetric-dual-encoders/m7src/multieval.py:75).
- With the new memoized pool IDs/vectors at [heldout.py](/home/dylan/asymetric-dual-encoders/m7src/heldout.py:139), the shared-object assertion in `multieval` is now sound. Before that concurrent change, the full run would have failed.
- `k=100` is safe for nDCG@10 after dropping one self-hit. The API becomes unsafe if called with `k=10`; assert `k >= cut+1` when self-hit filtering is enabled.
- “The weighted-mean denominator keeps length normalization unchanged” at [table.py](/home/dylan/asymetric-dual-encoders/m7src/table.py:151) is misleading: the denominator cancels under final L2 normalization except for fallback thresholds.

## Lever-4 design coaching

The three-arm family is coherent, and Holm over the three fp16 comparisons is appropriate. Separate Holm for the three int8 comparisons is the safest interpretation of “independently.”

`cap2` is the least informative arm: it equals the baseline for counts one and two and changes only counts three or greater. A fixed BM25-like saturation such as `2c/(1+c)`, or `log1p(c)`, gives a smoother third point between binary and mean. However, you have already seen two-component smoke results. Do not silently change the family now; either keep the registered three or record a clearly post-smoke amendment and use Holm over four.

Deferring `Preproc.pool_mode` during the probe is defensible. If adopted, add it before fusion, gate, or release use, explicitly rebuild/re-save the release metadata, and rehash it. No trained artifact needs rerunning.

## Hostile-reader wording fixes

The two new registrations should clarify:

- “Before the numbers” means before the recomputed/full lever-4 numbers, not before the original lever outcomes.
- The bootstrap conditions on fixed long/non-long stratum sizes.
- The survival threshold is newly standardized, not historically the original rule.
- Which survivor “frozen candidate” refers to.
- Whether int8 gets separate Holm or is a winner-only robustness gate.
- Pooling counts post-truncation WordPiece occurrences, including specials, and selection uses fp16 macro.
- Smoke results were observed before any subsequent family amendment.

Verification performed: `test_dep_stats.py` passed, and all current changed Python files parse successfully. The Torch API smoke could not run because the read-only environment provides no writable temporary directory.
