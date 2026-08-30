# Codex adversarial review — M9.0 screen lock (v1 draft)

Reviewer: `gpt-5.6-sol`, reasoning effort high, read-only. Brief: `work/briefs/m9_lock_review.md`.
Date: 2026-08-30. Read-exclusion carried in the brief; the log was grepped afterwards and
shows **zero** reserved-set reads (every match is quoted mandate/ledger text).
Disposition of all 19 findings + the post-number-freedom table: `m9/LEDGER.md` §10.

---

# Verdict: DO NOT COMMIT

The lock is not executable as written and does not satisfy its binding mandate. The worst failures are the three-component decision surface, arm 6’s direct violation of the equal-token requirement, and a guard that permits changing the lock after a run starts.

## BLOCKER

1. **The decision surface directly contradicts the mandate.**

   The mandate defines tuning dev as all six pinned components and says the arms share an equal-component tuning-dev macro ([instructions-m9.md:108](/home/dylan/asymetric-dual-encoders/instructions-m9.md:108), [instructions-m9.md:169](/home/dylan/asymetric-dual-encoders/instructions-m9.md:169)). The ledger instead decides every arm on three components ([LEDGER.md:186](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:186)).

   This cannot be cured by explaining the compute constraint: M9.0 must either obey the mandate or obtain an owner-approved mandate amendment before commitment.

   **Fix:** Amend the mandate explicitly. Preferably:

   - teacher selection may use the three feasible components;
   - if stella remains, student/prompt/mix use all six dev components;
   - if a challenger wins, either fund the full-dev document encode or define and justify a challenger-specific proxy surface before any run.

2. **Arm 6 violates the mandate’s identical non-pad-token budget and does not isolate data mix.**

   The mandate requires identical non-pad-token budgets ([instructions-m9.md:108](/home/dylan/asymetric-dual-encoders/instructions-m9.md:108)). The ledger explicitly refuses to equalize them ([LEDGER.md:162](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:162)).

   Replacing short queries with long documents while holding example count fixed changes at least:

   - role composition;
   - query exposure;
   - non-pad-token dose;
   - padding/FLOPs;
   - likely microbatching and optimizer noise.

   With the probe’s representative lengths, 30% document *examples* could constitute roughly 85% of token mass. A win or loss cannot be attributed to document regression.

   **Fix:** Define arm 6 as a fixed-compute contrast: identical optimizer updates and total non-pad tokens, with a locked document-token share, sampler, microbatching, role batching and loss weighting. If the question is instead “are documents worth extra compute?”, add a separately labelled additive-dose contrast.

3. **The registration guard does not freeze the protocol that produced a number.**

   It checks only the current state at result-write time ([guard9.py:10](/home/dylan/asymetric-dual-encoders/m9src/guard9.py:10)). A session can:

   1. start training under lock A;
   2. observe console/checkpoint results;
   3. amend and push lock B;
   4. write the artifact successfully under B.

   It also:

   - guards only two files, not training/evaluation code or data;
   - accepts HEAD on any remote branch, not specifically `m9-work`;
   - exposes `strict=False` ([guard9.py:51](/home/dylan/asymetric-dual-encoders/m9src/guard9.py:51));
   - permits direct writes—`data.py` already uses them ([data.py:154](/home/dylan/asymetric-dual-encoders/m9src/data.py:154)).

   **Fix:** Create a run-start manifest containing exact lock commit, branch, code blob hashes, data hashes and arm ID. Result writing must require that token and verify nothing changed. Remove `strict=False` from selectable runs and make any warning-bearing artifact mechanically ineligible.

4. **The six arms are not reproducibly specified.**

   Only the incumbent has a model revision. Challenger and student repository aliases are unpinned ([LEDGER.md:111](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:111)). Also absent are exact:

   - tokenizer/config revisions;
   - framework and optimizer implementations;
   - initializer;
   - shuffle and epoch-boundary behavior;
   - partial-batch behavior;
   - mixed-role batch schedule;
   - gradient accumulation;
   - LR scheduler formula;
   - dtype used in retrieval.

   Worse, `M7_ENCODER` uses `setdefault`, allowing an operator-provided value to override the claimed incumbent ([m9base.py:21](/home/dylan/asymetric-dual-encoders/m9src/m9base.py:21)).

   The challenger branch also does not define arm 6’s document targets. “Existing frozen pool vector” is valid only for stella; a selected challenger needs those same raw documents re-encoded in its own space.

   **Fix:** Pin all revisions, dependency versions, exact training/evaluation implementation hashes and branch-specific target construction. Reject conflicting `M7_ENCODER`; do not inherit it from the environment.

5. **The statistic permits both systems to omit the same queries.**

   `align()` verifies only that A and B have the same qid set. If both omit the same difficult queries, it accepts the shrunken surface ([screen_stats.py:30](/home/dylan/asymetric-dual-encoders/m9src/screen_stats.py:30)). The registry has only total query count, not per-component qid manifests or hashes.

   The machine implementation also has no teacher-level function enforcing “evaluate both challengers, then select the larger passing point.” It evaluates one oriented contrast supplied by its caller.

   **Fix:** Pin ordered qid hashes and exact counts for every component. Assert each arm equals that manifest. Implement a single arm-report function that loads registered arm IDs, fixes contrast orientation, shares resamples and applies the two-challenger outcome table.

6. **The parity sample arithmetic is impossible or ambiguous.**

   The ledger asks for 256 query and 256 document texts while also asking for 64/64 from each of five length bins ([LEDGER.md:251](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:251)). Five times 64 is 320 per side, not 256.

   The fp16 rule similarly conflates:

   - a 10,000-text training-target numerical sample; and
   - a SCREEN-DEV retrieval shift requiring qids/qrels.

   Training texts cannot themselves produce a SCREEN-DEV retrieval macro.

   **Fix:** Materialize and hash exact sample IDs at M9.0. Define, for example, 51/51/51/51/52 per side. Specify a separate all-5,367-query retrieval parity check.

7. **Nearly half the intended query corpus is deferred until after recipe selection.**

   `nqopen + triviaqa = 220,632` of an intended `463,418` non-FEVER texts: **47.6%**. Yet the ledger lets M9.2 decide their re-screening after all six screen outcomes are known ([LEDGER.md:45](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:45)).

   This means the selected student/prompt/mix is evaluated on one training distribution and built on a potentially very different one. Rebuilding the missing extended containment index also appears to require the protected bytes M9.1 may not access.

   **Fix:** Either solve and pin their admissibility before arm 1, or exclude both from all of M9 through an explicit mandate amendment. “Deferred, not dropped” is not a lock.

## MAJOR

1. **The surface heavily selects for CQA behavior, not the six-set claim.**

   CQA supplies 35.7% of queries but 66.7% of macro weight. A programmers query has 3.94× the weight of an NQ query; physics has 3.32×.

   This is especially problematic because:

   - the confirmatory six contain no CQA;
   - reserved NDO-3 is 50% CQA;
   - LoTTE is entirely forum-derived.

   It does not contaminate the eventual one-shot six-set result, but it invalidates “the screen found the generally better recipe.” Reserved CQA results would be held-out-slice evidence, not fresh family-level robustness.

   **Fix:** Combine the two CQA components into one family weight, report decision sensitivity under query-pooled and equal-family weights, and require direction stability. Disclose the reserved CQA rows as family-informed.

2. **The 0.004 rule is not literally impossible, but its effective threshold is unstated and probably much larger.**

   For common per-query difference SD \(s\),

   \[
   SE_{\text{macro}}
   =\frac{1}{3}\sqrt{\frac{s^2}{3452}+\frac{s^2}{876}+\frac{s^2}{1039}}
   =0.01631s.
   \]

   A plausible planning range \(s=0.10\)–0.25 gives macro SE **0.00163–0.00408**. Therefore a 97.5% lower bound above zero requires an observed point around **0.0032–0.0080**. At the nominal point `0.004`, the bound can pass only if \(s<0.125\).

   Rough 80% power needs true effects around **0.0054–0.0114**, not 0.004. Student contrasts are likely toward the high-variance end; prompt contrasts may be lower variance.

   The planning memo itself says M8’s 0.004 floor was table-specific and nano needed its own floor ([PLANNING.md:96](/home/dylan/asymetric-dual-encoders/m9/PLANNING.md:96)); the ledger nevertheless imports 0.004 unchanged.

   **Fix:** Use existing DEV per-query contrasts to preregister empirical SDs, MDE and power before training. If power is inadequate, say explicitly that defaults are locked and arms are diagnostic, or replace the rule with a power-feasible selection policy.

3. **The screen dose is an early-training ranking, not evidence about the final-dose ranking.**

   M9 uses 1.94M example presentations versus roughly 201M for LEAF: 0.97%. With batch 128 versus LEAF’s 32, it has roughly 0.24% as many optimizer updates. The planning evidence itself says batch 32 beat 256 because more updates mattered ([PLANNING.md:54](/home/dylan/asymetric-dual-encoders/m9/PLANNING.md:54)).

   Early rankings can systematically favor:

   - the retrieval-tuned bge initialization;
   - easier teacher manifolds;
   - query-only over mix, because mix removes query updates;
   - one prompt policy with faster convergence.

   **Fix:** Use the existing 2/4/6/8-epoch checkpoints for a locked rank-stability test. Continue both sides of any close or trending contrast to 16 epochs, with a predeclared crossing/slope rule. Also pilot batch 32 versus 128; the current choice contradicts the evidence offered for the recipe.

4. **The bootstrap CI ignores training uncertainty.**

   It quantifies query resampling conditional on one seed-0 fitted artifact. It does not establish that a student, prompt or mix is superior as a recipe, nor that retraining seed 0 at a larger dose preserves the result.

   **Fix:** Either describe decisions as artifact-specific, or include multiple screen seeds and aggregate over both seed and query uncertainty. Full-dose replicas of only the eventual winner do not repair winner selection.

5. **FineWeb cannot legally be admitted, but the replacement changes the experiment’s meaning.**

   Counts, a pool-row mask and query fingerprints cannot screen arbitrary FineWeb documents against reserved documents. Any attempt to reinterpret them as satisfying the mandate would be a protocol breach.

   The resulting document arm is instead a narrow Wikipedia/Amazon/SQuAD-derived pool, and it repeats only 72,836 documents eight times. It cannot answer whether LEAF-style broad document regression helps.

   **Fix:** State that arm 6 estimates only repeated regression over the pre-screened M7 pool. Better, sample roughly 582,688 unique eligible documents once at the same dose. Change “reopens within M9” to “may reopen in M10”; spending the M9.4 reserved access occurs too late to change M9 training.

6. **LoTTE “confirmation” is not confirmation.**

   The rule retains the selected recipe whenever it is no more than 0.004 worse by point estimate, with no uncertainty calculation ([LEDGER.md:263](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:263)). That is an asymmetric tolerance veto, not evidence that the screen choice reproduced.

   **Fix:** Rename it a veto/non-inferiority sanity check and define uncertainty. Pin equal-dose, already-trained checkpoint hashes for selected and fallback artifacts before the batch; adoption must not trigger post-LoTTE retraining choices.

7. **Long-query coverage is misstated.**

   Training on long `"passage: "` documents does not exercise long-query input behavior. The sentence claiming the mix arm’s documents cover this limitation ([LEDGER.md:40](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:40)) is false.

   **Fix:** Add safe training-side long queries or a fully specified head+tail probe now. Otherwise state that long-query training coverage is absent and forbid M9.2 changes based on the descriptive `heldout-longq` result.

8. **The throughput fallback is manipulable.**

   “10,000–50,000 real texts” leaves sample size, source composition, length mix and batching open ([LEDGER.md:253](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:253)). Those choices can move the estimate across the 12-hour threshold and halve every arm’s dose.

   **Fix:** Pin one sample manifest, exact size, length/source strata, batches, warmup exclusion and formula for total GPU-hours.

## Explicit post-number freedoms

| Sentence | Permitted reinterpretation | Required fix |
|---|---|---|
| “Carried into M9.2 as an open item” ([LEDGER.md:43](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:43)) | Choose long-query handling after seeing `heldout-longq`. | Lock outcome→action now or forbid changes. |
| “Re-screening them is an M9.2 task with its own G2 amendment” ([LEDGER.md:48](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:48)) | Include/exclude 47.6% of final data after screen results. | Resolve before arm 1 or exclude for M9. |
| “Hash recorded at run time” ([LEDGER.md:59](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:59)) | Change pool/mask/sample and bless the resulting hash afterward. | Compute and register the row hash at M9.0. |
| “Non-pad token counts are reported per arm, not equalized” ([LEDGER.md:165](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:165)) | Call a dose-confounded result a mix effect. | Equalize token/FLOP dose. |
| “unless the head+tail probe runs” ([LEDGER.md:239](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:239)) | Decide whether and how to run it after seeing long-query performance. | Fully specify it or state it will not run. |
| “10,000–50,000 real texts” | Select a workload that triggers or avoids the fallback. | Pin the manifest and exact count. |
| “if both pass, the larger point estimate wins” ([LEDGER.md:212](/home/dylan/asymetric-dual-encoders/m9/LEDGER.md:212)) | Exact tie has no action; machine registry does not encode the joint rule. | Add an exact tie default and implement it. |

## Arithmetic audit

| Quantity | Result |
|---|---|
| `8 × 242,786` | Correct: **1,942,288** |
| `ceil(1,942,288 / 128)` | Correct: **15,175**, but full batches contain 1,942,400 slots—handling of the extra 112 is unspecified |
| Checkpoints | Correct as cumulative ceilings at 2/4/6/8 epochs: **3,794 / 7,588 / 11,381 / 15,175** |
| Four-epoch fallback | Counts/checkpoints are correct, but the last-step partial-batch policy is again missing |
| Query split | Correct: `5×242,786 + 145,672 = 1,359,602` |
| Document split | Wrong: `72,836×8 = 582,688`, not 582,686 |
| Claimed mix total | The stated query count plus true eight doc epochs is **1,942,290**, two over target |
| bge theoretical fp16 weights | `(33.4M + 0.394M)×2 = 67.588 MB` |
| MiniLM theoretical fp16 weights | `(22.7M + 0.394M)×2 = 46.188 MB` |

The size multiplication is internally consistent only for rounded parameter counts and decimal MB. “Both fit” is not yet established for the shipping asset: ONNX initializer dtype, graph overhead, tokenizer and configuration are not included. Define the cap as total shipped repository bytes and measure it.

## MINOR

- The `70 MB` unit and asset boundary are undefined; fix decimal bytes and included files.
- `B=10,000` leaves only about 125 replicates below the teacher’s 1.25% quantile; increase B or report Monte Carlo sensitivity near the decision boundary.
- `"passage: "` does not make query/document inputs disjoint “by construction”; a raw query can begin with that string. Use a reserved role token or assert the pinned tokenized-input sets are disjoint.
- The eligible-document arithmetic assumes `banned_pool_rows` has zero overlap with the FEVER span. Pin the union count and mask hash rather than relying on subtraction.

Read-exclusion honored: I accessed only the named files and did not list, open, grep or inspect any held-out evaluation path or cache.
tokens used
105,017
