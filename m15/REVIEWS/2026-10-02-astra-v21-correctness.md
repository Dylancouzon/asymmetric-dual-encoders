# v21 correctness review (GPT-6-Astra)

Fresh session 01a0fa53, v21 at 9d50946. Access log audited: result reads limited to files named in PAPER_EVIDENCE_MAP.md (m15_e22_gap_predictors, m15_e8x_towers, m15_e2_ann_msmarco1m). Claude reproduced all six findings from those receipts before fixing. Findings 4 and 6 also existed in v20.

## Brief

# Brief: independent correctness review of v21 (GPT-6-Astra)

Repository: /Users/dylanc/Documents/GitHub/asymetric-dual-encoders, branch m15-whitepaper, HEAD 9d50946. Read-only: change no files, run no experiments, no network.

## Context

m15/PAPER.md is v21 of a research paper for seasoned search engineers and IR researchers. GPT-6.1-Sol wrote it from v20 (commit 51759b2) under decisions agreed with Claude (m15/REVISION_PLAN_V21.md). The rewrite moved material out of the paper and reworded most of the prose. No new experiments were run; every number should come from v20 or a committed receipt. You are the independent correctness reviewer: the writer's sign-off is not independent.

## Read

1. CLAUDE.md and instructions-m15.md (project and paper rules, including registered/exploratory labelling and the required registered test).
2. m15/OWNER_PREFERENCES.md (owner rulings; the "Settled answers" and "v21 direction" sections are closed decisions, do not reopen them).
3. m15/PAPER.md in full (v21).
4. v20 for comparison: `git show 51759b2:m15/PAPER.md`.
5. To check a claim: m15/PAPER_EVIDENCE_MAP.md, m15/EVIDENCE.md, m15/MEASUREMENTS.md, and only the result files PAPER_EVIDENCE_MAP.md names.

Read files with cat/sed/head/git show on named paths only. No recursive or repo-wide content search. Never read results/frozen_eval/untouched-*, reserved qrels caches, work/m9reserve, raw M18 confirmation data, or sealed M19 confirmation data. Open nothing under results/ or work/ unless PAPER_EVIDENCE_MAP.md names that exact file. Never write results/perquery.json.

## Task

Find what is wrong, not what could be phrased differently. In particular:
1. Claim drift: any v21 sentence whose scope, direction, denominator, or certainty differs from v20 or from its receipt (for example a recipe-specific result stated generally, an exploratory result presented as registered, a descriptive difference presented as a test, a time share presented as measured throughput, recovery presented as relevance).
2. Wrong or mismatched numbers, units, workloads, or partitions.
3. Required content lost: the registered test (clean-four with all six beside it, the held-out four, Zero's failed Holm BM25 test), exposure disclosures, licence role of MS MARCO, the exploratory/registered labels, the prospective hash commitment.
4. Internal contradictions between sections, or between text and tables or figure captions.
5. Anything a hostile IR reviewer would use to dismiss the paper that the text could fix without new experiments.

## Rules

Essential-only (CLAUDE.md, Dylan 2026-09-15): report only what changes a result, a claim, evidence integrity, or reproducibility. No wording, style, or preference restructuring. If you find nothing essential in a category, say so plainly. For each finding give: severity (P0 wrong or misleading result; P1 claim drift or lost required content; P2 minor but real), the v21 sentence quoted, the evidence (v20 text or receipt with path), and a concrete fix that keeps the agreed structure. Under 1,000 words. No em dashes.

## Findings

I found six corrections. Two affect claim scope or evidence integrity; four are smaller numerical or reproducibility issues. All can be fixed within the agreed structure, without experiments.

1. **P1: Cross-workload results are attributed to the wrong predictor.**

   **v21, §5:** “Training on two workloads ranks the third at +0.50, but its error is 2.55 points against 2.33 for the fold-mean baseline.”

   The surrounding passage concerns the original-query distance ratio. However, `results/m15_e22_gap_predictors.json` assigns these numbers to `all_features_leave_workload_out`. The distance-only result is `single_qdoc_ratio_teacher`, evaluated across held-out families: correlation 0.6421 and MAE 1.2640. These are different models and validation procedures. Section 7 compounds this by saying the distance ratio predicts the penalty across workloads “less precisely.”

   **Fix:** Identify the cross-workload result explicitly as the model using all features and covariates. Limit the distance-only conclusion to its demonstrated family-held-out prediction.

2. **P1: The original roster’s failed fit has disappeared.**

   **v21, §3.1:** “Ten spaces were registered before results, and 16 were added as an exploratory expansion.”

   V20 Appendix B disclosed: “One expanded-roster checkpoint failed the convergence gate and is excluded.” V21 removes this disclosure while retaining the separate prospective failure. `results/m15_e8x_towers.json`, under `stopped`, records `arctic-embed-m-v1` with development status `STOPPED` and six-set status `missing`; `m15/EVIDENCE.md`, C3b, names it.

   **Fix:** Add one sentence in Appendix B stating that arctic-embed-m-v1 stopped at the convergence gate and was excluded from the original 26-space analysis. This preserves the accounting of attempted versus completed fits.

3. **P2: The selection-regret procedure is no longer reproducible from the paper.**

   **v21, §4:** “On the original roster, the conditional model’s top choices lose 0.249 nDCG@10 for the head and 0.107 for the table relative to the best available students.”

   V20 Table C and Appendix B explicitly say each candidate was predicted by a separate model fitted on the other 25 checkpoints. `m15/MEASUREMENTS.md`, E21, specifies “M2 prediction fitted leave-one-out.” V21 describes family-held-out prediction elsewhere but never identifies this different procedure for selection. Applying the preceding family-held-out procedure is not the reported selector.

   **Fix:** Insert “using leave-one-checkpoint-out predictions, each fitted on the other 25 spaces” into this paragraph.

4. **P2: The fixed-effort latency bound is false. This error already exists in v20.**

   **v21, §6:** “Search time at a fixed `ef` differs by less than 5% across paths.”

   In `results/m15_e2_ann_msmarco1m.json`, uncompressed `search_p50_ms` at `ef=256` is 2.3076 for Zero versus 2.1653 for Stella, about 6.6% higher. At `ef=16`, Nano is 0.9997 versus Stella’s 0.8405, about 19% higher. Even restricting attention to the selected effort range does not support the bound.

   **Fix:** Replace the bound with: “At `ef=128–512`, fixed-effort search medians differ by roughly 4–7% across paths; the larger difference at the selected operating points primarily reflects their different `ef` settings.”

5. **P2: The binary comparison silently changes encoding measurements and omits its exploratory status.**

   **v21, §6:** “In one binary-quantized collection, Zero’s 63-fold encoding advantage over Nano becomes 2.2-fold after search, at different quality levels.”

   Table 2’s encoding medians imply about 51-fold, not 63-fold. The latter comes from the diagnostic’s own encoding measurements: 1.6114/0.025708 ms in `results/m15_e2_ann_msmarco1m.json`. `m15/EVIDENCE.md`, C10, explicitly labels the fixed-quantization selection an “Exploratory shared-index illustration,” choosing the fastest recorded settings within 1% of each path’s exact score.

   **Fix:** Name the one-million-passage diagnostic, its own encoding timings, the exploratory selection, and the 1% criterion in this passage. The numbers themselves need not change.

6. **P2: The routing improvement range excludes a reported result. This also predates v21.**

   **v21, §6:** “Yet query length, subwords per word, and pooled-vector norm improve only 0.002 to 0.005 over random routing at matched Nano share.”

   `m15/EVIDENCE.md`, C8, reports query-length routing at the 0.50 budget improving by **0.0005**, not at least 0.002. Its other query-length improvements are 0.0017 and 0.0018.

   **Fix:** Say “at most 0.005” or report the approximately 0.0005–0.005 range. The weak-selector conclusion remains intact.

The required clean-four/all-six contrasts, held-out four, failed Holm BM25 test, exposure disclosures, MS MARCO licence role, and prospective hash commitment remain present. Beyond the attribution and timing issues above, I found no additional essential internal contradiction or recovery-versus-relevance substitution. No files were changed, experiments run, network accessed, or protected data opened.
## Fixes (diff)

```diff
80c80
< Ranking prediction and choosing the best student are different tests: on the original roster, the conditional model's top choices lose 0.249 nDCG@10 for the head and 0.107 for the table relative to the best available students. Selecting by directly measured development retrieval loses 0.035 and 0.000; on the prospective candidates, both rules select the best head and a table 0.043 below the best.
---
> Ranking prediction and choosing the best student are different tests: on the original roster, the conditional model's top choices, each predicted by a fit on the other 25 spaces, lose 0.249 nDCG@10 for the head and 0.107 for the table relative to the best available students. Selecting by directly measured development retrieval loses 0.035 and 0.000; on the prospective candidates, both rules select the best head and a table 0.043 below the best.
100c100
< The original-query ratio marks susceptibility, and its cause stays open: the student-minus-original change predicts poorly on held-out families. Transfer across workloads is also weaker than transfer across families. Training on two workloads ranks the third at +0.50, but its error is 2.55 points against 2.33 for the fold-mean baseline. This predictor helps rank susceptible spaces within the measured setting; the penalty's magnitude still depends on the workload.
---
> The original-query ratio marks susceptibility, and its cause stays open: the student-minus-original change predicts poorly on held-out families. Transfer across workloads is weaker than transfer across families: the model with all features and covariates, trained on two workloads, ranks the third at +0.50, but its error is 2.55 points against 2.33 for the fold-mean baseline. This predictor helps rank susceptible spaces within the measured setting; the penalty's magnitude still depends on the workload.
124c124
< Summing these component medians and extrapolating to one million queries gives Nano and Zero time shares of 13% and 11% of Stella's, respectively. Large savings versus the original query path remain, but Zero's additional search largely consumes its encoding advantage over Nano. Search time at a fixed `ef` differs by less than 5% across paths; the added work comes from the larger settings required to reach their respective targets. These component-time shares measure neither sustained throughput nor equal absolute relevance.
---
> Summing these component medians and extrapolating to one million queries gives Nano and Zero time shares of 13% and 11% of Stella's, respectively. Large savings versus the original query path remain, but Zero's additional search largely consumes its encoding advantage over Nano. At a fixed `ef` from 32 to 512, search medians differ by at most 7% across paths; the added work comes from the larger settings required to reach their respective targets. These component-time shares measure neither sustained throughput nor equal absolute relevance.
130c130
< The frontier depends on the collection configuration as well as the encoder. In one binary-quantized collection, Zero's 63-fold encoding advantage over Nano becomes 2.2-fold after search, at different quality levels. Figure 3 retains the relevance coordinate that a speed ratio omits. It also shows why an encoder's own exact score is the appropriate reference for approximation loss, while the absolute score remains necessary to compare operating choices.
---
> The frontier depends on the collection configuration as well as the encoder. In an exploratory selection on one binary-quantized collection of the million-passage diagnostic, using that diagnostic's own encoding timings and the fastest settings within 1% of each path's exact score, Zero's 63-fold encoding advantage over Nano becomes 2.2-fold after search, at different quality levels. Figure 3 retains the relevance coordinate that a speed ratio omits. It also shows why an encoder's own exact score is the appropriate reference for approximation loss, while the absolute score remains necessary to compare operating choices.
134c134
< Automatic relevance-based allocation is a separate problem. In the registered routing analysis over 12 datasets, a label-aware oracle selecting Zero or Nano scores 0.5473 against 0.5161 for always-Nano. Yet query length, subwords per word, and pooled-vector norm improve only 0.002 to 0.005 over random routing at matched Nano share. The family exposes useful headroom, but these simple features do not recover it. Request-context selection uses an available operating signal; the oracle requires relevance information unavailable at request time.
---
> Automatic relevance-based allocation is a separate problem. In the registered routing analysis over 12 datasets, a label-aware oracle selecting Zero or Nano scores 0.5473 against 0.5161 for always-Nano. Yet query length, subwords per word, and pooled-vector norm improve at most 0.005 over random routing at matched Nano share. The family exposes useful headroom, but these simple features do not recover it. Request-context selection uses an available operating signal; the oracle requires relevance information unavailable at request time.
144c144
< These results have three consequences for a search system. First, a candidate query encoder is cheap to fit and to measure on development queries for the target index. On our original roster, that measurement selects better than original query quality, and the vector-agreement counterexamples show that cosine fidelity cannot certify the choice. Second, a replacement query path has its own search requirements. The original queries' distance ratio flags susceptible spaces before any replacement is fitted, although it predicts the size of the penalty across workloads less precisely. Third, a compatible family turns query compute into a per-request setting over the same stored documents. Allocation by request context needs no predictor; predicting which encoder will retrieve best for a given query remains unresolved.
---
> These results have three consequences for a search system. First, a candidate query encoder is cheap to fit and to measure on development queries for the target index. On our original roster, that measurement selects better than original query quality, and the vector-agreement counterexamples show that cosine fidelity cannot certify the choice. Second, a replacement query path has its own search requirements. The original queries' distance ratio flags susceptible spaces before any replacement is fitted; the size of the penalty on a new workload still needs measurement on that workload. Third, a compatible family turns query compute into a per-request setting over the same stored documents. Allocation by request context needs no predictor; predicting which encoder will retrieve best for a given query remains unresolved.
217c217
< The anchor includes the original model's pooling and output projection. Student tokenization includes special tokens, has no query prefix, and truncates at 512 tokens; mean pooling and L2 normalization follow lookup. Block conjugate gradient solves the normal equations, requiring every output column to pass relative-residual tolerance 1e-6 in fp32 or 1e-8 in fp64. Penalties range from 1e-4 to 1e-1, with one decade extension when development selection lands at an edge. Exact development nDCG@10 selects the penalty before six-set scoring. The fit list was cleaned after an earlier version had 1.31% held-out-query overlap. Encoder exposure was documented for the registered roster; the exploratory expansion was not individually audited.
---
> The anchor includes the original model's pooling and output projection. Student tokenization includes special tokens, has no query prefix, and truncates at 512 tokens; mean pooling and L2 normalization follow lookup. Block conjugate gradient solves the normal equations, requiring every output column to pass relative-residual tolerance 1e-6 in fp32 or 1e-8 in fp64. Penalties range from 1e-4 to 1e-1, with one decade extension when development selection lands at an edge. Exact development nDCG@10 selects the penalty before six-set scoring. The fit list was cleaned after an earlier version had 1.31% held-out-query overlap. One expanded-roster checkpoint, arctic-embed-m-v1, stopped at the convergence gate and is excluded from the 26. Encoder exposure was documented for the registered roster; the exploratory expansion was not individually audited.
```

## Focused re-review

1. **Fixed.** Cross-workload numbers now identify the all-feature model, and §7 no longer attributes cross-workload magnitude prediction to the distance ratio.

2. **Fixed.** Appendix B restores the failed arctic-embed-m-v1 fit and its exclusion from the 26-space analysis.

3. **Fixed.** The selection-regret passage now specifies predictions fitted on the other 25 spaces.

4. **Fixed.** “At most 7%” over `ef=32–512` matches the receipt and explicitly excludes the larger spread at `ef=16`.

5. **Fixed.** The binary comparison now identifies its exploratory status, workload, distinct encoding measurements, and selection criterion.

6. **Fixed.** “At most 0.005” includes the smaller query-length routing gains without changing the conclusion.

**No fix introduced a new correctness error.** This assessment covers the supplied diff against the previously reviewed text and receipts.
## Focused check after the v21.1 changes

**P2: The restored gte-small retention has an ambiguous subject and denominator.**

> “Its head student retains 1.008 of bge-small’s original retrieval score, and gte-small, nearly a linear image of the backbone, retains 0.978.”

The second ratio describes **gte-small’s fitted head student relative to gte-small’s own retrieval score**, not gte-small relative to bge-small. The preceding explicit bge-small denominator makes the compressed continuation ambiguous.

Evidence: `m15/EVIDENCE.md`, C16, reports gte-small head nDCG@10 of 0.4733 against its own 0.4838, giving 0.978 retention.

**Fix:** “The bge-small head student retains 1.008 of bge-small’s own retrieval score; the gte-small head student retains 0.978 of gte-small’s own score.”

Otherwise, I found nothing essential in the supplied changes. The restored numbers match v20 and the checked receipts; the terminology replacements preserve the measured references; removing the retrospective validation does not remove a required test. Required disclosures and the six previous fixes remain intact.