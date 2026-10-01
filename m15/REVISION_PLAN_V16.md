# Proposed v16 revision plan (Astra + Fable, 2026-10-01, revised after owner round 3)

**Status: proposed, awaiting owner approval of the full plan.** Experiment A is owner-approved; B was conditional on Astra's agreement, which is recorded in `REVIEWS/2026-10-01-astra-fable-plan-rounds.md`. Owner direction is in `OWNER_PREFERENCES.md` (2026-10-01 entries). The earlier draft of this file (two-part "use a family / build a family" outline) is superseded by this version; its content survives in the plan rounds file.

## Goal the plan serves

Credibility in IR research: findings a search engineer or IR researcher would cite and argue about. Not model usage, not a benchmark report. The owner judged v15 thin on whitepaper-grade findings; today it has one (the teacher reversal). The plan adds a second and generalizes both using the project's unique asset: 26 teacher spaces with fitted tables, cached fit-query vectors, and per-teacher six-set document indexes.

## Spine and title

Working title: **Constella: Teacher Selection and Search Cost in Query-Side Distillation.**

Spine (Astra's wording): inexpensive student probes make teacher selection empirically testable; the selected query representation must then be evaluated against both exact retrieval and approximate-search cost. Cost is the enabling method, not a novelty claim. QED, EmbedDistill, LEAF, and pyNIFE also keep the document tower frozen and are attributed; what is ours is closed-form fitting without gradients in minutes, which made a 26-teacher screen affordable, and what that screen revealed. Fitting time stays separate from target generation, document encoding, and index construction.

## Outline (about 3,000 main-text words)

| Section | Purpose |
|---|---|
| Abstract | Frozen index, cheap students, two findings: teacher quality does not predict student quality under the tested recipes; cheap query vectors need more search effort over the unchanged index. |
| 1. Teacher selection and the cost of testing it | Define Stella, Zero, Nano. Why the index is the asset (measured 39.9 h to encode 23.7M BEIR-15 documents with Stella on one RTX 3080). Why cheap construction matters: it makes screening affordable. Prior art with attribution. |
| 2. The family and the setup | Only what a reader needs to interpret results: space, pooling, parameter counts, compatibility contract, exact quality and encoding cost, per-dataset retention figure (named datasets, spread, exposure annotations, one exploratory word-order correlate). |
| 3. Finding A: teacher quality does not rank student quality | Existing 26-checkpoint table screen; new cross-recipe transfer (experiment A); surrogate failures (loss and cosine do not certify retrieval); complete cost boundaries. Outcome-contingent wording: transferable rankings, or recipe-specific screening. Challenges the common strongest-teacher heuristic; does not claim to contradict a stated prior claim. |
| 4. Finding B: cheap queries change the search problem | Effort multiplier on the uncompressed Stella collections (ef 512/256/128; 3.58/2.18/1.53 ms); geometry as exploratory evidence; new cross-space result (experiment B); query precision as supporting evidence, no promised remedy. |
| 5. What the shared index then allows | Brief: three budgets over one populated collection; fusion recovery with the lexical-implementation distinction; oracle headroom versus tested routers and the failed blend. |
| 6. Limits and implications | One consolidated limits section; one caveat sentence per finding in the body is the target. |
| Artifact statement | Released models, qualified runtime (one sentence on the CUDA fp16 export failure), reproducibility materials, and any inputs still missing. |

Appendices: protocol; registered clean-four/all-six contrasts and the reserved-four table in full with bge-small, LEAF, BM25 as reference points and one plain sentence per table; complete per-dataset scores and CQADupStack aggregation; construction details and exposure; search and precision controls; experiment A and B tables. Add HNSW and DBSF references; BCT and FCT leave with the serving section.

## Cuts

The 62-fold to 2.2-fold ratio headline; the hypothetical 0.68 / 3 ms example; the standalone serving section; the main-text fusion-depth table (one appendix line); Nano-versus-bge-small argument in the main text; repeated per-paragraph caveats; any economical-specialization framing without a result.

## Experiments (designs and decision rules in `FOLLOWUP.md`)

- **A. Cross-recipe teacher screen** (owner-approved): frozen bge-small backbone, closed-form ridge head per teacher, lambda frozen on the two dev forums before six-set scoring, same 26 checkpoints, each against its own index. Pre-specified analysis written before scoring. Optional MiniLM-L6 backbone only if it adds a different test.
- **B. Cross-space ANN effort** (Astra agreed; awaiting owner): uncompressed HNSW on FiQA and TREC-COVID for all 26 spaces from cached E8 document vectors, two graph builds each, ef sweep 16 to 512 for teacher queries and table queries, per-space geometry. Censor unreachable targets; builds are repetitions, not samples; related checkpoints limit independence.
- Step zero for both: start the pod, verify the E8 caches (fit-query vectors, six-set document vectors, tables) exist; if not, re-encode (about 8 to 10 A100 hours).
- Rejected by owner: native scalar8 query precision (a knob, not knowledge). Deferred by owner: domain-fit table (future work).

## Pre-circulation task

Recover and archive the cleaned 337,981-query fit list and the 1M sampled-ID list, or disclose them as unavailable in the artifact statement.

## Logging

Every decision in this plan is dated in `OWNER_PREFERENCES.md` (owner direction) or `LOG.md` (work record). Experiment methods go to `MEASUREMENTS.md` as E19 (A) and E20 (B) before scoring; receipts to `results/m15_e19_*.json` and `results/m15_e20_*.json`; cards to `EVIDENCE.md`.
