# Scientific review for the owner's decision, 2026-10-01

Reviewed draft v10 against the goal of useful, trustworthy knowledge for search engineers. This is an independent essential-only review, not a formatting review. No training, model inference, raw evaluation access, protected reads, or changes to the paper were performed. Later in the review I inspected the parent's new exploratory E15 calculation and receipt.

## Findings, ranked by scientific consequence

### 1. The useful teacher-choice result stops at the closed-form recipe

**Consequence: central framing decision.** The paper asks which index to build if cheap queries may be wanted later. Its strongest experiment establishes a narrower and useful answer: under this particular ridge-table recipe and six-set macro objective, directly screening the candidate tables beats choosing the tower with the highest observed retrieval score. It does not establish how to choose a teacher for the separately optimized Zero recipe or for Nano. The paper already discloses that limitation; this is the main missing scientific bridge, not an undisclosed bug.

The concrete counterexample is stronger than “correlation near zero.” The development screen chooses Stella, whose table scores 0.3974; selecting the best public tower in hindsight chooses gte-large, whose table scores 0.2455. E15 correctly calls the latter a hindsight rule. Do not describe this as a prospective test of following a model leaderboard.

The target also matters. On clean-4, the selected Stella table trails the best of the registered ten by 0.00481 and the best of the pooled 26 by 0.02778. Across the 26 it is fifth on SciFact and eleventh on TREC-COVID, despite winning the six-set macro. These exploratory calculations are in E15 and agree with my independent aggregation of E8/E8x. The screen is useful evidence for choosing against a representative target workload, not a recommendation to choose Stella for arbitrary future workloads.

**Concrete fix:** Make the supported decision explicit in the main argument: evaluate the intended cheap encoder on representative development queries before committing to the document index. Present the observed teacher/student reversal and target sensitivity. Keep the scope “closed-form tables” beside the recommendation. Do not imply that the experiment validated ridge screening as a surrogate for training Zero or Nano. No new experiment is needed to make this narrower paper correct.

**Small decisive experiment if the broader recommendation remains central:** run the served Zero training recipe on the contrasting gte-large and bge-small towers, preserving the existing data and budget policy, and compare with Stella using absolute table scores and equal data/optimization budgets. This tests whether the dramatic ridge ranking survives the change of recipe. It would be an exploratory two-anchor stress test, not a validation of a 26-model predictor. Nano requires separate evidence; do not infer it from this experiment.

### 2. Shuffling is a useful diagnostic, but the controls do not establish the missing capability

**Consequence: do not make “word order explains the table's loss” a central finding.** Section 5 correctly calls the result associative, but “general fragility barely predicts it,” the abstract's contrast with fragile queries, and the interpretation that Nano reads part of the order Stella reads go beyond the controls.

The two quantities are `S - Z` and `S - S_shuffled`; they share the same outcome `S`. The implemented control uses broad quartiles of positive Stella scores, plus a zero-score bin (`e12_failure_modes.py:76-90`), not exact conditioning on Stella score or conditioning jointly on dataset. This is a sensitivity check, not elimination of mathematical coupling. I did not establish that coupling explains the observed result; the retained association may be real.

The “fragility” control makes one isotropic perturbation in the 1024-dimensional space at the same cosine distance as one shuffle. For a document-score difference direction, a uniform tangent perturbation has projection variance proportional to `1 / 1023`; a language-derived perturbation need not. Equal vector distance therefore does not match the ranking-relevant perturbation. The experiment establishes that this particular isotropic-noise control harms retrieval less. It does not rule out general semantic or ranking fragility. Fertility similarly measures subword fragmentation, not token rarity or training support.

**Concrete fix:** retain “shuffle-sensitive queries carry most of the observed net Stella–Zero gap” as a bounded diagnostic. Call the control isotropic vector noise, and describe quartile stratification accurately. Remove the inference about a specific missing capability from the main conclusion. Moving this material to a diagnostic appendix, as the parent proposes, is scientifically sensible. There is no need to add more controls unless a mechanism claim remains a main contribution.

If a mechanism claim is retained, first use existing per-query records to condition more tightly on dataset and Stella score and report `Nano - Zero`, which does not share the Stella outcome. A targeted relation-preserving versus relation-changing query experiment would then be more decisive than more random vector noise.

### 3. The dimension/retention headline exaggerates the independent size evidence

**Consequence: adjust the result used to motivate teacher choice.** The abstract highlights dimension versus retention at -0.76 as evidence that bigger towers distill worse. Retention divides student quality by teacher quality, and dimension correlates positively with that denominator.

Recomputed from the 26 E8/E8x checkpoints, excluding the alternate arctic readout:

| Association | Spearman |
|---|---:|
| Dimension versus absolute table score | -0.44158 |
| Dimension versus tower score | +0.61217 |
| Dimension versus table/tower ratio | -0.76203 |

These numbers agree with E15. There is an actual negative association with absolute table quality, so denominator coupling does not dispose of the observation. It does make the -0.76 a poor standalone measure of the size effect. Dimension also differs from parameter count and is confounded with the checkpoint's training and readout.

**Concrete fix:** lead with absolute table quality and the within-family examples; treat retention as descriptive context. Describe “higher-dimensional checkpoints tended to produce weaker tables under this recipe,” without a causal size claim. The parent can use E15 as the reproducible exploratory receipt. No further experiment is required.

### 4. Build cost can support approachability only when its accounting boundary is explicit

**Consequence: the proposed new framing must not turn partial costs into a turnkey build claim.** E15's Nano figure is 57.26998 training hours multiplied by a recorded historical rental rate, giving $95.27497. It excludes teacher-target preparation, data generation, recipe search, evaluation, and engineering. Zero's 20 minutes and 8–12 hours are historical planning estimates, not measured end-to-end receipts. The 4–7 minute screen timings are coarse intervals between output files with cached material, not cold-start timings.

**Concrete fix:** use “training-loop time and priced rental cost” for Nano, “historical estimate” for Zero, and give the excluded upstream work beside the numbers. These are useful budget anchors, but do not claim that an engineer can reproduce the full result from scratch for $95 or build a new table in 20 minutes. E15's accounting labels are appropriate; I checked its arithmetic and source-field logic but did not independently open the build receipts or historical ledger.

## What I would keep at the center

The strongest supported contribution is a concrete engineering failure mode: **choosing a strong document tower does not guarantee a useful cheap query path, and measuring the candidate cheap path changes the choice materially.** The same project then shows a second decision boundary: impressive encoding savings shrink once the query searches the index. Build-cost accounting connects these decisions if its scope stays honest. This yields a practical paper about designing an index for cheap future queries, with a precise recipe-specific teacher result and a measured deployment result.

I would not present the current evidence as a discovered law of teacher size, a validated general teacher-selection method, or an explanation of what fraction of loss is caused by word order. The oracle and frozen routers are honestly bounded in the present draft; the 15% oracle is explicitly label-aware and the real margin router falls well below always-Nano. They need not carry the main argument. No additional oracle experiment is essential.

No new leakage defect was established in this review. The cleaned E8 fit list, freeze order, exploratory expansion, and E9 smoke disclosure are stated openly. Repeated use of known-test data still makes the broader paper's narrative exploratory; a registered method inside it does not make the overall story a fresh confirmation. The existing registered/exploratory labels should be preserved.

## Access and calculations

Opened only these local content paths (repository relative):

- `CLAUDE.md`, `instructions-m15.md`
- `m15/HANDOFF.md`, `m15/LOG.md` (tail), `m15/PAPER.md`, `m15/MEASUREMENTS.md`, `m15/EVIDENCE.md`
- `m15src/e12_failure_modes.py`, `m15src/e12b_fragility.py`, `m15src/e14_screen_checks.py`, `m15src/e15_decision_audit.py`
- `results/m15_e8_towers.json`, `results/m15_e8x_towers.json`, `results/m15_e12_failure_modes.json`, `results/m15_e12b_fragility.json`, `results/m15_e13_robustness.json`, `results/m15_e14_screen_checks.json`, `results/m15_e10_router.json`, `results/m15_e5_oracle.json`, `results/m15_e15_decision_audit.json`

Also ran `pwd`, `git status --short`, and a filename-only search for guidance files. Independent exploratory CPU calculations used only E8/E8x: absolute/ratio dimension correlations, per-dataset choice ranks, and clean-4 screen correlation. A 10,000-permutation illustration held tower scores/dimensions fixed and permuted student scores; its resulting dimension/retention correlations had median -0.344. This was solely a mathematical-coupling diagnostic, not an inferential test or proposed paper result. All calculation output went to the tool stream; no result file was overwritten. The paper-useful calculations now have the parent's separate E15 receipt. No `work/` content, protected evaluations, or M18 confirmation material was opened.
