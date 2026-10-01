# Fable (fresh agent) review of v16 register and scientific weight (2026-10-01)

Brief: `briefs-2026-10-01/v16-scientific-register.md`. Read: brief, PAPER.md, REVISION_PLAN_V16.md, EVIDENCE cards C1/C3/C3b/C10/C13/C15/C16/C17, LEARNINGS.md, FOLLOWUP.md, both 2026-10-01 reviews, RELATED_WORK.md. No protected paths touched, nothing modified. Verbatim.

## 1. Register diagnosis

The owner is right. Specific tells:

- Section titles are slogans, not functions: "The index stays; the query encoder changes", "What the shared index then allows". A paper has Method, Results, Analysis.
- No research question or hypothesis precedes any number. Finding A opens "so we tested the heuristic at scale" with no stated prediction and no falsification rule. Finding B says "which is the hypothesis this section tests" after two paragraphs of background and never writes the hypothesis down.
- Second-person advice: "Screen candidates with the student recipe you intend to use." "What a search engineer can take from this: screen the student you intend to build. Expect a cheap student to need a larger search budget". That is a blog takeaway box.
- Reviewer-directed prose inside results: "A referee's first objection is that this is a property of ridge-fit tables." "That is a plausible explanation, not a reason to set the result aside." A paper reports; it does not argue with an imagined referee mid-table.
- Evidence lives in prose. The 26-teacher roster (C16: dim, family, readout, teacher score, table, head, retention) is the paper's primary dataset and never appears as a table. Readers cannot test the width story themselves.
- Only effect size is Spearman. The within-width-band correlations (+0.47, +0.67, +0.36) are printed without intervals, and the paper never fits a model to its own 26 points.
- Contribution by negation: "We claim no priority for the architecture. What this paper adds is what becomes measurable once the student is cheap to build."

## 2. Scientific weight

Finding A today is a correlation anecdote: Spearman +0.09 with a CI from −0.39 to +0.52 over 26 related checkpoints from eight families. That interval does not reject moderate positive correlation. What would make me sit up is already in the data: pooled null, within-width positive, and retention falling with width at −0.73/−0.76. That is a Simpson-shaped structure. A model, student = f(teacher score, width, family), is the actual finding: width taxes closed-form students, and conditional on width stronger teachers help. That is sharper, more citable, and consistent with gte-large (1024d) producing the worst table. The registered-ten/exploratory-16 split is a free held-out test: fit on ten, predict 16. The paper has the split and does not use it.

Finding B today is a measurement with a null predictor. "In all 25 spaces" is true on two workloads and false on SCIDOCS (8 of 25). Without a predictor it is a benchmark observation. The OOD-ANN literature gives candidate predictors the paper did not try: ratio of query-to-nearest-document distance to document-to-document NN distance, hubness of the returned set, overlap between table and teacher exact top-10. These are computable from cached vectors. Where the evidence cannot support more: three workloads confound corpus size with domain, and the TREC-COVID relative-loss result (median 0.25) is noise over 50 queries. The current data cannot say whether the SCIDOCS attenuation is size.

## 3. Decisions to reopen

Reverse: width as a qualifier. It is the result. Reverse: geometry null in one sentence. A table of tried predictors with CIs is required if the paper claims the penalty is "not yet predictable". Keep: demotion of fusion and routing; I would cut Section 5 to one sentence of future work. Keep: not running native query precision; but then Appendix C and the "separate budget" paragraph are half-in and should go to one line. Title: "Constella" names the released models, but the science is about tables fitted to 26 other teachers. Two papers are stapled together: a model release with registered evaluations, and a methodological study. Yes, a narrower paper is stronger. One paper: teacher selection for closed-form query students, with the width model and a held-out prediction. Finding B becomes one paragraph of consequence unless experiment 2 below gives it a predictor.

## 4. Proposed shape (about 3,500 main-text words)

1. Introduction, 500: problem, RQ1 (does teacher retrieval quality predict student quality, and what does), RQ2 (does student substitution change graph-search effort, and is it predictable), contributions.
2. Related work, 400.
3. Setup, 600: Table A, the 26 teachers (family, dim, readout, teacher score). Two student recipes. Fit list, dev forums, exact scoring.
4. RQ1, 900: H1 written before data ("strongest teacher gives strongest student"). Table B: per-teacher table, head, retention. Table C: correlations and a width-conditioned model, all with CIs, registered-ten fit and 16-checkpoint prediction. Figure 1: student vs teacher scatter, colour by width, both recipes.
5. RQ2, 700: H2 stated. Table D per workload: spaces above 1, censored, multiplier quantiles, recovery gap with across-space CI. Figure 2: recovery gap by space grouped by family, three panels. Table E: predictors tried, correlation, CI.
6. The Constella instance, 300: Table 1 plus registered contrasts, as the worked example.
7. Limitations, 250.
Appendices: registered evaluations, fusion/routing, precision control.

## 5. Three analyses, ranked

1. Width-conditioned model of student quality on the existing 26 points. Claim: conditional on width, teacher quality predicts student quality; width dominates. Required: partial coefficient on teacher score with CI excluding zero after width, replicated by registered-ten fit predicting the 16. Negative: nothing survives family effects, and A stays "unpredictable". Cost: zero GPU, one day.
2. Pre-registered predictors for the recovery gap across 75 space-by-workload points: query-to-doc distance ratio, hubness, table/teacher top-10 overlap, plus effective rank of each teacher's cached fit-query cloud. Claim: the gap is predictable from query placement. Negative: nothing beats the null; B stays a measurement. Cost: cached vectors, CPU, under $20.
3. Deconfound corpus size: subsample FiQA to SCIDOCS size (25k) and rebuild for 25 spaces, same queries. Claim: the SCIDOCS attenuation is size, not domain. Negative: the gap persists at 25k, so domain. Cost: hours of local Qdrant, about $10 pod time to pull vectors.

## 6. Sentences I would not sign

- "they retain 100%, 90.5%, and 81.4% of exact-search quality": Stella is the reference; its retention is a tautology. Write "Nano and Zero retain 90.5% and 81.4% of Stella's exact nDCG@10 (0.5614)".
- "a teacher's own retrieval score was a poor guide to its student's quality under either of two student recipes": the paper's own within-band numbers contradict this unconditionally.
- "in all 25 measured spaces on two corpora" and "a median four times the graph-search budget": the abstract takes the two favourable workloads of three, the secondary measure, and a median over reached spaces with four censored. State all three workloads and the primary measure.
- "Because the cheap students are fit in closed form in minutes, teacher selection becomes testable": it was testable; it became affordable.
- "Four times the ef is not four times the latency, and nothing here measures a production request": stacked hedges in a results section; move to Limitations.
- "That is a plausible explanation, not a reason to set the result aside": delete.
