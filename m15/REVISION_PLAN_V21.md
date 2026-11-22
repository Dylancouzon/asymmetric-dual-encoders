# v21 decisions (agreed by Claude and GPT-6.1-Sol, 2026-10-02)

Owner direction and the full rounds are in `OWNER_PREFERENCES.md` ("v21 direction", "Settled answers") and `REVIEWS/2026-10-02-v21-sol-rounds.md`. This file records what was agreed so later revisions do not reopen it.

## Thesis

The document index is a durable asset; the query encoder is a recurring cost that can be chosen separately. Two empirical findings and one demonstrated capability:

1. Quality: the space decides how good a cheap query student gets. The strongest original encoder gives the weakest registered table student; conditioning on dimensionality returns the expected quality association; a prospective test supports the head. Fitting and measuring the student is cheap and selects better than the forecast or vector agreement.
2. Search cost: cheap query vectors need about two to four times the graph effort to recover their own exact neighbors, across 25 spaces and in LightRetriever's jointly trained lookup path; relevance consequences depend on the corpus; the original queries' distance ratio flags susceptible spaces.
3. Capability: one index serves a family of query budgets chosen per request, with construction cost, measured quality and encoding, the search setting each path needs, and request-context selection. Section 6 carries the same weight as Sections 4 and 5; it is the payoff, not an illustration.

## Agreed choices

- Problem-first abstract and introduction (owner rules of 2026-10-02); surprises stated with effect sizes after the question.
- Research order; Constella defined in the introduction.
- Main tables: the conditional quality model and the Constella table. Main figures: quality by dimensionality, cross-space recovery gaps, quality against encoding plus search time.
- Tables 1 and 2 of v20 stay separate measurements: merging them would imply one operating point (Sol, round 3).
- Search cost in prose with its workload named first; time shares called extrapolated component-time shares.
- Kept: routing (oracle headroom versus weak simple selectors), one hybrid paragraph with its workload-dependent sign, the vector-fidelity counterexample.
- Moved to the repository: precision control, QED retention comparison (unmatched protocols), gap-predictor feature tables, selection-choice table, Table B1, fusion-depth and routing detail, secondary LightRetriever prompts.
- Appendices: registered test in full; reproduction recipes; the 26-space roster with its scores (the data behind RQ1) and the prospective table.
- Discussion: combined meaning, consequences for a search system in third person, and the settled re-embedding answer.

## Voice contract changes (replace the v17 rules where they conflict)

- "One thread" becomes one question with several axes; a result that changes no production or research decision leaves the paper.
- "Evidence in tables" becomes: a table only when the reader compares rows; otherwise the carrying numbers go in the sentence.
- Limits once means no repeated qualifications; essential scope (fitted versus trained students, which workload each timing comes from) comes before the numbers.
- Kept: third person, effect sizes, surprise stated plainly, evidence labels at experiment level, no internal labels in reader prose, no em dashes, positive statements over "rather than" framing.
