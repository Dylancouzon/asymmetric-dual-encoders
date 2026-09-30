# Owner-directed paper review and revision, 2026-10-01

## Judgment

Draft v10 contained enough evidence for a useful applied retrieval paper, but the reader had to infer its purpose from five partly independent contributions. Its title made the ridge screen sound like a validated teacher-selection method for the trained models. The strongest result was buried in correlations, while training and preparation costs were compressed into a setup table. The paper risked reading as a careful project report rather than a study that changes an engineer's decision.

The revised question is: **when is it worth building a cheap query encoder for a document index that will remain fixed?** The answer connects three decisions: evaluate the actual cheap representation rather than infer its quality from its teacher, account for target preparation separately from fitting, and retune approximate search before quoting a system speedup. For an already built index, the teacher is fixed and the first decision is feasibility. Teacher selection applies before indexing, or when an index change is otherwise acceptable.

This is a practical empirical argument. Cheap asymmetric distillation itself is established prior art. The approachability angle makes the study useful, but it cannot honestly carry an architectural novelty claim.

## What is most worth sharing or citing

1. **A consequential teacher/table reversal under a controlled recipe.** The strongest registered teacher, gte-large, scores 0.5970 but produces the weakest registered table, 0.2455. Stella scores 0.5745 as a teacher and 0.3974 as a table. The 0.1519 difference makes teacher choice concrete. It is a fixed-recipe result, not an intrinsic property of either model and not a prospective evaluation of following a model leaderboard.
2. **The representation changes the cost of searching the same index.** At unquantized HNSW ef 16 on the 1M diagnostic, Zero loses 16.3% of its exact quality, against 7.0% for Nano and 4.4% for Stella. After per-tier tuning, roughly 60-fold encoding savings over Nano become 2.2-3.1-fold end-to-end savings. These comparisons preserve each tier's own exact score, not equal absolute quality. This is a reusable search-engine lesson; Qdrant earns credibility by exposing and measuring the poor result as well as the useful one.
3. **A bounded cost anchor for fitting a compatible query model.** Nano's recorded 206,171.92 training seconds equal 57.27 A100 hours. At the recorded historical rental rate that is $95.27 for the training duration. This is useful evidence that the final optimization can fit on one rented GPU. It excludes teacher targets, data generation, recipe search, failed runs, evaluation, export, and engineering. Zero's 20-minute retraining and 8-12-hour re-encoding numbers are approximate ledger estimates, not a complete cold-build quote.

The phrase “screen the student” alone does not explain why the work matters. The choice reversal and the search penalty do. Development-set selection is ordinary practice; the empirical contribution is showing why teacher quality is a misleading proxy under this cheap representation and how large the consequence can be.

## Assumptions challenged

- **The best teacher is the best foundation for cheap queries.** Not under the measured ridge recipe. Nor is Stella a universal recommendation: the pooled-roster screen's chosen table trails the best clean-four table by 0.0278 and is eleventh on TREC-COVID. Representative development data is part of the decision.
- **The ridge ranking applies to trained Zero or Nano.** Not demonstrated. Only Stella has the served recipes. The paper now carries this limit beside the central result, not only at its end.
- **Bigger teachers intrinsically distill worse.** The -0.76 dimension/retention association partly includes denominator coupling. The direct dimension/absolute-table association is -0.44; training and readout remain confounded. The size panel was removed from the main figure.
- **Shuffle sensitivity identifies missing word-order capability.** The gaps share Stella's full score, broad score bins do not remove all coupling, and one isotropic vector perturbation does not match a language perturbation's ranking-relevant direction. The diagnostic remains interesting but does not establish a cause.
- **A cheap training loop makes the full build easy.** Teacher-target preparation and data provenance remain real work. Historical compute prices are not current quotes. The revised paper distinguishes measured time, priced time, and estimates.
- **A frozen index is automatically the best deployment.** Nano is weaker than some small systems on their own indexes. Preserving the index is valuable when it is a genuine constraint; its migration savings were not measured here.
- **The research must sell Qdrant to improve its reputation.** The community gets more value from honest, reproducible retrieval measurements than from a product claim. The engine is named as the measured implementation, without claiming the architecture is unique to it.

## Changes made in draft v11

The title is now *Cheap Queries, Fixed Indexes: The Costs and Limits of Query Encoder Distillation*. The abstract and introduction state the question and decision consequences. A new build section separates preparation, fitting, and deployment, with an explicit cost table and a process another team can follow. The main text shrank from 4,941 to approximately 3,700 words before the appendices.

The teacher comparison leads with the gte-large/Stella reversal, then correlations, then target sensitivity. The main teacher figure now has two panels and no causal-looking size headline. The system section states that its targets preserve each tier's own exact quality. Routing, blending, shuffle diagnostics, synthetic prefixes, and the absorbability proof are supporting appendices. Mixed-workload router latency arithmetic was removed. Registered contrast outcomes remain, and all four held-out aggregate rows are now printed directly in the appendix.

All 14 reference entries received a primary-source pass. LightRetriever and QPP placeholders are replaced with real entries. NanoVDR's positive teacher-quality result varies datasets for one teacher; it does not contradict this cross-teacher experiment. The novelty map is rewritten around the actual empirical contribution. The generated evidence file uses the same narrower interpretation as the paper.

## Goals assessment

| Owner goal | Draft v10 issue | Draft v11 assessment |
|---|---|---|
| New knowledge for IR | Useful cross-teacher result competed with many lesser findings | Fixed-recipe reversal and same-index ANN interaction are central; general trained-student selection remains untested |
| Research paper, not benchmark report | Contribution inventory and frontier often carried the argument | One practical question with three linked decision boundaries; comparator history in appendix |
| Interesting and clear | Owner could not identify the goal after skimming | Concrete failure example near the beginning, explicit build process, shorter main body; still needs the owner's read-through |
| Checkable claims | Sources existed, but some interpretations exceeded controls | E15 adds immutable selection/cost receipt; size, shuffle, and cost scope narrowed; references verified |
| Community reputation | Risk of architecture priority or product framing | Prior art credited; weak and negative results retained; engine behavior measured candidly |
| Lean | Routing/proxy/prefix inventories diluted the main story | Moved to supporting appendices; one small CPU audit, no training, cloud run, or new raw evaluation access |

## Review dispositions

| Review | Finding | Disposition |
|---|---|---|
| Astra correctness | Ridge-to-served-recipe inference unsupported | Scope made explicit throughout; no broad prediction claim |
| Astra correctness | Size/retention coupling | Direct absolute association added via E15; size headline and figure panel removed |
| Astra correctness | Shuffle controls do not isolate mechanism | Diagnostic appendix states coupling and isotropic-control limitations |
| Astra correctness | Partial build cost could imply turnkey reproduction | Measured/priced/estimated boundaries and exclusions beside the figures |
| Sol target reader | Unclear decision, contribution overload | New title, question, structure, and concrete selection consequence; supporting analyses moved |
| Sol literature | Broad approachability/index reuse/routing already prior art | Related work and novelty claims narrowed; no new architecture claim |
| Sol literature | NanoVDR comparison misleading | Different units of analysis stated directly |
| Sol literature | Incomplete bibliography | All printed keys now resolve to full entries and primary links |

## The most valuable next experiment

If extending the empirical claim is worth another experiment, train the served Zero recipe on a contrasting teacher, such as gte-large, and a small-table control, such as bge-small, under comparable data and optimization budgets. Stella supplies the existing anchor. This tests whether the ridge reversal survives the training recipe. A reversal that survives is stronger evidence; a reversal that disappears identifies a useful limit of cheap screening. Either outcome is more informative than adding more ridge checkpoints.

That experiment is not required for the current ridge-scoped paper. No claim about Nano's teacher ranking should follow from it. The current work also does not provide a from-scratch build invoice or a standalone reproduction kit; those would need direct measurements and packaging, not more confident prose.

## Audit and verification

Review files: `2026-10-01-owner-correctness.md`, `2026-10-01-owner-reader.md`, and `2026-10-01-owner-literature.md`. Their local access lists were inspected; each stayed within named-file reads and the protected exclusions. E15 opens only five published JSON receipts and the historical Zero ledger. No protected raw held-out material, M18 confirmation material, or `results/perquery.json` was opened or modified.

E15's selection/correlation calculations match Astra's independent aggregate calculations. Its cost arithmetic was checked against the recorded duration and rate. Evidence and the changed figure were regenerated from committed receipts. The LaTeX build uses the Markdown title and supports wrapping source paths. Final PDF inspection and focused v11 reviews are recorded in `m15/LOG.md` and the focused review files.
