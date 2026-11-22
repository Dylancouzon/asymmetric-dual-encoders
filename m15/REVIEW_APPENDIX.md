## Reviewer appendix

This appendix accompanies the v22 internal review copy and is excluded from the publication manuscript. Comments may challenge the thesis, selection, interpretation, or prose.

**Source snapshot:** [540e2bb0](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/540e2bb089a30c83225156945a76b1537beedcd3). **Working branch:** [m15-whitepaper](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/m15-whitepaper). The Google Doc and repository are not automatically synchronized.

### Review and revision workflow

Comment on the sentence, table, or figure you want to challenge; use suggestions for proposed prose. For evidence objections, name the claim and result and say what would change your assessment. Broad thesis discussions can be anchored to the introduction or conclusion. After agreement, accepted changes go into the repository, with consequential decisions and their rationale logged. Update this same Doc with targeted edits so existing discussion stays attached to its context.

Start with [the coworker review guide](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/REVIEW_GUIDE.md); it includes a Codex prompt, owner expectations, and routes for deeper review. [Owner preferences](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/OWNER_PREFERENCES.md) records the audience and editorial goals. Prior agent reviews are not author acceptance.

### Why this direction and what else we considered

The v22 paper asks what an existing document representation predicts about student retrieval quality and how replacing the query encoder changes graph-search effort. Constella illustrates three query budgets over one unchanged Stella index. The current editorial rationale and full-history learning catalog explain alternatives and material retained outside the manuscript. Agent reviews remain opinions about their named snapshots, rather than coauthor acceptance.

See [the editorial rationale](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EDITORIAL_RATIONALE.md), [v21 structural decisions](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/REVISION_PLAN_V21.md), and [the working log](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/LOG.md).

### Availability and provenance

The branch contains the paper, committed results and methods, experiment code, numerical evidence cards, negative findings, historical learning synthesis, and review dispositions. All 55 local targets linked by the current manuscript evidence map are tracked. [The provenance audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/PROVENANCE_AUDIT.md) provides exact historical main-script and method versions, including six exploratory runs with dirty source that was committed afterward.

A fresh clone supports substantive review; it is not a self-contained rerun bundle. Large vectors, model files, environments, and some raw logs are external. The cleaned teacher-screen fit list is untracked and absent on the Mac; its hash alone cannot recover it. The million-passage sampled-ID list is local and untracked. M20 has a verified separate local archive, but its object-storage copy is outstanding. Exact reruns need artifact recovery or verified reconstruction first.

Never read reserved-four raw evaluations, raw closed M18 confirmation, or sealed M19 confirmation during review. Use published aggregate receipts. No repo-wide content search across results/ or work/: select a named-file allowlist from the evidence map first. A listed filename is not authorization to read protected material.

### Evidence for each manuscript claim

The following reproduces the current v22 evidence-map rows. Earlier maps remain available in the linked source as historical context.

**Abstract and §1 encoding medians 31.6, 2.25, 0.044 ms; Table 2 encoding**

[real-query latency](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e1_latency.json), evidence C2. Registered timing; medium queries, batch one, four threads, Apple M5 Pro.

**Figure 1; abstract, contribution 3, and §6 binary latencies 21.3, 2.48, 1.11 ms at nDCG@10 0.7226, 0.6893, 0.6116**

[1M search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e2_ann_msmarco1m.json), evidence C10; plotted by `f0_teaser` in [make_figures.py](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/figures/make_figures.py). Exploratory selection from the registered sweep: binary1 rows, fastest setting within 1% of each encoder's own exact score; encoding on the diagnostic's own queries, measured separately from search.

**Reversal (gte-large 0.5970/0.2455; Stella 0.5745/0.3974), 26-model correlation**

[registered towers](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e8_towers.json), [second recipe](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e19_head_screen.json), evidence C3/C16. Registered comparison; correlation over 26 is exploratory.

**Table 1, quality-only baselines, clean-four sensitivity**

[width model](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e21_width_model.json), evidence C18. Exploratory, analysis pre-specified.

**Prospective test, Appendix C prospective table**

[E24 amended](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e24_prospective_amend1.json), evidence C22. Predictions hash-committed before fitting; transformer student supported, table uncertain.

**§4 selection gaps 0.249/0.107 and 0.035/0.000; prospective 0.043**

[width model](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e21_width_model.json), [E24 amended](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e24_prospective_amend1.json); method E21 in [MEASUREMENTS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/MEASUREMENTS.md). Leave-one-checkpoint-out predictions on the original 26.

**§4 fit times; vector-agreement counterexample**

evidence C13, [vector diagnostics](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e11_mechanism.json). Component costs; descriptive diagnostic.

**§5 recovery gaps, multipliers, censoring, tas-b exclusion, Figure 3**

[E20](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e20_ann_spaces.json), evidence C17. Exploratory, pre-specified, with dated amendments.

**§5 corpus-size control**

[E23](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e23_fiqa25k.json), evidence C21. Exploratory, pre-specified.

**§5 distance ratio and cross-workload all-features model**

[E22](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e22_gap_predictors.json), evidence C20. Exploratory, pre-specified; ratio is family-held-out; cross-workload numbers are the all-features model.

**§5 LightRetriever recurrence**

[E25](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e25_lightretriever.json), evidence C19. Exploratory, specified before encoding.

**Table 2 quality, per-dataset retention, BM25 fusion 0.4933, bge-small/LEAF references**

[BEIR-15 aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m20_beir15_run.json), evidence C1. Registered descriptive evaluation.

**§6 uncompressed ef settings, 13%/11% shares, fixed-ef spread at most 7%**

[1M search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e2_ann_msmarco1m.json), [latency](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e1_latency.json), evidence C10/C2. Registered sweep; shares are extrapolated component sums.

**§6 routing oracle 0.5473/0.5161, simple selectors at most 0.005**

[oracle](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e5_oracle.json), [routers](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m15_e10_router.json), evidence C5/C8. Registered.

**§3.1 and Appendix B recipes, 338,076 pairs, 3,486,034 and 834,462 queries; §6 construction costs**

Sources listed under "Architecture and training exposition" below; evidence C13. Nano optimization measured; Zero retraining a historical estimate.

**§2 retention context (QED 92.5%, LEAF 97.7%, LightRetriever about 95%)**

[related-work notes](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/RELATED_WORK.md). Published figures under other protocols.

**Appendix A registered tests and held-out four**

[Nano final run](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m10_final_run.json), [Zero final run](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m7_final_run.json), [published reserved aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m13_reserved_run.json), evidence C11. Registered; reserved raw content not read.

### All numerical evidence cards

These links include every current evidence card, including unfavorable and superseded findings. Cards describe their own methods and limits; historical paper-section references do not govern v22.

- [C1. Retention across the three tiers](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L5)
- [C2. Query-encode latency](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L34)
- [C3. Tower quality does not predict the table; a dev screen does](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L52)
- [C3b. The tower result on 26 checkpoints](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L92)
- [C4. Three candidate proxies for table quality](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L121)
- [C5. The table's loss is concentrated](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L145)
- [C6. Synthetic prefix retention by query tier](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L170)
- [C7. A fertility router captures a tenth of the headroom](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L229)
- [C8. Routers on Zero's own signals](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L243)
- [C9. Blending two query vectors in one search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L272)
- [C10. Does the saving survive the search (fiqa)](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L297)
- [C10. Does the saving survive the search (msmarco1m)](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L319)
- [C12. The table's gap concentrates on shuffle-sensitive queries](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L356)
- [C11. The one-shot held-out test](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L380)
- [C13. Teacher-choice consequences and component build costs](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L399)
- [C14. Quantized scoring on the same collection](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L429)
- [C15. Query precision with document codes fixed](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L457)
- [C16. The teacher screen under a second student recipe](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L508)
- [C17. ANN effort for table queries across teacher spaces](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L568)
- [C18. Width hides the teacher-quality signal](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L612)
- [C19. LightRetriever's own lookup path under graph search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L651)
- [C20. Predictors of the cross-space recovery gap](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L681)
- [C21. Corpus size and the recovery gap](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L712)
- [C22. Prospective test of the width model](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md#L748)

### Learnings beyond the current paper

[LEARNINGS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/LEARNINGS.md) synthesizes 33 potential lessons from the full research history. It gives evidence, practical consequence, strength, limits, milestone coverage, and candidate manuscript use. It includes failed projections, data/recipe probes, fusion and routing negatives, label-resolution problems, runtime/device failures, and build-cost lessons. These are alternatives for editorial selection, not a claim that every finding belongs in the paper.

[RELATED_WORK.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/RELATED_WORK.md) contains the primary-literature verification trail. [V15 review dispositions](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/REVIEWS/2026-10-01-v15-synthesis.md) links the focused reader and correctness work; the inventory also lists older reviews. [FOLLOWUP.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/FOLLOWUP.md) identifies bounded research that could strengthen a particular claim without full-model retraining.

### Repository contents

[The branch inventory](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/BRANCH_INVENTORY.md) lists all M15 paper, review, code, and result paths and describes the rest of the project. [The complete baseline file listing](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/BRANCH_FILES.tsv) gives all 1,841 tracked baseline paths, blob IDs and byte sizes. It is a metadata inventory, not an input-data archive. The reviewer-package additions are listed separately in the inventory.

| Material | Entry point |
|---|---|
| Current manuscript and rendered PDF | [PAPER.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/PAPER.md); [paper.pdf](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/latex/paper.pdf) |
| Author roster | [AUTHORS.json](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/AUTHORS.json) |
| Current claim/source mapping and numerical cards | [PAPER_EVIDENCE_MAP.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/PAPER_EVIDENCE_MAP.md); [EVIDENCE.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/EVIDENCE.md) |
| Full-history findings | [LEARNINGS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/LEARNINGS.md) |
| Methods and amendments | [MEASUREMENTS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/MEASUREMENTS.md) |
| M15 experiment code and result records | [m15src](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/540e2bb089a30c83225156945a76b1537beedcd3/m15src); [result-file inventory](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/BRANCH_INVENTORY.md) |
| Original Nano/Zero registered tests | [Nano final run](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m10_final_run.json); [Zero final run](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m7_final_run.json) |
| Broad benchmarks and allowed reserved aggregates | [BEIR-15 aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m20_beir15_run.json); [reserved aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/results/m13_reserved_run.json); [canonical benchmarks](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m21/BENCHMARKS.md) |
| Figures, build and evidence generator | [m15/figures](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/540e2bb089a30c83225156945a76b1537beedcd3/m15/figures); [PDF build](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/latex/build.sh); [evidence generator](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/make_evidence.py) |
| Milestones, harness and licensing/history | [ROADMAP.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/ROADMAP.md); [HARNESS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/HARNESS.md); [research](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/540e2bb089a30c83225156945a76b1537beedcd3/research) |
| Owner decisions, current handoff, work log | [OWNER_PREFERENCES.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/OWNER_PREFERENCES.md); [HANDOFF.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/HANDOFF.md); [LOG.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/LOG.md) |
| Reviews and source-version availability | [m15/REVIEWS](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/540e2bb089a30c83225156945a76b1537beedcd3/m15/REVIEWS); [PROVENANCE_AUDIT.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/540e2bb089a30c83225156945a76b1537beedcd3/m15/PROVENANCE_AUDIT.md) |

Historical plans/drafts and the superseded EVIDENCE_INDEX.md remain for auditability. Use the current map and full-history synthesis when interpreting earlier candidate claims. Query measurements use ms; training-component durations use minutes/hours.
