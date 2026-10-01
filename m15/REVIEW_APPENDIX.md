## Reviewer appendix

This is source material for a future Google Doc. Creation is deferred until after the owner's
separate Astra review; refresh the appendix and source snapshot when that review is incorporated.

This appendix is for collaboration and is not part of the publication manuscript. The paper above is v15, unchanged from the pinned repository snapshot. Comments may challenge the thesis, selection, interpretation, or prose.

**Source snapshot:** [5ce1ced9](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/5ce1ced947ec5a75171aaac65f116df2e72c0127). **Working branch:** [m15-whitepaper](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/m15-whitepaper). The Google Doc and repository are not automatically synchronized.

### Review and revision workflow

Comment on the sentence, table, or figure you want to challenge; use suggestions for proposed prose. For evidence objections, name the claim and result and say what would change your assessment. Broad thesis discussions can be anchored to the introduction or conclusion. After agreement, accepted changes go into the repository, with consequential decisions and their rationale logged. Update this same Doc with targeted edits so existing discussion stays attached to its context.

Start with [the coworker review guide](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/REVIEW_GUIDE.md); it includes a Codex prompt, owner expectations, and routes for deeper review. [Owner preferences](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/OWNER_PREFERENCES.md) records the audience and editorial goals. Prior agent reviews are not author acceptance.

### Why this direction and what else we considered

The draft leads with selectable query computation over fixed document vectors because it connects
Constella's actual artifacts to a recognizable relevance/cost decision. The strongest measured
consequence is that encoding savings can be spent again on ANN search. The teacher/table reversal
remains a separate finding about construction before choosing an index.

The strongest objection is that these are partly separate studies: the teacher screen may be the
more surprising research result, while fusion and runtime cases can make the broader paper feel
like a collection of project findings. The current framing is an editorial recommendation, not owner
acceptance or coauthor consensus. Earlier cost/teacher-first emphasis was challenged and revised;
previous agent endorsements did not settle the question.

Alternative directions and their present limits:

- **Teacher selection first:** a memorable 26-checkpoint table result and a tighter research question;
  the tested ridge recipe does not validate teacher ranking for trained Zero/Nano.
- **Training cost and approachability:** useful measured optimization cost; the $95 Nano component
  excludes preparation, failed runs, evaluation, and engineering, and does not price a turnkey build.
- **Benchmark/model release:** clear model tradeoffs; selected comparators are historical targets,
  and a leaderboard is not the owner's requested contribution. Unfavorable results stay visible.
- **ANN/query precision:** a focused systems question with recoverable candidate information;
  native replication is workload-sensitive and the exhaustive control has no native latency remedy.
- **Adaptive query compute:** compatible selection and oracle headroom; no successful production
  router or timed loading/failover is established by the current experiments.
- **Construction recipes and failed shortcuts:** head/data/width/fidelity lessons from older milestones;
  local screens do not isolate the whole rebuild's cause or establish a universal static-model ceiling.
- **Fusion first:** useful quality recovery and a candidate-depth ordering reversal; the historical
  operator/lexical implementation and development scope limit generalization.
- **Serving/edge/adaptation case study:** real caller/device/dtype and label-resolution lessons;
  old synthetic prototypes and inconclusive patches cannot establish released-model edge or relevance gains.

The full [editorial rationale and alternatives](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EDITORIAL_RATIONALE.md) gives the case for each direction,
why it was scoped or omitted, its evidence, proposed strengthening work, and what could reopen the
choice. The [33-lesson catalog](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/LEARNINGS.md) preserves the other findings for selection.

Reviewers should decide whether the fixed-index question should lead, whether the teacher study
belongs here or in a separate paper, which finding they would share with another engineer, which
omission would change the conclusion, and whether one bounded experiment would materially improve
the paper. Another thesis, a split, or substantial removals are valid recommendations.

### Availability and provenance

The branch contains the paper, committed results and methods, experiment code, numerical evidence cards, negative findings, historical learning synthesis, and review dispositions. All 55 local targets linked by the current manuscript evidence map are tracked. [The provenance audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/PROVENANCE_AUDIT.md) provides exact historical main-script and method versions, including six exploratory runs with dirty source that was committed afterward.

A fresh clone supports substantive review; it is not a self-contained rerun bundle. Large vectors, model files, environments, and some raw logs are external. The cleaned teacher-screen fit list is untracked and absent on the Mac; its hash alone cannot recover it. The million-passage sampled-ID list is local and untracked. M20 has a verified separate local archive, but its object-storage copy is outstanding. Exact reruns need artifact recovery or verified reconstruction first.

Never read reserved-four raw evaluations, raw closed M18 confirmation, or sealed M19 confirmation during review. Use published aggregate receipts. No repo-wide content search across results/ or work/: select a named-file allowlist from the evidence map first. A listed filename is not authorization to read protected material.

### Evidence for each manuscript claim

The authoritative current map is [PAPER_EVIDENCE_MAP.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/PAPER_EVIDENCE_MAP.md). The entries below reproduce every map row, with source links pinned to this snapshot.

**§2 compatibility, architectures, same populated collections.** [Zero recipe](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m7/RECIPE.md), [Nano build](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m13_build_record.json), [FiQA search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e2_ann_fiqa.json), [1M search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e2_ann_msmarco1m.json) — Fixed Stella space; three query paths; encoder batches, not timed loading/concurrent switching.

**Table 1 quality and selected-reference Appendix A.** [BEIR-15 aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m20_beir15_run.json), evidence C1 — Descriptive macro; exact original-vector retrieval. Registered earlier tests remain distinct.

**Table 1 and Zero load/first query.** [real-query latency](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e1_latency.json), evidence C2 — Registered timing; medium bucket, three fresh processes, four threads, one Mac. Assets are not RSS.

**§5.2 Nano dose/time/component price; Zero and probe estimates.** [decision audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e15_decision_audit.json), [build record](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m13_build_record.json), [allocation](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m13_build_allocation.json), [Zero ledger](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m7/LEDGER.md) — Exploratory receipt-based accounting; Nano loop measured, Zero historical estimates, probe coarse intervals. Full build not priced.

**§5.1 teacher reversal, 26-checkpoint correlation, task sensitivity.** [registered towers](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e8_towers.json), [frozen penalties](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e8_frozen_lambdas.json), [expansion](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e8x_towers.json), [robustness](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e13_robustness.json), [selection audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e15_decision_audit.json) — One ridge-table recipe, each teacher's own index; 10 registered +16 exploratory; does not validate selector for trained students.

**§5.1 fidelity versus retrieval, loss versus retrieval.** [vector diagnostics](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e11_mechanism.json), [margin diagnostic](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e11b_margin.json), [M7 findings](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m7/FINDINGS.md), [M17 findings](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m17/FINDINGS.md), [M17 status](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m17/STATUS.md) — Multiple local counterexamples. Historical records are summarized, not newly rerun; no loss-class rejection.

**§3.1 original-vector ANN loss, Table 2, Figure 1, query-sample distinction.** [1M search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e2_ann_msmarco1m.json), [FiQA search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e2_ann_fiqa.json), [shared-index audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e15_decision_audit.json), evidence C10 — Registered sweep; exploratory same-quantization restriction. Tier-relative loss, different absolute quality; per-query sum of separate encode/search phases.

**§3.1 early independent model observation.** [M1–M6 findings](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/research/m1-m6-findings.md), [early audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/REVIEWS/2026-10-01-learning-audit-early.md) — Historical LightRetriever/HNSW observation, varying by dataset; not merged into current timing protocol.

**§3.2 and Appendix B precision control.** [fixed-code scoring](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e18_query_precision.json), evidence C15, [methods](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/MEASUREMENTS.md) E18 — Exploratory, fixed codes/vectors/targets, exhaustive coverage and expected boundary ties. No native latency or relevance gain.

**Appendix B strong native pilot and weaker replication.** [pilot](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e16_quantization_pilot.json), [replication](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e17_quantization_replication.json), [semantics audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/REVIEWS/2026-10-01-quantization-semantics.md), evidence C14 — Exploratory native options on one fixed graph per workload; primary ef64 replica uncertainty retained.

**§4.1 lexical fusion macro and domain reversals.** [BEIR-15 aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m20_beir15_run.json), evidence C1 — Descriptive dense/fused routes; no universal fusion improvement.

**§4.1 diagnostic fused quality/cost.** [1M search receipt](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e2_ann_msmarco1m.json), evidence C10 — Separate uncompressed dense+sparse collection versus shared binary dense example; not isolated fusion overhead.

**§4.1 Table 3 depth reversal and operator distinction.** [M12 findings](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m12/FINDINGS.md), [tier 1 receipt](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m12/tier1.json), [tier 2 receipt](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m12/tier2.json), [six-set DBSF](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m12/six_dbsf.json) — Development depth curve is descriptive. bm25s-lucene, convex weight .8; no equivalence claim. Original deep-operator match failed; later DBSF policy override disclosed.

**§4.1 specialized dense gain did not clear fused gate.** [M18 findings](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m18/FINDINGS.md), [M18 status](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m18/STATUS.md), [postmortem](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/research/m17-m19-postmortem-2026-09-13.md) — Published summaries only; small model-judged internal dev. No raw confirmation access or improved model claim.

**§4.2 oracle, weak query selectors, post-search margin.** [oracle](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e5_oracle.json), [fertility router](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e6_router.json), [other routers](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e10_router.json), evidence C5/C7/C8 — Registered methods; label-aware oracle upper bound versus frozen actual policies. Different set counts, repeat search required.

**§4.2 blend negative.** [blend](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e9_blend.json), evidence C9, [methods](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/MEASUREMENTS.md) E9/amendments — Registered descriptive; development-selected blend; pre-freeze SciFact curve print disclosed. No architecture-wide rejection.

**§6 wrong prompt/padding/caller/device.** [M11 status](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m11/STATUS.md), [M14 findings](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m14/FINDINGS.md), [M14 status](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m14/STATUS.md), [middle audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/REVIEWS/2026-10-01-learning-audit-middle.md) — Historical bugs/candidate failures, reference parity qualification; .662 minimum CUDA cosine belongs to rejected fp16 document graph.

**§6 dtype and normalization.** [M21 benchmarks](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m21/BENCHMARKS.md), [M21 status](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m21/STATUS.md), [upstream audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m22/UPSTREAM_FASTEMBED.md), [dtype receipt](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m22_dtype_issue_measurements.json) — Actual caller/dtype counterexamples. Output bytes and values checked separately; fixed preview versus upstream status distinguished.

**§7 source-object leakage and uncertain judgments.** [M18 findings](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m18/FINDINGS.md), [M19 findings](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m19/FINDINGS.md), [postmortem](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/research/m17-m19-postmortem-2026-09-13.md), [late audit](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/REVIEWS/2026-10-01-learning-audit-late.md) — Brief limit; no M19 quality gain claimed and no protected confirmation opened. Detailed patch evidence stays in learning catalog.

**Appendix A original Nano contrasts.** [final run](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m10_final_run.json), [canonical benchmarks](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m21/BENCHMARKS.md) — Fixed-sequence registered rules, clean-four/all-six both present. Broader average is not a replacement test.

**Appendix A original Zero family and clean-four sensitivity.** [final Zero run](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m7_final_run.json), [Zero status](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m7/STATUS.md), [canonical benchmarks](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m21/BENCHMARKS.md) — Failed dense bar, failed Holm BM25 contrast, unresolved original convex/OpenSearch contrast; no equivalence.

**Appendix A originally reserved rows and NDO summaries.** [published aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m13_reserved_run.json), evidence C11 — One-shot registered held-out evaluation; copied published aggregates only. Raw content remains unread.

**Appendix A exposure/fit cleanup.** [registered exposure](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/e8_exposure.json), [screen checks](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m15_e14_screen_checks.json), [fit-cleanup method](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/MEASUREMENTS.md) — Earlier 1.31% overlap disclosed; cleaned fit list for current screen; exploratory teacher exposures not individually audited.

### All numerical evidence cards

[The readable evidence cards](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md) include findings retained and omitted from the manuscript. Each card gives numbers, registered/exploratory status, source, and limits. C10 has two workloads: name FiQA or the million-passage diagnostic when commenting.

- [C1. Retention across the three tiers](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L5) — registered (M20).
- [C2. Query-encode latency](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L34) — registered (E1).
- [C3. Tower quality does not predict the table; a dev screen does](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L52) — registered (E8; lambdas frozen before the six sets were scored).
- [C3b. The tower result on 26 checkpoints](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L92) — exploratory (E8x, E13).
- [C4. Three candidate proxies for table quality](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L121) — exploratory (E11, E11b).
- [C5. The table's loss is concentrated](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L145) — registered (E5).
- [C6. Synthetic prefix retention by query tier](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L170) — registered (E4, E7 folded in).
- [C7. A fertility router captures a tenth of the headroom](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L229) — registered (E6).
- [C8. Routers on Zero's own signals](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L243) — registered (E10).
- [C9. Blending two query vectors in one search](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L272) — registered (E9).
- [C10. Does the saving survive the search (fiqa)](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L297) — registered (E2).
- [C10. Does the saving survive the search (msmarco1m)](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L319) — registered (E2).
- [C12. The table's gap concentrates on shuffle-sensitive queries](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L356) — exploratory (E12, E12b).
- [C11. The one-shot held-out test](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L380) — registered (spent 2026-09-19).
- [C13. Teacher-choice consequences and component build costs](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L399) — exploratory (E15).
- [C14. Quantized scoring on the same collection](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L429) — exploratory (E16, E17).
- [C15. Query precision with document codes fixed](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md#L457) — exploratory (E18).

### Learnings beyond the current paper

[LEARNINGS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/LEARNINGS.md) synthesizes 33 potential lessons from the full research history. It gives evidence, practical consequence, strength, limits, milestone coverage, and candidate manuscript use. It includes failed projections, data/recipe probes, fusion and routing negatives, label-resolution problems, runtime/device failures, and build-cost lessons. These are alternatives for editorial selection, not a claim that every finding belongs in the paper.

[RELATED_WORK.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/RELATED_WORK.md) contains the primary-literature verification trail. [V15 review dispositions](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/REVIEWS/2026-10-01-v15-synthesis.md) links the focused reader and correctness work; the inventory also lists older reviews. [FOLLOWUP.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/FOLLOWUP.md) identifies bounded research that could strengthen a particular claim without full-model retraining.

### Repository contents

[The branch inventory](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/BRANCH_INVENTORY.md) lists all M15 paper, review, code, and result paths and describes the rest of the project. [The complete baseline file listing](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/BRANCH_FILES.tsv) gives all 1,841 tracked baseline paths, blob IDs and byte sizes. It is a metadata inventory, not an input-data archive. The reviewer-package additions are listed separately in the inventory.

| Material | Entry point |
|---|---|
| Current manuscript and rendered PDF | [PAPER.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/PAPER.md); [paper.pdf](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/latex/paper.pdf) |
| Author roster | [AUTHORS.json](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/AUTHORS.json) |
| Current claim/source mapping and numerical cards | [PAPER_EVIDENCE_MAP.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/PAPER_EVIDENCE_MAP.md); [EVIDENCE.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/EVIDENCE.md) |
| Full-history findings | [LEARNINGS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/LEARNINGS.md) |
| Methods and amendments | [MEASUREMENTS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/MEASUREMENTS.md) |
| M15 experiment code and result records | [m15src](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15src); [result-file inventory](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/BRANCH_INVENTORY.md) |
| Original Nano/Zero registered tests | [Nano final run](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m10_final_run.json); [Zero final run](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m7_final_run.json) |
| Broad benchmarks and allowed reserved aggregates | [BEIR-15 aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m20_beir15_run.json); [reserved aggregate](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/results/m13_reserved_run.json); [canonical benchmarks](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m21/BENCHMARKS.md) |
| Figures, build and evidence generator | [m15/figures](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/figures); [PDF build](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/latex/build.sh); [evidence generator](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/make_evidence.py) |
| Milestones, harness and licensing/history | [ROADMAP.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/ROADMAP.md); [HARNESS.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/HARNESS.md); [research](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/5ce1ced947ec5a75171aaac65f116df2e72c0127/research) |
| Owner decisions, current handoff, work log | [OWNER_PREFERENCES.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/OWNER_PREFERENCES.md); [HANDOFF.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/HANDOFF.md); [LOG.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/LOG.md) |
| Reviews and source-version availability | [m15/REVIEWS](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/REVIEWS); [PROVENANCE_AUDIT.md](https://github.com/Dylancouzon/asymmetric-dual-encoders/blob/5ce1ced947ec5a75171aaac65f116df2e72c0127/m15/PROVENANCE_AUDIT.md) |

Historical plans/drafts and the superseded EVIDENCE_INDEX.md remain for auditability. Use the current map and full-history synthesis when interpreting earlier candidate claims. Query measurements use ms; training-component durations use minutes/hours.
