# V15 manuscript evidence map

2026-10-01. File-level provenance belongs here, outside the standalone manuscript. Existing result
payloads were not changed for this rewrite. [EVIDENCE.md](EVIDENCE.md) gives numerical cards;
[LEARNINGS.md](LEARNINGS.md) covers alternative lessons from the complete history.

| Manuscript location / claim | Source of record | Method and scope |
|---|---|---|
| §2 compatibility, architectures, same populated collections | [Zero recipe](../m7/RECIPE.md), [Nano build](../results/m13_build_record.json), [FiQA search](../results/m15_e2_ann_fiqa.json), [1M search](../results/m15_e2_ann_msmarco1m.json) | Fixed Stella space; three query paths; encoder batches, not timed loading/concurrent switching |
| Table 1 quality and selected-reference Appendix A | [BEIR-15 aggregate](../results/m20_beir15_run.json), evidence C1 | Descriptive macro; exact original-vector retrieval. Registered earlier tests remain distinct |
| Table 1 and Zero load/first query | [real-query latency](../results/m15_e1_latency.json), evidence C2 | Registered timing; medium bucket, three fresh processes, four threads, one Mac. Assets are not RSS |
| §5.2 Nano dose/time/component price; Zero and probe estimates | [decision audit](../results/m15_e15_decision_audit.json), [build record](../results/m13_build_record.json), [allocation](../results/m13_build_allocation.json), [Zero ledger](../m7/LEDGER.md) | Exploratory receipt-based accounting; Nano loop measured, Zero historical estimates, probe coarse intervals. Full build not priced |
| §5.1 teacher reversal, 26-checkpoint correlation, task sensitivity | [registered towers](../results/m15_e8_towers.json), [frozen penalties](../results/m15_e8_frozen_lambdas.json), [expansion](../results/m15_e8x_towers.json), [robustness](../results/m15_e13_robustness.json), [selection audit](../results/m15_e15_decision_audit.json) | One ridge-table recipe, each teacher's own index; 10 registered +16 exploratory; does not validate selector for trained students |
| §5.1 fidelity versus retrieval, loss versus retrieval | [vector diagnostics](../results/m15_e11_mechanism.json), [margin diagnostic](../results/m15_e11b_margin.json), [M7 findings](../m7/FINDINGS.md), [M17 findings](../m17/FINDINGS.md), [M17 status](../m17/STATUS.md) | Multiple local counterexamples. Historical records are summarized, not newly rerun; no loss-class rejection |
| §3.1 original-vector ANN loss, Table 2, Figure 1, query-sample distinction | [1M search](../results/m15_e2_ann_msmarco1m.json), [FiQA search](../results/m15_e2_ann_fiqa.json), [shared-index audit](../results/m15_e15_decision_audit.json), evidence C10 | Registered sweep; exploratory same-quantization restriction. Tier-relative loss, different absolute quality; per-query sum of separate encode/search phases |
| §3.1 early independent model observation | [M1–M6 findings](../research/m1-m6-findings.md), [early audit](REVIEWS/2026-10-01-learning-audit-early.md) | Historical LightRetriever/HNSW observation, varying by dataset; not merged into current timing protocol |
| §3.2 and Appendix B precision control | [fixed-code scoring](../results/m15_e18_query_precision.json), evidence C15, [methods](MEASUREMENTS.md) E18 | Exploratory, fixed codes/vectors/targets, exhaustive coverage and expected boundary ties. No native latency or relevance gain |
| Appendix B strong native pilot and weaker replication | [pilot](../results/m15_e16_quantization_pilot.json), [replication](../results/m15_e17_quantization_replication.json), [semantics audit](REVIEWS/2026-10-01-quantization-semantics.md), evidence C14 | Exploratory native options on one fixed graph per workload; primary ef64 replica uncertainty retained |
| §4.1 lexical fusion macro and domain reversals | [BEIR-15 aggregate](../results/m20_beir15_run.json), evidence C1 | Descriptive dense/fused routes; no universal fusion improvement |
| §4.1 diagnostic fused quality/cost | [1M search receipt](../results/m15_e2_ann_msmarco1m.json), evidence C10 | Separate uncompressed dense+sparse collection versus shared binary dense example; not isolated fusion overhead |
| §4.1 Table 3 depth reversal and operator distinction | [M12 findings](../m12/FINDINGS.md), [tier 1 receipt](../m12/tier1.json), [tier 2 receipt](../m12/tier2.json), [six-set DBSF](../m12/six_dbsf.json) | Development depth curve is descriptive. bm25s-lucene, convex weight .8; no equivalence claim. Original deep-operator match failed; later DBSF policy override disclosed |
| §4.1 specialized dense gain did not clear fused gate | [M18 findings](../m18/FINDINGS.md), [M18 status](../m18/STATUS.md), [postmortem](../research/m17-m19-postmortem-2026-09-13.md) | Published summaries only; small model-judged internal dev. No raw confirmation access or improved model claim |
| §4.2 oracle, weak query selectors, post-search margin | [oracle](../results/m15_e5_oracle.json), [fertility router](../results/m15_e6_router.json), [other routers](../results/m15_e10_router.json), evidence C5/C7/C8 | Registered methods; label-aware oracle upper bound versus frozen actual policies. Different set counts, repeat search required |
| §4.2 blend negative | [blend](../results/m15_e9_blend.json), evidence C9, [methods](MEASUREMENTS.md) E9/amendments | Registered descriptive; development-selected blend; pre-freeze SciFact curve print disclosed. No architecture-wide rejection |
| §6 wrong prompt/padding/caller/device | [M11 status](../m11/STATUS.md), [M14 findings](../m14/FINDINGS.md), [M14 status](../m14/STATUS.md), [middle audit](REVIEWS/2026-10-01-learning-audit-middle.md) | Historical bugs/candidate failures, reference parity qualification; .662 minimum CUDA cosine belongs to rejected fp16 document graph |
| §6 dtype and normalization | [M21 benchmarks](../m21/BENCHMARKS.md), [M21 status](../m21/STATUS.md), [upstream audit](../m22/UPSTREAM_FASTEMBED.md), [dtype receipt](../results/m22_dtype_issue_measurements.json) | Actual caller/dtype counterexamples. Output bytes and values checked separately; fixed preview versus upstream status distinguished |
| §7 source-object leakage and uncertain judgments | [M18 findings](../m18/FINDINGS.md), [M19 findings](../m19/FINDINGS.md), [postmortem](../research/m17-m19-postmortem-2026-09-13.md), [late audit](REVIEWS/2026-10-01-learning-audit-late.md) | Brief limit; no M19 quality gain claimed and no protected confirmation opened. Detailed patch evidence stays in learning catalog |
| Appendix A original Nano contrasts | [final run](../results/m10_final_run.json), [canonical benchmarks](../m21/BENCHMARKS.md) | Fixed-sequence registered rules, clean-four/all-six both present. Broader average is not a replacement test |
| Appendix A original Zero family and clean-four sensitivity | [final Zero run](../results/m7_final_run.json), [Zero status](../m7/STATUS.md), [canonical benchmarks](../m21/BENCHMARKS.md) | Failed dense bar, failed Holm BM25 contrast, unresolved original convex/OpenSearch contrast; no equivalence |
| Appendix A originally reserved rows and NDO summaries | [published aggregate](../results/m13_reserved_run.json), evidence C11 | One-shot registered held-out evaluation; copied published aggregates only. Raw content remains unread |
| Appendix A exposure/fit cleanup | [registered exposure](e8_exposure.json), [screen checks](../results/m15_e14_screen_checks.json), [fit-cleanup method](MEASUREMENTS.md) | Earlier 1.31% overlap disclosed; cleaned fit list for current screen; exploratory teacher exposures not individually audited |

## Editorial disposition

- Foregrounded L01/L10 (fixed-index budgets and ANN), L02/L03/L07 (construction), L15–L19
  (fusion and real versus ideal routing), L26/L27 (actual serving contract).
- Retained L11/L12 as bounded query-precision research, with the weaker native replica visible.
- Used L24 to inform a brief evaluation lesson; removed the underexplained internal patch story
  from the main text. Full L21 remains available for a later case study.
- Cut main-text/appendix bulk on absorption algebra, synthetic prefixes, shuffle examples,
  scalar teacher diagnostics, router threshold tables, and release chronology. The catalog and
  existing evidence retain them. Removal is editorial, not a change in result status.
- Reader-requested hypothetical quality/time example is a worked interpretation of Table 2,
  not a new measured policy or prospectively chosen threshold.

The companion repository is the manuscript's artifact-availability reference. Internal paths are
not printed as scholarly citations or substituted for methods. Primary-literature provenance is
in [RELATED_WORK.md](RELATED_WORK.md).

Owner reiterated the whitepaper genre during this revision. The final manuscript replaces recurring instructional prompts with findings and discussion; the learning catalog retains decision-oriented author notes.

## v16 additions (2026-10-01)

Section numbers in the table above refer to v15; v16 reorders the manuscript (1 motivation, 2 family
and setup, 3 Finding A, 4 Finding B, 5 shared-index options, 6 limits, Appendices A to D). The rows
below cover the material v16 adds; every other v15 row still maps to the same receipts.

| Manuscript location / claim | Source of record | Method and scope |
|---|---|---|
| §1 production index-migration motivation (updated 2026-10-02) | [Primary-source migration notes](RELATED_WORK.md), Qdrant embedding-model migration procedures cited in the manuscript | Qualitative migration requirements and resource implications; no production time, cost, or downtime estimate. Replaces the BEIR encoding-duration illustration; historical M20 measurements remain in the repository |
| §2 Figure 1 per-dataset retention; word-order correlate | [BEIR-15 aggregate](../results/m20_beir15_run.json), evidence C1; [failure modes](../results/m15_e12_failure_modes.json), evidence C12 | Descriptive per-dataset ratios; exploratory per-query correlation sharing Stella's score |
| §3 second student recipe, Figure 3, correlation table, strongest-teacher comparison, width qualifier | [E19 result](../results/m15_e19_head_screen.json), evidence C16; [method E19](MEASUREMENTS.md); width analysis in [LOG.md](LOG.md) | Exploratory, analysis pre-specified before scoring; frozen bge-small head; same fit list, selection order, indexes as E8/E8x; width bands exploratory |
| §4 cross-space ANN effort, Figure 5, multipliers, recovery gaps, geometry null, tas-b exclusion | [E20 result](../results/m15_e20_ann_spaces.json), evidence C17; [method E20 and amendments 1 to 4](MEASUREMENTS.md); [pre-spend review](REVIEWS/2026-10-01-astra-e19-e20-prespend.md) | Exploratory, analysis pre-specified; loss-based multiplier primary, recovery secondary; two builds; one engine; latency not a claim |
| §4 search time similar at fixed ef; ef 128/256/512 within 1% | [1M search](../results/m15_e2_ann_msmarco1m.json) rows with quant none | Registered sweep rows read directly |
| Appendix A Nano decision bounds, Zero clean-four intervals, NDO-3 definition | [canonical benchmarks](../m21/BENCHMARKS.md), [published reserved aggregate](../results/m13_reserved_run.json) | Registered outcomes copied in full |
| Appendix B fidelity versus retrieval | [vector diagnostics](../results/m15_e11_mechanism.json), evidence C4; [M7 findings](../m7/FINDINGS.md), [M17 findings](../m17/FINDINGS.md) | Multiple local counterexamples |
| Appendix D fusion and routing | evidence C1, C5, C7, C8, C9, C10 as in the v15 rows for §4.1 and §4.2 | Unchanged receipts, moved out of the main text |

Review records for v16: `REVIEWS/2026-10-01-andrey-review-v16.md`, `REVIEWS/2026-10-01-astra-correctness-v16.md`, `REVIEWS/2026-10-01-sol-reader-v16.md`.

## v17 additions (2026-10-01, frozen-index frame)

Section numbers now: 1 introduction, 2 related work, 3 setup, 4 RQ1, 5 RQ2, 6 instance, 7 limits,
Appendices A to D. Rows above map the same receipts; new or moved material:

| Manuscript location / claim | Source of record | Method and scope |
|---|---|---|
| Abstract, §1, §4.2 to 4.4: width-conditioned model, held-out ten-to-16 prediction, partial correlations, selection regret | [E21 result](../results/m15_e21_width_model.json), evidence C18; [method E21](MEASUREMENTS.md) | Exploratory, pre-specified before running; 26 related checkpoints, three width levels; hypothesis formed after seeing all 26, disclosed |
| §2 joint-training contrast (LightRetriever about 95%) | [related-work notes](RELATED_WORK.md), item 4 | Published figure from the cited paper; different protocol and corpus |
| §3.1 Table A per-width summary; Appendix B full roster | [E19 result](../results/m15_e19_head_screen.json) configs; E8/E8x pooling fields | Derived from the committed rows |
| §6 what each path computes and where it loses; internal jargon workload below BM25 | [BEIR-15 aggregate](../results/m20_beir15_run.json), evidence C1; [M18 findings](../m18/FINDINGS.md) dev table | Descriptive; M18 is a small internal model-judged surface, summary only |
| §6 type-pause-submit capability; prefix retention | [E2 receipts](../results/m15_e2_ann_fiqa.json) (same populated collections); [prefix card](../results/m15_e4_prefix.json), evidence C6 | Capability statement; prefixes are synthetic, no session measured |
| §6 Table 2 CPU hours per million queries | [latency](../results/m15_e1_latency.json) C2; [1M search](../results/m15_e2_ann_msmarco1m.json) quant none rows at ef 128/256/512 | Arithmetic on registered medians; not a throughput measurement |
| §6 build costs | [decision audit](../results/m15_e15_decision_audit.json), evidence C13; E19 timings in [LOG.md](LOG.md) | Component costs only |
| §7 three specialization attempts without an improved encoder | [M17](../m17/FINDINGS.md), [M18](../m18/FINDINGS.md), [M19](../m19/FINDINGS.md) findings | Summaries only; sealed material unopened |
| §5.4 Table E, abstract, §2 and §6 LightRetriever sentences, §7 | [E25 result](../results/m15_e25_lightretriever.json), evidence C19; [method E25](MEASUREMENTS.md); [pre-spend review](REVIEWS/2026-10-01-astra-e25-prespend.md) | Exploratory, pre-specified; released adapter at the pinned revision over Qwen2.5-1.5B; our dense reproduction; one engine; GPU timing is context only |
| §4.3 prospective test, Table C prospective rows, Appendix B prospective roster, Figure 1 triangles | [E24 amended assembly](../results/m15_e24_prospective_amend1.json) (first assembly [kept](../results/m15_e24_prospective.json)), evidence C22; [method E24 and amendment 1](MEASUREMENTS.md); prediction commitment recorded in the result | Exploratory; predictions hashed before any student was fitted; one table fit stopped at the convergence gate, so head n=8 and table n=7 |
| §5.3 Table F gap predictors; Appendix C feature definitions | [E22 result](../results/m15_e22_gap_predictors.json), evidence C20; [method E22 and amendment 1](MEASUREMENTS.md) | Exploratory, features declared before computation; clustered intervals; fold-mean baseline |
| §5.2 corpus-size paragraph (E23) | [E23 result](../results/m15_e23_fiqa25k.json), evidence C21; [method E23](MEASUREMENTS.md) | Exploratory; seeded subsample keeping judged-relevant documents |

## Owner narration pass (2026-10-02)

The main-text claims continue to use the same receipts. Family-adjusted E21 fits are now in
Appendix B; per-dataset Constella commentary is in Appendix A; the internal-workload example is
in Appendix D. Build-component detail remains in Appendix B, and secondary placement features
remain in Appendix C. Scientific tables and the whole manuscript's numeric-token set are
unchanged. Claim wording now emphasizes predictive associations and the distinction between
the screen's fitted baselines and the separately trained release models. E24 assembly history
remains in MEASUREMENTS.md and the evidence cards; the manuscript gives the completed-fit rule.

## Architecture and training exposition (2026-10-02)

| Manuscript location / claim | Source of record | Method and scope |
|---|---|---|
| §3.2, Appendix B fitted table objective, mean pooling, contextual teacher-output anchor | [screen implementation](../m15src/e8_towers.py), [count matrix and ridge objective](../m7src/stage0_ridge.py), [contextual initialization](../m8src/init_m8.py), [query preprocessing](../m7src/table.py), [E8 methods](MEASUREMENTS.md) | Corrects former square-root pooling and input-token-embedding descriptions; measurements unchanged. Fitted table learns rows only |
| §3.2, Appendix B frozen-feature head and trace-scaled ridge | [head screen implementation](../m15src/e19_head_screen.py), [E19 result](../results/m15_e19_head_screen.json) | Frozen bge-small features, biased projection, same fit list and selection order as table |
| §3.5, Appendix B released Zero architecture, two objectives, data, optimization, export | [selected recipe](../m7/RECIPE.md), [distillation phase receipt](../results/m7_run_p35b-2m.json), [contrastive phase receipt](../results/m7_run_p35w-2m-s2500.json), [training implementation](../m7src/train.py), [pooling selection](../results/m7_lever4_pooling_p35w-2m-s2500.json), [release identity](../m11/STATUS.md), [usable pair count](../m7/LEDGER.md) | Frozen Stella targets; learned rows and weights; mean pooling in training, square-root count pooling adopted afterwards. The historical release card's L2-only summary is incomplete; receipts and code govern this exposition |
| §3.5, Appendix B released Nano architecture, data counts, trainable backbone, dose and selected checkpoint | [build record](../results/m13_build_record.json), [build configuration](../m13/build_config.json), [architecture and objectives](../m10src/nano10.py), [trainer](../m10src/trainer10.py), [corpus sampler](../m10src/corpus_loader.py), [recipe selection](../results/m10_screen_verdicts.json), [development estimator](../m10src/cov_macro.py), [generation recipe](../m10src/gen.py), [generation driver](../scripts/m10_gen_build.py), [filtering](../m10src/assemble10.py), [data provenance](../results/m10_data_manifest.json) | Three-layer linear head, ridge warm start, joint squared-L2 optimization; query/document targets remain Stella's. Exact dose is example presentations, not distinct texts. Registered screening defaults do not prove a benefit for each component |

No experiment, immutable result, release artifact, or protected raw evaluation was changed or
reopened. The 26 original encoders remain identified by their roster properties and source
configuration; the expanded recipes concern the students fitted or trained in this project.

| v19 §4.4 / Table C panels and Appendix B chosen encoders | [E21 aggregate](../results/m15_e21_width_model.json), [E21 selector implementation](../m15src/e21_width_model.py), [E24 prospective aggregate](../results/m15_e24_prospective.json) | Absolute best-minus-selected score gap; E21 predictions leave each candidate out; E24 trains on the original 26. Layout and explanation changed, not results. |

| v20 §4.2/Appendix B clean-four sensitivity | [E21 aggregate](../results/m15_e21_width_model.json) pooled/table/clean4 | Quality coefficient interval slightly crosses zero; dimensionality interval remains negative. Development/public rank agreement .88 all-six versus .68 clean-four; partition sensitivity is not a causal exposure estimate. |
| v20 §6 and Appendix C timing-workload scope | [E2 method](MEASUREMENTS.md), [MS MARCO diagnostic receipt](../results/m15_e2_ann_msmarco1m.json), [encoding receipt](../results/m15_e1_latency.json) | Positive-preserving 1M subset, 6,980 development queries; separate 100 medium-query encoding sample. Optimistic relevance relative to full corpus; no transfer of ANN magnitude/direction implied. |
| v20 Appendix C precision-table endpoint | [E18 receipt](../results/m15_e18_query_precision.json) primary_candidate_budget=40 | Displayed coverage is at 40 candidates; 10/40/100 are the broader control sweep. |
