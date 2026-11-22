# V15 scientific and production-scope review

2026-10-01. **Disposition: the central argument is coherent and supported as an applied study. One narrow method disclosure should be restored for the newly foregrounded fusion result.** No new experiment is needed for the present claims.

## Required clarification: identify the historical fusion implementation

Section 5 correctly says candidate depth and lexical implementation matter, but its numerical depth table does not identify the implementation actually measured. `m12/FINDINGS.md` explicitly states **bm25s-lucene, not Qdrant/bm25**, and warns that a different lexical implementation changes the scores entering DBSF. The convex comparator is the fixed **convex0 weight 0.8** carried through the depth sweep, rather than a freshly optimized convex operator at every displayed budget.

With repository references removed, the manuscript should supply this information in Section 5 or its methods appendix. For example: “The historical depth comparison uses bm25s-lucene scores and the previously selected convex0 weight 0.8 at every depth; substituting Qdrant/bm25 was not evaluated.” Describe the convex normalization in the appendix if the table is meant to be independently reproducible.

This matters to the production claim: naming a stock fusion operator does not establish that an otherwise different native lexical pipeline reproduces these quality rows. The existing cautions about development selection, small differences, policy override, and self-exclusion are correct and should remain. The finding still supports testing operators at the actual budget; it does not prescribe a universally superior operator.

## Does the paper now answer a useful question?

**Yes, more clearly than v14.** It starts with the retained-index constraint and identifies the measured options, then explains three reasons encoder timing alone cannot decide deployment: student ranking, candidate-search work, and the actual served path. It explicitly branches teacher selection into the before-indexing decision. This avoids pretending the 26-teacher experiment can choose a replacement space for an already populated Stella index.

The production advice is supported because it asks for measurements that observed failures show can change the decision. It does not require proving those failures are common. This remains an empirical engineering guide with concrete counterexamples, not a breakthrough architecture, a universal recipe, or a generally optimal policy. The conclusion presents it at that strength.

The new runtime and judgment examples are useful precisely because they show actual failure modes: CPU-only qualification missed the CUDA export failure, and a bounded patch could not be promoted through unresolved labels. They should remain compact examples rather than a growing generic checklist. The large milestone catalog need not all enter this paper.

## Essential checks passed

- **Available budgets:** Table 1 separates exact quality, encoding latency, and assets. The lower bge-small/LEAF broad ranking remains disclosed. Index compatibility is available aligned encoder selection, not timed model loading or concurrent failover.
- **Teacher screen:** Student/teacher retrieval, vector imitation, and optimization loss are distinct. The retrospective strongest-teacher comparison, checkpoint expansion, domain sensitivity, and lack of validation as a selector for trained Zero/Nano remain explicit.
- **ANN:** The shared binary example fixes the physical collection; the full figure permits separately selected collections. The 62-fold and 51-fold encoding ratios refer to different samples. The 1% targets are tier-relative and the implication explicitly starts with the application's absolute quality requirement.
- **Precision:** Candidate coverage targets each tier's own original neighbors. The exhaustive control does not become native scalar8, relevance, or latency evidence. The weaker native replication and Zero's greater miss headroom remain disclosed in the appendix/main text.
- **Fusion and policies:** Negative workload-level fusion outcomes, extra index/search cost, weak routers, and the failed development-selected blend prevent an unconditional “fuse” or “route” recommendation. The 4.71-versus-1.11 ms comparison is correctly identified as different configurations, not isolated fusion overhead.
- **Historical deployment examples:** The 0.662 CUDA cosine is consistent with the named release record and is attributed to a candidate export, not all fp16 models. Repaired loader/pooling defects and unresolved judgment evidence are not sold as present release failures or new relevance wins.
- **Unfavorable evaluations:** Nano's unresolved clean-four LEAF contrast, negative broader/held-out comparisons, Zero's missed original bar, and failed multiplicity-adjusted BM25 claim remain visible. The parent is adding the complete original Zero-family figures and exact Nano dose; those in-progress additions were not independently rechecked here.

## Access record

Read the v15 main text and appendices in `m15/PAPER.md`; checked corrected selection/L33 passages in `m15/LEARNINGS.md`; targeted named-source passages in `m11/STATUS.md`, `m12/FINDINGS.md`, `m20/FINDINGS.md`, `m20/STATUS.md`, and `m21/BENCHMARKS.md`. Reused the prior experiment correctness reviews and all-history audits. No new result receipt, protected content, raw query/qrel, or sealed confirmation was opened. Only this review was written; no experiment, manuscript edit, commit, or push.

## Focused closure after correction and reorder

**The required fusion disclosure is resolved.** Section 4.1 now identifies bm25s with Lucene settings, the fixed development-fitted convex weight 0.8, and the need to compare lexical substitutions separately. The statement no longer implies that naming DBSF certifies a different native lexical pipeline.

The added original Zero-family contrasts, raw intervals, Holm outcome, and descriptive clean-four sensitivities match the named canonical benchmark/status summaries. The exact Nano dose, 199,999,721 presentations, matches its build-status record; it does not misstate compliance as exactly 200M.

The hypothetical 0.68 quality floor and 3 ms budget are a correct reading of the three displayed rows: Nano alone satisfies both; none satisfies both at 2 ms. The paragraph limits the comparison to those measured settings and explicitly labels the thresholds hypothetical. It does not infer an SLO, tail-latency guarantee, globally optimal setting, or tested selection policy.

Moving the pre-index teacher choice after the fixed-index search/fusion chain preserves the necessary decision boundary. The standalone evidence map retains source and protocol distinctions without making repository filenames substitute for manuscript explanations.

**No essential issue remains from this focused closure.** Checked only the changed passages and `m15/PAPER_EVIDENCE_MAP.md`, plus targeted aggregate-summary lines in `m21/BENCHMARKS.md`, `m7/STATUS.md`, `m13/STATUS.md`, and `m15/EVIDENCE.md`. No result payload, protected/raw content, experiment, or general optional re-review.
