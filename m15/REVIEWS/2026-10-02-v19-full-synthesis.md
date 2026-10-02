# Independent full-paper review synthesis, v19

Owner-requested parallel GPT-6-Astra and GPT-6.1-Sol reviews of manuscript commit 68509e714d869fa492ae1b573234765d532bad0e. Both received the intended audience, goals, full preferences, current revision history and source boundaries, and read the whole paper including appendices/references. Their independent detailed records are [Astra](2026-10-02-astra-v19-full.md) and [Sol](2026-10-02-sol61-v19-full.md). These are review findings, not new owner decisions or whole-paper acceptance. No manuscript edits in this review batch.

## Agreed verdict

A coherent, credible paper exists here, and the evidence is more interesting than a model-release benchmark. It is not the best version yet. The argument is coherent at the experimental-design level but under-explained at the narrative level. Neither reviewer recommends splitting the paper, importing historical inventories, or starting another experiment campaign.

The common thesis: independent query-encoder choice faces an exact-quality constraint and an approximate-search constraint. Cross-space fits measure both; separately trained Constella illustrates combined costs in one space. There is no established common causal mechanism or validated transfer of RQ1 predictions to trained releases.

## Parent assessment and proposed bounded next revision

1. **Narrative synthesis and bridge (both; necessary).** State the exact-then-approximate experimental logic near the RQs and transitions. Introduce Constella as a separately trained deployment illustration. End with a short combined interpretation before limitations. Avoid repeatedly restating boundaries in every section.
2. **Outcome and scope alignment (both; necessary).** RQ1 models absolute student nDCG, while retention is a ratio. E22 predicts recovery gap at ef=64, not required ef or latency. Distinguish inexpensive construction from measured cheap inference. Make prospective support head-specific; retain table uncertainty. Describe original relevance as a measured query/document-pair outcome, not a property inherited solely from document encoder.
3. **Standalone methods omissions (Astra; necessary).** Explain positive-preserving 1M MS MARCO diagnostic before its timing payoff, including separate encoding query sample and scope. Identify 40-candidate budget in the precision table. Parent confirmed the sampling limitation in MEASUREMENTS.md E2 and the budget/coverage match in the E18 named receipt.
4. **Interpret geometry predictively (both; necessary clarification).** The original-query distance ratio is a useful susceptibility marker; student-minus-original changes predict poorly. Separate proposed cause from measured association, including H2's because clause. No causal study promised.
5. **Trim secondary material (both; optional).** Move §4.4's regression/backbone tail to Appendix B. Consider relocating Table A. Remove internal specialization and synthetic-prefix fragments from the manuscript if they distract; preserve companion evidence. Keep trained recipes and adverse registered outcomes. Do not expand fusion/routing into another main branch.
6. **Sensitivity and cost continuity (Astra; worthwhile).** A concise clean-four qualification would expose the less stable table quality association while retaining robust dimensionality result. Parent confirmed clean-four table quality beta interval crosses zero slightly in E21. Label screening fitting versus release optimization clearly; do not compare different stages as a matched training invoice.

## Disagreement and editorial choices

Title is the largest live disagreement. Sol sees its unsupported defect promise as a credibility problem; Astra treats reopening as optional because the owner deliberately chose it. Parent agrees it overpromises but will not silently reverse the owner's title choice. A conditional hook remains an option for discussion.

Sol recommends moving Table A; Astra prioritizes keeping the scientific center and shortening §4.4 without insisting on that move. Neither says every appendix result is worthless: relevance to this paper's question, not mere historical interest, should govern retention.

## What to preserve

Two RQs and current sequence; prospective head validation and table uncertainty; unfavorable top-choice performance; mixed relevance versus consistent recovery effects; LightRetriever as bounded replication; complete adverse registered outcomes; standalone architecture/training exposition; precise measured-cost scope. No broad comparator frontier or causal mechanism invented.

Fully trained cross-space transfer, causal dimensionality experiments, calibrated required-effort prediction and production-cost evidence would be additional research. They are not blockers for a clearly scoped empirical whitepaper. Manuscript remains v19 pending owner direction on implementation.
