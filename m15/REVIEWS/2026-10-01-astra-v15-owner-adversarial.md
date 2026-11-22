# Astra adversarial review of v15 (owner-directed, 2026-10-01)

Model: gpt-6-astra via codex-companion, thread 01a0f58f-423d-7571-a86f-4b76b87fe20d. Brief: owner questions on weight, relevance, generalizability, appendices, references, sections 4/5.2/6, coherence, and experiments. Output below is verbatim.

---

**V15 is careful enough to take seriously, but it is not yet focused enough to achieve the intended reputation-building goal.** Its main weakness is contribution selection, not inflated benchmark numbers. It contains a useful systems study, a separate teacher-selection study, and several engineering observations competing for attention.

Reviewed `m15-whitepaper` at `220bae41576b`. No files changed, experiments run, or protected evaluation content opened. Prior reviews informed the disposition check; they were not treated as scientific evidence.

1. **P1 — The paper still has two research questions, despite the reordered sections.**

   **Claim/section:** The abstract, four introductory findings, and §5.1.

   **Evidence:** [EDITORIAL_RATIONALE.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/EDITORIAL_RATIONALE.md), alternatives 1 and 4; [EVIDENCE.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/EVIDENCE.md), C3/C3b/C13. The teacher screen changes document spaces and predicts its own ridge-table recipe. It does not predict the separately trained Zero or Nano.

   **Effect:** A reader drawn in by replacing queries over an existing index encounters a substantial experiment about choosing an index before building it. Moving that experiment later fixes the interruption but not the intellectual split. I disagree with the earlier disposition that reordering sufficiently resolves this.

   **Fix:** Organize this paper around the consequences of replacing the query encoder while retaining document vectors. Move the full teacher screen into a companion study, retaining a short construction discussion. If publishing only one paper without additional experiments, a tightly scoped teacher-screen paper is also defensible: its reversal is the strongest standalone research result currently available. Do not give both equal billing.

2. **P1 — The 62×→2.2× headline conflates ordinary shared retrieval cost with additional search difficulty.**

   **Claim/section:** Abstract and §3.1: search effort consumes the encoding advantage.

   **Evidence:** [The million-passage receipt](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_e2_ann_msmarco1m.json), `encode_ms` and the Table 2 binary rows. Encoding medians are approximately **0.026 ms versus 1.61 ms**; search medians are **1.086 ms versus 0.869 ms** for Zero and Nano. C10 also preserves the uncompressed rows, where their selected search medians differ much more: approximately **3.583 versus 2.178 ms**.

   **Effect:** Adding substantial retrieval time to both encoders would dramatically shrink their ratio even without a Zero-specific penalty. The binary example establishes total-cost dilution; the ratio alone does not establish how much dilution comes from harder queries. That distinction matters because the additional ANN burden is the more interesting research contribution.

   **Fix:** Separate two findings explicitly: retrieval imposes a shared cost, and the substituted encoder can require additional search work. Show encoding and search components beside the quality curves. Lead the encoder-specific argument with the matched-setting loss measurements and effort needed to recover quality. Keep 62×→2.2× as an illustration, not an explanation. The existing unequal-quality caveat and hypothetical threshold example are correct; they address a different issue.

3. **P1 — The artifact package has a material reproducibility gap for both leading studies.**

   **Claim/section:** Artifact availability and the evidence supporting §§3 and 5.

   **Evidence:** [PROVENANCE_AUDIT.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PROVENANCE_AUDIT.md), “What is outside Git or still incomplete”; [MEASUREMENTS.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/MEASUREMENTS.md), E2 and E8. The cleaned 337,981-query fit list is absent from the Mac and untracked. The million-passage sampled IDs are local and untracked. Hashes identify these inputs but do not recover them.

   **Effect:** The results are reviewable, but the precise experimental populations are not reliably recoverable from the advertised package. “Not a standalone trainer for an arbitrary teacher” does not disclose this narrower problem: reproducing the paper’s own principal experiments.

   **Fix:** Before circulation, recover and archive the fit list and sampled IDs, or provide a verified reconstruction with pinned source identities. State the remaining unavailable inputs directly in the artifact statement. This is preservation work, not a reason to rerun experiments or reopen protected data.

4. **P2 — Section 4 does not yet establish which alternative is worth the query budget.**

   **Claim/section:** “Lexical fusion is a practical alternative” and the routing discussion.

   **Evidence:** EVIDENCE C1, C7–C10; [MEASUREMENTS.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/MEASUREMENTS.md), E2 “Fused cost”; [m12/FINDINGS.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m12/FINDINGS.md), depth curve and implementation caveats. Broad fusion quality uses bm25s/Lucene-style retrieval; native diagnostic timing uses Qdrant BM25 on a different collection. Router comparisons match Nano share, not measured total request cost.

   **Effect:** All these results are useful individually, but they do not establish a common quality/cost comparison among fusion, additional encoder compute, and routing. The paper correctly warns about some configuration differences, yet gives this collection of observations the prominence of a resolved budget decision.

   **Fix:** Keep the dense/fused quality changes and the failed blend. Move the operator-depth table to supporting material unless it becomes part of a matched serving comparison. State explicitly that the broad fusion quality and native timing describe different lexical implementations. Describe the routers as weak tested selectors with unmeasured end-to-end benefit, rather than a general negative result for adaptive compute. The earlier disclosure fix was necessary; it did not establish a common comparison.

5. **P2 — Section 5.2 cannot support a purpose-built specialization claim.**

   **Claim/section:** The proposed expansion from component training cost to inexpensive, use-case-specific encoders.

   **Evidence:** EVIDENCE C13 and [the decision audit](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_e15_decision_audit.json), `nano_build` and its exclusions; [m18/FINDINGS.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m18/FINDINGS.md), development results; [m19/FINDINGS.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m19/FINDINGS.md), judgment outcome and sensitivity.

   **Effect:** The repository demonstrates compatible compute tiers and a priced optimization component. It does not demonstrate successful, economical customization. M18 promoted no specialized encoder; M19 remains inconclusive. Cheap fitting and successful specialization are separate propositions.

   **Fix:** Keep the cost paragraph modest or move it into construction details. Treat specialization as a research opportunity, accompanied by its unsuccessful and inconclusive evidence. Do not turn approximately $95 of final optimization into an affordability story for bespoke retrieval.

6. **P2 — Section 6 gives routine qualification failures too much research weight.**

   **Claim/section:** Serving compatibility as a main finding.

   **Evidence:** [m11/STATUS.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m11/STATUS.md), prompt, padding, and rejected CUDA export; [m14/FINDINGS.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m14/FINDINGS.md), native pooling; [m22/UPSTREAM_FASTEMBED.md](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m22/UPSTREAM_FASTEMBED.md), dtype audit.

   **Effect:** These are legitimate operational counterexamples. They establish why the evaluated runtime must be identified, but they provide no systematic estimate of failure prevalence or distinctive consequence of query-side distillation. A catalog of repaired integration mistakes weakens the research ending. I disagree with the previous recommendation to retain this as a main section.

   **Fix:** Remove §6 as a standalone section. Preserve one short methods paragraph identifying the qualified serving path and one striking counterexample in an implementation appendix. Keep the detailed failure history in the repository.

**The remaining evidence does not require manufactured objections.** I found no essential issue in the current distinctions between exact relevance, ANN recovery, and precision-control coverage; the weaker native replication is disclosed; failed and unresolved original tests remain visible. Those are strengths.

The appendices are not excessive, but Appendix A gives more space to project acceptance gates than to explaining the headline evaluation. Add the complete per-dataset headline results and a compact description of the 15-dataset composition, including CQADupStack aggregation. Preserve the registered outcomes. Appendix B earns its space because its controls constrain the interpretation of §3.

There are **20 references, not 19**. That is not too many. QED, pyNIFE, LEAF, the ANN references, and QA-Cos perform necessary attribution work. BCT/FCT are weakly connected to wrapper defects and should leave with that main-text discussion. Model2Vec, StaticEmb, and NanoVDR can be compressed into related work. Citation count is not the problem; several references accompany material that does not need a main-text slot.

No omitted historical finding clearly deserves to replace the ANN study. LEARNINGS L04’s head initialization and L05’s distribution/width screens could strengthen a construction-focused paper, but their local scope makes them poor additions here. L21’s bounded patch is interesting precisely because operational control and relevance improvement separate; its unresolved quality prevents a successful-specialization headline.

**5. Section 4:** Restructure and shorten it. Keep fusion as a measured relevance alternative and routing/blending as a compact limit on the appeal of interchangeable encoders. Move the depth sweep out of the main narrative. It should answer what additional capability has actually been demonstrated, not introduce another research agenda.

**6. Section 5.2:** The owner’s idea is credible as a hypothesis; presenting it as an established capability would be marketing without sufficient data. The repository proves that query computation can change while documents remain fixed. It does not prove that arbitrary new needs can be served well in that fixed representation, or that query-side changes make an expensive index cheap to search. Evidence needed: successful adaptation on an independently judged use case, comparison with the unadapted encoder and existing teacher, preservation of unrelated queries, measured serving cost, and complete incremental build accounting.

**7. Section 6:** Cut the standalone section. The failures belong in reproducibility and implementation qualification. They become a research contribution only with a systematic question and evidence beyond examples of incorrect serving behavior.

**8. Thesis and order:** Use: **“Replacing the query encoder changes both relevance and the search effort required by an unchanged document index; encoding speed alone therefore misstates the available tradeoff.”** Order: fixed-index question and prior work; construction and evaluation setup; exact quality and encoding cost; ANN interaction and cost decomposition; query-precision control; compact fusion/adaptation limits; conclusion. Put detailed teacher selection in a companion study and serving qualification in an appendix.

**9. Ranked experiments:** These are proposed planning caps, not rental quotes; none requires full-model retraining.

1. **Native query precision at matched retrieval quality.** Claim: preserving magnitudes can reduce total search cost over the same document representation. Compare native precision choices against retuned binary `ef`/oversampling, retaining original-vector targets and controlling graph/segment changes. Required outcome: a reproducible latency improvement at matched recovery and nDCG, including scoring and rescoring costs. Plausible negative: better candidate coverage costs more than it saves. **Budget: $0–$100 compute, roughly 1–3 engineering days.**

2. **A contrasting teacher-space ANN replication.** Claim: additional search difficulty is not peculiar to Stella. Use one contrasting teacher and its existing fitted table, a complete nonreserved corpus, and repeated graph builds; prioritize uncompressed search so quantization does not obscure the question. Required outcome: increased effort at matched original-neighbor recovery that survives graph variation. Plausible negative: the penalty disappears or reverses, narrowing the conclusion to this family. **Budget: $50–$250 compute, conditional on recovering the necessary artifacts.**

3. **One bounded specialization test with adequate judgments.** Claim: a small query-side adaptation can improve a specific use case while preserving the index and a defined cost budget. Use a frozen-backbone head or row-only update, fresh development/evaluation separation, and contextual judgments; leave the sealed M19 surface untouched. Required outcome: a useful held-out gain over the unadapted route with unrelated-query preservation and measured total cost. Plausible negative: no gain, collateral losses, or insufficient information in the fixed document space. **Budget: $50–$150 compute plus approximately 8–16 expert judgment hours.** Pursue this only if specialization is intended to become a central claim.

Codex session ID: 01a0f58f-423d-7571-a86f-4b76b87fe20d
Resume in Codex: codex resume 01a0f58f-423d-7571-a86f-4b76b87fe20d
