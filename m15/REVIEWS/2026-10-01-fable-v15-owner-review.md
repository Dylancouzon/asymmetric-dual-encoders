# Fable review of v15 for the owner (2026-10-01)

Independent of the Astra review in `2026-10-01-astra-v15-owner-adversarial.md`; both ran from the same owner brief. Read: CLAUDE.md, instructions-m15.md, REVIEW_GUIDE, OWNER_PREFERENCES, PAPER.md, EDITORIAL_RATIONALE, LEARNINGS, NOVELTY, FOLLOWUP, PAPER_EVIDENCE_MAP, EVIDENCE.md, the v15 reader/correctness/synthesis reviews, HANDOFF, LOG tail, m12/m13/m17/m18/m19/m20 findings heads, m9/EDGE_COST_MAC.md, RELATED_WORK head, both figures, and the two E2 receipts (`results/m15_e2_ann_*.json`). No protected path opened; no result changed; no experiment run.

## Verdict

The numbers hold. The paper does not yet, because it is three studies in one wrapper and the prose hedges every finding into the ground. A seasoned search engineer would trust it and not finish it.

## Findings

1. **The 62-fold to 2.2-fold headline is the wrong number.** In the 1M receipt, search time at a fixed `ef` is the same for all three encoders within 5% (ef=128: 1.53/1.53/1.60 ms; ef=512: 3.45/3.37/3.58 ms). The ratio collapses because search dominates any encoder under 2 ms. The encoder-specific finding is that Zero needs ef=512 to reach the same 1% relative loss that Nano reaches at 256 and Stella at 128, and that at ef=16 under binary quantization Zero loses 20% of its exact score versus 9% for Nano and 6% for Stella. Lead with the effort multiplier and the matched-ef loss, not the ratio.

2. **The paper has the mechanism for finding 1 and does not use it.** The E2 receipts record per-encoder geometry: median top-1 cosine 0.60 (Zero) / 0.74 (Nano) / 0.78 (Stella) on 1M, 0.49 / 0.68 / 0.73 on FiQA; median top-1 minus top-10 score gap 0.086 / 0.112 / 0.137 on 1M, 0.057 / 0.072 / 0.086 on FiQA. Lookup-table queries sit farther from every document and have flatter top-k margins, so greedy graph search has less signal to follow. This is exploratory and descriptive, but it is the explanation a reader wants, and it is already committed. Section 3.2 says "the mechanism remains unresolved" about precision while the ANN mechanism sits unused in the same receipt.

3. **Retention is reported as one macro; the per-dataset pattern is the generalizable result.** From C1, Zero retention spans 0.67 (TREC-COVID, FiQA) to 0.96 (climate-fever, Quora). The low end is topical and scientific sets (TREC-COVID 0.67, FiQA 0.67, SCIDOCS 0.70, NFCorpus 0.76); the high end is entity and duplicate-question sets (Quora 0.95, ArguAna 0.93, climate-fever 0.96, HotpotQA 0.88, FEVER 0.85). Nano is flat at 0.86 to 0.99 except FEVER (0.76, excluded from its pool). A one-panel figure of 15 datasets by two tiers tells an engineer where a static query table is usable. It costs nothing. The shuffle-sensitivity card (C12) supports the same reading.

4. **Sections 4, 5.2 and 6 are the stapled part.** See the answers below. Section 6 is a repaired-bug list. Section 5.2 is one priced component. Section 4's Table 3 is a development-only, four-component, Qdrant-operator curve.

5. **Hedging density.** Nearly every paragraph ends with what the result cannot show. The ratio of caveat to claim is roughly two to one. Honesty is the goal; this reads as a legal disclaimer and hides the findings. Move limits to one section and one sentence per finding.

6. **The reader's first objection is never answered with a number.** Nano (0.5081) is below bge-small on its own index (0.5171) at the same query cost. The only reason to prefer Nano is the cost of re-encoding. M20 measured Stella document throughput at 12.9k to 15.8k tokens/s on an RTX 3080, which is 39.9 h for the 23.7M-document BEIR-15 corpus. State the re-indexing cost in the introduction; it is the whole motivation.

7. **Appendix A is an acceptance-gate dump.** "Established below the 0.4583 bar" and "fails Holm .0083" are project language. Split into evaluation protocol, registered tests with one explanatory sentence per table, and teacher-screen details.

8. **References.** Twenty is fine. HNSW has no citation and DBSF has none; add both. BCT and FCT leave with section 6.

9. **Figure captions mix ms and relative loss; Table 2's hypothetical 0.68 / 3 ms example reads as contrived.** Cut the example; the per-encoder effort multiplier replaces it.

## Answers to the owner's questions

- **Section 4.** Keep 4.2 (oracle bound and failed routers/blend) as one paragraph: it answers the question every reader of a multi-tier design asks. Keep the fusion recovery number (0.4572 to 0.4933) as one paragraph. Cut Table 3 to one sentence or an appendix.
- **Section 5.2 and the "catered encoder over a monolithic index" idea.** The vision is right and it belongs in the introduction as the motivation, two or three sentences. It cannot be a section: three bounded specialization attempts (M17 no survivor, M18 no improvement and failed fused gate, M19 inconclusive) produced no improved encoder, and on M18's internal corpus Zero v1 dense scored below BM25 (0.087 vs 0.109 nDCG@10). Expanding 5.2 without a new result is marketing. The experiment that would earn it is listed below.
- **Section 6.** Cut. One sentence in the artifact statement: "CPU parity did not catch a CUDA fp16 export failure (minimum cosine 0.662); released paths are fp32 and qualified through the published loader."
- **Coherence.** One frame fixes the split: the document index is the long-lived asset, the query encoder is the part you change. Before the index exists, screen the student, not the teacher (5.1). After it exists, the cheap student's queries are out of distribution for the graph (3). Everything else is a paragraph under "what the shared index then allows".

## Experiments, ranked

1. Native magnitude-preserving query encoding (scalar8 query on the binary collection) at matched recovery and nDCG with measured search cost, graph rebuild controlled. Local, hours. Turns 3.2 from a diagnostic into a remedy in the engine the authors own.
2. Repeat the ANN effort sweep for two other teachers' tables on FiQA (vectors from the E8 screen if still cached). Tests whether "lookup queries need more ef" is a Stella property or a lookup-table property. Local or under $50.
3. Domain-fit table: fit the closed-form table on in-domain query vectors and compare against the generic table on that domain's test set. Minutes per fit. Licensing fork: SciFact train is CC BY-NC; NQ and HotpotQA are CC BY-SA and already in Zero's pool. Owner ruling needed on exploratory use of a non-commercial train split.

## Disagreement with Astra

Astra would move the teacher screen to a companion paper. I would not: it is the most shareable figure in the draft, and two thin papers serve the reputation goal worse than one coherent one. The lifecycle frame above keeps it.
