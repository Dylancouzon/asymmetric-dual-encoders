# Codex gpt-6-astra, correctness review of PAPER.md draft v4

Date 2026-09-30. Brief: `briefs-2026-09-30/astra_paper_v4.md`. Access: named files only (plus its own skill file). All 16 findings applied. Verbatim below.

---

1. **“At every word cut the three dense tiers keep the same share within 0.02.”** False for the registered 75% cut. `results/m15_e4_prefix.json` gives macro retention **0.846440 / 0.828958 / 0.812476**, a **0.033964** spread. **Fix:** qualify this as “at the word cuts displayed here,” and report the exception.

2. **“On these three sets, what a prefix loses is information the user has not typed yet, and a larger query encoder does not recover it.”** E4 measures truncated test queries, not user typing or the cause of loss; similar retention does not establish equivalence. **Fix:** say the displayed cuts produced similar descriptive retention ratios, without the causal conclusion. Apply this qualification to the abstract and conclusion.

3. **“squared L2 to Stella's query vectors over 199,999,721 examples”** The dose is correct, but the targets are misdescribed. `results/m13_build_record.json` records **75% query batches and 25% document batches**. **Fix:** “squared L2 to Stella’s query and document vectors,” with the mixture stated.

4. **“Every fitted choice in this paper (ridge weights, router thresholds, blend weights) is made on two CQADupStack development forums, physics and programmers, and evaluated elsewhere.”** This overstates the scope. The build record uses **six development components**; `m7/RECIPE.md` describes broader development selection. **Fix:** restrict this statement to the named M15 ridge, routing and blending choices.

5. **“Four datasets (FEVER, DBpedia-entity and two CQADupStack forums) were held out for the registered test of Section 8 and play no other role.”** `results/m20_beir15_run.json` reuses their aggregate results in BEIR-15, including Section 4. **Fix:** say they have no subsequent fitting or diagnostic role, while their frozen results enter descriptive aggregates.

6. **“These works report one student per teacher; we hold the index fixed and span three tiers.”** `m15/RELATED_WORK.md` describes multiple EmbedDistill student sizes and NanoVDR-S/S-Multi. **Fix:** remove “one student per teacher”; describe the actual distinction without that assertion.

7. **“We fit this one recipe to 11 configurations of ten checkpoints from five families, all sharing one BERT WordPiece vocabulary, on 337,981 training queries screened against every held-out set.”** E8 names **six** families: Stella, Arctic, BGE, E5, GTE and Mixedbread. **Fix:** change “five” to “six,” or explicitly define a different grouping.

8. **“Second, the table's loss is concentrated: the two cheap tiers score the same on 43% of queries, and an oracle that sends 15% of queries to the transformer matches sending all of them.”** E5’s **43.3021%** is an equal-dataset average; the query-pooled tie rate is **56.1210%**. Its frontier applies the budget within each dataset. **Fix:** state both aggregation conventions; label Section 6.1’s percentages accordingly.

9. **“Vector fidelity, the quantity a distillation loss optimizes, is not ranking fidelity, which is why only a ranking screen predicts.”** E8/E11/E11b provide descriptive correlations at **n = 10**, including **−0.418** for error/margin versus quality. They establish neither exclusive predictiveness nor causation. **Fix:** report observed associations; qualify the accompanying e5 margin explanation as a hypothesis.

10. **“Closing the gap therefore needs a different query function, not a different post-processing of this one.”** `m15/PROOF_ABSORB.md` proves unchanged representational capacity, not optimality of the fitted table; it explicitly allows post-processing to help as a prior or initialization. **Fix:** say these operations add no representational capacity, without ruling out quality improvements.

11. **“A cheaper query vector makes approximate search harder.”** E2 observes larger ANN losses alongside lower nearest-document cosines (**0.599 / 0.739 / 0.780**); it does not establish cost or distance from a “document manifold” as the cause. **Fix:** describe Zero’s observed ANN penalty and mark the geometric explanation as a hypothesis.

12. **“Adding Qdrant's server-side BM25 and DBSF over two prefetches of 100 costs about 4.7 ms per query on the 1M collection, and on this subset it adds nothing to Zero (0.6183 against 0.6169 exact) and costs Nano 0.037.”** E2’s **4.684/4.627 ms** are total fused-search medians, not incremental costs. **Fix:** “The fused search takes about 4.7 ms”; describe Zero’s **+0.0014** as an observed difference.

13. **“A swap is legal when the replacement emits vectors of the same width, normalization and similarity”** Those properties do not establish alignment: an arbitrary rotation preserves them but changes scores against the fixed index. `zero_encoder.py` requires the exact teacher’s space. **Fix:** add alignment with that document space as a requirement.

14. **“Nano - bge-small | held-out NDO-3, dataset-weighted | +0.0032 [-0.0069, +0.0134] | unresolved”** The numbers match, but `results/m13_reserved_run.json` specifies **descriptive, zero alpha, no gate**. **Fix:** label this and all reserved rows “registered descriptive,” separating them from confirmatory outcomes.

15. **“Three SciFact queries show the pattern (exploratory, rank of the first relevant document):”** None of the permitted evidence files supplies these three rank measurements or the **2.29** fragmentation value. **Fix:** attach a committed diagnostic source containing the measurements, or remove the unsupported examples and explanation.

16. **“Each result file carries a receipt: script hash, git commit, model and dataset revisions, seed and machine.”** This is false for, for example, `results/m20_beir15_run.json`, which lacks that complete receipt. **Fix:** identify which files contain complete receipts and explicitly map other results to their provenance records.

Files opened:

- `m15/PAPER.md`, `m15/EVIDENCE.md`, `m15/PROOF_ABSORB.md`, `m15/RELATED_WORK.md`, `m15/e8_exposure.json`, `m15/MEASUREMENTS.md`, `m15/PLAN.md`, `m15/EVIDENCE_INDEX.md`, `m15/REVIEWS/2026-09-30-sonnet-literature.md`.
- Under `results/`: `m20_beir15_run.json`, `m13_reserved_run.json`, `m15_e1_latency.json`, `m15_e2_ann_msmarco1m.json`, `m15_e4_prefix.json`, `m15_e5_oracle.json`, `m15_e6_router.json`, `m15_e8_towers.json`, `m15_e8_frozen_lambdas.json`, `m15_e11_mechanism.json`, `m15_e11b_margin.json`, `m13_build_record.json`, `m7_absorb_check.json`.
- `m21/BENCHMARKS.md`, `m12/FINDINGS.md`, `m7/RECIPE.md`, `m11/STATUS.md`, `m11/release/zero_encoder.py`.
- Workflow instructions: `/Users/dylanc/.codex/plugins/cache/sendbird/cc/1.5.0/skills/review/SKILL.md`.
