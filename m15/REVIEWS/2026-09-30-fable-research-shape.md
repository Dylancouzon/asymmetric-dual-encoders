# Fable consultant on the research-paper shape (v4)

Date 2026-09-30. Access: FINDINGS files and plain-English page only. Returned text below, verbatim. Dispositions are in `m15/LOG.md` (2026-09-30) and `m15/PLAN.md`.

---

**Read:** spine.md, plain-English page, m7/m8/m9/m10 findings, m1-m6 findings, m20/STATUS.md. Nothing under results/ or work/.

## 1. Research question

**Pick:** *How much retrieval quality does one frozen document index keep as query-side compute falls from the tower's own encoder to a 34.5M transformer to a table lookup, and what decides where the quality is lost?*

This is the question the evidence answers. Finding 1 gives the frontier, findings 2 to 4 give the mechanism (tower distillability, structural ceiling of the table, coverage), and finding 6 gives the operational cost. A search engineer reads it as "what do I give up at each tier, and can I predict it before training".

**Alternative:** *Can one document index serve query encoders at several cost tiers without re-embedding, and what does the swap contract cost?* This is the product framing. It is weaker today because the tiering claim has no measurement, and the paper would lean on the system prototype, which the spine says is synthetic. Keep it as the implications section, not the lead.

## 2. Title and abstract

**Title:** Retention as Query Compute Drops: Cheap Query Encoders on a Frozen Document Index

**Alternative:** One Index, Three Query Budgets: What a Frozen Document Tower Keeps When the Query Side Shrinks

**Abstract (150 words):**

A dense index is built once with an expensive document encoder, but every query pays for its own encoder. We ask how much retrieval quality one frozen document index keeps as the query encoder shrinks from the tower's own 400M-parameter path to a 34.5M transformer to an int8 token lookup table, and what decides where quality is lost. On BEIR-15 with exact search, the transformer retains 90.5% of the tower's nDCG@10 at 7.3 ms per query on CPU; the table retains 81.4% at 0.11 ms and beats BM25 on 11 of 15 datasets. Three results explain the losses. Distillability is a property of the tower, and tower quality does not predict it (Spearman 0.00 over eight towers). The table's ceiling is structural: every post-pooling transform is absorbable into its rows. A student keeps what its training queries cover; an out-of-domain development slice predicted held-out retention within 0.009. A pre-registered test placed the transformer above bge-small on the registered partition (+0.018) and unresolved on a one-shot held-out set.

## 3. Section structure (12 pages plus appendix)

1. **Introduction (1 page).** The asymmetry: documents encoded once, queries encoded forever. State the question and the three encoders as three cost points, not as products.
2. **Setup (1 page).** Frozen tower, the space contract (prompt, pooling, dtype, normalization), exact search, BEIR-15, the registered partitions. Comparators named once as reference points.
3. **Finding: the retention frontier (2 pages). Spine.** Finding 1. Macro and per-dataset retention for each tier, BM25 and DBSF fusion beside them. Say plainly that the teacher's latency was never measured under the same protocol, or measure it (see section 4).
4. **Finding: distillability belongs to the tower (1.5 pages). Spine.** Finding 2 plus the base-beats-large screening rule from m7. Practical implication: select a teacher on the artifact you will ship, with a closed-form ridge probe that costs minutes.
5. **Finding: the lookup table has a structural ceiling (1.5 pages). Spine.** Finding 3. The absorbability proof, the inert objective (median KL 4.7e-07, positive first in 99.75% of training queries), the 12 levers under 0.005, and fusion as the one lever that moved. The fertility correlation that was not a lever (m8) belongs here as the cleanest example.
6. **Finding: a student keeps what its queries cover (1.5 pages). Spine.** Finding 4. The 93.8% versus 50.1% split, the FEVER reversal where the table beats the transformer, and the out-of-domain slice predicting held-out within 0.009 while the macro overstated by 0.16. This is the section a practitioner acts on: build the query pool before touching the architecture.
7. **Evaluation: the pre-registered test (1 page). Supporting.** Finding 7, stated as a test with its outcome: established on clean-4 against bge-small, unresolved on the reserved four, behind LEAF there. No equivalence claim. Dev-reuse counts published as numbers.
8. **Implications: what one index with swappable query encoders makes possible (1.5 pages). Supporting.** Use cases from section 4 below, each with its measurement or one sentence.
9. **The swap contract fails silently (0.5 page). Supporting.** Finding 6, as a checklist with the cosine numbers.
10. **Limitations and what would change the verdict (0.5 page).** Query-resampling intervals only, exact search, English only, three points on the frontier.

**Appendix:** fusion's dataset dependence (DBSF helps zero +0.089 on HotpotQA, costs nano 0.036 on MS MARCO), the full per-dataset table, the M9 rank-bottleneck probe, the training data licence audit, the nuisance-parameter band (m7 finding 8), registration and receipt hashes.

**Out:** M17, M18, M19 specialized-table attempts (no result), LightRetriever reproduction gap, the sparse competitor sweep, edge container prototype as a headline (see below).

## 4. Use cases

| Use case | Treatment | Cheapest measurement |
|---|---|---|
| New query encoders for an existing index | Section (it is section 8's core) | Already measured: the table is a closed-form ridge solve; nano was 57 h, about $95, on one A100. Add the ridge wall-clock for one table. |
| High QPS | Paragraph | Queries per second per core for zero, nano, bge-small at batch 1 and batch 64 on the M5 Pro, plus one 1M-vector Qdrant run with the archive vectors so the paper can state what fraction of end-to-end time the encoder is. This also fills the stella-query latency hole. |
| Search-as-you-type | Section, if the curve is flat; paragraph if it is not | Prefix retention: truncate public queries (SciFact, NFCorpus, NQ, Quora; never the reserved four) to k words, report nDCG@10 and overlap@10 against the full-query ranking, per encoder. The m7 length probe suggests the table may degrade gracefully, which would be a real result. Cost per keystroke: 0.11 ms versus 7.3 ms is the framing. |
| Per-request tiering | Paragraph | Oracle tiering from existing per-query rows: fraction of BEIR-15 queries where zero's nDCG@10 is at least nano's, and the macro of per-query max. No router needed; it bounds the headroom. |
| Edge, on-device | Paragraph | The 256 MB container result exists. Re-run on the real 1M-vector archive with binary quantization and record recall against exact search, since the quality tables are exact. |
| Environments without an ML runtime | One sentence in section 8 | None. |

## 5. Figures

1. **Retention frontier.** Log-x query latency, y macro retention, one point per encoder with a vertical strip showing per-dataset spread. BM25 and the fused points beside them. This is the figure people cite.
2. **Tower quality does not predict distillability.** Scatter of eight towers, x the tower's own score, y its table's score, Spearman 0.00 annotated, the top-ceiling tower ranked fifth.
3. **Where each tier loses.** Per-dataset retention heatmap for zero and nano, datasets ordered by BM25 strength, FEVER and HotpotQA highlighted where the table beats the transformer.
4. **Prefix retention curve** if measured. Otherwise the swap-contract figure: cosine to the correct vector under each misconfiguration (0.80 missing prompt, 0.35 padding to 512).

## 6. "I did not know that", ranked

1. Tower quality does not predict how well it distills into a table (Spearman 0.00, best tower ranked fifth). Every team picks a teacher from a leaderboard.
2. Post-pooling transforms are exactly absorbable into a mean-pooled table, so centering, whitening, PC removal, and IDF weights cannot raise its ceiling.
3. The distillation objective was inert at 99.75% positive-first, and 12 table-side levers moved under 0.005. The table's quality is set by the tower and the pool, not the recipe.
4. An int8 lookup beats BM25 on 11 of 15 datasets at 0.11 ms.
5. Coverage beats capacity: the table beats the 34.5M transformer on FEVER because its pool included FEVER-train, and the out-of-domain slice predicted held-out within 0.009.
6. DBSF's value tracks BM25's strength on the dataset, so a fixed fusion policy helps one tier and hurts another.
7. Padding to 512 drops cosine to 0.35 with no error raised.

Two pushbacks. The frontier has three points, so call it a frontier, not a curve, unless the M5 Pro run adds bge-small and int8 or fp16 variants of nano. And the teacher's latency gap is the one hole a reviewer will find first; the same M5 Pro session that measures QPS closes it.
