# Fable consultant on the v3 (comparator-framed) shape

Date 2026-09-30. Access: m20/STATUS.md, m20/FINDINGS.md, README.md, CLAUDE.md, branch PAPER.md and NOVELTY.md only. Returned text below, verbatim. Dispositions are in `m15/LOG.md` (2026-09-30) and `m15/PLAN.md`.

---

**Short version:** you have a frontier paper, not a "we beat bge-small" paper. Lead with the frontier, put the held-out non-replication in the abstract, release the vectors, and run the real-index measurement before submitting. Details by question.

## 1. Thesis and title

**Top pick:** *"Swap the Query, Keep the Index: A Registered Retention-Cost Frontier for Query Encoders over a Frozen Document Tower."*

Thesis: with one frozen 1024-d index, the query side is a menu. A lookup table retains 0.81 of the tower (0.88 fused with BM25) at 0.1 ms; a 34.5M student trained on commercially licensed data only retains 0.905 and lands in the bge-small band on every dataset neither model trained on; and the 5x encoder gap collapses to about 2x once search is included. All of it measured under pre-registration with a one-shot held-out test that is reported whether or not it agreed.

Why it gets cited: nobody has the reference numbers for "how much do I lose if I put a static table or a small student on a third-party index." pyNIFE has NanoBEIR on one tier; LEAF has one student per teacher. Engineers choosing a query tier will cite the frontier figure. Methodologists will cite the contact-partitioned reading of BEIR and the pre-registered held-out design, because both are rare in IR and both are reusable.

**Runner-up:** *"Contact-Partitioned BEIR: Comparing Query Encoders When the Baselines Trained on the Benchmark."* The 10-dataset no-contact partition (nano 0.4518 > bge-small 0.4395, LEAF 0.4652) versus the 5 comparator-trained sets (bge-small 0.6724 vs nano 0.6208) is the cleanest picture in your data. It is a smaller paper but a very citable one, since every small-model comparison on BEIR has this problem and nobody has a standard way to report it. I would fold it into the top pick as Section 4.3 and Figure 2, not run it alone.

## 2. Structure and figures (12 pages plus appendix)

1. Introduction, 1.5 p. Prior art (LEAF, pyNIFE, BCT) named in paragraph two. Four contributions: frontier, contact-partitioned comparison with held-out replication, system-level cost, transferable method results.
2. Setting and swap contract, 1.5 p. Tower, three tiers, comparators, index bytes per 1M docs. The four silent failures (prompt, padding, dtype, operator) as a half-page box. Engineers love this and it is short.
3. Protocol, 1.5 p. Registration timeline per family, clean-4, reserved-four one-shot, the contact partition of BEIR-15, exact search, statistics. One timeline figure or table of "what was registered when" belongs here.
4. Retrieval quality, 3.5 p. 4.1 registered contrasts; 4.2 held-out reserved four; 4.3 BEIR-15 by contact class; 4.4 where the table beats the student (fever, climate-fever, hotpotqa); 4.5 the fusion rule.
5. System cost, 2 p. Encoder-only, in-system by query length, quantization and memory, all on the real-vector index (see Q5).
6. Results that transfer, 1 p. OOD dev slice as predictor; absorbability lemma in three sentences with a pointer to the appendix proof.
7. Limitations, 0.75 p. One tower, one training seed, intervals exclude seed variation, no diagnosis for 4.4.
8. Release, 0.5 p.
Appendix: full BEIR-15 table, per-query statistics, registration ledger, teacher sweep, objective exhaustion, closed avenues (one table, six rows).

Figures that carry the paper:

- **F1, the frontier.** x = system p50 latency at 20-word queries (log scale), y = BEIR-15 macro nDCG@10, one point per system, marker size = index GB per 1M docs. stella-query, LEAF, bge-small, nano, nano+BM25, zero+BM25, zero, BM25. This is the picture people will screenshot.
- **F2, contact partition.** Fifteen datasets as rows grouped by class (comparator-trained, teacher-disclosed, clean), bars for nano minus bge-small and nano minus LEAF. It shows the whole contamination argument without a paragraph.
- **F3, fusion rule.** Fusion gain versus (BM25 minus dense) per dataset, both tiers, 30 points, r = 0.73. Annotate trec-covid (+0.169 for zero) and NQ (-0.084 for nano). This is the one actionable rule in the paper.

## 3. New, known, cut

New to an advanced search engineer:
- The contact-partitioned reading and the fact that a no-MS-MARCO 34.5M student sits in the bge-small band on out-of-contact data. Anyone who cannot use non-commercial training data will cite this.
- The system-level collapse with the query-length dependence (1.96x at short, 5.1x at long). Folklore, but nobody has the number.
- The fusion-gain rule (r = 0.73). Practical and quotable.
- The table beating the transformer on claim-form and multi-hop queries. Surprising. Report as an observation with the coverage hypothesis, no claim, unless Q5's diagnostic lands.
- The OOD dev slice predicting held-out retention within 0.009 while the full macro overstated by 0.16. Anyone distilling a query encoder wants this.
- The 142.6 GiB vector archive (see Q6).

Nice but known, keep short:
- The swap mechanism itself. One paragraph, cite LEAF and pyNIFE.
- DBSF monotone in prefetch depth, binary quantization as the memory precondition, rescoring cost. One table row each.
- The absorbability lemma. Algebraically obvious once stated, but it closes off a class of post-hoc tricks that Model2Vec-style pipelines still apply. Three sentences in the body, proof in the appendix.

Cut or appendix:
- Teacher ceiling Spearman 0.000 over n = 8. Too small to support the sentence you want. Appendix.
- Objective exhaustion (4.73e-07 nats) and the fertility non-lever. Appendix.
- The DBSF versus convex0 debate. One footnote.
- The artifact-size disagreement (90 MiB vs 270 MB). Resolve it under one packaging definition before submission, or cut it. Reporting confusion is not a result.
- LightRetriever bar and OpenSearch bar contrasts. Different towers, different indexes, hard to read. Appendix.
- Docker noise ratios (1.11x). Cut.

## 4. The reserved four

Asset, provided you put it in the abstract. The registered clean-4 win did not replicate on the held-out three (+0.0032, CI straddles zero, query-pooled negative), and LEAF is ahead there by 0.039. Say so in sentence four of the abstract. Then say what does survive: across 10 datasets no comparator trained on, nano and bge-small are within 0.012, and nano is behind LEAF by 0.013. The honest claim is "in the bge-small band, behind LEAF, on a frozen third-party index, trained without MS MARCO," and that claim is stronger than "+0.017648" because it survived a one-shot test.

Do not headline +0.017648 anywhere. A reviewer who finds the reserved-four result in Section 4.2 after reading that number in the abstract will stop trusting the paper. Report the registered NDO-3 weighting and the query-pooled weighting side by side, with the registration date. Pre-registered held-out replication is almost unheard of in IR benchmark papers; that is the paper's credibility argument, and a non-confirming result is what makes it believable.

## 5. Measurements

Necessary:
- **Real-vector 1M collection with recall.** Section 5 cannot stand on random 1024-d vectors; HNSW on random vectors has no cluster structure, and a reviewer will say the 0.85 ms search figure is meaningless. Take 1M msmarco stella vectors from the archive, measure fp16, int8, binary recall@10 against exact search plus p50 latency, all encoders in one server lifetime. TurboQuant is optional.
- **stella-query and LEAF latency under the common protocol.** F1 has no reference point without them. The 400M tower's CPU batch-1 number is what makes the two tiers interesting.

Optional but high value:
- **Query-form diagnostic.** Bound it to one week. If it converts the fever/hotpotqa reversal into a mechanism (declarative claims versus question-form training queries), Section 4.4 becomes a finding people cite. If not, it stays an observation.

## 6. Release

Yes, and it is probably worth more citations than any single result. Artifacts get cited more than findings, and encoding BEIR-15 on a 400M tower cost you 40 hours per tower. Anyone who wants to test a new query encoder against a frozen index today has to redo that.

Minimal credible release: one Hugging Face dataset repo with the stella document vectors for all BEIR-15 corpora as fp16 shards, doc-id order files, the manifest with SHA-256 hashes, and one script that takes a query-vector file and emits exact-search nDCG@10. That is the "bring your own query encoder" harness. Add the bge-small and LEAF document towers in a second push. "Bring your own document tower" is the encode script plus the registry, with no promise of support. Check redistribution terms per corpus before upload; msmarco's terms differ from the rest, and derived vectors need a stated position.

## 7. The attack

"This is LEAF with a worse result. LEAF-asym beats nano on the held-out set (by 0.039) and on BEIR-15 macro (0.5402 vs 0.5081), with a 23M student and a 25% smaller index. The pre-registration is ceremony around a negative result."

Pre-empt it in three moves. First, concede in the abstract that LEAF is ahead and put it on F1 as a point. Second, state exactly what LEAF's student trained on; if its data includes MS MARCO, NQ, or HotpotQA, the contact partition applies to it too, and the no-contact gap is 0.013, not 0.032. Third, make the licensing constraint explicit as a design axis, since a student with no non-commercial data in any role, sitting within 0.013 of one that has it, is a result for every team that cannot ship MS MARCO derivatives. The paper's claim is the frontier over one index with all tiers priced, and LEAF is one point on it with its own index cost. A paper that says "we lost to LEAF and here is why the comparison still matters" reads as trustworthy. One that buries it reads as a product launch.

Two smaller attacks to close in Limitations: one training seed with intervals that exclude seed variation, and one tower. Say both plainly and stop.
