# M15 paper proposal v3 (for owner review, 2026-09-30)

## Thesis

Index a corpus once with a large frozen tower (stella_en_400M_v5, 1024-d). The query encoder then becomes a per-request choice over the same index. We measure the whole menu on that one index, on the MTEB English retrieval set (15 BEIR datasets), under exact search, with a pre-registered confirmatory family and a one-shot held-out test:

- lookup table (zero), no neural network at query time: BEIR-15 macro 0.4572, retention 0.814 of the tower's single-prompt query path (0.5614); beats BM25 on 11 of 15.
- zero + BM25 (Qdrant DBSF@100): 0.4933, retention 0.879.
- 34.5M distilled student (nano), trained with no MS MARCO in any role: 0.5081, retention 0.905.
- nano + BM25 DBSF@100: 0.5110.
- tower's own query path: 0.5614.
- external: LEAF-asym 0.5402 (own 768-d arctic-m index), bge-small 0.5171 (own 384-d index), BM25 0.4006.

## Title candidates

1. How Much Query Encoder Does a Frozen Document Index Need? A pre-registered quality-cost frontier from a lookup table to the tower itself. (recommended)
2. Swap the Query, Keep the Index: a registered retention-cost frontier for query encoders over a frozen document tower. (Fable)

## Headline and the held-out result

- Registered confirmatory headline (CLAUDE.md and m20 registry fix it): nano − bge-small clean-4 +0.017648 (one-sided 2.5% lower bound +0.003674), all-six +0.027449; nano − LEAF all-six +0.016181, clean-4 −0.001063 unresolved.
- One-shot held-out reserved four (registered descriptive, alpha 0): R1 nano − bge-small NDO-3 +0.0032 [−0.0069, +0.0134], query-pooled −0.0121 [−0.0219, −0.0024]; R2 nano − LEAF −0.0389 [−0.0488, −0.0290].
- The abstract states both in adjacent sentences. The held-out result is the paper's credibility argument, not a footnote.
- The m20 registry forbids superiority, equivalence, significance or release claims from BEIR-15 or reserved rows; all BEIR-15 and reserved language is descriptive.

## External reproduction checks (new, from published cards and the MTEB results repo)

- bge-small 0.5171 vs card "Retrieval (15)" 51.68.
- LEAF-asym 0.5402 vs card 54.03.
- BM25 0.4006 vs MTEB baseline-bm25s 39.84.
- stella-query 0.5614 vs card 58.97. The card used per-task e5-mistral instructions, max_len 400, bf16 (author, HF discussion 21). Our protocol uses the single `s2p_query` prompt, the deployable setting and the one our tiers distilled against. Gap concentrates on climate-fever −0.145, fever −0.089, nq −0.054, dbpedia −0.039, trec-covid −0.029; msmarco, whose e5 instruction equals s2p, is +0.000.

## Contributions

1. The frontier on one index: five query-side configurations plus three external systems, BEIR-15 quality, retention against the tower's own query path, and system cost on a real 1M-vector Qdrant collection (E1, E2).
2. A pre-registered protocol with a one-shot held-out replication, reported whether or not it agreed; plus a contact-partitioned reading of BEIR-15 that labels every dataset by teacher disclosure and by train-split contact for every system, ours included.
3. Where the cheap tiers fail and what fixes it: on claim-verification and multi-hop sets (fever, climate-fever, hotpotqa) the lookup table beats the student; fusion gain over dense tracks the BM25 − dense gap (Pearson 0.73 zero, 0.74 nano, 15 datasets), giving a per-corpus rule for when to fuse.
4. Release: models (already public), the full registration trail (public repo), every paper table regenerated from committed JSON, and the Stella BEIR-15 document vectors with a scoring script so anyone can score a new query encoder against the same frozen index without the ~40 h encode.

Short method section (body 1 page, detail in appendix): out-of-domain dev slice predicted held-out retention within 0.009 while the full dev macro overstated it by 0.16; post-pooling transformations are exactly absorbable into a mean-pooled table (proof in appendix).

## Contact partition (descriptive, post-hoc, rule stated before computing)

The registry labels comparator training and teacher disclosure but not our own tiers' train-split contact. Our tiers saw: zero, FEVER-train, HotpotQA, NQ-open, SQuAD, TriviaQA, Mr. TyDi, ESCI (m7/RECIPE.md); nano, HotpotQA-train 81,743, NQ-open, 415,454 claim-form rows (m13 build record). The partition must be symmetric before it goes in the paper. Current observed macros, comparator-trained excluded (10 sets): stella 0.4940, LEAF 0.4652, nano 0.4518, bge 0.4395, zero 0.3842. Neither comparator-trained nor teacher-disclosed (7): nano+BM25 0.4692, LEAF 0.4678, nano 0.4528, bge 0.4385.

## Structure (12 pages + appendix)

1. Introduction (1.5 p). Index is the sticky asset, query side is a menu. Prior art in paragraph two: LEAF, pyNIFE, LightRetriever, 2306.11550, BCT/FCT, Drift-Adapter. Contributions.
2. Setting and the swap contract (1.5 p). Tower, tiers, comparators, index bytes per 1M docs. Box: four silent failures (prompt, padding to 512, float64 pooling, fusion operator and depth).
3. Protocol (1.5 p). Registration timeline per family (table), clean-4, reserved-four one-shot, contact labels, exact search, statistics, what "unresolved" means.
4. Quality (3.5 p). 4.1 registered contrasts. 4.2 held-out reserved four. 4.3 BEIR-15 by contact class (F2). 4.4 where the table beats the student. 4.5 the fusion rule (F3).
5. System cost (2 p). Encoder-only by query length, in-system on the real 1M index, quantization recall and latency including TurboQuant, memory limits, the live swap in one server lifetime (F1, F4).
6. Results that transfer (1 p). OOD dev slice; absorbability.
7. Limitations (0.75 p). One tower, one seed, intervals exclude seed variation, single-prompt reference, 4.4 undiagnosed unless E3 lands.
8. Release and reproducibility (0.5 p).
Appendix: full BEIR-15 table with per-dataset contact labels, per-forum cqadupstack, statistics details, registration ledger, teacher sweep (n=8, Spearman 0.000), objective exhaustion, M7 bar contrasts (LightRetriever, OpenSearch), closed avenues (one table, six rows maximum).

## Figures

- F1 frontier: x = measured system p50 latency (log), y = BEIR-15 macro, marker size = index GB per 1M docs. Needs E1 + E2.
- F2 contact partition: nano − bge-small and nano − LEAF per dataset, grouped by contact class.
- F3 fusion rule: fusion gain vs BM25 − dense gap, 30 points.
- F4 system latency vs query length, per encoder, on the real index.

## Cut or moved from v2

- M9 system-latency table (sections 6.2 and 6.4 in v2): its "nano" was an untrained MiniLM-L6 fp16 ONNX and its zero table and index were synthetic (bench/edge_prototype_pair.py:4-11, 41-42). It cannot be presented as nano/zero system cost. Replaced by E2.
- Artifact-size contradiction (90.1 MiB vs 270.1 MB): resolved by E4 under one definition, or cut.
- Docker 1.11x bound, DBSF vs convex0 discussion (footnote), depth-10 inversion (dev only): cut or footnote.
- Teacher-sweep Spearman, objective exhaustion, fertility non-lever, M7 bar contrasts: appendix.
- M8/M17/M18/M19: one appendix table row each at most.

## Measurement package (owner go needed; register method first, one Astra review, no protected access, no training)

- E1 encoder latency, one harness on the Apple M5 Pro: zero, nano, bge-small, mdbr-leaf-ir, stella-query; common protocol (3 fresh processes, batch 1, 4 threads, 5 warmups, 20 samples); query-length buckets 6-10, 21-50, 51-120 words. About 2 h.
- E2 real 1M index: 1M document vectors of one public corpus from the M20 archive for each of the three document towers (Stella ~2 GB, bge-small ~0.8 GB, arctic-m ~1.5 GB fp16), copied from the RTX box D: drive to the Mac. Qdrant server 1.18 in Docker: original vectors, scalar int8, binary (rescore on/off), TurboQuant bits4; recall@10 against exact search, p50/p99, RSS, disk; memory limits 256 MB / 512 MB / 2 GB; every query encoder swapped live against one Stella collection in one server lifetime with returned ids recorded. About 1 day including transfer.
- E3 label-free diagnostic (optional, about 2 h): per-dataset mean cosine between each tier's query vector and the tower's, on public non-reserved query sets (no qrels, no documents), correlated with per-dataset retention. Method frozen before computing.
- E4 packaging: one definition of shipped query-side bytes for each encoder.
- E5 prompt probe (optional, RTX box): stella-query with the e5 claim instruction on climate-fever, to test whether the claim-set failure is single-prompt inheritance.

## Release package

- R1 paper/ with LaTeX source, figures, and one command that regenerates every table from committed result JSON.
- R2 Hugging Face dataset: Stella BEIR-15 document vectors (fp16, doc-id order, SHA-256 manifest) and a short script that scores a query-vector file with exact nDCG@10 and pairs it against our per-query rows. Licence gating: quora and climate-fever are licence-unverified, msmarco is non-commercial; exclude or gate those. Owner and possibly Qdrant legal decide.
- R3 bring-your-own-tower kit: deferred. Estimate ~4.5k LOC lifted from m7src/m10src/m7src stats plus 0.6-1k new (data loader, BEIR runner, de-hardcoded serving script, tokenizer handling). Cite the code in the paper with no support promise.

## Also found

- Zero model card says "L2 regression"; the recipe is cosine + KL ranking + InfoNCE (m7/RECIPE.md:46-91 vs m11/release/MODEL_CARD.md:184). M22 card fix.
- Build cost for a new tower: zero ~20 min training on an RTX 3080 plus 8-12 h re-encoding; nano 57.27 h on one A100 (about $95 at $1.66/h), 199,999,721 examples.
