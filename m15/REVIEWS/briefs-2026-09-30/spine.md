# Research spine under consideration (M15 preprint, 2026-09-30)

Owner direction (Dylan, 2026-09-30): the paper is a RESEARCH paper, not a benchmark report or competitor overview. It leads with findings, implications and what the design makes possible. The project's comparator bars (nano vs bge-small / LEAF) were internal release gates, not the paper's goals; they appear in the evaluation section as the pre-registered test, and bge-small / LEAF / BM25 only as reference points. Failed approaches enter only if they carry significant value. Target reader: an advanced search engineer. Aim: the best, most-cited research paper this evidence can support, published as an arXiv preprint.

System: one frozen document tower, stella_en_400M_v5 (1024-d), documents encoded once. Query encoders that never co-trained with it emit into the same index:
- zero: per-token int8 lookup table distilled against the tower; tokenization + row lookup + weighted mean + L2 norm; no neural network at query time (construction prior art: pyNIFE 2025).
- nano: 34.5M transformer (pretrained bge-small backbone, layers 12/8/4 concat, linear head to 1024-d), distilled by squared L2 to Stella targets on 199,999,721 examples, ~57 h on one A100 (~$95).
- stella-query: the tower's own single-prompt (s2p) query path.
- BM25 and Qdrant DBSF@100 fusion.

Candidate findings (all numbers committed):
1. Retention vs query compute on BEIR-15 (MTEB English retrieval), exact search, macro nDCG@10: stella-query 0.5614 (1.000), nano 0.5081 (0.905) at ~7.25 ms warm p50 (CPU, batch 1, 4 threads, 20 words), zero 0.4572 (0.814) at 0.1119 ms, zero+BM25 0.4933 (0.879), nano+BM25 0.5110, BM25 0.4006. Zero beats BM25 on 11/15. Stella-query latency never measured under the same protocol. Per-dataset zero retention ranges 0.667 (trec-covid) to 0.958 (climate-fever); nano 0.759 (fever) to 0.988 (quora).
2. Distillability depends on the tower, and tower quality did not predict it: 7 towers with complete rows, table retention 43%..72% of each tower's own score on two dev forums; Spearman(tower ceiling, table quality) = 0.000 over 8 candidates; highest-ceiling teacher's table ranked 5th.
3. The lookup table's ceiling: post-pooling transforms (centering, whitening, PC removal, IDF/SIF weights) are exactly absorbable into a mean-pooled table (max abs diff ~1e-16), so they cannot raise it; word order and count saturation are outside what it can represent; the distillation objective was inert (median KL 4.73e-07 nats, positive already first for 99.75% of training queries); 12 table-side levers moved the dev endpoint < 0.005; BM25 fusion is the lever that worked.
4. A student keeps what its query distribution covers: first student 93.8% retention on a dev slice near training vs 50.1% on a programming forum; final nano weakest where its pool excluded FEVER (fever 0.623 vs zero 0.698, which trained on FEVER-train); out-of-domain dev slice predicted held-out retention within 0.009 while full dev macro overstated by 0.16.
5. System cost: encoder-only zero is ~65x faster than nano, but inside a system the ANN search dominates; the only same-harness system data is a synthetic-index prototype (not shipped models) so a real 1M-vector measurement is still needed.
6. The swap contract fails silently: missing prompt (cosine 0.80), tokenizer padding to 512 (cosine 0.35), float64 pooling, fusion operator/depth.
7. Pre-registered test: registered clean-4 nano − bge-small +0.017648 (established); one-shot held-out reserved four +0.0032 [−0.0069, +0.0134] unresolved; nano − LEAF −0.0389 there.

Use cases the owner finds interesting: high QPS (query encoding cost ~0.1 ms, no GPU, no ML runtime), search-as-you-type (encode every keystroke), plus edge/on-device, per-request tiering on one index, new query encoders for an existing index in hours, query encoding in environments without an ML runtime.
Nothing measured yet on throughput/QPS under load, prefix/partial queries, or multi-core scaling.

Cheap measurements possible on an Apple M5 Pro (no training, no protected data): encoder latency and throughput per core; real 1M-vector Qdrant collection (vectors from an archive) with quantization recall/latency, all Stella-space encoders swapped live; prefix-query retention (truncate public-dataset queries to k characters/words and measure nDCG vs full query, per encoder).
