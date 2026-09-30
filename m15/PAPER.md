# How Much Query Encoder Does a Frozen Document Index Need?

**Draft v3, 2026-09-30. Not for circulation.** Written from `m15/PLAN.md` v5.3. Markers `[E2]`, `[E9]`
and `[E10]` wait on measurements registered in `m15/MEASUREMENTS.md` and running overnight. Every
number traces to a committed file under `results/`; the source is named where the number first
appears. v2 (a comparator study) is in git history.

## Abstract

A dense retrieval system pays for its document index once and for its query encoder on every
request. We hold one document index fixed, 1024-dimensional vectors from `stella_en_400M_v5`, and
change only the query side, across a 700-fold range of query compute: the tower's own 400M query
path (31.6 ms per query on an Apple M5 Pro CPU), a 34.5M-parameter distilled transformer (2.25 ms),
and an int8 token lookup table with no neural network at query time (0.044 ms). On 15 BEIR datasets
under exact search the three keep 100%, 90.5% and 81.4% of the tower's own nDCG@10. The loss is not
spread evenly. On 43% of queries the table and the transformer score the same, and an oracle that
sends only 15% of queries to the transformer matches sending all of them. Short and unfinished
queries cost every tier the same share of its score, so that loss belongs to the query, not the
encoder. When choosing a tower to distill into a table, the tower's own retrieval quality predicted
nothing (Spearman -0.09 over 10 checkpoints on six public BEIR sets), while a cheap screen of the
table on two development forums predicted the public ranking (0.90). `[E9]` `[E10]` `[E2]` We report
a registered head-to-head test in full and release the index vectors and the harness.

## 1. Introduction

Retrieval systems are trained as one model and deployed as two. The document index costs a
corpus-scale encode, lives in a vector search engine, and is rebuilt reluctantly. The query encoder
runs once per request and stores nothing. If a cheaper query encoder can write into the same vector
space, the expensive half of the system never moves while the cheap half is swapped, retrained,
specialized or routed per request.

The setting is not new. LEAF ships asymmetric checkpoints that pair a small query model with a
larger document model [LEAF]. Query-encoder distillation onto a frozen document encoder appeared in
2023 [QED]. pyNIFE distills a per-token lookup table against a frozen off-the-shelf tower [pyNIFE].
Backward- and forward-compatible training named the goal of upgrading a model without re-embedding
a corpus [BCT, FCT]. What the literature lacks is a measurement of the whole range on one index:
how much quality survives as query compute falls from the tower's own path, to a small transformer,
to a lookup table, and what decides where it is lost.

We answer with one index and three query encoders that all regress onto the tower's query output,
so their vectors land in the space the index already holds. Because they share that space, a system
can also mix them per query or blend their vectors in one search. The results, in order:

1. **Retention as query compute drops** (Section 3). 100%, 90.5% and 81.4% of the tower's BEIR-15
   macro nDCG@10 at 31.6, 2.25 and 0.044 ms per query.
2. **Choosing a tower for a table** (Section 4). Across 11 configurations the tower's own quality did
   not predict the quality of its table; a development screen of the table did. In all four model
   families we measured, the smaller checkpoint made the better table.
3. **Where the loss sits and what routing can recover** (Section 5). Per-query headroom is large and
   concentrated; a free lexical feature captures about a tenth of it. `[E10]` `[E9]`
4. **Short queries** (Section 6). Prefixes cost the three tiers the same share of their score.
5. **Does the saving survive the search** (Section 7). `[E2]`
6. **The pre-registered test** (Section 8), reported in full.

## 2. Setup

**The frozen side.** One document tower, `NovaSearch/stella_en_400M_v5` at revision `ffeb2b7e`,
1024 dimensions, fp32 compute, L2-normalized vectors stored fp16, cosine similarity. Every dense
result searches the same document vectors.

**The query side.**

| Tier | What it is | Built by | Build cost |
|---|---|---|---|
| Stella query | The tower's own query path with its `s2p_query` prompt | nothing | none |
| Nano | 34,540,672-parameter transformer: bge-small backbone, layers 12, 8 and 4 concatenated, linear head to 1024, mean pooled | squared L2 to the tower's query vectors, 199,999,721 examples | 57.3 A100 hours, about $95 |
| Zero | 30,522 x 1024 int8 token table; a query is a normalized, count-saturated mean of its token rows | cosine and ranking loss against the tower, then int8 | about 20 minutes of training after 8-12 hours of encoding on one RTX 3080 |

Build details are in `results/m13_build_record.json` and `m7/RECIPE.md`. Neither build changed the
document tower. BM25 (bm25s, Lucene k1 = 1.2, b = 0.75) appears as a reference and as the lexical
half of two fused systems (DBSF over two top-100 prefetches, Qdrant's operator).

**Data and protocol.** Quality comes from exact search on BEIR-15 (`results/m20_beir15_run.json`).
Four datasets (FEVER, DBpedia-entity, CQADupStack android and english) were reserved for a
registered one-shot test and are now spent; they enter no measurement in this paper beyond their
committed aggregate result. Every new measurement follows a method file frozen before it ran
(`m15/MEASUREMENTS.md`), fits on two CQADupStack development forums (physics and programmers), and
evaluates elsewhere. Our single-prompt Stella path scores 0.5614 on BEIR-15; the model card reports
58.97 with per-task instructions, a 400-token limit and bf16. Our bge-small (0.5171), LEAF (0.5402)
and BM25 (0.4006) reproduce their published numbers.

## 3. Retention as Query Compute Drops

**Table 1.** BEIR-15 macro nDCG@10, exact search. Retention is a ratio of macros against the
Stella query path. Encode latency is the median over real test queries of 5-12 words, batch one, four
CPU threads, ONNX Runtime (Zero on NumPy), Apple M5 Pro (`results/m15_e1_latency.json`).

| Query side | nDCG@10 | Retention | Encode p50 |
|---|---:|---:|---:|
| Stella query path | 0.5614 | 1.000 | 31.6 ms |
| Nano + BM25, DBSF@100 | 0.5110 | 0.910 | |
| Nano | 0.5081 | 0.905 | 2.25 ms |
| Zero + BM25, DBSF@100 | 0.4933 | 0.879 | |
| Zero | 0.4572 | 0.814 | 0.044 ms |
| BM25 | 0.4006 | 0.713 | |
| *Reference, own index:* LEAF (query side) | 0.5402 | | 1.39 ms |
| *Reference, own index:* bge-small | 0.5171 | | 2.57 ms |

Figure F1 plots retention against encode latency on a log axis. Nano keeps nine tenths of the
tower's quality at one fourteenth of its query compute; Zero keeps four fifths at one seven-hundredth.
Zero beats BM25 on 11 of the 15 datasets while costing about as much to encode as tokenizing the query.

Retention varies by dataset more than by tier (Figure F2). Zero keeps 0.667 of the tower on
TREC-COVID and 0.958 on Climate-FEVER; Nano ranges from 0.759 (FEVER) to 0.988 (Quora). Zero scores
above Nano on FEVER, HotpotQA and Climate-FEVER, three claim-style or multi-hop sets. We do not read
that as a property of the table: Zero trained on FEVER-train, Nano's training pool excluded FEVER, and
Nano starts from bge-small.

Published retention numbers are not comparable to ours without care. Query-encoder distillation
reports 92.5% as an average of per-dataset relative scores [QED]; LEAF reports 97.7% with its own
tower and benchmark [LEAF]; ours is a ratio of macros on BEIR-15 against a single-prompt tower path.

## 4. Choosing a Tower to Distill Into a Table

A lookup table can be fit in closed form against any tower that shares its vocabulary: a ridge
regression from token-count bags to the tower's query vectors, anchored at each token's own tower
embedding (`m8src/teacher_screen.py`). We refit that recipe for 11 configurations of 10 checkpoints,
all sharing one BERT WordPiece vocabulary (one ordered-vocabulary hash across all 11), on 337,981
training queries screened against every protected set. We chose each ridge weight on the two
development forums, froze all of them in a committed file (`results/m15_e8_frozen_lambdas.json`),
and only then scored six public BEIR sets once (`results/m15_e8_towers.json`).

**Table 2.** Closed-form tables. "Tower" is each checkpoint's own query path on its own documents.
Six-set macro nDCG@10; retention is table over tower.

| Checkpoint | Dim | Dev table | Six-set tower | Six-set table | Retention |
|---|---:|---:|---:|---:|---:|
| stella_en_400M_v5 | 1024 | 0.3437 | 0.5745 | 0.3974 | 0.692 |
| bge-base-en-v1.5 | 768 | 0.3073 | 0.5259 | 0.3529 | 0.671 |
| arctic-embed-m-v1.5 | 768 | 0.3019 | 0.5263 | 0.3279 | 0.623 |
| gte-base-en-v1.5 | 768 | 0.2748 | 0.5331 | 0.3252 | 0.610 |
| arctic-embed-l | 1024 | 0.2586 | 0.5289 | 0.3034 | 0.574 |
| e5-base-v2 | 768 | 0.2341 | 0.4669 | 0.2930 | 0.627 |
| bge-large-en-v1.5 | 1024 | 0.2759 | 0.5329 | 0.2845 | 0.534 |
| mxbai-embed-large-v1 | 1024 | 0.2505 | 0.5368 | 0.2605 | 0.485 |
| e5-large-v2 | 1024 | 0.2192 | 0.4735 | 0.2585 | 0.546 |
| gte-large-en-v1.5 | 1024 | 0.2039 | 0.5970 | 0.2455 | 0.411 |
| *arctic-embed-l, mean readout (control)* | 1024 | 0.2217 | 0.5073 | 0.2764 | 0.545 |

Three results.

**The tower's quality did not transfer.** Over the 10 checkpoints, Spearman correlation between a
tower's own six-set score and its table's six-set score is -0.09 (0.14 on the four sets with no
disclosed exposure). On the development forums the same correlation is 0.45, so a tower that looks
strong where the table was screened does not stay strong where it is used. The strongest tower on these sets, gte-large (0.5970), made the weakest table
(0.2455, retention 0.411). The pattern matches the finding that a stronger teacher does not always
make a better student [Cho and Hariharan 2019], here for embedding towers and a table student.

**A cheap screen of the table transferred.** The development-forum score of each table predicted its
six-set rank with Spearman 0.90 (0.68 on the four exposure-free sets). The screen costs one encode of
the fit list and two forums per tower; the tower's leaderboard row costs nothing and told us nothing.

**Smaller checkpoints made better tables in every family.** bge-base over bge-large, e5-base over
e5-large, gte-base over gte-large, and arctic-m-v1.5 over arctic-l, on the development forums and on
the six sets alike. The arctic pair also differs in version, so it is the weakest of the four.
Reading the same arctic-l checkpoint with mean pooling instead of its CLS token lowered the table
(0.2764 against 0.3034), so pooling alone does not explain which towers distill well.

**What this does and does not show.** One closed-form recipe, 10 checkpoints, six public sets; no
significance claim. The towers differ in data, size, dimension and readout, so none of this isolates
a cause. Training exposure to these sets is disclosed by no checkpoint's authors; most checkpoints
trained on the source families of the science and finance sets, and one states it removed MTEB test
overlap (`m15/e8_exposure.json`). An earlier screen of this kind in this project used a fit list
with 1.31% overlap with protected queries and found a correlation of 0.000 over eight configurations;
this refit uses the cleaned list.

## 5. What a Table Cannot Do, and What Routing Can

### 5.1 The Table's Limit

A mean-pooled lookup table absorbs any affine map applied after pooling and any fixed per-token
weight: the transformed system is another table, so such levers cannot add capacity. We checked this
numerically against explicitly rebuilt tables (maximum absolute difference 9.31e-14,
`results/m7_absorb_check.json`); it fails under sum pooling, and nonlinear transforms are outside it.
Training evidence points the same way. The shipped objective was nearly exhausted on its own data
(median KL 1.08e-07 nats; the positive ranked first for 99.75% of 4,000 training queries against the
training bank), and no table-side lever improved the development score by more than about 0.005.
A table also cannot see word order. Shuffling a query's words costs the Stella path 6.9% and Nano
4.4% of their own nDCG@10 on SciFact, NFCorpus and FiQA, and Zero nothing, by construction
(`results/m15_e4_prefix.json`). On FiQA the Stella path loses 13.7%.

### 5.2 Where the Loss Sits

Averages hide how the loss is distributed. Across the 12 BEIR-15 datasets outside the reserved
test, Zero and Nano tie on 43% of queries, Zero wins on 18% and Nano on 39%
(`results/m15_e5_oracle.json`). A per-query oracle that picks the better of the two reaches 0.5473
macro nDCG@10, close to the Stella path (0.5580 on the same 12), against 0.5161 for always-Nano. If the
oracle may send only a fraction of queries to Nano, taking the largest gains first, 15% already
matches always-Nano (0.5163) and 25% reaches 0.5334 (Figure F5).

That is the case for tiering: most queries do not need the transformer, and the ones that do are
few. The question is whether a system can find them without running the transformer first.

### 5.3 Routing on Signals the Table Already Has

**Fertility.** The table fell behind the tower by 0.050 nDCG@10 per extra subword per word in earlier
work, so we routed queries with high fertility to Nano. With thresholds fixed on the two development
forums for Nano budgets of 10%, 25% and 50%, the router beat random routing at the same Nano share by
0.003 to 0.005 on the 12 evaluation datasets, against an oracle margin of 0.03 to 0.05: about a tenth
of the headroom (`results/m15_e6_router.json`). Fertility locates the table's weakness against the
tower, not its weakness against Nano.

`[E10]` Zero's own confidence: the norm of its pooled vector before normalization and the margin
between its first and tenth retrieved document, fitted and frozen the same way, plus an escalation
target of the E9 blend instead of plain Nano.

### 5.4 Blending Two Query Vectors in One Search

`[E9]` Because Zero and Nano write into the same space, their vectors can be averaged and searched
once. Weight fixed on the development forums; six public sets; against Nano alone and against
Nano + BM25 fusion.

## 6. Short Queries and Prefixes

Search-as-you-type and short queries are where a near-free query encoder would matter most, so we
cut each test query of SciFact, NFCorpus and FiQA and asked how much of its own full-query score each
tier keeps (`results/m15_e4_prefix.json`, macro over the three sets):

| Cut | Stella query | Nano | Zero | BM25 |
|---|---:|---:|---:|---:|
| First word | 0.293 | 0.288 | 0.297 | 0.362 |
| First two words | 0.389 | 0.381 | 0.387 | 0.448 |
| First three words | 0.479 | 0.468 | 0.461 | 0.528 |
| First half of the words | 0.610 | 0.597 | 0.604 | 0.656 |
| First half of the characters | 0.549 | 0.533 | 0.547 | 0.488 |
| Words shuffled | 0.931 | 0.956 | 1.000 | 1.000 |

The three dense tiers keep the same share within 0.02 at every word cut. What a prefix loses is
information the query has not yet supplied, and a bigger query encoder cannot supply it. A table is
as good a search-as-you-type encoder, relative to its own ceiling, as the tower is. BM25 keeps a
larger share of its lower score on word prefixes and a smaller share when the cut lands inside a word,
where a lexical match fails and a subword table still sees the fragment. At half the words, prefixes lose
to random word subsets of the same length on FiQA and beat them on SciFact and NFCorpus, so where the
informative words sit depends on the dataset.

## 7. System Cost

### 7.1 The Encoder in Isolation

| Encoder | p50, 1-4 words | p50, 5-12 | p50, 13-64 | Peak RSS | Model files |
|---|---:|---:|---:|---:|---:|
| Zero (NumPy) | 0.033 ms | 0.044 ms | 0.064 ms | 360 MiB | 90.1 MiB |
| LEAF query (ONNX) | 1.19 ms | 1.39 ms | 1.69 ms | 243 MiB | 88.2 MiB |
| Nano (ONNX) | 2.00 ms | 2.25 ms | 2.84 ms | 317 MiB | 132.3 MiB |
| bge-small (ONNX) | 2.28 ms | 2.57 ms | 3.22 ms | 296 MiB | 127.8 MiB |
| Stella query (ONNX) | 22.6 ms | 31.6 ms | 49.0 ms | 2,812 MiB | 1,669.6 MiB |

Three fresh processes per encoder, 100 real test queries per length bucket, every ONNX graph matched
to its torch path at minimum cosine 0.9999 before timing. Zero's model file holds both its int8 and
fp16 tables. Computing the fertility feature of Section 5.3 costs 0.018 ms.

### 7.2 Does the Saving Survive the Search?

`[E2]` One Qdrant collection per setting over FiQA (57,638 documents) and a 1M-passage MS MARCO
subset that keeps every judged positive; HNSW `ef` sweep and scalar, binary and TurboQuant
quantization; for each encoder the cheapest setting within 1%, 2% and 5% of its own exact nDCG@10,
and the end-to-end latency it needs there.

### 7.3 Fusion Depends on Candidate Depth

DBSF normalizes each prefetch by its own score distribution, so the depth of the prefetch is part of
the operator. Zero + BM25 on the four exposure-free sets scores 0.4625, 0.4856, 0.4912 and 0.4974 at
prefetch depths 10, 50, 100 and 1000 (`m12/FINDINGS.md`). The registered depth of 100 was chosen for
latency before the evaluation and costs 0.006 against depth 1000.

## 8. The Pre-Registered Test

Before any evaluation data was read, the project registered head-to-head contrasts for Nano against
bge-small and LEAF on six BEIR sets, with a clean-4 partition (NFCorpus, SCIDOCS, SciFact,
TREC-COVID) that excludes the sets Stella's authors disclose, and a one-shot test on four reserved
sets. We report both in full.

| Contrast | Partition | Difference in nDCG@10 | Registered outcome |
|---|---|---:|---|
| Nano - bge-small | clean-4 | +0.017648 (lower bound +0.003674) | established |
| Nano - bge-small | all six | +0.027449 | established |
| Nano - LEAF | all six | +0.016181 | established |
| Nano - LEAF | clean-4 | -0.001063 | unresolved |
| Nano - bge-small | reserved NDO-3, dataset-weighted | +0.0032 [-0.0069, +0.0134] | unresolved |
| Nano - bge-small | reserved NDO-3, query-pooled | -0.0121 [-0.0219, -0.0024] | descriptive |
| Nano - LEAF | reserved NDO-3 | -0.0389 [-0.0488, -0.0290] | descriptive |

NDO-3 is DBpedia-entity weighted 0.5 and the two reserved CQADupStack forums 0.25 each; FEVER is
reported separately. "Unresolved" is not equivalence: no equivalence interval was computed anywhere.
The comparators search their own indexes (bge-small at 384 dimensions, LEAF against arctic-embed-m at
768) while Nano searches Stella's, so these contrasts compare systems, not query encoders alone.

## 9. Limitations

One document tower and one family of query students; the Section 4 result covers one closed-form
recipe and 10 checkpoints. BEIR-15 and E8 results are descriptive. Query-resampling intervals exclude
training-seed variation. Latencies come from one edge-class CPU and one runtime; server throughput and
GPU serving are not measured. The E2 subset is a diagnostic, not full MS MARCO. Several BEIR sets
overlap the training families of the towers and of Nano's bge-small backbone, which trained on MS
MARCO. Stella's exposure to ArguAna, FiQA and FEVER rests on community metadata.

## Appendix A. What Does Not Travel With a Swap

A swap is legal when the replacement emits vectors of the same width, normalization and similarity;
the served paths match their references to 4.5e-08 (Zero), 1.2e-07 (Nano) and minimum cosine
1.00000000 (Stella through the published document graph). Three things fail silently: Stella without
its prompt scores at cosine 0.80 against the correct vector; Stella's tokenizer pads to 512 by default,
which drops cosine to 0.35; and one serving library pooled Nano in float64 until fixed upstream.

## Appendix B. Reproducibility

Every table regenerates from committed JSON under `results/m15_*`, each with a receipt of script
hash, git commit, model revisions, dataset revisions, seed and machine. The method file and its
reviews are in `m15/MEASUREMENTS.md` and `m15/REVIEWS/`.

## References

- [BCT] Shen, Y., Xiong, Y., Xia, W., and Soatto, S. Towards Backward-Compatible Representation
  Learning. CVPR 2020.
- [FCT] Ramanujan, V., Vasu, P. K. A., Farhadi, A., Tuzel, O., and Pouransari, H. Forward Compatible
  Training for Large-Scale Embedding Retrieval Systems. arXiv:2112.02805.
- [QED] Query Encoder Distillation via Embedding Alignment is a Strong Baseline Method to Boost Dense
  Retriever Online Efficiency. arXiv:2306.11550.
- [LEAF] LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations.
  arXiv:2509.12539.
- [pyNIFE] Tulkens, S. pyNIFE. https://github.com/stephantul/pynife
- [Model2Vec] https://github.com/MinishLab/model2vec
- [LightRetriever] arXiv:2505.12260.
- [Cho and Hariharan 2019] On the Efficacy of Knowledge Distillation. ICCV 2019. arXiv:1910.01348.

Verify every entry against the published record before submission.
