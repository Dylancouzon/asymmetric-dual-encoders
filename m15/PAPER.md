# A Strong Tower Is Not a Distillable Tower: Cheap Query Encoders for a Frozen Document Index

**Draft v4, 2026-09-30. Not for circulation.** Slots marked `[E2]`, `[E9]` and `[E10]` wait on
measurements running tonight (`m15/MEASUREMENTS.md`). Each claim links to the committed file that
proves it; `m15/EVIDENCE.md` restates every number in plain language with its source and caveats.
Results are labelled **registered** (method frozen before observation) or **exploratory**.

## Abstract

A vector search engine pays for its document index once and for its query encoder on every
request. If a cheaper query encoder can write into the same vector space, the expensive half of the
system never moves while the cheap half is swapped, specialized or routed per request. We study this
over one frozen index of `stella_en_400M_v5` vectors with three query encoders that land in its
space: the tower's own 400M query path (31.6 ms per query on a laptop CPU), a 34.5M-parameter
distilled transformer (2.25 ms), and an int8 token lookup table with no neural network at query time
(0.044 ms). They keep 100%, 90.5% and 81.4% of the tower's nDCG@10 on 15 BEIR datasets.

Three findings go beyond that frontier. First, when we fit the same closed-form lookup table to ten
different embedding towers, the tower's own retrieval quality told us nothing about the table it
would yield (Spearman -0.09 on six public BEIR sets), and neither did three plausible mechanisms we
measured, while a few-minute screen of the table itself on two development forums predicted the public
ranking (0.90). In each of the four families where we tried two sizes, the smaller checkpoint made the better table.
Second, the table's loss is concentrated: averaged over datasets, the two cheap tiers score the same on 43% of queries, and an oracle that sends 15% of each dataset's queries to the transformer matches sending all of them. `[E10]` `[E9]`
Third, on truncated queries the three tiers keep similar shares of their own score, so a cheap tier loses no more to short or unfinished queries than the tower does. `[E2]`

## 1. Introduction

Suppose you run semantic search over a large corpus and your document vectors come from a strong,
slow embedding model. Re-embedding the corpus is the expensive, risky operation; encoding a query is
cheap in comparison but happens on every request, on every device, at every keystroke if you offer
search-as-you-type. You would like a much cheaper query encoder that writes into the index you
already have. Two practical questions follow, and neither is answered by the usual efficiency
frontier:

1. **If you might want cheap queries later, which embedding model should the index be built on?**
   The natural guess is "the best one on the leaderboard", because the cheap query encoder is
   distilled from it.
2. **Which queries actually need the expensive encoder?** If most do not, a system can serve the
   cheap tier by default and escalate, all against one index.

We measure both on one frozen index. Our main result is that the natural guess in question 1 is
wrong in an informative way. Across ten towers, the tower's own quality, how well a bag of tokens
reproduces its query vectors, how much it reads word order, and how far its top result stands above
its tenth all fail to predict the quality of the lookup table distilled from it. What does predict it
is cheap: fit the table and score it on a small development set. We call this "screen the student".

For question 2 we find large, concentrated headroom and measure how much of it a system can
recover with signals the cheap tier already produces (Sections 6 and 7). Because every tier writes
into one space, a system can also do something that is impossible across independently trained
retrievers: blend two query vectors and search once (Section 7).

**Contributions.**

- A measurement of how much quality a frozen index keeps as the query encoder shrinks by three
  orders of magnitude in latency, from the tower's own path to a lookup table, on 15 datasets, with
  the deployment cost under approximate search and quantization in a vector search engine
  (Section 4). `[E2]`
- Evidence that a tower's quality does not predict its distillability into a lookup table, that
  three natural mechanisms do not explain it either, and that a cheap direct screen does
  (Section 5).
- A decomposition of where the table loses to the transformer, with worked examples, and the
  observation that prefix and short-query losses are the query's, not the encoder's (Section 6).
- Two ways to exploit a shared query space: routing on the cheap tier's own signals and blending
  query vectors in one search (Section 7). `[E9]` `[E10]`
- A pre-registered head-to-head test of the transformer tier, reported in full (Section 8), and a
  public harness in which every number regenerates from committed files.

## 2. Related Work

**Cheap query encoders over a frozen document side.** Query-encoder distillation onto a frozen
document encoder keeps 92.5% of the teacher on BEIR, averaged per dataset [QED]. EmbedDistill
aligns embeddings with a frozen teacher document encoder and reports 95-97% on MS MARCO and NQ
[EmbedDistill]. LEAF releases 23M-parameter query models that pair with a larger document model and
keep 97.7% [LEAF]. NanoVDR distills a visual-document retriever's query side and keeps 95.1%
[NanoVDR]. These works study a student, or a few student sizes, against its teacher; we hold one index fixed and span three tiers, down to a table with no network at query time.

**Lookup-table query encoders.** Static embeddings compute a query as an average of token vectors
[Model2Vec, StaticEmb]. pyNIFE fits a per-token table against a frozen off-the-shelf tower and
reuses its index [pyNIFE]. LightRetriever serves queries by embedding lookup and keeps about 95% on
BEIR, but trains its document LLM jointly with the query table [LightRetriever]. Our table is fit to
a frozen tower that was never trained for it, which is the setting where the index already exists.

**Teacher quality and distillation.** A stronger teacher does not always make a better student
[Cho and Hariharan 2019], including in dense retrieval distillation [PROD]. We are not aware of a
study that fits the same query-side student to many frozen embedding towers and asks what predicts
its quality.

**Routing and query performance prediction.** Choosing a retriever per query between sparse and
dense retrieval [Arabzadeh et al.] and predicting retrieval quality before or after retrieval [QPP]
are established. Our routing is between two query encoders over one index, using signals the cheap
encoder already computes. We found no prior work that blends the query vectors of two encoders that
share a document index.

**Compatible representations.** Backward- and forward-compatible training upgrade an embedding
model without re-encoding a gallery [BCT, FCT]. Our setting is the query-side special case with the
document tower frozen by construction.

## 3. Setup

**The frozen side.** `NovaSearch/stella_en_400M_v5` at revision `ffeb2b7e`: 1024-dimensional
document vectors, fp32 compute, normalized, stored fp16, cosine similarity. All results in Sections
4, 6 and 7 search these document vectors. (Section 5 fits tables to other towers over their own
documents; its references bge-small and LEAF search their own indexes.)

**Three query tiers.**

| Tier | What runs at query time | How it was built | Build cost |
|---|---|---|---|
| Stella query path | the tower itself, with its `s2p_query` prompt | nothing to build | none |
| Nano | a 34,540,672-parameter transformer: bge-small layers 12, 8 and 4 concatenated, a linear head to 1024, mean pooling | squared L2 to Stella's vectors over 199,999,721 examples (75% query, 25% document batches) | 57.3 A100 hours, about $95 |
| Zero | a 30,522 x 1024 int8 table; the query vector is the normalized, count-saturated mean of its tokens' rows | cosine and ranking losses against Stella, then int8 | about 20 minutes of training after 8-12 hours of encoding on one RTX 3080 |

Sources: `results/m13_build_record.json`, `m7/RECIPE.md`. BM25 (bm25s, Lucene, k1 = 1.2, b = 0.75)
appears as a reference and as the lexical half of fused systems (DBSF over two top-100 prefetches,
the fusion operator of Qdrant's Query API).

**Evaluation.** Quality is nDCG@10 under exact search on 15 BEIR datasets
(`results/m20_beir15_run.json`), with per-query rows committed in `results/m20_beir15_scores/`.
Approximate-search results appear only in Section 4.2 and are labelled as deployment behavior. Four
datasets (FEVER, DBpedia-entity and two CQADupStack forums) were held out for the registered test of Section 8; their frozen results enter the BEIR-15 aggregates, and they play no fitting or diagnostic role anywhere in this paper. Every choice fitted for this paper (ridge weights, router thresholds, blend weights) is made on two CQADupStack development forums, physics and programmers, and evaluated elsewhere; the tiers themselves were built earlier on the project's development suite.

**How faithful is our Stella baseline?** Our single-prompt Stella path scores 0.5614 on BEIR-15; the
model card reports 58.97 with task-specific instructions, a 400-token limit and bf16. Our bge-small
(0.5171 against the card's 51.68), LEAF (0.5402 against 54.03) and BM25 (0.4006 against 39.84 for
bm25s in MTEB) match their public numbers within half a point (`m15/REVIEWS/2026-09-30-sonnet-literature.md`).

## 4. The Cost-Quality Frontier

### 4.1 What Survives as the Query Encoder Shrinks

**Table 1.** BEIR-15 macro nDCG@10 under exact search, and query-encode latency (median over 100
real test queries of 5-12 words, batch one, four CPU threads, ONNX Runtime, Zero on NumPy, Apple M5
Pro; `results/m15_e1_latency.json`). Retention is a ratio of macros. Registered.

| Query side over the Stella index | nDCG@10 | Retention | Encode p50 |
|---|---:|---:|---:|
| Stella query path | 0.5614 | 1.000 | 31.6 ms |
| Nano | 0.5081 | 0.905 | 2.25 ms |
| Nano + BM25 (DBSF@100) | 0.5110 | 0.910 | 2.25 ms + BM25 |
| Zero | 0.4572 | 0.814 | 0.044 ms |
| Zero + BM25 (DBSF@100) | 0.4933 | 0.879 | 0.044 ms + BM25 |
| BM25 alone | 0.4006 | 0.713 | |

![Figure 1](figures/f1_frontier.png)

**Figure 1.** Retention against encode latency. Hollow markers are reference systems on their own
indexes (bge-small 0.5171 at 2.57 ms, LEAF 0.5402 at 1.39 ms).

Nano keeps nine tenths of the tower at one fourteenth of its latency; Zero keeps four fifths at one
seven-hundredth, and beats BM25 on 11 of 15 datasets. Fusing Zero with BM25 recovers about a third
of the gap between Zero and the tower. The spread across datasets is wider than the spread across tiers
(Figure 2): Zero keeps 0.667 of the tower on TREC-COVID and 0.958 on Climate-FEVER.

![Figure 2](figures/f2_per_dataset.png)

**Figure 2.** Per-dataset retention. Zero is above Nano on FEVER, HotpotQA and Climate-FEVER; Zero
trained on FEVER-train while Nano's pool excluded it, so we do not read this as a property of tables.

Published retention numbers use other definitions (per-dataset averages [QED], other towers
[LEAF]), so we do not rank them against ours.

### 4.2 Does the Saving Survive the Search?

A cheap query encoder is only cheap if the search that follows does not have to work harder for it.
We served each tier from Qdrant (v1.19.1, native macOS build) over two collections: FiQA (57,638
documents) and a one-million-passage MS MARCO subset that keeps every judged positive and fills the
rest at random (a diagnostic, not full MS MARCO). For each tier we swept HNSW `ef` and three
quantizations (int8 scalar, 1-bit binary, 4-bit TurboQuant, each with rescoring and 1x-4x
oversampling) and asked for the cheapest end-to-end setting within 1%, 2% and 5% of the tier's own
exact nDCG@10. Exploratory in the sense that it answers a deployment question, registered in method.

**Zero's query vectors lose more to approximate search.** On the one-million-passage collection,
at the same unquantized HNSW setting (`ef` 16), Zero loses 16.3% of its own exact nDCG@10 to
approximate search, Nano 7.0% and the Stella path 4.4%; their top-10 lists agree with exact search
on 82%, 91% and 93% of neighbors (`results/m15_e2_ann_msmarco1m.json`). The geometry is consistent with a simple hypothesis: the median cosine between a query and its nearest document is 0.60 for Zero, 0.74 for Nano and 0.78 for the Stella path. A table's query vector is an average of token rows and lands farther from the documents the graph was built on, which we would expect to require a wider graph search; we did not test that mechanism directly.

**The saving survives, but shrinks.** Within 1% of each tier's exact score, Zero needs `ef` 512 with
binary quantization and 2x oversampling (search p50 1.09 ms), Nano `ef` 256 with 4x oversampling
(0.87 ms), and the Stella path `ef` 64 with TurboQuant (0.66 ms). End to end, encode plus search,
that is 1.11 ms for Zero, 2.48 ms for Nano and 21.3 ms for the Stella path (MS MARCO queries are
short: encode p50 0.026, 1.61 and 20.7 ms). Nano's cost is 2.2 times Zero's at the 1% target and 3.1
times at 5%, down from 62 times for the encoders alone. Figure 4 shows the whole frontier.

![Figure 4](figures/f4_system.png)

**Figure 4.** End-to-end p50 against the share of each tier's exact nDCG@10 lost to approximate
search; each point is the cheapest setting at that loss. Left: 1M MS MARCO subset; right: FiQA.

Fusion is not free inside the engine either. A fused query, Qdrant's server-side BM25 and DBSF over two prefetches of 100, takes about 4.7 ms per query on the 1M collection; on this subset it scores 0.6183 for Zero against 0.6169 for Zero's exact dense search, and 0.037 below Nano's. `[E2 FiQA rerun on an idle
machine replaces the FiQA panel.]`

## 5. Which Towers Admit a Cheap Query Encoder?

A lookup table can be fit in closed form to any tower that shares its tokenizer: a ridge regression
from each query's token counts to the tower's query vector, anchored at each token's own embedding
(`m8src/teacher_screen.py`). We fit this one recipe to 11 configurations of ten checkpoints from six model families, all sharing one BERT WordPiece vocabulary, on 337,981 training queries screened against
every held-out set. The ridge weight of each was chosen on the two development forums and frozen in a
committed file (`results/m15_e8_frozen_lambdas.json`) before any public set was scored
(`results/m15_e8_towers.json`). Registered.

![Figure 3](figures/f3_towers.png)

**Figure 3.** Left: a tower's own nDCG@10 against its table's, on six public BEIR sets. Right: the
table's score on two development forums against its public score. Gray lines join base and large
checkpoints of one family.

**The tower's quality did not transfer.** Across the ten checkpoints, Spearman correlation between a
tower's own score and its table's is -0.09 on the six public sets. The strongest tower there,
gte-large-en-v1.5 (0.5970), made the weakest table (0.2455, keeping 41% of its tower). Stella,
the second-strongest tower, made the best table and kept 69%.

**A cheap screen of the table did.** The table's score on the development forums ranks the public
scores with Spearman 0.90. The screen costs one encode of the training queries and two small
corpora per tower, a few minutes on one A100; the leaderboard costs nothing and, here, told us
nothing.

**In every family with two sizes, the smaller checkpoint made the better table.** bge-base over bge-large (0.3529 against
0.2845 on the public sets), e5-base over e5-large (0.2930 against 0.2585), gte-base over gte-large
(0.3252 against 0.2455) and arctic-m over arctic-l (0.3279 against 0.3034), on the development forums
as well. The arctic pair also differs in version.

**Three mechanisms that do not explain it.** Exploratory, `results/m15_e11_mechanism.json` and
`results/m15_e11b_margin.json`. We measured, per tower, on the real test queries:

| Candidate explanation | Measurement | Spearman with table quality |
|---|---|---:|
| The tower is "almost a bag of words" | mean cosine between the table's and the tower's query vector | -0.07 |
| The tower reads word order, which a table cannot | mean cosine between a query's vector and its shuffled version's | -0.27 |
| The table's error is small relative to the tower's ranking margins | vector error divided by the median gap between the tower's 1st and 10th score | -0.42 |

The first two point the wrong way; the third points the right way but weakly, and against retention
it is -0.13. The e5 checkpoints show why fidelity misleads: their tables reproduce the tower's query
vectors best (cosine 0.90 and 0.89, against 0.78 for Stella) and still rank documents worse, because
e5's own score margins are the narrowest (a median gap of 0.026 between its 1st and 10th document,
against 0.084 for Stella), so a small vector error reorders its results. Margins alone do not rescue the prediction either: gte-base has the widest margins (0.109) and a middling table. With ten checkpoints these are associations, not causes, and the e5 reading is a hypothesis; what the data do show is that vector fidelity, the quantity a distillation loss optimizes, did not track ranking quality here, while a direct ranking screen did. Vector fidelity, the quantity a
distillation loss optimizes, is not ranking fidelity, which is why only a ranking screen predicts.

**What this does and does not show.** One recipe, ten checkpoints, six public sets, no significance
claim. The towers differ in data, size, dimension and readout. No checkpoint's authors disclose
training on these six test sets; most trained on the source families of the science and finance
sets, and one states it removed MTEB test overlap (`m15/e8_exposure.json`). An earlier screen in
this project used a training list with 1.31% overlap with held-out queries and found a correlation of
0.000 over eight configurations; this refit uses the cleaned list.

## 6. Where Context Matters

### 6.1 The Loss Is Concentrated

Averages hide how the table loses. On the 12 non-held-out BEIR-15 datasets, as the mean of the per-dataset shares (pooling all queries instead gives 56% ties, because Quora is large and tied on 72% of its queries; `results/m20_beir15_scores/`, exploratory decomposition):

| Per-query outcome, Zero against Nano | Share of queries |
|---|---:|
| both score 0 | 17% |
| both score the same, above 0 | 26% |
| Zero higher | 18% |
| Nano higher | 39% |

A per-query oracle that picks the better tier reaches 0.5473 macro nDCG@10 against 0.5161 for
always-Nano and 0.5580 for the Stella path. If it may send only a fraction of each dataset's queries to Nano, largest gains first, 15% already matches always-Nano and 25% reaches 0.5334
(`results/m15_e5_oracle.json`, registered). Most queries do not need the transformer; the ones that
do are few.

### 6.2 What the Table Cannot Read

Three SciFact queries show the pattern (exploratory, rank of the first relevant document; `results/m15_examples.json`):

| Query | Zero | Nano |
|---|---:|---:|
| ADAR1 binds to Dicer to cleave pre-miRNA. | 1 | 1 |
| Ivermectin is used to treat lymphatic filariasis. | not in top 100 | 2 |
| Albendazole is used to treat lymphatic filariasis. | not in top 100 | 2 |

Zero's tokenizer splits "ivermectin" into `iv ##er ##me ##ct ##in` and "filariasis" into
`fi ##lar ##ias ##is`. A table can only average those rows, each shared with thousands of unrelated
words; Nano composes them in context. Fragmentation is not the whole story: "ADAR1" and "Dicer" split
almost as badly (2.29 subwords per word) and Zero still ranks the answer first, because the fragments
it averages happen to be distinctive.

A table also cannot read word order. Shuffling a query's words costs the Stella path 6.9% and Nano
4.4% of their own nDCG@10 on SciFact, NFCorpus and FiQA, and Zero nothing, by construction; on FiQA
the Stella path loses 13.7% (`results/m15_e4_prefix.json`). And a table absorbs, rather than
benefits from, any affine map after pooling or any fixed per-token weighting such as IDF: the
result is another table (Appendix A proves this). Such post-processing adds no representational capacity to a table; it can still help as an initialization or prior, but it cannot express a function a table could not.

### 6.3 Short and Unfinished Queries

Search-as-you-type is where a near-free encoder would matter most. We cut each test query of
SciFact, NFCorpus and FiQA and measured how much of its own full-query nDCG@10 each tier keeps
(macro over the three sets; registered):

| Cut | Stella query | Nano | Zero | BM25 |
|---|---:|---:|---:|---:|
| First word | 0.293 | 0.288 | 0.297 | 0.362 |
| First two words | 0.389 | 0.381 | 0.387 | 0.448 |
| First three words | 0.479 | 0.468 | 0.461 | 0.528 |
| First half of the words | 0.610 | 0.597 | 0.604 | 0.656 |
| First half of the characters | 0.549 | 0.533 | 0.547 | 0.488 |

At the cuts shown the three dense tiers keep the same share within 0.02; at 75% of the words the spread is 0.034 (0.846, 0.829 and 0.812 for Stella, Nano and Zero). On these three sets, a larger query encoder does not recover measurably more from a truncated query than a lookup table does. BM25 keeps a larger share on word prefixes but a smaller one when the cut lands inside a word,
where an exact term match fails and a subword table still sees the fragment. These are cut test
queries, not real typing sessions.

## 7. Can a System Exploit the Shared Space?

### 7.1 Routing on Signals the Cheap Tier Already Has

A router that sends a query to Nano must decide before running Nano. We fitted each router's
direction and threshold on the two development forums for Nano budgets of 10%, 25% and 50%, froze
them, and compared each against random routing at the same realized Nano share and against the
oracle at that share (`results/m15_e6_router.json`, `results/m15_e10_router.json`, registered).

Fertility, the number of subwords per word, beats random routing by 0.003 to 0.005 macro nDCG@10 on
12 datasets, against an oracle margin of 0.03 to 0.05: about a tenth of the headroom. `[E10]` The
norm of Zero's pooled vector before normalization, its word count, the margin between its first and
tenth retrieved document, and the agreement between its top 10 and BM25's.

![Figure 5](figures/f5_routing.png)

### 7.2 Blending Two Query Vectors in One Search

`[E9]` Because Zero and Nano write into one space, a system can search once with
normalize((1 - a) Nano + a Zero), paying Zero's 0.044 ms on top of Nano. The weight is fitted on the
development forums and evaluated on six public sets, against Nano alone and against Nano + BM25
fusion; the same for the Stella path.

## 8. The Pre-Registered Test

Before any evaluation data were read, the project registered head-to-head contrasts of Nano against
bge-small and LEAF on six BEIR sets, with a clean-4 partition (NFCorpus, SCIDOCS, SciFact,
TREC-COVID) that excludes the sets Stella's recorded training data names, and a one-shot test on four
held-out sets. All results, including the unfavorable ones (`m21/BENCHMARKS.md`,
`results/m13_reserved_run.json`):

| Contrast | Partition | Difference in nDCG@10 | Registered outcome |
|---|---|---:|---|
| Nano - bge-small | clean-4 | +0.017648 (one-sided lower bound +0.003674) | established |
| Nano - bge-small | all six | +0.027449 | established |
| Nano - LEAF | all six | +0.016181 | established |
| Nano - LEAF | clean-4 | -0.001063 | unresolved |
| Nano - bge-small | held-out NDO-3, dataset-weighted | +0.0032 [-0.0069, +0.0134] | registered descriptive |
| Nano - bge-small | held-out NDO-3, query-pooled | -0.0121 [-0.0219, -0.0024] | registered descriptive |
| Nano - LEAF | held-out NDO-3 | -0.0389 [-0.0488, -0.0290] | registered descriptive |

The held-out test was registered as descriptive, with no significance gate, so its rows report differences and intervals rather than outcomes. NDO-3 weights DBpedia-entity 0.5 and the two held-out CQADupStack forums 0.25 each; FEVER's row is in
Appendix C. "Unresolved" is not equivalence; we computed no equivalence interval. Nano searches
Stella's 1024-dimensional index while bge-small and LEAF search their own, so these contrast systems,
not query encoders alone. On the held-out sets Nano is below LEAF, and we report it as such.

## 9. Limitations

One document tower and one family of students. Section 5 covers one closed-form recipe on ten
checkpoints; its correlations have no significance test. BEIR-15 results are descriptive, and
query-resampling intervals exclude training-seed variation. Latencies come from one laptop CPU and
one runtime; we did not measure server throughput or GPU serving. The MS MARCO subset of Section 4.2
removes most competing passages, so its absolute nDCG is optimistic. Several BEIR sets share source
families with the towers' training data, and Nano's bge-small backbone trained on MS MARCO.

## 10. Conclusion: Decisions This Supports

- **Choosing the index tower when you may want cheap queries:** do not pick by leaderboard; fit the
  cheap query encoder and score it on a small development set. In our data the smaller checkpoint of a family was the better choice in all four families with two sizes.
- **Choosing a query tier:** a 34.5M transformer keeps nine tenths of a 400M tower at one fourteenth
  of the latency; a lookup table keeps four fifths at one seven-hundredth, and on truncated test queries keeps about the same share of its score as the tower. `[E2]`
- **Serving both:** most queries do not need the transformer, but a cheap router has to find the
  ones that do. `[E10]` `[E9]`

## Appendix A. What a Mean-Pooled Table Absorbs

The full proof is in `m15/PROOF_ABSORB.md`; we summarize it. Let a table with rows $w_t$ pool a query
with weights $\alpha_t(q) \ge 0$ that sum to one (mean pooling, and Zero's count-saturated pooling),
then normalize. (1) Applying $x \mapsto Ax + b$ after pooling equals pooling the table
$w'_t = A w_t + b$, because $\sum_t \alpha_t (A w_t + b) = A \sum_t \alpha_t w_t + b$. The sum-to-one
condition matters: under sum pooling the offset scales with query length. (2) Reweighting tokens by
fixed $g_t > 0$ and renormalizing equals the table $g_t w_t$ up to a positive scalar that the final
normalization removes. So centering, whitening, principal-component removal, IDF or SIF weighting,
and their compositions are all tables of the same shape, and cannot extend what the architecture
represents. They do not cover nonlinear maps, pooling weights that depend on count patterns, or word
order. A numerical check against explicitly rebuilt tables agrees to 9.31e-14
(`results/m7_absorb_check.json`).

## Appendix B. What Does Not Travel With a Swap

A swap is legal when the replacement emits vectors aligned with the index's document space, with the same width, normalization and similarity;
our served paths match their references to 4.5e-08 (Zero), 1.2e-07 (Nano) and a minimum cosine of
1.00000000 (Stella through the published document graph). Three things fail silently: Stella without
its prompt scores at cosine 0.80 against the correct vector; Stella's tokenizer pads to 512 by
default, which drops cosine to 0.35; and one serving library pooled Nano in float64 until we fixed it
upstream.

## Appendix C. Full Tables

Per-dataset BEIR-15 rows for all eight systems, the FEVER row of the held-out test, E8 per-dataset
tables, and the full E1 latency inventory: `m15/EVIDENCE.md`.

## Appendix D. Reproducibility

Every table and figure regenerates from committed JSON (`python m15/figures/make_figures.py`).
Each M15 result file (`results/m15_*.json`) carries a receipt: script hash, git commit, model and dataset revisions, seed and machine; earlier results (BEIR-15, the held-out test, the tier builds) carry the provenance records of the milestone that produced them. Methods and reviews: `m15/MEASUREMENTS.md`, `m15/REVIEWS/`.

## References

Full citations with the passages we rely on: `m15/RELATED_WORK.md`.

- [QED] Query Encoder Distillation via Embedding Alignment is a Strong Baseline Method to Boost
  Dense Retriever Online Efficiency. arXiv:2306.11550.
- [EmbedDistill] Kim, S., Rawat, A. S., Zaheer, M., et al. EmbedDistill: A Geometric Knowledge Distillation for Information
  Retrieval. arXiv:2301.12005.
- [LEAF] LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned
  Representations. arXiv:2509.12539.
- [NanoVDR] Liu, Z., Zhang, Y., and Xiao, Y. NanoVDR: Distilling a 2B Vision-Language Retriever
  into a 70M Text-Only Encoder for Visual Document Retrieval. arXiv:2603.12824.
- [LightRetriever] arXiv:2505.12260.
- [pyNIFE] Tulkens, S. pyNIFE. https://github.com/stephantul/pynife
- [Model2Vec] MinishLab. https://github.com/MinishLab/model2vec
- [StaticEmb] Aarsen, T. Train 400x faster Static Embedding Models with Sentence Transformers.
  Hugging Face blog, 2025.
- [Cho and Hariharan 2019] On the Efficacy of Knowledge Distillation. ICCV 2019. arXiv:1910.01348.
- [PROD] Lin, Z., et al. PROD: Progressive Distillation for Dense Retrieval. WWW 2023.
  arXiv:2209.13335.
- [Arabzadeh et al.] Arabzadeh, N., Yan, X., and Clarke, C. L. A. Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval
  Strategy Selection. CIKM 2021. arXiv:2109.10739.
- [QPP] See `m15/RELATED_WORK.md`, item 8.
- [BCT] Shen et al. Towards Backward-Compatible Representation Learning. CVPR 2020.
- [FCT] Ramanujan et al. Forward Compatible Training for Large-Scale Embedding Retrieval Systems.
  arXiv:2112.02805.
