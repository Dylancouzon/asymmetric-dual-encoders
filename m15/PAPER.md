# Screen the Student: Cheap Query Encoders for a Frozen Document Index

**Draft v6, 2026-09-30. Not for circulation.** Each claim names the committed file that holds its
numbers; `m15/EVIDENCE.md` restates every number in plain language with its source and caveats.
Results are labelled **registered** (method frozen before observation, `m15/MEASUREMENTS.md`) or
**exploratory**.

## Abstract

If a cheap query encoder can write into the vector space of an existing document index, the expensive
half of a retrieval system never has to move. We fit the same lookup-table query encoder, a token
table with no neural network at query time, to ten embedding towers and ask which towers make good
tables. The tower's own retrieval quality did not rank them: Spearman -0.09 between a tower's score
and its table's on six public BEIR sets. A few-minute screen of the table on two development forums
did (0.90, and between 0.87 and 0.98 when any one model family is left out). Over one frozen
`stella_en_400M_v5` index, the table keeps 81.4% of the tower's nDCG@10 on 15 BEIR datasets at 0.044
ms per query on a laptop CPU, and a 34.5M-parameter distilled transformer keeps 90.5% at 2.25 ms,
against 31.6 ms for the tower. The table's loss is concentrated on queries the tower reads
differently when their words are shuffled, not on queries that are merely fragile or full of rare
subwords, and routing on the table's own retrieval margin recovers a quarter of the headroom an oracle
router would. Inside a vector search engine the table's queries need a wider graph search, so its
encode-time saving over the transformer shrinks from about 60-fold to 2- to 3-fold end to end on a
one-million-passage collection.

## 1. Introduction

Suppose your document vectors come from a strong, slow embedding model. Re-embedding the corpus is
the expensive, risky operation; encoding a query is cheap by comparison but happens on every request,
on every device, and at every keystroke if you offer search-as-you-type. You want a much cheaper query
encoder that writes into the index you already have. Two decisions follow, and an efficiency frontier
answers neither:

1. **If you may want cheap queries later, which embedding model should the index be built on?** The
   natural guess is the best one on the leaderboard, since the cheap encoder is distilled from it.
2. **Which queries actually need the expensive encoder?** If few do, a system can serve the cheap tier
   by default and escalate, all against one index.

We measure both. The natural guess in question 1 is wrong in an informative way. We fit one
closed-form lookup-table recipe to ten towers: the towers' own retrieval quality, and three scalar
proxies for why a tower might distill well, did not rank the tables they produced, while scoring each
table on a small development set did. We call this *screen the student*. For question 2, a per-query
oracle shows the headroom is large and concentrated, the table's losses are largest on queries the tower reads differently when their words are shuffled, and the table's own retrieval margin is the cheapest
signal we found that locates those queries.

**Contributions.**

- A controlled comparison of ten towers under one lookup-table recipe, with lambdas frozen on
  development data before six public sets were scored, showing that tower quality did not rank table
  quality and a direct development screen did (Section 4).
- A per-query localization of what a table loses: order-dependent queries, controlled for the tower's own score and for query fragility, with worked examples (Section 5).
- An oracle upper bound and five frozen routers built on signals the table already has, and a test of
  blending two tiers' query vectors in one search (Section 6).
- The deployment cost in a vector search engine: approximate search and quantization cost the table
  more than the transformer, and the end-to-end saving is what remains (Section 7).
- A registered head-to-head test of the transformer tier reported in full (Appendix B), and a public
  harness in which every number regenerates from committed files.

## 2. Prior Work and What Is New Here

**Cheap query encoders over a frozen document side.** Query-encoder distillation onto a frozen
document encoder keeps 92.5% of the teacher on BEIR, averaged per dataset [QED]. EmbedDistill aligns
student and frozen teacher embeddings and reports 95-97% on MS MARCO and NQ [EmbedDistill]. LEAF
releases 23M-parameter query models that pair with a larger document model and keep 97.7% [LEAF].
NanoVDR distills a visual-document retriever's query side and keeps 95.1% [NanoVDR].

**Lookup-table query encoders.** Static embeddings compute a query as an average of token vectors
[Model2Vec, StaticEmb]. pyNIFE fits a per-token table against a frozen off-the-shelf tower, reuses the
tower's index, and already proposes a fast static path with a slow contextual fallback for queries that
need context [pyNIFE]. LightRetriever serves queries by embedding lookup and keeps about 95% on BEIR,
but trains its document LLM jointly with the query table [LightRetriever].

**Teachers, routing and compatibility.** A stronger teacher does not always make a better student
[Cho and Hariharan 2019], including in dense retrieval distillation [PROD]. Choosing a retriever per
query between sparse and dense retrieval [Arabzadeh et al.] and predicting retrieval quality [QPP] are
established. Backward- and forward-compatible training upgrade an embedding model without
re-encoding a gallery [BCT, FCT].

**What is new here.** None of the above holds one recipe fixed across many frozen towers and asks what
predicts the student; we do, with the development choices frozen before public evaluation. We add a
per-query analysis of what the table loses with controls for the obvious confounds, routers tested
against an oracle at matched traffic, the observation that the static tier pays a larger
approximate-search penalty, and, as far as we are aware, the first test of blending two query
encoders' vectors over one index. The architecture is not new.

## 3. Setup

**The frozen index.** `NovaSearch/stella_en_400M_v5` at revision `ffeb2b7e`: 1024-dimensional document
vectors, fp32 compute, normalized, stored fp16, cosine similarity. Sections 5 to 7 search these
vectors. Section 4 fits tables to other towers over each tower's own document vectors.

**Three query tiers over that index.**

| Tier | What runs at query time | How it was built | Build cost |
|---|---|---|---|
| Stella query path | the tower itself, with its `s2p_query` prompt | nothing to build | none |
| Nano | a 34,540,672-parameter transformer: bge-small layers 12, 8 and 4 concatenated, a linear head to 1024, mean pooling | squared L2 to Stella's vectors over 199,999,721 examples (75% query, 25% document batches) | 57.3 A100 hours, about $95 |
| Zero | a 30,522 x 1024 int8 table; the query vector is the normalized, count-saturated mean of its tokens' rows | cosine and ranking losses against Stella, then int8 | about 20 minutes of training after 8-12 hours of encoding on one RTX 3080 |

Sources: `results/m13_build_record.json`, `m7/RECIPE.md`. BM25 (bm25s, Lucene, k1 = 1.2, b = 0.75) is a
reference and the lexical half of fused systems (DBSF over two top-100 prefetches, as in Qdrant's
Query API).

**Evaluation.** Quality is nDCG@10 under exact search on 15 BEIR datasets
(`results/m20_beir15_run.json`, per-query rows in `results/m20_beir15_scores/`). Approximate search
appears only in Section 7. Four datasets (FEVER, DBpedia-entity and two CQADupStack forums) were held
out for a registered one-shot test; their frozen results enter the BEIR-15 aggregates and they play no
fitting or diagnostic role here. Every choice fitted for this paper (ridge weights, router thresholds,
blend weights) is made on two other CQADupStack forums, physics and programmers, and evaluated
elsewhere; the tiers themselves were built earlier on the project's development suite. The two tiers did not see the same training data: Zero's included FEVER-train and HotpotQA, and Nano's pool excluded FEVER, so Zero-versus-Nano comparisons on claim-style sets (FEVER, Climate-FEVER) mix architecture with data. Our
single-prompt Stella path scores 0.5614 on BEIR-15 (the model card reports 58.97 with task-specific
instructions, a 400-token limit and bf16); our bge-small (0.5171), LEAF (0.5402) and BM25 (0.4006)
match their public numbers within half a point.

## 4. A Strong Tower Need Not Yield a Strong Table

A lookup table can be fit in closed form to any tower that shares its tokenizer: a ridge regression
from each query's token counts to the tower's query vector, anchored at each token's own tower
embedding (`m8src/teacher_screen.py`). We fit this one recipe to 11 configurations of ten checkpoints
from six model families, all sharing one BERT WordPiece vocabulary, on 337,981 training queries
screened against every held-out set. Each ridge weight was chosen on the two development forums and
frozen in a committed file (`results/m15_e8_frozen_lambdas.json`) before any public set was scored
(`results/m15_e8_towers.json`, registered).

![Figure 1](figures/f3_towers.png)

**Figure 1.** Left: each tower's own nDCG@10 against its table's, on six public BEIR sets. Right: the
table's score on the two development forums against its public score. Gray lines join the base and
large checkpoints of one family.

**The tower's score did not rank the tables.** Across the ten checkpoints, the Spearman correlation
between a tower's own score and its table's is -0.09 on the six public sets, and +0.14 on the four of
them with no disclosed training exposure. Leaving any one family out moves it between -0.43 and +0.24
(`results/m15_e13_robustness.json`). The strongest tower there, gte-large-en-v1.5 (0.5970), made the
weakest table (0.2455, keeping 41% of its tower); Stella, second-strongest, made the best table and
kept 69%.

**Scoring the table itself did.** The table's development-forum score ranks the public scores with
Spearman 0.90 (0.68 on the four exposure-free sets), and between 0.87 and 0.98 when any one family is
left out. The screen took 4 to 7 minutes per tower on one A100, measured between successive output files; for every tower but Stella, whose training-query vectors were cached, that includes encoding the 337,981 training queries and both forums. In each of the four families where we fit two sizes,
the smaller checkpoint scored higher: bge-base over bge-large (0.3529 against 0.2845 on the public
sets), e5-base over e5-large (0.2930 against 0.2585), gte-base over gte-large (0.3252 against
0.2455) and arctic-m over arctic-l (0.3279 against 0.3034); the arctic pair also differs in version,
and within each pair size is confounded with training and dimension.

**Three proxies that did not rank them either.** We measured, per tower, on the real test queries
(exploratory; `results/m15_e11_mechanism.json`, `results/m15_e11b_margin.json`):

| Candidate explanation | Measurement | Spearman with table quality |
|---|---|---:|
| The tower is "almost a bag of words" | mean cosine between the table's and the tower's query vector | -0.07 |
| The tower reads word order, which a table cannot | mean cosine between a query's vector and its shuffled version's | -0.27 |
| The table's error is small relative to the tower's ranking margins | RMS vector error over the median gap between the tower's 1st and 10th score, averaged over six sets | -0.42 |

The first two point the wrong way and the third points the right way but weakly (and at -0.13 against
retention). The e5 checkpoints show how vector fidelity can mislead: their tables reproduce the
tower's query vectors best (cosine 0.90 and 0.89, against 0.78 for Stella), yet rank documents worse.
One reading, a hypothesis at n = 10, is that e5's own score margins are the narrowest (a median
1st-to-10th gap of 0.026 averaged over the sets, against 0.084 for Stella), so a small vector error
reorders its results; gte-base, with the widest margins (0.109), still makes a middling table. What
the data show is narrower: across these towers, the vector fidelity a distillation loss optimizes did
not track ranking quality, while a direct ranking screen did.

**Scope.** One recipe, ten checkpoints, six public sets, no significance test. No checkpoint's
authors disclose training on these six test sets; most trained on their source families
(`m15/e8_exposure.json`). An earlier screen in this project used a training list with 1.31% overlap
with held-out queries and found a correlation of 0.000 over eight configurations; this refit uses the
cleaned list.

## 5. What the Table Cannot Represent

### 5.1 What Post-Processing It Absorbs

A mean-pooled table absorbs any affine map applied after pooling and any fixed per-token weighting
such as IDF or SIF: in each case the transformed system is another table of the same shape (Appendix
A proves this; a numerical check agrees to 9.31e-14, `results/m7_absorb_check.json`). Such
post-processing therefore adds no representational capacity, although it can still help as an
initialization. Nor did adding rows: in earlier work, closed-form rows for word pairs and finer
segmentations did not improve the development score (best -0.0028, `m8/FINDINGS.md`). And every such
table is blind to word order by construction.

### 5.2 The Loss Sits Where the Tower Reads Word Order

A table cannot tell "A treats B" from "B treats A", which makes word order the obvious suspect for its
loss; rare words split into many subwords are a second. We measured both on the 3,727 test queries of
the six public sets (exploratory; `results/m15_e12_failure_modes.json`,
`results/m15_e12b_fragility.json`). A query's *order dependence* is how much the Stella path's own
nDCG@10 drops when its words are shuffled; its *fertility* is subwords per word.

| Per-query association with the Stella-minus-Zero gap | Spearman |
|---|---:|
| Order dependence, all 3,727 queries | 0.46 |
| Order dependence, within bins of the Stella path's own score | 0.46 |
| Order dependence, controlling for fragility, 3,069 queries Stella scores above 0 | 0.42 |
| Fertility, all queries | 0.10 |

On the 31% of queries where shuffling hurts the Stella path, it beats Zero by 0.215 nDCG@10; on the
rest, by 0.038. The gap grows less with fertility, from 0.071 in the lowest tercile to 0.133 in the
highest.

Two confounds could produce this. Both quantities can only be large where the Stella path scores well,
so we repeated the correlation within bins of that score; it does not move. And some queries might
lose under any perturbation. To test that, we moved each Stella query vector by exactly as much as
shuffling moved it, but in a random direction. Among the 3,069 queries the Stella path scores above zero, that move costs 0.009 nDCG@10 on average against 0.047 for shuffling, and its damage correlates with Zero's gap at only 0.07. Shuffle sensitivity, not
general fragility, is what goes with the table's loss. This is an association: shuffling changes a
query in more ways than word order alone. Nano's advantage over Zero also concentrates on
order-dependent queries, less strongly (0.22), which fits a transformer that reads part of the order
the tower reads (shuffling costs Nano 4.4% of its score, and the Stella path 6.9%, on SciFact, NFCorpus
and FiQA; `results/m15_e4_prefix.json`).

Two FiQA questions show what that looks like (`results/m15_e13_robustness.json`; rank of the first
relevant answer):

| Query | Stella | Stella, words shuffled | Nano | Zero |
|---|---:|---:|---:|---:|
| What credit card information are offline US merchants allowed to collect for purposes other than the transaction? | 1 | 59 | 11 | not in top 100 |
| What can I do with a physical stock certificate for a now-mutual company? | 1 | 8 | 13 | 29 |

Both ask about a relation between their parts ("allowed to collect ... for purposes other than", "do with ... for a now-mutual company"), and in both the tower's ranking collapses when the words are shuffled; a bag of tokens sees the shuffled and unshuffled versions as the same query. Fragmentation shows up too, in rarer, sharper cases (`results/m15_examples.json`): on SciFact,
"Ivermectin is used to treat lymphatic filariasis" splits into `iv ##er ##me ##ct ##in` and
`fi ##lar ##ias ##is`, and Zero misses the relevant document entirely while Nano ranks it second. Yet
"ADAR1 binds to Dicer to cleave pre-miRNA" splits almost as badly (2.29 subwords per word) and Zero
ranks the answer first.

## 6. Which Queries Need Context, and Can a System Find Them?

### 6.1 An Upper Bound

On the 12 BEIR-15 datasets outside the held-out test, averaging per-dataset shares, Zero and Nano both
score 0 on 17% of queries, tie above 0 on 26%, and Zero is higher on 18% and Nano on 39% (pooling all
queries gives 56% ties, because Quora is large and tied on 72% of its queries). A per-query oracle
that knows the relevance labels and picks the better tier reaches 0.5473 macro nDCG@10, against 0.5161
for always-Nano and 0.5580 for the Stella path. Allowed to send only 15% of each dataset's queries to
Nano, largest gains first, it already matches always-Nano; at 25% it reaches 0.5334
(`results/m15_e5_oracle.json`, registered). Without Climate-FEVER, the one FEVER-family set among the 12, always-Nano scores 0.5406 and the oracle needs 20% of queries to match it (0.5453). The oracle is an upper bound, not a router: it shows the
gain is concentrated, not that a system can find it.

### 6.2 Routers Built on What the Table Already Knows

A router must decide before running Nano. For each signal we fitted a direction and threshold on the
two development forums for Nano budgets of 10%, 25% and 50%, froze them, and compared the router with
random routing at the same realized Nano share and with the oracle at that share
(`results/m15_e6_router.json`, `results/m15_e10_router.json`, registered).

| Signal the table computes anyway | Known | Router minus random, 25% budget | Share of oracle gain recovered | Sets |
|---|---|---:|---:|---|
| Subwords per word (fertility) | before search | +0.0049 | 0.10 | 12 |
| Norm of the pooled vector before normalization | before search | +0.0016 | 0.03 | 12 |
| Word count | before search | +0.0018 | 0.04 | 12 |
| Gap between Zero's 1st and 10th document score | after Zero's search | +0.0175 | 0.25 | 6 |
| Documents Zero's and BM25's top 10 share | after both searches | +0.0154 | 0.31 | 6 |

![Figure 2](figures/f5_routing.png)

**Figure 2.** Left: the upper bound from label-aware routing on the 12 evaluation sets, against random
routing. Right: the share of the oracle's gain over random routing each frozen router recovers, per
Nano budget; orange signals are known before the search, green ones after Zero's own search.

The three query-only signals we tried carry little. Zero's own retrieval confidence carries more:
routing the queries with the narrowest margin to Nano (28% of each dataset's queries on average, 24%
of all queries pooled) scores 0.4865 on the six sets, against 0.4690 for random routing at that share
and 0.5317 for always-Nano. The gain is largest where Zero is weakest: +0.043 on SciFact and +0.032 on
TREC-COVID. Sending the routed queries to the blend of Section 6.3 instead of plain Nano scores 0.4834,
0.0031 lower. The agreement signal counts shared documents, and at the 10% budget its frozen threshold
routes no queries.

As a cost estimate, combining E1's encode medians with the MS MARCO search medians of Section 7 at
unquantized `ef` 128, this margin router spends about 2.7 ms per query against 3.8 ms for always-Nano.
Put plainly, this router saves about 1.1 ms per query and gives up 0.045 nDCG@10 against always-Nano on these sets, and because Section 7 finds Nano only 2 to 4 times as expensive as Zero end to end, no router between these two tiers can save more than roughly half of Nano's cost. Routing is worth it where the transformer is expensive relative to search, as on a device; the oracle shows how much a better signal could still recover.

### 6.3 Blending Two Query Vectors in One Search

Because Zero and Nano write into one space, a system can search once with
normalize((1 - a) Nano + a Zero), paying Zero's 0.044 ms on top of Nano and no second search. On the
two development forums the blend helped: a = 0.3 raised Nano from 0.4247 to 0.4295. On the six public
sets the frozen blend lowered Nano by 0.0051 (95% interval -0.0092 to -0.0011), and by 0.0070 on the
four exposure-free sets; it gained only on ArguAna (+0.005) and lost most on TREC-COVID (-0.016). For the Stella path the forums chose a = 0 (`results/m15_e9_blend.json`, registered descriptive; a code check printed
SciFact's curve before the weights were frozen, which changed nothing in the procedure and is recorded
in `m15/MEASUREMENTS.md`).

The development-selected blend did not improve the six-set result, so the same two forums that ranked
the towers in Section 4 did not pick a useful blend weight. One untested reading: CQADupStack's
duplicate questions reward lexical overlap, which is the property a blend with a table adds.

## 7. Does Cheap Encoding Survive Retrieval?

**Table 1.** BEIR-15 macro nDCG@10 under exact search over the Stella index, and query-encode latency
(median over 100 real test queries of 5-12 words, batch one, four CPU threads, ONNX Runtime, Zero on
NumPy, Apple M5 Pro; `results/m15_e1_latency.json`). Retention is a ratio of macros. Registered.

| Query side | nDCG@10 | Retention | Encode p50 |
|---|---:|---:|---:|
| Stella query path | 0.5614 | 1.000 | 31.6 ms |
| Nano | 0.5081 | 0.905 | 2.25 ms |
| Nano + BM25 (DBSF@100) | 0.5110 | 0.910 | 2.25 ms + BM25 |
| Zero | 0.4572 | 0.814 | 0.044 ms |
| Zero + BM25 (DBSF@100) | 0.4933 | 0.879 | 0.044 ms + BM25 |
| BM25 alone | 0.4006 | 0.713 | |

Nano keeps nine tenths of the tower at one fourteenth of its encode latency; Zero keeps four fifths at one seven-hundredth and beats BM25 on 11 of 15 datasets. Nano is not the strongest small query encoder: on BEIR-15 it scores below bge-small on bge-small's own index (0.5171) and below LEAF on its arctic-embed-m index (0.5402, at 1.39 ms), and its 90.5% retention is below the 92.5% to 97.7% reported for other distilled query encoders against their own teachers. Its reason to exist is that it shares the Stella index with Zero and the Stella path, which is what makes the routing and blending of Section 6 possible; its distance to a 400M tower is larger than the teacher-student gaps in that prior work. Retention varies more across datasets than
across tiers, from 0.667 (Zero on TREC-COVID) to 0.988 (Nano on Quora); Figures 4 and 5 in Appendix D
show the frontier and the per-dataset spread.

**A table's query needs a wider graph search.** We served each tier from Qdrant (v1.19.1, native
macOS build) over a one-million-passage MS MARCO subset that keeps every judged positive and fills the
rest at random (a diagnostic, not full MS MARCO; MS MARCO is used for validation only) and over FiQA
(57,638 documents). For each tier we swept HNSW `ef` and three quantizations (int8 scalar, 1-bit
binary and 4-bit TurboQuant, each with rescoring and 1x-4x oversampling), and found the cheapest
end-to-end setting within 1%, 2% and 5% of the tier's own exact nDCG@10 (registered method). At the
same unquantized setting on the 1M collection (`ef` 16), approximate search costs Zero 16.3% of its
exact score, Nano 7.0% and the Stella path 4.4%, and their top-10 lists agree with exact search on 82%,
91% and 93% of neighbors (`results/m15_e2_ann_msmarco1m.json`). The median cosine between a query and
its nearest document is 0.60 for Zero, 0.74 for Nano and 0.78 for the Stella path: a table's averaged
query vector sits farther from the documents the graph was built on, which is consistent with, though
not a test of, a harder graph search.

![Figure 3](figures/f4_system.png)

**Figure 3.** nDCG@10 under approximate search against end-to-end p50 (encode plus search); each point is the fastest setting reaching that quality, and dotted lines mark each tier's exact score. Left: 1M MS MARCO subset; right: FiQA.

**The saving survives, and shrinks.** Within 1% of its exact score on the 1M collection, Zero needs
`ef` 512 with binary quantization and 2x oversampling (search p50 1.09 ms), Nano `ef` 256 with 4x
oversampling (0.87 ms), and the Stella path `ef` 64 with TurboQuant (0.66 ms). With encoding, that is
1.11 ms for Zero, 2.48 ms for Nano and 21.3 ms for the Stella path; Nano costs 2.2 times Zero at the 1%
target and 3.1 times at 5%, down from 62 times for the encoders alone on these short queries. On FiQA
the same ordering holds (at `ef` 16: 6.8%, 3.2% and 1.5% lost) and Nano costs 3.8 to 4.3 times Zero end
to end (`results/m15_e2_ann_fiqa.json`). Absolute laptop latencies move with background load: an
earlier FiQA run, during which another job used the CPU, measured Nano's encode at 1.75 ms against
2.83 ms in the rerun while the ratio stayed within 3.4 to 4.3 (`results/m15_e2_ann_fiqa_run1.json`),
so we read absolute latencies from E1 and the MS MARCO run.

**Fusion is not free in the engine either.** A fused query (Qdrant's server-side BM25 and DBSF over two
prefetches of 100) takes about 4.7 ms on the 1M collection and, on this subset, scores 0.6183 for Zero
against 0.6169 for Zero's exact dense search, and 0.037 below Nano's.

## 8. Robustness and Limitations

**Short and unfinished queries.** We cut each test query of SciFact, NFCorpus and FiQA and measured
how much of its own full-query nDCG@10 each tier keeps (macro over the three sets, registered,
`results/m15_e4_prefix.json`):

| Cut | Stella query | Nano | Zero | BM25 |
|---|---:|---:|---:|---:|
| First word | 0.293 | 0.288 | 0.297 | 0.362 |
| First two words | 0.389 | 0.381 | 0.387 | 0.448 |
| First three words | 0.479 | 0.468 | 0.461 | 0.528 |
| First half of the words | 0.610 | 0.597 | 0.604 | 0.656 |
| First half of the characters | 0.549 | 0.533 | 0.547 | 0.488 |

The dense tiers' shares are similar at every cut shown (within 0.02) and at 75% of the words differ by
0.034 (0.846, 0.829 and 0.812 for Stella, Nano and Zero). This is a descriptive similarity on cut test
queries, not an equivalence test and not real typing. BM25 keeps a larger share on word prefixes and a
smaller one when the cut lands inside a word.

**The registered test.** Before any evaluation data were read, the project registered head-to-head
contrasts of Nano against bge-small and LEAF. Nano beat bge-small on the four exposure-free sets
(+0.0176, one-sided lower bound +0.0037) and on all six, and beat LEAF on all six; Nano against LEAF
on the four exposure-free sets was unresolved (-0.0011). On the one-shot held-out sets, registered as
descriptive, Nano's dataset-weighted difference from bge-small was +0.0032, with a 95% interval (-0.0069 to +0.0134) that spans zero, and Nano was below LEAF (-0.0389). Appendix B has the full table. These compare systems on
different indexes, not query encoders alone.

**Limitations.** One document tower and one family of students; Section 4 covers one closed-form
recipe on ten checkpoints, and its correlations carry no significance test. BEIR-15 results are
descriptive, and query-resampling intervals exclude training-seed variation. Latencies come from one
laptop CPU and one runtime; server throughput and GPU serving are not measured. The MS MARCO subset
removes most competing passages, so its absolute nDCG is optimistic. Several BEIR sets share source
families with the towers' training data, and Nano's bge-small backbone trained on MS MARCO.

## 9. Conclusion

A frozen index can serve queries from a lookup table that gives up about a fifth of the tower's quality for a seven-hundredth of its encode cost, or from a small transformer that gives up a tenth for a fourteenth. The practical lessons are three. Pick the tower by screening the table you will serve,
not by the tower's own score: in our ten towers the leaderboard order did not survive distillation,
while a few-minute development screen predicted it. Expect the table's largest losses on queries the tower reads differently when their words are shuffled, and route on the table's own retrieval margin, which recovers about a quarter of the oracle's gain over matched random routing. And measure the saving inside the search engine: a table's query costs more graph search, so on our one-million-passage collection its roughly 60-fold encode saving over Nano became 2.2- to 3.1-fold end to end within 1% to 5% of each tier's exact quality.

## Appendix A. What a Mean-Pooled Table Absorbs

The full proof is in `m15/PROOF_ABSORB.md`. Let a table with rows $w_t$ pool a query with weights
$\alpha_t(q) \ge 0$ that sum to one (mean pooling, and Zero's count-saturated pooling), then
normalize. (1) Applying $x \mapsto Ax + b$ after pooling equals pooling the table $w'_t = A w_t + b$,
because $\sum_t \alpha_t (A w_t + b) = A \sum_t \alpha_t w_t + b$; under sum pooling the offset would
scale with query length instead. (2) Reweighting tokens by fixed $g_t > 0$ and renormalizing equals the
table $g_t w_t$ up to a positive scalar that the final normalization removes. Centering, whitening,
principal-component removal, IDF or SIF weighting and their compositions are therefore tables of the
same shape. The lemmas do not cover nonlinear maps, pooling weights that depend on count patterns, or
word order.

## Appendix B. The Registered Test in Full

| Contrast | Partition | Difference in nDCG@10 | Registered outcome |
|---|---|---:|---|
| Nano - bge-small | clean-4 | +0.017648 (one-sided lower bound +0.003674) | established |
| Nano - bge-small | all six | +0.027449 | established |
| Nano - LEAF | all six | +0.016181 | established |
| Nano - LEAF | clean-4 | -0.001063 | unresolved |
| Nano - bge-small | held-out NDO-3, dataset-weighted | +0.0032 [-0.0069, +0.0134] | registered descriptive |
| Nano - bge-small | held-out NDO-3, query-pooled | -0.0121 [-0.0219, -0.0024] | registered descriptive |
| Nano - LEAF | held-out NDO-3 | -0.0389 [-0.0488, -0.0290] | registered descriptive |

Clean-4 is NFCorpus, SCIDOCS, SciFact and TREC-COVID, the six-set members Stella's recorded training
data does not name. NDO-3 weights DBpedia-entity 0.5 and the two held-out CQADupStack forums 0.25 each;
FEVER's row is in `m15/EVIDENCE.md`. "Unresolved" is not equivalence; no equivalence interval was
computed. Sources: `m21/BENCHMARKS.md`, `results/m13_reserved_run.json`.

## Appendix C. What Does Not Travel With a Swap

A swap is legal when the replacement emits vectors aligned with the index's document space, with the
same width, normalization and similarity; our served paths match their references to 4.5e-08 (Zero),
1.2e-07 (Nano) and a minimum cosine of 1.00000000 (Stella through the published document graph).
Three things fail silently: Stella without its prompt scores at cosine 0.80 against the correct
vector; Stella's tokenizer pads to 512 by default, which drops cosine to 0.35; and FastEmbed's mean
pooling returns float64 for mean-pooled models, Nano among them, which doubles memory and changes the
raw bytes (qdrant/fastembed issue #752, open upstream; our measurements pool in float32 with ONNX
Runtime directly).

## Appendix D. Frontier and Per-Dataset Figures, Full Tables

![Figure 4](figures/f1_frontier.png)

**Figure 4.** BEIR-15 retention against encode latency. Hollow markers are reference systems on their
own indexes (bge-small 0.5171 at 2.57 ms, LEAF 0.5402 at 1.39 ms).

![Figure 5](figures/f2_per_dataset.png)

**Figure 5.** Per-dataset retention of Zero and Nano. Zero is above Nano on FEVER, HotpotQA and
Climate-FEVER; Zero trained on FEVER-train while Nano's pool excluded it.

Per-dataset BEIR-15 rows for all eight systems, E8's per-dataset tables and the full E1 latency
inventory are in `m15/EVIDENCE.md`.

## Appendix E. Reproducibility

Every figure regenerates from committed JSON (`python m15/figures/make_figures.py`) and every number in
`m15/EVIDENCE.md` from `python m15/make_evidence.py`. M15 result files (`results/m15_*.json`), except E8's frozen-lambda file (written on the pod and bound by hash into E8's result), carry the script hash, git commit, seed and machine; model and dataset revisions are recorded in the result, in
the scripts it names, or in the result it was derived from (E8's frozen lambdas are bound by hash
into E8's result; E12 and E12b take their pins from the committed M20 rows). Earlier results carry the
provenance records of the milestone that produced them. Methods and reviews: `m15/MEASUREMENTS.md`,
`m15/REVIEWS/`.

## References

Full citations with the passages we rely on: `m15/RELATED_WORK.md`.

- [QED] Query Encoder Distillation via Embedding Alignment is a Strong Baseline Method to Boost
  Dense Retriever Online Efficiency. arXiv:2306.11550.
- [EmbedDistill] Kim, S., Rawat, A. S., Zaheer, M., et al. EmbedDistill: A Geometric Knowledge
  Distillation for Information Retrieval. arXiv:2301.12005.
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
- [Arabzadeh et al.] Arabzadeh, N., Yan, X., and Clarke, C. L. A. Predicting Efficiency/Effectiveness
  Trade-offs for Dense vs. Sparse Retrieval Strategy Selection. CIKM 2021. arXiv:2109.10739.
- [QPP] See `m15/RELATED_WORK.md`, item 8.
- [BCT] Shen et al. Towards Backward-Compatible Representation Learning. CVPR 2020.
- [FCT] Ramanujan et al. Forward Compatible Training for Large-Scale Embedding Retrieval Systems.
  arXiv:2112.02805.
