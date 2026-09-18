# Swapping the Query Encoder Over a Frozen Document Index

**Draft, 2026-09-17. Not for circulation.** Sections marked `[M20]` wait on the reserved four and
BEIR-15. Numbers trace to committed result files through `EVIDENCE_INDEX.md`, whose spot-check
table records which batches a reviewer has verified against the source.

## Abstract

A dual-encoder retrieval system has one expensive asset and one cheap one. Encoding a corpus costs
a full pass over every document and produces an index that resists change. Encoding a query costs
one forward pass and produces nothing that has to be stored. We treat the two sides accordingly:
we hold `stella_en_400M_v5` frozen as the document tower, 1024 dimensions, and we replace only the
query side. Three dense query encoders emit into that one index: a per-token lookup table with no
query-time neural network, a 34,540,672-parameter distilled transformer, and the tower's own query
path. BM25 joins them as the lexical half of a fused system, over its own inverted index.

We report what each tier costs and what it retains, under a benchmark partition registered before
any number existed. The distilled student establishes superiority over bge-small on both the
contamination-clean partition and the full six (+0.017648 and +0.027449 nDCG@10) and over LEAF-asym
on the full six (+0.016181), while the clean partition against LEAF stays unresolved (-0.001063).
The lookup table retains 0.755 of the teacher's quality on the same six at 0.1119 ms warm median
query time, 61 to 65 times faster than the two transformer tiers we compare it against.

The measurement that surprised us is the one about cost. The two tiers differ by roughly 50 times as
encoders and by 1.96 times as whole systems at typical query length, because both pay the same
search. The gap widens with query length, reaching 5.10 times at 51 to 120 words, and it moves
between 1.11 and 3.28 times across index configurations. The near-zero-compute tier also carries the
larger artifact: 270.1 MB against 46.1 MB. A near-zero query encoder buys much less than its own
latency number promises, and the query length and the index decide how much.

## 1. Introduction

Retrieval systems are usually evaluated as a whole model. In deployment they come apart. The
document index is built once, costs a corpus-scale encode, and cannot be rebuilt whenever a better
query model appears. The query encoder runs once per request and holds no state. Changing it is
cheap. Changing the index is not.

This asymmetry is not new as an engineering observation, and it is not new as an inference mode.
LEAF ships distilled students explicitly for mixed-checkpoint inference, where the teacher encodes
documents and the student encodes queries against the same index. pyNIFE trains a per-token lookup
table by distillation against a frozen off-the-shelf teacher and reuses that teacher's index
unchanged. Backward-compatible and forward-compatible training name the general goal of upgrading a
model without re-embedding a corpus. We build on all of it, and we claim none of it.

What is missing is the measurement. Every published asymmetric pair trains its cheap query tower
against the document encoder it will eventually retire alongside. None of them asks the question a
team with an existing index actually asks: given an index I already built, which query encoder
should I put in front of it, and what does each choice cost me? Answering it needs more than one
cheap tier, one held-fixed index, and a protocol fixed before the numbers arrive.

That is what this paper provides.

**Contributions.**

1. A two-tier quality-against-compute frontier over one frozen third-party index, with a
   near-zero-compute tier and a sub-35M transformer tier measured under one protocol (Sections 4, 5).
2. Registered head-to-head results for the transformer tier against bge-small and against LEAF-asym,
   with the unresolved contrast reported as unresolved (Section 5).
3. Evidence that the cost advantage of a near-zero query encoder shrinks by one to two orders of
   magnitude at the system level, depends on query length, and arrives with the larger deployed
   artifact (Section 6).
4. Two method results with reach beyond this system: across the eight teachers we distilled, a
   teacher's own retrieval quality carried no signal about the quality of what distilled out of it,
   and a named class of post-hoc embedding transformations is exactly absorbable into a lookup table
   under mean pooling, so it cannot add capacity (Section 7).
5. An account of what a swap costs operationally, including the four places where it is not free
   (Section 3).

## 2. Setting

**The frozen side.** One document tower, `stella_en_400M_v5`, 1024-dimensional, L2-normalized,
cosine similarity. Every result in this paper searches the same document vectors. The tower's
CUDA and CPU paths agree to a minimum cosine of 1.000000 and a maximum absolute difference of
9.07e-05, so an index built on one device holds on the other.

**The swappable side.** Three dense query encoders emit 1024-dimensional vectors into the frozen
index:

| Tier | What it is | Query-time compute |
|---|---|---|
| `zero` | A per-token lookup table distilled against the frozen tower, served int8 | Row lookup and a normalized weighted mean. No neural network |
| `nano` | A 34,540,672-parameter transformer student, distilled against the same tower | One small forward pass |
| Stella query | The tower's own query path, with its `s2p_query` prompt | One 400M forward pass |

BM25 appears throughout as a comparator and as the lexical half of the fused systems. It is not a
fourth query encoder: it scores against its own inverted index and emits nothing into the dense
space. Section 5.3 reports what changes when a fused operator replaces a dense one.

The two tiers we built never saw the document tower's weights change. Their training target is the
tower's own query-side output, so their output lands in a space the index already indexes.

## 3. What a Swap Fixes, and Where It Is Not Free

A swap is legal when the replacement produces vectors the index can score: same dimensionality,
same normalization, same similarity function. Both tiers meet that by construction, and we verify
it rather than assume it. The lookup table's served path agrees with its numpy reference to
4.470e-08. The student's served path agrees with its reference at a minimum cosine of 1.0 and a
maximum comparison error of 1.1548399925231934e-07. The Stella query tower, run through the published
document graph with its prompt, reproduces the torch query path at minimum cosine 1.00000000.

Four things do not come along for free, and all four fail silently.

**The prompt.** Stella as a query encoder needs its `s2p_query` prefix. Without it the vector still
has the right shape and the right norm, and it scores at cosine 0.80 against the correct one.
The lookup table and the student need no prefix, so a serving layer that adds prompts by model type
gets one of the three wrong.

**Tokenization.** Stella's shipped tokenizer pads to 512 by default. Left alone it feeds roughly 500
PAD rows into the encoder, and cosine against the correct vector falls to 0.35.

**Dtype.** FastEmbed's mean pooling promoted output to float64 for models of the student's shape,
while the lookup table and the document tower pool inside ONNX and stay float32. Two query paths
against one index did not share a vector width until we fixed it upstream.

**The operator.** Swapping a dense tier for a fused one changes the retrieval operator, not the
encoder. Section 5 reports what that costs and how much of it depends on prefetch depth.

## 4. Protocol

The headline partition is `clean-4`: nfcorpus, scidocs, scifact, and trec-covid, the four of our six
BEIR datasets with no disclosed teacher overlap. Stella discloses exposure to ArguAna and FiQA. We
registered `clean-4` as the headline before any six-set number existed, we report all six beside it
in every table, and we report the difference between the two partitions as its own row. Datasets
were never added to or removed from the headline after a number was seen.

Two confirmatory families ran, under two different registered procedures, and we keep them apart.
The student's family (Section 5.1) tests four contrasts in a fixed sequence, each at a one-sided
alpha of 0.025, and its decision statistic is the one-sided 2.5% lower bound. The table's family
(Section 5.2) tests three contrasts with sign-flip p-values under Holm control across the family.

**Established** means the contrast satisfied its whole registered rule, which for the table's family
includes the Holm threshold and not only the interval. **Unresolved** means it did not, for any
reason. Unresolved is not equivalence. We computed no equivalence interval for any contrast in this
paper, so nothing here supports a claim that two systems match.

Quality numbers come from exact search. Approximate-search recall never enters a quality comparison;
it appears only in Section 6, where the question is deployment behavior.

## 5. Retrieval Quality

### 5.1 The Transformer Tier

| Contrast | Partition | Delta nDCG@10 | One-sided 2.5% lower bound | Sign-flip p | Verdict |
|---|---|---:|---:|---:|---|
| Nano minus bge-small | clean-4 | +0.017648 | +0.003674 | 0.006410 | Established |
| Nano minus bge-small | all six | +0.027449 | +0.017271 | 9.99990000099999e-06 | Established |
| Nano minus LEAF-asym | all six | +0.016181 | +0.006504 | 0.000540 | Established |
| Nano minus LEAF-asym | clean-4 | -0.001063 | -0.014456 | 0.559474 | Unresolved |

Three of the four contrasts resolve in the student's favor. The fourth does not, and the honest
reading is that on the four datasets with no disclosed teacher exposure we cannot separate a 34.5M
student trained against Stella from a 23M student trained against a different teacher. The student's
weakest dataset against LEAF is trec-covid, at -0.042982, and a macro average hides that.

Per-dataset nDCG@10 for the student: nfcorpus 0.363080, scidocs 0.217710, scifact 0.721097,
trec-covid 0.787116, ArguAna 0.623296, FiQA 0.477765. The last two carry Stella's disclosed exposure
and we mark them wherever they appear.

### 5.2 The Lookup-Table Tier

The table reaches 0.4339 average nDCG@10 over the six, against 0.5744 for the symmetric teacher.
Retention is 0.755. Three registered contrasts:

| Contrast | Delta | 95% CI | Sign-flip p | Verdict |
|---|---:|---|---:|---|
| Table against LightRetriever's 0.4583 bar | -0.0243 | [-0.0405, -0.0086] | 0.997 | Resolved below the bar |
| Table against BM25's 0.4174 | +0.0165 | [+0.0017, +0.0311] | 0.0149 against Holm 0.0083 | Unresolved |
| Fused system against OpenSearch's 0.4868 | +0.0043 | [-0.0063, +0.0151] | 0.219 | Unresolved |

The second row is the one worth dwelling on. The confidence interval excludes zero, and the
familywise correction still refuses the claim. We report it as unresolved because that is what the
registered rule says, and a paper that keeps the interval and drops the correction is choosing its
rule after seeing the number.

On `clean-4` the picture is worse and we report it: the table sits below BM25 by 0.0311
[-0.0517, -0.0109]. The system's two strongest datasets, ArguAna at 0.5916 and FiQA at 0.3728, are
exactly the two Stella discloses.

### 5.3 Fusion, and What the Registered Depth Cost Us

Fusing the table with BM25 lifts the six-set average to 0.4911, which is +0.057 over the dense table
alone, carried by trec-covid at +0.153 and scifact at +0.097. That fused system descriptively tops
every comparator in our matrix, including OpenSearch at 0.4868, while its query side remains a
lookup table and a term count.

The fusion operator that produced 0.4911 is `convex0` with a dev-fitted weight, and no shipping
Qdrant operator reproduces it. The deployable substitute is Qdrant's distribution-based score fusion
at prefetch depth 100, with zero fitted parameters. The two split differently across partitions:

| Operator | All six | clean-4 |
|---|---:|---:|
| DBSF at prefetch 100, zero fitted parameters | 0.4887 | 0.4912 |
| convex0 at depth 1000, dev-fitted, not available in Qdrant | 0.4911 | 0.4866 |

Neither operator is established over the other, and we computed no equivalence interval for the
pair, so the paper takes no position on which is better. The two rows are descriptive. The
deployable recommendation is the parameter-free operator, which is a product choice and not a
measured superiority.

DBSF increases monotonically with prefetch depth: 0.4660, 0.4849, 0.4887, and 0.4898 on all six, and
0.4625, 0.4856, 0.4912, and 0.4974 on `clean-4`, at depths 10, 50, 100, and 1000. **The registered
depth cost us the best number in the table.** DBSF at depth 1000 on `clean-4` reaches 0.4974, which
is 0.0062 above the registered headline. We fixed depth 100 in advance on a realism argument, before
any six-set access, and we report the deeper number rather than quietly keeping the one we
registered. At depth 10 the ranking between the three fusion operators inverts outright, so prefetch
depth is a real parameter of a fused swap and not an implementation detail.

### 5.4 Breadth `[M20]`

The reserved four and the BEIR-15 descriptive validation are running. Rows and their caveats go
here, including FEVER's double-contamination disclosure. This section adds breadth; it does not
reinterpret the registered gates above.

## 6. Deployment Cost

### 6.1 Query Encoders in Isolation

One protocol for all three: three fresh processes per model, batch size one, four CPU threads, five
warmups, 20 synthetic 20-word queries, medians over three trials.

| Model | Hydration | First query | Warm p50 | Peak RSS | Assets |
|---|---:|---:|---:|---:|---:|
| `zero` | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| `nano` | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

This is where a near-zero-compute claim looks strongest. The table answers a query in a tenth of a
millisecond, and the two transformers need about 65 times longer.

### 6.2 The Same Encoders Inside a System

Now add the search. On an Apple M5 Pro against a synthetic one-million-vector index, four threads,
default `ef`, the two paths compare like this:

| Query length | `zero`, encode + search | `nano`, encode + search | Ratio |
|---|---|---|---:|
| 6 to 10 words | 0.234 + 0.845 = 1.09 ms | 1.173 + 0.935 = 2.13 ms | 1.96x |
| 21 to 50 words | 0.566 + 0.805 = 1.37 ms | 3.201 + 0.942 = 4.14 ms | 3.02x |
| 51 to 120 words | 0.696 + 0.775 = 1.46 ms | 6.451 + 1.010 = 7.46 ms | 5.10x |

Two conditions govern the ratio. Search is a floor of about 0.85 ms that both paths pay, so at
typical query length the encoders' 50-fold difference becomes a 1.96-fold system difference. Query
length then moves it: a table lookup is linear in tokens and a transformer is not, so the gap reaches
5.10 times at 51 to 120 words. A separate sweep over index configurations and memory limits, at 6 to
10-word queries, puts the ratio between 1.11 and 3.28 times, and under memory pressure the two paths
converge further.

This is the most useful negative result in the paper. The tenth-of-a-millisecond figure is the one a
reader will quote, and any claim about query-side cost has to name a query length and an index.

### 6.3 The Cheap Encoder Is the Big Artifact

The int8 lookup table occupies 270.1 MB and loads in 0.51 s. The student's fp16 ONNX graph occupies
46.1 MB and loads in 0.17 s. The tier with no query-time neural network is 5.9 times larger on disk
and three times slower to hydrate. A lookup table moves cost from compute to storage; it does not
remove it.

### 6.4 Quantization Is a Precondition

A one-million-document index serves inside a 256 MB container, and only under binary quantization.
With binary quantization and rescoring disabled the whole system answers in 3.387 ms with the table
and 4.469 ms with the student, using 202 MB. The uncompressed fp16 index technically answers at
every tested limit and is unusable at all of them: 532 ms at 256 MB, and still 225 ms with 2 GB.
Binary quantization is 16 times smaller than the originals and also the fastest configuration
measured, so on this hardware it costs nothing in latency or footprint. What it costs in retrieval
quality is not measured here. This experiment records compression, feasibility, and latency, and no
quality claim follows from it. The comparison this section needs, TurboQuant against binary, int8,
and fp16 on latency, footprint, and recall at up to one million documents, is deferred to this paper
and has not yet run. `[deferred measurement]`

## 7. Two Results That Transfer

### 7.1 A Teacher's Own Quality Carried No Signal About Its Distilled Table

We measured 11 teachers in two sweeps and compared each teacher's own retrieval ceiling against the
quality of the table distilled from it. Over the eight candidates of the first sweep the Spearman
correlation between the two is 0.000. The table distilled from the highest-ceiling candidate ranked
fifth on the metric that ships, landed 0.0480 [-0.0608, -0.0349] below the incumbent's table, and we
withdrew that teacher the same day we approved it. Only one candidate beat the incumbent, at +0.0365 [0.0249, 0.0481], and it became the
document tower this paper freezes.

Two related mechanisms failed the same way. Pooling does not explain the ranking: the same weights
read out as a mean move the ratio from 0.526 to 0.472. Cosine agreement with the teacher does not
either: it rises with the distillation weight while nDCG@10 falls, so it mis-ranks candidates.

The sweep is closed-form, flat, dev-only, and scored on two components of one dataset family
against each teacher's own documents, so it ranks candidates rather than predicting their scores.
Within that scope the reading is a counterexample and not a law. Eight observations in one
distillation setting cannot show that teacher quality has no predictive value across architectures,
student classes, or training regimes. They do show that in this setting it had none, which is enough
to make leaderboard-order teacher selection an unjustified shortcut. The only reliable signal we
found is distilling the candidate and measuring the student. For a lookup-table student that is cheap, which is what makes a sweep of
this width affordable at all.

### 7.2 A Lookup Table Absorbs Its Own Post-Processing, Under One Condition

The architecture is a row lookup, a weighted mean over the query's token multiset, and an L2
normalization. Under that pooling, the standard post-processing stack is exactly absorbable into the
rows, because a mean commutes with the transformation: `mean(W[t] - mu) = mean(W[t]) - mu` for every
multiset. We checked each transformation numerically against an explicitly reconstructed table, on
ragged multisets with repeats, at vocabulary 500 and 64 dimensions:

| Transformation applied after pooling | Table that reproduces it | Maximum absolute difference |
|---|---|---:|
| Centering | `W' = W - mu` | 1.67e-16 |
| Whitening or any linear map | `W' = W A^T` | 3.33e-16 |
| Top principal component removal after centering | `W' = (W - mu) P^T` | 3.33e-16 |
| Per-token scalar weights, such as IDF or SIF | `W' = c_t W_t` | 9.31e-14 |
| The whole SIF recipe at once | `W' = c_t (W - mu) P^T` | 2.82e-14 |

The consequence is a ceiling argument. None of these levers can raise what the architecture can
reach, because a trained table could already have represented the result. They can still help as a
prior or an initialization, which is a much weaker claim than the one the lever list started with.

**The condition matters and we state it.** Absorbability of an affine transformation needs pooling
whose token coefficients sum to one, which mean and weighted-mean pooling satisfy. Under sum pooling
it fails: replacing each row by `A e(t) + b` yields `Aq + nb` rather than `Aq + b`, so the offset
scales with query length. Per-token weighting additionally needs weights fixed by token identity
alone. Pure linear maps carry no such condition.

Two things are genuinely outside the architecture's reach, and one is a decoy. Count saturation,
which reads each token once regardless of repeats, depends on a query's multiplicity vector while a
row is shared across all queries; it differs from the plain mean by 0.129 and no choice of rows
fixes it. An n-gram or phrase row adds a feature no unigram bag can express, because two queries
with the same multiset in a different order are identical to a unigram table. The decoy is
length-dependent scaling: any positive scalar function of query length is removed by the final L2
normalization, so it is a no-op rather than a lever.

### 7.3 How Much a Development Macro Overstates

Our development macro read 0.6153. The six-set result was 0.4339. Out-of-domain development
retention was 0.764 against a six-set retention of 0.755, while the all-six development retention
read 0.915. The development suite's in-distribution bias was real, we measured it in advance, and
the out-of-domain subset predicted the held-out result to within 0.009 retention while the full
development macro overstated it by 0.16.

## 8. Limitations

**Contamination.** Stella discloses exposure to ArguAna, FiQA, and FEVER. `clean-4` excludes the
first two. The all-six-minus-clean-4 difference is a partition sensitivity, not a causal estimate of
contamination.

**Unresolved is not equivalence.** Three confirmatory contrasts fail to resolve: the student against
LEAF-asym on clean-4, the table against BM25, and the fused system against OpenSearch. We computed
no equivalence interval for any of them, and the descriptive comparison between the two fusion
operators has no interval at all. None of these four comparisons supports a claim that two systems
match.

**Interval scope.** The reported intervals come from query resampling. They exclude
training-seed variation, so they understate total uncertainty for any claim about a recipe rather
than about a specific trained artifact.

**Coverage against capacity.** An earlier student retained 93.8% on Wikipedia-style questions and
50.1% on a programming forum with the same parameter budget. That pattern points at training-data
coverage rather than model capacity, and our evidence does not separate the two cleanly.

**Swap scale.** The runtime swap is demonstrated against one small in-process collection. The
quality tables come from offline exact search over the same frozen vectors. We have not measured a
swap against a large persistent collection in one server lifetime. `[open, see HOTSWAP.md]`

**One teacher.** Every result here holds the same document tower fixed. Whether the frontier's shape
survives a different frozen tower is untested.

## 9. Reproducibility

Artifacts, revisions, hashes, and harness commands. The student is published at
`DylanCouzon/constella-nano`, weights frozen at revision
`6bb167dc6f60d3992602235b8e8aaa374a309168`. The table and the document tower ship alongside it.
The student saw exactly 199,999,721 training examples; this is not 200,000,000 and we do not round it.

Training sources permit commercial derived weights. MS MARCO is excluded from training in every
role and appears only as a validation diagnostic. This matters for reading Section 5: the neural
comparators there train on MS MARCO and our two tiers do not, so any comparison against them carries
a training-exposure asymmetry in the comparators' favor. BM25 trains on nothing and is unaffected.

## Appendix A: Avenues That Closed

One line each, for readers who want to know what was tried.

- A probe milestone measured eight levers against the lookup table's quality gap and none moved the
  development endpoint by more than about 0.005. The gap is architectural, not a tuning deficit.
- Extending the table's vocabulary produced five arms between 0.021 and 0.035 nDCG@10 below the
  untrained baseline. The inversion was never diagnosed and the released table stands.
- A domain-specialized table passed its dense gate at +0.007591 and failed its fused gate at
  +0.002953. No specialized table shipped.
- A deterministic short-query variant closed inconclusive under a label-sensitive metric.

