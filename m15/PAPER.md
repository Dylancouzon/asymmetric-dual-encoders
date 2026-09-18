# Swapping the Query Encoder Over a Frozen Document Index

**Draft v2, 2026-09-17. Not for circulation.** Sections marked `[M20]` wait on the reserved four and
BEIR-15. Sections marked `[open]` name a measurement this paper owes. Numbers trace to committed
result files through `EVIDENCE_INDEX.md`, whose spot-check table records what a reviewer verified
against the source and when.

## Abstract

A dual-encoder retrieval system has one expensive asset and one cheap one. Encoding a corpus costs a
full pass over every document and produces an index that resists change. Encoding a query costs one
forward pass and produces nothing anyone has to store. We hold `stella_en_400M_v5` frozen as the
document tower, 1024 dimensions, and replace only the query side. Three dense query encoders emit
into that one index: a per-token lookup table with no query-time neural network, a
34,540,672-parameter distilled transformer, and the tower's own query path. BM25 joins them as the
lexical half of a fused system.

Running a cheap query encoder against a frozen teacher's index is not new. LEAF ships it, pyNIFE
published the lookup-table construction, and backward-compatible training named the goal years
earlier. What has not been done is the measurement: prior work ships one cheap query encoder per
index, and nobody has put two tiers, the teacher's own tower, and a lexical channel on one index and
priced retention against system cost under a protocol registered before the numbers existed.

The distilled student establishes superiority over bge-small on both the contamination-clean
partition and the full six (+0.017648 and +0.027449 nDCG@10). The lookup table retains 0.755 of the
teacher's quality on the same six at 0.1119 ms warm median query time.

The result that changes how the frontier reads is about cost. Inside one harness, two query paths
that differ by about five times as encoders differ by 1.96 times as whole systems at typical query
length, because both pay the same approximate search. The gap widens to 5.10 times at 51 to 120
words and narrows under memory pressure. Which tier ships the smaller artifact depends on how you
package it, and the two packagings we measured disagree. Below the transformer tier, query-encoder
compute stops deciding system cost; the index and its quantization decide it.

## 1. Introduction

Retrieval systems are evaluated as one model and deployed as two. The document index is built once,
costs a corpus-scale encode, and cannot be rebuilt whenever a better query model appears. The query
encoder runs once per request and holds no state. Replacing it changes no stored bytes.

That property is established. LEAF releases `-asym` checkpoints for exactly this mode and calls it
mixed-checkpoint inference. pyNIFE distills a per-token lookup table against a frozen off-the-shelf
teacher and reuses that teacher's index unchanged. Query-encoder distillation onto a frozen document
encoder appeared in 2023. Backward-compatible and forward-compatible training named the general goal
of upgrading a model without re-embedding a corpus. We build on all of it and claim none of it.

The gap is narrower and it is about measurement. Prior work ships one cheap query encoder per index.
Nobody has measured several query encoders on one held-fixed index, across a near-zero-compute tier,
a distilled transformer tier, the teacher's own tower, and a lexical channel, and priced what each
retains against what it costs in a running system, under a protocol registered before the numbers
arrived. Two questions fall out of that setting and out of no single-encoder study:

- How much quality does each tier retain from the frozen tower it shares?
- At what point does query-encoder compute stop deciding system cost?

**Contributions.**

1. A retention-against-cost frontier over one frozen index, with a near-zero-compute tier, a
   sub-35M transformer tier, and a lexical channel on the same document vectors (Sections 5, 6).
2. Registered head-to-head results for the transformer tier against bge-small and LEAF-asym, with
   the unresolved contrast reported as unresolved and the document-tower confound disclosed (5.1).
3. Evidence that the encoder-level cost advantage of a lookup table shrinks to under two times at
   the system level, depends on query length, and does not settle which tier ships the smaller
   artifact (Section 6).
4. Three method results with reach beyond this system: an out-of-domain development subset predicted
   held-out retention within 0.009 while the full development macro overstated it by 0.16; across
   eight distilled teachers, a teacher's own retrieval quality carried no signal about its table; and
   a named class of post-hoc transformations is exactly absorbable into a lookup table under mean
   pooling (Section 7).

## 2. Setting

**The frozen side.** One document tower, `stella_en_400M_v5`, 1024-dimensional, L2-normalized,
cosine similarity. Every dense result in this paper searches the same document vectors. The tower's
CUDA and CPU paths agree to a minimum cosine of 1.000000 and a maximum absolute difference of
9.07e-05, so an index built on one device holds on the other.

The choice of tower sets a standing cost the query side cannot undo. Per one million documents at
fp16, a 1024-dimensional index is about 2.05 GB. The comparators sit at 0.77 GB for bge-small's
384 dimensions, about 1.4 GB for OpenSearch's doc-side postings, 1.54 GB for LEAF and arctic-m at
768 dimensions, and 3.07 GB for LightRetriever at 1536. A query encoder that is free still rides an
index that is not.

**The swappable side.** Three dense query encoders emit 1024-dimensional vectors into that index:

| Tier | What it is | Query-time compute |
|---|---|---|
| `zero` | A per-token lookup table distilled against the frozen tower, served int8 | Row lookup and a normalized weighted mean. No neural network |
| `nano` | A 34,540,672-parameter transformer student, distilled against the same tower | One small forward pass |
| Stella query | The tower's own query path, with its `s2p_query` prompt | One 400M forward pass |

BM25 appears as a comparator and as the lexical half of the fused systems. It is not a fourth query
encoder: it scores against its own inverted index and emits nothing into the dense space.

Neither tier we trained ever changed the document tower. Both regress onto the tower's own
query-side output, so their vectors land in the space the index already indexes.

## 3. The Swap Contract

A swap is legal when the replacement emits vectors the index can score: same dimensionality, same
normalization, same similarity. We verify rather than assume it. The table's served path agrees with
its numpy reference to 4.470e-08. The student's served path agrees with its reference at a minimum
cosine of 1.0 and a maximum comparison error of 1.1548399925231934e-07. The Stella query tower, run
through the published document graph with its prompt, reproduces the torch query path at a minimum
cosine of 1.00000000.

Four things do not travel with the swap, and all four fail silently rather than loudly.

**The prompt.** Stella as a query encoder needs its `s2p_query` prefix. Without it the vector keeps
the right shape and the right norm and scores at cosine 0.80 against the correct one. The table and
the student need no prefix, so a serving layer that adds prompts by model family gets one of the
three wrong. FastEmbed's own `query_embed` does not add it.

**Tokenization.** Stella's shipped tokenizer pads to 512 by default. Left alone it feeds roughly 500
PAD rows into the encoder and cosine against the correct vector falls to 0.35.

**Dtype.** FastEmbed's mean pooling promoted output to float64 for models of the student's shape,
while the table and the document tower pool inside ONNX and stay float32. Two query paths against
one index did not share a vector width until we fixed it upstream.

**The operator.** Replacing a dense tier with a fused one changes the retrieval operator rather than
the encoder, and brings a prefetch depth with it. Section 5.3 prices that.

## 4. Protocol

The headline partition is `clean-4`: nfcorpus, scidocs, scifact, and trec-covid, the four of our six
BEIR datasets with no disclosed teacher overlap. Stella discloses exposure to ArguAna, FiQA, and
FEVER. We report all six beside `clean-4` in every table and report the difference between the two
partitions as its own row. No dataset entered or left the headline after a number was seen.

**Registration dates differ by family and we state them.** The partition itself was pre-registered
for both families, in `m7/LEDGER.md`, on the contamination argument alone. For the student's family
the headline designation also predates every six-set number. For the table's family, whose six-set
confirmatory run completed on 2026-08-28, `clean-4` was pre-registered as an exposure-restricted
descriptive analysis and was designated the headline afterwards. It moved against the table, which
is the direction that makes the designation credible rather than convenient, and we report both
partitions everywhere so a reader can apply either rule.

Two confirmatory families ran under two registered procedures. The student's family tests four
contrasts in a fixed sequence, each at a one-sided alpha of 0.025, with the one-sided 2.5% lower
bound as the decision statistic. The table's family tests three contrasts with sign-flip p-values
under Holm control.

**Established** means a contrast satisfied its whole registered rule, which for the table's family
includes the Holm threshold and not only the interval. **Unresolved** means it did not, for any
reason. Unresolved is not equivalence. We computed no equivalence interval anywhere in this paper,
so nothing here supports a claim that two systems match.

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

Per-dataset nDCG@10 for the student: nfcorpus 0.363080, scidocs 0.217710, scifact 0.721097,
trec-covid 0.787116, ArguAna 0.623296, FiQA 0.477765. The last two carry Stella's disclosed exposure.

**The LEAF rows are a system comparison, not a shared-index comparison, and that changes the
reading.** LEAF-asym encodes its documents with `snowflake-arctic-embed-m-v1.5`, 109M parameters at
768 dimensions, and its queries with the 23M `mdbr-leaf-ir`. Our student sits on a 400M tower at
1024 dimensions. The two document towers are not the same and the comparison prices whole systems.
On our six, Stella's symmetric ceiling is 0.5744 and arctic-m-v1.5's is 0.5264, a gap of 0.048. The
established all-six delta is +0.016181, a third of the gap between the towers. Read as retention of
its own frozen tower, LEAF-asym keeps 0.979 and our student keeps 0.9256 (derived from the six rows
above against 0.5744). A 400M tower and a 50% larger student win on the full six and fail to separate
on `clean-4`, and the distillation-efficiency comparison points the other way. We report the
contrast because it was registered and we report what it does not show.

The student's weakest dataset against LEAF is trec-covid, at -0.042982. A macro average hides that.

### 5.2 The Lookup-Table Tier

The table reaches 0.4339 average nDCG@10 over the six against the symmetric teacher's 0.5744, a
retention of 0.755. Three registered contrasts:

| Contrast | Delta | 95% CI | Sign-flip p | Verdict |
|---|---:|---|---:|---|
| Table against LightRetriever's 0.4583 bar | -0.0243 | [-0.0405, -0.0086] | 0.997 | Resolved below the bar |
| Table against BM25's 0.4174 | +0.0165 | [+0.0017, +0.0311] | 0.0149 against Holm 0.0083 | Unresolved |
| Fused system against OpenSearch's 0.4868 | +0.0043 | [-0.0063, +0.0151] | 0.219 | Unresolved |

The second row is the one to dwell on. Its confidence interval excludes zero and the familywise
correction still refuses the claim. We report it as unresolved because that is what the registered
rule says. A paper that keeps the interval and drops the correction is choosing its rule after
seeing the number.

On `clean-4` the picture is worse and we report it: the table sits 0.0311 [-0.0517, -0.0109] below
BM25. The system's two strongest datasets, ArguAna at 0.5916 and FiQA at 0.3728, are exactly the two
Stella discloses.

### 5.3 The Lexical Channel Is Worth Ten Times the Query-Side Levers

Fusing the table with BM25 lifts the six-set average to 0.4911, which is +0.057 over the dense table
alone, carried by trec-covid at +0.153 and scifact at +0.097. Put that beside Section 7.4 and
Appendix A: every lever we measured on the table itself moved the development endpoint by under
0.005. The lexical channel is worth an order of magnitude more than the query-side tuning that
produced the table, and a zero-compute product should ship fused.

The result worth stating plainly: a frozen third-party dense index, a lookup table, and term counts
reach 0.4887 on all six and 0.4912 on `clean-4` with stock Qdrant fusion and no fitted parameters.
That is the band of OpenSearch's inference-free sparse system at 0.4868, which needs a 133M-parameter
document encoder and about 1.4 GB of postings per million documents, and it is above LightRetriever
hybrid per-task at 0.4720. The query side is a table lookup and a term count.

The operator that produced 0.4911 is `convex0` with a dev-fitted weight, and no shipping Qdrant
operator reproduces it. The deployable substitute is distribution-based score fusion at prefetch
depth 100 with zero fitted parameters:

| Operator | All six | clean-4 |
|---|---:|---:|
| DBSF at prefetch 100, zero fitted parameters | 0.4887 | 0.4912 |
| convex0 at depth 1000, dev-fitted, not available in Qdrant | 0.4911 | 0.4866 |

Neither is established over the other, and we computed no equivalence interval, so the paper takes
no position on which is better. Choosing the parameter-free operator is a product decision.

DBSF increases monotonically with prefetch depth: 0.4660, 0.4849, 0.4887, and 0.4898 on all six, and
0.4625, 0.4856, 0.4912, and 0.4974 on `clean-4`, at depths 10, 50, 100, and 1000. **The registered
depth cost us the best number in the table.** Depth 1000 on `clean-4` reaches 0.4974, 0.0062 above
the registered headline. We fixed depth 100 in advance on a realism argument, before any six-set
access, and we report the deeper number rather than quietly keeping the registered one. On the
development suite, where four components allow a wider sweep, the ranking between the three fusion
operators inverts at depth 10; that inversion is a development-set observation and no six-set
evidence supports or contradicts it.

### 5.4 The Frontier

Retention is each system's six-set macro divided by its own frozen document tower's symmetric
ceiling. Cells marked *derived* are arithmetic over committed rows rather than a committed
aggregate, and they carry no interval.

| Query side | Frozen tower | Six-set macro | Retention | Warm p50, 20 words |
|---|---|---:|---:|---:|
| Stella query tower | Stella 400M | 0.5744 | 1.000 | unmeasured `[open]` |
| `nano`, 34.5M | Stella 400M | 0.5317 *derived* | 0.9256 *derived* | 7.2511 ms |
| bge-small | its own | 0.5042 | — | 6.8400 ms |
| `mdbr-leaf-ir`, 23M | arctic-m-v1.5 109M | 0.5155 | 0.979 | unmeasured here |
| `zero` table | Stella 400M | 0.4339 | 0.755 | 0.1119 ms |
| `zero` + BM25, DBSF at 100 | Stella 400M | 0.4887 | — | 0.1119 ms plus lexical |
| BM25 | none | 0.4174 | — | no neural network |

The shape is the paper. Dropping from a 400M query tower to a 34.5M student costs 0.074 retention.
Dropping from the student to a table costs another 0.171 and saves about seven milliseconds of
encoder time, which Section 6 shows is worth about one millisecond of system time at typical query
length. Adding a term count to the table recovers most of the drop at no query-side compute.

### 5.5 Breadth `[M20]`

The reserved four and the BEIR-15 descriptive validation are running. Rows and caveats go here,
including FEVER's double-contamination disclosure. This section adds breadth and does not
reinterpret the registered gates above.

## 6. Deployment Cost

### 6.1 Query Encoders in Isolation

One protocol for three models: three fresh processes per model, batch size one, four CPU threads,
five warmups, 20 synthetic 20-word queries, medians over three trials.

| Model | Hydration | First query | Warm p50 | Peak RSS | Served assets |
|---|---:|---:|---:|---:|---:|
| `zero` | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| `nano` | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

This is where a near-zero-compute claim looks strongest: 61 to 65 times faster than the two
transformer tiers. The Stella query tower is absent because we never measured it under this protocol,
which is the gap Section 9 names.

### 6.2 The Same Encoders Inside a System

Add the search. On an Apple M5 Pro against a **synthetic** one-million-vector index of random
1024-dimensional rows, four threads, default `ef`, latency and architecture only:

| Query length | `zero`, encode + search | `nano`, encode + search | Ratio |
|---|---|---|---:|
| 6 to 10 words | 0.234 + 0.845 = 1.09 ms | 1.173 + 0.935 = 2.13 ms | 1.96x |
| 21 to 50 words | 0.566 + 0.805 = 1.37 ms | 3.201 + 0.942 = 4.14 ms | 3.02x |
| 51 to 120 words | 0.696 + 0.775 = 1.46 ms | 6.451 + 1.010 = 7.46 ms | 5.10x |

Inside this one harness the encoders differ by about five times, not the 61 to 65 of Section 6.1,
because the two protocols measure different builds on different hardware. The collapse to report is
the same-harness one: **five times as encoders becomes 1.96 times as systems** at typical query
length, because both paths pay the same 0.85 ms of approximate search. Query length then moves it,
because a table lookup is linear in tokens and a transformer is not.

A separate sweep over index configurations and hard memory limits, at 6 to 10-word queries, puts the
ratio between 1.11 and 3.28 times; under memory pressure the paths converge further. The 1.11 end
comes from a 3.366 ms against 3.732 ms pair inside Docker, and the source flags sub-millisecond
Docker differences as noise, so treat that bound as soft.

Two conditions therefore govern any query-side cost claim: the query length, and the index it runs
against. The tenth-of-a-millisecond figure is the one a reader will quote, and on its own it is not
a system claim.

### 6.3 Which Tier Is the Bigger Artifact Depends on the Packaging

The two packagings we measured disagree, and we report both rather than pick the flattering one.

| Packaging | `zero` | Student | Ordering |
|---|---:|---:|---|
| Served bundle under the common protocol (Section 6.1) | 90.1 MiB | 132.3 MiB | The table is smaller |
| Edge artifacts in the M9 prototype: int8 token table against fp16 ONNX | 270.1 MB | 46.1 MB | The table is 5.9 times larger |

The release also lists a 94 MB numpy reference and a roughly 31 MB int8 ONNX path for the table, so
there is no single canonical asset size for it. The rows above measure different things and come
from different milestones, and reconciling them under one packaging definition is open work. What
survives both readings is the mechanism: a lookup table moves cost from compute to storage, and on
an edge device storage is the scarce resource.

### 6.4 Quantization Is a Precondition, and Rescoring Is a Trap

A one-million-document index serves inside a 256 MB container, and only under binary quantization.
With rescoring disabled the whole system answers in 3.387 ms with the table and 4.469 ms with the
student, using 202 MB. The uncompressed fp16 index answers at every tested limit and is unusable at
all of them: 532 ms at 256 MB, and still 225 ms with 2 GB. Binary quantization is 16 times smaller
than the originals and also the fastest configuration measured.

Rescoring against memory-mapped originals costs 8.467 ms against 0.441 ms without it, 19 times, so
it is a trap rather than a tuning choice on this hardware.

What none of this measures is retrieval quality. These runs record compression, feasibility, and
latency on a synthetic index, so no quality claim follows. The comparison the section needs,
TurboQuant against binary, int8, and fp16 on latency, footprint, and recall at up to one million
documents, is deferred to this paper and has not yet run. `[open]`

## 7. Results That Transfer

### 7.1 What a Development Macro Overstates, and What Predicts the Held-Out Number

Our development macro read 0.6153. The six-set result was 0.4339. The full development macro
therefore overstated held-out retention by 0.16, reading 0.915 against an eventual 0.755. The
out-of-domain subset of the same development suite read 0.764 and missed the held-out retention by
0.009.

We measured and disclosed that bias before the held-out run, so this is a prediction rather than a
postmortem. For anyone distilling a query encoder, the operational form is short: the out-of-domain
slice of a development suite is worth more than the whole of it, and the whole of it will flatter a
lookup table by roughly the margin above. One system, one suite, so treat 0.16 as an instance and
0.009 as an existence proof that a cheap in-house predictor of held-out retention is available.

### 7.2 A Teacher's Own Quality Carried No Signal About Its Distilled Table

We measured 11 teachers across two sweeps and compared each teacher's own retrieval ceiling against
the quality of the table distilled from it. Over the eight candidates of the first sweep the Spearman
correlation between the two is 0.000. The table distilled from the highest-ceiling candidate ranked
fifth on the metric that ships, landed 0.0480 [-0.0608, -0.0349] below the incumbent's table, and we
withdrew that teacher the same day we approved it. One candidate beat the incumbent, at +0.0365
[0.0249, 0.0481], and it became the tower this paper freezes.

Two plausible mechanisms failed the same way. Pooling does not explain the ranking: the same weights
read out as a mean move the ratio from 0.526 to 0.472. Cosine agreement with the teacher does not
either: it rises with the distillation weight while nDCG@10 falls, so it mis-ranks candidates.

The sweep is closed-form, flat, development-only, and scored on two components of one dataset family
against each teacher's own documents, so it ranks candidates rather than predicting their scores.
Eight observations in one setting cannot show that teacher quality never predicts student quality,
and the distillation literature already doubts the link. What they do show is that in this setting it
had none, which makes leaderboard-order teacher selection an unjustified shortcut and makes
distilling the candidate the only signal we trust.

### 7.3 A Lookup Table Absorbs Its Own Post-Processing, Under One Condition

The architecture is a row lookup, a weighted mean over the query's token multiset, and an L2
normalization. Under that pooling the standard post-processing stack is exactly absorbable into the
rows, because the mean commutes with the transformation: `mean(W[t] - mu) = mean(W[t]) - mu` for
every multiset. Checked numerically against an explicitly reconstructed table, on ragged multisets
with repeats, at vocabulary 500 and 64 dimensions:

| Transformation applied after pooling | Table that reproduces it | Maximum absolute difference |
|---|---|---:|
| Centering | `W' = W - mu` | 1.67e-16 |
| Whitening or any linear map | `W' = W A^T` | 3.33e-16 |
| Top principal component removal after centering | `W' = (W - mu) P^T` | 3.33e-16 |
| Per-token scalar weights, such as IDF or SIF | `W' = c_t W_t` | 9.31e-14 |
| The whole SIF recipe at once | `W' = c_t (W - mu) P^T` | 2.82e-14 |

None of these levers can raise what the architecture reaches, because a trained table could already
have represented the result. They can still help as a prior or an initialization.

**The condition matters.** Absorbing an affine transformation needs pooling whose token coefficients
sum to one, which mean and weighted-mean pooling satisfy. Under sum pooling it fails: replacing each
row by `A e(t) + b` yields `Aq + nb` rather than `Aq + b`, so the offset scales with query length.
Per-token weighting additionally needs weights fixed by token identity alone. Pure linear maps carry
no condition.

Two things sit outside the architecture's reach and one is a decoy. Count saturation, which reads
each token once regardless of repeats, depends on a query's multiplicity vector while a row is shared
across queries; it differs from the plain mean by 0.129 and no choice of rows fixes it. An n-gram row
adds a feature no unigram bag can express, because two queries with the same multiset in a different
order are identical to a unigram table. The decoy is length-dependent scaling: any positive scalar
function of query length is removed by the final L2 normalization, so it is a no-op.

### 7.4 The Distillation Objective Was Already Exhausted

The shipped objective is inert on its own training data. Median KL against a uniform bank is
4.73e-07 nats, and the table already ranks the positive first for 99.75% of training queries. The
loss has nothing left to teach, which is a property of the bank rather than of the KL class: the same
objective against a teacher's top-200 bank measures 0.777 nats.

This reframes a negative result. A lookup table that plateaus may have an exhausted objective rather
than an exhausted architecture, and the two look identical from the outside. The diagnostic is cheap:
measure how often the objective still separates the positive.

A second diagnostic in the same family warns against acting on a correlation. The table falls 0.050
nDCG@10 behind the teacher per additional subword per word (t = 4.61), which reads like a mechanism.
Moving fertility by 0.164 to 0.176 did not move the metric. A correlated channel is not a lever, and
this is the cleanest instance of that we produced.

## 8. Limitations

**Contamination.** Stella discloses exposure to ArguAna, FiQA, and FEVER. `clean-4` excludes the
first two. The difference between the partitions is a sensitivity, not a causal contamination
estimate. The distillation targets come from Stella, so any exposure in its lineage reaches our tiers
through the teacher even though neither tier trains on those corpora directly.

**Unresolved is not equivalence.** Three confirmatory contrasts fail to resolve: the student against
LEAF-asym on `clean-4`, the table against BM25, and the fused system against OpenSearch. We computed
no equivalence interval for any of them, and the descriptive comparison between the two fusion
operators has no interval at all.

**Interval scope.** Intervals come from query resampling. They exclude training-seed variation, so
they understate uncertainty for any claim about a recipe rather than a specific trained artifact.

**Synthetic index.** Every number in Sections 6.2 and 6.4 comes from an index of random
1024-dimensional vectors. Approximate search on random vectors is not a proxy for a real embedding
distribution, and no recall is reported for any configuration.

**Cross-harness comparison.** Sections 6.1 and 6.2 measure different builds on different hardware.
Ratios are only compared inside a harness.

**Coverage against capacity.** An earlier student retained 93.8% on Wikipedia-style questions and
50.1% on a programming forum at the same parameter budget. That points at training-data coverage
rather than capacity, and our evidence does not separate the two.

**Swap scale.** The runtime swap runs against one small in-process collection. The quality tables
come from offline exact search over the same frozen vectors. No measurement covers a swap against a
large persistent collection in one server lifetime.

**One tower.** Every result holds the same document tower fixed. Whether the frontier's shape
survives a different frozen tower is untested.

## 9. Open Measurements

Three gaps are named above. One experiment closes most of them, and it needs no training and no new
benchmark access.

Build one persistent Qdrant collection from committed Stella document vectors, then query it in one
server lifetime with the Stella query tower, the student, the table, and the table fused with BM25.
Per index configuration, fp16, int8, binary with and without rescoring, and under the 256 MB and
512 MB container limits, record approximate-search recall@10 against exact search, p50 latency per
query-length bucket, and returned point identifiers per encoder.

That run prices the Stella query tower, which is the swap a team with a Stella index would actually
make and the one number the frontier in Section 5.4 is missing. It replaces the synthetic index with
real vectors. It turns the quantization section from a latency claim into a quality-and-latency
claim, which is what the deferred TurboQuant comparison requires. And it demonstrates the swap at a
scale the paper currently asserts.

## 10. Reproducibility

The student is published at `DylanCouzon/constella-nano`, weights frozen at revision
`6bb167dc6f60d3992602235b8e8aaa374a309168`. The table and the document tower ship alongside it. The
student saw exactly 199,999,721 training examples; this is not 200,000,000 and we do not round it.

Training sources permit commercial derived weights. MS MARCO is excluded from training in every role
and appears only as a validation diagnostic. That matters for reading Section 5: the neural
comparators there train on MS MARCO and our two tiers do not, so those comparisons carry a
training-exposure asymmetry in the comparators' favor. BM25 trains on nothing.

## Appendix A: Avenues That Closed

- Twelve probes ran against the table's quality gap and none moved the development endpoint by more
  than about 0.005. The milestone closed on an owner decision not to spend further, so this records
  that no lever tested at equal budget worked, and the strongest remaining lead, a hard-candidate
  listwise objective measuring 0.777 nats where the shipped one measures 4.73e-07, was never run.
- Extending the table's vocabulary produced five arms between 0.021 and 0.035 nDCG@10 below the
  untrained baseline. The inversion was never diagnosed and the released table stands.
- A domain-specialized table passed its dense gate at +0.007591 and failed its fused gate at
  +0.002953. No specialized table shipped.
- A deterministic short-query variant closed inconclusive under a label-sensitive metric.
