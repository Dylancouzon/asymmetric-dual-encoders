# M15 measurement method, E1-E8

Frozen 2026-09-30, before any M15 run. Scope and questions come from `m15/PLAN.md` v5.3. A change to
an estimand, a decision rule or a data role after a run starts is a dated amendment at the end of
this file, and the original text stays in git. Revised the same day, before any run, for Astra's three
P1 findings (`REVIEWS/2026-09-30-astra-measurements-review.md`).

## Rules for every measurement

- **Data roles.** The reserved four (FEVER, DBpedia-entity, cqadupstack-android, cqadupstack-english)
  have no role in any measurement, including fitting, thresholds and diagnostics. Every loader in
  `m15src/` refuses those four names before it opens anything. No script reads
  `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`,
  `work/dev/cqadup-android.json` or `work/dev/cqadup-english.json`. MS MARCO is validation only:
  it is never a gradient, target, negative or generation seed.
- **Six-set access.** E2 and E4 use FiQA, SciFact and NFCorpus, and E8 uses all six M7 sets. They are
  public BEIR members that M20 already scored openly. This file is the registration that the project
  CLAUDE.md rule "no six-set evaluation outside its registered transaction" asks for, and each
  section names the only access it makes. Six-set loads append to `m7/SIX_ACCESS.log`, which is
  committed with the results.
- **Quality numbers come from exact search.** ANN numbers (neighbour recovery, nDCG@10 under ANN,
  latency) describe deployment behavior and carry an "ANN" label in every output field name.
- **Descriptive only.** No result here supports a superiority, equivalence or significance claim.
  Where an interval is given, it is a query-resampling bootstrap (10,000 draws, seed 20260930) and it
  excludes training-seed variation.
- **Machine.** Apple M5 Pro, 24 GB unified memory, 15 cores, macOS, `.venv-mac` (Python with torch
  2.8.0, transformers 4.57.6, onnxruntime 1.29.0, sentence-transformers 5.7.0, bm25s 0.3.11). One MPS
  job at a time with both MPS watermark variables set. CPU latency runs only when no other job runs.
  E8 alone runs on Runpod.
- **Receipts.** Each result is one JSON under `results/m15_*.json`, written atomically. It records
  status, the script path and sha256, the git commit, UTC start and end, the machine and package
  versions, every model repository with revision and file hash, every dataset source with revision,
  the input file hashes and the seed. A script refuses to overwrite an existing result. Scripts
  live in `m15src/`. Bulky intermediates (vectors, runs) go to gitignored `work/m15/`.
- **Encoder identities.** Stella query path: `NovaSearch/stella_en_400M_v5` at `ffeb2b7e`, the
  registered `s2p_query` prompt (`m20src/roster.py:63`), fp32, max length 512, dynamic padding.
  Zero: `DylanCouzon/constella-zero` at `ebee6ea9`, int8 variant, the frozen M7 table. Nano: the
  published constella-nano ONNX at a pinned revision, cast to float32. bge-small: `BAAI/bge-small-en-v1.5`
  at `5c38ec7c` with its query prefix. LEAF: `MongoDB/mdbr-leaf-ir` at `4262131b`, `query` prompt.
  Document vectors: the registered Stella document path, fp32 compute, normalized, stored fp16.
- **Reproduction gate (E2, E4).** Before a dataset's local Stella document vectors enter any result,
  the local exact pipeline must reproduce the committed M20 per-query rows
  (`results/m20_beir15_scores/<dataset>/`) for `stella-query`, `zero-dense` and `nano-dense`: mean
  nDCG@10 within 5e-4 of the committed mean for each system, and at least 98% of queries within
  1e-6. This one check validates the vectors, the Mac query paths and the published Nano ONNX
  together. A failure stops that dataset; its vectors are re-encoded on the Mac and the gate reruns.
- **Before a long run.** Smoke the real path on a small slice, then check the first progress line,
  its rate, host free memory and the log for `Traceback|Error|FAILED|OOM|Killed|assert`.

## E5, oracle headroom between Zero and Nano

**Estimand.** For each of the 12 non-reserved BEIR-15 datasets (scifact, nfcorpus, scidocs,
trec-covid, fiqa, arguana, msmarco, nq, hotpotqa, webis-touche2020, quora, climate-fever), the mean
per-query nDCG@10 of a per-query oracle that picks the better of Zero and Nano, and the oracle
frontier: the best mean reachable when Nano serves a fraction b of queries, for b from 0 to 1 in
steps of 0.05 (queries sorted by Nano minus Zero gain, largest first).

**Data.** Committed per-query rows `results/m20_beir15_scores/<dataset>/{zero-dense,nano-dense,stella-query}.json`
for those 12 datasets only. No encoding, no search.

**Outputs.** `results/m15_e5_oracle.json`: per dataset and as the macro over 12, the Zero, Nano and
oracle means; the share of queries where Zero is above, equal to and below Nano; the minimum Nano
fraction at which the oracle reaches its maximum; the frontier; and, as a reference, the share of
queries where Zero is at or above the Stella query path.

**What it can show.** An upper bound on what any per-query router between the two tiers can gain,
and where the gain sits. **What it cannot show.** A router that reaches the bound. Ties are common
because nDCG@10 is discrete; the equal share is reported and never assigned to either side.

## E6, a fertility router, fitted outside the evaluation datasets

**Hypothesis.** Queries whose words split into more subwords lose more under Zero (M8 found -0.050
nDCG@10 per +1 subword per word for Zero against the Stella path, `m8/FINDINGS.md:78`). E6 tests
whether the same feature routes queries between Zero and Nano. No result has shown this yet.

**Feature.** Fertility = Zero tokenizer tokens (no special tokens, whole query) / max(1, whitespace
words), the definition of `m8src/d2_pre.py:620`. The direction is fixed in advance: a query goes to
Nano when its fertility is above the threshold.

**Fit.** On the pooled test queries of cqadup-physics and cqadup-programmers (the two M7 dev forums,
outside the 12 evaluation datasets), the threshold t_B for each Nano budget B in {0.10, 0.25, 0.50}
is the (1 - B) quantile of fertility. The three thresholds are written to
`results/m15_e6_thresholds.json` before any evaluation dataset is read, and the final result binds
that file's sha256. Neither E5 nor the 12 evaluation datasets selects anything.

**Evaluation.** On each of the 12 datasets, and as the macro, for each B: the router's mean nDCG@10,
its realized Nano fraction f, always-Zero, always-Nano, the expected mean of random routing at the
same f (computed exactly as (1 - f) Zero + f Nano, no sampling), the E5 oracle at the same f, and the
efficiency (router - random) / (oracle - random). A 10,000-draw query bootstrap gives an interval for
router minus random per dataset: each draw resamples aligned (route, Zero, Nano) rows with the
threshold fixed and recomputes f, the router mean and the random mean. Feature cost comes from E1.

**Outputs.** `results/m15_e6_router.json`, with the thresholds, the fit-forum in-sample numbers and
the evaluation table.

**What it can show.** Whether a zero-cost feature beats random traffic splitting at the same Nano
share, and how much of the oracle headroom it captures. **What it cannot show.** That fertility is
the best feature; one feature is tested. A router near random is a reported negative result.

## E1, encoder latency under one protocol

**Estimand.** Per-query encode latency on the M5 Pro CPU for Zero, Nano, the Stella query path,
bge-small and the LEAF query encoder, at batch one, by query-length bucket; plus hydration time,
first-query time, peak RSS, asset size, and the time to compute the E6 fertility feature.

**Queries.** Real test query texts (text only, no labels) pooled from the 12 E5 datasets, bucketed
by whitespace words: short 1-4, medium 5-12, long 13-64. 100 queries per bucket, drawn with seed
20260930. The drawn list and its hash go in the result.

**Runtime.** Zero runs its published NumPy encoder. The four transformer encoders run in ONNX Runtime
1.29.0 on CPU with the same session options: 4 intra-op threads, 1 inter-op thread, full graph
optimization, dynamic padding, batch one, fp32. Stella uses the published document ONNX graph with
the `s2p_query` prompt (min-cos 1.0 against the torch path, `m11/STATUS.md`); bge-small and LEAF are
exported as in `scripts/m13_serving_costs.py`. Each ONNX encoder must match its registered torch
path on 32 drawn queries at minimum cosine 0.9999 before timing; a failure stops that encoder.

**Procedure.** Three fresh processes per encoder, run in sequence on an otherwise idle machine. In
each: hydrate (imports, hash check, load), one cold query, five warm-ups, then every drawn query
once per bucket, timed with `time.perf_counter`. Report p50 and p95 per bucket per trial and the
median over trials.

**Outputs.** `results/m15_e1_latency.json`. F1's x-axis is the medium-bucket p50.

**What it can show.** Relative query-side compute on one edge-class CPU under one runtime. **What it
cannot show.** Server throughput, GPU cost or end-to-end search latency (E2 owns that). An ONNX
Runtime latency is not a FastEmbed latency; the M13 WSL numbers stay separate.

## E4, prefixes, incomplete queries and word order

**Estimand.** For Zero, Nano, the Stella query path and BM25, nDCG@10 under exact search when each
test query is cut, and retention = mean nDCG@10 of the cut query / mean nDCG@10 of the full query,
per encoder (a ratio of means).

**Data.** SciFact, NFCorpus and FiQA test queries and qrels at the M20 pinned revisions. Stella
document vectors: the local SciFact and FiQA vectors in `artifacts/demo-stella/`, and NFCorpus
encoded on the Mac (3,633 documents); all three pass the reproduction gate first. BM25 is the
registered bm25s configuration of `m7src/fusion.py`.

**Conditions.** Word prefixes with the first k words, k in {1, 2, 3, 5}, and the first 25%, 50% and
75% of words (rounded up); a character prefix at 50% of characters, which usually ends inside a
word; control 1, a random word subset of the same length as each word prefix, order kept, seed
20260930; control 2 (this is E7), the full query with its words shuffled, seed 20260930. A query
shorter than a condition's length keeps its full text and is counted. Zero ignores word order by
construction, so control 2 measures Stella's and Nano's order sensitivity against a zero-sensitivity
baseline.

**Outputs.** `results/m15_e4_prefix.json`: per dataset, encoder and condition, the mean nDCG@10,
retention and a 10,000-draw query bootstrap interval for retention; for the shuffle condition also
the mean cosine between the shuffled and original query vectors.

**What it can show.** Whether the cheap tiers lose less or more than the Stella path when the query
is short or unfinished, and whether prefixes are worse than random subsets of the same length.
**What it cannot show.** Real search-as-you-type behavior: real partial queries are typed by people
and differ from cut test queries.

## E2, does the saving survive the search

**Question.** On one Stella collection served by Qdrant, how much quality does each query encoder
lose to approximate search and quantization, and what end-to-end latency does each encoder need to
stay within a fixed loss?

**Data.** (a) FiQA, 57,638 documents, 648 test queries, local vectors after the reproduction gate.
(b) A positive-preserving 1M MS MARCO subset diagnostic: every passage judged relevant for a BEIR
MS MARCO dev query, plus passages drawn uniformly without replacement from the rest of the
8,841,823-passage corpus (seed 20260930) up to exactly 1,000,000. The ID list is written to
`work/m15/msmarco1m_ids.json` and its sha256 goes in the result. All 6,980 dev queries. The subset
keeps every positive and removes most competitors, so its nDCG is optimistic against the full
corpus, and its ANN penalty need not transfer in size or sign; no output presents either as a
full-corpus number. MS MARCO carries its validation-only role, and Nano's backbone (bge-small) was
trained on MS MARCO.

**Encoding.** The 1M passages are encoded on the Mac MPS with the registered Stella document path,
batch 64, resumable in shards, under `nohup`. Before the run, 256 FiQA documents encoded on MPS must
match the local FiQA vectors at minimum cosine 0.9999. The exact scores for Zero, Nano and the Stella
query path on the subset come from the same numpy exact search as E4.

**Qdrant.** The native macOS Qdrant binary at the newest release at or above 1.18, version and sha256
recorded; the Python client 1.19.0 over gRPC; one client, sequential queries. Collection: cosine,
the stored vectors as float32, HNSW m = 16, ef_construct = 100, the index fully built before timing.
Settings: search `hnsw_ef` in {16, 32, 64, 128, 256, 512}; quantization none, scalar int8, binary
1-bit and TurboQuant 4-bit, each quantized setting with rescoring on the original vectors and
oversampling in {1, 2, 4}. A Qdrant `exact=true` search per encoder checks the collection against
the numpy exact run (top-10 overlap at least 0.999). Every setting runs every query once after 100
warm-up queries.

**Metrics per encoder and setting.** Neighbour recovery@10 against that encoder's exact top 10;
ANN nDCG@10 and its loss against the same encoder's exact nDCG@10; search latency p50 and p95;
end-to-end latency = E1-protocol encode time of that query + search time, both measured in the same
process; query-to-nearest-document geometry (top-1 cosine and the top-1 minus top-10 cosine gap).

**Decision rule, fixed now.** For each encoder and each loss target e in {1%, 2%, 5%} of its own
exact nDCG@10, report the setting with the lowest end-to-end p50 whose ANN nDCG@10 is at least
(1 - e) times exact, and that p50. The saving survives at target e when Zero's p50 at its setting is
below Nano's p50 at Nano's setting; the result reports the ratio for each e on each dataset.
Settings that miss every target stay in the table.

**Fused cost.** Zero + BM25 and Nano + BM25 under Qdrant's Query API: dense and BM25 prefetch of 100
each, DBSF fusion, on the unquantized collection with `hnsw_ef` 128, plus a sparse BM25 vector per
passage with the IDF modifier. The latency and nDCG@10 of this Qdrant BM25 are labelled as such and
are not the bm25s rows of M20.

**Outputs.** `results/m15_e2_ann_fiqa.json` and `results/m15_e2_ann_msmarco1m.json`. F4 plots
end-to-end p50 against ANN nDCG@10 per encoder, each dataset in its own panel.

**What it can show.** Whether the query-compute saving survives at a fixed quality loss on one
index, and whether Zero's queries need more search effort. **What it cannot show.** Full-corpus MS
MARCO performance, multi-client throughput, or memory-limited behavior (not measured; Docker is not
used).

**E2 presentation audit, 2026-10-01 (exploratory, after observation).** For the broader Constella
paper, restrict the committed MS MARCO 1M sweep to binary quantization, then show the lowest
encode-plus-search p50 within 1% of each tier's own exact nDCG@10. Repeat that restriction for
the unquantized collection to illustrate how search configuration changes the remaining saving.
These are selections of existing receipt rows, not new measurements or equal-quality contrasts.
`m15/make_evidence.py` regenerates the selections in C10. The source shows all three precomputed
query-vector batches searching each populated collection without a rebuild between encoders;
it does not time model loading or a concurrent request dispatcher. The fusion example uses its
separate dense-plus-sparse collection. This note does not alter E2's original decision rule.

## E7

Folded into E4 as the shuffled-word control. No separate run.

## E8, tower generality under the closed-form recipe (Runpod)

**Question.** Under the one fixed closed-form table recipe, does a table ranking selected on the two
dev forums predict the table ranking on six public BEIR sets, and does tower quality predict either?
E8 tests this recipe only. It does not test whether distillability is intrinsic to a tower, because
the configurations differ in checkpoint, training exposure, dimension and readout.

**Configurations.** The 11 of `results/m7_learnability_report.json`, at the revisions pinned in
`m7src/encoders.py:85-188`: stella-400M-v5, arctic-embed-l, arctic-embed-l-mean (same checkpoint,
mean readout, reported as a separate control and left out of checkpoint-level statistics),
arctic-embed-m-v1.5, bge-base-en-v1.5, bge-large-en-v1.5, e5-base-v2, e5-large-v2, gte-base-en-v1.5,
gte-large-en-v1.5, mxbai-embed-large-v1.

**Fit list.** M8's cleaned `work/m8_trainq_texts.json`: 337,981 queries, 21,327,603 bytes, sha256
`da0f208ef29bae3833ffead070eb34b61f390f9d72ed96d448d45ae6b52072c2` (`results/m8_trainq_manifest.json`),
checked on the pod before anything runs. A hash mismatch stops E8. M7's `work/trainq_texts.json` is
never read, and `m8src/protected_filter.py` and `scripts/encode_trainq.py dump` never run: both read
protected fingerprints, and the list already exists.

**Recipe (fixed).** The M8 screen `m8src/teacher_screen.py:screen_one`, unchanged in its math: X is
the row-normalized token-count bag of each fit query under the tower's tokenizer, Y is the tower's
own fp16 query vector with its registered prefix and readout, W0 is the teacher-init rows, and W
solves min ||XW - Y||^2 + lambda ||W - W0||^2 by block CG (identical to the direct solve, `m8/RESULTS.md`
B7). Lambda grid {1e-4, 1e-3, 1e-2, 1e-1}; when the best value sits at a grid edge, the grid extends
by one decade on that side once, and the edge flag is reported. **Convergence gate:** a solve counts
only when block CG reports `converged` with every column's relative residual at or below the
solver's own tolerance (`m8src/blockcg.py:83`, 1e-6 in fp32, 1e-8 in fp64), at every lambda and at the
final re-solve. A failed solve is never scored or selected; it stops that configuration, and the
result lists it with its diagnostics. Every solve's iterations, worst residual and seconds go in the
result. `m8base`'s path guard is imported.

**Document path.** Every E8 score, dev and BEIR, table and ceiling, uses each tower's registered
document path (`spec.doc_prefix`, fp16 storage). M7 scored tables with an empty document prefix but
ceilings with `passage: ` for e5-base-v2 and e5-large-v2 (`m7src/dev_eval.py:71`,
`m7src/teacher_probe.py:75`); E8 removes that mismatch, so the two e5 dev numbers can differ from M7.

**Order, fixed.** (1) For each configuration, fit at every lambda and score the two dev forums
(cqadup-physics and cqadup-programmers, `work/dev/`, macro of the two forum means, nDCG@10, exact
search); also score each tower's own query path on the same forums (dev ceiling). (2) Freeze each
configuration's lambda as its best dev macro and write all frozen lambdas and dev scores to
`work/m15/e8/frozen_lambdas.json`, with its sha256 printed to the log, before any BEIR set loads.
(3) Re-solve W at the frozen lambda and score scifact, nfcorpus, fiqa, arguana, scidocs and
trec-covid once, at the M20 pinned revisions, exact search, nDCG@10; score the tower's own query path
on the same sets (BEIR ceiling). No lambda, configuration or dataset changes after step 2.

**Estimands.** Primary: Spearman rho between the dev table macro and the six-set table macro over the
10 checkpoints. Secondary: rho between the six-set ceiling and the six-set table macro; rho between
the dev ceiling and the dev table macro. All three also over the 11 configurations, labelled. Per
configuration and dataset: absolute table nDCG@10, ceiling, and retention (table macro / ceiling
macro, a ratio of macros). The six-set macros are reported both over all six and over clean-4
(nfcorpus, scidocs, scifact, trec-covid). With n = 10, rho is descriptive, with no p-value.

**Exposure and tokenizer checks.** A per-configuration, per-dataset matrix of disclosed or unknown
training exposure for the six sets, from each model card at the pinned revision and the notes in
`research/m7-teacher-shortlist-2026-08-26.md`; "unknown" is the default and is never read as clean.
On the pod, a tokenizer check records for each configuration the vocab size, CLS id, the sha256 of
the ordered vocabulary, and the lowercase and accent normalizer settings; a configuration outside
the shared 30,522 BERT WordPiece vocabulary with CLS id 101 is reported and excluded.

**Six-set access.** Step 3 only, run once.

**Compute and spend.** Runpod pod `k3aee2m68765em` (one A100 80 GB, $1.59/h, owner ruling
2026-09-30). Encodes: about 337,981 fit queries and 272,117 BEIR documents per tower for 10 towers
(Stella's are cached), plus dev documents where no cache exists. Estimate 8 to 10 GPU hours, about
$16. Spend cap $40 including the retained pods' storage for the run's duration; at a projected $35
the run stops and the owner decides. The pod runs a branch commit pushed before the start. Only
`results/m15_e8_towers.json`, its receipt fields and `m7/SIX_ACCESS.log` come back; W matrices and
caches stay on the pod volume. The pod is stopped (not deleted) when the result is copied.

**Outputs.** `results/m15_e8_towers.json`: the frozen lambdas and their file hash, the lambda curves,
per-configuration dev and BEIR tables, ceilings, retention, the three rho values on both rosters, the
exposure matrix and the tokenizer check, plus the standard receipt.

**What it can show.** Whether dev-forum selection of a tower transfers to public BEIR under this
recipe, and whether a tower's own retrieval quality predicted its table on public BEIR. **What it
cannot show.** Anything about a trained table (Zero is trained, not closed-form), any recipe other
than this one, or a causal effect of tower properties.

## E9, blended query vectors on one index (added 2026-09-30, before any E9 observation)

**Question.** Zero and Nano write query vectors into the same Stella document space. Does adding the
nearly free Zero vector to the Nano vector, and running one search, beat Nano alone? The same for the
Stella query path. This is possible only because the tiers share one index; no second search, no
score fusion.

**Blends.** q(a) = normalize((1 - a) nano + a zero) and q(b) = normalize((1 - b) stella + b zero),
with a, b in {0, 0.1, 0.2, 0.3, 0.4, 0.5}.

**Fit, then freeze.** a* and b* maximize the macro nDCG@10 of the two M7 dev forums (cqadup-physics,
cqadup-programmers); a tie goes to the smaller weight. Both are written to
`results/m15_e9_frozen.json` before any evaluation dataset is scored.

**Evaluation.** The six M7 public sets (scifact, nfcorpus, fiqa, arguana, scidocs, trec-covid), exact
search over Stella document vectors that pass the reproduction gate. Per dataset, the macro over six
and over clean-4: nDCG@10 of the frozen blend, of Nano (or Stella) alone, and a paired 10,000-draw
query bootstrap interval for blend minus alone. Reference row, not a contrast: Nano + BM25 DBSF@100
from the committed M20 rows. The full weight curve on the evaluation sets is reported as a
post-hoc description and never selects anything. Added cost: Zero's encode (E1).

**Outputs.** `results/m15_e9_blend.json`. **Can show:** whether the table vector carries signal the
transformer lacks, recoverable in one search. **Cannot show:** ANN behavior of the blend, results
beyond six public sets, or why.

## E10, routers from signals Zero already has (added 2026-09-30, before any E10 observation)

**Question.** E6's fertility router captured about 10% of the oracle headroom. Do signals that Zero
produces anyway route better?

**Features.** f1 fertility (E6's, as a reference); f2 the L2 norm of Zero's pooled vector before
normalization (low when the token rows disagree); f3 whitespace word count; f4 Zero's retrieval
margin, cosine of its top 1 minus its top 10 document (post-retrieval, from Zero's own search).

**Fit, then freeze.** On the pooled test queries of the two dev forums, each feature's direction is
the sign of its Spearman correlation with Nano minus Zero per-query nDCG@10, and its threshold for a
Nano budget B in {0.10, 0.25, 0.50} is the matching quantile in that direction. Directions and
thresholds go to `results/m15_e10_frozen.json` before any evaluation dataset is read.

**Evaluation.** f1-f3 on the 12 E5 datasets from committed M20 rows; f4 on the six E9 sets, whose
per-query Zero and Nano rows come from the gated local exact runs (identical to M20 rows at the
gate). Statistics as E6: router, random at the same realized fraction, oracle at that fraction,
efficiency, and the bootstrap with the fraction recomputed per draw. All four features are
reported; none is chosen on evaluation data. System cost of an f4 router: Zero's path always, plus
Nano's for the routed fraction, computed from E1 and E2.

**Outputs.** `results/m15_e10_router.json`. **Can show:** whether a zero-cost tier can tell when it
is wrong. **Cannot show:** that these features are the best available.

## Amendments

- **2026-09-30, before E1 ran.** Nano's torch checkpoint is not on this Mac. Nano's E1 parity gate is
  its ONNX sha256 equal to the M13 freeze (`results/m13_build_record.json`, `freeze.onnx.sha256`,
  `9ba0acf5...`); E4's reproduction gate checks the Nano query path end to end.
- **2026-09-30, before E8 ran (driver `m15src/e8_towers.py`).** Two details the E8 section left
  implicit, fixed now: every E8 encode (fit targets, documents, ceiling queries) is fp16 compute and
  storage, as the M7/M8 screen did for every tower; and every E8 search, table and ceiling, retrieves
  the exact top 100 with self-hits dropped before nDCG@10 (M7's ceiling probe retrieved 10). The
  Stella fit targets reuse the pod's `trainq-337981-fp16-7423fa42cd3e` cache, whose key binds the
  text hash of the cleaned list.
- **2026-09-30, before E9 or E10 ran (Astra review `REVIEWS/2026-09-30-astra-e9e10.md`).** E9 runs
  the reproduction gate on every dataset it scores and stops on a failure; f4 uses Zero's 1st and 10th
  score after the same self-hit drop the search applies; E9 adds macro intervals (datasets resampled
  independently, equal weight); E10 reports the f4 router's cost from E1 encode p50 (medium bucket) and
  E2 search p50 (msmarco1m, unquantized, `hnsw_ef` 128). Added, on Astra's suggestion: an f4 router that
  escalates routed queries to the E9 blend at its frozen weight instead of to plain Nano, with the
  same thresholds and statistics.
- **2026-09-30, before E9 or E10 ran.** E10 adds f5, the number of documents shared by Zero's and
  BM25's top 10 (registered bm25s, same self-hit drop), a post-retrieval agreement signal; fitted,
  frozen and evaluated on the six E9 sets exactly as f4. BM25 is part of the fused tier already, so
  f5 costs one sparse search.
- **2026-09-30, disclosure, before the E9 freeze.** A smoke of `e9_blend.dataset_scores` on SciFact,
  run to check the code path, printed SciFact's blend curves (Nano alone 0.7211, weight 0.1 0.7193;
  Stella alone 0.7796, weight 0.1 0.7799) before the weights were frozen on the dev forums. SciFact is
  an E9 evaluation set. Nothing in the registration or the code changed after this; the weights are
  still chosen by the code on the two dev forums only. The paper reports this peek.

## E15, exploratory decision and cost audit (2026-10-01)

`m15src/e15_decision_audit.py` derives selection consequences from the published E8/E8x
aggregates: the development-screen winner, the strongest public teacher, the best public table,
their all-six differences, clean-four regret, and per-dataset ranks. The strongest public teacher
is a hindsight comparator, not a prospectively tested model-card selection rule. Dimension is
correlated with absolute table quality as well as retention to expose the latter's denominator
coupling. Nano's measured training seconds are priced at its recorded historical rental rate;
this is a training-loop estimate excluding upstream preparation and the broader research cost.
Zero's ledger estimates are labelled planning estimates, not measured timings. This analysis
was specified after seeing the original results and is exploratory throughout. It uses only
five committed JSON receipts and `m7/LEDGER.md`, hashes those inputs, and opens no raw data.

## E16, same-graph binary-quantization pilot (exploratory, 2026-10-01)

Owner welcomes justified additional compute/research, excludes full model retraining, and suggests
quantization with fair comparisons. This bounded diagnostic uses no cloud rental or training.

- **Question:** does binary scoring/candidate selection add a different neighbor-recovery penalty
  for Zero, Nano, and Stella when the physical document graph is held fixed? This is not a new
  quantizer, a general manifold explanation, or exhaustive compressed-score decomposition.
- **Data:** cached 57,638 FiQA document vectors from E2, original fp16 values cast fp32 and normalized.
  Select 192 of the 648 public test queries by SHA256(seed:qid), first hashes, seed 20260930. No
  selection by quality. Encoders are E1's frozen released artifacts; no new document encoding.
- **Index:** one isolated Qdrant 1.19.1 binary-quantized collection, HNSW m16/ef_construct100, all
  documents indexed. Every query tier uses exactly that graph and codes. Dedicated free ports and
  a new storage directory prevent contact with existing collections. Collection configuration and
  point/indexed counts must stay identical before and after the query ablation.
- **Conditions:** ef16,64,256; ignore quantization (original scoring), binary without rescoring,
  binary with original-vector rescoring and oversampling1, and binary with oversampling4. Every
  condition uses limit10. Queries share each tier's exact-original top10 reference. Native
  exact=true/ignore=true is only a parity gate against normalized NumPy original scores.
- **Execution:** pre-encode the three query-vector sets, run ten warmups per condition, then
  interleave all tier/condition pairs in a seeded random order for each selected query. Record
  neighbor IDs, per-query recovery and nDCG@10, search p50/p95, vector hashes, graph/config counts,
  model pins, source hashes, dataset pins, and the dedicated native binary hash.
- **Analysis:** ef64 is the primary pilot display; ef16 and ef256 are sensitivity displays. Compute
  binary-rescore1 minus original-scoring recovery within each query/tier, then paired differences
  of those losses for Zero versus Nano and Stella. Report 10,000-draw query-bootstrap intervals;
  these do not cover graph-build or training variation. Rescore/no-rescore and oversampling effects
  show bounded native request behavior; no exact-quantized ranking or causal manifold claim follows.
- **Decision:** a substantial repeated extra Zero loss would justify a larger same-graph study with
  a second dataset/size and precision/candidate-budget controls. A small or absent interaction
  redirects attention to base graph difficulty. No pilot result is promoted to a confirmatory test.
- **Output:** immutable `results/m15_e16_quantization_pilot.json`, generated by
  `m15src/e16_quantization_pilot.py`. Protected reserved and raw confirmation material remain closed.

## E17, second-scale same-graph replication (exploratory, 2026-10-01)

After the E16 pilot, repeat its fixed query-ID sampling rule (192 by SHA256(seed:qid)), three frozen
encoders, one fixed binary collection, ef16/64/256, and four native request treatments on the cached
positive-preserving 1M MS MARCO diagnostic. Primary display remains ef64. Original exact references
are recalculated against that document set, with normalized decoded fp16 vectors and native parity.
No hyperparameter or sampling change follows the pilot. This remains a validation-only diagnostic,
not full MS MARCO, and bge-small pretraining exposure remains relevant. Source:
`m15src/e17_quantization_replication.py`, using the hash-bound shared E16 implementation. Output:
immutable `results/m15_e17_quantization_replication.json`. No new training, encoding, or rental.

Read the native source check in `REVIEWS/2026-10-01-quantization-semantics.md`: Qdrant 1.19.1's
`query_encoding:null` uses SameAsStorage, hence sign-bit queries with one-bit documents. Native
rescoring can change collection-level candidate selection through segment merging; rescore/no-rescore
is not a fixed global candidate-pool experiment. This replication tests the native option interaction,
not an exhaustive quantized-score decomposition or an identified geometric mechanism.

## E18, controlled query-precision candidate scoring (exploratory, 2026-10-01)

After E16/E17 and tagged-source verification of sign-coded query defaults, hold document sign codes
fixed and compare exhaustive scores from sign(q)·sign(d) with normalized original q·sign(d). Positive
components map to+1, others−1, matching the tagged one-bit threshold. Each tier's original-vector
exact top10 and deterministic 192 query IDs come from its immutable E16/E17 receipt; freshly encoded
query vectors must match those recorded hashes. Use both FiQA and the 1M validation-only diagnostic.

Global candidate budgets are10,40,100;40 is the primary display. For quantized-score ties at the cutoff,
report expected original-top10 inclusion under uniform tie selection: a target strictly above the
cutoff has probability1; at the cutoff its probability is(C−number strictly above)/number tied.
This equals expected exact-neighbor recovery after original-target reranking of that candidate pool,
but neither measures relevance nDCG nor native search latency. Report paired float-minus-sign gains
and Zero-minus-other-tier gain differences, with query-bootstrap intervals. Show all budgets/results.

No graph is used; document codes, original target and query vectors are unchanged between precision
conditions. The float-query scorer is an explicitly defined magnitude-preserving control, not native
Qdrant scalar8, an ANN remedy, or a guaranteed upper bound. Global candidate budgets are not native
per-segment oversampling. No new encoding of documents, training, cloud rental, protected access or
query selection occurs. Source:`m15src/e18_query_precision.py`; immutable output:
`results/m15_e18_query_precision.json`.

## E19, cross-recipe teacher screen (exploratory, pre-specified 2026-10-01, before any scoring)

Owner-approved under `REVISION_PLAN_V16.md`. E8/E8x ranked 26 teachers by the six-set quality of a
closed-form token table fitted to each teacher's query vectors. E19 asks whether that ranking is a
property of the table recipe or of the teachers: it repeats the screen with a second, independent
student family and the same teachers, fit list, targets, document indexes, selection order and
scoring, changing only the student.

- **Question:** does the teacher ranking transfer from the closed-form table (recipe 1) to a
  frozen-backbone ridge head (recipe 2), and does recipe 2's ranking also ignore the teacher's own
  retrieval quality?
- **Fixed:** the 26 pooled checkpoints of E8x plus the E8 control `arctic-embed-l-mean`; the
  registered 337,981-query fit list (sha256 `da0f208e...`); each teacher's cached fit-query targets
  (`trainq-337981`, the teacher's query prefix, fp16) and cached dev/six-set document vectors
  (`e8-dev-*-docs`, `e8-six-*-docs`); exact nDCG@10 via `evalkit.score` at k=100 with the
  registered self-hit drop; the dev-then-freeze-then-six order.
- **Recipe 2:** frozen `BAAI/bge-small-en-v1.5` at the Nano build's pinned revision. Features are
  the masked-mean-pooled hidden states of layers 12, 8 and 4 concatenated (1152) plus a bias column,
  the M10 feature path, computed once for the fit list, the dev queries and the six-set queries at
  max length 512. The head is a dense ridge solve from features to the teacher's targets with
  penalty `lambda * trace(G)/d`, grid 1e-4, 1e-3, 1e-2, 1e-1, one decade extension at an edge, as
  E8. Student query vectors are L2-normalized. Lambda is chosen on the two dev forums by retrieval,
  then all lambdas are frozen to `work/m15/e19/frozen_lambdas.json` (hash in the result) before any
  six-set load. Each teacher is scored once per set; `m7/SIX_ACCESS.log` records the loads.
- **Special case:** `bge-small-en-v1.5` is also a teacher; for that row the student shares the
  teacher's backbone. It is reported, and every correlation is given with and without it.
- **Pre-specified analysis, descriptive, no p-values:** (1) Spearman between recipe-2 and recipe-1
  six-set macro over the registered ten and over the 26, with E8x's 10,000-draw checkpoint
  bootstrap and leave-one-family-out; (2) Spearman between recipe-2 six-set macro and the teacher's
  six-set score, same rosters and intervals; (3) recipe-2 dev screen versus recipe-2 six-set macro;
  (4) the strongest-registered-teacher comparison for recipe 2 (gte-large-en-v1.5 versus the
  recipe-2 screen choice), as C13 did for recipe 1; (5) clean-four variants of (1) to (3).
- **Reading:** rankings agree and both ignore teacher quality: the strongest-teacher heuristic is
  challenged for cheap query students beyond one recipe. Rankings disagree: the screen is
  recipe-specific. Neither outcome makes the screen a selector for the trained Zero or Nano, and a
  frozen-backbone head is a floor for a trained transformer student (M9: 50.8% of its ceiling).
- **Cost and access:** the retained A100 pod volume holds every cache; if it is unavailable the
  fit-list targets and six-set documents are re-encoded (8 to 10 A100 hours). No reserved set, no
  training, no change to any existing result.
- **Output:** immutable `results/m15_e19_head_screen.json` from `m15src/e19_head_screen.py`.

## E20, cross-space ANN effort (exploratory, pre-specified 2026-10-01, before any scoring)

Owner-approved under `REVISION_PLAN_V16.md` after Astra's agreement. E2 found that Zero needs more
graph-search effort than Stella's own queries over the same Stella index. E20 asks whether that is a
property of Stella or of table-based query substitution.

- **Question:** across 26 teacher spaces, does a fitted table's query path need more HNSW effort
  than the teacher's own query path to reach the same relative quality over the teacher's unchanged
  index, and do measurable query-geometry differences track the size of that penalty?
- **Fixed:** FiQA (57,638 documents) and TREC-COVID (171,332 documents); per teacher, the cached
  six-set document vectors and teacher query vectors from E8/E8x, and the table re-solved at the
  frozen E8/E8x lambda (recipe 1). If E19 is complete, its recipe-2 student queries form a third
  path per space, reported as secondary. Qdrant 1.19.1, cosine, HNSW m=16, ef_construct=100,
  indexing threshold 1 KB, no quantization, limit 11 with the registered self-hit drop. Two graph
  builds per space and workload, differing in a seeded insertion order. Exact parity against NumPy
  on 200 queries per path before any ANN row.
- **ef grid:** 16, 32, 64, 128, 256, 512.
- **Measured:** per space, workload, build, path and ef: recall@10 against that path's own exact
  top-10, nDCG@10 and relative loss against that path's own exact nDCG@10, and search p50/p95.
  Per space, workload and path: median top-1 cosine and median top-1 minus top-10 score gap from the
  exact scores. Stella's rows must agree with E2's FiQA rows within the two-build spread; this is a
  consistency check, not a gate.
- **Pre-specified analysis:** the teacher reference is ef=64. The effort multiplier is the smallest
  grid ef at which the table's relative nDCG loss is at or below the teacher's at ef=64, divided by
  64; a table that does not reach it by ef=512 is censored, not assigned 512. Recovery-based
  multipliers are computed the same way as a secondary display. Report the multiplier distribution
  per workload with builds averaged, the paired table-minus-teacher recovery gap at ef=64, and
  exploratory Spearman between that gap and the geometry differences (table minus teacher top-1
  cosine; table minus teacher margin). Builds are repetitions; related checkpoints are shown by
  family and are not independent samples. Latency is recorded on whichever host runs the sweep and
  is not a latency claim.
- **Reading:** a recurring penalty generalizes the Stella observation to table-based query
  substitution; absent or reversed penalties bound it to some spaces. Both are reportable.
- **Output:** immutable `results/m15_e20_ann_spaces.json` from `m15src/e20_ann_spaces.py`.
  No reserved set, no training, no change to any existing result.

**E20 amendment (2026-10-01, before any E20 scoring).** TREC-COVID has 50 test queries, too few
for a per-space relative-loss estimate on its own. SCIDOCS (25,657 documents, 1,000 queries) is
added as a third workload so the scale axis (TREC-COVID, 171,332 documents) and the query-count axis
(FiQA 648, SCIDOCS 1,000) are both covered. The ef grid, builds, analysis and outputs are unchanged.

**E20 amendment 2 (2026-10-01, during the run, before any space's rows were read).** The exact
parity gate is 0.995 instead of E2's 0.999. The control `arctic-embed-l-mean` table on FiQA scored
0.9985: fp16-valued document vectors produce tied exact scores that Qdrant and NumPy order
differently. Every parity value is recorded per space, workload, build and path in the result.

**E20 amendment 3 (2026-10-01, during the run; one space's partial rows were discarded, none were
analysed).** Some tables produce exact-score ties at rank 10 (13 of 200 FiQA queries for
`arctic-embed-m-v1.5`; identical fp16-valued documents), so two exact scorers legitimately return
different members of a tied set and set-overlap recovery cannot reach 1 even for exact search.
Recovery@10 is therefore tie-aware: a returned document counts when its exact score is at or above
the exact 10th neighbour's score. This equals E2's set definition when no ties exist. The set
version and the per-path count of tied queries are kept in every row, and the parity gate returns
to 0.999 under the tie-aware measure. Amendment 2's 0.995 gate is superseded. All sweeps restart
from scratch so every space is measured identically.
The tie tolerance is 1e-4 in cosine score: float32 accumulation order differs between Qdrant and
NumPy, and the control table's rank-10 gaps sit at that level (parity 0.9985 at 1e-6, same three
slots). Every space, path and ef uses the same tolerance.
Parity gate: 0.995 under the tie-aware measure. After 22 spaces passed at 0.999, `tas-b`'s teacher
path on FiQA reached 0.998 (four slots in 2,000); the gate exists to catch a broken collection, and
every parity value is reported per space, workload, build and path in the result.

**E20 amendment 4 (2026-10-01, during the run, 22 spaces complete and unread).** A space whose
exact parity cannot reach 0.995 even under the tie-aware measure is excluded from the multiplier
distribution and listed in the result with its parity values, instead of stopping the run. The
case: `tas-b` on FiQA has a median rank-10-to-11 exact score gap of 8e-4 and 17 of 200 check
queries under 1e-4, so its exact top-10 is undefined at float32 precision and recall against it
is not measurable. Completed spaces are unaffected; the roster size in the result is the included
count.

## E21, conditional teacher-selection analysis (exploratory, pre-specified 2026-10-01, existing data only)

Owner round 4 asks for scientific weight rather than observed rankings. E8/E8x and E19 give, for
26 checkpoints and two student recipes, the teacher's six-set score, the student's six-set score,
the student's dev-forum score, the teacher's width, and its model family. E21 asks whether student
quality is predictable from teacher properties once width and family are accounted for, and whether
the recipe-specific dev screen adds predictive value beyond them. No new encoding or scoring.

- **Units:** the 26 pooled checkpoints (control `arctic-embed-l-mean` excluded), two recipes.
  Family labels: arctic, bge, e5, gte, minilm (four MiniLM variants), mxbai, stella, contriever
  (two), tas-b. Width is log2 of the teacher's output dimension (384, 768, 1024).
- **Outcome:** absolute student six-set macro nDCG@10 (not retention, whose denominator is the
  teacher score). Clean-four macro is a sensitivity outcome.
- **Nested linear models, fit by least squares:** M0 intercept; M1 teacher score; M2 teacher score
  + width; M3 teacher score + width + family dummies; M4 M2 + the recipe's dev-forum score; M5 M3 +
  dev score. Standardized coefficients reported with a 10,000-draw checkpoint bootstrap interval.
- **Validation:** leave-one-family-out prediction (fit on the other families, predict the held-out
  family), reporting Spearman between predicted and actual over all 26 held-out predictions and
  mean absolute error. Models with family dummies cannot predict a held-out family and are reported
  only in-sample. Leave-one-checkpoint-out is a secondary display.
- **Selection regret:** for each rule (teacher score; teacher score within the widest band; M2
  prediction fitted leave-one-out; dev screen), the six-set gap between the rule's pick and the best
  student, over the registered ten and the pooled 26.
- **Partial association:** Spearman between student and teacher score after linearly removing
  width from both; the same within each width band (384 n=9, 768 n=10, 1024 n=7).
- **Reading:** a large negative width coefficient with a positive teacher coefficient, and
  held-out prediction well above the pooled correlation, means the pooled near-zero association is
  two opposing effects; the dev screen earns its place if M4/M5 beat M2/M3 out of sample. Twenty-six
  related checkpoints and three width levels support a hypothesis about what cheap students pay for
  width, not a mechanism; the backbone-affinity checkpoint (bge-small-en-v1.5) is reported with and
  without.
- **Output:** immutable `results/m15_e21_width_model.json` from `m15src/e21_width_model.py`.

## E22, predictors of the cross-space recovery gap (exploratory, pre-specified 2026-10-01, cached vectors)

- **Question:** is the table-minus-teacher recovery gap at ef=64 (E20, two builds averaged)
  predictable from properties of query placement computable before building a graph?
- **Units:** the 25 included spaces by three workloads, 75 points; held-out validation leaves one
  family out, and separately one workload out.
- **Declared features, per space and workload, from the E20 cached vectors and saved exact
  top-100 ids and scores.** (1) Mean ratio of a table query's distance to its nearest document to
  the mean nearest-neighbour distance among documents (document NN distances from a 5,000-document
  sample); the same for teacher queries; and their difference. (2) Hubness of the table's exact
  top-10: the Gini coefficient of document occurrence counts across queries, minus the teacher's.
  (3) Overlap between the table's and the teacher's exact top-10 per query, averaged. (4) Effective
  rank (participation ratio of the covariance spectrum) of the teacher's cached fit-query cloud and
  of the table's fit-query cloud, and their ratio. (5) The two E20 geometry summaries, as controls.
  (6) log2 width and corpus size, as covariates.
- **Analysis:** Spearman of each feature with the gap over 75 points and within each workload,
  with 10,000-draw bootstrap intervals over spaces; a ridge model on standardized features with
  leave-one-family-out and leave-one-workload-out prediction, compared with the constant predictor
  by Spearman and MAE. Features are declared here before any is computed.
- **Reading:** a feature that beats the constant predictor out of sample makes RQ2 a predictive
  finding; none does, and RQ2 is reported as a measured regularity with the predictor table.
- **Output:** `results/m15_e22_gap_predictors.json` from `m15src/e22_gap_predictors.py`.

## E23, corpus-size deconfound (exploratory, pre-specified 2026-10-01)

- **Question:** does the smaller recovery gap on SCIDOCS (25,657 documents) come from corpus size?
- **Design:** subsample FiQA's 57,638 documents to 25,657 by a seeded uniform draw that keeps every
  document judged relevant to a test query; rebuild the uncompressed HNSW index per space with the
  E20 parameters and two builds; rerun the E20 sweep for the teacher and table paths on the same
  648 queries, each path against its own exact top-10 over the subsampled corpus.
- **Analysis:** per-space recovery gap at ef=64 on FiQA-25k versus FiQA-57k and versus SCIDOCS;
  paired across spaces with a bootstrap interval over spaces.
- **Reading:** a gap that shrinks toward the SCIDOCS value supports size; a gap that persists at
  25k points to domain or query form.
- **Output:** `results/m15_e23_fiqa25k.json` from `m15src/e23_fiqa25k.py`.

## E24, prospective teacher-selection test (exploratory, pre-specified 2026-10-01)

- **Question:** does the E21 rule, fitted on the 26 scored checkpoints, rank checkpoints it has
  never seen?
- **Roster, fixed here before any encoding:** eight BERT-WordPiece-compatible checkpoints not in
  E8/E8x, spanning widths: `intfloat/e5-large` (1024), `BAAI/bge-large-en` (v1, 1024),
  `intfloat/e5-base-unsupervised` (768), `nomic-ai/nomic-embed-text-v1` (768),
  `sentence-transformers/msmarco-bert-base-dot-v5` (768),
  `sentence-transformers/msmarco-distilbert-base-v4` (768),
  `sentence-transformers/all-MiniLM-L12-v1` (384), `sentence-transformers/paraphrase-MiniLM-L6-v2`
  (384). The E8 tokenizer check applies; a checkpoint that fails it is reported and excluded,
  not replaced.
- **Predictions written before scoring:** for each recipe, the teacher + width model fitted on the
  26 predicts each new checkpoint's six-set student score from its six-set teacher score and width.
  The teacher score is measured first (it is an input), then the prediction file is committed with
  its hash, then the students are fitted and scored under the E8 and E19 recipes and selection order.
- **Endpoints:** Spearman between predicted and actual over the eight, with a checkpoint bootstrap
  interval, against teacher-only prediction; selection regret of the strongest-teacher rule, the
  model pick, and the dev screen over the eight.
- **Reading:** prediction above teacher-only with an interval excluding zero makes RQ1 prospective;
  failure keeps the held-out fit and drops the prospective claim.
- **Output:** `results/m15_e24_prospective.json` from `m15src/e24_prospective.py`.

**E22 amendment 1 (2026-10-01, before any feature was computed; pre-spend review finding 5).**
Feature (4) is restated. The table's fit-query cloud is not available without re-solving every
table, so the declared quantities are: the effective rank of a seeded 20,000-row sample of the
teacher's cached fit-query vectors, and the ratio of the effective rank of the table's workload
query vectors to that of the teacher's workload query vectors. The table fit-cloud rank is
omitted, not approximated. Also fixed before computing: hubness is the Gini of top-10 occurrence
counts over the full document support including zeros; bootstrap intervals resample spaces with
their three workload rows together and are reported within each workload as well; the held-out
baseline is each training fold's mean, reported beside every held-out model.

## E25, LightRetriever's own lookup path under graph search (exploratory, pre-specified 2026-10-01, before any encoding)

Owner-approved (round 7). RQ2 found that a lookup-table query path needs more HNSW effort than the
encoder's own query path over the same index, across 25 encoder spaces built by closed-form
fitting. LightRetriever [ICLR 2026] trains its lookup query side jointly with an LLM document tower
and reports a query-encoding speedup of over 1000-fold. E25 asks whether the search-effort penalty
also appears in that model's own setting, where the lookup path is learned, not fitted.

- **Model:** the released `lightretriever/lightretriever-qwen2.5-1.5b` adapter (Hub revision
  `59776f218c4d13a1fbb075409514a82c452d27ae`) merged into `Qwen/Qwen2.5-1.5B`, bfloat16, as the
  project's earlier faithful reproduction loaded it (`bench/run_lightretriever.py`; dense scores
  reproduced the paper to within reporting precision after the BOS fix recorded in
  `research/m1-m6-findings.md`). Dense path only; 1536-dimensional last-token pooling, normalized.
- **Document vectors:** FiQA (57,638) and SCIDOCS (25,657) encoded by the model's document path (no
  instruction, `add_special_tokens=True`, max length 512). One frozen index per corpus.
- **Two query paths over each index.** (1) Lookup: the released construction, one table per
  instruction built by forwarding `[prompt] + [token] + [eos]` and taking the EOS state, query vector
  = normalized mean of the query's token rows (`add_special_tokens=False`); primary instruction is the
  "websearch" prompt, the per-task prompt is secondary. (2) Full: the same model encoding
  `[prompt] + query tokens + [eos]` and taking the EOS state, normalized, with the same instruction.
  Path (2) is the model's own contextual query encoding; path (1) is what the paper ships as the
  cheap side.
- **Search protocol:** E20's, unchanged: Qdrant 1.19.1, HNSW m=16, ef_construct=100, cosine,
  uncompressed, two builds per corpus with different seeded insertion orders, ef 16 to 512, limit 11
  with the self-hit drop, each path against its own exact top-10, tie-aware recovery@10, tie-aware
  exact parity at 0.995 on 200 queries per path and build.
- **Pre-specified quantities:** the relative-nDCG-loss multiplier (primary) and the recovery
  multiplier (secondary) of the lookup path against the full path at `ef=64`, averaged over builds,
  censored at 512; the recovery gap at `ef=64` with a query-bootstrap interval; the exact nDCG@10 of
  both paths (quality context, not a reproduction claim); encoding time per query for both paths on
  the pod GPU, reported as context only.
- **Reading:** a multiplier above one with a recovery gap whose interval excludes zero places the
  RQ2 regularity in a jointly trained model's own space; a multiplier at or below one bounds RQ2 to
  closed-form students. Either outcome is reported; neither disputes the paper's retention or its
  encoding-speedup figure, which is measured in a different place.
- **Not claimed:** latency on a shared GPU pod; the paper's exact numbers (our dense reproduction
  carried a documented BOS fix and the sparse side is out of scope); anything about the per-task
  instruction beyond a secondary row.
- **Cost:** one fresh A100 pod, about $10 to $15; no project cache is needed.
- **Output:** immutable `results/m15_e25_lightretriever.json` from `m15src/e25_lightretriever.py`.
