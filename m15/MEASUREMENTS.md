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

## Amendments

- **2026-09-30, before E1 ran.** Nano's torch checkpoint is not on this Mac. Nano's E1 parity gate is
  its ONNX sha256 equal to the M13 freeze (`results/m13_build_record.json`, `freeze.onnx.sha256`,
  `9ba0acf5...`); E4's reproduction gate checks the Nano query path end to end.
