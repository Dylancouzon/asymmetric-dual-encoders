# M18 findings — internal Qdrant project memory

## Decision

M18 is complete with two distinct outcomes:

- `SYSTEM_READY`: a usable local Qdrant snapshot, Stella document index, BM25 index and
  Zero-v1/BM25 DBSF@100 query path are available for internal use.
- `ENCODER_NO_IMPROVEMENT`: none of the bounded specialized query tables passed the locked dense
  and fused development margins. Released Zero v1 remains the selected query encoder. No M18
  encoder is promoted or published.

This is the smallest defensible version for the use case. The project-memory system does not
depend on a training win, and the experiment stopped after the registered T0/T2 screen rather than
adding architectures, teachers, generated evaluation data or tuning sweeps.

## Snapshot, corpus and index

The source is `qdrant/qdrant` at commit `5e32ea89cb5ea9a6827a68b27291d2e2251e1b1c`, with GitHub
objects created by the `2026-09-13T00:36:26Z` cutoff. The acquisition retained 71,937 unique GitHub
objects and the pinned repository tree. One issue-events endpoint reached an apparent 30,000-object
server cap; those events are relationship metadata, not searchable text, and the limitation is
preserved in the source manifest.

Parsing produced 105,025 records across 11,574 artifact families. Of those, 79,269 natural-language
units are searchable; 25,756 issue events are non-indexable relationship metadata. Obvious
credential-shaped assignments were redacted, quoted reply lines were removed, and 4,850 exact
duplicates were removed. Search text includes issue and PR openings/bodies, comments, reviews,
repository prose and code comments—not titles alone.

The document tower is the unchanged pinned Stella 400M v5 model. Its normalized fp16 matrix has
shape 79,269 × 1,024 and is paired with a deterministic Lucene-style BM25 index. The primary query
path retrieves exact dense and BM25 candidates to depth 100 and fuses them with the existing DBSF
operator. No vector database or approximate index was introduced because this local corpus is
small enough for the registered exact path and changing the document contract was out of scope.

## Query protocol and training data

Astra adjudicated 344 prospective source-grounded query/answer pairs without seeing model scores:
140 were clear, 142 partial and 62 bad. Only clear pairs entered evaluation. The resulting held-out
protocol has 100 development and 40 one-shot confirmation queries across five strata. It is smaller
than the 250-query target because the instructions require reporting shortages rather than
manufacturing filler. All train/held-out family, artifact and answer-digest overlap checks are zero,
and each query-bearing source object's chunks are excluded before scoring.

Training contains 785 views: 298 non-held-out issue/PR-derived query-only views, 482 alias views and
five narrowly amended `k8s` views. The query-only rows use Stella and candidate-list supervision;
they are not declared qrels. The final exact vocabulary rows are `qdrant`, `shard`, `grpc`,
`snapshot`, `deps`, `json`, `flaky`, `docker`, `kubernetes`, `hnsw`, `config`, `s3`, `turboquant`,
`arm64`, `k8s` and `gridstore`.

The CTO-requested `k8s` amendment occurred after the initial baseline but before any optimizer
step. A pinned Qdrant source explicitly states “Kubernetes (K8s)”; five deterministic substitutions
of `kubernetes` in distinct already-admitted training contexts augment one natural `k8s` context.
They are teacher-only, do not alter qrels or held-out bytes, and did not require Qwen. Astra approved
this narrow revision. The original lock, preparation and baseline remain archived for provenance.

## Development results and encoder decision

Equal-weight stratum-macro development results on 100 queries were:

| Route | MRR@10 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|
| BM25 | 0.077111 | 0.109231 | 0.217564 |
| released Zero v1 dense | 0.066703 | 0.087336 | 0.157116 |
| released Zero v1 + DBSF | 0.080038 | 0.103431 | 0.177704 |
| Stella dense ceiling | 0.106519 | 0.136560 | 0.231981 |
| Stella + DBSF ceiling | 0.105902 | 0.140992 | 0.256083 |

The Stella ceiling showed measurable headroom, so the bounded encoder run was justified. T0 passed
the required step-500 parity diagnostic: inherited rows stayed unchanged, training-forward versus
serving error was at most `1.49e-8`, loader error was zero, and int8 error was at most `8.27e-4`.
The full 4,000-step T0 run reduced loss from 0.56220 to 0.25998. Its best dense checkpoint was step
4,000 at +0.007591 nDCG@10 versus v1 (95% paired interval [-0.000844, 0.018995]), clearing the
+0.005 dense screen. Its best fused result was the earlier step 250/500 tie at only +0.002953
(interval [0, 0.008858]), below the required +0.010.

T1 changed zero training and zero development queries and was skipped as registered. T2 changed six
development tokenizations and produced genuinely different query vectors, but every per-query
ranking metric equaled T0 at every checkpoint. It therefore failed the +0.005 replacement rule.
V0 was equal to v1 on observed rankings. No trained form passed both practical margins; hence a
second seed was not applicable. Astra independently checked the evidence and issued GO for
`ENCODER_NO_IMPROVEMENT`. T0 step 250 is retained only as the earliest checkpoint tied for best
first-seed fusion and cannot affect selection or deployment.

## One-shot confirmation audit

After the released-v1 decision, M18 locked the selected model/config/tokenizer hashes and all
development evidence, then read the 40-query confirmation set exactly once. The complete receipt
binds the result hash. Because no specialized candidate was eligible, this is an absolute audit of
the finished v1-backed system—not a post-hoc chance to reopen selection.

| Route | MRR@10 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|
| BM25 | 0.081167 | 0.109163 | 0.200000 |
| released Zero v1 dense | 0.132500 | 0.153176 | 0.220000 |
| released Zero v1 + DBSF | 0.104556 | 0.151445 | 0.300000 |

Fusion increased confirmation Recall@10 relative to either single route, though its nDCG@10 was
slightly below dense alone. Alias/jargon and concept/how-to benefited from dense/fused retrieval;
BM25 remained strongest on exact/version/numeric queries. Error/troubleshooting nDCG@10 was zero
for BM25, dense and fusion across its ten confirmation queries. That is the clearest quality gap in
the shipped internal system and should guide any future protocol or retrieval work.

Query-bearing source objects appeared in the unfiltered top ten for 97.5% of BM25 queries and 100%
of dense queries. Excluding every chunk of those objects is therefore essential: without that rule,
the evaluation would mostly measure verbatim self-retrieval.

## Bare terms and live query path

The report-only 21-term probe compares released v1 with the unselected T0-step250 bundle and has no
qrels. All 16 admitted terms changed dense and fused rank order; their mean top-ten document overlap
was 8.06. The unselected table's `k8s` dense list surfaced “Data persistence for k8s deployment” at
rank two, while both fused lists ranked it first. For `s3`, fusion ranked “Implement S3 snapshot
manager” first. Unsupported `mmap`, `cuda`, `tls`, `rocksdb` and `wal` rankings were unchanged.
These impressions show that the added rows affect plausible targets, but cannot override the fixed
evaluation gates.

Three end-to-end smoke queries also returned plausible top results: `k8s readiness probes` led to
the corresponding liveness/readiness issue, `How do I restore an S3 snapshot?` led to snapshot
discussion, and `HNSW configuration` led to vector-specific HNSW work. Observed full-query latency
was 173–264 ms on this host; the encoder alone was about 0.12 ms/query in confirmation. The wider
number is dominated by exact 79k-document dense scoring plus lexical search, so it is a local
functional measurement rather than a production latency claim.

Run the finished system from this worktree with:

```bash
/home/dylan/asymetric-dual-encoders/.venv/bin/python m18src/system.py query \
  "k8s readiness probes" \
  --bundle /home/dylan/asymetric-dual-encoders/work/release/zero-v1 \
  --limit 10
```

## Limitations

- The 140 clear held-out pairs are smaller and uneven by stratum; paired intervals are descriptive,
  and the confirmation error/troubleshooting failure is based on ten queries.
- Relevance is source-only Astra adjudication of repository relationships, not independent human
  judgments or exhaustive multi-relevance qrels.
- The `k8s` row has one natural plus five deterministic source-evidenced contexts. It is useful
  targeted coverage, not broad evidence about Kubernetes language.
- No complete alias pair survived the trainable-query filter, so the registered alias-consistency
  loss was inactive. Cosine/listwise supervision still trained the eligible new rows.
- Exact dense search is intentionally simple and reproducible but not a scale-out serving design.
- The snapshot is internally useful and reproducible at its pinned identities; it is not live,
  public, or evidence that Zero v1 improved.
