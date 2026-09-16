# M20 pre-observation registration

**Written and pushed 2026-09-16, before any protected read and before any executor change was run
against real data.** Authority: owner rulings **R20**, **R22** and **R23** (`m13/RULINGS.md`).
The machine-readable companion is `m20/beir15_registry.json`; the dated amendment to the reserved
block of `m10/final_run_registry.json` is `_amended_2026_09_16`.

Reserved access was verified **UNSPENT** when this was written: no `m8-reserved-spent` tag locally
or on `origin`, and `results/m10_final_run.json` ends `INCOMPLETE_RESERVED`. The executing session
re-verifies this immediately before the tag.

## What this registration does and does not change

It **adds** five descriptive systems to the reserved roster and registers a broad BEIR-15
descriptive pass and an archive. It **does not** change R1, R2, the NDO-3 weights, `B`, the seed,
the interval method, `alpha`, the four reserved datasets, the trigger, the six-set decision, the
partitions, the clean-4 headline or any release rule. It produces no new estimand. Every row it
adds is descriptive with `alpha = 0`.

The reserved four are scored exactly once, inside the tagged transaction. BEIR-15 copies those rows
into its tables; it never opens a reserved payload and never re-scores a reserved dataset.

## 1. Roster — eight systems

Registered order, identities pinned literally in `m20/beir15_registry.json` §`systems`:

| # | system | query side | document tower | new work |
|---|---|---|---|---|
| 1 | `nano-dense` | frozen M13 checkpoint, fp32 on CPU | Stella `ffeb2b7e` | inherited |
| 2 | `zero-dense` | released `constella-zero`, int8 table | Stella `ffeb2b7e` (reused) | R20 |
| 3 | `stella-query` | Stella query tower with its s2p prompt | Stella `ffeb2b7e` (reused) | R22 |
| 4 | `bge-small-en-v1.5` | bge-small `5c38ec7c` | same model | inherited |
| 5 | `leaf-ir-asym` | `MongoDB/mdbr-leaf-ir` `4262131b` | Arctic-M `e58a8f75` | inherited |
| 6 | `bm25` | `bm25s`, M7 frozen defaults | corpus text | R22 |
| 7 | `zero+bm25 dbsf@100` | derived | — | R22 |
| 8 | `nano+bm25 dbsf@100` | derived | — | R22 |

Three systems share one Stella document index. That reuse is the point of the project and is
registered explicitly: **Zero and Stella-query do not authorize a second corpus-scale Stella
encode.** The inherited prohibition on *substituting* `zero+bm25` for the dense Zero row stands —
both are reported, neither replaces the other.

### `constella-zero` (R20)

Hub revision **`ebee6ea94999f26182505548ffc5df035e727351`**, variant **int8**, loaded through
`m11/release/zero_encoder.py`. Per-file SHA-256 for all eleven files is in the registry. The
published `model.npz` hashes to `a7007b1a6af120b976f093fd69ddcb5001996ec0b84b5864b4fd25d7af878abf`,
which is `m7/FREEZE.json`'s `table_sha256`: the released bundle and the frozen M7 artifact are the
same bytes. Preprocessing is the frozen M7 rule — `pool_mode` sqrt, empty prefix, special tokens
on, max length 512, no padding. int8 is registered because int8 is the artifact every published
zero number was measured on.

Parity requirement, checked before the run: the released encoder must reproduce
`m7src/table.py:load_table(variant="int8").encode` under `m7/FREEZE.json` preprocessing to
max-abs ≤ 1e-5 on a fixed non-benchmark probe set. Measured **1.49e-08** on 2026-09-16.

### `stella-query` (R22)

`NovaSearch/stella_en_400M_v5` at `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`, through
`m7src/teacher.py:encode`. The query prompt is pinned literally, exactly as
`m7src/encoders.py` has carried it since M7, prepended to the raw query with no separator:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery:
```

(trailing space after the colon). Tokenization: `add_special_tokens=True`, `truncation=True`,
`max_length=512`, right truncation, padding to the longest member of each batch. Compute: fp32 on
CUDA, TF32 matmul disabled, mean pooling, `2_Dense_1024` post-pooling Dense, L2 normalize — the
same contract as the document encode, so query and document live in one space. This is the
symmetric teacher, a descriptive ceiling reference, not a deployable comparator.

### `bm25` and the two fusion rows (R22)

BM25 is `bm25s`, method `lucene`, `k1 = 1.2`, `b = 0.75`, English stopwords, Snowball PyStemmer,
retrieval depth 1000, zero-score rows dropped, self-hits dropped — `m7src/fusion.py`'s frozen
defaults, unchanged. The zero-score filter and the self-hit drop are part of the frozen function,
not a post-hoc filter.

Fusion is `m12src/qfusion.py:dbsf`, parameter-free, at prefetch depth 100. **Each input run is
truncated to its top 100 before DBSF is applied**, because DBSF's mu and sigma are computed over
whatever the prefetch returned; truncating afterwards would be a different function. A document
absent from one prefetch contributes 0 to that prefetch's term — Qdrant's behaviour and M12's
registered choice, not an oversight.

Before DBSF runs on any new data, the extended code must reproduce a published `m12/six_dbsf.json`
per-dataset `dbsf@100` value to 1e-9. `m20src/dbsf_reproduction.py` is that prerequisite and
`results/m20_dbsf_reproduction.json` is its receipt. It answers three separate questions rather
than one: whether extending the executor moved the operator, what substituting the released
`constella-zero` bundle for the frozen table costs a reported number, and whether retrieving dense
at depth 100 equals retrieving at 1000 and truncating. Run on SciFact and NFCorpus on 2026-09-16,
all three deltas were exactly **0.000e+00**.

Convex0 remains M7's confirmatory fusion operator. DBSF@100 remains a disclosed product-policy
override with no equivalence interval. Nothing here changes either statement.

## 2. Datasets — BEIR-15

Fifteen public BEIR datasets, pinned by HuggingFace repository and revision in
`m20/beir15_registry.json` §`datasets`, all revisions re-read on 2026-09-16. Excluded, as R22
requires: BioASQ, Signal-1M, TREC-NEWS, Robust04 (not freely redistributable).

**CQADupStack counts as one dataset.** Its score is the **mean of the twelve per-forum nDCG@10
dataset scores** (BEIR convention), not a query-pooled mean. Per-forum rows are also reported. Two
forums (android, english) come from the reserved transaction; ten are scored in the BEIR-15 pass.

Every row carries its contact labels: `clean`, `reserved`, `M7-dev`, `teacher-disclosed`,
`comparator-training`, `source-family-contact`, `licence-unverified`.

Licence roles, disclosed on every table:

- **MS MARCO** — validation only. Primary source `microsoft.github.io/msmarco/Notice.html`,
  "non-commercial research purposes only", recorded in `research/m7-data-licensing.md` row 1 and
  its 2026-09-04 validation-use rule change.
- **Quora** — no primary-source licence; **evaluation-only row under R22**.
- **Climate-FEVER** — no affirmative licence; **evaluation-only row under R22**.
- **Touché-2020** — source-family contact with ArguAna; licence status disclosed.

HuggingFace wrapper licence tags are not licence evidence. The training-source rule is untouched:
no dataset in BEIR-15 supplies gradients, targets, negatives or generation seeds in any role.

Climate-FEVER's corpus is FEVER's Wikipedia. The reserved FEVER document shards are reused for it
**only if** its corpus id and text hashes match FEVER's exactly; otherwise it is encoded
separately. Its queries and qrels are public and are not reserved.

## 3. Crash rule for the reserved transaction

**`reserved.crash` in `m10/final_run_registry.json` governs**, per owner ruling **R23**
(2026-09-16): per-system atomic outputs; a crash after `m8-reserved-spent` resumes at the first
incomplete system under the identical tagged code, registry and pre-encode identity; a persisted
complete system is never re-scored or re-opened; no aggregate is emitted until all systems
complete; no flag permits a rerun. **R14 governed the six-set run only and is unchanged for it.**
No executor may treat a `reserved.crash` resume as a second access.

Before the tag, a crash is free: nothing has been spent and the pre-encode resumes at its first
missing shard.

## 4. Archive destinations (R22, deliverable 5)

Both targets receive the same hash-manifested tarballs, and re-hashing at both targets is the
verification. `results/m20_archive_manifest.json` lists every file, its size and its SHA-256.

- **Local:** `/mnt/d/constella-archive/beir15/` (Windows `D:\constella-archive\beir15\`).
- **Object storage:** prefix `constella-archive/beir15/` in an S3-compatible bucket on the owner's
  account, driven by `rclone`. **The bucket name and provider are recorded here by the executing
  session at creation time and before the upload**; no credentials enter git, logs, or the pod.

Layout below the prefix, identical on both targets:

```
beir15/
  MANIFEST.json                                   # copy of results/m20_archive_manifest.json
  datasets/<corpus>/corpus.jsonl.gz               # every corpus, from the pinned HF revision
  datasets/<corpus>/queries.jsonl.gz              # public corpora only
  datasets/<corpus>/qrels.jsonl.gz                # public corpora only
  vectors/stella-ffeb2b7e/<corpus>/manifest.json
  vectors/stella-ffeb2b7e/<corpus>/shard_00000.npy ...   # the hash-recorded fp16 shards, as written
  vectors/bge-small-5c38ec7c/<corpus>/...
  vectors/arctic-m-e58a8f75/<corpus>/...
```

gzip rather than zstd: it is in the standard library on every host involved, and the archive's size
is dominated by fp16 vectors that do not compress. Payload bytes are written with `mtime=0` so the
manifest hash is reproducible.

**The reserved four's queries and qrels are exported by the tagged transaction, not by the
archiving pass.** R22 asks for queries and qrels for every evaluated dataset, and dropping the
reserved ones would be a scope reduction no executor may make. They also cannot be read by a later
archiving pass without a second protected access. So the transaction writes them into the archive
layout itself, from the payloads it has already opened and authenticated for scoring: no additional
protected read, and the deliverable intact. The archive builder **requires** those files for a
reserved corpus and fails if they are absent; it never opens a reserved payload to fill the gap.
The frozen originals stay in `results/frozen_eval/`, hash-pinned in `results/eval_manifest.json`.

Verification is a re-hash at **both** targets: `--verify` re-hashes the local copy against the
manifest, and `--verify-remote` asks object storage for its own SHA-256 of every stored object and
compares. `rclone copy --checksum` alone is a transfer check, not the registered verification.

Stopped Runpod volumes are not the archive.

**Open dependency:** at registration time the box had no object-storage account, bucket or
credentials, and no `rclone` binary. The archive step cannot complete until the owner provides
them. It blocks neither the reserved transaction nor BEIR-15, which are ordered before it, and the
local `D:` half of the archive can be built and verified without it.

## 5. Budget

The $1,000 project ceiling is a ceiling, not a target. `results/m13_build_record.json` records
`committed_usd = 657.04` and `headroom_usd = 342.96` against it, with the reserved batch's 55.2
hours already inside the committed figure.

### The inherited 55.2-hour figure priced one document tower of three

`results/m13_encode_benchmark.json` derived `reserved_batch_allowance = 55.2 h` as 10M passages at
the slower measured Stella rate, doubled, plus setup. The reserved batch also needs **bge-small**
and **Arctic-M** document vectors for the same 10.1M passages, and those two encodes were never in
that arithmetic. M20 therefore registers its own stage caps rather than pretending one number
covers work it never priced. **M13's registered 55.2-hour figure is not changed**; it is applied
where its derivation is now comfortably sufficient — the tagged scoring stage.

This is a cost-bookkeeping correction, made pre-observation and dated. It changes no estimand,
weight, threshold, contrast or release rule.

### Measured inputs

| input | value | source |
|---|---|---|
| A100 Stella fp32 document rate (slower of two sizes) | 100.76 passages/s | `results/m13_encode_benchmark.json` |
| bge-small cost relative to Stella | 0.1349 | `results/m20_tower_rate_benchmark.json` |
| Arctic-M cost relative to Stella | 0.3308 | `results/m20_tower_rate_benchmark.json` |
| all three towers, combined | 1.4657 × Stella | derived |
| reserved four documents | 10,115,709 | `results/eval_manifest.json` |
| BEIR-15 documents not already in the reserved set | 23,744,806 | registry `budget.document_volumes`, verified at download |
| target pod compute + its storage | $1.6636111111/h | `results/m13_build_record.json` |
| the two other retained stopped volumes | $0.1389/h | 2 × 500 GB at $0.10/GB-month |
| all-retained hourly during an M20 session | $1.8025/h | derived |

The tower ratios were measured on the RTX 3080 box on SQuAD training passages, one tower per
process, no protected access. Only the ratio is carried; the absolute Stella rate is the A100
measurement. A first attempt measured Arctic-M at 15.7 passages/s because all three towers were
left resident on a 10 GB card at once — an allocator artifact that would have gone straight into
this cap. The corrected measurement is the registered one and the artifact is recorded in
`results/m20_tower_rate_benchmark.json`.

**Climate-FEVER is encoded, not reused.** Its corpus is FEVER's Wikipedia, but the published
document counts differ (5,416,593 against 5,416,568), so the reserved FEVER shards do not cover
it. Its 5.42M documents are inside the cap below. `m20/PLAN.md`'s sketch omitted them; this
registration supersedes that sketch. The executor still compares its corpus hashes to FEVER's and
records the comparison rather than assuming either answer.

### Registered stage caps

| stage | protected? | expected | **cap** |
|---|---|---:|---:|
| A. reserved corpora download, pre-encode | no, pre-tag | ≈ 45 h | **52 h** |
| B. tagged reserved transaction | yes | ≈ 5 h | **55.2 h** (inherited, unchanged) |
| C. BEIR-15: download, pre-encode, score | no | ≈ 113 h | **120 h** |
| D. archive build, transfer, re-hash verification | no | ≈ 8 h | **10 h** |
| **M20 total** | | **≈ 171 h** | **237.2 h** |

New spend is priced as

```
m20_total_hours × all_retained_hourly − inherited_reserved_allowance_hours × target_pod_hourly
= 237.2 × $1.8025 − 55.2 × $1.6636111111 = $335.72
```

against $342.96 of recorded headroom, so committed plus new is **$992.76** of the $1,000 ceiling.
The subtraction removes only the target-pod hours already inside `committed_usd`; the two sibling
volumes bill during those hours too and are **not** subtracted. An earlier version of this
registration priced 187 hours at the all-retained rate and so omitted 55.2 hours of sibling
storage, $7.67, which put the plan $1.77 over the ceiling. Corrected pre-observation after review.

The controller recomputes this from the **live** quote at launch and refuses if the live price
makes the registered plan cost more than registered, or if committed plus new would exceed the
ceiling. Each stage has its own wall clock: the controller hands the pre-encode processes one
absolute stage deadline and wraps each stage in `timeout`, so a slow pre-encode cannot eat the
tagged transaction's budget.

**Stage C's cap is about 1.06× its expectation, because the ceiling rather than the estimate sets
it.** BEIR-15's 23.7M new documents across three fp32 towers is the dominant cost of M20 and there
is no slack to absorb a slower-than-projected run. The reserved transaction is therefore ordered
first, so the irreplaceable access lands before this risk is taken. If the projection gate trips in
stage C the run stops cleanly and **the owner decides** between raising the ceiling, narrowing
BEIR-15, or accepting a partial descriptive table. No executor may decide that, and no executor may
change the fp32 encoding contract to buy time: fp32 with TF32 disabled is what the frozen
comparator vectors were made under, and mixing precisions inside one table is exactly the silent
inconsistency this project exists to avoid.

### In-run projection gate

A cap alone does not stop a run that is silently five times slower than planned. After every shard
the pre-encode projects, from the rate it has actually measured, when the **whole stage** will
finish: its own remaining documents at its own rate, plus every tower that has not run yet,
converted through the registered cost ratios. If that lands past the stage's absolute deadline the
run stops at a shard boundary and reports, rather than burning the cap to discover the same thing
later. Every shard already written is hash-recorded and resumable. Before the tag this costs
nothing; after it, the reserved result is already durable per completed system.

The registered document volumes it projects against are the ones the cap was built from,
10,115,709 for the reserved four and 23,744,806 for BEIR-15, not the corpora downloaded so far,
which would make the projection most optimistic exactly when least is known.

## 6. Order of execution

1. Stage A — reserved corpora pre-encode and BM25 index build. Unprotected, resumable, no access
   spent.
2. Stage B — the single tagged reserved transaction, eight systems, four datasets. `reserved.crash`
   governs. On completion, `results/m10_final_run.json` becomes `COMPLETE`.
3. Stage C — BEIR-15, clearly labelled, never mixed into M13's gates.
4. Stage D — archive to both targets, verified by re-hash.
5. Pod STOP confirmed `EXITED`; receipts, ledger boundary note, `m20/STATUS.md`.

The reserved transaction runs **first** so that the irreplaceable access lands durably before the
much larger and entirely repeatable BEIR-15 work.

## 7. Ledger boundary

Once `m8-reserved-spent` is on origin, FEVER, DBpedia-entity, cqadup-android and cqadup-english
join the known-test class for future projects and are **unusable for re-deciding anything in this
project**. `m20/STATUS.md` and the ledger record that boundary at completion.
