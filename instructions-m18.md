# M18 — Constella Zero for Qdrant project memory

**Active for planning, 2026-09-12 (Dylan).** Build an internal, fully specialized Constella Zero
query encoder for searching the `qdrant/qdrant` GitHub repository, issues and pull requests. Run
locally on this machine and reuse the released Zero and the M17 machinery wherever possible. This
session creates this instruction file only. It does not authorize M18 data collection, training,
evaluation reads, publication, or changes to another milestone.

M18 starts from `main` and is independent of the closed M17 experiment. Do not reopen M17, change
its records, or read any spent/protected M7–M13 evaluation surface. Create fresh M18 data,
registries, work directories and evaluation sets when execution is authorized.

## Owner rulings and scope

- The artifact is for internal evaluation by Andrey and the team. Do not publish or represent it
  as a released general-purpose model.
- Use the Qdrant OSS repository and its GitHub issues and pull requests as the complete initial
  source domain. Stack Exchange and broad software/DevOps corpora are out of the first M18 run.
  No licence investigation is required for this internal evaluation.
- Fully specialize for Qdrant project memory. General-domain retention is descriptive only and is
  not an eligibility gate.
- Keep Stella as the frozen document tower and keep the existing document-vector contract. A weak
  teacher ceiling is a recorded limitation, not permission to introduce a new document model or
  re-embedding architecture inside M18.
- Execute on this machine only. Do not use cloud training, hosted training jobs, or paid data.
- The primary deployed mode is hybrid retrieval using the existing BM25 + dense DBSF recipe.
  Dense-only quality must still be reported so lexical retrieval cannot conceal a broken table.

This narrow corpus is intentional. The desired query distribution and memory collection are one
project, and its repository discussions contain the project's real vocabulary, aliases, errors and
question forms. A broad software corpus would mainly buy out-of-project transfer while increasing
ingestion, deduplication and curriculum work. Add outside data only in a later milestone if a
measured Qdrant-specific gap cannot be filled from Qdrant sources.

## Product contract

Given a natural-language question, error excerpt, configuration term, API name, acronym or short
identifier about Qdrant, retrieve the answer-bearing project artifact from a pinned snapshot of:

- documentation, READMEs, changelogs and design material in `qdrant/qdrant`;
- issue titles, bodies and comments;
- pull-request titles, bodies, review comments and linked discussion; and
- useful source metadata such as paths, symbols and surrounding explanatory comments.

The candidate encoder must emit the same normalized 1,024-dimensional space expected by the
frozen Stella document index, export as an int8 resident lookup table, and preserve Zero's
near-zero query compute. The serving bundle must include the exact tokenizer and preprocessing
fingerprint used in training.

M18 has two independent deliverables. The project-memory system — pinned snapshot, parsed corpus,
Stella index, BM25 index, DBSF query path and evaluation set — ships for internal use even if no
specialized Zero table beats v1. The encoder experiment separately records whether v1, an
untrained vocabulary extension or a trained specialized table is the preferred dense component.

This remains a context-independent, order-insensitive pooled token table. It may retrieve prose,
errors and identifiers associated with code, but M18 must not claim semantic code search or robust
disambiguation of an overloaded term without query context. Hybrid retrieval is the default
because BM25 is well suited to exact literals, hashes, paths and rare identifiers.

## Reuse before invention

Use the existing implementation surfaces as building blocks rather than designing a new stack:

- `m7src`: frozen teacher/index contract, query table, retrieval, fusion and statistics;
- `m10src`: corpus admission and packing, target generation, student training and resumability;
- `m12src/qfusion.py`: Qdrant-compatible RRF/DBSF implementation;
- `m11/release`: Zero bundle and release verification patterns; and
- `m17src`: candidate cache, exact-token vocabulary extension, training, export, NumPy loader and
  evaluation patterns.

Put M18-specific source adapters and orchestration in a new `m18src` namespace. `m17src` is a
milestone-bound fork whose training, cache, export and evaluation paths enforce the closed M17
registry and protected-screen state; do not spend the run trying to import through or relax those
guards. As execution step 0, copy the minimum runnable M17 path into `m18src`, replace its
milestone-specific registry/lock hooks with fresh M18 hooks, and retain the numerical logic and
tests that still apply. Record the copied files and semantic differences. Import unbound helpers
such as `m12src/qfusion.py` directly where practical.

The released Zero v1 bundle is the initialization and required baseline. M17's five trained
endpoints are negative historical evidence, not M18 initialization candidates. M17 showed that a
falling training loss can coexist with worse retrieval; M18 therefore evaluates early checkpoints
instead of assuming the final step is best.

## Source snapshot and corpus

Before any generation or training, create a registry that pins:

- the exact `qdrant/qdrant` commit and default branch;
- the GitHub API/export cutoff time, query parameters, pagination receipts and raw-data hashes;
- parser, deduplication, chunking and redaction versions;
- the released Zero v1, Stella teacher and BM25 configurations; and
- seeds, tokenizer fingerprint, preprocessing fingerprint and planned evaluation reads.

Acquire repository text from the pinned local Git checkout and discussion data through the
authenticated GitHub API with explicit pagination. Include open and closed issues/PRs, issue
comments, review comments and cross-reference/timeline links needed to connect questions to fixes.
Canonicalize by GitHub node/object ID so pull requests returned by issue endpoints are not indexed
twice. Save resumable page receipts and verify the observed object counts before parsing.

Keep raw snapshots immutable. Derived records must retain their repository path or GitHub object
ID, timestamp, author-role metadata when available, outbound links and parent thread. Strip bot
boilerplate, quoted-history duplication, generated logs that contain no useful explanation and
obvious credentials or secrets. Preserve useful stack traces, commands and error codes.

Prefer natural-language-bearing units: documentation sections, issue/PR openings, answer-bearing
comments, review threads, changelog entries and explanatory code comments. Keep code blocks with
their surrounding prose and store exact paths/symbols as searchable metadata. Avoid flooding the
dense index with isolated source lines; exact source search remains a lexical strength.

Index every qualifying issue, pull request and repository artifact in the pinned snapshot. Do not
sample away project artifacts to meet a document-unit target. If fine-grained chunking would exceed
200,000 document units, merge adjacent low-information chunks within the same artifact while
retaining artifact-level coverage and stable identifiers. Cap training query views, not the search
collection, at 300,000 in the first run; sample those views by artifact and time with fixed seeds,
retain all rare-token strata, and record the dropped view counts. Do not fill unused capacity with
unrelated data.

## Leakage-safe examples and splits

Split by GitHub thread or repository artifact before producing query variants. Prefer a temporal
split so later issues and PRs test retrieval over older project memory. Near-duplicate text,
backports, mirrored release notes and quoted comments must remain in one split.

Training pairs should come from evidence already present in the project:

- issue/PR questions or titles paired with answer-bearing maintainer comments, linked fixes,
  documentation or closing artifacts;
- documentation headings and sections paired with short question, task, error and acronym forms;
- review questions paired with the resolving reply or changed explanatory material; and
- aliases, renamed settings and old/new terminology paired only when the project gives evidence
  for the relationship.

Do not create trivial self-retrieval labels by putting the same title or query sentence verbatim in
both query and target. Separate the question from the answer-bearing span, and disclose exact-text
overlap rates. Use repository links, resolution structure and maintainer answers as weak labels.
Synthetic query generation with the already-cached local Qwen teacher is optional only to fill a
measured shortage in a query stratum; generated text must remain tied to a source span and cannot
define audit relevance by itself.

Create a fresh Qdrant-specific development-selection set and a sealed confirmation/audit set.
Target 200–300 total queries. Put at least 25 queries from each available stratum in development
and at least 10 in confirmation; report smaller realized strata rather than manufacturing filler.
Use these strata:

1. concepts and how-to questions;
2. errors and troubleshooting;
3. configuration, API and operational questions;
4. exact identifiers, versions and numeric distinctions; and
5. abbreviations, aliases and project jargon.

Use deterministic structural qrels first. A local model may adjudicate ambiguous candidates from
the source text without seeing model identity; human review is optional and capped at a small
spot-check rather than being an execution dependency. Record relevance rules and model-judgment
provenance. If Andrey or the team voluntarily provides 30–50 real questions before the split is
sealed, hold them out as a query-shape slice; do not wait for them or require anyone to author
questions. The confirmation/audit set is read once after the recipe, checkpoint rule and
eligibility threshold are locked.

## Vocabulary and tokenizer policy

Run the M17 fragmentation/support inventory on M18 training text only. Add exact `AddedToken`
entries with stable old token IDs for frequent, retrieval-relevant units that the base tokenizer
shatters, including supported examples such as `qdrant`, `hnsw`, `k8s`, `s3`, `grpc`, `rocksdb`,
`wal`, `mmap`, `cuda`, versioned model or API names, dotted settings and common Qdrant compounds.
An example is eligible because it appears with enough support and residual value in the pinned
training source, not merely because it is on this list.

Extend M17's term extraction to cover colon-delimited identifiers and useful dotted, underscored,
slashed, plus, hash and hyphenated compounds. Unit-test representative terms such as
`std::vector`, `torch.compile`, `foo_bar`, `CUDA_ERROR_OUT_OF_MEMORY` and `gpt-4o-mini`. Do not add
whole UUIDs, commit hashes, timestamps, arbitrary paths, random pod suffixes or other near-unique
strings as vocabulary rows. Report added rows, table bytes, support distribution and tokenization
before/after. Zero has no 35M-parameter cap; resident bytes and measured latency are the applicable
limits.

Token shattering is fixed primarily by learning an exact row for a meaningful whole term. The
first M18 candidate therefore keeps raw numbers and uses exact-token extension. It does not make
global digit removal or global numeric normalization the default.

### Tokenizer/preprocessing variants

Build these narrowly defined variants. `V0-Q` is a first-class baseline: it uses the T0 tokenizer,
keeps every inherited v1 row unchanged, initializes each new exact-token row with M17's
count-weighted composition rule, and performs no optimizer step. It may be selected as the final
internal encoder if training does not improve it.

- **T0 — exact tokens, raw text (primary):** vocabulary extension as above; preserve every digit.
- **T1 — typed ephemeral normalization:** start from T0, but normalize only UUIDs, long commit or
  hex hashes, timestamps, memory addresses and demonstrably generated random suffixes into
  separate typed placeholders. Preserve versions, ports, HTTP/gRPC status codes, quantities,
  dimensions and alphanumeric product or API names such as `S3`, `EC2`, `IPv4`, `SHA256`, `A100`,
  `GPT-4` and `CUDA 12.6`. The Stella teacher and indexed documents receive raw text; student
  training and serving queries receive the identical registered normalizer.
- **T2 — continuation-digit suppression (diagnostic):** start from T0 and mask only tokenizer rows
  matching `^##[0-9]+$` during student pooling. Preserve standalone numeric tokens and all
  alphanumeric exact tokens. Remove a masked piece from both the pooled sum and token count, and
  apply the identical rule at serving time. T2 is a falsifier for the claim that residual
  continuation-digit fragments are harmful; it is not presumed to be a fix.

The phrase *digit-row suppression* is otherwise ambiguous. In M18 it means the exact T2 rule
above. Suppressing a pooled row largely removes that fragment's contribution after final L2
normalization; it does not combine `s` and `##3` into a meaningful `s3` representation. It can
collapse distinctions such as `S3` versus `S4` or `EC2` versus `EC3`, so the exact-token solution
is preferred and the numeric/identifier slice must gate T2.

Numeric normalization reduces sparsity by mapping many values to a placeholder, at the cost of
making distinct inputs more alike. T1 limits that trade to values that usually identify an event
instance rather than a product, protocol, version or error. Before training T1, automatically emit
a collision audit with representative raw examples, normalized forms and relevance groups. Reject
any pattern that merges meaningfully distinct named entities or answer sets; do not require a
manual review unless the automatic report exposes an ambiguous high-frequency pattern.

## Fixed training recipe

Avoid a new hyperparameter search. Start from the released M7 Zero v1 unfolded checkpoint. Freeze
every inherited v1 token row and the inherited global pooling/scalar parameters. Train only the new
exact-token or typed-placeholder rows introduced by the selected variant. This single gradient mask
is the required M18 training change: it lets project terms specialize while preserving the ordinary
language rows that make differently phrased questions work. Do not add an anchor; frozen inherited
rows make it unnecessary.

Use M17's cosine target, hard-candidate listwise objective and supported alias consistency on the
trainable rows. Use these fixed values unless an implementation incompatibility is documented:

- cosine distillation weight `1.0`;
- hard-candidate listwise weight `1.0`;
- alias-consistency weight `0.1` when evidence-backed alias pairs exist;
- initialization-anchor weight `0.0`;
- new-row learning rate `3e-4`;
- batch size `256`, warmup `200`, maximum `4,000` optimizer steps; and
- checkpoint reads at step `0`, `250`, `500`, `1,000`, `2,000` and `4,000`.

Confirm these values against the reusable driver before locking the registry; if a named option has
a different meaning in code, document the mapping rather than approximating it silently. Build the
Stella target/candidate cache once from raw queries and share it across T0–T2. Each tokenizer arm
gets separate student token IDs, caches, fingerprints and export paths. Because inherited rows are
frozen, training batches must contain at least one trainable new row per query; report how many
source queries are eligible rather than padding the pool with zero-gradient examples.

Before a full screen, run T0 only through step 500 and read V0-Q/step 0, step 250 and step 500 on the
development-selection set. If dense quality is below step 0 at both trained checkpoints and fusion
has not improved, halt learned-arm training. Produce one automated parity report comparing:

- step-0 training-time forward output with the exported serving loader on identical inputs;
- V0-Q with v1 on queries that contain no added token, within the existing int8 tolerance; and
- the training-time forward output with serving output at the failing trained checkpoint.

If parity fails, fix the mismatch and repeat this 500-step diagnostic once. If parity passes, record
that learned specialization did not help and continue with V0-Q and the project-memory system; do
not spend the remaining time on open-ended diagnosis. This is the one required early training
check because it directly catches the unexplained M17 failure mode.

If T0 passes the diagnostic, screen T0–T2 with one fixed seed and select checkpoints on the
development-selection set. The T1 and T2 implementations should first be evaluated at step 0; an
arm whose transformation affects fewer than 1% of development queries may be recorded as inert and
skipped. Confirm only the selected trained form with one additional seed. Do not launch
architecture searches, broad source-mixture sweeps, new teachers or an arm farm. Every run must be
resumable, use atomic cache writes and record peak RAM, VRAM, disk, elapsed time and table size.

## Evaluation and decision rules

Before student training, measure the frozen Stella ceiling on the fresh development-selection set.
Report BM25, released Zero v1 dense, V0-Q dense, Stella dense, and DBSF@100 fusion for each dense
model. Interpret the ceiling by stratum: dense retrieval should add the most value for concepts,
how-to questions and differently phrased troubleshooting, while BM25 may properly dominate exact
identifiers. A weak macro ceiling is diagnostic and does not stop construction of the system or
automatically stop the bounded encoder run. Stop only for a demonstrated parity/corpus defect; do
not change the document tower inside M18.

Primary selection metric: fused nDCG@10 using the existing DBSF@100 recipe. Also report fused
Recall@10, dense nDCG@10/Recall@10, MRR@10, latency, resident bytes and results for every query
stratum. BM25 parameters and candidate depth remain identical across comparisons.

A candidate, including V0-Q, is eligible only when, on development-selection data:

- its fused nDCG@10 exceeds released Zero v1 + DBSF by at least `0.010` absolute;
- its dense nDCG@10 exceeds released Zero v1 dense by at least `0.005` absolute;
- no adequately sized registered stratum has both a negative point delta and a paired bootstrap
  interval wholly below zero versus the v1 counterpart;
- for a trained candidate, the direction of fused improvement agrees across both training seeds;
  and
- export parity, tokenizer parity, int8 error, latency and resident-size gates pass.

T1 or T2 replaces the corresponding raw-text candidate only if it improves fused nDCG@10 by at
least `0.005` absolute and does not reduce the exact-identifier/numeric stratum. Otherwise prefer
the simpler raw-text form. If no specialized candidate is eligible, use released Zero v1 in the
finished hybrid project-memory system and record the encoder result as `NO_IMPROVEMENT`.

After choosing the recipe and checkpoint without confirmation/audit access, read that set once.
Require a positive fused and dense delta versus v1, no stratum whose paired interval shows a clear
regression, and serving/export parity. Treat the fixed development margins as practical selection
screens rather than statistical claims; report paired bootstrap intervals. General-domain v1
checks are unnecessary unless they diagnose a concrete serving problem.

## Local execution envelope

Plan for a maximum of 24 elapsed local-compute hours on the RTX 3080 host, excluding time waiting
on the GitHub export and any optional human question contribution. Use the existing cached Stella
and Qwen weights. Budget the work in this order:

1. copy the minimum M17 execution path into M18 and run one synthetic end-to-end rehearsal;
2. snapshot, parse, deduplicate, index and freeze splits;
3. build the fresh development/confirmation protocol and system baselines;
4. build shared teacher/candidate caches and V0-Q;
5. run the T0 step-500 diagnostic;
6. if it passes, screen T0–T2 and confirm the winner with the second seed; and
7. export the best eligible encoder, finish the hybrid system and write the internal report.

Stop before training if source provenance, split leakage or qrels cannot be made credible. The
project-memory system still lands if training stops or no arm qualifies. One synthetic rehearsal
and the step-500 diagnostic are sufficient; add another smoke test only when it distinguishes a
specific live failure. Do not spend unused time inventing additional recipes.

The execution agent should make routine implementation choices autonomously. When two compatible
paths exist, choose the smaller change that reuses a tested repository primitive, record the choice
in the M18 ledger and continue. Ask Dylan only before expanding beyond Qdrant data, changing the
frozen document tower/index contract, using non-local compute, publishing an artifact, or relaxing
an eligibility/audit rule after observing results.

## Required M18 artifacts when execution is authorized

Create and maintain at least:

- `m18/STATUS.md`, `m18/PLANNING.md`, `m18/registry.json`, `m18/LEDGER.md` and `m18/CODEMAP.md`;
- immutable source receipts, manifests, split/deduplication reports and a tokenizer collision audit;
- a one-command resumable rehearsal that uses synthetic fixtures and no protected data;
- machine-readable baseline, checkpoint, seed, stratum and audit results;
- an internal int8 bundle with tokenizer/preprocessing/config hashes and a NumPy loader; and
- `m18/FINDINGS.md` stating what worked, what failed, limitations and the exact decision.

The final status records both outcomes: `SYSTEM_READY` or `SYSTEM_BLOCKED`, plus
`ENCODER_INTERNAL_CANDIDATE` or `ENCODER_NO_IMPROVEMENT`. `SYSTEM_READY` includes a usable snapshot,
index and hybrid query path even when its dense component remains released Zero v1. An internal
candidate remains non-public and does not replace the public Zero v1 release without a separate
owner ruling.
