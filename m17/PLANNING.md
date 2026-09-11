# M17 — Zero v1.1 plan

**Recommendation: keep Zero's serving design, add a small supported vocabulary across domains,
and compare joint table training with and without hard-candidate listwise distillation.** The
owner-approved plan also includes alias consistency training, fixed late-checkpoint averaging
and an int8 resident-row loader comparison. Spend the future 72 hours on the local RTX 3080,
counting preparation and validation in that window.
The likely deliverable is better coverage with a modest quality improvement, if measured;
matching the teacher on every query type is an aspiration, not an established attainable bar.

**Revision A2, 2026-09-11.** Dylan accepted all six review recommendations: route the screen on
the pinned development suite and keep the small M17 panel descriptive; build the judged panel
before the 72-hour clock; admit Kubernetes documentation in principle as a small technical slice;
cut the final dose and add a train/held-out divergence check; treat the anchor as inherited and
spread the averaged snapshots; record candidate entropy at cache build. Two extras were added:
an untrained sum-init export (V0) and a per-domain cap on new rows for breadth. The S3/k8s
vocabulary is an **internal need, not a headline claim**; new rows must stay broad across domains.
The earlier constants remain in git at `b9d355e`.

This session is planning only. The proposed constants live in [registry.json](registry.json),
marked `DRAFT_NOT_EXECUTABLE`. No trained candidate or quality result exists. Small diagnostic
observations live in `results/m17_*.json`; [FINDINGS.md](FINDINGS.md) explains their limits.

## Product contract and Nano

Zero v1.1 still emits one normalized 1024-dimensional vector via tokenization, row lookup,
sqrt-count pooling and normalization. It keeps the same query API, empty-input behavior,
512-token limit, int8 option and fixed Stella document space. A new tokenizer ships with its
matching new rows; every existing token ID remains stable. No document retraining, document
re-encoding, new production index, new query network or query reranker is part of this plan.

Nano's M13 model and protocol remain unchanged. Its bge-small tokenizer has exactly the same
30,522-token mapping as Zero v1 locally. Stella also uses that WordPiece vocabulary. Nano and
Stella have contextual transformers, whereas Zero combines context-independent rows. Thus all
three can tokenize `s3` and `k8s` today; the question is whether their representations capture
the intended concepts. A larger Zero tokenizer neither updates Nano nor requires Nano to
change. Preserve M13's frozen Zero v1 comparator; any later v1.1 comparison is a separate row.

The quality goal is retrieval, with latency and memory as regression checks. More rows must earn
their bytes through useful behavior. Keep the existing M12 DBSF@100 product fusion unchanged
and measure its effect with the new table; no fusion sweep.

## Planned levers

| Option | Evidence and decision |
|---|---|
| Supported vocabulary + joint row training | Priority. M8 tested frozen-row closed-form extensions and lost. A trained joint tokenizer/table variant remains unrun in `instructions-m17.md`. It changes the experiment rather than overturning that result. |
| Hard-candidate listwise distillation (`R-LIST`) | The best supported quality hypothesis. M8's actual entropy artifact shows nearly one-hot teacher distributions under the old uniform bank; teacher-neighbor candidates carry more information. No measured gain from training this variant yet. |
| Alias consistency | Added to the plan by Dylan, 2026-09-11. One matched extra arm teaches independently verified equivalent query forms to agree, with no inference rewriting. |
| Late-checkpoint averaging | Added to the plan. Compare one fixed average with the ordinary final checkpoint; export one table, with no runtime ensemble. |
| Int8 resident-row loading | Added to the plan. Decode only queried rows; retain the implementation only after parity and measured RAM/latency checks. |
| More general data with the old objective alone | Included as the matched continuation control. M8's data-only gains were small; do not use the entire budget merely repeating an exhausted loss. |
| Wholesale tokenizer replacement / 64K rows | Defer. More retraining and matching risk; 64K full-width rows exceed the current parameter cap. It would not mathematically require a new document index, but is unnecessary for the first bounded extension. |
| Prefix changes, pooling search, projections, additive frozen n-grams | Defer. Preserve existing behavior; several are closed or absorbable. No broad hyperparameter sweep. |
| Static late interaction | Research conclusion below: genuine token-level matching changes the document representation or serving, outside this update's contract. |

The external literature supports the *method class*, not a gain forecast for this system.
[RocketQAv2](https://aclanthology.org/2021.emnlp-main.224/) studies listwise distillation with
candidate lists in a different teacher/student setup. The local reason to test it is
[M8's entropy audit](../results/m8_b2_entropy.json), not borrowing its benchmark gains.
Source notes: [vocabulary](../research/m17-vocabulary-2026-09-11.md),
[interaction/loss](../research/m17-static-interaction-2026-09-11.md),
[data/training](../research/m17-data-training-2026-09-11.md),
[accepted follow-up ideas](../research/m17-additional-avenues-2026-09-11.md).

## Vocabulary: coverage beyond DevOps

Allow **up to 3,072 new lexical units**, including a technical priority allocation and an
evidence-driven remainder. With one learned scalar per row, the largest proposed table has
34,433,850 trainable parameters, within the 35M cap. The int8 codes/scales grow from about
31.4 to 34.5 MB. The current numpy loader expands them to fp32 RAM: about 125.0 to 137.6 MB
for rows alone. These are array sizes, not total process RSS or full bundle sizes.

Audit six domain slices: cloud/software, everyday queries, science/engineering, medicine,
finance and legal. Also stratify by abbreviations, identifiers/punctuation, proper names,
multiword concepts, repeated terms and long queries. English is the teacher's stated scope;
incidental foreign terms can be inspected without promising a multilingual release.

Use admitted **training text only** to discover candidates. Record document/context support,
original token fragmentation, existing row update exposure where available, and the teacher-to-
Zero vector residual on those training queries. Fragmentation is a diagnostic, not the objective:
M8 reduced it without improving retrieval. Inspect high-residual, already-single-token terms too;
their remedy may be better examples and row updates, with no vocabulary addition.

Rank candidates deterministically by support-weighted teacher residual, then document support,
then lexical tie-break. Apply the registry's support minima and technical priority allocation;
fill remaining slots from all domains. No single domain, cloud/software included, may supply
more than the registry's per-domain cap: the technical allocation is a ceiling, not a target. Count distinct source documents and query contexts after
deduplication, not repeated template exposures. Add fewer rows when support is thin. Do not fill
the cap with obscure terms, reassign `[unused]` IDs, or select words from the audit panel.

Prefer complete lexical units such as `s3`, `k8s`, `kubernetes` and `kubectl`. Add phrase rows
only where the same simple tokenizer matching path supports them and source data establishes
need; an elaborate phrase parser is not a deliverable. Preserve standard WordPiece fallback for
unlisted terms. `single_word=True` prevents inside-word matches; normalization and punctuation
still need fixture checks in the serialized runtime.
[Official AddedToken semantics](https://huggingface.co/docs/tokenizers/api/added-tokens).

Test `S3`, `s3://bucket`, `s3-compatible`, `s3_client`, `xs3`, `k8s-client`, C++/C#/.NET,
and ambiguous senses such as an S3 heart sound versus storage. Do not globally rewrite `S3`
to “Amazon storage”: that would discard legitimate meanings. A static row cannot disambiguate
all senses by itself. Existing context rows and mixed-sense training examples still matter.

Initialize a new row from the **sum of effective constituent rows**, then jointly update old
and new rows. This is an inexpensive warm start, not an accuracy improvement. Under sqrt-count
pooling, constituents shared with other terms break universal sum-initialization equivalence;
record such drift before training. Repetition of an isolated phrase alone need not break it.
For unfolded checkpoints, account for learned scalars exactly once; initialize each new scalar
to one. The historical frozen-row D2 experiment remains closed.

Before any training, export the extended tokenizer with sum-initialized rows exactly as a
release would be folded and quantized (**V0**) and read it once on the development suite. It
costs minutes and separates what tokenization alone does from what joint training adds. V0 is
descriptive and never a shippable candidate.

## Data and training: reuse first

Start from the verified **unfolded** `p35w-2m-s2500` training checkpoint and its learned weights,
under a new M17 run ID and optimizer. Verify that its folded int8 serving output reproduces v1
before the first update. Do not feed a folded release back into the resume loader or mutate its
sidecars. Keep the teacher revision, query instruction, document pooling/projection and dtype
consistent with existing caches. Changing the student tokenizer does not invalidate cached
teacher targets for unchanged raw text.

Build a bounded query pool using existing admitted sources. General replay dominates, with a
smaller coverage slice spanning technical and other underrepresented terms. Prefer real QA
queries and varied title/heading/span views of admitted documents. The latter supply teacher
targets, not automatically trustworthy relevance labels. Reuse existing admitted generated
queries only with their lineage and filtering intact; no large LLM-generation project.

Include independently checked, context-valid pairs such as `k8s ingress` / `Kubernetes ingress`
and `S3 bucket policy` / `Amazon S3 bucket policy`. Half the targeted-coverage slots are paired
views, as defined in the registry; the general replay share stays unchanged. Both views count
toward the query cap and total exposure, and **every arm sees the same views in the same order**.
Keep both views of a same-intent query family in one split; different held-out contexts may
use an alias that appeared in training. Do not infer equivalence from teacher agreement alone,
relabel a teacher hit as relevant, or expand ambiguous acronyms globally. Pair admission,
provenance, decontamination and per-domain/sense counts are part of the data exit.

The first data exit is an admitted-source support manifest with per-domain document and query
counts after deduplication. An unpopulated domain is a recorded gap, not grounds to silently
fill from unapproved sources. Source-group and near-duplicate separation must precede training, vocabulary discovery and
candidate-bank construction. Use existing protected fingerprint/ban products through their
approved interfaces; do not open protected payloads to rebuild those products. If the artifacts
cannot safely screen new data, defer those inputs until a separately authorized process exists.
No non-commercial text may become a target, negative, training input or generation seed.

Approved Wikipedia/dataset material is the default and supplies the general replay. The
currently admitted sources (HotpotQA, SQuAD, FEVER-train, Mr.TyDi, ESCI) are unlikely to give
terms such as `k8s` or `kubectl` twenty distinct documents, so on 2026-09-11 Dylan **admitted in
principle** official Kubernetes documentation (CC BY 4.0, attribution required) as a small
technical slice, capped by the registry's new-source share. One further CC BY or Apache
project-documentation source may be named at the support-manifest stage under the same
standard. Licence evidence is recorded in `research/m7-data-licensing.md` before any download.
AWS public pages are not presumed licensed training material. If admitted data still cannot
support a domain, report the coverage gap rather than filling it from unapproved sources.

Use a fixed admitted document-vector bank, sampled across sources and including every known
positive used by the selected labeled-query subset **inside** the bank's document cap. Choose
that subset before lock; do not silently discard a label to squeeze its query into the bank.
Mine once using frozen v1 and frozen Stella query vectors. Each training query gets a
small deterministic union of teacher neighbors, v1 neighbors, random documents and its known
positive, as defined in the registry. Query-only examples use a separate declared mixture with
no positive slot. Cache `has_label` and nullable `positive_id`; a teacher hit never becomes a
relevance label. Dedupe and fill to the fixed size. The bank is a training
approximation: its top hits are not necessarily the full corpus's hard neighbors.

Store candidate **IDs and teacher scores**, not a copy of every document vector per query.
The maximum bank's fp16 vectors occupy 512 MiB; query-specific candidate vectors can otherwise
grow to tens of GB. Gather by ID in batches. The bank is still about eight times smaller than
M7's, so candidates are softer than the B2 top-200 measurement; record teacher entropy and
p_max quantiles at cache build in the B2 format regardless of the stop rule. Score with the teacher's existing query vector dot
frozen document vector. No new cross-encoder or late-interaction teacher.

For the listwise arm, use normalized-query cosine loss plus KL from the teacher distribution
to the student distribution over that identical candidate list, with one fixed temperature.
The registry states the logits, KL direction/reduction, cosine target and effective-row anchor
exactly. Both losses use normalized query vectors and the same frozen document-vector cache;
the cosine target is Stella's query, not a positive document or document centroid.
Keep the initialization anchor inherited from M7's recipe for continuity of the control; at
the registered weight and the table's row scale it is effectively inert and is not drift
protection. Soft distributions may assign mass to several
relevant documents: do not relabel every nonpositive candidate as a hard negative or blindly
apply the old InfoNCE false-negative mask to the soft target. Reject known contradictory labels
or invalid/self documents under the admitted dataset's rules. If candidate distributions remain
near-one-hot under the registered diagnostic, stop the listwise branch; do not spend hours on
an inert loss or tune temperature repeatedly on evaluation data.

Only the alias arm adds the registry's fixed-weight mean pairwise cosine disagreement between
the two student query vectors. Both views retain ordinary teacher and candidate-list losses.
Paired views occupy existing batch slots, so this requires no separate student forward; their
teacher encodes and candidate construction still count against preparation time. Pair IDs and
the admitted equivalence manifest are included in cache/resume provenance. This is a bounded
loss comparison conditional on vocabulary expansion and listwise supervision, not a full
three-factor sweep.

## Small experiment matrix and decisions

The frozen int8 v1 is the product baseline. Five short runs share raw examples, order, seeds,
step budget, warm start, pooling, optimizer and candidate cache:

| Arm | Vocabulary | Loss | Question |
|---|---|---|---|
| C | Original | Cosine + initialization anchor | Does continuation/data alone help? |
| V | Extended | Same as C | Does joint vocabulary training add value over C? |
| L | Original | C + informative listwise KL | Does candidate supervision help over C? |
| VL | Extended | Same as L | Does their combination beat simpler choices? |
| VL-A | Extended | VL + alias consistency | Does explicit equivalence supervision help over VL? |

All rows and scalars can update; V/VL/VL-A do not freeze the original table. No automatic return to
the historical B→A schedule or final contrastive A phase: that would confound the new loss and
could erase a small gain. C is a matched continuation control, not a claimed reproduction of
M7's entire historical training recipe. Compare V−C, L−C, VL−V, VL−L and VL-A−VL descriptively,
and each against frozen v1. These are exploratory comparisons, not confirmatory superiority tests.

Screen decisions are read on the **pinned development suite**: its software components for
technical routing and its full macro for general regression. The M17 panel's selection half is
read once per arm and reported beside it, but with about fifty queries per domain its paired
standard error is roughly ten times the tie band, so it does not decide.

At the short-run endpoint choose at most one expanded candidate and its closest simpler
control, using the registry's fixed control map (VL-A pairs with VL). Prefer fewer rows/fewer
loss terms inside the declared tie band. VL-A must improve technical retrieval over VL beyond
that band, satisfy general dense/fused tolerances against v1, and avoid a drop on the independently
judged ambiguous-sense selection slice versus VL. A vocabulary arm must help technical retrieval
or a predeclared independently labeled coverage slice, while
staying inside general regression tolerances; fewer tokens or higher embedding cosine alone
do not qualify it. If V/VL/VL-A lose to C/L, record that vocabulary expansion failed; a better L-only
model is a partial outcome, not completion of the requested vocabulary upgrade.

Run the chosen candidate and matched control under the same longer schedule, with seed 0 and
one independent seed. Restart both from their declared initialization for that longer schedule;
do not append steps to an exhausted scheduler. The final budget is the registry's step count
per fresh full run, cut on 2026-09-11 from 16,000 to 6,000 steps so each query is seen about
two to three times rather than fourteen; a lookup table can memorize repeated queries. Log the
training and a fixed held-out training-source loss at the registry's interval and flag
divergence; the flag is reported, not acted on after lock. The five short screen runs are a
separate, already spent allocation. Keep the endpoint fixed and require repeatable direction. Each full run contributes only the ordinary
endpoint and the fixed averaged form described below. No checkpoint shopping on the untouched
audit. A stopped/failed arm keeps its result record.

## Fixed checkpoint averaging

For each finalist/control full run and each seed, save the registry's three step-bound
snapshots, spread over the last quarter of the schedule so they actually differ, in addition to
recovery checkpoints. Fold each snapshot's learned scalar into its
row, average effective rows in float32 with equal weights, then quantize once. Never average
rows and scalars separately, mix tokenizers/runs, or tune the averaging window after scoring.
Use the original effective-row units without per-row or global rescaling; record their RMS to
make scale drift visible. This is late-checkpoint averaging, not a reproduction of SWA's
different optimizer schedule or an ensemble of normalized query embeddings.

Read selection metrics once for each endpoint and each averaged form: two forms per run,
eight reads at most across the candidate/control and two seeds. Choose the averaged candidate
only under the registry's fixed rule: positive technical gain in both seeds, mean gain above
the existing tie band, general dense/fused tolerances against v1, and no ambiguous-sense drop
versus the endpoint in either seed. Otherwise retain the endpoint. Missing snapshots or
measurements are an incomplete comparison, not permission to pick a different window.

Keep seed 0 as the artifact. Apply the chosen endpoint/averaged form to its matched seed-0
control as well, and bind both hashes before the audit. Seed 1 reports training sensitivity.
An audit miss does not reopen endpoint-versus-average selection. The inference model is still
one table and one pooled query vector.

## Int8 resident-row loader

Implement this comparison in the standalone M17 numpy loader: retain int8 codes and per-row
scales, gather the query's unique token IDs, and reconstruct only those rows in float32 before
the existing sqrt pooling. Keep the same bundle format, tokenizer, API, normalization and
empty/degenerate fallback. Do not edit the frozen M11 release implementation.

Compare with eager full-table fp32 loading using the **same selected artifact**. Use fresh
processes for cold-start and steady/peak RSS, and the registry's fixed length/batch strata for
warmed p50/p95 latency and throughput. Exercise repeated tokens, punctuation, special tokens,
truncation, empty input and near-degenerate sums. Adopt resident int8 only if conformance passes,
measured steady RSS falls, and every stratum stays within the existing p95 latency ratio limit.
Otherwise retain eager loading and record the outcome. Array-size arithmetic alone is not the
RSS result, and no inference speedup is assumed.

This is an implementation comparison, not a quality-selection arm. Perform fixture conformance
before any benchmark use; the final post-audit serving check uses the already locked artifact
and cannot select different model weights. Existing ONNX/FastEmbed parity remains required,
but numpy memory results do not imply savings in those runtimes or authorize a new port design.

## Measurement without consuming M13's evidence

Use the existing **ordinary development** suite for broad dense/fused regression checks,
explicitly labeled heavily reused selection evidence; it has substantial training/domain
adjacency and M7/M8 accumulated hundreds of reads. It cannot certify universal improvement.
Do not score the six, reserved four or LoTTE, or alter any of their locks/access receipts.

Build a modest M17 panel across the declared domains, split by source document/query family
into selection and a sealed audit, **before the 72-hour clock starts**. Judging hundreds of
queries is days of human work and does not belong inside the training window; the clock begins
only once the panel manifest hash is committed. The registry sets a target size, not an
assertion that this panel already exists. The panel is descriptive and audit evidence; the
registry records its expected standard error so nobody reads a coin flip as a result. Use affirmatively licensed QA
labels where suitable and independently checked search-query relevance judgments for technical
material. Include acronyms, expanded forms, context and ambiguous senses. Every query needs a
preassigned slice/family label; count variant families together in sampling and uncertainty.
Each query also needs a defined corpus and relevant-document IDs. Teacher rankings and seed-document identity alone
are not human relevance judgments. Log zero-shot v1/teacher performance on the panel only after
its definitions are fixed, and only expose the selection partition during development.

At the final audit, score the chosen seed-0 artifact/form, its fixed matched control in that
same form, v1, and Stella on the same query IDs with **exact dense retrieval** over the panel's
full declared corpus.
Predeclare seed 0 as the artifact rather than choosing a lucky seed. Report nDCG@10 and Recall@10
per domain, general and technical macros, plus teacher-to-student agreement as a separate
diagnostic. For DBSF, fuse exact dense top-100 and fixed BM25 top-100 runs using the existing M12
implementation. ANN latency experiments must not enter these quality numbers.

Report paired query-family bootstrap intervals on candidate−v1, with domain stratification;
report training-seed sensitivity separately. A small panel may be unable to resolve a modest
gain or rule out a small regression. No significant difference is not equivalence, and no
teacher gap on one panel demonstrates coverage of all query types. Draft point-estimate
preferences in the registry route the screen; they are **not approved release/superiority bars**.
The owner must ratify the prospective M17 decision protocol before real quality observations.

If a trustworthy panel or adequate intervals cannot be obtained within the budget, keep the
artifact as a development candidate and state what was measured. Do not replace judgments with
teacher agreement or borrow M13's held-out access to force a v1.1 release. Publication remains
under the existing release policy; this session authorizes neither publication nor a policy
override. Preserve v1 for rollback even if a later v1.1 is approved.

## The 72-hour allocation

These are **ceilings and stop points**, not throughput predictions or instructions to occupy
the GPU for the entire period. Source selection and all costly work count against the clock.

| Elapsed window | Work | Exit |
|---|---|---|
| Pre-clock | Judged panel built, split and sealed; new-source licence evidence recorded; manifest hashes committed | Clock may start |
| 0–8 h | Bounded M17 driver/loader design, manifests, tokenizer and real-path smoke, two reviews | Verified warm start, pair handling, checkpoint/resume, tokenizer and measured rates; otherwise no long run |
| 8–24 h | Coverage and verified alias pairs, reusable teacher targets, one candidate cache | Support and provenance recorded; price both views and reduce draft dose before lock if needed |
| 24–36 h | Five matched short runs and one endpoint comparison each | Choose a candidate/control under the fixed alias comparison or stop for no useful signal |
| 36–52 h | Candidate/control full runs, seed replication, fixed endpoint/average comparisons | Complete schedules and all declared selection reads; fix form and seed-0 hashes |
| 52–60 h | Locked M17 audit; exact dense and DBSF checks | Report gains/regressions and uncertainty; no M13 access or form reselection |
| 60–66 h | Int8 resident/eager loader comparison, separate bundle and numpy/ONNX parity | Same API/index; measured RSS/latency decides loader, not model weights |
| 66–72 h | Recovery reserve | Repair an interrupted planned phase; no new hypothesis search |

Keep implementation to one M17 driver, one candidate-cache schema with pair IDs, a fixed
averaging export helper and the bounded numpy loader change, using existing table/retrieval/
fusion primitives. Defer optional card/UI polish if preparation slips; missing
mandatory parity or judgments means an incomplete development artifact, not a release.

If the first implementation/review phase overruns, account for it and recompute the remaining
allocation **before** locking the training schedule. Real teacher encoding, bank mining and
evaluation must be timed at two sizes; P0's synthetic step rate cannot price them. One model
owns the GPU at a time. Reuse caches only after identity checks; don't reshape M13's arms or
consume its cloud allocation. Stop early on a null result, exhausted support or a completed
deliverable. Save checkpoints at the registry's time interval, including optimizer, scheduler,
RNG, stream position and tokenizer identity; smoke interruption/resume before a long run.

After lock, an incomplete required phase stops dependent work. Checkpointed recovery may use
the remaining reserve; it cannot silently remove an arm, shorten a dose or drop seed replication.
Do not select a winner from an accidentally incomplete matrix. Unused time can move forward to
the same registered work, within the hard total. These rules are recorded in the registry.

## Static late interaction: scope decision

ColBERT represents each document with token vectors and performs per-query-token MaxSim
against them. Its index differs from the single Stella vector already stored here.
[ColBERT primary implementation](https://github.com/stanford-futuredata/ColBERT).

With only document vector `d`, a sum of token scores is exactly
`sum_i a_i (u_i dot d) = (sum_i a_i u_i) dot d`. Query normalization is constant across documents,
so this gives the same ranking as pooling first. A nonlinear max over query tokens could alter
ranking using existing candidate vectors, but changes the scorer and serving cost; it is not
ColBERT and would need separate evaluation. A lexical/token side index also adds a deployment
component. None is needed for the chosen v1.1 plan.

This is not a claim that static late interaction cannot work. It is a fit decision for a
drop-in update with one unchanged document index. Hard-candidate training can use richer
supervision while keeping the shipped query path simple.

## Ready for the next session

The [follow-up shortlist](../research/m17-additional-avenues-2026-09-11.md) supplies the evidence
for the three additions Dylan accepted on 2026-09-11. Their comparisons, constants and rebalanced
allocation now live in this plan and the draft registry. Their inclusion needs no further
approval; implementation, measured feasibility and the future execution go remain outstanding.
The recovery reserve, frozen index, M13 scope and protected-access rules stand.

Follow [STATUS.md](STATUS.md) and [CODEMAP.md](CODEMAP.md). Ratify the prospective M17 protocol,
pin admitted data and a real measured allocation, implement the small driver, and obtain the
two required independent reviews before expensive execution. No `CLAUDE.md` exception is
needed for the recommended scope. Larger tables, new data rights or changed release criteria
would require a specific owner ruling; none is silently adopted here.
