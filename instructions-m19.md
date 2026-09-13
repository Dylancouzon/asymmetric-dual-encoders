# M19 — short-query Qdrant Zero feasibility

**Planning only, 2026-09-13 (Dylan).** Determine whether a new Constella Zero query table can
materially improve bare and short Qdrant-jargon retrieval without degrading the useful hybrid
system. Stay on branch `m18-qdrant-project-memory` and reuse M18's pinned Qdrant snapshot, document
vectors, BM25 index and tested execution primitives. This planning session creates and reviews this
instruction file; it does not authorize M19 execution, a new evaluation read, external training
data, publication or changes to any closed result.

M18 remains closed at `SYSTEM_READY` + `ENCODER_NO_IMPROVEMENT`. That outcome means no M18
candidate met its locked eligibility rules. It does not establish equivalence with Zero v1, rule
out improvement on bare or short jargon queries, or prove a capacity limit for static token rows.
Do not rewrite M18's thresholds, results, confirmation receipt or decision after the fact.

## Owner objective and scope

M19 answers a narrower question than M18:

> Can a frozen-inherited-row Zero extension improve dense retrieval for bare identifiers and
> two-to-five-word Qdrant queries across multiple useful artifacts, while preserving the released
> v1 behavior and the artifact-collapsed BM25+dense hybrid path?

The immediate product and model questions remain separate:

- **Project-memory system:** return a diverse top ten of useful Qdrant artifacts with a supporting
  passage. Hybrid retrieval may solve this even if no new encoder qualifies.
- **Zero encoder:** demonstrate that a new table improves the targeted dense route on more than a
  handful of queries. BM25 or fusion must not conceal a broken or unchanged dense model.

Owner constraints:

- Keep the pinned Stella 400M v5 document tower and normalized 1,024-dimensional document space.
  Reuse M18's vectors; do not re-embed the corpus or change the document model.
- Reuse the released `DylanCouzon/constella-zero` bundle as initialization and baseline. Freeze
  every inherited row and inherited pooling parameter.
- Keep Zero a context-independent, order-insensitive token-row lookup and pooled sum. M19 gets one
  bounded exact-row recipe, not an architecture search or a transformer query encoder.
- Run locally on this machine. No cloud or paid compute.
- Qdrant remains the evaluation domain. Outside software text is not authorized for training by
  this file; it requires a measured support failure, a commercial-use licence record and an owner
  ruling before acquisition or use.
- Internal evaluation only. Do not publish a model, index, corpus or Hugging Face repository from
  M19 without a separate owner ruling after the result.
- Do not ask Andrey to repeat the dense-only verdict already supplied. Existing `s3` and `k8s`
  examples are development diagnostics. Voluntarily supplied queries may become prospective
  protocol input, not a post-result approval poll.

## What M18 established

Bind these facts as prior evidence, not hypotheses to rediscover:

- The M18 corpus has 79,269 indexable passages from 11,574 Qdrant artifact families. Its Stella
  document matrix and BM25 index are complete and integrity checked.
- M18's held-out questions are issue/PR titles and review questions, not issue bodies. Their median
  length was nine words overall and twelve in the exact/numeric stratum. They evaluate retrieval of
  one distinct answer span, while the deployed product returns useful artifacts.
- The base WordPiece vocabulary shatters relevant terms: examples include `s3` to `s`, `##3` and
  `k8s` to `k`, `##8`, `##s`. Stella can contextualize fragments; Zero pools fixed rows and cannot.
  This is a general Zero representation limitation whose impact is concentrated in short jargon,
  not a tokenizer implementation bug or a universal retrieval failure.
- M18 trained sixteen exact rows from only 148 gradient-eligible queries. No complete alias pair
  survived, so alias consistency was inactive. T1 was inert and T2 changed vectors without changing
  any registered ranking metric. Do not repeat either arm.
- M18 T0's best dense gain was `+0.007591` nDCG@10, but its best fused gain was `+0.002953`; that
  fused gain came from one development win and 99 ties. The candidate correctly remained ineligible.
- The report-only `k8s` dense result improved qualitatively, and released-v1 fusion already ranks
  plausible `s3` snapshot and `k8s` persistence artifacts first. Those examples are encouraging,
  unjudged development evidence—not a successful model result.
- An exclusion-corrected post-run diagnostic found development stratum-macro Stella Recall of
  approximately `0.524@100` and `0.782@1000` (77/100 target spans found by 1,000). The M18 qrels are
  difficult and too narrow for the new task, but they are not shown to be mostly unreachable or
  invalid. Reproduce and persist this diagnostic in M19 before relying on it.
- Chunk repetition is a real presentation problem but was exaggerated in the external diagnosis.
  Released-v1 DBSF averaged 9.19 distinct artifacts in the 21-term top ten, minimum seven. Artifact
  collapse is still the correct product unit; it cannot by itself remove all irrelevant artifacts.

M18 development queries, the 21 inspected bare terms and every M18 candidate checkpoint are spent
development evidence. They may inform M19 design and appear in diagnostics, but cannot serve as
fresh M19 confirmation. Never reopen or reread M18's raw confirmation queries/qrels. Its published
aggregate result, lock and receipt remain readable provenance.

## Hypotheses and falsifiers

M19 has three explicit hypotheses:

1. On bare and short Qdrant-jargon queries, an exact token row can carry a useful direction that the
   fixed fragment rows cannot express cleanly.
2. Direct bare-term supervision plus varied short contexts can improve more queries than M18's
   sparse natural-query extraction did, while frozen inherited rows preserve unrelated v1 behavior.
3. Collapsing passages into artifact results improves result diversity and aligns evaluation with
   the project-memory product, but does not substitute for relevance judgment.

Stop or falsify the attempt when any of the following occurs:

- fresh pooled judgments cannot be made credible or the target query slice cannot be populated;
- Stella lacks a meaningful dense advantage over v1 on the fresh target slice;
- the Qdrant-only support inventory cannot provide the minimum distinct contexts and eligible views;
- the step-500 candidate changes too few target rankings or does not improve targeted dense quality;
- serving/tokenizer/export parity fails after one corrective attempt; or
- the first-seed candidate misses the fixed eligibility rule.

Do not turn a failed gate into an architecture, teacher, corpus-mixture or hyperparameter search.

## Frozen inputs and protected boundaries

At execution start, create an M19 registry and inheritance manifest that verify, at minimum:

- M18 source, corpus, index and system-manifest file hashes plus their internal identities;
- Qdrant commit `5e32ea89cb5ea9a6827a68b27291d2e2251e1b1c` and GitHub cutoff
  `2026-09-13T00:36:26Z`;
- ordered document IDs, corpus text hash, Stella vector hash/shape/dtype and BM25 parameters;
- released Zero-v1 model, tokenizer and config hashes;
- M19 parser, artifact-collapse, query-set, pooling, judgment, vocabulary, training, export and
  evaluation versions; and
- every seed, planned read, practical margin and confirmation transaction state.

Reference immutable M18 artifacts rather than copying or rebuilding large arrays. M19 may create
new manifests, query caches, training data, checkpoints and result files only under `m19/`,
`m19src/`, `work/m19/` and `results/m19_*`.

The historical protections remain absolute. Do not read or write:

- `results/perquery.json`, `results/frozen_eval/untouched-*`, reserved qrels caches,
  `work/m9reserve`, LoTTE payloads or any spent M7–M13 evaluation surface;
- raw M18 confirmation queries or qrels, directly or through a helper;
- a model-generated reconstruction of protected examples; or
- any closed registry, lock, decision, receipt or published result as though it were mutable input.

Implement path-level read guards and tests before data work. A reviewer prompt must repeat these
exclusions and name the files it may inspect.

## Retrieval unit and serving contract

Documents remain M18 passages, but ranked results become artifacts. Define one deterministic
collapse rule before any new quality read:

1. Retrieve up to 500 passages per route, with the existing deterministic score/document-ID ties.
2. Map each passage to its stable `artifact_id` (`gh:thread:<number>` for a GitHub thread and the
   registered repository-artifact identity for repository text).
3. Keep the highest-scoring passage per artifact for that route; tie by ascending passage ID.
4. Retain the first 100 unique artifacts per route. Record both the artifact score and its best
   supporting passage.
5. Apply the existing DBSF operator to artifact scores at depth 100. For display, retain the best
   lexical and dense supporting passage and identify which route contributed each.
6. Return at most one row per artifact. Never silently mark every sibling passage relevant.

If 500 passages do not supply 100 unique artifacts for a route, use all available artifacts and
report the shortage. The passage depth is a serving/retrieval constant, not a tuning parameter.
Measure latency and memory for passage scoring, collapse and fusion separately. Preserve a
chunk-level diagnostic only to quantify the effect; artifact-level retrieval is the M19 gate.

For an evaluation query authored from one specific artifact, exclude that complete artifact before
collapse and record the exclusion. Excluding only the opening while returning a sibling comment
would become trivial self-artifact retrieval at the new relevance unit. A real internal query or a
bare term discovered across multiple artifacts has no synthetic source artifact to exclude. The
relevance judge assesses each remaining artifact and supporting evidence; relevance must never be
inferred from a thread relationship.

## Fresh query protocol

Create 60 development-selection queries and 30 sealed confirmation queries when evidence permits.
Use fixed family/deduplication assignment and target the following composition:

| Stratum | Development | Confirmation | Purpose |
|---|---:|---:|---|
| bare identifiers/project terms | 12 | 6 | directly exercise token shattering |
| short task or intent, 2–5 words | 18 | 8 | test a term in minimal context |
| aliases, renamed settings and old/new jargon | 10 | 5 | test project language relationships |
| exact/numeric/version distinctions | 10 | 5 | prevent lexical and numeric regressions |
| longer natural-language controls | 10 | 6 | show whether gains are narrow and safe |

Minimum viable execution is 40 development and 20 confirmation queries, with at least eight/four
in every available target stratum. Report shortages; do not manufacture filler to hit a table.
Bare means one project term or identifier after punctuation-preserving lexical splitting. Short
means two to five lexical terms. Controls are six or more terms.

Query sources, in preference order:

1. real internal query text supplied prospectively with permission, stripped of user identity and
   secrets;
2. short intents independently authored from pinned Qdrant artifacts before retrieval results are
   shown; and
3. deterministic bare terms and identifier-plus-intent forms supported by multiple Qdrant
   artifacts.

The two known `s3`/`k8s` complaints and all terms inspected in M18 are development-only. Do not use
them in M19 confirmation. Do not use Qwen or another model to author evaluation questions or decide
relevance. Store the query text separately from qrels, union near duplicates before splitting, and
exclude source families across development and confirmation.

## Pooled multi-relevance judgments

Structural single-answer qrels are not the primary M19 evaluation. Build artifact pools and judge
what a project-memory user could use.

For development, pool the top ten unique artifacts from these fixed systems:

- BM25;
- released Zero v1 dense;
- released Zero v1 + DBSF;
- Stella dense and Stella + DBSF; and
- M18 T0 step 250 as a named historical diversity contributor, never as an eligible M19 model.

Deduplicate the union by artifact. Randomize presentation order and remove model name, score, rank
and route. Show the query, artifact title/metadata and enough top passages to judge usefulness.
Use graded labels:

- `2`: directly answers, resolves or is the intended artifact;
- `1`: materially useful context for the query; and
- `0`: not useful, coincidental term match or misleading.

Multiple artifacts may be relevant. A provenance record must bind query, artifact, displayed
passage hashes, label, reviewer identity/version and rationale. `Unjudgeable` is separate from
zero. Do not label by thread relationship, title overlap or source-system membership.

Candidate-aware pool completion is predeclared and metric-blind. At the step-500 pause, union any
unjudged top-ten artifacts from steps 250/500, blind and judge them once, then compute the diagnostic
gate. If training continues, union unjudged top-ten artifacts from every remaining registered
checkpoint before reading any of their retrieval metrics, blind and judge that union once, freeze
the final development qrels, and recompute all baseline/checkpoint metrics on the same qrels. Never
choose a checkpoint first and expand the pool only for its results.

Confirmation initially seals query text and family assignments, not incomplete baseline-only
qrels. After the model/checkpoint decision is locked, a one-shot transaction retrieves the fixed
baseline and selected-candidate pools, blinds and adjudicates their union, freezes confirmation
qrels, scores all systems and writes a complete receipt. The transaction may be rehearsed on
synthetic fixtures but never partially run on real confirmation. If no candidate is eligible,
confirmation audits the selected v1-backed system without adding an M19 candidate.

### Judgment review gate

- One primary reviewer judges the complete blinded development pool.
- A fresh independent reviewer judges a fixed seeded 20% sample, oversampling relevance labels and
  every `unjudgeable` item.
- Report exact agreement for relevant (`1/2`) versus irrelevant (`0`), graded agreement and all
  disagreements. Require at least 90% binary agreement and no unresolved material rubric defect.
- Adjudicate disagreements without system identities. If the threshold fails after one rubric
  clarification and fresh sample, stop before training.
- Neither reviewer may see confirmation content before its one-shot transaction.

Human review is preferred for at least the short-query sample because usefulness is a product
judgment. Astra may perform source-grounded primary or independent review when human labels are not
available, but its model/version, prompt, inputs and access log must be recorded. Model judgment is
evidence, not ground truth, and must be disclosed.

## Baselines and headroom gate

Before training, measure on the frozen artifact-level development protocol:

- BM25, released Zero v1 dense and released Zero v1 + DBSF;
- Stella dense and Stella + DBSF;
- untrained M19 V0 and V0 + DBSF; and
- the historical M18 T0-step250 diagnostic, clearly marked ineligible.

Report graded nDCG@10, binary Recall@10, MRR@10, at-least-one-grade-2 success@10, wins/ties/losses
versus v1, per-stratum results, bootstrap intervals, query latency, distinct artifacts, stored bytes
and runtime resident bytes. Report Recall@100 and Recall@1000 for the legacy M18 development qrels
once as a postmortem diagnostic; it is not an M19 selection metric.

Freeze exact practical margins after the judgment-quality report and before any candidate result or
optimizer step. The default headroom requirement, which may only be made stricter at that point, is:

- on the combined bare/short/alias target slice, Stella dense exceeds v1 dense by at least `0.05`
  artifact nDCG@10; and
- Stella produces at least ten percentage points more target-query wins than losses versus v1.

If either fails, do not train. Report that the frozen teacher does not demonstrate enough teachable
headroom for this recipe. A strong v1 hybrid result may close the product problem, but does not by
itself prove that dense Zero improved.

## Vocabulary and training-support gate

Discover vocabulary only from M19 training sources. A row is eligible when the released tokenizer
splits a meaningful Qdrant term into multiple pieces and the term has retrieval demand in the
prospective development distribution or a registered operational reason. Record tokenization,
fragment rows, support, residual value and before/after text behavior. Do not add already-whole
tokens, UUIDs, hashes, timestamps, random suffixes or one-off paths.

Start with Qdrant-only support and improve extraction before requesting new data:

- documentation headings and their sections;
- issue/PR titles and concise intent-bearing sentences;
- review questions and resolving explanations outside held-out families;
- aliases/renames explicitly evidenced in project text; and
- deterministic bare-term plus two-to-five-word forms tied to distinct source artifacts.

A bare term may receive a direct Stella query target, but repeating it does not count as additional
support. Context counts are by distinct artifact and normalized query text. Synthetic variants do
not increase natural-source counts.

The default training gate is:

- 8–64 added exact rows;
- at least 12 distinct Qdrant artifacts and 20 natural contexts for every ordinary row;
- separately disclosed owner-requested rows may use a minimum of six natural Qdrant artifacts only
  when an explicit alias/expansion is source-evidenced;
- at least 1,000 distinct gradient-eligible short query views overall;
- median of at least 30 distinct eligible views per row; and
- at least 50 complete evidence-backed alias pairs before enabling alias loss.

If the total or per-row gate fails, stop before training and report the exact shortage. Only then may
the owner authorize an affirmatively commercial-use-licensed software corpus for training support.
Qdrant remains the evaluation corpus. Do not silently weaken support minima or count generated
paraphrases as independent evidence.

The cached local Qwen teacher is optional only after a measured shortage in phrasing diversity, not
source support. It may generate at most five short training queries per source artifact and no more
than 25% of eligible training views. Each generation must be answerable from its bound Qdrant span,
contain an admitted target term, pass deduplication and receive a blinded quality audit. Qwen text
never defines evaluation queries, qrels, aliases or natural-support counts.

## Single candidate recipe

M19 has one trained form, `T0-short`:

- released Zero v1 tokenizer plus the admitted exact AddedTokens;
- raw text with digits preserved and no T1/T2 normalization;
- released-v1 effective rows as initialization; inherited rows and pooling parameters frozen;
- count-weighted fragment composition for each new row at V0;
- direct cosine supervision on each bare term and varied short contexts;
- the existing teacher-cosine and hard-candidate listwise objectives on source-grounded queries;
- row-balanced deterministic sampling so a frequent term cannot dominate and a scarce row cannot
  be amplified beyond four times its natural share;
- alias-consistency weight `0.1` only if the support gate yields at least 50 complete eligible pairs,
  otherwise weight `0.0` prospectively and truthfully;
- new-row learning rate `3e-4`, batch `256`, warmup `200`, maximum `4,000` optimizer steps;
- checkpoints at steps `0`, `250`, `500`, `1,000`, `2,000`, `4,000`; and
- primary seed `19001`, confirmation seed `19002`, bootstrap seed `19019`.

Confirm inherited-row immutability algebraically and by hash. Keep the M18 duplicate-safe
cross-epoch sampler, but require the eligible pool to exceed batch size so routine batches do not
mostly repeat queries. Cache raw Stella targets and candidate lists once. Make every long stage
atomic, resumable and identity-bound. Record loss components, actual alias pairs, row sampling,
unique queries per batch, elapsed time, peak RAM/VRAM, disk and table sizes.

Do not add an anchor, new teacher, T1/T2 arms, learned fusion, negative-mining sweep, checkpoint
averaging, document fine-tuning or late interaction. Any such idea belongs to a later milestone
after this feasibility result.

## Step-500 diagnostic and stopping rule

Before the full run, evaluate V0/step 0, step 250 and step 500 on development. The real-path
diagnostic must also verify:

- training-forward versus exported NumPy loader parity;
- unchanged inherited model/tokenizer/config hashes and inherited row bytes;
- exact added-token matching and collision behavior;
- int8 quantization error and runtime resident representation;
- deterministic resume from the step-250 checkpoint; and
- artifact-collapse/fusion parity between evaluation and the documented query command.

Continue beyond 500 only when all gates pass and, on the target slice, step 250 or 500:

- improves dense artifact nDCG@10 over step 0 by at least `0.01`;
- produces at least three more query wins than losses; and
- changes the top-ten artifact order for at least 15% of target queries.

If not, stop the trained attempt. One parity defect may be corrected and the identical diagnostic
rerun once; a quality miss is not permission to change the recipe.

## Encoder eligibility and selection

The model objective is dense short-query improvement. The final exact margins must be sealed after
the pre-training protocol/headroom audit and before optimizer step one. Unless that prospective
audit makes them stricter, a candidate is eligible on development only if:

- target-slice dense artifact nDCG@10 exceeds released v1 by at least `0.03` absolute;
- the paired 95% bootstrap interval for that target dense delta has a lower endpoint above zero;
- target-query wins minus losses are at least ten percentage points, with gains occurring in at
  least two target strata;
- candidate + DBSF does not trail v1 + DBSF by more than `0.005` artifact nDCG@10 overall or on the
  exact/numeric control stratum;
- longer-control dense and hybrid deltas are not clearly negative by paired interval;
- no adequately sized target stratum has a negative point delta with an interval wholly below zero;
  and
- tokenizer, export, int8, latency and resident-size gates pass.

Select the earliest checkpoint within `0.005` dense nDCG@10 of the best eligible checkpoint, unless
the later checkpoint has a clear paired advantage. This avoids selecting a late step on noise.
V0 can qualify under the same quality and serving rules without a second seed. A trained form runs
seed `19002` only after it is eligible on seed `19001`; require the target dense and hybrid-safety
directions to agree. No eligible first seed means no second seed.

An M19 trained candidate is not a public Zero successor. It is an internal candidate until fresh
confirmation passes and the owner separately approves release.

## One-shot confirmation

Before confirmation access, commit and push a decision lock containing:

- selected model/checkpoint and exact bundle hashes;
- all query/protocol/judgment versions and development evidence hashes;
- source/corpus/index inheritance identities;
- artifact-collapse, retrieval, fusion and metric recipes;
- eligibility computation, second-seed result and serving gates; and
- the confirmation pooling/adjudication transaction code and synthetic rehearsal receipt.

Then execute the confirmation transaction exactly once. For a candidate to pass:

- target dense artifact nDCG@10 delta versus v1 is positive;
- target wins exceed losses and improvement is not confined to one query;
- hybrid overall and exact/numeric deltas are no worse than `-0.005`;
- no adequately sized stratum has a paired interval wholly below zero; and
- the decision-locked serving bytes and evaluation loader remain identical.

Confirmation is a directional audit on a small fresh sample, not a chance to tune margins, change
qrels, select a different checkpoint or run another candidate. A failed confirmation retains
released v1 and closes the encoder result.

## Reviewer process

Use adversarial review at the decisions that can invalidate downstream work:

1. **Plan review:** one Astra review of this instruction file for task alignment, leakage, feasible
   statistics and overengineering. Correct every P0/P1 or explicitly reject it with evidence.
2. **Inheritance/implementation review:** before any new quality read, one implementation reviewer
   and one fresh Astra reviewer inspect the real artifact-collapse, protected-read guards, pooling,
   scoring and synthetic transaction—not isolated helpers.
3. **Data review:** primary blinded judgment plus the independent 20% audit described above. A
   reviewer who authored a query may not be its sole relevance judge.
4. **Pre-training lock review:** Astra verifies the frozen query sets, judgment quality, support
   gate, headroom, exact thresholds and executable identities before optimizer step one.
5. **Step-500 review:** reproduce parity and stopping arithmetic before continuing the fixed run.
6. **Selection review:** a fresh reviewer independently recomputes eligibility and bundle hashes
   before the one-shot confirmation transaction.
7. **Final review:** reconcile outcomes, confirmation receipt, limitations and publication status.

Only one Astra reviewer runs at a time. Give every reviewer a bounded file list, the protected-read
exclusions and a read-only mandate. Save prompts, reviewer/model identity, findings, dispositions
and verification evidence under `m19/reviews/` or `results/m19_review_*`. Reviewers never silently
edit results or relax a gate. Re-review material fixes. Two independent reviewers must issue GO
before expensive training or the irreversible confirmation read.

## Ordered execution

1. Create `m19/`, `m19src/`, `work/m19/` and fresh guards/registry; bind M18 inheritance.
2. Implement and synthetically rehearse artifact collapse, pooled judgment import, metrics, resume,
   export and the one-shot confirmation transaction.
3. Reproduce the M18 Recall@100/@1000 postmortem and artifact-diversity diagnostics.
4. Build, deduplicate and seal fresh M19 query text/families; keep confirmation inaccessible.
5. Build and blind the development artifact pools; adjudicate and pass independent review.
6. Read fixed baselines, measure teacher headroom and stop if the headroom gate fails.
7. Build the Qdrant-only training inventory; stop or request owner direction if support fails.
8. Build V0/cache/candidate data, seal the complete execution recipe and obtain two review GOs.
9. Run the T0-short step-500 diagnostic; continue only if its fixed gate passes.
10. Complete the one-seed screen and run the second seed only for an eligible trained form.
11. Decision-lock the selected encoder/system without confirmation access.
12. Run the fresh confirmation pool, adjudication and scoring transaction once.
13. Finalize the artifact-collapsed internal query path and record both outcomes.

No stage proceeds merely because compute remains available.

## Required artifacts and final outcomes

Create and maintain at least:

- `m19/STATUS.md`, `PLANNING.md`, `registry.json`, `LEDGER.md`, `FINDINGS.md`, `CODEMAP.md`;
- `m19/inheritance-lock.json` and an immutable pre-training execution lock;
- protected-read tests and one-command synthetic rehearsal;
- query/family manifests, pool manifests, blinded judgment records and independent-review receipts;
- artifact-collapse parity and legacy M18 recall/diversity diagnostics;
- vocabulary/support/collision manifests and explicit natural/generated counts;
- machine-readable baseline, headroom, checkpoint, seed, stratum, win/tie/loss and bootstrap results;
- int8 export gates, exact bundle hashes, runtime memory/latency and a NumPy loader;
- a decision lock plus one-shot confirmation lock/result/receipt; and
- a final system manifest and exact local query command.

Final status records both axes:

- `SYSTEM_READY` or `SYSTEM_BLOCKED`; and
- `ENCODER_INTERNAL_CANDIDATE` or `ENCODER_NO_MEASURABLE_IMPROVEMENT`.

`SYSTEM_READY` requires a usable artifact-collapsed BM25+dense DBSF path over the inherited snapshot.
`ENCODER_INTERNAL_CANDIDATE` requires every development, second-seed, confirmation and serving gate.
Otherwise released Zero v1 remains selected. Never call an unresolved interval equivalence, never
promote a qualitative top ten as a measured win, and never let the historical M18 label imply that
the broader short-query goal is impossible.
