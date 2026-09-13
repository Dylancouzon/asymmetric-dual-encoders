# M19 — deterministic short-query Zero feasibility

**Planning only, 2026-09-13 (Dylan).** Determine whether one deterministic exact-row extension of
Constella Zero can materially improve bare and short Qdrant-jargon retrieval without degrading the
useful hybrid system. Stay on branch `m18-qdrant-project-memory` and reuse M18's pinned Qdrant
snapshot, document vectors, BM25 index and tested serving primitives. This planning session creates
and reviews this instruction file; it does not authorize M19 execution, evaluation access, outside
training data, publication or mutation of a closed result.

M18 remains closed at `SYSTEM_READY` + `ENCODER_NO_IMPROVEMENT`. That means no M18 candidate met its
locked eligibility rules. It does not establish equivalence with Zero v1, rule out improvement on
bare or short jargon queries, or prove a capacity limit for static token rows. Never rewrite M18's
thresholds, results, decision, confirmation lock or one-read receipt.

## Next-session bootstrap and repository conventions

Begin execution in a newly cleared context. The M18 closure session accidentally let one broad
repository text search match a protected historical evaluation file; no returned content was used,
no M19 experiment had begun and the incident is recorded in M18's ledger. A clean context is
therefore a mandatory M19 boundary, not optional housekeeping.

At the beginning of that execution session, read `CLAUDE.md` completely, then this file, M18's
short `STATUS.md`, `CODEMAP.md` and final `FINDINGS.md`. Inspect only the specific M18 manifests
needed to bind inheritance. Do not recursively ingest historical results or protected data. Never
run a repo-wide content search over `results/` or `work/`; use explicit admitted paths and install
the protected-read guards before any query/evaluation work.

Follow the repository's established milestone structure and rules:

- use `m19/` for status, planning, registry, ledger, findings, codemap, locks and review records;
- use `m19src/` for the minimum milestone-bound implementation and synthetic tests;
- use gitignored `work/m19/` for raw/derived caches and runs, and tracked `results/m19_*` for compact
  machine-readable evidence without raw private query text;
- fork the smallest tested M18 primitives and record source commit plus semantic differences;
- keep constants in the registry, measurements in JSON, decisions in the ledger, pitfalls in the
  codemap and durable interpretation in findings;
- use atomic/resumable writes for every expensive or one-shot stage, protect user changes, and use
  `apply_patch` for hand edits;
- run affected tests before each coherent commit, commit and push often, and never force-push away
  the M18/M19 paper trail; and
- use `/home/dylan/asymetric-dual-encoders/.venv/bin/python` unless the worktree later receives its
  own verified environment. Do not assume bare `python` resolves correctly.

If this file and `CLAUDE.md` conflict, the newer, milestone-specific ruling here controls M19 while
the global protected-data and evidence-preservation rules in `CLAUDE.md` remain absolute.

## Objective and scope

M19 asks one narrower question:

> Can replacing a shattered Qdrant term with one teacher-targeted exact row improve Zero's dense
> retrieval on previously unseen short contexts for that term, across multiple useful artifacts?

This is a mechanism-feasibility milestone, not a claim that one Qdrant experiment validates every
technical vertical. A positive result would justify a later cross-project milestone for deriving
small vertical-specific vocabularies; that later work must repeat demand/support selection and use
fresh queries in each domain. A negative result rejects this deterministic teacher-row construction
for the tested setting, not vocabulary extension in general.

The project-memory system and encoder decision remain separate:

- **System:** return a diverse top ten of useful Qdrant artifacts with supporting passages. The
  artifact-collapsed BM25+dense DBSF path can succeed even when no new encoder qualifies.
- **Encoder:** demonstrate dense improvement across terms and fresh short intents. BM25/fusion may
  not conceal an unchanged or broken table.

Owner constraints:

- Keep the pinned Stella 400M v5 document tower and normalized 1,024-dimensional document space.
  Reuse M18's vectors; do not re-embed the corpus or change the document encoder.
- Initialize from released `DylanCouzon/constella-zero`. Preserve every inherited token ID, row and
  pooling parameter byte-for-byte in the new bundle.
- Keep Zero a context-independent, order-insensitive token-row lookup and pooled sum. M19 constructs
  one deterministic exact-row candidate. It has no optimizer, training sweep or model architecture
  arm.
- Run locally on this machine. No cloud, paid compute or new source acquisition.
- Qdrant remains the only corpus and evaluation domain. M19 needs no Qwen, outside software corpus,
  alias loss, checkpoint schedule or second training seed.
- Internal evaluation only. Do not publish a model, corpus, index or Hugging Face repository without
  a separate owner ruling after the result.
- Do not ask Andrey to repeat the dense-only verdict already supplied. Existing `s3` and `k8s`
  examples are spent development diagnostics. Voluntarily supplied queries may be prospective
  protocol input, not a post-result approval poll.

## What M18 established

Bind these observations as prior evidence:

- The pinned Qdrant corpus contains 79,269 searchable passages from 11,574 artifact families. Its
  Stella document matrix and BM25 index are complete and integrity checked.
- M18's questions are issue/PR titles and review questions, not issue bodies. A nine-word overall
  and twelve-word exact/numeric median was measured on the earlier 175-query v4 protocol; it was not
  re-measured after adjudication reduced the final development set to 100. Recompute any final-set
  length claim before using it.
- The base WordPiece vocabulary shatters relevant terms: `s3` becomes `s`, `##3`; `k8s` becomes
  `k`, `##8`, `##s`. Stella can contextualize fragments; Zero pools fixed rows and cannot. This is
  a general representation limitation whose impact concentrates in short jargon, not a tokenizer
  implementation bug or a failure on every query.
- M18 added sixteen exact rows but only 148 queries provided gradients. No complete alias pair
  survived, T1 was inert and T2 changed vectors without changing registered rankings. Do not repeat
  those arms.
- M18 T0's best dense gain was `+0.007591` nDCG@10. Its best fused gain was `+0.002953`, produced by
  one development win and 99 ties, so the candidate correctly remained ineligible.
- Released-v1 DBSF already ranks plausible S3 snapshot and Kubernetes persistence artifacts first.
  M18's report-only `k8s` dense list improved qualitatively. These are encouraging, unjudged and
  spent examples—not evidence that an M18 model qualified.
- A local, exclusion-corrected post-run calculation found legacy M18 development stratum-macro
  Stella Recall near `0.524@100` and `0.782@1000`, with 77/100 registered target spans found by
  1,000. This is provisional until reproduced into an M19 result. It refutes neither the qrels nor
  the observation that answer-span evaluation differs from useful-artifact search.
- Released-v1 DBSF averaged 9.19 distinct artifacts in the M18 21-term top ten, minimum seven.
  Chunk repetition is a real product defect but not the whole quality problem.

M18 development queries, its 21 inspected bare terms and all M18 candidates are spent development
evidence. They may inform design and appear in labeled diagnostics, but cannot provide fresh M19
confirmation. Never read raw M18 confirmation queries or qrels again. Published M18 aggregates,
manifests, locks and receipts remain provenance.

## Hypothesis and falsifiers

For a bare term, an exact row is the only free semantic component after fixed special-token rows.
It can therefore be chosen so the unquantized pooled Zero vector points exactly in the pinned
Stella query direction. M19 tests whether that direct repair transfers from the training-exposed
bare term to previously unseen short contexts and useful artifact rankings.

The attempt stops when any of these holds:

- fewer than eight genuinely demanded fragmented terms can be registered;
- fresh short contexts or credible useful-artifact judgments cannot be assembled;
- Stella lacks material headroom over v1 on queries the new rows can affect;
- the deterministic candidate fails algebraic, tokenizer, quantization or serving parity;
- development cannot resolve the registered effect and preservation claims; or
- the candidate fails any fixed development eligibility rule.

Do not answer a failed gate with contextual row fitting, Qwen, more terms, another teacher, a new
fusion method or an architecture search. Such work would require a later milestone and evidence
that this simpler test failed for the relevant reason.

## Frozen inheritance and protected boundaries

At execution start, create an M19 registry and inheritance lock binding at minimum:

- M18 source, corpus, index and system-manifest file hashes plus their internal identities;
- Qdrant commit `5e32ea89cb5ea9a6827a68b27291d2e2251e1b1c` and GitHub cutoff
  `2026-09-13T00:36:26Z`;
- ordered document IDs, corpus hash, Stella vector hash/shape/dtype and BM25/DBSF recipe;
- released-v1 model, tokenizer, config and effective-row hashes;
- term roster, exact-token policy, deterministic row formula and scale convention;
- query/split/pool/judgment/artifact-collapse/metric versions and seeds; and
- all planned reads, practical margins, numerical gates and confirmation states.

Reference immutable M18 artifacts rather than copying or rebuilding large arrays. M19 may create
files only under `m19/`, `m19src/`, `work/m19/` and `results/m19_*`. Keep the M18 worktree inputs
read-only and verify them before every quality transaction.

Historical protections remain absolute. Do not read or write:

- `results/perquery.json`, `results/frozen_eval/untouched-*`, reserved qrels caches,
  `work/m9reserve`, LoTTE payloads or any spent M7–M13 evaluation surface;
- raw M18 confirmation queries or qrels, directly or through a helper;
- model-generated reconstructions of protected examples; or
- any closed registry, decision, lock, receipt or result as mutable experiment state.

Implement path-level guards and tests before query work. Every reviewer brief names the admitted
files, repeats the exclusions and requires read-only operation.

## Deterministic artifact retrieval

Documents remain M18 passages, but M19 results are artifacts. Freeze this rule before quality data:

1. Retrieve up to 500 passages per route with descending score/ascending passage-ID ties.
2. Map passages to stable `artifact_id` values.
3. For each route, retain the highest-scoring passage per artifact; tie by passage ID.
4. Keep the first 100 unique artifacts per route. Record artifact score and supporting passage.
5. Apply the unchanged DBSF operator to artifact scores at depth 100.
6. Return one row per artifact. Preserve the best dense and lexical passage and their contributions
   so the interface can explain why the artifact ranked.

If 500 passages supply fewer than 100 artifacts, use all available and report it. Passage depth 500
is a registered serving constant, not a tuning knob. Measure passage scoring, collapse and fusion
latency separately. Keep chunk-level results only as a diagnostic.

For a query authored from one specific artifact, exclude that complete artifact and text-equivalent
copies before passage truncation. Otherwise a sibling comment becomes trivial self-artifact
retrieval. Real internal queries and bare terms supported across artifacts have no synthetic source
artifact. A distinct linked answer is not excluded merely for linking to the source. Each authored
query must have at least one useful remaining artifact or be labeled unanswerable and removed before
the split is frozen.

## Prospective term roster

Before authoring evaluation queries or reading retrieval quality, inventory the M18 corpus and
released tokenizer. Select 8–12 terms that:

- split into at least two released-tokenizer pieces;
- have one stable, project-relevant meaning;
- occur naturally in at least five distinct Qdrant artifacts;
- have credible demand from the recorded use case or prospective internal query input; and
- can support at least three distinct short intents and multiple useful result artifacts.

`s3` and `k8s` may be roster terms if they pass the same meaning and multi-artifact checks, but
their already inspected bare queries remain development-only. Do not admit ordinary whole tokens,
UUIDs, hashes, timestamps, random suffixes or one-off paths. Case-folding and AddedToken boundary
semantics must match released serving behavior. Freeze the roster, original fragment IDs and all
support counts before candidate construction.

The roster is a demand/meaning gate, not a claim that five occurrences estimate a 1,024-dimensional
row. The candidate receives its direction from Stella, not a fitted sample. Do not impose M18-style
training-view counts or generate paraphrases to inflate support.

## Deterministic exact-row candidate

Build exactly two vocabulary-extension bundles with the same frozen inherited components:

- **V0-compose:** exact AddedTokens initialized by M18's count-weighted composition of the original
  fragment rows. It isolates tokenizer replacement without a new semantic direction and is a
  descriptive baseline, not selectable.
- **T0-teacher:** the sole candidate. Each exact row makes the bare-term query point in the pinned
  Stella query direction under the actual Zero pooling rule.

For roster term `t`, let:

- `u_t` be the unit Stella query vector for the raw bare term using the pinned query prefix;
- `a_t` be the weighted sum of all frozen rows that remain when the original fragment pieces are
  replaced by one exact token—normally `[CLS]` and `[SEP]`;
- `r_comp,t` be the weighted sum of the original fragment rows under released sqrt-count pooling;
- `s_old,t = a_t + r_comp,t`; and
- `alpha_t = ||s_old,t||_2`.

Set the new effective row to:

`r_t = alpha_t * u_t - a_t`.

The final normalization cancels the pooling denominator, so the new unquantized bare query sum is
`alpha_t * u_t`. Preserving `||s_old,t||` fixes the otherwise arbitrary row scale and limits the
change in relative contribution when the term appears with context. Do not tune `alpha_t` from
retrieval results. Refuse a term if actual tokenizer/pooling behavior makes this derivation false or
non-finite.

Verify for every term through the real training-time math, exported NumPy loader and int8 path:

- exact bare-token match with stable inherited token IDs;
- pre-quantization cosine to `u_t` at least `0.999999`;
- int8 cosine to `u_t` at least `0.999` and maximum coordinate error at most `0.02`;
- byte-identical inherited row codes/scales and pooling parameters;
- unchanged tokenization and encoder output within `1e-6` for queries without an added-token match;
- boundary, punctuation, case, plural/possessive and substring collision fixtures; and
- no eager float32 expansion of the full table in the M19 loader.

The baseline input files keep their original hashes; the extended bundle files necessarily have
new hashes. Record both. Resident table bytes must equal inherited int8 codes/scales plus the exact
per-row increment; measure temporary allocations separately.

## Query protocol and split estimand

The estimand is improvement for the fixed term roster. Report bare-term adaptation separately from
generalization to unseen short contexts.

For each roster term, create prospectively:

- one bare development query, explicitly marked training-exposed because it defined `u_t`;
- three development short intents of two to five lexical terms; and
- two sealed confirmation short intents of two to five lexical terms.

Add one longer target-control query per term to development and one per two terms to confirmation.
Each contains exactly one roster term and tests whether its new row still composes in a more
specific question. Across the roster, designate at least four development and two confirmation
target controls covering numeric/version distinctions across at least two terms; if credible cases
cannot be authored, stop rather than leave the safety slice empty. With 8–12 terms this yields
40–60 development and 20–30 confirmation queries. If fewer than eight terms survive, stop. Do not
fill shortages with more templates for the same term.

Primary query classes are mutually exclusive: `bare_adaptation`, `short_context` and
`longer_control`. `alias`, `renamed`, `numeric` and `version` are secondary tags only. Short means
two to five punctuation-preserving lexical terms; longer means six or more. Every target query,
including each longer `target_control`, must contain exactly one roster term matched as an
AddedToken. Separate no-added-token collision fixtures belong to tokenizer/ranking parity checks,
not the quality estimate.

Fresh confirmation uses previously uninspected short intents whose normalized text and
near-duplicate intent family are absent from training, M18 diagnostics and M19 development.
Familiar vocabulary does not make a new context spent. Bare queries used to construct teacher rows
cannot count as unseen-query confirmation evidence and never enter confirmation.

Partition source/intent families before extracting query variants. Hold query-bearing source
artifacts out across development and confirmation and remove normalized/near-duplicate intersections
after every transformation. The same term may appear in both splits; the intent, wording and source
family may not. Record term-level dependence explicitly.

Query sources, in preference order:

1. prospective internal query text supplied with permission and stripped of identity/secrets;
2. short intents independently authored from pinned Qdrant evidence before results are shown; and
3. deterministic identifier-plus-intent forms supported by multiple artifacts.

Qwen may not author evaluation queries or judge relevance. Astra may perform source-grounded
relevance judgment and independent auditing under a frozen prompt/access policy; disclose that the
labels are model judgments. A query author may not be its sole relevance judge.

## Blinded useful-artifact judgments

Use binary artifact relevance for this bounded feasibility run:

- `1`: the artifact is materially useful for satisfying the query, supported by the frozen evidence
  packet; or
- `0`: it is coincidental, misleading or not materially useful.

Binary labels avoid an unaudited `1` versus `2` distinction. Multiple artifacts may be relevant.
`Unjudgeable` is not zero: resolve it under the frozen procedure or mark the protocol incomplete and
stop before metrics.

For both development and confirmation, pool the top ten unique artifacts from BM25, v1 dense,
v1+DBSF, V0-compose dense, T0-teacher dense, T0-teacher+DBSF and Stella dense. Build the candidate
before pooling; there is no checkpoint or later endpoint. Every artifact that can affect a reported
top-ten metric is therefore judged prospectively in one union.

Randomize artifacts with seed `19019` and remove system identity, route, score, rank and first-seen
phase. Mix concealed repeated audit items into later packets so their origin cannot be inferred.
Freeze an evidence packet containing:

- query ID/text and source-exclusion identity;
- artifact ID, title, URL/path and kind;
- the route-independent top three passages by best reciprocal rank across the blinded pool routes,
  tie-broken by passage ID, each capped at 1,200 characters with full-text hash;
- sufficient parent-thread or repository metadata to interpret the passages; and
- no model identity or numeric retrieval score.

An artifact label is route-independent. For the selected system, independently verify that the
displayed supporting passage itself justifies every artifact responsible for a measured win; an
artifact-level label cannot certify a misleading snippet.

One primary reviewer judges the complete pool. A fresh independent reviewer judges every primary
positive and unjudgeable plus seeded negatives sufficient to cover every query and term and bring
the audit to at least 20% of the pool. Thus 20% is a floor, not an impossible fixed size. Require at
least 90% exact binary agreement on audited items, review every disagreement blind, and freeze
adjudicated labels. If one rubric clarification is necessary, apply it to the complete pool and
draw a fresh audit sample. A second failure stops the run.

Pilot pool construction on ten development queries before full judgment. Report unique artifacts,
packet bytes and reviewer minutes. Cap development at 3,000 unique query-artifact judgments and
confirmation at 1,500; exceeding a cap stops for owner direction rather than truncating a top ten.

Confirmation packets and labels remain inaccessible until the one-shot confirmation claim.

## Metrics and statistical unit

Freeze the metric implementation and practical margins before any V0 or T0 quality result. Score
only judged top-ten artifact results.

Primary metric:

- binary useful-artifact `Precision@10` on fresh `short_context` queries, averaged within term and
  then equally across terms. A list shorter than ten has ten as its denominator.

Secondary metrics:

- binary nDCG@10 with gain `g(rel)=rel` and IDCG computed from all judged positive artifacts in the
  frozen pool, capped at ten;
- pooled Recall@10 (named pooled because corpus-wide relevance is not exhaustive);
- MRR@10 and at-least-one-useful-artifact success@10;
- per-term and secondary-tag results, wins/ties/losses, supporting-passage audit, distinct artifacts,
  latency, stored bytes and runtime resident bytes; and
- spent bare-adaptation and longer-control results, reported outside the primary estimate.

A query win/loss is the sign of its candidate-minus-v1 Precision@10 difference with exact zero a
tie. A term win/loss is the sign of its mean short-context difference. The independent unit is the
roster term: bootstrap terms with seed `19019`, carrying all their query intents together. Report
percentile 95% intervals. Development intervals guide a locked selection; they are not
post-selection proof.

Baseline/headroom systems are BM25, v1 dense, v1+DBSF, V0-compose dense, Stella dense and
T0-teacher routes after the pool freezes. Call Stella a headroom reference, not a ceiling.

Training-free execution proceeds to candidate scoring only when:

- at least eight terms and two fresh development short intents per term remain judgeable;
- Stella dense exceeds v1 dense by at least `0.05` term-macro Precision@10 on the modifiable
  `short_context` slice; and
- Stella wins on at least three more distinct terms than it loses versus v1.

If not, retain v1 and close without claiming the teacher proves a capacity limit.

## Numerical and serving gates

Before quality scoring, require:

- loader parity maximum absolute error at most `1e-6` before quantization;
- int8 query-vector maximum absolute error at most `0.02`;
- pre/int8 bare-direction cosine gates stated above;
- exact no-added-token ranking parity with released v1 on fixed fixtures;
- candidate resident codes-plus-scales bytes no greater than v1's corresponding compact arrays plus
  `(1024 + 4)` bytes per added row;
- no full-table float32 runtime copy;
- median and p95 encoder latency no worse than both `1.20x` and `+0.05 ms/query` versus v1 on one
  fixed 10,000-query sequence after warmup; and
- artifact-collapsed end-to-end median/p95 latency no worse than `1.20x` v1 on the same query order.

Pin host, thread counts, warmup, repetitions and query-sequence hash. Report process RSS and
temporary peak allocations descriptively; do not confuse stored, resident and eager-float bytes.

## Development eligibility

T0-teacher is eligible only if all numerical/judgment gates pass and, versus released v1:

- fresh-short-context Precision@10 improves by at least `0.03` absolute;
- the term-bootstrap 95% interval for that primary delta has a lower endpoint above zero;
- more than half of roster terms win and winning terms outnumber losing terms by at least three;
- fresh-short-context binary nDCG@10 delta is positive;
- T0-teacher+DBSF Precision@10 is noninferior to v1+DBSF under a one-sided 95% term-bootstrap lower
  bound of `-0.02`, with point delta no worse than `-0.01`;
- neither the required numeric/version target-control slice nor the complete longer-control slice
  has a negative point delta with an interval wholly below zero; and
- every artifact responsible for a primary win passes the supporting-passage audit.

V0-compose is descriptive and cannot be selected. There is one deterministic candidate, no
checkpoint selection, seed variance or second seed. If the primary interval or hybrid
noninferiority claim is unresolved, retain v1 and record `ENCODER_INCONCLUSIVE`, not equivalence or
improvement. A clear miss records `ENCODER_NO_MEASURABLE_IMPROVEMENT`.

## Resumable one-shot confirmation

Before accessing confirmation, commit and push a decision lock containing the selected candidate,
exact bundle hashes, frozen development qrels/results, term roster, formula/scale, artifact-collapse,
pool/evidence/judgment/metric recipes, numerical gates and all inheritance identities.

One-shot means one decision-locked confirmation experiment, not one uninterrupted process. Use
immutable states:

`locked -> claimed -> pools-frozen -> judgments-in-progress -> qrels-frozen -> scored -> complete`.

Claim before exposing query content. Each pool, packet, judgment batch and adjudication is hash-bound
and atomically checkpointed. An interruption may resume the same state with unchanged inputs; it may
not alter the candidate, rubric, pool recipe, thresholds or reviewer identities. Expose no quality
metric before qrels freeze. An unrecoverable failure marks confirmation consumed/incomplete and
retains v1. Rehearse interruption/resume across the judgment boundary on synthetic fixtures.

Name the primary confirmation judge and independent auditor in the decision lock. Apply the same
complete-pool, at-least-20% audit, 90% agreement, disagreement and supporting-passage rules as
development.

Confirmation passes only when:

- primary fresh-short Precision@10 delta is at least `+0.03` with a term-bootstrap 95% lower
  endpoint above zero;
- more than half of roster terms win and winning terms outnumber losing terms by at least two;
- fresh-short nDCG@10 delta is positive;
- hybrid satisfies the same point `-0.01` and one-sided lower-bound `-0.02` noninferiority margins;
- no registered safety slice has an interval wholly below zero; and
- the decision-locked bundle/serving identities remain unchanged and the new confirmation
  supporting-passage audit passes the same frozen rubric.

If the bounded confirmation sample cannot resolve improvement or noninferiority, record
`ENCODER_INCONCLUSIVE` and retain v1. Confirmation cannot change the candidate, terms, qrels policy,
metrics or thresholds.

## Reviewer process

Use the smallest review set that protects consequential transitions:

1. **Plan:** Astra adversarially reviews this file for task alignment, leakage, statistics,
   feasibility and overengineering. Resolve every P0/P1 and re-review material changes.
2. **End-to-end implementation/protocol:** before real judgments, one implementation reviewer and
   one fresh Astra reviewer inspect inheritance guards, actual row algebra, tokenizer/export,
   artifact collapse, pool blinding, metrics and the synthetic confirmation-resume transaction.
3. **Judgments:** one primary blinded judge plus the fresh independent audit defined above. A query
   author may not be its sole judge.
4. **Decision/confirmation:** independently recompute development eligibility and all hashes before
   the irreversible claim. Reconcile confirmation receipt and final outcomes afterward.

Only one Astra reviewer runs at a time. Reviewer prompts name a bounded file set, forbid protected
reads and require read-only operation. Save prompts, reviewer/model identities, access logs,
findings, dispositions and verification under `m19/reviews/` or `results/m19_review_*`. Reviewers
never edit results or relax gates. Two independent GOs are required before real judging and before
confirmation. Automate final identity reconciliation rather than adding another narrative review.

## Ordered execution

1. Create `m19/`, `m19src/`, `work/m19/`, fresh guards/registry and the inheritance lock.
2. Inventory fragmented terms and freeze the 8–12-term roster before quality retrieval.
3. Implement row construction, int8 export/loader, artifact collapse, pooling, metrics and the
   resumable confirmation state machine; pass synthetic end-to-end rehearsal.
4. Build, deduplicate and seal development/confirmation query text and families. Keep confirmation
   inaccessible.
5. Build V0-compose and T0-teacher; pass every algebraic, tokenizer, export, memory and latency gate.
6. Run the ten-query pool-cost pilot. Stop for owner direction if the fixed caps would be exceeded.
7. Build and blind the complete development pool; judge, independently audit and freeze qrels.
8. Score all development systems once, apply headroom and eligibility rules, and decision-lock the
   selected encoder/system. Do not access confirmation if T0 is ineligible unless a separately
   registered system-only audit needs it.
9. For an eligible T0, run the resumable fresh confirmation transaction once.
10. Finalize the artifact-collapsed internal query command and record both outcome axes.

No stage proceeds merely because compute or context remains.

## Required artifacts and outcomes

Create and maintain at least:

- `m19/STATUS.md`, `PLANNING.md`, `registry.json`, `LEDGER.md`, `FINDINGS.md`, `CODEMAP.md`;
- inheritance/execution/decision locks and path-protection tests;
- one-command synthetic rehearsal including interrupted confirmation resume;
- term/support/tokenization manifest and per-row algebra/parity receipt;
- query/family/pool/evidence-packet manifests and blinded judgments/audits;
- artifact-collapse parity and legacy M18 recall/diversity postmortem;
- machine-readable baseline, headroom, per-query/per-term, interval and eligibility results;
- int8 bundle hashes, loader, stored/resident/temporary bytes and latency results;
- confirmation state/lock/result/receipt when applicable; and
- final system manifest with the exact local query command.

Final status records:

- `SYSTEM_READY` or `SYSTEM_BLOCKED`; and
- `ENCODER_INTERNAL_CANDIDATE`, `ENCODER_NO_MEASURABLE_IMPROVEMENT` or `ENCODER_INCONCLUSIVE`.

`SYSTEM_READY` requires a usable artifact-collapsed BM25+dense DBSF path over the inherited snapshot.
`ENCODER_INTERNAL_CANDIDATE` requires every development, confirmation and serving gate. Otherwise
released Zero v1 remains selected. Never call an unresolved interval equivalence, never promote a
qualitative top ten as a measured win, and never claim that this one deterministic repair exhausts
the broader tokenizer-shattering problem.
