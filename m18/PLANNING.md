# M18 execution plan

M18 builds two independent internal artifacts: a pinned Qdrant project-memory system and a
specialized Zero query table. Execution is complete: the system ships with released Zero v1 because
no specialized table was eligible. Nothing in M18 changes the public release. Realized held-out
counts are 100 development and 40 confirmation queries; the smaller clear-only pool is disclosed
instead of being filled.

## Ordered execution

1. Fork the minimum M17 vocabulary/cache/train/export/loader path into `m18src`, remove closed-M17
   locks and general-domain arms, implement inherited-row freezing, then pass one synthetic
   rehearsal including resume and export parity.
2. Acquire the exact repository commit and complete GitHub REST collections named in the registry.
   Raw pages are immutable, content-addressed, explicitly paginated and checked against observed
   API totals. Parser inputs never follow mutable branch heads.
3. Parse natural-language-bearing repository and discussion units, redact obvious credentials,
   deduplicate within stable artifact families, retain source metadata, and keep artifact-level
   coverage even if adjacent chunks must be merged below 200,000 units.
4. Split query-label families before producing variants. Earlier artifact families train; the
   newest credible maintainer-answer families supply a 175-query development set and a sealed
   75-query confirmation set (35/15 per stratum when available). Near duplicates are unioned
   before assignment. Confirmation text is stored separately and its bytes locked before the
   first development read.
5. Encode the full document collection once with pinned Stella; build a deterministic BM25 index.
   Freeze structural qrels and report overlap/leakage checks. Read Stella, BM25, v1, V0-Q and their
   DBSF@100 baselines on development.
6. Build the training-only support inventory, exact-token extension and shared raw-query
   Stella/candidate cache. T0 uses a 4,000-step schedule with an execution stop at step 500 so the
   diagnostic trajectory is identical to a resumable full run.
7. Read T0 steps 250/500. If both dense and fusion trail V0-Q, run the mandated parity report and
   stop learned arms when parity passes. Otherwise evaluate T1/T2 step 0, skip inert transforms,
   screen the fixed arms, and confirm an eligible chosen form with the second seed. No form was
   eligible in the realized run, so the second seed was not applicable.
8. Lock the recipe/checkpoint, read confirmation exactly once, export the best eligible internal
   int8 bundle (or retain v1), finalize the hybrid query command and record both final outcomes.

## Evaluation construction

Structural questions come from issue titles/bodies or review questions; targets are distinct
maintainer answers, resolving replies, linked closing artifacts, or documentation spans. The query
sentence is removed from its target, and exact-text overlap is reported. Synthetic generations may
augment training only and never define development/confirmation relevance.

Strata are concepts/how-to, errors/troubleshooting, configuration/API/operations, exact
identifiers/versions/numeric distinctions, and abbreviations/aliases/project jargon. Selection is
fused nDCG@10 under the margins and paired-bootstrap rules in `instructions-m18.md`; dense metrics
remain independently visible.

## Local budget

All work runs on this RTX 3080 host. Preparation is sized at two sample scales before a long job.
The maximum local-compute clock is 24 hours, excluding GitHub export wait. Atomic stage receipts,
resumable pagination, cache identities, checkpoints, host/GPU peaks and elapsed time are required.
