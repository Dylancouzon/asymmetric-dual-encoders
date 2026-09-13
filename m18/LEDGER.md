# M18 ledger

## 2026-09-12 — E0 execution authorization and initial lock

- Dylan authorized executing `instructions-m18.md`, including local collection, training and the
  fresh M18 evaluation surfaces. M7–M13 protected/spent evaluation content remains forbidden.
- Pinned `qdrant/qdrant` default branch `master` at commit
  `5e32ea89cb5ea9a6827a68b27291d2e2251e1b1c`; GitHub export cutoff
  `2026-09-13T00:36:26Z` (UTC).
- The staged released Zero v1 is accepted only because its `model.npz`, tokenizer and config hashes
  are pinned in the registry and the model hash equals `m7/FREEZE.json:table_sha256`.
- The M7 unfolded checkpoint is the only training initialization. M17 endpoints are historical
  negative evidence and are not M18 inputs.
- GitHub collection uses repository-wide REST list endpoints for issues/PR openings, issue
  comments, review comments and issue events. This preserves explicit page receipts without one
  request per thread. Cross-reference/closing links are taken from event/link fields; unsupported
  timeline payloads are not silently inferred.
- Evaluation target is 250 queries: development 35 and confirmation 15 per realized stratum. The
  mandated minima govern short strata. Split assignment happens on deduplicated artifact families.
- The T0 diagnostic uses a 4,000-step scheduler with `stop_after=500`; a separate 500-step schedule
  would change the optimization trajectory and make the later resume incomparable.
- The primary evaluation is held-out question retrieval over the complete pinned snapshot, not a
  claim that only documents predating each question are searchable. Query-bearing openings remain
  distractors and are never relevant solely because they share a thread; qrels point to distinct
  answer spans and the self-hit rate is reported.
- API cutoff means objects/events created no later than the cutoff, with mutable bodies captured as
  observed during the recorded acquisition interval. M18 does not claim GitHub reconstructs body
  versions at the exact cutoff instant; post-cutoff updates are counted in reconciliation.
- No M18 result has been observed at this entry.

## 2026-09-12 — E1 minimum fork and synthetic rehearsal

- Copied from M17 at `4213920`: `common.py` (`caa1e927…`), `vocab.py` (`9ab9a30a…`),
  `cache.py` (`4d1e0ca4…`), `train.py` (`78f7ab12…`), `export.py` (`c3a377b5…`),
  `loader_np.py` (`140cdf1b…`) and the rehearsal structure (`7bbe396f…`). Evaluation copied only
  the generic exact-search/metrics/bootstrap ideas; none of M17's dev/panel readers came across.
- Semantic differences: fresh M18 paths/guards; project-only support rather than M17 domain caps
  and pins; colon compound extraction; only appended rows are Parameters; no learned-scalar
  optimizer; fixed 4,000-step identity plus execution-only stop; T1/T2 serving preprocessing;
  atomic cache/snapshot/bundle publication; no checkpoint averaging or parameter cap; fresh
  M18 structural surfaces and dense/BM25/DBSF evaluator.
- The canonical synthetic rehearsal passed. It parsed 42 document units in 21 artifacts, created
  10 development and 5 sealed-confirmation questions across all five strata, resumed steps 2→4
  to bit-identical rows versus an uninterrupted run, kept inherited rows unchanged, moved a new
  row, passed int8 loader gates, and refused a second confirmation claim. Receipt:
  `results/m18_rehearsal.json`. These fixture metrics are not quality observations.
- A planning Astra review found the schedule/freeze/atomicity/protocol issues above. It reviewed
  no implementation and does not count as either pre-expensive-run implementation review.

## 2026-09-13 — E2 first implementation review and remediation

- The first Astra implementation review was a no-go for live acquisition/GPU work. It found that
  title-only splitting could separate one artifact family and conceal duplicate-answer leakage;
  broad answer cues admitted information requests; confirmation reads bypassed the one-read rule;
  live payload edits broke byte-level REST reconciliation; and training/export did not bind every
  realized array, tokenizer and table identity.
- Remediation connected families across artifact IDs, exact answer digests, title/backport shapes
  and explicit Qdrant thread links. Split reports now compute actual artifact/family/answer-digest
  intersections. Generic log/detail requests are rejected, targets must be later answer-bearing
  chunks, discussion objects are chunked before qrels, and verbatim documentation self-pairs were
  removed.
- REST reconciliation now compares the cutoff-admitted canonical identity set while recording
  mutable payload changes and a growing post-cutoff tail; the first-pass bytes remain immutable.
  Corpus parsing requires the combined source manifest and all page receipts in real execution,
  and repository bytes are read from the pinned Git object rather than the worktree.
- Confirmation can only be loaded inside a decision-locked, hash-verified, exclusive one-shot
  transaction. Training resume binds ordered query IDs/student IDs, teacher vectors, document
  bank, candidates, scores, aliases and initial rows. Snapshot load reconstructs and verifies the
  concatenated table. Pair-aware batches co-locate supported alias views. Export binds variant,
  tokenizer, preprocessing and float-table hashes and compares the actual torch training forward
  path to serving.
- The revised synthetic rehearsal passes 26 network-free tests. It now reports zero artifact,
  connected-family, answer-digest and development/confirmation overlap; confirmation is exercised
  only through the one-shot transaction. A second independent implementation review is pending.

## 2026-09-13 — E3 second review and live pagination correction

- The second independent review approved live raw acquisition but held development reads, full
  Stella encoding and training behind stale-cache, label, index and execution-lock fixes. Those
  focused blockers were corrected and covered by 28 network-free tests; no broad framework was
  introduced.
- Live GitHub acquisition reached issue page 100 and received HTTP 422: GitHub now requires the
  opaque `after` cursor from its Link header for large collections. The adapter was corrected to
  carry that cursor while retaining explicit page numbers and immutable receipts. The first 99
  validated pages resumed in place because their endpoint, order, page and payload identities are
  unchanged; page 100 onward records the cursor in each request receipt.

## 2026-09-13 — E4 realized corpus and protocol review

- The reconciled pinned snapshot contains 71,937 canonical GitHub objects. Parsing the Git tree and
  discussion snapshot yielded 105,025 records across 11,574 artifacts: 79,269 indexable document
  units and 25,756 non-indexable relationship events. The terminal issue-event page was full at the
  server's 30,000-object cap; this is disclosed auxiliary metadata and never supplies a retrieval
  document or sole qrel justification.
- The first realized protocol contained 1,594 training views, 175 development questions and 75
  sealed confirmation questions. It had zero measured artifact, connected-family, answer-digest or
  development/confirmation leakage. Confirmation content remained unread.
- Astra's deterministic 40-development-qrel audit found 24 clear answers, eight partial/diagnostic
  targets and eight wrong targets. It also found that 174/175 query-bearing source documents contain
  the query verbatim, making their unjudged presence a systematic evaluation confound. Corpus scale,
  provenance, deduplication and leakage controls were accepted; scoring was not.
- The prospective v3 qrel rule no longer admits thread closure plus generic words as an answer. It
  requires substantive resolution, linked explanation or closing-event plus strong explanation;
  rejects information/clarification and unrelated CI/codespell/rebase/push status; and retains the
  cleaner explicit review-reply relation. Evaluation removes, but never marks relevant, each
  query's own source document and records its pre-exclusion top-10 rate.
- Task-style PR titles no longer fill the concept/how-to audit stratum. The rebuilt development
  composition is 25 review questions and 10 issue openings there. Overall, the v3 protocol has 921
  structural candidates, 1,043 training views, 175 development questions and 75 sealed confirmation
  questions; all five strata remain at 35/15 and all measured leakage intersections remain zero.
  Full Stella encoding remains paused until Astra re-audits this rebuilt development set.

## 2026-09-13 — E5 final narrow qrel hardening

- A focused diff review found that bare “need/want logs” requests and interrogative or negated
  resolution language could still pass the v3 predicates. The v4 rule rejects those forms in both
  opening/comment and explicit review-reply candidates, with falsifying fixtures for each case.
- Generic review questions without project-specific error, configuration, identifier or jargon
  context were also removed from the concept/how-to audit stratum. They depended on an unseen code
  diff even when their replies were substantively correct. That stratum now consists of 35 issue
  openings; no synthetic replacements were made.
- The rebuilt v4 protocol has 733 structural candidates, 901 training views, 175 development
  questions and 75 still-sealed confirmation questions. All five strata retain their 35/15 target,
  every measured leakage intersection remains zero, and 39 network-free tests pass. Previously
  flagged IDs `5110`, `7632` and `9159` remain only because v4 selects later, substantive answers
  for them; their earlier diagnostic/status targets are gone.

## 2026-09-13 — E6 use-case alignment and adjudication scope

- The session-external review in `m18/REVIEW.md` correctly identified that the numeric/identifier
  audit slice does not reproduce bare terms such as `s3` and `k8s`. Its premise that issue bodies
  are evaluation queries is incorrect: issue/PR queries are titles, while bodies stay in the corpus
  and provide adjudication context. Measured median title/question length is nine words overall and
  12 in exact/numeric; only one exact/numeric development query has five or fewer words.
- Adopt the review's smallest direct acceptance artifact after vocabulary preparation: a fixed,
  report-only side-by-side top-10 for v1 and the selected candidate on the supported bare terms. It
  has no manufactured qrels and does not replace the registered metrics. Do not add a sixth gated
  stratum or wait on owner-authored questions; voluntarily supplied questions can still be added in
  a later milestone as originally ruled.
- Astra's v4 audit was still no-go: 27/40 clear, eight partial and five bad. More lexical exceptions
  would not test whether an answer actually addresses a question. The prospective pool is therefore
  capped at the newest 80 connected families per proposed stratum (344 pairs total after source-text
  duplicate union, bot-review removal and prose-question validation). Astra sees only query, source
  context and proposed target, before the final split exists; it sees no retrieval scores or model
  identity. Only `clear` pairs survive. This is a one-pass M18 data correction, not a reusable
  judgment framework.

## 2026-09-13 — E7 prospective adjudication executed

- One Astra judge reviewed all 344 content-bound prospective pairs without retrieval output, model
  identity or a pre-existing audit split. It rated 141 clear, 141 partial and 62 bad, and corrected
  94 plainly mismatched strata. The decision file covers the exact candidate ID set and copies each
  source/query/target identity hash; its SHA-256 is
  `d98144096f17a718bb70073655fbcd4f28319c75f41d670f0a376118aa505904`.
- Only the 141 clear pairs were split: 101 development and 40 sealed confirmation. Alias, concept
  and exact/numeric have fewer than 35 available clear families, so their smaller realized counts
  are disclosed rather than filled with partial labels. Confirmation content remains unread.
- The surviving clear pairs all enter held-out surfaces, leaving no adjudicated labeled structural
  training rows. To preserve project-language support without reintroducing bad positives, 297
  non-held-out issue/PR titles are retained through the existing teacher-only/query-only cache path;
  review comments are excluded from this fallback. Together with 482 evidence-backed alias views,
  the training protocol has 779 views. No rejected target is used as a relevance label.
- The v5 manifest again reports zero artifact, connected-family, answer-digest and
  development/confirmation overlap. Query-source documents remain excluded before evaluation
  metrics. Forty-three network-free tests pass. A fresh Astra reviewer must still spot-check the
  final development surface before full Stella encoding.
