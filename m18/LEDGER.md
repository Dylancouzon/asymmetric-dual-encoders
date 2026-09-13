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
