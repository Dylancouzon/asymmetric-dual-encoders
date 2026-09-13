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
