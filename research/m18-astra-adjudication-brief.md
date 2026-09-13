# M18 prospective qrel adjudication brief

Judge every row in `results/m18_adjudication_candidates.jsonl` from its source text only. No
retrieval output exists and no encoder identity may be considered. This pool predates the final
development/confirmation split.

For each candidate, emit exactly one JSONL decision with `query_id`, the unchanged
`candidate_sha256`, `label`, `stratum`, and a short `reason`.

- `clear`: the target directly answers the query with a substantive explanation, resolution, or
  actionable project guidance, and the query is understandable without an unseen diff.
- `partial`: the target is topical but diagnostic, speculative, deferred, incomplete, or mainly a
  request for more information.
- `bad`: the target answers another question, is status-only, or the query lacks standalone
  meaning.

Only `clear` is retained. Use one of the five registered strata. Correct the proposed stratum only
when the query is plainly misclassified; do not manufacture balance or upgrade uncertain pairs.
