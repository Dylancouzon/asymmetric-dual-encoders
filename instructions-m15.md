# M15 — whitepaper and evidence package

Previously M14; moved 2026-09-10. One paper is the default. Runs after M13's measurement and
normally M14's release; runs on the zero-only frontier if nano does not land. Release is optional,
credible evidence is not. Working files belong under `m15/` when writing starts.

## Deliverable

An empirical study of replacing the query encoder while preserving a pretrained document index:
quality and deployment cost for a lookup table and compact transformer, with reproducible evidence,
negative results, limitations and the comparator table intentionally kept off model cards.

## Evidence and claims

- Headline = pre-registered clean-4 (`nfcorpus`, `scidocs`, `scifact`, `trec-covid`) for both zero
  and nano; always show all six. No re-picking after results. Disclose stella's ArguAna/FiQA/FEVER
  exposure. Clean-4 means no disclosed overlap; all-six minus clean-four is partition sensitivity.
- Teacher choice is an empirical counterexample to selecting on tower quality, not a universal
  law: eleven teachers measured, Spearman over eight / seven comparable rows. Keep denominators
  explicit (`results/m7_learnability_report.json`, `results/m7_offfamily_report.json`).
- Include the baseline matrix, M7's missed dense bar, M8's negative probes, M9's dataset-dependent
  retention and two-lock build provenance, M10's screen and M12's deployable fusion result.
  `m10/FINDINGS.md` names the limits of the coverage, width, synthetic-data and seed claims.
- Zero makes the near-zero-query-compute claim. Nano must demonstrate quality at comparable edge
  cost while sharing the document index. Include assets, index, hydration, latency and RSS under
  the same harness; distinguish cloud fusion from measured Edge behavior.
- Unresolved superiority is not equivalence. Label descriptive differences and query-only
  intervals. Do not claim zero confirmatorily beat BM25: its M7 C2 failed the Holm threshold.
- Cite pyNIFE as prior art for the frozen-teacher lookup-table construction. Claim the empirical
  evidence and deployable implementation, not architectural priority.
- Any validation-only MS MARCO diagnostic carries its licence role and comparator-training
  exposure caveat. It never enters training or replaces the headline benchmark.

Every headline must trace to a committed result and its registered analysis. New measurements
belong in a separately scoped M13 follow-up, not in an improvised paper paragraph. Finish with
an independent review of the evidence and draft; disclose the findings and dispositions.

Full prior writing rules and withdrawn claims:
`research/archive/m10-cleanup-2026-09-10/instructions-m14.md`.
