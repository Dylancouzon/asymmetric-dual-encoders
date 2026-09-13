# M19 development-only uncertainty sensitivity amendment

Owner authorization: 2026-09-13. This is a bounded, explicitly post-hoc diagnostic of the closed
M19 development judgment stop. It does not rewrite `ENCODER_INCONCLUSIVE`, create qrels, reopen M18,
or authorize any M19 confirmation access.

## Fixed analysis

Read only the frozen M19 development pool, metric runs, query specifications, primary labels,
concealed audit packet and audit labels. Do not read individual evidence passages or emit query,
artifact or item identifiers.

Map each concealed audit repeat back to its frozen query-artifact identity. A label is **known**
only when it is a binary primary label that either was not audited or exactly matches a binary
auditor label. A query-artifact label is **unknown** when either reviewer used `unjudgeable` or when
two binary audit labels disagree. This deliberately includes more than the 60 auditor abstentions.

For every query, enumerate every binary assignment to its unknown labels. Use the already frozen
metric runs and `m19src.metrics.score_query`; never choose an assignment. Aggregate the exact
minimum and maximum attainable candidate-minus-v1 contrasts under the registered term macro:

- Stella-v1 short-context Precision@10 headroom;
- T0-v1 short-context dense Precision@10 and nDCG@10;
- T0-v1 short-context hybrid Precision@10;
- T0-v1 numeric/version and all longer-control Precision@10; and
- attainable short-context term win/loss/tie counts.

This per-query enumeration is the correct contrast bound. The simpler pair “all unknown zero” and
“all unknown one” is not a bound when an unknown artifact appears in only one of the compared
routes.

Classify each numeric gate as:

- `robust_pass`: every assignment passes the original threshold;
- `robust_fail`: no assignment passes it; or
- `label_sensitive`: some assignments pass and some fail.

Report resolved binary audit agreement and its denominator separately from uncertainty coverage.
The missing supporting-passage win audit remains `not_evaluated`; it may not be inferred from label
bounds. Publish aggregate-only `results/m19_development_sensitivity.json`, run the affected tests,
and stop. Confirmation stays sealed regardless of the result.
