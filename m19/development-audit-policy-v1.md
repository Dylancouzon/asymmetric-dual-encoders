# M19 development audit policy v1

Scope: independently label every item in the concealed-repeat audit packet `4e12bb8b97f479f4966ae1133407b5bacded71e18b1a8da180f305027cb56db4`.

Auditor identity: `astra-m19-development-auditor-01`.

Use only each item's query text, term, artifact title/type/location, parent metadata, and displayed passages:

- `1`: the artifact is materially useful for satisfying, implementing, diagnosing, or answering the query, supported by the displayed frozen evidence.
- `0`: the match is coincidental, misleading, materially different, or unsupported by the displayed evidence.
- `"unjudgeable"`: the artifact is plausibly material but the packet lacks enough context to choose `0` or `1`. Never coerce uncertainty to zero.

Judge independently. Do not seek or infer the primary item identity, primary label, audit-selection reason, route, model, score, rank, or experiment outcome. Do not access the internal audit sample, primary labels, pool manifest, metric runs, source repository, web, confirmation data, prior evaluations, or other files.

Process deterministic packet-order batches under the same auditor identity. Publish one final JSON object mapping every concealed packet `item_id` exactly once to `0`, `1`, or `"unjudgeable"`. Do not calculate agreement or aggregate relevance/quality metrics.
