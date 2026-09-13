# M19 development judgment policy v1

Scope: label every item in the frozen blinded development evidence packet `abd6b47496423de275f663d8620ab4e4b895e36af2eb5a0f234bc65641368d69`.

Primary reviewer identity: `astra-m19-development-primary-01`.

For each item, use only its query text, term, artifact title/type/location, parent metadata, and displayed passages:

- `1`: the artifact is materially useful for satisfying, implementing, diagnosing, or answering the query, and that utility is supported by the displayed frozen evidence.
- `0`: the match is coincidental, misleading, about a materially different problem, or the displayed evidence does not make the artifact materially useful for the query.
- `"unjudgeable"`: the artifact is plausibly material but the frozen packet lacks enough context to choose `0` or `1`. Never coerce uncertainty to zero.

Judge artifact relevance independently of any retrieval system. Multiple artifacts may be positive. Do not infer route, model, score, rank, or experiment outcome. Do not open the pool manifest, metric runs, source repository, web, confirmation data, prior evaluations, or any content outside the frozen evidence packet and this policy.

Process deterministic packet-order batches under the same reviewer identity. Each item must receive exactly one label. Publish one final JSON object mapping every packet `item_id` to the exact JSON value `0`, `1`, or `"unjudgeable"`. Do not calculate aggregate counts or quality metrics while judging.
