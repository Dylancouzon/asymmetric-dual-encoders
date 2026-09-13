# M19 development sensitivity result — Astra reconciliation

Date: 2026-09-13

Reviewer: `m19_sensitivity_result_astra` (`gpt-6-astra`, high reasoning)

Disposition: **GO; no P0/P1 findings.**

Astra checked the frozen amendment, implementation, tests, aggregate result, registry and pre-result
method review. The reported bounds and statuses match the registered thresholds. In particular,
dense Precision, winning terms and net wins are label-sensitive; headroom, dense nDCG, hybrid and
both controls robustly pass. Numeric/version preservation correctly passes at exact equality
`-0.02`. `LABEL_SENSITIVE` is warranted but does not imply one jointly eligible assignment.
Supporting passages remain `not_evaluated`, and the result preserves `ENCODER_INCONCLUSIVE`, creates
no qrels and makes no confirmation claim.

P2 recorded without remediation: the result hashes all six development inputs but not the tracked
registry that supplies thresholds. The method/result commit history and this review retain that
provenance; the diagnostic is not being expanded or rerun.

Access declaration: the reviewer read only the six authorized tracked method/result files. It did
not execute code or access `work/`, other results, individual queries, labels, packets, qrels,
confirmation or any protected/spent surface.
