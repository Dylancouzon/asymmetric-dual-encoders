# M19 findings

## Term feasibility gate

Twelve prioritized terms passed the corrected non-quality roster gate. The weakest observed
serving-consistent support was still 13 distinct non-equivalent Qdrant artifacts, so the registered
minimum of eight terms/five artifacts is not the limiting factor. Catalog and first support result
entered git together, so there is no independent pre-scan catalog receipt; no retrieval quality
informed selection. This establishes only source-qualified corpus demand and tokenizer
fragmentation, not Stella headroom or retrieval improvement.

## Deterministic row construction

The direct teacher-row algebra is numerically comfortable on all 12 terms: after independent int8
quantization, the worst T0 bare-term cosine is `0.9999541` and the largest coordinate error is
`0.0005364`, far inside the registered `0.999`/`0.02` gates. The added resident cost is exactly
12 × 1,028 bytes. Stella's cached custom model defaults to xformers, but the released Zero config
already pins its supported eager PyTorch path (`use_memory_efficient_attention=false`,
`unpad_inputs=false`); using those frozen settings produces deterministic teacher bytes locally.

## Serving cost

The superseding official 10,000-query benchmark finds no serving regression from the T0 row lookup
under the registered exact route. Against the actual released V1 loader, encoder median and p95 are
`0.9580x` and `1.0370x`; the full inherited hybrid path is `0.9956x` median and `0.9696x` p95.
Released-loader numerical parity is `2.24e-8`. These are cost and parity findings only; the
benchmark discarded rankings and did not inspect relevance. The earlier v1 measurement used a
shortcut route and is preserved only as superseded evidence.

## Pool-cost feasibility

The ten-query, ten-term pilot yields 224 query-artifact judgment items and a 265,914-byte blinded
packet. Linear projection to the complete 60-query development split is 1,344 items, well below the
registered 3,000 cap. The evidence-volume model projects about 12.7 reviewer hours for complete
primary labeling; that is a conservative planning estimate, not an observed judgment duration.
This supports mechanical pool feasibility but makes reviewer throughput the dominant execution
cost. No relevance was assessed in deriving it.

## Development judgment outcome

The full pool contained 1,376 items, close to the pilot projection of 1,344. Primary model judgment
completed all items, and the deterministic independent audit included 925 concealed repeats because
all primary positives and unjudgeables must be audited in addition to seeded negatives. The auditor
retained 60 items as unjudgeable after a dedicated evidence-only reconsideration: the displayed
passages did not contain enough context for a defensible binary decision. M19 therefore stopped
before agreement, qrels, headroom or candidate-quality scoring. This is `ENCODER_INCONCLUSIVE`, not
evidence of improvement, non-improvement, equivalence or a capacity limit. Released Zero v1 remains
selected; fresh confirmation stayed sealed.
