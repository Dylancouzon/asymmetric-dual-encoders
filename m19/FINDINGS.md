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

The official 10,000-query fixed-sequence benchmark finds no measurable serving regression from the
T0 row lookup. Encoder median and p95 are slightly lower than V1 (`0.9590x` and `0.9763x`), and the
full inherited hybrid path is effectively unchanged (`0.9944x` median, `0.9969x` p95). These are
cost and parity findings only; the benchmark discarded rankings and did not inspect relevance.
