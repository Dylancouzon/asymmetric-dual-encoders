# V14 essential scientific scope review, 2026-10-01

**Disposition: no essential scientific correction identified in the revised main text.** This is a focused scope check using the previously reviewed numbers and experimental controls, not a repeated result audit.

- **Readable definitions preserve the measurements.** Exact search, macro nDCG@10, retention, ANN effort, rescoring, and oversampling are described without conflating relevance with exact-neighbor recovery. Zero's square-root count explanation agrees with the release encoder. Compatibility means available aligned vectors can query an unchanged collection; no concurrent switching or loading performance is claimed.
- **The timing protocols are now distinguishable.** Section 4 identifies the approximately 62-fold encoding ratio as measured on its own diagnostic queries and explicitly separates it from Table 1's approximately 51-fold ratio. The 2.2-fold ratio includes separately measured encoding and retrieval and preserves each encoder's own quality target. The shared binary example and separately selected compression configurations are identified. Neither is an equal-absolute-quality comparison.
- **The precision inference stays within its control.** The native replication's weaker primary interaction remains visible. The exhaustive experiment targets each encoder's own original-vector top ten, averages expected inclusion at score ties, and reports absolute coverage gains at the primary global candidate budget. It does not imply native 8-bit performance, latency gains, greater proportional sensitivity, or a demonstrated causal explanation for Zero's larger absolute gain.
- **Section 6 distinguishes counterexamples from prevalence claims.** Repetition across two workloads is explicitly within one Stella-aligned family. Query intervals do not represent retraining or teacher-family uncertainty. Compatibility not certifying search settings and teacher quality not certifying the tested table are defensible practical counterexamples; the paper does not infer their frequency across architectures.
- **The teacher screen remains recipe-specific.** The registered ten versus later exploratory 16, wide teacher/table correlation interval, hindsight strongest-teacher choice, clean-set regret, domain sensitivity, incomplete exploratory exposure audit, and lack of validation as a selector for trained Zero/Nano are retained.

## Comparator visibility

The main text explicitly states that Nano scores below bge-small and LEAF on BEIR-15, explains their separate document indexes, and discloses the failed registered Zero–BM25 test. Appendix B retains Nano's unfavorable held-out LEAF result, the negative query-pooled bge comparison, unresolved comparisons, Zero's missed registered dense bar, and the unresolved fusion comparison. That is sufficient visibility to make Figure 1 focus on the three literal same-index alternatives without implying universal superiority.

At review time Figure 1's text still contained the hollow external-reference markers. Moving those references to Appendix B is scientifically acceptable provided the existing main disclosure and the appendix's unfavorable outcomes remain. The planned replacement image itself was not inspected in this review.

No additional experiment is required to support the present scoped claims. Native query-precision testing or another aligned teacher family would be needed for the corresponding stronger deployment or cross-family claims, which this draft already withholds.

## Access record

Opened only `m15/PAPER.md`, targeted count-pooling lines in `m11/release/zero_encoder.py`, and `m15/REVIEWS/2026-10-01-v13-correctness.md`. Previously verified numbers and controls were reused. No result file, protected content, raw query/qrels, or new experiment was accessed. Only this review file was written; no commit was made.
