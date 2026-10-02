# v20 dispositions: coherent and value-dense

Owner ruling: retain the title; implement the Astra/Sol recommendations while keeping the paper interesting and value-dense. Source reviewed: v19 at 68509e7; reports and synthesis committed in 7ae7a6b. This is implementation disposition, not a new independent review or publication sign-off.

## Applied

- Both: explicit exact-quality then additional graph-search logic near the RQs and at the §5 transition; trained Constella illustration at §6; short discussion synthesizing quality/selection, recurring recovery, and the measured encoding/search trade-off.
- Both: RQ1 and §4 now concern absolute student retrieval quality, with retention ratios used explicitly in Constella. RQ2's prediction endpoint is the recovery gap at fixed effort. Preserved exact/ANN/relevance/effort/timing distinctions and all uncertainty.
- Sol: prospective support is head-specific in the introduction; table uncertainty remains. Abstract now names linearly fitted representations rather than asserting all are smaller/faster.
- Astra: bge-head construction versus inference distinction stated where architecture is introduced; original relevance described as a measured query-path score rather than inherited solely from document encoder.
- Astra: identify positive-preserving MS MARCO diagnostic, 6,980 development queries and separate encoding sample before Table 2. Full sampling, seed, corpus size and consequences in Appendix C. Precision table explicitly reports the 40-candidate budget. Named E2/E18 sources govern.
- Both: geometry is a predictive susceptibility marker, with poor prediction by the student-minus-original change explained. H2 separates measured effort from the proposed placement explanation; H2d tests recurrence in joint training. No causal mechanism claimed.
- Astra: clean-four sensitivity stated concisely, with E21 coefficient intervals and development rank changes in Appendix B. Construction costs label cross-space screening and separate release optimization, including excluded evaluation costs.
- Optional cuts adopted: original Table A relocates intact to Appendix B as Table B1; §4.4 regression/backbone tail moves there. Synthetic-prefix and internal-specialization fragments leave the manuscript, retained in historical source and companion evidence. Repaired fp16 export anecdote removed; actual served precision retained.

## Preserved deliberately

The title, strongest-teacher/weakest-student reversal, dimensionality-conditioned result, prospective test, unfavorable top-choice performance, mixed relevance-loss behavior, LightRetriever replication, separately trained model recipes, registered adverse reference comparisons and uncertainty. Fusion/routing remain subordinate. No historical inventory promoted and no new experiments launched.

The new discussion uses concrete results and their combined implication, rather than adding generic positioning. Main text including tables grows from 5,273 to 5,379 words (about 2%); the added synthesis and missing methods explanation offset relocated detail. This is a value-density judgment, not proof of readability or owner acceptance.

## Evidence and validation

Title and every scientific table line, including the relocated dimensionality summary, are exactly preserved as a multiset against v19. New sensitivity values checked in results/m15_e21_width_model.json; timing sampling in MEASUREMENTS.md E2; primary precision budget in results/m15_e18_query_precision.json. No result payload changes, protected reads, retraining, or new measured values. PDF rendering, whitespace and build syntax checks complete the local validation; detailed final result in LOG.md.
