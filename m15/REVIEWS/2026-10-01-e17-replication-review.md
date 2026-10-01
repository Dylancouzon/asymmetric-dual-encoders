# E17 replication review and E18 assessment, 2026-10-01

**Outcome:** no essential execution/control defect found. E17 supports a binary-search penalty for every tier on the 1M diagnostic, but does **not** clearly reproduce E16's large extra Zero penalty at the primary setting. Report that attenuation directly.

## Source and protocol check

The E17 wrapper calls the shared E16 procedure with the 1M diagnostic, the fixed 192-query hash selection, and unchanged primary `ef=64` plus secondary 16/256 settings. Both the wrapper hash and the receipt-bound shared implementation hash match the inspected files. The original exact gate is 1.0 for all tiers. The collection has one million points/indexed vectors; before/after recorded configuration and counts agree. No model training or document re-encoding was introduced. The method's “no new encoding” should be read as no new **document** encoding: the implementation does encode selected queries with existing models.

The same-graph treatment remains a native option ablation, with the segment-merging and query-sign-encoding limitations documented in `2026-10-01-quantization-semantics.md`. It is not exhaustive quantized scoring or a graph/quantization decomposition.

## Primary result

At `ef=64`, binary + rescoring1 minus original-score traversal changes recovery by:

| Tier | Recovery change |
|---|---:|
| Zero | -3.9063 percentage points |
| Nano | -3.0729 percentage points |
| Stella | -2.1875 percentage points |

The extra Zero loss versus Nano is only -0.8333 points, with query-bootstrap interval [-2.8125, +1.1458]. Against Stella it is -1.7188 points, interval approximately [-3.4896, 0]. The stored upper endpoint of about -8.5e-19 is floating-point zero; it is not evidence that an interval strictly excludes zero. These exploratory intervals do not include graph-build uncertainty.

At secondary `ef=256`, extra Zero loss is -2.50 points versus Nano and -3.8542 versus Stella. These observations are useful evidence of setting dependence, but selecting that setting as the replication success would replace the stated primary analysis after seeing results. At `ef=16`, both interaction intervals also include zero. The correct combined account is a large FiQA pilot interaction, an attenuated/uncertain primary 1M interaction, and sensitivity to search effort.

E17 should therefore redirect the research from a universal “Zero is especially damaged by binary quantization” claim toward locating the actual source of candidate loss. It does not erase E2's original-score ANN difficulty, and neighbor recovery should not be equated with judged relevance: E17's nDCG changes sometimes have the opposite sign from recovery changes.

## E18: worthwhile and more discriminating

The proposed exhaustive comparison of **the same one-bit document codes** scored with (a) sign queries and (b) original graded queries is a good next bounded diagnostic. It holds corpus, document codes, original-tier target neighbors, and candidate budget fixed, while removing graph traversal and segment merging. It directly tests the consequence of discarding query-coordinate magnitudes against those fixed codes.

Required interpretation and controls:

- Use exactly the verified document sign convention, including zero coordinates. Sign-query dot sign-document scores are equivalent to Hamming ranking up to constants; graded-query dot sign-document scores are an asymmetric surrogate retaining query magnitudes. This is not native scalar8 query encoding and need not be an upper bound on that mode or the optimal score for binary documents.
- Keep each tier's original exact top 10 as its target. Compare candidate inclusion at C = 10, 40, and 100 for both scoring paths. These are common candidate budgets, not native oversampling settings or measured search costs.
- Treat binary-score ties explicitly. For a cutoff with `a` strictly better documents and `b` documents tied at the boundary, each boundary-tied target has inclusion probability `(C-a)/b`. Sum target inclusion probabilities and divide by 10. This is expected recall under uniform random boundary tie-breaking, not recall of an arbitrary ID-order sort. Include tie size or a best/worst bound so the source of uncertainty is visible.
- Preserve the exact-original target definition used in E16/E17; handle any exact-original cutoff ties consistently. Do not use labels or outcome-dependent tie-breaking to favor either arm. Graded scores may have their own ties; apply the same rule where applicable.
- Compute paired graded-minus-sign inclusion changes within query, then differences of those changes between tiers. Report all three budgets and both datasets. Reusing the known E16/E17 samples makes this exploratory mechanism follow-up, not independent confirmation.
- An improvement would show that query magnitude carries information lost by sign encoding for recovering original neighbors. It would not show that HNSW can find those candidates cheaply, that graded queries preserve nDCG, or that switching a live Qdrant collection to scalar8 yields the same result. A native follow-up would still need graph/configuration comparability and measured latency.

If graded scoring helps every tier similarly, report a general asymmetric-scoring effect rather than a Zero-specific mechanism. If it helps Zero markedly more on FiQA but not the 1M sample, preserve that domain/size dependence. If it does not help, do not add arbitrary calibration until an alternative explanation is specified. This diagnostic is justified despite the attenuated E17 primary result because it identifies a distinct factor that the native option ablation could not isolate; no broader sweep or training is required first.

## Access

Read `m15src/e17_quantization_replication.py`, the shared E16 selection/encoding/main setup, the last E16/E17 method sections in `m15/MEASUREMENTS.md`, `m15/REVIEWS/2026-10-01-quantization-semantics.md`, and E17 aggregate status, measurements, contrasts, configuration equality, and provenance. No neighbor arrays were examined, and no raw/protected material or `work/` content was opened. Only this review file was written; no code or result mutation, commits, or runs.
