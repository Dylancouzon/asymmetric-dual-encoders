# E18 query-precision review, 2026-10-01

**Outcome:** no essential implementation or tie-handling defect found. The primary paired contrasts support a repeated, scoped **absolute candidate-coverage cost of discarding query magnitudes**, larger for Zero than for Nano or Stella in these experiments. This is a controlled scoring finding distinct from E17's uncertain primary native-search interaction.

## Controls and mathematical checks

The source computes document codes once per dataset and reuses those exact sign codes for every tier and both query conditions. Each tier retains its own original exact-top-10 target from E16/E17. Sign queries and normalized original queries differ only in retained coordinate magnitudes; normalization supplies a query-wide positive scale and does not itself change rankings. There is no graph, model fitting, correction learned on outcomes, or encoder-specific document quantizer.

The source requires freshly encoded normalized query-vector bytes to match E16/E17's recorded hashes. E18 completed with those checks. The inspected E18 script hash and both receipt-bound E16/E17 file hashes agree with the committed receipt. The declared primary budget is 40, with 10 and 100 also reported.

`expected_inclusion` selects the correct Cth-largest cutoff. A target above the cutoff enters with probability one; a target at the cutoff enters with probability `(C - number_strictly_above) / number_tied`. Linearity of expectation makes their mean the expected original-target coverage under uniform boundary tie selection. I extracted this pure function without running the experiment and checked it against exhaustive enumeration on synthetic all-tie, cutoff-tie, and untied arrays at three budgets. All checks agreed. Sign-score dot products are integer-valued here, avoiding arbitrary floating tolerances for the important binary ties. Original-query scores use their computed float values and receive the same exact-equality tie treatment.

Expected candidate inclusion also corresponds to expected recovery after original-score reranking when the original target's exact-cutoff ordering is preserved. Any exact-original score ties must use the same target convention; the experiment appropriately fixes the recorded target set rather than selecting favorable targets anew. It does not estimate nDCG.

## Primary evidence: C = 40

| Workload | Tier | Sign-query coverage | Original-query coverage | Gain, percentage points |
|---|---|---:|---:|---:|
| FiQA | Zero | 0.8217 | 0.9297 | +10.80 |
| FiQA | Nano | 0.9137 | 0.9771 | +6.33 |
| FiQA | Stella | 0.9638 | 0.9922 | +2.84 |
| 1M diagnostic | Zero | 0.9010 | 0.9677 | +6.67 |
| 1M diagnostic | Nano | 0.9484 | 0.9833 | +3.49 |
| 1M diagnostic | Stella | 0.9685 | 0.9932 | +2.47 |

The paired differences of gains are:

| Workload | Contrast | Extra Zero gain, percentage points | Query-bootstrap 95% interval |
|---|---|---:|---:|
| FiQA | Zero minus Nano | +4.46 | [+2.53, +6.43] |
| FiQA | Zero minus Stella | +7.96 | [+6.13, +9.78] |
| 1M diagnostic | Zero minus Nano | +3.17 | [+1.58, +4.75] |
| 1M diagnostic | Zero minus Stella | +4.20 | [+2.65, +5.80] |

These are paired on the same query identities and use fixed targets/budgets. The primary effects consistently favor a larger absolute Zero gain. They are exploratory, conditional on the selected known-test queries, these encoders, and these document spaces; no architectural population claim follows. Keep all budget rows available: at C=10 on the 1M diagnostic, the Zero-minus-Nano interval spans zero, so “at every tested budget against both transformers” would be false.

## The important remaining interpretation boundary

Say **absolute percentage-point gain**, not that Zero is universally or proportionally more sensitive to query quantization. The baselines have different headroom. A read-only exploratory calculation from the same C40 aggregates gives fractions of sign-mode misses recovered of 60.6%/73.4%/78.4% for Zero/Nano/Stella on FiQA and 67.4%/67.7%/78.5% on the 1M diagnostic. The absolute effect is real and useful, but the normalized story differs. These additional ratios are review calculations, not a proposed new confirmatory result.

Likewise, the mean cosine between each original query and its sign direction is about 0.795–0.798 for every tier. E18 therefore does not show that Zero undergoes a larger average angular perturbation. It measures a larger absolute consequence for recovering its own original neighbors. Explaining that consequence through rank gaps, query-coordinate distribution, or geometry remains further work.

A safe paper claim is:

> With document sign codes and original-neighbor targets fixed, preserving query-coordinate magnitudes improved exact candidate coverage more for Zero in absolute terms at the primary 40-candidate budget on both workloads. This isolates a query-scoring contribution to lost candidate fidelity, independently of graph traversal.

Do not call the original-query scorer a guaranteed upper bound, native Qdrant scalar8, a measured serving improvement, or a repair of E17's full native pipeline. It is a defined asymmetric full-query/sign-document score. The result does not establish equal-quality speedups, relevance improvements, or causation of the entire original-score ANN penalty. Global C is not native per-segment oversampling.

This can be a substantive new quantization finding in the paper if its controlled estimand is made clear and E17's weaker primary replication remains visible. It supports investigating query precision as a design choice when sharing compressed document vectors; it does not require relabeling E17 as successful confirmation. Practical claims need a separate native query-precision and latency experiment with graph identity controlled.

## Access and verification

Read `m15src/e18_query_precision.py`, the last E16–E18 sections of `m15/MEASUREMENTS.md`, and E18 metadata, aggregate rows, precision deltas, tier interactions, magnitude summaries, and provenance. E16/E17 receipt files were hashed without inspecting their neighbor arrays. The only execution was read-only aggregate arithmetic and synthetic enumeration of the pure tie function; no full benchmark, raw query/qrel read, protected access, or `work/` content access occurred. Only this review file was written.
