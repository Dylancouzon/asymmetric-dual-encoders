# Research follow-up: query compute and search precision

2026-10-01. The owner welcomes justified additional research and compute, with no full-model
retraining and no publication rush. The owner also proposes quantization, provided comparisons
remain fair. This file records the scientific question, experiments, findings, and next decisions.
The v12 paper is a credible applied study; stronger discoveries require evidence, not stronger wording.

## The question worth spending on

**Does replacing only the query encoder change the search precision and candidate budget needed
by a fixed document index?** If so, can a cheap query-side intervention or request setting repair
the penalty without changing the original retrieval target or rebuilding documents?

OOD graph-search difficulty and query-aware quantized scoring have substantial prior art. Our
specific condition is a static, compact-transformer, and original-teacher query path over the same
frozen text document space. Novelty would come from a reproducible encoder/precision interaction
and an explanation or useful remedy under that fixed-index constraint, not another quantizer sweep.
Verified sources and competing explanations are in `REVIEWS/2026-10-01-followup-literature.md` and
`REVIEWS/2026-10-01-followup-design.md`.

## E16: a same-graph pilot is complete

Source: `m15src/e16_quantization_pilot.py`; immutable receipt:
`results/m15_e16_quantization_pilot.json`; method: the E16 section of `MEASUREMENTS.md`.

- 192 FiQA queries selected deterministically by query-ID hash, not by performance; all 57,638
  cached documents. No document encoding, model training, or cloud rental.
- One disk-backed binary collection and fixed graph for every query tier and request option.
  Original exact search matches NumPy top10 perfectly for all three encoders. Collection
  configuration and point/indexed counts match before and after interleaved queries.
- Query-resampling intervals condition on the one graph and fixed models. These are exploratory
  pilot results, not general architecture laws or confirmatory tests.

At the primary `ef=64`, exact-neighbor recovery@10 is:

| Query path | Original-vector scoring | Binary + rescore, oversampling 1 | Binary + rescore, oversampling 4 |
|---|---:|---:|---:|
| Zero | 0.9802 | 0.8635 | 0.9521 |
| Nano | 0.9964 | 0.9453 | 0.9927 |
| Stella | 0.9995 | 0.9766 | 0.9990 |

The paired binary-minus-original recovery penalties are 11.67 percentage points for Zero,
5.10 for Nano, and 2.29 for Stella. Zero's extra penalty is 6.56 points versus Nano
(95% query-bootstrap interval 4.69–8.44) and 9.38 versus Stella (7.40–11.41).
The interaction persists at `ef=16` and `ef=256`. Increasing `ef` alone leaves a substantial binary
penalty; widening the rescoring candidate budget helps markedly. Neighbor recovery and relevance
quality are different outcomes; the receipt records nDCG beside recovery and does not equate them.

These native API treatments change traversal/scoring behavior and potentially candidate handling.
They do not isolate exhaustive quantization error or a pure rerank over an identical candidate pool.
In particular, native rescore/no-rescore return different neighbor sets even with oversampling 1.
The original stored vectors are fp16 values decoded to fp32, not fresh full-precision encodes.
The independent pilot review finds no essential control defect and recommends a second-scale test.

## Next sequence

1. **E17 replication:** repeat the paired same-graph request-option test on the cached 1M MS MARCO
   diagnostic, with the same deterministic 192-query sample rule and fixed condition grid. The
   subset preserves positives and removes competitors; it remains validation-only and is not full
   MS MARCO. Local execution needs no rental. This tests domain/scale sensitivity, not universality.
2. **If the interaction repeats:** verify the native binary query precision and candidate-handling
   semantics; distinguish quantized ranking/candidate omission from graph omission before claiming
   a mechanism. Use fixed pools/original reranking or validated exhaustive compressed scoring.
3. **One bounded remedy:** only after the diagnostic, test a small navigation correction selected
   on development queries. Keep the original Zero vector as the final scorer so improvement cannot
   come merely from changing the retrieval target. Include the transform and candidate handling
   costs, and compare against retuned `ef` and oversampling. No full encoder retraining.
4. **Alternatives:** a cost-aware router is useful if it beats simple/static policies after extra
   searches; a small contrasting-teacher fit can test the probe-to-trained-recipe bridge. Both are
   less directly connected to the strongest current shared-index result than the precision study.

Quantizing Nano's inference weights is a separate deployment experiment. It could improve the
available query budget without training, but weight quantization alone is established and is less
likely to add new IR knowledge. It should preserve the frozen document space and evaluate quality
and measured encoding cost together rather than compare unmatched precision runs.

## What would justify a stronger paper

A repeated controlled interaction plus a defensible decomposition would add new information about
serving distilled queries, even without a new algorithm. A cheap fixed-index remedy that improves
search cost at matched original-target recovery and retrieval quality would be stronger still.
A null remedy or a result explained by known query hardness is useful if reported accurately;
neither should be marketed as a breakthrough. More checkpoints or an undirected quantizer search
would add bulk before adding an argument.
