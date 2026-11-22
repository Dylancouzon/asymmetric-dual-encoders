# E16 pilot sanity review, 2026-10-01

**Outcome:** no essential control or implementation defect found within this bounded native request-option experiment. The result justifies a targeted replication/decomposition, not a breakthrough claim or a causal explanation of query geometry.

## Controls checked

The sample is deterministic before outcomes; all three released encoder paths share the same normalized cached document vectors, graph, and binary codes. One dedicated server/collection is built once, and requests are interleaved in a seeded order per query. The original-score exact parity gate passes at 1.0 for every tier. All 57,638 points are indexed; before/after collection configuration and counts agree. The final receipt's E16 script hash matches the inspected source and records a clean source commit. The primary `ef=64` and secondary 16/256 match the written method.

The paired interaction compares each encoder with its own original exact-neighbor target on the same query IDs. It therefore asks whether changing the native search path incurs a different recovery penalty, without pretending that the three encoders have equal exact quality. Query bootstraps exclude graph-build and training variation, as disclosed.

## Finding and strength

At the primary `ef=64`:

| Tier | Original-score traversal recovery | Binary + rescore1 recovery | Change |
|---|---:|---:|---:|
| Zero | 0.9802 | 0.8635 | -0.1167 |
| Nano | 0.9964 | 0.9453 | -0.0510 |
| Stella | 0.9995 | 0.9766 | -0.0229 |

The Zero-minus-Nano interaction is -0.065625, query-bootstrap 95% interval [-0.084375, -0.046875]. Zero-minus-Stella is -0.09375 [-0.114063, -0.073958]. The ordering persists at `ef=16` and 256. This is substantial for a 192-query pilot and is not solely Zero's worse original-score graph recovery.

Increasing `ef` from 64 to 256 barely moves unrescored binary recovery: Zero 0.4984→0.4990, Nano 0.6073→0.6083, Stella 0.6708→0.6698. At `ef=64`, oversampling4 with original rescoring improves Zero to 0.9521, Nano to 0.9927, and Stella to 0.9990. This pattern motivates examining compressed-score/candidate limits. It does not establish an exhaustive compressed-retrieval ceiling or separate query quantization from document quantization.

## Inference boundaries and next decision

The correct statement is: **on this fixed FiQA graph and pilot sample, enabling the native binary path with rescoring1 adds a larger exact-neighbor recovery penalty for Zero than for Nano or Stella; oversampling mitigates it.**

Do not call the contrasts pure document-quantization distortion, pure graph-navigation loss, or pure reranking improvements over an identical candidate pool. `ignore`, `rescore`, and `oversampling` select native execution paths; segment-level candidate handling and traversal may differ. Collection/configuration counts are useful no-mutation checks, not a cryptographic verification of graph bytes. No graph-rebuild variability was measured. The original exact gate deliberately says nothing about exhaustive compressed scoring.

The primary conclusion concerns neighbor recovery. nDCG changes need not match it, and diagnostic timing does not establish an optimal equal-quality serving policy. In particular, a quantized path can be faster but lower-recall, while rescoring/oversampling can spend the speed saving; both quality and cost must enter any proposed remedy.

Proceed with one second-scale fixed-graph replication on a deterministic sample of the already permitted 1M diagnostic, and add the exact compressed-score/candidate decomposition only if the actual native representation can be accessed and verified economically. Keep the three encoders, common graph, and stated primary contrast. The high-ef plateau makes compressed candidate recovery a more immediate next question than a new learned router or a general “off-manifold” correction. No full retraining or large hyperparameter sweep is warranted by this pilot alone.

## Access

Read `m15src/e16_quantization_pilot.py`, the final E16 method section of `m15/MEASUREMENTS.md`, and only status, provenance, configuration equality, aggregate measurement fields, and paired contrasts from `results/m15_e16_quantization_pilot.json`. No neighbor arrays were printed or examined, no raw queries/qrels or protected content was read, and the running source and result were untouched. This review file is the only edit.
