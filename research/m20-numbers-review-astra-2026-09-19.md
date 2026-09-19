# M20 reserved-four numbers and stage-C readiness — Astra, 2026-09-19

Reviewer: Codex `gpt-6-astra`, read-only. Brief: `work/m20/logs/astra_numbers_brief.md`.
Transcript: `work/m20/logs/astra_numbers_review.log`. Reviewed at `736ab05`.

**Numbers: SOUND WITH CAVEATS.** I found no essential numerical defect in the permitted evidence.

**Starting stage C: NO-GO as currently documented.** One essential execution gap remains.

1. **P1 — The stage-C entry point does not implement the registered bounded stage.**  
   [beir15.py:372](/home/dylan/asymetric-dual-encoders/m20src/beir15.py:372) proceeds directly to query encoding and scoring. Its document-vector path, [beir15.py:122](/home/dylan/asymetric-dual-encoders/m20src/beir15.py:122), only loads existing shards. Missing shards cause failure; this entry point does not schedule their encoding.

   It also has no deadline, timeout, or projection-gate invocation. Consequently, the documented command neither completes the missing corpus work nor enforces the **single 120-hour download/encode/score cap** required by [REGISTRATION.md:237](/home/dylan/asymetric-dual-encoders/m20/REGISTRATION.md:237). Setting the allocator in this process does not configure a separately launched pre-encoder.

   **What breaks:** stage-C completion and registered budget protection. Before launch, supply a concrete invocation covering shard reuse, missing encodes, and scoring under one retained deadline, with the pre-encode projection gate armed. The requested additional launcher/pre-encoder reads were not approved, so I cannot certify an external orchestration path.

The numerical checks passed:

- **All 32 means reproduce exactly** from the saved per-query scores. Every system has 6,666 / 400 / 699 / 1,570 queries, identical query sets within each dataset, and finite scores within `[0,1]`.
- **Both contrasts and every reported sensitivity interval reproduce.** I independently regenerated the paired, within-dataset bootstrap: `B=10000`, seed `902`, sorted datasets and query IDs, shared draws across contrasts, and `inverted_cdf` quantiles. The draw-plan hash matches exactly; maximum point-estimate difference was below `7×10⁻¹⁸`.
- **NDO-3 uses 0.50/0.25/0.25 and excludes FEVER.** R1’s sign flip is explained completely by weighting: its DBpedia/Android/English deltas are `+0.01872 / +0.00134 / −0.02600`. Query pooling reduces DBpedia’s weight to 15% and increases English’s to 59%.
- **Fusion wiring and receipts agree.** Every derived row names the correct dense producer plus BM25, with matching producer run hashes and payload identities. The code loads those persisted runs and truncates both inputs to 100 before DBSF. All dense rows’ document-manifest hashes match the pre-encode receipt, including the shared Stella index.

The FEVER reversals do **not themselves establish a defect**. The inspected tower mapping and prompt routing match the registration. Stella-query is not a mathematical upper bound on BGE; the fusion reversal also follows the direction of the underlying dense results. These observations are compatible with dataset-specific performance and disclosed contamination, but they do not identify contamination as the cause.

Reporting in the result is appropriately descriptive: FEVER is labelled double-contaminated sensitivity, excluded from NDO-3 and its pooled sensitivities, with zero alpha and no gate or release claim. The transaction code preserves the six-set decision while adding reserved completion bookkeeping.

Stage C correctly copies reserved rows and computes CQADupStack only when all twelve forum means exist, using their unweighted mean. It **does not import M13’s six-set score outputs**: absent existing M20 score files, it scores those datasets again, as the BEIR registry specifies.

**Verification limits:** the actual persisted top-100 files, vector contents, underlying retrieval/fusion implementations, and before/after six-set result were outside the original allowlist. Their requested reads remained unanswered. Thus fusion provenance is verified through code and matching receipts, not independently rescored against labels. The intervals quantify query sampling, not training-seed variation.

**Access log — files read wholly or partially:**

```text
CLAUDE.md
m13/RULINGS.md
m20/STATUS.md
m20/REGISTRATION.md
m20/beir15_registry.json
m10/final_run_registry.json
m13src/reserved_support.py
m13src/score13.py
m13src/reserved_transaction.py
m20src/roster.py
m20src/beir15.py
results/m13_reserved_run.json
results/m20_local_run.json
results/m13_reserved_manifest.json
results/m13_reserved_preencode.json
results/m20_allocator_probe.json
