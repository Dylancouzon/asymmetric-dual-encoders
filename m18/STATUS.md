# M18 status

**Complete 2026-09-13 — `SYSTEM_READY` + `ENCODER_NO_IMPROVEMENT`.**

The internal Qdrant project-memory system is ready over a pinned 79,269-document snapshot. Its
primary query path is released Zero v1 dense retrieval plus BM25, fused with DBSF@100. The exact
query command and immutable identities are in `results/m18_system_manifest.json`.

The bounded query-table training completed. T0's best development dense gain was +0.007591
nDCG@10, but its best fused gain was only +0.002953 against the required +0.010. T1 was inert;
T2 changed representations but never changed a registered ranking metric relative to T0. No form
met both eligibility margins, so no second seed was applicable and released Zero v1 remains the
system encoder. Astra independently approved that decision.

The sole confirmation read is complete and receipted. On 40 confirmation queries, stratum-macro
nDCG@10 was 0.109163 for BM25, 0.153176 for v1 dense and 0.151445 for v1+DBSF; fused Recall@10 was
0.300. Error/troubleshooting scored zero for every route on this small slice, a material limitation
recorded in `m18/FINDINGS.md`. A report-only 21-term probe and three live hybrid queries include the
CTO-requested `k8s` coverage; they do not alter the encoder decision.

Pointers: `instructions-m18.md`, `m18/registry.json`, `m18/LEDGER.md`, `m18/CODEMAP.md`,
`m18/FINDINGS.md`.
