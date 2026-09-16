# Project status: 2026-09-16

**Constella Zero, the Stella document tower and Constella Nano are public; Nano is a research
preview.** M13 and M14 are closed. M20 (opened 2026-09-16, R22) owns the still-unspent reserved
evaluation and the BEIR-13 descriptive run; M22 owns the official release and M23 the upstream
FastEmbed PRs. Current plan: `ROADMAP.md`; canonical numbers: `m21/BENCHMARKS.md`.

## Research and product

| Area | Status |
|---|---|
| Shared index | The frozen 1024-d Stella document tower shipped in M11. Zero and Nano query the same index |
| Zero | Frozen and published in M11. Dense all-six nDCG@10 0.4339; M7 established it below LightRetriever's 0.4583 bar by −0.0243 |
| Fusion | Deployed recommendation is Qdrant DBSF@100: 0.4887 all-six / 0.4912 clean-4. The separate M7 convex0 result is 0.4911 / 0.4866; no superiority or equivalence comparison was established |
| First Nano (M9) | Frozen and not released. M13 completed its six-set close-out; the observed dataset dependence does not isolate coverage from capacity |
| Nano (M13/M14) | Built from exactly 199,999,721 training examples, evaluated and published at `DylanCouzon/constella-nano`. Weights are frozen at revision `6bb167dc6f60d3992602235b8e8aaa374a309168`; the card has been updated since and the model page carries the current one |
| Nano result | Established over bge-small by +0.017648 clean-4 and +0.027449 all-six; established over LEAF by +0.016181 all-six. Clean-4 vs LEAF was −0.001063, superiority unestablished with no equivalence claim |
| Zero v1.1 (M17) | Closed 2026-09-12, negative result: no screen arm met eligibility; released Zero v1 remains shipped |
| Project memory (M18) | Closed 2026-09-13 as `SYSTEM_READY` + `ENCODER_NO_IMPROVEMENT`; the system is usable with released Zero v1 |
| M19 | Closed 2026-09-13 as `SYSTEM_READY` + `ENCODER_INCONCLUSIVE`; development results were label-sensitive, confirmation remained sealed and released Zero v1 stays selected |
| M20 | Active: reserved four and BEIR-13 descriptive evaluation over an eight-system roster (Nano, Zero, BGE-small, LEAF, Stella-query, BM25, two DBSF@100 fusions), then archive. Release is M22, upstream PRs M23 |
| M21 | Closed 2026-09-15: research-preview documentation, canonical benchmark tables, model-card publication and FastEmbed polish completed |

## Verification and limits

- M13 closed after the Nano build, frozen six-set evaluation, M9 close-out and common serving-cost
  measurement. Its status records **267 M13 tests passed**.
- The common serving protocol used three fresh processes per query encoder, batch one and four CPU
  threads. Warm 20-word p50 was 0.1119 ms for Zero, 6.8400 ms for bge-small and 7.2511 ms for Nano.
- M14's publication gates verified the frozen and staged artifacts, native FastEmbed registration,
  boundary-length parity, a fresh revision download and anonymous access. The published revision is
  immutable and remains the research-preview artifact.
- Absolute per-dataset bge-small and LEAF rows are not published. The committed evidence supports
  Nano's absolute row and the registered comparator deltas without re-deriving aggregates.
- The reserved four and BEIR-13 remain **UNSPENT**. They have no result and belong to M20; this
  preview does not imply one. "BEIR-18" in older files means this BEIR-13 (R22).
- M20 must run those evaluations under their registered access rules; M22 then completes the
  official release and M23 opens the upstream FastEmbed PRs. M19 closed without opening
  confirmation; its development evidence does not establish improvement, non-improvement or
  equivalence.

Historical paths, frozen comparators and registrations retain their original names. In particular,
**never overwrite `results/perquery.json`**. Research lessons remain in the milestone `FINDINGS.md`
files; reusable checks and component boundaries are mapped in `HARNESS.md`.
