# Project status — 2026-09-15

**Constella Zero, the Stella document tower and Constella Nano are public; Nano is a research
preview.** M13 and M14 are closed. M20 owns the still-unspent reserved evaluation, official release
work and upstream FastEmbed PR. Current plan: `ROADMAP.md`; canonical numbers: `m21/BENCHMARKS.md`.

## Research and product

| Area | Status |
|---|---|
| Shared index | The frozen 1024-d Stella document tower shipped in M11. Zero and Nano query the same index |
| Zero | Frozen and published in M11. Dense all-six nDCG@10 0.4339; M7 established it below LightRetriever's 0.4583 bar by −0.0243 |
| Fusion | Deployed recommendation is Qdrant DBSF@100: 0.4887 all-six / 0.4912 clean-4. The separate M7 convex0 result is 0.4911 / 0.4866; no superiority or equivalence comparison was established |
| First Nano (M9) | Frozen and not released. M13 completed its six-set close-out; the observed dataset dependence does not isolate coverage from capacity |
| Nano (M13/M14) | Built from exactly 199,999,721 training examples, evaluated and published at `DylanCouzon/constella-nano` revision `6bb167dc6f60d3992602235b8e8aaa374a309168` |
| Nano result | Established over bge-small by +0.017648 clean-4 and +0.027449 all-six; established over LEAF by +0.016181 all-six. Clean-4 vs LEAF was −0.001063, superiority unestablished with no equivalence claim |
| Zero v1.1 (M17) | Closed 2026-09-12, negative result: no screen arm met eligibility; released Zero v1 remains shipped |
| Project memory (M18) | Closed 2026-09-13 as `SYSTEM_READY` + `ENCODER_NO_IMPROVEMENT`; the system is usable with released Zero v1 |
| M19 | Deterministic short-query feasibility plan reviewed but not executed; no result |
| M20 | Next execution milestone: reserved four, BEIR-18, official Nano release and upstream FastEmbed PR |
| M21 | Research-preview documentation, canonical-table and FastEmbed polish in progress |

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
- The reserved four and BEIR-18 remain **UNSPENT**. They have no result and belong to M20; this
  preview does not imply one.
- M20 must run those pending evaluations under their registered access rules, complete official
  release work and open the clean upstream FastEmbed PR. M19 likewise requires a new execution
  session before it can produce evidence.

Historical paths, frozen comparators and registrations retain their original names. In particular,
**never overwrite `results/perquery.json`**. Research lessons remain in the milestone `FINDINGS.md`
files; reusable checks and component boundaries are mapped in `HARNESS.md`.
