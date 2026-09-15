# M10 ledger — current index

## Scope amendment — 2026-09-10

Dylan requested a cleanup and clarified that M10 was preparation before cloud GPUs.
M10 closes that scope. Pending cloud E, build and evaluation move to M13; release/paper move to M14/M15. `ROADMAP.md` maps every future milestone.
No result, screen registry, completed decision, access receipt or frozen comparator was changed.

## Authoritative records

| Record | Home |
|---|---|
| Screen arms, doses, contrasts, rules and amendments | `m10/screen_registry.json` |
| Corpus identities and disclosures | `results/m10_data_manifest.json` |
| Recipe handoff | `m10/M102_LOCK.md` |
| Contrast outputs and selection | `results/m10_contrast_*.json`, `results/m10_screen_verdicts.json` |
| Descriptive observations | `results/m10_descriptive_*.json` |
| Open execution findings | `m13/STATUS.md`, `m13/EXECUTION.md` |
| Measured lessons and rejected interpretations | `m10/FINDINGS.md`, `m10/EXPLORED.md` |

The complete 1,789-line ledger through commit `46ea18d`, including every owner ruling,
withdrawn claim, delegated decision, read count and review disposition, is preserved verbatim at
`research/archive/m10-cleanup-2026-09-10/m10/LEDGER.md`. Its §0/§1/§2/§3/§4/§5 anchors and
line references refer to that copy. Earlier git commits remain the provenance authority.

The 2026-09-10 recipe/judging split remains recorded. It did not remove the pre-build LoTTE veto
or permit M9 close-out while an M10-derived recipe decision remains open. M13 owns enforcement.

- 2026-09-14T23:48:04.125687+00:00 — **FINAL-RUN-BEGIN** freeze `3e49e0bfaa63` pid 330139 host `Str0keWtf`
- 2026-09-14T23:48:07.992314+00:00 — **FINAL-RUN access** (six) `scifact`: 5,183 docs / 300 queries, corpus hashes verified against a fresh HF download, labels read from `results/frozen_eval/scifact.json`.
- 2026-09-14T23:57:22.097572+00:00 — **FINAL-RUN access** (six) `nfcorpus`: 3,633 docs / 323 queries, corpus hashes verified against a fresh HF download, labels read from `results/frozen_eval/nfcorpus.json`.
- 2026-09-15T00:03:48.386068+00:00 — **FINAL-RUN access** (six) `fiqa`: 57,638 docs / 648 queries, corpus hashes verified against a fresh HF download, labels read from `results/frozen_eval/fiqa.json`.
- 2026-09-15T01:43:20.208051+00:00 — **FINAL-RUN access** (six) `arguana`: 8,674 docs / 1,406 queries, corpus hashes verified against a fresh HF download, labels read from `results/frozen_eval/arguana.json`.
- 2026-09-15T02:02:06.080102+00:00 — **FINAL-RUN access** (six) `scidocs`: 25,657 docs / 1,000 queries, corpus hashes verified against a fresh HF download, labels read from `results/frozen_eval/scidocs.json`.
- 2026-09-15T02:46:34.850774+00:00 — **FINAL-RUN access** (six) `trec-covid`: 171,332 docs / 50 queries, corpus hashes verified against a fresh HF download, labels read from `results/frozen_eval/trec-covid.json`.
- 2026-09-15T07:39:09.603594+00:00 — **FINAL-RUN-END** (full, INCOMPLETE_RESERVED) result sha256 `f5b5ad8a63060fbe` outcome `REJECTED:C1b,C1a,C2a`
