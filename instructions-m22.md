# M22 — official Nano release

**Created 2026-09-16 under owner ruling R22 (Dylan).** Takes the release half of the original M20
mandate so that M20 stays evaluation-only. Runs after `m20/STATUS.md` reports the reserved four and
BEIR-15 complete. No new measurement, no training, no protected access.

## Deliverables

1. **Card revision** of `Qdrant/constella-nano` (moved from `DylanCouzon/` under R25, `m22/HUB_TRANSFER.md`) as a later revision of the existing
   repository: the frozen weights (`6bb167dc`) do not change. The card carries the reserved-four
   and BEIR-15 descriptive tables with their contact labels and FEVER caveat, the M13 registered
   results unchanged, and drops the "research preview" framing. Numbers come from
   `m21/BENCHMARKS.md`, extended from `results/m13_reserved_run.json` and
   `results/m20_beir15_run.json`; nothing is recomputed or re-partitioned. Zero and document-tower
   cards get the same tables where they apply.
2. **Release discipline** inherited from `m14/HANDOFF.md` steps 4–8 and 10: fresh staging from the
   verified bytes, documented packaging transforms only, length-stratified parity including the
   511/512/513 boundary, `m14/verify_card.py` offline against staged bytes, private-first upload of
   any changed file, LFS-oid and downloaded-byte verification, then the public revision.
3. **Storage-retirement request.** Only after Hub download verification and after M20's archive is
   verified at both targets may M22 recommend retiring the three Runpod volumes. Deletion requires
   an explicit owner ruling; it is not inferable from any handoff.

## Constraints

Standing `CLAUDE.md` rules. Comparator per-dataset absolutes remain unpublished on cards unless the
owner rules otherwise (2026-09-15 decision); the whitepaper (M15) carries the full comparator table.
Descriptive rows are labelled as such; unresolved superiority is never written as equivalence.
