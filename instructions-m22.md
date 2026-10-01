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
   The install line on all three cards, the README and the plain-English page switches to plain
   `pip install fastembed` once a PyPI release contains qdrant/fastembed#751 (owner decision,
   2026-09-30, `m22/UPSTREAM_FASTEMBED.md`); the "requires the Constella preview branch" banner
   sentence goes with it.
3. **Zero card training description (logged 2026-09-30, found during M15).** The Zero card says the
   table "was trained by L2 regression against Stella query embeddings"
   (`m11/release/MODEL_CARD.md:184`, and the live `Qdrant/constella-zero` card). The recipe is
   different: phase B uses cosine to the teacher's query vector plus a top-32 KL ranking loss plus
   InfoNCE over a 2M-vector bank (temperature 0.02, false-negative margin 0.02), with learned
   per-token weights seeded from IDF and folded into the rows at export; phase A continues for
   2,500 steps (`m7/RECIPE.md:46-91`). Correct the card from the recipe. The card's 338,076 usable
   pairs against the recipe's 340,850 pairs also needs a one-line reconciliation from
   `m7/LEDGER.md`. `research/constella-in-plain-english.md` ("trained by regression") gets the same
   fix, and its line 141 ("seven candidates with complete learnability rows") should say nine
   complete pairs, per `results/m7_learnability_report.json`.
4. **Storage-retirement request.** Only after Hub download verification and after M20's archive is
   verified at both targets may M22 recommend retiring the three Runpod volumes. Deletion requires
   an explicit owner ruling; it is not inferable from any handoff.

## Constraints

Standing `CLAUDE.md` rules. Comparator per-dataset absolutes remain unpublished on cards unless the
owner rules otherwise (2026-09-15 decision); the whitepaper (M15) carries the full comparator table.
Descriptive rows are labelled as such; unresolved superiority is never written as equivalence.

**Owner note (2026-10-01).** Showcase hot-swappability in the official model cards: all three query
paths (Stella, Nano, Zero) verified against the same populated Qdrant collections with no rebuild
between encoder batches, so a request can select any path; cite the M15 E2 receipts. Describe it as a
capability (compatible encoder selection), not timed model loading or failover.
