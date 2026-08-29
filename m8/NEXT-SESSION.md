# M8 next-session worklist (rewritten 2026-08-29 after the overnight session)

**Read first: `m8/STATUS.md` (its wake-up note has five decisions for Dylan), then `m8/LEDGER.md`
(the protocol authority), then `m8/registry.json` (the executable half). This file is the
remaining worklist only — everything the overnight session finished has moved into LEDGER,
RESULTS, EXPLORED and CODEMAP, and is not repeated here.**

The previous version of this file said "delete when consumed". It is **not** consumed: items 8,
9 and 11 are partly or wholly outstanding, and five decisions are with Dylan. Deleting it would
have lost the remainder, so it is rewritten instead.

## Blocked on Dylan (STATUS's wake-up note has the full case for each)

1. **E14 — doc-side co-adaptation.** The largest unopened lever. A literature sweep confirms
   nobody has isolated it (LightRetriever never freezes its document tower). Yes/no is his.
2. **E10 — the shadow.** All ten LoTTE slices reject. Three options are costed in STATUS: keep the
   slice-level bar and lose the shadow; authorise a per-question remedy; or substitute the eight
   unused CQADupStack subforums. **The pipeline's STOP gate is missing until this is answered.**
3. **harrier's ruling** — now with three blockers, not one, so probably not worth his time until
   the other challengers are screened.
4. **HUPD's licence interpretation** and the **PatentsView API key** (M9 reserve).
5. **The E12 LR-dense pre-encode bill** — pre-agree the published-numbers fallback.

## Executable without Dylan, in order

1. **Fit list — DONE** (`m8src/protected_filter.py fitlist`): 337,981 kept of 338,076, all 95
   removals from M9-reserve. Re-run it if the filter's coverage changes (e.g. if Dylan authorises
   a LoTTE remedy, the surviving shadow queries must enter it).
2. **T1 — DONE for three of five candidates; NO SWAP** (LEDGER §21). The incumbent was re-probed
   in the identical frame and stands at 0.3438; granite-r2 (−0.052) and gte-modernbert (−0.109)
   both lose CI-resolved. What remains, and only if there is a reason to want it:
   - **stella-1.5B** — needs its degenerate-query fallback row REGISTERED first: its
     `config.json` and `tokenizer_config.json` disagree about BOS (151643 against null), so there
     is no defensible default. Needs `trust_remote_code` (same-repo, so `revision` pins it).
     int8-only at 155 MB.
   - **harrier** — needs Dylan's ruling AND new code: last-token pooling, which
     `m7src/teacher.py` raises on. It publishes no retrieval-only number either, so a screen
     result would have nothing to check against. Lowest priority of anything on this list.
   `m8src/challengers.py` registers Specs at runtime; add a row and `validate_encoder.py` must
   pass before any encode.
3. **The remaining gap-list obligations** (LEDGER §4.4). The pre-encode allowlist entry is DONE.
   What is left:
   - **the crossed B×A seed design — NEW, and the B-leg floor's own review asked for it.** The
     three chains vary ONE seed across both legs, so B-seed and A-seed effects are aliased and may
     partially cancel — which would make the floor an *under*-estimate, the anti-conservative
     direction. Crossing the three B checkpoints (`p35b-2m`, `m8nfb-seed1-b`, `m8nfb-seed2-b`)
     against three A seeds separates B variance, A variance given B, and their interaction. The
     three chains on disk are the diagonal, so **six more A legs (~30 min) complete a 3×3**. This
     is the cheapest remaining thing that would upgrade a convention into a bound.
   - **the B-leg noise floor — DONE** (`results/m8_noise_floor_bleg.json`, LEDGER §15). It is not
     larger than the A-leg floor, so R-PHASE and the pool/init levers read the same 0.0040
     planning minimum; the one exception is `int8/mean` worst-group and OOD macro, which take
     0.004369. All three floors now exist.
   - **`m8src/freeze.py` and `m8src/final_run.py`, and only then their test suites.** The tests
     are in the gap list but they cannot precede the modules, and the modules are a real port —
     M7's `freeze.py` alone is 34,659 bytes of accumulated refusals. This is the largest piece of
     engineering left in the milestone and it is weeks from being needed.
4. **Wave-1 probes.** S0, T1, B2, B7 and both noise floors have run; B17 ran and was disowned
   (§20). **B3 — DONE, verdict UNINFORMATIVE** (Phase A is not meaningfully pair-starved; reaching the bar would need ~17.6x the pool). It was rebuilt twice — the ICT lever was retired before any arm
   existed (registry `B3-ICT`), replaced by real-pair pool scaling, then pinned to computable
   scalars after a second review; the bar is code (`m8src/b3_decide.py`) with a test per branch.
   Nothing remains on B3. **B6-pre — DONE and PASSED**
   (`results/m8_b6_pre.json`): one file, 3,415 nodes, zero custom-domain ops, parity min-cosine
   0.99999994. D1 stays on the Stage-S menu.
5. **FineWeb arm prep (E13)** — untouched. Span sampler, full contamination/near-dup filters,
   teacher-encode. The arm itself stays refused until its bar is frozen.

## Session rules (unchanged, and they earned their keep)

Sonnet subagents for mechanical/retrieval work; Codex and Fable adversarially for anything about
to become expensive or permanent — tonight they caught a wrong P(ship) table, two divergent
qualifying-key vocabularies, a live route to the reserved qrels, and four tests the ledger claimed
existed. Commit and push after every completed item. **Smoke every new path**: tonight's smokes
caught a teacher-mismatch that killed five arms and a driver that ignored their failure.
`m8/CODEMAP.md` has fourteen pitfalls, several earned in the last twelve hours.
