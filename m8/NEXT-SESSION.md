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

1. **Regenerate the fit list.** `m8src/fitlist.py` — written and smoke-ready, not yet run. It
   derives the TRAIN query list from the current kept pairs and screens it through the filter that
   now covers M9-reserve too. **Every teacher-screen number is ranking-only until this exists**;
   `m8src/teacher_screen.py` refuses the stale list unless explicitly overridden.
2. **T1, the teacher screens.** B7 removed the arithmetic that blocked them (LEDGER §18) and the
   four candidates' Specs are established from primary sources
   (`research/m8-planning/challenger-specs-2026-08-29.md`). What remains:
   - write `m8src/challengers.py` — the four `Spec`s, inserted into `encoders.REGISTRY` at
     RUNTIME (no `m7src` edit; G3);
   - run `validate_encoder.py` for each before any encode — skipping it is how a comparison
     silently runs the wrong model;
   - screen granite-r2 and gte-modernbert FIRST: no `trust_remote_code`, byte-identical
     tokenizers to each other, 38.7 MB tables, and they are the registered CG-frame controls;
   - stella-1.5B next (int8-only at 155 MB; register its fallback row explicitly — its
     `config.json` and `tokenizer_config.json` disagree about BOS);
   - harrier last, and only after Dylan rules — it also needs last-token pooling, which
     `m7src/teacher.py` does not implement.
   `m8src/teacher_screen.py` is written and takes one candidate per process.
3. **The remaining gap-list obligations** (LEDGER §4.4): `test_final_guard.py`,
   `test_freeze_binding.py`, a **B-leg-varying noise floor** (the one floor still unmeasured —
   R-PHASE and any pool/init change flowing through the B leg cannot be read without it), and the
   **reserved-4 pre-encode allowlist entry** (G2 currently refuses `beir/fever` outright, which is
   the right default and will block the pre-encode at pipeline step 13 unless registered first).
4. **Wave-1 probes still unrun.** B3's bar is frozen at 0.0040 (both endpoints) so its arms can
   run. B6-pre (the doc-side ONNX fuse feasibility gate) has never been attempted and gates D1.
5. **FineWeb arm prep (E13)** — untouched. Span sampler, full contamination/near-dup filters,
   teacher-encode. The arm itself stays refused until its bar is frozen.

## Session rules (unchanged, and they earned their keep)

Sonnet subagents for mechanical/retrieval work; Codex and Fable adversarially for anything about
to become expensive or permanent — tonight they caught a wrong P(ship) table, two divergent
qualifying-key vocabularies, a live route to the reserved qrels, and four tests the ledger claimed
existed. Commit and push after every completed item. **Smoke every new path**: tonight's smokes
caught a teacher-mismatch that killed five arms and a driver that ignored their failure.
`m8/CODEMAP.md` has fourteen pitfalls, several earned in the last twelve hours.
