# M8 — v2 of the zero-compute query table (tiny mandate)

*Created 2026-08-28, before M7's final run, when Dylan renumbered the milestones: M8 is the
learnings-driven v2 of M7's released table; the LEAF-style distilled tower moved to
`instructions-m9.md`.*

Run after M7's final run, same box, same session rules. M8 takes M7's frozen recipe, its final
numbers, and everything in `m7/FINDINGS.md` / `m7/EXPLORED.md`, and builds a stronger v2 of the
same artifact class: a Qdrant lookup-table query encoder, zero transformer at query time.

Everything binds from `instructions-m7.md` unchanged: decision authority, licensing and
decontamination rules, dev-only selection, the freeze/ledger protocol for one final confirmatory
access, adversarial reviews as routine instruments, Sonnet subagents for research, and the
headless git contract — working files under `m8/`, same four-file split. Scope, levers and bars
are set in `m8/LEDGER.md` AFTER M7's final number is read — that is the point of a learnings v2 —
except for the items below, which are only legal to fix while M7's number does not yet exist and
are therefore fixed now.

## Pre-registered NOW (2026-08-28, before any M7 six-set number)

1. **M8's confirmatory evaluation runs on the reserved untouched-final four** — FEVER,
   DBpedia-entity, CQADupStack-android, CQADupStack-english — already hash-pinned in
   `results/eval_manifest.json` + `results/frozen_eval/` with the overlap disclosures in
   `m7/LEDGER.md` (FEVER 11.3% / DBpedia 9.32% TRAIN-document overlap; the CQADupStack pair is
   within-family transfer). **M7's final run therefore defaults to skipping the untouched tail**,
   reserving these sets un-scored; Dylan may override before the tail would run, and an override
   burns them for M8 (they become development-visible the moment they are scored). This default
   costs M7 nothing confirmatory — the tail was always descriptive-only and deferrable.
2. **The M7-vs-M8 comparison is confirmatory only on the reserved sets, scored paired in M8's own
   one-shot access**: frozen M7 artifact and frozen M8 artifact through the same harness, same
   statistics family as M7's tier rule (Holm + raw CI + simultaneous bound, dependence-preserving
   where nested), levels fixed in `m8/LEDGER.md` before the access. M7's six-set number is a
   REFERENCE for M8, never re-earned: M8 may score the six descriptively for continuity, but every
   six-set claim carries the label "development-informed at milestone level" because M8's design
   reads M7's per-dataset results.
3. **Comparator bars on the reserved sets are fixed in `m8/LEDGER.md` before any M8 training run**,
   from: BM25 via the frozen `fusion.bm25_run` builder (the one lexical function), the frozen M7
   released system, and published numbers as labelled context only (no frozen per-query vectors
   exist for these four; that limitation is stated, not papered over).
4. **The clean-stack tax variant stays an M7 task** (post-final-run, already registered in
   `m7/LEDGER.md`); its result is an M8 input, not an M8 experiment.

## Known levers carried in from M7, each requiring its own pre-registration in `m8/LEDGER.md`

- Bigram/n-gram rows **trained through the forward** (closed in M7 only for closed-form
  integration; the joint retrain is the one capacity direction the algebra left open).
- Training through the `sqrt` pooling rule at scale (lever #6 failed at arm (a) under a frozen
  falsifier; a v2 may redesign the arm, not revive the dead one).
- doc2query doc-side expansion at full dose — **blocked on Dylan's licensing ruling for a
  commercially clean generator**, still open.
- The negatives/step-count confound (`m7/LEDGER.md`: "the dev suite cannot separate the negatives
  source from the step count") — resolvable with a matched-steps design if worth the compute.
- A teacher revisit is allowed by the swap bar in `m7/LEDGER.md` (closed-form table criterion,
  off-family read, Dylan's sign-off) — a NEW milestone's pre-registration, exactly as the
  one-access rule requires.

Deliverables: decided in `m8/LEDGER.md` after M7's report; the release bar must be at least "beats
the frozen M7 released system CI-resolved on the reserved sets" for a v2 to ship as a replacement.
