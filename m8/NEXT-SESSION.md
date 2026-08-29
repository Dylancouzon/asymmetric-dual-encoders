# M8 overnight session plan (written 2026-08-29, Dylan offline ~12h)

**Read first: `m8/STATUS.md`, then `m8/PLAN-DRAFT.md` (v5 — the authoritative plan), then this
file. Delete this file when its worklist is consumed (its content graduates into LEDGER/STATUS).**

Session rules: everything below is executable WITHOUT Dylan. Anything not below that seems
necessary goes to the wake-up note (top of STATUS), not into a decision. Commit+push after every
completed item (standing grant, branch `m8-planning`). Sonnet subagents for mechanical/retrieval
work; Opus/Codex only where judgment is the product; reviews read-only. Use Codex/Fable
adversarial review before anything expensive or irreversible (standing grant).

## Hard guardrails (from the Opus review §C; violations = stop and write the wake-up note)

1. **No probe runs before its bar is committed and pushed.** Build `m8src/probe_guard.py` FIRST:
   reads `m8/LEDGER.md`, refuses any probe id lacking bar/endpoint/comparator/multiplicity/
   no-survivor outcome at the current commit. Every probe entry point calls it.
2. **Path guard**: no script under `m8src/` may open `results/frozen_eval/untouched-*`, the LoTTE
   payloads, or M9-reserve payloads. Shared-import guard + a grep test over `m8src/`.
3. **Nothing irreversible overnight**: no freeze/tag/access; no HF upload; no writes to
   `results/perquery.json`, `results/eval_manifest.json`, `results/frozen_eval/`, or `m7/`; no
   final M9-reserve hash-pin (inventories only); no amendment to any §0 FROZEN item; AMENDABLE
   changes need a dated, reasoned LEDGER entry.
4. **Noise floor before any bar is frozen** (two matched null replicates per endpoint: seed change
   and ±10% A-steps; publish; bars ≥2x floor or the B3 sign+seed template).
5. **Long-run discipline, mechanically**: smoke every new path (~90 steps); `setsid nohup` for
   anything >10 min (harness interrupts kill background tasks); monitors grep
   `Traceback|Error|FAILED|OOM|Killed|assert` alongside the progress marker; write the wall-clock
   estimate BEFORE launch and kill any job exceeding it 2x.
6. **Publish the serial GPU/RAM/disk schedule before the first probe** (include the reserved-4
   pre-encode line: ~10.12M docs ≈ 20.6 GB fp16 per system — NOT run tonight, just scheduled).
7. **Ordering interlock**: no teacher download/probe until the protected-query filter covering
   six + reserved + shadow + M9-reserve inventories is built, hashed, committed.
8. **Dev-reuse counter from evaluation #1** (`m8_dev_reuse_count.json`).
9. **Wake-up note discipline**: blocking questions to the top of STATUS; never decided alone.

## Worklist, in order

1. **LEDGER transcription** (`m8/LEDGER.md` from PLAN-DRAFT v5): the §0 inheritance table, the
   twelve rulings, pipeline order, Stage R degrees of freedom with fallbacks, probe registrations
   (bars as TBD-pending-noise-floor where §2b says so), the ship rule incl. the six-set
   no-regression guard and E12 comparators, workstream T rules incl. the swap bar, DATA
   enforcement spec, ONNX parity spec, the inherited-obligation matrix. Then **gate it**: one
   Codex read-only pass over the LEDGER text before anything runs under it.
2. **Registration deliverables**: (a) executable confirmatory decision code (draws/seed/stratified
   paired resampling/qid alignment/Holm/α-3 bound/raw-CI rule, unrounded); (b) the joint power
   simulation of the full ship rule → minimum detectable effects + **P(ship)** → wake-up note;
   (c) `rule_audit.py` ported to M8.
3. **Guards** (guardrails 1–2 above) + port the M7 guard/freeze-binding test suites to M8 paths.
4. **Protected freezes**: LoTTE download + **overlap measurement** (community intersection +
   doc-hash overlap vs reserved android/english and dev physics/programmers; any hit → drop slice
   + wake-up note; material overlap → reopen E10 in the note) → hash-pin the surviving slices.
   M9-reserve: EUR-Lex + USPTO corpus and query-text INVENTORIES only, PROVISIONAL sanity report.
5. **Protected-query filter** over six + reserved + shadow + M9 inventories; regenerate the
   closed-form fit list through it; commit hashes.
6. **Noise-floor measurement** (guardrail 4) → freeze wave-1 bars in LEDGER.
7. **Benchmark pass** (10K-doc/1K-query per new path; the timed B+A chain) → publish the schedule.
8. **Workstream T screens** (CG solver first, then: incumbent re-probe in the CG frame,
   granite-r2, gte-modernbert, stella-1.5B, harrier-0.6b; fixed student frame; provenance rows;
   ONNX feasibility evidence per finalist). Teacher freeze ONLY if the incumbent wins or a
   challenger clears the swap bar arithmetic trivially — any actual swap case goes to the wake-up
   note for Dylan's sign-off (the freeze then waits for him).
9. **Wave-1 probes** (B2, B3, B17, B9, B10) under probe_guard, if their bars are frozen and the
   teacher question is settled or provably teacher-invariant (B2/B3/B9/B10/B17 all read the
   incumbent frame; they are re-run only if a swap later lands — note this in LEDGER).
10. **STATUS update + wake-up note** before session end: what ran, what's blocked, P(ship), LoTTE
    result, teacher-screen table, any surprises. One screen.

11. **FineWeb arm prep (ruling E13, 2026-08-29 — measure first, ship-decide later):** build the
    Qdrant/FineWeb-10B span sampler (~1–2M spans), run it through the full contamination/near-dup
    filters vs ALL protected partitions, teacher-encode the survivors (~17–35 min). The arm itself
    joins the registered data probe (same bar, matched exposure, never released, refused by
    `assert_releasable`). If its result clears the bar by a shippable margin → wake-up note with
    the number; the licensing ruling returns to Dylan, never inferred.

Stretch (only if all above lands cleanly): B7 solver benchmark at 64K, stella ONNX export attempt
(config-flag recipe in `research/m8-planning/onnx-feasibility-2026-08-29.md`), the constella-zero
query-side ONNX graph prototype + conformance fixtures.
