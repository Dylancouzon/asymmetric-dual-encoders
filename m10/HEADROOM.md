# Headroom plan — where the freed capacity goes (Dylan, 2026-09-04: "spend it wherever you see fit")

Revised after a Fable pre-execution review (verbatim in `m10/LEDGER.md` §3 T2-4), which corrected the
diagnosis, the requirement and the ordering, and found the two registered items that matter more
than anything I had proposed. **Nothing below had run when it was written.**

**The honest framing first.** The freed GPU-hours are **not spent here**. Every lever that would
consume them — doses, arms, confirmations, quotas — is a Tier-3 registered constant. They roll into
`max_extension_cycles` on the selected recipe at the M10.2 lock, which is where they buy the most.
What follows is CPU work that removes defects, plus two registered admissions that are worth more
than either.

**Dylan's constraints, and what they rule out.** Model usable in production, benchmarks repeatable.
So: **no architecture change** (bge-small + per-token linear head, ONNX-exportable, FastEmbed-served
exactly as `zero` is); **no evaluation-path change** (exact search, frozen comparators, registered
statistics); **no new serving or scoring dependency.** `torch.compile` is a training-step speedup
only — the rules that keep it that way are registered in §T below and bind the step-6 trainer port.

## H1 RESULT — LEDGER **ADMITTED** (a refusal made, then withdrawn on fact)

**First refused, wrongly.** I read `load_dataset_builder(...).info.features` — a stale 8-column
list — concluded "no qrels, no corpus, report-level relevance only", and wrote that into `LEDGER.md`
§2 and a commit. The loaded dataset has **13 columns including `qrels`**. Every inference after that
was sound reasoning on a false premise. **Read the artefact, not the metadata about it.**

**Admitted, structure verified** (`m10src/cov_ledger.py`, `work/m10cov/ledger_structure.json`):
10,000 queries · 494 reports → **47,820 pages** · **116,912 graded qrels**, chunk rule = the
dataset's own `<--- Page Split --->` marker, and **all 116,912 qrel page ids resolve in the split**
(0 missing), 96.8 pages/report, under the 100K cap without applying it. Screen: queries 0/0; pages
0 exact / 1 near above the 8-word floor (the raw 710 is the same sub-8-word artefact as BRIGHT).

**Why it mattered:** the COV surface goes **3,416 → 13,416 queries** and gains its **fourth family**.
§Surfaces expected most B–G contrasts to be unresolved at MDE 0.0056 and named LEDGER as the one
candidate large enough to move the surface's power. It was one stale metadata call from being lost.

## H1 — LEDGER admission (the highest-value item, and I had omitted it)

§Surfaces expects the screen's resolution distance at **0.009–0.0135** against an **MDE of 0.0056**,
so **most B–G contrasts are expected to be UNRESOLVED** on the currently admitted surface. LEDGER
(118,048 questions) is named there as "the one candidate large enough to move the surface's power".
Nothing else available this weekend changes whether the screen produces information at all.
Admit it as an ordinary admission: pin the revision, fix the chunk rule, apply the 100K cap, screen,
encode. If its structure does not verify, it is refused and the screen runs at three families with
the power disclosure saying so.

## H2 — CUREv1 admission (decision 12, adopted, and it has no §2 row)

2,000 real clinician queries, CC BY-NC (validation-only, admissible under Dylan's 2026-09-04 rule),
**never selection-bearing**. It is the only biomedical read beside MedicalQA, so it is also the
diagnostic that says whether H3 bought anything.

## H3 — the seed-supply defect, correctly diagnosed

Not corpus thinness. **Router recall.** English Wikipedia holds ~50K medicine articles; 5.23M intro
paragraphs yielding 8,844 health seeds is my 17-keyword regex under-matching, not an absent topic.
So the first lever is **recall at fixed precision**, not a relaxed floor.

**Health is ranked first, and processed first:** two of the four clean-4 headline datasets
(`nfcorpus`, `trec-covid`) are biomedical. `fiqa` is not in clean-4 and is stella-disclosed, so
`finance` ranks second.

**The requirement is ~32–33K seeds per form, not 28.6K.** 143K/5 is a floor that ignores contract
failures, exact dedup, the 5-gram seed-copy screen, A8's near-duplicate cut and the 500-seed
FORMS-12 hold-out. **The "5 queries per seed" is NOT registered anywhere** — it is the smoke's
constant and `forms.prompt`'s default. The projections below are also pre-screen and pre-`used`.

**Ladder, fixed here before P1 counts, stopping at the first rung that clears.** Rungs 2–4 amend
T2-3 (which registered `ROUTE` + `min_score ≥ 4` for the build draw); logged as such.

1. **More approved stores.** `hotpotqa-corpus`, `squad-ctx`, `mrtydi-docs`. **`esci-prod` is
   excluded from the topical scan** — product listings cannot serve howto/health/finance.
   Cross-store text-fingerprint dedup, since all three are Wikipedia slices.
2. **The widened `ROUTE` lists in `m10src/seeds.ROUTE_WIDE`** — fixed and committed *before* the
   scan, never tuned to the number it produces. Recall lever, precision held.
3. **Relax `min_score` 4 → 3, then → 2**, for the short form only, stopping at the first floor that
   passes. Precision lever, so it is gated: **the full 200 queries, uniform-random, judged by an
   independent Fable subagent against the frozen `RUBRIC`, must hold ≥ 80%.** (50 queries has
   SE ≈ 5.7 points and cannot tell 80 from 72.) No trying floor 2 "to see" once 3 passes.
4. **Raise queries-per-seed 5 → 8** for the short form. Last: it changes the prompt text `n` is
   formatted into, so it needs its **own 200-query re-smoke and a fresh six-hour veto window**,
   because it changes the basis the approved prompt-hash was approved on.

A form that still cannot reach quota reports its realized count; **no top-up from another form**.
Disclose the interaction with form-balanced sampling: a short form is drawn with replacement, so its
texts are seen more often.

## H3 RESULT — v1 WITHDRAWN as defective; v2 rescan in flight

`results/m10_seed_supply.json`, `m10src/seeds.supply()`. Full stores, 5,348,204 documents scanned,
5,339,995 unique after cross-store fingerprint dedup, 2,615,015 length-eligible. Controlled: both
rows are the SAME stores and the SAME `min_score = 4`, so the only variable is the keyword list.

| form | orig `ROUTE` | `ROUTE_WIDE` | gain | vs the ~32–33K need |
|---|---|---|---|---|
| `health` | 10,399 | **42,380** | **4.1×** | clears |
| `finance` | 22,375 | **39,918** | 1.8× | clears |
| `howto` | 37,927 | **37,154** | 1.0× | clears |

**The table above is WITHDRAWN.** It is v1's, and v1's widened lists kept the form
`\b(alt|alt|…)\w*\b`, which wildcards every alternative: `pain` matched "paints", `nurse`
"nursery", `chronic` "chronicle", `capital` "capital city", `credit` "credited". A painter and a
truck-painting business were routed as `health` seeds, so the 4.1× is part recall gain and part
precision loss — and I had claimed "precision held" without measuring it, which is the whole defect.
**v2 lists explicit word forms** and is being rescanned. Registered with v2, because asserting
precision is what failed: a sample of the passages the widening newly admits is judged on-topic by
an independent Fable subagent before the widened routing is used for a build draw. The diagnosis
(router recall, not corpus thinness) still stands; its size does not, and rungs 3–4 may yet be
needed for `finance`.

`howto` slips 773 seeds because health and finance now claim shared passages ahead of it in the
fixed priority order — the `used` exclusion behaving as designed, disclosed rather than tuned away.

**Registered consequence:** the BUILD seed draw uses `ROUTE_WIDE` (`seeds.draw(route=...)` defaults
to it) and `SCREEN_VERSION` is bumped so any cached draw made under the old routing is invalidated.
The smoke's seeds were drawn under the original `ROUTE`; that is unchanged and remains a gate
artifact that never enters a build corpus.

## For Dylan — one question, default unchanged

**Raise the `health` quota above ≈143K?** It is 1 of 7 generated forms but carries 2 of the 4
headline datasets. Quotas are Tier 3, so this is his and the default is 143K. Not decided here.
Supply is no longer the constraint: at 42,380 seeds health could support ~212K queries at 5 each.

## §T — `torch.compile` rules, registered now, binding the step-6 trainer port

1. Checkpoints save the **eager** module (`_orig_mod.state_dict()`), never the compiled wrapper.
2. Export, parity test, teacher-target encodes, stella encodes and all COV/DEV evaluation run
   **eager**; compile appears only in the training step.
3. The parity test loads a checkpoint produced by a **compiled** run.

Under these no released weight and no reported number passes through compile. Compiled training is
not bit-reproducible run to run — neither is eager bf16 — and "repeatable benchmarks" is satisfied
by exact search over frozen vectors, which is unchanged.

## Deliberately NOT doing

- **Not raising generated volume.** Four curves saturate below 1.0M (`PLANNING` §12) and **A2−A1
  already measures volume** (A2 carries the full 4.037M PAQ against A1's 463K) at ~2 GPU-hours.
- Not adding forms, unapproved corpora, or a further screen arm; not touching the student, head,
  objective or export path.
