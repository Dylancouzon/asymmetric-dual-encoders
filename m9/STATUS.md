# M9 status — M9.1 in flight

**Stage: M9.0 LOCKED (2026-08-30), M9.1 running.** Branch `m9-work`. Nothing has touched the six
or the reserved four; LoTTE unread. `results/perquery.json` untouched.

## Needs Dylan (two items, both logged, neither blocking tonight)

1. **Mandate amendment — ratify or reject.** `instructions-m9.md` fixes the screen's tuning-dev
   macro at all six pinned components. That is impossible for a *challenger teacher*: re-encoding
   full dev in one costs 11.72M document encodes — 46.5 GPU-hours for stella-1.5B, 22.3 for
   Qwen3-0.6B on this box, ~69 for the pair, against a whole-screen budget of ~6. So the teacher
   contrast alone runs on a 3-component family-weighted surface; **student, prompt and mix keep all
   six**, exactly as the mandate says. Made before any number was observed and written down with
   the arithmetic (`m9/LEDGER.md` §0). If a challenger were to win, the run **stops** and comes back
   to you rather than proxying.
2. **FineWeb: approved by you, not exercised.** The mandate allows it only against *pre-existing*
   reserved-set **document** fingerprints. There are none — M7 persisted query-side fingerprints
   and R3 *counts* only, and the document index it streamed was discarded. Building one now would
   open reserved corpora, which the mandate forbids. FineWeb is out for M9; doc-side text comes
   from the 6.15M pre-screened M7 pool rows instead. Reopens in M10. (`m9/LEDGER.md` §1.3)

## What M9.0 locked

242,786 non-FEVER query texts · 6,149,679 eligible document rows · LEAF plain-L2 regression, mean
pooling, `Linear(hidden, 1024)` · 16 epochs = 3,884,576 examples = 30,349 steps = 59,507,872 non-pad
tokens · two surfaces (DEV-6 equal weight decides student/prompt/mix; family-weighted SCREEN-3
decides the teacher) · **MDE = max(0.0051, 2F)** where F is a *measured* seed-replica floor, not an
imported constant. `nqopen`/`triviaqa` excluded from all of M9 rather than left as a post-hoc
freedom. Full text: `m9/LEDGER.md`; machine copy: `m9/registry.json`.

## Reviews

`research/m9-codex-lock-2026-08-30.md` — gpt-5.6-sol on the first draft: **DO NOT COMMIT**,
7 BLOCKER / 8 MAJOR / 4 MINOR + a post-number-freedom table. All actioned (`m9/LEDGER.md` §10).
The five that changed the design: the mandate-surface conflict; arm 6 confounding data mix with
token dose; a guard that let a session amend the lock *after* seeing a number; an unstated
effective threshold; and a screen dose too small to be read as a final-dose ranking.
A second pass on the amended lock **and the code** ran before any compute was spent.

## Measured so far (no decisions read these yet)

| quantity | value |
|---|---|
| stella-400M symmetric ceiling, DEV-6 | **0.6724** |
| stella-400M symmetric ceiling, SCREEN-3 (family weights) | **0.6822** |
| student throughput, bge-small @ bs128 | ~1,990 ex/s (real texts) |
| teacher encode, stella-400M / Qwen3-0.6B | 2,076 / 1,152 q/s · 210 / 146 doc/s |
| retention after 2,000 steps (diagnostic smoke) | 0.124 of the SCREEN-3 ceiling |

That last row is the honest headline risk: **LEAF's published 97.9% retention came from ~100
A100-hours and 6.7M unique texts; M9's affordable dose is ~1% of it on 243K unique queries.** The
checkpoint curve across four doses is the registered instrument for saying what retention this
budget actually buys, and it is the main thing M9.1 is for.

## Files

| file | contract |
|---|---|
| `LEDGER.md` | the M9.0 lock: protocol, rulings, and every number a rule reads |
| `registry.json` | the machine copy of those constants |
| `RESULTS.md` | runs, in order |
| `PLANNING.md` · `BRIEF.md` | the pre-M9.0 evidence and context |
| `CODEMAP.md` | modules and the pitfalls this milestone earned |
