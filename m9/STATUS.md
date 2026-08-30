# M9 status — M9.1 in flight

**Stage: M9.0 LOCKED (2026-08-30), M9.1 stage A running.** Branch `m9-work`. Nothing has touched
the six or the reserved four; LoTTE unread. `results/perquery.json` untouched.

M9.1 is **staged**, on Codex's recommendation after it reviewed the lock *and the code*: stage A is
the four gates plus the **anchor arm** and its warm-start contrast; stage B — the seed replica and
the five contrast arms — runs only if the anchor clears a pre-registered **adequacy gate**
(retention ≥ 0.60 of the DEV-6 ceiling and a late-checkpoint slope ≤ 0.02). Its words were: *"the
only defensible next GPU action is a corrected, fully guarded anchor curve — not all nine arms."*
Spending six GPU-hours on contrasts measured at a dose where the student sits far from the teacher
surface would have ranked early imitability, not the factors under test.

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
decides the teacher) · **MDE = 0.0056**, one number derived from 2,031 historical dev contrasts ·
the head **warm-started in closed form** for every arm, with λ chosen on a training-only holdout. `nqopen`/`triviaqa` excluded from all of M9 rather than left as a post-hoc
freedom. Full text: `m9/LEDGER.md`; machine copy: `m9/registry.json`.

## Reviews — three adversarial passes, all before any arm ran

| pass | target | verdict | disposition |
|---|---|---|---|
| 1 | the first draft | **DO NOT COMMIT** — 7 BLOCKER / 8 MAJOR / 4 MINOR + a post-number-freedom table | `m9/LEDGER.md` §10 |
| 2 | the amended lock **and the code** | **DO NOT COMMIT. DO NOT SPEND THE 6 GPU-HOURS** — the v1 fixes had moved failures out of the prose and into the implementation | `m9/LEDGER.md` §11 |
| 3 | v3, the warm start and the adequacy gate | **v3 is broken; do not let `m9s1` open stage B** — it caught a **false statement in the lock**: the warm-start ridge λ was described as chosen on the training residual and had in fact been chosen on SCREEN-3, a dev surface | `m9/LEDGER.md` §12 |

The third pass is why the first anchor run was **killed at 11,000 of 30,349 steps and thrown
away**: it had been trained with a dev-selected λ, so nothing it produced could be preregistered
evidence. λ selection moved to a training-only holdout under the real normalized objective and the
anchor was re-run from scratch. That is the cost of the standing directive working as intended —
one wasted GPU-hour instead of a milestone built on a number chosen after seeing dev.

What the reviews actually changed: the mandate-surface conflict; two arms that confounded their own
factor with token dose; a guard that let a session amend the lock after seeing a number; a decision
threshold whose effective value was unstated; a "noise floor" built from two seeds; a batch pilot
that would have measured update count; a sorted document sample that was store-biased; and a
deterministic crash in the mandatory port pilot.

## Measured so far (no decisions read these yet)

| quantity | value |
|---|---|
| stella-400M symmetric ceiling, DEV-6 | **0.6724** |
| stella-400M symmetric ceiling, SCREEN-3 (family weights) | **0.6822** |
| student throughput, bge-small @ bs128 | ~1,990 ex/s (real texts) |
| teacher encode, stella-400M / Qwen3-0.6B | 2,076 / 1,152 q/s · 210 / 146 doc/s |
| closed-form head on a **frozen** backbone (diagnostic) | **0.3463** SCREEN-3 = **50.8%** of the ceiling |
| the same backbone, random head, 2,000 trained steps | 12.4% |
| fp16 target gate | PASS — min-cos 0.999959, max-abs 2.0e-4 |
| bridge-tolerance dry run | PASS — zero qid drift, max \|Δ nDCG@10\| **0.0** across processes |
| ONNX export, both students | PASS — min-cos 0.9999993, opset 17, **zero custom-domain ops** |
| aborted anchor at 4 epochs (quarantined, λ was dev-selected) | SCREEN-3 0.4481 = 65.7% of the ceiling |

The head-probe rows are the session's most consequential finding so far, and they changed the
recipe: at ~1% of LEAF's dose a randomly-initialized projection head spends a large share of the
whole budget re-deriving a linear map that has a closed form. Every arm now warm-starts it, and arm
`m9s1c` repeats the anchor without it to price exactly what that is worth.

The honest headline risk is unchanged: **LEAF's published 97.9% retention came from ~100 A100-hours
and 6.7M unique texts; M9's affordable dose is ~1% of that on 243K unique queries.** The
four-checkpoint curve is the registered instrument for saying what retention this budget buys, and
it is the main thing stage A exists to produce.

## Files

| file | contract |
|---|---|
| `LEDGER.md` | the M9.0 lock: protocol, rulings, and every number a rule reads |
| `registry.json` | the machine copy of those constants |
| `RESULTS.md` | runs, in order |
| `PLANNING.md` · `BRIEF.md` | the pre-M9.0 evidence and context |
| `CODEMAP.md` | modules and the pitfalls this milestone earned |
