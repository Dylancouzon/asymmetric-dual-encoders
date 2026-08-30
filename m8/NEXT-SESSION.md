# M8 worklist

**Read `m8/STATUS.md`, then `m8/LEDGER.md` §24 (`D2-PRE`'s verdict) and §15's bar-freeze amendment.
This file is the remaining work only.**

`D2-PRE` returned **DO NOT AUTHORISE** on 2026-08-29: all four new-row classes negative out-of-fold.
**`D2` and the additive n-gram class are both CLOSED** (§24, `m8/EXPLORED.md`). Do not reopen either
without a mechanism §24 does not already exclude — coverage, compile, leakage and λ were each
checked and excluded, and two-thirds of the added rows were measurably inert.

## 1. `B8` — run FIRST. ~15 minutes.

Closed-form document-centroid target against R0's target, bar **0.0040** on the closed-form dev
group vector, DIAGNOSTIC ONLY (nothing ships from a closed-form fit — the rule `D2-PRE` ran under).
Its bar was frozen 2026-08-29; it had been unrunnable for the whole milestone. The ledger calls
deferring it "a false economy" and it was deferred behind a multi-hour tokenizer tournament anyway.

## 2. `R-LIST` — `B2` triggered it directly

Hard-candidate listwise distillation. `B2` measured the recipe's uniform-bank KL at 4.73e-07 nats
(inert) but its `teacher_top200` arm at **0.777 nats**, so the KL CLASS IS NOT CLOSED — only the
recipe's degenerate instance is. Bar **0.0040**, A-leg-only (reuses R0's Phase-B checkpoint); an arm
that also retrains the B leg reads 0.00519 instead, fixed before it runs.

## 3. `VECTOR-PRF` — unregistered, and ranked above `B10`. Write the row first.

`q' = normalize(αq + β·mean(d_1..k))` at the PUBLISHED fixed config α=0.4, β=0.6, k=3
(arXiv 2205.00235) — no tuning grid, no training, no document-index growth. It attacks the
bag-vs-context mismatch directly rather than hoping vocabulary granularity repairs it, which is the
hypothesis `D2-PRE` just falsified. **It is a SYSTEM change, not a better table** — Dylan's
2026-08-29 ruling permits that (condition 4), but the row must say so and the report must decompose
it. Costs a second ANN pass and carries query-drift risk. One frozen config, dense AND fused read.

## 4. `B10` — the named alternate, weak prior

`pool_mode`, redesigned arm, bar **0.00519** (trains through the served rule → chain-varying), raw
CI > 0 in BOTH precisions. M7's lever #6 arm (a) measured +0.0011, p=0.051/0.073, CI straddling zero.

## 5. `E10` — decide it, do not rebuild it

The remedy artifact stays unpinned; the review's BLOCKERs stand (`m8/LEDGER.md` §15, "E10
REOPENED"). A shadow can stop a bad artifact shipping but may never be a selection surface, so it
cannot raise the number. **Either** rebuild it properly — union screen across roles, an independent
acceptance detector, a canary that proves acceptance can fail, length-adaptive matching — **or**
declare M8 ran without a shadow and record that as a stated limitation. A third rebuild attempt is
not worth a session.

## 6. G8's dev-reuse counter — DONE (2026-08-29)

`results/m8_dev_reuse_count.json` exists (`m8src/dev_reuse_m8.py`). **494 cumulative in-training dev
evaluations** across M7+M8; M8 alone: 35 trained arms, 172 in-training evaluations, 112 eval-only
variants. Quote it beside any dev-read verdict: nested selection protects the *reserved* sets, it
does not make the out-of-domain macro a fresh surface.

## 7. The exit — now actually executable

`D2` has run and missed. The exit still needs **`B8`**, **`R-LIST`** and **`B10`** to run and miss,
then a re-run of CLAUDE.md's standing directive. All three were REFUSED by `probe_guard` until
2026-08-29 (stale `TBD-noise-floor` bars), so the "pre-committed exit" was un-executable for the
whole milestone; bars are frozen now (§15) and all three are runnable. Only then does the default —
do not spend the access — apply. `VECTOR-PRF` is a live lever and belongs before the exit too.

## 8. Deferred — do not start these

`freeze.py` / `final_run.py` (§4.4's gap list), `B9`, `B13`, `B14`, `B15`, `B6`, `D-GENRE`,
`D-SYNTH`, `R-PHASE`, `D-FINEWEB` (pool-varying, bar not computable), `R1-ASSEMBLY`, `S-SELECT`.
Reasons are on each registry row's `plan_status`. **No new floors, guards or registry machinery
until a capacity lever has a number** — the one exception is the D×downstream floor `E14-LORA`
needs, which is a precondition for writing its bar at all.

## Unregistered ideas, recorded so they are not lost

**Vector-PRF has been PROMOTED to §3** — do not leave it here.

**Query-side small-k** (not reviewed; an external review ranked it BELOW Vector-PRF: max-over-k is an
OR operator that rewards a document matching any fragment and can destroy conjunction, which is
least plausible on the short CQADupStack OOD pair. If tested, use deterministic contiguous grouping,
never per-query k-means with another pile of choices): cluster the fired rows into 2–4 query
vectors, score max-over-k against the *single* document vector. Not the banned token-level late
interaction; leaves the document index untouched; same table bytes; costs k ANN queries. Attacks
the single-vector ceiling (LIMIT, logged in M1) that is the ArguAna multi-topic failure mode.
Testable with **no training** on existing artifacts. **Small-k document facets** stay out: §8
already records the objection — they multiply the document index, which is the one resource the
edge product cannot spend.

## Session rules

Sonnet subagents for mechanical work; **Codex adversarially before anything becomes permanent** —
it has returned BLOCKERs on an implementation, a STOP on a shadow, and two over-claims in an
already-committed write-up. **Commit and push before launching guarded work**, and treat
`m8/LEDGER.md` and `m8/registry.json` as frozen for the duration of any guarded run (CODEMAP 24).
Smoke every new path; check the RATE, not just that it runs. `setsid nohup` for anything long.
`git status m7/ results/` after anything out of `m7src/`.
