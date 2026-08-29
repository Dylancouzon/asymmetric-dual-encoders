# M8 worklist

**Read `m8/STATUS.md`, then `m8/LEDGER.md` §15's audit entry (2026-08-29), then the `D2` row in
`m8/registry.json`. This file is the remaining work only.**

Re-routed 2026-08-29 by a milestone audit, then twice by adversarial review. **Two capacity routes
are live: `D2` (table side, gated by `D2-PRE`) and `E14-LORA` (document side, authorised but its row
is unwritten).** Everything else is deferred. Nothing is blocked on Dylan.

## 1. `D2-PRE` — the closed-form preflight. Run this FIRST; it can reverse the plan.

Under an hour, no training chains, on the block-CG solver `B7` already proved. The registry row is
the authority. Staged, each stage able to stop the probe:

1. **Opportunity** — retokenize dev+train queries, measure the actual fertility reduction on the
   high-gap queries. Far below ~0.104 out-of-domain and even the optimistic association cannot
   reach the bar.
2. **Compile/floor** — sum-initialize new rows, zero residual, score against R0: must reproduce it
   within 0.001 dense. **Score mean-init beside it as a negative control** — if mean is not worse,
   the pooling model in the registration is wrong and everything downstream stops.
3. **Residual** — solve only the new rows against existing teacher-query targets, **cross-fitted**
   so the scored dataset's own queries never enter the fit (`B17` measured its own 957-query fit
   set and was disowned; do not repeat it).
4. **Four arms at equal row budget**: D2 segmentation · additive overlapping word n-grams ·
   additive character n-grams · D2 with zero-residual fallback.

**The router is NUMERIC and frozen** (registry `D2-PRE.bar` — an earlier draft said "clearly
positive / plausibly above / adequate / no material", which is judgement, not a rule). Authorise
full `D2` chains iff ALL of: cross-fitted `g_best ≥ +0.0052` (the chain bar itself — a screen
routing a five-chain spend must clear the bar those chains will face); `g > 0` in ≥ 4 of 5 folds;
zero-update occurrence mass ≤ 20%; fused ≥ −0.0020; and the sum-init compile reproduced R0 within
0.001 **with mean-init strictly worse**. **Reversal margin, frozen:** additive replaces D2 if it
leads by ≥ 0.0020 at equal row budget, and **ties break to additive** — a zero-residual additive row
recovers R0 exactly, a segmentation change does not. Nothing ships from a closed-form fit.

## 2. `D2` — full chains, ONLY if `D2-PRE` clears

1. **Coverage precondition** on the **fixed** pool — occurrence mass at 0/<5/<20 effective updates,
   plus the performance gate (sum-init compile within 0.001 of R0). **No pool expansion inside D2**;
   if a vocabulary cannot be covered, take the smaller one or NO-GO.
2. **Smoke both chains at ~90 steps** before any full arm (G5). New code path, no execution
   history — that is where the bug is.
3. **Selection**: one seed per vocabulary, read on wikipedia + heldout only. Deterministic seeded
   tokenizer, sha256 into the result.
4. **Reported set**: finalist at 3 seeds vs R0. **R0's three chains already exist** as §23's
   diagonal — `m8nf-seed0`, `m8nfb-seed{1,2}-{b,a}`. Verify they read as full chains; never
   substitute an A-leg-only arm.
5. **Bar 0.00519 dense.** Rows are trained as a **residual on the sum init**, so an under-updated
   row stays at zero and reproduces R0 exactly.

**Kill criterion, and it must be allowed to fire.** One continuation at doubled steps if the
adequacy gate trips, paired with a doubled-step R0. Then the class closes. No third budget, no
vocabulary escalation after a number, no re-tokenized retry.

## 3. `E14-LORA` — AUTHORISED, and the other capacity route. Write the row before any arm.

Dylan ruled M8 may ship a better **system**, so a document-side win now carries a v2 (condition 4,
amended in prose and code). Licence closed — stella is MIT. **Not runnable yet, and `probe_guard`
will refuse it: its bar is TBD by design.** Order, and it is not negotiable:

1. **Measure the D×downstream floor first.** `0.00519` does **not** apply — it bounds B×A under a
   *fixed* document tower (§23); a LoRA adds a document-training leg whose variance nothing has
   measured. Same rule that governed the B leg: no bar may read such an arm until its floor exists.
2. **Then the comparator.** Frozen R0 is **not** honest — it confounds "the tower changed" with
   "the table was retrained against a new tower". Primary control is a **stock-stella tower whose
   table is jointly retrained under the identical recipe, seeds and budget**; R0 is secondary.
3. **Then the bar**, and only then an arm. One **fixed** LoRA config — the pre-registered-lr rule
   binds, no config search — two OOD components against their own re-encoded corpora, three paired
   end-to-end seeds. **No 10.12M pre-encode until that clears.**

Bounds: derived from stella only (LoRA/adapter/last-block/head), never from scratch; query side
stays a pure lookup table (E1).

## 4. `NF-CROSSED-FUSED` — MANDATORY before any `D2` success claim

Nine fused scoring passes over cells already on disk, no training. Fused is read as a pre-declared
non-inferiority comparison against the floor it measures — not a sign test. A dense-only win is a
**mechanism** success, never a release success.

## 5. `E10` — time-boxed, and it does not gate `D2`

The remedy artifact stays unpinned; the review's BLOCKERs stand (`m8/LEDGER.md` §15, "E10
REOPENED"). A shadow can stop a bad artifact shipping but may never be a selection surface, so it
cannot raise the number. **Either** rebuild it properly — union screen across roles, an independent
acceptance detector, a canary that proves acceptance can fail, length-adaptive matching — **or**
declare M8 ran without a shadow and record that as a stated limitation. A third rebuild attempt is
not worth a session.

## 6. G8's dev-reuse counter is MISSING — build it

`results/m8_dev_reuse_count.json` is promised by §14 G8 and absent from HEAD. M7 logged 322
in-training dev evaluations. Nested selection protects the *reserved* sets; it does not make
wikipedia/heldout selection independent of a CQADupStack dev bar, and a 0.005-scale development
difference deserves the counter that was promised. Small, and it should exist before D2's number is
interpreted.

## 7. If the capacity levers miss — the exit CANNOT fire yet

Registered order before the exit is even eligible: **`B10`** (`pool_mode`, redesigned arm, own bar),
**`B8`** (closed-form document-centroid target, ~15 min — deferring it was a false economy), and
**`R-LIST`** (hard-candidate listwise distillation, which `B2` directly triggered: its
`teacher_top200` arm is 0.777 nats, so the KL class is NOT closed). Then re-run CLAUDE.md's standing
directive. Only then does the default — do not spend the access — apply.

## 8. Deferred — do not start these

`freeze.py` / `final_run.py` (§4.4's gap list), `B9`, `B13`, `B14`, `B15`, `B6`, `D-GENRE`,
`D-SYNTH`, `R-PHASE`, `D-FINEWEB` (pool-varying, bar not computable), `R1-ASSEMBLY`, `S-SELECT`.
Reasons are on each registry row's `plan_status`. **No new floors, guards or registry machinery
until a capacity lever has a number** — the one exception is the D×downstream floor `E14-LORA`
needs, which is a precondition for writing its bar at all.

## Unregistered ideas, recorded so they are not lost

**Query-side small-k** (mine, not in the repo, not reviewed): cluster the fired rows into 2–4 query
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
