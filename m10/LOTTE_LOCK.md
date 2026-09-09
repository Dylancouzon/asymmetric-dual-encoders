# LoTTE read #1 — pre-registration for M10

**Written 2026-09-09, while LoTTE is ENTIRELY UNREAD for M10.** M9's read #1 was withdrawn
unexecuted at its close-out (2026-09-01) precisely so M10 would inherit a fresh surface, so this is
a **first** access, not a repeat.

**Classification: PREREGISTRATION**, not an amendment. It is fixed before the build has run, before
any M10 checkpoint exists, and before any LoTTE byte is read. M9's equivalent had to be classified
an amendment because it was written during M9.3; this one does not, and the distinction must not
blur.

## The veto rule — unchanged from M9 §7, deliberately

The registered two-model non-inferiority veto runs and is reported **exactly** as locked:

> the selection is vetoed iff the selected recipe's 7-slice macro is worse than the anchor's by
> more than **0.004** AND the one-sided 97.5% paired-bootstrap upper bound (B = 10,000, seed 903,
> paired within slice) on (selected − anchor) is below **−0.004**.

A veto means **the anchor recipe builds**. Margin, bootstrap rule and consequence: untouched. **No
other action may follow from read #1.** Adoption may not trigger any retraining choice, any recipe
edit, or any change to a screen verdict.

**M10's specific case.** The screen selected no non-default component, so *the selected recipe and
the anchor recipe are the same in every axis except `batch`* (`M102_LOCK.md`). If family E selects
bs32, they are identical and the veto is vacuous — it can only compare a recipe to itself. **That is
registered here as the expected outcome, not discovered later:**

- if E selects **bs32** → the two arms of the veto are the same recipe. Read #1 is **SKIPPED** and
  reported skipped, with this paragraph as the reason. Skipping a vacuous comparison is not
  discretion; running it would manufacture a number with no estimand.
- if E selects **bs128** → the veto runs as written, selected = bs128, anchor = bs32.

## Why a read happens at all

LoTTE-clean is the only genuinely fresh out-of-domain surface available before the final run, and
the six are development-informed. The M7 precedent is the reason: an in-domain dev read
overestimated six-set retention badly (0.915 dev against 0.755 actual) while the out-of-domain read
(0.764) was accurate. M10's dev surfaces have the same shape of problem — COV shares no dataset
with the release bar, and DEV-6 contains NQ-adjacent data that is in the training mix.

So the candidate's LoTTE number is **prospective out-of-domain observational evidence**. It
supplements; it does not convert an unactionable veto into an actionable one, and it **cannot
retroactively validate the screen selection**.

## Final-checkpoint identity — REGISTERED NOW, before any LoTTE access

Deterministic, and **may not be changed using LoTTE or any LoTTE-derived output**:

1. If the build reaches its registered end (the plateau rule fires, or `max_extension_cycles` is
   exhausted, or the dose completes), the candidate is the **final annealed cycle-end checkpoint**.
   No dev selection.
2. If it terminates early (kill rule, non-finite loss or gradient, operator STOP, crash), the
   candidate is the annealed cycle-end checkpoint carrying the best COV macro `m_k`, and the report
   must disclose that it was **dev-selected**.

Training config, seed and stopping rules are those of `m10/M102_LOCK.md`, unchanged. Once training
ends, the **actual artifact hashes** are committed and pushed in a second manifest commit **before**
the evaluator may touch LoTTE.

## Firewall

- **Read #1 happens AFTER the recipe lock is pushed** and after the build, never before. Its only
  output is the veto verdict and the observational row.
- **Read #2** is the pre-freeze audit, and is audit only. **No third read.**
- LoTTE is **not** a selection surface, a bar, a tie-break, or an input to any C-conjunct. It
  appears in the final-run registry nowhere, and that is not an oversight.
- The brief for any adversarial review of this file carries the standing read-exclusion; LoTTE
  joins `results/frozen_eval/untouched-*`, the reserved qrels caches and `work/m9reserve` on it
  until read #1 executes.
- **M9's LoTTE state is not inherited.** M9's read #1 was withdrawn unexecuted; nothing about
  M10's access is conditioned on anything M9 did or did not see.

## Manifest — filled in the second commit, after training ends

| field | value |
|---|---|
| candidate checkpoint | *pending build* |
| candidate sha256 | *pending build* |
| anchor checkpoint | *pending E1; see the vacuity rule above* |
| anchor sha256 | *pending* |
| slices | LoTTE-clean, 7 slices, macro at equal weight |
| bootstrap | B = 10,000, seed 903, paired within slice, one-sided 97.5% upper bound |
| veto margin | 0.004 |
| executed | **NO — LoTTE is entirely unread for M10 as of 2026-09-09** |
