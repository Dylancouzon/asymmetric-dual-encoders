# LoTTE read #1 — amendment and pre-registration

**Written 2026-08-31, during M9.3, while LoTTE is ENTIRELY UNREAD.** Approved by adversarial
review (`research/codex_lotte.log`). In `m9/` rather than `m9/LEDGER.md` because protocol-scope
files are frozen for the build's duration (see `m9/BUILD_LOG.md`); merge after the build.

**This is a material amendment — an added model and a changed timing — not execution of the
original registration unchanged.** It must never be described as the latter.

## What is unchanged

The registered two-model non-inferiority veto runs and is reported **exactly** as locked in
`m9/LEDGER.md` §7: `m9s6` (selected: stella-400M × bge-small × prompt b × 5/5/90) vs `m9s1`
(fallback: query-only) at equal dose; the selection stands unless the selected recipe's 7-slice
macro is worse by more than **0.004** AND the 97.5% one-sided paired-bootstrap upper bound on
(selected − fallback) is below **−0.004**; adoption may not trigger any retraining choice.
Margin, bootstrap rule and fallback consequence: untouched.

## What is added, and why

A **third** hash-pinned checkpoint — the build's final candidate — scored in the *same* atomic
batch. LoTTE-clean is the only genuinely fresh surface available before the final run, and the six
are development-informed. Our dev read (SCREEN-3) weights NQ at 0.50 while NQ-adjacent data is in
the training mix, and the M7 precedent is that an in-domain dev read overestimated six-set
retention badly (0.915 dev vs 0.755 actual) while the out-of-domain read (0.764) was accurate.

**Correctly characterised** (per the review): this SUPPLEMENTS the veto; it does not convert an
unactionable veto into an actionable one. The candidate's LoTTE number is **prospective
out-of-domain observational evidence**. It cannot retroactively validate the screen selection.

## Final-checkpoint identity rule — REGISTERED NOW, before any LoTTE access

Deterministic, and **may not be changed using LoTTE or any LoTTE-derived output**:

1. If the run reaches `cooldown complete`, the candidate is the **final post-cooldown
   checkpoint**. No dev selection.
2. If it terminates before cooldown completion (regression, non-finite, operator STOP, watchdog
   give-up), the candidate is the checkpoint carrying `best` by SCREEN-3, and the report must
   disclose that it was **dev-selected**.

Training config, seed and stopping rules are those of `m9/M92_LOCK.md` and
`work/m9long/config.json`, unchanged. Once training ends, the **actual artifact hashes** are
committed and pushed in a second manifest commit **before** the evaluator may touch LoTTE.

## Firewall — the conditions of approval

- **No LoTTE-derived output may influence anything downstream.** No retraining, no checkpoint
  substitution, no change to gates, bars, statistics or claims, and no decision about whether to
  execute C1/C2. **C1/C2 proceed as already committed, unconditionally.**
- Read #1 may select a fusion weight from the locked grid; any such weight is **LoTTE-tuned and
  descriptive**, must be labelled so, and must never be presented as fresh validation. Fusion is
  not gated.
- All three models fixed before execution, identical preprocessing and inference dose, one batch.
- **No partial inspection**: no logs, per-slice scores or intermediate outputs may be examined
  until all three evaluations complete.
- **Predeclared failure handling:** an infrastructure failure before any score is written permits
  exactly one rerun of the identical batch; a failure after any score exists ends read #1 and the
  partial result is discarded unread and reported as spent. A partial failure may never become an
  opportunity to inspect two models and reconsider the third.
- Nothing may be added to the batch after any LoTTE-derived output exists.
- **Read #2 keeps only its registered authority**: audit-only, pre-freeze, may stop M9 or remove a
  claim; it may not select models, weights, thresholds, checkpoints or training. No third read.

## Analysis plan (fixed here)

7 slices, macro over slices, never pooled. Paired bootstrap: unit = query, paired within slice,
B = 10,000, seed `903`, one-sided 97.5% upper bound for the veto; the candidate's rows are
descriptive with two-sided 95% `ci95_raw` at full precision. Retention denominator for the
candidate: the same teacher (stella-400M) scored on the identical slices. Fusion grid and tie rule
as registered in `m9/LEDGER.md` §7. Evaluator code commit and environment recorded in the manifest
commit.
