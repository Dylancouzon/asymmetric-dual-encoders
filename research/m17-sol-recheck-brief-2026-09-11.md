# Brief: P1-only re-check of the `m17src/` fixes (M17 step 4, the single re-check)

You are re-checking code, read-only. You may run only read-only viewing commands (`cat`,
`sed -n`, `nl -ba`, `head`, `tail`, `wc`, `ls` on a named path, non-recursive `grep` over
explicitly named files or the glob `m17src/*.py`) and `git log --oneline -12`,
`git show --stat <sha>` and `git diff <sha1> <sha2> -- m17src` for the shas listed below.
No other commands, no writes, no recursive searches.

**Forbidden reads:** anything under `results/frozen_eval/untouched-*`, any reserved qrels cache,
anything under `work/m9reserve`, six-set or LoTTE payloads, and anything under `work/` or
`results/` at all for this re-check. List every file you read at the end.

## Context

Two independent implementation reviews of `m17src/` (Codex Astra, then Codex Sol) produced 29 and
22 findings. Fixes landed in commits `e2db77e`, `a2a08b9`, `c567424`, `cc88fc1` (the last one is
the Sol fix commit). The owner ruled today: do not over-engineer, this is a fairly easy
re-training; fixes are the smallest that close each real bug, and findings whose only trigger is
an adversary were dropped or deferred to the step-5 data builder. `m17/REVIEW.md` records every
disposition (read its last two sections, "Codex Astra implementation review" and "Codex Sol
implementation review").

## Files to read

1. `m17/REVIEW.md` (the two sections above), `m17/registry.json` (`training.*`,
   `data.measured_dose_after_pre_lock_rule`, `checkpoint_averaging`, `execution_entry_missing`).
2. `m17src/train.py`, `m17src/export.py`, `m17src/cache.py`, `m17src/common.py`,
   `m17src/evaluate.py`, `m17src/vocab.py`, `m17src/panel_build.py`, `m17src/alias_test_build.py`,
   `m17src/alias_pairs.py`, `m17src/support_manifest.py`, `m17src/loader_np.py`,
   `m17src/rehearse17.py`, and the ten `m17src/test_*.py` files.

## What to do

Only P1s. For each P1 in the two REVIEW.md tables that is marked **Fixed**, confirm the fix is
real: name the line, state the input that used to pass wrongly, and say whether it is now
refused and whether a test covers it. If a fix is incomplete or introduces a new fail-open
path or a regression in the happy path (the rehearsal must still run: `--rehearsal` bypasses
only the status gate and the FREEZE hash, announcing both), say so as a **remaining P1** with
the exact fix. For each P1 marked **Dropped** or **Deferred**, say in one sentence whether a
careless single researcher on one box (not an attacker) could plausibly trigger it before step 5
closes; if yes, that is a remaining P1.

Do not report P2/P3, style, or anything already dispositioned as accepted. Output: a table of
the P1s with "confirmed fixed" / "remaining", then the remaining P1s in detail (at most one
paragraph each with file:line and the smallest fix), then the file list. If nothing remains,
say so plainly.
