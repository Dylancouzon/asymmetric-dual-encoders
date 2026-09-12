# Brief: second review of the M17 step-6 lock after the Astra fixes (`m17src/lock.py`)

You are reviewing code, read-only. You may run only read-only viewing commands (`cat`,
`sed -n`, `nl -ba`, `head`, `tail`, `wc`, `ls` on a named path, non-recursive `grep` over
explicitly named files or the glob `m17src/*.py`) and `git log --oneline -8`,
`git show --stat 2c157be`, `git show 2c157be -- m17src`. No other commands, no writes, no
recursive searches, no Python.

**Forbidden reads:** anything under `results/frozen_eval/untouched-*`, any reserved qrels cache,
anything under `work/m9reserve`, six-set or LoTTE payloads, and anything under `work/` or
`results/` at all for this review. List every file you read at the end.

## Context

Read `m17/REVIEW.md`, last section "Codex Astra step-6 lock review (2026-09-11)": it states the
two-half lock design, Astra's eleven findings and each disposition. The fixes are commit
`2c157be`. The owner's triage rule: smallest fix per real bug; findings whose only trigger is an
adversary editing files behind the manifests are dropped. Do not re-litigate a dropped finding
unless a careless single researcher on one box (not an attacker) could plausibly trigger it
between now and the first real training run.

The build that the pre half will price is running now (`prepare_data.py --out
work/m17/prepared/full`, no `--size`). Nothing has been locked; the registry is still
`DRAFT_NOT_EXECUTABLE`.

## Files to read

1. `m17/REVIEW.md` (last section), `m17/registry.json` (`status`, `training.decision_protocol`,
   `training.untrained_vocab_export_v0`, `allocation_hours`, `execution_entry_missing`).
2. `m17src/lock.py`, `m17src/test_lock.py`.
3. `m17src/common.py` (`require_executable`), `m17src/train.py` (`run`, `main`,
   `_load_prepared`, `_check_executed_lock`, `_check_locked_recipe`, `_save_table`),
   `m17src/export.py` (`IDENTITY_FIELDS`, `snapshot_identity`, `average_snapshots`,
   `effective_rows`, `build_bundle`), `m17src/prepare_data.py` (`TextVectorCache`,
   `stage_protected`'s return dict, `stage_manifests`' `hashes` block, `stage_identity`,
   `recipe_identity`, `build`).
4. `m17src/test_prepare_data.py` (the three `test_teacher_cache_*` tests),
   `m17src/test_train.py` and `m17src/test_export.py` only where they touch identity fields.

## What to do

1. **Confirm or refute each Astra fix.** For each row marked Fixed in the REVIEW table, name
   the refusing line, the input that used to pass, and whether the test in `test_lock.py` or
   `test_prepare_data.py` actually exercises it (read the test; do not assume from its name).
   A fix that is incomplete, introduces a fail-open path, or breaks the happy path (the
   rehearsal must still run; the on-clock `--protected-screen` rebuild must still be admitted
   under `EXECUTABLE`; the pre half must accept the real full build's `prepared.json` shape as
   `stage_manifests` writes it) is a **remaining P1** with the exact smallest fix.
2. **The real manifest shape.** `lock_pre`/`lock_executed` read `ext["size"]`, `ext["seed"]`,
   `ext["form"]`, `ext["hashes"][...]`, `ext["stages"]["vocab"]["selected"]`,
   `ext["protected_screen"][...]`, `base["hashes"][...]`, `build_record.json`'s
   `complete_build`/`wall_clock_kind`/`wall_clock_seconds`. Check every one of those against
   what `prepare_data.py` actually writes (field names, types, `None` vs missing, the receipt
   path convention `sm.rel`). A KeyError or TypeError on the real build is a P1.
3. **`train._check_executed_lock` and the two forms.** `EXECUTED_HASHES` includes `new_rows`
   and `student_ids`; the base manifest has no `new_rows` key. Does the comparison handle
   `None` vs missing the same way in the lock and in the loader? Would the control arm's real
   run be refused wrongly, or an extended arm admitted wrongly?
4. **`protocol_identity`'s whole-registry hash** now includes `execution_entry_missing`,
   `accepted_plan_revision_*` and any text field. Is there a field the lock ITSELF or the
   executed half legitimately changes (other than `status`, `lock`, `allocation_hours`) that
   would make the executed half refuse its own pre half? Is there a field that must change
   between the halves by design?
5. **V0.** `export_v0` authenticates tokenizer (`sha_file`), `new_rows` (`sha_array`) and warm
   start (`sha_file`) against `prepared.json`. Confirm each convention matches the builder's
   (`stage_vocab`, `_materialize_warm_start`, `stage_manifests`). Confirm the folded rows equal
   what `train.extend_model` would produce at step 0 (new-row scalars 1.0).
6. **Anything new the fixes broke.** In particular `require_executable(training=True)` call
   sites, `IDENTITY_FIELDS` growth versus older snapshots and the rehearsal, and the cache
   constructor's suffix drop (can it ever drop COMMITTED rows?).

Output: a table of the Astra fixes with "confirmed" / "remaining", then remaining P1s in detail
(file:line, input, smallest fix, at most one paragraph each), then any NEW P1/P2 the fixes
introduced, then the file list. Do not report style or P3s. If nothing remains, say so plainly.
