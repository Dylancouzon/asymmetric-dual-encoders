# Codex Astra brief — M17 step 6b re-check: widen the read list, answer Q1–4 (2026-09-11)

Read-only. Adversarial. This continues `research/m17-astra-6b-fix-brief-2026-09-11.md` (read it
first for the situation and questions). Your first pass (`research/m17-astra-6b-fix-review-2026-09-11.md`)
found two P2s, both fixed in commit `bfa7644`:

- `common.reassert_path_order()` now also evicts a `sys.modules["train"]` whose `__file__` is
  under `m7src`.
- `test_stage_protected_restores_the_path_order_after_importing_protected10` calls the production
  site with a stub `protected10` whose `build()` stops the stage; it fails when the production call
  is removed (mutation-checked).

You said Q1–4 needed more source. Granted, read-only, in addition to the first brief's list:
`m10src/decontam.py`, every module `m10src/protected10.py` imports (follow its imports one level;
name what you read), `m17src/train.py` (`_load_prepared`, `_check_cache_belongs_here`,
`_check_locked_recipe`, `load_warm_start`), `m17src/export.py`, `m17src/evaluate.py`,
`m17src/cache.py` (import lines and `load`), `m17src/vocab.py` (import lines).
`git show bfa7644 -- m17src/common.py m17src/test_prepare_data.py`.

**Read exclusion (mandatory, unchanged):** nothing under `results/frozen_eval/untouched-*`, any
reserved qrels cache, `work/m9reserve`, `work/m17/prepared/`, or `work/m17/logs/` except
`work/m17/logs/prepare_full_screen_crash1.log` and `work/m17/logs/prepare_full_screen.log`
(stage lines only; they hold no query text).

## State since the first brief

The resumed build finished: `prepared -> work/m17/prepared/full (3779.4 s this invocation)`,
zero `Traceback|Error|FAILED|REFUSED` lines, `protected_screen.state == complete` with a receipt,
1,512 rows dropped by the screen and refilled to 600,000, seed 0, `size` null,
`registry_status_at_build == EXECUTABLE`, `hashes.prepare_data_py == aefcf585…` (the amended
value). Nothing under the prepared directory has been read as a quality surface. The executed half
has NOT been run yet; it has no dry run by design.

## Answer these

1. (Q1) With the wider list: is there any import of a name shared between `m7src`/`m10src` and
   `m17src` that could resolve wrongly in the on-clock process after `protected10` is imported,
   before or after the fix? List each shared basename you found across the three trees.
2. (Q2) Did any stage before `parity` (pool, domain, protected, teacher, bank, v1) import `train`
   or another shared-name module AFTER `protected10` was imported in the crashed invocation? If
   none did, the six reused markers are sound; if any did, say which and what could differ.
3. (Q3/Q4) Does `lock_executed`, `train._load_prepared`, `export`, or `evaluate` anywhere hold or
   compare the OLD builder hash, refuse a `partial rebuild` build record, or read the wall clock
   from it? Is the `lock.amendments` entry sufficient as a dated amendment?
4. Two verdicts on one line each: (a) may the resumed build proceed to `lock.py --phase executed`
   as it stands; (b) anything that must be fixed BEFORE that command runs, versus after.

Findings only, ranked P1/P2/P3 with file:line and a failure scenario. Do not restate the first
review.
