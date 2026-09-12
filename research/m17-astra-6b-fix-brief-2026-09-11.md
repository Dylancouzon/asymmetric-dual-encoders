# Codex Astra brief — M17 step 6b: on-clock builder fix and pre-half amendment (2026-09-11)

Read-only. Adversarial: try to break the fix and the amendment. Do not confirm; report defects
with file:line and a concrete failure scenario, ranked P1/P2/P3. Nothing found is a valid answer
only after you checked every question below.

## Files (read these only; no recursive searches over the repo)

- `m17src/common.py` (`reassert_path_order`)
- `m17src/prepare_data.py` — `stage_protected` (~line 1250–1262), `stage_parity` (~1772),
  `_rehydrate`/resume block (~2101–2140), `stage_identity` (~2169), `_invalidate_after`/stale
  markers (~2206–2225), build record (~2275–2290)
- `m17src/lock.py` — `INVARIANT_HASHES` (~54–60), `measured_allocation` (112–148),
  `lock_executed` (260–328)
- `m17/registry.json` — the `lock` block only (`lock.amendments`, `lock.invariant_build_inputs_sha256`)
- `m17src/test_prepare_data.py` — the last test,
  `test_train_resolves_to_m17_after_a_legacy_module_puts_m7src_first`, and
  `test_the_protected_screen_gates_on_the_registry_before_importing_protected10`
- `m10src/protected10.py` lines 15–24 only
- `git log --oneline -6` and `git show d8f2b43 --stat`; `git show d8f2b43 -- m17src/common.py m17src/prepare_data.py`

**Read exclusion (mandatory):** do not open anything under `results/frozen_eval/untouched-*`,
any reserved qrels cache, `work/m9reserve`, `work/m17/prepared/`, or `work/m17/logs/` other
than `work/m17/logs/prepare_full_screen_crash1.log` (the traceback is at its end).

## What happened

The pre half of the lock (commit `e0a1a40`) bound `sha256(m17src/prepare_data.py)` as an
invariant build input. The first on-clock screened rebuild
(`prepare_data.py --out work/m17/prepared/full --protected-screen --force`) died at `parity`:
`AttributeError: module 'train' has no attribute 'load_warm_start'`. Cause: `m10src/protected10`
does `sys.path.insert(0, m7src)` at import; `stage_parity`'s lazy `import train` then resolved
to the legacy `m7src/train.py`. The unscreened full build never imported `protected10`, and the
synthetic rehearsal does not run the real screen, so this path was never exercised.

Fix (commit `d8f2b43`): `common.reassert_path_order()` (the module-load ordering loop, now
callable) is called immediately after `import protected10` in `stage_protected`. The pre half's
`prepare_data_py` hash was amended by hand, dated, with old/new/reason/commit recorded in
`lock.amendments`; the original block is at `e0a1a40`. No recipe, constant, stage logic or
identity field changed; no quality surface was read (the build died before vocab/cache/manifests).

The build was then resumed WITHOUT `--force`: stages pool, domain, protected, teacher, bank, v1
were `cached, skipped` (their markers' `stage_identity` does not include the builder source hash),
parity passed, and vocab/cache/manifests are running now.

## Questions to break

1. Is the fix complete? After `protected10.build()` returns, can anything else during
   `stage_protected` or later stages re-insert `m7src` ahead of `m17src`, or import a wrong module
   under a name both trees share (`train` is the only shared basename between `m7src` and `m17src`;
   check `m10src` imports that `protected10` pulls in — e.g. `decontam` — for shared names with
   `m17src`)? Could a module already cached in `sys.modules` under a legacy identity be reused?
2. Is resuming without `--force` sound? The six reused markers were produced by the invocation
   that died. Their outputs on disk (screened pool, receipt, teacher_q, bank, v1) were written by
   code identical to the fixed builder except for the two-line path fix. Is there any way the
   wrong-`train` resolution could have affected a stage BEFORE parity (i.e. did anything before
   parity import `train`, `cache`, `vocab`, `export`, `evaluate` or `common` after `protected10`
   was imported)? If yes, the six stages must be rebuilt.
3. The amendment: does `lock_executed` compare `ext["hashes"]["prepare_data_py"]` (computed by
   the manifests stage of the resumed run from the FIXED source) with the amended lock value, and
   does anything else in the lock or in `train._load_prepared` still hold the OLD hash? Is
   `registry_sha256_excluding_lock_fields` unaffected by an edit inside `lock`? Is the amendment
   format acceptable as a "dated amendment preserving the original registration in git"?
4. The resumed build record will say `wall_clock_kind: "partial rebuild"`. Does `lock_executed`
   or anything downstream (train, export, evaluate) refuse a partial-rebuild record or read the
   wall clock from it? Does the on-clock preparation cost for the LEDGER need the two invocations
   summed (crash1 ran pool→v1 plus the failed parity; resume runs parity→manifests)?
5. The regression test uses `importlib.machinery.PathFinder.find_spec`. Does it actually prove the
   production call site is protected, or only that the helper works? Would a test that imports
   `protected10` be feasible without materializing protected payloads (the P1-1 gate)?
6. Anything in the amendment reasoning that is wrong or overstated.

Output: findings only, ranked, with the exact line and scenario; then a one-line verdict on
whether the resumed build may proceed to the executed half.
