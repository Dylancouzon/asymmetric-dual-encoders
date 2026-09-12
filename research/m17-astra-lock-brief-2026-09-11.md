# Brief: break the M17 step-6 lock (`m17src/lock.py`) and the teacher-cache flush change

You are reviewing code and one protocol-sequencing decision, read-only. You may run only
read-only viewing commands (`cat`, `sed -n`, `nl -ba`, `head`, `tail`, `wc`, `ls` on a named
path, non-recursive `grep` over explicitly named files or the glob `m17src/*.py`) and
`git log --oneline -8`, `git show --stat 7f96762`, `git show 7f96762 -- m17src`. No other
commands, no writes, no recursive searches, no Python.

**Forbidden reads:** anything under `results/frozen_eval/untouched-*`, any reserved qrels cache,
anything under `work/m9reserve`, six-set or LoTTE payloads, and anything under `work/` or
`results/` at all for this review. List every file you read at the end.

## Context

M17 retrains the zero query table (vocabulary extension + listwise + alias consistency) against
the frozen stella index under a 72-hour clock. Step 5 built `m17src/prepare_data.py` (ten
cached stages: pool, domain, protected, teacher, bank, v1, parity, vocab, cache, manifests). The
registry `m17/registry.json` is `DRAFT_NOT_EXECUTABLE`; `common.require_executable` makes
`train.py` and the protected screen refuse until the lock is committed.

Step 6 is the lock. `training.decision_protocol.lock` says: "This block, the vocabulary list
hash, tokenizer hash, sampler seed, candidate-cache identity and V0 export hash are committed
before the V0 read or any real development read."

**The decision under review.** The protected screen (`prepare_data.py --protected-screen`)
may only run on the clock (it materializes protected payloads in-process; `m17/LEDGER.md`). The
protected flag is part of every stage marker's identity (`prepare_data.stage_identity`), so the
screened rebuild re-runs teacher, bank, v1, parity, vocab, cache and manifests. Therefore the
vocabulary list, extended tokenizer, new rows and candidate cache that actually train are
produced ON the clock, after the screen, and the unscreened full build (running now, ~1.5 h)
can only price them. I split the lock into two dated halves, both in `m17src/lock.py`:

- `--phase pre` (pre-clock): binds the decision-protocol/arms/contrasts hashes, the builder
  recipe identity, seeds/steps/batch/snapshot window, the fixed manifests, the invariant build
  inputs, the unscreened build's identities as `pre_screen_build` (disclosed as a price), the
  measured allocation (phase ceiling = two full builds rounded up; the freed hours are left
  unallocated and recorded), and sets status `EXECUTABLE`.
- `--phase executed` (on the clock, after the screened rebuild, BEFORE any read): requires the
  screen receipt, unchanged protocol/recipe/invariant inputs, a build stamped `EXECUTABLE`;
  records the executed identities and which ones the screen moved; exports V0 through the five
  gates and records its hash with `read: false`; sets `LOCKED_EXECUTABLE`.

I believe this satisfies the registered lock text (hashes committed before any read) and is the
only coherent reading given the stage identities. Break that belief if you can.

The same commit (`7f96762`) makes `prepare_data.TextVectorCache.get` encode misses in chunks of
25,000 and flush atomically after each, because a CUDA illegal-memory-access at 120k of 550k
texts lost every encoded vector (nothing was written until the end).

## Files to read

1. `m17src/lock.py`, `m17src/test_lock.py` (new).
2. `m17src/common.py` (`require_executable`, `EXECUTABLE_STATUSES`, `admit_read/write`),
   `m17src/prepare_data.py` (`stage_identity`, `recipe_identity`, `TextVectorCache`,
   `stage_protected`, `stage_manifests`, `build`), `m17src/train.py` (`_load_prepared`,
   `_check_locked_config`, `_check_locked_recipe`, `extend_model`, `load_warm_start`),
   `m17src/export.py` (`effective_rows`, `build_bundle`, `run_gates`, `IDENTITY_FIELDS`).
3. `m17/registry.json`: `status`, `training.decision_protocol`, `training.untrained_vocab_export_v0`,
   `training.cache_identity`, `allocation_hours`, `budget_policy`, `execution_entry_missing`.
4. `m17/STATUS.md`, `m17/LEDGER.md` (the step-3 and step-5 sections), `m17/CODEMAP.md` (the
   `lock.py` row).

## What to break

1. **Fail-open paths in `lock.py`.** Any input that lets the executed half stamp
   `LOCKED_EXECUTABLE` without the screen having actually run over THIS build, or with a cache,
   tokenizer or vocabulary that is not the one `train.py` will load. Compare `EXECUTED_HASHES`
   and `INVARIANT_HASHES` with what `train._load_prepared` verifies and what `stage_manifests`
   records: name any artifact that trains but is not bound (e.g. `candidates.npz` arrays,
   `heldout_idx`, `positives`, `bank.npy` alignment, base-form tokenizer, `student_ids`).
2. **The sequencing decision.** Is there a reading of the registry, LEDGER or PLANNING under
   which committing the executed identities on the clock (after the screen, before any read)
   violates a registered rule? Does anything the pre half does count as a development read?
   Does the V0 export itself (folded int8 table, five gates on synthetic fixtures) read a
   development component, the panel or a protected surface? Is defining "clock start" as the
   first `--protected-screen` invocation consistent with `allocation_hours` and `budget_policy`?
3. **The allocation edit.** `lock_pre` rewrites the preparation row's `hours` to
   `min(placeholder, ceil(2 × measured))` and records the rest as unallocated. Is that an
   unauthorized protocol change, or a permitted "measured preparation costs ... replacing the
   placeholder hours before lock" (`execution_entry_missing`)? Is the ×2 ceiling defensible
   given the on-clock rebuild has warm teacher/document caches but re-runs every post-screen stage?
4. **V0 correctness.** `export_v0` concatenates `export.effective_rows(warm_start.npz)` with
   `new_rows.npy` and quantizes with the ext tokenizer. Is that the registered V0 ("unchanged
   old rows, count-weighted new rows, folded and quantized exactly like a release, no optimizer
   step")? Does `train.extend_model` produce the same effective rows (new-row scalars at 1.0)?
   Any row/tokenizer misalignment `build_bundle` would not catch?
5. **The chunked cache flush.** Does chunking change any cached vector versus the unchunked path
   (order, normalization, fp16 rounding, duplicates within/between chunks)? Can a crash between
   `np.save(tmp)`/`replace` and `write_json(index)` leave an inconsistent cache that the
   constructor then refuses, costing everything anyway? Does the resume path re-encode correctly?
6. **Tests.** Which of the above is untested, and what is the smallest test that would catch it?

Output: a numbered list of findings, each with severity (P1 = fail-open, wrong identity or a
protocol violation; P2 = real defect with a plausible careless-researcher trigger; P3 = other),
file:line, the concrete input that goes wrong, and the smallest fix. Then a one-paragraph verdict
on the sequencing decision. Then the file list. Do not pad; if a section has nothing, say so.
