# Brief: P1-only re-check of the M17 step-5 fixes (the single re-check; review loop closes here)

You are re-checking code, read-only. You may run only read-only viewing commands (`cat`,
`sed -n`, `nl -ba`, `head`, `tail`, `wc`, `ls` on a named path, non-recursive `grep` over
explicitly named files or the glob `m17src/*.py`), `git log --oneline -14`,
`git show --stat <sha>` and `git diff 6fc3b6a 850d169 -- m17src m17/REVIEW.md m17/CODEMAP.md`.
Exactly one test command from the repository root is allowed:
`.venv/bin/python -m pytest -q -ra m17src` (if your sandbox has no writable temp directory, say
so and skip it). No other commands, no writes, no recursive searches.

**Forbidden reads:** anything under `results/frozen_eval/untouched-*`, any reserved qrels cache,
anything under `work/m9reserve`, six-set or LoTTE payloads, `work/m17/panel/`, `work/m17/alias/`,
`work/train/stores/`. Under `work/` you may open only
`work/m17/prepared/s2000/build_record.json`, `work/m17/prepared/s2000/ext/prepared.json`,
`work/m17/prepared/s2000/shared/cache.json`, `work/m17/runs/smoke-resume-VL-A-2/run_record.json`.
List every file you read at the end.

## Context

Two independent reviews of the step-5 prepared-data builder: Codex Astra (21 findings, fixed in
`c76de64`, `22e296d`, `9d8e3e4`) then Codex Sol on the fixed code (11 findings, fixed in
`9a90ec7`, `cee9d84`, `850d169`). Owner rulings A4, A5 and A6 are in `m17/LEDGER.md` and the
registry (`accepted_plan_revision_a4/a5/a6`). Dispositions: `m17/REVIEW.md`, sections "Codex
Astra step-5 review" and "Codex Sol step-5 review". The owner's standing instruction: do not
over-engineer; the smallest fix per real bug; adversarial-only findings are dropped.

## Files to read

1. `m17/REVIEW.md` (the two step-5 sections), `m17/registry.json` (`data.cap_fill_priority`,
   `data.judging_amendment_a6`, `training.labeled_positive_budget_share`,
   `training.candidate_construction`, `training.cache_identity`, `execution_entry_missing`).
2. `m17src/prepare_data.py`, `m17src/cache.py`, `m17src/train.py`, `m17src/vocab.py`,
   `m17src/support_manifest.py`, `m17src/common.py`, `m17src/test_prepare_data.py`,
   `m17src/test_cache.py`, `m17src/test_train.py`.
3. `results/m17_prepare_timing.json` (the three-size time and memory result).

## What to do

Only P1s. For each of the 14 P1s in the two tables (Astra P1-1 … P1-10, Sol P1-1 … P1-4) that
is marked **Fixed**, confirm the fix is real: name the line, state the input that used to pass
wrongly, say whether it is now refused and whether a test covers it. If a fix is incomplete, opens
a new fail-open path, or breaks the happy path (the three rebuilt prepared directories and the
100-step resume smoke are the evidence it still runs), report it as a **remaining P1** with the
exact smallest fix. Also check one thing neither reviewer could: the A5 cap-fill code
(`cap_fill_priority`) — does it really take every distinct general query first, then ALL distinct
alias pairs, then coverage to exactly `training_query_cap`, deterministically, and does the
after-screen refill preserve that order?

Do not report P2/P3, style, or anything already dispositioned. Output: a table of the 14 P1s
with "confirmed fixed" / "remaining", then any remaining P1s in detail (at most one paragraph
each with file:line and the smallest fix), then the file list. If nothing remains, say so plainly.
