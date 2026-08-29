# M8 worklist (rewritten 2026-08-29, after the E14-HEAD review)

**Read `m8/STATUS.md` first, then `m8/LEDGER.md` (the protocol authority), then
`m8/registry.json` (the executable half). This file is the remaining work only — everything
finished lives in LEDGER/RESULTS/EXPLORED/CODEMAP and is not repeated here.**

**Nothing is blocked on Dylan.**

## In order

1. **`E14-HEAD` — DONE, 2026-08-29. NO SURVIVOR.** Both heads harm (dense −0.0244 / −0.0293
   against a +0.0040 bar); the patch stack measured as a null; the mechanism control shows the
   hypothesis was real and the instrument wrong. Results in `m8/RESULTS.md` and STATUS; artifact
   `results/m8_e14_head.json`. **Nothing further to run here.** `E14-LORA` remains registered and
   REFUSED with a TBD bar, and this result is an input to that bar, not a licence to write it.

2. **`E10-REMEDY` — code is written and registered, nothing has run.**
   `protected_filter.py remedy` does the per-item removal and the zero-tolerance re-screen;
   `freeze_lotte.py pin` hashes the survivors and refuses any slice the remedy artifact does not
   list. Then: add the pinned partition to `paths_guard`, feed the surviving queries into
   `protected_filter`'s index, and **regenerate the fit list** (the current 337,981 predates them).
   **Smoke it with `--limit` first** — it streams 5.2M documents against a 134K-document index.
   **Review the remediation before trusting it**: a shadow that is quietly still contaminated is
   worse than no shadow, because it gives false reassurance right before the one-shot access.

3. **`D-FINEWEB` (E13).** Bar unfrozen, so the guard refuses it. The more interesting data lever
   now that B3 closed the Phase-A pair-count route, because this is a **B-side** one — it feeds the
   pseudo-query pool, and ~12% of the table's 30,522 rows are never updated in training.
   Two strings attached: E13's standing ruling is *measure first, ship-decide later*, and **"a
   wrapper tag — including our own — is not a licence"**; and this arm **triggers the deferred
   patent question** (a general web crawl is the one planned source that could contain patent
   text), so settle that before web-crawl data enters training.

4. **`m8src/freeze.py` and `final_run.py`, then their suites** (§4.4 gap list). The largest
   engineering left, weeks from being needed. The tests cannot precede the modules.

5. **`E14-LORA`** — registered and refused, TBD bar by design. Only after `E14-HEAD` reports, and
   it needs a fresh ruling: the bill (doubled 10.12M pre-encode, hours of pool re-encoding per arm,
   the stella derived-weights licence check, a forced C2 redefinition) is **not** covered by
   "measure it small first".

## Session rules (they have earned their keep every time)

Sonnet subagents for mechanical work; Codex and Fable adversarially for anything about to become
expensive or permanent. Commit and push after every completed item. **Smoke every new path** and
**check the rate**, not just that something is running. `setsid nohup` for anything long — harness
interrupts killed plain background waiters on 2026-08-29 while the detached jobs survived.
**Run `git status m7/ results/` after anything out of `m7src/`**: `sweep.one` appends a row to
`m7/RESULTS.md` every run, and `m8src/reclaim_results_rows.sh "<heading>"` is the fix.
