# M8 worklist (rewritten 2026-08-29, after the E14-HEAD review)

**Read `m8/STATUS.md` first, then `m8/LEDGER.md` (the protocol authority), then
`m8/registry.json` (the executable half). This file is the remaining work only — everything
finished lives in LEDGER/RESULTS/EXPLORED/CODEMAP and is not repeated here.**

**Nothing is blocked on Dylan.**

## In order

1. **`E14-HEAD` — the milestone's main bet. The registration is now correct; the code is not
   written.** A Codex pass found three BLOCKERs in the design *before* any arm ran and the row was
   amended (LEDGER §15, 2026-08-29). **Implement from the registry row and that entry — NOT from
   `research/m8-planning/e14-head-design-2026-08-29.md`, which is the reviewed brief and is wrong
   in three specific places.** What already exists: `m8src/e14_head.py` holds both heads
   (`lin` primary, `mlp` its nonlinearity control), the verbatim `infonce` copy with the
   false-negative mask kept in raw teacher space, and a self-test that is bit-identical to
   `m7src`'s loss on all six branch cases.
   What is left, in dependency order:
   - the per-arm training driver: rebind `train.infonce` and `torch.optim.Adam` in a subprocess,
     persist the head with its sha256 provenance bindings;
   - the **`R0N`** comparator arms (head frozen at identity, 3 seeds) — these are also the
     end-to-end null on the whole patch stack, so run them first and check they land within the
     floor of the existing R0 arms;
   - the **dev-blind** lr ladder on tuning seed 3, plus the step-adequacy continuation to 5,000
     steps with its pre-registered plateau rule;
   - the streamed (lazy slice-transforming) scoring path — a materializing patch needs ~21.4 GB on
     HotpotQA and does not fit;
   - the mechanism control (headed documents against frozen *teacher* queries).
   **Re-brief a review on the implementation before the campaign runs.** The design review paid
   for itself; the previous session's B3 review did too.

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
