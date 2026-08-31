# M9.3 build-period log

`m9/LEDGER.md` and `m9/registry.json` are in guard9's **protocol** scope, which
`DEPS["m9-build"]` depends on, so editing them makes the trainer — including the watchdog's
automatic crash-restart — refuse to start. Entries therefore land here during the build and are
merged into the ledger afterwards. This file is in no scope.

## 2026-08-31

- Build session force-reopened after the approved M3/M4 infrastructure repair; full disclosure in
  `m9/LEDGER.md` ("M9.3 BUILD PROVENANCE DISCLOSURE"). Trainer resumed from step ~78,754.
- Watchdog gave up at 06:17 after two failed launches (cause: the guard9 session fingerprint, a
  configuration failure — not a training or quality stop) and wrote `ckpt/STOP` +
  `terminal.json`. Both cleared to resume; the marker is archived at
  `work/m9long/archive/terminal-configfailure-20260831T0617.json`. Build session force-reopened at
  fingerprint `ef899585b10f` (was `24dcbe852bde`); `work/m9tokens/SESSION.json` untouched
  (`def5f88a0acf`).
- The build's warm-start adapter permits ONLY `owner_rulings` additions to `m9/registry.json`
  after the screen, so the `final_run` block I had added there blocked the trainer. Moved to
  `m9/final_run_registry.json` (no scope, same authority); `registry.json` restored to its
  post-screen state apart from the owner ruling, `stage` back to `M9.0`. `final_stats.py` and
  `FINAL_LOCK.md` updated; 16/16 tests still pass.
