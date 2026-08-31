# M9.3 build-period log

`m9/LEDGER.md` and `m9/registry.json` are in guard9's **protocol** scope, which
`DEPS["m9-build"]` depends on, so editing them makes the trainer — including the watchdog's
automatic crash-restart — refuse to start. Entries therefore land here during the build and are
merged into the ledger afterwards. This file is in no scope.

## 2026-08-31

- Build session force-reopened after the approved M3/M4 infrastructure repair; full disclosure in
  `m9/LEDGER.md` ("M9.3 BUILD PROVENANCE DISCLOSURE"). Trainer resumed from step ~78,754.
