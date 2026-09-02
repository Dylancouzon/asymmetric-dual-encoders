# M9 status — CLOSED as a measurement (2026-09-01, Dylan); six-set close-out pending

Build complete at 3.743B tokens (plateau rule → cooldown); candidate frozen (`m9/FREEZE.json`,
sha `9d631b2c…`). SCREEN-3 0.5606 = 82.2% retention, **93.8% on NQ vs 50–71% on CQADupStack** —
a coverage failure. Both bars need ≥87.8% on avg-6; the candidate is not released. **Read
`m9/FINDINGS.md`**, then `m9/EXPLORED.md`. M10 is the retry (`instructions-m10.md`).

## Close-out work (box required)

| # | item | state |
|---|---|---|
| 1 | Dylan ratifies the final-lock amendment (`final_run_registry.json.ratified_by_owner`) | **pending** — nothing below runs without it |
| 2 | `m9src/final9.py` step-4 scoring path (reuse `m7src/final_run.py` `verify_and_load`/`score_set`), then a whole-module adversarial review | **not written**; two access-control passes done |
| 3 | The registered six-set transaction on the frozen candidate (`m9/FINAL_LOCK.md`), **amended before execution to six-only: the `if C1 then execute` reserved conditional is struck**, so M9 cannot spend the reserved access M10 needs | to run **after M10.2's lock is pushed** (no M9 six-set output may exist while an M10 design decision is open); the amendment is ratified with item 1; its rows are a forecasting calibration and a whitepaper frontier point |
| 4 | Merge `m9/BUILD_LOG.md` into `m9/LEDGER.md` **after** #3 (protocol scope stays frozen until the transaction is done) | after #3 |
| 5 | `freeze.assert_releasable` record gap | moot — the candidate is not released; document only |
| 6 | Cost rows on the box | moved to M11 (`m9/EDGE_COST_MAC.md` has the Mac numbers) |

## Decisions taken at close-out (2026-09-01, planning session, delegated authority; Dylan to confirm)

- **LoTTE read #1 (`m9/LOTTE_LOCK.md`) is WITHDRAWN unexecuted.** Its veto (m9s6 vs m9s1) decides
  nothing now, and executing it would spend the only fresh out-of-domain surface on a candidate
  that misses. LoTTE-clean passes to M10 unread. No LoTTE-derived output exists.
- **The six-set transaction stays registered and runs after M10's recipe lock** (item 3). The
  frozen candidate misses on dev; the six-set rows are still the only calibration between the dev
  surfaces and avg-6, and the whitepaper needs the 35M/3.7B-token frontier point — but they may
  not inform an open M10 decision, so the lock comes first.
- `m9-status` (the watchdog's status-only orphan branch) has done its job; deleting it on origin
  is Dylan's call.

## Never do

Never run withdrawn `m9s1b`/`m9s1c`; never force/re-open `work/m9tokens/SESSION.json`; never
delete screen artifacts or their tokens; never edit a `guard9` protocol-scope file
(`m9/LEDGER.md`, `m9/registry.json`, `instructions-m9.md`, `run_m9_stage.sh`, …) before item 3
runs; never access the six, reserved four, LoTTE or confirmatory data outside `final9.py`.
Helper tools on the box run via `m9src/sacrificial.sh`; kill by exact PID, never by `pgrep`.

Pointers: `M92_LOCK.md` recipe · `RESULTS.md` runs · `RUN_STATUS.md` final watchdog state ·
`BUILD_LOG.md` build-period notes · `LEDGER.md` protocol · `CODEMAP.md` implementation.
