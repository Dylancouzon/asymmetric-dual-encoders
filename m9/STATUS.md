# M9 status — build RUNNING (launched 2026-08-30 23:14)

Trainer pid 225232, watchdog + `guardian.sh` + `sentinel.sh` supervising. Recipe `M92_LOCK.md`,
unchanged. Absolute deadline persisted in `work/m9long/deadline.json` — never reset by a restart.

Live state: `m9/RUN_STATUS.md`, refreshed every 30 min and pushed to branch `m9-status`.

## OPEN WORK — carry this forward across any context clear

| # | item | state |
|---|---|---|
| 1 | Codex M3/M4/M5 trainer repair (deadline-truncates-cooldown, corrupt-`last.pt` fallback, uncontained eval exceptions) | patch written, **Codex-reviewed before applying**; needs one trainer restart at a checkpoint boundary |
| 2 | M9.4 final-run scorer (`final_run.py`): bridge as phase 1, C1/C2, B=10,000 bootstrap + B=100,000 sign-flip, Holm at α=0.025 | **not written.** Must be Codex-reviewed before it touches the six — M7 precedent: a rounded CI endpoint was caught there |
| 3 | LoTTE read #1 batch manifest (fusion weight + non-inferiority veto) | not built; one atomic batch, no second chance |
| 4 | Cost rows (query asset / doc index / hydration + doc-encode, ONNX batch-1 latency protocol) | not measured |
| 5 | Report artifact | after M9.4 |

Owner rulings in force: seed replicas **waived** (report must state variability unmeasured);
GO at the locked 168 h horizon.

## Supervision — three layers, because silence is not health

- `watchdog.py` restarts the trainer, enforces staleness/deadline, pushes status.
- `guardian.sh` restarts the **watchdog** (it was an unsupervised single point of failure).
  Safe by flock: a relaunch while one lives exits immediately. Verified against SIGKILL.
- `sentinel.sh` is the inverse alarm: it is SILENT when healthy and speaks only on a stall,
  a missing process, a throughput collapse, low disk, or the run ending. A log-tailing monitor
  cannot detect "nothing was ever written again"; this can.

**Never kill by `pgrep` pattern** — a pattern matched the operator's own shell during the
2026-08-31 repair and killed it. Use exact PIDs from `ps -eo pid,args`.

## Stop, cool down, restart

1. Stop safely: `touch work/m9long/ckpt/STOP`. Keep the watchdog running;
   it supervises until `terminal.json` confirms the trainer exited.
2. Cool down: after that terminal marker appears, run
   `setsid nohup .venv/bin/python m9src/watchdog.py --cooldown --hours 4 >> logs/m9_watchdog.log 2>&1 &`.
   The cooldown command safely consumes the acknowledged STOP and terminal markers, resumes
   `last.pt` in decay, and supervises it through `cooldown complete`.
3. Restart after a crash: if the watchdog is alive, do nothing; it restarts the trainer exactly.
   If the watchdog died, rerun the original watchdog launch command. It reuses `deadline.json`,
   attaches to a live trainer or resumes `last.pt`, and never resets the seven-day horizon.

Never run withdrawn `m9s1b` or `m9s1c`. Never force/re-open `work/m9tokens/SESSION.json`. Never
delete screen artifacts or their run tokens. Never use the old destructive screen reset or
`run_m9_stage.sh`. Do not access the six, reserved four, LoTTE shadow, or confirmatory data.

Pointers: `M92_LOCK.md` recipe · `RUN_STATUS.md` live state · `LEDGER.md` protocol · `CODEMAP.md`
implementation.
