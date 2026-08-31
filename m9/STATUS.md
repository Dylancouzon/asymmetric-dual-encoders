# M9 status — build RUNNING (resumed 2026-08-31 06:22 after an approved repair)

Trainer resumed at step 78,000; watchdog + `guardian.sh` + `sentinel.sh` supervising. Recipe
`M92_LOCK.md` unchanged. **This build's provenance spans two build lock states** — see the
provenance disclosure in `m9/LEDGER.md`; never describe it as a single-lock run.

**PROTOCOL FREEZE while the build runs:** do NOT write `m9/LEDGER.md`, `m9/registry.json`, or any
guard9 protocol-scope file. They are in `DEPS["m9-build"]`, so an edit makes the trainer refuse to
start — including the watchdog's automatic crash-restart, which silently disables unattended
recovery. Build-period notes go to `m9/BUILD_LOG.md` (no scope) and merge afterwards.

Live state: `m9/RUN_STATUS.md`, pushed to branch `m9-status` every 30 min.

## OPEN WORK

| # | item | state |
|---|---|---|
| 1 | `m9src/final9.py` — the six-set transaction: access state machine, bridge as phase 1, then `final_stats.run_contrasts` | **not written.** Spec is complete and reviewed in `m9/FINAL_LOCK.md` + `m9/final_run_registry.json`; statistics done and tested (`final_stats.py`, 16/16). Remaining work is GPU-bound encoding, untestable until the build frees the card. Reuse `m7src/final_run.py` (state machine, `verify_and_load`, `score_set`) and `m7src/boot.py` |
| 2 | LoTTE read #1 batch | amended + pre-registered (`m9/LOTTE_LOCK.md`). Needs the batch code and, after training, a second commit pinning the final checkpoint's hash **before** the evaluator touches LoTTE |
| 3 | Cost rows on THIS box | deferred deliberately: batch-1 CPU latency measured while training is contaminated. Mac numbers exist (`m9/EDGE_COST_MAC.md`) |
| 4 | dev→six retention calibration | optional forecast; LoTTE read #1 supersedes most of its value |
| 5 | Report artifact | after M9.4 |
| 6 | **Dylan: ratify** the final-lock amendment (`final_run_registry.json.ratified_by_owner: false`) and the LoTTE amendment | pending his return |

## Curve so far (SCREEN-3, dev; ceiling 0.68223)

0.34619 → 0.46710 → 0.50148 → 0.51802 → 0.52510 → 0.53334 → 0.53775 (step 90,000, 0.737B tokens).
`m9src/dose_curve.py` brackets the endpoint: floor 0.782 / **saturating 0.843** / loglinear 0.961
retention. SCREEN-3 is NOT avg-6 — the aim needs 89.7% there — and the fit sees only stable-LR
checkpoints, so it cannot see the final anneal.

## Supervision — four layers, because silence is not health

`watchdog.py` → trainer · `guardian.sh` → watchdog · `sentinel.sh` → alarms on stalls, dead
processes, throughput collapse, low disk and run end (SILENT when healthy; one daily proof-of-life
line) · `m9-status` branch → Dylan.

Helper tools MUST run via `m9src/sacrificial.sh`: the trainer's `oom_score` is 1073 (18.5 GB of
reclaimable mmap pages; real anon footprint 2.1 GB), so it is the kernel's top OOM victim and
helpers must volunteer to die first.

**Never kill by `pgrep` pattern** — one matched the operator's own shell on 2026-08-31. Use exact
PIDs from `ps -eo pid,args`.

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
