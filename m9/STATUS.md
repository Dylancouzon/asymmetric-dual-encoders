# M9 status — build ready; not launched

No trainer or watchdog is running and nothing has trained. Screen, targets, corpora, manifest,
config, and resume prerequisites are complete. The locked build is stella-400M-v5 ×
bge-small-en-v1.5 × bare prompt × 5/5/90; see `M92_LOCK.md`.

Before launch, the owner must commit and push this pre-launch repair (this session was forbidden to
do so), then require `.venv/bin/python m9src/guard9.py` to print `problems: []`. Launch only then:

```bash
setsid nohup .venv/bin/python m9src/watchdog.py --hours 168 --eval-stale 18000 \
  --ckpt-stale 7200 > logs/m9_watchdog.log 2>&1 &
```

Monitor from a fresh session:

```bash
pgrep -af "watchdog[.]py"
pgrep -af "longrun[.]py (train|decay)"
.venv/bin/python m9src/longrun.py status
tail -n 200 logs/m9_watchdog.log logs/m9_build.log
rg -n "Traceback|Error|FAILED|OOM|Killed|assert" logs/m9_watchdog.log logs/m9_build.log
```

`m9/RUN_STATUS.md` refreshes every 30 minutes. Expect exactly one watchdog and one trainer, then
step-0 eval, step 500, checkpoints every 3,000 steps, and evals every 15,000 steps.

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
