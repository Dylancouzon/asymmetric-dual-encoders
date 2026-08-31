# M9 status — build ready; not launched

No trainer or watchdog is running and nothing has trained. Screen, targets, corpora, manifest,
config, and resume prerequisites are complete. The locked build is stella-400M-v5 ×
bge-small-en-v1.5 × bare prompt × 5/5/90; see `M92_LOCK.md`.

The pre-launch repair is committed and pushed; `guard9.py` prints `problems: []` and
`test_resume.py` passes bitwise. **Everything is verified and the only thing missing is Dylan's
GO.** Re-check the guard, then launch:

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

`m9/RUN_STATUS.md` refreshes every 30 minutes on branch `m9-status`. Expect exactly one watchdog
and one trainer, then step-0 eval, step 500, checkpoints every 3,000 steps (~22 min), and evals
every 15,000 steps (~1.8 h; 3.6 h at the 50% throughput floor, against a 5 h staleness limit).

**Verify the launch before walking away** — these paths were reviewed sixteen times but the
watchdog supervising a live trainer, and its push to `m9-status`, had never actually run:

```bash
tail -f logs/m9_build.log          # want: warm-start adapter line, EVAL step 0, step 500
cat work/m9long/heartbeat.json     # want: state train, advancing step
git ls-remote origin m9-status     # want: the sha to move within ~35 min
```

If any of that fails, stop it (below) rather than leave it running.

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
