# M9 status — launch prerequisites running

Branch `m9-work`. The six and reserved four remain untouched; LoTTE unread;
`results/perquery.json` untouched.

## Decision

- **Screen COMPLETE and eligible:** `results/m9_screen_decisions.json`.
- **Recipe LOCKED:** `m9/M92_LOCK.md` → stella-400M × bge-small × prompt (b) × 5/5/90.
- `m9s6` selected `query-only`; Dylan's owner ruling authorizes 5/5/90. Machine gate:
  `m9/registry.json` → `owner_rulings.m9s6_mix_override`.
- `m9s1b` and `m9s1c` are **WITHDRAWN: never run them**.

The prior destructive screen reset and `run_m9_stage.sh` chain are not launch instructions.
Starting a new screen would void the current lock.

## Handoff / launch

Remaining, in order:

1. The active `targets` → `manifest` → `verify` chain is running now; do not start a duplicate.
2. After it succeeds, generate the config and prove resume equivalence:

```bash
.venv/bin/python m9src/make_config.py && \
.venv/bin/python m9src/test_resume.py
```

3. Only after both pass, launch the watchdog:

```bash
setsid nohup .venv/bin/python m9src/watchdog.py --hours 168 > logs/m9_watchdog.log 2>&1 &
```

4. Verify the launch. Expect exactly one watchdog and trainer, `EVAL step 0` within ~30 minutes,
the first `step 500` line, no failure signature, and `origin/m9-status` to move within ~35 minutes.

```bash
pgrep -af "watchdog[.]py"
pgrep -af "longrun[.]py train"
tail -n 200 logs/m9_watchdog.log logs/m9_build.log
rg -n "EVAL step 0|step 500|Traceback|Error|FAILED|OOM|Killed|assert" \
  logs/m9_watchdog.log logs/m9_build.log
git ls-remote origin m9-status
```

`m9/RUN_STATUS.md` is republished on `m9-status` every 30 minutes.

## Pointers

| file | contract |
|---|---|
| `m9/M92_LOCK.md` | locked M9.2 recipe |
| `m9/registry.json` | machine constants and owner ruling |
| `m9/LEDGER.md` | protocol and rulings |
| `m9/RESULTS.md` | run record |
| `m9/RUN_STATUS.md` | live build status |
| `m9/CODEMAP.md` | implementation map and pitfalls |
