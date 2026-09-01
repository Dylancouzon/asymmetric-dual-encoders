# M9.3 build — live status

_Updated 2026-09-01 08:54:33 by `m9src/watchdog.py`._

state **eval** · heartbeat 1s old

**Best SCREEN-3 0.55570 — retention 0.815** of the 0.68223 teacher ceiling.

| step | B tokens | SCREEN-3 | retention |
|---|---|---|---|
| 135,000 | 1.105 | 0.54462 | 0.7983 |
| 150,000 | 1.228 | 0.54487 | 0.7987 |
| 165,000 | 1.351 | 0.54532 | 0.7993 |
| 180,000 | 1.473 | 0.54676 | 0.8014 |
| 195,000 | 1.596 | 0.54938 | 0.8053 |
| 210,000 | 1.719 | 0.54830 | 0.8037 |
| 225,000 | 1.842 | 0.54860 | 0.8041 |
| 240,000 | 1.965 | 0.55216 | 0.8093 |
| 255,000 | 2.087 | 0.55109 | 0.8078 |
| 270,000 | 2.210 | 0.55188 | 0.8089 |
| 285,000 | 2.333 | 0.55424 | 0.8124 |
| 300,000 | 2.456 | 0.55123 | 0.808 |
| 315,000 | 2.579 | 0.55570 | 0.8145 |
| 330,000 | 2.702 | 0.55225 | 0.8095 |
| 345,000 | 2.824 | 0.55345 | 0.8112 |

## Incidents

| when | event | detail |
|---|---|---|
| 2026-08-31T06:14:05 | watchdog_start | period 60s, mode train, absolute deadline 1788750840.314 (reused) |
| 2026-08-31T06:14:25 | launch | initial trainer start; pids [] |
| 2026-08-31T06:15:46 | restart_failed | dead; nothing came up. See logs/m9_build.log. 1 consecutive. |
| 2026-08-31T06:17:09 | restart_failed | dead; nothing came up. See logs/m9_build.log. 2 consecutive. |
| 2026-08-31T06:17:09 | give_up_stop_requested | two consecutive launches produced no process; configuration failure; wrote /home/dylan/asy |
| 2026-08-31T06:17:09 | giving_up | two consecutive launches produced no process; configuration failure; wrote STOP; remaining |
| 2026-08-31T06:17:11 | watchdog_stop | 2 restarts |
| 2026-08-31T06:18:19 | watchdog_start | period 60s, mode train, absolute deadline 1788750840.314 (reused) |
| 2026-08-31T06:18:39 | launch | initial trainer start; pids [] |
| 2026-08-31T06:20:00 | restart_failed | dead; nothing came up. See logs/m9_build.log. 1 consecutive. |
| 2026-08-31T06:21:22 | watchdog_start | period 60s, mode train, absolute deadline 1788750840.314 (reused) |
| 2026-09-01T06:22:21 | daily | 17 evals; tokens 0.74B -> 2.70B; SCREEN-3 0.53775 -> 0.55225 (+0.01451); best 0.55570 |

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
