# M9.3 build — live status

_Updated 2026-08-31 18:23:22 by `m9src/watchdog.py`._

**step 206,004** · **1.686 B tokens** (14.8% of the cap) · 24,100 tok/s · phase **stable** · heartbeat 10s old

**Best SCREEN-3 0.54938 — retention 0.805** of the 0.68223 teacher ceiling.

| step | B tokens | SCREEN-3 | retention |
|---|---|---|---|
| 0 | 0.000 | 0.34619 | 0.5074 |
| 15,000 | 0.123 | 0.46710 | 0.6847 |
| 30,000 | 0.246 | 0.50148 | 0.7351 |
| 45,000 | 0.368 | 0.51802 | 0.7593 |
| 60,000 | 0.491 | 0.52510 | 0.7697 |
| 75,000 | 0.614 | 0.53334 | 0.7818 |
| 90,000 | 0.737 | 0.53775 | 0.7882 |
| 105,000 | 0.860 | 0.53903 | 0.7901 |
| 120,000 | 0.982 | 0.54421 | 0.7977 |
| 135,000 | 1.105 | 0.54462 | 0.7983 |
| 150,000 | 1.228 | 0.54487 | 0.7987 |
| 165,000 | 1.351 | 0.54532 | 0.7993 |
| 180,000 | 1.473 | 0.54676 | 0.8014 |
| 195,000 | 1.596 | 0.54938 | 0.8053 |

## Incidents

| when | event | detail |
|---|---|---|
| 2026-08-31T00:49:55 | watchdog_start | period 60s, mode train, absolute deadline 1788750840.314 (reused) |
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
