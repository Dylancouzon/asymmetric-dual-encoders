# M13 job monitoring

`scripts/m13_monitor.py` checks the explicit jobs in `m13/monitor_config.json` every
five minutes. It reads receipts, controller process identities and log activity. Failures,
dead controllers, missing evidence and 15-minute inactivity are visible in
`work/m13-monitor/health.json`; state changes append to `alerts.jsonl` beside it.
Waiting for an active upstream job is handled separately from a stalled handoff.

The WSL user service `m13-monitor.service` runs independently of the chat and restarts
on monitor failure. Windows toast notifications announce state changes. Notification API
acceptance does not guarantee that Windows displays a banner: notification settings and
Do Not Disturb still apply. Delivery failures are recorded in monitor output.

Check the service and evidence:

```sh
systemctl --user status m13-monitor.service
cat work/m13-monitor/health.json
journalctl --user -u m13-monitor.service -n 20 --no-pager
```

Before the next long job, add its exact receipt, logs and controller PID file to the
configuration and restart the service. Confirm a fresh health timestamp and the expected
running/waiting state before leaving it unattended. Use an inactivity allowance suited to
the stage; a long encoding call may legitimately stay quiet for more than 15 minutes.

This watchdog alerts locally; it cannot wake the assistant or send messages to this chat.
It does not restart workloads, change experiments, or control cloud billing. Existing
execution controllers retain their timeout and STOP responsibilities. Keep Windows/WSL
awake: a local watchdog cannot detect its own host being asleep. A recovered failure is
recognized only when the recovery receipt binds the exact failed receipt hash; the original
failure remains in the evidence trail.

Installed 2026-09-11 at `/home/dylan/.config/systemd/user/m13-monitor.service`, with this
worktree as WorkingDirectory, `/usr/bin/python3 -u scripts/m13_monitor.py --config
m13/monitor_config.json` as the command, `Restart=on-failure`, `RestartSec=5`, and
`UMask=0077`. Enabled for the user session. Ten synthetic tests passed, independently
reviewed to GO. A deliberate SIGKILL of the monitor produced one automatic restart and
a fresh health receipt. Windows toast API calls succeeded; visual delivery was not verified.

The full build now logs at most one sixth of its rolling-checkpoint step interval: about
five minutes at the supplied planning rate, rather than total_steps/50 (about75 minutes
at the measured rate). This changes only diagnostic printing. LoTTE already flushes shard
starts and progress around each20,000 passages, so it needs no additional heartbeat.

## Independent fallback — 2026-09-15

User-requested m13-backup-watchdog.timer invokes scripts/m13_timer_watchdog.py
every5minutes, separately from the persistent primary monitor. Its check includes
the evaluation child CPU counters and log progress; it alerts on15minute stalls,
stale supervisor heartbeat, process loss and14hour runtime. It restores a stopped
or stale primary monitor but never restarts or kills the scoring process.
Timer service is bounded to45seconds. State/events: timer-health.json and
timer-alerts.jsonl under work/m13-monitor; systemd journal is an additional record.
Actual stopped-primary recovery and synthetic alarm cases passed.
Installed units are copies of m13/m13-backup-watchdog.{service,timer}.
