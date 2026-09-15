#!/usr/bin/env python3
"""Independent timer check of the monitor and final evaluation. Never restarts scoring."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'work/m13-monitor/timer-health.json'


def process(pid, expected):
    p = Path('/proc') / str(int(pid))
    args = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if expected not in args: raise ValueError('PID belongs to another command')
    fields = (p / 'stat').read_text().split(') ', 1)[1].split()
    if fields[0] == 'Z': raise ValueError('process is a zombie')
    return {'identity': f'{pid}:{fields[19]}', 'cpu_ticks': int(fields[11])+int(fields[12])}


def judge(now, receipt, child, previous, log_mtime):
    if receipt['status'] != 'RUNNING':
        return receipt['status'], 'Evaluation finished; check its result and remaining tasks.', now
    progress = previous.get('last_progress_at', now)
    if child != previous.get('child') or log_mtime > previous.get('log_mtime', 0):
        progress = now
    if now - receipt['updated_at'] > 180:
        return 'ALERT', 'Evaluation supervisor heartbeat is stale.', progress
    if now - progress > 900:
        return 'ALERT', 'No child CPU or log progress for over 15 minutes.', progress
    if now - receipt['started_at'] > 14*3600:
        return 'ALERT', 'Evaluation exceeds 14 hours; inspect it, do not restart it.', progress
    return 'RUNNING', 'Evaluation child is live; monitoring independently.', progress


def main():
    now = time.time()
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    event = {'name':'independent timer watchdog', 'timestamp':now}
    try:
        health = ROOT / 'work/m13-monitor/health.json'
        active = subprocess.run(['systemctl','--user','is-active','--quiet','m13-monitor.service'],
                                timeout=5).returncode == 0
        if not active or not health.exists() or now-health.stat().st_mtime > 900:
            subprocess.run(['systemctl','--user','restart','m13-monitor.service'],
                           timeout=10, check=True)
            event['restarted_primary_monitor'] = True
        receipt = json.loads((ROOT / 'work/m13-final/execution.json').read_text())
        child = None
        if receipt['status'] == 'RUNNING':
            process(receipt['pid'], 'm13_final_cpu.py')
            child = process(receipt['child_pid'], 'm13src/score13.py')
        log_mtime = (ROOT / 'logs/m13-final-six.log').stat().st_mtime
        status, detail, progress = judge(now,receipt,child,previous,log_mtime)
        event.update(state=status,detail=detail,child=child,last_progress_at=progress,
                     log_mtime=log_mtime)
    except Exception as exc:
        event.update(state='ALERT',detail=f'Watchdog check failed: {type(exc).__name__}: {exc}')
    changed = any(event.get(k) != previous.get(k) for k in ('state','detail'))
    if changed or event.get('restarted_primary_monitor'):
        try:
            subprocess.run(['/usr/bin/python3',str(ROOT/'scripts/m13_windows_alert.py')],
                           input=json.dumps(event),text=True,timeout=15,check=True,
                           capture_output=True)
        except Exception as exc: event['notification_error'] = str(exc)
        with (STATE.parent/'timer-alerts.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
    tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(event,indent=2)+'\n'); tmp.replace(STATE)
    print(json.dumps(event),flush=True)


if __name__ == '__main__': main()
