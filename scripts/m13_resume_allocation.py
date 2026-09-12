#!/usr/bin/env python3
"""Reconcile the one upload-only reboot interruption without resetting its allowance."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import urllib.request
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
KEYS = Path('/home/dylan/.config/runpod')
ORIGINAL = 'results/m13_build_allocation.json'
ATTEMPT = 'results/m13_cloud_build.json'
PAUSE = 'results/m13_reboot_pause.json'
OUTPUT = REPO / 'results/m13_build_resume_allocation.json'
ATTEMPT_SHA = '47715c8852ed27ed1975ba032a255abe5620a16ca61118d3ba6baeebdffefc23'
PODS = ('k3aee2m68765em', 'exulxoxelug5um', 'wnzk8eeqrrkw4m')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def live():
    headers = {'Authorization': 'Bearer ' + (KEYS/'api_key').read_text().strip(),
               'Content-Type': 'application/json', 'User-Agent': 'm13-build-allocation/1.0'}
    def get(url, body=None):
        req = urllib.request.Request(url, headers=headers,
            data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(req, timeout=30) as response: return json.load(response)
    pods = [get('https://rest.runpod.io/v1/pods/' + pod) for pod in PODS]
    balance = float(get('https://api.runpod.io/graphql',
        {'query': 'query { myself { clientBalance } }'})['data']['myself']['clientBalance'])
    return pods, balance


def remaining(original, attempt, pause, balance, continuation_hours=0):
    if not math.isfinite(balance) or not 0 <= balance <= original['account_balance_usd']:
        raise RuntimeError('Funding changed or invalid balance; refuse unexplained ledger')
    if not math.isclose(original['account_balance_usd']+original['spent_to_date_usd'],505,abs_tol=1e-6):
        raise RuntimeError('Original funding ledger changed')
    if original['max_hours'] != 144 or original['max_cost_usd'] != 239.56 or original['funding_baseline_usd'] != 505:
        raise RuntimeError('Original reviewed allowance changed')
    if (attempt['status'] != 'FAILED' or attempt['stage'] != 'finished'
        or attempt.get('pod_final_status') != 'EXITED' or not attempt.get('backup_verified')
        or pause['status'] != 'PASSED' or pause.get('training_started') is not False
        or pause.get('stage_at_request') != 'uploading-build-inputs'
        or pause.get('artifact_sha256', {}).get(ATTEMPT) != ATTEMPT_SHA
        or any(pause['pod_statuses'].get(p) != 'EXITED' for p in PODS)):
        raise RuntimeError('Not the verified upload-only interruption')
    elapsed = (pause['verified_at'] - attempt['started_at']) / 3600
    if not math.isfinite(elapsed) or not 0 < elapsed < 144:
        raise RuntimeError('Invalid consumed duration')
    if not math.isfinite(continuation_hours) or continuation_hours < 0:
        raise RuntimeError('Invalid continuation duration')
    elapsed += continuation_hours
    paid = original['account_balance_usd'] - balance
    price = original['total_price_usd_per_hour']
    consumed = max(paid, elapsed * price)
    # Round down whole seconds so neither original hours nor dollars can expand.
    hours = math.floor(min(144-elapsed, (239.56-consumed)/price)*3600)/3600
    cap = hours * price
    projected = original['projected_total_usd'] + consumed  # conservatively retains original whole stage
    if (hours < 200000000/original['rate_ex_per_s']/3600+4 or cap > balance
        or projected > 1000 or cap > 1000-(505-balance)):
        raise RuntimeError('Remaining stage or project funds insufficient')
    return {'max_hours': hours, 'max_cost_usd': cap, 'prior_elapsed_hours': elapsed,
        'paid_since_original_allocation_usd': paid, 'stage_consumed_usd': consumed,
        'projected_total_usd': math.ceil(projected*100)/100,
        'remaining_headroom_usd': math.floor((1000-projected)*100)/100,
        'account_balance_usd': balance, 'spent_to_date_usd': 505-balance,
        'remaining_budget_usd': 1000-(505-balance)}


def main():
    if OUTPUT.exists(): raise RuntimeError('Existing continuation allocation must be preserved')
    subprocess.run(['git','diff','--quiet','HEAD'], cwd=REPO, check=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    pushed = subprocess.check_output(['git','ls-remote','--exit-code','origin',
        'refs/heads/m13-stage1-execution-prep'],cwd=REPO,text=True,timeout=30).split()[0]
    if head != pushed: raise RuntimeError('Require exact pushed prerequisites')
    load = lambda name: json.loads((REPO/name).read_text())
    original, attempt, pause = map(load, (ORIGINAL, ATTEMPT, PAUSE))
    if sha(REPO/ATTEMPT) != ATTEMPT_SHA or sha(REPO/ORIGINAL) != attempt['allocation_sha256']:
        raise RuntimeError('Preserved attempt/allocation identity changed')
    bound = original['artifact_sha256']
    for name, digest in bound.items():
        if sha(REPO/name) != digest: raise RuntimeError('Changed registered input: '+name)
    pods, balance = live()
    if any(p['desiredStatus'] != 'EXITED' for p in pods): raise RuntimeError('Require all retained Pods stopped')
    pod = pods[-1]
    if (pod['id'] != original['pod_id'] or not 0 < float(pod['costPerHr']) <= 1.59
        or pod.get('volumeInGb') != 500 or pod.get('containerDiskInGb') != 30
        or pod.get('volumeMountPath') != '/home/dylan'):
        raise RuntimeError('Pod identity, storage or quote changed')
    result = dict(original)
    result.update(remaining(original,attempt,pause,balance))
    result.update(status='PASSED', measured_utc=datetime.now(timezone.utc).isoformat(),
        prerequisite_commit=head, allocation_script_sha256=sha(__file__),
        continuation_of={name:sha(REPO/name) for name in (ORIGINAL,ATTEMPT,PAUSE)},
        limitations='Original stage hours and dollars reduced by prior runtime and all balance charges including paused storage; conservative original project projection plus new paid charges. Same Pod, no new observations or recipe changes.')
    for name, digest in bound.items():
        if sha(REPO/name) != digest: raise RuntimeError('Input changed during reconciliation')
    with OUTPUT.open('x') as stream: json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','max_hours','max_cost_usd','account_balance_usd','projected_total_usd')}))

if __name__ == '__main__': main()
