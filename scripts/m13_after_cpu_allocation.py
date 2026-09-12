#!/usr/bin/env python3
"""Read-only admission after verified zero-GPU upload; never reset original stage spend."""
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import urllib.request
import m13_resume_allocation as recovery

REPO = Path(__file__).resolve().parents[1]
CPU = 'results/m13_storage_upload.json'
CPU_FAILED = 'results/m13_cpu_upload.json'
FAILED = 'results/m13_cloud_build_resume.json'
FIRST_ALLOC = 'results/m13_build_resume_allocation.json'
OUTPUT = REPO/'results/m13_build_after_cpu_allocation.json'
CHAIN = (recovery.ORIGINAL,recovery.ATTEMPT,recovery.PAUSE,FIRST_ALLOC,FAILED,CPU_FAILED,CPU)


def chain_hours(repo=REPO):
    load=lambda name:json.loads((repo/name).read_text())
    original,attempt,pause,first,failed,cpu_failed,cpu=map(load,CHAIN)
    if recovery.sha(repo/recovery.ATTEMPT)!=recovery.ATTEMPT_SHA or recovery.sha(repo/recovery.ORIGINAL)!=attempt['allocation_sha256']:
        raise RuntimeError('Original attempt/allowance changed')
    if (failed.get('status')!='FAILED' or failed.get('pod_final_status')!='EXITED'
        or failed.get('allocation_sha256')!=recovery.sha(repo/FIRST_ALLOC)
        or failed.get('transfer_verified') or failed.get('preflight_sha256')
        or cpu_failed.get('status')!='FAILED' or cpu_failed.get('pod_final_status')!='EXITED'
        or cpu_failed.get('training_started') is not False or cpu_failed.get('transfer_verified')
        or cpu_failed.get('pod_id')!=original['pod_id']
        or cpu.get('storage_only_operation') is not True
        or cpu.get('status')!='PASSED' or cpu.get('pod_final_status')!='EXITED'
        or cpu.get('training_started') is not False or cpu.get('transfer_verified') is not True
        or cpu.get('verified_files')!=394 or not cpu.get('checksum_list_sha256')
        or cpu.get('pod_id')!=original['pod_id'] or failed.get('pod_id')!=original['pod_id']):
        raise RuntimeError('Require verified upload and stopped infrastructure-only chain')
    for name in CHAIN[:-1]:
        if cpu.get('artifact_sha256',{}).get(name)!=recovery.sha(repo/name):
            raise RuntimeError('CPU receipt does not bind prior evidence: '+name)
    if cpu['artifact_sha256'].get('m13/build_transfer_manifest.json')!=recovery.sha(repo/'m13/build_transfer_manifest.json'):
        raise RuntimeError('CPU upload inventory changed')
    hours=0
    for receipt,limit in ((failed,1),(cpu_failed,1),(cpu,10.2)):
        elapsed=(receipt['finished_at']-receipt['started_at'])/3600
        if not math.isfinite(elapsed) or not 0<=elapsed<=limit:raise RuntimeError('Invalid prior stage duration')
        hours+=elapsed
    if not 0 < cpu['maximum_hours'] <= 10 or cpu['price_ceiling_usd_h'] != original['total_price_usd_per_hour']:
        raise RuntimeError('CPU stage cap changed')
    return hours


def gpu_capacity():
    headers={'Authorization':'Bearer '+(recovery.KEYS/'api_key').read_text().strip(),
        'Content-Type':'application/json','User-Agent':'m13-build-allocation/1.0'}
    query='query { myself { pods { id desiredStatus gpuCount machine { gpuAvailable } } } }'
    req=urllib.request.Request('https://api.runpod.io/graphql',headers=headers,data=json.dumps({'query':query}).encode())
    with urllib.request.urlopen(req,timeout=30) as response:data=json.load(response)
    pods=data['data']['myself']['pods']
    pod=next(p for p in pods if p['id']==recovery.PODS[-1])
    count=(pod.get('machine') or {}).get('gpuAvailable')
    if type(count) not in (int,float) or not math.isfinite(count) or count<0:raise RuntimeError('Unknown GPU capacity')
    return pod,count


def main():
    if OUTPUT.exists():raise RuntimeError('Preserve existing after-CPU allocation')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=REPO,check=True)
    subprocess.run(['git','ls-files','--error-unmatch',*CHAIN],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    pushed=subprocess.check_output(['git','ls-remote','--exit-code','origin','refs/heads/m13-stage1-execution-prep'],cwd=REPO,text=True,timeout=30).split()[0]
    if head!=pushed:raise RuntimeError('Require exact pushed chain')
    hours=chain_hours()
    original,attempt,pause=[json.loads((REPO/p).read_text()) for p in (recovery.ORIGINAL,recovery.ATTEMPT,recovery.PAUSE)]
    bindings={p:recovery.sha(REPO/p) for p in CHAIN}
    for name,digest in original['artifact_sha256'].items():
        if recovery.sha(REPO/name)!=digest:raise RuntimeError('Changed registered input: '+name)
    pods,balance=recovery.live()
    if any(p['desiredStatus']!='EXITED' for p in pods):raise RuntimeError('Require all retained Pods stopped')
    pod=pods[-1]
    if pod.get('volumeInGb')!=500 or pod.get('containerDiskInGb')!=30 or pod.get('volumeMountPath')!='/home/dylan':
        raise RuntimeError('Retained storage changed')
    capacity,count=gpu_capacity()
    if capacity['desiredStatus']!='EXITED' or count<1:raise RuntimeError('No free GPU on retained host; no paid retry')
    result=dict(original)
    result.update(recovery.remaining(original,attempt,pause,balance,continuation_hours=hours))
    result.update(status='PASSED',measured_utc=datetime.now(timezone.utc).isoformat(),
        prerequisite_commit=head,allocation_script_sha256=recovery.sha(__file__),
        continuation_of=bindings,prior_continuation_hours=hours,gpu_capacity_observed=count,
        limitations='Original interruption, failed GPU start and complete CPU upload deducted at conservative original rate or actual balance charges, whichever is greater. Same retained Pod; explicit GPU resume required; no scientific resume or extra dose.')
    if any(recovery.sha(REPO/p)!=h for p,h in bindings.items()):raise RuntimeError('Chain changed during reconciliation')
    with OUTPUT.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','max_hours','max_cost_usd','account_balance_usd','projected_total_usd')}))

if __name__=='__main__':main()
