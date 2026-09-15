#!/usr/bin/env python3
"""Bounded verification-only recovery of the exact retained upload; STOP, never train."""
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request
import m13_resume_allocation as budget
import m13_upload_only as upload

REPO=upload.REPO
FAILED='results/m13_storage_upload.json'
FAILED_SHA='413c7856c1d6c53473c425fdf51b4f373bbc346e4c70d09c7a312310ce4bfcdc'
RESULT=REPO/'results/m13_storage_verification.json'


def prior_hours(repo=REPO):
    if budget.sha(repo/FAILED)!=FAILED_SHA:raise RuntimeError('Upload timeout evidence changed')
    failed=json.loads((repo/FAILED).read_text())
    if (failed.get('status')!='FAILED' or failed.get('pod_final_status')!='EXITED'
        or failed.get('training_started') is not False or failed.get('transfer_verified')
        or not failed.get('storage_only_operation') or failed.get('pod_id')!=upload.POD
        or 'timed out after 1840 seconds' not in failed.get('error','')):
        raise RuntimeError('Not the captured destination verification timeout')
    for name,digest in failed['artifact_sha256'].items():
        if budget.sha(repo/name)!=digest:raise RuntimeError('Upload chain changed: '+name)
    total=0
    for name,limit in ((upload.FAILED,1),('results/m13_cpu_upload.json',1),(FAILED,10.2)):
        r=json.loads((repo/name).read_text());hours=(r['finished_at']-r['started_at'])/3600
        if not math.isfinite(hours) or not 0<=hours<=limit:raise RuntimeError('Invalid consumed duration')
        total+=hours
    return total


# Remote stdlib only; exact manifest destinations, no recursive reads or extra payloads.
REMOTE_CODE='''import hashlib,json,pathlib,time
entries=json.loads(pathlib.Path('/tmp/m13-verify-manifest.json').read_text())['entries']
assert len(entries)==394
start=time.monotonic();total=0;last=start
for number,row in enumerate(entries,1):
 p=pathlib.Path(row['destination']);assert p.stat().st_size==row['bytes'], 'Size mismatch: '+str(p)
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):
   h.update(block);total+=len(block)
   if time.monotonic()-last>=15:
    print(json.dumps({'files_completed':number-1,'bytes_read':total,'elapsed_s':round(time.monotonic()-start,1)}),flush=True);last=time.monotonic()
 assert h.hexdigest()==row['sha256'], 'Checksum mismatch: '+str(p)
 print(json.dumps({'verified_file':number,'bytes_read':total,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
print(json.dumps({'status':'PASSED','verified_files':len(entries),'verified_bytes':total}),flush=True)
'''


def main():
    if RESULT.exists():raise RuntimeError('Preserve existing verification evidence')
    upload.run(['git','diff','--quiet','HEAD'],cwd=REPO)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    pushed=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+upload.BRANCH],cwd=REPO,text=True,timeout=30).split()[0]
    if head!=pushed:raise RuntimeError('Require exact pushed HEAD')
    consumed=prior_hours()
    original,attempt,pause=[json.loads((REPO/n).read_text()) for n in (budget.ORIGINAL,budget.ATTEMPT,budget.PAUSE)]
    for n,h in original['artifact_sha256'].items():
        if budget.sha(REPO/n)!=h:raise RuntimeError('Registered input changed')
    manifest=REPO/'m13/build_transfer_manifest.json'
    print('Checking pinned local inventory',flush=True)
    _,expected=upload.inventory(json.loads(manifest.read_text()))
    pods,balance=budget.live()
    if any(p['desiredStatus']!='EXITED' for p in pods):raise RuntimeError('Require retained Pods stopped')
    allowance=budget.remaining(original,attempt,pause,balance,continuation_hours=consumed)
    hours=min(3,allowance['max_hours']-200000000/original['rate_ex_per_s']/3600-4,
              10-(json.loads((REPO/FAILED).read_text())['finished_at']-json.loads((REPO/FAILED).read_text())['started_at'])/3600)
    if hours<=.5:raise RuntimeError('Insufficient bounded verification time')
    pod=pods[-1]
    if pod['id']!=upload.POD or pod.get('volumeInGb')!=500 or pod.get('containerDiskInGb') not in (5,30) or pod.get('volumeMountPath')!='/home/dylan':raise RuntimeError('Retained disk changed')
    bound=dict(json.loads((REPO/FAILED).read_text())['artifact_sha256']);bound[FAILED]=FAILED_SHA
    receipt={'status':'RUNNING','stage':'starting-zero-gpu','pid':os.getpid(),'pod_id':upload.POD,
             'code_commit':head,'started_at':time.time(),'maximum_hours':hours,
             'price_ceiling_usd_h':original['total_price_usd_per_hour'],'allocation':allowance,
             'training_started':False,'storage_only_operation':True,'verification_only':True,
             'artifact_sha256':bound,'prior_continuation_hours':consumed,
             'verifier_source_sha256':__import__('hashlib').sha256(REMOTE_CODE.encode()).hexdigest()}
    with RESULT.open('x') as out:json.dump(receipt,out)
    def save(stage=None):
        if stage:receipt['stage']=stage
        receipt['updated_at']=time.time();tmp=RESULT.with_suffix('.pending.json');tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(RESULT);print(receipt['stage'],flush=True)
    headers={'Authorization':'Bearer '+(budget.KEYS/'api_key').read_text().strip(),'Content-Type':'application/json','User-Agent':'m13-build-allocation/1.0'}
    def request(url,body=None,method=None):
        req=urllib.request.Request(url,headers=headers,data=None if body is None else json.dumps(body).encode(),method=method)
        with urllib.request.urlopen(req,timeout=30) as out:
            raw=out.read();return json.loads(raw) if raw else {}
    def api(suffix='',method='GET'):return request('https://rest.runpod.io/v1/pods/'+upload.POD+suffix,method=method)
    def interrupted(sig,frame):raise KeyboardInterrupt('Verification interrupted '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    deadline=time.monotonic()+hours*3600;signal.alarm(int(hours*3600)-180)
    ssh=['ssh','-n','-F',str(upload.CONFIG),upload.ALIAS];scp=['scp','-F',str(upload.CONFIG)]
    started=False;success=False
    try:
        started=True
        response=request('https://api.runpod.io/graphql',{'query':'mutation { podResumeZeroGpu(input: {podId:"'+upload.POD+'"}) { id gpuCount } }'})
        pod=(response.get('data') or {}).get('podResumeZeroGpu') or {}
        if response.get('errors') or pod.get('id')!=upload.POD or pod.get('gpuCount')!=0:raise RuntimeError('Exact zero GPU resume failed')
        ready=min(deadline-300,time.monotonic()+600)
        while time.monotonic()<ready:
            pod=api();price=float(pod['costPerHr'])+530*.1/720
            if not math.isfinite(price) or not 0<=price<=receipt['price_ceiling_usd_h']:raise RuntimeError('Price exceeds allowance')
            port=(pod.get('portMappings') or {}).get('22')
            if pod.get('publicIp') and port:
                upload.CONFIG.write_text('\n'.join('  HostName '+pod['publicIp'] if line.strip().startswith('HostName ') else '  Port '+str(port) if line.strip().startswith('Port ') else line for line in upload.CONFIG.read_text().splitlines())+'\n')
                if subprocess.run(ssh+['true'],timeout=20,capture_output=True).returncode==0:break
            time.sleep(5)
        else:raise RuntimeError('SSH readiness timeout')
        receipt['observed_total_price_usd_h']=price
        upload.run(ssh+['mountpoint -q /home/dylan'])
        helper=REPO/'work/m13cloud-launchers/verify-uploaded-remote.py';helper.write_text(REMOTE_CODE)
        upload.run(scp+[str(helper),upload.ALIAS+':/tmp/m13-verify-uploaded.py'])
        upload.run(scp+[str(manifest),upload.ALIAS+':/tmp/m13-verify-manifest.json'])
        upload.run(ssh+['sha256sum /tmp/m13-verify-uploaded.py /tmp/m13-verify-manifest.json'],capture_output=True) # identity checked below
        check='import hashlib,pathlib; assert hashlib.sha256(pathlib.Path("/tmp/m13-verify-uploaded.py").read_bytes()).hexdigest()=='+repr(budget.sha(helper))+'; assert hashlib.sha256(pathlib.Path("/tmp/m13-verify-manifest.json").read_bytes()).hexdigest()=='+repr(budget.sha(manifest))
        upload.run(ssh+['python3 -c '+shlex.quote(check)])
        save('verifying-destination-hashes')
        seconds=int(deadline-time.monotonic())-240
        if seconds<=0:raise RuntimeError('No verification time remains')
        upload.run(ssh+['timeout --kill-after=30s '+str(seconds)+'s python3 -u /tmp/m13-verify-uploaded.py'],timeout=seconds+40)
        if any(budget.sha(REPO/n)!=h for n,h in bound.items()):raise RuntimeError('Evidence changed during verification')
        sums='\n'.join(h+'  '+p for p,h in expected.items())+'\n'
        receipt.update(transfer_verified=True,verified_files=394,checksum_list_sha256=__import__('hashlib').sha256(sums.encode()).hexdigest())
        success=True
    except BaseException as error:receipt['error']=type(error).__name__+': '+str(error)
    finally:
        signal.alarm(0)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
        if started:
            for _ in range(6):
                try:
                    api('/stop','POST');receipt['pod_final_status']=api().get('desiredStatus')
                    if receipt['pod_final_status']=='EXITED':break
                except Exception as error:receipt['stop_error']=type(error).__name__+': '+str(error)
                time.sleep(3)
            else:success=False
        receipt.update(status='PASSED' if success else 'FAILED',finished_at=time.time());save('finished')
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
