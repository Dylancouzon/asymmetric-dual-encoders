#!/usr/bin/env python3
"""One bounded zero-GPU staging attempt on the retained Pod; never train or score."""
import hashlib
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

REPO = Path(__file__).resolve().parents[1]
REMOTE = '/home/dylan/asymetric-dual-encoders'
RESULT = REPO/'results/m13_cpu_upload.json'
FAILED = 'results/m13_cloud_build_resume.json'
ALLOC = 'results/m13_build_resume_allocation.json'
CONFIG = budget.KEYS/'m13_gate_chain_ssh_config'
ALIAS = 'm13-gate-chain'
POD = 'wnzk8eeqrrkw4m'
BRANCH = 'm13-stage1-execution-prep'


def run(argv, **kw):
    return subprocess.run(argv, check=True, timeout=kw.pop('timeout',60), **kw)


def inventory(transfer):
    if len(transfer['entries']) != 394 or transfer['files'] != 394:
        raise RuntimeError('Unexpected inventory')
    groups={'root': [], 'tree': []}; expected={}; sources=set()
    for item in transfer['entries']:
        source,dest=map(Path,(item['source'],item['destination']))
        if (not source.is_absolute() or not dest.is_absolute() or '..' in source.parts
            or '..' in dest.parts or str(source) in sources or str(dest) in expected):
            raise RuntimeError('Invalid or duplicate transfer path')
        if any(x in str(dest) for x in ('work/lotte','work/m9reserve','frozen_eval')) or dest.name in ('perquery.json','cqadup-android.json','cqadup-english.json'):
            raise RuntimeError('Protected payload excluded')
        if source.is_relative_to(REPO):
            rel=source.relative_to(REPO)
            if dest != Path(REMOTE)/rel: raise RuntimeError('Worktree mapping changed')
            groups['tree'].append(str(rel))
        else:
            if dest != source or not source.is_relative_to('/home/dylan'): raise RuntimeError('Root mapping changed')
            groups['root'].append(str(source).lstrip('/'))
        if source.stat().st_size != item['bytes'] or budget.sha(source) != item['sha256']:
            raise RuntimeError('Local transfer identity changed: '+str(source))
        sources.add(str(source));expected[str(dest)]=item['sha256']
    return groups,expected


def main():
    if RESULT.exists(): raise RuntimeError('Preserve existing upload attempt; no automatic retry')
    run(['git','diff','--quiet','HEAD'],cwd=REPO)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    pushed=subprocess.check_output(['git','ls-remote','--exit-code','origin','refs/heads/'+BRANCH],cwd=REPO,text=True,timeout=30).split()[0]
    if head != pushed: raise RuntimeError('Require exact pushed HEAD')
    load=lambda p:json.loads((REPO/p).read_text())
    original,attempt,pause=map(load,(budget.ORIGINAL,budget.ATTEMPT,budget.PAUSE))
    failed=load(FAILED);alloc=load(ALLOC)
    if (failed.get('status')!='FAILED' or failed.get('pod_final_status')!='EXITED'
        or failed.get('pod_id')!=POD or failed.get('preflight_sha256') or failed.get('transfer_verified')
        or failed.get('allocation_sha256')!=budget.sha(REPO/ALLOC)
        or budget.sha(REPO/budget.ATTEMPT)!=budget.ATTEMPT_SHA
        or budget.sha(REPO/budget.ORIGINAL)!=attempt['allocation_sha256']):
        raise RuntimeError('Not the stopped infrastructure-only failure')
    for name,digest in original['artifact_sha256'].items():
        if budget.sha(REPO/name)!=digest:raise RuntimeError('Registered input changed: '+name)
    transfer=load('m13/build_transfer_manifest.json')
    print('Verifying exact local upload inventory',flush=True)
    groups,expected=inventory(transfer)
    failed_hours=(failed['finished_at']-failed['started_at'])/3600
    if not 0 <= failed_hours < 1:raise RuntimeError('Unexpected failed restart duration')
    pods,balance=budget.live()
    if any(p['desiredStatus']!='EXITED' for p in pods):raise RuntimeError('Require all Pods stopped')
    allowed=budget.remaining(original,attempt,pause,balance,continuation_hours=failed_hours)
    # Preserve conservative training/finalization time within the original stage.
    hours=min(10,allowed['max_hours']-200000000/original['rate_ex_per_s']/3600-4)
    if hours <= 1:raise RuntimeError('Insufficient staging allowance')
    price=original['total_price_usd_per_hour']
    pod=pods[-1]
    if pod['id']!=POD or pod.get('volumeInGb')!=500 or pod.get('containerDiskInGb')!=30 or pod.get('volumeMountPath')!='/home/dylan':
        raise RuntimeError('Persistent storage identity changed')
    headers={'Authorization':'Bearer '+(budget.KEYS/'api_key').read_text().strip(),
        'Content-Type':'application/json','User-Agent':'m13-build-allocation/1.0'}
    def request(url,body=None,method=None):
        req=urllib.request.Request(url,headers=headers,data=None if body is None else json.dumps(body).encode(),method=method)
        with urllib.request.urlopen(req,timeout=30) as f:
            raw=f.read();return json.loads(raw) if raw else {}
    def api(suffix='',method='GET'):
        return request('https://rest.runpod.io/v1/pods/'+POD+suffix,method=method)
    receipt={'status':'RUNNING','stage':'starting-zero-gpu','pid':os.getpid(),'pod_id':POD,
        'code_commit':head,'started_at':time.time(),'maximum_hours':hours,'price_ceiling_usd_h':price,
        'allocation':allowed,'training_started':False,'artifact_sha256':{name:budget.sha(REPO/name)
        for name in (FAILED,ALLOC,budget.ORIGINAL,budget.ATTEMPT,budget.PAUSE,'m13/build_transfer_manifest.json')}}
    with RESULT.open('x') as f:json.dump(receipt,f)
    def save(stage=None):
        if stage:receipt['stage']=stage
        receipt['updated_at']=time.time();p=RESULT.with_suffix('.pending.json')
        p.write_text(json.dumps(receipt,indent=2)+'\n');p.replace(RESULT);print(receipt['stage'],flush=True)
    def interrupted(sig,frame):raise KeyboardInterrupt('Upload interrupted '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    deadline=time.monotonic()+hours*3600
    signal.alarm(int(hours*3600)-180)
    started=False;success=False
    ssh=['ssh','-n','-F',str(CONFIG),ALIAS];scp=['scp','-F',str(CONFIG)]
    try:
        started=True
        response=request('https://api.runpod.io/graphql',{'query':'mutation { podResume(input: { podId: "'+POD+'", gpuCount: 0 }) { id desiredStatus gpuCount } }'})
        if response.get('errors'):raise RuntimeError('Zero-GPU resume refused: '+json.dumps(response['errors']))
        if response.get('data',{}).get('podResume',{}).get('id')!=POD:raise RuntimeError('Wrong resumed Pod')
        if response['data']['podResume'].get('gpuCount') != 0:
            raise RuntimeError('Upload requires zero GPUs')
        ready_deadline=min(deadline-300,time.monotonic()+600)
        while time.monotonic()<ready_deadline:
            pod=api()
            live_price=float(pod['costPerHr'])+530*.1/720
            if not math.isfinite(live_price) or not 0<=live_price<=price:raise RuntimeError('CPU quote exceeds conservative allocation')
            port=(pod.get('portMappings') or {}).get('22')
            if pod.get('publicIp') and port:
                CONFIG.write_text('\n'.join('  HostName '+pod['publicIp'] if line.strip().startswith('HostName ') else '  Port '+str(port) if line.strip().startswith('Port ') else line for line in CONFIG.read_text().splitlines())+'\n')
                if subprocess.run(ssh+['true'],timeout=20,capture_output=True).returncode==0:break
            time.sleep(5)
        else:raise RuntimeError('Zero-GPU SSH readiness timeout')
        receipt['observed_total_price_usd_h']=live_price
        save('deploying-upload-tools')
        bundle=REPO/'work/m13cloud-launchers/cpu-upload.bundle'
        run(['git','bundle','create',str(bundle),'HEAD'],cwd=REPO,timeout=120)
        run(scp+[str(bundle),ALIAS+':/tmp/m13-cpu-upload.bundle'],timeout=180)
        setup='set -eu; mountpoint -q /home/dylan; cd '+REMOTE+'; git diff --quiet HEAD; git fetch /tmp/m13-cpu-upload.bundle HEAD; git merge --ff-only FETCH_HEAD; git update-ref refs/remotes/origin/'+BRANCH+' '+head+'; test "$(git rev-parse HEAD)" = '+head+'; if ! command -v rsync >/dev/null; then apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rsync; fi'
        run(ssh+['timeout --kill-after=30s 600s bash -c '+shlex.quote(setup)],timeout=640)
        save('uploading-build-inputs')
        for label,source,dest,flags in [('root','/',ALIAS+':/',[]),('tree',str(REPO)+'/',ALIAS+':'+REMOTE+'/',['--copy-links'])]:
            listing=REPO/('work/m13cloud-launchers/cpu-upload-'+label+'.files');listing.write_text('\n'.join(groups[label])+'\n')
            run(['rsync','-a','--no-owner','--no-group','--info=progress2','--partial-dir=/home/dylan/.m13-build-partial-'+label,*flags,'--files-from='+str(listing),'-e','ssh -F '+shlex.quote(str(CONFIG)),source,dest],timeout=max(1,int(deadline-time.monotonic())-120))
        save('verifying-destination-hashes')
        sums=REPO/'work/m13cloud-launchers/cpu-upload-sha256.txt';sums.write_text('\n'.join(h+'  '+p for p,h in expected.items())+'\n')
        run(scp+[str(sums),ALIAS+':/tmp/m13-cpu-upload-sha256.txt'])
        run(ssh+['timeout --kill-after=30s 1800s sha256sum --check --status /tmp/m13-cpu-upload-sha256.txt'],timeout=min(1840,max(1,int(deadline-time.monotonic())-120)))
        receipt.update(transfer_verified=True,verified_files=len(expected),checksum_list_sha256=budget.sha(sums))
        success=True
    except BaseException as e:receipt['error']=type(e).__name__+': '+str(e)
    finally:
        signal.alarm(0)
        for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
        if started:
            for _ in range(6):
                try:
                    api('/stop','POST');receipt['pod_final_status']=api().get('desiredStatus')
                    if receipt['pod_final_status']=='EXITED':break
                except Exception as e:receipt['stop_error']=type(e).__name__+': '+str(e)
                time.sleep(3)
            else:success=False
        receipt['status']='PASSED' if success else 'FAILED';receipt['finished_at']=time.time();save('finished')
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
