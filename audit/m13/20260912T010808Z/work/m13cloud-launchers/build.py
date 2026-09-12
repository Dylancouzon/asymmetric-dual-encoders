#!/usr/bin/env python3
"""Guarded 200M build supervisor. Requires reviewed staging/preflight and allocation artifacts."""
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

REPO = Path('/home/dylan/asymetric-dual-encoders/work/m13cloud')
REMOTE = '/home/dylan/asymetric-dual-encoders'
KEYS = Path('/home/dylan/.config/runpod')
RESULT = REPO / 'results/m13_cloud_build.json'
ALLOC = REPO / 'results/m13_build_allocation.json'
BACKUP = REPO / 'work/m13cloud-build-backup'
ROOTS = ('work/m13build/BUILD-200M', 'results/m13_build_record.json')
BRANCH = 'm13-stage1-execution-prep'
ENV = ('export HF_HOME=/home/dylan/.cache/huggingface HF_HUB_CACHE=/home/dylan/.cache/huggingface/hub '
       'HF_DATASETS_CACHE=/home/dylan/.cache/huggingface/datasets XDG_CACHE_HOME=/home/dylan/.cache '
       'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 '
       'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; ')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def run(cmd, timeout=60, **kwargs):
    return subprocess.run(cmd, timeout=timeout, check=True, **kwargs)


def main():
    if RESULT.exists() or BACKUP.exists() or (REPO/'logs/m13-build.log').exists() or (REPO/'results/m13_build_preflight.json').exists() or any((REPO/p).exists() for p in ROOTS):
        raise RuntimeError('Existing build evidence; no automatic rerun')
    run(['git','diff','--quiet','HEAD'], cwd=REPO)
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO,text=True).strip()
    pushed = subprocess.check_output(['git','ls-remote','--exit-code','origin','refs/heads/'+BRANCH],cwd=REPO,text=True,timeout=30).split()[0]
    if head != pushed: raise RuntimeError('Require exact pushed execution HEAD')
    allocation = json.loads(ALLOC.read_text())
    required = ('m13/LOTTE_GATE.json','m13/LOTTE_GATE_MANIFEST.json','results/m8_lotte_pin.json',
        'm13/LOTTE_GATE_REGISTRATION.json','m13/build_config.json','m10/screen_registry.json',
        'results/m10_screen_verdicts.json','results/m13_cloud_gate_chain_upload.json',
        'm13/build_transfer_manifest.json','results/m10_arm_E-bs32.json',
        'results/m13_encode_benchmark_fp16.json','results/m13_gate_chain_allocation.json')
    run(['git','ls-files','--error-unmatch',str(ALLOC.relative_to(REPO)),*required],cwd=REPO,stdout=subprocess.DEVNULL)
    if allocation.get('status')!='PASSED' or allocation.get('budget_ceiling_usd')!=1000 or set(allocation.get('artifact_sha256',{}))!=set(required):
        raise RuntimeError('Require reviewed allocation and exact input bindings')
    for p in required:
        if sha(REPO/p)!=allocation['artifact_sha256'][p]: raise RuntimeError('Changed build input: '+p)
    for k in ('max_hours','max_cost_usd','total_price_usd_per_hour','rate_ex_per_s','remaining_budget_usd'):
        if type(allocation.get(k)) not in (int,float) or not math.isfinite(allocation[k]) or allocation[k]<=0: raise RuntimeError('Invalid allocation '+k)
    hours, rate, price = (allocation[k] for k in ('max_hours','rate_ex_per_s','total_price_usd_per_hour'))
    if not 200000000/rate/3600+4 <= hours <= 144 or hours*price>allocation['max_cost_usd'] or allocation['max_cost_usd']>allocation['remaining_budget_usd']:
        raise RuntimeError('Build cap does not cover training, freeze and backup under the ceiling')
    gate = json.loads((REPO/required[7]).read_text())
    if gate.get('status')!='PASSED' or not gate.get('backup_verified') or gate.get('pod_final_status')!='EXITED': raise RuntimeError('Gate and its backup/STOP must pass first')
    pod_id=allocation['pod_id']
    if pod_id!=gate.get('pod_id'): raise RuntimeError('Build must use the verified gate Pod')
    transfer = json.loads((REPO/'m13/build_transfer_manifest.json').read_text())
    if len(transfer.get('entries', [])) != 394:
        raise RuntimeError('Require reviewed exact build input inventory')
    run(['git','ls-files','--error-unmatch','scripts/m13_build_preflight.py'],cwd=REPO,stdout=subprocess.DEVNULL)
    ssh_config=Path(allocation['ssh_config']); alias=allocation['ssh_alias']
    if not ssh_config.is_relative_to(KEYS) or not alias.replace('-','').isalnum(): raise RuntimeError('Invalid SSH configuration')
    ssh=['ssh','-n','-F',str(ssh_config),alias]; scp=['scp','-F',str(ssh_config)]
    headers={'Authorization':'Bearer '+(KEYS/'api_key').read_text().strip(),'User-Agent':'m13-build/1.0'}
    receipt={'status':'RUNNING','stage':'starting','pid':os.getpid(),'pod_id':pod_id,'started_at':time.time(),
        'code_commit':head,'allocation_sha256':sha(ALLOC),'rolling_backups':[]}
    with RESULT.open('x') as f: json.dump(receipt,f)
    deadline=time.monotonic()+hours*3600
    started=ready=success=False; process=None; job=head[:12]+'-'+str(os.getpid()); pidfile='/tmp/m13-build-'+job+'.pid'
    def save(stage=None):
        if stage: receipt['stage']=stage
        receipt['updated_at']=time.time(); p=RESULT.with_suffix('.pending.json'); p.write_text(json.dumps(receipt,indent=2)+'\n'); p.replace(RESULT)
        print(receipt['stage'],flush=True)
    def api(suffix='',method='GET'):
        with urllib.request.urlopen(urllib.request.Request('https://rest.runpod.io/v1/pods/'+pod_id+suffix,headers=headers,method=method),timeout=30) as f:
            raw=f.read(); return json.loads(raw) if raw else {}
    def interrupt(sig,frame): raise KeyboardInterrupt('Build supervisor interrupted: '+str(sig))
    for sig in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(sig,interrupt)
    signal.alarm(int(hours*3600)-3600)
    try:
        pod=api()
        if pod.get('desiredStatus')!='EXITED' or not 0<float(pod['costPerHr'])<=1.59: raise RuntimeError('Require stopped admitted-price Pod')
        if float(pod['costPerHr'])+530*.1/720 > price+1e-9: raise RuntimeError('Live GPU plus storage exceeds allocated price')
        started=True; save(); api('/start','POST')
        endpoint_deadline=time.monotonic()+600
        while time.monotonic()<endpoint_deadline:
            pod=api(); port=(pod.get('portMappings') or {}).get('22')
            if pod.get('publicIp') and port:
                ssh_config.write_text('\n'.join('  HostName '+pod['publicIp'] if line.strip().startswith('HostName ') else '  Port '+str(port) if line.strip().startswith('Port ') else line for line in ssh_config.read_text().splitlines())+'\n')
                r=subprocess.run(ssh+['true'],timeout=20,capture_output=True)
                if r.returncode==0: ready=True; break
            time.sleep(5)
        else: raise RuntimeError('SSH readiness timeout')
        save('deploying')
        bundle=REPO/'work/m13cloud-launchers/build.bundle'
        run(['git','bundle','create',str(bundle),'HEAD'],cwd=REPO,timeout=120)
        run(scp+[str(bundle),alias+':/tmp/m13-build.bundle'],timeout=180)
        # Preserve newly generated gate receipts before the commit supplies their tracked twins.
        known={name:sha(REPO/name) for name in ('m13/LOTTE_GATE.json','results/m13_encode_benchmark_fp16.json')}
        preserve="""import pathlib,hashlib,subprocess,shutil
for name,want in KNOWN.items():
 p=pathlib.Path(name)
 if p.exists() and subprocess.run(['git','ls-files','--error-unmatch',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=want: raise RuntimeError('Gate artifact backup identity mismatch')
  d=pathlib.Path('work/m13-build-preserved-gate')/name; d.parent.mkdir(parents=True,exist_ok=True)
  if d.exists(): raise RuntimeError('Existing gate artifact archive')
  shutil.copy2(p,d)
  if d.read_bytes()!=p.read_bytes(): raise RuntimeError('Gate artifact archive mismatch')
  p.unlink()
""".replace('KNOWN',repr(known))
        bootstrap=(REPO/'scripts/m13_cloud_bootstrap.sh').read_text()
        replacements={'UV_CACHE_DIR=/home/dylan/.cache/uv':'UV_CACHE_DIR=/opt/m13-runtime/uv-cache',
            'UV_PYTHON_INSTALL_DIR=/home/dylan/.local/share/uv/python':'UV_PYTHON_INSTALL_DIR=/opt/m13-runtime/python',
            'uv_tools=/home/dylan/.local/share/m13-uv-0.12.5':'uv_tools=/opt/m13-runtime/uv-tools',
            'venv="$repo/.venv"':'venv=/opt/m13-runtime/venv'}
        for a,b in replacements.items():
            if bootstrap.count(a)!=1: raise RuntimeError('Bootstrap template changed')
            bootstrap=bootstrap.replace(a,b)
        local_bootstrap=REPO/'work/m13cloud-launchers/build-runtime.sh'; local_bootstrap.write_text(bootstrap)
        receipt['runtime_bootstrap_sha256']=sha(local_bootstrap)
        run(scp+[str(local_bootstrap),alias+':/tmp/m13-build-runtime.sh'])
        setup='set -eu; mountpoint -q /home/dylan; cd '+REMOTE+'; git diff --quiet HEAD; python3 -c '+shlex.quote(preserve)
        setup+='; git fetch /tmp/m13-build.bundle HEAD; git merge --ff-only FETCH_HEAD; git update-ref refs/remotes/origin/'+BRANCH+' '+head
        setup+='; test "$(git rev-parse HEAD)" = '+head+'; mkdir -p /opt/m13-runtime /home/dylan/setup-logs; '
        setup+='bash /tmp/m13-build-runtime.sh '+REMOTE+' > /home/dylan/setup-logs/bootstrap-build.log 2>&1; test "$(readlink .venv)" = /opt/m13-runtime/venv'
        run(ssh+['timeout --kill-after=30s 1500s bash -c '+shlex.quote(setup)],timeout=1540)
        save('uploading-build-inputs')
        root_files=[]; tree_files=[]; expected={}
        for row in transfer['entries']:
            src=Path(row['source']); dst=Path(row['destination'])
            if not dst.is_absolute() or '..' in dst.parts or not src.is_absolute(): raise RuntimeError('Invalid staged path')
            if src.is_relative_to(REPO):
                rel=src.relative_to(REPO)
                if dst!=Path(REMOTE)/rel: raise RuntimeError('Unexpected worktree destination')
                tree_files.append(str(rel))
            else:
                if dst!=src or not src.is_relative_to(Path('/home/dylan')): raise RuntimeError('Unexpected root destination')
                root_files.append(str(src).lstrip('/'))
            if ('work/lotte' in str(dst) or 'work/m9reserve' in str(dst) or 'frozen_eval' in str(dst) or dst.name in ('cqadup-android.json','cqadup-english.json')):
                raise RuntimeError('Protected data excluded from build staging')
            digest=row.get('sha256')
            if not isinstance(digest,str) or len(digest)!=64 or sha(src)!=digest:
                raise RuntimeError('Local build input differs from committed transfer pin')
            expected[str(dst)]=digest
        staging_deadline=time.monotonic()+36000
        for label,names,source,dest,copy_links in (('root',root_files,'/',alias+':/',False),('tree',tree_files,str(REPO)+'/',alias+':'+REMOTE+'/',True)):
            listing=REPO/('work/m13cloud-launchers/build-'+label+'.files'); listing.write_text('\n'.join(names)+'\n')
            flags=['--copy-links'] if copy_links else []
            run(['rsync','-a','--no-owner','--no-group','--info=progress2',*flags,'--files-from='+str(listing),'-e','ssh -F '+shlex.quote(str(ssh_config)),source,dest],timeout=max(1,int(staging_deadline-time.monotonic())))
        sums=REPO/'work/m13cloud-launchers/build-input-sha256.txt'; sums.write_text('\n'.join(h+'  '+p for p,h in expected.items())+'\n')
        run(scp+[str(sums),alias+':/tmp/m13-build-input-sha256.txt'])
        run(ssh+['timeout --kill-after=30s 1800s sha256sum --check --status /tmp/m13-build-input-sha256.txt'],timeout=1840)
        receipt['transfer_verified']=True; receipt['transfer_manifest_sha256']=sha(sums)
        save('full-build-preflight')
        command=ENV+'cd '+REMOTE+'; .venv/bin/python -u scripts/m13_build_preflight.py'
        run(ssh+['timeout --kill-after=30s 7200s bash -c '+shlex.quote(command)],timeout=7240)
        preflight_path=REPO/'results/m13_build_preflight.json'
        if preflight_path.exists(): raise RuntimeError('Existing local build preflight evidence')
        run(scp+[alias+':'+REMOTE+'/results/m13_build_preflight.json',str(preflight_path)])
        digest=subprocess.check_output(ssh+['sha256sum '+REMOTE+'/results/m13_build_preflight.json'],timeout=30,text=True).split()[0]
        preflight=json.loads(preflight_path.read_text())
        if digest!=sha(preflight_path) or preflight.get('status')!='PASSED' or not preflight.get('full_uncut_assembly_verified') or not preflight.get('dev6_cache_reuse_verified') or not preflight.get('cov_cache_integrity_verified') or preflight.get('git_head')!=head:
            raise RuntimeError('Full assembly and existing DEV6 preflight failed')
        receipt['preflight_sha256']=digest
        save('runtime-preflight')
        check='set -eu; cd '+REMOTE+'; test "$(git rev-parse HEAD)" = '+head+'; git diff --quiet HEAD; '
        check+='test ! -e work/m13build/BUILD-200M; test ! -e results/m13_build_record.json; '
        check+=ENV+'.venv/bin/python -c '+shlex.quote('import torch; assert torch.cuda.is_available(); assert "A100" in torch.cuda.get_device_name(0)')
        run(ssh+['timeout --kill-after=30s 120s bash -c '+shlex.quote(check)],timeout=160)
        seconds=int(deadline-time.monotonic())-3600
        if seconds < 200000000/rate + 7200:
            raise RuntimeError('Setup left insufficient planned training and freeze allowance')
        cmd=ENV+'cd '+REMOTE+'; .venv/bin/python -u m13src/build13.py --config m13/build_config.json --device cuda --rate '+str(rate)+' --price '+str(price)
        owned='env M13_BUILD_JOB_ID='+job+' setsid bash -c '+shlex.quote('echo $$ > '+pidfile+'; exec timeout --kill-after=30s '+str(seconds)+'s bash -c '+shlex.quote(cmd))
        BACKUP.mkdir(); (BACKUP/'rolling').mkdir(); save('training')
        with (REPO/'logs/m13-build.log').open('x') as log:
            process=subprocess.Popen(ssh+[owned],stdout=log,stderr=subprocess.STDOUT)
            last_backup=time.monotonic()-3600
            backup_failures=0
            while process.poll() is None:
                if time.monotonic()-last_backup>=3600:
                    snapshot='/tmp/m13-build-'+job+'-rolling.pt'
                    command='set -eu; p='+REMOTE+'/work/m13build/BUILD-200M/ckpt.pt; if test -f "$p"; then cp --reflink=auto "$p" '+snapshot+'; sha256sum '+snapshot+'; fi'
                    for attempt in range(2):
                        local=None
                        try:
                            output=subprocess.check_output(ssh+[command],timeout=180,text=True).strip()
                            if output:
                                expected=output.split()[0]; local=BACKUP/'rolling'/(str(time.time_ns())+'.pt')
                                run(scp+[alias+':'+snapshot,str(local)],timeout=600)
                                if sha(local)!=expected: raise RuntimeError('Rolling backup mismatch')
                                receipt['rolling_backups'].append({'path':str(local.relative_to(REPO)),'sha256':expected})
                                live=[r for r in receipt['rolling_backups'] if not r.get('pruned')]
                                for old in live[:-2]:
                                    old_path=REPO/old['path']
                                    if old_path.parent!=BACKUP/'rolling' or sha(old_path)!=old['sha256']: raise RuntimeError('Refuse changed rolling backup cleanup')
                                    old_path.unlink(); old['pruned']=True; old['pruned_at']=time.time()
                                backup_failures=0; save()
                            break
                        except Exception as e:
                            if local is not None and local.exists() and not any(r['path']==str(local.relative_to(REPO)) for r in receipt['rolling_backups']): local.unlink()
                            receipt['rolling_backup_warning']=type(e).__name__+': '+str(e)
                            if attempt==1:
                                backup_failures+=1; receipt['consecutive_rolling_backup_failures']=backup_failures; save()
                            else: time.sleep(5)
                    last_backup=time.monotonic()
                time.sleep(30)
            if process.returncode: raise RuntimeError('Build command failed with exit '+str(process.returncode))
        # One retry is permitted only for completed training whose finalization
        # explicitly remained non-terminal. Retain the first outcome as evidence.
        first=BACKUP/'first-finalization-record.json'
        run(scp+[alias+':'+REMOTE+'/results/m13_build_record.json',str(first)])
        first_sha=subprocess.check_output(ssh+['sha256sum '+REMOTE+'/results/m13_build_record.json'],timeout=30,text=True).split()[0]
        if sha(first)!=first_sha: raise RuntimeError('First finalization record backup mismatch')
        record=json.loads(first.read_text())
        if record.get('status')=='frozen_unverified':
            if (record.get('terminal') is not False or record.get('complete') is not False
                or record.get('dose_run_examples')!=200000000 or record.get('git_head')!=head
                or not record.get('final_checkpoint_sha256')):
                raise RuntimeError('Invalid non-terminal finalization record')
            state_check='import json; from pathlib import Path; s=json.loads(Path("work/m13build/BUILD-200M/state.json").read_text()); assert s.get("base_done") is True; assert s.get("examples_run")==200000000'
            run(ssh+['cd '+REMOTE+'; .venv/bin/python -c '+shlex.quote(state_check)])
            retry_seconds=min(7200,int(deadline-time.monotonic())-3600)
            if retry_seconds<=0: raise RuntimeError('No finalization retry time remains')
            receipt['freeze_retry']={'started_at':time.time(),'first_record_sha256':first_sha,
                'recipe_fingerprint':record['recipe_fingerprint'],'final_checkpoint_sha256':record['final_checkpoint_sha256'],'training_repeated':False}
            save('freeze-retry')
            retry='env M13_BUILD_JOB_ID='+job+' setsid bash -c '+shlex.quote('echo $$ > '+pidfile+'; exec timeout --kill-after=30s '+str(retry_seconds)+'s bash -c '+shlex.quote(cmd+' --resume'))
            with (REPO/'logs/m13-build.log').open('a') as log:
                run(ssh+[retry],timeout=retry_seconds+40,stdout=log,stderr=subprocess.STDOUT)
        success=True
    except BaseException as e: receipt['error']=type(e).__name__+': '+str(e)
    finally:
        signal.alarm(0)
        for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT): signal.signal(sig,signal.SIG_IGN)
        try:
            if ready:
                available=int(deadline-time.monotonic())-720
                if available<=0: raise RuntimeError('Backup deadline reached; remote volume retained')
                signal.alarm(available); save('backup')
                quiesce="""import os,pathlib,signal,time
p=pathlib.Path(PIDFILE)
if p.exists():
 pid=int(p.read_text()); proc=pathlib.Path('/proc')/str(pid)
 if proc.exists():
  if MARKER not in (proc/'environ').read_bytes().split(bytes([0])) or os.getpgid(pid)!=pid: raise RuntimeError('Unowned build process')
  os.killpg(pid,signal.SIGTERM); time.sleep(3)
  try: os.killpg(pid,signal.SIGKILL)
  except ProcessLookupError: pass
""".replace('PIDFILE',repr(pidfile)).replace('MARKER',repr(('M13_BUILD_JOB_ID='+job).encode()))
                run(ssh+['python3 -c '+shlex.quote(quiesce)],timeout=60)
                script="""import pathlib,hashlib,json
out={}
for name in ROOTS:
 root=pathlib.Path(name)
 for p in ([root] if root.is_file() else root.rglob('*')):
  if p.is_symlink(): raise RuntimeError('Symlink in build outputs')
  if p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''): h.update(b)
   out[str(p)]=h.hexdigest()
pathlib.Path('/tmp/m13-build-output-manifest.json').write_text(json.dumps(out))
""".replace('ROOTS',repr(ROOTS))
                run(ssh+['timeout --kill-after=30s 540s bash -c '+shlex.quote('cd '+REMOTE+'; python3 -c '+shlex.quote(script))],timeout=600)
                BACKUP.mkdir(exist_ok=True); manifest=BACKUP/'manifest.json'
                run(scp+[alias+':/tmp/m13-build-output-manifest.json',str(manifest)])
                hashes=json.loads(manifest.read_text())
                if any('..' in Path(p).parts or not any(p==root or p.startswith(root+'/') for root in ROOTS) for p in hashes): raise RuntimeError('Unexpected backup path')
                listing=BACKUP/'files.txt'; listing.write_text('\n'.join(hashes)+'\n')
                run(['rsync','-a','--no-owner','--no-group','--info=progress2','--files-from='+str(listing),'-e','ssh -F '+shlex.quote(str(ssh_config)),alias+':'+REMOTE+'/',str(BACKUP)+'/'],timeout=1800)
                if any(sha(BACKUP/p)!=h for p,h in hashes.items()): raise RuntimeError('Build backup checksum mismatch')
                receipt['backup_verified']=True; receipt['backup_manifest_sha256']=sha(manifest)
                if success:
                    record=json.loads((BACKUP/ROOTS[1]).read_text())
                    if record.get('git_head')!=head or record.get('smoke') or record.get('status')!='complete' or record.get('complete') is not True or record.get('dose_run_examples')!=200000000 or (record.get('freeze') or {}).get('verified') is not True: raise RuntimeError('Build did not complete the registered dose and freeze')
                    if receipt.get('freeze_retry') and any(record.get(k)!=receipt['freeze_retry'][k] for k in ('recipe_fingerprint','final_checkpoint_sha256')): raise RuntimeError('Finalization retry identity changed')
                    if record.get('config_sha256')!=allocation['artifact_sha256']['m13/build_config.json'] or record.get('registry_sha256')!=allocation['artifact_sha256']['m10/screen_registry.json']: raise RuntimeError('Final build input identity mismatch')
                    final=record['final_checkpoint']
                    if final not in hashes or not final.startswith(ROOTS[0]+'/') or sha(BACKUP/final)!=record['final_checkpoint_sha256']: raise RuntimeError('Final checkpoint identity mismatch')
                    receipt['final_checkpoint_sha256']=record['final_checkpoint_sha256']
        except BaseException as e: success=False; receipt['backup_error']=type(e).__name__+': '+str(e)
        finally:
            signal.alarm(0)
            if started:
                for _ in range(6):
                    try:
                        api('/stop','POST'); receipt['pod_final_status']=api().get('desiredStatus')
                        if receipt['pod_final_status']=='EXITED': break
                    except Exception as e: receipt['stop_error']=str(e)
                    time.sleep(3)
                else: success=False
        receipt['status']='PASSED' if success else 'FAILED'; receipt['finished_at']=time.time(); save('finished')
    return 0 if receipt['status']=='PASSED' else 1


if __name__=='__main__': raise SystemExit(main())
