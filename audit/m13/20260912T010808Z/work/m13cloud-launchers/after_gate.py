#!/usr/bin/env python3
"""Success-only handoff from the fixed gate supervisor to the reviewed build."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path('/home/dylan/asymetric-dual-encoders/work/m13cloud')
LAUNCH=REPO/'work/m13cloud-launchers'
RESULT=REPO/'results/m13_after_gate.json'
GATE='results/m13_cloud_gate_chain_upload.json'
ACTIVE='results/m13_gate_chain_upload_active_pod.json'
BENCH='results/m13_encode_benchmark_fp16.json'
MANIFEST=LAUNCH/'gate-chain-upload-outputs.json'
POD='wnzk8eeqrrkw4m'
PID=136163
BRANCH='m13-stage1-execution-prep'
DEADLINE=1789170672.323684+17*3600+300
ARTIFACTS=('m13/LOTTE_GATE.json',BENCH,GATE,ACTIVE)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def run(args,timeout=120):
    return subprocess.run(args,cwd=REPO,check=True,timeout=timeout)


def git_output(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True,timeout=60).strip()


def exact_pushed():
    head=git_output('rev-parse','HEAD')
    if git_output('ls-remote','--exit-code','origin','refs/heads/'+BRANCH).split()[0]!=head:
        raise RuntimeError('Require exact pushed branch HEAD')
    return head


def commit_only(paths,message):
    # No unrelated staged edits can enter the scientific record commit.
    if git_output('diff','--cached','--name-only'): raise RuntimeError('Unrelated staged changes')
    changed=set(git_output('diff','--name-only','HEAD').splitlines())
    if changed-set(paths): raise RuntimeError('Unrelated tracked changes')
    run(['git','add','--',*paths])
    run(['git','commit','-m',message,'--',*paths])
    run(['git','push','origin','HEAD:refs/heads/'+BRANCH],timeout=180)
    return exact_pushed()


def main():
    if RESULT.exists() or (REPO/'results/m13_cloud_build.json').exists() or (REPO/'results/m13_build_allocation.json').exists():
        raise RuntimeError('Existing handoff/build evidence; no automatic retry')
    run(['git','diff','--quiet','HEAD'])
    initial_head=exact_pushed()
    scripts={name:sha(LAUNCH/name) for name in ('after_gate.py','build.py','build_allocation.py')}
    receipt={'status':'RUNNING','stage':'waiting-for-gate','pid':os.getpid(),'started_at':time.time(),
             'initial_head':initial_head,'launcher_sha256':scripts,'gate_pid':PID,'pod_id':POD}
    with RESULT.open('x') as f: json.dump(receipt,f)
    def save(stage=None):
        if stage: receipt['stage']=stage
        receipt['updated_at']=time.time(); tmp=RESULT.with_suffix('.pending.json')
        tmp.write_text(json.dumps(receipt,indent=2)+'\n'); tmp.replace(RESULT)
        print(receipt['stage'],flush=True)
    try:
        while time.time()<DEADLINE:
            gate=json.loads((REPO/GATE).read_text())
            if gate.get('pid')!=PID or gate.get('pod_id')!=POD: raise RuntimeError('Unexpected gate controller')
            if gate.get('status')=='PASSED': break
            if gate.get('status') not in ('STARTING','RUNNING'): raise RuntimeError('Gate did not pass; preserved without continuation')
            proc=Path('/proc')/str(PID)
            if not proc.exists() or b'gate_chain_upload.py' not in (proc/'cmdline').read_bytes():
                raise RuntimeError('Gate controller is absent or changed')
            save(); time.sleep(min(300,max(1,DEADLINE-time.time())))
        else: raise RuntimeError('Original gate deadline elapsed')
        if not gate.get('backup_verified') or gate.get('pod_final_status')!='EXITED' or not gate.get('fp16_backup_verified'):
            raise RuntimeError('Gate backup and STOP must be confirmed')
        # Receipt is the last supervisor write; wait for its process to exit.
        for _ in range(30):
            proc=Path('/proc')/str(PID)
            if not proc.exists() or '\nState:\tZ' in (proc/'status').read_text(): break
            time.sleep(1)
        else: raise RuntimeError('Gate controller remains live')
        if any(sha(LAUNCH/name)!=digest for name,digest in scripts.items()): raise RuntimeError('Reviewed launcher changed while waiting')
        if exact_pushed()!=initial_head: raise RuntimeError('Reviewed source HEAD changed while waiting')
        if sha(MANIFEST)!=gate.get('backup_manifest_sha256'): raise RuntimeError('Gate backup manifest changed')
        outputs=json.loads(MANIFEST.read_text())
        # The gate supervisor already hashed every payload; only metadata is reread here.
        if sha(REPO/ARTIFACTS[0])!=outputs.get(ARTIFACTS[0]): raise RuntimeError('Gate record differs from verified backup')
        row=json.loads((REPO/ARTIFACTS[0]).read_text())
        if row.get('executed') is not True or row.get('git_head')!=gate.get('code_commit') or row.get('decision') not in ('skipped','veto','no_veto'):
            raise RuntimeError('Invalid executed gate record')
        benchmark=json.loads((REPO/BENCH).read_text()); active=json.loads((REPO/ACTIVE).read_text())
        if sha(REPO/BENCH)!=gate.get('benchmark_sha256') or benchmark.get('status')!='PASSED' or benchmark.get('git_head')!=gate['code_commit'] or benchmark.get('encode_dtype')!='fp16':
            raise RuntimeError('Invalid verified fp16 benchmark')
        if active.get('status')!='PASSED' or active.get('pod_id')!=POD: raise RuntimeError('Invalid active Pod identity')
        run(['git','merge-base','--is-ancestor',gate['code_commit'],'HEAD'])
        receipt['gate_artifact_sha256']={p:sha(REPO/p) for p in ARTIFACTS}
        save('committing-gate')
        receipt['gate_commit']=commit_only(ARTIFACTS,'Record verified M13 LoTTE gate and bounded cloud execution')
        save('allocating-build')
        run([sys.executable,str(LAUNCH/'build_allocation.py')],timeout=180)
        allocation='results/m13_build_allocation.json'
        receipt['allocation_sha256']=sha(REPO/allocation)
        receipt['build_commit']=commit_only((allocation,),'Allocate bounded M13 full build with verified gate inputs')
        save('running-build')
        log=REPO/'logs/m13-build-controller.log'
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,str(LAUNCH/'build.py')],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        (LAUNCH/'build.pid').write_text(str(child.pid)+'\n')
        receipt['build_pid']=child.pid; save()
        for _ in range(60):
            if child.poll() is not None: raise RuntimeError('Build supervisor exited before verified handoff')
            evidence=REPO/'results/m13_cloud_build.json'
            if evidence.exists():
                try: started=json.loads(evidence.read_text())
                except json.JSONDecodeError:
                    time.sleep(1); continue
                if started.get('pid')==child.pid and started.get('pod_id')==POD and started.get('status')=='RUNNING': break
                raise RuntimeError('Unexpected build supervisor receipt')
            time.sleep(1)
        else: raise RuntimeError('Build supervisor did not publish startup receipt; inspect owned child PID')
        # The detached build has its own absolute cost bound and STOP owner.
        receipt['status']='PASSED'; receipt['outcome']='handed_off'; save('finished')
        return 0
    except BaseException as e:
        receipt['status']='FAILED'; receipt['error']=type(e).__name__+': '+str(e); save('finished')
        return 1


if __name__=='__main__': raise SystemExit(main())
