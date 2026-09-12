#!/usr/bin/env python3
"""Wait for verified storage upload, publish it, and hand off once to the reviewed GPU build.

The build supervisor owns the paid resume and STOP. Failed startup has a STOP-only fallback.
"""
import hashlib
import json
import os
import signal
import urllib.request
from pathlib import Path
import subprocess
import time
import m13_migration_allocation as admission
import m13_resume_allocation as recovery

REPO=Path(__file__).resolve().parents[1]
RESULT=REPO/'results/m13_after_migration_recovery.json'
GPU_RESULT=REPO/'results/m13_cloud_build_migrated.json'
GPU_PID=REPO/'work/m13cloud-launchers/build_migrated.pid'
GPU_LOG=REPO/'logs/m13-build-migrated-controller.log'
SUPERVISOR='scripts/m13_build_migrated.py'
CODE=('scripts/m13_after_migration_recovery.py','scripts/m13_migration_allocation.py',
      'scripts/m13_migrate_inputs.py',SUPERVISOR,'scripts/m13_resume_allocation.py',
      'scripts/m13_after_cpu_allocation.py','scripts/m13_verify_uploaded.py','m13/monitor_config.json')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True,timeout=45).strip()


def require_pushed():
    if git('rev-parse','HEAD')!=git('ls-remote','--exit-code','origin','refs/heads/m13-stage1-execution-prep').split()[0]:
        raise RuntimeError('Require exact pushed branch HEAD')


def publish(paths,message):
    changed=set(git('diff','--name-only','HEAD').splitlines())
    if changed-set(paths):raise RuntimeError('Unrelated tracked changes; refuse automatic publication')
    subprocess.run(['git','add','--',*paths],cwd=REPO,check=True,timeout=30)
    if git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m',message],cwd=REPO,check=True,timeout=45)
    subprocess.run(['git','push','origin','m13-stage1-execution-prep'],cwd=REPO,check=True,timeout=60)
    require_pushed()


def cancel_unconfirmed(process):
    """Quiesce our child before a STOP-only fallback; no late paid start can escape."""
    termination_error=None
    try:
        if process.poll() is None:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=30)
    except Exception as error:
        termination_error=type(error).__name__+': '+str(error)
        # Retry a forced quiescence even if an earlier termination operation failed.
        try:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=30)
            termination_error=None
        except ProcessLookupError: termination_error=None
        except Exception as error: termination_error+='; '+type(error).__name__+': '+str(error)
    finally:
        stopped=stop_failed_dispatch()
    if termination_error:raise RuntimeError('STOP completed but child termination unconfirmed: '+termination_error)
    return stopped


def stop_failed_dispatch():
    # This Pod was stopped before our single child dispatch. Never target other Pods.
    headers={'Authorization':'Bearer '+(recovery.KEYS/'api_key').read_text().strip(),
             'User-Agent':'m13-after-upload-stop/1.0'}
    url='https://rest.runpod.io/v1/pods/'+admission.TARGET
    last=None
    for _ in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url+'/stop',headers=headers,method='POST'),timeout=30) as response: response.read()
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as response: pod=json.load(response)
            if pod.get('desiredStatus')=='EXITED':return 'EXITED'
        except Exception as error: last=type(error).__name__+': '+str(error)
        time.sleep(3)
    raise RuntimeError('Failed dispatch STOP unconfirmed: '+str(last))


def main():
    if RESULT.exists() or GPU_RESULT.exists() or GPU_PID.exists() or GPU_LOG.exists() or admission.OUTPUT.exists():
        raise RuntimeError('Existing migration/build evidence; no automatic retry')
    if git('diff','--name-only','HEAD'):raise RuntimeError('Require clean tracked tree')
    require_pushed()
    bound={p:recovery.sha(REPO/p) for p in CODE}
    original_inputs=json.loads((REPO/recovery.ORIGINAL).read_text())['artifact_sha256']
    def unchanged():
        if any(recovery.sha(REPO/p)!=h for p,h in {**bound,**original_inputs}.items()):
            raise RuntimeError('Reviewed migration/build source or input changed')
    receipt={'status':'WAITING','stage':'waiting','pid':os.getpid(),'started_at':time.time(),
             'initial_commit':git('rev-parse','HEAD'),'source_sha256':bound}
    with RESULT.open('x') as out:json.dump(receipt,out)
    def save(stage=None):
        if stage:receipt['stage']=stage
        receipt['updated_at']=time.time();tmp=RESULT.with_suffix('.pending.json')
        tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(RESULT);print(receipt['stage'],flush=True)
    process=None
    target_lease_active=False
    try:
        wait_deadline=time.monotonic()+70*60
        while time.monotonic()<wait_deadline:
            unchanged()
            if (REPO/admission.READY).exists():break
            progress=REPO/'results/m13_migration.json'
            if progress.exists() and json.loads(progress.read_text()).get('status')=='FAILED':
                raise RuntimeError('Migration failed; preserve evidence')
            save();time.sleep(10)
        else:raise RuntimeError('Migration ready timeout')
        # The migration controller has left this retained target billing.  If
        # any validation, publication, or allocator step fails before the
        # build child owns it, the coordinator must still issue the exact-Pod
        # STOP fallback.
        target_lease_active=True
        admission.chain_hours(REPO,now=time.time())
        receipt['migration_ready_sha256']=recovery.sha(REPO/admission.READY)
        save('publishing-verified-migration')
        publish([admission.READY], 'Record verified cloud migration with retained GPU lease')
        unchanged()
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        subprocess.run([str(REPO/'.venv/bin/python'),'scripts/m13_migration_allocation.py','--after'],cwd=REPO,env=env,check=True,timeout=180)
        publish([str(admission.OUTPUT.relative_to(REPO))], 'Record cumulative migrated M13 build allowance')
        unchanged()
        receipt['allocation_sha256']=recovery.sha(admission.OUTPUT)
        save('launching-reviewed-build')
        subprocess.run(['systemctl','--user','is-active','--quiet','m13-monitor.service'],check=True,timeout=15)
        with GPU_LOG.open('x') as log, GPU_PID.open('x') as pid:
            process=subprocess.Popen(['/usr/bin/python3','-u',SUPERVISOR],cwd=REPO,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            pid.write(str(process.pid)+'\n')
        receipt['build_pid']=process.pid
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if GPU_RESULT.exists():
                try:build=json.loads(GPU_RESULT.read_text())
                except json.JSONDecodeError:time.sleep(1);continue
                if (build.get('status')=='RUNNING' and process.poll() is None
                    and build.get('migration_ready_sha256')==receipt['migration_ready_sha256']):
                    receipt.update(status='PASSED',finished_at=time.time());save('handed_off');return 0
                if build.get('status')=='FAILED':raise RuntimeError('Build admission failed; preserve evidence')
            if process.poll() is not None:raise RuntimeError('Build controller exited before adoption')
            time.sleep(1)
        raise RuntimeError('Build adoption timeout')
    except BaseException as error:
        if process is not None:
            try:save('aborting-unconfirmed-start')
            except Exception:pass
            try:receipt['failed_dispatch_pod_status']=cancel_unconfirmed(process)
            except Exception as cleanup_error:receipt['cleanup_error']=type(cleanup_error).__name__+': '+str(cleanup_error)
        elif target_lease_active:
            try: receipt['failed_dispatch_pod_status']=stop_failed_dispatch()
            except Exception as cleanup_error: receipt['cleanup_error']=type(cleanup_error).__name__+': '+str(cleanup_error)
        receipt.update(status='FAILED',error=type(error).__name__+': '+str(error),finished_at=time.time());save('finished');return 1

if __name__=='__main__':raise SystemExit(main())
