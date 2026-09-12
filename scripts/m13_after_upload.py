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
import m13_after_cpu_allocation as admission
import m13_resume_allocation as recovery

REPO=Path(__file__).resolve().parents[1]
RESULT=REPO/'results/m13_after_verification.json'
GPU_RESULT=REPO/'results/m13_cloud_build_after_cpu.json'
GPU_PID=REPO/'work/m13cloud-launchers/build_after_cpu.pid'
GPU_LOG=REPO/'logs/m13-build-after-cpu-controller.log'
PATCH='m13/after_cpu_supervisor.patch'
SUPERVISOR='scripts/m13_resume_build.py'
AFTER_SHA='846e93b71fca78f6b4174b9dfc0e667175db11cb0bc453046e50b8d5d49c67d7'
CODE=('scripts/m13_after_upload.py','scripts/m13_after_cpu_allocation.py',
      'scripts/m13_resume_allocation.py',SUPERVISOR,PATCH,'scripts/m13_upload_only.py',
      'm13/monitor_config.json','scripts/m13_verify_uploaded.py')


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
    url='https://rest.runpod.io/v1/pods/'+recovery.PODS[-1]
    last=None
    for _ in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url+'/stop',headers=headers,method='POST'),timeout=30) as response: response.read()
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as response: pod=json.load(response)
            if pod.get('desiredStatus')=='EXITED':return 'EXITED'
        except Exception as error: last=type(error).__name__+': '+str(error)
        time.sleep(3)
    raise RuntimeError('Failed dispatch STOP unconfirmed: '+str(last))


def restore_container():
    """Restore only ephemeral runtime capacity while all retained Pods are stopped."""
    pods,_=recovery.live()
    if any(p['desiredStatus']!='EXITED' for p in pods):raise RuntimeError('Container restoration requires all Pods stopped')
    pod=pods[-1]
    if (pod['id']!=recovery.PODS[-1] or pod.get('volumeInGb')!=500
        or pod.get('volumeMountPath')!='/home/dylan' or pod.get('containerDiskInGb') not in (5,30)):
        raise RuntimeError('Unexpected retained disk configuration')
    if pod['containerDiskInGb']==30:return {'changed':False,'containerDiskInGb':30,'pod_status':'EXITED'}
    headers={'Authorization':'Bearer '+(recovery.KEYS/'api_key').read_text().strip(),
             'Content-Type':'application/json','User-Agent':'m13-container-restore/1.0'}
    url='https://rest.runpod.io/v1/pods/'+recovery.PODS[-1]
    try:
        # Exact single-field update; never resize, replace or detach persistent storage.
        req=urllib.request.Request(url,headers=headers,data=json.dumps({'containerDiskInGb':30}).encode(),method='PATCH')
        with urllib.request.urlopen(req,timeout=30) as response:response.read()
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as response:current=json.load(response)
        if (current.get('id')!=pod['id'] or current.get('desiredStatus')!='EXITED'
            or current.get('volumeInGb')!=500 or current.get('volumeMountPath')!='/home/dylan'
            or current.get('containerDiskInGb')!=30):raise RuntimeError('Container restoration not confirmed stopped and intact')
    except BaseException:
        stop_failed_dispatch()
        raise
    return {'changed':True,'containerDiskInGb':30,'pod_status':'EXITED','restored_at':time.time()}


def main():
    if RESULT.exists() or GPU_RESULT.exists() or GPU_PID.exists() or GPU_LOG.exists() or admission.OUTPUT.exists():
        raise RuntimeError('Existing handoff/build evidence; no automatic retry')
    if git('diff','--name-only','HEAD'):raise RuntimeError('Require clean tracked tree')
    require_pushed()
    previous=REPO/'results/m13_after_upload.json'
    if recovery.sha(previous)!='2fcad24c9ed0cb6130e8af4a2bfcfdf7fbadab0876b23a9a55d99f5b2e407ec6':
        raise RuntimeError('Previous stopped handoff changed')
    bound={p:recovery.sha(REPO/p) for p in CODE}
    original_inputs=json.loads((REPO/recovery.ORIGINAL).read_text())['artifact_sha256']
    def unchanged():
        if any(recovery.sha(REPO/p)!=h for p,h in {**bound,**original_inputs}.items()):
            raise RuntimeError('Reviewed handoff source or input changed')
    receipt={'status':'WAITING','stage':'waiting','pid':os.getpid(),'started_at':time.time(),
             'initial_commit':git('rev-parse','HEAD'),'source_sha256':dict(bound)}
    with RESULT.open('x') as stream:json.dump(receipt,stream)
    def save(stage=None):
        if stage:receipt['stage']=stage
        receipt['updated_at']=time.time();tmp=RESULT.with_suffix('.pending.json')
        tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(RESULT);print(receipt['stage'],flush=True)
    process=None
    try:
        # No admission until the source controller finishes with verified hashes and STOP.
        upload_deadline=time.monotonic()+4*3600
        while time.monotonic()<upload_deadline:
            unchanged()
            if (REPO/admission.CPU).exists():
                cpu=json.loads((REPO/admission.CPU).read_text())
                if cpu.get('status')=='FAILED':raise RuntimeError('Storage upload failed; preserve it')
                if cpu.get('status')=='PASSED':break
            save();time.sleep(30)
        else:raise RuntimeError('Storage upload completion timeout')
        admission.chain_hours(REPO)
        receipt['storage_upload_sha256']=recovery.sha(REPO/admission.CPU)
        save('publishing-verified-upload')
        publish([admission.CPU], 'Record verified M13 storage upload and STOP')
        unchanged()
        save('restoring-runtime-container')
        receipt['container_restoration']=restore_container()
        save()
        # The reviewed patch is applied only after upload completion, never during its bundle.
        subprocess.run(['git','apply','--check',PATCH],cwd=REPO,check=True,timeout=30)
        subprocess.run(['git','apply',PATCH],cwd=REPO,check=True,timeout=30)
        if recovery.sha(REPO/SUPERVISOR)!=AFTER_SHA:raise RuntimeError('Applied supervisor differs from reviewed code')
        bound[SUPERVISOR]=AFTER_SHA
        receipt['applied_supervisor_sha256']=AFTER_SHA
        publish([SUPERVISOR], 'Activate reviewed M13 build handoff after verified storage upload')
        save('waiting_gpu')
        capacity_deadline=time.monotonic()+12*3600
        while time.monotonic()<capacity_deadline:
            unchanged()
            if recovery.sha(REPO/admission.CPU)!=receipt['storage_upload_sha256']:
                raise RuntimeError('Completed upload receipt changed')
            try:
                pod,count=admission.gpu_capacity()
                if pod['desiredStatus']!='EXITED':raise RuntimeError('Retained Pod is unexpectedly active')
                receipt['available_gpus']=count
                if count>=1:break
            except (OSError,KeyError,ValueError) as error:
                receipt['capacity_lookup_warning']=type(error).__name__+': '+str(error)
            save();time.sleep(300)
        else:raise RuntimeError('No retained-host GPU capacity within 12h; no paid retry')
        save('reconciling-allocation')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        subprocess.run([str(REPO/'.venv/bin/python'),'scripts/m13_after_cpu_allocation.py'],cwd=REPO,env=env,check=True,timeout=240)
        publish([str(admission.OUTPUT.relative_to(REPO))], 'Record live post-upload M13 build allowance')
        unchanged()
        receipt['allocation_sha256']=recovery.sha(admission.OUTPUT)
        save('launching-reviewed-build')
        subprocess.run(['systemctl','--user','is-active','--quiet','m13-monitor.service'],check=True,timeout=15)
        with GPU_LOG.open('x') as stream:
            process=subprocess.Popen(['/usr/bin/python3','-u',SUPERVISOR,'--after-cpu'],cwd=REPO,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with GPU_PID.open('x') as stream:stream.write(str(process.pid)+'\n')
        receipt['build_pid']=process.pid
        start_deadline=time.monotonic()+120
        while time.monotonic()<start_deadline:
            if GPU_RESULT.exists():
                try: build=json.loads(GPU_RESULT.read_text())
                except json.JSONDecodeError:
                    time.sleep(1);continue
                if build.get('status')=='RUNNING' and process.poll() is None:
                    receipt.update(status='PASSED',finished_at=time.time());save('handed_off');return 0
                if build.get('status')=='FAILED':raise RuntimeError('Reviewed build refused or failed; inspect its own receipt')
            if process.poll() is not None:raise RuntimeError('Build controller exited before a live receipt')
            time.sleep(1)
        raise RuntimeError('Build startup receipt timeout; independently check build controller')
    except BaseException as error:
        if process is not None:
            try: save('aborting-unconfirmed-start')
            except Exception: pass  # Receipt I/O must never bypass child/Pod cleanup.
            try: receipt['failed_dispatch_pod_status']=cancel_unconfirmed(process)
            except Exception as cleanup_error:
                receipt['failed_dispatch_cleanup_error']=type(cleanup_error).__name__+': '+str(cleanup_error)
        receipt.update(status='FAILED',error=type(error).__name__+': '+str(error),finished_at=time.time())
        save('finished');return 1

if __name__=='__main__':raise SystemExit(main())
