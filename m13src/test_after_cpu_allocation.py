"""Receipt-chain admission on scratch evidence only."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m13_after_cpu_allocation as a


def make_chain(root,monkeypatch):
    def put(name,data):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
    pod=a.recovery.PODS[-1]
    put(a.recovery.ORIGINAL,dict(pod_id=pod,total_price_usd_per_hour=1.6636111111111112))
    put(a.recovery.ATTEMPT,dict(allocation_sha256=a.recovery.sha(root/a.recovery.ORIGINAL)))
    monkeypatch.setattr(a.recovery,'ATTEMPT_SHA',a.recovery.sha(root/a.recovery.ATTEMPT))
    put(a.recovery.PAUSE,{})
    put(a.FIRST_ALLOC,{})
    put(a.FAILED,dict(status='FAILED',pod_final_status='EXITED',pod_id=pod,
        allocation_sha256=a.recovery.sha(root/a.FIRST_ALLOC),started_at=0,finished_at=5))
    put(a.CPU_FAILED,dict(status='FAILED',pod_final_status='EXITED',pod_id=pod,
        training_started=False,started_at=10,finished_at=12))
    put('m13/build_transfer_manifest.json',{})
    storage=dict(status='FAILED',pod_final_status='EXITED',pod_id=pod,training_started=False,
        storage_only_operation=True,error='timed out after 1840 seconds',started_at=20,finished_at=3620,
        artifact_sha256={p:a.recovery.sha(root/p) for p in a.CHAIN[:-2]})
    storage['artifact_sha256']['m13/build_transfer_manifest.json']=a.recovery.sha(root/'m13/build_transfer_manifest.json')
    put(a.STORAGE_FAILED,storage)
    monkeypatch.setattr(a.verification,'FAILED_SHA',a.recovery.sha(root/a.STORAGE_FAILED))
    cpu=dict(status='PASSED',pod_final_status='EXITED',training_started=False,transfer_verified=True,
        verified_files=394,checksum_list_sha256='a'*64,pod_id=pod,maximum_hours=3,verification_only=True,
        price_ceiling_usd_h=1.6636111111111112,storage_only_operation=True,
        started_at=3700,finished_at=7300,
        artifact_sha256={p:a.recovery.sha(root/p) for p in a.CHAIN[:-1]})
    cpu['artifact_sha256']['m13/build_transfer_manifest.json']=a.recovery.sha(root/'m13/build_transfer_manifest.json')
    put(a.CPU,cpu)
    return cpu,put


def test_every_completed_interval_is_charged(tmp_path,monkeypatch):
    make_chain(tmp_path,monkeypatch)
    assert a.chain_hours(tmp_path)==pytest.approx((5+2+3600+3600)/3600)


@pytest.mark.parametrize('field,value',[('status','FAILED'),('verified_files',393),('transfer_verified',False),('training_started',True),('pod_final_status','RUNNING')])
def test_incomplete_or_non_staging_outcomes_refuse(tmp_path,monkeypatch,field,value):
    cpu,put=make_chain(tmp_path,monkeypatch);cpu[field]=value;put(a.CPU,cpu)
    with pytest.raises(RuntimeError):a.chain_hours(tmp_path)


def test_changed_prior_receipt_refuses(tmp_path,monkeypatch):
    cpu,put=make_chain(tmp_path,monkeypatch);put(a.CPU_FAILED,{})
    with pytest.raises(RuntimeError):a.chain_hours(tmp_path)
