"""Synthetic recovery allowance checks: no cloud calls or experiment reads."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('resume_allocation', Path(__file__).resolve().parents[1]/'scripts/m13_resume_allocation.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture():
    original = dict(max_hours=144,max_cost_usd=239.56,funding_baseline_usd=505,
        account_balance_usd=489.58,spent_to_date_usd=15.42,total_price_usd_per_hour=1.6636111111111112,
        projected_total_usd=706.20,rate_ex_per_s=441.9)
    attempt = dict(status='FAILED',stage='finished',pod_final_status='EXITED',backup_verified=True,started_at=1000)
    pause = dict(status='PASSED',training_started=False,stage_at_request='uploading-build-inputs',
        artifact_sha256={r.ATTEMPT:r.ATTEMPT_SHA},pod_statuses={p:'EXITED' for p in r.PODS},verified_at=8200)
    return original,attempt,pause


def test_unbilled_upload_time_is_still_deducted():
    a,b,c=fixture(); d=r.remaining(a,b,c,489.58)
    assert d['max_hours'] <= 142
    assert d['max_cost_usd'] + 2*a['total_price_usd_per_hour'] <= a['max_cost_usd']


def test_all_paid_charges_including_pause_storage_reduce_dollars():
    a,b,c=fixture(); d=r.remaining(a,b,c,480)
    assert d['max_cost_usd']+9.58 <= a['max_cost_usd']+1e-9
    assert d['projected_total_usd']+1e-9 >= a['projected_total_usd']+9.58


@pytest.mark.parametrize('balance',[490,505,-1,float('nan'),float('inf'),300])
def test_invalid_unfunded_or_changed_ledger_refuses(balance):
    with pytest.raises(RuntimeError): r.remaining(*fixture(),balance)


@pytest.mark.parametrize('field,value',[('training_started',True),('verified_at',900),('status','FAILED')])
def test_only_verified_upload_pause_is_admitted(field,value):
    a,b,c=fixture();c[field]=value
    with pytest.raises(RuntimeError):r.remaining(a,b,c,489.58)


def test_a_paused_pod_that_was_not_stopped_refuses():
    a,b,c=fixture();c['pod_statuses'][r.PODS[0]]='RUNNING'
    with pytest.raises(RuntimeError):r.remaining(a,b,c,489.58)


def test_staging_time_is_charged_even_before_balance_updates():
    a,b,c=fixture(); d=r.remaining(a,b,c,489.58,continuation_hours=3)
    assert d['max_hours'] <= 139
    assert d['stage_consumed_usd'] == 5*a['total_price_usd_per_hour']
