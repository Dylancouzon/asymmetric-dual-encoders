from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import m13_preflight_retry_allocation as retry


def failed(tmp_path, start=100.0, finish=200.0):
    allocation = tmp_path / retry.MIGRATION_ALLOCATION
    allocation.parent.mkdir(parents=True, exist_ok=True)
    allocation.write_text('{"status":"PASSED"}\n')
    return ({
        'status': 'FAILED', 'stage': 'finished', 'pod_id': retry.TARGET,
        'pod_final_status': 'EXITED', 'backup_verified': True,
        'backup_manifest_sha256': retry.EMPTY_BACKUP_MANIFEST_SHA256,
        'preflight_sha256': None, 'final_checkpoint_sha256': None,
        'training_started': False, 'started_at': start, 'finished_at': finish,
        'allocation_sha256': retry.migration.sha(allocation),
    }, allocation)


def test_failed_interval_stops_at_confirmed_pod_stop(tmp_path):
    receipt, _ = failed(tmp_path, 100, 3700)
    assert retry.failed_hours(receipt, now=5000, repo=tmp_path) == pytest.approx(1)
    with pytest.raises(RuntimeError, match='precedes'):
        retry.failed_hours(receipt, now=3000, repo=tmp_path)


def test_cumulative_replay_adds_terminal_interval_once(tmp_path):
    receipt, _ = failed(tmp_path, 100, 3700)
    assert retry.cumulative_hours(2.5, receipt, now=5000, repo=tmp_path) == pytest.approx(3.5)


def test_failed_retry_rejects_nonempty_or_training_evidence(tmp_path):
    receipt, _ = failed(tmp_path)
    with pytest.raises(RuntimeError):
        retry._validate_failed(dict(receipt, backup_manifest_sha256='0' * 64), tmp_path)
    with pytest.raises(RuntimeError):
        retry._validate_failed(dict(receipt, training_started=True), tmp_path)


def test_target_retry_requires_stopped_exact_a100():
    target = {'id': retry.TARGET, 'desiredStatus': 'EXITED', 'gpuCount': 1,
              'volumeInGb': 500, 'containerDiskInGb': 30,
              'volumeMountPath': '/home/dylan'}
    capacity = {'id': retry.TARGET, 'desiredStatus': 'EXITED', 'gpuCount': 1,
                'machine': {'gpuTypeId': 'NVIDIA A100-SXM4-80GB', 'gpuAvailable': 1}}
    retry.validate_target_state(target, capacity)
    with pytest.raises(RuntimeError):
        retry.validate_target_state(dict(target, desiredStatus='RUNNING'), capacity)
    with pytest.raises(RuntimeError):
        retry.validate_target_state(target, dict(capacity, gpuCount=2))
