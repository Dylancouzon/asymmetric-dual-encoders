import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import m13_migration_allocation as a


def original():
    return {
        'max_hours': 144,
        'max_cost_usd': 239.56,
        'funding_baseline_usd': 505,
        'account_balance_usd': 500,
        'spent_to_date_usd': 5,
        'total_price_usd_per_hour': a.ORIGINAL_PRICE,
        'projected_total_usd': 800,
        'rate_ex_per_s': 441.9,
    }


def ready(source_hours=.25, target_hours=.5):
    return {
        'status': 'PASSED',
        'source_id': a.SOURCE,
        'target_id': a.TARGET,
        'source_status': 'EXITED',
        'target_status': 'RUNNING',
        'target_gpu_count': 1,
        'target_model': 'A100SXM80GB',
        'verified_files': 394,
        'transfer_verified': True,
        'source_receipt_sha256': 's' * 64,
        'manifest_sha256': 'm' * 64,
        'source_volume_in_gb': 500,
        'target_volume_in_gb': 500,
        'target_container_disk_in_gb': 30,
        'target_volume_mount_path': '/home/dylan',
        'source_started_at': 1000,
        'source_finished_at': 1000 + source_hours * 3600,
        'target_started_at': 1000,
        'target_finished_at': 1000 + target_hours * 3600,
        'source_price_usd_per_h': a.SOURCE_PRICE,
        'target_price_usd_per_h': a.TARGET_PRICE,
    }


def test_concurrent_migration_uses_aggregate_cost_equivalent():
    result = a.migration_cost(1, 1)
    assert result['wall_hours'] == 1
    assert result['equivalent_hours'] == pytest.approx(
        (a.SOURCE_PRICE + a.TARGET_PRICE) / a.ORIGINAL_PRICE)


def test_target_lease_is_charged_after_source_stops():
    result = a.migration_hours(ready(.25, .75), now=3700)
    assert result['target_hours'] == pytest.approx(.75)
    assert result['equivalent_hours'] == pytest.approx(
        max(.75, (.25 * a.SOURCE_PRICE + .75 * a.TARGET_PRICE) / a.ORIGINAL_PRICE))


def test_migration_lease_cannot_exceed_one_hour():
    with pytest.raises(RuntimeError, match='target lease'):
        a.migration_hours(ready(.25, 1.01))


def test_target_lease_must_cover_source_interval():
    with pytest.raises(RuntimeError, match='ended before'):
        a.migration_hours(ready(.75, .5))


def test_before_state_requires_exact_stopped_disks_and_capacity():
    pod = {'id': a.SOURCE, 'desiredStatus': 'EXITED', 'volumeInGb': 500,
           'containerDiskInGb': 30, 'volumeMountPath': '/home/dylan'}
    target = dict(pod, id=a.TARGET)
    a.validate_before_state(pod, target, {'gpuAvailable': 1})
    with pytest.raises(RuntimeError):
        a.validate_before_state(pod, target, {'gpuAvailable': 0})
    with pytest.raises(RuntimeError):
        a.validate_before_state(pod, dict(target, volumeInGb=499), 1)


def test_ready_requires_exact_inventory_and_target_gpu():
    a.validate_ready(ready())
    with pytest.raises(RuntimeError):
        a.validate_ready(dict(ready(), verified_files=393))
    with pytest.raises(RuntimeError):
        a.validate_ready(dict(ready(), target_gpu_count=2))


def test_before_allocation_preserves_training_reserve_and_floors_seconds():
    result = a.before_allocation(original(), 490, .25)
    assert result['status'] == 'PASSED'
    assert result['prior_elapsed_hours'] == pytest.approx(
        .25 + result['migration']['equivalent_hours'])
    assert round(result['max_hours'] * 3600) == math.floor(
        result['max_hours'] * 3600 + 1e-9)
    assert result['max_hours'] >= a.TRAINING_RESERVE_HOURS


def test_after_allocation_accounts_for_target_lease():
    result = a.after_allocation(original(), 490, .25, ready(.25, .5), now=2800)
    assert result['status'] == 'PASSED'
    assert result['pod_id'] == a.TARGET
    assert result['ssh_alias'] == 'm13-runpod'
    assert result['migration_measured_at'] == 2800
    assert result['prior_continuation_hours'] == result['prior_elapsed_hours']


def test_price_ceiling_rejects_unbounded_target_quote():
    with pytest.raises(RuntimeError, match='price'):
        a.migration_cost(.1, .1, target_price=a.ORIGINAL_PRICE + .01)
