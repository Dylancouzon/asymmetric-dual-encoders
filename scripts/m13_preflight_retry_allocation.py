#!/usr/bin/env python3
"""Allocate one bounded preflight retry after a stopped migrated build failure.

This module performs read-only provider checks.  It preserves the original
144-hour/$239.56 stage and charges the failed migrated supervisor through its
confirmed STOP before authorizing another target-Pod attempt.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import m13_migration_allocation as migration

REPO = Path(__file__).resolve().parents[1]
FAILED = 'results/m13_cloud_build_migrated.json'
MIGRATION_ALLOCATION = 'results/m13_migration_after_allocation.json'
OUTPUT = REPO / 'results/m13_preflight_retry_allocation.json'
TARGET = migration.TARGET
SOURCE = migration.SOURCE
READY = migration.READY
SSH_CONFIG = migration.SSH_CONFIG
SSH_ALIAS = migration.SSH_ALIAS
CHAIN = migration.CHAIN + (MIGRATION_ALLOCATION, FAILED)
EMPTY_BACKUP_MANIFEST_SHA256 = '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'
FAILED_SHA256 = '2089ceb3769a9779541d14c6c3b0c4820426eb4a379d795ed5d73c93c28e750a'
_live_target_capacity = migration._live_target_capacity


def _load(repo, name):
    return json.loads((Path(repo) / name).read_text())


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError('Invalid ' + label)
    return value


def _validate_failed(receipt, repo=REPO):
    """Validate the immutable preflight failure and its prior allocation pin."""
    if (receipt.get('status') != 'FAILED' or receipt.get('stage') != 'finished'
            or receipt.get('pod_id') != TARGET
            or receipt.get('pod_final_status') != 'EXITED'
            or receipt.get('backup_verified') is not True
            or receipt.get('backup_manifest_sha256') != EMPTY_BACKUP_MANIFEST_SHA256
            or (receipt.get('backup_manifest') is not None and receipt.get('backup_manifest') != {})
            or (receipt.get('backup_files') is not None and receipt.get('backup_files') != 0)
            or receipt.get('preflight_sha256')
            or receipt.get('final_checkpoint_sha256')
            or receipt.get('training_started') is True):
        raise RuntimeError('Require stopped preflight-only migrated build failure')
    started = _finite(receipt.get('started_at'), 'failed build start')
    finished = _finite(receipt.get('finished_at'), 'failed build STOP')
    if not 0 < finished - started < 144 * 3600:
        raise RuntimeError('Invalid failed build interval')
    allocation_path = Path(repo) / MIGRATION_ALLOCATION
    if receipt.get('allocation_sha256') != migration.sha(allocation_path):
        raise RuntimeError('Failed build allocation binding changed')
    return started, finished


def failed_hours(receipt, now=None, repo=REPO):
    """Return the failed supervisor interval through its recorded STOP."""
    _, finished = _validate_failed(receipt, repo)
    if now is not None and _finite(now, 'current time') < finished:
        raise RuntimeError('Current time precedes failed build STOP')
    return (finished - _finite(receipt['started_at'], 'failed build start')) / 3600


def cumulative_hours(migration_elapsed_hours, receipt, now=None, repo=REPO):
    """Add the terminal failed-build interval to a replayed migration clock."""
    migration_elapsed_hours = _finite(migration_elapsed_hours, 'migration elapsed duration')
    if migration_elapsed_hours < 0:
        raise RuntimeError('Invalid migration elapsed duration')
    return migration_elapsed_hours + failed_hours(receipt, now=now, repo=repo)


def chain_hours(repo=REPO, now=None):
    """Return migration elapsed through failed start plus failed build time."""
    repo = Path(repo)
    if migration.sha(repo / FAILED) != FAILED_SHA256:
        raise RuntimeError('Failed build receipt changed from the reviewed failure')
    failed = _load(repo, FAILED)
    started, _ = _validate_failed(failed, repo)
    migration_allocation = _load(repo, MIGRATION_ALLOCATION)
    if (migration_allocation.get('status') != 'PASSED'
            or migration_allocation.get('pod_id') != TARGET
            or migration_allocation.get('migration_ready_sha256') != migration.sha(repo / READY)):
        raise RuntimeError('Migrated allocation is not bound to the verified target')
    for name in migration.CHAIN:
        if migration_allocation.get('continuation_of', {}).get(name) != migration.sha(repo / name):
            raise RuntimeError('Migrated allocation chain changed: ' + name)
    # The failed build binds its preceding migration allocation through the
    # dedicated allocation_sha256 field; its continuation map binds the
    # migration evidence itself.
    for name in migration.CHAIN:
        if failed.get('continuation_of', {}).get(name) != migration.sha(repo / name):
            raise RuntimeError('Failed build chain changed: ' + name)
    return cumulative_hours(migration.chain_hours(repo, now=started), failed,
                             now=now, repo=repo)


def validate_target_state(target, capacity):
    """Require the exact stopped retained target and a free matching A100."""
    if (target.get('id') != TARGET or target.get('desiredStatus') != 'EXITED'
            or target.get('gpuCount') != 1 or target.get('volumeInGb') != 500
            or target.get('containerDiskInGb') != 30
            or target.get('volumeMountPath') != '/home/dylan'):
        raise RuntimeError('Retry target identity or storage changed')
    if (capacity.get('id') != TARGET or capacity.get('desiredStatus') != 'EXITED'
            or capacity.get('gpuCount') != 1
            or (capacity.get('machine') or {}).get('gpuTypeId') != 'NVIDIA A100-SXM4-80GB'
            or _finite((capacity.get('machine') or {}).get('gpuAvailable'), 'GPU capacity') < 1):
        raise RuntimeError('Retry target lacks exact free A100 capacity')


def _write(path, result):
    path = Path(path)
    if path.exists():
        raise RuntimeError('Existing retry allocation must be preserved')
    path.write_text(json.dumps(result, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    args = parser.parse_args(argv)
    repo = args.repo
    output = repo / OUTPUT.relative_to(REPO)
    if output.exists():
        raise RuntimeError('Existing retry allocation must be preserved')
    import m13_resume_allocation as recovery
    import m13_after_cpu_allocation as after_cpu
    if migration.sha(repo / FAILED) != FAILED_SHA256:
        raise RuntimeError('Failed build receipt changed from the reviewed failure')
    failed = _load(repo, FAILED)
    started, finished = _validate_failed(failed, repo)
    migration_allocation = _load(repo, MIGRATION_ALLOCATION)
    pods, balance = recovery.live()
    by_id = {pod.get('id'): pod for pod in pods}
    if any(pod.get('desiredStatus') != 'EXITED' for pod in pods):
        raise RuntimeError('Require all retained Pods stopped before retry allocation')
    target_capacity, _ = _live_target_capacity()
    validate_target_state(by_id.get(TARGET, {}), target_capacity)
    quote = _finite(by_id[TARGET].get('costPerHr'), TARGET + ' GPU quote')
    if quote <= 0 or quote + 530 * .1 / 720 > migration.TARGET_PRICE + 1e-9:
        raise RuntimeError('Retry target all-in quote exceeds ceiling')
    prior = chain_hours(repo, now=time.time())
    original = _load(repo, recovery.ORIGINAL)
    measured = time.time()
    result = dict(original)
    result.update(recovery.remaining(original, _load(repo, recovery.ATTEMPT),
                                     _load(repo, recovery.PAUSE), balance,
                                     continuation_hours=prior))
    result.update(status='PASSED', measured_utc=measured,
                  prerequisite_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  allocation_script_sha256=migration.sha(__file__),
                  continuation_of={name: migration.sha(repo / name) for name in CHAIN},
                  prior_continuation_hours=prior,
                  migration_measured_at=measured,
                  failed_build_started_at=started,
                  failed_build_finished_at=finished,
                  failed_build_duration_hours=failed_hours(failed, repo=repo),
                  failed_build_sha256=migration.sha(repo / FAILED),
                  migration_allocation_sha256=migration.sha(repo / MIGRATION_ALLOCATION),
                  migration_ready_sha256=migration.sha(repo / READY),
                  pod_id=TARGET, source_id=SOURCE, target_id=TARGET,
                  ssh_config=str(SSH_CONFIG), ssh_alias=SSH_ALIAS,
                  training_reserve_hours=migration.TRAINING_RESERVE_HOURS,
                  limitations='Original interruption, migration lease, failed preflight-only migrated supervisor through confirmed STOP, and all observed balance charges are deducted. Same retained target; no recipe or new Pod.')
    _write(output, result)
    print(json.dumps({key: result[key] for key in
                      ('status', 'max_hours', 'max_cost_usd', 'account_balance_usd',
                       'projected_total_usd')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
