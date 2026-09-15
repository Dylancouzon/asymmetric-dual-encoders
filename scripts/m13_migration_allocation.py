#!/usr/bin/env python3
"""Pure budget and receipt checks for the reviewed retained-Pod migration.

This module performs no provider calls and no mutations.  Callers supply the
already-observed balance, Pod state, and immutable migration receipt.
"""
import argparse
import hashlib
import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path

ORIGINAL_PRICE = 1.6636111111111112
SOURCE_PRICE = 0.8686111111111111
TARGET_PRICE = ORIGINAL_PRICE
SOURCE = 'wnzk8eeqrrkw4m'
TARGET = 'k3aee2m68765em'
READY = 'results/m13_migration_ready.json'
OUTPUT = Path('/home/dylan/asymetric-dual-encoders/work/m13cloud/results/m13_migration_after_allocation.json')
PRE_OUTPUT = Path('/home/dylan/asymetric-dual-encoders/work/m13cloud/results/m13_migration_before_allocation.json')
SSH_CONFIG = Path('/home/dylan/.config/runpod/m13_ssh_config')
SSH_ALIAS = 'm13-runpod'
TARGET_SSH_CONFIG = SSH_CONFIG
TARGET_SSH_ALIAS = SSH_ALIAS
# Kept as an explicit public chain for allocation/builder receipt binding.  The
# source/target handoff receipt is appended after the earlier after-CPU chain.
CHAIN = (
    'results/m13_build_allocation.json',
    'results/m13_cloud_build.json',
    'results/m13_reboot_pause.json',
    'results/m13_build_resume_allocation.json',
    'results/m13_cloud_build_resume.json',
    'results/m13_cpu_upload.json',
    'results/m13_storage_upload.json',
    'results/m13_storage_verification.json',
    READY,
)
MIGRATION_WALL_CAP_HOURS = 1.0
STAGE_HOURS = 144.0
STAGE_DOLLARS = 239.56
TRAINING_HOURS = 200000000 / 441.9 / 3600
TRAINING_RESERVE_HOURS = TRAINING_HOURS + 4.0


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError('Invalid ' + label)
    return value


def sha(path):
    """Hash a local receipt or manifest without following any protected data."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _original(original):
    if (original.get('max_hours') != STAGE_HOURS
            or original.get('max_cost_usd') != STAGE_DOLLARS
            or original.get('funding_baseline_usd') != 505
            or not math.isclose(original['account_balance_usd']
                                + original['spent_to_date_usd'], 505,
                                abs_tol=1e-6)
            or original.get('total_price_usd_per_hour') != ORIGINAL_PRICE):
        raise RuntimeError('Original reviewed allowance changed')
    return original


def stage_remaining(original, balance, equivalent_elapsed_hours):
    """Return the original stage remainder after equivalent elapsed charges."""
    _original(original)
    balance = _finite(balance, 'balance')
    if not 0 <= balance <= original['account_balance_usd']:
        raise RuntimeError('Funding changed or invalid balance')
    elapsed = _finite(equivalent_elapsed_hours, 'equivalent elapsed duration')
    if not 0 <= elapsed < STAGE_HOURS:
        raise RuntimeError('Stage duration exhausted or invalid')
    paid = original['account_balance_usd'] - balance
    consumed = max(paid, elapsed * ORIGINAL_PRICE)
    raw_seconds = min(STAGE_HOURS - elapsed,
                      (STAGE_DOLLARS - consumed) / ORIGINAL_PRICE) * 3600
    seconds = math.floor(raw_seconds)
    hours = seconds / 3600
    cap = hours * ORIGINAL_PRICE
    projected = original['projected_total_usd'] + consumed
    if hours < TRAINING_RESERVE_HOURS or consumed > STAGE_DOLLARS \
            or cap > balance \
            or projected > 1000 or cap > 1000 - (505 - balance):
        raise RuntimeError('Remaining stage cannot preserve training reserve')
    return {
        'max_hours': hours,
        'max_cost_usd': cap,
        'equivalent_elapsed_hours': elapsed,
        'paid_since_original_allocation_usd': paid,
        'stage_consumed_usd': consumed,
        'projected_total_usd': math.ceil(projected * 100) / 100,
        'remaining_headroom_usd': math.floor((1000 - projected) * 100) / 100,
        'account_balance_usd': balance,
        'spent_to_date_usd': 505 - balance,
        'remaining_budget_usd': 1000 - (505 - balance),
    }


def _interval(start, finish, label, maximum=None):
    start = _finite(start, label + ' start')
    finish = _finite(finish, label + ' finish')
    hours = (finish - start) / 3600
    if not math.isfinite(hours) or hours < 0 or (maximum is not None and hours > maximum):
        raise RuntimeError('Invalid ' + label + ' interval')
    return hours


def migration_cost(source_hours, target_hours, source_price=SOURCE_PRICE,
                   target_price=TARGET_PRICE):
    """Convert concurrent source/target migration billing to stage hours."""
    source_hours = _finite(source_hours, 'source migration duration')
    target_hours = _finite(target_hours, 'target migration duration')
    if not 0 <= source_hours <= MIGRATION_WALL_CAP_HOURS \
            or not 0 <= target_hours <= MIGRATION_WALL_CAP_HOURS:
        raise RuntimeError('Migration exceeds one-hour bound')
    source_price = _finite(source_price, 'source price')
    target_price = _finite(target_price, 'target price')
    if source_price < 0 or target_price < 0 \
            or source_price > ORIGINAL_PRICE + 1e-9 \
            or target_price > ORIGINAL_PRICE + 1e-9:
        raise RuntimeError('Migration price exceeds conservative ceiling')
    wall = max(source_hours, target_hours)
    dollars = source_hours * source_price + target_hours * target_price
    return {
        'source_hours': source_hours,
        'target_hours': target_hours,
        'wall_hours': wall,
        'aggregate_cost_usd': dollars,
        'equivalent_hours': max(wall, dollars / ORIGINAL_PRICE),
        'source_price_usd_per_h': source_price,
        'target_price_usd_per_h': target_price,
    }


def migration_hours(ready, now=None):
    """Compute migration plus target-lease equivalent hours from a ready receipt.

    ``ready`` contains source_started_at/source_finished_at and
    target_started_at.  target_finished_at may be omitted while its lease is
    active; ``now`` then closes that interval for a read-only admission check.
    """
    if now is None:
        now = time.time()
    source_hours = _interval(ready['source_started_at'],
                             ready['source_finished_at'], 'source migration',
                             MIGRATION_WALL_CAP_HOURS)
    # Billing starts when the target is resumed.  The target lease receipt may
    # use a later handoff marker, but that marker cannot erase already billed
    # target time.  Keep the entire concurrent operation within the reviewed
    # one-hour wall bound, including the 20-minute claim lease.
    target_hours = _interval(ready['target_started_at'],
                             ready.get('target_finished_at', now),
                             'target lease', MIGRATION_WALL_CAP_HOURS)
    if target_hours + 1e-9 < source_hours:
        raise RuntimeError('Target lease ended before source migration')
    return migration_cost(source_hours, target_hours,
                          ready.get('source_price_usd_per_h',
                                    ready.get('source_price_ceiling_usd_h', SOURCE_PRICE)),
                          ready.get('target_price_usd_per_h',
                                    ready.get('target_price_ceiling_usd_h', TARGET_PRICE)))


def chain_hours(repo=Path('/home/dylan/asymetric-dual-encoders/work/m13cloud'), now=None):
    """Return prior after-CPU hours plus migration/target lease equivalent hours.

    The earlier chain validator is imported lazily so this module remains a
    pure helper and can be used by its tests without provider access.
    """
    import json
    import sys
    scripts = str(Path(repo) / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import m13_after_cpu_allocation as after_cpu
    prior = after_cpu.chain_hours(repo)
    ready_path = Path(repo) / READY
    ready = json.loads(ready_path.read_text())
    validate_ready(ready, now=now)
    # The handoff receipt is immutable evidence.  Validate the two inputs it
    # claims to have copied before using its elapsed time in the budget chain.
    if ready.get('source_receipt_sha256') != sha(Path(repo) / 'results/m13_storage_verification.json'):
        raise RuntimeError('Migration source verification binding changed')
    manifest = Path(repo) / 'm13/build_transfer_manifest.json'
    manifest_sha = sha(manifest)
    if ready.get('manifest_sha256', ready.get('destination_sha256')) != manifest_sha:
        raise RuntimeError('Migration manifest binding changed')
    return prior + migration_hours(ready, now)['equivalent_hours']


def validate_before_state(source, target, capacity):
    """Require stopped source/retained disks and a free exact target A100."""
    if source.get('id') != SOURCE or source.get('desiredStatus') != 'EXITED':
        raise RuntimeError('Source Pod is not the stopped approved Pod')
    if target.get('id') != TARGET or target.get('desiredStatus') != 'EXITED':
        raise RuntimeError('Target Pod is not the stopped retained Pod')
    for pod, label in ((source, 'source'), (target, 'target')):
        if (pod.get('volumeInGb') != 500 or pod.get('containerDiskInGb') != 30
                or pod.get('volumeMountPath') != '/home/dylan'):
            raise RuntimeError(label + ' persistent disk identity changed')
    if isinstance(capacity, dict):
        count = capacity.get('gpuAvailable', capacity.get('gpu_available'))
    else:
        count = capacity
    count = _finite(count, 'target GPU capacity')
    if count < 1:
        raise RuntimeError('No free target GPU')


def validate_ready(ready, now=None):
    """Require the immutable post-migration source/target handoff receipt."""
    source_id = ready.get('source_id', ready.get('source_pod_id'))
    target_id = ready.get('target_id', ready.get('target_pod_id'))
    source_status = ready.get('source_status', ready.get('source_pod_final_status'))
    target_status = ready.get('target_status', ready.get('target_pod_final_status'))
    if (ready.get('status') != 'PASSED'
            or source_id != SOURCE
            or target_id != TARGET
            or source_status != 'EXITED'
            or target_status != 'RUNNING'
            or ready.get('target_gpu_count') != 1
            or ready.get('target_model') not in ('A100 SXM80GB', 'A100-SXM4-80GB',
                                                  'A100SXM80GB', 'NVIDIA A100-SXM4-80GB')
            or ready.get('verified_files') != 394
            or ready.get('transfer_verified') is not True
            or ready.get('source_price_usd_per_h',
                         ready.get('source_price_ceiling_usd_h', SOURCE_PRICE)) < 0
            or ready.get('target_price_usd_per_h',
                         ready.get('target_price_ceiling_usd_h', TARGET_PRICE)) > ORIGINAL_PRICE + 1e-9
            or not ready.get('source_receipt_sha256')
            or not (ready.get('manifest_sha256') or ready.get('destination_sha256'))):
        raise RuntimeError('Migration handoff receipt is not admitted')
    if (ready.get('source_volume_in_gb') != 500
            or ready.get('target_volume_in_gb') != 500
            or ready.get('target_container_disk_in_gb') != 30
            or ready.get('target_volume_mount_path') != '/home/dylan'):
        raise RuntimeError('Migration handoff persistent disk changed')
    migration_hours(ready, ready.get('target_finished_at',
                                     time.time() if now is None else now))


def before_allocation(original, balance, prior_hours, source_price=SOURCE_PRICE,
                     target_price=TARGET_PRICE):
    """Allocate the bounded concurrent one-hour source/target migration."""
    migration = migration_cost(1.0, 1.0, source_price, target_price)
    result = stage_remaining(original, balance,
                             _finite(prior_hours, 'prior elapsed')
                             + migration['equivalent_hours'])
    result['prior_elapsed_hours'] = result.pop('equivalent_elapsed_hours')
    result.update(status='PASSED', migration=migration,
                  source_id=SOURCE, target_id=TARGET,
                  pod_id=TARGET, ssh_config=str(SSH_CONFIG), ssh_alias=SSH_ALIAS,
                  training_reserve_hours=TRAINING_RESERVE_HOURS,
                  limitations='One-hour concurrent source/target migration, charged at aggregate equivalent original-price hours; no new Pod or recipe change.')
    return result


def after_allocation(original, balance, prior_hours, ready, now=None):
    """Allocate the target GPU build after the source is stopped."""
    validate_ready(ready)
    migration = migration_hours(ready, now)
    result = stage_remaining(original, balance,
                             _finite(prior_hours, 'prior elapsed')
                             + migration['equivalent_hours'])
    result['prior_elapsed_hours'] = result.pop('equivalent_elapsed_hours')
    measured = time.time() if now is None else float(now)
    result.update(status='PASSED', migration=migration,
                  source_id=SOURCE, target_id=TARGET,
                  pod_id=TARGET, ssh_config=str(SSH_CONFIG), ssh_alias=SSH_ALIAS,
                  ready_sha256=ready.get('receipt_sha256'),
                  migration_ready_sha256=ready.get('receipt_sha256'),
                  migration_measured_at=measured,
                  prior_continuation_hours=result['prior_elapsed_hours'],
                  training_reserve_hours=TRAINING_RESERVE_HOURS,
                  limitations='Cumulative source interval and target lease charged at conservative equivalent original-price hours; source stopped, exact target retained, no new Pod or recipe change.')
    return result


def _load(repo, name):
    return json.loads((Path(repo) / name).read_text())


def _remaining(repo, balance, continuation_hours):
    """Use the reviewed recovery ledger so CLI output has its exact schema."""
    scripts = str(Path(repo) / 'scripts')
    import sys
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import m13_resume_allocation as recovery
    original = _load(repo, recovery.ORIGINAL)
    attempt = _load(repo, recovery.ATTEMPT)
    pause = _load(repo, recovery.PAUSE)
    return recovery.remaining(original, attempt, pause, balance,
                             continuation_hours=continuation_hours)


def _live_target_capacity():
    """Read target capacity through the existing read-only GraphQL query."""
    scripts = str(Path(__file__).resolve().parent)
    import sys
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import m13_resume_allocation as recovery
    headers = {'Authorization': 'Bearer ' + (recovery.KEYS / 'api_key').read_text().strip(),
               'Content-Type': 'application/json', 'User-Agent': 'm13-migration-allocation/1.0'}
    query = 'query { myself { pods { id desiredStatus gpuCount machine { gpuAvailable gpuTypeId } } } }'
    request = urllib.request.Request('https://api.runpod.io/graphql', headers=headers,
                                     data=json.dumps({'query': query}).encode())
    with urllib.request.urlopen(request, timeout=30) as response:
        rows = json.load(response)['data']['myself']['pods']
    row = next((item for item in rows if item.get('id') == TARGET), None)
    if row is None:
        raise RuntimeError('Target Pod is absent from capacity response')
    return row, (row.get('machine') or {}).get('gpuAvailable')


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Existing migration allocation must be preserved')
    path.write_text(json.dumps(data, indent=2) + '\n')


def _before(repo):
    """Create the pre-migration allocation after read-only state checks."""
    repo = Path(repo)
    pre_output = repo / PRE_OUTPUT.relative_to(Path('/home/dylan/asymetric-dual-encoders/work/m13cloud')) \
        if PRE_OUTPUT.is_absolute() else repo / PRE_OUTPUT
    if pre_output.exists():
        raise RuntimeError('Existing pre-migration allocation must be preserved')
    scripts = str(repo / 'scripts')
    import sys
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import m13_resume_allocation as recovery
    import m13_after_cpu_allocation as after_cpu
    pods, balance = recovery.live()
    by_id = {p.get('id'): p for p in pods}
    if any(p.get('desiredStatus')!='EXITED' for p in pods):raise RuntimeError('All retained Pods must be stopped')
    target_capacity, count = _live_target_capacity()
    recovery_original = _load(repo, recovery.ORIGINAL)
    validate_before_state(by_id.get(SOURCE, {}), by_id.get(TARGET, {}), count)
    if (target_capacity.get('machine') or {}).get('gpuTypeId')!='NVIDIA A100-SXM4-80GB':
        raise RuntimeError('Target GPU model changed')
    if not 0 < float(by_id[TARGET]['costPerHr']) <= 1.59:raise RuntimeError('Target quote exceeds admitted GPU rate')
    if target_capacity.get('desiredStatus') != 'EXITED':
        raise RuntimeError('Target must be stopped before migration allocation')
    prior = after_cpu.chain_hours(repo)
    storage_price = 530 * .1 / 720
    source_total = max(SOURCE_PRICE,
                       _finite(by_id[SOURCE].get('costPerHr'), 'source GPU quote') + storage_price)
    target_total = max(TARGET_PRICE,
                       _finite(by_id[TARGET].get('costPerHr'), 'target GPU quote') + storage_price)
    migration = migration_cost(1.0, 1.0, source_total, target_total)
    continuation = prior + migration['equivalent_hours']
    result = dict(recovery_original)
    result.update(_remaining(repo, balance, continuation))
    result.update(status='PASSED', measured_utc=time.time(),
                  prerequisite_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  allocation_script_sha256=sha(__file__),
                  continuation_of={name: sha(repo / name) for name in CHAIN[:-1]},
                  prior_continuation_hours=continuation, migration=migration,
                  migration_measured_at=None, source_id=SOURCE, target_id=TARGET,
                  pod_id=TARGET, ssh_config=str(SSH_CONFIG), ssh_alias=SSH_ALIAS,
                  training_reserve_hours=TRAINING_RESERVE_HOURS,
                  limitations='One-hour concurrent source/target migration; target capacity and both stopped Pods checked immediately before dispatch.')
    _write(pre_output, result)
    return result


def _after(repo):
    """Create the post-handoff GPU allocation, charging the live target lease."""
    repo = Path(repo)
    scripts = str(repo / 'scripts')
    import sys
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import m13_resume_allocation as recovery
    import m13_after_cpu_allocation as after_cpu
    ready = _load(repo, READY)
    validate_ready(ready)
    pods, balance = recovery.live()
    by_id = {p.get('id'): p for p in pods}
    if any(p.get('desiredStatus') != ('RUNNING' if p.get('id') == TARGET else 'EXITED') for p in pods):
        raise RuntimeError('Only the target Pod may be running at handoff')
    target = by_id.get(TARGET, {})
    if (target.get('volumeInGb') != 500 or target.get('containerDiskInGb') != 30
            or target.get('volumeMountPath') != '/home/dylan'):
        raise RuntimeError('Target storage identity changed at handoff')
    prior = after_cpu.chain_hours(repo)
    measured = time.time()
    continuation = chain_hours(repo, now=measured)
    original = _load(repo, recovery.ORIGINAL)
    result = dict(original)
    result.update(_remaining(repo, balance, continuation))
    result.update(status='PASSED', measured_utc=measured,
                  prerequisite_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  allocation_script_sha256=sha(__file__),
                  continuation_of={name: sha(repo / name) for name in CHAIN},
                  prior_continuation_hours=continuation,
                  migration_measured_at=measured,
                  migration_ready_sha256=sha(repo / READY),
                  source_id=SOURCE, target_id=TARGET, pod_id=TARGET,
                  ssh_config=str(SSH_CONFIG), ssh_alias=SSH_ALIAS,
                  training_reserve_hours=TRAINING_RESERVE_HOURS,
                  limitations='Source verification, migration interval and every second of the target lease charged at conservative equivalent original-price hours; target is the sole running retained A100 Pod.')
    if result.get('prior_continuation_hours') != continuation or continuation < prior:
        raise RuntimeError('Migration continuation arithmetic changed during write')
    output = repo / OUTPUT.relative_to(Path('/home/dylan/asymetric-dual-encoders/work/m13cloud')) \
        if OUTPUT.is_absolute() else repo / OUTPUT
    _write(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--before', action='store_true', help='write the pre-migration allocation')
    mode.add_argument('--after', action='store_true', help='write the post-handoff GPU allocation')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    return _before(args.repo) if args.before else _after(args.repo)


if __name__ == '__main__':
    main()
