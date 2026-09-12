#!/usr/bin/env python3
"""Cloud-to-cloud transfer of the pinned M13 build inputs.

The source Pod is resumed with zero GPUs and the target Pod with one GPU.  The
target pulls the exact manifest entries directly from the source over an
ephemeral, forced-command rsync key; build inputs never pass through the local
machine.  This supervisor is bounded, records a migration lease, and leaves
the target running only after a reviewed build controller claims that lease.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
KEYS = Path('/home/dylan/.config/runpod')
SOURCE = 'wnzk8eeqrrkw4m'
TARGET = 'k3aee2m68765em'
REMOTE = '/home/dylan/asymetric-dual-encoders'
MANIFEST = REPO / 'm13/build_transfer_manifest.json'
SOURCE_RECEIPT = REPO / 'results/m13_storage_verification.json'
PRIOR_RECEIPT = REPO / 'results/m13_after_verification.json'
PRE_ALLOCATION = REPO / 'results/m13_migration_before_allocation.json'
ALLOCATION_SCRIPT = REPO / 'scripts/m13_migration_allocation.py'
RESULT = REPO / 'results/m13_migration.json'
READY = REPO / 'results/m13_migration_ready.json'
HANDOFF = REPO / 'results/m13_migration_handoff.json'
FAILURE = REPO / 'results/m13_migration_failure.json'
LOG = REPO / 'logs/m13-migrate-inputs.log'
PIDFILE = REPO / 'work/m13cloud-launchers/migrate_inputs.pid'
SOURCE_CONFIG = KEYS / 'm13_gate_chain_ssh_config'
SOURCE_ALIAS = 'm13-gate-chain'
TARGET_CONFIG = KEYS / 'm13_ssh_config'
TARGET_ALIAS = 'm13-runpod'
SOURCE_RECEIPT_SHA = None  # Bound in the receipt at invocation; never overwritten.
STORAGE_PRICE = 530 * 0.1 / 720
SOURCE_ALL_IN_CEILING = 0.8686111111111111
TARGET_ALL_IN_CEILING = 1.6636111111111112


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(argv, timeout=60, **kwargs):
    return subprocess.run(argv, check=True, timeout=timeout, **kwargs)


def update_ssh_endpoint(config, pod):
    """Bind the reviewed SSH config to the endpoint returned after resume."""
    host = pod.get('publicIp')
    port = (pod.get('portMappings') or {}).get('22')
    if not host or not port:
        raise RuntimeError('Pod has no SSH endpoint')
    lines = config.read_text().splitlines()
    if not any(line.strip().startswith('HostName ') for line in lines):
        raise RuntimeError('SSH config has no HostName binding')
    if not any(line.strip().startswith('Port ') for line in lines):
        raise RuntimeError('SSH config has no Port binding')
    config.write_text('\n'.join(
        '  HostName ' + str(host) if line.strip().startswith('HostName ') else
        '  Port ' + str(port) if line.strip().startswith('Port ') else line
        for line in lines) + '\n')


def pod_model(pod):
    value = pod.get('gpuTypeId') or pod.get('gpuType') or pod.get('machineType')
    if isinstance(value, dict):
        value = value.get('name') or value.get('id')
    if not isinstance(value, str):
        raise RuntimeError('Target GPU model is absent')
    normalized = re.sub(r'[^A-Z0-9]', '', value.upper())
    if 'A100' not in normalized or '80GB' not in normalized:
        raise RuntimeError('Target is not the admitted A100 80GB model')
    return value


def capacity_rows(request):
    payload = {'query': ('query { myself { pods { id desiredStatus gpuCount '
                         'machine { gpuAvailable gpuTypeId } } } }')}
    result = request('https://api.runpod.io/graphql', payload)
    if result.get('errors'):
        raise RuntimeError('Capacity query returned errors')
    return {row.get('id'): row for row in (result.get('data') or {}).get('myself', {}).get('pods', [])}


def all_in_price(pod):
    try:
        compute = float(pod['costPerHr'])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError('Pod compute quote is absent') from error
    total = compute + STORAGE_PRICE
    if not math.isfinite(total):
        raise RuntimeError('Pod all-in quote is invalid')
    return total


def validate_manifest(data):
    """Return root/tree files and destination hashes from the committed pin."""
    entries = data.get('entries')
    if data.get('status') != 'HASH_PINNED_NOT_LAUNCH_APPROVAL' or data.get('files') != 394:
        raise RuntimeError('Unexpected transfer manifest')
    if not isinstance(entries, list) or len(entries) != 394:
        raise RuntimeError('Require exactly 394 transfer entries')
    root, tree, expected, sources = [], [], {}, set()
    for item in entries:
        if not isinstance(item, dict):
            raise RuntimeError('Malformed transfer entry')
        source, destination = Path(item.get('source', '')), Path(item.get('destination', ''))
        digest = item.get('sha256')
        if (not source.is_absolute() or not destination.is_absolute() or
                '..' in source.parts or '..' in destination.parts or
                str(source) in sources or str(destination) in expected):
            raise RuntimeError('Invalid or duplicate transfer path')
        if ('work/lotte' in str(destination) or 'work/m9reserve' in str(destination) or
                'frozen_eval' in str(destination) or destination.name in
                ('perquery.json', 'cqadup-android.json', 'cqadup-english.json')):
            raise RuntimeError('Protected payload in transfer manifest')
        if source.is_relative_to(REPO):
            rel = source.relative_to(REPO)
            if destination != Path(REMOTE) / rel:
                raise RuntimeError('Worktree destination mapping changed')
            tree.append(str(rel))
        else:
            if destination != source or not source.is_relative_to('/home/dylan'):
                raise RuntimeError('Root destination mapping changed')
            root.append(str(source).lstrip('/'))
        if (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) or
                not source.is_file() or source.stat().st_size != item.get('bytes') or
                sha(source) != digest):
            raise RuntimeError('Local source differs from transfer pin: ' + str(source))
        sources.add(str(source))
        expected[str(destination)] = digest
    if len(root) + len(tree) != 394 or len(expected) != 394:
        raise RuntimeError('Manifest partition is not exactly 394 files')
    return {'root': root, 'tree': tree}, expected


def forced_rsync_helper():
    """Source-side helper: permit only rsync sender commands."""
    return '''#!/usr/bin/env python3
import os, shlex, sys
command = os.environ.get("SSH_ORIGINAL_COMMAND", "")
try:
    parts = shlex.split(command)
except ValueError:
    raise SystemExit(2)
if len(parts) < 4 or parts[0] != "rsync" or "--server" not in parts or "--sender" not in parts:
    raise SystemExit(2)
if any(flag in parts for flag in ("--delete", "--remove-source-files", "--backup")):
    raise SystemExit(2)
if parts[-1] not in (".", "/", "/home/dylan/asymetric-dual-encoders/"):
    raise SystemExit(2)
os.execvp(parts[0], parts)
'''


def validate_forced_command(command):
    """Pure testable equivalent of the source forced-command policy."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return (len(parts) >= 4 and parts[0] == 'rsync' and '--server' in parts and
            '--sender' in parts and not any(x in parts for x in
            ('--delete', '--remove-source-files', '--backup')) and
            parts[-1] in ('/', '/home/dylan/asymetric-dual-encoders/'))


def read_ssh_config(config, alias):
    raw = subprocess.check_output(['ssh', '-G', '-F', str(config), alias], text=True, timeout=30)
    values = {}
    for line in raw.splitlines():
        key, _, value = line.partition(' ')
        if key in ('user', 'hostname', 'port', 'identityfile') and key not in values:
            values[key] = value.strip()
    if not all(values.get(k) for k in ('user', 'hostname', 'port')):
        raise RuntimeError('SSH config lacks endpoint identity')
    return values


def api_client():
    token = (KEYS / 'api_key').read_text().strip()
    if not token:
        raise RuntimeError('Runpod API key is empty')
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
               'User-Agent': 'm13-input-migration/1.0'}

    def request(url, body=None, method=None):
        request_obj = urllib.request.Request(url, headers=headers,
            data=None if body is None else json.dumps(body).encode(), method=method)
        with urllib.request.urlopen(request_obj, timeout=30) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}

    def pod(pod_id):
        return request('https://rest.runpod.io/v1/pods/' + pod_id)

    def stop(pod_id):
        request('https://rest.runpod.io/v1/pods/' + pod_id + '/stop', method='POST')

    return headers, request, pod, stop


def resume(request, pod_id, gpu_count):
    if gpu_count == 0:
        operation = 'podResumeZeroGpu'
        fields = 'id gpuCount'
        options = 'podId: "' + pod_id + '"'
    else:
        operation = 'podResume'
        fields = 'id gpuCount'
        options = 'podId: "' + pod_id + '", gpuCount: ' + str(gpu_count)
    payload = {'query': 'mutation { ' + operation + '(input: {' + options +
               '}) {' + fields + '} }'}
    result = request('https://api.runpod.io/graphql', payload)
    row = (result.get('data') or {}).get(operation) or {}
    if result.get('errors') or row.get('id') != pod_id or row.get('gpuCount') != gpu_count:
        raise RuntimeError('Resume did not return exact Pod/GPU count')
    return row


def pin_host_key(ssh_values, ssh_cmd):
    """Read the public host key through authenticated SSH, then pin its endpoint."""
    raw = subprocess.check_output(ssh_cmd + ['cat /etc/ssh/ssh_host_ed25519_key.pub'],
                                  text=True, timeout=30).strip().splitlines()
    if len(raw) != 1:
        raise RuntimeError('Unexpected source host key response')
    fields = raw[0].split()
    if len(fields) < 2 or fields[0] != 'ssh-ed25519' or not re.fullmatch(r'[A-Za-z0-9+/=]+', fields[1]):
        raise RuntimeError('Invalid source host key')
    return '[' + ssh_values['hostname'] + ']:' + ssh_values['port'] + ' ' + fields[0] + ' ' + fields[1] + '\n'


def write_receipt(receipt):
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    temp = RESULT.with_suffix('.pending.json')
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    temp.replace(RESULT)


def write_ready(receipt):
    """Publish the immutable source-stop/destination-hash adoption binding."""
    READY.parent.mkdir(parents=True, exist_ok=True)
    temp = READY.with_suffix('.pending.json')
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    os.link(temp,READY)
    temp.unlink()


def terminate_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, default=SOURCE_CONFIG)
    parser.add_argument('--source-alias', default=SOURCE_ALIAS)
    parser.add_argument('--target-config', type=Path, default=TARGET_CONFIG)
    parser.add_argument('--target-alias', default=TARGET_ALIAS)
    args = parser.parse_args()
    if RESULT.exists() or READY.exists() or HANDOFF.exists() or LOG.exists() or PIDFILE.exists():
        raise RuntimeError('Existing migration evidence; refuse automatic retry')
    run(['git', 'diff', '--quiet', 'HEAD'], cwd=REPO)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    pushed = subprocess.check_output(['git', 'ls-remote', '--exit-code', 'origin',
        'refs/heads/m13-stage1-execution-prep'], cwd=REPO, text=True, timeout=30).split()[0]
    if head != pushed:
        raise RuntimeError('Require exact pushed migration source')
    source_receipt_sha = sha(SOURCE_RECEIPT)
    source_receipt = json.loads(SOURCE_RECEIPT.read_text())
    if (source_receipt.get('status') != 'PASSED' or source_receipt.get('transfer_verified') is not True or
            source_receipt.get('verified_files') != 394 or source_receipt.get('pod_final_status') != 'EXITED' or
            source_receipt.get('pod_id') != SOURCE):
        raise RuntimeError('Require passed verified source upload and STOP')
    if not PRIOR_RECEIPT.is_file():
        raise RuntimeError('Missing prior stopped verification evidence')
    prior_receipt_sha = sha(PRIOR_RECEIPT)
    prior_receipt = json.loads(PRIOR_RECEIPT.read_text())
    if prior_receipt.get('status') != 'FAILED' or prior_receipt.get('pod_final_status') not in (None, 'EXITED'):
        raise RuntimeError('Prior verification receipt is not the captured stopped attempt')
    groups, expected = validate_manifest(json.loads(MANIFEST.read_text()))
    source_ssh = read_ssh_config(args.source_config, args.source_alias)
    target_ssh = read_ssh_config(args.target_config, args.target_alias)
    _, request, get_pod, stop = api_client()
    source_started = target_started = False
    source_ssh_cmd = ['ssh', '-n', '-F', str(args.source_config), args.source_alias]
    target_ssh_cmd = ['ssh', '-n', '-F', str(args.target_config), args.target_alias]
    receipt = {'status': 'RUNNING', 'stage': 'starting', 'pid': os.getpid(),
               'source_pod_id': SOURCE, 'target_pod_id': TARGET, 'code_commit': head,
               'source_receipt_sha256': source_receipt_sha, 'manifest_sha256': sha(MANIFEST),
               'prior_verification_receipt_sha256': prior_receipt_sha,
               'started_at': time.time(), 'expected_files': 394}
    write_receipt(receipt)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()) + '\n')
    deadline = time.monotonic() + 3600
    process = None
    ready_published = False

    def save(stage=None):
        if stage:
            receipt['stage'] = stage
        receipt['updated_at'] = time.time()
        write_receipt(receipt)

    def stop_exact(pod_id):
        last = None
        for _ in range(6):
            try:
                stop(pod_id)
                if get_pod(pod_id).get('desiredStatus') == 'EXITED':
                    return
            except Exception as error:
                last = type(error).__name__ + ': ' + str(error)
            time.sleep(3)
        raise RuntimeError('STOP unconfirmed for ' + pod_id + ': ' + str(last))

    def interrupt(sig, frame):
        raise KeyboardInterrupt('Migration interrupted: ' + str(sig))

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    signal.alarm(3600 - 180)
    try:
        source = get_pod(SOURCE); target = get_pod(TARGET)
        if any(p.get('desiredStatus') != 'EXITED' for p in (source, target)):
            raise RuntimeError('Source and target must both be stopped before migration')
        for pod, label in ((source, 'Source'), (target, 'Target')):
            if (pod.get('volumeInGb') != 500 or pod.get('containerDiskInGb') != 30 or
                    pod.get('volumeMountPath') != '/home/dylan'):
                raise RuntimeError(label + ' storage identity changed')
        import m13_migration_allocation as allocation_module
        import m13_resume_allocation as recovery
        import m13_after_cpu_allocation as prior_allocation
        live_rows=capacity_rows(request); target_row=live_rows[TARGET]
        target_model=(target_row.get('machine') or {}).get('gpuTypeId')
        if (target_model!='NVIDIA A100-SXM4-80GB' or target_row.get('gpuCount')!=1
            or target_row.get('desiredStatus')!='EXITED'
            or (target_row.get('machine') or {}).get('gpuAvailable',0)<1):
            raise RuntimeError('No free exact retained A100 target')
        source_quote=float(source['costPerHr'])+STORAGE_PRICE
        if not math.isfinite(source_quote) or not 0<=source_quote<=SOURCE_ALL_IN_CEILING+1e-9:
            raise RuntimeError('Source all-in quote exceeds admission before resume')
        target_gpu_price=float(target['costPerHr'])
        if not math.isfinite(target_gpu_price) or not 0<target_gpu_price<=1.59:
            raise RuntimeError('Target GPU quote exceeds admission')
        source_price=SOURCE_ALL_IN_CEILING; target_price=TARGET_ALL_IN_CEILING
        # The pre-migration allocation is created, reviewed, committed, and
        # pushed before this paid supervisor starts.  Never regenerate it here:
        # doing so would erase the evidence that authorized this dispatch.
        run(['git', 'ls-files', '--error-unmatch', str(PRE_ALLOCATION.relative_to(REPO))],
            cwd=REPO, stdout=subprocess.DEVNULL)
        if not PRE_ALLOCATION.is_file():
            raise RuntimeError('Missing committed pre-migration allocation')
        preallocation = json.loads(PRE_ALLOCATION.read_text())
        if (preallocation.get('status') != 'PASSED' or preallocation.get('source_id') != SOURCE or
                preallocation.get('target_id') != TARGET or preallocation.get('pod_id') != TARGET or
                preallocation.get('migration', {}).get('wall_hours') != 1.0 or
                preallocation.get('allocation_script_sha256') != sha(ALLOCATION_SCRIPT)):
            raise RuntimeError('Pre-migration allocation is not bound to both retained Pods')
        prerequisite = preallocation.get('prerequisite_commit')
        if (not isinstance(prerequisite, str) or len(prerequisite) != 40 or
                any(c not in '0123456789abcdef' for c in prerequisite) or
                subprocess.run(['git', 'merge-base', '--is-ancestor', prerequisite, head],
                               cwd=REPO).returncode != 0):
            raise RuntimeError('Pre-migration allocation prerequisite is not in pushed history')
        run(['git','merge-base','--is-ancestor',preallocation['prerequisite_commit'],head],cwd=REPO)
        for name in allocation_module.CHAIN[:-1]:
            if preallocation.get('continuation_of', {}).get(name) != sha(REPO / name):
                raise RuntimeError('Pre-migration allocation source chain changed: ' + name)
        pods,balance=recovery.live()
        if any(p['desiredStatus']!='EXITED' for p in pods):raise RuntimeError('All retained Pods must be stopped')
        if balance>preallocation['account_balance_usd']:raise RuntimeError('Funding changed after allocation')
        original=json.loads((REPO/recovery.ORIGINAL).read_text())
        for name,digest in original['artifact_sha256'].items():
            if sha(REPO/name)!=digest:raise RuntimeError('Registered input changed')
        planned=preallocation['migration']
        expected_plan=allocation_module.migration_cost(1,1,planned['source_price_usd_per_h'],planned['target_price_usd_per_h'])
        if planned!=expected_plan:raise RuntimeError('Committed migration reserve arithmetic changed')
        full_reserve=prior_allocation.chain_hours(REPO)+planned['equivalent_hours']
        receipt['fresh_migration_reserve']=allocation_module._remaining(REPO,balance,full_reserve)
        receipt.update(target_model=target_model, target_gpu_count=1,
                       source_volume_in_gb=source.get('volumeInGb'), target_volume_in_gb=target.get('volumeInGb'),
                       target_container_disk_in_gb=target.get('containerDiskInGb'),
                       target_volume_mount_path=target.get('volumeMountPath'),
                       source_gpu_price_usd_per_h=SOURCE_ALL_IN_CEILING-STORAGE_PRICE,
                       target_gpu_price_usd_per_h=target_gpu_price,
                       source_price_usd_per_h=source_price, target_price_usd_per_h=target_price,
                       preallocation_sha256=sha(PRE_ALLOCATION))
        save('resuming-target')
        target_started = True; target_started_at = time.time()
        resume(request, TARGET, 1)
        save('resuming-source-zero-gpu')
        source_started = True; source_started_at = time.time()
        resume(request, SOURCE, 0)
        endpoint_deadline = min(deadline - 300, time.monotonic() + 600)
        while time.monotonic() < endpoint_deadline:
            source = get_pod(SOURCE); target = get_pod(TARGET)
            for pod,ceiling in ((source,SOURCE_ALL_IN_CEILING),(target,TARGET_ALL_IN_CEILING)):
                quote=float(pod['costPerHr'])+STORAGE_PRICE
                if not math.isfinite(quote) or not 0<=quote<=ceiling+1e-9:raise RuntimeError('Resumed quote exceeds allocation')
                if pod.get('volumeInGb')!=500 or pod.get('volumeMountPath')!='/home/dylan':raise RuntimeError('Resumed persistent disk changed')
            if target.get('containerDiskInGb')!=30:raise RuntimeError('Target runtime disk changed')
            if source.get('publicIp') and (source.get('portMappings') or {}).get('22') and target.get('publicIp') and (target.get('portMappings') or {}).get('22'):
                break
            time.sleep(5)
        else:
            raise RuntimeError('SSH endpoint readiness timeout')
        # A resumed Pod may receive a new endpoint.  Bind both reviewed configs
        # before any authenticated transfer or host-key pinning.
        update_ssh_endpoint(args.source_config, source)
        update_ssh_endpoint(args.target_config, target)
        source_ssh = read_ssh_config(args.source_config, args.source_alias)
        target_ssh = read_ssh_config(args.target_config, args.target_alias)
        ssh_deadline = min(deadline - 300, time.monotonic() + 180)
        while time.monotonic() < ssh_deadline:
            source_ok = subprocess.run(source_ssh_cmd + ['true'], timeout=20,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            target_ok = subprocess.run(target_ssh_cmd + ['true'], timeout=20,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if source_ok and target_ok:
                break
            time.sleep(5)
        else:
            raise RuntimeError('SSH authentication readiness timeout')
        target_row=capacity_rows(request)[TARGET]
        if (target_row.get('gpuCount')!=1 or target_row.get('desiredStatus')!='RUNNING'
            or (target_row.get('machine') or {}).get('gpuTypeId')!='NVIDIA A100-SXM4-80GB'):
            raise RuntimeError('Resumed target is not exact A100 GPU')
        for command in (source_ssh_cmd,target_ssh_cmd):
            run(command+['set -eu; mountpoint -q /home/dylan; if ! command -v rsync >/dev/null; then apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rsync; fi'],timeout=300)
        receipt['target_started_at'] = target_started_at
        receipt['source_started_at'] = source_started_at
        source_key_active = False
        with tempfile.TemporaryDirectory(prefix='m13-migration-') as temporary:
            temp = Path(temporary)
            key = temp / 'source_pull_ed25519'
            run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(key)], timeout=30)
            public = key.with_suffix('.pub').read_text().strip()
            helper = temp / 'm13-rsync-readonly.py'; helper.write_text(forced_rsync_helper())
            run(['scp', '-F', str(args.source_config), str(helper), args.source_alias + ':/tmp/m13-rsync-readonly.py'], timeout=60)
            auth = ('command="python3 /tmp/m13-rsync-readonly.py",no-agent-forwarding,'
                    'no-port-forwarding,no-X11-forwarding,no-pty,no-user-rc ' + public + ' m13-migration')
            install = ('umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; '
                       'grep -Fqx -- ' + shlex.quote(auth) + ' ~/.ssh/authorized_keys || '
                       'printf "%s\\n" ' + shlex.quote(auth) + ' >> ~/.ssh/authorized_keys')
            run(source_ssh_cmd + [install], timeout=60)
            source_key_active = True
            remove = ('sed -i -e ' + shlex.quote(r'/^command="python3 \/tmp\/m13-rsync-readonly.py".*m13-migration$/d') +
                      ' ~/.ssh/authorized_keys; rm -f /tmp/m13-rsync-readonly.py')
            try:
                host_line = pin_host_key(source_ssh, source_ssh_cmd)
                known = temp / 'known_hosts'; known.write_text(host_line)
                target_key = '/tmp/m13-source-pull-ed25519'
                target_known = '/tmp/m13-source-known-hosts'
                run(['scp', '-F', str(args.target_config), str(key), args.target_alias + ':' + target_key], timeout=60)
                run(['scp', '-F', str(args.target_config), str(known), args.target_alias + ':' + target_known], timeout=60)
                target_manifest = temp / 'manifest.json'; target_manifest.write_text(MANIFEST.read_text())
                run(['scp', '-F', str(args.target_config), str(target_manifest), args.target_alias + ':/tmp/m13-migrate-manifest.json'], timeout=60)
                for label in ('root', 'tree'):
                    listing = temp / (label + '.files'); listing.write_text('\n'.join(groups[label]) + '\n')
                    run(['scp', '-F', str(args.target_config), str(listing), args.target_alias + ':/tmp/m13-migrate-' + label + '.files'], timeout=60)
                source_host = source_ssh['user'] + '@' + source_ssh['hostname']
                source_ssh_opts = ('-i ' + shlex.quote(target_key) + ' -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes '
                                   '-o UserKnownHostsFile=' + shlex.quote(target_known) + ' -p ' + shlex.quote(source_ssh['port']))
                commands = []
                for label, source_root, destination, copy_links in (
                    ('root', '/', '/', False), ('tree', REMOTE + '/', REMOTE + '/', True)):
                    links = '--copy-links ' if copy_links else ''
                    commands.append('rsync -a --no-owner --no-group --info=progress2 --partial-dir=/home/dylan/.m13-migrate-partial-' + label +
                                    ' ' + links + '--files-from=/tmp/m13-migrate-' + label + '.files -e ' + shlex.quote('ssh ' + source_ssh_opts) +
                                    ' ' + shlex.quote(source_host + ':' + source_root) + ' ' + shlex.quote(destination))
                target_command = ('set -eu; command -v python3 >/dev/null; '
                                  'command -v rsync >/dev/null || '
                                  '(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rsync); '
                                  'command -v rsync >/dev/null; ' + ' && '.join(commands))
                save('transferring-394-inputs')
                LOG.parent.mkdir(parents=True, exist_ok=True)
                transfer_cutoff = min(deadline - 20 * 60, time.monotonic() + 40 * 60)
                with LOG.open('x') as log:
                    process = subprocess.Popen(target_ssh_cmd + [target_command], stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    if process.wait(timeout=max(1, int(transfer_cutoff - time.monotonic()))) != 0:
                        raise RuntimeError('Target cloud-to-cloud rsync failed')
                    process = None
                save('verifying-destination-hashes')
                from m13_verify_uploaded import REMOTE_CODE
                verifier = REPO / 'work/m13cloud-launchers/migrate-verify-remote.py'
                verifier.write_text(REMOTE_CODE.replace('/tmp/m13-verify-manifest.json','/tmp/m13-migrate-manifest.json'))
                verifier_sha = sha(verifier)
                run(['scp', '-F', str(args.target_config), str(verifier),
                     args.target_alias + ':/tmp/m13-migrate-verify.py'], timeout=60)
                check = ('import hashlib,pathlib; assert hashlib.sha256(pathlib.Path("/tmp/m13-migrate-verify.py").read_bytes()).hexdigest()=='
                         + repr(verifier_sha) + '; assert hashlib.sha256(pathlib.Path("/tmp/m13-migrate-manifest.json").read_bytes()).hexdigest()== '
                         + repr(sha(MANIFEST)))
                # The remote helper streams each of the exact 394 manifest paths,
                # reports progress, and has no directory walk or payload discovery.
                verify_command = 'set -eu; python3 -c ' + shlex.quote(check) + ' && python3 /tmp/m13-migrate-verify.py'
                remaining = max(1, int(transfer_cutoff - time.monotonic()))
                with LOG.open('a') as log:
                    run(target_ssh_cmd + [verify_command], timeout=remaining, stdout=log, stderr=subprocess.STDOUT)
                receipt.update(transfer_verified=True, verified_files=394,
                    destination_manifest_sha256=sha(MANIFEST), verifier_sha256=verifier_sha)
                # Remove the read key while the source endpoint is still live;
                # only then is it safe to STOP the source Pod.
                run(source_ssh_cmd + [remove], timeout=60)
                source_key_active = False
                stop_exact(SOURCE); source_started = False
                receipt.update(source_pod_final_status='EXITED', source_finished_at=time.time())
                # Freeze the adoption binding.  The ready receipt is hashed by
                # the build controller and must never be rewritten during the
                # claim wait or after handoff.
                receipt.update(status='PASSED', stage='awaiting-build-claim',
                               target_lease_started_at=time.time(),
                               target_expected_status='RUNNING', source_id=SOURCE,
                               target_id=TARGET, source_status='EXITED',
                               target_status='RUNNING', target_pod_final_status='RUNNING')
                write_ready(receipt)
                ready_published = True
                ready_sha = sha(READY)
                # Claim is made by the reviewed build supervisor; migration leaves TARGET running only then.
                lease_deadline = time.monotonic() + 20 * 60
                while time.monotonic() < lease_deadline:
                    claim = REPO / 'results/m13_cloud_build_migrated.json'
                    if claim.exists():
                        data = json.loads(claim.read_text())
                        if (data.get('status') == 'RUNNING' and
                                data.get('migration_ready_sha256') == ready_sha and data.get('pod_id') == TARGET and
                                data.get('pid') and Path('/proc', str(data['pid'])).exists()):
                            receipt.update(status='PASSED', stage='handed-off-target-running',
                                           finished_at=time.time(), target_pod_final_status='RUNNING')
                            write_receipt(receipt)
                            HANDOFF.parent.mkdir(parents=True, exist_ok=True)
                            HANDOFF.write_text(json.dumps({
                                'status': 'PASSED', 'stage': 'handed-off-target-running',
                                'migration_ready_sha256': ready_sha, 'target_pod_id': TARGET,
                                'claim_sha256': sha(claim), 'claimed_at': time.time(),
                            }, indent=2) + '\n')
                            return 0
                    time.sleep(15)
                raise RuntimeError('Migration lease claim timeout')
            finally:
                try:
                    if source_key_active:
                        run(source_ssh_cmd + [remove], timeout=60)
                except Exception as error:
                    receipt['source_key_cleanup_error'] = type(error).__name__ + ': ' + str(error)
                try:
                    run(target_ssh_cmd + ['rm -f /tmp/m13-source-pull-ed25519 /tmp/m13-source-known-hosts /tmp/m13-migrate-manifest.json /tmp/m13-migrate-root.files /tmp/m13-migrate-tree.files /tmp/m13-migrate-verify.py'], timeout=60)
                except Exception as error:
                    receipt['target_ephemeral_cleanup_error'] = type(error).__name__ + ': ' + str(error)
    except BaseException as error:
        receipt.update(status='FAILED', stage='finished', error=type(error).__name__ + ': ' + str(error), finished_at=time.time())
        signal.alarm(0)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        terminate_process(process)
        for pod_id, active in ((SOURCE, source_started), (TARGET, target_started)):
            if active:
                try: stop_exact(pod_id)
                except Exception as stop_error: receipt['stop_error_' + pod_id] = str(stop_error)
        write_receipt(receipt)
        return 1
    finally:
        signal.alarm(0)
        # Keep the PID evidence bound to the mutable receipt; a retry must be
        # an explicit new evidence set rather than silently replacing this run.


if __name__ == '__main__':
    raise SystemExit(main())
