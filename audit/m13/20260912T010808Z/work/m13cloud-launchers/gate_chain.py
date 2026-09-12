#!/usr/bin/env python3
"""One new Pod: admitted fp16 timing then registered LoTTE read, backup and STOP."""
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path('/home/dylan/asymetric-dual-encoders/work/m13cloud')
REMOTE = '/home/dylan/asymetric-dual-encoders'
KEYS = Path('/home/dylan/.config/runpod')
OLD = 'k3aee2m68765em'
PRIOR = (OLD, 'exulxoxelug5um')
NAME = 'm13-gate-chain-20260911'
RESULT = REPO / 'results/m13_cloud_gate_chain.json'
ALLOCATION = REPO / 'results/m13_gate_chain_allocation.json'
ROOTS = ('work/lotte/gate13', 'work/lotte/enc', 'm13/LOTTE_GATE.json')
ACTIVE = REPO / 'results/m13_gate_chain_active_pod.json'
BENCH = REPO / 'results/m13_encode_benchmark_fp16.json'
SSH_CONFIG = KEYS / 'm13_gate_chain_ssh_config'
SSH = ['ssh', '-n', '-F', str(SSH_CONFIG), 'm13-gate-chain']
SCP = ['scp', '-F', str(SSH_CONFIG)]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def run(cmd, timeout=60, **kw):
    return subprocess.run(cmd, check=True, timeout=timeout, **kw)


def main():
    if BENCH.exists() or ACTIVE.exists() or (REPO / 'logs/m13-fp16-encode.log').exists() or (REPO / 'logs/m13-lotte-gate.log').exists() or any((REPO / p).exists() for p in ROOTS):
        raise RuntimeError('Refuse existing benchmark or replacement assignment')
    run(['git', 'diff', '--quiet', 'HEAD'], cwd=REPO)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    pushed = subprocess.check_output(['git', 'ls-remote', '--exit-code', 'origin',
        'refs/heads/m13-stage1-execution-prep'], cwd=REPO, timeout=30, text=True).split()[0]
    if pushed != head:
        raise RuntimeError('Require exact pushed branch HEAD')
    run(['git', 'ls-files', '--error-unmatch', str(ALLOCATION.relative_to(REPO))], cwd=REPO, stdout=subprocess.DEVNULL)
    alloc = json.loads(ALLOCATION.read_text())
    required = ('results/m13_encode_benchmark.json', 'm13/build_config.json',
        'm13/LOTTE_GATE_MANIFEST.json', 'results/m8_lotte_pin.json',
        'm13/LOTTE_GATE_REGISTRATION.json', 'results/m10_screen_verdicts.json',
        'm10/screen_registry.json', 'm13/runpod_pod_config.json',
        'results/m10_arm_E-bs32.json', 'results/m10_arm_E-bs128.json')
    if alloc.get('status') != 'PASSED' or alloc.get('budget_ceiling_usd') != 1000 or alloc.get('max_hours') != 17 or alloc.get('gate_max_hours') != 15:
        raise RuntimeError('Require reviewed fixed 17h chain / 15h gate allocation')
    for key in ('max_cost_usd', 'total_price_usd_per_hour', 'remaining_budget_usd'):
        if type(alloc.get(key)) not in (int, float) or not math.isfinite(alloc[key]) or alloc[key] <= 0:
            raise RuntimeError('Invalid allocation ' + key)
    if 17 * alloc['total_price_usd_per_hour'] > alloc['max_cost_usd'] or alloc['max_cost_usd'] > alloc['remaining_budget_usd']:
        raise RuntimeError('Insufficient stage allocation')
    if set(alloc.get('artifact_sha256', {})) != set(required):
        raise RuntimeError('Allocation lacks exact registered artifact bindings')
    for name in required:
        if sha(REPO / name) != alloc['artifact_sha256'][name]:
            raise RuntimeError('Changed allocation input: ' + name)
    pin_commit_time = subprocess.check_output(['git','log','-1','--format=%cI','--','results/m8_lotte_pin.json'], cwd=REPO, text=True).strip()
    local_day = datetime.now(ZoneInfo('America/New_York')).date()
    if datetime.fromisoformat(pin_commit_time).astimezone(ZoneInfo('America/New_York')).date() != local_day:
        raise RuntimeError('Pin must be committed on the local day of the read')
    registration = json.loads((REPO / 'm13/LOTTE_GATE_REGISTRATION.json').read_text())
    keys = registration['surface']['slices']
    if len(keys) != 7 or any(len(k.split('/')) != 2 or '..' in k or k.startswith('/') for k in keys):
        raise RuntimeError('Invalid registered slice keys')
    protected_files = ['work/lotte/remediated/' + k + '/' + name for k in sorted(keys)
        for name in ('collection.tsv', 'questions.forum.tsv', 'qas.forum.jsonl')]
    for name in protected_files:
        path = REPO / name
        if not path.is_file() or path.is_symlink() or any(q.is_symlink() for q in path.parents):
            raise RuntimeError('Require independently staged regular registered input')
    with RESULT.open('x') as f:
        json.dump({'status': 'STARTING'}, f)
    receipt = {'status': 'RUNNING', 'stage': 'preflight', 'pid': os.getpid(),
               'started_at': time.time(), 'code_commit': head, 'original_pod_id': OLD,
               'name': NAME, 'maximum_execution_seconds': 61200, 'allocation_sha256': sha(ALLOCATION), 'owned_pod_ids': []}
    headers = {'Authorization': 'Bearer ' + (KEYS / 'api_key').read_text().strip(),
               'Content-Type': 'application/json', 'User-Agent': 'm13-gate-chain/1.0'}
    owned, create_attempted, success = [], False, False
    ready = False
    gate_started = False
    execution_deadline = time.monotonic() + 61200

    def save(stage=None):
        if stage:
            receipt['stage'] = stage
        receipt['updated_at'] = time.time()
        p = RESULT.with_suffix('.pending.json')
        p.write_text(json.dumps(receipt, indent=2) + '\n')
        p.replace(RESULT)
        print(receipt['stage'], flush=True)

    def api(path, method='GET', body=None):
        req = urllib.request.Request('https://rest.runpod.io/v1/' + path,
              data=None if body is None else json.dumps(body).encode(), headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as f:
            raw = f.read()
            return json.loads(raw) if raw else None

    def interrupted(sig, frame):
        raise RuntimeError('Interrupted: ' + str(sig))

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    signal.alarm(57000)  # Reserve 70 minutes for durable backup and STOP.
    try:
        save()
        for old_id in PRIOR:
            if api('pods/' + old_id).get('desiredStatus') != 'EXITED':
                raise RuntimeError('Both prior Pods must remain stopped')
        if any(p.get('name') == NAME for p in api('pods')):
            raise RuntimeError('Replacement name already exists; no duplicate create')
        quote = {'query': 'query { gpuTypes(input:{id:"NVIDIA A100-SXM4-80GB"}) { lowestPrice(input:{gpuCount:1,secureCloud:true,minDisk:530,minMemoryInGb:100,minVcpuCount:8}) { stockStatus uninterruptablePrice } } }'}
        req = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps(quote).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=30) as f:
            q = json.load(f)['data']['gpuTypes'][0]['lowestPrice']
        if not q or q['stockStatus'] == 'None' or not 0 < float(q['uninterruptablePrice']) <= 1.59:
            raise RuntimeError('No admissible A100 quote')
        receipt['quote'] = q
        spec = json.loads((REPO / 'm13/runpod_pod_config.json').read_text())
        spec.pop('dataCenterIds', None)
        spec['name'] = NAME
        spec['env']['PUBLIC_KEY'] = (KEYS / 'm13_ssh.pub').read_text().strip()
        save('creating')
        create_attempted = True
        pod = api('pods', 'POST', spec)  # Exactly once; ambiguity is handled by discovery + STOP.
        owned.append(pod['id'])
        receipt['owned_pod_ids'] = owned
        if pod['id'] in PRIOR:
            raise RuntimeError('Unexpected original Pod id')
        if not 0 < float(pod['costPerHr']) <= 1.59 or float(pod['costPerHr']) + (530 * .10 / 720) > alloc['total_price_usd_per_hour']:
            raise RuntimeError('Created Pod exceeds GPU price ceiling')
        receipt['pod_id'] = pod['id']
        receipt['gpu_price_usd_h'] = float(pod['costPerHr'])
        ACTIVE.write_text(json.dumps({'pod_id': pod['id'], 'original_pod_id': OLD,
            'status': 'PROVISIONED', 'gpu_price_ceiling_usd_h': 1.59,
            'name': NAME, 'ssh_config': str(SSH_CONFIG), 'ssh_alias': 'm13-gate-chain'}, indent=2) + '\n')
        save('ssh-readiness')
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            pod = api('pods/' + owned[0])
            port = (pod.get('portMappings') or {}).get('22')
            if pod.get('publicIp') and port:
                SSH_CONFIG.write_text('Host m13-gate-chain\n  HostName ' + pod['publicIp'] + '\n  Port ' + str(port) +
                    '\n  User root\n  IdentityFile ' + str(KEYS / 'm13_ssh') +
                    '\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n  ConnectTimeout 10\n')
                SSH_CONFIG.chmod(0o600)
                r = subprocess.run(SSH + ['true'], timeout=20, capture_output=True)
                if r.returncode == 0:
                    ready = True
                    break
            time.sleep(5)
        else:
            raise RuntimeError('SSH deadline exceeded')
        save('bootstrap')
        bundle = REPO / 'work/m13cloud-launchers/gate-chain.bundle'
        run(['git', 'bundle', 'create', str(bundle), 'HEAD'], timeout=120, cwd=REPO)
        run(SCP + [str(bundle), 'm13-gate-chain:/tmp/m13-gate-chain.bundle'], timeout=180)
        setup = 'set -eu; mountpoint -q /home/dylan; mkdir -p /home/dylan/setup-logs; test ! -e ' + REMOTE
        setup += '; git clone /tmp/m13-gate-chain.bundle ' + REMOTE + '; cd ' + REMOTE
        setup += '; git checkout --detach ' + shlex.quote(head)
        setup += '; git update-ref refs/remotes/origin/m13-stage1-execution-prep ' + head
        setup += '; bash scripts/m13_cloud_bootstrap.sh > /home/dylan/setup-logs/bootstrap-gate-chain.log 2>&1'
        run(SSH + ['timeout --kill-after=30s 1500s bash -c ' + shlex.quote(setup)], timeout=1540)
        save('uploading')
        prefixes = ('home/dylan/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/',
                    'home/dylan/.cache/huggingface/hub/models--NovaSearch--stella_en_400M_v5/')
        source_name = 'home/dylan/asymetric-dual-encoders/work/train/stores/squad-ctx.json'
        files = [n for n in (REPO / 'm13/cloud_transfer_files.txt').read_text().splitlines()
                 if n.startswith(prefixes) or n == source_name]
        if source_name not in files:
            raise RuntimeError('Missing benchmark source in transfer list')
        listing = REPO / 'work/m13cloud-launchers/gate-chain-files.txt'
        listing.write_text('\n'.join(files) + '\n')
        run(['rsync', '-a', '--files-from=' + str(listing), '-e', 'ssh -F ' + shlex.quote(str(SSH_CONFIG)), '/', 'm13-gate-chain:/'], timeout=2400)
        hashes = {n: sha(Path('/') / n) for n in files}
        for arm in ('E-bs32', 'E-bs128'):
            ck = REPO / 'work/m10arms' / arm / 'cycle3.pt'
            rec = json.loads((REPO / 'results' / ('m10_arm_' + arm + '.json')).read_text())
            if sha(ck) != rec['final_checkpoint_sha256']:
                raise RuntimeError('Checkpoint checksum mismatch: ' + arm)
            target = REMOTE + '/work/m10arms/' + arm
            run(SSH + ['mkdir -p ' + shlex.quote(target)])
            run(SCP + [str(ck), 'm13-gate-chain:' + target + '/cycle3.pt'], timeout=600)
            hashes[(Path(target) / 'cycle3.pt').as_posix().lstrip('/')] = rec['final_checkpoint_sha256']
        check = REPO / 'work/m13cloud-launchers/gate-chain-sha256.txt'
        check.write_text('\n'.join(h + '  /' + n for n, h in hashes.items()) + '\n')
        run(SCP + [str(check), 'm13-gate-chain:/tmp/m13-gate-chain-sha256.txt'])
        run(SSH + ['sha256sum --check --status /tmp/m13-gate-chain-sha256.txt'], timeout=180)
        receipt['uploaded_file_count'] = len(hashes)
        receipt['transfer_verified'] = True
        save('encoding')
        env = 'export HF_HOME=/home/dylan/.cache/huggingface HF_HUB_CACHE=/home/dylan/.cache/huggingface/hub HF_DATASETS_CACHE=/home/dylan/.cache/huggingface/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; cd ' + REMOTE + '; .venv/bin/python -u scripts/m13_encode_benchmark.py --dtype fp16'
        with (REPO / 'logs/m13-fp16-encode.log').open('x') as f:
            run(SSH + ['timeout --kill-after=30s 600s bash -c ' + shlex.quote(env)], timeout=640, stdout=f, stderr=subprocess.STDOUT)
        run(SCP + ['m13-gate-chain:' + REMOTE + '/results/' + BENCH.name, str(BENCH)], timeout=60)
        got = subprocess.check_output(SSH + ['sha256sum ' + REMOTE + '/results/' + BENCH.name], timeout=30, text=True).split()[0]
        data = json.loads(BENCH.read_text())
        if got != sha(BENCH) or data.get('status') != 'PASSED' or data.get('encode_dtype') != 'fp16' or data.get('git_head') != head:
            raise RuntimeError('Benchmark result verification failed')
        receipt['benchmark_sha256'] = got
        receipt['fp16_backup_verified'] = True
        measurements = data.get('measurements', [])
        if [r.get('passages') for r in measurements] != [1000, 10000]:
            raise RuntimeError('Require both admitted timing sizes')
        seconds_per_row = []
        for row in measurements:
            sec = row.get('seconds')
            if type(sec) not in (int, float) or not math.isfinite(sec) or sec <= 0:
                raise RuntimeError('Invalid measured timing')
            seconds_per_row.append(sec / row['passages'])
        documents = sum(r['docs_after_remedy'] for r in keys.values())
        projected = max(seconds_per_row) * documents * 2 / 3600 + 1
        receipt['measured_gate_allowance_hours'] = projected
        if projected > 15:
            raise RuntimeError('Measured fp16 gate allowance exceeds registered stage cap')
        save('uploading-registered-inputs')
        inputs = {name: sha(REPO / name) for name in protected_files}
        listing = REPO / 'work/m13cloud-launchers/gate-chain-lotte.files'
        listing.write_text('\n'.join(protected_files) + '\n')
        run(['rsync', '-a', '--timeout=120', '--files-from=' + str(listing), '-e',
            'ssh -F ' + shlex.quote(str(SSH_CONFIG)), str(REPO) + '/', 'm13-gate-chain:' + REMOTE + '/'], timeout=900)
        check = REPO / 'work/m13cloud-launchers/gate-chain-lotte-sha256.txt'
        check.write_text('\n'.join(h + '  ' + REMOTE + '/' + n for n,h in inputs.items()) + '\n')
        run(SCP + [str(check), 'm13-gate-chain:/tmp/m13-gate-chain-lotte-sha256.txt'])
        run(SSH + ['sha256sum --check --status /tmp/m13-gate-chain-lotte-sha256.txt'], timeout=300)
        # No code/config mutation after this point. The pinned executor owns the access.
        if datetime.now(ZoneInfo('America/New_York')).date() != local_day:
            raise RuntimeError('The local pin day elapsed before the read')
        environment = env.split('; cd ', 1)[0] + '; cd ' + REMOTE + '; '
        save('gate-preflight')
        run(SSH + ['timeout --kill-after=30s 180s bash -c ' + shlex.quote(environment + '.venv/bin/python -u m13src/lotte_gate13.py --preflight-only --device cuda')], timeout=220)
        remaining = int(execution_deadline - time.monotonic()) - 4200
        gate_seconds = min(54000, remaining)
        if gate_seconds < projected * 3600:
            raise RuntimeError('Setup used too much of the fixed stage allocation')
        job = head[:12] + '-' + str(os.getpid())
        receipt['remote_job_id'] = job
        save('gate-read')
        gate_started = True
        pidfile = '/tmp/m13-gate-chain-' + job + '.pid'
        command = 'timeout --kill-after=30s ' + str(gate_seconds) + 's bash -c ' + shlex.quote(environment + '.venv/bin/python -u m13src/lotte_gate13.py --device cuda')
        launch = 'env M13_LOTTE_JOB_ID=' + job + ' setsid bash -c ' + shlex.quote('echo $$ > ' + pidfile + '; exec ' + command)
        with (REPO / 'logs/m13-lotte-gate.log').open('x') as f:
            run(SSH + [launch], timeout=gate_seconds + 40, stdout=f, stderr=subprocess.STDOUT)
        success = True
    except BaseException as e:
        receipt['error'] = type(e).__name__ + ': ' + str(e)
    finally:
        signal.alarm(0)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        # Quiesce the owned gate process, then back up only registered output roots.
        if ready and gate_started:
            try:
                available = int(execution_deadline - time.monotonic()) - 720
                if available <= 0:
                    raise RuntimeError('Backup deadline reached; preserve remote volume')
                signal.alarm(available)
                quiesce = """import os,pathlib,signal,time
p=pathlib.Path(PIDFILE)
if p.exists():
    pid=int(p.read_text()); proc=pathlib.Path('/proc')/str(pid)
    if proc.exists():
        if MARKER not in (proc/'environ').read_bytes().split(bytes([0])) or os.getpgid(pid)!=pid: raise RuntimeError('Unowned process')
        os.killpg(pid,signal.SIGTERM); time.sleep(3)
        try: os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError: pass
""".replace('PIDFILE', repr(pidfile)).replace('MARKER', repr(('M13_LOTTE_JOB_ID=' + job).encode()))
                run(SSH + ['python3 -c ' + shlex.quote(quiesce)], timeout=60)
                save('backup')
                manifest_code = """import hashlib,json,pathlib
out={}
for name in ROOTS:
    root=pathlib.Path(name)
    for p in ([root] if root.is_file() else root.rglob('*')):
        if p.is_symlink(): raise RuntimeError('Symlink in output')
        if not p.is_file(): continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''): h.update(b)
        out[str(p)]=h.hexdigest()
pathlib.Path('/tmp/m13-gate-chain-outputs.json').write_text(json.dumps(out))
""".replace('ROOTS', repr(ROOTS))
                run(SSH + ['timeout --kill-after=30s 540s bash -c ' + shlex.quote('set -eu; cd ' + REMOTE + '; python3 -c ' + shlex.quote(manifest_code))], timeout=600)
                manifest = REPO / 'work/m13cloud-launchers/gate-chain-outputs.json'
                run(SCP + ['m13-gate-chain:/tmp/m13-gate-chain-outputs.json', str(manifest)])
                outputs = json.loads(manifest.read_text())
                for name in outputs:
                    if '..' in Path(name).parts or not any(name == root or name.startswith(root + '/') for root in ROOTS):
                        raise RuntimeError('Unexpected backup path')
                    if (REPO / name).exists():
                        raise RuntimeError('Backup would overwrite existing evidence')
                listing = REPO / 'work/m13cloud-launchers/gate-chain-backup.files'
                listing.write_text('\n'.join(outputs) + '\n')
                run(['rsync', '-a', '--timeout=120', '--files-from=' + str(listing), '-e',
                    'ssh -F ' + shlex.quote(str(SSH_CONFIG)), 'm13-gate-chain:' + REMOTE + '/', str(REPO) + '/'], timeout=2400)
                if any(sha(REPO / n) != digest for n,digest in outputs.items()):
                    raise RuntimeError('Gate backup checksum mismatch')
                receipt['backup_verified'] = True
                receipt['backup_manifest_sha256'] = sha(manifest)
                receipt['backup_files'] = len(outputs)
                if success:
                    gate = json.loads((REPO / 'm13/LOTTE_GATE.json').read_text())
                    if gate.get('executed') is not True or gate.get('git_head') != head or gate.get('decision') not in ('skipped','veto','no_veto'):
                        raise RuntimeError('Invalid completed gate record')
                    receipt['decision'] = gate['decision']
            except BaseException as e:
                success = False
                receipt['backup_error'] = type(e).__name__ + ': ' + str(e)
            finally:
                signal.alarm(0)
        if create_attempted and not owned:
            for _ in range(6):
                try:
                    owned = [p['id'] for p in api('pods') if p.get('name') == NAME and p['id'] not in PRIOR]
                    if owned:
                        break
                except Exception as e:
                    receipt['discovery_error'] = str(e)
                time.sleep(10)
            if not owned:
                receipt['creation_ambiguity_unresolved'] = True
                receipt['discovery_error'] = 'Creation outcome unresolved; inspect unique Pod name before any retry'
        receipt['owned_pod_ids'] = owned
        stopped = {}
        for podid in owned:
            if podid in PRIOR:
                continue
            for _ in range(6):
                try:
                    api('pods/' + podid + '/stop', 'POST')
                    stopped[podid] = api('pods/' + podid).get('desiredStatus')
                    if stopped[podid] == 'EXITED':
                        break
                except Exception as e:
                    receipt['stop_error'] = str(e)
                time.sleep(3)
        receipt['pod_final_statuses'] = stopped
        receipt['pod_final_status'] = 'EXITED' if owned and all(stopped.get(p) == 'EXITED' for p in owned) else 'UNCONFIRMED'
        receipt['status'] = 'PASSED' if success and owned and all(stopped.get(p) == 'EXITED' for p in owned) else 'FAILED'
        receipt['finished_at'] = time.time()
        if receipt['status'] == 'PASSED':
            active = json.loads(ACTIVE.read_text())
            active['status'] = 'PASSED'
            ACTIVE.write_text(json.dumps(active, indent=2) + '\n')
        save('finished')
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
