#!/usr/bin/env python3
"""Bounded real-data E smoke recovery check; never launches a registered arm.

Run with .venv/bin/python scripts/m13_cloud_smoke.py. Existing outputs refuse;
all logs and interrupted checkpoints are retained, including after failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'work/m13cloud-smoke'
RESULT = REPO / 'results/m13_cloud_resume_smoke.json'
ARMS = {'E-bs32': 32, 'E-bs128': 128}
STEPS = 600
PROCESS_SECONDS = 600
OVERALL_SECONDS = 1800


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_result(result):
    # Only called after this process exclusively creates RESULT and AUDIT.
    temporary = AUDIT / 'result.pending.json'
    with temporary.open('w') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, RESULT)


def stop_child(child):
    if child.poll() is None:
        child.kill()  # SIGKILL works even when the owned child is SIGSTOPped.
    child.wait(timeout=30)


def check_deadline(deadline):
    require(time.monotonic() < deadline, 'Cloud smoke deadline exceeded')


def handle_termination(signum, _frame):
    raise SystemExit(f'Supervisor received signal {signum}')


def run_process(command, log, overall_deadline, checkpoint=None, audit=None):
    """Return audited checkpoint metadata, or completed-process elapsed seconds."""
    started = time.monotonic()
    deadline = min(overall_deadline, started + PROCESS_SECONDS)
    child = None
    with log.open('x') as output:
        try:
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=output,
                                     stderr=subprocess.STDOUT)
            while True:
                check_deadline(deadline)
                code = child.poll()
                if code is not None:
                    require(code == 0, f'Child exited {code}; see {log}')
                    require(checkpoint is None, 'Smoke completed before interruption')
                    return {'wall_seconds': round(time.monotonic() - started, 3)}
                if checkpoint is not None and checkpoint.is_file():
                    child.send_signal(signal.SIGSTOP)
                    # Confirm the process is stopped before inspecting/copying its
                    # last atomic checkpoint; never signal an unowned PID.
                    while True:
                        check_deadline(deadline)
                        require(child.poll() is None, 'Child exited before SIGSTOP audit')
                        status = Path(f'/proc/{child.pid}/status').read_text()
                        state = next(line for line in status.splitlines()
                                     if line.startswith('State:')).split()[1]
                        if state in ('T', 't'):
                            break
                        time.sleep(0.05)
                    require(not (checkpoint.parent / 'record.json').exists(),
                            'Unexpected terminal smoke record before interruption')
                    # This .pt was written by the child launched above, not an
                    # externally supplied pickle. Loading stays on CPU.
                    import torch
                    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    step = ck['step']
                    require(0 < step < STEPS and step % 100 == 0,
                            f'Invalid interruption step {step}')
                    require(ck['extra']['stopped'] is None, 'Checkpoint already stopped')
                    require(len(ck['extra']['losses']) == step, 'Incomplete loss history')
                    require(ck.get('cuda_rng') is not None, 'Missing CUDA RNG state')
                    shutil.copyfile(checkpoint, audit)
                    digest = sha(audit)
                    require(digest == sha(checkpoint), 'Interrupted checkpoint copy differs')
                    check_deadline(deadline)
                    child.kill()
                    require(child.wait(timeout=30) == -signal.SIGKILL,
                            'Expected intentional SIGKILL exit')
                    return {'step': step, 'sha256': digest,
                            'fingerprint': ck['extra']['fingerprint'],
                            'wall_seconds': round(time.monotonic() - started, 3)}
                time.sleep(0.1)
        finally:
            if child is not None:
                stop_child(child)


def validate_completion(arm, interrupted, audit):
    import torch
    out = REPO / 'work/m10arms/smoke' / arm
    rec = json.loads((out / 'record.json').read_text())
    require(rec['arm'] == arm and rec['complete'] and rec['terminal']
            and rec['status'] == 'complete' and rec['smoke'], 'Smoke did not complete')
    require(rec['smoke_steps'] == STEPS and rec['max_len'] == 512
            and rec['device'] == 'cuda', 'Smoke shape differs')
    require(rec['evaluation_mode'] == 'stub' and rec['dev6'] is None,
            'Unexpected evaluation mode')
    training = rec['training']
    start = interrupted['step']
    require(training['start_step'] == start and training['steps_run'] == STEPS - start
            and training['total_steps'] == STEPS, 'Resume did not execute remaining steps')
    require(training['examples'] == STEPS * ARMS[arm]
            and training['examples_this_run'] == (STEPS - start) * ARMS[arm],
            'Resume example counters differ')
    for cycle in range(1, 4):
        path = out / f'cycle{cycle}.pt'
        require(sha(path) == rec['checkpoints'][f'cycle{cycle}']['sha256'],
                f'Cycle {cycle} checkpoint hash differs')
    final_path = out / 'cycle3.pt'
    require(sha(final_path) == rec['final_checkpoint_sha256'], 'Final hash differs')
    initial = torch.load(audit, map_location='cpu', weights_only=False)
    final = torch.load(final_path, map_location='cpu', weights_only=False)
    require(final['step'] == STEPS, 'Final checkpoint is short')
    require(final['extra']['fingerprint'] == interrupted['fingerprint']
            == rec['recipe_fingerprint'], 'Resume fingerprint differs')
    require(final['extra']['losses'][:start] == initial['extra']['losses'],
            'Resume lost prior loss history')
    return {'passed': True, 'max_len': rec['max_len'], 'batch': rec['recipe']['batch'],
            'student': rec['recipe']['student'], 'student_source': rec['student_source'],
            'training': training, 'final_checkpoint_sha256': rec['final_checkpoint_sha256'],
            'record_sha256': sha(out / 'record.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=list(ARMS))
    args = parser.parse_args()
    require(len(args.arms) == len(set(args.arms)), 'Duplicate arms refused')
    require(sys.platform == 'linux', 'Supervisor requires Linux SIGSTOP and /proc')
    signal.signal(signal.SIGTERM, handle_termination)
    require(not os.path.lexists(RESULT) and not os.path.lexists(AUDIT),
            'Existing cloud smoke evidence refused; preserve/archive deliberately')
    for arm in args.arms:
        require(not os.path.lexists(REPO / 'work/m10arms/smoke' / arm),
                f'Existing {arm} smoke directory refused')
    python = REPO / '.venv/bin/python'
    require(python.is_file(), 'Missing checkout virtual environment')
    AUDIT.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'scope': 'smoke only; stub evaluations', 'arms': {}}
    with RESULT.open('x') as f:
        json.dump(result, f)
    started = time.monotonic()
    deadline = started + OVERALL_SECONDS
    try:
        for arm in args.arms:
            check_deadline(deadline)
            command = [str(python), '-u', 'm10src/run_arm.py', arm, '--device', 'cuda',
                       '--smoke-steps', str(STEPS), '--max-len', '512', '--ckpt-every', '100']
            audit = AUDIT / f'{arm}.interrupted.pt'
            entry = result['arms'][arm] = {'command': command, 'status': 'running'}
            save_result(result)
            print(f'{arm}: starting 512-token smoke and checkpoint interruption', flush=True)
            entry['interrupted'] = run_process(command, AUDIT / f'{arm}.initial.log',
                deadline, REPO / 'work/m10arms/smoke' / arm / 'ckpt.pt', audit)
            save_result(result)
            print(f"{arm}: resuming from step {entry['interrupted']['step']}", flush=True)
            entry['resume'] = run_process(command + ['--resume'],
                                          AUDIT / f'{arm}.resume.log', deadline)
            entry.update(validate_completion(arm, entry['interrupted'], audit), status='complete')
            check_deadline(deadline)
            save_result(result)
            print(f'{arm}: smoke recovery passed', flush=True)
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['wall_seconds'] = round(time.monotonic() - started, 3)
        save_result(result)


if __name__ == '__main__':
    main()
