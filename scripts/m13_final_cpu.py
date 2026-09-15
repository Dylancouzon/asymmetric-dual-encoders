#!/usr/bin/env python3
"""Run the registered final executor once on CPU; preserve its exit status and log."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'work/m13-final/execution.json'


def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', M7_DEVICE='cpu',
               M7_ENCODER='stella-400M-v5', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONUNBUFFERED='1')
    command = [str(ROOT / '.venv/bin/python'), 'm13src/score13.py']
    if STATE.exists():
        raise RuntimeError('Preserve prior execution; no automatic retry')
    subprocess.run(command + ['--preflight-only'], cwd=ROOT, env=env, check=True)
    record = dict(status='RUNNING', stage='final-six', pid=os.getpid(),
                  started_at=time.time(), device='cpu', paid_compute=False)
    with STATE.open('x') as stream:
        json.dump(record, stream, indent=2)
    (ROOT / 'work/m13-final/execution.pid').write_text(str(os.getpid()))

    def save():
        record['updated_at'] = time.time()
        tmp = STATE.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(STATE)

    with (ROOT / 'logs/m13-final-six.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        record['child_pid'] = child.pid
        save()
        while child.poll() is None:
            time.sleep(30)
            save()
        record.update(exit_code=child.returncode, finished_at=time.time(),
                      stage='finished',
                      status='PASSED' if child.returncode == 0 else 'FAILED')
        if child.returncode == 5:
            record['outcome'] = 'INCOMPLETE_RESERVED'
        save()
    return child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
