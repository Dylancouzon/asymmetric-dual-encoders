#!/usr/bin/env python3
"""One CPU-only diagnostic of frozen M9 on the already-open COV surface."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', M7_DEVICE='cpu', M7_ENCODER='stella-400M-v5',
                  HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  TOKENIZERS_PARALLELISM='false')
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
for name in ('scripts', 'm7src', 'm9src', 'm10src'):
    sys.path.insert(0, str(REPO / name))
OUTPUT = REPO / 'results/m13_m9_cov_diagnostic.json'
INPUT = REPO / 'results/m13_cov_mid1041666.json'
CHECKPOINT = Path('/home/dylan/asymetric-dual-encoders/work/m9long/ckpt/last.pt')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    if OUTPUT.exists(): raise RuntimeError('Preserve existing diagnostic; no automatic rerun')
    plan = json.loads((REPO / 'm13/COV_DIAGNOSTIC.json').read_text())
    if sha(INPUT) != plan['current_cov_sha256']: raise RuntimeError('Current COV evidence changed')
    freeze = json.loads((REPO / 'm9/FREEZE.json').read_text())
    if sha(CHECKPOINT) != freeze['checkpoint_sha256']: raise RuntimeError('Frozen M9 checkpoint changed')
    started = time.time()
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available(): raise RuntimeError('Diagnostic must have no GPU visibility')
    import m9base
    import cov_probe
    import cov_eval10
    import teacher9
    import nano
    import datasets
    import huggingface_hub.constants
    # Older ledger imports clear offline flags; restore both env and library state.
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    datasets.config.HF_DATASETS_OFFLINE = True
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    import transformers.utils.hub
    transformers.utils.hub._is_offline_mode = True
    import _paths
    if _paths.DEVICE != 'cpu': raise RuntimeError('Retrieval must use CPU')
    def refuse(*args, **kwargs):
        raise RuntimeError('Diagnostic refuses teacher encoding; require existing caches')
    teacher9.encode = refuse
    from m13_build_preflight import check_cov
    print('Verifying admitted COV document caches', flush=True)
    cache_check = check_cov()
    print('Loading hash-verified frozen M9 on CPU', flush=True)
    model = nano.Nano(freeze['student']).to('cpu')
    blob = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if blob.get('step') != freeze['step']: raise RuntimeError('M9 step identity mismatch')
    model.load_state_dict(blob['model'], strict=True)
    del blob
    model.eval()
    units = cov_probe.units()
    import numpy as np
    def encode(texts):
        pieces = []
        for i in range(0, len(texts), 128):
            batch = [freeze.get('student_query_prefix', '') + t for t in texts[i:i+128]]
            pieces.append(model.encode_queries(batch, batch_size=16))
            print(f'Encoded {min(i+128,len(texts))}/{len(texts)} queries in current COV unit', flush=True)
        return np.concatenate(pieces)
    per = cov_eval10.score_student(encode, units=units, verbose=True)
    macro, families, by_unit = cov_eval10.macro(per, units=units)
    current = json.loads(INPUT.read_text())
    teacher = json.loads((REPO / 'results/m10_cov_teacher_ceiling.json').read_text())
    if current['step'] != plan['current_cov_step']: raise RuntimeError('Current step differs from plan')
    if set(by_unit) != set(current['by_unit']) or set(by_unit) != set(teacher['by_unit']):
        raise RuntimeError('Comparison unit surfaces differ')
    if set(families) != set(current['by_family']) or set(families) != set(teacher['by_family']):
        raise RuntimeError('Comparison family surfaces differ')
    out = {'status':'PASSED', 'scope':plan['scope'], 'decision_effect':'none; no stopping rule or recipe changed',
           'started_at':started, 'finished_at':time.time(), 'device':'cpu', 'threads':4,
           'checkpoint_sha256':freeze['checkpoint_sha256'], 'm9_step':freeze['step'],
           'current_cov_sha256':sha(INPUT), 'plan_sha256':sha(REPO/'m13/COV_DIAGNOSTIC.json'),
           'script_sha256':sha(__file__), 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
           'teacher_baseline_sha256':sha(REPO/'results/m10_cov_teacher_ceiling.json'),
           'cache_integrity':cache_check, 'm9':{'macro':macro,'by_family':families,'by_unit':by_unit,'per_unit_query':per},
           'm13':{k:current[k] for k in ('step','macro','by_family','by_unit')},
           'teacher':{k:teacher[k] for k in ('macro','by_family','by_unit')},
           'm13_minus_m9':current['macro']-macro,
           'caveats':plan['caveats']}
    with OUTPUT.open('x') as f: json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:out[k] for k in ['status','m13_minus_m9','started_at','finished_at']}),flush=True)
    print('M9 macro',macro,'M13 macro',current['macro'],'teacher macro',teacher['macro'],flush=True)

if __name__ == '__main__': main()
