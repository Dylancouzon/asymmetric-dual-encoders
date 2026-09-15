#!/usr/bin/env python3
"""Evaluate a pinned M13 rolling checkpoint on CPU against the existing M9 COV baseline."""
import m13_m9_cov_diagnostic as base  # installs CPU-only and offline environment before torch
import json
import os
from pathlib import Path
import subprocess
import time

REPO = base.REPO
PLAN = REPO / 'm13/COV_CURRENT_CPU.json'
OUTPUT = REPO / 'results/m13_current_cov_cpu.json'
BASELINE = REPO / 'results/m13_m9_cov_diagnostic.json'


def main():
    if OUTPUT.exists(): raise RuntimeError('Preserve prior result; refuse repeat')
    plan = json.loads(PLAN.read_text())
    checkpoint = REPO / plan['checkpoint']
    if base.sha(checkpoint) != plan['checkpoint_sha256']: raise RuntimeError('Checkpoint hash mismatch')
    if base.sha(BASELINE) != plan['m9_baseline_sha256']: raise RuntimeError('M9 baseline changed')
    started = time.time()
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available(): raise RuntimeError('CPU diagnostic must not see GPUs')
    import m9base
    import cov_probe
    import cov_eval10
    import nano10
    import teacher9
    import datasets
    import huggingface_hub.constants
    import transformers.utils.hub
    teacher9.ENC9 = Path('/home/dylan/asymetric-dual-encoders/work/enc9')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    datasets.config.HF_DATASETS_OFFLINE = True
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    transformers.utils.hub._is_offline_mode = True
    import _paths
    if _paths.DEVICE != 'cpu': raise RuntimeError('Retrieval must use CPU')
    def refuse(*args, **kwargs): raise RuntimeError('Refuse teacher encoding; existing caches only')
    teacher9.encode = refuse
    from m13_build_preflight import check_cov
    print('Verifying admitted COV caches', flush=True)
    integrity = check_cov()
    model = nano10.Nano10(**plan['recipe']).to('cpu')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if blob['step'] != plan['checkpoint_step']: raise RuntimeError('Checkpoint step mismatch')
    model.load_state_dict(blob['model'], strict=True)
    if not model.under_cap(): raise RuntimeError('Model exceeds registered cap')
    examples = blob['extra']['examples']
    fingerprint = blob['extra']['fingerprint']
    del blob
    model.eval()
    units = cov_probe.units()
    import numpy as np
    def encode(texts):
        pieces = []
        for i in range(0, len(texts), 128):
            pieces.append(model.encode_queries(texts[i:i+128], batch_size=16))
            print(f'Encoded {min(i+128,len(texts))}/{len(texts)} queries in current COV unit', flush=True)
        return np.concatenate(pieces)
    per = cov_eval10.score_student(encode, units=units, verbose=True)
    macro, families, by_unit = cov_eval10.macro(per, units=units)
    baseline = json.loads(BASELINE.read_text())
    m9 = baseline['m9']; teacher = baseline['teacher']
    if set(per) != set(m9['per_unit_query']) or set(by_unit) != set(teacher['by_unit']):
        raise RuntimeError('Evaluation unit sets differ')
    if any(set(v) != set(m9['per_unit_query'][k]) for k,v in per.items()):
        raise RuntimeError('Evaluation query sets differ')
    out = {'status':'PASSED','scope':plan['scope'],'caveats':plan['caveats'],
           'started_at':started,'finished_at':time.time(),'device':'cpu','threads':4,
           'plan_sha256':base.sha(PLAN),'script_sha256':base.sha(__file__),
           'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
           'checkpoint_sha256':plan['checkpoint_sha256'],'checkpoint_step':plan['checkpoint_step'],
           'examples':examples,'recipe_fingerprint':fingerprint,
           'm9_baseline_sha256':base.sha(BASELINE),'cache_integrity':integrity,
           'm13':{'macro':macro,'by_family':families,'by_unit':by_unit,'per_unit_query':per},
           'm9':{k:m9[k] for k in ('macro','by_family','by_unit')},'teacher':teacher,
           'm13_minus_m9':macro-m9['macro'],'teacher_retention':macro/teacher['macro']}
    with OUTPUT.open('x') as f: json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:out[k] for k in ('status','examples','m13_minus_m9','teacher_retention')}),flush=True)
    print('M13 macro',macro,'M9 macro',m9['macro'],flush=True)

if __name__ == '__main__': main()
