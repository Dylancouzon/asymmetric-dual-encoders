#!/usr/bin/env python3
"""Registered descriptive Nano-minus-M9 row, from completed saved scores only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', M7_DEVICE='cpu', OPENBLAS_NUM_THREADS='1')
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'm13src'))
import access13 as A
import score13 as S
import m9_closeout13 as M
from m9src import final_stats as F
import numpy as np


def summarize(nano, m9, conf):
    aligned = F.align(nano, m9)
    plan, digest = F.draw_plan(aligned, conf['bootstrap']['B'], conf['bootstrap']['seed'])
    differences = {ds: x-y for ds, (_,x,y) in aligned.items()}
    means = {ds: d[plan[ds]].mean(axis=1) for ds,d in differences.items()}
    out = {'draw_plan_sha256': digest, 'partitions': {}, 'per_dataset': {}}
    for ds,d in differences.items():
        out['per_dataset'][ds] = {'delta': float(d.mean()), 'n': len(d),
            'nano_higher': int((d>0).sum()), 'm9_higher': int((d<0).sum()),
            'ties': int((d==0).sum())}
    for name, datasets in conf['partitions'].items():
        if name not in ('all6','clean4'): continue
        draws = np.mean([means[ds] for ds in datasets], axis=0)
        out['partitions'][name] = {
            'delta': float(np.mean([differences[ds].mean() for ds in datasets])),
            'ci95': np.quantile(draws,[.025,.975],method='inverted_cdf').tolist()}
    return out


def main():
    output = ROOT / 'results/m13_paired_m9_nano.json'
    if output.exists(): raise RuntimeError('Preserve prior paired result')
    nano_cfg, m9_cfg = A.production(), M.production()
    conf = json.loads(nano_cfg.registry_path.read_text())
    scores, inputs = {}, {}
    for label,cfg in [('nano',nano_cfg),('m9',m9_cfg)]:
        result = json.loads(cfg.result_path.read_text())
        man = json.loads(cfg.state_path.read_text())
        per = S.persisted(cfg, conf['partitions']['all6'])
        if set(per) != set(conf['partitions']['all6']):
            raise RuntimeError(f'{label}: all six completed rows required')
        if any(v.get('bridge_ok') is not True for v in per.values()):
            raise RuntimeError(f'{label}: failed bridge')
        if label == 'm9':
            M.validate_saved(cfg,json.loads(cfg.registry_path.read_text()),man,per)
        else:
            problems = S._identity_problems(cfg,conf,man)
            for ds,row in per.items(): problems += S.row_identity_problems(ds,row,man)
            if problems: raise RuntimeError(problems)
        if result['freeze_sha256'] != man['freeze_sha256']:
            raise RuntimeError(f'{label}: result/checkpoint identity mismatch')
        scores[label] = {ds:row['scores'] for ds,row in per.items()}
        inputs[label] = {'result_sha256':A.sha256_file(cfg.result_path),
            'checkpoint_sha256':man['freeze_sha256'],
            'score_sha256':{ds:A.sha256_file(cfg.score_path(ds)) for ds in per}}
    out = {'status':'PASSED','scope':'Descriptive Nano minus M9; no p-value or gate. Different recipes, doses and training histories prevent causal attribution.',
           'inputs':inputs,'B':conf['bootstrap']['B'],'seed':conf['bootstrap']['seed'],
           **summarize(scores['nano'],scores['m9'],conf)}
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['partitions']))


if __name__ == '__main__': main()
