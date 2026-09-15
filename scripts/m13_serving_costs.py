#!/usr/bin/env python3
"""Prepare serving assets and measure all three systems sequentially on an idle CPU.

Synthetic inputs only. No retrieval labels, no paid compute. Separate worker processes
report hydration, first-call and warm latency, peak RSS, and actual on-disk asset size.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', M7_DEVICE='cpu', OMP_NUM_THREADS='4',
                  MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                  HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('/home/dylan/asymetric-dual-encoders')
ASSETS = ROOT / 'work/m13-costs'
NANO = ROOT / 'work/m13cloud-build-preflight-retry-backup/work/m13build/BUILD-200M/onnx'
ZERO = PARENT / 'work/release/zero-v1'
BGE = ASSETS / 'bge-small'
WORDS = 'query document retrieval medicine finance law science search passage relevant'.split()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def texts(n):
    return [' '.join(WORDS[(i + j) % len(WORDS)] for j in range(n)) for i in range(20)]


def prepare():
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1)
    BGE.mkdir(parents=True, exist_ok=True)
    revision = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
    tok = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5', revision=revision,
                                       local_files_only=True)
    model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5', revision=revision,
                                      local_files_only=True).eval()
    model.config.save_pretrained(BGE)
    class Tokens(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = model
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    wrapper = Tokens().eval()
    tok.save_pretrained(BGE)
    (BGE / 'special_tokens_map.json').write_text(json.dumps(
        {k: str(v) for k,v in tok.special_tokens_map.items()}))
    samples = texts(8)[:2]
    batch = tok(samples, padding=True, return_tensors='pt')
    torch.onnx.export(wrapper, (batch['input_ids'], batch['attention_mask']),
                      str(BGE / 'model.onnx'), opset_version=17, dynamo=False,
                      input_names=['input_ids', 'attention_mask'],
                      output_names=['last_hidden_state'],
                      dynamic_axes={k: {0: 'batch', 1: 'sequence'} for k in
                                    ('input_ids', 'attention_mask', 'last_hidden_state')})
    # The actual FastEmbed reader must honor the pinned max length and dynamic padding.
    for name in ('tokenizer_config.json', 'tokenizer.json'):
        p = BGE / name; value = json.loads(p.read_text())
        if name == 'tokenizer_config.json': value['model_max_length'] = 512
        else: value['padding'] = None
        p.write_text(json.dumps(value))
    import numpy as np
    served = loader('bge-small')
    samples = texts(5)[:2] + texts(120)[:2] + [' '.join(['long'] * 600)]
    got = served(samples)
    batch = tok(samples, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.inference_mode():
        ref = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0].float(), dim=-1)
    cosine = (got * ref.numpy()).sum(1)
    if float(cosine.min()) < 0.9999:
        raise RuntimeError('BGE serving export parity failed')
    receipt = {'revision': revision, 'sha256': sha(BGE / 'model.onnx'),
               'parity_min_cos': float(cosine.min()), 'n_texts': len(samples)}
    (ASSETS / 'bge_export.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


def loader(name):
    if name == 'zero':
        sys.path.insert(0, str(ROOT / 'm11/release'))
        from zero_encoder import ZeroQueryEncoder
        frozen = json.loads((ROOT / 'm7/FREEZE.json').read_text())
        if sha(ZERO / 'model.npz') != frozen['table_sha256']:
            raise RuntimeError('Zero release table differs from freeze')
        return ZeroQueryEncoder(ZERO).encode
    import numpy as np
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType
    path, dim, pooling = (NANO, 1024, PoolingType.MEAN) if name == 'nano' else (BGE, 384, PoolingType.CLS)
    if name == 'nano':
        record = json.loads((ROOT / 'results/m13_build_record.json').read_text())
        expected = record['freeze']['onnx']['sha256']
    else: expected = json.loads((ASSETS / 'bge_export.json').read_text())['sha256'] if (ASSETS / 'bge_export.json').exists() else sha(path / 'model.onnx')
    if sha(path / 'model.onnx') != expected: raise RuntimeError('Serving artifact hash changed')
    model_name = 'm13-cost/' + name
    TextEmbedding.add_custom_model(model=model_name, pooling=pooling, normalization=True,
        sources=ModelSource(hf=model_name), dim=dim, model_file='model.onnx',
        description='M13 local serving-cost measurement', license='mit',
        size_in_gb=(path / 'model.onnx').stat().st_size / 1e9)
    encoder = TextEmbedding(model_name=model_name, specific_model_path=str(path), threads=4)
    return lambda values: np.stack(list(encoder.embed(values, batch_size=1)))


def worker(name):
    import numpy as np
    started = time.perf_counter(); encode = loader(name)
    hydration = time.perf_counter() - started
    prefix = 'Represent this sentence for searching relevant passages: ' if name == 'bge-small' else ''
    first = time.perf_counter(); encode([prefix + texts(20)[0]]); cold = (time.perf_counter() - first) * 1000
    result = {}
    for length in (5, 10, 20, 50, 120):
        values = [prefix + v for v in texts(length)]
        for _ in range(5): encode(values[:1])
        latencies = []
        for value in values:
            start = time.perf_counter(); vector = encode([value])
            latencies.append((time.perf_counter() - start) * 1000)
            if not np.isfinite(vector).all(): raise RuntimeError('Nonfinite served vector')
        result[str(length)] = {'p50_ms': float(np.quantile(latencies, .5)),
                              'p95_ms': float(np.quantile(latencies, .95)),
                              'samples_ms': latencies}
    path = {'zero': ZERO, 'nano': NANO, 'bge-small': BGE}[name]
    assets = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file() and
              (p.suffix in ('.json', '.onnx', '.npz', '.txt') or p.name == 'zero_encoder.py')}
    if name == 'zero':
        assets = {k:v for k,v in assets.items() if k in
                  ('model.npz', 'tokenizer.json', 'config.json', 'zero_encoder.py')}
    return {'model': name, 'hydration_s': hydration, 'first_query_ms': cold,
            'warm_by_words': result, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'asset_bytes': assets, 'asset_total_bytes': sum(assets.values()),
            'runtime': 'numpy/tokenizers' if name == 'zero' else 'FastEmbed/ONNX Runtime',
            'batch_size': 1, 'threads': 4}


def measure():
    live = ROOT / 'work/m13-final/execution.json'
    if live.exists() and json.loads(live.read_text())['status'] == 'RUNNING':
        raise RuntimeError('Refuse latency measurement during active final CPU evaluation')
    output = ROOT / 'results/m13_serving_costs.json'
    if output.exists(): raise RuntimeError('Preserve previous measurement')
    rows = []
    for name in ('zero', 'bge-small', 'nano'):
        for trial in range(3):
            proc = subprocess.run([sys.executable, __file__, '--worker', name],
                                  cwd=ROOT, capture_output=True, text=True, check=True)
            row = json.loads(proc.stdout.splitlines()[-1]); row['trial'] = trial
            rows.append(row); print('Measured', name, trial, flush=True)
    cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
               if line.startswith('model name'))
    blob = {'status': 'PASSED', 'cpu': cpu, 'platform': platform.platform(),
            'protocol': '3 fresh processes/model; hydration includes loader imports, verification and load, excluding interpreter startup; first inference; five warmups and20 synthetic batch1 samples/length. OS disk cache is not flushed.',
            'caveats': ['Synthetic latency inputs are not a query-distribution estimate.',
                        'Zero uses its published NumPy encoder; transformers use FastEmbed CPU.',
                        'Index figures exclude graph/IDs/metadata and apply to dense vectors only.'],
            'dense_fp32_bytes_per_million_documents': {'zero': 4096000000, 'nano': 4096000000,
                                                       'bge-small': 1536000000},
            'dense_fp32_bytes_per_query_vector': {'zero':4096,'nano':4096,'bge-small':1536},
            'rows': rows, 'script_sha256': sha(__file__)}
    output.write_text(json.dumps(blob, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--measure', action='store_true')
    group.add_argument('--worker', choices=['zero','nano','bge-small'])
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.worker: print(json.dumps(worker(args.worker)))
    else: measure()
