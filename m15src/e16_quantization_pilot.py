"""Exploratory paired E16: binary scoring and rescoring on one fixed FiQA HNSW graph.

Uses cached original document vectors and the released frozen encoders. No training, new document
encoding, reserved evaluation, exhaustive-quantized-scoring claim, or production timing claim.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import numpy as np
from qdrant_client import QdrantClient, models as m

from common import REPO, SEED, bootstrap_mean, receipt, sha_file, utc_now, write_result
import e2_ann as A
import encoders15 as E
import vectors15 as V

N = 192
EFS = (16, 64, 256)
TIERS = ('zero', 'nano', 'stella-query')
PATTERNS = {
    'original_traversal': {'ignore': True},
    'binary_no_rescore': {'ignore': False, 'rescore': False, 'oversampling': 1.0},
    'binary_rescore_1': {'ignore': False, 'rescore': True, 'oversampling': 1.0},
    'binary_rescore_4': {'ignore': False, 'rescore': True, 'oversampling': 4.0},
}
OUT = REPO / 'results/m15_e16_quantization_pilot.json'
RUN = REPO / 'work/m15/e16-quantization-pilot'


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def start_server(run_dir):
    run_dir.mkdir(parents=True, exist_ok=False)
    http, grpc = free_port(), free_port()
    while http == grpc:
        grpc = free_port()
    log = (run_dir / 'qdrant.log').open('w')
    env = os.environ.copy()
    env.update({'QDRANT__STORAGE__STORAGE_PATH': str(run_dir / 'storage'),
                'QDRANT__TELEMETRY_DISABLED': 'true',
                'QDRANT__SERVICE__HTTP_PORT': str(http),
                'QDRANT__SERVICE__GRPC_PORT': str(grpc)})
    proc = subprocess.Popen([str(A.QDRANT / 'qdrant')], cwd=A.QDRANT,
                            env=env, stdout=log, stderr=log)
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError('Dedicated Qdrant server exited; inspect its log')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{http}/readyz', timeout=1)
                return proc, log, QdrantClient(host='127.0.0.1', port=http, grpc_port=grpc,
                                               prefer_grpc=True, timeout=600)
            except OSError:
                time.sleep(0.5)
        raise RuntimeError('Dedicated Qdrant did not become ready')
    except BaseException:
        proc.terminate()
        proc.wait(timeout=10)
        log.close()
        raise


def get_ids(client, name, vector, ef, pattern, exact=False):
    params = m.SearchParams(hnsw_ef=ef, exact=exact,
                           quantization=m.QuantizationSearchParams(**PATTERNS[pattern]))
    t0 = time.perf_counter()
    got = client.query_points(name, query=vector.tolist(), using='dense', limit=10,
                              search_params=params)
    return [int(p.id) for p in got.points], (time.perf_counter() - t0) * 1000


def snapshot(client, name):
    info = client.get_collection(name)
    return {'points': info.points_count, 'indexed_vectors': info.indexed_vectors_count,
            'status': info.status.value, 'config': info.config.model_dump(mode='json')}


def main(dataset='fiqa', measurement='E16', label='exploratory pilot',
         out_path=OUT, run_dir=RUN, script=__file__):
    if out_path.exists() or run_dir.exists():
        raise SystemExit('Output or isolated run directory exists; refusing to overwrite')
    started = utc_now()
    data, dv, provenance = A.load(dataset)
    chosen = sorted(range(len(data['q_ids'])), key=lambda i: hashlib.sha256(
        f"{SEED}:{data['q_ids'][i]}".encode()).hexdigest())[:N]
    qids = [data['q_ids'][i] for i in chosen]
    texts = [data['q_texts'][i] for i in chosen]
    docs = dv.astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    qv, exact, geometry, model_pins = {}, {}, {}, {}
    for tier in TIERS:
        encoder = E.make(tier)
        qv[tier] = encoder.encode(texts).astype(np.float32)
        qv[tier] /= np.linalg.norm(qv[tier], axis=1, keepdims=True)
        scores = qv[tier] @ docs.T
        ids = np.argpartition(-scores, kth=10, axis=1)[:, :11]
        order = np.argsort(-np.take_along_axis(scores, ids, axis=1), axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        top = np.take_along_axis(scores, ids, axis=1)
        exact[tier] = ids[:, :10].tolist()
        geometry[tier] = {'top1_cos': top[:, 0].tolist(),
                          'top10_top11_margin': (top[:, 9] - top[:, 10]).tolist(),
                          'top1_top10_margin': (top[:, 0] - top[:, 9]).tolist()}
        model_pins[tier] = {
            'query_vectors_sha256': hashlib.sha256(qv[tier].tobytes()).hexdigest(),
            'artifact_sha256': (sha_file(encoder.dir / 'model.npz') if tier == 'zero'
                                else encoder.onnx_sha256)}
        del encoder, scores
        print(f'encoded {tier}: {N} queries; computed original exact neighbors', flush=True)
    proc, log, client = start_server(run_dir)
    name = f'{measurement.lower()}-{dataset}-fixed-binary'
    try:
        if client.collection_exists(name):
            raise RuntimeError('Fresh isolated server unexpectedly contains the pilot collection')
        build = A.build(client, name, docs, quant='binary1')
        before = snapshot(client, name)
        print('built one shared graph', build, flush=True)
        # Native exact=true is used only as an original-scoring parity gate, never as compressed exact.
        parity = {}
        for tier in TIERS:
            got = [get_ids(client, name, v, 64, 'original_traversal', exact=True)[0]
                   for v in qv[tier]]
            parity[tier] = float(np.mean([len(set(a) & set(b)) / 10
                                         for a, b in zip(got, exact[tier])]))
            if parity[tier] < 0.999:
                raise RuntimeError(f'Original exact parity failed for {tier}: {parity[tier]}')
        print('original exact parity', parity, flush=True)
        measurements = {}
        rng = np.random.default_rng(SEED)
        keys = [(tier, pattern) for tier in TIERS for pattern in PATTERNS]
        for ef in EFS:
            observed = {key: {'ids': [], 'lat': []} for key in keys}
            for tier, pattern in keys:
                for v in qv[tier][:10]:
                    get_ids(client, name, v, ef, pattern)
            for i in range(N):
                for position in rng.permutation(len(keys)):
                    tier, pattern = keys[int(position)]
                    ids, elapsed = get_ids(client, name, qv[tier][i], ef, pattern)
                    observed[(tier, pattern)]['ids'].append(ids)
                    observed[(tier, pattern)]['lat'].append(elapsed)
            for (tier, pattern), got in observed.items():
                recovery = [len(set(a) & set(b)) / 10
                            for a, b in zip(got['ids'], exact[tier])]
                run = A.top10(got['ids'], data['doc_ids'], qids)
                ndcg = V.ndcg10(run, data['qrels'], qids)
                measurements[f'{ef}/{tier}/{pattern}'] = {
                    'ef': ef, 'tier': tier, 'pattern': pattern,
                    'recovery_at10': float(np.mean(recovery)),
                    'per_query_recovery': recovery,
                    'ndcg10': float(np.mean(list(ndcg.values()))),
                    'per_query_ndcg10': [ndcg[q] for q in qids],
                    'search_p50_ms': float(np.median(got['lat'])),
                    'search_p95_ms': float(np.quantile(got['lat'], .95)),
                    'neighbor_ids': got['ids']}
            print('completed paired settings at ef', ef, flush=True)
        after = snapshot(client, name)
        if before != after:
            raise RuntimeError('Collection configuration or counts changed during query ablation')
        contrasts = {}
        for ef in EFS:
            deltas = {}
            for tier in TIERS:
                original = np.asarray(measurements[f'{ef}/{tier}/original_traversal']['per_query_recovery'])
                quant = np.asarray(measurements[f'{ef}/{tier}/binary_rescore_1']['per_query_recovery'])
                deltas[tier] = quant - original
                contrasts[f'{ef}/{tier}/binary_minus_original'] = {
                    'mean': float(deltas[tier].mean()), 'query_bootstrap95': bootstrap_mean(deltas[tier])}
            for other in ('nano', 'stella-query'):
                interaction = deltas['zero'] - deltas[other]
                contrasts[f'{ef}/zero_minus_{other}/quantization_interaction'] = {
                    'mean': float(interaction.mean()), 'query_bootstrap95': bootstrap_mean(interaction)}
        result = {'status': 'COMPLETE', 'measurement': measurement, 'label': label,
                  'dataset': dataset, 'n_queries': N, 'n_docs': len(docs),
                  'sample': {'method': 'first 192 IDs by SHA256(seed:qid)', 'q_ids': qids,
                             'q_ids_sha256': hashlib.sha256('\n'.join(qids).encode()).hexdigest()},
                  'original_vectors': 'cached fp16 values, cast fp32 and normalized; no fresh document encoding',
                  'primary_ef': 64, 'secondary_efs': [16, 256], 'search_patterns': PATTERNS,
                  'qdrant': {'version': client.info().version, 'binary_sha256': sha_file(A.QDRANT / 'qdrant'),
                              'build': build, 'before': before, 'after': after},
                  'original_exact_parity': parity, 'exact_top10': exact, 'geometry': geometry,
                  'model_pins': model_pins, 'measurements': measurements, 'contrasts': contrasts,
                  'limits': ['one dataset and one graph build; not training-seed or graph-build intervals',
                             'native request-option ablation, not exhaustive compressed-score decomposition',
                             'query-resampling intervals and selected pilot scope; not a confirmatory test',
                             'search timing is diagnostic only; no encoder loading or application dispatch',
                             'no general manifold cause, architecture law, equal-quality dominance, or new quantizer claimed'],
                  'receipt': receipt(script, [provenance, {'dataset_pins': data['pins']},
                                              {'encoders': 'm15src/encoders15.py', 'sha256': sha_file(E.__file__)},
                                              {'e2_build_helper': 'm15src/e2_ann.py', 'sha256': sha_file(A.__file__)},
                                              {'shared_pilot_source': 'm15src/e16_quantization_pilot.py',
                                               'sha256': sha_file(__file__)}],
                                     started, extra_packages=('qdrant-client', 'onnxruntime', 'datasets'))}
        write_result(out_path, result)
        print(json.dumps(contrasts, indent=2), flush=True)
    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


if __name__ == '__main__':
    main()
