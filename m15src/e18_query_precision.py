"""E18 exploratory exact candidate scoring: same sign-coded documents, different query precision.

No graph, new document encoding, training, native scalar8 simulation, or serving-speed claim.
Each tier retains its own E16/E17 original-vector exact top10 target. Ties use expected inclusion.
"""
import hashlib
import json

import numpy as np

from common import REPO, bootstrap_mean, receipt, sha_file, utc_now, write_result
import e2_ann as A
import encoders15 as E

OUT = REPO / 'results/m15_e18_query_precision.json'
BUDGETS = (10, 40, 100)
TIERS = ('zero', 'nano', 'stella-query')
SOURCES = {'fiqa': 'm15_e16_quantization_pilot.json',
           'msmarco1m': 'm15_e17_quantization_replication.json'}


def expected_inclusion(scores, target_ids):
    """Expected target-top10 inclusion under uniform tie breaking at the candidate cutoff."""
    result = {c: [] for c in BUDGETS}
    for row, ids in zip(scores, target_ids):
        positions = [len(row) - c for c in BUDGETS]
        partitioned = np.partition(row, positions)
        target_scores = row[np.asarray(ids)]
        for c, position in zip(BUDGETS, positions):
            threshold = partitioned[position]
            greater = int(np.count_nonzero(row > threshold))
            equal = int(np.count_nonzero(row == threshold))
            boundary_probability = (c - greater) / equal
            if not 0 < boundary_probability <= 1:
                raise RuntimeError('Invalid candidate-cutoff tie probability')
            inclusion = (target_scores > threshold).astype(np.float64)
            inclusion += (target_scores == threshold) * boundary_probability
            result[c].append(float(inclusion.mean()))
    return result


def main():
    if OUT.exists():
        raise SystemExit('E18 receipt exists; refusing to overwrite')
    started = utc_now()
    datasets, inputs = {}, []
    for dataset, filename in SOURCES.items():
        source_path = REPO / 'results' / filename
        source = json.loads(source_path.read_text())
        data, dv, provenance = A.load(dataset)
        if source['status'] != 'COMPLETE' or source['n_docs'] != len(dv):
            raise RuntimeError('Source/reference document set mismatch')
        qids = source['sample']['q_ids']
        positions = {q: i for i, q in enumerate(data['q_ids'])}
        texts = [data['q_texts'][positions[q]] for q in qids]
        # Qdrant tagged source encodes positive dimensions as1, all others as0. +/-1 is rank-equivalent.
        doc_codes = np.where(dv > 0, np.float32(1), np.float32(-1))
        code_hash = hashlib.sha256(doc_codes.tobytes()).hexdigest()
        rows, magnitudes, precision_deltas = {}, {}, {}
        for tier in TIERS:
            encoder = E.make(tier)
            vectors = encoder.encode(texts).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            if hashlib.sha256(vectors.tobytes()).hexdigest() != source['model_pins'][tier]['query_vectors_sha256']:
                raise RuntimeError(f'{dataset}/{tier}: query vectors differ from E16/E17 reference')
            sign_query = np.where(vectors > 0, np.float32(1), np.float32(-1))
            fidelity = np.sum(vectors * sign_query, axis=1) / np.sqrt(vectors.shape[1])
            magnitudes[tier] = {'mean_cosine_to_sign_direction': float(fidelity.mean()),
                                'per_query_cosine_to_sign_direction': fidelity.tolist()}
            for mode, q in [('sign_query', sign_query), ('float_query', vectors)]:
                scores = q @ doc_codes.T
                coverage = expected_inclusion(scores, source['exact_top10'][tier])
                for c, values in coverage.items():
                    rows[f'{tier}/{mode}/{c}'] = {
                        'tier': tier, 'query_precision': mode, 'candidate_budget': c,
                        'expected_original_top10_coverage': float(np.mean(values)),
                        'per_query_expected_coverage': values}
                del scores
            for c in BUDGETS:
                delta = (np.asarray(rows[f'{tier}/float_query/{c}']['per_query_expected_coverage'])
                         - np.asarray(rows[f'{tier}/sign_query/{c}']['per_query_expected_coverage']))
                precision_deltas[f'{tier}/{c}'] = {
                    'float_minus_sign': float(delta.mean()), 'query_bootstrap95': bootstrap_mean(delta)}
            del encoder
            print(dataset, tier, {c: [round(rows[f'{tier}/{mode}/{c}']['expected_original_top10_coverage'], 4)
                                     for mode in ('sign_query', 'float_query')] for c in BUDGETS}, flush=True)
        interactions = {}
        for c in BUDGETS:
            zero_delta = (np.asarray(rows[f'zero/float_query/{c}']['per_query_expected_coverage'])
                          - np.asarray(rows[f'zero/sign_query/{c}']['per_query_expected_coverage']))
            for other in ('nano', 'stella-query'):
                other_delta = (np.asarray(rows[f'{other}/float_query/{c}']['per_query_expected_coverage'])
                               - np.asarray(rows[f'{other}/sign_query/{c}']['per_query_expected_coverage']))
                delta = zero_delta - other_delta
                interactions[f'zero_minus_{other}/{c}'] = {'mean': float(delta.mean()),
                                                          'query_bootstrap95': bootstrap_mean(delta)}
        datasets[dataset] = {'n_docs': len(dv), 'n_queries': len(qids), 'q_ids': qids,
                             'document_sign_codes_sha256': code_hash,
                             'source_receipt': filename, 'rows': rows, 'precision_deltas': precision_deltas,
                             'tier_interactions': interactions, 'query_magnitude_diagnostic': magnitudes}
        inputs.extend([provenance, {'path': str(source_path.relative_to(REPO)), 'sha256': sha_file(source_path)},
                       {'dataset_pins': data['pins']}])
        del doc_codes, dv, data
    write_result(OUT, {'status': 'COMPLETE', 'measurement': 'E18', 'label': 'exploratory controlled scoring',
                       'primary_candidate_budget': 40, 'candidate_budgets': list(BUDGETS),
                       'scoring': {'sign_query': 'sign(q) dot sign(d), positive=>+1, other=>-1',
                                   'float_query': 'normalized original q dot SAME sign(d)'},
                       'tie_rule': 'expected target inclusion under uniform boundary-tie selection',
                       'datasets': datasets,
                       'limits': ['analytically defined exact scorers, not native Qdrant scalar8 or HNSW',
                                  'float-query precision is a control, not a guaranteed best quantizer',
                                  'global candidate budget is not native per-segment oversampling',
                                  'original-top10 candidate coverage is not nDCG or measured serving latency',
                                  'fixed previously selected query samples, known-test exploratory analysis',
                                  'one teacher family, two workloads, one positive-preserving subset'],
                       'receipt': receipt(__file__, inputs + [{'encoders': 'm15src/encoders15.py',
                                                              'sha256': sha_file(E.__file__)}], started,
                                          extra_packages=('onnxruntime', 'datasets'))})


if __name__ == '__main__':
    main()
